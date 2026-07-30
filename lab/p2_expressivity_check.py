#!/usr/bin/env python
"""DIAGNOSTIC: does the Phase-0 mixer family actually separate by algebraic class?

PLAN2 Phase-0 probe 4 reads an *ordering* of diag[0,1] / diag[-1,1] /
DeltaNet[-1,1] off the competition task.  That reading is only meaningful if the
three implementations demonstrably differ on problems whose algebraic class is
known.  This script establishes that calibration on synthetic word problems, so
that a null on the real task can be attributed to the task rather than to a bug.

Tasks (prefix-product / running-state, one target per position):
  parity : Z_2            -- needs eigenvalue -1; diag[0,1] provably cannot
  mod3   : Z_3            -- abelian, cyclic; needs non-triangular / complex
  a5     : A_5 (non-solvable, order 60) -- NC^1-complete word problem

No competition data is touched.  Everything here is synthetic and generated in
this process.

Usage:
  python lab/p2_expressivity_check.py --lengths 32 --steps 3000
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO = Path(__file__).resolve().parent.parent
SUBS = REPO / "submissions" / "plan2-phase0"


# --------------------------------------------------------------------------- #
# synthetic monoids
# --------------------------------------------------------------------------- #
def _a5_table() -> tuple[int, torch.Tensor]:
    """Cayley table of A_5 as permutations of 5 points."""
    perms = [p for p in itertools.permutations(range(5))]

    def sign(p):
        s = 1
        for i in range(5):
            for j in range(i + 1, 5):
                if p[i] > p[j]:
                    s = -s
        return s

    even = [p for p in perms if sign(p) == 1]
    index = {p: i for i, p in enumerate(even)}
    n = len(even)
    table = torch.zeros((n, n), dtype=torch.long)
    for i, a in enumerate(even):
        for j, b in enumerate(even):
            table[i, j] = index[tuple(a[b[k]] for k in range(5))]
    return n, table


_A5_N, _A5_TABLE = _a5_table()


def make_batch(task: str, batch: int, length: int, device) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return (tokens, per-position targets, vocab_size, n_classes)."""
    if task == "parity":
        x = torch.randint(0, 2, (batch, length), device=device)
        y = torch.cumsum(x, dim=1) % 2
        return x, y, 2, 2
    if task == "mod3":
        x = torch.randint(0, 2, (batch, length), device=device)
        y = torch.cumsum(x, dim=1) % 3
        return x, y, 2, 3
    if task == "a5":
        table = _A5_TABLE.to(device)
        x = torch.randint(0, _A5_N, (batch, length), device=device)
        y = torch.empty_like(x)
        acc = torch.zeros(batch, dtype=torch.long, device=device)
        for t in range(length):
            acc = table[acc, x[:, t]]
            y[:, t] = acc
        return x, y, _A5_N, _A5_N
    raise ValueError(task)


# --------------------------------------------------------------------------- #
def load_arch(name: str):
    path = SUBS / name / "submission.py"
    spec = importlib.util.spec_from_file_location(f"p2sub_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Probe(nn.Module):
    """The shipped scaffold + a fresh linear head sized to the synthetic task."""

    def __init__(self, module, vocab: int, length: int, n_classes: int) -> None:
        super().__init__()
        spec = module.ModelSpec(
            vocab_size=vocab, max_seq_len=length, maximum_model_state_elements=10**9
        )
        self.body = module.Model(spec)
        self.head = nn.Linear(module.D_MODEL, n_classes)

    def forward(self, x, mask):
        b = self.body
        positions = torch.arange(x.shape[1], device=x.device)
        h = b.token_embedding(x) + b.position_embedding(positions)
        for block in b.blocks:
            h = block(h, mask)
        return self.head(b.final_norm(h))


def run_cell(name: str, task: str, length: int, steps: int, lr: float, seed: int, device) -> dict:
    torch.manual_seed(seed)
    module = load_arch(name)
    _, _, vocab, n_classes = make_batch(task, 2, length, device)
    model = Probe(module, vocab, length, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    mask = torch.ones((256, length), dtype=torch.bool, device=device)
    t0 = time.time()
    for _ in range(steps):
        x, y, _, _ = make_batch(task, 256, length, device)
        logits = model(x, mask)
        loss = F.cross_entropy(logits.reshape(-1, n_classes), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        x, y, _, _ = make_batch(task, 2048, length, device)
        m = torch.ones((2048, length), dtype=torch.bool, device=device)
        pred = model(x, m).argmax(-1)
        tok_acc = (pred == y).float().mean().item()
        last_acc = (pred[:, -1] == y[:, -1]).float().mean().item()
        seq_acc = (pred == y).all(dim=1).float().mean().item()
    return {
        "arch": name.split("_")[0],
        "task": task,
        "length": length,
        "seed": seed,
        "steps": steps,
        "token_acc": round(tok_acc, 4),
        "final_pos_acc": round(last_acc, 4),
        "seq_acc": round(seq_acc, 4),
        "chance": round(1.0 / n_classes, 4),
        "loss": round(loss.item(), 4),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arch",
        nargs="+",
        default=["diag01", "diagpm1", "delta01", "deltapm1", "lstm", "gru", "attn", "mlp"],
    )
    ap.add_argument("--tasks", nargs="+", default=["parity", "mod3", "a5"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[32])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[74])
    ap.add_argument("--out", default="lab/p2/expressivity_check.jsonl")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        for task in args.tasks:
            for length in args.lengths:
                for name in args.arch:
                    for seed in args.seeds:
                        row = run_cell(
                            f"{name}_d128_L2", task, length, args.steps, args.lr, seed, device
                        )
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
