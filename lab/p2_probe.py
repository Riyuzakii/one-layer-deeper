#!/usr/bin/env python
"""DIAGNOSTIC: synthetic state-tracking probe for §3.2 / §3.3.

Why this exists.  ~950 experiments in this project read MAX_T=0 on the scored
task.  A sweep that reads "flat" there is uninterpretable unless the sweep is
first shown to have *resolving power* on a task where theory says it must.  So
before touching the evaluator, run the exact same layers on three word problems
with known algebraic class:

  parity  -- Z2.        Needs an eigenvalue of -1.  Provably impossible for a
                        transition operator with spectrum in [0,1].
  mod3    -- Z3.        Needs a non-triangular transition (rotation by 2pi/3).
  A5      -- non-solvable.  NC^1-complete word problem (Barrington); the
                        canonical hard case, and what DeltaProduct is evaluated
                        on in the literature.

The recurrence here is CAUSAL and runs over the synthetic sequence axis, which
IS the composition axis for these tasks (unlike the real task -- see BRIEF2 §2a).

Nothing here touches data/generated/.  Nothing here is a submission.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.p2_layers import (  # noqa: E402
    DeltaProductMixer,
    MatrixScanMixer,
    PDSSMMixer,
    RMSNorm,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).resolve().parent / "p2_probe_results.jsonl"


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #


def _a5_table() -> tuple[list[tuple[int, ...]], list[int], torch.Tensor]:
    """Cayley table of A5 plus a 5-element generating alphabet."""
    perms = [p for p in itertools.permutations(range(5))]

    def sign(p):
        s = 1
        for i in range(5):
            for j in range(i + 1, 5):
                if p[i] > p[j]:
                    s = -s
        return s

    even = [p for p in perms if sign(p) == 1]
    idx = {p: i for i, p in enumerate(even)}
    n = len(even)  # 60
    table = torch.empty(n, n, dtype=torch.long)
    for a, pa in enumerate(even):
        for b, pb in enumerate(even):
            table[a, b] = idx[tuple(pa[pb[i]] for i in range(5))]
    # generating alphabet: identity + four 3-cycles / 5-cycles
    gens = [
        (0, 1, 2, 3, 4),
        (1, 2, 0, 3, 4),
        (0, 2, 3, 1, 4),
        (1, 2, 3, 4, 0),
        (0, 1, 3, 4, 2),
    ]
    alphabet = [idx[g] for g in gens]
    return even, alphabet, table


_A5 = None


def make_batch(task: str, batch: int, length: int, gen: torch.Generator):
    global _A5
    if task == "parity":
        x = torch.randint(0, 2, (batch, length), generator=gen)
        y = x.cumsum(1) % 2
        return x.to(DEV), y.to(DEV), 2, 2
    if task == "mod3":
        x = torch.randint(0, 2, (batch, length), generator=gen)
        y = x.cumsum(1) % 3
        return x.to(DEV), y.to(DEV), 2, 3
    if task == "a5":
        if _A5 is None:
            _A5 = _a5_table()
        _, alphabet, table = _A5
        a = torch.tensor(alphabet)
        sym = torch.randint(0, len(alphabet), (batch, length), generator=gen)
        elems = a[sym]
        acc = torch.zeros(batch, dtype=torch.long)
        ys = []
        for t in range(length):
            acc = table[acc, elems[:, t]]
            ys.append(acc.clone())
        y = torch.stack(ys, dim=1)
        return sym.to(DEV), y.to(DEV), len(alphabet), table.shape[0]
    raise ValueError(task)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


class ProbeModel(nn.Module):
    def __init__(self, vocab: int, n_classes: int, mixer: nn.Module, d: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.n1 = RMSNorm(d)
        self.mixer = mixer
        self.n2 = RMSNorm(d)
        self.up = nn.Linear(d, 2 * d)
        self.down = nn.Linear(2 * d, d)
        self.nf = RMSNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        h = self.emb(x)
        h = h + self.mixer(self.n1(h))
        h = h + self.down(F.gelu(self.up(self.n2(h))))
        return self.head(self.nf(h))


def build(cfg: dict, vocab: int, n_classes: int) -> ProbeModel:
    d = cfg["d_model"]
    if cfg["arch"] == "delta":
        mixer = DeltaProductMixer(
            d,
            n_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            n_h=cfg["n_h"],
            eig_range=cfg["eig_range"],
            bidirectional=False,
        )
    elif cfg["arch"] == "pdssm":
        mixer = PDSSMMixer(
            d,
            n_heads=cfg["n_heads"],
            state_size=cfg["state_size"],
            tau=cfg["tau"],
            ste=cfg["ste"],
            bidirectional=False,
        )
    elif cfg["arch"] == "matscan":
        mixer = MatrixScanMixer(
            d,
            n_heads=cfg["n_heads"],
            state_size=cfg["state_size"],
            bidirectional=False,
        )
    else:
        raise ValueError(cfg["arch"])
    return ProbeModel(vocab, n_classes, mixer, d).to(DEV)


# --------------------------------------------------------------------------- #
def run_one(cfg: dict) -> dict:
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 1000)
    eval_gen = torch.Generator().manual_seed(999_000)  # held out by construction
    x, y, vocab, n_classes = make_batch(cfg["task"], 8, cfg["length"], gen)
    model = build(cfg, vocab, n_classes)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    t0 = time.time()
    for step in range(1, cfg["steps"] + 1):
        x, y, _, _ = make_batch(cfg["task"], cfg["batch"], cfg["length"], gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, n_classes), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    train_s = time.time() - t0

    model.eval()
    with torch.no_grad():
        # in-distribution length
        xe, ye, _, _ = make_batch(cfg["task"], 1024, cfg["length"], eval_gen)
        pred = model(xe).argmax(-1)
        acc = (pred == ye).float().mean().item()
        seq_acc = (pred == ye).all(-1).float().mean().item()
        last_acc = (pred[:, -1] == ye[:, -1]).float().mean().item()
        # collapse detector: output diversity at the last position
        div = len(torch.unique(pred[:, -1])) / n_classes
        # length extrapolation (2x)
        xl, yl, _, _ = make_batch(cfg["task"], 512, cfg["length"] * 2, eval_gen)
        predl = model(xl).argmax(-1)
        ext_last = (predl[:, -1] == yl[:, -1]).float().mean().item()
        # realised eigenvalue range on eval inputs
        extra = {}
        h = model.n1(model.emb(xe))
        if cfg["arch"] == "delta":
            ev = model.mixer.transition_eigenvalues(h.float())
            extra = {
                "eig_min": ev.min().item(),
                "eig_max": ev.max().item(),
                "eig_frac_neg": (ev < 0).float().mean().item(),
                "eig_p01": torch.quantile(ev, 0.01).item(),
                "eig_p99": torch.quantile(ev, 0.99).item(),
            }
        elif cfg["arch"] == "pdssm":
            extra = model.mixer.transition_stats(h.float())
    row = dict(cfg)
    row.update(
        {
            "n_classes": n_classes,
            "params": nparam,
            "final_loss": loss.item(),
            "tok_acc": acc,
            "seq_acc": seq_acc,
            "last_acc": last_acc,
            "chance": 1.0 / n_classes,
            "out_diversity": div,
            "ext2x_last_acc": ext_last,
            "train_seconds": round(train_s, 2),
        }
    )
    row.update(extra)
    return row


DEFAULT = dict(
    d_model=64,
    n_heads=1,
    head_dim=32,
    state_size=16,
    n_h=1,
    eig_range="neg",
    tau=1.0,
    ste="hard",
    lr=3e-3,
    steps=1200,
    batch=128,
    length=20,
    seed=0,
    arch="delta",
    task="parity",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="all")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--tasks", nargs="+", default=["parity", "mod3", "a5"])
    args = ap.parse_args()

    cfgs: list[dict] = []

    def add(**kw):
        c = dict(DEFAULT)
        c.update(kw)
        c["steps"] = args.steps
        cfgs.append(c)

    for task in args.tasks:
        for seed in args.seeds:
            if args.grid in ("all", "eig"):
                # A. eigenvalue range: the §3.3 trap, measured
                for eig in ("pos", "neg"):
                    add(task=task, seed=seed, arch="delta", n_h=1, eig_range=eig,
                        label="A_eig")
            if args.grid in ("all", "nh"):
                # B. the n_h sweep -- the cleanest single-axis experiment
                for n_h in (1, 2, 3, 4):
                    add(task=task, seed=seed, arch="delta", n_h=n_h,
                        eig_range="neg", label="B_nh")
                # B'. same sweep under the WRONG eigenvalue range (control)
                for n_h in (1, 2, 4):
                    add(task=task, seed=seed, arch="delta", n_h=n_h,
                        eig_range="pos", label="Bp_nh_pos")
            if args.grid in ("all", "tau"):
                # C. PD-SSM temperature sweep + the soft control
                for tau in (0.1, 0.3, 1.0, 3.0, 10.0):
                    add(task=task, seed=seed, arch="pdssm", tau=tau, ste="hard",
                        label="C_tau")
                add(task=task, seed=seed, arch="pdssm", tau=1.0, ste="none",
                    label="C_soft")
            if args.grid in ("all", "base"):
                add(task=task, seed=seed, arch="matscan", label="D_matscan")

    print(f"{len(cfgs)} configs -> {OUT}", flush=True)
    with OUT.open("a") as fh:
        for i, cfg in enumerate(cfgs, 1):
            row = run_one(cfg)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(
                f"[{i}/{len(cfgs)}] {row['label']:<11} {row['task']:<6} "
                f"{row['arch']:<8} n_h={row['n_h']} eig={row['eig_range']:<3} "
                f"tau={row['tau']:<5} ste={row['ste']:<4} s={row['seed']} | "
                f"last={row['last_acc']:.3f} seq={row['seq_acc']:.3f} "
                f"chance={row['chance']:.3f} div={row['out_diversity']:.3f} "
                f"ext2x={row['ext2x_last_acc']:.3f} ({row['train_seconds']}s)",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
