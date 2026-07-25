#!/usr/bin/env python
"""Offline learnability study -- LAB ONLY, not a submission, no dataset access.

Question this answers: given one of my architectures and *e1-sized* supervision,
can gradient descent find a solution that generalises to held-out x at all?  If
not even in a clean, self-generated setting, the architecture (or the data
budget) is the blocker and no amount of evaluator-side tuning will help.

Everything here is synthesised from the *public generator spec*
(`data/squaring_mod.py`: prompt = `[N] digits(N) [X] digits(x) [T] digits(T)`,
target = tail-aligned digits of `x^(2^T) mod N`).  Nothing under
`data/generated/` is opened.  The modulus and the split sizes are passed on the
command line; they are public configuration, not data.

  python lab/probe_learnability.py \
      --submission submissions/group-rotation/gp_k8h16/submission.py \
      --modulus 323 --p 17 --train 250 --steps 20000
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import torch
import torch.nn.functional as F

DIGIT_OFFSET = 7
TOK_N, TOK_X, TOK_T = 2, 3, 4


def number_tokens(value: int) -> list[int]:
    return [DIGIT_OFFSET + int(c) for c in str(value)]


def build_prompt(modulus: int, x: int, time_steps: int) -> list[int]:
    return (
        [TOK_N]
        + number_tokens(modulus)
        + [TOK_X]
        + number_tokens(x)
        + [TOK_T]
        + number_tokens(time_steps)
    )


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("gr_probe_arch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_tensors(rows, max_len, device):
    n = len(rows)
    ids = torch.zeros(n, max_len, dtype=torch.long)
    mask = torch.zeros(n, max_len, dtype=torch.bool)
    labels = torch.full((n, max_len), -100, dtype=torch.long)
    for i, (prompt, answer) in enumerate(rows):
        ids[i, : len(prompt)] = torch.tensor(prompt)
        mask[i, : len(prompt)] = True
        tail = number_tokens(answer)
        labels[i, len(prompt) - len(tail) : len(prompt)] = torch.tensor(tail)
    return ids.to(device), mask.to(device), labels.to(device)


def exact_accuracy(logits, labels):
    pred = logits.argmax(-1)
    valid = labels != -100
    ok = ((pred == labels) | ~valid).all(dim=1)
    return ok.float().mean().item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--modulus", type=int, default=323)
    ap.add_argument("--p", type=int, default=17)
    ap.add_argument("--train-x", type=int, default=250, help="distinct x seen in training")
    ap.add_argument("--time-steps", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--eval-time-steps", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log-every", type=int, default=1000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    modulus, p = args.modulus, args.p
    q = modulus // p
    phi = (p - 1) * (q - 1)
    units = [x for x in range(1, modulus) if math.gcd(x, modulus) == 1]
    perm = torch.randperm(len(units), generator=torch.Generator().manual_seed(args.seed))
    train_x = [units[i] for i in perm[: args.train_x].tolist()]
    held_x = [units[i] for i in perm[args.train_x :].tolist()]
    print(
        f"modulus={modulus} units={len(units)} train_x={len(train_x)} held_x={len(held_x)}"
    )

    def rows_for(xs, ts_list):
        out = []
        for t in ts_list:
            for x in xs:
                y = pow(x, pow(2, t, phi), modulus)
                out.append((build_prompt(modulus, x, t), y))
        return out

    train_rows = rows_for(train_x, args.time_steps)
    max_len = max(
        len(r[0])
        for r in train_rows + rows_for(held_x, args.eval_time_steps)
    )

    module = load_module(Path(args.submission).resolve())
    spec = module.ModelSpec(
        vocab_size=17, max_seq_len=max_len, maximum_model_state_elements=500_000_000
    )
    device = torch.device(args.device)
    model = module.build_model(spec).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"params={n_params:,}  max_seq_len={max_len}")

    ids, mask, labels = make_tensors(train_rows, max_len, device)
    evals = {}
    for t in args.eval_time_steps:
        evals[("held", t)] = make_tensors(rows_for(held_x, [t]), max_len, device)
        evals[("seen", t)] = make_tensors(rows_for(train_x[:64], [t]), max_len, device)

    decay = [p_ for p_ in model.parameters() if p_.ndim >= 2]
    no_decay = [p_ for p_ in model.parameters() if p_.ndim < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.wd},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    for step in range(1, args.steps + 1):
        model.train()
        logits, _ = model(ids, mask)
        valid = labels != -100
        loss = F.cross_entropy(logits[valid].float(), labels[valid])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0:
            model.eval()
            with torch.no_grad():
                train_acc = exact_accuracy(model(ids, mask)[0], labels)
                parts = []
                for (kind, t), (a, b, c) in evals.items():
                    parts.append(f"{kind}T{t}={exact_accuracy(model(a, b)[0], c):.3f}")
            print(
                f"step={step:>7} loss={loss.item():.5f} train_exact={train_acc:.3f} "
                + " ".join(parts),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
