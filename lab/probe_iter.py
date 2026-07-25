#!/usr/bin/env python
"""LAB ONLY: does *iterating* one shared squaring block pin the frequency?

probe_step.py showed the blocker: a phase-based squaring map generalises to
held-out x only if its frequencies are commensurate with N to ~1e-5, and the
single-step training loss is flat along that direction.

This probe tests the one mechanism inside the group-rotation family that could
supply the missing constraint.  If the SAME block is applied T times, with the
intermediate value decoded to digits and re-encoded, then

    decode(phase(v^2))  must round-trip,

and a round-trip is only consistent when the phase is genuinely periodic mod N.
The T=1,2,3 rows of e1 would then not merely be extra data -- they would be the
constraint that fixes the scale.

Self-generated data only; nothing under data/generated/ is opened.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from torch import nn


class SquaringBlock(nn.Module):
    """One shared step: soft digits -> pairwise phase -> soft digits."""

    def __init__(self, slots: int, n_freq: int, n_harm: int):
        super().__init__()
        self.slots = slots
        self.pairs = [(i, j) for i in range(slots) for j in range(i, slots)]
        self.table = nn.Parameter(torch.randn(len(self.pairs), 10, 10, n_freq) * 0.5)
        self.n_harm = n_harm
        feat = 2 * n_harm * n_freq
        self.readout = nn.Parameter(torch.randn(slots, feat, 10) * feat**-0.5)
        self.bias = nn.Parameter(torch.zeros(slots, 10))

    def forward(self, s):
        theta = 0.0
        for n, (i, j) in enumerate(self.pairs):
            theta = theta + torch.einsum("ba,bc,acf->bf", s[:, i], s[:, j], self.table[n])
        parts = []
        for h in range(1, self.n_harm + 1):
            parts.append(torch.cos(theta * h))
            parts.append(torch.sin(theta * h))
        feats = torch.cat(parts, dim=-1)
        return torch.einsum("bf,ifd->bid", feats, self.readout) + self.bias


def digits_of(value: int, slots: int) -> list[int]:
    out = []
    for _ in range(slots):
        out.append(value % 10)
        value //= 10
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulus", type=int, default=323)
    ap.add_argument("--p", type=int, default=17)
    ap.add_argument("--train-x", type=int, default=250)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--freqs", type=int, default=32)
    ap.add_argument("--harm", type=int, default=8)
    ap.add_argument("--time-steps", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--eval-time-steps", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--temp", type=float, default=1.0, help="softmax temperature between steps")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    modulus, p = args.modulus, args.p
    q = modulus // p
    phi = (p - 1) * (q - 1)
    units = [x for x in range(1, modulus) if math.gcd(x, modulus) == 1]
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(units), generator=g).tolist()
    train_x = [units[i] for i in perm[: args.train_x]]
    held_x = [units[i] for i in perm[args.train_x :]]
    device = torch.device(args.device)

    def onehot(xs):
        out = torch.zeros(len(xs), args.slots, 10)
        for r, x in enumerate(xs):
            for i, d in enumerate(digits_of(x, args.slots)):
                out[r, i, d] = 1.0
        return out.to(device)

    def targets(xs, t):
        out = torch.zeros(len(xs), args.slots, dtype=torch.long)
        for r, x in enumerate(xs):
            y = pow(x, pow(2, t, phi), modulus)
            for i, d in enumerate(digits_of(y, args.slots)):
                out[r, i] = d
        return out.to(device)

    xin, hin = onehot(train_x), onehot(held_x)
    xtg = {t: targets(train_x, t) for t in args.time_steps}
    htg = {t: targets(held_x, t) for t in set(args.time_steps) | set(args.eval_time_steps)}
    stg = {t: targets(train_x, t) for t in set(args.time_steps) | set(args.eval_time_steps)}

    block = SquaringBlock(args.slots, args.freqs, args.harm).to(device)
    print(
        f"modulus={modulus} train={len(train_x)} held={len(held_x)} "
        f"params={sum(p_.numel() for p_ in block.parameters()):,} "
        f"train_T={args.time_steps}"
    )
    opt = torch.optim.AdamW(block.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.95))

    def rollout(s0, max_t):
        outs = {}
        s = s0
        for t in range(1, max_t + 1):
            logits = block(s)
            outs[t] = logits
            s = F.softmax(logits / args.temp, dim=-1)
        return outs

    max_train_t = max(args.time_steps)
    max_eval_t = max(max(args.eval_time_steps), max_train_t)

    for step in range(1, args.steps + 1):
        outs = rollout(xin, max_train_t)
        loss = sum(
            F.cross_entropy(outs[t].reshape(-1, 10), xtg[t].reshape(-1))
            for t in args.time_steps
        ) / len(args.time_steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                ho = rollout(hin, max_eval_t)
                so = rollout(xin, max_eval_t)
                parts = []
                for t in args.eval_time_steps:
                    ha = (ho[t].argmax(-1) == htg[t]).all(1).float().mean().item()
                    sa = (so[t].argmax(-1) == stg[t]).all(1).float().mean().item()
                    parts.append(f"T{t}:seen={sa:.3f},held={ha:.3f}")
            print(f"step={step:>7} loss={loss.item():.5f} " + " ".join(parts), flush=True)

    # --- mechanistic check: did the block learn mod-N structure? ---
    # (N-v)^2 = v^2 mod N, so a block that genuinely computes "square mod N" must
    # give the SAME answer for v and N-v.  This is a property of the model's own
    # activations on self-generated inputs; it needs no labels.
    all_units = [v for v in units if math.gcd(modulus - v, modulus) == 1]
    a = onehot(all_units)
    b = onehot([modulus - v for v in all_units])
    with torch.no_grad():
        pa = block(a).argmax(-1)
        pb = block(b).argmax(-1)
        agree = (pa == pb).all(1).float().mean().item()
        tgt_all = targets(all_units, 1)
        acc_all = (pa == tgt_all).all(1).float().mean().item()
    print(
        f"[mech] pred(v) == pred(N-v) on {len(all_units)} units: {agree:.3f}   "
        f"(1.0 => the block respects mod-N structure; ~0.0 => it is a value lookup)   "
        f"exact over all units: {acc_all:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
