#!/usr/bin/env python
"""LAB ONLY -- constructed ceiling of the SHIPPED submission's `AddStep` on a
MIXED-modulus batch, plus the leading-zero floor of digit accuracy.

WHY.  `hard/curriculum-hf1` reports that a fixed-slot ALU on a mixed-modulus
dataset can overflow the quotient alphabet under the `t <= S` reduction
schedule, so that the constructed ceiling is NOT 1.000 across modulus sizes,
and that `redall` (reduce after every place) fixes it.  hf1 contains three ID
sizes and three OOD-N sizes in one training set, so this has to be checked on
the object that will actually be trained -- `submissions/hard-add-only`'s
`AddStep`, which takes N per example.

It also measures the constant-zero digit-accuracy floor, because a
digit-accuracy metric is fooled by leading zeros and my repair-basin profile
quotes digit accuracy.

Self-generated operands only; nothing under data/generated/ is opened.
`--construct` is a LAB DIAGNOSTIC and can never be a submission.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_addsearch import digits_le, sample_semiprime  # noqa: E402

BIG = 30.0


def load_submission(path):
    spec = importlib.util.spec_from_file_location("subm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def construct(step, sub):
    """LAB DIAGNOSTIC -- set the submission's learned tables to the truth."""
    Ca, Cb = step.Ca, step.Cb
    t = torch.full((10, 10, Ca, 10 + Ca), -BIG)
    for u in range(10):
        for v in range(10):
            for c in range(Ca):
                s = u + v + c
                t[u, v, c, s % 10] = BIG
                t[u, v, c, 10 + s // 10] = BIG
    step.Tadd.copy_(t.to(step.Tadd))
    t = torch.full((10, 10, Cb, 10 + Cb), -BIG)
    for u in range(10):
        for v in range(10):
            for c in range(Cb):
                s = u - v - c
                t[u, v, c, s % 10] = BIG
                t[u, v, c, 10 + (1 if s < 0 else 0)] = BIG
    step.Tsub.copy_(t.to(step.Tsub))
    t = torch.full((10, 10), -BIG)
    for d in range(10):
        t[d, d] = BIG
    step.Tpick.copy_(t.to(step.Tpick))
    for P_, n in ((step.zero, 10), (step.carry0, Ca), (step.borrow0, Cb)):
        v = torch.full((n,), -BIG)
        v[0] = BIG
        P_.copy_(v.to(P_))
    w = torch.zeros(1, 2 * Cb)
    w[0, 0], w[0, 1] = BIG / 2, -BIG / 2
    w[0, Cb + 0], w[0, Cb + 1] = -BIG / 2, BIG / 2
    step.sel.weight.copy_(w.to(step.sel.weight))
    step.sel.bias.zero_()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission",
                    default="submissions/hard-add-only/submission.py")
    ap.add_argument("--slots", type=int, default=7)
    ap.add_argument("--bits", default="16,17,18,19,20,21")
    ap.add_argument("--per-bits", type=int, default=3,
                    help="distinct moduli sampled per bit size")
    ap.add_argument("--x-per-modulus", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--redall", action="store_true",
                    help="reduce after EVERY place instead of t <= S")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    sub = load_submission(args.submission)
    dev = torch.device(args.device)
    S = args.slots
    step = sub.AddStep(S).to(dev)
    construct(step, sub)
    if args.redall:
        # monkey-patch the schedule for the diagnostic only
        orig = sub.AddStep.forward

        def redall_forward(self, s, nd):
            b, S_, W, Fw = s.shape[0], self.S, self.W, self.Fw
            z = F.softmax(self.zero, -1)[None].expand(b, 10)
            zr = z[:, None].expand(b, W, 10)
            hn = self.chain(zr, nd, self.needed)
            mults = torch.stack([hn[m] for m in self.needed], 1)
            xr = torch.cat([s, z[:, None].expand(b, W - S_, 10)], 1)
            hx = self.chain(zr, xr, self.xneeded)
            Xm = torch.stack([hx[m] for m in self.xneeded], 1)
            pk = F.softmax(self.Tpick, -1)
            leaves = []
            for i in range(S_):
                w = torch.einsum("bu,um->bm", s[:, i], pk)
                v = torch.einsum("bm,bmwo->bwo", w, Xm)
                cols = [z] * Fw
                for j in range(W):
                    if i + j < Fw:
                        cols[i + j] = v[:, j]
                leaves.append(torch.stack(cols, 1))
            Pr = self.tree_sum(leaves)
            r = zr
            for t in range(Fw - 1, -1, -1):
                r = torch.cat([Pr[:, t:t + 1], r[:, :W - 1]], dim=1)
                r = self.quot_reduce(r, mults)      # EVERY place
            return torch.log(r[:, :S_] + 1e-9)
        sub.AddStep.forward = redall_forward
        del orig

    rng = random.Random(args.seed)
    rows = []
    per_bits = {}
    for bits in [int(b) for b in args.bits.split(",")]:
        for k in range(args.per_bits):
            N, p, q = sample_semiprime(bits, rng)
            units = [x for x in range(1, N) if math.gcd(x, N) == 1]
            g = torch.Generator().manual_seed(1000 * bits + k)
            perm = torch.randperm(len(units), generator=g).tolist()
            xs = [units[i] for i in perm[: args.x_per_modulus]]
            for x in xs:
                rows.append((bits, N, x, (x * x) % N))
            per_bits.setdefault(bits, []).append(N)
    rng2 = random.Random(args.seed + 1)
    rng2.shuffle(rows)                       # a genuinely MIXED batch

    W = S + 1
    inp = torch.zeros(len(rows), S, 10)
    nd = torch.zeros(len(rows), W, 10)
    tgt = torch.zeros(len(rows), S, dtype=torch.long)
    for i, (_, N, x, y) in enumerate(rows):
        for j, d in enumerate(digits_le(x, S)):
            inp[i, j, d] = 1.0
        for j, d in enumerate(digits_le(N, W)):
            nd[i, j, d] = 1.0
        for j, d in enumerate(digits_le(y, S)):
            tgt[i, j] = d
    inp, nd, tgt = inp.to(dev), nd.to(dev), tgt.to(dev)

    tag = "redall" if args.redall else "t<=S"
    print(f"[mixed:{tag}] S={S} rows={len(rows)} moduli="
          f"{ {b: len(v) for b, v in per_bits.items()} } "
          f"sizes={sorted(per_bits)}", flush=True)

    ok = torch.zeros(len(rows), dtype=torch.bool, device=dev)
    dig = torch.zeros(len(rows), S, dtype=torch.bool, device=dev)
    chunk = 512
    with torch.no_grad():
        for i in range(0, len(rows), chunk):
            lg = step(inp[i:i + chunk], nd[i:i + chunk])
            pr = lg.argmax(-1)
            hit = pr == tgt[i:i + chunk]
            dig[i:i + chunk] = hit
            ok[i:i + chunk] = hit.all(-1)

    bits_arr = torch.tensor([r[0] for r in rows], device=dev)
    print(f"[mixed:{tag}] MIXED-BATCH constructed exact={ok.float().mean():.4f} "
          f"digit={dig.float().mean():.4f}", flush=True)
    for b in sorted(per_bits):
        m = bits_arr == b
        # constant-zero predictor floor on this cohort
        zero_floor = (tgt[m] == 0).float().mean().item()
        print(f"[mixed:{tag}]   {b:2d}-bit  n={int(m.sum())}  "
              f"exact={ok[m].float().mean():.4f}  "
              f"digit={dig[m].float().mean():.4f}  "
              f"const-zero digit floor={zero_floor:.4f}", flush=True)
    zf = (tgt == 0).float().mean().item()
    print(f"[mixed:{tag}] CONSTANT-ZERO PREDICTOR digit floor (all) = "
          f"{zf:.4f}  (exact floor = "
          f"{(tgt == 0).all(-1).float().mean().item():.4f})", flush=True)
    return 0 if ok.float().mean().item() == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
