#!/usr/bin/env python
"""LAB ONLY -- the digit-compositional readout at SAMPLED moduli (the e5 / OOD-N
regime), and the parameter-count argument stated as a measurement.

`probe_alu.py --construct` shows held-out 1.000 at four fixed moduli with an
identical 6,817 parameters.  The reason is that no tensor in the readout has an
index that ranges over Z_N -- the modulus enters only as *input digits*.  If
that is true, the SAME parameter vector, with no retraining, must also be exact
on moduli it has never seen, and on operands it has never seen, at any size the
slot count allows.  That is what this script checks, and it is the property the
OOD-N tie-break needs.

Self-generated values only; nothing under data/generated/ is opened.
"""

from __future__ import annotations

import argparse
import math
import random

import torch

from probe_alu import DigitALU, digits_le


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def sample_semiprime(bits: int, rng: random.Random) -> int:
    half = bits // 2
    while True:
        p = rng.randrange(1 << (half - 1), 1 << half) | 1
        q = rng.randrange(1 << (bits - half - 1), 1 << (bits - half)) | 1
        if p != q and is_prime(p) and is_prime(q):
            n = p * q
            if n.bit_length() == bits:
                return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[10, 11],
                    help="modulus bit widths (e5 uses 10 and 11)")
    ap.add_argument("--moduli", type=int, default=8, help="moduli per bit width")
    ap.add_argument("--x-per-modulus", type=int, default=400)
    ap.add_argument("--slots", type=int, default=0, help="0 = from the widest N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = torch.device(args.device)
    mods = []
    for b in args.bits:
        seen = set()
        while len(seen) < args.moduli:
            seen.add(sample_semiprime(b, rng))
        mods += sorted(seen)
    S = args.slots or max(len(str(m)) for m in mods)
    model = DigitALU(S).to(device)
    model.construct()
    model.eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"one CONSTRUCTED parameter vector, params={n_par:,}, slots={S}, "
          f"applied to {len(mods)} unseen moduli\n")
    print(f"{'N':>7} {'bits':>4} {'units':>7} {'tested':>7} {'exact':>7}")
    print("-" * 38)
    tot_ok = tot_n = 0
    for m in mods:
        units = [x for x in range(1, m) if math.gcd(x, m) == 1]
        xs = units if len(units) <= args.x_per_modulus else \
            rng.sample(units, args.x_per_modulus)
        inp = torch.zeros(len(xs), S, 10)
        tgt = torch.zeros(len(xs), S, dtype=torch.long)
        for r, x in enumerate(xs):
            for i, d in enumerate(digits_le(x, S)):
                inp[r, i, d] = 1.0
            for i, d in enumerate(digits_le((x * x) % m, S)):
                tgt[r, i] = d
        nd = torch.zeros(S + 1, 10)
        for i, d in enumerate(digits_le(m, S + 1)):
            nd[i, d] = 1.0
        with torch.no_grad():
            lg = model(inp.to(device), nd.to(device))
        ok = (lg.argmax(-1) == tgt.to(device)).all(dim=1).float().mean().item()
        tot_ok += ok * len(xs)
        tot_n += len(xs)
        print(f"{m:>7} {m.bit_length():>4} {len(units):>7} {len(xs):>7} {ok:>7.3f}")
    print("-" * 38)
    print(f"{'TOTAL':>7} {'':>4} {'':>7} {tot_n:>7} {tot_ok/tot_n:>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
