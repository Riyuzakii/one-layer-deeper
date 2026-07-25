#!/usr/bin/env python
"""LAB ONLY -- the closed-form counterpart of the coverage ceiling, for TWO
readout families.

`explore/group-rotation` §7 established, and verified against a closed-form
combinatorial prediction at three moduli, that the oracle held-out accuracy of a
readout that is *an arbitrary function of the residue* is

    P[ x^2 mod N  already occurs as  x_train^2 mod N ]

  N=323 -> 1.000,  N=899 -> 0.614,  N=2021 -> 0.337,  N=10403 -> 0.072.

This script reproduces that number and computes the analogous quantity for a
readout that is *compositional in the digits*: one whose learned objects are
tables indexed by a small, place-independent local alphabet (a digit pair, a
digit + carry, a digit + borrow, a comparator state) rather than by the value.

The reference schedule below is a LAB ANALYSIS routine -- it is never part of a
forward pass and no submission imports it.  Its only role is to enumerate which
local table entries each x exercises, exactly as §7 enumerated which residues
each x exercises.

Self-generated values only; nothing under data/generated/ is opened.
"""

from __future__ import annotations

import argparse
import math
import random

import torch


def digits_le(value: int, slots: int, base: int = 10) -> list[int]:
    """Little-endian digits, slot 0 = units."""
    out = []
    for _ in range(slots):
        out.append(value % base)
        value //= base
    return out


def split(modulus: int, train_x: int, seed: int,
          held_sample: int = 0) -> tuple[list[int], list[int]]:
    """Identical split to lab/probe_step.py so the numbers are comparable.

    Above ~1e6 the unit group cannot be enumerated, so `held_sample` draws the
    held-out cohort by rejection instead.  Coverage is a per-example question,
    so a sample estimates it unbiasedly."""
    rng = random.Random(seed)
    if held_sample:
        seen: set[int] = set()
        while len(seen) < train_x + held_sample:
            v = rng.randrange(1, modulus)
            if math.gcd(v, modulus) == 1:
                seen.add(v)
        allx = sorted(seen)
        rng.shuffle(allx)
        return allx[:train_x], allx[train_x:]
    units = [x for x in range(1, modulus) if math.gcd(x, modulus) == 1]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(units), generator=g).tolist()
    return [units[i] for i in perm[:train_x]], [units[i] for i in perm[train_x:]]


# --------------------------------------------------------------------------
# atom enumeration for a place-shared (digit-compositional) pipeline
# --------------------------------------------------------------------------
#
#  y = x^2 mod N  computed as
#
#    c[k]  = sum_{i+j=k} d_i d_j                       (place-shared pair table)
#    r     = 0
#    for k = 2S-2 .. 0:
#        acc = 10*r + c[k]                             (shift + ripple add)
#        q   = acc // N                                (compare scan, q <= 10)
#        r   = acc - q*N                               (digit*multidigit, then
#                                                       subtract with borrow)
#    y = r
#
#  Every learned table in this pipeline is indexed by a tuple that does NOT
#  contain the residue: (a,b) for the pair product, (u,v,carry) for the add,
#  (q,n,carry) for the single-digit multiply, (u,v,borrow) for the subtract, and
#  (u,v,state) for the comparator.  Those are the "atoms".


def atoms_for(x: int, modulus: int, slots: int, base: int = 10) -> dict[str, set]:
    """`base` is the INTERNAL radix of the tables, not the prompt's alphabet.

    explore/alu-depth: a larger internal radix (pair adjacent decimal digits
    into a base-100 symbol) halves the slot count and so roughly quarters the
    sequential depth.  It also squares every table's index space, and this
    routine is what prices that trade: it is the same coverage question, asked
    of a base-B alphabet.
    """
    S = slots
    d = digits_le(x, S, base)
    A: dict[str, set] = {k: set() for k in
                         ("pair", "add", "mul", "sub", "cmp")}

    # stage A -- place-shared partial-product table, indexed by a digit pair
    K = 2 * S - 1
    c = [0] * K
    for i in range(S):
        for j in range(S):
            A["pair"].add((d[i], d[j]))
            c[i + j] += d[i] * d[j]

    W = S + 1  # base*r needs one slot more than r
    r = 0
    for k in range(K - 1, -1, -1):
        # ---- shift + ripple add:  acc = base*r + c[k]
        shifted = digits_le(base * r, W + 1, base)
        addend = digits_le(c[k], W + 1, base)
        carry = 0
        for m in range(W + 1):
            A["add"].add((shifted[m], addend[m], carry))
            t = shifted[m] + addend[m] + carry
            carry = t // base
        acc = base * r + c[k]

        # ---- quotient by comparison against multiples of N (digit scan,
        #      MSB->LSB, three-valued state).  q <= base because r < N.
        q = acc // modulus
        accd = digits_le(acc, W + 1, base)
        cand = digits_le(q * modulus, W + 1, base)
        state = 0  # 0 = equal-so-far, 1 = greater, 2 = less
        for m in range(W, -1, -1):
            A["cmp"].add((accd[m], cand[m], state))
            if state == 0:
                state = 0 if accd[m] == cand[m] else (1 if accd[m] > cand[m] else 2)

        # ---- q * N  (single digit x multi digit, shared cell)
        nd = digits_le(modulus, W + 1, base)
        carry = 0
        for m in range(W + 1):
            A["mul"].add((q, nd[m], carry))
            t = q * nd[m] + carry
            carry = t // base

        # ---- acc - q*N  (subtract with borrow, shared cell)
        prodd = digits_le(q * modulus, W + 1, base)
        borrow = 0
        for m in range(W + 1):
            A["sub"].add((accd[m], prodd[m], borrow))
            t = accd[m] - prodd[m] - borrow
            borrow = 1 if t < 0 else 0

        r = acc - q * modulus
    assert r == (x * x) % modulus
    return A


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["323:3", "899:3", "2021:4", "10403:5"])
    ap.add_argument("--train-x", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radix", type=int, default=10,
                    help="INTERNAL radix of the learned tables.  --radix 100 "
                         "pairs adjacent decimal digits, halving the slot "
                         "count and so the sequential depth; slots in "
                         "--configs are then base-`radix` slots.")
    ap.add_argument("--held-sample", type=int, default=0,
                    help="sample the held-out cohort instead of enumerating "
                         "(needed above ~1e6 units)")
    args = ap.parse_args()

    print(f"train_x={args.train_x} seed={args.seed} radix={args.radix}\n")
    hdr = (f"{'N':>7} {'S':>2} {'units':>6} {'held':>5} "
           f"{'residue ceiling':>15} {'digit ceiling':>13}   atoms seen / needed")
    print(hdr)
    print("-" * len(hdr))

    for cfg in args.configs:
        modulus, slots = (int(v) for v in cfg.split(":"))
        train_x, held_x = split(modulus, args.train_x, args.seed,
                                args.held_sample)

        # ---- family 1: readout = arbitrary function of the residue (§7)
        seen_res = {(x * x) % modulus for x in train_x}
        res_cov = sum((x * x) % modulus in seen_res for x in held_x) / len(held_x)

        # ---- family 2: readout = place-shared digit tables
        seen: dict[str, set] = {k: set() for k in ("pair", "add", "mul", "sub", "cmp")}
        for x in train_x:
            for k, v in atoms_for(x, modulus, slots, args.radix).items():
                seen[k] |= v
        ok = 0
        needed: dict[str, set] = {k: set() for k in seen}
        for x in held_x:
            a = atoms_for(x, modulus, slots, args.radix)
            for k, v in a.items():
                needed[k] |= v
            if all(a[k] <= seen[k] for k in a):
                ok += 1
        dig_cov = ok / len(held_x)

        detail = " ".join(
            f"{k}={len(seen[k])}/{len(seen[k] | needed[k])}" for k in
            ("pair", "add", "mul", "sub", "cmp"))
        units_n = len(train_x) + len(held_x) if not args.held_sample else -1
        print(f"{modulus:>7} {slots:>2} {units_n:>6} "
              f"{len(held_x):>5} {res_cov:>15.3f} {dig_cov:>13.3f}   {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
