#!/usr/bin/env python
"""Constructed ceiling reported PER MODULUS SIZE, never pooled.

`hard/curriculum-hf1` found that a fixed-slot ALU with the published `t <= S`
reduction schedule overflows the quotient alphabet for the *smaller* moduli in a
mixed-size dataset, so an aggregate 1.000 can hide a broken bit size.  hf1 mixes
16/18/20-bit (ID) and 17/19/21-bit (OOD-N) moduli in one training set, so every
size is checked separately here, soft and argmax-hard.

DIAGNOSTIC: uses `--construct`.
"""
import math, random, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from o1reduce import O1ReduceALU, int_to_digits, digits_to_int, true_mu
from probe_monoid import sample_semiprime

DEV = torch.device("cuda:0")
BITS = [(16, "ID"), (18, "ID"), (20, "ID"), (17, "OOD-N"), (19, "OOD-N"), (21, "OOD-N")]
S = 7            # set by the widest in-distribution modulus (20 bits = 7 digits)
N_MOD, N_X, T = 6, 64, 1
fail = []

for recip in ("div", "oracle"):
    model = O1ReduceALU(S, recip=recip).to(DEV)
    model.construct_()
    for bits, kind in BITS:
        rng = random.Random(1000 + bits)
        mods, seen = [], set()
        while len(mods) < N_MOD:
            m = sample_semiprime(bits, rng)
            if m not in seen:
                seen.add(m); mods.append(m)
        pairs = []
        for m in mods:
            xs = []
            while len(xs) < N_X:
                v = rng.randrange(2, m)
                if math.gcd(v, m) == 1:
                    xs.append(v)
            pairs += [(m, x) for x in xs]
        ns = [p[0] for p in pairs]
        x = int_to_digits([p[1] for p in pairs], model.L, DEV)
        n = int_to_digits(ns, model.L, DEV)
        if recip == "oracle":
            model.set_mu(true_mu(ns, S, model.Lmu, DEV))
        want = []
        for m, v in pairs:
            for _ in range(T):
                v = v * v % m
            want.append(v)
        line = [f"bits={bits:>2} ({kind:5}) digits={len(str(mods[0]))} mods={N_MOD} x={N_X}"]
        for hard in (False, True):
            with torch.no_grad():
                y = model(x, n, hard=hard)
            got = digits_to_int(y[:, :S]).tolist()
            ok = sum(int(a) == b for a, b in zip(got, want))
            line.append(f"{'hard' if hard else 'soft'}={ok / len(want):.3f}")
            if ok != len(want):
                fail.append((recip, bits, hard, ok, len(want)))
        print(f"  recip={recip:7} " + "  ".join(line), flush=True)

print("ALL SIZES EXACT" if not fail else f"FAILURES: {fail}")
raise SystemExit(1 if fail else 0)
