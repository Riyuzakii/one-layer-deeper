#!/usr/bin/env python
"""How much of the corruptible table surface does the training set exercise?

A cell no example touches receives no gradient and cannot be repaired at any
conditioning, so the repair basin has to be read against this.
"""
import math, random, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from o1reduce import O1ReduceALU, int_to_digits, true_mu
from probe_monoid import sample_semiprime

DEV = torch.device("cuda:0")


def report(tag, moduli, S, n_x=250):
    rng = random.Random(0)
    pairs = []
    for m in moduli:
        xs = []
        while len(xs) < n_x:
            v = rng.randrange(2, m)
            if math.gcd(v, m) == 1:
                xs.append(v)
        pairs += [(m, x) for x in xs]
    model = O1ReduceALU(S, recip="oracle").to(DEV)
    model.construct_()
    x = int_to_digits([p[1] for p in pairs], model.L, DEV)
    n = int_to_digits([p[0] for p in pairs], model.L, DEV)
    mu = true_mu([p[0] for p in pairs], S, model.Lmu, DEV)
    u = model.usage(x, n, mu)
    print(f"[{tag}] S={S} rows={len(pairs)}")
    for name, p, shape, dim, b in model._cells_logit():
        m = u[name]
        live = int((m > 1e-6).sum())
        top = float(m.max())
        print(f"   {name:14} rows={shape[0]:>4}  exercised={live:>4} "
              f"({live / shape[0]:.2f})  max_row_mass={top:.3f}")


report("e1-323", [323], 3)
report("hf1", sum(([sample_semiprime(b, random.Random(7 + b)) for _ in range(8)]
                   for b in (16, 18, 20)), []), 7, n_x=100)
