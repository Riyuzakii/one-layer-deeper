#!/usr/bin/env python
"""DIAGNOSTIC addendum: give PD-SSM a state size large enough for A5.

PD-SSM's guarantee is "any N-state FSA with ONE layer of dimension N".  A5 has
60 states, so the main probe's N=16 cells are under-provisioned by construction
and their A5 null says nothing about PD-SSM.  This runs N=64 (>= 60) so the
comparison against DeltaProduct is fair.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.p2_probe import DEFAULT, OUT, run_one  # noqa: E402

cfgs = []
for seed in (0, 1):
    for n in (64, 32):
        c = dict(DEFAULT)
        c.update(task="a5", arch="pdssm", state_size=n, tau=1.0, ste="hard",
                 seed=seed, steps=1500, label=f"F_pdssm_N{n}")
        cfgs.append(c)
    for n in (64,):
        c = dict(DEFAULT)
        c.update(task="a5", arch="matscan", state_size=n, seed=seed,
                 steps=1500, label=f"F_matscan_N{n}")
        cfgs.append(c)
with OUT.open("a") as fh:
    for i, cfg in enumerate(cfgs, 1):
        r = run_one(cfg)
        fh.write(json.dumps(r) + "\n"); fh.flush()
        print(f"[{i}/{len(cfgs)}] {r['label']:<14} N={cfg['state_size']} s={r['seed']} "
              f"last={r['last_acc']:.3f} seq={r['seq_acc']:.3f} chance={r['chance']:.3f} "
              f"div={r['out_diversity']:.3f} ext2x={r['ext2x_last_acc']:.3f} "
              f"({r['train_seconds']}s)", flush=True)
