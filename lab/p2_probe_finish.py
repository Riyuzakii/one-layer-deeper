#!/usr/bin/env python
"""Re-run the 3 probe cells lost when the main probe process was killed at 93/96:
seed-1 A5 cells for tau=10, the soft control, and the matscan reference."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.p2_probe import DEFAULT, OUT, run_one  # noqa: E402

cfgs = []
c = dict(DEFAULT); c.update(task="a5", arch="pdssm", tau=10.0, ste="hard",
                            seed=1, steps=1500, label="C_tau"); cfgs.append(c)
c = dict(DEFAULT); c.update(task="a5", arch="pdssm", tau=1.0, ste="none",
                            seed=1, steps=1500, label="C_soft"); cfgs.append(c)
c = dict(DEFAULT); c.update(task="a5", arch="matscan", seed=1, steps=1500,
                            label="D_matscan"); cfgs.append(c)
with OUT.open("a") as fh:
    for i, cfg in enumerate(cfgs, 1):
        r = run_one(cfg)
        fh.write(json.dumps(r) + "\n"); fh.flush()
        print(f"[{i}/{len(cfgs)}] {r['label']:<11} {r['arch']:<8} tau={r['tau']} "
              f"ste={r['ste']:<4} s={r['seed']} last={r['last_acc']:.3f} "
              f"seq={r['seq_acc']:.3f} chance={r['chance']:.3f} "
              f"div={r['out_diversity']:.3f} ext2x={r['ext2x_last_acc']:.3f}", flush=True)
