#!/usr/bin/env python
"""Collate lab/runs/grid*.jsonl into the per-rung tables the report needs."""
from __future__ import annotations

import glob
import json
import sys

LAD = [1, 2, 4, 8, 16, 32, 64]


def main() -> int:
    pats = sys.argv[1:] or ["lab/runs/grid*.jsonl"]
    rows = []
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            for ln in open(f):
                rows.append(json.loads(ln))
    by = {}
    for r in rows:
        by.setdefault(r["tag"], []).append(r)
    print(f"{'cell':26} {'N':>8} {'trainT':>10} {'seeds':>6} "
          f"{'MAX_T mix':>22} {'MAX_T arg':>22}")
    print("-" * 100)
    for tag, rs in by.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        mix = ",".join(str(r.get("max_t_mix", r["max_t"])) for r in rs)
        arg = ",".join(str(r.get("max_t_arg", "-")) for r in rs)
        print(f"{tag:26} {str(rs[0]['modulus']):>8} "
              f"{str(rs[0]['train_t']):>10} {len(rs):>6} {mix:>22} {arg:>22}")
    print()
    for tag, rs in by.items():
        print(f"### {tag}")
        for r in sorted(rs, key=lambda r: r["seed"]):
            e = r["exact"]
            a = r.get("exact_arg", {})
            print(f"  s{r['seed']} mix " + " ".join(
                f"{e[str(t)] if str(t) in e else e[t]:.3f}" for t in LAD)
                + ("  arg " + " ".join(
                    f"{a[str(t)] if str(t) in a else a[t]:.3f}" for t in LAD)
                   if a else "")
                + f"   [{r.get('diag','')}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
