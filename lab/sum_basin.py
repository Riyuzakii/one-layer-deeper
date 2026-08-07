#!/usr/bin/env python
"""Tabulate the repair-basin runs written by `lab/probe_o1.py --corrupt`."""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("corrupted") is not None or "repaired" in r:
                rows.append(r)
    return rows


def main(paths):
    final = {}
    for p in paths:
        for r in load(p):
            key = r["tag"]
            if key not in final or r["step"] >= final[key]["step"]:
                final[key] = r
    # --- headline: repair vs k -------------------------------------------- #
    by_k = defaultdict(list)
    for tag, r in sorted(final.items()):
        a = r.get("args", {})
        k = a.get("corrupt")
        mode = a.get("corrupt_mode", "uniform")
        recip = a.get("recip", "?")
        lr = a.get("lr", None)
        by_k[(recip, mode, k, lr == 0)].append(r)

    print(f"{'recip':7} {'mode':10} {'k':>4} {'lr0':>4} | "
          f"{'repaired':>10} {'exercised':>10} | {'tbl':>6} {'tr_hard':>8} {'he_hard':>8} {'he_div':>7}")
    for (recip, mode, k, lr0), rs in sorted(by_k.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] or 0)):
        rep = f"{sum(r['repaired'] for r in rs)}"
        den = sum(r["corrupted"] for r in rs)
        nai = f"{sum(r.get('repaired_live', 0) for r in rs)}"
        nden = sum(r.get("live", 0) for r in rs)
        print(f"{recip:7} {mode:10} {k:>4} {str(lr0):>4} | {rep:>5}/{den:<4} {nai:>5}/{nden:<4} | "
              f"{min(r['tbl_all'] for r in rs):.3f}-{max(r['tbl_all'] for r in rs):.3f} "
              f"{min(r['train_exact_hard'] for r in rs):.3f}-{max(r['train_exact_hard'] for r in rs):.3f} "
              f"{min(r['held_exact_hard'] for r in rs):.3f}-{max(r['held_exact_hard'] for r in rs):.3f} "
              f"{min(r['held_div'] for r in rs):.2f}-{max(r['held_div'] for r in rs):.2f}"
              f"  n={len(rs)}")

    # --- the new measurement: repair resolved by learned-op depth ---------- #
    print("\nrepair by LEARNED-OP DEPTH from the loss (pooled over seeds)")
    pooled = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for tag, r in final.items():
        a = r.get("args", {})
        grp = (a.get("recip", "?"), a.get("corrupt_mode", "uniform"),
               a.get("corrupt"), a.get("lr") == 0)
        for d, (ok, tot) in r.get("by_depth_live", r.get("by_depth", {})).items():
            pooled[grp][int(d)][0] += ok
            pooled[grp][int(d)][1] += tot
    for grp, d in sorted(pooled.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] or 0)):
        cells = " ".join(f"d{k}:{v[0]}/{v[1]}" for k, v in sorted(d.items()))
        tot = [sum(v[0] for v in d.values()), sum(v[1] for v in d.values())]
        print(f"  recip={grp[0]:7} mode={grp[1]:10} k={grp[2]:<4} lr0={str(grp[3]):5} "
              f"{cells}   all={tot[0]}/{tot[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["lab/runs/basin.jsonl"]))
