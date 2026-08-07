#!/usr/bin/env python
"""Summarise the repair-basin ladders into the comparison table."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    p = os.path.join(HERE, path)
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def kof(row):
    if "k" in row:
        return row["k"]
    argv = row.get("argv", [])
    if "--repair" in argv:
        return int(argv[argv.index("--repair") + 1])
    return None


def ncells(row):
    if "n_cells" in row:
        return row["n_cells"]
    return None


def table(rows, label, cells_hint=None):
    by = defaultdict(list)
    for r in rows:
        k = kof(r)
        if k is None:
            continue
        by[k].append(r)
    print(f"\n### {label}")
    print("| k | reps | reached 1.000 | mean train_exact_hard | "
          "mean held_exact_hard | mean cell_agree |")
    print("|---|---|---|---|---|---|")
    for k in sorted(by):
        rs = by[k]
        n = len(rs)
        hit = sum(1 for r in rs if r["train_exact_hard"] >= 0.99999)
        te = sum(r["train_exact_hard"] for r in rs) / n
        he = sum(r["held_exact_hard"] for r in rs) / n
        def _ca(r):
            v = r["cell_agree"]
            return v["all"] if isinstance(v, dict) else v
        ca = sum(_ca(r) for r in rs) / n
        nc = ncells(rs[0]) or cells_hint or "?"
        print(f"| {k} of {nc} | {n} | **{hit}/{n}** | {te:.3f} | {he:.3f} | "
              f"{ca:.3f} |")


def main():
    add = load("basin_runs.jsonl") + load("basin2_runs.jsonl")
    dal = load("basin_runs_dalu.jsonl") + load("basin2_runs_dalu.jsonl")
    groups = defaultdict(list)
    for r in add:
        key = (r.get("pick", "learned"), tuple(r.get("ties", [])))
        groups[key].append(r)
    for (pick, ties), rs in sorted(groups.items()):
        table(rs, f"ADD-ONLY pick={pick} ties={','.join(ties) or 'none'}")
    dgrp = defaultdict(list)
    for r in dal:
        argv = r.get("argv", [])
        ties = argv[argv.index("--tie") + 1] if "--tie" in argv else ""
        dgrp[ties].append(r)
    for ties, rs in sorted(dgrp.items()):
        table(rs, f"DigitALU (CONTROL) ties={ties or 'none'}",
              337 if ties else 1007)
    return 0


if __name__ == "__main__":
    sys.exit(main())
