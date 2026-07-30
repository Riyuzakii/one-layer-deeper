#!/usr/bin/env python
"""Aggregate `probe_rnn.py` rows into the markdown tables the report needs.

Always prints the whole metric row (BRIEF2 §6.3): train and held-out exact accuracy
side by side, plus the collapse detector (`div` = distinct predicted answers / held-out
examples, `top` = share of held-out examples given the single most common answer).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(paths):
    rows = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def agg(values):
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f}±{statistics.pstdev(values):.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=["lab/probe_rnn.jsonl"])
    ap.add_argument("--group", nargs="+", default=["tag", "dataset", "params"])
    args = ap.parse_args()

    rows = load(args.files)
    buckets = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k) for k in args.group)
        buckets[key].append(row)

    header = " | ".join(args.group)
    print(f"| {header} | n | steps | train_exact | held_exact | div | top_share "
          f"| held_ce |")
    print("|" + "---|" * (len(args.group) + 7))
    for key in sorted(buckets, key=lambda k: tuple(str(x) for x in k)):
        group = buckets[key]
        cells = " | ".join(str(x) for x in key)
        print(
            f"| {cells} | {len(group)} | {group[0]['steps']} "
            f"| {agg([r['train_exact'] for r in group])} "
            f"| {agg([r['held_exact'] for r in group])} "
            f"| {agg([r['held_diversity'] for r in group])} "
            f"| {agg([r['held_top_share'] for r in group])} "
            f"| {agg([r['held_ce'] for r in group])} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
