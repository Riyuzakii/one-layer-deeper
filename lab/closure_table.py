#!/usr/bin/env python
"""Render lab/archive.jsonl rows for one or more tags as markdown tables.

usage: .venv/bin/python lab/closure_table.py CL1-lambda CL2-seeds ...
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ARCHIVE = Path(__file__).resolve().parent / "archive.jsonl"
LADDER = [1, 2, 4, 8, 16, 32, 64]


def rows(tags):
    out = []
    for line in ARCHIVE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("tag") in tags and row.get("status") == "ok":
            out.append(row)
    return out


def train_at(row, step):
    curve = row.get("train_curve") or []
    best = None
    for s, _loss, acc in curve:
        if s <= step:
            best = acc
    return best


def main() -> int:
    tags = sys.argv[1:]
    if not tags:
        print(__doc__)
        return 1
    data = rows(tags)
    header = (
        "| tag | cfg | manifest | MAX_T | "
        + " | ".join(f"T={t}" for t in LADDER)
        + " | test acc | test CE | ood CE | train acc | kl_c | kl_r | idm | idx |"
    )
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    for row in data:
        rung = row.get("rung_exact_accuracy") or {}
        acc = row.get("split_exact_accuracy") or {}
        ce = row.get("split_loss") or {}
        diag = (row.get("diag_curve") or [[None] * 8])[-1]
        cells = [
            row["tag"],
            row.get("note", ""),
            row["manifest"],
            str(row.get("max_certified_t", 0)),
        ]
        for t in LADDER:
            v = rung.get(str(t), rung.get(t))
            cells.append("-" if v is None else f"{v:.3f}")
        cells.append(f"{acc.get('test', float('nan')):.3f}")
        cells.append(
            "-" if ce.get("test") is None else f"{ce['test']:.3f}"
        )
        cells.append("-" if ce.get("ood") is None else f"{ce['ood']:.3f}")
        ta = train_at(row, 10**9)
        cells.append("-" if ta is None else f"{ta:.3f}")
        for i in (1, 2, 4, 7):
            v = diag[i] if len(diag) > i else None
            cells.append("-" if v is None else f"{v:.3f}")
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
