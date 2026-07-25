#!/usr/bin/env python
"""Tabulate tied-recurrence runs out of lab/archive.jsonl as markdown.

Usage:  python lab/tied_table.py T2-iter [T3-extrap ...]
        python lab/tied_table.py --all
Reads only the archive (never generated data).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parent / "archive.jsonl"
LADDER = [1, 2, 4, 8, 16, 32, 64]


def fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}".rstrip("0").rstrip(".") if v else "0"


def main() -> int:
    args = sys.argv[1:] or ["--all"]
    rows = [json.loads(line) for line in ARCHIVE.read_text().splitlines() if line.strip()]
    if "--all" not in args:
        rows = [r for r in rows if r.get("tag") in args]
    rows = [r for r in rows if r.get("status") == "ok"]
    if not rows:
        print("no matching rows")
        return 1
    head = (
        "| run | manifest | steps | MAX_T | "
        + " | ".join(f"T={t}" for t in LADDER)
        + " | test | mean | eval_s |"
    )
    print(head)
    print("|" + "---|" * (len(LADDER) + 8))
    for r in rows:
        name = Path(r["submission"]).parent.name
        rung = {int(k): v for k, v in (r.get("rung_exact_accuracy") or {}).items()}
        steps = r.get("completed_training_steps") or []
        ev = r.get("evaluation_seconds") or []
        splits = r.get("split_exact_accuracy") or {}
        print(
            f"| `{name}` | {r['manifest']} | {steps[0] if steps else '-'} | "
            f"{r.get('max_certified_t')} | "
            + " | ".join(fmt(rung.get(t)) for t in LADDER)
            + f" | {fmt(splits.get('test'))} | {fmt(r.get('mean_exact_accuracy'))} | "
            f"{round(ev[0], 1) if ev else '-'} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
