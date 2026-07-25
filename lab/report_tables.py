#!/usr/bin/env python
"""Render per-rung markdown tables from lab/archive.jsonl for a given tag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parent / "archive.jsonl"
LADDER = [1, 2, 4, 8, 16, 32, 64]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in ARCHIVE.open()]
    rows = [r for r in rows if r.get("tag") == args.tag and r.get("status") == "ok"]
    if not rows:
        print(f"(no runs with tag {args.tag})")
        return 0
    head = "| variant | MAX_T | " + " | ".join(f"T={t}" for t in LADDER) + " | mean acc |"
    print(head)
    print("|" + "---|" * (len(LADDER) + 3))
    for r in rows:
        rung = {int(k): v for k, v in (r.get("rung_exact_accuracy") or {}).items()}
        cells = " | ".join(
            f"{rung[t]:.3f}" if t in rung else "-" for t in LADDER
        )
        note = r["note"].replace("|", "/")
        print(
            f"| {note} | **{r['max_certified_t']}** | {cells} "
            f"| {r['mean_exact_accuracy']:.4f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
