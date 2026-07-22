#!/usr/bin/env python
"""Tabulate lab/archive.jsonl: one row per run, grouped by tag, sorted by mean_acc.
Read-only over our own archive (never touches generated data)."""
from __future__ import annotations
import json
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parent / "archive.jsonl"


def main() -> int:
    if not ARCHIVE.exists():
        print("no archive yet")
        return 0
    rows = [json.loads(l) for l in ARCHIVE.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]
    bad = [r for r in rows if r.get("status") != "ok"]
    hdr = f"{'tag':<14}{'manifest':<22}{'mean_acc':>9}{'steps':>8}  splits"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(ok, key=lambda r: (r.get("tag", ""), -(r.get("mean_exact_accuracy") or 0))):
        steps = r.get("completed_training_steps") or []
        steps_s = steps[0] if len(steps) == 1 else steps
        splits = r.get("split_exact_accuracy") or {}
        splits_s = " ".join(f"{k}={v:.3f}" for k, v in sorted(splits.items()))
        note = (r.get("note") or "")[:40]
        print(
            f"{r.get('tag',''):<14}{r.get('manifest',''):<22}"
            f"{(r.get('mean_exact_accuracy') or 0):>9.4f}{str(steps_s):>8}  {splits_s}   {note}"
        )
    if bad:
        print(f"\n{len(bad)} failed run(s):")
        for r in bad:
            print(f"  [{r.get('tag','')}] {r.get('manifest','')} rc={r.get('returncode')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
