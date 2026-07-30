#!/usr/bin/env python
"""Tabulate the PLAN2 Phase-0 runs out of lab/archive.jsonl.

Reports the metric ROW, not the cell (BRIEF2 s6.3):
  train  final train-batch exact accuracy from the runner's own log line
  test   held-out prompts
  r1     depth rung T=1 -- fresh (N,x) pairs never used in train/test/ood
  MAXT   the ranking metric
Read-only over our own archive; never touches generated data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parent / "archive.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="*", default=None)
    ap.add_argument("--grep", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in ARCHIVE.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("status") == "ok" and r.get("tag", "").startswith("P")]
    if args.tag:
        rows = [r for r in rows if r.get("tag") in args.tag]
    if args.grep:
        rows = [r for r in rows if args.grep in (r.get("note") or "")]

    hdr = (
        f"{'tag':<16}{'note':<38}{'manifest':<24}"
        f"{'steps':>6}{'train':>7}{'test':>7}{'ood':>7}{'r1':>7}{'r2':>7}{'mean':>8}{'MAXT':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        curve = r.get("train_curve") or []
        train = curve[-1][2] if curve else float("nan")
        sp = r.get("split_exact_accuracy") or {}
        rung = {int(k): v for k, v in (r.get("rung_exact_accuracy") or {}).items()}
        steps = (r.get("completed_training_steps") or [0])[0]
        print(
            f"{r.get('tag',''):<16}{(r.get('note') or '')[:37]:<38}"
            f"{r.get('manifest',''):<24}{steps:>6}"
            f"{train:>7.3f}{sp.get('test', float('nan')):>7.3f}"
            f"{sp.get('ood', float('nan')):>7.3f}"
            f"{rung.get(1, float('nan')):>7.3f}{rung.get(2, float('nan')):>7.3f}"
            f"{(r.get('mean_exact_accuracy') or 0):>8.4f}"
            f"{r.get('max_certified_t', 0):>5}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
