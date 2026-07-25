#!/usr/bin/env python
"""Tabulate grok-optimization runs from lab/archive.jsonl.

  python lab/grok_table.py --tag A-wd            # summary table
  python lab/grok_table.py --tag A-wd --curve    # + train-accuracy curve tail
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parent / "archive.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--contains", default=None)
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--last", type=int, default=200)
    args = ap.parse_args()

    rows = []
    for line in ARCHIVE.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if args.tag and r.get("tag") != args.tag:
            continue
        if args.contains and args.contains not in (r.get("note", "") + r.get("submission", "")):
            continue
        rows.append(r)
    rows = rows[-args.last :]

    hdr = f"{'sub':<34} {'mfst':<20} {'st':>7} {'MAXT':>4} {'r1':>6} {'r2':>6} {'r4':>6} {'test':>6} {'mean':>6} {'loss':>8} {'sec':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("status") != "ok":
            print(f"{Path(r['submission']).parent.name:<34} {r['manifest']:<20} FAILED rc={r.get('returncode')}")
            continue
        rung = {int(k): v for k, v in (r.get("rung_exact_accuracy") or {}).items()}
        sp = r.get("split_exact_accuracy") or {}
        name = Path(r["submission"]).parent.name
        steps = (r.get("completed_training_steps") or [0])[0]
        sec = (r.get("training_seconds") or [0])[0] or 0.0
        fl = (r.get("final_train_loss") or [None])[0]
        print(
            f"{name:<34} {r['manifest']:<20} {steps:>7} {r.get('max_certified_t',0):>4} "
            f"{rung.get(1,float('nan')):>6.3f} {rung.get(2,float('nan')):>6.3f} "
            f"{rung.get(4,float('nan')):>6.3f} {sp.get('test',float('nan')):>6.3f} "
            f"{r.get('mean_exact_accuracy',float('nan')):>6.3f} "
            f"{(fl if fl is not None else float('nan')):>8.4f} {sec:>7.1f}"
        )
        if args.curve:
            c = r.get("train_curve") or []
            pts = ", ".join(f"{s}:{a:.2f}" for s, _l, a in c[-14:])
            print(f"    curve(step:trainacc) {pts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
