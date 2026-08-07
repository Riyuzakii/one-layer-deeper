#!/usr/bin/env python
"""Render lab/logs/curric.jsonl as the report's tables.

Every row is labelled LEGAL or DIAGNOSTIC and reports the metric ROW
(train_exact_hard, per-bucket digit accuracy against its measured trivial
floor, local_ce against its measured random-init value, output diversity
against the measured constructed reference), never a single cell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

DIAGNOSTIC = ("only-bits", "--construct")


def label(row) -> str:
    argv = " ".join(row.get("argv", []))
    return "DIAGNOSTIC" if "--only-bits" in argv and "--only-bits 0" not in argv \
        else "LEGAL"


def load(path):
    rows = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def get(row, coh, kind, key):
    d = row.get(f"{coh}_{kind}")
    return None if d is None else d.get(key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(HERE / "logs" / "curric.jsonl"))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    rows = [r for r in load(args.jsonl) if "train_hard" in r]
    if args.only:
        rows = [r for r in rows if args.only in r["tag"]]

    hdr = ("| run | legal | steps | train_exact_hard | heldN_exact_hard | "
           "train dacc 16/18/20 | heldN dacc 16/18/20 | local_ce | "
           "state_sharp | heldN div |")
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for r in rows:
        argv = r.get("argv", [])
        steps = argv[argv.index("--steps") + 1] if "--steps" in argv else "?"
        td = "/".join(str(get(r, "train", "soft", f"d{b}")) for b in (16, 18, 20))
        hd = "/".join(str(get(r, "held_n", "soft", f"d{b}")) for b in (16, 18, 20))
        print(f"| {r['tag']} | {label(r)} | {steps} | "
              f"{get(r,'train','hard','all')} | {get(r,'held_n','hard','all')} | "
              f"{td} | {hd} | {r.get('local_ce')} | "
              f"{r.get('state_sharpness')} | {get(r,'held_n','hard','div')} |")

    print()
    print("| run | mul_fn | mul_gauge | add_shift | sub_shift | ms/step |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['tag']} | {r.get('mul_fn')} | {r.get('mul_gauge')} | "
              f"{r.get('add_shift')} | {r.get('sub_shift')} | "
              f"{r.get('ms_per_step')} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
