#!/usr/bin/env python
"""Collapse lab/runs/monoid_*.jsonl into the report's metric ROWS."""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

KEYS = ["train_exact", "held_exact", "train_exact_hard", "held_exact_hard",
        "held_ce", "held_div", "tbl_all"]


def load(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--group", default="family",
                    help="arg field to group by (family/d/impl/corrupt/lr)")
    ap.add_argument("--final", action="store_true", help="last step only")
    ap.add_argument("--init", action="store_true", help="step 0 only")
    args = ap.parse_args()

    rows = load(args.files)
    by_tag = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r)
    sel = []
    for tag, rs in by_tag.items():
        rs.sort(key=lambda r: r["step"])
        if args.init:
            sel.append(rs[0])
        elif args.final:
            sel.append(rs[-1])
        else:
            sel += [rs[0], rs[-1]]

    groups = defaultdict(list)
    for r in sel:
        a = r["args"]
        key = (a.get(args.group), r["step"] == 0)
        groups[key].append(r)

    hdr = ["group", "when", "n"] + KEYS + ["repaired/total"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for (g, is_init), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), not kv[0][1])):
        cells = [str(g), "init" if is_init else "final", str(len(rs))]
        for k in KEYS:
            v = [r[k] for r in rs if r.get(k) is not None]
            if not v:
                cells.append("-")
            elif len(v) > 1:
                cells.append(f"{st.median(v):.3f} [{min(v):.3f},{max(v):.3f}]")
            else:
                cells.append(f"{v[0]:.3f}")
        rep = [(r.get("repaired"), r.get("corrupted")) for r in rs if r.get("corrupted")]
        if rep:
            cells.append(f"{st.median([x for x, _ in rep]):.0f}/{rep[0][1]}"
                         f" [{min(x for x, _ in rep)},{max(x for x, _ in rep)}]")
        else:
            cells.append("-")
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
