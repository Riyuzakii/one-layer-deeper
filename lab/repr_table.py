#!/usr/bin/env python
"""Per-rung table for the representation experiments in lab/archive.jsonl.

Usage: python lab/repr_table.py [tag-prefix ...]
Groups by (submission tag, dataset) and reports mean +- spread over seeds.
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parent / "archive.jsonl"
LADDER = [1, 2, 4, 8, 16, 32, 64]


def main() -> int:
    prefixes = sys.argv[1:] or ["repr"]
    rows = [
        json.loads(line)
        for line in ARCHIVE.read_text().splitlines()
        if line.strip()
    ]
    rows = [
        r
        for r in rows
        if any(r.get("tag", "").startswith(p) for p in prefixes)
        and r.get("status") == "ok"
    ]
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        m = re.match(r"(\S+) @ (\S+)", r.get("note", ""))
        name = m.group(1) if m else r.get("note", "?")[:20]
        dataset = m.group(2) if m else r.get("manifest", "?")
        groups.setdefault((dataset, name), []).append(r)

    for (dataset, name), rs in sorted(groups.items()):
        per_rung = {t: [] for t in LADDER}
        means, maxts, steps = [], [], []
        for r in rs:
            acc = {int(k): v for k, v in r.get("rung_exact_accuracy", {}).items()}
            for t in LADDER:
                if t in acc:
                    per_rung[t].append(acc[t])
            means.append(r.get("mean_exact_accuracy") or 0.0)
            maxts.append(r.get("max_certified_t") or 0)
            steps.extend(r.get("completed_training_steps") or [])
        cells = []
        for t in LADDER:
            v = per_rung[t]
            if not v:
                cells.append("  -  ")
            elif len(v) == 1:
                cells.append(f"{v[0]:.3f}")
            else:
                cells.append(f"{st.mean(v):.3f}±{st.pstdev(v):.3f}")
        print(
            f"{dataset:22s} {name:16s} n={len(rs)} steps={max(steps) if steps else '?':<6} "
            f"MAXT={max(maxts)} T1={cells[0]:12s} "
            + " ".join(f"T{t}={c}" for t, c in zip(LADDER[1:], cells[1:]))
            + f"  mean={st.mean(means):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
