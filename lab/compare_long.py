#!/usr/bin/env python
"""Print the e5 and m1 40,000-step runs side by side at identical checkpoints.

The coordinator asked for the m1 curve "at the same checkpoints you used for the e5 40k
run so the two are directly comparable".  `probe_rnn.py` records at `steps // 8`, so both
runs sample at 1, 5k, 10k … 40k with no alignment work needed; this just tabulates them.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROWS = [
    json.loads(line)
    for line in (REPO / "lab" / "probe_rnn.jsonl").read_text().splitlines()
    if line.strip()
]

LONG = [r for r in ROWS if r["tag"] in ("e5-long40k", "m1-long40k")]
LONG.sort(key=lambda r: (r["dataset"], r["overrides"]["D_H"], r["seed"]))

print("## 40,000-step runs, identical checkpoints\n")
print("| dataset | D_H | params/row | seed | " + " | ".join(
    f"step {s // 1000}k" if s else "step 1" for s in
    [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
) + " |")
print("|" + "---|" * 13)
ROWS_PER_DS = {"e5": 4800, "m1": 27000}
for r in LONG:
    d_h = r["overrides"]["D_H"]
    per_row = r["params"] / ROWS_PER_DS[r["dataset"]]
    cells = " | ".join(f"{c[2]:.3f}" for c in r["curve"])
    print(f"| {r['dataset']} | {d_h} | {per_row:.1f} | {r['seed']} | {cells} |")

print("\n## final train / held, and the loss the run reached\n")
print("| dataset | D_H | seed | final loss | train_exact | held_exact | div | top | held_ce | n_held |")
print("|" + "---|" * 10)
for r in LONG:
    print(
        f"| {r['dataset']} | {r['overrides']['D_H']} | {r['seed']} | {r['curve'][-1][1]:.4f} "
        f"| **{r['train_exact']:.4f}** | **{r['held_exact']:.4f}** | {r['held_diversity']:.3f} "
        f"| {r['held_top_share']:.4f} | {r['held_ce']:.3f} | {r['held_n']} |"
    )
