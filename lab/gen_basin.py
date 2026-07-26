#!/usr/bin/env python
"""Emit report section 6.1's basin table from lab/logs/basin3.log."""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "lab/logs/basin3.log")
pat = re.compile(r"k=\s*(\d+) dig=([\d.]+)\+-([\d.]+) exact=([\d.]+) ce=([\d.]+) \| "
                 r"fold=([\d.]+)\+-[\d.]+ red=([\d.]+)\+-[\d.]+ horn=([\d.]+)\+-[\d.]+ "
                 r"rel=([\d.]+)\+-[\d.]+ asc=([\d.]+)\+-[\d.]+ mas=([\d.]+)")
rows = []
for line in open(path):
    m = pat.search(line)
    if m:
        rows.append([float(x) for x in m.groups()])
print("| `k` | `dig` (label) | **`asc`** | `fold` | `red` | `horn` | `rel` | `mas` |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    k, dig, _sd, _ex, _ce, fold, red, horn, rel, asc, mas = r
    print(f"| {int(k)} | {dig:.3f} | **{asc:.1f}** | {fold:.1f} | {red:.1f} | "
          f"{horn:.1f} | {rel:.1f} | {mas:.1f} |")
if rows and rows[-1][0] < 1000:
    print(f"\n> **Partial:** the ladder reached `k = {int(rows[-1][0])}` of 1,000 "
          "before the session ended; see §12 for the resume command.")
