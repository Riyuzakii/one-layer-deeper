#!/usr/bin/env python
"""LAB ONLY -- render lab/pop_runs.jsonl as the report's tables."""
import json
import sys

rows = []
for line in open(sys.argv[1] if len(sys.argv) > 1 else "lab/pop_runs.jsonl"):
    d = json.loads(line)
    if "train_exact_hard_best" in d:
        rows.append(d)

sel = sys.argv[2] if len(sys.argv) > 2 else ""
print("| tag | P | best | mix | argmax | ce-argmin | held(best) | held(argmax) "
      "| n_basin | n(lce<.006) | lce_min | star_rank | ms/step |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in rows:
    if sel and not d["tag"].startswith(sel):
        continue
    print(f"| `{d['tag']}` | {d['pop']} | {d['train_exact_hard_best']:.3f} "
          f"| {d['train_exact_hard_mix']:.3f} | {d['train_exact_hard_argmax']:.3f} "
          f"| {d.get('train_exact_hard_ce_argmin', float('nan')):.3f} "
          f"| {d['held_exact_hard_best']:.3f} | {d['held_exact_hard_argmax']:.3f} "
          f"| {d['n_basin']}/{d['pop']} | {d.get('n_lce_006', '-')}/{d['pop']} "
          f"| {d['local_ce_best']:.4f} | {d.get('star_rank', '-')} "
          f"| {d['ms_per_step']:.0f} |")
