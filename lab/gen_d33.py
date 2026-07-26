#!/usr/bin/env python
"""Emit report section 3.3's table from lab/rel_runs.jsonl."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
seen = {}
for l in open(os.path.join(ROOT, "lab/rel_runs.jsonl")):
    d = json.loads(l)
    if "train_exact_hard" in d:
        seen[d["tag"]] = d
rows = [("`--rel affine` (**flagged**)", "d_aff_s0"),
        ("`--rel affine` (**flagged**)", "d_aff_s1"),
        ("`--rel affine`, no non-degeneracy", "d_aff_nd0_s0"),
        ("`--rel free` (conservative)", "d_free_s0"),
        ("`--rel free` (conservative)", "d_free_s1")]
print("**LEGAL — 2,000 steps, m1 scale.** `--rel-nondeg` adds a generic "
      "penalty on `inc` collapsing to the learned additive identity.\n")
print("| variant | seed | `train_exact` | **`train_exact_hard`** | "
      "`held_exact_hard` | `local_ce` | `add_shift` | `out_div` |")
print("|---|---|---|---|---|---|---|---|")
any_missing = []
for label, tag in rows:
    d = seen.get(tag)
    sd = tag.rsplit("_s", 1)[1]
    if not d:
        any_missing.append(tag)
        continue
    print(f"| {label} | {sd} | {d.get('train_exact', 0):.3f} | "
          f"**{d['train_exact_hard']:.3f}** | {d['held_exact_hard']:.3f} | "
          f"{d.get('local_ce', -1):.3f} | {d.get('add_shift', -1):.3f} | "
          f"{d.get('out_div', -1):.3f} |")
if any_missing:
    print(f"\n> **Not measured:** `{'`, `'.join(any_missing)}` were still in "
          "flight when the session ended. See §12.")
