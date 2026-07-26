#!/usr/bin/env python
"""Emit the report's markdown tables from lab/rel_runs.jsonl and
lab/assoc_runs.jsonl.  Last run of a tag wins."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path, key):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        d = json.loads(line)
        if key in d:
            out[d["tag"]] = d
    return out


def row(t, d):
    return (f"| `{t}` | {d.get('train_exact', 0):.3f} | "
            f"**{d['train_exact_hard']:.3f}** | {d['held_exact_hard']:.3f} | "
            f"{d.get('local_ce', -1):.3f} | {d.get('add_shift', -1):.3f} | "
            f"{d.get('sub_shift', -1):.3f} | {d.get('mul_gauge', -1):.2f} | "
            f"{d.get('out_div', -1):.3f} |")


HEAD = ("| run | `train_exact` | **`train_exact_hard`** | `held_exact_hard` | "
        "`local_ce` | `add_shift` | `sub_shift` | `mul_gauge` | `out_div` |\n"
        "|---|---|---|---|---|---|---|---|---|")


def main():
    rel = load(os.path.join(ROOT, "lab/rel_runs.jsonl"), "train_exact_hard")
    asc = load(os.path.join(ROOT, "lab/assoc_runs.jsonl"), "obj")
    pref = sys.argv[1:] or ["a_", "b_", "c_", "d_", "e_", "h_"]
    for p in pref:
        sel = {t: d for t, d in rel.items() if t.startswith(p)}
        if not sel:
            continue
        print(f"\n### {p}\n")
        print(HEAD)
        for t in sorted(sel):
            print(row(t, sel[t]))
    if asc:
        print("\n### discrete search (probe_assoc)\n")
        print("| run | obj | held_obj | `add_shift` | `n_shifts` | `cell_agree` |")
        print("|---|---|---|---|---|---|")
        for t in sorted(asc):
            h = asc[t]["hist"][-1]
            print(f"| `{t}` | {asc[t]['obj']:.5f} | {asc[t]['held_obj']:.5f} | "
                  f"{h['add_shift']} | {h['n_shifts']} | {h['cell']} |")


if __name__ == "__main__":
    main()
