#!/usr/bin/env python
"""Tabulate lab/p2_probe_results.jsonl (synthetic) and lab/p2_grid_log.jsonl (real task)."""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

LAB = Path(__file__).resolve().parent


def load(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def agg(vals):
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{st.mean(vals):.3f}+-{st.pstdev(vals):.3f}"


def probe_tables() -> None:
    rows = load(LAB / "p2_probe_results.jsonl")
    rows = [r for r in rows if r.get("steps", 0) >= 1000]
    if not rows:
        print("(no probe rows yet)")
        return
    tasks = ["parity", "mod3", "a5"]

    print("\n### PROBE A -- realised eigenvalue range (DeltaNet n_h=1), last-token accuracy")
    print("| task | chance | eig in [0,1] | eig in [-1,1] | 2x-length, [-1,1] |")
    print("|---|---|---|---|---|")
    for task in tasks:
        sel = [r for r in rows if r["label"] == "A_eig" and r["task"] == task]
        if not sel:
            continue
        ch = sel[0]["chance"]
        pos = [r["last_acc"] for r in sel if r["eig_range"] == "pos"]
        neg = [r["last_acc"] for r in sel if r["eig_range"] == "neg"]
        ext = [r["ext2x_last_acc"] for r in sel if r["eig_range"] == "neg"]
        print(f"| {task} | {ch:.3f} | {agg(pos)} | {agg(neg)} | {agg(ext)} |")

    print("\n### PROBE B -- the n_h sweep (DeltaProduct), last-token accuracy")
    print("| task | chance | eig | n_h=1 | n_h=2 | n_h=3 | n_h=4 |")
    print("|---|---|---|---|---|---|---|")
    for task in tasks:
        for lab, eig in (("B_nh", "neg"), ("Bp_nh_pos", "pos")):
            sel = [r for r in rows if r["label"] == lab and r["task"] == task]
            if not sel:
                continue
            ch = sel[0]["chance"]
            cells = []
            for n_h in (1, 2, 3, 4):
                v = [r["last_acc"] for r in sel if r["n_h"] == n_h]
                cells.append(agg(v))
            rng = "[-1,1]" if eig == "neg" else "[0,1]"
            print(f"| {task} | {ch:.3f} | {rng} | " + " | ".join(cells) + " |")

    print("\n### PROBE C -- PD-SSM straight-through temperature sweep, last-token accuracy")
    print("| task | chance | tau=0.1 | tau=0.3 | tau=1 | tau=3 | tau=10 | soft (no STE) |")
    print("|---|---|---|---|---|---|---|---|")
    for task in tasks:
        sel = [r for r in rows if r["label"] in ("C_tau", "C_soft") and r["task"] == task]
        if not sel:
            continue
        ch = sel[0]["chance"]
        cells = []
        for tau in (0.1, 0.3, 1.0, 3.0, 10.0):
            v = [r["last_acc"] for r in sel if r["label"] == "C_tau" and r["tau"] == tau]
            cells.append(agg(v))
        soft = [r["last_acc"] for r in sel if r["label"] == "C_soft"]
        cells.append(agg(soft))
        print(f"| {task} | {ch:.3f} | " + " | ".join(cells) + " |")

    print("\n### PROBE D -- §3.1 dense matrix-scan reference on the same tasks")
    print("| task | chance | matscan last-acc | out diversity |")
    print("|---|---|---|---|")
    for task in tasks:
        sel = [r for r in rows if r["label"] == "D_matscan" and r["task"] == task]
        if not sel:
            continue
        print(f"| {task} | {sel[0]['chance']:.3f} | "
              f"{agg([r['last_acc'] for r in sel])} | "
              f"{agg([r['out_diversity'] for r in sel])} |")

    print("\n### PROBE E -- realised eigenvalue range AFTER training (delta cells)")
    print("| eig setting | n_h | eig_min | eig_max | frac negative |")
    print("|---|---|---|---|---|")
    for eig in ("pos", "neg"):
        for n_h in (1, 2, 3, 4):
            sel = [r for r in rows if r.get("arch") == "delta"
                   and r["eig_range"] == eig and r["n_h"] == n_h and "eig_min" in r]
            if not sel:
                continue
            print(f"| {'[-1,1]' if eig=='neg' else '[0,1]'} | {n_h} | "
                  f"{agg([r['eig_min'] for r in sel])} | "
                  f"{agg([r['eig_max'] for r in sel])} | "
                  f"{agg([r['eig_frac_neg'] for r in sel])} |")


_SUM = re.compile(r"MAX_T=(\d+) OOD_N_MAX_T=(\d+) mean_acc=([\d.]+) steps=\[(\d+)\] "
                  r"train_s=\[([\d.]+)\]")
_R1 = re.compile(r"rungs\(seen_n\)=\{1: ([\d.]+)")


def grid_table() -> None:
    rows = load(LAB / "p2_grid_log.jsonl")
    if not rows:
        print("\n(no grid rows yet)")
        return
    by = defaultdict(list)
    for r in rows:
        s = r.get("summary", "")
        m = _SUM.search(s)
        if not m:
            continue
        key = r["cell"].rsplit("_s", 1)[0]
        d = r.get("diag_final", {})
        by[key].append({
            "tag": r["tag"],
            "max_t": int(m.group(1)), "ood_t": int(m.group(2)),
            "mean_acc": float(m.group(3)), "steps": int(m.group(4)),
            "train_s": float(m.group(5)),
            "rung1": float(_R1.search(" ".join(r.get("detail", []))).group(1))
            if _R1.search(" ".join(r.get("detail", []))) else None,
            "diversity": d.get("out_diversity"),
            "top_share": d.get("top_token_share"),
            "eig_min": d.get("eig_min"), "eig_max": d.get("eig_max"),
            "eig_frac_neg": d.get("eig_frac_neg"),
            "p_max_prob": d.get("p_max_prob"), "p_perm_frac": d.get("p_perm_frac"),
            "wall": r["wall_seconds"],
        })

    print("\n### REAL TASK -- e5, fixed_step 1500, multi-seed")
    print("| cell | n | MAX_T | OOD_N | rung-1 | mean_acc | out_div | top_share | "
          "train_s | extra |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(by):
        v = by[key]
        r1 = [x["rung1"] for x in v if x["rung1"] is not None]
        dv = [x["diversity"] for x in v if x["diversity"] is not None]
        tsh = [x["top_share"] for x in v if x["top_share"] is not None]
        extra = ""
        em = [x["eig_min"] for x in v if x["eig_min"] is not None]
        if em:
            extra = (f"eig[{st.mean(em):+.3f},"
                     f"{st.mean([x['eig_max'] for x in v]):+.3f}] "
                     f"neg={st.mean([x['eig_frac_neg'] for x in v]):.3f}")
        pm = [x["p_max_prob"] for x in v if x["p_max_prob"] is not None]
        if pm:
            extra = (f"p_max={st.mean(pm):.3f} "
                     f"perm={st.mean([x['p_perm_frac'] for x in v]):.3f}")
        print(f"| {key} | {len(v)} | {max(x['max_t'] for x in v)} | "
              f"{max(x['ood_t'] for x in v)} | {agg(r1)} | "
              f"{agg([x['mean_acc'] for x in v])} | {agg(dv)} | {agg(tsh)} | "
              f"{st.mean([x['train_s'] for x in v]):.0f} | {extra} |")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "probe"):
        probe_tables()
    if what in ("all", "grid"):
        grid_table()
