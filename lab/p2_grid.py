#!/usr/bin/env python
"""Drive the §3.2 / §3.3 sweep on the real task through lab/run_experiment.py.

Every cell is a fixed-step manifest run (BRIEF §5: three siblings share the GPU,
so wall-clock manifests are not comparable).  Each cell also writes a per-step
diagnostic stream (collapse detector + realised eigenvalue range) to
lab/p2_diag/<cell>.jsonl via the submission's lab-only P2_DIAG_FILE hook.

Usage:  $VENV lab/p2_grid.py --grid nh        # or: eig tau base lr0 repeat all
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = "/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python"
DIAG = ROOT / "lab" / "p2_diag"
LOG = ROOT / "lab" / "p2_grid_log.jsonl"


def cell_name(cfg: dict) -> str:
    a = cfg["P2_ARCH"]
    parts = [a]
    if a == "delta":
        parts += [f"nh{cfg['P2_NH']}", cfg["P2_EIG"]]
    elif a == "pdssm":
        parts += [f"tau{cfg['P2_TAU']}", cfg["P2_STE"]]
        if str(cfg.get("P2_STATE", 16)) != "16":
            parts.append(f"N{cfg['P2_STATE']}")
    if str(cfg.get("P2_REPEAT", 1)) != "1":
        parts.append(f"rep{cfg['P2_REPEAT']}")
    if str(cfg.get("P2_LR", "0.001")) == "0":
        parts.append("lr0")
    return "_".join(parts)


def run(cfg: dict, manifest: str, tag: str, note: str) -> dict:
    DIAG.mkdir(exist_ok=True)
    name = f"{cell_name(cfg)}_{manifest.split('_')[-1]}"
    diag = DIAG / f"{name}.jsonl"
    if diag.exists():
        diag.unlink()
    env = dict(os.environ)
    env.update({k: str(v) for k, v in cfg.items()})
    env["P2_DIAG_FILE"] = str(diag)
    env["CUDA_CACHE_PATH"] = env.get("CUDA_CACHE_PATH", "/home/scratch.arohan_hw/.nv_cache")
    t0 = time.time()
    proc = subprocess.run(
        [
            VENV, "lab/run_experiment.py",
            "--submission", "submissions/plan2-pd-ssm-delta/submission.py",
            "--manifest", f"lab/manifests/{manifest}.json",
            "--tag", tag, "--note", note, "--timeout", "3000",
        ],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    wall = time.time() - t0
    out = proc.stdout
    rec = {"cell": name, "cfg": cfg, "manifest": manifest, "tag": tag,
           "wall_seconds": round(wall, 1), "rc": proc.returncode}
    for ln in out.splitlines():
        if ln.startswith("[OK]"):
            rec["summary"] = ln.strip()
        if ln.startswith("      "):
            rec.setdefault("detail", []).append(ln.strip())
    if proc.returncode != 0:
        rec["stderr_tail"] = "\n".join(proc.stderr.splitlines()[-15:])
    # fold in the last diagnostic record
    if diag.exists():
        lines = diag.read_text().splitlines()
        if lines:
            rec["diag_final"] = json.loads(lines[-1])
            rec["diag_first"] = json.loads(lines[0])
    with LOG.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"  [{wall:6.1f}s] {name:<28} {rec.get('summary', 'FAILED')}", flush=True)
    if "diag_final" in rec:
        d = rec["diag_final"]
        extras = " ".join(
            f"{k}={d[k]:.3f}" for k in
            ("out_diversity", "top_token_share", "eig_min", "eig_max",
             "eig_frac_neg", "p_max_prob", "p_perm_frac", "d_abs_mean")
            if k in d
        )
        print(f"          diag: {extras}", flush=True)
    return rec


def manifests(seeds, steps):
    return [f"p2_e5_fs{steps}_s{s}" for s in seeds]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", nargs="+", default=["all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[74, 7, 21])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--nh", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--no-soft", action="store_true")
    args = ap.parse_args()
    g = set(args.grid)
    all_ = "all" in g
    mans = manifests(args.seeds, args.steps)

    jobs: list[tuple[dict, str, str, str]] = []

    if all_ or "nh" in g:
        for n_h in args.nh:
            for m in mans:
                jobs.append((
                    {"P2_ARCH": "delta", "P2_NH": n_h, "P2_EIG": "neg"},
                    m, "S1-nh",
                    f"LEGAL §3.3 DeltaProduct n_h={n_h} eig=[-1,1] {m}",
                ))
    if all_ or "eig" in g:
        for n_h in args.nh:
            for m in mans:
                jobs.append((
                    {"P2_ARCH": "delta", "P2_NH": n_h, "P2_EIG": "pos"},
                    m, "S2-eig",
                    f"LEGAL §3.3 control: n_h={n_h} eig=[0,1] the flagged trap {m}",
                ))
    if all_ or "tau" in g:
        for tau in args.taus:
            for m in mans:
                jobs.append((
                    {"P2_ARCH": "pdssm", "P2_TAU": tau, "P2_STE": "hard"},
                    m, "S3-tau",
                    f"LEGAL §3.2 PD-SSM straight-through tau={tau} {m}",
                ))
        for m in ([] if args.no_soft else mans):
            jobs.append((
                {"P2_ARCH": "pdssm", "P2_TAU": 1.0, "P2_STE": "none"},
                m, "S3-tau", f"LEGAL §3.2 PD-SSM soft control no discretisation {m}",
            ))
    if all_ or "base" in g:
        for m in mans:
            jobs.append((
                {"P2_ARCH": "matscan"}, m, "S4-base",
                f"LEGAL §3.1 dense matrix-scan reference {m}",
            ))
    if all_ or "lr0" in g:
        for cfg in (
            {"P2_ARCH": "delta", "P2_NH": 2, "P2_EIG": "neg"},
            {"P2_ARCH": "pdssm", "P2_TAU": 1.0, "P2_STE": "hard"},
            {"P2_ARCH": "matscan"},
        ):
            c = dict(cfg)
            c["P2_LR"] = 0
            jobs.append((c, mans[0], "S5-lr0",
                         f"CONTROL --lr 0 BRIEF2 6.1 random init no training {mans[0]}"))
    if "small" in g:
        # §3.2 at the granularity where the task plausibly HAS a small FSA:
        # digit-carry's sharpest finding is "the state alphabet must be small"
        # (a 32-dim carry just re-encodes the value).  N=11 is the carry
        # alphabet size for base-10 add-with-carry.
        for n in (11, 8):
            for m in mans:
                jobs.append((
                    {"P2_ARCH": "pdssm", "P2_STATE": n, "P2_TAU": 1.0,
                     "P2_STE": "hard"},
                    m, "S7-small",
                    f"LEGAL §3.2 PD-SSM with a small state alphabet N={n} "
                    f"(learned FSA at carry granularity)",
                ))
    if all_ or "repeat" in g:
        for rep in (2, 4):
            jobs.append((
                {"P2_ARCH": "delta", "P2_NH": 2, "P2_EIG": "neg", "P2_REPEAT": rep},
                mans[0], "S6-rep",
                f"LEGAL §3.3 tied unrolled internal depth R={rep} "
                f"(composition depth R*L, not the token axis alone)",
            ))

    print(f"{len(jobs)} cells", flush=True)
    for i, (cfg, m, tag, note) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {tag} {cfg} {m}", flush=True)
        run(cfg, m, tag, note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
