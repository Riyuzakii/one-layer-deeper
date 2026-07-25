#!/usr/bin/env python
"""Phase 1 instrumentation: run a submission on a manifest and archive the result.

Runs `python -m benchmark.runner` as a SUBPROCESS (fresh process per run -> clean
GPU memory + honest timing), parses the final `RESULT_JSON=` line, and appends one
row to lab/archive.jsonl. Failures are archived too (the plan: negative results are
recorded, not discarded).

COMPLIANCE: this harness only ever reads the *submission* source and the runner's
stdout. It never opens anything under data/generated/. Do not add data inspection.

HARDWARE NOTE: we run on a B300 (sm_103), not the H100 the competition scores on.
`completed_training_steps` and any wall-clock figure here are B300 numbers and will
be HIGHER than H100 for the same budget. Treat step counts / timings as B300-local;
treat accuracy-at-fixed-step-count and accuracy-vs-depth as hardware-independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# runner logs e.g. "step=100 loss=1.234567 accuracy=0.250000 elapsed=..s budget=..s"
_STEP_RE = re.compile(r"step=(\d+)\s+loss=([\d.eE+-]+)\s+accuracy=([\d.eE+-]+)")


def parse_training_curve(stdout: str, max_points: int = 40) -> list:
    pts = []
    for ln in stdout.splitlines():
        m = _STEP_RE.search(ln)
        if m:
            pts.append([int(m.group(1)), float(m.group(2)), float(m.group(3))])
    if len(pts) <= max_points:
        return pts
    stride = len(pts) // max_points
    return pts[::stride][:max_points] + [pts[-1]]

REPO = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO / "benchmark" / "manifests"
ARCHIVE = Path(__file__).resolve().parent / "archive.jsonl"


def resolve_manifest(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.is_file():
        return p.resolve()
    cand = MANIFEST_DIR / name_or_path
    if cand.is_file():
        return cand
    cand = MANIFEST_DIR / f"{name_or_path}.json"
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"manifest not found: {name_or_path}")


def sha8(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def parse_result(stdout: str) -> dict | None:
    line = None
    for ln in stdout.splitlines():
        if ln.startswith("RESULT_JSON="):
            line = ln[len("RESULT_JSON=") :]
    if line is None:
        return None
    return json.loads(line)


def flatten(result: dict) -> dict:
    """Pull the fields we care about out of the nested runner result.

    THE RANKING METRIC (since the 2026-07-24 scoring change) is the depth
    profile: `max_certified_time_steps` (largest T whose rung and every lower
    rung are 100% exact), tie-broken by `ood_n_max_certified_time_steps`.
    `mean_exact_accuracy` is now only a diagnostic.
    """
    score = result.get("score", {})
    seeds = result.get("seeds", [])
    # one seed per manifest in this competition, but stay general
    splits: dict[str, float] = {}
    completed = []
    tsec = []
    esec = []
    state_elems = None
    opt_elems = None
    final_loss = []
    # per-token cross-entropy on each scored split.  exact_accuracy is
    # all-or-none per example, so it cannot distinguish "the model knows nothing"
    # from "the model knows most digits".  ln(17) = 2.833 is the
    # uniform-over-vocab reference.  (The runner does not expose a per-rung loss,
    # only per-rung exact accuracy, so this is measured on `test`/`ood`.)
    split_loss_acc: dict[str, list[float]] = {}
    for s in seeds:
        completed.append(s.get("completed_training_steps"))
        tsec.append(s.get("training_seconds"))
        esec.append(s.get("evaluation_seconds"))
        state_elems = s.get("model_state_elements", state_elems)
        opt_elems = s.get("optimizer_state_elements_after_first_step", opt_elems)
        final_loss.append(s.get("final_train_loss"))
        for split, m in s.get("evaluation", {}).items():
            splits.setdefault(split, [])
            splits[split].append(m.get("exact_accuracy"))
            if m.get("loss") is not None:
                split_loss_acc.setdefault(split, []).append(m["loss"])
    # mean per split across seeds
    split_acc = {k: (sum(v) / len(v) if v else None) for k, v in splits.items()}

    profile = result.get("depth_profile") or {}
    rung_acc: dict[str, dict[int, float]] = {"seen_n": {}, "ood_n": {}}
    # per-seed detail (multi-seed manifests aggregate with min(), which hides the
    # spread; rungs are small so the spread is the thing that matters)
    per_seed_max_t = [
        (s.get("depth_profile") or {}).get("max_certified_time_steps") or 0 for s in seeds
    ]
    per_seed_rung = [
        {
            r["time_steps"]: r.get("exact_accuracy")
            for r in ((s.get("depth_profile") or {}).get("rungs") or [])
        }
        for s in seeds
    ]
    for s in seeds:
        seed_profile = s.get("depth_profile") or {}
        for key, rungs in (
            ("seen_n", seed_profile.get("rungs") or []),
            ("ood_n", seed_profile.get("ood_n_rungs") or []),
        ):
            for rung in rungs:
                acc = rung.get("exact_accuracy")
                if acc is None:
                    continue
                t = rung["time_steps"]
                rung_acc[key].setdefault(t, []).append(acc)
    rung_acc = {
        key: {t: sum(v) / len(v) for t, v in sorted(by_t.items())}
        for key, by_t in rung_acc.items()
    }

    return {
        # --- ranking metric ---
        "max_certified_t": profile.get("max_certified_time_steps") or 0,
        "ood_n_max_certified_t": profile.get("ood_n_max_certified_time_steps") or 0,
        "rung_exact_accuracy": rung_acc["seen_n"],
        "ood_n_rung_exact_accuracy": rung_acc["ood_n"],
        "per_seed_max_certified_t": per_seed_max_t,
        "per_seed_rung_exact_accuracy": per_seed_rung,
        "split_loss": {k: sum(v) / len(v) for k, v in sorted(split_loss_acc.items())},
        # --- diagnostics ---
        "mean_exact_accuracy": score.get("mean_exact_accuracy"),
        "split_exact_accuracy": split_acc,
        "completed_training_steps": completed,
        "training_seconds": tsec,
        "evaluation_seconds": esec,
        "model_state_elements": state_elems,
        "optimizer_state_elements": opt_elems,
        "final_train_loss": final_loss,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--manifest", required=True, help="name (e.g. h100_easy_e1) or path")
    ap.add_argument("--note", default="", help="free-text: what is being tested")
    ap.add_argument("--tag", default="", help="axis label, e.g. A-depth")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    sub = Path(args.submission).resolve()
    manifest = resolve_manifest(args.manifest)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    # sm_103 (B300) isn't in torch's compiled arch list -> kernels JIT from sm_100
    # PTX on first use. Persist that JIT cache on scratch (home is quota-limited) so
    # it survives across runs. B300-local workaround only; H100 (sm_90) never JITs.
    env.setdefault("CUDA_CACHE_PATH", "/home/scratch.arohan_hw/.nv_cache")
    env.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 1024 * 1024 * 1024))

    cmd = [
        sys.executable,
        "-m",
        "benchmark.runner",
        "--manifest",
        str(manifest),
        "--submission-file",
        str(sub),
    ]
    started = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout, stderr, rc = (e.stdout or ""), (e.stderr or "") + "\n[TIMEOUT]", -9

    result = parse_result(stdout) if rc == 0 else None
    row = {
        "ts": started.isoformat(),
        "tag": args.tag,
        "note": args.note,
        "submission": str(sub.relative_to(REPO)) if str(sub).startswith(str(REPO)) else str(sub),
        "submission_sha8": sha8(sub),
        "manifest": manifest.stem,
        "status": "ok" if result is not None else "failed",
        "returncode": rc,
    }
    if result is not None:
        row.update(flatten(result))
        row["train_curve"] = parse_training_curve(stdout)  # [step, loss, acc] downsampled
    else:
        row["stderr_tail"] = "\n".join((stderr or "").splitlines()[-25:])

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with ARCHIVE.open("a") as f:
        f.write(json.dumps(row) + "\n")

    # concise human summary
    if result is not None:
        acc = row["mean_exact_accuracy"]
        steps = row["completed_training_steps"]
        ts = row["training_seconds"]
        rungs = {t: round(a, 3) for t, a in row["rung_exact_accuracy"].items()}
        print(
            f"[OK] {manifest.stem} sha={row['submission_sha8']} "
            f"MAX_T={row['max_certified_t']} OOD_N_MAX_T={row['ood_n_max_certified_t']} "
            f"mean_acc={acc:.4f} steps={steps} train_s={ts}"
        )
        print(f"      rungs(seen_n)={rungs}")
        print(f"      per_seed_MAX_T={row['per_seed_max_certified_t']}")
        print(
            "      split_loss(ln17=2.833)="
            f"{ {k: round(v, 3) for k, v in row['split_loss'].items()} }"
        )
        print(
            "      per_seed_rung1="
            f"{[None if d.get(1) is None else round(d[1], 3) for d in row['per_seed_rung_exact_accuracy']]}"
        )
        print(
            f"      rungs(ood_n)="
            f"{ {t: round(a, 3) for t, a in row['ood_n_rung_exact_accuracy'].items()} }"
        )
        print(
            f"      splits={ {k: round(v,3) for k,v in row['split_exact_accuracy'].items()} }"
        )
    else:
        print(f"[FAIL rc={rc}] {manifest.stem} sha={row['submission_sha8']}")
        print("--- stderr tail ---")
        print(row.get("stderr_tail", ""))
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
