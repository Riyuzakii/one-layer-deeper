#!/usr/bin/env python
"""Emit a lab manifest.

Two screening modes:

* ``--mode fixed_step`` (default) — a huge wall-clock budget with a hard
  ``max_steps`` ceiling.  The clock never binds, so results are *immune to GPU
  contention* and comparable across parallel agents.  Use this for every
  controlled architecture/optimizer comparison.
* ``--mode wallclock`` — a tier-faithful budget (60/600/3600s).  Only
  meaningful on an idle GPU, and only ever a local approximation of the H100
  the competition scores on.  Use this to check timing feasibility, never to
  compare two ideas while other jobs are running.

Manifests are evaluator-owned in the real competition; these are local
screening copies that keep every scored knob (dtype, amp, grad_clip, state
ceiling) identical to the official tier manifests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LAB_MANIFESTS = REPO / "lab" / "manifests"

# data roots of the official public datasets, by short name
DATA_ROOTS = {
    "e1": "data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_323_t123",
    "e2": "data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_899_t124",
    "e3": "data/generated/squaring_mod_new11_easy_bidirectional_fixed_t_b1011_t2",
    "e4": "data/generated/squaring_mod_new11_easy_bidirectional_fixed_t_b1112_t2",
    "e5": "data/generated/squaring_mod_new11_easy_bidirectional_variable_b1011_t123",
    "m1": "data/generated/squaring_mod_new11_medium_bidirectional_fixed_n_10403_t4816",
    "m2": "data/generated/squaring_mod_new11_medium_bidirectional_fixed_n_38021_t4816",
    "m3": "data/generated/squaring_mod_new11_medium_bidirectional_fixed_t_b111315_t2",
    "m4": "data/generated/squaring_mod_new11_medium_bidirectional_fixed_t_b141822_t8",
    "m5": "data/generated/squaring_mod_new11_medium_bidirectional_variable_b121416_t248",
    # Hard-FAITHFUL proxies (see lab/gen_hard_faithful.sh) -- split_group=modulus with
    # separate_ood_splits, which is the structure the hosted h1 run revealed. Train and
    # test moduli are DISJOINT. These are the ones to use for Hard.
    "hf1": "data/generated/proxy_hard_modulus_b161820_t4816",
    "hf1s": "data/generated/proxy_hard_modulus_b161820_t4816_small",
    # SUPERSEDED hard proxies (see lab/gen_hard_proxy.sh). These use split_group=prompt,
    # where train and test SHARE moduli, so they do not model h1. Kept for continuity
    # with earlier results only -- do not use for new Hard work.
    "hp1": "data/generated/proxy_hard_fixed_n_p2003_q2011_t4816",
    "hp2": "data/generated/proxy_hard_sampled_b3032_t4816",
    "hp3": "data/generated/proxy_hard_sampled_b202428_t8",
}

TIER_SECONDS = {"e": 60.0, "m": 600.0, "h": 3600.0}


def build(
    *,
    dataset: str,
    mode: str,
    max_steps: int,
    seconds: float | None,
    seeds: list[int],
    batch_size: int,
    eval_batch_size: int,
    name: str | None,
) -> dict:
    data_root = DATA_ROOTS[dataset]
    if seconds is None:
        seconds = 100_000.0 if mode == "fixed_step" else TIER_SECONDS[dataset[0]]
    suffix = f"s{'_'.join(str(s) for s in seeds)}"
    label = name or (
        f"lab_{dataset}_{'fs' + str(max_steps) if mode == 'fixed_step' else 'wc'}_{suffix}"
    )
    return {
        "name": label,
        "data": {
            "kind": "squaring_mod",
            "data_root": data_root,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "shuffle_train": True,
            "shuffle_eval": False,
            "num_workers": 2,
            "pin_memory": True,
            "drop_last": True,
            "seed": 45,
        },
        "runtime": {
            "device": "cuda:0",
            "dtype": "bfloat16",
            "amp": True,
            "compile": False,
            "total_training_time_seconds": seconds,
            "max_steps": max_steps,
            "seeds": seeds,
            "grad_clip": 1,
            "log_every": 100,
        },
        "model_state": {"maximum_elements": 500000000},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATA_ROOTS))
    ap.add_argument("--mode", default="fixed_step", choices=("fixed_step", "wallclock"))
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[74])
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    manifest = build(
        dataset=args.dataset,
        mode=args.mode,
        max_steps=args.max_steps,
        seconds=args.seconds,
        seeds=args.seeds,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        name=args.name,
    )
    out = Path(args.out) if args.out else LAB_MANIFESTS / f"{manifest['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
