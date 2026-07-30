#!/usr/bin/env python
"""DIAGNOSTIC: collapse detector for the Phase-0 architecture family.

BRIEF2 s6.3 -- "report the metric row, not the cell, with a collapse detector".
The evaluator reports exact accuracy only, which cannot distinguish "learned the
digit marginals and predicts one string for everything" from "learned nothing".
This script replays the evaluator's *semantics* (one optimizer.step per batch,
same manifest knobs, same public dataloaders, same target_positions slicing) and
then measures, on held-out cohorts:

  exact_acc        fraction of examples whose whole answer is right
  token_acc        per-digit accuracy
  const_share      share of the single most common predicted answer string
  distinct_frac    distinct predicted answer strings / examples
  entropy_bits     Shannon entropy of the predicted-answer distribution

A model that has collapsed to a constant map reads const_share ~ 1.0 and
distinct_frac ~ 1/n.  A model that has only learned per-position digit marginals
reads high token_acc relative to exact_acc and low entropy.

This is a LAB DIAGNOSTIC and is never part of a submission: it uses its own
training loop, which BRIEF s4.4 forbids inside a submission.  It never reads,
prints or summarizes any dataset row -- only this model's own predictions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(REPO))

from benchmark.api import ModelSpec, OptimizerSpec  # noqa: E402
from benchmark.batches import prepare_batch  # noqa: E402
from benchmark.manifest import load_manifest  # noqa: E402
from data import infer_max_seq_len, infer_vocab_size, make_dataloaders  # noqa: E402


def load_submission(path: Path):
    spec = importlib.util.spec_from_file_location("p2_diag_sub", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(model, loader, device, manifest) -> dict:
    model.eval()
    counter: Counter[str] = Counter()
    exact = tok_ok = tok_n = rows = 0
    with torch.no_grad():
        for batch in loader:
            input_ids, targets, mask, positions = prepare_batch(batch, device)
            with torch.autocast(
                device_type=device.type, dtype=getattr(torch, manifest.runtime.dtype)
            ):
                logits, _ = model(input_ids, attention_mask=mask)
            idx = torch.arange(logits.shape[0], device=device)[:, None]
            token_logits = logits[idx, positions.clamp_min(0)].float()
            pred = token_logits.argmax(-1)
            valid = targets != -100
            ok = (pred == targets) | ~valid
            exact += int(ok.all(dim=1).sum().item())
            tok_ok += int(((pred == targets) & valid).sum().item())
            tok_n += int(valid.sum().item())
            rows += int(input_ids.shape[0])
            masked = torch.where(valid, pred, torch.full_like(pred, -1))
            for row in masked.tolist():
                counter["".join(str(v) for v in row if v >= 0)] += 1
    model.train()
    total = sum(counter.values())
    probs = [c / total for c in counter.values()]
    return {
        "exact_acc": round(exact / rows, 4),
        "token_acc": round(tok_ok / max(tok_n, 1), 4),
        "examples": rows,
        "const_share": round(max(counter.values()) / total, 4),
        "distinct_frac": round(len(counter) / total, 4),
        "entropy_bits": round(-sum(p * math.log2(p) for p in probs), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=74)
    ap.add_argument("--splits", nargs="+", default=["train", "test", "depth_t_1", "depth_t_2"])
    ap.add_argument("--out", default="lab/p2/diversity.jsonl")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    manifest = load_manifest(Path(args.manifest))
    torch.manual_seed(args.seed)
    module = load_submission(Path(args.submission))
    sub = module.SUBMISSION

    loaders = make_dataloaders(
        replace(
            manifest.data,
            seed=args.seed,
            batch_size=sub.batch_size or manifest.data.batch_size,
            eval_batch_size=sub.eval_batch_size or manifest.data.eval_batch_size,
        ),
        device=device,
    )
    train_loader = loaders["train"]
    spec = ModelSpec(
        vocab_size=infer_vocab_size(manifest.data),
        max_seq_len=infer_max_seq_len(manifest.data),
        maximum_model_state_elements=manifest.model_state.maximum_elements,
    )
    model = sub.build_model(spec).to(device)
    bundle = sub.build_optimizer(model, OptimizerSpec(60.0, device.type))
    opt = bundle.optimizer

    curve = []
    it = iter(train_loader)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        input_ids, targets, mask, positions = prepare_batch(batch, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=getattr(torch, manifest.runtime.dtype)
        ):
            logits, _ = model(input_ids, attention_mask=mask)
            idx = torch.arange(logits.shape[0], device=device)[:, None]
            token_logits = logits[idx, positions.clamp_min(0)].float()
            valid = targets != -100
            loss = F.cross_entropy(token_logits[valid], targets[valid])
            pred = token_logits.argmax(-1)
            acc = (((pred == targets) | ~valid).all(dim=1)).float().mean().item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), manifest.runtime.grad_clip)
        opt.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            curve.append([step, round(loss.item(), 4), round(acc, 4)])

    row = {
        "submission": args.submission,
        "manifest": Path(args.manifest).stem,
        "steps": args.steps,
        "seed": args.seed,
        "note": args.note,
        "kind": "DIAGNOSTIC",
        "train_curve": curve,
        "train_exact_final": curve[-1][2],
        "seconds": round(time.time() - t0, 1),
        "splits": {},
    }
    for name in args.splits:
        if name in loaders:
            row["splits"][name] = measure(model, loaders[name], device, manifest)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
