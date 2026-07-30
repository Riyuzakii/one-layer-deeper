#!/usr/bin/env python
"""Offline screen for the PLAN2 §3.6 sequential-RNN candidate.

Reports, side by side and at every configuration (BRIEF2 §6.3, "report the metric row,
not the cell"):

  * ``train_exact``  — exact-example accuracy on the training split
  * ``held_exact``   — exact-example accuracy on the ``test`` split
  * ``diversity``    — distinct predicted answer strings / number of held-out examples
  * ``top_share``    — share of held-out examples receiving the single most common
                       predicted answer.  A constant map reads diversity ~= 1/n and
                       top_share ~= 1.0; this is the collapse detector.
  * ``held_ce``      — held-out cross-entropy, recorded but *not* ranked on
                       (RESUME.md: label smoothing moves it with zero algebraic content)

This is a LAB DIAGNOSTIC, not a submission: it uses its own training loop, which a
submission may not (BRIEF.md §4.4).  Everything it measures is a property of the
model's *predictions*; no dataset record is ever read, printed or summarised here.

Usage:
  probe_rnn.py --dataset e5 --d-h 64 --steps 2000 --seeds 0 1
  probe_rnn.py --dataset e5 --d-h 64 --steps 2000 --lr 0      # the mandatory control
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmark.api import ModelSpec, OptimizerSpec  # noqa: E402
from benchmark.batches import prepare_batch  # noqa: E402
from data import infer_max_seq_len, infer_vocab_size, make_dataloaders  # noqa: E402
from data.config import DataConfig  # noqa: E402

sys.path.insert(0, str(REPO / "lab"))
from make_manifest import DATA_ROOTS  # noqa: E402


def load_submission(path: Path):
    spec = importlib.util.spec_from_file_location("_probe_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def evaluate(model, loader, device, amp: bool, max_batches: int | None = None):
    """Exact accuracy + prediction-collapse statistics on one split."""
    model.eval()
    n_rows = 0
    n_exact = 0
    ce_sum = 0.0
    ce_n = 0
    answer_counts: dict[tuple, int] = {}
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        ids, targets, mask, positions = prepare_batch(batch, device)
        ctx = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if amp
            else torch.autocast(device_type=device.type, enabled=False)
        )
        with ctx:
            logits, _ = model(ids, attention_mask=mask)
        logits = logits.float()
        rows = torch.arange(logits.shape[0], device=device)[:, None]
        token_logits = logits[rows, positions.clamp_min(0)]
        valid = targets != -100
        preds = token_logits.argmax(dim=-1)
        exact = ((preds == targets) | ~valid).all(dim=1)
        has = valid.any(dim=1)
        n_rows += int(has.sum())
        n_exact += int(exact[has].sum())
        flat_logits = token_logits[valid]
        flat_labels = targets[valid]
        ce_sum += float(F.cross_entropy(flat_logits, flat_labels, reduction="sum"))
        ce_n += int(valid.sum())
        # collapse detector: the predicted answer *string*, masked to valid slots
        masked = torch.where(valid, preds, torch.full_like(preds, -1)).cpu()
        for row in masked.tolist():
            key = tuple(row)
            answer_counts[key] = answer_counts.get(key, 0) + 1
    model.train()
    if n_rows == 0:
        return dict(exact=0.0, diversity=0.0, top_share=1.0, ce=float("nan"), n=0)
    return dict(
        exact=n_exact / n_rows,
        diversity=len(answer_counts) / n_rows,
        top_share=max(answer_counts.values()) / n_rows,
        ce=ce_sum / max(1, ce_n),
        n=n_rows,
    )


def cache_split(loader, device, width: int | None = None):
    """Materialise one split as padded GPU tensors.

    The evaluator's DataLoader respawns its 2 workers every epoch (RESUME.md), which
    on these tiny splits dominates the step time and makes a screening sweep needlessly
    slow.  Caching is legal here because this is a lab probe, not a submission; no
    record content is inspected, only moved to the device.
    """
    chunks = []
    for batch in loader:
        ids, targets, mask, positions = prepare_batch(batch, device)
        chunks.append((ids, targets, mask, positions))
    w_in = max(c[0].shape[1] for c in chunks)
    w_tg = max(c[1].shape[1] for c in chunks)
    if width is not None:
        w_in = max(w_in, width)

    def pad(t, w, value):
        if t.shape[1] == w:
            return t
        out = torch.full(
            (t.shape[0], w), value, dtype=t.dtype, device=t.device
        )
        out[:, : t.shape[1]] = t
        return out

    ids = torch.cat([pad(c[0], w_in, 0) for c in chunks])
    targets = torch.cat([pad(c[1], w_tg, -100) for c in chunks])
    mask = torch.cat([pad(c[2], w_in, False) for c in chunks])
    positions = torch.cat([pad(c[3], w_tg, -1) for c in chunks])
    return ids, targets, mask, positions


@torch.no_grad()
def evaluate_cached(model, cache, device, batch_size=4096):
    model.eval()
    ids_all, tgt_all, mask_all, pos_all = cache
    n_rows = 0
    n_exact = 0
    ce_sum = 0.0
    ce_n = 0
    answer_counts: dict[tuple, int] = {}
    for start in range(0, ids_all.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        ids, targets, mask, positions = (
            ids_all[sl],
            tgt_all[sl],
            mask_all[sl],
            pos_all[sl],
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(ids, attention_mask=mask)
        logits = logits.float()
        rows = torch.arange(logits.shape[0], device=device)[:, None]
        token_logits = logits[rows, positions.clamp_min(0)]
        valid = targets != -100
        preds = token_logits.argmax(dim=-1)
        exact = ((preds == targets) | ~valid).all(dim=1)
        has = valid.any(dim=1)
        n_rows += int(has.sum())
        n_exact += int(exact[has].sum())
        ce_sum += float(
            F.cross_entropy(token_logits[valid], targets[valid], reduction="sum")
        )
        ce_n += int(valid.sum())
        masked = torch.where(valid, preds, torch.full_like(preds, -1)).cpu()
        for row in masked.tolist():
            key = tuple(row)
            answer_counts[key] = answer_counts.get(key, 0) + 1
    model.train()
    if n_rows == 0:
        return dict(exact=0.0, diversity=0.0, top_share=1.0, ce=float("nan"), n=0)
    return dict(
        exact=n_exact / n_rows,
        diversity=len(answer_counts) / n_rows,
        top_share=max(answer_counts.values()) / n_rows,
        ce=ce_sum / max(1, ce_n),
        n=n_rows,
    )


def run_one(args, seed: int, overrides: dict) -> dict:
    device = torch.device("cuda")
    torch.manual_seed(seed)

    module = load_submission(Path(args.submission))
    for key, value in overrides.items():
        if hasattr(module, key):
            setattr(module, key, value)

    data_root = DATA_ROOTS[args.dataset]
    config = DataConfig(
        kind="squaring_mod",
        data_root=data_root,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        shuffle_train=True,
        shuffle_eval=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        seed=45,
    )
    loaders = make_dataloaders(config, device=device)
    vocab = infer_vocab_size(config)
    seq_len = infer_max_seq_len(config)

    spec = ModelSpec(
        vocab_size=vocab, max_seq_len=seq_len, maximum_model_state_elements=500_000_000
    )
    model = module.build_model(spec).to(device)
    bundle = module.build_optimizer(model, OptimizerSpec(60.0, "cuda"))
    optimizer = bundle.optimizer
    if args.lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    n_params = sum(p.numel() for p in model.parameters())
    train_cache = cache_split(loaders["train"], device)
    held_key = "test" if "test" in loaders else sorted(loaders)[0]
    held_cache = cache_split(loaders[held_key], device, width=train_cache[0].shape[1])
    n_train = train_cache[0].shape[0]
    generator = torch.Generator(device=device).manual_seed(seed)

    t0 = time.perf_counter()
    curve = []
    for step in range(1, args.steps + 1):
        sel = torch.randint(
            0, n_train, (min(args.batch_size, n_train),), device=device,
            generator=generator,
        )
        ids, targets, mask, positions = (t[sel] for t in train_cache)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(ids, attention_mask=mask)
            rows = torch.arange(logits.shape[0], device=device)[:, None]
            token_logits = logits[rows, positions.clamp_min(0)].float()
            valid = targets != -100
            loss = F.cross_entropy(token_logits[valid], targets[valid])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % max(1, args.steps // 8) == 0 or step == 1:
            with torch.no_grad():
                preds = token_logits.argmax(dim=-1)
                tr = float(((preds == targets) | ~valid).all(dim=1).float().mean())
            point = [step, round(float(loss.detach()), 4), round(tr, 4)]
            if args.eval_every:
                # optional held-out evaluation at each checkpoint.  Off by default so
                # that runs stay comparable with the ones already recorded; turn it on
                # when the held-out *trajectory* (not just its endpoint) is the question.
                point.append(
                    round(
                        evaluate_cached(
                            model, held_cache, device, args.eval_batch_size
                        )["exact"],
                        4,
                    )
                )
            curve.append(point)
    train_seconds = time.perf_counter() - t0

    train_stats = evaluate_cached(model, train_cache, device, args.eval_batch_size)
    held_stats = evaluate_cached(model, held_cache, device, args.eval_batch_size)

    return dict(
        dataset=args.dataset,
        seed=seed,
        steps=args.steps,
        lr=args.lr if args.lr is not None else module.LR,
        overrides=overrides,
        params=n_params,
        seq_len=seq_len,
        ms_per_step=1e3 * train_seconds / args.steps,
        train_n=train_stats["n"],
        train_diversity=round(train_stats["diversity"], 4),
        train_exact=round(train_stats["exact"], 4),
        held_exact=round(held_stats["exact"], 4),
        held_diversity=round(held_stats["diversity"], 4),
        held_top_share=round(held_stats["top_share"], 4),
        held_ce=round(held_stats["ce"], 4),
        held_n=held_stats["n"],
        curve=curve,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="e5")
    ap.add_argument(
        "--submission",
        default=str(REPO / "submissions" / "p2-sequential-rnn" / "submission.py"),
    )
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-batch-size", type=int, default=4096)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--lr", type=float, default=None, help="override; use 0 for control")
    ap.add_argument(
        "--eval-every",
        action="store_true",
        help="also evaluate held-out at each curve checkpoint (appends a 4th column)",
    )
    ap.add_argument("--d-h", type=int, nargs="+", default=[64])
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--align", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="extra module-constant overrides, e.g. K_STEPS=20 GRAD_NOISE_ETA=0.0",
    )
    ap.add_argument("--out", default=str(REPO / "lab" / "probe_rnn.jsonl"))
    args = ap.parse_args()

    extra = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        try:
            extra[key] = int(raw)
        except ValueError:
            try:
                extra[key] = float(raw)
            except ValueError:
                extra[key] = raw

    out = Path(args.out)
    for d_h in args.d_h:
        for seed in args.seeds:
            overrides = dict(
                D_H=d_h,
                D_EMB=d_h,
                CHANNELS=d_h,
                LOOPS=args.loops,
                ALIGN=bool(args.align),
                **extra,
            )
            row = run_one(args, seed, overrides)
            row["tag"] = args.tag
            with out.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                f"[{args.tag}] {args.dataset} d_h={d_h:<4} seed={seed} "
                f"lr={row['lr']:<8g} params={row['params']:<8} "
                f"train={row['train_exact']:.4f} held={row['held_exact']:.4f} "
                f"div={row['held_diversity']:.4f} top={row['held_top_share']:.4f} "
                f"ce={row['held_ce']:.3f} {row['ms_per_step']:.1f}ms/step",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
