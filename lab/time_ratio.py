#!/usr/bin/env python
"""Step-cost RATIO at Hard's shape, measured by interleaving the candidates.

BRIEF2 §7: absolute wall clock does not transfer off sm_107/sm_103, and cost
ratios do not transfer across tier shapes either -- so every timing claim has to
be a ratio to a fixed reference model, taken at the tier's own shape.  This box
also has three sibling agents on the same GPU, so a ratio measured by two
separate runs is not a ratio at all.  This script interleaves the measurements
round-robin so both candidates see the same contention.

Reference = the official baseline architecture at D_MODEL=128, batch 512, L=19.
The hosted H100 calibration is 38.6 ms/step for a D=128 x 8-loop stack, i.e.
~4.8 ms/step per loop, so the reference here converts to H100 milliseconds.
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmark.api import ModelSpec, OptimizerSpec  # noqa: E402

H100_MS_PER_LOOP = 38.6 / 8.0     # hosted Hard run, D=128 recurrent stack


def load(path: str):
    spec = importlib.util.spec_from_file_location("sub", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make(path: str, vocab: int, seq: int):
    mod = load(path)
    ms = ModelSpec(vocab_size=vocab, max_seq_len=seq,
                   maximum_model_state_elements=500_000_000)
    model = mod.SUBMISSION.build_model(ms).cuda()
    bundle = mod.SUBMISSION.build_optimizer(
        model, OptimizerSpec(training_time_seconds=3600.0, device_type="cuda"))
    bs = mod.SUBMISSION.batch_size or 512
    return mod, model, bundle, bs


def one_step(mod, model, bundle, ids, mask, tgt):
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, aux = model(ids, attention_mask=mask)
        flat = logits[:, : tgt.shape[1]].reshape(-1, logits.shape[-1]).float()
        if mod.SUBMISSION.training_loss is None:
            loss = F.cross_entropy(flat, tgt.reshape(-1))
        else:
            loss = mod.SUBMISSION.training_loss(flat, tgt.reshape(-1), aux)
    bundle.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    bundle.optimizer.step()
    return float(loss.detach())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seq", type=int, default=19)
    ap.add_argument("--vocab", type=int, default=17)
    ap.add_argument("--tgt", type=int, default=7)
    args = ap.parse_args()

    built = []
    for p in args.subs:
        mod, model, bundle, bs = make(p, args.vocab, args.seq)
        ids = torch.randint(0, args.vocab, (bs, args.seq), device="cuda")
        mask = torch.ones(bs, args.seq, dtype=torch.bool, device="cuda")
        tgt = torch.randint(0, 10, (bs, args.tgt), device="cuda")
        built.append((p, mod, model, bundle, bs, ids, mask, tgt))
        for _ in range(args.warmup):
            one_step(mod, model, bundle, ids, mask, tgt)
        torch.cuda.synchronize()

    times = {p: [] for p, *_ in built}
    for _ in range(args.reps):
        for p, mod, model, bundle, bs, ids, mask, tgt in built:
            torch.cuda.synchronize()
            t0 = time.time()
            one_step(mod, model, bundle, ids, mask, tgt)
            torch.cuda.synchronize()
            times[p].append(time.time() - t0)

    ref = statistics.median(times[args.subs[0]])
    print(f"interleaved, {args.reps} reps, L={args.seq}, "
          f"reference = {args.subs[0]}")
    for p, mod, model, bundle, bs, *_ in built:
        med = statistics.median(times[p])
        h100 = med / ref * H100_MS_PER_LOOP
        print(f"  {Path(p).parent.name:<34} batch={bs:<5} "
              f"median={med * 1000:9.1f} ms  ratio={med / ref:8.2f}x  "
              f"implied H100 ms/step={h100:9.1f}  "
              f"steps in 1800s={1800 / (h100 / 1000):10.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
