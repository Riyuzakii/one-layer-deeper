#!/usr/bin/env python
"""Wall-clock ratios at HARD's shape, not Easy's.

`plan2/sequential-rnn` measured that cost gaps collapse toward 1 at Easy's
shape (L=13, batch 128) because everything is kernel-launch bound there: the
Neural GPU is 22x the fused LSTM at L=21/batch 512 but only 1.7x at e5's shape.
So any wall-clock number used in a falsifier has to be taken at Hard's shape and
reported as a RATIO to a fixed reference (this box is sm_107; absolutes do not
transfer, ratios were validated to).

Reference (1.00x) = the official baseline transformer at batch 512, L=21.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from monoid import MonoidALU, int_to_digits  # noqa: E402


def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / iters * 1000.0


def reference(device, batch, length):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ref", REPO / "submissions" / "baseline_adamw" / "submission.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from benchmark import ModelSpec

    m = mod.build_model(ModelSpec(17, length, 500_000_000)).to(device)
    ids = torch.randint(0, 17, (batch, length), device=device)
    mask = torch.ones(batch, length, dtype=torch.bool, device=device)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _ = m(ids, mask)
            loss = lg.float().logsumexp(-1).mean()
        loss.backward()

    return timeit(step)


def monoid(device, batch, slots, d, impl):
    alu = MonoidALU(slots, d=d, family="colsoftmax", impl=impl).to(device)
    x = int_to_digits([12345678] * batch, alu.L, device)
    n = int_to_digits([99999989] * batch, alu.L, device)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = alu(x, n)
            loss = y.float().log().mean()
        loss.backward()

    return timeit(step, iters=10, warmup=3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seq", type=int, default=21, help="Hard's max_seq_len")
    ap.add_argument("--slots", type=int, default=10, help="digit slots at Hard scale")
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)

    ref = reference(device, args.batch, args.seq)
    print(f"reference transformer  batch={args.batch} L={args.seq}: "
          f"{ref:.1f} ms/step  = 1.00x")
    for impl in ("scan", "serial"):
        try:
            ms = monoid(device, args.batch, args.slots, args.d, impl)
            print(f"MonoidALU impl={impl:<6} S={args.slots} L={2*args.slots} d={args.d}: "
                  f"{ms:.1f} ms/step  = {ms/ref:.2f}x")
        except torch.OutOfMemoryError:
            print(f"MonoidALU impl={impl}: OOM at batch {args.batch}")
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
