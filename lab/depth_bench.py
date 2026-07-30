#!/usr/bin/env python
"""LAB ONLY -- optimizer steps per second for each DigitALU graph shape.

The coordinator's step-famine number ("tier-faithful Easy affords 14-18
optimizer steps") makes throughput, not just soft-step count, the quantity that
decides whether a shape is usable.  This times ONE optimizer step (forward +
backward + AdamW) of the readout alone, under the manifests' bf16 autocast and
batch size, and converts it to the step budget each tier affords.

Caveats, stated so the numbers are not over-read:
  * this is the READOUT only.  A submission also pays for the parser and the
    T-loop, so the real step count is lower.
  * the evaluator's clock starts at import and eval takes a further half of the
    training budget, neither of which is charged here.
  * this box is sm_107 and is SHARED with sibling agents; absolute wall clock
    does not transfer to the scoring H100.  The RATIO between shapes does, since
    the workload is kernel-launch bound and every shape launches the same kind
    of kernel.  Run with --quiet-check to see the contention estimate.

Self-generated values only; nothing under data/generated/ is opened.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from probe_alu import DigitALU, digits_le

TIERS = {"Easy": 60.0, "Medium": 600.0, "Hard": 3600.0}


def bench(S: int, mul: str, red: str, scan: str, batch: int, iters: int,
          warmup: int, device: torch.device, amp: bool,
          modulus: int) -> tuple[float, dict]:
    model = DigitALU(S, mul_mode=mul, reduce_mode=red, scan_mode=scan).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    s = F.one_hot(torch.randint(0, 10, (batch, S), device=device), 10).float()
    tgt = torch.randint(0, 10, (batch, S), device=device)
    nd = torch.zeros(S + 1, 10, device=device)
    for i, d in enumerate(digits_le(modulus, S + 1)):
        nd[i, d] = 1.0
    ctx = torch.autocast("cuda", torch.bfloat16) if amp else \
        torch.autocast("cuda", enabled=False)

    def one():
        with ctx:
            lg = model(s, nd)
            loss = F.cross_entropy(lg.reshape(-1, 10).float(), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        one()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters, model.depth()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--modulus", type=int, nargs="+", default=[323, 10403])
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--cells", nargs="+",
                    default=["horner:serial", "tree:serial", "horner:binary",
                             "tree:binary", "horner:quotient", "tree:quotient"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)

    print(f"batch={args.batch} amp={'off' if args.no_amp else 'bf16'} "
          f"iters={args.iters} device={torch.cuda.get_device_name(device)}\n")
    hdr = (f"{'S':>2} {'modulus':>8} {'shape':>25} {'depth':>6} {'ms/step':>9} "
           f"{'steps/s':>8} " + " ".join(f"{t:>7}" for t in TIERS))
    print(hdr)
    print("-" * len(hdr))
    for S, mod in zip(args.slots, args.modulus):
        base = None
        for cell in args.cells:
            parts = cell.split(":")
            mul, red = parts[0], parts[1]
            scan = parts[2] if len(parts) > 2 else "serial"
            dt, d = bench(S, mul, red, scan, args.batch, args.iters,
                          args.warmup, device, not args.no_amp, mod)
            base = base if base is not None else dt
            budget = " ".join(f"{TIERS[t] / dt:>7.0f}" for t in TIERS)
            print(f"{S:>2} {mod:>8} {cell:>25} {d['main']:>6} "
                  f"{dt * 1e3:>9.2f} {1 / dt:>8.1f} {budget}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
