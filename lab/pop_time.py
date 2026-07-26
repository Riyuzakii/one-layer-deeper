#!/usr/bin/env python
"""LAB ONLY -- ms/step vs replica count, in ONE process.

The GPU is shared with a sibling branch, so an inter-process sweep measures
whoever else is running.  This driver interleaves the replica counts across
several rounds and reports the MINIMUM per-step time for each, which is the
only contention-robust statistic available on a shared device.

Times a full training step: teacher-forced reference tape (no_grad) + forced
forward + backward + AdamW step, exactly as lab/probe_pop.py runs it.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_alu_depth import digits_le                      # noqa: E402
from probe_pop import PopALU                               # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--slow-quot", action="store_true")
    ap.add_argument("--sel", action="store_true",
                    help="also run the free-running mixture forward")
    ap.add_argument("--modulus", type=int, default=10403)
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device)
    S, N = args.slots, args.modulus
    W = S + 1
    nd = torch.zeros(W, 10)
    for i, d in enumerate(digits_le(N, W)):
        nd[i, d] = 1.0
    nd = nd.to(dev)
    bi = torch.zeros(args.batch, S, 10, device=dev)
    bt = torch.randint(0, 10, (args.batch, S), device=dev)
    for r in range(args.batch):
        for i, d in enumerate(digits_le(1 + r, S)):
            bi[r, i, d] = 1.0

    pops = [int(p) for p in args.pops.split(",")]
    best = {p: math.inf for p in pops}
    mem = {p: 0.0 for p in pops}
    ref = PopALU(1, S, 2, 2, 10).to(dev)
    ref.fast = not args.slow_quot
    ref.construct()
    for p in ref.parameters():
        p.requires_grad_(False)

    for rnd in range(args.rounds):
        for P in pops:
            torch.manual_seed(0)
            m = PopALU(P, S, 2, 2, 10).to(dev)
            m.fast = not args.slow_quot
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            torch.cuda.reset_peak_memory_stats()

            def one():
                with torch.no_grad():
                    ref.mode, ref.tape = 'record', []
                    ref(bi, nd)
                    ref.mode = None
                m.mode, m.tape = 'force', ref.tape
                m.tf_loss, m.tf_n, m.tf_p = torch.zeros(P, device=dev), 0, 1.0
                m(bi, nd)
                m.mode = None
                loss = (m.tf_loss / max(m.tf_n, 1)).mean()
                if args.sel:
                    lg = m(bi, nd)
                    mp = m.mix_probs(lg).clamp_min(1e-9)
                    loss = loss + torch.nn.functional.nll_loss(
                        mp.log().reshape(-1, 10), bt.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            for _ in range(3):
                one()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.iters):
                one()
            torch.cuda.synchronize()
            ms = 1000 * (time.time() - t0) / args.iters
            best[P] = min(best[P], ms)
            mem[P] = max(mem[P], torch.cuda.max_memory_allocated() / 2 ** 30)
            del m, opt
            torch.cuda.empty_cache()
        print(f"# round {rnd}: " + " ".join(f"{p}:{best[p]:.1f}" for p in pops),
              flush=True)

    print(f"\n# batch={args.batch} slow_quot={args.slow_quot} sel={args.sel}")
    print("pop  ms/step  ms/replica  vs_P1   peak_GiB")
    for P in pops:
        print(f"{P:>4} {best[P]:8.1f} {best[P] / P:10.2f} "
              f"{best[P] / best[pops[0]]:6.2f}x {mem[P]:9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
