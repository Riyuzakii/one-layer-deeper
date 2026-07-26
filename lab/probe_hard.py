#!/usr/bin/env python
"""LAB ONLY -- make the DISCRETE solution the one the optimiser descends into.

`alu-depth` established that `DigitALU` learns a continuous relaxation riding
the W x 10 simplex: `train_exact_hard` (every inter-step state snapped to its
argmax) is 0.000-0.012 at EVERY chain length from 257 steps down to 39, while
the constructed solution survives snapping at 1.000.  Ranking on `train_exact`
actively selects for that failure mode.

So the target is not "fit the data", it is "fit the data with one-hot states".
This probe holds `alu-depth`'s cheap graph fixed (`--mul-mode tree
--reduce-mode quotient`, 39 sequential steps, ceiling 1.000) and varies only
the pressure applied to the states:

  --state-ent W    penalise the entropy of EVERY inter-step state, every step.
                   This is the direct lever: it costs nothing per step and it
                   makes soft states expensive rather than free.
  --sharp-target T same, but as a hinge on max-probability -- pressure only
                   while a state is softer than T, none once it is sharp.
  --tau-final X    temperature annealing that ENDS hard (X << 1), not merely
                   sharpened.
  --gumbel G       Gumbel-softmax noise, annealed to the discrete limit
                   alongside tau.
  --hard-at F      switch to straight-through after fraction F -- an anneal
                   INTO ST rather than ST from scratch (which alu-depth
                   falsified at every depth).
  --tau-mul/add/sub  per-module temperature.

Everything here is a training-procedure change: `--construct` is unchanged and
must still report 1.000, and every reported number is `train_exact_hard`.

Self-generated operands only; nothing under data/generated/ is opened.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_alu_depth import DigitALU, digits_le  # noqa: E402


class HardALU(DigitALU):
    """`alu-depth`'s graph, unchanged, with hooks for state-level pressure.

    Only `_sm` is overridden, so every reduce/mul/scan mode still works and the
    constructed ceiling is untouched.
    """

    def __init__(self, *a, **kw):
        self.noise = 0.0
        self.ent_sum = None
        self.ent_n = 0
        self.tau_mod = {}
        self.sharp_target = 0.0
        super().__init__(*a, **kw)

    def _sm(self, logits):
        p = F.softmax(logits / self.tau, -1)
        if getattr(self, "stat", None) is not None and p.dim() > 1:
            self.stat[0] += p.max(-1).values.mean().item()
            self.stat[1] += 1
        if self.noise > 0.0 and self.training:
            u = torch.rand_like(logits).clamp_(1e-9, 1 - 1e-9)
            p = F.softmax((logits - self.noise * torch.log(-torch.log(u)))
                          / self.tau, -1)
        # accumulate STATE entropy (batched softmaxes only -- the three learned
        # constants are 1-D and are not inter-step states)
        if self.ent_sum is not None and p.dim() > 1 and self.training:
            if self.sharp_target > 0:
                # hinge: pressure only while the state is softer than the
                # target, none once it is sharp.  Avoids the over-sharpening
                # that plain entropy pressure causes.
                self.ent_sum = self.ent_sum + \
                    (self.sharp_target - p.max(-1).values).clamp_min(0).mean()
            else:
                self.ent_sum = self.ent_sum + \
                    -(p * p.clamp_min(1e-9).log()).sum(-1).mean()
            self.ent_n += 1
        if self.hard:
            h = F.one_hot(p.argmax(-1), p.shape[-1]).to(p.dtype)
            p = h + p - p.detach()
        return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulus", type=int, default=323)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--train-x", type=int, default=250)
    ap.add_argument("--reduce", type=int, default=11)
    ap.add_argument("--reduce-mode", default="quotient",
                    choices=["serial", "binary", "quotient"])
    ap.add_argument("--mul-mode", default="tree", choices=["horner", "tree"])
    ap.add_argument("--scan-mode", default="serial", choices=["serial", "prefix"])
    ap.add_argument("--max-quot", type=int, default=10)
    ap.add_argument("--carry", type=int, default=2)
    ap.add_argument("--borrow", type=int, default=2)
    # ---- the levers
    ap.add_argument("--state-ent", type=float, default=0.0,
                    help="weight on the mean entropy of every inter-step state")
    ap.add_argument("--state-ent-warm", type=float, default=0.0,
                    help="ramp the state-entropy weight in over this fraction")
    ap.add_argument("--sharp-target", type=float, default=0.0,
                    help="hinge form: penalise only while max-prob < this")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--tau-final", type=float, default=None)
    ap.add_argument("--gumbel", type=float, default=0.0)
    ap.add_argument("--gumbel-final", type=float, default=0.0)
    ap.add_argument("--hard-at", type=float, default=None,
                    help="anneal INTO straight-through at this fraction")
    ap.add_argument("--identity-init", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--construct", action="store_true")
    ap.add_argument("--jsonl", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    modulus, S = args.modulus, args.slots
    units = [x for x in range(1, modulus) if math.gcd(x, modulus) == 1]
    g = torch.Generator().manual_seed(args.split_seed)
    perm = torch.randperm(len(units), generator=g).tolist()
    train_x = [units[i] for i in perm[: args.train_x]]
    held_x = [units[i] for i in perm[args.train_x:]]
    device = torch.device(args.device)

    def tensors(xs):
        inp = torch.zeros(len(xs), S, 10)
        tgt = torch.zeros(len(xs), S, dtype=torch.long)
        for r, x in enumerate(xs):
            for i, d in enumerate(digits_le(x, S)):
                inp[r, i, d] = 1.0
            for i, d in enumerate(digits_le((x * x) % modulus, S)):
                tgt[r, i] = d
        return inp.to(device), tgt.to(device)

    xin, xt = tensors(train_x)
    hin, ht = tensors(held_x)
    W = S + 1
    ndig = torch.zeros(W, 10)
    for i, d in enumerate(digits_le(modulus, W)):
        ndig[i, d] = 1.0
    ndig = ndig.to(device)

    model = HardALU(S, args.carry, args.borrow, args.reduce, args.tau,
                    args.identity_init, False, args.reduce_mode,
                    args.max_quot, args.mul_mode, args.scan_mode).to(device)
    model.sharp_target = args.sharp_target
    if args.construct:
        model.construct()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] N={modulus} S={S} mul={args.mul_mode} "
          f"reduce={args.reduce_mode} scan={args.scan_mode} "
          f"params={n_par:,} train={len(train_x)} held={len(held_x)}",
          flush=True)

    @torch.no_grad()
    def evaluate(inp, tgt, discrete=False):
        model.eval()
        was, model.hard = model.hard, True if discrete else model.hard
        lg = model(inp, ndig)
        ok = (lg.argmax(-1) == tgt).all(dim=1).float().mean().item()
        ce = F.cross_entropy(lg.reshape(-1, 10), tgt.reshape(-1)).item()
        model.hard = was
        model.train()
        return ok, ce

    @torch.no_grad()
    def sharpness():
        model.stat = [0.0, 0]
        evaluate(xin[:256], xt[:256])
        s = model.stat[0] / max(model.stat[1], 1)
        model.stat = None
        return s

    if args.construct:
        tr, _ = evaluate(xin, xt)
        trh, _ = evaluate(xin, xt, discrete=True)
        heh, _ = evaluate(hin, ht, discrete=True)
        print(f"[{args.tag}] CONSTRUCTED train_exact={tr:.3f} "
              f"train_exact_hard={trh:.3f} held_exact_hard={heh:.3f} "
              f"state_sharpness={sharpness():.3f}", flush=True)
        return 0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd, betas=(0.9, 0.95))
    t0 = time.time()
    best_hard = 0.0
    for step in range(1, args.steps + 1):
        f = step / args.steps
        if args.tau_final is not None:
            model.tau = math.exp((1 - f) * math.log(args.tau)
                                 + f * math.log(args.tau_final))
        if args.gumbel > 0:
            model.noise = args.gumbel + (args.gumbel_final - args.gumbel) * f
        if args.hard_at is not None:
            model.hard = f >= args.hard_at
        model.ent_sum, model.ent_n = torch.zeros((), device=device), 0
        logits = model(xin, ndig)
        loss = F.cross_entropy(logits.reshape(-1, 10), xt.reshape(-1))
        main_ce = loss.item()
        ent = model.ent_sum / max(model.ent_n, 1)
        if args.state_ent > 0:
            w = args.state_ent
            if args.state_ent_warm > 0:
                w = w * min(1.0, f / args.state_ent_warm)
            loss = loss + w * ent
        model.ent_sum = None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            tr, tr_ce = evaluate(xin, xt)
            trh, _ = evaluate(xin, xt, discrete=True)
            he, _ = evaluate(hin, ht)
            heh, _ = evaluate(hin, ht, discrete=True)
            best_hard = max(best_hard, trh)
            print(f"[{args.tag}] step={step:>5} tau={model.tau:.3f} "
                  f"ce={main_ce:.4f} ent={ent.item():.3f} "
                  f"train_exact={tr:.3f} train_exact_hard={trh:.3f} "
                  f"held_exact={he:.3f} held_exact_hard={heh:.3f} "
                  f"sharp={sharpness():.3f} ({time.time()-t0:.0f}s)", flush=True)
    tr, _ = evaluate(xin, xt)
    trh, _ = evaluate(xin, xt, discrete=True)
    he, _ = evaluate(hin, ht)
    heh, _ = evaluate(hin, ht, discrete=True)
    best_hard = max(best_hard, trh)
    sh = sharpness()
    print(f"[{args.tag}] FINAL train_exact={tr:.3f} "
          f"train_exact_hard={trh:.3f} best_hard={best_hard:.3f} "
          f"held_exact={he:.3f} held_exact_hard={heh:.3f} "
          f"sharp={sh:.3f} ({time.time()-t0:.0f}s)", flush=True)
    if args.jsonl:
        with open(args.jsonl, "a") as fh:
            fh.write(json.dumps({"tag": args.tag, "argv": sys.argv[1:],
                                 "train_exact": tr, "train_exact_hard": trh,
                                 "best_hard": best_hard, "held_exact": he,
                                 "held_exact_hard": heh, "sharpness": sh,
                                 "secs": round(time.time() - t0, 1)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
