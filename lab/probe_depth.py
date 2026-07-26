#!/usr/bin/env python
"""LAB ONLY -- the depth controller in isolation.

`alu-compose` measured that with the squaring step CONSTRUCTED (exact) the
pipeline still certifies only MAX_T = 2 on Easy and MAX_T = 0 on Medium, and
localised the whole loss to the component that decides *how many times* to apply
the step.  Two causes were diagnosed there:

  (a) coverage in the T field -- every head is indexed by (place, digit) of T,
      and the tier's training T values never present the pairs the ladder needs;
  (b) a flat loss inside the correct bin -- the soft window is insensitive once
      `loc` is anywhere in the right bin.

This probe holds the parser and the ALU at their construction (LAB DIAGNOSTIC,
never in a submission) and trains ONLY the controller, so the controller is the
only thing measured.  On top of `probe_compose.py --mode select` it adds the
levers that were named but never attempted:

  --train-loops L     decouple the training mixture depth from max(train T).
                      probe_compose used L = max(train T), and `ws[-1] += rest`
                      dumps unhalted mass on the last index -- so "never halt"
                      is EXACTLY CORRECT for the deepest training T and carries
                      no gradient.  This flag removes that degenerate optimum.
  --halt-pen w        penalise unspent halting mass (PonderNet-style).
  --cons w            SELF-CONSISTENCY: the controller's own increment orbit.
                      w(inc(r)) must equal shift_right(w(r)) for every register
                      r, which needs no labels and no extra data.  One anchored
                      T plus this chain determines the map at every T.
  --cons-j J          orbit length (how far above/below the training T values).
  --hard-sel          straight-through discrete selection (attacks cause (b)).
  --window-margin m   margin loss on `loc` (attacks cause (b)).

Every modulus used here has its lambda and its ladder collapse printed, because
lambda(323) = lcm(16,18) = 144 makes 4, 16 and 64 applications THE SAME MAP and
any depth number measured on e1/e2 is therefore uninterpretable.

Nothing under data/generated/ is opened; prompts and targets are synthesised
from the public generator spec.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from math import gcd

import torch
import torch.nn.functional as F
from torch import nn

from probe_compose import (LADDER, Composed, CounterSelector, Selector,
                           build_pool, make_batch, targets, window)

BIG = 30.0


# --------------------------------------------------------------------------
# number theory: which ladder rungs does this modulus collapse?
# --------------------------------------------------------------------------
def factor(n: int):
    f = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            f.append(d)
            n //= d
        d += 1
    if n > 1:
        f.append(n)
    return f


def carmichael_semiprime(n: int):
    f = factor(n)
    if len(f) != 2:
        return None
    p, q = f
    return (p - 1) * (q - 1) // gcd(p - 1, q - 1)


def ladder_classes(n: int, ladder=LADDER):
    lam = carmichael_semiprime(n)
    if lam is None:
        return None, None
    cls = {}
    for t in ladder:
        cls.setdefault(pow(2, t, lam), []).append(t)
    return lam, cls


def lam_banner(mods, ladder=LADDER):
    seen = []
    for m in sorted(set(mods)):
        lam, cls = ladder_classes(m, ladder)
        if lam is None:
            seen.append(f"N={m} (not a semiprime)")
            continue
        coll = [v for v in cls.values() if len(v) > 1]
        seen.append(f"N={m} lambda={lam} distinct={len(cls)}/{len(ladder)}"
                    + (f" COLLAPSES={coll}" if coll else " (no collapse)"))
    return " | ".join(seen)


# --------------------------------------------------------------------------
# digit-register arithmetic shared with the ALU (used to build the orbit)
# --------------------------------------------------------------------------
def _st_round(r, hard: bool):
    if not hard:
        return r
    h = F.one_hot(r.argmax(-1), r.shape[-1]).to(r.dtype)
    return h + r - r.detach()


def reg_add(r, alu, unit, hard=False):
    """r + unit, digit-serial with the ALU's OWN learned Tadd/carry0/zero."""
    b, n_t = r.shape[0], r.shape[1]
    one = alu._sm(unit).to(r.dtype)
    zer = alu._sm(alu.zero).to(r.dtype)
    add = torch.stack([one] + [zer] * (n_t - 1), 0)[None].expand(b, n_t, 10)
    c = alu._sm(alu.carry0).to(r.dtype).expand(b, alu.Ca)
    outs = []
    for m in range(n_t):
        o = torch.einsum("bu,bv,bc,uvco->bo", r[:, m], add[:, m], c,
                         alu.Tadd.to(r.dtype))
        outs.append(alu._sm(o[:, :10]))
        c = alu._sm(o[:, 10:])
    return _st_round(torch.stack(outs, 1), hard)


def reg_sub(r, alu, unit, hard=False):
    """r - unit, digit-serial with the ALU's OWN learned Tsub/borrow0/zero."""
    b, n_t = r.shape[0], r.shape[1]
    one = alu._sm(unit).to(r.dtype)
    zer = alu._sm(alu.zero).to(r.dtype)
    sub = torch.stack([one] + [zer] * (n_t - 1), 0)[None].expand(b, n_t, 10)
    c = alu._sm(alu.borrow0).to(r.dtype).expand(b, alu.Cb)
    outs = []
    for m in range(n_t):
        o = torch.einsum("bu,bv,bc,uvco->bo", r[:, m], sub[:, m], c,
                         alu.Tsub.to(r.dtype))
        outs.append(alu._sm(o[:, :10]))
        c = alu._sm(o[:, 10:])
    return _st_round(torch.stack(outs, 1), hard)


# --------------------------------------------------------------------------
# controllers
# --------------------------------------------------------------------------
class CounterSel2(nn.Module):
    """Counted halting, re-derived so the training depth is a free parameter.

    Identical mechanism to probe_compose.CounterSelector (decrement a digit
    register with the ALU's own Tsub/borrow, halt when it matches the ALU's own
    learned zero digit) with three differences that the alu-compose measurements
    argue for:

      * `dump` controls whether unspent halting mass is piled onto the last
        index.  probe_compose always dumps, which makes "never halt" the exact
        optimum for the deepest training T.
      * the chain is computed ONCE and every prefix is reused, so w(r) for the
        whole increment orbit of r costs O(L) rather than O(L^2).
      * `one` init scale is exposed: softmax(randn*0.5) is nearly uniform, so
        the decrement is mush at init and the register never becomes a count.

    Learned: `one` (10), `gain`, `thresh` -> 12 scalars, NONE of them indexed by
    a place or a digit of T.
    """

    def __init__(self, n_t: int, init_scale: float = 0.5,
                 dump: bool = True) -> None:
        super().__init__()
        self.n_t = n_t
        self.dump = dump
        self.one = nn.Parameter(torch.randn(10) * init_scale)
        self.gain = nn.Parameter(torch.tensor(2.0))
        self.thresh = nn.Parameter(torch.tensor(float(n_t) - 0.5))

    @torch.no_grad()
    def construct(self) -> None:
        o = torch.full((10,), -BIG)
        o[1] = BIG
        self.one.copy_(o)
        self.gain.fill_(BIG)
        self.thresh.fill_(self.n_t - 0.5)

    def is_zero(self, c, alu):
        z = alu._sm(alu.zero).to(c.dtype)
        m = torch.einsum("btd,d->b", c, z)[:, None]
        return torch.sigmoid(self.gain.to(c.dtype) * (m - self.thresh.to(c.dtype)))

    def forward(self, st, alu, loops, hard=False):
        c = st[:, : self.n_t]
        rest = torch.ones_like(self.is_zero(c, alu))
        ws = []
        for _ in range(loops):
            c = reg_sub(c, alu, self.one, hard)
            p = self.is_zero(c, alu)
            ws.append(rest * p)
            rest = rest * (1 - p)
        if self.dump:
            ws[-1] = ws[-1] + rest
        return torch.cat(ws, dim=-1)


SEL_KINDS = ("construct", "mlp", "linear", "place", "placev",
             "counter", "counterz", "counter2")


class Depth(Composed):
    """Composed pipeline with a controller that also owns a `unit` digit for
    the self-consistency orbit."""

    def __init__(self, slots, n_t=2, reduce_steps=11, tau=1.0,
                 sel_kind="counter2", one_init=0.5, dump=True):
        super().__init__(slots, n_t, reduce_steps, tau,
                         "construct" if sel_kind == "counter2" else sel_kind)
        if sel_kind == "counter2":
            self.sel = CounterSel2(n_t, one_init, dump)
            self.sel_kind = "counter2"
        # a learned unit digit for the consistency orbit; for the counters it
        # IS the counter's own `one`, so the orbit and the countdown are tied.
        if sel_kind.startswith("counter"):
            self.unit = None
        else:
            self.unit = nn.Parameter(torch.randn(10) * one_init)

    def unit_vec(self):
        return self.sel.one if self.unit is None else self.unit

    def weights(self, st, loops, sigma, hard=False):
        if self.sel_kind.startswith("counter"):
            if isinstance(self.sel, CounterSel2):
                return self.sel(st, self.alu, loops, hard)
            return self.sel(st, self.alu, loops)
        return window(self.sel(st), loops, sigma, st.dtype)

    def loc(self, st, loops, sigma, hard=False):
        if self.sel_kind.startswith("counter"):
            return self.weights(st, loops, sigma, hard).argmax(-1).float()
        return self.sel(st).squeeze(-1)


# --------------------------------------------------------------------------
# self-consistency: w(inc(r)) == shift_right(w(r)), and its `loc` form
# --------------------------------------------------------------------------
def shift_right(w):
    return torch.cat([torch.zeros_like(w[:, :1]), w[:, :-1]], dim=1)


def shift_left(w):
    return torch.cat([w[:, 1:], torch.zeros_like(w[:, :1])], dim=1)


def orbit_regs(reg, model, j_up, j_dn, hard):
    """[dec^j_dn(reg) ... reg ... inc^j_up(reg)] as a list, anchor at index
    j_dn."""
    alu, unit = model.alu, model.unit_vec()
    up, r = [], reg
    for _ in range(j_up):
        r = reg_add(r, alu, unit, hard)
        up.append(r)
    dn, r = [], reg
    for _ in range(j_dn):
        r = reg_sub(r, alu, unit, hard)
        dn.append(r)
    return list(reversed(dn)) + [reg] + up


def cons_loss(model, regs, anchor_i, loops, sigma, space, hard):
    """Teacher flows AWAY from the anchor in both directions; the target is
    detached, so this is a chain of one-step consistency constraints rather
    than a joint smoothness penalty that can be satisfied by collapse."""
    tot = regs[0].new_zeros(())
    if space == "loc":
        locs = [model.sel(r).squeeze(-1) for r in regs]
        for i in range(anchor_i, len(regs) - 1):
            tot = tot + ((locs[i + 1] - locs[i].detach() - 1.0) ** 2).mean()
        for i in range(anchor_i, 0, -1):
            tot = tot + ((locs[i - 1] - locs[i].detach() + 1.0) ** 2).mean()
        return tot
    ws = [model.weights(r, loops, sigma, hard) for r in regs]
    for i in range(anchor_i, len(regs) - 1):
        tgt = shift_right(ws[i]).detach()
        tot = tot + ((ws[i + 1] - tgt) ** 2).sum(-1).mean()
    for i in range(anchor_i, 0, -1):
        tgt = shift_left(ws[i]).detach()
        tot = tot + ((ws[i - 1] - tgt) ** 2).sum(-1).mean()
    return tot


# --------------------------------------------------------------------------
# train + evaluate one controller cell
# --------------------------------------------------------------------------
def run_cell(args, model, train, held, device, S):
    amp = args.dtype == "bf16"
    tr_mods, tr_xs = train
    mods, xs = held
    L = args.train_loops or max(args.train_t)

    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("sel.") or n == "unit")
    tr_par = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tr_par, lr=args.sel_lr)

    # the ALU is frozen and constructed, so the candidate stack is a CONSTANT:
    # compute it once at the training depth L and reuse it every step.
    pre = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=amp):
        for t in args.train_t:
            ids, mask = make_batch(tr_mods, tr_xs, t, device)
            tg = targets(tr_mods, tr_xs, t, S, device)
            sx, sn, st = model.slots_of(ids, mask)
            stack, _ = model.iterate(sx, sn, L)
            pre.append((st.float(), stack.float(), tg))
    torch.cuda.empty_cache()

    sigma0 = args.sel_sigma0 or max(1.0, L / 2.0)
    t0 = time.time()
    for step in range(1, args.sel_steps + 1):
        frac = min(1.0, step / max(1, args.sel_anneal))
        sigma = max(args.sel_sigma1, sigma0 - (sigma0 - args.sel_sigma1) * frac)
        loss = torch.zeros((), device=device)
        for st, stack, tg in pre:
            w = model.weights(st, L, sigma, args.hard_sel)
            mixed = torch.einsum("bt,btsd->bsd", w, stack)
            loss = loss + F.cross_entropy(mixed.reshape(-1, 10), tg.reshape(-1))
            if args.halt_pen > 0:
                loss = loss + args.halt_pen * ((1.0 - w.sum(-1)) ** 2).mean()
        if args.cons > 0:
            # registers depend only on T, so one representative row per T
            for st, _, _ in pre:
                reg = st[:1, : model.n_t]
                regs = orbit_regs(reg, model, args.cons_j, args.cons_jd,
                                  args.cons_hard)
                loss = loss + args.cons * cons_loss(
                    model, regs, args.cons_jd, args.cons_loops, sigma,
                    args.cons_space, args.hard_sel)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tr_par, 1.0)
        opt.step()
        if args.sel_log and (step % args.sel_log == 0 or step == 1):
            print(f"  step={step:>5} loss={float(loss):.5f} sigma={sigma:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    with torch.no_grad():
        fitted = []
        for (st, _, _), t in zip(pre, args.train_t):
            fitted.append(float(model.loc(st, max(L, 64), 0.15,
                                          args.hard_sel).float().mean()))
    diag = param_diag(model)
    print(f"  fitted loc on train T={args.train_t}: "
          + " ".join(f"{t}->{v:.3f}" for t, v in zip(args.train_t, fitted))
          + f"   [{diag}]", flush=True)

    print(f"\n{'T':>4} {'loc':>8} {'route':>7} {'exact':>7} {'sec':>6}")
    print("-" * 40)
    results, routes = {}, {}
    for t in args.ladder:
        ids, mask = make_batch(mods, xs, t, device)
        tg = targets(mods, xs, t, S, device)
        t1 = time.time()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=amp):
            sx, sn, st = model.slots_of(ids, mask)
            stack, _ = model.iterate(sx, sn, args.eval_loops)
            w = model.weights(st.float(), args.eval_loops, args.sigma,
                              args.hard_sel).to(stack.dtype)
            mixed = torch.einsum("bt,btsd->bsd", w, stack)
        route = (w.argmax(-1) == (t - 1)).float().mean().item()
        ok = (mixed.float().argmax(-1) == tg).all(dim=1).float().mean().item()
        results[t], routes[t] = ok, route
        print(f"{t:>4} {float(w.argmax(-1).float().mean()):>8.3f} {route:>7.4f} "
              f"{ok:>7.4f} {time.time()-t1:>6.2f}", flush=True)
    maxt = 0
    for t in args.ladder:
        if results.get(t, 0.0) >= 1.0:
            maxt = t
        else:
            break
    print(f"\nMAX_T = {maxt}", flush=True)
    return results, routes, maxt, fitted, diag


def param_diag(model):
    bits = []
    sel = model.sel
    if hasattr(sel, "one"):
        p = F.softmax(sel.one.float(), -1)
        bits.append(f"one=argmax{int(p.argmax())}@{float(p.max()):.3f}")
    if hasattr(sel, "gain"):
        bits.append(f"gain={float(sel.gain):.2f}")
    if hasattr(sel, "thresh"):
        bits.append(f"thresh={float(sel.thresh):.3f}")
    if getattr(model, "unit", None) is not None:
        p = F.softmax(model.unit.float(), -1)
        bits.append(f"unit=argmax{int(p.argmax())}@{float(p.max()):.3f}")
    if hasattr(sel, "c"):
        bits.append("c=" + ",".join(f"{float(v):.3f}" for v in sel.c))
        bits.append(f"b={float(sel.b):.3f}")
    return " ".join(bits)


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="fixed", choices=("fixed", "sampled"))
    ap.add_argument("--modulus", type=int, default=329)
    ap.add_argument("--bits", type=int, nargs="+", default=[10, 11])
    ap.add_argument("--moduli", type=int, default=6)
    ap.add_argument("--slots", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=76)
    ap.add_argument("--reduce", type=int, default=11)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--ladder", type=int, nargs="+", default=LADDER)
    ap.add_argument("--dtype", default="bf16", choices=("fp32", "bf16"))
    ap.add_argument("--selector", default="counter2", choices=SEL_KINDS)
    ap.add_argument("--train-t", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--eval-loops", type=int, default=64)
    ap.add_argument("--train-loops", type=int, default=0,
                    help="0 = max(train T), i.e. probe_compose's setting")
    ap.add_argument("--no-dump", action="store_true",
                    help="do NOT pile unspent halting mass on the last index")
    ap.add_argument("--halt-pen", type=float, default=0.0)
    ap.add_argument("--hard-sel", "--reg-hard", dest="hard_sel",
                    action="store_true",
                    help="straight-through one-hot on the counter's digit "
                         "register after every decrement: the count stays "
                         "exactly discrete over 64 steps while the gradient "
                         "still reaches `one` (attacks cause (b) at its source)")
    ap.add_argument("--one-init", type=float, default=0.5)
    ap.add_argument("--cons", type=float, default=0.0)
    ap.add_argument("--cons-j", type=int, default=8)
    ap.add_argument("--cons-jd", type=int, default=0)
    ap.add_argument("--cons-loops", type=int, default=0)
    ap.add_argument("--cons-space", default="w", choices=("w", "loc"))
    ap.add_argument("--cons-hard", action="store_true")
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--sel-steps", type=int, default=1500)
    ap.add_argument("--sel-lr", type=float, default=0.1)
    ap.add_argument("--sel-anneal", type=int, default=800)
    ap.add_argument("--sel-sigma0", type=float, default=0.0)
    ap.add_argument("--sel-sigma1", type=float, default=0.1)
    ap.add_argument("--sel-log", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.cons_loops:
        args.cons_loops = args.eval_loops

    device = torch.device(args.device)
    rows = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        rng = random.Random(seed)
        train, held, S = build_pool(args, rng)
        n_t = max(2, max(len(str(t)) for t in args.ladder))
        model = Depth(S, n_t, args.reduce, args.tau, args.selector,
                      args.one_init, not args.no_dump).to(device)
        # parser + ALU constructed (LAB DIAGNOSTIC); the controller is random
        model.construct(parse=True, alu=True, sel=False)
        model.eval()
        n_par = sum(p.numel() for p in model.parameters()
                    if p.requires_grad or True)
        n_sel = sum(p.numel() for n, p in model.named_parameters()
                    if n.startswith("sel.") or n == "unit")
        print(f"\n[{args.tag}] seed={seed} selector={args.selector} "
              f"train_T={args.train_t} train_loops="
              f"{args.train_loops or max(args.train_t)} eval_loops="
              f"{args.eval_loops} cons={args.cons} j={args.cons_j}/"
              f"{args.cons_jd} space={args.cons_space} dump={not args.no_dump} "
              f"halt_pen={args.halt_pen} hard={args.hard_sel} S={S} "
              f"sel_params={n_sel} train={len(train[1])} held={len(held[1])}",
              flush=True)
        print(f"[lambda] {lam_banner(set(held[0]), args.ladder)}", flush=True)
        res, route, maxt, fitted, diag = run_cell(args, model, train, held,
                                                  device, S)
        rows.append(dict(tag=args.tag, seed=seed, selector=args.selector,
                         modulus=(args.modulus if args.regime == "fixed"
                                  else args.bits),
                         train_t=args.train_t,
                         train_loops=args.train_loops or max(args.train_t),
                         cons=args.cons, cons_j=args.cons_j,
                         cons_jd=args.cons_jd, cons_space=args.cons_space,
                         dump=not args.no_dump, halt_pen=args.halt_pen,
                         hard=args.hard_sel, exact=res, route=route,
                         max_t=maxt, fitted=fitted, diag=diag))
        del model
        torch.cuda.empty_cache()
    print("\n=== summary ===")
    for r in rows:
        print(f"  seed={r['seed']} MAX_T={r['max_t']:>2}  "
              + " ".join(f"{t}:{r['exact'][t]:.3f}" for t in args.ladder))
    if args.out:
        with open(args.out, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
