#!/usr/bin/env python
"""LAB ONLY -- the endgame question: given a PERFECT single squaring step, what
MAX_T does the full pipeline certify?

Three components have each been verified in isolation on other branches:

  * marker-relative parser  (digit-carry §3)   prompt -> place-valued digit slots
  * DigitALU                (digit-carry §2.2) digits(x), digits(N) -> digits(x^2 mod N)
  * weight-tied iteration   (tied-recurrence)  one block applied T times

This script COMPOSES them in one real forward pass, with the one unsolved
component (training the transducer) replaced by its constructed ceiling, and
measures the whole ladder T = 1,2,4,8,16,32,64.

`--construct` sets the digit tables and the pointer to the exact solution.  That
is a LAB DIAGNOSTIC and never appears in a submission (BRIEF §4 rule 7).  It is
the direct analogue of probe_alu.py --construct and probe_step.py --oracle.

Prompts are synthesised from the public generator spec
(`data/squaring_mod.py`: `[N] d(N) [X] d(x) [T] d(T)`, target = tail-aligned
digits of x^(2^T) mod N).  Nothing under data/generated/ is opened.

Modes:
  --mode compose   exactness of the composed pipeline, forced loops = T
  --mode select    the ordered-pointer depth selector on the full ladder
  --mode time      throughput / eval-budget arithmetic at every loop count
"""

from __future__ import annotations

import argparse
import math
import random
import time

import torch
import torch.nn.functional as F
from torch import nn

from probe_alu import DigitALU, digits_le
from probe_alu_oodn import sample_semiprime
from probe_parse import MarkerPointer, build_prompt, number_tokens

VOCAB = 17
DIGIT_OFFSET = 7
BIG = 30.0
LADDER = [1, 2, 4, 8, 16, 32, 64]


# --------------------------------------------------------------------------
# a DigitALU whose modulus digits are per-example (the sampled-N / e5 regime)
# --------------------------------------------------------------------------
class BatchedALU(DigitALU):
    def cond_sub(self, r, ndig):
        b = r.shape[0]
        c = self._sm(self.borrow0).expand(b, self.Cb)
        outs = []
        for m in range(self.W):
            nd = ndig[:, m] if ndig.dim() == 3 else ndig[m].expand(b, 10)
            o = torch.einsum("bu,bv,bc,uvco->bo", r[:, m], nd, c, self.Tsub)
            outs.append(self._sm(o[:, :10] + self.copy_scale * r[:, m]))
            c = self._sm(o[:, 10:] + self.copy_scale * c)
        t = torch.stack(outs, 1)
        g = torch.sigmoid(self.gate(c))[:, :, None]
        return g * t + (1 - g) * r


# --------------------------------------------------------------------------
# depth selectors: T's digit slots -> a scalar location on the step axis
# --------------------------------------------------------------------------
class Selector(nn.Module):
    """`kind` decides how much structure the T -> step-count map is given.

    construct : loc = sum_p 10^p * sum_d d*st[p,d] - 1     (DIAGNOSTIC)
    mlp       : Linear(ST*10,64) -> GELU -> Linear(64,1)   (the shipped dcp head)
    linear    : Linear(ST*10, 1)
    placev    : loc = sum_p c_p * (sum_d v_d st[p,d]) + b, v learned  (10+ST+1)
    place     : same with the ORDINAL prior v_d = d fixed             (ST+1)
    """

    def __init__(self, kind: str, n_t: int) -> None:
        super().__init__()
        self.kind = kind
        self.n_t = n_t
        if kind == "mlp":
            self.net = nn.Sequential(nn.Linear(n_t * 10, 64), nn.GELU(),
                                     nn.Linear(64, 1))
        elif kind == "linear":
            self.net = nn.Linear(n_t * 10, 1)
        elif kind in ("place", "placev", "construct"):
            self.c = nn.Parameter(torch.randn(n_t) * 0.5)
            self.b = nn.Parameter(torch.zeros(1))
            self.v = nn.Parameter(torch.randn(10) * 0.5)
        else:
            raise ValueError(kind)

    @torch.no_grad()
    def construct(self) -> None:
        self.c.copy_(torch.tensor([10.0 ** p for p in range(self.n_t)]))
        self.b.fill_(-1.0)
        self.v.copy_(torch.arange(10, dtype=torch.float32))

    def forward(self, st):                       # st: (B, n_t, 10)
        if self.kind in ("mlp", "linear"):
            return self.net(st.flatten(1))
        v = torch.arange(10, device=st.device, dtype=st.dtype) \
            if self.kind == "place" else self.v.to(st.dtype)
        val = torch.einsum("btd,d->bt", st, v)
        return (val * self.c.to(st.dtype)).sum(-1, keepdim=True) + self.b.to(st.dtype)


class CounterSelector(nn.Module):
    """Depth control by COUNTING DOWN T's digits with the same borrow scan the
    modular reduction already uses.

    The ordered-pointer selectors above all decode T into a scalar with a head
    whose parameters are indexed by (place, digit).  Training only ever shows
    them the T values of the tier, so the digits and places the ladder needs
    but training never presents are unconstrained -- that is the measured
    failure in §4.  A counter has no such head: it subtracts a learned unit
    digit from a digit register, using `Tsub` and `borrow0` SHARED with the
    arithmetic, which the modular reduction exercises on every digit pair.

        z_k    = P(register has reached zero after k decrements)   (monotone)
        w_k    = z_{k+1} - z_k                                     (a PonderNet
                 marginal over iterations; readout = sum_k w_k out_k)

    Learned here: the unit digit `one`, and a per-slot zero detector.  That is
    22 parameters, none of them indexed by a place of T.
    """

    def __init__(self, n_t: int, shared_zero: bool = False) -> None:
        super().__init__()
        self.n_t = n_t
        self.shared_zero = shared_zero
        self.one = nn.Parameter(torch.randn(10) * 0.5)
        self.det = nn.Linear(10, 1)
        self.det_b = nn.Parameter(torch.full((1,), -2.0))
        # `counterz`: the zero detector is a match against the ALU's OWN
        # learned zero digit, so it has no per-digit parameter of its own and
        # inherits whatever grounding the arithmetic gives that vector.
        self.gain = nn.Parameter(torch.tensor(2.0))
        self.thresh = nn.Parameter(torch.tensor(float(n_t) - 0.5))

    @torch.no_grad()
    def construct(self) -> None:
        o = torch.full((10,), -BIG)
        o[1] = BIG
        self.one.copy_(o)
        w = torch.full((1, 10), -BIG)
        w[0, 0] = BIG
        self.det.weight.copy_(w)
        self.det.bias.zero_()
        self.det_b.fill_(-BIG * (self.n_t - 0.5))
        self.gain.fill_(BIG)
        self.thresh.fill_(self.n_t - 0.5)

    def is_zero(self, c, alu=None):
        if self.shared_zero:
            z = alu._sm(alu.zero).to(c.dtype)
            m = torch.einsum("btd,d->b", c, z)[:, None]
            return torch.sigmoid(self.gain.to(c.dtype)
                                 * (m - self.thresh.to(c.dtype)))
        return torch.sigmoid(self.det(c).sum(1) + self.det_b.to(c.dtype))

    def decrement(self, c, alu):
        b = c.shape[0]
        one = alu._sm(self.one).to(c.dtype)
        zer = alu._sm(alu.zero).to(c.dtype)
        sub = torch.stack([one] + [zer] * (self.n_t - 1), 0)[None].expand(
            b, self.n_t, 10)
        brw = alu._sm(alu.borrow0).to(c.dtype).expand(b, alu.Cb)
        outs = []
        for m in range(self.n_t):
            o = torch.einsum("bu,bv,bc,uvco->bo", c[:, m], sub[:, m], brw,
                             alu.Tsub.to(c.dtype))
            outs.append(alu._sm(o[:, :10]))
            brw = alu._sm(o[:, 10:])
        return torch.stack(outs, 1)

    def forward(self, st, alu, loops):
        # PonderNet marginal: p_k = P(the register is zero after k+1
        # decrements), w_k = p_k * prod_{j<k} (1 - p_j).  Better conditioned
        # than a difference of monotone detectors -- at p = 1/2 it is a proper
        # geometric distribution rather than an all-zero weight vector.
        c = st[:, : self.n_t]
        rest = torch.ones_like(self.is_zero(c, alu))
        ws = []
        for _ in range(loops):
            c = self.decrement(c, alu)
            p = self.is_zero(c, alu)
            ws.append(rest * p)
            rest = rest * (1 - p)
        ws[-1] = ws[-1] + rest          # remaining mass on the deepest step
        return torch.cat(ws, dim=-1)


# --------------------------------------------------------------------------
# the composed pipeline
# --------------------------------------------------------------------------
class Composed(nn.Module):
    def __init__(self, slots: int, n_t: int = 2, reduce_steps: int = 11,
                 tau: float = 1.0, sel_kind: str = "construct") -> None:
        super().__init__()
        self.S = slots
        self.W = slots + 1
        self.n_t = n_t
        self.tau = tau
        self.emb = nn.Parameter(torch.eye(VOCAB), requires_grad=False)
        self.n_slot = self.S + self.W + n_t
        self.ptr = MarkerPointer(VOCAB, self.n_slot, o_hi=max(9, self.W + 1))
        self.digit = nn.Linear(VOCAB, 10)
        self.alu = BatchedALU(slots, reduce_steps=reduce_steps)
        self.sel_kind = sel_kind
        self.sel = (CounterSelector(n_t, sel_kind == "counterz")
                    if sel_kind.startswith("counter")
                    else Selector(sel_kind, n_t))

    # ---------------- construction (LAB DIAGNOSTIC ONLY) ----------------
    @torch.no_grad()
    def construct(self, parse: bool = True, alu: bool = True,
                  sel: bool = True) -> None:
        if parse:
            # (terminating anchor, offset, opening anchor);
            # anchors: 0=[N] 1=[X] 2=[T] 3=end-of-sequence
            spec = ([(2, 1 + p, 1) for p in range(self.S)]
                    + [(1, 1 + p, 0) for p in range(self.W)]
                    + [(3, p, 2) for p in range(self.n_t)])
            self.ptr.construct(self.emb, spec)
            w = torch.zeros(10, VOCAB)
            for t in range(VOCAB):
                w[:, t] = -BIG
                w[(t - DIGIT_OFFSET) if t >= DIGIT_OFFSET else 0, t] = BIG
            self.digit.weight.copy_(w)
            self.digit.bias.zero_()
        if alu:
            self.alu.construct()
        if sel and hasattr(self.sel, "construct"):
            self.sel.construct()

    def slots_of(self, ids, mask):
        e = self.emb[ids]
        s = self.ptr(e, mask)[0]
        dg = F.softmax(self.digit(s) / self.tau, dim=-1)
        return (dg[:, : self.S], dg[:, self.S: self.S + self.W],
                dg[:, self.S + self.W:])

    def weights(self, st, loops, sigma):
        """(B, loops) mixture over iterations."""
        if self.sel_kind.startswith("counter"):
            return self.sel(st, self.alu, loops)
        return window(self.sel(st), loops, sigma, st.dtype)

    def iterate(self, sx, sn, loops, keep_all=True):
        state, outs = sx, []
        for _ in range(loops):
            out = self.alu(state, sn)
            if keep_all:
                outs.append(out)
            state = F.softmax(out, dim=-1)
        return (torch.stack(outs, 1) if keep_all else out.unsqueeze(1)), state


def window(loc, loops, sigma, dtype):
    loc = loc.float()
    grid = torch.arange(loops, device=loc.device, dtype=loc.dtype)[None, :]
    return F.softmax(-((grid - loc) ** 2) / (2 * sigma ** 2), dim=-1).to(dtype)


# --------------------------------------------------------------------------
# data (self-generated from the public generator spec)
# --------------------------------------------------------------------------
def make_batch(mods, xs, t, device):
    rows = [build_prompt(m, x, t) for m, x in zip(mods, xs)]
    L = max(len(r) for r in rows)
    ids = torch.zeros(len(rows), L, dtype=torch.long)
    mask = torch.zeros(len(rows), L, dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = True
    return ids.to(device), mask.to(device)


def targets(mods, xs, t, slots, device):
    tgt = torch.zeros(len(xs), slots, dtype=torch.long)
    for i, (m, x) in enumerate(zip(mods, xs)):
        y = pow(x, 1 << t, m)
        for p, d in enumerate(digits_le(y, slots)):
            tgt[i, p] = d
    return tgt.to(device)


def build_pool(args, rng):  # noqa: C901
    if args.regime == "sampled":
        pool = [sample_semiprime(b, rng) for b in args.bits
                for _ in range(args.moduli)]
        mods, xs = [], []
        for _ in range(args.n_eval):
            m = rng.choice(pool)
            mods.append(m)
            xs.append(rng.randrange(1, m))
        held = (mods, xs)
        trm, trx = [], []
        for _ in range(args.n_train):
            m = rng.choice(pool)
            trm.append(m)
            trx.append(rng.randrange(1, m))
        S = args.slots or max(len(str(m)) for m in pool)
        return (trm, trx), held, S
    m = args.modulus
    if m > 100_000:
        # enumerating the unit group is intractable past ~24 bits; sample
        # instead (exactness is a per-example question, so a sample is unbiased)
        pick = rng.sample(range(1, m), args.n_train + args.n_eval)
        train_x, held_x = pick[: args.n_train], pick[args.n_train:]
    else:
        units = [x for x in range(1, m) if math.gcd(x, m) == 1]
        g = torch.Generator().manual_seed(args.split_seed)
        perm = torch.randperm(len(units), generator=g).tolist()
        train_x = [units[i] for i in perm[: args.n_train]]
        held_x = [units[i] for i in perm[args.n_train:]][: args.n_eval]
    S = args.slots or len(str(m))
    return ([m] * len(train_x), train_x), ([m] * len(held_x), held_x), S


# --------------------------------------------------------------------------
# mode: compose -- does exactness compose, with loops forced to T?
# --------------------------------------------------------------------------
def mode_compose(args, model, held, device, S):
    mods, xs = held
    amp = args.dtype == "bf16"
    print(f"{'T':>4} {'n':>5} {'exact':>7} {'digit_acc':>9} {'min_max_p':>9} "
          f"{'sec':>6}")
    print("-" * 48)
    results = {}
    for t in args.ladder:
        ids, mask = make_batch(mods, xs, t, device)
        tgt = targets(mods, xs, t, S, device)
        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=amp):
            model.alu.hard = args.eval_hard
            sx, sn, _ = model.slots_of(ids, mask)
            _, state = model.iterate(sx, sn, t, keep_all=False)
        lg = state.float()
        ok = (lg.argmax(-1) == tgt).all(dim=1).float().mean().item()
        dig = (lg.argmax(-1) == tgt).float().mean().item()
        drift = lg.max(-1).values.min().item()
        results[t] = ok
        print(f"{t:>4} {len(xs):>5} {ok:>7.4f} {dig:>9.4f} {drift:>9.5f} "
              f"{time.time()-t0:>6.2f}")
    maxt = 0
    for t in args.ladder:
        if results.get(t, 0.0) >= 1.0:
            maxt = t
        else:
            break
    print(f"\nMAX_T(constructed, oracle loop count) = {maxt}")
    return results


# --------------------------------------------------------------------------
# mode: select -- does the depth selector route T at the top of the ladder?
# --------------------------------------------------------------------------
def mode_select(args, model, train, held, device, S):
    amp = args.dtype == "bf16"
    tr_mods, tr_xs = train
    mods, xs = held
    loops_tr = max(args.train_t)

    if args.sel_steps > 0:
        # freeze parser + ALU at the construction; train ONLY the selector, and
        # only on the T values this tier actually trains on.
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith("sel."))
        opt = torch.optim.AdamW([p for p in model.parameters()
                                 if p.requires_grad], lr=args.sel_lr)
        cache = []
        for t in args.train_t:
            ids, mask = make_batch(tr_mods, tr_xs, t, device)
            cache.append((ids, mask, targets(tr_mods, tr_xs, t, S, device)))
        with torch.no_grad():
            pre = []
            for ids, mask, tg in cache:
                sx, sn, st = model.slots_of(ids, mask)
                stack, _ = model.iterate(sx, sn, loops_tr)
                pre.append((st, stack, tg))
        sigma0 = args.sel_sigma0 or max(1.0, loops_tr / 2.0)
        for step in range(1, args.sel_steps + 1):
            frac = min(1.0, step / max(1, args.sel_anneal))
            sigma = max(0.1, sigma0 - (sigma0 - 0.1) * frac)
            loss = 0.0
            for st, stack, tg in pre:
                w = model.weights(st, loops_tr, sigma).to(stack.dtype)
                mixed = torch.einsum("bt,btsd->bsd", w, stack)
                loss = loss + F.cross_entropy(mixed.reshape(-1, 10),
                                              tg.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % args.sel_log == 0 or step == 1:
                print(f"  sel step={step:>5} loss={float(loss):.5f} "
                      f"sigma={sigma:.2f}", flush=True)
        with torch.no_grad():
            if model.sel_kind.startswith("counter"):
                locs = [float(model.weights(st, loops_tr, 0.15).argmax(-1)
                              .float().mean()) for st, _, _ in pre]
            else:
                locs = [float(model.sel(st).mean()) for st, _, _ in pre]
        print(f"  fitted loc on train T={args.train_t}: "
              + " ".join(f"{t}->{l:.3f}" for t, l in zip(args.train_t, locs)))

    print(f"\n{'T':>4} {'loc':>8} {'route':>7} {'exact':>7} {'sec':>6}")
    print("-" * 40)
    results = {}
    for t in args.ladder:
        ids, mask = make_batch(mods, xs, t, device)
        tgt = targets(mods, xs, t, S, device)
        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=amp):
            model.alu.hard = args.eval_hard
            sx, sn, st = model.slots_of(ids, mask)
            stack, _ = model.iterate(sx, sn, args.eval_loops)
            w = model.weights(st, args.eval_loops, args.sigma).to(stack.dtype)
            loc = (w.argmax(-1).float() if model.sel_kind.startswith("counter")
                   else model.sel(st))
            mixed = torch.einsum("bt,btsd->bsd", w, stack)
        route = (w.argmax(-1) == (t - 1)).float().mean().item()
        ok = (mixed.float().argmax(-1) == tgt).all(dim=1).float().mean().item()
        results[t] = ok
        print(f"{t:>4} {float(loc.mean()):>8.3f} {route:>7.4f} {ok:>7.4f} "
              f"{time.time()-t0:>6.2f}")
    maxt = 0
    for t in args.ladder:
        if results.get(t, 0.0) >= 1.0:
            maxt = t
        else:
            break
    print(f"\nMAX_T(constructed step, selector={args.selector}, "
          f"train_T={args.train_t}) = {maxt}")
    return results


# --------------------------------------------------------------------------
# mode: time -- throughput and the eval-budget arithmetic
# --------------------------------------------------------------------------
def mode_time(args, model, held, device, S):
    mods, xs = held
    amp = args.dtype == "bf16"

    def timed(fn, reps=3):
        fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / reps

    print(f"{'batch':>6} {'loops':>6} {'mode':>12} {'ms':>9} {'ms/loop':>8}")
    print("-" * 46)
    rows = []
    for bs in args.batch:
        mm = [mods[i % len(mods)] for i in range(bs)]
        xx = [xs[i % len(xs)] for i in range(bs)]
        ids, mask = make_batch(mm, xx, 64, device)
        tgt = targets(mm, xx, 1, S, device)
        for loops in args.loops:
            def ev():
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16,
                                                     enabled=amp):
                    model.alu.hard = args.eval_hard
                    sx, sn, st = model.slots_of(ids, mask)
                    stack, _ = model.iterate(sx, sn, loops)
                    w = model.weights(st, loops, args.sigma).to(stack.dtype)
                    return torch.einsum("bt,btsd->bsd", w, stack)
            ms = timed(ev) * 1e3
            rows.append((bs, loops, "eval-mixture", ms))
            print(f"{bs:>6} {loops:>6} {'eval-mixture':>12} {ms:>9.1f} "
                  f"{ms/loops:>8.2f}")

            def evh():
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16,
                                                     enabled=amp):
                    model.alu.hard = True
                    sx, sn, st = model.slots_of(ids, mask)
                    k = model.weights(st, loops, args.sigma).argmax(
                        -1, keepdim=True) if model.sel_kind.startswith("counter") \
                        else model.sel(st).round().clamp(0, loops - 1).long()
                    n = int(k.max().item()) + 1
                    stack, _ = model.iterate(sx, sn, n)
                    return stack.gather(
                        1, k[:, :, None, None].expand(-1, 1, S, 10)).squeeze(1)
            ms = timed(evh) * 1e3
            rows.append((bs, loops, "eval-hardsel", ms))
            print(f"{bs:>6} {loops:>6} {'eval-hardsel':>12} {ms:>9.1f} "
                  f"{ms/loops:>8.2f}")

            if loops <= args.train_loops_max:
                for p in model.parameters():
                    p.requires_grad_(True)
                model.emb.requires_grad_(False)

                def tr():
                    with torch.autocast("cuda", torch.bfloat16, enabled=amp):
                        model.alu.hard = False
                        sx, sn, st = model.slots_of(ids, mask)
                        stack, _ = model.iterate(sx, sn, loops)
                        w = model.weights(st, loops, args.sigma).to(stack.dtype)
                        mixed = torch.einsum("bt,btsd->bsd", w, stack)
                        loss = F.cross_entropy(mixed.float().reshape(-1, 10),
                                               tgt.reshape(-1))
                    loss.backward()
                    model.zero_grad(set_to_none=True)
                ms = timed(tr, reps=2) * 1e3
                rows.append((bs, loops, "train+bwd", ms))
                print(f"{bs:>6} {loops:>6} {'train+bwd':>12} {ms:>9.1f} "
                      f"{ms/loops:>8.2f}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="compose",
                    choices=("compose", "select", "time"))
    ap.add_argument("--regime", default="fixed", choices=("fixed", "sampled"))
    ap.add_argument("--modulus", type=int, default=323)
    ap.add_argument("--bits", type=int, nargs="+", default=[10, 11])
    ap.add_argument("--moduli", type=int, default=6)
    ap.add_argument("--slots", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--reduce", type=int, default=11)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--ladder", type=int, nargs="+", default=LADDER)
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--eval-hard", action="store_true",
                    help="discrete (argmax) states at eval -- no backward pass "
                         "there, so this costs nothing in credit assignment")
    ap.add_argument("--selector", default="construct",
                    choices=("construct", "mlp", "linear", "place", "placev",
                             "counter", "counterz"))
    ap.add_argument("--train-t", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--eval-loops", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--sel-steps", type=int, default=0)
    ap.add_argument("--sel-lr", type=float, default=3e-2)
    ap.add_argument("--sel-anneal", type=int, default=400)
    ap.add_argument("--sel-sigma0", type=float, default=0.0)
    ap.add_argument("--sel-log", type=int, default=200)
    ap.add_argument("--batch", type=int, nargs="+", default=[38, 512])
    ap.add_argument("--loops", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--train-loops-max", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    train, held, S = build_pool(args, rng)
    n_t = max(2, max(len(str(t)) for t in args.ladder))
    model = Composed(S, n_t, args.reduce, args.tau, args.selector).to(device)
    model.construct(parse=True, alu=True, sel=(args.sel_steps == 0))
    model.eval()
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.tag}] mode={args.mode} regime={args.regime} "
          f"modulus={args.modulus if args.regime == 'fixed' else args.bits} "
          f"S={S} W={S+1} K={2*S-1} R={args.reduce} n_t={n_t} "
          f"dtype={args.dtype} eval_hard={args.eval_hard} "
          f"selector={args.selector} params={n_par:,} "
          f"train={len(train[1])} held={len(held[1])}", flush=True)

    if args.mode == "compose":
        mode_compose(args, model, held, device, S)
    elif args.mode == "select":
        mode_select(args, model, train, held, device, S)
    else:
        mode_time(args, model, held, device, S)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
