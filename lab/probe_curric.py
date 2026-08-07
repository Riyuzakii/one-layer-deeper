#!/usr/bin/env python
"""LAB ONLY -- a LOSS-SIDE CURRICULUM over modulus size and operand magnitude,
measured on a MIXED-MODULUS DigitALU at Hard-faithful scale.

WHY THIS PROBE EXISTS
---------------------
`hf1` puts three modulus sizes (16, 18, 20 bits) in one training set, and the
DigitALU's tables are modulus-independent by construction, so every example --
whatever its modulus -- trains the *same* ~6.8k parameters.  That makes
"upweight the small moduli early, anneal toward the large ones" a genuine
curriculum over difficulty on shared parameters, and it is implementable
entirely inside `training_loss`, which is the only place the evaluator's fixed
loop leaves open.

`explore/alu-credit` measured a *magnitude* curriculum as null, but at e1 scale
on a single 3-digit modulus, where there is no modulus-size axis at all.  This
probe rebuilds the idea in the setting where the axis actually exists.

WHAT IS NEW HERE MECHANICALLY
-----------------------------
The published `DigitALU` / `PopALU` take ONE modulus per forward (`ndig` is a
single (W,10) tensor).  A curriculum over modulus size needs several moduli in
one batch, so this version carries a PER-EXAMPLE modulus: `ndig` is (b, W, 10),
`multiples` is (b, M, W, 10), and the multiples-contracted subtract table is
built once per forward.  The graph is otherwise
`probe_alu_depth.DigitALU(mul_mode='tree', reduce_mode='quotient',
scan_mode='serial')`.

One correctness consequence, and it is load-bearing: with a FIXED slot count S
shared across modulus sizes, the published schedule "reduce only for t <= S"
is wrong for the small moduli -- after 2S-1-S unreduced places the register can
exceed 10*N for a 16-bit N, so the quotient alphabet Q+1 no longer covers the
true quotient.  This probe reduces at EVERY place (`redall`), which keeps
r < 10*N everywhere and makes the constructed ceiling 1.000 at every modulus
size.  `--red-skip k` restores the cheaper schedule for timing comparisons.

DIFFICULTY IS DERIVED FROM THE INPUT TENSORS, NEVER FROM THE DATASET
--------------------------------------------------------------------
The curriculum weight is a monotone function of `log10` of the modulus and of
the operand, both computed from the one-hot digit tensors the model is handed
at runtime -- exactly the quantity a submission can compute from its parsed
slots inside `training_loss`.  There is no bit-size threshold anywhere in the
loss: the difficulty scalar is standardised within the batch, so the only
hyperparameters are a slope `beta` and its schedule.  Nothing under
data/generated/ is opened; every modulus and operand here is self-generated.

COMPLIANCE.  `--construct` sets the tables to the truth and is a LAB DIAGNOSTIC
(rules 2 and 7); it is used only to certify the ceiling and to calibrate the
collapse detector.  `--only-bits` trains on a subset and is LEGAL (it is a loss
weight of 0/1 computed from the input), but it is reported as the curriculum's
extreme point, not as a candidate.  Everything else is LEGAL.
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
from torch import nn

BIG = 30.0


def digits_le(value: int, slots: int) -> list[int]:
    out = []
    for _ in range(slots):
        out.append(value % 10)
        value //= 10
    return out


# ---------------------------------------------------------------- the model
class CurrALU(nn.Module):
    """tree:quotient DigitALU with a PER-EXAMPLE modulus.

    State alphabet is unchanged (digit 10, carry 2, borrow 2, quotient Q+1) and
    no tensor is indexed by anything ranging over Z_N: the modulus enters only
    as input digits.  Parameter count is independent of the modulus and of how
    many moduli the batch contains.
    """

    def __init__(self, slots: int, n_carry: int = 2, n_borrow: int = 2,
                 max_quot: int = 10, tau: float = 1.0, hard: bool = False,
                 init_scale: float = 0.5, red_skip: int = 0):
        super().__init__()
        self.S = slots
        self.W = slots + 1
        self.Ca, self.Cb = n_carry, n_borrow
        self.Q = max_quot
        self.tau, self.hard = tau, hard
        self.red_skip = red_skip
        self.needed = list(range(max_quot + 2))

        def rp(*shape):
            return nn.Parameter(torch.randn(*shape) * init_scale)

        self.Tmul = rp(10, 10, 20)
        self.Tadd = rp(10, 10, n_carry, 10 + n_carry)
        self.Tsub = rp(10, 10, n_borrow, 10 + n_borrow)
        self.zero = rp(10)
        self.carry0 = rp(n_carry)
        self.borrow0 = rp(n_borrow)
        self.sel_w = rp(2 * n_borrow)
        self.sel_b = nn.Parameter(torch.zeros(()))
        self.copy_scale = nn.Parameter(torch.zeros(()))
        # ---- instrumentation tap (DIAGNOSTIC `local_ce` only) ----
        # `record` on a CONSTRUCTED copy stores the true inter-step register
        # states; `force` replays them and accumulates the per-step CE.  This is
        # never used inside a training loss here -- it is the measuring
        # instrument `alu-population` / `alu-optimizer` calibrated (random init
        # 2.08-2.24, basin cliff 0.006, legal training drives it to 3.5-4.2).
        self.mode = None
        self.tape: list = []
        self.tpos = 0
        self.tf_loss = None
        self.tf_n = 0

    # ---------------- construction (LAB DIAGNOSTIC ONLY) ----------------
    @torch.no_grad()
    def construct(self):
        t = torch.full((10, 10, 20), -BIG)
        for a in range(10):
            for b in range(10):
                t[a, b, (a * b) % 10] = BIG
                t[a, b, 10 + (a * b) // 10] = BIG
        self.Tmul.copy_(t.to(self.Tmul))
        t = torch.full((10, 10, self.Ca, 10 + self.Ca), -BIG)
        for u in range(10):
            for v in range(10):
                for c in range(2):
                    s = u + v + c
                    t[u, v, c, s % 10] = BIG
                    t[u, v, c, 10 + s // 10] = BIG
        self.Tadd.copy_(t.to(self.Tadd))
        t = torch.full((10, 10, self.Cb, 10 + self.Cb), -BIG)
        for u in range(10):
            for v in range(10):
                for c in range(2):
                    s = u - v - c
                    t[u, v, c, s % 10] = BIG
                    t[u, v, c, 10 + (1 if s < 0 else 0)] = BIG
        self.Tsub.copy_(t.to(self.Tsub))
        for P_, n in ((self.zero, 10), (self.carry0, self.Ca),
                      (self.borrow0, self.Cb)):
            v = torch.full((n,), -BIG)
            v[0] = BIG
            P_.copy_(v.to(P_))
        w = torch.zeros(2 * self.Cb)
        w[0], w[1] = BIG / 2, -BIG / 2
        w[self.Cb + 0], w[self.Cb + 1] = -BIG / 2, BIG / 2
        self.sel_w.copy_(w.to(self.sel_w))
        self.sel_b.zero_()
        self.copy_scale.zero_()

    # ---------------- machinery ----------------
    def _sm(self, logits):
        p = F.softmax(logits / self.tau, -1)
        if getattr(self, "stat", None) is not None and p.dim() > 1:
            self.stat[0] += p.max(-1).values.mean().item()
            self.stat[1] += 1
        if self.hard:
            h = F.one_hot(p.argmax(-1), p.shape[-1]).to(p.dtype)
            p = h + p - p.detach()
        return p

    def _tap(self, x):
        """x: (..., 10) soft digit states on the critical path."""
        if self.mode == "record":
            self.tape.append(x.detach())
            return x
        if self.mode == "force":
            truth = self.tape[self.tpos]
            self.tpos += 1
            tgt = truth.argmax(-1)
            ce = F.cross_entropy(x.clamp_min(1e-9).log().reshape(-1, 10),
                                 tgt.reshape(-1))
            self.tf_loss = self.tf_loss + ce
            self.tf_n += 1
            return truth
        return x

    def add_scan(self, r, addend):
        """r, addend: (n, D, 10) -> (n, D, 10)."""
        cs = self.copy_scale
        c = self._sm(self.carry0).expand(r.shape[0], self.Ca)
        outs = []
        for m in range(r.shape[1]):
            o = torch.einsum("nu,nv,nc,uvco->no",
                             r[:, m], addend[:, m], c, self.Tadd)
            outs.append(self._sm(o[..., :10] + cs * r[:, m]))
            c = self._sm(o[..., 10:] + cs * c)
        return self._tap(torch.stack(outs, 1))

    def tree_sum(self, regs):
        while len(regs) > 1:
            carry = [regs[-1]] if len(regs) % 2 else []
            pairs = [(regs[i], regs[i + 1]) for i in range(0, len(regs) - 1, 2)]
            n = regs[0].shape[0]
            out = self.add_scan(torch.cat([a for a, _ in pairs], 0),
                                torch.cat([c for _, c in pairs], 0))
            regs = [out[i * n:(i + 1) * n] for i in range(len(pairs))] + carry
        return regs[0]

    def multiples(self, ndig):
        """{m*N} for m in `needed`, built from digits(N) by the SAME learned
        Tadd.  ndig: (b, W, 10) -> (b, M, W, 10).  Zero new parameters."""
        b, W = ndig.shape[0], self.W
        z = self._sm(self.zero).view(1, 1, 10).expand(b, W, 10)
        have = {0: z, 1: ndig}
        target = set(self.needed)
        depth = 0
        while not target <= set(have):
            newly = []
            for t in sorted(target - set(have)):
                cand = [a for a in have if a <= t - a and (t - a) in have]
                if cand:
                    newly.append((t, max(cand), t - max(cand)))
            if not newly:
                mx = max(have)
                newly = [(2 * mx, mx, mx)]
            A = torch.cat([have[a] for _, a, _ in newly], 0)
            B = torch.cat([have[b_] for _, _, b_ in newly], 0)
            out = self.add_scan(A, B)
            for i, (t, _, _) in enumerate(newly):
                have[t] = out[i * b:(i + 1) * b]
            depth += 1
        self.mult_depth = depth
        return torch.stack([have[m] for m in self.needed], 1)

    def quot_reduce(self, r):
        """One learned quotient digit + one subtraction, batched over the M
        candidate multiples.  r: (b, W, 10) -> (b, W, 10).

        `_Tm[b,m,w,u,c,o] = sum_v mults[b,m,w,v] * Tsub[u,v,c,o]` is contracted
        ONCE per forward (the multiples do not change between reductions), so
        the M-way candidate scan costs one small einsum per slot.
        """
        b, M = r.shape[0], self._Tm.shape[1]
        cs = self.copy_scale
        c = self._sm(self.borrow0).view(1, 1, self.Cb).expand(b, M, self.Cb)
        outs = []
        for w in range(self.W):
            rw = r[:, w]                                        # (b,10)
            o = torch.einsum("bu,bmc,bmuco->bmo", rw, c, self._Tm[:, :, w])
            outs.append(self._sm(o[..., :10] + cs * rw[:, None]))
            c = self._sm(o[..., 10:] + cs * c)
        t = self._tap(torch.stack(outs, 2))                     # (b,M,W,10)
        pair = torch.cat([c[:, :-1], c[:, 1:]], dim=-1)         # (b,M-1,2Cb)
        q = self._sm(torch.einsum("bmk,k->bm", pair, self.sel_w) + self.sel_b)
        self.last_q = q
        return self._tap(torch.einsum("bm,bmwo->bwo", q, t[:, :-1]))

    def forward(self, s, ndig):
        """s: (b, S, 10) one-hot digits of x (LSB first).
           ndig: (b, W, 10) one-hot digits of the modulus.
           returns log-probs (b, S, 10)."""
        b, S, W = s.shape[0], self.S, self.W
        F2 = 2 * S
        self.tpos = 0
        z = self._sm(self.zero).view(1, 10).expand(b, 10)
        mults = self.multiples(ndig)                            # (b,M,W,10)
        self._Tm = torch.einsum("bmwv,uvco->bmwuco", mults, self.Tsub)
        prod = {}
        for i in range(S):
            for j in range(S):
                o = torch.einsum("bu,bv,uvo->bo", s[:, i], s[:, j], self.Tmul)
                prod[(i, j)] = (self._sm(o[..., :10]), self._sm(o[..., 10:]))
        buckets = {}
        for (i, j), lh in prod.items():
            buckets.setdefault(i + j, []).append(lh)
        leaves = []
        for par in (0, 1):
            offs = [k for k in sorted(buckets) if k % 2 == par]
            if not offs:
                continue
            for t in range(max(len(buckets[k]) for k in offs)):
                cols = [z] * F2
                for k in offs:
                    if t < len(buckets[k]):
                        cols[k], cols[k + 1] = buckets[k][t]
                leaves.append(torch.stack(cols, 1))
        Pr = self.tree_sum(leaves)                              # (b,F2,10)
        r = z[:, None].expand(b, W, 10)
        for t in range(F2 - 1, -1, -1):
            r = torch.cat([Pr[:, t:t + 1], r[:, :W - 1]], dim=1)
            if t < F2 - self.red_skip:
                r = self.quot_reduce(r)
        return torch.log(r[:, :S] + 1e-9)

    def depth(self):
        """Chained softmax steps on the critical path."""
        S, W, F2 = self.S, self.W, 2 * self.S
        counts = [min(k + 1, S, 2 * S - 1 - k) for k in range(2 * S - 1)]
        n_leaf = (max([counts[k] for k in range(0, 2 * S - 1, 2)] or [0])
                  + max([counts[k] for k in range(1, 2 * S - 1, 2)] or [0]))
        adds = math.ceil(math.log2(n_leaf)) * F2 if n_leaf > 1 else 0
        red = (F2 - self.red_skip) * (W + 1)
        prefix = 0
        m = 1
        while m < max(self.needed):
            m *= 2
            prefix += 1
        prefix *= W
        return {"main": 1 + adds + red, "adds": adds, "reduce": red,
                "prefix": prefix, "leaves": n_leaf}

    def alphabet(self):
        return {"digit": 10, "carry": self.Ca, "borrow": self.Cb,
                "quotient": self.Q + 1}


# ---------------------------------------------------------------- data
def _primes(lo: int, hi: int) -> list[int]:
    sieve = bytearray([1]) * (hi + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(hi ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p:hi + 1:p] = bytearray(len(range(p * p, hi + 1, p)))
    return [i for i in range(max(lo, 2), hi + 1) if sieve[i]]


def moduli_for_bits(bits: int, count: int, rng: torch.Generator,
                    exclude: set[int]) -> list[int]:
    """Balanced semiprimes with the generator's own bit split (p_bits =
    bits//2, q_bits = bits - p_bits).  Source: data/squaring_mod.py:1259."""
    pb, qb = bits // 2, bits - bits // 2
    P = _primes(1 << (pb - 1), (1 << pb) - 1)
    Q = _primes(1 << (qb - 1), (1 << qb) - 1)
    lo, hi = 1 << (bits - 1), 1 << bits
    cand = sorted({p * q for p in P for q in Q if p != q and lo <= p * q < hi})
    cand = [n for n in cand if n not in exclude]
    perm = torch.randperm(len(cand), generator=rng).tolist()
    return [cand[i] for i in perm[:count]]


def sample_units(N: int, k: int, rng, seen: set[int]) -> list[int]:
    out, guard = [], 0
    while len(out) < k and guard < 200 * k:
        guard += 1
        x = int(torch.randint(1, N, (1,), generator=rng).item())
        if x in seen or math.gcd(x, N) != 1:
            continue
        seen.add(x)
        out.append(x)
    return out


class Cohort:
    """One evaluation or training cohort: (x, N) pairs and their answers."""

    def __init__(self, rows, S: int, W: int, dev, shuffle_gen=None):
        if shuffle_gen is not None:
            perm = torch.randperm(len(rows), generator=shuffle_gen).tolist()
            rows = [rows[i] for i in perm]
        self.n = len(rows)
        xd = torch.zeros(self.n, S, dtype=torch.uint8)
        nd = torch.zeros(self.n, W, dtype=torch.uint8)
        tg = torch.zeros(self.n, S, dtype=torch.long)
        bits = torch.zeros(self.n, dtype=torch.long)
        for r, (N, x, b) in enumerate(rows):
            xd[r] = torch.tensor(digits_le(x, S), dtype=torch.uint8)
            nd[r] = torch.tensor(digits_le(N, W), dtype=torch.uint8)
            tg[r] = torch.tensor(digits_le((x * x) % N, S), dtype=torch.long)
            bits[r] = b
        self.xd = xd.to(dev)
        self.nd = nd.to(dev)
        self.tg = tg.to(dev)
        self.bits = bits.to(dev)

    def batch(self, idx):
        return (F.one_hot(self.xd[idx].long(), 10).float(),
                F.one_hot(self.nd[idx].long(), 10).float(),
                self.tg[idx], self.bits[idx])


def build_data(args, dev):
    """Mirror hf1's structure with SELF-GENERATED moduli and operands.

    ID bits [16,18,20] with DISJOINT train / held modulus pools (hf1 uses
    split_group=modulus, so its `test` split is unseen moduli), plus an OOD-N
    pool at [17,19,21].  Nothing under data/generated/ is touched.
    """
    g = torch.Generator().manual_seed(args.split_seed)
    id_bits = [int(b) for b in args.id_bits.split(",")]
    ood_bits = [int(b) for b in args.ood_bits.split(",")]
    used: set[int] = set()
    tr_mod, he_mod, on_mod = {}, {}, {}
    for b in id_bits:
        ms = moduli_for_bits(b, args.n_mod + args.n_mod_held, g, used)
        used |= set(ms)
        tr_mod[b] = ms[: args.n_mod]
        he_mod[b] = ms[args.n_mod:]
    for b in ood_bits:
        on_mod[b] = moduli_for_bits(b, args.n_mod_held, g, used)

    def rows_for(mods_by_bits, n_x, seen_by_mod=None, skip=0):
        rows = []
        for b, ms in mods_by_bits.items():
            for N in ms:
                seen = seen_by_mod.setdefault(N, set()) if seen_by_mod is not None \
                    else set()
                xs = sample_units(N, n_x, g, seen)
                rows += [(N, x, b) for x in xs[skip:]]
        return rows

    seen: dict[int, set[int]] = {}
    train_rows = rows_for(tr_mod, args.n_x, seen)
    heldx_rows = rows_for(tr_mod, args.n_held_x, seen)     # same N, unseen x
    heldn_rows = rows_for(he_mod, args.n_held_x, None)     # unseen N
    oodn_rows = rows_for(on_mod, args.n_held_x, None)      # unseen N size
    S, W = args.slots, args.slots + 1
    # shuffle so that a `--eval-n` prefix is representative of every bit size
    return {
        "train": Cohort(train_rows, S, W, dev, g),
        "held_x": Cohort(heldx_rows, S, W, dev, g),
        "held_n": Cohort(heldn_rows, S, W, dev, g),
        "ood_n": Cohort(oodn_rows, S, W, dev, g),
    }, tr_mod, he_mod, on_mod


# ---------------------------------------------------------------- curriculum
_POW = None


def magnitudes(x_oh, n_oh):
    """log10 of the operand and of the modulus, from the INPUT TENSORS.

    This is the quantity a submission computes inside `training_loss` from the
    parsed digit slots carried in `aux`; nothing here reads a dataset field.
    """
    global _POW
    if _POW is None or _POW.device != x_oh.device:
        _POW = {}
    dev = x_oh.device
    key = (x_oh.shape[1], n_oh.shape[1])
    dig = torch.arange(10, device=dev, dtype=torch.float32)
    px = 10.0 ** torch.arange(x_oh.shape[1], device=dev, dtype=torch.float32)
    pn = 10.0 ** torch.arange(n_oh.shape[1], device=dev, dtype=torch.float32)
    xval = ((x_oh.float() * dig).sum(-1) * px).sum(-1)
    nval = ((n_oh.float() * dig).sum(-1) * pn).sum(-1)
    del key
    return (torch.log10(xval.clamp_min(1.0)),
            torch.log10(nval.clamp_min(1.0)))


def beta_at(step: int, total: int, args) -> tuple[float, float]:
    """Schedule shape for the curriculum slope."""
    f = args.anneal_frac
    if args.sched == "const":
        s = 1.0
    elif args.sched == "step":
        s = 1.0 if step < f * total else 0.0
    elif args.sched == "linear":
        s = max(0.0, 1.0 - step / max(f * total, 1.0))
    elif args.sched == "exp":
        s = math.exp(-step / max(f * total, 1.0))
    else:
        raise ValueError(args.sched)
    return args.beta_n * s, args.beta_x * s


def curric_weights(x_oh, n_oh, step, total, args):
    """Per-example loss weights, mean 1.  Larger modulus / operand -> lower
    weight while beta > 0; beta -> 0 recovers the plain objective exactly."""
    bn, bx = beta_at(step, total, args)
    if bn == 0.0 and bx == 0.0:
        return torch.ones(x_oh.shape[0], device=x_oh.device), 0.0, 0.0
    lx, ln = magnitudes(x_oh, n_oh)
    d = bn * _std(ln) + bx * _std(lx)
    w = torch.exp(-d.clamp(-20, 20))
    return w / w.mean(), bn, bx


def _std(v):
    return (v - v.mean()) / (v.std() + 1e-6)


# ---------------------------------------------------------------- diagnostics
@torch.no_grad()
def local_ce(model, ref, coh, n=512):
    """DIAGNOSTIC per-step CE against a constructed reference's register tape.

    Calibration inherited from `explore/alu-population` and
    `explore/alu-optimizer`: random init reads 2.08-2.24, the basin cliff is at
    ~0.006, and the plain legal objective *trains it away* to 3.5-4.2.  Exact
    match is pinned at 0.000 for every legal procedure in this project, so this
    is the only instrument with resolution below the cliff.
    """
    idx = torch.arange(min(n, coh.n), device=coh.tg.device)
    xo, no, _, _ = coh.batch(idx)
    was_h, model.hard = model.hard, False
    ref.mode, ref.tape = "record", []
    ref(xo, no)
    ref.mode = None
    model.mode, model.tape = "force", ref.tape
    model.tf_loss, model.tf_n = torch.zeros((), device=xo.device), 0
    model(xo, no)
    model.mode = None
    model.hard = was_h
    return round(float(model.tf_loss / max(model.tf_n, 1)), 5)


@torch.no_grad()
def struct_scores(model, ndigits):
    """Gauge-invariant parameter-level structure scores (same definitions as
    `alu-population`'s `struct_scores`).  Random baseline 0.23-0.28."""
    out = {}
    am = model.Tmul[..., :10].argmax(-1)
    groups = {}
    for a in range(10):
        for b in range(10):
            groups.setdefault((a * b) % 10, []).append(int(am[a, b]))
    k = sum(v.count(max(set(v), key=v.count)) for v in groups.values())
    out["mul_fn"] = round(k / 100, 3)
    pi = {c: max(set(v), key=v.count) for c, v in groups.items()}
    out["mul_gauge"] = round(len(set(pi.values())) / len(pi), 3)
    for nm, T, cols in (("add_shift", model.Tadd, range(10)),
                        ("sub_shift", model.Tsub, sorted(set(ndigits)))):
        k = n = 0
        for v in cols:
            for c in range(T.shape[2]):
                a = T[:, v, c, :10].argmax(-1)
                k += max(sum(1 for u in range(10) if int(a[u]) == (u + s) % 10)
                         for s in range(10))
                n += 10
        out[nm] = round(k / n, 3)
    return out


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, coh, chunk=512, discrete=False, limit=None,
             diversity=False):
    """Per-bit-size exact accuracy.  `discrete=True` snaps every inter-step
    state to its argmax (the `train_exact_hard` headline)."""
    model.eval()
    was, model.hard = model.hard, True if discrete else model.hard
    n = coh.n if limit is None else min(coh.n, limit)
    ok = torch.zeros(n, device=coh.tg.device)
    dg = torch.zeros(n, device=coh.tg.device)
    ce = torch.zeros(n, device=coh.tg.device)
    ans = []
    for i in range(0, n, chunk):
        idx = torch.arange(i, min(i + chunk, n), device=coh.tg.device)
        xo, no, tg, _ = coh.batch(idx)
        lg = model(xo, no)
        hit = (lg.argmax(-1) == tg)
        ok[i:i + idx.shape[0]] = hit.all(-1).float()
        dg[i:i + idx.shape[0]] = hit.float().mean(-1)
        ce[i:i + idx.shape[0]] = F.cross_entropy(
            lg.reshape(-1, 10), tg.reshape(-1), reduction="none"
        ).view(idx.shape[0], -1).mean(1)
        if diversity:
            ans.append(lg.argmax(-1))
    model.hard = was
    model.train()
    bits = coh.bits[:n]
    out = {"all": float(ok.mean()), "ce": float(ce.mean()), "n": int(n),
           "dacc": round(float(dg.mean()), 4)}
    for b in sorted(set(bits.tolist())):
        m = bits == b
        out[f"b{b}"] = round(float(ok[m].mean()), 4)
        out[f"d{b}"] = round(float(dg[m].mean()), 4)
        out[f"n{b}"] = int(m.sum())
    if diversity:
        a = torch.cat(ans, 0)
        codes = (a * (10 ** torch.arange(a.shape[1], device=a.device))).sum(-1)
        out["div"] = round(len(set(codes.tolist())) / n, 4)
    out["all"] = round(out["all"], 4)
    out["ce"] = round(out["ce"], 4)
    return out


@torch.no_grad()
def table_coverage(coh, limit=None):
    """How much of the SHARED 10x10 product table one example exercises.

    The curriculum's premise is that a small-modulus example is a cheaper source
    of the same signal.  It is the same function, but it is not the same amount
    of signal: a 16-bit operand has five significant decimal digits inside a
    seven-slot register, so 24 of its 49 digit pairs are (0, .) or (., 0) and it
    touches far fewer of the 100 shared `Tmul` cells than a 20-bit operand does.
    This counts that, per bit size.
    """
    n = coh.n if limit is None else min(coh.n, limit)
    xd, bits = coh.xd[:n].long(), coh.bits[:n]
    S = xd.shape[1]
    pair = xd[:, :, None] * 10 + xd[:, None, :]              # (n, S, S)
    flat = pair.reshape(n, S * S)
    oh = torch.zeros(n, 100, device=xd.device)
    oh.scatter_(1, flat, 1.0)
    per_ex = oh.sum(1)
    nz = (xd != 0).sum(1).float()
    out = {}
    for b in sorted(set(bits.tolist())):
        m = bits == b
        out[f"b{b}"] = {"cells_per_example": round(float(per_ex[m].mean()), 2),
                        "sig_digits": round(float(nz[m].mean()), 2),
                        "table_covered": int((oh[m].sum(0) > 0).sum())}
    return out


@torch.no_grad()
def basin(model, data, args, dev):
    """DIAGNOSTIC: sensitivity of the END-OF-CHAIN label to k corrupted `Tmul`
    cells, resolved by modulus size.

    Same instrument as `alu-relational`'s / `matrix-scan`'s basin measurement,
    asked of a new question: is the label a STRONGER signal at the easy end of
    the curriculum than at the hard end?  If a 16-bit example's answer is less
    disturbed by a wrong product cell than a 20-bit example's, then upweighting
    16-bit examples upweights the WEAKER gradient, and the curriculum is pushing
    the wrong way for a mechanical reason rather than a tuning one.
    """
    was, model.hard = model.hard, True
    n = args.basin_n
    print(f"[{args.tag}] BASIN (DIAGNOSTIC) n={n} reps={args.basin_reps} "
          f"module=Tmul cells=200", flush=True)
    for k in [int(v) for v in args.basin_ks.split(",")]:
        acc = {}
        for rep in range(args.basin_reps):
            g = torch.Generator(device=dev).manual_seed(1000 * k + rep)
            model.construct()
            if k:
                v = model.Tmul.view(100, 2, 10)
                pick = torch.randperm(200, generator=g, device=dev)[:k]
                for p in pick.tolist():
                    cell, half = p // 2, p % 2
                    cur = int(v[cell, half].argmax())
                    new = int(torch.randint(0, 10, (1,), generator=g,
                                            device=dev))
                    while new == cur:
                        new = int(torch.randint(0, 10, (1,), generator=g,
                                                device=dev))
                    v[cell, half] = -BIG
                    v[cell, half, new] = BIG
            r = evaluate(model, data["train"], args.eval_chunk, True,
                         n, False)
            for key in ("dacc", "all", "d16", "d18", "d20",
                        "b16", "b18", "b20"):
                acc.setdefault(key, []).append(r.get(key, float("nan")))
        mean = {key: round(float(torch.tensor(v).mean()), 4)
                for key, v in acc.items()}
        print(f"[{args.tag}] BASIN k={k:>4} exact={mean['all']} "
              f"(b16={mean['b16']} b18={mean['b18']} b20={mean['b20']}) "
              f"dacc={mean['dacc']} (d16={mean['d16']} d18={mean['d18']} "
              f"d20={mean['d20']})", flush=True)
    model.hard = was


@torch.no_grad()
def trivial_floors(coh, limit=None):
    """Measured floors for `dacc`, so per-bucket digit accuracy is read against
    a reference rather than against 0.1.

    A 16-bit modulus leaves the top two of seven slots identically zero, so a
    constant-zero predictor already scores ~0.36 there and ~0.23 at 18/20 bits.
    ONLY within-bucket, across-configuration comparisons of `dacc` are
    meaningful; the cross-bucket ordering is an artifact of leading zeros.
    """
    n = coh.n if limit is None else min(coh.n, limit)
    tg, bits = coh.tg[:n], coh.bits[:n]
    out = {}
    for b in sorted(set(bits.tolist())):
        m = bits == b
        zero = float((tg[m] == 0).float().mean())
        maj = float(torch.stack([
            torch.bincount(tg[m][:, s], minlength=10).max() / m.sum()
            for s in range(tg.shape[1])]).mean())
        out[f"b{b}"] = (round(zero, 4), round(maj, 4))
    return out


def fmt(name, r):
    per = " ".join(f"{k}={v}" for k, v in r.items() if k[0] in "bd"
                   and k not in ("bits", "dacc") and k[1:].isdigit())
    extra = f" div={r['div']}" if "div" in r else ""
    return (f"{name}={r['all']} dacc={r['dacc']} ({per}) ce={r['ce']}{extra}")


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=7)
    ap.add_argument("--max-quot", type=int, default=10)
    ap.add_argument("--red-skip", type=int, default=0,
                    help="skip the top-k reductions (cheaper, but only valid "
                         "when every modulus has ~S significant digits)")
    ap.add_argument("--id-bits", default="16,18,20")
    ap.add_argument("--ood-bits", default="17,19,21")
    ap.add_argument("--n-mod", type=int, default=16,
                    help="training moduli PER bit size")
    ap.add_argument("--n-mod-held", type=int, default=8,
                    help="held-out moduli per bit size (disjoint pool)")
    ap.add_argument("--n-x", type=int, default=1024, help="train x per modulus")
    ap.add_argument("--n-held-x", type=int, default=64)
    # curriculum
    ap.add_argument("--beta-n", type=float, default=0.0,
                    help="curriculum slope on log10(modulus)")
    ap.add_argument("--beta-x", type=float, default=0.0,
                    help="curriculum slope on log10(operand)")
    ap.add_argument("--sched", default="linear",
                    choices=["const", "step", "linear", "exp"])
    ap.add_argument("--anneal-frac", type=float, default=0.5)
    ap.add_argument("--only-bits", type=int, default=0,
                    help="DIAGNOSTIC extreme: weight 1 on this bit size, 0 "
                         "elsewhere, for the whole run")
    # training
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--init-scale", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--eval-n", type=int, default=1536)
    ap.add_argument("--lce-n", type=int, default=512)
    ap.add_argument("--eval-chunk", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--construct", action="store_true")
    ap.add_argument("--basin", action="store_true",
                    help="DIAGNOSTIC: label sensitivity to k corrupted Tmul "
                         "cells, resolved by modulus size")
    ap.add_argument("--basin-ks", default="0,1,2,5,10,20,50,100,200")
    ap.add_argument("--basin-n", type=int, default=1536)
    ap.add_argument("--basin-reps", type=int, default=3)
    ap.add_argument("--grad-equiv", action="store_true",
                    help="GATE: per-row loss weight == per-row gradient scale "
                         "on the logits (the form a submission must use)")
    ap.add_argument("--depth-only", action="store_true")
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--jsonl", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    S, W = args.slots, args.slots + 1
    model = CurrALU(S, 2, 2, args.max_quot, args.tau, False,
                    args.init_scale, args.red_skip).to(dev)
    d = model.depth()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] tree:quotient S={S} W={W} Q={args.max_quot} "
          f"params={n_par:,} DEPTH main={d['main']} (adds={d['adds']} "
          f"reduce={d['reduce']}) N-prefix={d['prefix']} leaves={d['leaves']} "
          f"alphabet={model.alphabet()}", flush=True)
    if args.depth_only:
        return 0

    t_data = time.time()
    data, tr_mod, he_mod, on_mod = build_data(args, dev)
    print(f"[{args.tag}] moduli train="
          f"{ {b: len(v) for b, v in tr_mod.items()} } held="
          f"{ {b: len(v) for b, v in he_mod.items()} } oodn="
          f"{ {b: len(v) for b, v in on_mod.items()} } rows "
          + " ".join(f"{k}={v.n}" for k, v in data.items())
          + f" ({time.time()-t_data:.0f}s)", flush=True)

    # structural sizing check on self-generated values (diagnostic print only)
    qmax = 0
    for b, ms in list(tr_mod.items()) + list(on_mod.items()):
        N = max(ms)
        qmax = max(qmax, (10 * (N - 1) + 9) // N)
    print(f"[{args.tag}] max true quotient under red-skip={args.red_skip}: "
          f"{qmax} (alphabet covers 0..{args.max_quot})", flush=True)
    for nm in ("train", "held_n"):
        print(f"[{args.tag}] dacc trivial floors ({nm}, zero-predictor, "
              f"per-slot-majority): {trivial_floors(data[nm], args.eval_n)}",
              flush=True)
    print(f"[{args.tag}] Tmul coverage per example: "
          f"{table_coverage(data['train'], args.eval_n)}", flush=True)

    try:
        from benchmark import ModelSpec, assert_model_state
        spec = ModelSpec(vocab_size=17, max_seq_len=19,
                         maximum_model_state_elements=500_000_000)
        n_state = assert_model_state(model, spec)
        print(f"[{args.tag}] assert_model_state OK: {n_state:,} / 500,000,000",
              flush=True)
    except Exception as exc:                       # pragma: no cover
        print(f"[{args.tag}] assert_model_state FAILED: {exc}", flush=True)

    if args.construct:
        model.construct()
        for name, coh in data.items():
            r = evaluate(model, coh, args.eval_chunk, True, args.eval_n, True)
            print(f"[{args.tag}] CONSTRUCTED(hard) {fmt(name, r)}", flush=True)
        for name, coh in data.items():
            r = evaluate(model, coh, args.eval_chunk, False, args.eval_n, True)
            print(f"[{args.tag}] CONSTRUCTED(soft) {fmt(name, r)}", flush=True)
        return 0

    if args.basin:
        basin(model, data, args, dev)
        return 0

    if args.grad_equiv:
        # GATE: a per-row loss weight and a per-row GRADIENT SCALE applied to
        # the logits in the forward are the same update.  The submission has to
        # use the second form because `training_loss` receives the tensors
        # flattened by a ragged valid mask (see the submission docstring), so
        # this equivalence is what makes the curriculum expressible at all.
        tr0 = data["train"]
        idx = torch.arange(256, device=dev)
        xo, no, tg, _ = tr0.batch(idx)
        w = torch.rand(256, device=dev) * 2.0
        w = w / w.mean()
        g = {}
        for name in ("weighted_loss", "grad_scale"):
            model.zero_grad(set_to_none=True)
            lg = model(xo, no)
            if name == "weighted_loss":
                ce = F.cross_entropy(lg.reshape(-1, 10), tg.reshape(-1),
                                     reduction="none").view(256, -1).mean(1)
                loss = (w * ce).mean()
            else:
                wv = w.view(-1, 1, 1)
                lg = wv * lg + (1.0 - wv) * lg.detach()
                loss = F.cross_entropy(lg.reshape(-1, 10), tg.reshape(-1))
            loss.backward()
            g[name] = torch.cat([p.grad.reshape(-1).clone()
                                 for p in model.parameters()])
            g[name + "_loss"] = float(loss)
        a, b = g["weighted_loss"], g["grad_scale"]
        rel = float((a - b).norm() / (a.norm() + 1e-12))
        print(f"[{args.tag}] GRAD-EQUIV weighted_loss={g['weighted_loss_loss']:.6f} "
              f"grad_scale_loss={g['grad_scale_loss']:.6f} "
              f"grad_rel_diff={rel:.3e} cos="
              f"{float(F.cosine_similarity(a[None], b[None])):.8f}", flush=True)
        model.zero_grad(set_to_none=True)
        return 0

    ref = CurrALU(S, 2, 2, args.max_quot, 1.0, False, args.init_scale,
                  args.red_skip).to(dev)
    ref.construct()
    for p in ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd, betas=(0.9, 0.95))
    tr = data["train"]
    t0 = time.time()
    tstep = 0.0
    curve = []
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, tr.n, (args.batch,), device=dev, generator=gen)
        xo, no, tg, bt = tr.batch(idx)
        if step == 11:
            torch.cuda.synchronize()
            tstep = time.time()
        logits = model(xo, no)
        ce = F.cross_entropy(logits.reshape(-1, 10), tg.reshape(-1),
                             reduction="none").view(args.batch, -1).mean(1)
        if args.only_bits:
            w = (bt == args.only_bits).float()
            w = w / w.mean().clamp_min(1e-6)
            bn = bx = float("nan")
        else:
            w, bn, bx = curric_weights(xo, no, step, args.steps, args)
        loss = (w * ce).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            lv = float(loss.detach())
            rt = evaluate(model, tr, args.eval_chunk, False, args.eval_n)
            rth = evaluate(model, tr, args.eval_chunk, True, args.eval_n)
            rh = evaluate(model, data["held_n"], args.eval_chunk, True,
                          args.eval_n)
            lce = local_ce(model, ref, tr, args.lce_n)
            curve.append({"step": step, "loss": round(lv, 4),
                          "train_exact": rt["all"],
                          "train_dacc": rt["dacc"],
                          "train_exact_hard": rth["all"],
                          "train_dacc_hard": rth["dacc"],
                          "held_n_exact_hard": rh["all"],
                          "held_n_dacc_hard": rh["dacc"],
                          "train_ce": rt["ce"], "local_ce": lce,
                          "d16": rt.get("d16"), "d18": rt.get("d18"),
                          "d20": rt.get("d20")})
            print(f"[{args.tag}] step={step:>5} loss={lv:.4f} "
                  f"beta=({bn:.2f},{bx:.2f}) local_ce={lce} "
                  f"{fmt('train_soft', rt)} | {fmt('train_hard', rth)} | "
                  f"{fmt('heldN_hard', rh)} ({time.time()-t0:.0f}s)",
                  flush=True)
    torch.cuda.synchronize()
    ms = 1000 * (time.time() - tstep) / max(args.steps - 10, 1) \
        if args.steps > 10 else float("nan")

    out = {"tag": args.tag, "argv": sys.argv[1:], "ms_per_step": round(ms, 2),
           "secs": round(time.time() - t0, 1), "depth": d["main"],
           "curve": curve}
    if not args.timing_only:
        for name, coh in data.items():
            lim = args.eval_n
            rh = evaluate(model, coh, args.eval_chunk, True, lim, True)
            rs = evaluate(model, coh, args.eval_chunk, False, lim, True)
            out[f"{name}_hard"] = rh
            out[f"{name}_soft"] = rs
            print(f"[{args.tag}] FINAL hard {fmt(name, rh)}", flush=True)
            print(f"[{args.tag}] FINAL soft {fmt(name, rs)}", flush=True)
        model.stat = [0.0, 0]
        evaluate(model, tr, args.eval_chunk, False, 512)
        out["state_sharpness"] = round(model.stat[0] / max(model.stat[1], 1), 4)
        model.stat = None
        out["local_ce"] = local_ce(model, ref, tr, args.lce_n)
        nd_dig = []
        for b, ms_ in tr_mod.items():
            nd_dig += digits_le(ms_[0], W)
        out.update(struct_scores(model, nd_dig))
        print(f"[{args.tag}] FINAL state_sharpness={out['state_sharpness']} "
              f"local_ce={out['local_ce']} mul_fn={out['mul_fn']} "
              f"mul_gauge={out['mul_gauge']} add_shift={out['add_shift']} "
              f"sub_shift={out['sub_shift']} ms_per_step={ms:.1f}", flush=True)
    else:
        print(f"[{args.tag}] TIMING ms_per_step={ms:.2f} batch={args.batch} "
              f"peak_mem={torch.cuda.max_memory_allocated()/2**30:.2f}GiB",
              flush=True)
    out["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    if args.jsonl:
        with open(args.jsonl, "a") as fh:
            fh.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CUDA_CACHE_PATH",
                          "/home/scratch.arohan_hw/.nv_computecache")
    raise SystemExit(main())
