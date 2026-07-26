#!/usr/bin/env python
"""LAB ONLY -- a POPULATION of independent ALU replicas trained by one
optimizer.step(), plus a differentiable mixture that selects among them.

WHY.  `alu-credit` Stage 1 measured that with m1-scale operands, the
tree:quotient graph and a per-step (teacher-forced) signal, the DigitALU
reaches `train_exact_hard` 0.951 / `held_exact_hard` 0.946 -- but only in
1 of 10 seeds.  The evaluator runs ONE seed.  A 1-in-10 procedure is worth
nothing unless the restarts happen INSIDE the run.

THE IDEA.  One ALU is ~6,820 parameters; the evaluator's model-state ceiling is
5e8 elements, so tens of thousands of independent replicas fit.  The contract
constrains the LOOP (one forward, one backward, one optimizer.step() per batch),
not the model's internal width.  Give the model a replica dimension P: every
replica has its own tables, all see the same batch, their gradients are
disjoint, so one optimizer.step() trains P independent runs simultaneously.

SELECTION.  P learned logits `alpha` define a mixture over the replicas'
output distributions.  The mixture is scored by the ORDINARY end-of-chain task
loss, so the loss itself upweights whichever replica is right; the gradient path
from the loss to every replica's parameters is unbroken.  At eval we report BOTH
the mixture and the argmax replica -- `depth-controller` measured that
committing to the mode rather than the blend took Medium from 0 to 4.

COMPLIANCE.  `--tf` (teacher forcing) replays a tape recorded from a
CONSTRUCTED copy of the model, so every row that uses it is a LAB DIAGNOSTIC
(rules 2 and 7) and can never be a submission.  The population machinery and
the mixture head are LEGAL by themselves; `--tf 0` runs the legal end-of-chain
objective only.  Self-generated operands only; nothing under data/generated/ is
opened.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_alu_depth import digits_le  # noqa: E402

BIG = 30.0


class PopALU(nn.Module):
    """tree:quotient DigitALU with a leading replica dimension.

    Every state tensor is (P, n, ...) and every table read carries a `p` index,
    so replica p's forward touches only replica p's parameters.  The graph is
    identical to `probe_alu_depth.DigitALU(mul_mode='tree',
    reduce_mode='quotient', scan_mode='serial')`; only the replica axis is new.
    """

    def __init__(self, P: int, slots: int, n_carry: int = 2, n_borrow: int = 2,
                 max_quot: int = 10, tau: float = 1.0, hard: bool = False,
                 tie: bool = False, tie_sub: bool = False,
                 init_scale: float = 0.5, scale_spread: float = 0.0):
        super().__init__()
        self.P, self.S = P, slots
        self.W = slots + 1
        self.Ca, self.Cb = n_carry, n_borrow
        self.Q = max_quot
        self.tau, self.hard = tau, hard
        self.tie, self.tie_sub = tie, tie_sub
        self.fast = True
        self.needed = list(range(max_quot + 2))
        # per-replica init scale (a population can cover a hyperparameter too)
        if scale_spread > 0:
            lo = math.log(init_scale / scale_spread)
            hi = math.log(init_scale * scale_spread)
            sc = torch.exp(torch.linspace(lo, hi, P))
        else:
            sc = torch.full((P,), float(init_scale))
        self.register_buffer("init_scale", sc, persistent=False)

        def rp(*shape):
            t = torch.randn(P, *shape)
            return nn.Parameter(t * sc.view(P, *([1] * len(shape))))

        self.Tmul = rp(10, 10, 20)
        self.Tadd = rp(10, 10, n_carry, 10 + n_carry)
        self.Tsub = rp(10, 10, n_borrow, 10 + n_borrow)
        self.zero = rp(10)
        self.carry0 = rp(n_carry)
        self.borrow0 = rp(n_borrow)
        self.sel_w = rp(2 * n_borrow)
        self.sel_b = nn.Parameter(torch.zeros(P))
        self.copy_scale = nn.Parameter(torch.zeros(P))
        # mixture logits over replicas -- the differentiable selector
        self.alpha = nn.Parameter(torch.zeros(P))
        self.sel_tau = 1.0
        self.sel_hard = False
        # taps
        self.mode = None            # None | 'record' | 'force' | 'shape'
        self.tape: list = []
        self.tpos = 0
        self.tf_loss = None         # (P,)
        self.tf_n = 0
        self.tf_p = 1.0

    # ---- replica-wise parameter groups, for reinit / diagnostics ----
    def replica_params(self):
        return [self.Tmul, self.Tadd, self.Tsub, self.zero, self.carry0,
                self.borrow0, self.sel_w, self.sel_b, self.copy_scale]

    # ---- weight ties (a reparameterisation; the forward is unchanged) ----
    @property
    def Tmul_eff(self):
        return 0.5 * (self.Tmul + self.Tmul.transpose(1, 2)) if self.tie \
            else self.Tmul

    @property
    def Tadd_eff(self):
        return 0.5 * (self.Tadd + self.Tadd.transpose(1, 2)) if self.tie \
            else self.Tadd

    @property
    def Tsub_eff(self):
        if not self.tie_sub:
            return self.Tsub
        dig = self.Tadd_eff[..., :10].permute(0, 4, 2, 3, 1)   # (P,w,v,c,u)
        return torch.cat([dig, self.Tsub[..., 10:]], -1)

    # ---------------- construction (LAB DIAGNOSTIC ONLY) ----------------
    @torch.no_grad()
    def construct(self):
        t = torch.full((10, 10, 20), -BIG)
        for a in range(10):
            for b in range(10):
                t[a, b, (a * b) % 10] = BIG
                t[a, b, 10 + (a * b) // 10] = BIG
        self.Tmul.copy_(t.to(self.Tmul).expand_as(self.Tmul))
        t = torch.full((10, 10, self.Ca, 10 + self.Ca), -BIG)
        for u in range(10):
            for v in range(10):
                for c in range(2):
                    s = u + v + c
                    t[u, v, c, s % 10] = BIG
                    t[u, v, c, 10 + s // 10] = BIG
        self.Tadd.copy_(t.to(self.Tadd).expand_as(self.Tadd))
        t = torch.full((10, 10, self.Cb, 10 + self.Cb), -BIG)
        for u in range(10):
            for v in range(10):
                for c in range(2):
                    s = u - v - c
                    t[u, v, c, s % 10] = BIG
                    t[u, v, c, 10 + (1 if s < 0 else 0)] = BIG
        self.Tsub.copy_(t.to(self.Tsub).expand_as(self.Tsub))
        for P_, n in ((self.zero, 10), (self.carry0, self.Ca),
                      (self.borrow0, self.Cb)):
            v = torch.full((n,), -BIG)
            v[0] = BIG
            P_.copy_(v.to(P_).expand_as(P_))
        w = torch.zeros(2 * self.Cb)
        w[0], w[1] = BIG / 2, -BIG / 2
        w[self.Cb + 0], w[self.Cb + 1] = -BIG / 2, BIG / 2
        self.sel_w.copy_(w.to(self.sel_w).expand_as(self.sel_w))
        self.sel_b.zero_()
        self.copy_scale.zero_()

    # ---------------- machinery ----------------
    def _sm(self, logits):
        p = F.softmax(logits / self.tau, -1)
        if self.hard:
            h = F.one_hot(p.argmax(-1), p.shape[-1]).to(p.dtype)
            p = h + p - p.detach()
        return p

    def _tap(self, x):
        """x: (P, n, D, 10)."""
        if self.mode == 'record':
            self.tape.append(x.detach())
            return x
        if self.mode == 'force':
            truth = self.tape[self.tpos]
            self.tpos += 1
            tgt = truth.argmax(-1)                       # (1, n, D)
            lg = x.clamp_min(1e-9).log()
            ce = F.cross_entropy(lg.reshape(-1, 10),
                                 tgt.expand(x.shape[0], *tgt.shape[1:])
                                    .reshape(-1), reduction="none")
            self.tf_loss = self.tf_loss + ce.view(x.shape[0], -1).mean(1)
            self.tf_n += 1
            if self.tf_p >= 1.0:
                return truth.expand_as(x)
            m = torch.rand(x.shape[0], x.shape[1],
                           *([1] * (x.dim() - 2)), device=x.device) < self.tf_p
            return torch.where(m, truth.expand_as(x), x)
        return x

    def add_scan(self, r, addend):
        """r, addend: (P, n, W, 10) -> (P, n, W, 10)."""
        P, n = r.shape[0], r.shape[1]
        cs = self.copy_scale.view(P, 1, 1)
        c = self._sm(self.carry0)[:, None].expand(P, n, self.Ca)
        outs = []
        Ta = self.Tadd_eff
        for m in range(r.shape[2]):
            o = torch.einsum("pnu,pnv,pnc,puvco->pno",
                             r[:, :, m], addend[:, :, m], c, Ta)
            outs.append(self._sm(o[..., :10] + cs * r[:, :, m]))
            c = self._sm(o[..., 10:] + cs * c)
        return self._tap(torch.stack(outs, 2))

    def sub_scan(self, r, sub):
        P, n = r.shape[0], r.shape[1]
        cs = self.copy_scale.view(P, 1, 1)
        c = self._sm(self.borrow0)[:, None].expand(P, n, self.Cb)
        outs = []
        Ts = self.Tsub_eff
        for m in range(r.shape[2]):
            o = torch.einsum("pnu,pnv,pnc,puvco->pno",
                             r[:, :, m], sub[:, :, m], c, Ts)
            outs.append(self._sm(o[..., :10] + cs * r[:, :, m]))
            c = self._sm(o[..., 10:] + cs * c)
        return self._tap(torch.stack(outs, 2)), c

    def tree_sum(self, regs):
        while len(regs) > 1:
            carry = [regs[-1]] if len(regs) % 2 else []
            pairs = [(regs[i], regs[i + 1]) for i in range(0, len(regs) - 1, 2)]
            n = regs[0].shape[1]
            out = self.add_scan(torch.cat([a for a, _ in pairs], 1),
                                torch.cat([c for _, c in pairs], 1))
            regs = [out[:, i * n:(i + 1) * n] for i in range(len(pairs))] + carry
        return regs[0]

    def multiples(self, ndig):
        """{m*N} built from digits(N) with the SAME learned Tadd. (P,M,W,10)."""
        P, W = self.P, self.W
        z = self._sm(self.zero)
        have = {0: z[:, None, None, :].expand(P, 1, W, 10),
                1: ndig[None, None].expand(P, 1, W, 10)}
        target = set(self.needed)
        while not target <= set(have):
            newly = []
            for t in sorted(target - set(have)):
                cand = [a for a in have if a <= t - a and (t - a) in have]
                if cand:
                    newly.append((t, max(cand), t - max(cand)))
            if not newly:
                mx = max(have)
                newly = [(2 * mx, mx, mx)]
            A = torch.cat([have[a] for _, a, _ in newly], 1)
            B = torch.cat([have[b] for _, _, b in newly], 1)
            out = self.add_scan(A, B)
            for i, (t, _, _) in enumerate(newly):
                have[t] = out[:, i:i + 1]
        return torch.cat([have[m] for m in self.needed], 1)

    def quot_reduce(self, r, mults):
        """One learned quotient digit + one subtraction, batched over the M
        candidate multiples.

        `fast` is an ALGEBRAICALLY IDENTICAL reassociation, not a different
        graph: the subtrahend digit `mults[p,m,w,v]` does not depend on the
        example, so it can be contracted into Tsub once per forward
        (Tm[p,m,w,u,c,o] = sum_v mults*Tsub, ~17k elements per replica) instead
        of being broadcast to every one of the b*M scan rows.  That removes the
        (u,v,c) outer product over b*M rows, which is the whole cost of the
        population at large P.  `--slow-quot` keeps the naive form so the
        report can quote both.
        """
        P, b, M = r.shape[0], r.shape[1], mults.shape[1]
        W = self.W
        if not self.fast:
            rr = r[:, :, None].expand(P, b, M, W, 10).reshape(P, b * M, W, 10)
            ss = mults[:, None].expand(P, b, M, W, 10).reshape(P, b * M, W, 10)
            t, c = self.sub_scan(rr, ss)
            t = t.view(P, b, M, W, 10)
            c = c.view(P, b, M, self.Cb)
        else:
            Tm = self._Tm                                # (P,M,W,10,Cb,10+Cb)
            cs = self.copy_scale.view(P, 1, 1, 1)
            c = self._sm(self.borrow0)[:, None, None].expand(P, b, M, self.Cb)
            outs = []
            for m in range(W):
                rw = r[:, :, m]                          # (P,b,10)
                o = torch.einsum("pbu,pbmc,pmuco->pbmo", rw, c, Tm[:, :, m])
                outs.append(self._sm(o[..., :10] + cs * rw[:, :, None]))
                c = self._sm(o[..., 10:] + cs * c)
            t = self._tap(torch.stack(outs, 3))          # (P,b,M,W,10)
        pair = torch.cat([c[:, :, :-1], c[:, :, 1:]], dim=-1)   # (P,b,M-1,2Cb)
        w = self._sm(torch.einsum("pbmk,pk->pbm", pair, self.sel_w)
                     + self.sel_b[:, None, None])
        self.last_q = w
        return self._tap(torch.einsum("pbm,pbmwo->pbwo", w, t[:, :, :-1]))

    def forward(self, s, ndig):
        """s: (b, S, 10) shared across replicas.  ndig: (W, 10).
        returns log-probs (P, b, S, 10)."""
        P, S, W = self.P, self.S, self.W
        b = s.shape[0]
        self.tpos = 0
        F2 = 2 * S
        z = self._sm(self.zero)[:, None].expand(P, b, 10)
        mults = self.multiples(ndig)
        self._Tm = torch.einsum("pmwv,puvco->pmwuco", mults, self.Tsub_eff) \
            if self.fast else None
        se = s[None].expand(P, b, S, 10)
        Tm = self.Tmul_eff
        prod = {}
        for i in range(S):
            for j in range(S):
                o = torch.einsum("pbu,pbv,puvo->pbo", se[:, :, i], se[:, :, j],
                                 Tm)
                prod[(i, j)] = (self._sm(o[..., :10]), self._sm(o[..., 10:]))
        # pack partial products into full-width leaves
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
                leaves.append(torch.stack(cols, 2))
        Pr = self.tree_sum(leaves)
        r = z[:, :, None].expand(P, b, W, 10)
        for t in range(F2 - 1, -1, -1):
            r = torch.cat([Pr[:, :, t:t + 1], r[:, :, :W - 1]], dim=2)
            if t <= S:
                r = self.quot_reduce(r, mults)
        return torch.log(r[:, :, :S] + 1e-9)

    # ---- mixture over replicas (the differentiable selector) ----
    def mix_probs(self, logits):
        """logits: (P,b,S,10) log-probs -> (b,S,10) mixture probabilities.

        `sel_hard` commits to the mode with a straight-through estimator: the
        forward uses ONLY the argmax replica (so a correct replica is never
        diluted by 31 wrong ones) while the backward is the soft mixture's, so
        every replica still receives gradient through alpha.  This is the
        population analogue of `depth-controller`'s "commit to the mode, not
        the blend", which took its Medium result from 0 to 4."""
        w = F.softmax(self.alpha / self.sel_tau, 0)
        if self.sel_hard:
            h = F.one_hot(w.argmax(0), w.shape[0]).to(w.dtype)
            w = h + w - w.detach()
        return torch.einsum("p,pbso->bso", w, logits.exp())


# ---------------------------------------------------------------- diagnostics
@torch.no_grad()
def struct_scores(model, p, ndigits):
    """Gauge-invariant structure scores for replica p (same definitions as
    lab/probe_tf2.py)."""
    out = {}
    am = model.Tmul_eff[p, ..., :10].argmax(-1)
    groups = {}
    for a in range(10):
        for b in range(10):
            groups.setdefault((a * b) % 10, []).append(int(am[a, b]))
    k = sum(v.count(max(set(v), key=v.count)) for v in groups.values())
    out["mul_fn"] = round(k / 100, 3)
    pi = {c: max(set(v), key=v.count) for c, v in groups.items()}
    out["mul_gauge"] = round(len(set(pi.values())) / len(pi), 3)
    for nm, T, cols in (("add_shift", model.Tadd_eff[p], range(10)),
                        ("sub_shift", model.Tsub_eff[p], sorted(set(ndigits)))):
        k = n = 0
        for v in cols:
            for c in range(T.shape[2]):
                a = T[:, v, c, :10].argmax(-1)
                k += max(sum(1 for u in range(10) if int(a[u]) == (u + s) % 10)
                         for s in range(10))
                n += 10
        out[nm] = round(k / n, 3)
    return out


@torch.no_grad()
def eval_pop(model, inp, tgt, nd, discrete=False, chunk=256):
    """Returns per-replica exact accuracy (P,), mixture accuracy, argmax-replica
    accuracy, oracle-best accuracy, and the replica ranking by alpha."""
    model.eval()
    was, model.hard = model.hard, True if discrete else model.hard
    P = model.P
    ok = torch.zeros(P, device=inp.device)
    ok_mix = 0.0
    ce_p = torch.zeros(P, device=inp.device)
    n = inp.shape[0]
    for i in range(0, n, chunk):
        lg = model(inp[i:i + chunk], nd)                    # (P,c,S,10)
        t = tgt[i:i + chunk]
        ok += (lg.argmax(-1) == t[None]).all(-1).float().sum(1)
        ce_p += -lg.gather(-1, t[None].expand(P, *t.shape)[..., None]) \
            .squeeze(-1).sum((1, 2))
        mp = model.mix_probs(lg)
        ok_mix += (mp.argmax(-1) == t).all(-1).float().sum().item()
    model.hard = was
    model.train()
    acc = (ok / n)
    star = int(model.alpha.argmax().item())
    ce = ce_p / (n * tgt.shape[1])
    cstar = int(ce.argmin().item())
    return {"per": acc, "mix": ok_mix / n, "argmax": float(acc[star]),
            "best": float(acc.max()), "star": star,
            "ce": ce, "ce_star": cstar, "ce_argmin": float(acc[cstar])}


# ---------------------------------------------------------------- optimizer
class AdamWReinit(torch.optim.AdamW):
    """AdamW that re-randomises the worst replicas every `period` steps.

    COMPLIANCE NOTE (flagged, not shipped): re-initialising parameters inside
    optimizer.step() does not cut the autograd path -- the path from loss to
    parameters is rebuilt and intact on every step -- but it IS a discrete,
    non-gradient jump applied to the parameters that produce the prediction,
    and the replica that survives to eval may have been re-randomised at an
    arbitrary point.  I read that as a gray area under rule 8 and report it as
    DIAGNOSTIC.  A submission should prefer the mixture head, which is
    unambiguous.
    """

    def __init__(self, params, model, period=0, frac=0.5, warm=0, **kw):
        super().__init__(params, **kw)
        self.model, self.period, self.frac, self.warm = model, period, frac, warm
        self.n_steps = 0
        self.n_reinit = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        self.n_steps += 1
        m = self.model
        if self.period and self.n_steps > self.warm \
                and self.n_steps % self.period == 0:
            P = m.P
            k = max(1, int(P * self.frac))
            worst = torch.argsort(m.alpha)[:k]          # lowest mixture logits
            for p in m.replica_params():
                sc = m.init_scale[worst].view(-1, *([1] * (p.dim() - 1)))
                fresh = torch.randn(k, *p.shape[1:], device=p.device,
                                    dtype=p.dtype) * sc
                if p is m.sel_b or p is m.copy_scale:
                    fresh = torch.zeros_like(fresh)
                p[worst] = fresh
                st = self.state.get(p)
                if st:
                    st["exp_avg"][worst] = 0
                    st["exp_avg_sq"][worst] = 0
            m.alpha[worst] = m.alpha.mean()
            st = self.state.get(m.alpha)
            if st:
                st["exp_avg"][worst] = 0
                st["exp_avg_sq"][worst] = 0
            self.n_reinit += k
        return loss


def pop_clip_(model, max_norm):
    """Per-replica gradient clipping.  A GLOBAL clip couples replicas: one
    exploding replica shrinks everyone's step.  This keeps each replica's update
    identical to what it would be at P=1.  (The evaluator clips globally --
    see the report; with Adam the difference is a common scale factor.)"""
    ps = [p for p in model.replica_params() if p.grad is not None]
    if not ps:
        return
    sq = torch.stack([p.grad.reshape(p.shape[0], -1).pow(2).sum(1) for p in ps])
    nrm = sq.sum(0).sqrt()                                   # (P,)
    scale = (max_norm / (nrm + 1e-6)).clamp(max=1.0)
    for p in ps:
        p.grad.mul_(scale.view(-1, *([1] * (p.dim() - 1))))


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--modulus", type=int, default=10403)
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--train-x", type=int, default=8000)
    ap.add_argument("--held-x", type=int, default=1024)
    ap.add_argument("--max-quot", type=int, default=10)
    ap.add_argument("--tie", action="store_true")
    ap.add_argument("--tie-sub", action="store_true")
    ap.add_argument("--tf", type=float, default=1.0,
                    help="weight on the DIAGNOSTIC teacher-forced local CE; "
                         "0 disables teacher forcing entirely (LEGAL)")
    ap.add_argument("--tf-p", type=float, default=1.0)
    ap.add_argument("--sel", type=float, default=1.0,
                    help="weight on the LEGAL end-of-chain mixture CE")
    ap.add_argument("--sel-detach", action="store_true",
                    help="detach replica outputs in the mixture loss so only "
                         "alpha learns from it (isolates selection)")
    ap.add_argument("--sel-tau", type=float, default=1.0)
    ap.add_argument("--sel-hard", action="store_true",
                    help="straight-through commit to the argmax replica in "
                         "the mixture loss instead of blending")
    ap.add_argument("--sel-tau-final", type=float, default=None)
    ap.add_argument("--sel-lr", type=float, default=None)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--lr-final", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--init-scale", type=float, default=0.5)
    ap.add_argument("--scale-spread", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--global-clip", action="store_true",
                    help="clip the whole population jointly, as the evaluator "
                         "does, instead of per replica")
    ap.add_argument("--reinit-period", type=int, default=0)
    ap.add_argument("--reinit-frac", type=float, default=0.5)
    ap.add_argument("--reinit-warm", type=int, default=0)
    ap.add_argument("--avg-replicas", action="store_true",
                    help="DIAGNOSTIC: also report the accuracy of the "
                         "parameter-averaged replica (weight averaging)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--eval-chunk", type=int, default=256)
    ap.add_argument("--eval-n", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--construct", action="store_true")
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--slow-quot", action="store_true",
                    help="naive quotient reduction (broadcast the multiples to "
                         "every scan row) -- the straightforward implementation")
    ap.add_argument("--jsonl", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    N, S, P = args.modulus, args.slots, args.pop
    units = [x for x in range(1, N) if math.gcd(x, N) == 1]
    g = torch.Generator().manual_seed(args.split_seed)
    perm = torch.randperm(len(units), generator=g).tolist()
    tr_x = [units[i] for i in perm[: args.train_x]]
    he_x = [units[i] for i in perm[args.train_x: args.train_x + args.held_x]]
    dev = torch.device(args.device)

    def tens(xs):
        inp = torch.zeros(len(xs), S, 10)
        tgt = torch.zeros(len(xs), S, dtype=torch.long)
        for r, x in enumerate(xs):
            for i, d in enumerate(digits_le(x, S)):
                inp[r, i, d] = 1.0
            for i, d in enumerate(digits_le((x * x) % N, S)):
                tgt[r, i] = d
        return inp.to(dev), tgt.to(dev)

    xin, xt = tens(tr_x)
    hin, ht = tens(he_x)
    W = S + 1
    nd = torch.zeros(W, 10)
    for i, d in enumerate(digits_le(N, W)):
        nd[i, d] = 1.0
    nd = nd.to(dev)
    nd_dig = digits_le(N, W)

    model = PopALU(P, S, 2, 2, args.max_quot, 1.0, False, args.tie,
                   args.tie_sub, args.init_scale, args.scale_spread).to(dev)
    ref = PopALU(1, S, 2, 2, args.max_quot, 1.0, False, args.tie,
                 args.tie_sub).to(dev)
    model.fast = ref.fast = not args.slow_quot
    model.sel_hard = args.sel_hard
    ref.construct()
    for p in ref.parameters():
        p.requires_grad_(False)

    n_par = sum(p.numel() for p in model.parameters())
    per_rep = (n_par - P) // P
    print(f"[{args.tag}] POP={P} N={N} S={S} tree:quotient tie={args.tie} "
          f"tie_sub={args.tie_sub} params={n_par:,} (~{per_rep:,}/replica) "
          f"train={len(tr_x)} held={len(he_x)} units={len(units)}", flush=True)

    # ---- state-ceiling check against the real evaluator API ----
    try:
        from benchmark import ModelSpec, assert_model_state
        spec = ModelSpec(vocab_size=17, max_seq_len=64,
                         maximum_model_state_elements=500_000_000)
        n_state = assert_model_state(model, spec)
        print(f"[{args.tag}] assert_model_state OK: {n_state:,} / 500,000,000 "
              f"({100 * n_state / 5e8:.4f}%)", flush=True)
    except Exception as exc:                       # pragma: no cover
        print(f"[{args.tag}] assert_model_state FAILED: {exc}", flush=True)
        return 1

    if args.construct:
        model.construct()
        r = eval_pop(model, xin[:args.eval_n], xt[:args.eval_n], nd, True,
                     args.eval_chunk)
        rh = eval_pop(model, hin, ht, nd, True, args.eval_chunk)
        print(f"[{args.tag}] CONSTRUCTED train_exact_hard={r['best']:.3f} "
              f"held_exact_hard={rh['best']:.3f} mix={r['mix']:.3f}", flush=True)
        return 0

    groups = [{"params": [p for p in model.parameters() if p is not model.alpha],
               "lr": args.lr},
              {"params": [model.alpha],
               "lr": args.sel_lr if args.sel_lr is not None else args.lr}]
    okw = dict(weight_decay=args.wd, betas=(0.9, 0.95))
    if args.reinit_period:
        opt = AdamWReinit(groups, model, args.reinit_period, args.reinit_frac,
                          args.reinit_warm, **okw)
    else:
        opt = torch.optim.AdamW(groups, **okw)

    t0 = time.time()
    tstep = 0.0
    for step in range(1, args.steps + 1):
        if args.sel_tau_final is not None:
            f = step / args.steps
            model.sel_tau = math.exp((1 - f) * math.log(args.sel_tau)
                                     + f * math.log(args.sel_tau_final))
        else:
            model.sel_tau = args.sel_tau
        if args.lr_final is not None:
            f = step / args.steps
            lr = math.exp((1 - f) * math.log(args.lr) + f * math.log(args.lr_final))
            opt.param_groups[0]["lr"] = lr
        idx = torch.randint(0, xin.shape[0], (args.batch,), device=dev)
        bi, bt = xin[idx], xt[idx]
        if step == 11:
            torch.cuda.synchronize()
            tstep = time.time()
        loss = torch.zeros((), device=dev)
        if args.tf > 0:
            with torch.no_grad():
                ref.mode, ref.tape = 'record', []
                ref(bi, nd)
                ref.mode = None
            model.mode, model.tape = 'force', ref.tape
            model.tf_loss = torch.zeros(P, device=dev)
            model.tf_n, model.tf_p = 0, args.tf_p
            model(bi, nd)
            model.mode = None
            loss = loss + args.tf * (model.tf_loss / max(model.tf_n, 1)).mean()
        if args.sel > 0 or args.tf == 0:
            logits = model(bi, nd)
            lg = logits.detach() if args.sel_detach else logits
            mp = model.mix_probs(lg).clamp_min(1e-9)
            mix_ce = F.nll_loss(mp.log().reshape(-1, 10), bt.reshape(-1))
            loss = loss + max(args.sel, 0.0 if args.tf > 0 else 1.0) * mix_ce
            if args.tf == 0:
                # LEGAL run: also give every replica its own end-of-chain CE so
                # the population is trained, not just the selector.
                loss = loss + F.cross_entropy(
                    logits.reshape(-1, 10),
                    bt[None].expand(P, *bt.shape).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip:
            if args.global_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            else:
                pop_clip_(model, args.clip)
                torch.nn.utils.clip_grad_norm_([model.alpha], args.clip)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            r = eval_pop(model, xin[:args.eval_n], xt[:args.eval_n], nd, True,
                         args.eval_chunk)
            w = F.softmax(model.alpha / model.sel_tau, 0)
            # per-replica local CE on this batch, free from the training pass
            lce = (model.tf_loss / max(model.tf_n, 1)).detach() \
                if args.tf > 0 else torch.full((P,), float('nan'), device=dev)
            print(f"[{args.tag}] step={step:>5} loss={loss.item():.4f} "
                  f"best={r['best']:.3f} mix={r['mix']:.3f} "
                  f"argmax={r['argmax']:.3f} ce_argmin={r['ce_argmin']:.3f} "
                  f"star={r['star']} "
                  f"n_basin={(r['per'] > 0.5).sum().item()}/{P} "
                  f"lce_min={lce.min().item():.4f} "
                  f"n_lce={(lce < 0.006).sum().item()} "
                  f"wmax={w.max().item():.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    torch.cuda.synchronize()
    ms = 1000 * (time.time() - tstep) / max(args.steps - 10, 1) \
        if args.steps > 10 else float('nan')

    out = {"tag": args.tag, "argv": sys.argv[1:], "pop": P,
           "ms_per_step": round(ms, 2),
           "secs": round(time.time() - t0, 1)}
    if not args.timing_only:
        tr = eval_pop(model, xin[:args.eval_n], xt[:args.eval_n], nd, True,
                      args.eval_chunk)
        trs = eval_pop(model, xin[:args.eval_n], xt[:args.eval_n], nd, False,
                       args.eval_chunk)
        he = eval_pop(model, hin, ht, nd, True, args.eval_chunk)
        # per-replica local CE (DIAGNOSTIC: needs the constructed tape)
        with torch.no_grad():
            ref.mode, ref.tape = 'record', []
            ref(xin[:512], nd)
            ref.mode = None
            model.mode, model.tape = 'force', ref.tape
            model.tf_loss, model.tf_n, model.tf_p = \
                torch.zeros(P, device=dev), 0, 1.0
            model(xin[:512], nd)
            model.mode = None
            lce = (model.tf_loss / max(model.tf_n, 1)).cpu()
        acc = tr["per"].cpu()
        hacc = he["per"].cpu()
        order = torch.argsort(acc, descending=True)
        w = F.softmax(model.alpha / model.sel_tau, 0).detach().cpu()
        out.update({
            "train_exact_hard_best": round(float(acc.max()), 4),
            "train_exact_hard_mix": round(tr["mix"], 4),
            "train_exact_hard_argmax": round(tr["argmax"], 4),
            "train_exact_hard_ce_argmin": round(tr["ce_argmin"], 4),
            "train_exact_soft_mix": round(trs["mix"], 4),
            "held_exact_hard_best": round(float(hacc.max()), 4),
            "held_exact_hard_mix": round(he["mix"], 4),
            "held_exact_hard_argmax": round(he["argmax"], 4),
            "held_exact_hard_ce_argmin": round(float(hacc[tr["ce_star"]]), 4),
            "n_basin": int((acc > 0.5).sum()),
            "n_basin_20": int((acc > 0.2).sum()),
            "n_lce_006": int((lce < 0.006).sum()),
            "n_lce_010": int((lce < 0.010).sum()),
            "star": tr["star"],
            "ce_star": tr["ce_star"],
            "star_rank": int((order == tr["star"]).nonzero()[0, 0]),
            "alpha_wmax": round(float(w.max()), 4),
            "local_ce_best": round(float(lce.min()), 5),
            "local_ce_star": round(float(lce[tr["star"]]), 5),
            "acc_sorted": [round(float(v), 3) for v in acc[order][:16]],
            "lce_sorted": [round(float(lce[i]), 4) for i in order[:16]],
        })
        out.update(struct_scores(model, int(order[0]), nd_dig))
        if args.avg_replicas:
            avg = PopALU(1, S, 2, 2, args.max_quot, 1.0, False, args.tie,
                         args.tie_sub).to(dev)
            avg.fast = model.fast
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name == "alpha":
                        continue
                    getattr(avg, name).copy_(p.mean(0, keepdim=True))
            ra = eval_pop(avg, xin[:args.eval_n], xt[:args.eval_n], nd, True,
                          args.eval_chunk)
            out["avg_exact_hard"] = round(float(ra["per"][0]), 4)
        if args.reinit_period:
            out["n_reinit"] = opt.n_reinit
        print(f"[{args.tag}] FINAL " + " ".join(
            f"{k}={v}" for k, v in out.items() if k not in ("argv", "tag")),
            flush=True)
    else:
        print(f"[{args.tag}] TIMING ms_per_step={ms:.2f} pop={P} "
              f"batch={args.batch} "
              f"peak_mem={torch.cuda.max_memory_allocated()/2**30:.2f}GiB",
              flush=True)
    out["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    if args.jsonl:
        with open(args.jsonl, "a") as fh:
            fh.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
