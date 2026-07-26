#!/usr/bin/env python
"""Relational / dual-path training signals for `DigitALU`, at Stage-1 conditions.

`alu-credit` Stage 1 established that the discrete solution IS reachable by
gradient descent given the right per-step signal (train_exact_hard 0.951 with
teacher forcing at m1 scale on tree:quotient), and that end-of-chain
supervision alone never leaves the floor.  Target propagation -- the obvious
legal substitute -- collapses to a trivial mutual fixed point because
self-consistency has no content.

This probe tests signals that are NOT self-consistency:

  1. RELATIONAL (`--rel`).  Assert that the learned map has an additive-increment
     structure: there exist a learned unary `inc`, a learned composition (the
     model's own mod-N add) and a learned increment `D` with

         f(inc(x)) = f(x) (+) D(x)      for all x the model sees.

     Three forms of D, in increasing specificity (see --rel):
       free   -- D is a free learned MLP.  Asserts only "an additive-increment
                 structure exists".  LEGAL (conservative variant).
       affine -- D(x) = (a (x) x) (+) b with a, b learned registers, using the
                 model's OWN modular multiply/add.  Asserts the map is "affine
                 in the increment", i.e. quadratic.  LEGAL-BUT-FLAGGED: it
                 encodes a task-specific structural fact (degree 2), even though
                 no coefficient is supplied.  Reported separately, never mixed.
       true   -- a, b, c pinned to the digits of 2, 1, 1.  This IS the true
                 identity (x+1)^2 = x^2 + 2x + 1.  **ILLEGAL, DIAGNOSTIC ONLY**
                 (rule 2/7); it exists to measure the family's ceiling before
                 spending budget on legal variants.

  2. DUAL-PATH AGREEMENT (`--dual`).  Compute the same squaring twice by two
     structurally different graphs that SHARE every parameter, and require the
     two answers to agree.  Unlike target propagation there are no free
     auxiliary parameters, so agreement cannot be bought by inventing a latent;
     the only trivial solution is a degenerate table, which `mul_gauge` detects.
       fold   -- leaf sum by a chained left fold instead of a balanced tree
                 (an associativity constraint on the learned adder)
       redall -- reduce after EVERY product digit instead of the last S+1
                 (a constraint on the learned reduction schedule)
       horner -- the full 257-step Horner graph vs the 39-step tree
     LEGAL: no arithmetic is supplied; the constraint is that one set of tables
     computes one function regardless of the order it is composed in.

  3. ALGEBRAIC RE-SCREEN (`--sym`, `--inv`, `--assoc`).  `alu-credit` measured
     commutativity and Tsub.Tadd = id as null -- at e1 scale on the 257-step
     untied graph, a setting Stage 1 showed is the wrong one.  Re-run here.
     `--assoc` (register-level associativity of the learned multi-digit adder)
     is new.

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


class RelALU(DigitALU):
    """DigitALU + weight ties (from probe_tf2) + a general two-operand modular
    multiply and modular add built from the SAME tables, + alternate graph
    shapes, + a record/force tap used only to MEASURE local_ce."""

    def __init__(self, *a, tie=False, tie_sub=False, **kw):
        self.tie, self.tie_sub = tie, tie_sub
        self.mode = None            # None | 'record' | 'force'
        self.tape: list = []
        self.tpos = 0
        self.tf_loss = None
        self.tf_n = 0
        super().__init__(*a, **kw)

    # ---- weight ties (a reparameterisation; the forward is unchanged) ----
    @property
    def Tmul_eff(self):
        return 0.5 * (self.Tmul + self.Tmul.transpose(0, 1)) if self.tie \
            else self.Tmul

    @property
    def Tadd_eff(self):
        return 0.5 * (self.Tadd + self.Tadd.transpose(0, 1)) if self.tie \
            else self.Tadd

    @property
    def Tsub_eff(self):
        if not self.tie_sub:
            return self.Tsub
        dig = self.Tadd_eff[..., :10].permute(3, 1, 2, 0)      # (w,v,c,u)
        return torch.cat([dig, self.Tsub[..., 10:]], -1)

    # ---- the tap (measurement only: record the true trace, force it) ----
    def _tap(self, x):
        if self.mode == 'record':
            self.tape.append(x.detach())
            return x
        if self.mode == 'force':
            truth = self.tape[self.tpos]
            self.tpos += 1
            lg = x.clamp_min(1e-9).log()
            self.tf_loss = self.tf_loss + F.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), truth.argmax(-1).reshape(-1))
            self.tf_n += 1
            return truth
        return x

    # ---- scans, re-expressed against the tied tensors and tapped ----
    def add_scan(self, r, addend):
        c = self._sm(self.carry0).expand(r.shape[0], self.Ca)
        outs = []
        for m in range(r.shape[1]):
            o = torch.einsum("bu,bv,bc,uvco->bo", r[:, m], addend[:, m], c,
                             self.Tadd_eff)
            outs.append(self._sm(o[:, :10] + self.copy_scale * r[:, m]))
            c = self._sm(o[:, 10:] + self.copy_scale * c)
        return self._tap(torch.stack(outs, 1))

    def sub_scan(self, r, sub):
        b = r.shape[0]
        c = self._sm(self.borrow0).expand(b, self.Cb)
        outs = []
        for m in range(r.shape[1]):
            o = torch.einsum("bu,bv,bc,uvco->bo", r[:, m], sub[:, m], c,
                             self.Tsub_eff)
            outs.append(self._sm(o[:, :10] + self.copy_scale * r[:, m]))
            c = self._sm(o[:, 10:] + self.copy_scale * c)
        return self._tap(torch.stack(outs, 1)), c

    def quot_reduce(self, r, mults):
        b, M = r.shape[0], mults.shape[0]
        rr = r[:, None].expand(b, M, self.W, 10).reshape(b * M, self.W, 10)
        ss = mults[None].expand(b, M, self.W, 10).reshape(b * M, self.W, 10)
        t, c = self.sub_scan(rr, ss)
        t = t.view(b, M, self.W, 10)
        c = c.view(b, M, self.Cb)
        pair = torch.cat([c[:, :-1], c[:, 1:]], dim=-1)
        w = self._sm(self.sel(pair).squeeze(-1))
        self.last_q = w
        return self._tap(torch.einsum("bm,bmwo->bwo", w, t[:, :-1]))

    # ---------------- general two-operand ops, same tables ----------------
    def _prods(self, s, t):
        prod = {}
        for i in range(self.S):
            for j in range(self.S):
                o = torch.einsum("bu,bv,uvo->bo", s[:, i], t[:, j],
                                 self.Tmul_eff)
                prod[(i, j)] = (self._sm(o[:, :10]), self._sm(o[:, 10:]))
        return prod

    def _fold_sum(self, regs):
        acc = regs[0]
        for r in regs[1:]:
            acc = self.add_scan(acc, r)
        return acc

    def _combine(self, prod, z, ndig, mults, variant, b):
        Fw = 2 * self.S
        if variant == "horner":
            r = z[:, None].expand(b, self.W, 10)
            for k in range(self.K - 1, -1, -1):
                r = torch.cat([z[:, None], r[:, : self.W - 1]], dim=1)
                for i in range(self.S):
                    j = k - i
                    if 0 <= j < self.S:
                        lo, hi = prod[(i, j)]
                        slots = [lo[:, None], hi[:, None]] + \
                                [z[:, None]] * (self.W - 2)
                        r = self.add_scan(r, torch.cat(slots, dim=1))
                r = self.reduce(r, ndig, mults)
            return r
        leaves = self._leaves(prod, z, Fw)
        P = self._fold_sum(leaves) if variant == "fold" else self.tree_sum(leaves)
        r = z[:, None].expand(b, self.W, 10)
        for t in range(Fw - 1, -1, -1):
            r = torch.cat([P[:, t:t + 1], r[:, : self.W - 1]], dim=1)
            if variant == "redall" or t <= self.S:
                r = self.reduce(r, ndig, mults)
        return r

    def mulmod(self, s, t, ndig, mults, variant="tree"):
        """(s * t) mod N as a distribution over S digit slots.  The diagonal
        s == t is exactly the task, so every table used here is anchored by the
        task loss."""
        b = s.shape[0]
        z = self._sm(self.zero).expand(b, 10)
        return self._combine(self._prods(s, t), z, ndig, mults, variant, b)[:, :self.S]

    def addmod(self, s, t, ndig, mults):
        """(s + t) mod N, same learned adder and same learned reduction."""
        b = s.shape[0]
        z = self._sm(self.zero).expand(b, 10)
        pad = z[:, None].expand(b, self.W - self.S, 10)
        r = self.add_scan(torch.cat([s, pad], 1), torch.cat([t, pad], 1))
        return self.reduce(r, ndig, mults)[:, :self.S]

    def square(self, s, ndig, mults, variant="tree"):
        return self.mulmod(s, s, ndig, mults, variant)

    def forward(self, s, ndig):
        self.tpos = 0
        mults = self.multiples(ndig) if self.needed else None
        return torch.log(self.square(s, ndig, mults) + 1e-9)


class FreeD(torch.nn.Module):
    """A free learned increment map D(x).  Conservative variant: asserts only
    that SOME additive-increment structure exists, with no statement about its
    form.  Discarded at eval."""

    def __init__(self, slots, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(slots * 10, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, slots * 10))
        self.S = slots

    def forward(self, s):
        return F.softmax(self.net(s.flatten(1)).view(-1, self.S, 10), -1)


@torch.no_grad()
def scores(model, ref, ndigits):
    out, tot, ok = {}, 0, 0
    for name in ("Tmul_eff", "Tadd_eff", "Tsub_eff"):
        g, w = getattr(model, name), getattr(ref, name)
        a = (g.argmax(-1) == w.argmax(-1)).float()
        ok += a.sum().item()
        tot += a.numel()
    out["cell_agree"] = round(ok / tot, 3)
    am = model.Tmul_eff[..., :10].argmax(-1)
    groups = {}
    for a in range(10):
        for b in range(10):
            groups.setdefault((a * b) % 10, []).append(int(am[a, b]))
    k = sum(v.count(max(set(v), key=v.count)) for v in groups.values())
    out["mul_fn"] = round(k / 100, 3)
    pi = {c: max(set(v), key=v.count) for c, v in groups.items()}
    out["mul_gauge"] = round(len(set(pi.values())) / len(pi), 3)
    for nm, T, cols in (("add_shift", model.Tadd_eff, range(10)),
                        ("sub_shift", model.Tsub_eff, sorted(set(ndigits)))):
        k = n = 0
        for v in cols:
            for c in range(T.shape[2]):
                a = T[:, v, c, :10].argmax(-1)
                k += max(sum(1 for u in range(10) if int(a[u]) == (u + s) % 10)
                         for s in range(10))
                n += 10
        out[nm] = round(k / n, 3)
    return out


def sym_ce(p, q):
    """Symmetric cross-entropy between two digit-slot distributions.  Its
    minimum is at p = q = one-hot, so it prices agreement AND sharpness."""
    return (-(q.detach() * p.clamp_min(1e-9).log()).sum(-1).mean()
            - (p.detach() * q.clamp_min(1e-9).log()).sum(-1).mean())


def sym_kl(p, q):
    """Symmetric KL.  Zero whenever p = q at ANY entropy, so it prices
    agreement ONLY -- the control for "was it the sharpening pressure rather
    than the relational content that did the work?"."""
    lp, lq = p.clamp_min(1e-9).log(), q.clamp_min(1e-9).log()
    return (((p - q.detach()) * (lp - lq.detach())).sum(-1).mean()
            + ((q - p.detach()) * (lq - lp.detach())).sum(-1).mean())


@torch.no_grad()
def out_diversity(model, inp, nd, chunk=512, variant="tree"):
    """Collapse detector: fraction of DISTINCT predicted answers over a batch.
    A map collapsed to a constant reads ~1/n; the truth reads 1.0."""
    model.eval()
    was, model.hard = model.hard, True
    mults = model.multiples(nd) if model.needed else None
    a = model.square(inp[:chunk], nd, mults, variant).argmax(-1)
    model.hard = was
    model.train()
    return round(len({tuple(r.tolist()) for r in a}) / a.shape[0], 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulus", type=int, default=10403)
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--train-x", type=int, default=8000)
    ap.add_argument("--held-x", type=int, default=1024)
    ap.add_argument("--max-quot", type=int, default=10)
    ap.add_argument("--tie", action="store_true")
    ap.add_argument("--tie-sub", action="store_true")
    # --- family 1: relational increment law ---
    ap.add_argument("--rel", default="none",
                    choices=["none", "free", "affine", "true"])
    ap.add_argument("--rel-w", type=float, default=1.0)
    ap.add_argument("--rel-batch", type=int, default=128)
    ap.add_argument("--rel-nondeg", type=float, default=0.0,
                    help="penalty on inc being the learned additive identity")
    ap.add_argument("--rel-hidden", type=int, default=256)
    # --- family 2: dual-path agreement ---
    ap.add_argument("--dual", default="none",
                    choices=["none", "fold", "redall", "horner"])
    ap.add_argument("--dual-w", type=float, default=1.0)
    ap.add_argument("--dual-batch", type=int, default=0,
                    help="0 = same batch as the label loss")
    ap.add_argument("--dual-label", type=float, default=1.0,
                    help="weight of the label CE on path B")
    # --- family 3: algebraic re-screen ---
    ap.add_argument("--sym", type=float, default=0.0,
                    help="soft commutativity penalty on Tmul/Tadd")
    ap.add_argument("--inv", type=float, default=0.0,
                    help="soft Tsub o Tadd = id penalty")
    ap.add_argument("--assoc", type=float, default=0.0,
                    help="register-level associativity of the learned adder")
    ap.add_argument("--label-w", type=float, default=1.0,
                    help="weight on the task label CE.  0 = ALGEBRA ONLY: is "
                         "the adder identifiable from generic algebraic laws "
                         "alone, with no labels at all?  The chain is skipped, "
                         "so it is ~50x cheaper per step.")
    ap.add_argument("--cancel", type=float, default=0.0,
                    help="cancellativity: B -> A (+) B is injective.  A generic "
                         "algebraic non-degeneracy law.  It excludes the "
                         "CONSTANT adder, which is otherwise associative and "
                         "commutative and is the observed --assoc collapse.")
    ap.add_argument("--hard", action="store_true",
                    help="straight-through on every inter-step state.  The "
                         "basin diagnostic is measured with HARD states, so "
                         "this is the direct test of 'the discrete landscape "
                         "is graded -- can SGD see it through a ST estimator?'")
    ap.add_argument("--div", default="ce", choices=["ce", "kl"],
                    help="agreement divergence for --dual/--rel/--assoc. "
                         "ce prices agreement AND sharpness; kl prices "
                         "agreement only (the control)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--construct", action="store_true")
    ap.add_argument("--check-law", action="store_true",
                    help="correctness gate: does the CONSTRUCTED model satisfy "
                         "the relational law and the dual-path agreement?")
    ap.add_argument("--basin", action="store_true",
                    help="THE DECISIVE CHEAP MEASUREMENT.  Corrupt k cells of "
                         "the construction and ask how far from the solution "
                         "each objective can still see.  discrete-search "
                         "measured the LABEL objective as informative only "
                         "inside a Hamming ball of ~10-20 of 1,007 cells; if "
                         "the relational / dual-path objectives are informative "
                         "further out, the family is alive even where SGD "
                         "fails.  DIAGNOSTIC (uses the construction).")
    ap.add_argument("--basin-reps", type=int, default=4)
    ap.add_argument("--basin-module", default="all",
                    choices=["all", "Tmul", "Tadd", "Tsub"],
                    help="restrict corruption to one table, so the label "
                         "objective and the algebraic objectives are compared "
                         "at the SAME number of corrupted cells of the SAME "
                         "table")
    ap.add_argument("--basin-n", type=int, default=256)
    ap.add_argument("--jsonl", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    N, S = args.modulus, args.slots
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

    def mk():
        return RelALU(S, 2, 2, 11, 1.0, 0.0, args.hard, "quotient",
                      args.max_quot, "tree", "serial",
                      tie=args.tie, tie_sub=args.tie_sub).to(dev)

    model = mk()
    ref = mk()
    ref.hard = False
    ref.construct()
    for p in ref.parameters():
        p.requires_grad_(False)

    BIG = 30.0

    def const_reg(val):
        """One-hot register for an integer -- DIAGNOSTIC use only."""
        t = torch.full((S, 10), -BIG)
        for i, d in enumerate(digits_le(val, S)):
            t[i, d] = BIG
        return t.to(dev)

    # learned relational constants (random init unless --rel true)
    rel_c = torch.nn.Parameter(torch.randn(S, 10, device=dev) * 0.5)
    rel_a = torch.nn.Parameter(torch.randn(S, 10, device=dev) * 0.5)
    rel_b = torch.nn.Parameter(torch.randn(S, 10, device=dev) * 0.5)
    freeD = FreeD(S, args.rel_hidden).to(dev) if args.rel == "free" else None
    if args.rel == "true":                      # DIAGNOSTIC: the true identity
        with torch.no_grad():
            rel_c.copy_(const_reg(1))
            rel_a.copy_(const_reg(2))
            rel_b.copy_(const_reg(1))
        rel_c.requires_grad_(False)
        rel_a.requires_grad_(False)
        rel_b.requires_grad_(False)

    free = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] N={N} S={S} tree:quotient tie={args.tie} "
          f"tie_sub={args.tie_sub} rel={args.rel} dual={args.dual} "
          f"params={free:,} train={len(tr_x)} held={len(he_x)} "
          f"units={len(units)}", flush=True)

    @torch.no_grad()
    def ev(m, inp, tgt, discrete=False, chunk=512, variant="tree"):
        m.eval()
        was, m.hard = m.hard, True if discrete else m.hard
        ok = 0
        for i in range(0, inp.shape[0], chunk):
            s = inp[i:i + chunk]
            mults = m.multiples(nd) if m.needed else None
            lg = m.square(s, nd, mults, variant)
            ok += (lg.argmax(-1) == tgt[i:i + chunk]).all(-1).sum().item()
        m.hard = was
        m.train()
        return ok / inp.shape[0]

    if args.basin:
        # ---- DIAGNOSTIC: how far from the solution can each objective see? ----
        BIGC = 30.0
        allspecs = [("Tmul", (slice(None), slice(None), slice(0, 10))),
                 ("Tmul", (slice(None), slice(None), slice(10, 20))),
                 ("Tadd", (slice(None), slice(None), slice(None), slice(0, 10))),
                 ("Tadd", (slice(None), slice(None), slice(None), slice(10, None))),
                 ("Tsub", (slice(None), slice(None), slice(None), slice(0, 10))),
                 ("Tsub", (slice(None), slice(None), slice(None), slice(10, None)))]
        specs = [x for x in allspecs
                 if args.basin_module in ("all", x[0])]
        n_cells = {"all": 1000, "Tmul": 200, "Tadd": 400, "Tsub": 400}[
            args.basin_module]

        @torch.no_grad()
        def corrupt(m, k, gen):
            """Flip the argmax of k randomly chosen discrete cells."""
            cells = []
            for name, sl in specs:
                T = getattr(m, name)
                sub = T[sl]
                n_out = sub.shape[-1]
                flat = int(torch.tensor(sub.shape[:-1]).prod())
                cells += [(name, sl, i, n_out) for i in range(flat)]
            pick = torch.randperm(len(cells), generator=gen)[:k].tolist()
            for p in pick:
                name, sl, i, n_out = cells[p]
                T = getattr(m, name)
                sub = T[sl]
                v = sub.reshape(-1, n_out)
                new = int(torch.randint(0, n_out, (1,), generator=gen))
                while n_out > 1 and new == int(v[i].argmax()):
                    new = int(torch.randint(0, n_out, (1,), generator=gen))
                v[i] = -BIGC
                v[i, new] = BIGC
                T[sl] = sub

        n = args.basin_n
        s = xin[:n]
        tgt = xt[:n]
        cc = F.softmax(const_reg(1), -1)[None].expand(n, S, 10)
        aa = F.softmax(const_reg(2), -1)[None].expand(n, S, 10)
        ks = [k for k in (0, 1, 2, 5, 10, 20, 50, 100, 200, 400, 1000)
              if k <= n_cells]
        print(f"[{args.tag}] BASIN (DIAGNOSTIC) n={n} reps={args.basin_reps} "
              f"module={args.basin_module} cells={n_cells} -- read the "
              f"SHAPE and the spread", flush=True)
        rows = []
        for k in ks:
            acc = {key: [] for key in
                   ("dig", "ce", "exact", "fold", "red", "horn", "rel", "asc")}
            for rep in range(args.basin_reps):
                gen = torch.Generator().manual_seed(1000 * k + rep)
                model.construct()
                model.hard = True
                if k:
                    corrupt(model, k, gen)
                with torch.no_grad():
                    mults = model.multiples(nd)
                    r = model.square(s, nd, mults)
                    acc["dig"].append((r.argmax(-1) == tgt).float().mean().item())
                    acc["exact"].append(
                        (r.argmax(-1) == tgt).all(-1).float().mean().item())
                    acc["ce"].append(F.cross_entropy(
                        torch.log(r + 1e-9).reshape(-1, 10),
                        tgt.reshape(-1)).item())
                    for nm, var in (("fold", "fold"), ("red", "redall"),
                                    ("horn", "horner")):
                        rb = model.square(s, nd, mults, var)
                        acc[nm].append(sym_kl(r, rb).item())
                    inc = model.addmod(s, cc, nd, mults)
                    lhs = model.square(inc, nd, mults)
                    ax = model.mulmod(aa, s, nd, mults)
                    rhs = model.addmod(model.addmod(r, ax, nd, mults), cc,
                                       nd, mults)
                    acc["rel"].append(sym_kl(lhs, rhs).item())
                    g2 = torch.Generator(device=dev).manual_seed(7)
                    rr = torch.randint(0, 10, (3, 256, W), device=dev,
                                       generator=g2)
                    A, B, C = [F.one_hot(rr[i], 10).float() for i in range(3)]
                    l1 = model.add_scan(model.add_scan(A, B), C)
                    l2 = model.add_scan(A, model.add_scan(B, C))
                    acc["asc"].append(sym_kl(l1, l2).item())
            row = {"k": k}
            for key, v in acc.items():
                t = torch.tensor(v)
                row[key] = round(t.mean().item(), 4)
                row[key + "_sd"] = round(t.std().item() if len(v) > 1 else 0.0, 4)
            rows.append(row)
            print(f"[{args.tag}] k={k:>5} dig={row['dig']:.4f}+-{row['dig_sd']:.4f} "
                  f"exact={row['exact']:.4f} ce={row['ce']:.3f} | "
                  f"fold={row['fold']:.4f}+-{row['fold_sd']:.4f} "
                  f"red={row['red']:.4f}+-{row['red_sd']:.4f} "
                  f"horn={row['horn']:.4f}+-{row['horn_sd']:.4f} "
                  f"rel={row['rel']:.4f}+-{row['rel_sd']:.4f} "
                  f"asc={row['asc']:.4f}+-{row['asc_sd']:.4f}", flush=True)
        if args.jsonl:
            with open(args.jsonl, "a") as fh:
                fh.write(json.dumps({"tag": args.tag, "argv": sys.argv[1:],
                                     "basin": rows}) + "\n")
        return 0

    if args.construct or args.check_law:
        model.construct()
        with torch.no_grad():
            mults = model.multiples(nd)
            s = xin[:512]
            print(f"[{args.tag}] CONSTRUCTED tree soft={ev(model, xin[:1024], xt[:1024]):.3f} "
                  f"hard={ev(model, xin[:1024], xt[:1024], True):.3f} "
                  f"held_hard={ev(model, hin, ht, True):.3f}", flush=True)
            for v in ("fold", "redall", "horner"):
                print(f"[{args.tag}] CONSTRUCTED {v:>6} soft="
                      f"{ev(model, xin[:512], xt[:512], False, 512, v):.3f} "
                      f"hard={ev(model, xin[:512], xt[:512], True, 512, v):.3f}",
                      flush=True)
            if args.check_law:
                fx = model.square(s, nd, mults)
                inc = model.addmod(s, F.softmax(const_reg(1), -1)[None].expand(
                    s.shape[0], S, 10), nd, mults)
                lhs = model.square(inc, nd, mults)
                ax = model.mulmod(
                    F.softmax(const_reg(2), -1)[None].expand(s.shape[0], S, 10),
                    s, nd, mults)
                rhs = model.addmod(model.addmod(fx, ax, nd, mults),
                                   F.softmax(const_reg(1), -1)[None].expand(
                                       s.shape[0], S, 10), nd, mults)
                agree = (lhs.argmax(-1) == rhs.argmax(-1)).all(-1).float().mean()
                # and does inc actually increment?  (check against the truth)
                print(f"[{args.tag}] LAW CHECK (constructed, true constants): "
                      f"exact agreement lhs==rhs on {agree.item():.3f} of 512 "
                      f"operands; sym_ce={sym_ce(lhs, rhs).item():.4f}",
                      flush=True)
                a_tree = model.square(s, nd, mults, "tree")
                for v in ("fold", "redall", "horner"):
                    b_v = model.square(s, nd, mults, v)
                    ag = (a_tree.argmax(-1) == b_v.argmax(-1)).all(-1).float().mean()
                    print(f"[{args.tag}] DUAL CHECK tree vs {v:>6}: "
                          f"agree={ag.item():.3f} sym_ce={sym_ce(a_tree, b_v).item():.4f}",
                          flush=True)
        return 0

    params = list(model.parameters())
    if args.rel in ("free", "affine"):
        params += [rel_c]
        if args.rel == "affine":
            params += [rel_a, rel_b]
        if freeD is not None:
            params += list(freeD.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95))
    t0 = time.time()
    nd_dig = digits_le(N, W)
    agree = sym_ce if args.div == "ce" else sym_kl

    @torch.no_grad()
    def local_ce(n=512):
        """The coordinator's control variable: how well does each op fit its
        own local task given true inputs?  The cliff sits between 0.0050 and
        0.0073.  Far more sensitive than any exact-match metric, so it is
        tracked during training, not only at the end."""
        was = model.hard
        ref.mode, ref.tape = 'record', []
        ref(xin[:n], nd)
        ref.mode = None
        model.mode, model.tape = 'force', ref.tape
        model.tf_loss, model.tf_n = torch.zeros((), device=dev), 0
        model(xin[:n], nd)
        model.mode, model.hard = None, was
        return round((model.tf_loss / max(model.tf_n, 1)).item(), 4)

    # index maps for the soft algebraic penalties
    idx = torch.arange(10, device=dev)

    def sym_pen():
        p = ((model.Tmul_eff - model.Tmul_eff.transpose(0, 1)) ** 2).mean()
        p = p + ((model.Tadd_eff - model.Tadd_eff.transpose(0, 1)) ** 2).mean()
        return p

    def inv_pen():
        """Tsub(add_digit(u,v,c), v, c) should return u, with borrow = carry."""
        a = F.softmax(model.Tadd_eff, -1)                    # (u,v,c,10+Ca)
        w = a[..., :10]                                      # (u,v,c,w)
        s = F.softmax(model.Tsub_eff, -1)                    # (w,v,c,10+Cb)
        # expected sub output given the add's output digit distribution
        got = torch.einsum("uvcw,wvco->uvco", w, s)
        tgt_d = F.one_hot(idx, 10).float()[:, None, None, :].expand(
            10, 10, model.Ca, 10)
        return -(tgt_d * got[..., :10].clamp_min(1e-9).log()).sum(-1).mean()

    def assoc_pen(bs):
        """(A + B) + C == A + (B + C) at self-generated register values."""
        r = torch.randint(0, 10, (3, bs, W), device=dev)
        A, B, C = [F.one_hot(r[i], 10).float() for i in range(3)]
        l = model.add_scan(model.add_scan(A, B), C)
        rr = model.add_scan(A, model.add_scan(B, C))
        return agree(l, rr)

    def cancel_pen(bs):
        """A (+) B must differ from A (+) C whenever B differs from C.  A
        generic algebraic law (injectivity of translation); it supplies no
        values and excludes the constant adder."""
        r = torch.randint(0, 10, (3, bs, W), device=dev)
        A, B, C = [F.one_hot(r[i], 10).float() for i in range(3)]
        same = (r[1] == r[2]).all(-1).float()[:, None, None]
        p, q = model.add_scan(A, B), model.add_scan(A, C)
        ov = (p * q).sum(-1)                       # per-slot overlap
        return ((1 - same.squeeze(-1)) * ov).mean()

    hist = []
    for step in range(1, args.steps + 1):
        i0 = torch.randint(0, xin.shape[0], (args.batch,), device=dev)
        bi, bt = xin[i0], xt[i0]
        mults = model.multiples(nd) if model.needed else None
        parts = {}
        if args.label_w > 0:
            r = model.square(bi, nd, mults)
            loss = args.label_w * F.cross_entropy(
                torch.log(r + 1e-9).reshape(-1, 10), bt.reshape(-1))
        else:
            r = None
            loss = torch.zeros((), device=dev)

        if (args.dual != "none" or args.rel != "none") and r is None:
            r = model.square(bi, nd, mults)

        if args.dual != "none":
            nb = args.dual_batch or args.batch
            sub, subt = bi[:nb], bt[:nb]
            rb = model.square(sub, nd, mults, args.dual)
            d = agree(r[:nb], rb)
            parts["dual"] = d.item()
            loss = loss + args.dual_w * d
            if args.dual_label > 0:
                loss = loss + args.dual_label * F.cross_entropy(
                    torch.log(rb + 1e-9).reshape(-1, 10), subt.reshape(-1))

        if args.rel != "none":
            nb = min(args.rel_batch, args.batch)
            sub = bi[:nb]
            cc = F.softmax(rel_c, -1)[None].expand(nb, S, 10)
            inc = model.addmod(sub, cc, nd, mults)
            lhs = model.square(inc, nd, mults)
            fx = r[:nb]
            if args.rel == "free":
                D = freeD(sub)
            else:
                aa = F.softmax(rel_a, -1)[None].expand(nb, S, 10)
                bb = F.softmax(rel_b, -1)[None].expand(nb, S, 10)
                D = model.addmod(model.mulmod(aa, sub, nd, mults), bb, nd, mults)
            rhs = model.addmod(fx, D, nd, mults)
            rl = agree(lhs, rhs)
            parts["rel"] = rl.item()
            loss = loss + args.rel_w * rl
            if args.rel_nondeg > 0:
                zc = F.softmax(model.zero, -1)[None].expand(S, 10)
                ov = (F.softmax(rel_c, -1) * zc).sum(-1).mean()
                loss = loss + args.rel_nondeg * ov
                parts["nondeg"] = ov.item()

        if args.sym > 0:
            p = sym_pen()
            parts["sym"] = p.item()
            loss = loss + args.sym * p
        if args.inv > 0:
            p = inv_pen()
            parts["inv"] = p.item()
            loss = loss + args.inv * p
        if args.assoc > 0:
            p = assoc_pen(256)
            parts["assoc"] = p.item()
            loss = loss + args.assoc * p
        if args.cancel > 0:
            p = cancel_pen(256)
            parts["cancel"] = p.item()
            loss = loss + args.cancel * p

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            sc = scores(model, ref, nd_dig)
            te = ev(model, xin[:1024], xt[:1024])
            th = ev(model, xin[:1024], xt[:1024], True)
            hh = ev(model, hin, ht, True)
            hist.append((step, round(th, 4)))
            sc["out_div"] = out_diversity(model, xin, nd)
            sc["local_ce"] = local_ce()
            print(f"[{args.tag}] step={step:>5} loss={loss.item():.4f} "
                  f"{ {k: round(v, 4) for k, v in parts.items()} } "
                  f"train_exact={te:.3f} train_exact_hard={th:.3f} "
                  f"held_exact_hard={hh:.3f} {sc} ({time.time()-t0:.0f}s)",
                  flush=True)

    trh = ev(model, xin[:2048], xt[:2048], True)
    heh = ev(model, hin, ht, True)
    sc = scores(model, ref, nd_dig)
    sc["out_div"] = out_diversity(model, xin, nd)
    if args.dual != "none":
        sc["pathB_hard"] = round(
            ev(model, xin[:1024], xt[:1024], True, 512, args.dual), 4)
        sc["pathB_div"] = out_diversity(model, xin, nd, 512, args.dual)
    sc["local_ce"] = local_ce()
    tre = ev(model, xin[:2048], xt[:2048])
    hee = ev(model, hin, ht)
    print(f"[{args.tag}] FINAL train_exact={tre:.3f} train_exact_hard={trh:.3f} "
          f"held_exact={hee:.3f} held_exact_hard={heh:.3f} {sc} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if args.jsonl:
        with open(args.jsonl, "a") as fh:
            fh.write(json.dumps({"tag": args.tag, "argv": sys.argv[1:],
                                 "train_exact": tre, "held_exact": hee,
                                 "train_exact_hard": trh,
                                 "held_exact_hard": heh, **sc,
                                 "hist": hist,
                                 "secs": round(time.time() - t0, 1)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
