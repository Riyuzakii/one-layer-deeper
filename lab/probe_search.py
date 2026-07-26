#!/usr/bin/env python
"""LAB ONLY -- direct DISCRETE local search over `DigitALU`'s table cells.

`alu-credit` §10.2 named this the decisive diagnostic for the whole family:

    "The basin measurement says the neighbourhood of the solution is
     informative -- 10 of 500 cells wrong still scores train_exact 0.555 -- so a
     *discrete* local search has signal where the gradient does not.  It answers
     a question nobody has asked: is the discrete landscape itself benign, and
     only the gradient estimator bad?"

The graph is `alu-depth`'s `tree:quotient` (39 sequential soft steps, 6,820
parameters, constructed ceiling 1.000 soft AND hard).  With every inter-step
state snapped to its argmax -- which is the *only* honest evaluation of this
family, per `alu-depth` §2.3 -- the whole forward pass is a pure integer
transducer: every softmax is an argmax, every einsum against a one-hot is a
table lookup.  So this file re-implements that same graph in *integers*, which

  * is exactly equivalent to `probe_alu_depth.py` with `hard=True` (verified by
    `--verify`, which compares against the float model cell for cell), and
  * is ~1000x cheaper, and vectorises over a POPULATION of candidate table
    assignments on a leading dimension -- which is what makes a 1,007-cell x
    up-to-10-candidate coordinate sweep cost seconds instead of hours.

Search space (the argmax of every learned table = the discrete transducer):

    mul_lo  (10,10)      10 candidates    100 cells
    mul_hi  (10,10)      10 candidates    100 cells
    add_d   (10,10,2)    10 candidates    200 cells
    add_c   (10,10,2)     2 candidates    200 cells
    sub_d   (10,10,2)    10 candidates    200 cells
    sub_b   (10,10,2)     2 candidates    200 cells
    zero, carry0, borrow0                   3 cells
    sel     (2,2) score table              4 cells
    ------------------------------------------------
                                        1,007 cells / 6,808 candidate values

`sel` is the float model's 5-parameter quotient scorer written in its exact
canonical form: with one-hot borrow states the linear layer is a function of the
pair (borrow_m, borrow_{m+1}) only, so `score[a][b] = W[0,a] + W[0,Cb+b] + bias`
is an exact reparameterisation.  Ties are broken toward the smallest m.

MODES
  --verify        integer sim == float sim with hard states (correctness gate)
  --basin         corrupt k cells of the construction, measure what survives
  --repair K      corrupt K cells, then run the search: does it find its way
                  back?  This measures the RADIUS of the discrete basin.
  (default)       search from a uniformly random table assignment

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def digits_le(value: int, slots: int) -> list[int]:
    out = []
    for _ in range(slots):
        out.append(value % 10)
        value //= 10
    return out


# --------------------------------------------------------------- gather helpers
def _g2(tab, i, j):
    """tab (P,A,B) -> tab[p, i[p,n], j[p,n]]  ->  (P,N)"""
    P, A, B = tab.shape
    return tab.reshape(P, A * B).gather(1, i * B + j)


def _g3(tab, i, j, k):
    """tab (P,A,B,C) -> tab[p, i[p,n], j[p,n], k[p,n]]  ->  (P,N)"""
    P, A, B, C = tab.shape
    return tab.reshape(P, A * B * C).gather(1, (i * B + j) * C + k)


# ------------------------------------------------------------------- the model
CELLS = (                      # (name, shape, n_candidates)
    ("mul_lo", (10, 10), 10),
    ("mul_hi", (10, 10), 10),
    ("add_d", (10, 10, 2), 10),
    ("add_c", (10, 10, 2), 2),
    ("sub_d", (10, 10, 2), 10),
    ("sub_b", (10, 10, 2), 2),
    ("zero", (1,), 10),
    ("carry0", (1,), 2),
    ("borrow0", (1,), 2),
    ("sel", (2, 2), 2),
)


class IntALU:
    """`tree:quotient` DigitALU as an integer transducer, batched over a
    population of candidate table assignments."""

    def __init__(self, P, slots, ndig, n_carry=2, n_borrow=2, max_quot=10,
                 device="cuda:0"):
        self.P, self.S = P, slots
        self.W = slots + 1
        self.F = 2 * slots
        self.Ca, self.Cb, self.Q = n_carry, n_borrow, max_quot
        self.needed = list(range(max_quot + 2))
        self.M = len(self.needed)
        self.dev = torch.device(device)
        self.ndig = ndig.to(self.dev)                      # (W,) int
        self.t = {}
        for name, shape, _ in CELLS:
            dt = torch.float32 if name == "sel" else torch.int64
            self.t[name] = torch.zeros((P,) + shape, dtype=dt, device=self.dev)
        self._eps = torch.arange(self.M - 1, device=self.dev,
                                 dtype=torch.float32) * 1e-6
        self.ties = set()

    # ---- LEGAL hypothesis-class ties (weight sharing, no value supplied) ----
    def set_ties(self, ties):
        """`sym`: Tmul and Tadd are symmetric in their two digit indices
        (commutativity is a structural property, not a value).
        `inv`: Tsub is DERIVED from Tadd -- sub_d[add_d[u,v,c],v,c] = u and
        sub_b[...] = add_c[u,v,c].  This is weight tying: it supplies no
        arithmetic, it says the subtract table is the add table read backwards.
        Both keep the construction inside the class (verified by --construct)."""
        self.ties = set(t for t in ties if t)
        idx = torch.zeros(100, dtype=torch.long)
        for u in range(10):
            for v in range(10):
                idx[u * 10 + v] = min(u, v) * 10 + max(u, v)
        self.sym100 = idx.to(self.dev)
        self.sym200 = (idx[:, None] * 2
                       + torch.arange(2)[None]).reshape(-1).to(self.dev)

    def canon(self):
        if not self.ties:
            return
        P = self.P
        if "sym" in self.ties:
            for nm, ix in (("mul_lo", self.sym100), ("mul_hi", self.sym100),
                           ("add_d", self.sym200), ("add_c", self.sym200)):
                f = self.t[nm].reshape(P, -1)
                f.copy_(f.gather(1, ix[None].expand(P, -1)))
        if "inv" in self.ties:
            ad, ac = self.t["add_d"], self.t["add_c"]
            u = torch.arange(10, device=self.dev).view(1, 10, 1, 1).expand_as(ad)
            self.t["sub_d"].zero_().scatter_(1, ad, u)
            self.t["sub_b"].zero_().scatter_(1, ad, ac)

    # ---- population plumbing -------------------------------------------------
    def flat(self, name):
        return self.t[name].reshape(self.P, -1)

    def snapshot(self, member=0):
        return {k: v[member].clone() for k, v in self.t.items()}

    def load(self, snap):
        for k, v in snap.items():
            self.t[k].copy_(v.unsqueeze(0).expand_as(self.t[k]))

    def randomize(self, gen):
        for name, shape, n in CELLS:
            if name == "sel":
                v = torch.randint(0, n, (self.P,) + shape, generator=gen).float()
            else:
                v = torch.randint(0, n, (self.P,) + shape, generator=gen)
            self.t[name].copy_(v.to(self.dev))

    def construct(self):
        """LAB DIAGNOSTIC -- the exact solution, as integers."""
        lo = torch.zeros(10, 10, dtype=torch.int64)
        hi = torch.zeros(10, 10, dtype=torch.int64)
        for a in range(10):
            for b in range(10):
                lo[a, b] = (a * b) % 10
                hi[a, b] = (a * b) // 10
        ad = torch.zeros(10, 10, 2, dtype=torch.int64)
        ac = torch.zeros(10, 10, 2, dtype=torch.int64)
        sd = torch.zeros(10, 10, 2, dtype=torch.int64)
        sb = torch.zeros(10, 10, 2, dtype=torch.int64)
        for u in range(10):
            for v in range(10):
                for c in range(2):
                    s = u + v + c
                    ad[u, v, c], ac[u, v, c] = s % 10, s // 10
                    d = u - v - c
                    sd[u, v, c], sb[u, v, c] = d % 10, (1 if d < 0 else 0)
        sel = torch.zeros(2, 2)
        sel[0, 1] = 1.0
        snap = {"mul_lo": lo, "mul_hi": hi, "add_d": ad, "add_c": ac,
                "sub_d": sd, "sub_b": sb,
                "zero": torch.zeros(1, dtype=torch.int64),
                "carry0": torch.zeros(1, dtype=torch.int64),
                "borrow0": torch.zeros(1, dtype=torch.int64), "sel": sel}
        self.load({k: v.to(self.dev) for k, v in snap.items()})

    # ---- cell-exercise counting (identifiability diagnostic) ----------------
    def count_on(self):
        self.counts = {name: torch.zeros(
            int(torch.tensor(shape).prod()), dtype=torch.long, device=self.dev)
            for name, shape, _ in CELLS}

    def _rec(self, name, idx):
        c = getattr(self, "counts", None)
        if c is not None:
            c[name].scatter_add_(0, idx.reshape(-1),
                                 torch.ones_like(idx.reshape(-1)))

    def G2(self, name, i, j):
        tab = self.t[name]
        P, A, B = tab.shape
        idx = i * B + j
        self._rec(name, idx)
        return tab.reshape(P, A * B).gather(1, idx)

    def G3(self, name, i, j, k):
        tab = self.t[name]
        P, A, B, C = tab.shape
        idx = (i * B + j) * C + k
        self._rec(name, idx)
        return tab.reshape(P, A * B * C).gather(1, idx)

    # ---- scans ---------------------------------------------------------------
    def add_scan(self, r, a):
        """r,a (P,N,W) -> (P,N,W).  Mirrors DigitALU.add_scan with hard states."""
        P, N, W = r.shape
        c = self.t["carry0"][:, 0:1].expand(P, N)
        outs = []
        for m in range(W):
            u, v = r[:, :, m], a[:, :, m]
            outs.append(self.G3("add_d", u, v, c))
            c = self.G3("add_c", u, v, c)
        return torch.stack(outs, -1)

    def sub_scan(self, r, a):
        P, N, W = r.shape
        c = self.t["borrow0"][:, 0:1].expand(P, N)
        outs = []
        for m in range(W):
            u, v = r[:, :, m], a[:, :, m]
            outs.append(self.G3("sub_d", u, v, c))
            c = self.G3("sub_b", u, v, c)
        return torch.stack(outs, -1), c

    def tree_sum(self, regs):
        while len(regs) > 1:
            carry = [regs[-1]] if len(regs) % 2 else []
            pairs = [(regs[i], regs[i + 1]) for i in range(0, len(regs) - 1, 2)]
            n = regs[0].shape[1]
            out = self.add_scan(torch.cat([a for a, _ in pairs], 1),
                                torch.cat([b for _, b in pairs], 1))
            regs = [out[:, i * n:(i + 1) * n] for i in range(len(pairs))] + carry
        return regs[0]

    def multiples(self):
        """{m*N} for m in `needed`, built from digits(N) with the same add_scan."""
        P, W = self.P, self.W
        z = self.t["zero"][:, 0:1, None].expand(P, 1, W)
        have = {0: z.contiguous(),
                1: self.ndig.view(1, 1, W).expand(P, 1, W).contiguous()}
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
        return torch.cat([have[m] for m in self.needed], 1)          # (P,M,W)

    def quot_reduce(self, r, mults):
        P, B, W = r.shape
        M = mults.shape[1]
        rr = r[:, :, None].expand(P, B, M, W).reshape(P, B * M, W)
        ss = mults[:, None].expand(P, B, M, W).reshape(P, B * M, W)
        t, c = self.sub_scan(rr, ss)
        t = t.view(P, B, M, W)
        c = c.view(P, B, M)
        a = c[:, :, :-1].reshape(P, B * (M - 1))
        b = c[:, :, 1:].reshape(P, B * (M - 1))
        score = self.G2("sel", a, b).view(P, B, M - 1) - self._eps
        w = score.argmax(-1)
        return t.gather(2, w[:, :, None, None].expand(P, B, 1, W)).squeeze(2)

    def _leaves(self, prod, z):
        buckets = {}
        for (i, j), lh in prod.items():
            buckets.setdefault(i + j, []).append(lh)
        leaves = []
        for par in (0, 1):
            offs = [k for k in sorted(buckets) if k % 2 == par]
            if not offs:
                continue
            for t in range(max(len(buckets[k]) for k in offs)):
                cols = [z] * self.F
                for k in offs:
                    if t < len(buckets[k]):
                        cols[k], cols[k + 1] = buckets[k][t]
                leaves.append(torch.stack(cols, 2))
        return leaves

    @torch.no_grad()
    def forward(self, s):
        """s (B,S) int digits of x, LSB first -> (P,B,S) predicted digits."""
        self.canon()
        P, B, S, W = self.P, s.shape[0], self.S, self.W
        z = self.t["zero"][:, 0:1].expand(P, B)
        prod = {}
        for i in range(S):
            si = s[:, i].view(1, B).expand(P, B)
            for j in range(S):
                sj = s[:, j].view(1, B).expand(P, B)
                prod[(i, j)] = (self.G2("mul_lo", si, sj),
                                self.G2("mul_hi", si, sj))
        mults = self.multiples()
        Pr = self.tree_sum(self._leaves(prod, z))                    # (P,B,F)
        r = z[:, :, None].expand(P, B, W)
        for t in range(self.F - 1, -1, -1):
            r = torch.cat([Pr[:, :, t:t + 1], r[:, :, :W - 1]], 2)
            if t <= S:
                r = self.quot_reduce(r, mults)
        return r[:, :, :S]

    @torch.no_grad()
    def score(self, s, tgt, chunk=0):
        """-> (digit_acc (P,), exact_acc (P,))"""
        B = s.shape[0]
        chunk = chunk or B
        dg = torch.zeros(self.P, device=self.dev)
        ex = torch.zeros(self.P, device=self.dev)
        for i in range(0, B, chunk):
            p = self.forward(s[i:i + chunk])
            hit = (p == tgt[i:i + chunk].unsqueeze(0))
            dg += hit.float().sum((1, 2))
            ex += hit.all(-1).float().sum(1)
        return dg / (B * self.S), ex / B


# ----------------------------------------------------------- structure scores
def structure_scores(snap, ndigits):
    """Gauge-invariant parameter-level scores (alu-credit §3), on int tables.
    Random baseline 0.23-0.28; the construction reads 1.000."""
    out = {}
    for name, fn in (("mul_lo", lambda a, b: (a * b) % 10),
                     ("mul_hi", lambda a, b: (a * b) // 10)):
        am = snap[name].cpu()
        groups = {}
        for a in range(10):
            for b in range(10):
                groups.setdefault(fn(a, b), []).append(int(am[a, b]))
        ok = tot = 0
        for vs in groups.values():
            ok += vs.count(max(set(vs), key=vs.count))
            tot += len(vs)
        out[name] = round(ok / tot, 3)
    for name, key in (("add", "add_d"), ("sub", "sub_d")):
        T = snap[key].cpu()
        cols = range(10) if name == "add" else sorted(set(ndigits))
        ok = tot = 0
        for v in cols:
            for c in range(T.shape[2]):
                am = T[:, v, c]
                ok += max(sum(1 for u in range(10)
                              if int(am[u]) == (u + sh) % 10) for sh in range(10))
                tot += 10
        out[name + "_shift"] = round(ok / tot, 3)
    return out


def cell_agreement(snap, truth):
    """Fraction of cells equal to the constructed value (NOT gauge invariant --
    reported only for the repair experiment, where the gauge is fixed by the
    uncorrupted cells)."""
    ok = tot = 0
    for name, _, _ in CELLS:
        a, b = snap[name].reshape(-1), truth[name].reshape(-1)
        ok += int((a == b).sum())
        tot += a.numel()
    return round(ok / tot, 4)


# --------------------------------------------------------------------- search
ALIAS = {"mul": ("mul_lo", "mul_hi"), "add": ("add_d", "add_c"),
         "sub": ("sub_d", "sub_b"), "const": ("zero", "carry0", "borrow0"),
         "sel": ("sel",)}


def expand_modules(spec):
    if not spec:
        return None
    names = set()
    for part in spec.split(","):
        part = part.strip()
        names |= set(ALIAS.get(part, (part,)))
    return names


def free_cols(name, ties):
    """Flat columns that are FREE under the ties (the rest are derived)."""
    shape = dict((a, b) for a, b, _ in CELLS)[name]
    k = 1
    for d in shape:
        k *= d
    if "inv" in ties and name in ("sub_d", "sub_b"):
        return []
    if "sym" in ties and name in ("mul_lo", "mul_hi"):
        return [u * 10 + v for u in range(10) for v in range(u, 10)]
    if "sym" in ties and name in ("add_d", "add_c"):
        return [(u * 10 + v) * 2 + c for u in range(10)
                for v in range(u, 10) for c in range(2)]
    return list(range(k))


def cell_index(only=None, ties=()):
    """flat list of (name, col, n_candidates)"""
    out = []
    for name, shape, n in CELLS:
        if only is not None and name not in only:
            continue
        for c in free_cols(name, ties):
            out.append((name, c, n))
    return out


def _flog(msg):
    print(msg, flush=True)


def search(model, s, tgt, cells, obj="digit", block=64, sweeps=40, gen=None,
           log=_flog, tag="", patience=3, time_budget=0.0):
    """Block-greedy coordinate search with prefix verification.

    Each block evaluates every (cell, candidate) single mutation of the current
    assignment in ONE population forward (Jacobi).  The improving moves are then
    sorted by gain and applied as cumulative PREFIXES in a second population
    forward, and the best prefix is kept -- so interactions between
    simultaneously-accepted moves are always verified, never assumed.
    """
    CMAX = max(n for _, _, n in cells)
    P = model.P
    assert P >= block * CMAX, f"population {P} < block*{CMAX}"
    base = model.snapshot(0)

    def cur():
        model.load(base)
        d, e = model.score(s, tgt)
        return d[0].item(), e[0].item()

    d0, e0 = cur()
    t0 = time.time()
    order = list(range(len(cells)))
    for sw in range(sweeps):
        if gen is not None:
            perm = torch.randperm(len(cells), generator=gen).tolist()
            order = perm
        moved = 0
        start = (d0, e0)
        for bstart in range(0, len(order), block):
            blk = order[bstart:bstart + block]
            model.load(base)
            # ---- population of single mutations
            memb = []
            for gi, ci in enumerate(blk):
                name, col, n = cells[ci]
                f = model.flat(name)
                vals = torch.arange(CMAX, device=model.dev) % n
                rows = torch.arange(gi * CMAX, (gi + 1) * CMAX, device=model.dev)
                f[rows, col] = vals.to(f.dtype)
                for c in range(n):
                    memb.append((gi * CMAX + c, ci, c))
            d, e = model.score(s, tgt)
            sc = d if obj == "digit" else e
            if obj == "mix":
                sc = d + e
            sc = sc.cpu()
            b0 = (d0 if obj == "digit" else e0) if obj != "mix" else d0 + e0
            gains = {}
            for p, ci, c in memb:
                g = sc[p].item() - b0
                if g > 1e-9 and g > gains.get(ci, (0.0, 0))[0]:
                    gains[ci] = (g, c)
            if not gains:
                continue
            ranked = sorted(gains.items(), key=lambda kv: -kv[1][0])
            K = min(len(ranked), CMAX * block)
            # ---- prefix verification
            model.load(base)
            for j, (ci, (g, c)) in enumerate(ranked[:K]):
                name, col, n = cells[ci]
                f = model.flat(name)
                f[j:, col] = float(c) if name == "sel" else c
            d, e = model.score(s, tgt)
            sc2 = d if obj == "digit" else e
            if obj == "mix":
                sc2 = d + e
            sc2 = sc2[:K].cpu()
            k = int(sc2.argmax())
            if sc2[k].item() > b0:
                for j, (ci, (g, c)) in enumerate(ranked[:k + 1]):
                    name, col, n = cells[ci]
                    base[name].reshape(-1)[col] = (
                        float(c) if name == "sel" else c)
                moved += k + 1
                d0, e0 = cur()
        log(f"[{tag}] sweep={sw + 1} moves={moved} digit={d0:.4f} "
            f"exact={e0:.4f} ({time.time() - t0:.0f}s)")
        if time_budget and time.time() - t0 > time_budget:
            log(f"[{tag}] time budget reached")
            break
        if (d0, e0) == start:
            patience -= 1
            if patience <= 0:
                break
        else:
            patience = 3
    model.load(base)
    return base, d0, e0


# ------------------------------------------------------- genome / annealing
class Genome:
    """The whole discrete transducer as one flat int vector, so that a
    POPULATION member can be an independent search chain rather than a single
    mutation of a shared base."""

    def __init__(self, model):
        self.model = model
        self.spec = []
        off = 0
        for name, shape, n in CELLS:
            size = 1
            for d in shape:
                size *= d
            self.spec.append((name, shape, n, off, size))
            off += size
        self.n = off
        self.ncand = torch.zeros(off, dtype=torch.long, device=model.dev)
        for name, shape, n, o, s in self.spec:
            self.ncand[o:o + s] = n
        self.idx = torch.arange(off, device=model.dev)   # searchable positions

    def restrict(self, only):
        keep = []
        for name, shape, n, o, s in self.spec:
            if only is not None and name not in only:
                continue
            keep += [o + c for c in free_cols(name, self.model.ties)]
        self.idx = torch.tensor(keep, dtype=torch.long, device=self.model.dev)

    def push(self, g):
        P = g.shape[0]
        for name, shape, n, o, s in self.spec:
            v = g[:, o:o + s].reshape((P,) + shape)
            self.model.t[name].copy_(v.float() if name == "sel" else v)

    def pull(self):
        P = self.model.P
        cols = []
        for name, shape, n, o, s in self.spec:
            cols.append(self.model.t[name].reshape(P, s).long())
        return torch.cat(cols, 1)

    def random(self, P, gen):
        u = torch.rand(P, self.n, generator=gen).to(self.model.dev)
        return (u * self.ncand[None].float()).long().clamp_max_(
            self.ncand[None] - 1)

    def snap_to_dict(self, row):
        out = {}
        for name, shape, n, o, s in self.spec:
            v = row[o:o + s].reshape(shape)
            out[name] = v.float() if name == "sel" else v
        return out


def anneal(model, gm, s, tgt, steps, t0, t1, gen, log=_flog, tag="",
           init=None, log_every=500):
    """P independent Metropolis chains, one proposal each per forward pass.

    Every chain is a full table assignment; one random cell is re-drawn per
    chain per step and accepted with the Metropolis rule on the digit-accuracy
    objective.  P chains advance per forward, so a 640-chain run at ~30 ms per
    forward is ~20,000 objective evaluations per second."""
    P = model.P
    G = gm.random(P, gen).to(model.dev) if init is None else init.clone()
    gm.push(G)
    cur, _ = model.score(s, tgt)
    best, bestG = cur.clone(), G.clone()
    ar = torch.arange(P, device=model.dev)
    t_start = time.time()
    for st in range(steps):
        T = t0 * (t1 / t0) ** (st / max(steps - 1, 1))
        ci = gm.idx[torch.randint(0, gm.idx.numel(), (P,), device=model.dev)]
        v = (torch.rand(P, device=model.dev) * gm.ncand[ci].float()).long()
        v = torch.minimum(v, gm.ncand[ci] - 1)
        old = G[ar, ci].clone()
        G[ar, ci] = v
        gm.push(G)
        new, _ = model.score(s, tgt)
        dE = new - cur
        acc = (dE >= 0) | (torch.rand(P, device=model.dev) < (dE / T).exp())
        G[ar, ci] = torch.where(acc, v, old)
        cur = torch.where(acc, new, cur)
        imp = cur > best
        if imp.any():
            best = torch.where(imp, cur, best)
            bestG[imp] = G[imp]
        if (st + 1) % log_every == 0 or st == steps - 1:
            log(f"[{tag}] anneal step={st + 1}/{steps} T={T:.2e} "
                f"cur(max={cur.max():.4f} mean={cur.mean():.4f}) "
                f"best={best.max():.4f} ({time.time() - t_start:.0f}s)")
    return bestG, best


# ----------------------------------------------------------------------- main
def build_task(modulus, slots, train_x, split_seed, device):
    units = [x for x in range(1, modulus) if math.gcd(x, modulus) == 1]
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(units), generator=g).tolist()
    tr = [units[i] for i in perm[:train_x]]
    he = [units[i] for i in perm[train_x:]]

    def tens(xs):
        inp = torch.tensor([digits_le(x, slots) for x in xs], dtype=torch.long)
        tgt = torch.tensor([digits_le((x * x) % modulus, slots) for x in xs],
                           dtype=torch.long)
        return inp.to(device), tgt.to(device)

    W = slots + 1
    nd = torch.tensor(digits_le(modulus, W), dtype=torch.long)
    return tens(tr), tens(he), nd, tr, he


def verify(args, device):
    """Correctness gate: the integer sim must equal the float sim with hard
    states, on the construction AND on random inits."""
    from probe_alu_depth import DigitALU
    import torch.nn.functional as F
    (xin, xt), (hin, ht), nd, tr, he = build_task(
        args.modulus, args.slots, args.train_x, args.split_seed, device)
    W = args.slots + 1
    ndf = torch.zeros(W, 10, device=device)
    for i, d in enumerate(nd.tolist()):
        ndf[i, d] = 1.0
    xinf = F.one_hot(xin, 10).float()
    hinf = F.one_hot(hin, 10).float()
    ok = True
    for seed in range(args.verify_seeds):
        for use_c in ((True, False) if seed == 0 else (False,)):
            torch.manual_seed(seed)
            fm = DigitALU(args.slots, 2, 2, 11, 1.0, 0.0, True,
                          "quotient", args.max_quot, "tree", "serial").to(device)
            if use_c:
                fm.construct()
            fm.eval()
            with torch.no_grad():
                lg = fm(xinf, ndf)
                fex = (lg.argmax(-1) == xt).all(1).float().mean().item()
                fdg = (lg.argmax(-1) == xt).float().mean().item()
                lgh = fm(hinf, ndf)
                fexh = (lgh.argmax(-1) == ht).all(1).float().mean().item()
            im = IntALU(1, args.slots, nd, 2, 2, args.max_quot, device)
            snap = {}
            with torch.no_grad():
                snap["mul_lo"] = fm.Tmul[..., :10].argmax(-1)
                snap["mul_hi"] = fm.Tmul[..., 10:].argmax(-1)
                snap["add_d"] = fm.Tadd[..., :10].argmax(-1)
                snap["add_c"] = fm.Tadd[..., 10:].argmax(-1)
                snap["sub_d"] = fm.Tsub[..., :10].argmax(-1)
                snap["sub_b"] = fm.Tsub[..., 10:].argmax(-1)
                snap["zero"] = fm.zero.argmax().view(1)
                snap["carry0"] = fm.carry0.argmax().view(1)
                snap["borrow0"] = fm.borrow0.argmax().view(1)
                w, bs = fm.sel.weight[0], fm.sel.bias[0]
                sel = torch.zeros(2, 2, device=device)
                for a in range(2):
                    for b in range(2):
                        sel[a, b] = w[a] + w[2 + b] + bs
                snap["sel"] = sel
            im.load(snap)
            idg, iex = im.score(xin, xt)
            _, iexh = im.score(hin, ht)
            match = (abs(iex[0].item() - fex) < 1e-6
                     and abs(idg[0].item() - fdg) < 1e-6
                     and abs(iexh[0].item() - fexh) < 1e-6)
            ok = ok and match
            print(f"[verify] seed={seed} construct={use_c} "
                  f"float(exact={fex:.4f} digit={fdg:.4f} held={fexh:.4f}) "
                  f"int(exact={iex[0]:.4f} digit={idg[0]:.4f} "
                  f"held={iexh[0]:.4f}) MATCH={match}", flush=True)
    print(f"[verify] ALL MATCH = {ok}", flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulus", type=int, default=323)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--train-x", type=int, default=250)
    ap.add_argument("--max-quot", type=int, default=10)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--sweeps", type=int, default=40)
    ap.add_argument("--obj", default="digit", choices=["digit", "exact", "mix"])
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-seeds", type=int, default=3)
    ap.add_argument("--basin", action="store_true")
    ap.add_argument("--exercise", action="store_true",
                    help="identifiability diagnostic: how many table cells does "
                         "the training set actually read, and how many bits of "
                         "supervision are available per bit of table?")
    ap.add_argument("--repair", type=int, default=0,
                    help="corrupt this many cells of the construction, then search")
    ap.add_argument("--repair-reps", type=int, default=1)
    ap.add_argument("--ils", type=int, default=0,
                    help="iterated local search: this many perturb+re-search rounds")
    ap.add_argument("--ils-k", type=int, default=8,
                    help="cells randomly re-drawn at each ILS perturbation")
    ap.add_argument("--time-budget", type=float, default=0.0)
    ap.add_argument("--anneal", type=int, default=0,
                    help="run this many parallel-Metropolis steps per chain "
                         "(population = independent chains) before polishing")
    ap.add_argument("--t0", type=float, default=3e-3)
    ap.add_argument("--t1", type=float, default=2e-5)
    ap.add_argument("--anneal-rounds", type=int, default=1,
                    help="reheat cycles: each round restarts the schedule from "
                         "the current best chains")
    ap.add_argument("--pop", type=int, default=0,
                    help="population size for --anneal (default block*10)")
    ap.add_argument("--tie", default="",
                    help="LEGAL hypothesis-class ties: 'sym' (Tmul/Tadd "
                         "symmetric in the two digit indices), 'inv' (Tsub "
                         "derived from Tadd).  Weight sharing only -- the "
                         "construction stays in the class.")
    ap.add_argument("--modules", default="",
                    help="LAB DIAGNOSTIC: search ONLY these modules "
                         "(mul/add/sub/const/sel or raw table names); every "
                         "other table is set to the construction.  Measures "
                         "which modules the end-of-chain objective identifies.")
    ap.add_argument("--polish", type=int, default=3,
                    help="number of top chains to greedy-polish after annealing")
    ap.add_argument("--tag", default="")
    ap.add_argument("--jsonl", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.verify:
        return verify(args, device)

    (xin, xt), (hin, ht), nd, tr, he = build_task(
        args.modulus, args.slots, args.train_x, args.split_seed, device)
    only = expand_modules(args.modules)
    ties = tuple(t for t in args.tie.split(",") if t)
    cells = cell_index(only, ties)
    CMAX = max(n for _, _, n in cells)
    P = args.block * CMAX
    model = IntALU(P, args.slots, nd, 2, 2, args.max_quot, device)
    model.set_ties(ties)
    ndl = nd.tolist()
    print(f"[{args.tag}] N={args.modulus} S={args.slots} train={len(tr)} "
          f"held={len(he)} cells={len(cells)} "
          f"candidates={sum(n for _, _, n in cells)} pop={P}", flush=True)

    model.construct()
    truth = model.snapshot(0)
    d, e = model.score(xin, xt)
    dh, eh = model.score(hin, ht)
    print(f"[{args.tag}] CONSTRUCTED digit={d[0]:.4f} exact={e[0]:.4f} "
          f"held_exact={eh[0]:.4f} struct="
          f"{structure_scores(truth, ndl)}", flush=True)

    gen = torch.Generator().manual_seed(args.seed)

    if args.exercise:
        import math as _m
        for split, inp, tg in (("train", xin, xt), ("held", hin, ht)):
            one = IntALU(1, args.slots, nd, 2, 2, args.max_quot, device)
            one.construct()
            one.count_on()
            one.forward(inp)
            tot_cells = tot_used = 0
            bits_all = bits_used = 0.0
            per = []
            for name, shape, n in CELLS:
                c = one.counts[name]
                used = int((c > 0).sum())
                tot_cells += c.numel()
                tot_used += used
                bits_all += c.numel() * _m.log2(n)
                bits_used += used * _m.log2(n)
                per.append(f"{name}={used}/{c.numel()}")
            sup = inp.shape[0] * args.slots * _m.log2(10)
            print(f"[{args.tag}] {split}: cells_read={tot_used}/{tot_cells} "
                  f"({' '.join(per)})", flush=True)
            print(f"[{args.tag}] {split}: table_bits(all)={bits_all:.0f} "
                  f"table_bits(read)={bits_used:.0f} "
                  f"supervision_bits={sup:.0f} "
                  f"ratio_sup/read={sup / max(bits_used, 1e-9):.3f}", flush=True)
        return 0

    if args.basin:
        for k in (0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1007):
            ds, es = [], []
            for rep in range(1 if k == 0 else 5):
                model.construct()
                g = torch.Generator().manual_seed(1000 * k + rep)
                pick = torch.randperm(len(cells), generator=g)[:k].tolist()
                for ci in pick:
                    name, col, n = cells[ci]
                    v = int(torch.randint(0, n, (1,), generator=g))
                    model.flat(name)[:, col] = float(v) if name == "sel" else v
                d, e = model.score(xin, xt)
                ds.append(d[0].item())
                es.append(e[0].item())
            print(f"[{args.tag}] basin k={k:>4}/{len(cells)} "
                  f"digit={sum(ds)/len(ds):.4f} exact={sum(es)/len(es):.4f}",
                  flush=True)
        return 0

    if args.anneal:
        gm = Genome(model)
        gm.restrict(only)
        init = None
        if only is not None:
            # LAB DIAGNOSTIC: every table outside --modules starts (and stays)
            # at the construction; only the named ones are searched.
            model.construct()
            cg = gm.pull()
            r = gm.random(P, gen).to(model.dev)
            cg[:, gm.idx] = r[:, gm.idx]
            init = cg
        best = None
        for rnd in range(args.anneal_rounds):
            bg, bs = anneal(model, gm, xin, xt, args.anneal, args.t0, args.t1,
                            gen, tag=f"{args.tag}/rd{rnd}", init=init)
            order = bs.argsort(descending=True)
            init = bg[order[torch.arange(P, device=model.dev) % args.polish]]
            best = (bg, bs)
        bg, bs = best
        order = bs.argsort(descending=True)[:args.polish]
        rows = []
        for rank, idx in enumerate(order.tolist()):
            snap = gm.snap_to_dict(bg[idx])
            model.load(snap)
            d, e = model.score(xin, xt)
            print(f"[{args.tag}] anneal top{rank} digit={d[0]:.4f} "
                  f"exact={e[0]:.4f} -> polishing", flush=True)
            fin, d0, e0 = search(model, xin, xt, cells, args.obj, args.block,
                                 args.sweeps, gen, tag=f"{args.tag}/pol{rank}",
                                 time_budget=args.time_budget)
            model.load(fin)
            d, e = model.score(xin, xt)
            dh, ehd = model.score(hin, ht)
            ss = structure_scores(fin, ndl)
            ca = cell_agreement(fin, truth)
            row = {"tag": args.tag, "rep": rank, "argv": sys.argv[1:],
                   "train_digit": round(d[0].item(), 4),
                   "train_exact_hard": round(e[0].item(), 4),
                   "held_digit": round(dh[0].item(), 4),
                   "held_exact_hard": round(ehd[0].item(), 4),
                   "struct": ss, "cell_agree": ca}
            rows.append(row)
            print(f"[{args.tag}] top{rank} FINAL train_digit={d[0]:.4f} "
                  f"train_exact_hard={e[0]:.4f} held_exact_hard={ehd[0]:.4f} "
                  f"cell_agree={ca} struct={ss}", flush=True)
        if args.jsonl:
            with open(args.jsonl, "a") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
        return 0

    rows = []
    reps = args.repair_reps if args.repair else args.restarts
    for rep in range(reps):
        g = torch.Generator().manual_seed(args.seed * 100 + rep)
        if args.repair:
            model.construct()
            pick = torch.randperm(len(cells), generator=g)[:args.repair].tolist()
            for ci in pick:
                name, col, n = cells[ci]
                v = int(torch.randint(0, n, (1,), generator=g))
                model.flat(name)[:, col] = float(v) if name == "sel" else v
            base = model.snapshot(0)
            model.load(base)
            d, e = model.score(xin, xt)
            print(f"[{args.tag}] rep={rep} corrupted k={args.repair} "
                  f"digit={d[0]:.4f} exact={e[0]:.4f} "
                  f"cell_agree={cell_agreement(base, truth)}", flush=True)
        elif only is not None:
            # LAB DIAGNOSTIC: everything outside `--modules` is set to the
            # construction; only the named modules start random and are searched.
            model.construct()
            base = model.snapshot(0)
            rndm = IntALU(1, args.slots, nd, 2, 2, args.max_quot, device)
            rndm.set_ties(ties)
            rndm.randomize(g)
            rs = rndm.snapshot(0)
            for name in only:
                base[name] = rs[name]
            model.load(base)
            d, e = model.score(xin, xt)
            print(f"[{args.tag}] rep={rep} modules={sorted(only)} random, rest "
                  f"CONSTRUCTED: digit={d[0]:.4f} exact={e[0]:.4f}", flush=True)
        else:
            model.randomize(g)
            base = model.snapshot(0)
            model.load(base)
            d, e = model.score(xin, xt)
            print(f"[{args.tag}] rep={rep} random init digit={d[0]:.4f} "
                  f"exact={e[0]:.4f}", flush=True)

        tag = f"{args.tag}/r{rep}"
        best, d0, e0 = search(model, xin, xt, cells, args.obj, args.block,
                              args.sweeps, gen, tag=tag,
                              time_budget=args.time_budget)
        for it in range(args.ils):
            model.load(best)
            pick = torch.randperm(len(cells), generator=g)[:args.ils_k].tolist()
            pert = {k: v.clone() for k, v in best.items()}
            for ci in pick:
                name, col, n = cells[ci]
                v = int(torch.randint(0, n, (1,), generator=g))
                pert[name].reshape(-1)[col] = float(v) if name == "sel" else v
            model.load(pert)
            cand, dc, ec = search(model, xin, xt, cells, args.obj, args.block,
                                  args.sweeps, gen, tag=f"{tag}/ils{it}",
                                  time_budget=args.time_budget)
            key = (dc if args.obj == "digit" else ec)
            cur = (d0 if args.obj == "digit" else e0)
            if key > cur:
                best, d0, e0 = cand, dc, ec
                print(f"[{tag}] ils{it} ACCEPT digit={d0:.4f} exact={e0:.4f}",
                      flush=True)
        model.load(best)
        d, e = model.score(xin, xt)
        dh, ehd = model.score(hin, ht)
        ss = structure_scores(best, ndl)
        ca = cell_agreement(best, truth)
        row = {"tag": args.tag, "rep": rep, "argv": sys.argv[1:],
               "train_digit": round(d[0].item(), 4),
               "train_exact_hard": round(e[0].item(), 4),
               "held_digit": round(dh[0].item(), 4),
               "held_exact_hard": round(ehd[0].item(), 4),
               "struct": ss, "cell_agree": ca}
        rows.append(row)
        print(f"[{args.tag}] rep={rep} FINAL train_digit={d[0]:.4f} "
              f"train_exact_hard={e[0]:.4f} held_exact_hard={ehd[0]:.4f} "
              f"cell_agree={ca} struct={ss}", flush=True)

    if args.jsonl:
        with open(args.jsonl, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
