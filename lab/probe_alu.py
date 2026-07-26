#!/usr/bin/env python
"""LAB ONLY -- the oracle-equivalent of a DIGIT-COMPOSITIONAL readout.

group-rotation §7 measured the ceiling of a readout that is an arbitrary
function of the residue, by freezing the representation at the truth and
training only the readout:

    N=323 -> 1.000,  N=899 -> 0.614,  N=2021 -> 0.337,  N=10403 -> 0.031
    (closed form P[x^2 already seen]: 1.000 / 0.614 / 0.337 / 0.072)

This is the same measurement for a readout in which EVERY learned tensor is a
table indexed by a *digit tuple* with a small, modulus-independent alphabet.
Nothing is indexed by the residue, so the coverage argument cannot apply; the
question is what replaces it.

Three tiers are measured, mirroring how §7 separated representation from
optimisation:

  --construct   tables set to the exact solution (LAB DIAGNOSTIC, never in a
                submission).  Answers: is the ceiling of this readout family
                1.000 at every modulus?  This is the direct analogue of
                probe_step.py --oracle.
  (default)     tables learned from random init on the same 250 training x.
                Answers: is the family identifiable from e1-sized supervision?
  --freeze X    a middle tier: freeze some tables at the truth, learn the rest.

Structure of the forward pass (a shift, a scan order and a gate; all *values*
learned):

    lo_ij, hi_ij = Tmul[d_i, d_j]                      SHARED over places
    r = zero
    for k = K-1 .. 0:                                  Horner over places of x^2
        r = shift_up(r)                                structural, no params
        for (i,j) with i+j == k:  r = add_scan(r, [lo_ij, hi_ij])
        reduce(r, N)                                   see --reduce-mode
    y = r[:S]

    add_scan  : slot scan LSB->MSB, Tadd[digit, addend, carry] -> digit, carry
    cond_sub  : slot scan LSB->MSB, Tsub[digit, n_digit, borrow] -> digit, borrow
                then a learned gate on the final borrow state chooses between
                the scanned result and the input.

Total parameters ~7k, every one of them indexed by (digit, digit, small state).
There is no tensor with an index that ranges over Z_N.

--reduce-mode  (explore/alu-depth) -- the reduction is 80-86% of the sequential
depth and depth is the measured obstruction to trainability, so the reduction is
the thing to shorten.  All three modes keep the SAME state alphabet (digits 10,
carry 2, borrow 2) and add no index over Z_N:

  serial    R=11 weight-tied cond_sub(r, N).  10 of 11 are no-ops.  DEPTH K*R*W.
  binary    cond_sub against 8N,4N,2N,N.  The multiples are built from digits(N)
            by the SAME learned Tadd (2N = add_scan(N,N), ...), so this costs
            zero new parameters.  DEPTH K*4*W.
  quotient  all of 0*N .. (Q+1)*N are subtracted from r IN PARALLEL (one scan,
            batched over the multiple index), the final borrow state of each is
            a learned comparison bit, and a learned scorer over the adjacent
            pair (borrow_m, borrow_{m+1}) picks the largest m that does not
            borrow -- i.e. a learned quotient digit, alphabet Q+1.  One scan,
            one select.  DEPTH K*(W+1).

Self-generated values only; nothing under data/generated/ is opened.
"""

from __future__ import annotations

import argparse
import math
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


class DigitALU(nn.Module):
    def __init__(self, slots: int, n_carry: int = 2, n_borrow: int = 2,
                 reduce_steps: int = 11, tau: float = 1.0,
                 identity_init: float = 0.0, hard: bool = False,
                 reduce_mode: str = "serial", max_quot: int = 10,
                 mul_mode: str = "horner"):
        super().__init__()
        self.S = slots
        self.K = 2 * slots - 1
        self.W = slots + 1
        self.R = reduce_steps
        self.Ca, self.Cb = n_carry, n_borrow
        self.tau = tau
        self.hard = hard
        self.reduce_mode = reduce_mode
        self.mul_mode = mul_mode
        self.Q = max_quot
        if reduce_mode == "binary":
            p, self.bin_list = 1, []
            while p <= max_quot:
                self.bin_list.append(p)
                p *= 2
            self.bin_list.reverse()                       # [8, 4, 2, 1]
            self.needed = sorted({0} | set(self.bin_list))
        elif reduce_mode == "quotient":
            self.bin_list = []
            self.needed = list(range(max_quot + 2))       # 0 .. Q+1
        else:
            self.bin_list, self.needed = [], []
        self.Tmul = nn.Parameter(torch.randn(10, 10, 20) * 0.5)
        self.Tadd = nn.Parameter(torch.randn(10, 10, n_carry, 10 + n_carry) * 0.5)
        self.Tsub = nn.Parameter(torch.randn(10, 10, n_borrow, 10 + n_borrow) * 0.5)
        self.zero = nn.Parameter(torch.randn(10) * 0.5)
        self.carry0 = nn.Parameter(torch.randn(n_carry) * 0.5)
        self.borrow0 = nn.Parameter(torch.randn(n_borrow) * 0.5)
        self.gate = nn.Linear(n_borrow, 1)
        # learned quotient-digit scorer: reads the adjacent pair of comparison
        # (= final borrow) states and scores "m is the largest non-borrowing
        # multiple".  2*n_borrow inputs -> 1 logit; 5 parameters.
        self.sel = nn.Linear(2 * n_borrow, 1)
        # A 280-step soft chain from random init is badly conditioned: every
        # scan scrambles the register before any of them is right.  A learned
        # copy-through path makes each scan the IDENTITY at init, so the chain
        # starts well-conditioned and learning perturbs away from it.  This is
        # residual/identity initialisation, not a supplied arithmetic rule --
        # copy_scale is trainable and the tables can override it.
        self.copy_scale = nn.Parameter(torch.tensor(float(identity_init)))
        if identity_init > 0:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -4.0)  # reduction starts closed

    # ---------------- construction (LAB DIAGNOSTIC ONLY) ----------------
    @torch.no_grad()
    def construct(self, which=("mul", "add", "sub", "gate", "zero", "sel")):
        if "mul" in which:
            t = torch.full_like(self.Tmul, -BIG)
            for a in range(10):
                for b in range(10):
                    t[a, b, (a * b) % 10] = BIG
                    t[a, b, 10 + (a * b) // 10] = BIG
            self.Tmul.copy_(t)
        if "add" in which:
            assert self.Ca == 2
            t = torch.full_like(self.Tadd, -BIG)
            for u in range(10):
                for v in range(10):
                    for c in range(2):
                        s = u + v + c
                        t[u, v, c, s % 10] = BIG
                        t[u, v, c, 10 + s // 10] = BIG
            self.Tadd.copy_(t)
        if "sub" in which:
            assert self.Cb == 2
            t = torch.full_like(self.Tsub, -BIG)
            for u in range(10):
                for v in range(10):
                    for c in range(2):
                        s = u - v - c
                        t[u, v, c, s % 10] = BIG
                        t[u, v, c, 10 + (1 if s < 0 else 0)] = BIG
            self.Tsub.copy_(t)
        if "gate" in which:
            self.gate.weight.copy_(torch.tensor([[BIG, -BIG]]))
            self.gate.bias.zero_()
        if "sel" in which:
            # state 0 of the borrow alphabet is "no borrow" (see the Tsub
            # construction above).  The quotient digit is the unique m with
            # (no-borrow at m, borrow at m+1).
            w = torch.zeros(1, 2 * self.Cb)
            w[0, 0], w[0, 1] = BIG / 2, -BIG / 2
            w[0, self.Cb + 0], w[0, self.Cb + 1] = -BIG / 2, BIG / 2
            self.sel.weight.copy_(w)
            self.sel.bias.zero_()
        if "zero" in which:
            z = torch.full((10,), -BIG)
            z[0] = BIG
            self.zero.copy_(z)
            c = torch.full((self.Ca,), -BIG)
            c[0] = BIG
            self.carry0.copy_(c)
            b = torch.full((self.Cb,), -BIG)
            b[0] = BIG
            self.borrow0.copy_(b)
        self.copy_scale.zero_()

    # ---------------- scans ----------------
    def _sm(self, logits):
        p = F.softmax(logits / self.tau, -1)
        if getattr(self, "stat", None) is not None and p.dim() > 1:
            self.stat[0] += p.max(-1).values.mean().item()
            self.stat[1] += 1
        if self.hard:  # straight-through: close the continuous side-channel
            h = F.one_hot(p.argmax(-1), p.shape[-1]).to(p.dtype)
            p = h + p - p.detach()
        return p

    def add_scan(self, r, addend):
        c = self._sm(self.carry0).expand(r.shape[0], self.Ca)
        outs = []
        for m in range(r.shape[1]):
            o = torch.einsum("bu,bv,bc,uvco->bo", r[:, m], addend[:, m], c, self.Tadd)
            outs.append(self._sm(o[:, :10] + self.copy_scale * r[:, m]))
            c = self._sm(o[:, 10:] + self.copy_scale * c)
        return torch.stack(outs, 1)

    def tree_sum(self, regs):
        """Balanced-tree sum of equal-width registers.  ceil(log2 n) chained
        add_scans instead of n-1; every level is one batched scan."""
        while len(regs) > 1:
            carry = [regs[-1]] if len(regs) % 2 else []
            pairs = [(regs[i], regs[i + 1]) for i in range(0, len(regs) - 1, 2)]
            b = regs[0].shape[0]
            out = self.add_scan(torch.cat([a for a, _ in pairs], 0),
                                torch.cat([c for _, c in pairs], 0))
            regs = [out[i * b:(i + 1) * b] for i in range(len(pairs))] + carry
        return regs[0]

    def sub_scan(self, r, sub):
        """r - sub, slot scan LSB->MSB.  Returns (digits, final borrow state)."""
        b = r.shape[0]
        c = self._sm(self.borrow0).expand(b, self.Cb)
        outs = []
        for m in range(r.shape[1]):
            o = torch.einsum("bu,bv,bc,uvco->bo", r[:, m], sub[:, m], c, self.Tsub)
            outs.append(self._sm(o[:, :10] + self.copy_scale * r[:, m]))
            c = self._sm(o[:, 10:] + self.copy_scale * c)
        return torch.stack(outs, 1), c

    def cond_sub(self, r, sub):
        t, c = self.sub_scan(r, sub)
        g = torch.sigmoid(self.gate(c))[:, :, None]  # (B,1,1)
        return g * t + (1 - g) * r

    # ---------------- learned multiples of N (zero new parameters) ----------
    def multiples(self, ndig):
        """{m*N} built from digits(N) with the SAME learned Tadd.

        m*N = a*N + b*N with a+b=m, so a balanced schedule needs
        ceil(log2(max m)) chained add_scans -- a shared prefix paid once per
        forward, not per Horner place.  No new parameter, no new alphabet.
        """
        z = self._sm(self.zero)
        have = {0: z.view(1, 1, 10).expand(1, self.W, 10).contiguous(),
                1: ndig[None]}
        target = set(self.needed)
        depth = 0
        while not target <= set(have):
            newly = []
            for t in sorted(target - set(have)):
                cand = [a for a in have if a <= t - a and (t - a) in have]
                if cand:
                    newly.append((t, max(cand), t - max(cand)))
            if not newly:                       # need a bigger intermediate
                mx = max(have)
                newly = [(2 * mx, mx, mx)]
            A = torch.cat([have[a] for _, a, _ in newly], 0)
            B = torch.cat([have[b] for _, _, b in newly], 0)
            out = self.add_scan(A, B)
            for i, (t, _, _) in enumerate(newly):
                have[t] = out[i:i + 1]
            depth += 1
        self.mult_depth = depth
        return torch.cat([have[m] for m in self.needed], 0)      # (M, W, 10)

    def quot_reduce(self, r, mults):
        """One learned quotient digit + one subtraction.

        All M = Q+2 candidate subtractions r - m*N run in ONE scan (batched over
        m), so the depth is W, not M*W.  The final borrow state of candidate m
        is the learned comparison bit [m*N > r]; a learned scorer over the
        adjacent pair picks the largest non-borrowing m.  The quotient digit is
        a distribution over the Q+1 alphabet {0..Q} -- discrete and small.
        """
        b, M = r.shape[0], mults.shape[0]
        rr = r[:, None].expand(b, M, self.W, 10).reshape(b * M, self.W, 10)
        ss = mults[None].expand(b, M, self.W, 10).reshape(b * M, self.W, 10)
        t, c = self.sub_scan(rr, ss)
        t = t.view(b, M, self.W, 10)
        c = c.view(b, M, self.Cb)
        pair = torch.cat([c[:, :-1], c[:, 1:]], dim=-1)          # (B, M-1, 2Cb)
        w = self._sm(self.sel(pair).squeeze(-1))                 # (B, M-1)
        self.last_q = w
        return torch.einsum("bm,bmwo->bwo", w, t[:, :-1])

    def reduce(self, r, ndig, mults):
        b = r.shape[0]
        if self.reduce_mode == "serial":
            sub = ndig[None].expand(b, self.W, 10)
            for _ in range(self.R):
                r = self.cond_sub(r, sub)
            return r
        if self.reduce_mode == "binary":
            for m in self.bin_list:
                sub = mults[self.needed.index(m)][None].expand(b, self.W, 10)
                r = self.cond_sub(r, sub)
            return r
        return self.quot_reduce(r, mults)

    def _leaves(self, prod, z, F):
        """Pack the S^2 partial products into as few full-width registers as
        possible.  Two products whose offsets differ by >= 2 occupy disjoint
        slots, so they can share a register for free; that is what halves the
        number of tree leaves (S^2 -> ~S+1) before a single add is spent."""
        buckets = {}
        for (i, j), lh in prod.items():
            buckets.setdefault(i + j, []).append(lh)
        leaves = []
        for par in (0, 1):
            offs = [k for k in sorted(buckets) if k % 2 == par]
            if not offs:
                continue
            for t in range(max(len(buckets[k]) for k in offs)):
                cols = [z] * F
                for k in offs:
                    if t < len(buckets[k]):
                        cols[k], cols[k + 1] = buckets[k][t]
                leaves.append(torch.stack(cols, 1))
        return leaves

    def forward_tree(self, s, ndig, mults, prod, z):
        """Full 2S-digit product by a TREE (log depth, not chained), then one
        long division.  Trades the Horner chain's S^2 serial adds for
        ceil(log2 leaves) parallel ones; the divide is S+1 reductions."""
        b, S, F = s.shape[0], self.S, 2 * self.S
        P = self.tree_sum(self._leaves(prod, z, F))
        r = z[:, None].expand(b, self.W, 10)
        for t in range(F - 1, -1, -1):
            r = torch.cat([P[:, t:t + 1], r[:, : self.W - 1]], dim=1)
            if t <= S:
                r = self.reduce(r, ndig, mults)
        return torch.log(r[:, :S] + 1e-9)

    def forward(self, s, ndig):
        b = s.shape[0]
        z = self._sm(self.zero).expand(b, 10)
        mults = self.multiples(ndig) if self.needed else None
        prod = {}
        for i in range(self.S):
            for j in range(self.S):
                o = torch.einsum("bu,bv,uvo->bo", s[:, i], s[:, j], self.Tmul)
                prod[(i, j)] = (self._sm(o[:, :10]), self._sm(o[:, 10:]))
        if self.mul_mode == "tree":
            return self.forward_tree(s, ndig, mults, prod, z)
        r = z[:, None].expand(b, self.W, 10)
        for k in range(self.K - 1, -1, -1):
            r = torch.cat([z[:, None], r[:, : self.W - 1]], dim=1)  # x10
            for i in range(self.S):
                j = k - i
                if 0 <= j < self.S:
                    lo, hi = prod[(i, j)]
                    slots = [lo[:, None], hi[:, None]] + \
                            [z[:, None]] * (self.W - 2)
                    r = self.add_scan(r, torch.cat(slots, dim=1))
            r = self.reduce(r, ndig, mults)
        return torch.log(r[:, : self.S] + 1e-9)

    # ---------------- sequential-depth accounting ----------------
    def depth(self):
        """Chained soft (softmax) steps on the critical path.

        `main` is the x -> y path, the number the digit-carry report quotes as
        "~280 sequential soft table lookups".  `prefix` is the N-only multiples
        tree, a shared prefix merged in at the first reduction.
        """
        W, S, K = self.W, self.S, self.K
        if self.reduce_mode == "serial":
            per = self.R * W
        elif self.reduce_mode == "binary":
            per = len(self.bin_list) * W
        else:
            per = W + 1                        # one scan + one select
        if self.mul_mode == "tree":
            F = 2 * S
            counts = [min(k + 1, S, 2 * S - 1 - k) for k in range(2 * S - 1)]
            n_leaf = max([counts[k] for k in range(0, 2 * S - 1, 2)] or [0]) \
                + max([counts[k] for k in range(1, 2 * S - 1, 2)] or [0])
            adds = math.ceil(math.log2(n_leaf)) * F if n_leaf > 1 else 0
            red = (S + 1) * per
        else:
            adds = S * S * W                   # one add_scan per (i,j) pair
            red = K * per
        prefix = 0
        if self.needed:
            m = 1
            while m < max(self.needed):
                m *= 2
                prefix += 1
            prefix *= W
        return {"main": 1 + adds + red, "adds": adds, "reduce": red,
                "prefix": prefix}

    def alphabet(self):
        """Size of every discrete inter-step state (the family's constraint)."""
        a = {"digit": 10, "carry": self.Ca, "borrow": self.Cb}
        if self.reduce_mode == "quotient":
            a["quotient"] = self.Q + 1
        return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulus", type=int, default=323)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--train-x", type=int, default=250)
    ap.add_argument("--reduce", type=int, default=11)
    ap.add_argument("--reduce-mode", default="serial",
                    choices=["serial", "binary", "quotient"],
                    help="serial: R tied cond_sub(r,N) [digit-carry default]. "
                         "binary: cond_sub against 8N,4N,2N,N. "
                         "quotient: one learned quotient digit + one subtract")
    ap.add_argument("--mul-mode", default="horner", choices=["horner", "tree"],
                    help="horner: reduce after every place of x^2 [default]. "
                         "tree: full 2S-digit product by a log-depth tree, "
                         "then one long division")
    ap.add_argument("--max-quot", type=int, default=10,
                    help="alphabet of the learned quotient digit / largest "
                         "multiple of N the reduction can remove")
    ap.add_argument("--eval-hard", action="store_true",
                    help="also report train_exact with every inter-step state "
                         "snapped to argmax -- prices the continuous "
                         "side-channel through the soft register")
    ap.add_argument("--depth-only", action="store_true",
                    help="print the sequential soft-step count and exit")
    ap.add_argument("--carry", type=int, default=2)
    ap.add_argument("--borrow", type=int, default=2)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--tau-final", type=float, default=None,
                    help="anneal the state temperature from --tau to this")
    ap.add_argument("--identity-init", type=float, default=0.0,
                    help="copy-through logit scale; >0 makes every scan the "
                         "identity at init (trainable)")
    ap.add_argument("--hard", action="store_true",
                    help="straight-through discrete states: closes the "
                         "continuous side-channel through the soft digits")
    ap.add_argument("--construct", action="store_true",
                    help="LAB DIAGNOSTIC: set every table to the exact solution")
    ap.add_argument("--freeze", nargs="*", default=[],
                    help="subset of mul/add/sub/gate/zero to construct+freeze")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    modulus, S = args.modulus, args.slots
    units = [] if args.depth_only else \
        [x for x in range(1, modulus) if math.gcd(x, modulus) == 1]
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

    model = DigitALU(S, args.carry, args.borrow, args.reduce, args.tau,
                     args.identity_init, args.hard, args.reduce_mode,
                     args.max_quot, args.mul_mode).to(device)
    frozen = []
    if args.construct:
        model.construct()
        frozen = ["Tmul", "Tadd", "Tsub", "gate", "sel", "zero", "carry0",
                  "borrow0"]
    elif args.freeze:
        model.construct(tuple(args.freeze))
        name = {"mul": ["Tmul"], "add": ["Tadd"], "sub": ["Tsub"],
                "gate": ["gate"], "sel": ["sel"],
                "zero": ["zero", "carry0", "borrow0"]}
        for f in args.freeze:
            frozen += name[f]
    for n, p in model.named_parameters():
        if any(n.startswith(f) for f in frozen):
            p.requires_grad_(False)
    unused = {"serial": ["sel"], "binary": ["sel"], "quotient": ["gate"]}
    n_par = sum(p.numel() for n, p in model.named_parameters()
                if not any(n.startswith(u) for u in unused[args.reduce_mode]))
    n_tr = sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad
               and not any(n.startswith(u) for u in unused[args.reduce_mode]))
    d = model.depth()
    # the largest quotient this reduction is asked to produce -- a structural
    # sizing check on self-generated values, printed as a diagnostic only
    qmax = rmax = 0
    for x in (() if args.depth_only else units):
        rr = 0
        if args.mul_mode == "tree":
            pd = digits_le(x * x, 2 * S)
            for t in range(2 * S - 1, -1, -1):
                rr = rr * 10 + pd[t]
                rmax = max(rmax, rr)
                if t <= S:
                    qmax = max(qmax, rr // modulus)
                    rr %= modulus
        else:
            for k in range(model.K - 1, -1, -1):
                rr = rr * 10 + sum(digits_le(x, S)[i] * digits_le(x, S)[k - i]
                                   for i in range(S) if 0 <= k - i < S)
                rmax = max(rmax, rr)
                qmax = max(qmax, rr // modulus)
                rr %= modulus
    print(f"[{args.tag}] modulus={modulus} S={S} K={model.K} W={W} "
          f"mul={args.mul_mode} mode={args.reduce_mode} R={model.R} "
          f"Q={args.max_quot} "
          f"units={len(units)} train={len(train_x)} held={len(held_x)} "
          f"params={n_par:,} trainable={n_tr:,} frozen={frozen}", flush=True)
    print(f"[{args.tag}] DEPTH main={d['main']} (adds={d['adds']} "
          f"reduce={d['reduce']}) N-prefix={d['prefix']} "
          f"alphabet={model.alphabet()} true_max_quotient={qmax} "
          f"max_register={rmax} (W holds < {10 ** W})", flush=True)
    if args.depth_only:
        return 0

    @torch.no_grad()
    def evaluate(inp, tgt, chunk=4096, discrete=False):
        model.eval()
        was, model.hard = model.hard, (True if discrete else model.hard)
        ok, ce = 0, 0.0
        for i in range(0, inp.shape[0], chunk):
            lg = model(inp[i:i + chunk], ndig)
            ok += (lg.argmax(-1) == tgt[i:i + chunk]).all(dim=1).sum().item()
            ce += F.cross_entropy(lg.reshape(-1, 10), tgt[i:i + chunk].reshape(-1),
                                  reduction="sum").item()
        model.hard = was
        model.train()
        return ok / inp.shape[0], ce / (inp.shape[0] * tgt.shape[1])

    def extra(inp, tgt):
        """Is the learned solution actually a DISCRETE transducer?

        The register slots are 10-simplices, so a soft state is a continuous
        side-channel that can carry value information the digit alphabet
        cannot.  Re-running the SAME trained weights with every state snapped
        to its argmax prices that channel: if train_exact survives, the
        solution really is in the discrete family; if it collapses, the model
        is riding the continuum."""
        if not args.eval_hard:
            return ""
        h, _ = evaluate(inp, tgt, discrete=True)
        model.stat = [0.0, 0]
        evaluate(inp[:256], tgt[:256])
        s = model.stat[0] / max(model.stat[1], 1)
        model.stat = None
        return f" train_exact_hard={h:.3f} state_sharpness={s:.3f}"

    if args.construct:
        tr, tr_ce = evaluate(xin, xt)
        he, he_ce = evaluate(hin, ht)
        hd = ""
        if args.eval_hard:
            trh, _ = evaluate(xin, xt, discrete=True)
            heh, _ = evaluate(hin, ht, discrete=True)
            hd = f" train_exact_hard={trh:.3f} held_exact_hard={heh:.3f}"
        print(f"[{args.tag}] CONSTRUCTED  train_exact={tr:.3f} held_exact={he:.3f} "
              f"train_ce={tr_ce:.4f} held_ce={he_ce:.4f}{hd}", flush=True)
        return 0

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if args.tau_final is not None:
            f = step / args.steps
            model.tau = math.exp((1 - f) * math.log(args.tau)
                                 + f * math.log(args.tau_final))
        logits = model(xin, ndig)
        loss = F.cross_entropy(logits.reshape(-1, 10), xt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            tr, tr_ce = evaluate(xin, xt)
            he, he_ce = evaluate(hin, ht)
            print(f"[{args.tag}] step={step:>6} loss={loss.item():.5f} "
                  f"train_exact={tr:.3f} held_exact={he:.3f} "
                  f"train_ce={tr_ce:.3f} held_ce={he_ce:.3f}"
                  f"{extra(xin, xt)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
