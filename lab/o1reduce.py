"""Modular reduction at O(1) learned-op depth (RANKING.md method #3).

`plan2/matrix-scan` §7 measured why `MonoidALU` bottoms out at learned-op depth
12: long division emits quotient digits MSB-first and the state carried between
them is the *partial remainder*, an S-digit number.  That is `2S` learned-table
applications, and it is the majority of the graph and the whole of its growth
in `S`.  This module removes it.

THE CONSTRUCTION.  Barrett reduction (HAC 14.42) replaces the S serial
conditional subtractions with two multiplications and one subtraction:

    mu = floor(10^{2S} / N)            (depends on N only)
    q1 = floor(p  / 10^{S-1})
    q2 = q1 * mu
    q3 = floor(q2 / 10^{S+1})
    r  = p - q3*N                       and  0 <= r < 3N   (HAC 14.42)

Every truncation is a slice, so the *x*-dependent reduction is a fixed number of
learned operations no matter how many digits N has.

THE PRIMITIVE THAT MAKES A MULTIPLY 2 OPS.  `MonoidALU` paid
`1 + ceil(log2 2S)` for a multiply: one product-table read plus a balanced tree
of pairwise adds.  Here the partial products of a whole column are pooled into a
**bag of digit-pair indicators** -- a plain sum of one-hots, no arithmetic
assumed -- and a single learned table maps the bag to a distribution over that
column's total (`ColSum`).  A second learned table resolves carries across
columns as a linear monoid (`CarryMonoid`).  So

    multiply-accumulate  =  2 learned-table applications, constant in S

and subtraction is free inside the same bag: a contribution simply carries a
negative role.  A `ColSum` can therefore compute `p - (q3+c)*N` for several `c`
at once, and the carry monoid's *final state* is the sign, so the Barrett
correction costs one learned selector rather than a compare-and-subtract loop.

RESULTING LEARNED-OP DEPTH (`model.op_depth`, and constant in S):

    square      ColSum + CarryMonoid                     2
    quotient    ColSum + CarryMonoid                     2
    residual    ColSum + CarryMonoid + selector          3
                                                      -----
                                                         7

against `MonoidALU`'s `2 + ceil(log2 2S) + 2S` (12 at S=3, 21 at S=7) and
`DigitALU tree:quotient`'s 39.  The reduction alone goes from `2S` to **5**.

WHAT IS *NOT* O(1), AND THIS IS A RESULT NOT AN OVERSIGHT.  `mu` is a division.
`--recip div` computes it by learned long division with **private** parameters,
which is exactly constructible and is what certifies the class ceiling;
`--recip head` is a shallow learned map from N's digits (depth 1, legal, not
exactly constructible); `--recip oracle` supplies the true `mu` and is a
**DIAGNOSTIC** used to measure the reduction's repair basin under maximally
favourable conditioning.  See `lab/reports/hard-o1-reduction.md` §7.

COMPLIANCE.  Nothing here reads `data/generated/`.  Every tensor that decides an
output digit is a learned parameter; the only fixed operations are re-indexing
(a shift is a slice), pooling one-hot indicators into a bag, softmax, and
matmul.  `construct_()` / `corrupt_()` / `--recip oracle` write or supply the
truth and are **LAB DIAGNOSTICS ONLY** (BRIEF §4 rule 2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from monoid import (  # noqa: F401  (re-exported for probes)
    PREFIX_IMPLS,
    digits_to_int,
    prefix_scan,
    prefix_serial,
    snap,
    zero_digits,
)


def int_to_digits(values, L: int, device, dtype=torch.float32) -> Tensor:
    """One-hot digit encoding, LSB first.  Probe input only.

    `monoid.int_to_digits` writes one GPU element per digit inside a Python
    loop; on a shared GPU that dominated the constructed-ceiling gate (144 s for
    what the model itself does in 0.5 s).  Build on the host, transfer once.
    """
    idx = torch.empty(len(values), L, dtype=torch.long)
    for b, v in enumerate(values):
        v = int(v)
        for k in range(L):
            idx[b, k] = (v // 10**k) % 10
    return F.one_hot(idx, 10).to(device=device, dtype=dtype)

BIG = 12.0      # logit magnitude for constructed one-hot tables
LAM = 16.0      # curvature of the constructed ColSum quadratic (see below)


# --------------------------------------------------------------------------- #
# 1. column pooling helpers (fixed re-indexing; no arithmetic)                 #
# --------------------------------------------------------------------------- #


_PLAN_CACHE: dict = {}


def pair_plan(La: int, Lb: int, Lout: int, shift: int, device) -> tuple[Tensor, Tensor]:
    """Fixed binary maps sending digit-pair (i, j) to its output columns.

    Returns ``(A_lo, A_hi)`` of shape ``(La*Lb, Lout)``: the low half of the
    product of ``a_i`` and ``b_j`` lands in column ``i + j + shift``, the high
    half one column above.  Pure bookkeeping -- the *values* of the halves are
    what `ColSum` learns.

    Built on the host and cached: the boolean-mask index assignment forces a
    device sync, and rebuilding these per forward cost 0.68 s/step on a shared
    GPU (vs ~0.02 s once cached).
    """
    key = ("p", La, Lb, Lout, shift, str(device))
    if key not in _PLAN_CACHE:
        i = torch.arange(La).view(La, 1).expand(La, Lb).reshape(-1)
        j = torch.arange(Lb).view(1, Lb).expand(La, Lb).reshape(-1)
        lo, hi = torch.zeros(La * Lb, Lout), torch.zeros(La * Lb, Lout)
        cl, ch = i + j + shift, i + j + shift + 1
        rows = torch.arange(La * Lb)
        ok_l, ok_h = cl < Lout, ch < Lout
        lo[rows[ok_l], cl[ok_l]] = 1.0
        hi[rows[ok_h], ch[ok_h]] = 1.0
        _PLAN_CACHE[key] = (lo.to(device), hi.to(device))
    return _PLAN_CACHE[key]


def digit_plan(Ld: int, Lout: int, shift: int, device) -> Tensor:
    """Fixed binary map sending digit ``l`` of an addend to column ``l+shift``."""
    key = ("d", Ld, Lout, shift, str(device))
    if key not in _PLAN_CACHE:
        m = torch.zeros(Ld, Lout)
        c = torch.arange(Ld) + shift
        ok = c < Lout
        m[torch.arange(Ld)[ok], c[ok]] = 1.0
        _PLAN_CACHE[key] = m.to(device)
    return _PLAN_CACHE[key]


def pair_bag(a: Tensor, b: Tensor, A_lo: Tensor, A_hi: Tensor) -> Tensor:
    """Bag of digit-pair indicators per output column.

    ``a`` is ``(..., La, 10)`` and ``b`` is ``(..., Lb, 10)``; the result is
    ``(..., Lout, 2, 100)`` where index 0 is the low-half role and 1 the high.
    This is a *sum of one-hot indicators*, i.e. counting which digit pairs occur
    in a column -- it assumes nothing about what a digit means.
    """
    outer = (a.unsqueeze(-2).unsqueeze(-1) * b.unsqueeze(-3).unsqueeze(-2))
    outer = outer.flatten(-2).flatten(-3, -2)  # (..., La*Lb, 100)
    lo = torch.einsum("...pc,pl->...lc", outer, A_lo)
    hi = torch.einsum("...pc,pl->...lc", outer, A_hi)
    return torch.stack([lo, hi], dim=-2)


def digit_bag(d: Tensor, plan: Tensor) -> Tensor:
    """``(..., Ld, 10)`` addend digits -> ``(..., Lout, 10)`` bag."""
    return torch.einsum("...pc,pl->...lc", d, plan)


# --------------------------------------------------------------------------- #
# 2. the two learned primitives                                               #
# --------------------------------------------------------------------------- #


class ColSum(nn.Module):
    """Learned map: bag of contributions in a column -> the column's total.

    ONE learned-table application.  ``Wp[r]`` is a 100 x M table indexed by a
    digit *pair* (role ``r`` selects the low/high half and the sign), ``Wd[r]``
    a 10 x M table indexed by a single addend digit.  Because the bag is formed
    by summation *before* the table, the whole multi-row accumulation costs one
    application regardless of how many rows or columns there are -- this is what
    replaces `MonoidALU`'s ``ceil(log2 2S)``-deep tree of pairwise adds.

    The exact solution is inside the class: with the column total
    ``t = sum of contributions``, the score ``LAM*(m*t) - LAM*m^2/2`` equals
    ``-LAM/2 (m-t)^2`` up to a term constant in ``m``, so its softmax peaks at
    ``m = t``; both pieces are linear in the bag plus a bias.
    """

    def __init__(self, m_lo: int, m_hi: int, n_pair: int, n_dig: int,
                 init_scale: float = 0.5):
        super().__init__()
        self.m_lo, self.m_hi = m_lo, m_hi
        self.M = m_hi - m_lo + 1
        self.Wp = nn.Parameter(torch.randn(max(n_pair, 1), 100, self.M) * init_scale)
        self.Wd = nn.Parameter(torch.randn(max(n_dig, 1), 10, self.M) * init_scale)
        self.b = nn.Parameter(torch.randn(self.M) * init_scale)
        self.n_pair, self.n_dig = n_pair, n_dig

    def forward(self, pbags: Tensor | None, dbags: Tensor | None) -> Tensor:
        """``pbags`` ``(..., L, n_pair, 100)``, ``dbags`` ``(..., L, n_dig, 10)``."""
        # fp32 throughout: the constructed quadratic's logits reach ~1e6 while the
        # margin between adjacent totals is LAM/2, which bf16 cannot resolve.
        # (PLAN2 §5 makes the same demand of the prefix products.)
        with torch.autocast(device_type="cuda", enabled=False):
            out = self.b.float()
            if pbags is not None and self.n_pair:
                out = out + torch.einsum("...lrc,rcm->...lm", pbags.float(), self.Wp.float())
            if dbags is not None and self.n_dig:
                out = out + torch.einsum("...lrc,rcm->...lm", dbags.float(), self.Wd.float())
        return out

    @torch.no_grad()
    def construct_(self, pair_vals: list, dig_vals: list) -> None:
        """``pair_vals[r]`` is a length-100 int list, ``dig_vals[r]`` length 10."""
        m = torch.arange(self.m_lo, self.m_hi + 1, device=self.b.device, dtype=torch.float32)
        self.b.copy_(-LAM * m * m / 2.0)
        self.Wp.zero_()
        self.Wd.zero_()
        for r, vals in enumerate(pair_vals):
            v = torch.tensor(vals, device=self.b.device, dtype=torch.float32)
            self.Wp[r].copy_(LAM * v[:, None] * m[None, :])
        for r, vals in enumerate(dig_vals):
            v = torch.tensor(vals, device=self.b.device, dtype=torch.float32)
            self.Wd[r].copy_(LAM * v[:, None] * m[None, :])


class CarryMonoid(nn.Module):
    """Learned map: column totals -> output digits, with carries as a monoid.

    ONE learned-table application.  ``trans`` is read once per column *in
    parallel* and the states are obtained by a prefix product of those matrices
    -- the composition is fixed matmul, not a learned op, so the nonlinearity is
    paid once no matter how long the digit string is (the accounting
    `plan2/matrix-scan` used for `PairMonoid`, §2 of its report).
    """

    def __init__(self, M: int, c_lo: int, c_hi: int, init_scale: float = 0.5,
                 impl: str = "serial"):
        super().__init__()
        self.M, self.c_lo, self.c_hi = M, c_lo, c_hi
        self.C = c_hi - c_lo + 1
        self.impl = impl
        self.trans = nn.Parameter(torch.randn(M, self.C * self.C) * init_scale)
        self.emit = nn.Parameter(torch.randn(M, self.C * 10) * init_scale)

    record_usage = False   # when True, accumulate how much input mass each row sees

    def forward(self, tot: Tensor, *, hard: bool = False, ste: bool = False):
        """``tot`` ``(B, L, M)`` column-total distribution -> (digits, final)."""
        if self.record_usage:
            u = tot.detach().float().reshape(-1, tot.shape[-1]).sum(0)
            self.usage = u if not hasattr(self, "usage") else self.usage + u
        shape = tot.shape[:-2]
        L, M = tot.shape[-2], tot.shape[-1]
        t = tot.reshape(-1, L, M).float()
        B = t.shape[0]
        with torch.autocast(device_type="cuda", enabled=False):
            logits = (t @ self.trans.float()).view(B, L, self.C, self.C)
            Mm = logits.softmax(dim=-2)  # column-stochastic: a function on carries
            if hard:
                Mm = snap(Mm, -2, ste=ste)
            P = PREFIX_IMPLS[self.impl](Mm)
            c0 = torch.zeros(B, self.C, 1, device=t.device, dtype=t.dtype)
            c0[:, -self.c_lo, 0] = 1.0
            states = torch.cat(
                [c0.unsqueeze(1), P[:, :-1] @ c0.unsqueeze(1)], dim=1
            ).squeeze(-1)  # (B, L, C)
            if hard:
                states = snap(states, -1, ste=ste)
            el = (t @ self.emit.float()).view(B, L, self.C, 10)
            out = torch.einsum("blc,blcd->bld", states, el).softmax(-1)
            if hard:
                out = snap(out, -1, ste=ste)
            final = (P[:, -1] @ c0).squeeze(-1)
            if hard:
                final = snap(final, -1, ste=ste)
        return out.reshape(*shape, L, 10), final.reshape(*shape, self.C)

    @torch.no_grad()
    def construct_(self) -> None:
        dev = self.trans.device
        self.trans.fill_(-BIG)
        self.emit.fill_(-BIG)
        tv = torch.arange(self.M, device=dev) + self._t_lo
        cv = torch.arange(self.C, device=dev) + self.c_lo
        v = tv[:, None] + cv[None, :]                       # (M, C) running value
        cout = torch.div(v, 10, rounding_mode="floor").clamp(self.c_lo, self.c_hi)
        dig = v - 10 * torch.div(v, 10, rounding_mode="floor")
        mi = torch.arange(self.M, device=dev)[:, None].expand(self.M, self.C)
        ci = torch.arange(self.C, device=dev)[None, :].expand(self.M, self.C)
        self.trans.view(self.M, self.C, self.C)[mi, cout - self.c_lo, ci] = BIG
        self.emit.view(self.M, self.C, 10)[mi, ci, dig] = BIG

    _t_lo = 0  # set by the owner: the value the first total index represents


# --------------------------------------------------------------------------- #
# 3. a multiply-accumulate stage = ColSum + CarryMonoid                       #
# --------------------------------------------------------------------------- #


class AccStage(nn.Module):
    """One accumulate-and-normalise stage: **2 learned-op depth**, any S."""

    def __init__(self, m_lo: int, m_hi: int, n_pair: int, n_dig: int,
                 init_scale: float = 0.5, impl: str = "serial"):
        super().__init__()
        c_hi = max(0, (m_hi + 8) // 9)
        c_lo = min(0, -((-m_lo + 8) // 9))
        self.cols = ColSum(m_lo, m_hi, n_pair, n_dig, init_scale)
        self.carry = CarryMonoid(self.cols.M, c_lo, c_hi, init_scale, impl)
        self.carry._t_lo = m_lo

    def forward(self, pbags, dbags, *, hard=False, ste=False):
        tot = self.cols(pbags, dbags).softmax(-1)
        if hard:
            tot = snap(tot, -1, ste=ste)
        return self.carry(tot, hard=hard, ste=ste)

    @torch.no_grad()
    def construct_(self, pair_vals, dig_vals) -> None:
        self.cols.construct_(pair_vals, dig_vals)
        self.carry.construct_()


# value tables for the constructed solution -------------------------------- #
_PAIR_LO = [(a * b) % 10 for a in range(10) for b in range(10)]
_PAIR_HI = [(a * b) // 10 for a in range(10) for b in range(10)]
_DIG = list(range(10))


# --------------------------------------------------------------------------- #
# 4. reciprocal heads                                                          #
# --------------------------------------------------------------------------- #


class RecipHead(nn.Module):
    """LEGAL, learned-op depth 1(-2): a shallow map from N's digits to mu.

    Not exactly constructible -- see the module docstring and report §7.
    """

    def __init__(self, S: int, Lmu: int, hidden: int = 256, init_scale: float = 0.5):
        super().__init__()
        self.S, self.Lmu = S, Lmu
        self.w1 = nn.Parameter(torch.randn(S * 10, hidden) * (1.0 / (S * 10) ** 0.5))
        self.w2 = nn.Parameter(torch.randn(hidden, Lmu * 10) * (1.0 / hidden ** 0.5))

    @property
    def op_depth(self) -> int:
        return 2

    def forward(self, n: Tensor, *, hard: bool = False, ste: bool = False) -> Tensor:
        h = torch.relu(n[:, : self.S].flatten(-2) @ self.w1)
        out = (h @ self.w2).view(-1, self.Lmu, 10).softmax(-1)
        return snap(out, -1, ste=ste) if hard else out


class LongDivRecip(nn.Module):
    """LEGAL and exactly constructible: mu = floor(10^{2S}/N) by long division.

    Private parameters, and the path is a function of N alone.  Learned-op depth
    ``3*(S+1)`` -- deliberately NOT O(1); §7 of the report argues it cannot be.
    Each step subtracts every candidate ``d*N*10^k`` in parallel inside one
    `AccStage` and picks the largest that stays non-negative.
    """

    def __init__(self, S: int, Lmu: int, init_scale: float = 0.5, impl: str = "serial"):
        super().__init__()
        self.S, self.Lmu = S, Lmu
        # room for d*N*10^k at the highest quotient position
        self.Lr = Lmu + S + 1
        # contributions: pair roles (lo-, hi-) from d*N ; digit role (+) from R.
        # d is a single digit, so a column receives at most one low and one high
        # half: the total lives in [-17, 9].
        self.step = AccStage(-18, 9, 2, 1, init_scale, impl)
        self.sel = nn.Parameter(torch.randn(10, self.step.carry.C) * init_scale)

    @property
    def op_depth(self) -> int:
        return 3 * self.Lmu

    def forward(self, n: Tensor, *, hard: bool = False, ste: bool = False) -> Tensor:
        B, dev, dt = n.shape[0], n.device, n.dtype
        Lr = self.Lr
        r = zero_digits(B, Lr, dev, dt).clone()
        r[:, 2 * self.S, 0] = 0.0
        r[:, 2 * self.S, 1] = 1.0  # R = 10^{2S}
        dplan = digit_plan(Lr, Lr, 0, dev)
        eye = torch.eye(10, device=dev, dtype=dt).view(1, 10, 1, 10).expand(B, 10, 1, 10)
        digs = []
        for k in range(self.Lmu - 1, -1, -1):
            A_lo, A_hi = pair_plan(1, self.S, Lr, k, dev)
            pb = pair_bag(eye, n[:, : self.S].unsqueeze(1).expand(B, 10, self.S, 10),
                          A_lo, A_hi)                       # (B, 10, Lr, 2, 100)
            db = digit_bag(r, dplan).unsqueeze(1).unsqueeze(-2).expand(B, 10, Lr, 1, 10)
            cand, final = self.step(pb, db, hard=hard, ste=ste)   # (B,10,Lr,10), (B,10,C)
            logits = torch.einsum("bqc,qc->bq", final, self.sel)
            s = logits.softmax(-1)
            if hard:
                s = snap(s, -1, ste=ste)
            digs.append(s)
            r = torch.einsum("bq,bqld->bld", s, cand)
        return torch.stack(digs[::-1], dim=1)                # LSB first: k=0..Lmu-1

    @torch.no_grad()
    def construct_(self) -> None:
        self.step.construct_([[-v for v in _PAIR_LO], [-v for v in _PAIR_HI]], [_DIG])
        C, c_lo = self.step.carry.C, self.step.carry.c_lo
        self.sel.fill_(-BIG)
        for q in range(10):
            self.sel[q, -c_lo] = BIG * (1.0 + q)   # non-negative -> prefer the largest q


# --------------------------------------------------------------------------- #
# 5. the ALU                                                                  #
# --------------------------------------------------------------------------- #

RECIPS = ("oracle", "head", "div")


class O1ReduceALU(nn.Module):
    """``x^2 mod N`` with the modular reduction at O(1) learned-op depth."""

    def __init__(self, slots: int, recip: str = "div", init_scale: float = 0.5,
                 impl: str = "serial", hidden: int = 256, n_corr: int = 2):
        super().__init__()
        S = self.S = slots
        self.L = 2 * S                     # digit length of x, N (padded) and p
        # mu = floor(10^{2S}/N).  N may have fewer than S significant digits
        # (hf1 mixes 16/18/20-bit moduli), so mu is given the full 2S width --
        # otherwise the architecture would only be exact at one modulus size.
        self.Lmu = 2 * S
        self.Lq2 = 4 * S
        self.Lq3 = S + 1                   # q = floor(p*mu/10^{2S}) < N < 10^S
        self.Lr = 2 * S + 2
        self.n_corr = n_corr               # exact mu gives r in [0, 2N)
        self.recip_kind = recip
        self.impl = impl

        # -- stage A: p = x*x  (pairs of x digits, both halves positive)
        self.sq = AccStage(0, 17 * S, 2, 0, init_scale, impl)
        # -- stage B: q2 = p*mu
        self.qm = AccStage(0, 34 * S, 2, 0, init_scale, impl)
        # -- stage C: d_c = p - (q3+c)*N   (pairs negative, p positive, N negative)
        self.rs = AccStage(-17 * S - 9 * (n_corr - 1), 9, 2, 2, init_scale, impl)
        self.sel = nn.Parameter(torch.randn(n_corr, self.rs.carry.C) * init_scale)

        if recip == "head":
            self.recip = RecipHead(S, self.Lmu, hidden, init_scale)
        elif recip == "div":
            self.recip = LongDivRecip(S, self.Lmu, init_scale, impl)
        else:
            self.recip = None              # DIAGNOSTIC: supplied externally
        self.ste = False
        self._mu_oracle: Tensor | None = None

    # -- structural bookkeeping -------------------------------------------- #
    @property
    def reduce_op_depth(self) -> int:
        """Learned ops between the loss and a table, on the x-dependent
        reduction path only (quotient, residual, correction select)."""
        return 2 + 2 + 1

    @property
    def op_depth(self) -> int:
        """Longest learned-op chain from the loss to any parameter."""
        return 2 + self.reduce_op_depth

    @property
    def recip_op_depth(self) -> int:
        return 0 if self.recip is None else self.recip.op_depth

    @property
    def full_op_depth(self) -> int:
        """Including the N-only reciprocal branch (which is not O(1))."""
        return max(self.op_depth, self.recip_op_depth + self.reduce_op_depth)

    def table_depth(self) -> dict[str, int]:
        """Learned-op depth from the loss to each named table.

        This is the instrument for the depth-stratified repair basin: the tables
        are private per stage, so one corrupt-and-repair run reports the
        objective's exact-repair rate at five different depths at once.
        """
        d = {"rs_carry": 1, "rs_cols": 2, "sel": 1,
             "qm_carry": 3, "qm_cols": 4,
             "sq_carry": 5, "sq_cols": 6}
        if self.recip_kind == "div":
            d["recip_step_carry"] = 3 + self.recip.op_depth - 2
            d["recip_step_cols"] = 3 + self.recip.op_depth - 1
            d["recip_sel"] = 3 + self.recip.op_depth - 3
        return d

    # -- pieces ------------------------------------------------------------- #
    def square(self, x: Tensor, *, hard=False) -> Tensor:
        B, dev = x.shape[0], x.device
        A_lo, A_hi = pair_plan(self.S, self.S, self.L, 0, dev)
        pb = pair_bag(x[:, : self.S], x[:, : self.S], A_lo, A_hi)
        p, _ = self.sq(pb, None, hard=hard, ste=self.ste)
        return p

    def quotient(self, p: Tensor, mu: Tensor, *, hard=False) -> Tensor:
        """q = floor(p*mu / 10^{2S}), which is floor(p/N) or one less."""
        dev = p.device
        A_lo, A_hi = pair_plan(self.L, self.Lmu, self.Lq2, 0, dev)
        pb = pair_bag(p, mu, A_lo, A_hi)
        q2, _ = self.qm(pb, None, hard=hard, ste=self.ste)
        return q2[:, self.L:][:, : self.Lq3]                # a slice, not arithmetic

    def residual(self, p: Tensor, q3: Tensor, n: Tensor, *, hard=False) -> Tensor:
        """d_c = p - (q3 + c)*N for c = 0..n_corr-1; select the largest c >= 0."""
        B, dev = p.shape[0], p.device
        Lr, K = self.Lr, self.n_corr
        A_lo, A_hi = pair_plan(self.Lq3, self.S, Lr, 0, dev)
        pb = pair_bag(q3, n[:, : self.S], A_lo, A_hi)       # (B, Lr, 2, 100)
        pb = pb.unsqueeze(1).expand(B, K, Lr, 2, 100)
        dp = digit_bag(p, digit_plan(self.L, Lr, 0, dev))   # (B, Lr, 10)
        dn = digit_bag(n[:, : self.S], digit_plan(self.S, Lr, 0, dev))
        w = torch.arange(K, device=dev, dtype=dn.dtype).view(1, K, 1, 1)
        db = torch.stack(
            [dp.unsqueeze(1).expand(B, K, Lr, 10), dn.unsqueeze(1) * w], dim=-2
        )                                                   # (B, K, Lr, 2, 10)
        cand, final = self.rs(pb, db, hard=hard, ste=self.ste)
        logits = torch.einsum("bkc,kc->bk", final, self.sel)
        s = logits.softmax(-1)
        if hard:
            s = snap(s, -1, ste=self.ste)
        return torch.einsum("bk,bkld->bld", s, cand)

    def mu_of(self, n: Tensor, *, hard=False) -> Tensor:
        if self.recip is None:
            assert self._mu_oracle is not None, "--recip oracle needs set_mu()"
            return self._mu_oracle
        return self.recip(n, hard=hard, ste=self.ste)

    def set_mu(self, mu: Tensor) -> None:
        """DIAGNOSTIC: supply the true reciprocal (``--recip oracle``)."""
        self._mu_oracle = mu

    def forward(self, x: Tensor, n: Tensor, *, hard: bool = False) -> Tensor:
        p = self.square(x, hard=hard)
        mu = self.mu_of(n, hard=hard)
        q3 = self.quotient(p, mu, hard=hard)
        r = self.residual(p, q3, n, hard=hard)
        out = r[:, : self.L]
        if out.shape[1] < self.L:
            out = torch.cat(
                [out, zero_digits(x.shape[0], self.L - out.shape[1], x.device, out.dtype)], 1
            )
        return out

    # -- DIAGNOSTIC ORACLE (never legal in a submission) -------------------- #
    @torch.no_grad()
    def construct_(self) -> None:
        self.sq.construct_([_PAIR_LO, _PAIR_HI], [])
        self.qm.construct_([_PAIR_LO, _PAIR_HI], [])
        self.rs.construct_([[-v for v in _PAIR_LO], [-v for v in _PAIR_HI]],
                           [_DIG, [-v for v in _DIG]])
        c_lo = self.rs.carry.c_lo
        self.sel.fill_(-BIG)
        for c in range(self.n_corr):
            self.sel[c, -c_lo] = BIG * (1.0 + c)
        if self.recip_kind == "div":
            self.recip.construct_()

    # -- table bookkeeping -------------------------------------------------- #
    def _cells(self):
        """(name, parameter, view shape, argmax dim, bias) for every learned table.

        A *cell* is one row of a table -- one digit pair, one digit, one column
        total -- the unit `alu-relational` (400 cells) and `plan2/matrix-scan`
        (700 cells) used, so the repair counts are directly comparable.

        `bias` matters for `ColSum`: a row encodes a *value*, and the value it
        votes for is ``argmax_m (W[row, m] + b[m])`` -- reading `argmax W` alone
        would decode every positive row to the top of the range.  The bias is
        the stage's own learned `b`, so the criterion stays functional as
        training moves it.
        """
        out = []

        def add(prefix, stage, n_pair, n_dig):
            M, C, b = stage.cols.M, stage.carry.C, stage.cols.b
            if n_pair:
                out.append((f"{prefix}_cols_p", stage.cols.Wp, (n_pair * 100, M), -1, b))
            if n_dig:
                out.append((f"{prefix}_cols_d", stage.cols.Wd, (n_dig * 10, M), -1, b))
            out.append((f"{prefix}_carry_t", stage.carry.trans, (M, C, C), -2, None))
            out.append((f"{prefix}_carry_e", stage.carry.emit, (M, C, 10), -1, None))

        add("sq", self.sq, 2, 0)
        add("qm", self.qm, 2, 0)
        add("rs", self.rs, 2, 2)
        out.append(("sel", self.sel, (self.n_corr, self.rs.carry.C), -1, None))
        if self.recip_kind == "div":
            add("rc", self.recip.step, 2, 1)
            out.append(("rc_sel", self.recip.sel, (10, self.recip.step.carry.C), -1, None))
        return out

    @torch.no_grad()
    def _decode(self, ref=None) -> dict[str, Tensor]:
        """Per-cell decoded answer, for this model and (optionally) a reference."""
        src = ref or self
        out = {}
        for name, p, shape, dim, b in src._cells():
            v = p.view(*shape)
            if b is not None:
                v = v + b.view(*([1] * (v.ndim - 1)), -1)
            out[name] = v.argmax(dim)
        return out

    def _cell_depth(self) -> dict[str, int]:
        d = {"rs_cols_p": 2, "rs_cols_d": 2, "rs_carry_t": 1, "rs_carry_e": 1, "sel": 1,
             "qm_cols_p": 4, "qm_cols_d": 4, "qm_carry_t": 3, "qm_carry_e": 3,
             "sq_cols_p": 6, "sq_cols_d": 6, "sq_carry_t": 5, "sq_carry_e": 5}
        if self.recip_kind == "div":
            base = 5 + self.recip.op_depth
            d.update({"rc_cols_p": base, "rc_cols_d": base,
                      "rc_carry_t": base - 1, "rc_carry_e": base - 1,
                      "rc_sel": base - 2})
        return d

    @torch.no_grad()
    def _reference(self) -> "O1ReduceALU":
        ref = O1ReduceALU(self.S, self.recip_kind, impl=self.impl,
                          n_corr=self.n_corr).to(self.sel.device)
        ref.construct_()
        return ref

    # Tables whose constructed logits are +/-BIG, i.e. on exactly the scale
    # `alu-relational` (400 cells) and `plan2/matrix-scan` (700 cells) measured.
    # `ColSum` rows are NOT: the constructed quadratic reaches ~3e4, so a
    # corrupted row sits ~1e5 away from the truth and no learning rate that
    # trains the rest of the model can walk back to it in 2,000 steps.  Mixing
    # the two would measure parameter scale, not conditioning -- so the basin is
    # reported on this set and the ColSum rows are reported separately.
    LOGIT_TABLES = ("carry_t", "carry_e", "sel")

    # `hard/add-only` measured that SEPARABILITY, not size, is what the legal
    # label can exploit: a 10-cell separable selector is recovered exactly every
    # seed, while an arithmetic table's repair radius is 5-10 cells.  These two
    # sets isolate that axis at matched learned-op depth 1: `sel` is separable
    # (each row's answer is fixed by the label independently of the others),
    # `rs_carry_*` is not (its rows must agree across every carry state and they
    # interact through the carry chain).
    SEL_TABLES = ("sel",)
    D1_ARITH_TABLES = ("rs_carry_t", "rs_carry_e")

    def _cells_logit(self):
        return [c for c in self._cells() if c[0].endswith(self.LOGIT_TABLES)]

    def _cells_named(self, which: str):
        if which == "sel":
            return [c for c in self._cells() if c[0].endswith(self.SEL_TABLES)]
        if which == "d1arith":
            return [c for c in self._cells() if c[0] in self.D1_ARITH_TABLES]
        return self._cells() if which == "all" else self._cells_logit()

    @torch.no_grad()
    def corrupt_(self, k: int, generator=None, scale: float = 0.5,
                 mode: str = "uniform", tables: str = "logit") -> dict[str, Tensor]:
        """Randomise k cells (`uniform`) or k cells of *every* table (`per_table`).

        A corrupted row is replaced by noise at `scale` x the RMS of the
        constructed row, so the randomisation is on the table's own scale (this
        matters here because `ColSum` logits run to ~1e4 while the carry tables
        are +/-12; a fixed absolute noise level would mean two different
        experiments).  `per_table` is what makes the depth-stratified read
        balanced -- the tables differ in size by 10x.
        """
        cells = [(n, p, s) for n, p, s, _, _ in self._cells_named(tables)]
        hit = {}
        if mode == "uniform":
            total = sum(s[0] for _, _, s in cells)
            k = min(k, total)
            pick = torch.randperm(total, generator=generator)[:k]
            off = 0
            for name, p, shape in cells:
                n = shape[0]
                sel = pick[(pick >= off) & (pick < off + n)] - off
                if sel.numel():
                    hit[name] = sel
                off += n
        else:
            for name, p, shape in cells:
                n = shape[0]
                hit[name] = torch.randperm(n, generator=generator)[: min(k, n)]
        for name, p, shape in cells:
            if name not in hit:
                continue
            sel = hit[name].to(p.device)
            view = p.data.view(*shape)
            rms = view[sel].pow(2).mean().sqrt().clamp_min(1e-6)
            view[sel] = torch.randn(
                sel.numel(), *shape[1:], generator=generator
            ).to(p.device) * (scale * rms)
        return hit

    @torch.no_grad()
    def usage(self, x, n, mu=None, chunk: int = 256) -> dict[str, Tensor]:
        """Input mass each corruptible table row receives over a dataset.

        A cell that no example exercises receives no gradient and cannot be
        repaired whatever the conditioning is -- so the basin has to be read
        against this, not in isolation.  Row indices match `_cells()`.
        """
        mons = {"sq": self.sq.carry, "qm": self.qm.carry, "rs": self.rs.carry}
        if self.recip_kind == "div":
            mons["rc"] = self.recip.step.carry
        for m in mons.values():
            m.record_usage = True
            if hasattr(m, "usage"):
                del m.usage
        for i in range(0, x.shape[0], chunk):
            if mu is not None:
                self.set_mu(mu[i:i + chunk])
            self(x[i:i + chunk], n[i:i + chunk])
        out = {}
        for name, m in mons.items():
            m.record_usage = False
            u = m.usage / m.usage.sum().clamp_min(1e-9)
            out[f"{name}_carry_t"] = u
            out[f"{name}_carry_e"] = u
        out["sel"] = torch.ones(self.n_corr, device=x.device) / self.n_corr
        if self.recip_kind == "div":
            out["rc_sel"] = torch.ones(10, device=x.device) / 10
        return out

    @torch.no_grad()
    def cell_correct(self, ref=None) -> dict[str, Tensor]:
        ref = ref or self._reference()
        got, want = self._decode(), self._decode(ref)
        out = {}
        for name in got:
            m = got[name] == want[name]
            out[name] = m.flatten(1).all(-1) if m.ndim > 1 else m
        return out

    @torch.no_grad()
    def repaired(self, hit, ref=None, pre=None, live=None):
        """Cells that were WRONG right after corruption and are right now.

        Counting against the post-corruption state rather than against the set
        of touched cells is necessary here: a randomised `ColSum` row decodes to
        value 0, and ~half of the true `hi` rows *are* 0, so the naive count
        would report a large repair rate for cells the objective never touched.
        """
        now = self.cell_correct(ref)
        depth = self._cell_depth()
        per_depth: dict[int, list[int]] = {}
        ok = tot = 0
        for name, idx in hit.items():
            idx = idx.to(now[name].device)
            was_wrong = ~pre[name][idx] if pre is not None else torch.ones_like(now[name][idx])
            if live is not None and name in live:
                # a cell the training set never exercises gets no gradient, so it
                # cannot be repaired at any conditioning; excluding it gives the
                # honest denominator
                was_wrong = was_wrong & (live[name][idx] > 1e-6)
            fixed = now[name][idx] & was_wrong
            ok += int(fixed.sum())
            tot += int(was_wrong.sum())
            e = per_depth.setdefault(depth[name], [0, 0])
            e[0] += int(fixed.sum())
            e[1] += int(was_wrong.sum())
        return ok, tot, {k: tuple(v) for k, v in sorted(per_depth.items())}

    @torch.no_grad()
    def table_correct(self, ref=None) -> dict[str, float]:
        cc = self.cell_correct(ref)
        out, correct, total = {}, 0, 0
        for name, per_cell in cc.items():
            out[name] = per_cell.float().mean().item()
            correct += int(per_cell.sum())
            total += per_cell.numel()
        out["_all"] = correct / total
        out["_n_cells"] = total
        return out


def true_mu(values, S: int, Lmu: int, device, dtype=torch.float32) -> Tensor:
    """DIAGNOSTIC helper for ``--recip oracle``: digits of floor(10^{2S}/N)."""
    return int_to_digits([(10 ** (2 * S)) // int(v) for v in values], Lmu, device, dtype)


def reference_mod(x: int, n: int, S: int) -> int:
    """Barrett as this architecture evaluates it -- used by the unit test."""
    mu = (10 ** (2 * S)) // n
    p = x * x
    q = (p * mu) // (10 ** (2 * S))
    r = p - q * n
    return r
