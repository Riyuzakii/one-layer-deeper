"""PLAN2 §3.1 — learned monoid + matrix associative scan, over DIGIT POSITIONS.

BRIEF2 §2(c) corrects PLAN2's framing: the sequence axis is not the composition
axis (max_seq_len is 13-21 and T is not a sequence dimension), but **carry
propagation across digit positions is an associative prefix computation** — the
classic propagate/generate carry monoid.  So the scan lives on a digit-position
axis of length ~6-10, not on the prompt.

This module provides

  * three interchangeable prefix-product implementations (serial reference,
    log-depth Hillis-Steele doubling, and torch's associative_scan HOP) which
    are asserted equal in ``lab/test_monoid.py``;
  * constraint families for the transition matrices (dense / column-stochastic /
    straight-through one-hot / doubly-stochastic / orthogonal);
  * ``MonoidALU`` — a modulus-independent digit transducer for ``x^2 mod N`` in
    which every learned tensor is indexed by a digit tuple and the serial state
    updates are replaced by per-position table applications plus associative
    scans.

COMPLIANCE.  Nothing here reads ``data/generated/``.  ``MonoidALU.construct_()``
writes the true digit tables and is a **DIAGNOSTIC ORACLE** (BRIEF §4 rule 2) —
it may never appear in a submission; it exists to measure the class ceiling and
the repair basin.  A submission uses the same *structure* with every tensor
learned from random init.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# --------------------------------------------------------------------------- #
# 1. prefix products                                                          #
# --------------------------------------------------------------------------- #
#
# Convention, fixed once and asserted in the tests:
#
#     P[k] = M[k] @ M[k-1] @ ... @ M[0]          (INCLUSIVE prefix)
#
# so that with a column-vector state and c_{k+1} = M[k] c_k we get
# c_{k+1} = P[k] c_0.  PLAN2 §5 warns that left/right order and off-by-one
# errors here are silent; the serial function below is the reference.


def prefix_serial(M: Tensor) -> Tensor:
    """Reference implementation: K-1 sequential matmuls.  M: (B, K, d, d)."""
    outs = [M[:, 0]]
    for k in range(1, M.shape[1]):
        outs.append(M[:, k] @ outs[-1])
    return torch.stack(outs, dim=1)


def prefix_scan(M: Tensor) -> Tensor:
    """Log-depth doubling (Hillis-Steele).  ceil(log2 K) sequential matmuls."""
    B, K, d, _ = M.shape
    A = M
    eye = torch.eye(d, dtype=M.dtype, device=M.device).expand(B, 1, d, d)
    n = 1
    while n < K:
        A = A @ torch.cat([eye.expand(B, n, d, d), A[:, : K - n]], dim=1)
        n *= 2
    return A


def prefix_hop(M: Tensor) -> Tensor:
    """torch._higher_order_ops.associative_scan (BRIEF2 §5 says it imports)."""
    from torch._higher_order_ops.associative_scan import associative_scan

    def combine(earlier: Tensor, later: Tensor) -> Tensor:
        return later @ earlier

    return associative_scan(combine, M, dim=1, combine_mode="generic")


PREFIX_IMPLS = {"serial": prefix_serial, "scan": prefix_scan, "hop": prefix_hop}


def scan_depth(kind: str, K: int) -> int:
    """Sequential matmul levels used by an implementation over K positions."""
    if kind == "serial":
        return max(K - 1, 0)
    n, levels = 1, 0
    while n < K:
        n *= 2
        levels += 1
    return levels


# --------------------------------------------------------------------------- #
# 2. constraint families for the transition matrices                          #
# --------------------------------------------------------------------------- #

FAMILIES = ("colsoftmax", "sthard", "dsink", "orth", "dense")


def _sinkhorn(logits: Tensor, iters: int = 8) -> Tensor:
    x = logits - logits.amax(dim=(-2, -1), keepdim=True)
    for _ in range(iters):
        x = x - x.logsumexp(dim=-2, keepdim=True)
        x = x - x.logsumexp(dim=-1, keepdim=True)
    return x.exp()


def _cayley(logits: Tensor) -> Tensor:
    """Orthogonal matrix from a skew-symmetric part: (I-A)(I+A)^-1."""
    a = logits - logits.transpose(-1, -2)
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    return torch.linalg.solve((eye + a).transpose(-1, -2), (eye - a).transpose(-1, -2)).transpose(-1, -2)


def constrain(logits: Tensor, family: str) -> Tensor:
    """Map raw per-position logits (..., d, d) to a transition matrix."""
    if family == "colsoftmax":
        # column-stochastic: M[i, j] = P(next state i | current state j).
        # This is the natural family for a carry monoid (a *function* on states,
        # not a bijection) and it is numerically closed: products of
        # column-stochastic matrices stay column-stochastic, so the fp32 prefix
        # products can neither vanish nor explode.
        return logits.softmax(dim=-2)
    if family == "sthard":
        # PD-SSM style: column one-hot forward, softmax gradient backward.
        p = logits.softmax(dim=-2)
        hard = torch.zeros_like(p).scatter_(-2, p.argmax(dim=-2, keepdim=True), 1.0)
        return hard + (p - p.detach())
    if family == "dsink":
        return _sinkhorn(logits)
    if family == "orth":
        return _cayley(logits)
    if family == "dense":
        # unconstrained, renormalised per position so the prefix products of a
        # length-K chain neither vanish nor explode (PLAN2 §5 numerics note).
        n = logits.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
        return logits * (logits.shape[-1] ** 0.5 / n.clamp_min(1e-6))
    raise ValueError(f"unknown family {family!r}")


def snap_matrix(M: Tensor) -> Tensor:
    """Argmax-snap a transition matrix to a column one-hot (hard-state metric)."""
    hard = torch.zeros_like(M)
    return hard.scatter_(-2, M.argmax(dim=-2, keepdim=True), 1.0)


# --------------------------------------------------------------------------- #
# 3. digit-string helpers                                                     #
# --------------------------------------------------------------------------- #

Digits = Tensor  # (B, L, 10) — distributions over decimal digits, LSB first


def zero_digits(B: int, L: int, device, dtype) -> Digits:
    z = torch.zeros(B, L, 10, device=device, dtype=dtype)
    z[..., 0] = 1.0
    return z


def shift_up(v: Digits, i: int, L: int) -> Digits:
    """Multiply by 10**i in digit space: a pure re-index, no arithmetic."""
    if i == 0:
        out = v
    else:
        pad = zero_digits(v.shape[0], i, v.device, v.dtype)
        out = torch.cat([pad, v], dim=-2)
    if out.shape[-2] < L:
        out = torch.cat([out, zero_digits(v.shape[0], L - out.shape[-2], v.device, v.dtype)], dim=-2)
    return out[..., :L, :]


def digits_to_int(v: Digits) -> Tensor:
    """Argmax-decode a digit string (LSB first) to python-int-valued tensor."""
    idx = v.argmax(dim=-1)
    place = torch.tensor([10**k for k in range(v.shape[-2])], device=v.device, dtype=torch.float64)
    return (idx.to(torch.float64) * place).sum(-1)


def int_to_digits(values, L: int, device, dtype=torch.float32) -> Digits:
    """One-hot digit encoding, LSB first.  Used only to build probe inputs."""
    out = torch.zeros(len(values), L, 10, device=device, dtype=dtype)
    for b, v in enumerate(values):
        for k in range(L):
            out[b, k, (v // 10**k) % 10] = 1.0
    return out


# --------------------------------------------------------------------------- #
# 4. the monoid blocks                                                        #
# --------------------------------------------------------------------------- #

BIG = 12.0  # logit magnitude used by the constructed (diagnostic) tables


class PairMonoid(nn.Module):
    """A learned associative digit-position monoid over a pair of digit strings.

    At position k the module reads the two input digits ``(a_k, b_k)``, looks up
    a d x d transition matrix and a d x 10 emission table, then obtains the
    incoming carry/borrow state at *every* position at once with a prefix scan.

    Learned-op depth from the loss to ``trans``/``emit`` is **one** table
    application, regardless of how many digit positions there are — that is the
    conditioning claim this branch exists to test.
    """

    n_index = 100

    def __init__(self, d: int, family: str = "colsoftmax", init_scale: float = 0.5):
        super().__init__()
        self.d, self.family = d, family
        self.trans = nn.Parameter(torch.randn(self.n_index, d * d) * init_scale)
        self.emit = nn.Parameter(torch.randn(self.n_index, d * 10) * init_scale)

    def features(self, a: Digits, b: Digits) -> Tensor:
        return (a.unsqueeze(-1) * b.unsqueeze(-2)).flatten(-2)  # (..., 100)

    def forward(
        self,
        a: Digits,
        b: Digits,
        *,
        impl: str = "scan",
        hard: bool = False,
        reverse: bool = False,
        return_state: bool = False,
    ):
        shape = a.shape[:-2]
        L = a.shape[-2]
        feat = self.features(a, b).reshape(-1, L, self.n_index)
        if reverse:
            feat = feat.flip(-2)
        B = feat.shape[0]
        logits = (feat @ self.trans).view(B, L, self.d, self.d).float()
        M = constrain(logits, self.family)
        if hard:
            M = snap_matrix(M)
        P = PREFIX_IMPLS[impl](M)
        # exclusive prefix: state entering position k
        e0 = torch.zeros(B, self.d, 1, device=M.device, dtype=M.dtype)
        e0[:, 0, 0] = 1.0
        states = torch.cat([e0.unsqueeze(1).expand(B, 1, self.d, 1), P[:, :-1] @ e0.unsqueeze(1)], dim=1)
        c = states.squeeze(-1)  # (B, L, d)
        if hard:
            c = torch.zeros_like(c).scatter_(-1, c.argmax(-1, keepdim=True), 1.0)
        if return_state:
            final = (P[:, -1] @ e0).squeeze(-1)
            if hard:
                final = torch.zeros_like(final).scatter_(-1, final.argmax(-1, keepdim=True), 1.0)
            return c, final
        el = (feat @ self.emit).view(B, L, self.d, 10).float()
        out_logits = torch.einsum("bki,bkic->bkc", c, el)
        out = out_logits.softmax(dim=-1)
        if reverse:
            out = out.flip(-2)
        if hard:
            out = torch.zeros_like(out).scatter_(-1, out.argmax(-1, keepdim=True), 1.0)
        return out.reshape(*shape, L, 10)


class MulTable(nn.Module):
    """Learned single-digit product table: (a, b) -> (low digit, high digit)."""

    def __init__(self, init_scale: float = 0.5):
        super().__init__()
        self.lo = nn.Parameter(torch.randn(100, 10) * init_scale)
        self.hi = nn.Parameter(torch.randn(100, 10) * init_scale)

    def forward(self, a: Tensor, b: Tensor, *, hard: bool = False):
        feat = (a.unsqueeze(-1) * b.unsqueeze(-2)).flatten(-2)
        lo = (feat @ self.lo).softmax(-1)
        hi = (feat @ self.hi).softmax(-1)
        if hard:
            lo = torch.zeros_like(lo).scatter_(-1, lo.argmax(-1, keepdim=True), 1.0)
            hi = torch.zeros_like(hi).scatter_(-1, hi.argmax(-1, keepdim=True), 1.0)
        return lo, hi


# --------------------------------------------------------------------------- #
# 5. the ALU                                                                  #
# --------------------------------------------------------------------------- #


class MonoidALU(nn.Module):
    """``x^2 mod N`` from digit strings, as monoid scans over digit positions.

    Every learned tensor is indexed by a digit tuple; **no index ranges over
    Z_N**, so the parameter set is identical at every modulus (BRIEF2 §4 makes
    that mandatory for Hard, whose train/test moduli are disjoint).

    Shape of the computation, with S significant digit slots and L = 2S:

      multiply   1 product-table application  +  ceil(log2 2S) tree adds
      multiples  1 product-table application  +  1 add
      reduce     S x (1 compare + 1 subtract)

    so the number of *learned-table applications* on the gradient path is
    ``2 + ceil(log2 2S) + 1 + 2S`` — 12 at S=3, 14 at S=4 — against 39 for
    ``alu-depth``'s tree:quotient graph and 257 for the original ``DigitALU``.
    ``self.op_depth`` reports the measured number.
    """

    def __init__(
        self,
        slots: int,
        d: int = 16,
        family: str = "colsoftmax",
        impl: str = "scan",
        init_scale: float = 0.5,
        tie_mul: bool = True,
    ):
        super().__init__()
        self.S = slots
        self.L = 2 * slots
        self.d = d
        self.impl = impl
        self.family = family
        self.mul = MulTable(init_scale)
        self.mul_n = self.mul if tie_mul else MulTable(init_scale)
        self.add = PairMonoid(d, family, init_scale)
        self.sub = PairMonoid(d, family, init_scale)
        self.cmp = PairMonoid(d, family, init_scale)
        self.cmp_head = nn.Parameter(torch.randn(d) * init_scale)
        self.sel = nn.Parameter(torch.randn(10, 10) * init_scale)

    # -- structural bookkeeping -------------------------------------------- #
    @property
    def op_depth(self) -> int:
        n_tree = 0
        m = 2 * self.S
        while m > 1:
            m = (m + 1) // 2
            n_tree += 1
        return 1 + n_tree + 2 + 2 * self.S

    @property
    def graph_depth(self) -> int:
        """Sequential matmul levels (the raw autograd chain length)."""
        n_tree = 0
        m = 2 * self.S
        while m > 1:
            m = (m + 1) // 2
            n_tree += 1
        per_scan = scan_depth(self.impl, self.L)
        return 1 + n_tree * (1 + per_scan) + (1 + per_scan) + self.S * (2 + 2 * per_scan)

    # -- pieces ------------------------------------------------------------- #
    def _tree_add(self, rows: list[Digits], *, hard: bool) -> Digits:
        while len(rows) > 1:
            nxt = []
            pairs = [(rows[i], rows[i + 1]) for i in range(0, len(rows) - 1, 2)]
            if len(rows) % 2:
                leftover = rows[-1]
            else:
                leftover = None
            a = torch.stack([p[0] for p in pairs], dim=1)  # (B, P, L, 10)
            b = torch.stack([p[1] for p in pairs], dim=1)
            s = self.add(a, b, impl=self.impl, hard=hard)
            nxt = [s[:, i] for i in range(s.shape[1])]
            if leftover is not None:
                nxt.append(leftover)
            rows = nxt
        return rows[0]

    def multiply(self, x: Digits, *, hard: bool = False) -> Digits:
        B, L = x.shape[0], self.L
        xs = x[:, : self.S]  # (B, S, 10) multiplier digits
        lo, hi = self.mul(xs.unsqueeze(2), x.unsqueeze(1), hard=hard)  # (B,S,L,10)
        rows = []
        for i in range(self.S):
            rows.append(shift_up(lo[:, i], i, L))
            rows.append(shift_up(hi[:, i], i + 1, L))
        return self._tree_add(rows, hard=hard)

    def multiples(self, n: Digits, *, hard: bool = False) -> Digits:
        """All ten multiples q*N, q = 0..9, as (B, 10, L, 10)."""
        B, L = n.shape[0], self.L
        q = torch.eye(10, device=n.device, dtype=n.dtype).expand(B, 10, 10)
        lo, hi = self.mul_n(q.unsqueeze(2), n.unsqueeze(1), hard=hard)  # (B,10,L,10)
        hi_shift = torch.cat(
            [zero_digits(B * 10, 1, n.device, n.dtype).view(B, 10, 1, 10), hi[:, :, :-1]], dim=2
        )
        return self.add(lo, hi_shift, impl=self.impl, hard=hard)

    def compare_ge(self, a: Digits, b: Digits, *, hard: bool = False) -> Tensor:
        """P(a >= b) via the comparison monoid, scanned MSB-first."""
        shape = a.shape[:-2]
        _, final = self.cmp(a, b, impl=self.impl, hard=hard, reverse=True, return_state=True)
        return torch.sigmoid(final @ self.cmp_head).reshape(*shape)

    def reduce_mod(self, p: Digits, n: Digits, *, hard: bool = False) -> Digits:
        B, L = p.shape[0], self.L
        mult = self.multiples(n, hard=hard)  # (B, 10, L, 10)
        r = p
        for k in range(self.S - 1, -1, -1):
            cand = shift_up(mult.reshape(B * 10, L, 10), k, L).view(B, 10, L, 10)
            fits = self.compare_ge(r.unsqueeze(1).expand(-1, 10, -1, -1), cand, hard=hard)
            qlogits = fits @ self.sel.t()
            qsel = qlogits.softmax(-1)
            if hard:
                qsel = torch.zeros_like(qsel).scatter_(-1, qsel.argmax(-1, keepdim=True), 1.0)
            row = torch.einsum("bq,bqlc->blc", qsel, cand)
            r = self.sub(r, row, impl=self.impl, hard=hard)
        return r

    def forward(self, x: Digits, n: Digits, *, hard: bool = False) -> Digits:
        p = self.multiply(x, hard=hard)
        return self.reduce_mod(p, n, hard=hard)

    # -- DIAGNOSTIC ORACLE (never legal in a submission) -------------------- #
    @torch.no_grad()
    def construct_(self) -> None:
        """Write the exact digit tables.  BRIEF §4 rule 2: diagnostic only."""
        d = self.d
        for tbl in (self.mul, self.mul_n):
            tbl.lo.fill_(-BIG)
            tbl.hi.fill_(-BIG)
            for a in range(10):
                for b in range(10):
                    tbl.lo[a * 10 + b, (a * b) % 10] = BIG
                    tbl.hi[a * 10 + b, (a * b) // 10] = BIG
            if self.mul_n is self.mul:
                break

        # addition: state = carry in {0, 1}
        self.add.trans.fill_(-BIG)
        self.add.emit.fill_(-BIG)
        tr = self.add.trans.view(100, d, d)
        em = self.add.emit.view(100, d, 10)
        for a in range(10):
            for b in range(10):
                for c in range(d):
                    cc = c if c < 2 else 0
                    v = a + b + cc
                    tr[a * 10 + b, min(v // 10, d - 1), c] = BIG
                    em[a * 10 + b, c, v % 10] = BIG

        # subtraction: state = borrow in {0, 1}
        self.sub.trans.fill_(-BIG)
        self.sub.emit.fill_(-BIG)
        tr = self.sub.trans.view(100, d, d)
        em = self.sub.emit.view(100, d, 10)
        for a in range(10):
            for b in range(10):
                for c in range(d):
                    cc = c if c < 2 else 0
                    v = a - b - cc
                    tr[a * 10 + b, min(1 if v < 0 else 0, d - 1), c] = BIG
                    em[a * 10 + b, c, v % 10] = BIG

        # comparison: state 0 = EQ, 1 = LT, 2 = GT (needs d >= 3)
        self.cmp.trans.fill_(-BIG)
        tr = self.cmp.trans.view(100, d, d)
        for a in range(10):
            for b in range(10):
                for c in range(d):
                    if c == 0:
                        nxt = 0 if a == b else (2 if a > b else 1)
                    else:
                        nxt = c
                    tr[a * 10 + b, min(nxt, d - 1), c] = BIG
        self.cmp_head.fill_(0.0)
        self.cmp_head[0] = BIG
        if d > 1:
            self.cmp_head[1] = -BIG
        if d > 2:
            self.cmp_head[2] = BIG

        # quotient selection: logit_q = fits_q - fits_{q+1}
        self.sel.fill_(0.0)
        for q in range(10):
            self.sel[q, q] = BIG
            if q + 1 < 10:
                self.sel[q, q + 1] = -BIG

    @torch.no_grad()
    def corrupt_(self, k: int, generator: torch.Generator | None = None,
                 scale: float = 0.5) -> dict[str, Tensor]:
        """Randomise k cells of the constructed tables (repair-basin probe).

        Returns the index of every corrupted cell per table so the caller can
        count how many the objective puts back exactly.
        """
        cells = [(name, p) for name, p, _, _ in self._cells()]
        total = sum(p.shape[0] for _, p in cells)
        k = min(k, total)
        pick = torch.randperm(total, generator=generator)[:k]
        off, hit = 0, {}
        for name, p in cells:
            n = p.shape[0]
            sel = pick[(pick >= off) & (pick < off + n)] - off
            if sel.numel():
                p.data[sel.to(p.device)] = torch.randn(
                    sel.numel(), *p.shape[1:], generator=generator
                ).to(p.device) * scale
                hit[name] = sel
            off += n
        return hit

    @torch.no_grad()
    def repaired(self, hit: dict[str, Tensor], ref: "MonoidALU | None" = None) -> tuple[int, int]:
        """How many corrupted cells now argmax-match the truth exactly."""
        ref = ref or self._reference()
        rc = {n: q for n, q, _, _ in ref._cells()}
        ok = tot = 0
        for name, p, shape, dim in self._cells():
            if name not in hit:
                continue
            idx = hit[name].to(p.device)
            a = p.view(*shape)[idx].argmax(dim)
            b = rc[name].view(*shape)[idx].argmax(dim)
            m = (a == b)
            m = m.flatten(1).all(-1) if m.ndim > 1 else m
            ok += int(m.sum())
            tot += m.numel()
        return ok, tot

    # -- table bookkeeping -------------------------------------------------- #
    def _cells(self):
        """(name, parameter, view shape, argmax dim) for every digit-indexed table.

        A *cell* is one row — one (a, b) digit pair — matching the unit used by
        `alu-relational`'s repair-basin measurement (400 cells there, 700 here).
        """
        d = self.d
        out = [
            ("mul_lo", self.mul.lo, (100, 10), -1),
            ("mul_hi", self.mul.hi, (100, 10), -1),
            ("add_trans", self.add.trans, (100, d, d), -2),
            ("add_emit", self.add.emit, (100, d, 10), -1),
            ("sub_trans", self.sub.trans, (100, d, d), -2),
            ("sub_emit", self.sub.emit, (100, d, 10), -1),
            ("cmp_trans", self.cmp.trans, (100, d, d), -2),
        ]
        if self.mul_n is not self.mul:
            out += [("muln_lo", self.mul_n.lo, (100, 10), -1),
                    ("muln_hi", self.mul_n.hi, (100, 10), -1)]
        return out

    @torch.no_grad()
    def _reference(self) -> "MonoidALU":
        ref = MonoidALU(self.S, self.d, self.family, self.impl,
                        tie_mul=self.mul_n is self.mul).to(self.mul.lo.device)
        ref.construct_()
        return ref

    @torch.no_grad()
    def table_correct(self, ref: "MonoidALU | None" = None) -> dict[str, float]:
        """Fraction of table cells whose argmax matches the true solution."""
        ref = ref or self._reference()
        out, correct, total = {}, 0, 0
        rc = {n: q for n, q, _, _ in ref._cells()}
        for name, p, shape, dim in self._cells():
            a = p.view(*shape).argmax(dim)
            b = rc[name].view(*shape).argmax(dim)
            ok = (a == b)
            per_cell = ok.flatten(1).all(-1) if ok.ndim > 1 else ok
            out[name] = per_cell.float().mean().item()
            correct += int(per_cell.sum())
            total += per_cell.numel()
        out["_all"] = correct / total
        out["_n_cells"] = total
        return out
