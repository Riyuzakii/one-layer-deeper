"""Modular reduction at O(1) learned-op depth (RANKING.md method #3).

Architecture in one sentence: a marker-relative learned pointer reads the digit
slots of N and x out of the prompt, then ``x^2 mod N`` is computed as a Barrett
reduction in which every accumulation is one learned table applied to a *bag of
digit-pair indicators* plus one learned carry monoid -- so the number of learned
operations between the loss and any table is a constant, independent of how many
digits the modulus has.

WHY THIS SHAPE.  `plan2/matrix-scan` measured the repair basin of the legal
end-of-chain label against learned-op depth: 0/5 at depth 39 (`DigitALU`),
14/400 at depth 12 (`MonoidALU`), 50/400 for laws evaluated one op from a table.
Its §7 located the obstruction exactly -- long division carries an S-digit
partial remainder between quotient digits, which is `2S` learned applications
and the whole of the growth in S.  Barrett replaces that with two
multiplications and one subtraction, and the bag-of-pairs accumulator replaces
`MonoidALU`'s ceil(log2 2S)-deep tree of pairwise adds with one table.  Measured
here: reduction depth `2S` -> **5**, total 12/21 -> **7**.

Every learned tensor is indexed by a digit tuple or a column total; no index
ranges over Z_N, so the parameter set is identical at every modulus -- mandatory
on Hard, whose train and test moduli are disjoint.

COMPLIANCE.  No hard-coded arithmetic.  The column-value tables, the carry
monoids, the correction selector, the reciprocal's long-division tables, the
marker pointer, the digit readout and the depth selector are all learned from
random init in this run.  The only fixed operations are re-indexing (a shift by
a digit position is a slice), pooling one-hot indicators into a bag, softmax and
matmul.  One differentiable graph, no custom training loop, no
participant-controlled backward.  `construct_` and the oracle reciprocal used in
`lab/` to measure the class ceiling appear NOWHERE in this file.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    assert_model_state,
)

SLOTS = 7           # hf1 / h1: in-distribution moduli are <= 20 bits = 7 digits
N_LOOP = 3          # squaring applications the learned selector chooses between
D_MODEL = 32
DIGIT_OFFSET = 7    # vocab: PAD BOS N X T ANS EOS then the ten digits
O_LO, O_HI = -2, 9
NEG = -1e4          # finfo.min overflows bf16 under amp
INIT = 0.5
RECIP = "div"       # "div": learned long division (exactly expressible)
                    # "head": one shallow learned map from N's digits


# --------------------------------------------------------------------------- #
# digit helpers -- fixed re-indexing only                                     #
# --------------------------------------------------------------------------- #

_PLANS: dict = {}


def zero_digits(b: int, length: int, device, dtype) -> Tensor:
    z = torch.zeros(b, length, 10, device=device, dtype=dtype)
    z[..., 0] = 1.0
    return z


def pair_plan(la: int, lb: int, lout: int, shift: int, device):
    key = ("p", la, lb, lout, shift, str(device))
    if key not in _PLANS:
        i = torch.arange(la).view(la, 1).expand(la, lb).reshape(-1)
        j = torch.arange(lb).view(1, lb).expand(la, lb).reshape(-1)
        lo, hi = torch.zeros(la * lb, lout), torch.zeros(la * lb, lout)
        cl, ch = i + j + shift, i + j + shift + 1
        rows = torch.arange(la * lb)
        lo[rows[cl < lout], cl[cl < lout]] = 1.0
        hi[rows[ch < lout], ch[ch < lout]] = 1.0
        _PLANS[key] = (lo.to(device), hi.to(device))
    return _PLANS[key]


def digit_plan(ld: int, lout: int, shift: int, device) -> Tensor:
    key = ("d", ld, lout, shift, str(device))
    if key not in _PLANS:
        m = torch.zeros(ld, lout)
        c = torch.arange(ld) + shift
        m[torch.arange(ld)[c < lout], c[c < lout]] = 1.0
        _PLANS[key] = m.to(device)
    return _PLANS[key]


def pair_bag(a: Tensor, b: Tensor, a_lo: Tensor, a_hi: Tensor) -> Tensor:
    outer = (a.unsqueeze(-2).unsqueeze(-1) * b.unsqueeze(-3).unsqueeze(-2))
    outer = outer.flatten(-2).flatten(-3, -2)
    return torch.stack([torch.einsum("...pc,pl->...lc", outer, a_lo),
                        torch.einsum("...pc,pl->...lc", outer, a_hi)], dim=-2)


def digit_bag(d: Tensor, plan: Tensor) -> Tensor:
    return torch.einsum("...pc,pl->...lc", d, plan)


def prefix_serial(m: Tensor) -> Tensor:
    outs = [m[:, 0]]
    for k in range(1, m.shape[1]):
        outs.append(m[:, k] @ outs[-1])
    return torch.stack(outs, dim=1)


# --------------------------------------------------------------------------- #
# the two learned primitives                                                  #
# --------------------------------------------------------------------------- #


class ColSum(nn.Module):
    """Bag of contributions in a column -> the column's total.  ONE learned op.

    Pooling before the table is what makes a whole multi-row accumulation cost
    one application no matter how many rows or columns there are.  Subtraction
    is a role, not a separate module, so `p - (q+c)N` for every correction `c`
    is the same single application.
    """

    def __init__(self, m_lo: int, m_hi: int, n_pair: int, n_dig: int):
        super().__init__()
        self.m_lo, self.M = m_lo, m_hi - m_lo + 1
        self.n_pair, self.n_dig = n_pair, n_dig
        self.wp = nn.Parameter(torch.randn(max(n_pair, 1), 100, self.M) * INIT)
        self.wd = nn.Parameter(torch.randn(max(n_dig, 1), 10, self.M) * INIT)
        self.b = nn.Parameter(torch.randn(self.M) * INIT)

    def forward(self, pbags, dbags) -> Tensor:
        # fp32: the column-total logits span ~1e5 while adjacent totals differ
        # by O(1), which bf16 cannot resolve.
        with torch.autocast(device_type="cuda", enabled=False):
            out = self.b.float()
            if pbags is not None and self.n_pair:
                out = out + torch.einsum("...lrc,rcm->...lm", pbags.float(), self.wp.float())
            if dbags is not None and self.n_dig:
                out = out + torch.einsum("...lrc,rcm->...lm", dbags.float(), self.wd.float())
        return out


class CarryMonoid(nn.Module):
    """Column totals -> output digits, carries as a linear monoid.  ONE learned op.

    The transition matrices are read once per column in parallel and composed by
    fixed matmul, so the nonlinearity is paid once however long the digit string
    is; the final state is the carry-out, i.e. the sign of a subtraction.
    """

    def __init__(self, m: int, c_lo: int, c_hi: int):
        super().__init__()
        self.M, self.c_lo, self.C = m, c_lo, c_hi - c_lo + 1
        self.trans = nn.Parameter(torch.randn(m, self.C * self.C) * INIT)
        self.emit = nn.Parameter(torch.randn(m, self.C * 10) * INIT)

    def forward(self, tot: Tensor):
        shape = tot.shape[:-2]
        length, m = tot.shape[-2], tot.shape[-1]
        t = tot.reshape(-1, length, m).float()
        b = t.shape[0]
        with torch.autocast(device_type="cuda", enabled=False):
            mm = (t @ self.trans.float()).view(b, length, self.C, self.C).softmax(dim=-2)
            p = prefix_serial(mm)
            c0 = torch.zeros(b, self.C, 1, device=t.device, dtype=t.dtype)
            c0[:, -self.c_lo, 0] = 1.0
            states = torch.cat([c0.unsqueeze(1), p[:, :-1] @ c0.unsqueeze(1)], 1).squeeze(-1)
            el = (t @ self.emit.float()).view(b, length, self.C, 10)
            out = torch.einsum("blc,blcd->bld", states, el).softmax(-1)
            final = (p[:, -1] @ c0).squeeze(-1)
        return out.reshape(*shape, length, 10), final.reshape(*shape, self.C)


class AccStage(nn.Module):
    """Accumulate then normalise: 2 learned-op depth at any modulus size."""

    def __init__(self, m_lo: int, m_hi: int, n_pair: int, n_dig: int):
        super().__init__()
        c_hi = max(0, (m_hi + 8) // 9)
        c_lo = min(0, -((-m_lo + 8) // 9))
        self.cols = ColSum(m_lo, m_hi, n_pair, n_dig)
        self.carry = CarryMonoid(self.cols.M, c_lo, c_hi)

    def forward(self, pbags, dbags):
        return self.carry(self.cols(pbags, dbags).softmax(-1))


# --------------------------------------------------------------------------- #
# reciprocal                                                                  #
# --------------------------------------------------------------------------- #


class RecipHead(nn.Module):
    """mu = floor(10^{2S}/N) as one shallow learned map.  Learned-op depth 2."""

    def __init__(self, s: int, lmu: int, hidden: int = 256):
        super().__init__()
        self.s, self.lmu = s, lmu
        self.w1 = nn.Parameter(torch.randn(s * 10, hidden) * (s * 10) ** -0.5)
        self.w2 = nn.Parameter(torch.randn(hidden, lmu * 10) * hidden**-0.5)

    def forward(self, n: Tensor) -> Tensor:
        h = torch.relu(n[:, : self.s].flatten(-2) @ self.w1)
        return (h @ self.w2).view(-1, self.lmu, 10).softmax(-1)


class LongDivRecip(nn.Module):
    """mu by learned long division, with PRIVATE parameters on an N-only path.

    Deliberately not O(1): division of an S-digit number is not, in a digit
    representation.  Isolating it here is what keeps every table on the
    x-dependent reduction path at constant depth from the loss.
    """

    def __init__(self, s: int, lmu: int):
        super().__init__()
        self.s, self.lmu, self.lr = s, lmu, lmu + s + 1
        self.step = AccStage(-18, 9, 2, 1)
        self.sel = nn.Parameter(torch.randn(10, self.step.carry.C) * INIT)

    def forward(self, n: Tensor) -> Tensor:
        b, dev, dt = n.shape[0], n.device, n.dtype
        r = zero_digits(b, self.lr, dev, dt).clone()
        r[:, 2 * self.s, 0] = 0.0
        r[:, 2 * self.s, 1] = 1.0
        dplan = digit_plan(self.lr, self.lr, 0, dev)
        eye = torch.eye(10, device=dev, dtype=dt).view(1, 10, 1, 10).expand(b, 10, 1, 10)
        digs = []
        for k in range(self.lmu - 1, -1, -1):
            a_lo, a_hi = pair_plan(1, self.s, self.lr, k, dev)
            pb = pair_bag(eye, n[:, : self.s].unsqueeze(1).expand(b, 10, self.s, 10), a_lo, a_hi)
            db = digit_bag(r, dplan).unsqueeze(1).unsqueeze(-2).expand(b, 10, self.lr, 1, 10)
            cand, final = self.step(pb, db)
            s = torch.einsum("bqc,qc->bq", final, self.sel).softmax(-1)
            digs.append(s)
            r = torch.einsum("bq,bqld->bld", s, cand)
        return torch.stack(digs[::-1], dim=1)


# --------------------------------------------------------------------------- #
# the ALU                                                                     #
# --------------------------------------------------------------------------- #


class O1ReduceALU(nn.Module):
    """x^2 mod N with the reduction at constant learned-op depth (5 of 7)."""

    def __init__(self, slots: int, recip: str = RECIP, n_corr: int = 2):
        super().__init__()
        s = self.s = slots
        self.length = 2 * s
        self.lmu, self.lq2, self.lq3, self.lr = 2 * s, 4 * s, s + 1, 2 * s + 2
        self.n_corr = n_corr
        self.sq = AccStage(0, 17 * s, 2, 0)
        self.qm = AccStage(0, 34 * s, 2, 0)
        self.rs = AccStage(-17 * s - 9 * (n_corr - 1), 9, 2, 2)
        self.sel = nn.Parameter(torch.randn(n_corr, self.rs.carry.C) * INIT)
        self.recip = LongDivRecip(s, self.lmu) if recip == "div" else RecipHead(s, self.lmu)

    def forward(self, x: Tensor, n: Tensor) -> Tensor:
        b, dev = x.shape[0], x.device
        a_lo, a_hi = pair_plan(self.s, self.s, self.length, 0, dev)
        p, _ = self.sq(pair_bag(x[:, : self.s], x[:, : self.s], a_lo, a_hi), None)

        mu = self.recip(n)
        a_lo, a_hi = pair_plan(self.length, self.lmu, self.lq2, 0, dev)
        q2, _ = self.qm(pair_bag(p, mu, a_lo, a_hi), None)
        q3 = q2[:, self.length:][:, : self.lq3]

        k = self.n_corr
        a_lo, a_hi = pair_plan(self.lq3, self.s, self.lr, 0, dev)
        pb = pair_bag(q3, n[:, : self.s], a_lo, a_hi).unsqueeze(1).expand(b, k, self.lr, 2, 100)
        dp = digit_bag(p, digit_plan(self.length, self.lr, 0, dev))
        dn = digit_bag(n[:, : self.s], digit_plan(self.s, self.lr, 0, dev))
        w = torch.arange(k, device=dev, dtype=dn.dtype).view(1, k, 1, 1)
        db = torch.stack([dp.unsqueeze(1).expand(b, k, self.lr, 10), dn.unsqueeze(1) * w], -2)
        cand, final = self.rs(pb, db)
        sel = torch.einsum("bkc,kc->bk", final, self.sel).softmax(-1)
        r = torch.einsum("bk,bkld->bld", sel, cand)
        return r[:, : self.length]


# --------------------------------------------------------------------------- #
# front end                                                                   #
# --------------------------------------------------------------------------- #


class MarkerPointer(nn.Module):
    """Marker-relative digit-slot attention: anchor each slot on the marker that
    TERMINATES its own field, so slots do not move when T gains a digit."""

    def __init__(self, d_model: int, n_slot: int) -> None:
        super().__init__()
        self.n_off = O_HI - O_LO + 1
        self.probe = nn.Linear(d_model, 3, bias=False)
        self.r = nn.Parameter(torch.randn(n_slot, 4, self.n_off) * INIT)
        self.g = nn.Parameter(torch.randn(n_slot, 4) * INIT)

    def forward(self, emb: Tensor, mask: Tensor) -> Tensor:
        b, length, _ = emb.shape
        sc = self.probe(emb).transpose(1, 2).masked_fill(~mask[:, None, :], NEG)
        a = F.softmax(sc, dim=-1)
        idx = mask.long().sum(-1) - 1
        a = torch.cat([a, F.one_hot(idx, length).to(a.dtype)[:, None]], dim=1)
        rels = []
        for o in range(O_LO, O_HI + 1):
            rels.append(F.pad(a[:, :, o:], (0, o)) if o >= 0
                        else F.pad(a[:, :, : length + o], (-o, 0)))
        rel = torch.stack(rels, dim=-1)
        logit = torch.einsum("balo,sao->bsl", rel, self.r) + torch.einsum(
            "bal,sa->bsl", a.cumsum(-1), self.g)
        return F.softmax(logit.masked_fill(~mask[:, None, :], NEG), dim=-1)


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.s = SLOTS
        self.embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        nn.init.normal_(self.embedding.weight, std=0.02)   # PD-SSM's init fix
        self.digit = nn.Linear(D_MODEL, 10)
        self.pointer = MarkerPointer(D_MODEL, 2 * SLOTS + 2)
        self.alu = O1ReduceALU(SLOTS)
        self.loop = nn.Parameter(torch.randn(2 * 10, N_LOOP) * INIT)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        if attention_mask is None:
            mask = torch.ones_like(input_ids, dtype=torch.bool)
        elif attention_mask.dim() == 3:
            mask = attention_mask.any(dim=1)
        else:
            mask = attention_mask.bool()
        b, length = input_ids.shape
        emb = self.embedding(input_ids)
        dig = self.digit(emb).softmax(-1)
        attn = self.pointer(emb, mask)
        slots = torch.einsum("bsl,blc->bsc", attn, dig)
        pad = zero_digits(b, self.s, input_ids.device, slots.dtype)
        x = torch.cat([slots[:, : self.s], pad], dim=1)
        n = torch.cat([slots[:, self.s: 2 * self.s], pad], dim=1)
        t = slots[:, 2 * self.s:]

        states, y = [], x
        for _ in range(N_LOOP):
            y = self.alu(y, n)
            states.append(y)
        w = (t.flatten(1) @ self.loop).softmax(-1)
        y = torch.einsum("bk,bklc->blc", w, torch.stack(states, 1))

        logy = y[:, : self.s].clamp_min(1e-6).log()
        vals = y.new_full((b, self.s, self.config.vocab_size), -30.0)
        vals[..., DIGIT_OFFSET: DIGIT_OFFSET + 10] = logy
        last = mask.long().sum(-1) - 1
        pos = (last[:, None] - torch.arange(self.s, device=input_ids.device)[None, :]).clamp_min(0)
        logits = y.new_zeros(b, length, self.config.vocab_size)
        logits = logits.scatter(1, pos[:, :, None].expand(-1, -1, self.config.vocab_size), vals)
        return logits, None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(
        torch.optim.AdamW(
            model.parameters(),
            lr=1e-2,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=128,
    eval_batch_size=512,
)
