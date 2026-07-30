"""PLAN2 3.1 — learned monoid + matrix associative scan over DIGIT POSITIONS.

Architecture, in one sentence: a marker-relative learned pointer reads the digit
slots of N and x out of the prompt, then ``x^2 mod N`` is computed by learned
digit-indexed tables whose carry / borrow / comparison recurrences are *linear*
matrix products evaluated with a log-depth associative prefix scan, and a
learned selector picks how many squarings to apply.

Why this shape (BRIEF2 2c): carry propagation across digit positions is an
associative prefix computation, so the scan belongs on the digit axis (length
2S ~ 8), not on the prompt sequence (length 13-21, which is not the composition
axis).  Every learned tensor is indexed by a digit tuple, so the parameter set
is identical at every modulus -- required on Hard, whose train and test moduli
are disjoint.

COMPLIANCE.  No hard-coded arithmetic: the product table, the add/subtract carry
monoids, the comparison monoid, the quotient selector and the digit/marker
readouts are all learned from random init in this run.  The only fixed
operations are re-indexing (a shift by a digit position is a slice), the
softmax, and matrix multiplication.  Everything is one differentiable graph; no
custom training loop and no participant-controlled backward.
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

# ---- architecture constants ------------------------------------------------ #
SLOTS = 4          # digit slots of N / x  (covers e1's 3-digit and e5's 4-digit N)
D_STATE = 16       # carry/borrow/compare monoid state size (PLAN2's `d`)
FAMILY = "colsoftmax"
D_MODEL = 32
N_LOOP = 3         # squaring applications the selector chooses between
DIGIT_OFFSET = 7   # vocab: PAD BOS N X T ANS EOS then the ten digits
O_LO, O_HI = -2, 9
NEG = -1e4         # finfo.min overflows bf16 under amp
INIT = 0.5


# --------------------------------------------------------------------------- #
# prefix products                                                             #
# --------------------------------------------------------------------------- #
# P[k] = M[k] @ ... @ M[0].  Verified bit-exact against a serial reference on
# permutation matrices for every d in {3,8,16,32} and K in 1..17
# (lab/test_monoid.py); fp32 max abs err <= 2e-6 on random dense matrices.


def prefix_scan(M: Tensor) -> Tensor:
    b, k, d, _ = M.shape
    eye = torch.eye(d, dtype=M.dtype, device=M.device).expand(b, 1, d, d)
    n = 1
    while n < k:
        M = M @ torch.cat([eye.expand(b, n, d, d), M[:, : k - n]], dim=1)
        n *= 2
    return M


def constrain(logits: Tensor) -> Tensor:
    if FAMILY == "colsoftmax":
        return logits.softmax(dim=-2)
    if FAMILY == "sthard":
        p = logits.softmax(dim=-2)
        hard = torch.zeros_like(p).scatter_(-2, p.argmax(dim=-2, keepdim=True), 1.0)
        return hard + (p - p.detach())
    raise ValueError(FAMILY)


def zero_digits(b: int, n: int, device, dtype) -> Tensor:
    z = torch.zeros(b, n, 10, device=device, dtype=dtype)
    z[..., 0] = 1.0
    return z


def shift_up(v: Tensor, i: int, length: int) -> Tensor:
    """Multiply by 10**i: a slice, not an arithmetic operation."""
    out = v
    if i:
        out = torch.cat([zero_digits(v.shape[0], i, v.device, v.dtype), v], dim=-2)
    if out.shape[-2] < length:
        out = torch.cat(
            [out, zero_digits(v.shape[0], length - out.shape[-2], v.device, v.dtype)], dim=-2
        )
    return out[..., :length, :]


# --------------------------------------------------------------------------- #
# learned blocks                                                              #
# --------------------------------------------------------------------------- #


class PairMonoid(nn.Module):
    """Per-digit-position d x d transition + emission, composed by a scan."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d
        self.trans = nn.Parameter(torch.randn(100, d * d) * INIT)
        self.emit = nn.Parameter(torch.randn(100, d * 10) * INIT)

    def forward(self, a: Tensor, b: Tensor, reverse: bool = False, state: bool = False):
        shape, length = a.shape[:-2], a.shape[-2]
        feat = (a.unsqueeze(-1) * b.unsqueeze(-2)).flatten(-2).reshape(-1, length, 100)
        if reverse:
            feat = feat.flip(-2)
        n = feat.shape[0]
        # fp32 interior: PLAN2 5 -- bf16 silently destroys long matrix products
        m = constrain((feat @ self.trans).view(n, length, self.d, self.d).float())
        p = prefix_scan(m)
        e0 = torch.zeros(n, 1, self.d, 1, device=m.device, dtype=m.dtype)
        e0[:, :, 0] = 1.0
        c = torch.cat([e0, p[:, :-1] @ e0], dim=1).squeeze(-1)
        if state:
            return (p[:, -1] @ e0[:, 0]).squeeze(-1)
        el = (feat @ self.emit).view(n, length, self.d, 10).float()
        out = torch.einsum("bki,bkic->bkc", c, el).softmax(dim=-1)
        if reverse:
            out = out.flip(-2)
        return out.reshape(*shape, length, 10).to(a.dtype)


class MulTable(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lo = nn.Parameter(torch.randn(100, 10) * INIT)
        self.hi = nn.Parameter(torch.randn(100, 10) * INIT)

    def forward(self, a: Tensor, b: Tensor):
        feat = (a.unsqueeze(-1) * b.unsqueeze(-2)).flatten(-2)
        return (feat @ self.lo).softmax(-1), (feat @ self.hi).softmax(-1)


class MonoidALU(nn.Module):
    """y = x^2 mod N from digit strings, as monoid scans over digit positions."""

    def __init__(self, slots: int, d: int) -> None:
        super().__init__()
        self.s, self.length = slots, 2 * slots
        self.mul = MulTable()
        self.add = PairMonoid(d)
        self.sub = PairMonoid(d)
        self.cmp = PairMonoid(d)
        self.cmp_head = nn.Parameter(torch.randn(d) * INIT)
        self.sel = nn.Parameter(torch.randn(10, 10) * INIT)

    def _tree_add(self, rows: list[Tensor]) -> Tensor:
        while len(rows) > 1:
            spare = rows[-1] if len(rows) % 2 else None
            a = torch.stack([rows[i] for i in range(0, len(rows) - 1, 2)], 1)
            b = torch.stack([rows[i + 1] for i in range(0, len(rows) - 1, 2)], 1)
            s = self.add(a, b)
            rows = [s[:, i] for i in range(s.shape[1])]
            if spare is not None:
                rows.append(spare)
        return rows[0]

    def multiply(self, x: Tensor) -> Tensor:
        lo, hi = self.mul(x[:, : self.s].unsqueeze(2), x.unsqueeze(1))
        rows = []
        for i in range(self.s):
            rows.append(shift_up(lo[:, i], i, self.length))
            rows.append(shift_up(hi[:, i], i + 1, self.length))
        return self._tree_add(rows)

    def multiples(self, n: Tensor) -> Tensor:
        b, length = n.shape[0], self.length
        q = torch.eye(10, device=n.device, dtype=n.dtype).expand(b, 10, 10)
        lo, hi = self.mul(q.unsqueeze(2), n.unsqueeze(1))
        pad = zero_digits(b * 10, 1, n.device, n.dtype).view(b, 10, 1, 10)
        return self.add(lo, torch.cat([pad, hi[:, :, :-1]], dim=2))

    def forward(self, x: Tensor, n: Tensor) -> Tensor:
        b, length = x.shape[0], self.length
        r = self.multiply(x)
        mult = self.multiples(n)
        for k in range(self.s - 1, -1, -1):
            cand = shift_up(mult.reshape(b * 10, length, 10), k, length).view(b, 10, length, 10)
            fits = torch.sigmoid(
                self.cmp(r.unsqueeze(1).expand(-1, 10, -1, -1), cand, reverse=True, state=True)
                @ self.cmp_head
            ).view(b, 10)
            q = (fits @ self.sel.t()).softmax(-1)
            r = self.sub(r, torch.einsum("bq,bqlc->blc", q, cand))
        return r


class MarkerPointer(nn.Module):
    """Marker-relative digit-slot attention (anchor each slot on the marker that
    TERMINATES its own field, so the slots do not move when T gains a digit)."""

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
            rels.append(F.pad(a[:, :, o:], (0, o)) if o >= 0 else F.pad(a[:, :, : length + o], (-o, 0)))
        rel = torch.stack(rels, dim=-1)
        logit = torch.einsum("balo,sao->bsl", rel, self.r) + torch.einsum(
            "bal,sa->bsl", a.cumsum(-1), self.g
        )
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
        self.digit = nn.Linear(D_MODEL, 10)
        self.pointer = MarkerPointer(D_MODEL, 2 * SLOTS + 2)
        self.alu = MonoidALU(SLOTS, D_STATE)
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
        attn = self.pointer(emb, mask)                       # (B, 2S+2, L)
        slots = torch.einsum("bsl,blc->bsc", attn, dig)      # (B, 2S+2, 10)
        x = shift_up(slots[:, : self.s], 0, 2 * self.s)
        n = shift_up(slots[:, self.s : 2 * self.s], 0, 2 * self.s)
        t = slots[:, 2 * self.s :]

        states = []
        y = x
        for _ in range(N_LOOP):
            y = self.alu(y, n)
            states.append(y)
        w = (t.flatten(1) @ self.loop).softmax(-1)            # learned depth selector
        y = torch.einsum("bk,bklc->blc", w, torch.stack(states, 1))

        logy = y[:, : self.s].clamp_min(1e-6).log()
        vals = y.new_full((b, self.s, self.config.vocab_size), -30.0)
        vals[..., DIGIT_OFFSET : DIGIT_OFFSET + 10] = logy
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
            lr=0.0,  # BRIEF2 6.1 CONTROL: what the metric row reads at random init
            betas=(0.9, 0.95),
            weight_decay=0.0,
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=128,
    eval_batch_size=1024,
)
