"""digit-carry: marker-relative prompt parser + weight-tied squaring step.

Front end -- bottleneck (A).  A decimal digit's PLACE VALUE is defined by its
offset from the end of ITS OWN field.  The prompt is left aligned with
variable-length fields, so absolute position cannot express place value, and
distance from the end of the prompt expresses it only while the T field has a
fixed digit count -- it breaks on exactly the T=16/32/64 rungs.  This model
instead learns, for each digit slot, which marker token terminates its field and
how far before that marker to read:

    anchor_a[i] = softmax_i(<learned probe_a, embed(tok_i)>)   a in {N,X,T}
    anchor_end  = last valid position of the attention mask
    rel[a,i,o]  = anchor_a[i+o]
    attn[s,i]   = softmax_i( sum_ao rel[a,i,o] R[s,a,o] + sum_a cum_a[i] G[s,a] )

R and G are learned tables over (slot, anchor, offset); the whole thing is one
differentiable attention and no Python control flow reads input_ids.

Step -- bottleneck (B).  ONE shared block is applied LOOPS times with a
soft-digit round trip between applications, and an ORDERED pointer selector
(a learned scalar location on the step axis plus a window that anneals from
wide to sharp) routes the parsed T field to a step count.
"""

from __future__ import annotations

import math

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

D_MODEL = 64
N_FREQ = 32
N_HARM = 8
LOOPS = 4
FRONT = "marker"
READOUT = "phase"
VINIT = True
ANNEAL_STEPS = 2000
RAND_LOOPS = 1
LR = 0.01
WD = 0.1
DIGIT_TAU = 0.2
VINIT_SCALE = 8.0
_MAX_STEPS = None
_BATCH_SIZE = None
O_LO, O_HI = -2, 9


class MarkerPointer(nn.Module):
    def __init__(self, d_model: int, n_slot: int) -> None:
        super().__init__()
        self.n_off = O_HI - O_LO + 1
        self.probe = nn.Linear(d_model, 3, bias=False)
        self.R = nn.Parameter(torch.randn(n_slot, 4, self.n_off) * 0.5)
        self.G = nn.Parameter(torch.randn(n_slot, 4) * 0.5)

    def forward(self, emb: Tensor, mask: Tensor) -> Tensor:
        b, L, _ = emb.shape
        neg = torch.finfo(emb.dtype).min
        sc = self.probe(emb).transpose(1, 2)
        sc = sc.masked_fill(~mask[:, None, :], neg)
        a = F.softmax(sc, dim=-1)
        idx = mask.long().sum(-1) - 1
        a = torch.cat([a, F.one_hot(idx, L).to(a.dtype)[:, None]], dim=1)
        rels = []
        for o in range(O_LO, O_HI + 1):
            if o >= 0:
                rels.append(F.pad(a[:, :, o:], (0, o)))
            else:
                rels.append(F.pad(a[:, :, : L + o], (-o, 0)))
        rel = torch.stack(rels, dim=-1)
        logit = (torch.einsum("balo,sao->bsl", rel, self.R)
                 + torch.einsum("bal,sa->bsl", a.cumsum(-1), self.G))
        logit = logit.masked_fill(~mask[:, None, :], neg)
        return torch.einsum("bsl,bld->bsd", F.softmax(logit, dim=-1), emb)


class PosPointer(nn.Module):
    """Controls: `abs` = absolute position, `rev` = distance from the end."""

    def __init__(self, d_model: int, n_slot: int, max_len: int, rev: bool) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_slot, max_len + 8) * 0.5)
        self.rev = rev

    def forward(self, emb: Tensor, mask: Tensor) -> Tensor:
        b, L, _ = emb.shape
        neg = torch.finfo(emb.dtype).min
        if self.rev:
            last = mask.long().sum(-1) - 1
            pos = torch.arange(L, device=emb.device)[None, :]
            idx = (last[:, None] - pos).clamp(0, self.q.shape[1] - 1)
            logit = self.q[None].expand(b, -1, -1).gather(
                2, idx[:, None, :].expand(-1, self.q.shape[0], -1))
        else:
            logit = self.q[None, :, :L].expand(b, -1, -1)
        logit = logit.masked_fill(~mask[:, None, :], neg)
        return torch.einsum("bsl,bld->bsd", F.softmax(logit, dim=-1), emb)


class PhaseStep(nn.Module):
    """theta = sum_(i<=j) Q[i,j].(s_i (x) s_j) -> (cos,sin) -> per-slot digits."""

    def __init__(self, slots: int) -> None:
        super().__init__()
        self.pairs = [(i, j) for i in range(slots) for j in range(i, slots)]
        self.table = nn.Parameter(torch.randn(len(self.pairs), 10, 10, N_FREQ) * 0.5)
        feat = 2 * N_HARM * N_FREQ
        self.readout = nn.Parameter(torch.randn(slots, feat, 10) * feat ** -0.5)
        self.bias = nn.Parameter(torch.zeros(slots, 10))

    def forward(self, s: Tensor) -> Tensor:
        theta = None
        for n, (i, j) in enumerate(self.pairs):
            term = torch.einsum("ba,bc,acf->bf", s[:, i], s[:, j], self.table[n])
            theta = term if theta is None else theta + term
        parts = []
        for h in range(1, N_HARM + 1):
            parts.append(torch.cos(theta * h))
            parts.append(torch.sin(theta * h))
        feats = torch.cat(parts, dim=-1)
        return torch.einsum("bf,ifd->bid", feats, self.readout) + self.bias


class DigitStep(nn.Module):
    """Digit-compositional step: a partial-product table SHARED over place pairs
    feeds a slot scan whose cells are shared over slots.  No tensor here has an
    index that ranges over the residue, so it is not subject to the coverage
    ceiling that bounds any value-indexed readout."""

    def __init__(self, slots: int) -> None:
        super().__init__()
        self.S = slots
        self.K = 2 * slots - 1
        self.pair = nn.Parameter(torch.randn(10, 10, N_FREQ) / math.sqrt(N_FREQ))
        self.emb = nn.Linear(10, D_MODEL, bias=False)
        self.nemb = nn.Linear(10, D_MODEL, bias=False)
        self.finj = nn.Linear(N_FREQ, D_MODEL)
        self.fwd = nn.GRUCell(D_MODEL, D_MODEL)
        self.bwd = nn.GRUCell(D_MODEL, D_MODEL)
        self.upd = nn.Sequential(
            nn.Linear(3 * D_MODEL, 2 * D_MODEL), nn.GELU(),
            nn.Linear(2 * D_MODEL, D_MODEL))
        self.norm = nn.LayerNorm(D_MODEL)
        self.out = nn.Linear(D_MODEL, 10)
        self.zero = nn.Parameter(torch.zeros(10))

    def scan(self, h: Tensor) -> Tensor:
        b, w, _ = h.shape
        c = h.new_zeros(b, D_MODEL)
        fs = []
        for m in range(w):
            c = self.fwd(h[:, m], c)
            fs.append(c)
        c = h.new_zeros(b, D_MODEL)
        bs = []
        for m in range(w - 1, -1, -1):
            c = self.bwd(h[:, m], c)
            bs.append(c)
        u = torch.stack(fs, 1)
        v = torch.stack(bs[::-1], 1)
        return self.norm(h + self.upd(torch.cat([h, u, v], -1)))

    def forward(self, s: Tensor, ndig: Tensor) -> Tensor:
        b = s.shape[0]
        f = s.new_zeros(b, self.K, self.pair.shape[-1])
        for i in range(self.S):
            for j in range(self.S):
                f[:, i + j] = f[:, i + j] + torch.einsum(
                    "ba,bc,ach->bh", s[:, i], s[:, j], self.pair)
        z = F.softmax(self.zero, -1)[None, None].expand(b, 1, 10)
        r = z.expand(b, self.S, 10)
        nv = self.nemb(ndig)
        for k in range(self.K - 1, -1, -1):
            r = torch.cat([z, r[:, : self.S - 1]], dim=1)
            h = self.emb(r) + nv + torch.cat(
                [self.finj(f[:, k])[:, None], r.new_zeros(b, self.S - 1, D_MODEL)], 1)
            h = self.scan(h)
            r = F.softmax(self.out(h), -1)
        return torch.log(r + 1e-9)


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.vocab = spec.vocab_size
        self.S = max(3, (spec.max_seq_len - 3) // 2)
        self.emb = nn.Embedding(spec.vocab_size, D_MODEL)
        n_slot = 3 * self.S
        if FRONT == "marker":
            self.ptr = MarkerPointer(D_MODEL, n_slot)
            if VINIT:
                with torch.no_grad():
                    # structured but fully trainable: slot p of x reads 1+p
                    # before [T], of N reads 1+p before [X], of T reads p before
                    # the end; each field's opening marker is the fallback.
                    self.ptr.R.zero_()
                    self.ptr.G.zero_()
                    for p in range(self.S):
                        for s, (a, o, opn) in (
                            (p, (2, 1 + p, 1)),
                            (self.S + p, (1, 1 + p, 0)),
                            (2 * self.S + p, (3, p, 2)),
                        ):
                            self.ptr.R[s, a, o - O_LO] = VINIT_SCALE
                            self.ptr.R[s, opn, -O_LO] = 0.5 * VINIT_SCALE
                            self.ptr.G[s, opn] = VINIT_SCALE
        else:
            self.ptr = PosPointer(D_MODEL, n_slot, spec.max_seq_len,
                                  FRONT == "rev")
        self.digit = nn.Linear(D_MODEL, 10)
        self.step = PhaseStep(self.S) if READOUT == "phase" else DigitStep(self.S)
        self.loc = nn.Sequential(
            nn.Linear(self.S * 10, 64), nn.GELU(), nn.Linear(64, 1))
        self.head = nn.Linear(10, spec.vocab_size)
        self.register_buffer("nstep", torch.zeros((), dtype=torch.long),
                             persistent=False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        b, L = input_ids.shape
        if attention_mask is None:
            mask = torch.ones(b, L, dtype=torch.bool, device=input_ids.device)
        else:
            mask = attention_mask.to(torch.bool)
            if mask.dim() == 3:
                mask = mask[:, 0]
        e = self.emb(input_ids)
        slots = self.ptr(e, mask)
        # A slot must carry a DIGIT, not an arbitrary 10-dim code.  Left
        # unsharpened the projection is a continuous side-channel: the step
        # table can then index a value rather than a digit pair, which is
        # exactly the memorisation route the parse was meant to remove.
        dg = F.softmax(self.digit(slots) / DIGIT_TAU, dim=-1)
        sx, sn, st = dg[:, : self.S], dg[:, self.S: 2 * self.S], dg[:, 2 * self.S:]

        loops = LOOPS
        if self.training and RAND_LOOPS:
            self.nstep += 1
            loops = int(torch.randint(1, LOOPS + 1, (1,)).item())
        state = sx
        outs = []
        for _ in range(loops):
            out = self.step(state) if READOUT == "phase" else self.step(state, sn)
            outs.append(out)
            state = F.softmax(out, dim=-1)
        stacked = torch.stack(outs, dim=1)

        # ORDERED depth selector: a learned scalar location on the step axis
        # plus a window annealed from wide to sharp.
        loc = self.loc(st.flatten(1))
        grid = torch.arange(loops, device=e.device, dtype=loc.dtype)[None, :]
        frac = (self.nstep.float() / ANNEAL_STEPS).clamp(max=1.0)
        sigma = (1.0 - 0.9 * frac).clamp(min=0.1)
        w = F.softmax(-((grid - loc) ** 2) / (2 * sigma ** 2), dim=-1)
        mixed = torch.einsum("bt,btsd->bsd", w, stacked)      # (B, S, 10)

        # the answer is TAIL aligned: place p of the answer sits at the p-th
        # position back from the last valid token.
        logits = input_ids.new_zeros(b, L, self.vocab, dtype=mixed.dtype)
        last = mask.long().sum(-1) - 1
        vals = self.head(mixed)
        for p in range(self.S):
            pos = (last - p).clamp(min=0)
            logits.scatter_(1, pos[:, None, None].expand(-1, 1, self.vocab),
                            vals[:, p][:, None, :])
        return logits, None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return OptimizerBundle(
        torch.optim.AdamW(
            [{"params": decay, "weight_decay": WD},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=LR, betas=(0.9, 0.95),
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=_BATCH_SIZE,
    max_steps=_MAX_STEPS,
)
