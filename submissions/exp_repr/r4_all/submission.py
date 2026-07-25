"""Representation-axis submission template (see lab/make_repr_submission.py).

Hypothesis family: the ~1-5% exact-accuracy plateau on repeated modular squaring
is an INPUT/OUTPUT REPRESENTATION failure, not a capacity/optimisation failure.

Everything below is an inductive bias over the *public prompt format*
(`[N] digits [X] digits [T] digits`, most-significant digit first, answer
supervised at the last len(answer) positions).  No arithmetic is implemented
here; the model still has to learn x^(2^T) mod N from random init.

Axes (all independently switchable):
  USE_ABS    absolute-from-left learned position embedding (the baseline)
  USE_FIELD  learned "which field am I in" embedding (N / X / T / marker)
  USE_PLACE  learned place-value index *within its own field* (LSD = 0),
             one shared table across fields  (the abacus / index-hint idea)
  USE_RPOS   learned distance-from-end index.  Because the evaluator reads the
             answer off the LAST len(answer) positions, distance-from-end j is
             exactly the place value 10^j of the answer digit read there.
  LAYOUT     "flat"      -> the evaluator's own token order
             "slots_sep" -> digits re-laid-out LSD-first into place-aligned
                            slots: N places, then x places, then T digits
             "slots_sum" -> one token per place value, N and x digits summed
                            into the same slot (channel representation)
Slot layouts do their re-ordering inside forward() with gathers, and scatter the
per-place logits back onto the evaluator's own positions, so the contract is
unchanged and the whole path stays differentiable.
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

# --- CONFIG BEGIN ---
D_MODEL = 128
NUM_HEADS = 4
NUM_LOOPS = 8
FF_MULT = 4
_LR = 0.001
_WD = 0.1
_MAX_STEPS = None
_BATCH_SIZE = None
LAYOUT = 'flat'
USE_ABS = True
USE_FIELD = True
USE_PLACE = True
USE_RPOS = True
# --- CONFIG END ---

# public prompt-format token ids (data/squaring_mod.py TOKEN_IDS)
_N_TOK, _X_TOK, _T_TOK = 2, 3, 4
_DIGIT_OFFSET = 7
_ABSENT = 10  # extra "this place does not exist" digit symbol


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, FF_MULT * D_MODEL)
        self.down = nn.Linear(FF_MULT * D_MODEL, D_MODEL)

    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
        residual = x
        x = self.attention_norm(x)
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(device=x.device, dtype=torch.bool)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        x = residual + self.out(x)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


def _field_geometry(input_ids: Tensor, mask: Tensor):
    """Segment the prompt into its three public fields.

    Returns per-token (pos, is_marker, is_digit, field_index) and per-field
    (start, end, length) tensors of shape [B, 4] indexed by field id
    (1 = N, 2 = x, 3 = T; column 0 is the padding sink).
    """
    batch, length = input_ids.shape
    device = input_ids.device
    pos = torch.arange(length, device=device).unsqueeze(0).expand(batch, length)
    is_marker = ((input_ids >= _N_TOK) & (input_ids <= _T_TOK)) & mask
    field = torch.cumsum(is_marker.long(), dim=1)
    field = torch.where(mask, field, torch.zeros_like(field)).clamp(0, 3)
    is_digit = mask & ~is_marker & (field > 0)

    neg = torch.full_like(pos, -1)
    big = torch.full_like(pos, length + 1)
    fend = torch.full((batch, 4), -1, dtype=torch.long, device=device)
    fend.scatter_reduce_(
        1, field, torch.where(is_digit, pos, neg), reduce="amax", include_self=True
    )
    fstart = torch.full((batch, 4), length + 1, dtype=torch.long, device=device)
    fstart.scatter_reduce_(
        1, field, torch.where(is_digit, pos, big), reduce="amin", include_self=True
    )
    flen = (fend - fstart + 1).clamp(min=0)
    return pos, is_marker, is_digit, field, fstart, fend, flen


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.vocab_size = spec.vocab_size
        self.max_seq_len = spec.max_seq_len
        table = spec.max_seq_len + 2
        if LAYOUT == "flat":
            self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
            if USE_ABS:
                self.position_embedding = nn.Embedding(table, D_MODEL)
            if USE_FIELD:
                # 0..3 = digit of field f, 4..7 = the field marker token itself
                self.field_embedding = nn.Embedding(8, D_MODEL)
            if USE_PLACE:
                # index = place value within its own field; last slot = "not a digit"
                self.place_embedding = nn.Embedding(table, D_MODEL)
            if USE_RPOS:
                self.rpos_embedding = nn.Embedding(table, D_MODEL)
        else:
            self.place_embedding = nn.Embedding(table, D_MODEL)
            self.t_place_embedding = nn.Embedding(table, D_MODEL)
            if LAYOUT == "slots_sum":
                self.n_digit_embedding = nn.Embedding(11, D_MODEL)
                self.x_digit_embedding = nn.Embedding(11, D_MODEL)
                self.t_digit_embedding = nn.Embedding(11, D_MODEL)
            else:
                self.digit_embedding = nn.Embedding(11, D_MODEL)
                self.field_embedding = nn.Embedding(8, D_MODEL)
        self.block = Block()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)

    # -- flat layout -------------------------------------------------------
    def _embed_flat(self, input_ids: Tensor, mask: Tensor) -> Tensor:
        x = self.token_embedding(input_ids)
        length = input_ids.shape[1]
        if USE_ABS:
            pos = torch.arange(length, device=input_ids.device)
            x = x + self.position_embedding(pos)
        if not (USE_FIELD or USE_PLACE or USE_RPOS):
            return x
        pos, is_marker, is_digit, field, _, fend, _ = _field_geometry(input_ids, mask)
        if USE_FIELD:
            fidx = torch.where(is_marker, field + 4, field)
            x = x + self.field_embedding(fidx)
        if USE_PLACE:
            place = (fend.gather(1, field) - pos).clamp(min=0)
            special = torch.full_like(place, self.max_seq_len + 1)
            x = x + self.place_embedding(torch.where(is_digit, place, special))
        if USE_RPOS:
            seq_len = mask.sum(dim=1, keepdim=True)
            rpos = (seq_len - 1 - pos).clamp(min=0, max=self.max_seq_len + 1)
            x = x + self.rpos_embedding(rpos)
        return x

    # -- place-aligned slot layouts ---------------------------------------
    @staticmethod
    def _gather_places(input_ids: Tensor, fend: Tensor, flen: Tensor, f: int, places: Tensor) -> Tensor:
        length = input_ids.shape[1]
        index = fend[:, f : f + 1] - places.unsqueeze(0)
        ok = (places.unsqueeze(0) < flen[:, f : f + 1]) & (index >= 0)
        tok = input_ids.gather(1, index.clamp(0, length - 1))
        digit = torch.where(tok >= _DIGIT_OFFSET, tok - _DIGIT_OFFSET, torch.full_like(tok, _ABSENT))
        return torch.where(ok, digit, torch.full_like(digit, _ABSENT)).clamp(0, _ABSENT)

    def _embed_slots(self, input_ids: Tensor, mask: Tensor):
        device = input_ids.device
        _, _, _, _, _, fend, flen = _field_geometry(input_ids, mask)
        n_places = int(max(flen[:, 1].max().item(), flen[:, 2].max().item(), 1))
        t_places = int(max(flen[:, 3].max().item(), 1))
        places = torch.arange(n_places, device=device)
        t_idx = torch.arange(t_places, device=device)
        d_n = self._gather_places(input_ids, fend, flen, 1, places)
        d_x = self._gather_places(input_ids, fend, flen, 2, places)
        d_t = self._gather_places(input_ids, fend, flen, 3, t_idx)
        place_vec = self.place_embedding(places)
        if LAYOUT == "slots_sum":
            body = (
                self.n_digit_embedding(d_n)
                + self.x_digit_embedding(d_x)
                + place_vec
            )
            tail = self.t_digit_embedding(d_t) + self.t_place_embedding(t_idx)
            x = torch.cat([body, tail], dim=1)
            return x, n_places, 0
        n_tok = self.digit_embedding(d_n) + place_vec + self.field_embedding.weight[1]
        x_tok = self.digit_embedding(d_x) + place_vec + self.field_embedding.weight[2]
        t_tok = (
            self.digit_embedding(d_t)
            + self.t_place_embedding(t_idx)
            + self.field_embedding.weight[3]
        )
        x = torch.cat([n_tok, x_tok, t_tok], dim=1)
        return x, n_places, n_places  # read the answer off the x-aligned slots

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        if attention_mask is None:
            mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            mask = attention_mask.to(torch.bool)
        batch, length = input_ids.shape
        if LAYOUT == "flat":
            x = self._embed_flat(input_ids, mask)
            for _ in range(NUM_LOOPS):
                x = self.block(x, mask)
            return self.head(self.final_norm(x)), None

        x, n_places, read_offset = self._embed_slots(input_ids, mask)
        for _ in range(NUM_LOOPS):
            x = self.block(x, None)
        slot_logits = self.head(
            self.final_norm(x[:, read_offset : read_offset + n_places])
        )
        seq_len = mask.sum(dim=1, keepdim=True)
        places = torch.arange(n_places, device=input_ids.device).unsqueeze(0)
        target = (seq_len - 1 - places).clamp(0, length - 1)
        out = slot_logits.new_zeros(batch, length, self.vocab_size)
        out.scatter_(
            1, target.unsqueeze(-1).expand(batch, n_places, self.vocab_size), slot_logits
        )
        return out, None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=_LR,
        betas=(0.9, 0.95),
        weight_decay=_WD,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(opt, None)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=None,
    batch_size=_BATCH_SIZE,
    max_steps=_MAX_STEPS,
)
