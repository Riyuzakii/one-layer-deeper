"""PLAN2 §3.6 — sequential non-linear RNN (LSTM), aggressively fused.

RECURRENCE AXIS (stated explicitly, per BRIEF2 §2a).  The recurrence runs over the
**token / digit-place axis of the prompt** (length 13-21).  It does *not* run over the
task's composition depth `T <= 64`; `T` is read as input digits like everything else.
BRIEF2 §2b says T-fold composition is already solved and the one open bottleneck is a
single squaring on unseen operands, so the recurrence is aimed at that: the decoder
scans prompt positions **right-to-left, anchored to each row's last valid token**, so
its hidden state travels in the *carry* direction — least-significant answer digit
first.  That is the classic propagate/generate shape for multi-digit arithmetic.

FUSION.  Every scan is an `nn.LSTM`, which dispatches to cuDNN's fused
multi-timestep kernel.  Measured on this box at batch 512 / L=21: a hand-written
Python-loop LSTM costs 16.9 ms/step, the same math through `nn.LSTM` costs 2.10 ms/step
(8.1x), against 4.43 ms/step for the D=128 8-loop recurrent transformer that measured
38.6 ms/step on the competition H100.  No Triton is needed and none is used — see
`lab/reports/p2-sequential-rnn.md` §3.

Padding is handled by an end-anchored index permutation rather than
`pack_padded_sequence`, so nothing leaves the GPU and no CPU sync is introduced.

COMPLIANCE.  Every tensor is learned from random init.  No arithmetic, solver, lookup
table or data-dependent Python control flow in the forward pass; the only thing read
out of the batch besides `input_ids` is the evaluator-supplied `attention_mask`.
Plain cross-entropy (`training_loss=None`), evaluator-owned optimizer step.
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

# --- knobs (a generator rewrites these to produce variants) -------------------------
D_H = 64
D_EMB = 64
LOOPS = 4
ALIGN = True        # learned end-anchored place-relative mixing before the decoder
DROPOUT = 0.0
LR = 1e-3
WD = 0.1
MAX_STEPS = None    # None -> the manifest ceiling binds
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 4096


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


def _end_anchored_reverse_index(attention_mask: Tensor, length: int) -> Tensor:
    """Row-wise reversal of the *valid* region only.

    For a row of true length ``n`` in a padded width-``L`` batch this returns
    ``[n-1, n-2, ..., 0, n, n+1, ..., L-1]``.  Gathering with it puts each row's last
    real token at index 0, which is where the carry scan must start; gathering with it
    a second time undoes the permutation exactly (it is an involution).
    """
    lengths = attention_mask.sum(dim=1, keepdim=True)  # (B,1)
    positions = torch.arange(length, device=attention_mask.device)[None, :]
    return torch.where(positions < lengths, lengths - 1 - positions, positions)


def _permute(x: Tensor, index: Tensor) -> Tensor:
    return torch.gather(x, 1, index[..., None].expand(-1, -1, x.shape[-1]))


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.loops = LOOPS
        self.align = ALIGN
        self.token_embedding = nn.Embedding(spec.vocab_size, D_EMB)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_EMB)
        # encoder: two unidirectional fused LSTMs over the token axis.  Written as two
        # scans rather than `bidirectional=True` so the backward direction can start
        # from each row's own last valid token instead of from padding.
        self.enc_fwd = nn.LSTM(D_EMB, D_H, batch_first=True)
        self.enc_rev = nn.LSTM(D_EMB, D_H, batch_first=True)
        # decoder: the carry scan, in end-anchored coordinates.
        self.dec = nn.LSTM(2 * D_H, D_H, batch_first=True)
        if ALIGN:
            # learned mixing over *distance from the end of the prompt*, i.e. over
            # digit place.  One L x L matrix; no index ranges over Z_N.
            self.place_mix = nn.Parameter(torch.zeros(spec.max_seq_len, spec.max_seq_len))
        self.loop_proj = nn.Linear(D_H, D_EMB) if LOOPS > 1 else None
        self.norm = nn.LayerNorm(D_H)
        self.head = nn.Linear(D_H, spec.vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        batch, length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.any(dim=1)
        attention_mask = attention_mask.bool()

        rev = _end_anchored_reverse_index(attention_mask, length)
        positions = torch.arange(length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = x * attention_mask[..., None]
        state = x.new_zeros(batch, length, D_H)

        for _ in range(self.loops):
            fwd, _ = self.enc_fwd(x)
            back, _ = self.enc_rev(_permute(x, rev))
            enc = torch.cat([fwd, _permute(back, rev)], dim=-1)

            # end-anchored coordinates: index 0 is each row's last real token, which is
            # the least-significant answer digit.
            enc_rev = _permute(enc, rev)
            if self.align:
                mix = self.place_mix[:length, :length].to(enc_rev.dtype)
                enc_rev = enc_rev + torch.einsum("jk,bkd->bjd", mix, enc_rev)

            dec, _ = self.dec(enc_rev)          # the carry scan: LSB -> MSB
            state = _permute(dec, rev)          # back to prompt order
            if self.loop_proj is not None:
                x = self.loop_proj(state)

        return self.head(self.norm(state)), None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.dim() <= 1 else decay).append(param)
    return OptimizerBundle(
        torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": WD},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=LR,
            betas=(0.9, 0.95),
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=None,
    batch_size=BATCH_SIZE,
    max_steps=MAX_STEPS,
    eval_batch_size=EVAL_BATCH_SIZE,
)
