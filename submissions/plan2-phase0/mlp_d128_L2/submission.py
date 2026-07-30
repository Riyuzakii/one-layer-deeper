"""PLAN2 Phase-0 probe: ARCH=mlp, D=128, L=2, lr=0.001.

Shared scaffold, one swappable sequence mixer.  The recurrence runs over the
PROMPT-TOKEN axis (13-21 tokens), not over the task's composition depth T.
Every tensor is learned from random init; nothing about modular arithmetic is
hard-coded anywhere in the forward pass.
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

ARCH = "mlp"
D_MODEL = 128
N_LAYERS = 2
N_HEADS = 4
D_HEAD = 32
FF_MULT = 4
LR = 0.001
WEIGHT_DECAY = 0.1
_BATCH_SIZE = 512
_EVAL_BATCH_SIZE = 4096

_INNER = N_HEADS * D_HEAD


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


def _reverse_index(mask: Tensor) -> Tensor:
    """Row-wise reversal index that keeps trailing PADs in place.

    ``idx`` is an involution on ``[0, len_i)`` and the identity on
    ``[len_i, T)``.  Gathering with it turns a right-padded row into a
    right-padded *reversed* row, so a plain left-to-right scan run on the
    gathered sequence realises the reverse direction with no CPU sync and no
    pad contamination of valid positions.
    """

    lengths = mask.sum(dim=1, keepdim=True)
    idx = torch.arange(mask.shape[1], device=mask.device)[None, :]
    return torch.where(idx < lengths, lengths - 1 - idx, idx)


def _gather_seq(x: Tensor, index: Tensor) -> Tensor:
    return x.gather(1, index[..., None].expand(-1, -1, x.shape[-1]))


class LinRecMixer(nn.Module):
    """Matrix-state linear recurrence; the transition class is set by ARCH."""

    diagonal = ARCH.startswith("diag")

    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(D_MODEL, _INNER, bias=False)
        self.k = nn.Linear(D_MODEL, _INNER, bias=False)
        self.v = nn.Linear(D_MODEL, _INNER, bias=False)
        self.g = nn.Linear(D_MODEL, _INNER, bias=True)
        self.onorm = RMSNorm(2 * _INNER)
        self.out = nn.Linear(2 * _INNER, D_MODEL, bias=False)
        # matched init: both members of each eigenvalue pair start at ~0.5
        with torch.no_grad():
            if ARCH == "diag01":
                self.g.bias.fill_(0.0)          # sigmoid(0) = 0.5
            elif ARCH == "diagpm1":
                self.g.bias.fill_(1.0986123)    # 2*sigmoid(1.0986)-1 = 0.5
            elif ARCH == "delta01":
                self.g.bias.fill_(0.0)          # 1 - sigmoid(0) = 0.5
            elif ARCH == "deltapm1":
                self.g.bias.fill_(-1.0986123)   # 1 - 2*sigmoid(-1.0986) = 0.5

    def _scan(self, x: Tensor, mask: Tensor) -> Tensor:
        batch, length, _ = x.shape
        shape = (batch, length, N_HEADS, D_HEAD)
        q = F.normalize(self.q(x).view(shape).float(), dim=-1)
        k = F.normalize(self.k(x).view(shape).float(), dim=-1)
        v = self.v(x).view(shape).float()
        g = self.g(x).view(shape).float()
        m = mask[:, :, None, None].float()
        v = v * m
        if self.diagonal:
            a = torch.sigmoid(g)
            if ARCH == "diagpm1":
                a = 2.0 * a - 1.0
            a = a * m + (1.0 - m)               # PAD -> identity transition
        else:
            beta = torch.sigmoid(g.mean(dim=-1, keepdim=True)).squeeze(-1)
            c = (2.0 if ARCH == "deltapm1" else 1.0) * beta
            c = c * mask[:, :, None].float()    # PAD -> identity transition

        state = x.new_zeros((batch, N_HEADS, D_HEAD, D_HEAD), dtype=torch.float32)
        outs = []
        for t in range(length):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]
            if self.diagonal:
                state = state * a[:, t].unsqueeze(-2) + vt.unsqueeze(-1) * kt.unsqueeze(-2)
            else:
                sk = (state * kt.unsqueeze(-2)).sum(dim=-1)
                upd = (vt - sk) * c[:, t].unsqueeze(-1)
                state = state + upd.unsqueeze(-1) * kt.unsqueeze(-2)
            outs.append((state * qt.unsqueeze(-2)).sum(dim=-1))
        return torch.stack(outs, dim=1).reshape(batch, length, _INNER)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        rev = _reverse_index(mask)
        fwd = self._scan(x, mask)
        bwd = _gather_seq(self._scan(_gather_seq(x, rev), mask), rev)
        y = torch.cat([fwd, bwd], dim=-1).to(x.dtype)
        return self.out(self.onorm(y))


class RnnMixer(nn.Module):
    """Bidirectional non-linear gated RNN (maximal expressivity, O(T) serial)."""

    def __init__(self) -> None:
        super().__init__()
        cls = nn.LSTM if ARCH == "lstm" else nn.GRU
        self.fwd = cls(D_MODEL, _INNER, num_layers=1, batch_first=True)
        self.bwd = cls(D_MODEL, _INNER, num_layers=1, batch_first=True)
        self.onorm = RMSNorm(2 * _INNER)
        self.out = nn.Linear(2 * _INNER, D_MODEL, bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        rev = _reverse_index(mask)
        xm = x * mask[..., None].to(x.dtype)
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = xm.float()
            f, _ = self.fwd(xf)
            b, _ = self.bwd(_gather_seq(xf, rev))
        y = torch.cat([f, _gather_seq(b, rev)], dim=-1).to(x.dtype)
        return self.out(self.onorm(y))


class AttnMixer(nn.Module):
    """Softmax attention control (TC^0)."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * _INNER, bias=False)
        self.out = nn.Linear(_INNER, D_MODEL, bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, N_HEADS, D_HEAD)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        attn = attn.transpose(1, 2).reshape(batch, length, _INNER)
        return self.out(attn)


class NoMixer(nn.Module):
    """Positionwise only -- the no-sequence-mixing floor control."""

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:  # noqa: ARG002
        return torch.zeros_like(x)


def _make_mixer() -> nn.Module:
    if ARCH in ("diag01", "diagpm1", "delta01", "deltapm1"):
        return LinRecMixer()
    if ARCH in ("lstm", "gru"):
        return RnnMixer()
    if ARCH == "attn":
        return AttnMixer()
    if ARCH == "mlp":
        return NoMixer()
    raise ValueError(f"unknown ARCH {ARCH}")


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mixer_norm = RMSNorm(D_MODEL)
        self.mixer = _make_mixer()
        self.ff_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, FF_MULT * D_MODEL)
        self.down = nn.Linear(FF_MULT * D_MODEL, D_MODEL)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x + self.mixer(self.mixer_norm(x), mask)
        return x + self.down(F.gelu(self.up(self.ff_norm(x))))


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        batch, length = input_ids.shape
        if attention_mask is None:
            mask = input_ids.new_ones((batch, length), dtype=torch.bool)
        elif attention_mask.dim() == 3:
            mask = attention_mask.any(dim=1).bool()
        else:
            mask = attention_mask.bool()
        positions = torch.arange(length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.final_norm(x)), None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(
        torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            betas=(0.9, 0.95),
            weight_decay=WEIGHT_DECAY,
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=_BATCH_SIZE,
    eval_batch_size=_EVAL_BATCH_SIZE,
)
