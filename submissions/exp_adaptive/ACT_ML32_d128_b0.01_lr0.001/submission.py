"""Adaptive (PonderNet-lite) recurrent transformer: MAX_LOOPS=32 beta=0.01."""

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

D_MODEL = 128
NUM_HEADS = 4
MAX_LOOPS = 32
FF_MULT = 4
_BETA = 0.01          # ponder penalty weight (encourages fewer steps)
_LR = 0.001
_MAX_STEPS = 400
_BATCH_SIZE = None


class Config:
    def __init__(self, vocab_size, max_seq_len):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, FF_MULT * D_MODEL)
        self.down = nn.Linear(FF_MULT * D_MODEL, D_MODEL)

    def forward(self, x, attention_mask):
        residual = x
        x = self.attention_norm(x)
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        mask = None
        if attention_mask is not None:
            if attention_mask.shape == (batch, length):
                mask = attention_mask[:, None, None, :]
            elif attention_mask.shape == (batch, length, length):
                mask = attention_mask[:, None, :, :]
            else:
                raise ValueError("invalid attention_mask shape")
            mask = mask.to(device=x.device, dtype=torch.bool)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        x = residual + self.out(x)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


class Model(nn.Module):
    max_loops = MAX_LOOPS

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.block = Block()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        self.halt = nn.Linear(D_MODEL, 1)
        # init: halt slowly at first (use many steps early), let ponder trim later
        with torch.no_grad():
            self.halt.bias.fill_(-2.0)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        B, Lseq = input_ids.shape
        dev = input_ids.device
        positions = torch.arange(Lseq, device=dev)
        emb0 = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = attention_mask if attention_mask is not None else (input_ids != 0)
        maskf = mask.to(emb0.dtype)
        denom = maskf.sum(1, keepdim=True).clamp(min=1.0)          # [B,1]

        h = emb0
        acc_state = torch.zeros_like(emb0)                        # halt-weighted state sum
        remain = torch.ones(B, device=dev, dtype=emb0.dtype)      # prod (1-lam) so far
        exp_steps = torch.zeros(B, device=dev, dtype=emb0.dtype)  # E[t]
        for t in range(1, MAX_LOOPS + 1):
            h = self.block(h + emb0, attention_mask)              # input injection
            hn = self.final_norm(h)
            pooled = (hn * maskf[..., None]).sum(1) / denom       # [B,D]
            lam = torch.sigmoid(self.halt(pooled)).squeeze(-1)    # [B] halt prob
            if t == MAX_LOOPS:
                p_t = remain                                      # last step: remaining mass
            else:
                p_t = remain * lam
                remain = remain * (1.0 - lam)
            acc_state = acc_state + p_t[:, None, None] * hn
            exp_steps = exp_steps + p_t * float(t)
        logits = self.head(acc_state)                             # readout from halt-weighted state
        aux = exp_steps.mean()                                    # ponder cost (scalar, differentiable)
        return logits, aux


def training_loss(logits: Tensor, labels: Tensor, aux: object) -> Tensor:
    ce = F.cross_entropy(logits.float(), labels)
    ponder = aux if torch.is_tensor(aux) else logits.new_zeros(())
    return ce + _BETA * ponder


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(
        torch.optim.AdamW(
            model.parameters(), lr=_LR, betas=(0.9, 0.95), weight_decay=0.1,
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,
    batch_size=_BATCH_SIZE,
    max_steps=_MAX_STEPS,
)
