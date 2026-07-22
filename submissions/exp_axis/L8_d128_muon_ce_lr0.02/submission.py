"""Recurrent transformer L=8 opt=muon loss=ce (generated)."""

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
NUM_LOOPS = 8
FF_MULT = 4
_LR = 0.02
_MAX_STEPS = 2000
_BATCH_SIZE = None
_FOCAL_GAMMA = 2.0


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
    num_loops = NUM_LOOPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.block = Block()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for _ in range(NUM_LOOPS):
            x = self.block(x, attention_mask)
        return self.head(self.final_norm(x)), None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def _newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.mT
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Muon (orthogonalized momentum) on 2D hidden matrices; AdamW on the rest.
    One optimizer, two kinds of param groups (use_muon flag)."""

    def __init__(self, param_groups):
        for g in param_groups:
            if g["use_muon"]:
                g.setdefault("lr", 0.02)
                g.setdefault("momentum", 0.95)
                g.setdefault("weight_decay", 0.0)
                g.setdefault("ns_steps", 5)
            else:
                g.setdefault("lr", 3e-3)
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-10)
                g.setdefault("weight_decay", 0.1)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            if g["use_muon"]:
                mom = g["momentum"]
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "mom" not in st:
                        st["mom"] = torch.zeros_like(p.grad)
                    buf = st["mom"]
                    buf.mul_(mom).add_(p.grad)
                    upd = p.grad.add(buf, alpha=mom)  # nesterov
                    upd = _newtonschulz5(upd, steps=g["ns_steps"])
                    scale = max(upd.size(0), upd.size(1)) ** 0.5
                    p.add_(upd, alpha=-g["lr"] * 0.2 * scale)
            else:
                b1, b2 = g["betas"]
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "step" not in st:
                        st["step"] = 0
                        st["m"] = torch.zeros_like(p.grad)
                        st["v"] = torch.zeros_like(p.grad)
                    st["step"] += 1
                    t = st["step"]
                    m, v = st["m"], st["v"]
                    m.mul_(b1).add_(p.grad, alpha=1 - b1)
                    v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    mhat = m / (1 - b1 ** t)
                    vhat = v / (1 - b2 ** t)
                    if g["weight_decay"]:
                        p.mul_(1 - g["lr"] * g["weight_decay"])
                    p.addcdiv_(mhat, vhat.sqrt().add_(g["eps"]), value=-g["lr"])
        return loss


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 2D hidden matrices in the block -> Muon; embeddings/head/norms -> Adam
        if p.ndim == 2 and "embedding" not in name and "head" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)
    groups = [
        dict(params=muon_params, use_muon=True, lr=_LR),
        dict(params=adam_params, use_muon=False, lr=3e-3),
    ]
    return OptimizerBundle(MuonWithAuxAdam(groups))


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=None,
    batch_size=_BATCH_SIZE,
    max_steps=_MAX_STEPS,
)
