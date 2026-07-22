#!/usr/bin/env python
"""Generate standalone weight-tied recurrent submissions for the Phase 2A depth sweep.

One shared Block looped NUM_LOOPS times => param count is IDENTICAL across the sweep
(the clean control: only serial compute depth varies, not capacity). This is the
"one layer deeper" hypothesis: composition over k steps needs ~k serial steps.

Usage:
  python lab/make_recurrent.py --loops 1 2 4 8 16 32 --max-steps 800 --dmodel 128
Writes submissions/exp_recur/L{n}_d{dmodel}_s{maxsteps}/submission.py for each n.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TEMPLATE = '''"""Weight-tied recurrent transformer: one shared Block looped {num_loops}x."""

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

D_MODEL = {dmodel}
NUM_HEADS = {num_heads}
NUM_LOOPS = {num_loops}
FF_MULT = 4
_MAX_STEPS = {max_steps}
_BATCH_SIZE = {batch_size}


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
        self.block = Block()  # ONE shared block, applied NUM_LOOPS times
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for _ in range(NUM_LOOPS):
            x = self.block(x, attention_mask)
        return self.head(self.final_norm(x)), None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(
        torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=_BATCH_SIZE,
    max_steps=_MAX_STEPS,
)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loops", type=int, nargs="+", required=True)
    ap.add_argument("--dmodel", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="bake a max_steps cap into the submission (fixed-step mode)")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    out_root = REPO / "submissions" / "exp_recur"
    written = []
    for n in args.loops:
        code = TEMPLATE.format(
            num_loops=n,
            dmodel=args.dmodel,
            num_heads=args.num_heads,
            max_steps=repr(args.max_steps),
            batch_size=repr(args.batch_size),
        )
        tag = f"L{n}_d{args.dmodel}"
        if args.max_steps is not None:
            tag += f"_s{args.max_steps}"
        d = out_root / tag
        d.mkdir(parents=True, exist_ok=True)
        (d / "submission.py").write_text(code)
        written.append(str((d / "submission.py").relative_to(REPO)))
    print("wrote:")
    for w in written:
        print(" ", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
