#!/usr/bin/env python
"""Emit the REFERENCE-CLASS submission for the Hard-faithful tier.

This is the floor every sibling branch reads its results against.  It is the
official `submissions/baseline_adamw` architecture (one pre-norm transformer
block, tied head) with exactly two free wins applied and nothing else:

  1. ``EMB_INIT`` -- ``nn.Embedding``'s default ``N(0,1)`` with a tied head puts
     step-1 CE at ~80 instead of ``ln 17 = 2.833``.  Confirmed on the real
     evaluator: the hosted Hard run's ``metric.jsonl`` logs step-1 loss 79.936.
     ``submissions/baseline_adamw`` pays that toll.  Worth 19x on fitting and
     0.000 on generalisation (`plan2/phase0`), so it belongs in the *floor*,
     not in any claim.
  2. a chosen ``batch_size`` (the evaluator's DataLoader has no
     ``persistent_workers``; batch size is a free throughput knob).

Everything else is unchanged from the baseline, deliberately: the point of a
reference is that it is boring.

``--lr 0`` produces the initialisation control (rule 1 of the screening
discipline).  AdamW at ``lr=0`` also applies no decoupled weight decay, so the
parameters are bit-identical to ``build_model``'s output at every step.

LEGAL: everything is learned from random init; no oracle, no teacher forcing.

Usage:
  python lab/make_ref.py --dmodel 128 --lr 0 --tag lr0
  python lab/make_ref.py --dmodel 128 512 1024 --max-steps 40000
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TEMPLATE = '''"""Reference-class transformer for hf1 -- baseline architecture + free wins.

D_MODEL={dmodel} LOOPS={loops} EMB_INIT={emb_init} lr={lr} batch={batch_size}
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

D_MODEL = {dmodel}
NUM_HEADS = {num_heads}
NUM_LOOPS = {loops}
FF_MULT = 4
EMB_INIT = {emb_init}
_LR = {lr}
_WD = {wd}
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
        # THE FREE WIN: default N(0,1) embeddings with a tied head put step-1 CE
        # at ~80 instead of ln(vocab) = 2.833.
        nn.init.normal_(self.token_embedding.weight, std=EMB_INIT)
        nn.init.normal_(self.position_embedding.weight, std=EMB_INIT)
        self.blocks = nn.ModuleList([Block() for _ in range(NUM_LOOPS)])
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
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.head(self.final_norm(x)), None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(
        torch.optim.AdamW(
            model.parameters(),
            lr=_LR,
            betas=(0.9, 0.95),
            weight_decay=_WD,
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
    ap.add_argument("--dmodel", type=int, nargs="+", default=[128])
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--emb-init", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out_root = REPO / "submissions" / "hard-digitalu-hf1" / "ref"
    for d in args.dmodel:
        code = TEMPLATE.format(
            dmodel=d,
            num_heads=args.num_heads,
            loops=args.loops,
            emb_init=repr(args.emb_init),
            lr=repr(args.lr),
            wd=repr(args.wd),
            max_steps=repr(args.max_steps),
            batch_size=repr(args.batch_size),
        )
        name = f"d{d}_L{args.loops}_e{args.emb_init}_lr{args.lr}_b{args.batch_size}"
        if args.tag:
            name += f"_{args.tag}"
        path = out_root / name / "submission.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        print(path.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
