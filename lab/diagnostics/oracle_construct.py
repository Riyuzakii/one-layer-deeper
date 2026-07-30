"""*** DIAGNOSTIC ONLY -- THIS IS NOT AND CAN NEVER BE A SUBMISSION. ***

It hard-codes the arithmetic in the forward pass, which BRIEF.md s4.2 forbids
outright.  It lives under `lab/diagnostics/`, never under `submissions/`.

WHY IT EXISTS.  PLAN2 s6 says that if a maximally expressive non-linear RNN also
gets nothing, the harness, the loss and the label alignment must be re-examined
before another architecture is written.  The cheapest decisive way to do that is
a *positive control*: a model that is known to compute the right answer, run
through the real `benchmark.runner` path, on the real manifests, scored by the
real metric.  If it certifies MAX_T = 64 then the evaluator, the collate, the
`target_positions` slicing, the exact-match rule and the depth ladder are all
sound, and every null in this project is a fact about learning rather than a
harness bug.

WHAT IT DOES.  Decodes (N, x, T) from the prompt using only the public prompt
format (`[N] d(N) [X] d(x) [T] d(T)`, most-significant digit first, token ids
from `data/squaring_mod.py:25-37`), applies integer modular squaring T times,
and writes the answer digits as logits at the last len(answer) prompt positions
-- i.e. exactly where `collate_squaring_mod` puts `target_positions`.

The single learned parameter is a logit scale, so the graph is differentiable
and the evaluator's `loss.backward()` / `optimizer.step()` loop runs unmodified.

VALIDITY LIMIT: int64 overflow.  y*y must stay below 2**63, so N must be below
~3.0e9.  Use only on datasets with moduli under that bound (e1-e5, m1-m3, and
the p2grid / p2iso sets built here); NOT on hp2 (30/32-bit sampled).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    assert_model_state,
)

TOK_X = 3
TOK_T = 4
DIGIT_OFFSET = 7
MAX_T = 64
MAX_DIGITS = 12


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.scale = nn.Parameter(torch.tensor(8.0))

    @staticmethod
    def _field_value(digits: Tensor, field: Tensor) -> Tensor:
        value = torch.zeros(digits.shape[0], dtype=torch.int64, device=digits.device)
        for j in range(digits.shape[1]):
            value = torch.where(field[:, j], value * 10 + digits[:, j], value)
        return value

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        batch, length = input_ids.shape
        device = input_ids.device
        if attention_mask is None:
            mask = torch.ones_like(input_ids, dtype=torch.bool)
        elif attention_mask.dim() == 3:
            mask = attention_mask.any(dim=1).bool()
        else:
            mask = attention_mask.bool()

        pos = torch.arange(length, device=device)[None, :]
        lengths = mask.sum(dim=1)[:, None]
        ix = (input_ids == TOK_X).int().argmax(dim=1)[:, None]
        it = (input_ids == TOK_T).int().argmax(dim=1)[:, None]
        digits = (input_ids - DIGIT_OFFSET).clamp(min=0)

        modulus = self._field_value(digits, (pos > 0) & (pos < ix))
        x = self._field_value(digits, (pos > ix) & (pos < it))
        steps = self._field_value(digits, (pos > it) & (pos < lengths))

        y = x % modulus.clamp(min=1)
        for i in range(MAX_T):
            squared = (y * y) % modulus.clamp(min=1)
            y = torch.where(steps > i, squared, y)

        n_answer = torch.zeros_like(y)
        rest = y.clone()
        for _ in range(MAX_DIGITS):
            n_answer = n_answer + (rest > 0).to(torch.int64)
            rest = rest // 10
        n_answer = n_answer.clamp(min=1)

        onehot = torch.zeros(
            (batch, length, self.config.vocab_size), device=device, dtype=torch.float32
        )
        rows = torch.arange(batch, device=device)
        ten = torch.tensor(10, dtype=torch.int64, device=device)
        for j in range(MAX_DIGITS):
            valid = (j < n_answer).float()
            place = (n_answer - 1 - j).clamp(min=0)
            digit = (y // torch.pow(ten, place)) % 10
            slot = (lengths[:, 0] - n_answer + j).clamp(0, length - 1)
            update = torch.zeros(
                (batch, self.config.vocab_size), device=device, dtype=torch.float32
            )
            update.scatter_(1, (DIGIT_OFFSET + digit)[:, None], 1.0)
            onehot[rows, slot] = onehot[rows, slot] + update * valid[:, None]
        return onehot * self.scale, None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(
        torch.optim.AdamW(
            model.parameters(),
            lr=1e-2,
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=512,
    eval_batch_size=4096,
)
