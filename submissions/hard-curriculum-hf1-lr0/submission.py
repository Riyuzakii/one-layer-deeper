"""hard/curriculum-hf1 -- a LOSS-SIDE CURRICULUM over modulus size and operand
magnitude, expressed under the evaluator's fixed one-step-per-batch loop.

WHAT THIS FILE DEMONSTRATES
---------------------------
`hf1`-shaped data mixes several modulus sizes in one training set, and any
modulus-independent readout trains the same parameters on all of them, so
"upweight the small moduli early, anneal toward the large ones" is a genuine
curriculum on shared parameters.  The evaluator owns the loop (one forward, one
backward, one `optimizer.step()` per batch), so the only two places a curriculum
can live are `training_loss` and the model's forward.

A MECHANICAL FACT THAT DECIDES WHICH ONE (source-verified, see the report):

    `training_loss(logits, labels, aux)` receives `token_logits[valid]` and
    `token_targets[valid]`, i.e. the tensors ALREADY FLATTENED by a ragged
    mask.  On this dataset the mask really is ragged --
    `tokenize_squaring_mod_with_result` emits `number_tokens(result)` with no
    zero padding, so the number of supervised positions is the DECIMAL DIGIT
    COUNT OF THE ANSWER and varies row by row.  Nothing in
    (`logits`, `labels`, `aux`) recovers the row boundaries, and the model
    cannot predict them (that count is a property of the answer).  So a
    per-example weight CANNOT be applied inside `training_loss` here.

    The equivalent operation in the forward is a per-row GRADIENT SCALE:

        h = w * h + (1 - w) * h.detach()

    whose forward value is exactly `h` and whose gradient to every upstream
    parameter is scaled by `w` -- algebraically identical to multiplying that
    row's loss by `w`.  It is ordinary arithmetic (no custom autograd Function,
    no participant-controlled backward), the graph stays unbroken, and the
    reported training loss is the plain unweighted one, so the curriculum
    cannot flatter the logged number.

    `lab/probe_curric.py --grad-equiv` verifies the equivalence numerically.

DIFFICULTY SIGNAL.  `_CURRIC_SRC="len"` (default) uses the number of valid
input tokens -- the prompt is `[N] digits(N) [X] digits(x) [T] digits(T)`, so
its length is a monotone function of the modulus's and the operand's decimal
size and of nothing else.  That is a property of the input's SHAPE, not a
decode of its content, so it is unambiguous under rule 2/3.  `_CURRIC_SRC="ids"`
reads the digit tokens to form log10 magnitudes of the two number fields; it is
a strictly better signal but it does read the prompt format, so it is FLAGGED
as compliance-uncertain and is not the default.

SCHEDULE.  The step counter lives in a NON-PERSISTENT buffer, which
`benchmark/api.py:26-42` excludes from the 5e8 model-state ceiling.  `beta`
anneals to 0, at which point the objective is exactly the unweighted one.

STATUS.  Shipped so the mechanism is exercised end to end through the real
runner.  The scientific test of the curriculum is in
`lab/reports/hard-curriculum-hf1.md`; on the evidence there the curriculum is
NULL and this file is not a candidate for a hosted attempt.
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

D_MODEL = 128
NUM_HEADS = 4
NUM_LOOPS = 8
FF_MULT = 4
EMB_INIT = 0.02          # plan2/phase0: the default N(0,1) with a tied head
                         # gives an initial loss of ~80 instead of ln 17 = 2.83
_LR = 0.0
_MAX_STEPS = 2000
_BATCH_SIZE = None

# ---- curriculum -----------------------------------------------------------
_CURRIC_BETA = 0.0       # slope on the batch-standardised difficulty; 0 = off
_CURRIC_SCHED = "linear"  # "const" | "step" | "linear" | "exp"
_CURRIC_FRAC = 0.5       # anneal horizon as a fraction of _MAX_STEPS
_CURRIC_SRC = "len"      # "len" (default, unambiguous) | "ids" (FLAGGED)
_DIGIT_OFFSET = 7        # public generator constant, used only by "ids"


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


def _schedule(step: float) -> float:
    horizon = max(_CURRIC_FRAC * max(_MAX_STEPS, 1), 1.0)
    if _CURRIC_SCHED == "const":
        return 1.0
    if _CURRIC_SCHED == "step":
        return 1.0 if step < horizon else 0.0
    if _CURRIC_SCHED == "linear":
        return max(0.0, 1.0 - step / horizon)
    if _CURRIC_SCHED == "exp":
        return math.exp(-step / horizon)
    raise ValueError(_CURRIC_SCHED)


class Model(nn.Module):
    num_loops = NUM_LOOPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        nn.init.normal_(self.token_embedding.weight, std=EMB_INIT)
        nn.init.normal_(self.position_embedding.weight, std=EMB_INIT)
        self.block = Block()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        # NON-PERSISTENT: excluded from the 5e8 model-state ceiling
        # (benchmark/api.py:26-42).  This is the free place for a schedule.
        self.register_buffer("curric_step", torch.zeros((), dtype=torch.float32),
                             persistent=False)

    # ---- the curriculum ------------------------------------------------
    def _difficulty(self, input_ids: Tensor, attention_mask: Tensor | None):
        """A per-row difficulty scalar, from the input the evaluator hands us.

        "len": the number of valid input tokens.  The prompt is
        `[N] d(N) [X] d(x) [T] d(T)`, so its length is 3 + |d(N)| + |d(x)| +
        |d(T)| -- monotone in both the modulus size and the operand magnitude.
        "ids": log10 of the two number fields, read from the digit tokens.
        FLAGGED: it decodes the prompt format (BRIEF.md §4.3 gray area).  It is
        never used to produce an output, only to weight the loss.
        """
        if attention_mask is None:
            valid = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            valid = attention_mask.bool()
        if _CURRIC_SRC == "len":
            return valid.sum(-1).float()
        digits = (input_ids - _DIGIT_OFFSET).clamp_min(0)
        is_digit = (input_ids >= _DIGIT_OFFSET) & valid
        # field id: 0 before [X], 1 between [X] and [T], 2 after [T]
        field = ((input_ids == 3).cumsum(-1) + (input_ids == 4).cumsum(-1))
        out = 0.0
        for f, weight in ((0, 1.0), (1, 1.0)):
            m = is_digit & (field == f)
            n_dig = m.sum(-1).float()
            # `number_tokens` writes most-significant digit first, so the
            # leading digit is the first masked position of the field.
            first = m.float().argmax(-1, keepdim=True)
            top = digits.gather(1, first).squeeze(1).float().clamp_min(1.0)
            out = out + weight * (n_dig - 1.0 + torch.log10(top))
        return out

    def _row_weight(self, input_ids: Tensor, attention_mask: Tensor | None):
        step = float(self.curric_step.item())
        beta = _CURRIC_BETA * _schedule(step)
        if beta == 0.0:
            return None
        d = self._difficulty(input_ids, attention_mask)
        z = (d - d.mean()) / (d.std() + 1e-6)
        w = torch.exp(-(beta * z).clamp(-20.0, 20.0))
        return w / w.mean()

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for _ in range(NUM_LOOPS):
            x = self.block(x, attention_mask)
        logits = self.head(self.final_norm(x))
        aux = None
        if self.training and _CURRIC_BETA != 0.0:
            w = self._row_weight(input_ids, attention_mask)
            self.curric_step += 1
            if w is not None:
                wv = w.view(-1, 1, 1).to(logits.dtype)
                # Forward value unchanged; the gradient to EVERY upstream
                # parameter (head included -- so this must sit after the head,
                # not before it) is scaled by w, i.e. this row's loss is
                # weighted by w.
                logits = wv * logits + (1.0 - wv) * logits.detach()
                aux = w
        return logits, aux


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def training_loss(logits: Tensor, labels: Tensor, auxiliary) -> Tensor:
    """The plain objective.

    The curriculum is already applied, as a gradient scale, in the forward --
    see the module docstring for why it cannot be applied here: `logits` and
    `labels` arrive flattened by a RAGGED valid mask whose row boundaries are
    not recoverable from these arguments.  Keeping the loss unweighted also
    means the logged training loss is the honest one.
    """
    del auxiliary
    return F.cross_entropy(logits, labels)


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=_LR,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(opt, None)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,
    batch_size=_BATCH_SIZE,
    max_steps=_MAX_STEPS,
)
