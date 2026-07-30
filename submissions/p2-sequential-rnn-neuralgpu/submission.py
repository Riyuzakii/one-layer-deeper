"""PLAN2 §3.5 — Neural GPU / tied convolutional GRU.  SECONDARY, timeboxed.

Kaiser & Sutskever (2015).  A convolutional GRU with fully tied weights applied `K`
times to a 2-D state grid of shape `(width, length, channels)`.  Purpose-built for
algorithmic tasks; "one layer deeper" is literally its design axis.

RECURRENCE AXIS.  Depth `K` is a *fixed unrolled tied depth* (PLAN2 Axis B, fourth
bullet); the convolution mixes along the **digit-place axis** of the prompt.  As in
§3.6 the recurrence is not over the task's composition depth `T`, which enters as
input digits.

PLAN2's own warning for this entry is that it is notoriously hard to *optimise*, and
that it needs curriculum learning, gradient noise and careful init.  Two of the three
are available under the competition contract and are used here:

* **gradient noise** — Neelakantan et al.'s `sigma_t = eta / (1+t)^0.55`, delivered
  through a custom `torch.optim.Optimizer`, which BRIEF2 §7 explicitly permits
  ("a custom `torch.optim.Optimizer` is fine").  It is *not* a participant-controlled
  backward: autograd computes every gradient, the optimizer perturbs its own update.
* **careful init** — orthogonal candidate kernel, gate biases set so the carry gate
  starts near-open, which is what keeps a K-step tied recurrence from vanishing.

Curriculum learning is *not* available: the evaluator owns the data order and the
training loop (BRIEF.md §4.4).  That is a real handicap for this architecture and is
reported as such rather than worked around.

COMPLIANCE: every tensor learned from random init; no arithmetic, solver or lookup in
the forward pass; plain cross-entropy; evaluator-owned `optimizer.step()`.
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

# --- knobs -------------------------------------------------------------------------
CHANNELS = 48       # d
WIDTH = 4           # grid height w
K_STEPS = 12        # tied conv-GRU applications
KERNEL = 3
LR = 1e-3
WD = 0.01
GRAD_NOISE_ETA = 0.01   # 0.0 disables the noise; the paper's default regime is 0.01-1.0
GRAD_NOISE_GAMMA = 0.55
MAX_STEPS = None
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 4096


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class CGRU(nn.Module):
    """One tied convolutional GRU cell over a (B, C, W, L) state grid."""

    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        pad = kernel // 2
        self.gates = nn.Conv2d(channels, 2 * channels, kernel, padding=pad)
        self.cand = nn.Conv2d(channels, channels, kernel, padding=pad)
        # near-open carry gate: u = sigmoid(+1) ~ 0.73 keeps the K-step tied
        # recurrence from vanishing at init (Kaiser & Sutskever's "careful init").
        nn.init.zeros_(self.gates.weight)
        nn.init.zeros_(self.gates.bias)
        with torch.no_grad():
            self.gates.bias[:channels].fill_(1.0)    # update gate u
            self.gates.bias[channels:].fill_(1.0)    # reset gate r
        nn.init.orthogonal_(self.cand.weight.view(channels, -1), gain=1.0)
        nn.init.zeros_(self.cand.bias)

    def forward(self, s: Tensor) -> Tensor:
        u, r = self.gates(s).chunk(2, dim=1)
        u = torch.sigmoid(u)
        r = torch.sigmoid(r)
        c = torch.tanh(self.cand(r * s))
        return u * s + (1.0 - u) * c


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.k = K_STEPS
        self.width = WIDTH
        self.channels = CHANNELS
        self.token_embedding = nn.Embedding(spec.vocab_size, CHANNELS)
        self.position_embedding = nn.Embedding(spec.max_seq_len, CHANNELS)
        self.row_embedding = nn.Parameter(torch.zeros(WIDTH, CHANNELS))
        self.cell = CGRU(CHANNELS, KERNEL)
        self.norm = nn.LayerNorm(CHANNELS)
        self.head = nn.Linear(CHANNELS, spec.vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)
        tokens = self.token_embedding(input_ids) + self.position_embedding(positions)
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.any(dim=1)
            tokens = tokens * attention_mask.bool()[..., None]

        # (B, C, W, L): the prompt is written into row 0, other rows are free scratch.
        state = tokens.new_zeros(batch, self.width, length, self.channels)
        state = state + self.row_embedding[None, :, None, :]
        state[:, 0] = state[:, 0] + tokens
        state = state.permute(0, 3, 1, 2).contiguous()

        for _ in range(self.k):
            state = self.cell(state)

        out = state.permute(0, 2, 3, 1)[:, 0]   # read row 0 -> (B, L, C)
        return self.head(self.norm(out)), None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


class NoisyAdamW(torch.optim.AdamW):
    """AdamW plus annealed Gaussian gradient noise (Neelakantan et al., 2015).

    LEGAL: a participant-supplied `torch.optim.Optimizer` is explicitly permitted.
    Autograd still produces every gradient; this only perturbs the update the optimizer
    itself computes, exactly as weight decay or momentum do.
    """

    def __init__(self, *args, noise_eta: float = 0.0, noise_gamma: float = 0.55, **kw):
        super().__init__(*args, **kw)
        self._noise_eta = noise_eta
        self._noise_gamma = noise_gamma
        self._noise_step = 0

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        if self._noise_eta > 0.0:
            self._noise_step += 1
            sigma = math.sqrt(
                self._noise_eta / (1.0 + self._noise_step) ** self._noise_gamma
            )
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.add_(torch.randn_like(param.grad), alpha=sigma)
        return super().step(closure)


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        (no_decay if param.dim() <= 1 else decay).append(param)
    return OptimizerBundle(
        NoisyAdamW(
            [
                {"params": decay, "weight_decay": WD},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=LR,
            betas=(0.9, 0.95),
            capturable=spec.device_type == "cuda",
            noise_eta=GRAD_NOISE_ETA,
            noise_gamma=GRAD_NOISE_GAMMA,
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
