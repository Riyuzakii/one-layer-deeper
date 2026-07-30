"""PLAN2 §3.2 (PD-SSM) and §3.3 (DeltaNet / DeltaProduct) as a submission.

Self-contained: no imports beyond torch + the public ``benchmark`` API.

WHAT AXIS THE RECURRENCE RUNS OVER  (BRIEF2 §2a demands this be explicit)
------------------------------------------------------------------------
The **prompt-token axis**, length 13 (e1-e5) / 15 (m1) / 21 (m4).  Consequences:

* This is a 13-to-21-fold composition.  The task's composition depth ``T <= 64``
  is a *field inside the prompt*, not a sequence dimension, so nothing here
  performs T-fold squaring by scanning.  BRIEF2 §2b says T-fold composition is
  already solved elsewhere and is not the target, so that is deliberate.
* The token axis *contains* the digit-position axis: inside the ``[X] <digits>``
  field, consecutive tokens are consecutive digits of x.  Carry propagation
  across digit positions is a genuine associative prefix computation
  (BRIEF2 §2c), and a scan over prompt tokens subsumes a scan over digits.
  That is the mechanism this branch is actually betting on.
* ``P2_REPEAT`` applies the whole tied block R times, giving an *unrolled
  internal depth* axis on top of the token axis (composition depth R*L).

EIGENVALUE RANGE  (PLAN2 §3.3's flagged trap)
---------------------------------------------
``I - beta k k^T`` with ``||k|| = 1`` has spectrum ``{1-beta} u {1}^(d-1)``.
``P2_EIG=neg`` (the default) uses ``beta = 2*sigmoid(.) in (0,2)``, i.e. exactly
PLAN2's ``I - 2 beta k k^T`` with ``beta in [0,1]``, so eigenvalues live in
``(-1,1)``.  ``P2_EIG=pos`` is the deliberately-crippled ``[0,1]`` control.

CONFIGURATION.  Every knob has a default; with no environment set this file is a
fixed, deterministic submission (``delta``, ``n_h=2``, ``eig=neg``, 2 layers).
The ``P2_*`` environment variables exist so one file can serve a whole lab sweep.
``P2_DIAG_FILE`` (unset by default) is a LAB-ONLY diagnostic sink and is never
set in a scored run.

COMPLIANCE: no arithmetic, solver, or lookup in the forward pass; every tensor
is learned from random init; end-to-end differentiable; no custom training loop.
"""

from __future__ import annotations

import json
import math
import os

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


def _env(name: str, default, cast=str):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


ARCH = _env("P2_ARCH", "delta")            # delta | pdssm | matscan
N_H = _env("P2_NH", 2, int)                # DeltaProduct Householders per token
EIG = _env("P2_EIG", "neg")                # neg -> (-1,1)  |  pos -> (0,1)
TAU = _env("P2_TAU", 1.0, float)           # PD-SSM straight-through temperature
STE = _env("P2_STE", "hard")               # hard | none  (none = soft control)
N_LAYERS = _env("P2_LAYERS", 2, int)
REPEAT = _env("P2_REPEAT", 1, int)         # tied unrolled internal depth
D_MODEL = _env("P2_DMODEL", 128, int)
N_HEADS = _env("P2_HEADS", 4, int)
STATE = _env("P2_STATE", 16, int)          # PD-SSM / matscan state size N
HEAD_DIM = _env("P2_HEADDIM", 32, int)     # DeltaNet head dim
LR = _env("P2_LR", 1e-3, float)            # P2_LR=0 -> the mandatory control
WD = _env("P2_WD", 0.1, float)
BATCH = _env("P2_BATCH", 128, int)
BIDIR = _env("P2_BIDIR", 1, int) == 1
IMPL = _env("P2_IMPL", "fast")             # fast (chunkwise/log-depth) | seq (reference)
EMB_STD = _env("P2_EMB_STD", 0.02, float)  # tied head -> N(0,1) gives initial loss ~80
DIAG_FILE = os.environ.get("P2_DIAG_FILE")  # lab-only
DIAG_EVERY = _env("P2_DIAG_EVERY", 100, int)


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


# --------------------------------------------------------------------------- #
# §3.3  DeltaNet / DeltaProduct
# --------------------------------------------------------------------------- #
class DeltaProductMixer(nn.Module):
    def __init__(self, d_model, *, n_heads, head_dim, n_h, eig_range, bidirectional):
        super().__init__()
        self.n_heads, self.head_dim, self.n_h = n_heads, head_dim, n_h
        self.eig_range = eig_range
        self.bidirectional = bidirectional
        self.beta_max = 1.0 if eig_range == "pos" else 2.0
        inner = n_heads * head_dim
        self.n_dir = 2 if bidirectional else 1
        self.q_proj = nn.Linear(d_model, inner * self.n_dir, bias=False)
        self.k_proj = nn.Linear(d_model, inner * n_h * self.n_dir, bias=False)
        self.v_proj = nn.Linear(d_model, inner * n_h * self.n_dir, bias=False)
        self.b_proj = nn.Linear(d_model, n_heads * n_h * self.n_dir, bias=True)
        self.out_proj = nn.Linear(inner * self.n_dir, d_model, bias=False)
        self.out_proj.weight.data.mul_(0.05)
        nn.init.zeros_(self.b_proj.bias)

    def _betas(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        z = self.b_proj(x).float().view(b, t, self.n_dir, self.n_heads, self.n_h)
        return self.beta_max * torch.sigmoid(z)

    def transition_eigenvalues(self, x: Tensor) -> Tensor:
        return (1.0 - self._betas(x)).reshape(-1).detach().float()

    def _scan(self, q, k, v, beta):
        """Reference sequential delta rule.  O(S) kernel launches."""
        b, h, _, dk = k.shape
        t = q.shape[2]
        state = k.new_zeros(b, h, dk, dk)
        outs = []
        idx = 0
        for step in range(t):
            for _ in range(self.n_h):
                ki, vi = k[:, :, idx], v[:, :, idx]
                bi = beta[:, :, idx].unsqueeze(-1)
                sk = torch.einsum("bhij,bhj->bhi", state, ki)
                state = state + torch.einsum("bhi,bhj->bhij", bi * (vi - sk), ki)
                idx += 1
            outs.append(torch.einsum("bhij,bhj->bhi", state, q[:, :, step]))
        return torch.stack(outs, dim=2)

    def _scan_wy(self, q, k, v, beta):
        """Chunkwise (WY / UT-transform) delta rule -- one chunk = whole sequence.

        S_i = S_{i-1} + u_i k_i^T with u_i = beta_i (v_i - S_{i-1} k_i), so
            (I + diag(beta) tril(K K^T, -1)) U = diag(beta) V
        (unit lower triangular; S_0 = 0).  Then o_t = sum_{j<=(t+1)n_h} u_j (k_j.q_t).

        O(1) kernel launches instead of O(S) -- the whole point of §3.3's
        "mature chunkwise-parallel training algorithms exist".
        """
        b, h, s, dk = k.shape
        t = q.shape[2]
        kk = torch.matmul(k, k.transpose(-1, -2))  # (B,H,S,S)
        low = torch.tril(kk, diagonal=-1)
        m = torch.eye(s, device=k.device, dtype=k.dtype) + beta.unsqueeze(-1) * low
        u = torch.linalg.solve_triangular(
            m, beta.unsqueeze(-1) * v, upper=False, unitriangular=True
        )
        attn = torch.matmul(q, k.transpose(-1, -2))  # (B,H,T,S)
        j = torch.arange(s, device=k.device)
        tt = torch.arange(t, device=k.device)
        causal = (j.view(1, s) < ((tt.view(t, 1) + 1) * self.n_h)).to(k.dtype)
        return torch.matmul(attn * causal, u)

    def _one_direction(self, x, d):
        b, t, _ = x.shape
        h, dk, n_h = self.n_heads, self.head_dim, self.n_h
        inner = h * dk
        q = self.q_proj(x).float().view(b, t, self.n_dir, inner)[:, :, d]
        k = self.k_proj(x).float().view(b, t, self.n_dir, n_h * inner)[:, :, d]
        v = self.v_proj(x).float().view(b, t, self.n_dir, n_h * inner)[:, :, d]
        beta = self._betas(x)[:, :, d]
        q = F.normalize(q.view(b, t, h, dk).transpose(1, 2), dim=-1)
        k = F.normalize(
            k.view(b, t, n_h, h, dk).permute(0, 3, 1, 2, 4).reshape(b, h, t * n_h, dk),
            dim=-1,
        )
        v = v.view(b, t, n_h, h, dk).permute(0, 3, 1, 2, 4).reshape(b, h, t * n_h, dk)
        beta = beta.permute(0, 2, 1, 3).reshape(b, h, t * n_h)
        scan = self._scan_wy if IMPL == "fast" else self._scan
        return scan(q, k, v, beta).transpose(1, 2).reshape(b, t, inner)

    def forward(self, x, mask=None):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            if mask is not None:
                x = x * mask.unsqueeze(-1).to(x.dtype)
            outs = [self._one_direction(x, 0)]
            if self.bidirectional:
                outs.append(
                    torch.flip(
                        self._one_direction(torch.flip(x, dims=(1,)), 1), dims=(1,)
                    )
                )
            return self.out_proj(torch.cat(outs, dim=-1))


# --------------------------------------------------------------------------- #
# §3.2  PD-SSM
# --------------------------------------------------------------------------- #
class PDSSMMixer(nn.Module):
    def __init__(self, d_model, *, n_heads, state_size, tau, ste, bidirectional):
        super().__init__()
        self.n_heads, self.n, self.tau, self.ste = n_heads, state_size, tau, ste
        self.bidirectional = bidirectional
        self.n_dir = 2 if bidirectional else 1
        h, n = n_heads, state_size
        self.p_proj = nn.Linear(d_model, self.n_dir * h * n * n, bias=True)
        self.d_proj = nn.Linear(d_model, self.n_dir * h * n * 2, bias=True)
        self.b_proj = nn.Linear(d_model, self.n_dir * h * n * 2, bias=False)
        self.c_proj = nn.Linear(self.n_dir * h * n * 2, d_model, bias=False)
        self.c_proj.weight.data.mul_(0.05)
        with torch.no_grad():
            self.d_proj.bias.view(self.n_dir, h, n, 2)[..., 0].fill_(2.0)
            self.d_proj.bias.view(self.n_dir, h, n, 2)[..., 1].zero_()
            self.d_proj.weight.mul_(0.1)
            self.p_proj.weight.mul_(0.1)

    def transition_stats(self, x: Tensor) -> dict:
        b, t, _ = x.shape
        h, n = self.n_heads, self.n
        logits = self.p_proj(x).float().view(b, t, self.n_dir, h, n, n)
        soft = torch.softmax(logits / self.tau, dim=-2)
        d = self.d_proj(x).float().view(b, t, self.n_dir, h, n, 2)
        onehot = soft.argmax(dim=-2).reshape(-1, n)
        step = max(1, onehot.shape[0] // 256)
        uniq = [len(torch.unique(r)) / n for r in onehot[::step]]
        return {
            "p_max_prob": soft.amax(dim=-2).mean().item(),
            "p_entropy": (-(soft * (soft + 1e-9).log()).sum(-2)).mean().item(),
            "p_perm_frac": float(sum(uniq) / max(1, len(uniq))),
            "d_abs_mean": torch.sigmoid(d[..., 0]).mean().item(),
            "d_abs_max": torch.sigmoid(d[..., 0]).amax().item(),
            "d_theta_absmean": d[..., 1].abs().mean().item(),
        }

    @staticmethod
    def _affine_scan(are, aim, bre, bim):
        """Log-depth prefix scan of the affine complex recurrence
        ``h_t = A_t h_{t-1} + b_t``, A stored as (real, imag) parts.

        Element composition ``(A_j,b_j) o (A_i,b_i) = (A_j A_i, A_j b_i + b_j)``.
        ``ceil(log2 T)`` levels instead of ``T`` sequential steps -- the "one
        layer, log T depth" claim, actually realised.
        """
        t, n = are.shape[2], are.shape[-1]
        eye = torch.eye(n, device=are.device, dtype=are.dtype)
        eye = eye.expand(are.shape[0], are.shape[1], 1, n, n)
        zer = torch.zeros_like(eye)
        zb = torch.zeros_like(bre[:, :, :1])
        shift = 1
        while shift < t:
            pare = torch.cat((eye.expand(-1, -1, shift, -1, -1), are[:, :, :-shift]), 2)
            paim = torch.cat((zer.expand(-1, -1, shift, -1, -1), aim[:, :, :-shift]), 2)
            pbre = torch.cat((zb.expand(-1, -1, shift, -1), bre[:, :, :-shift]), 2)
            pbim = torch.cat((zb.expand(-1, -1, shift, -1), bim[:, :, :-shift]), 2)
            nre = torch.matmul(are, pare) - torch.matmul(aim, paim)
            nim = torch.matmul(are, paim) + torch.matmul(aim, pare)
            obre = bre + (
                torch.einsum("bhtij,bhtj->bhti", are, pbre)
                - torch.einsum("bhtij,bhtj->bhti", aim, pbim)
            )
            obim = bim + (
                torch.einsum("bhtij,bhtj->bhti", are, pbim)
                + torch.einsum("bhtij,bhtj->bhti", aim, pbre)
            )
            are, aim, bre, bim = nre, nim, obre, obim
            shift *= 2
        return bre, bim

    def _one_direction(self, x, dir_i):
        b, t, _ = x.shape
        h, n = self.n_heads, self.n
        logits = self.p_proj(x).float().view(b, t, self.n_dir, h, n, n)[:, :, dir_i]
        soft = torch.softmax(logits / self.tau, dim=-2)
        if self.ste == "hard":
            idx = soft.argmax(dim=-2, keepdim=True)
            hard = torch.zeros_like(soft).scatter_(-2, idx, 1.0)
            p = hard + soft - soft.detach()
        else:
            p = soft
        dpar = self.d_proj(x).float().view(b, t, self.n_dir, h, n, 2)[:, :, dir_i]
        r = torch.sigmoid(dpar[..., 0])
        theta = dpar[..., 1]
        dre, dim_ = r * torch.cos(theta), r * torch.sin(theta)
        bx = self.b_proj(x).float().view(b, t, self.n_dir, h, n, 2)[:, :, dir_i]

        if IMPL == "fast":
            # A_t = P_t D_t  ->  columns of P scaled by the diagonal entry
            are = p * dre.unsqueeze(-2)
            aim = p * dim_.unsqueeze(-2)
            hre, him = self._affine_scan(
                are.transpose(1, 2), aim.transpose(1, 2),
                bx[..., 0].transpose(1, 2), bx[..., 1].transpose(1, 2),
            )
            out = torch.stack((hre, him), dim=-1).transpose(1, 2)
            return out.reshape(b, t, h * n * 2)

        hre = x.new_zeros(b, h, n)
        him = x.new_zeros(b, h, n)
        outs = []
        for s in range(t):
            ure = dre[:, s] * hre - dim_[:, s] * him
            uim = dre[:, s] * him + dim_[:, s] * hre
            pm = p[:, s]
            hre = torch.einsum("bhij,bhj->bhi", pm, ure) + bx[:, s, ..., 0]
            him = torch.einsum("bhij,bhj->bhi", pm, uim) + bx[:, s, ..., 1]
            outs.append(torch.stack((hre, him), dim=-1))
        return torch.stack(outs, dim=1).reshape(b, t, h * n * 2)

    def forward(self, x, mask=None):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            if mask is not None:
                x = x * mask.unsqueeze(-1).to(x.dtype)
            outs = [self._one_direction(x, 0)]
            if self.bidirectional:
                outs.append(
                    torch.flip(
                        self._one_direction(torch.flip(x, dims=(1,)), 1), dims=(1,)
                    )
                )
            return self.c_proj(torch.cat(outs, dim=-1))


# --------------------------------------------------------------------------- #
# §3.1 reference -- dense matrix associative scan (local instantiation)
# --------------------------------------------------------------------------- #
class MatrixScanMixer(nn.Module):
    def __init__(self, d_model, *, n_heads, state_size, bidirectional, rank=4):
        super().__init__()
        self.n_heads, self.n, self.rank = n_heads, state_size, rank
        self.bidirectional = bidirectional
        self.n_dir = 2 if bidirectional else 1
        h, n = n_heads, state_size
        self.u_proj = nn.Linear(d_model, self.n_dir * h * n * rank, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_dir * h * n * rank, bias=False)
        self.b_proj = nn.Linear(d_model, self.n_dir * h * n, bias=False)
        self.c_proj = nn.Linear(self.n_dir * h * n, d_model, bias=False)
        self.c_proj.weight.data.mul_(0.05)
        self.u_proj.weight.data.mul_(0.5)
        self.v_proj.weight.data.mul_(0.5)

    @staticmethod
    def _prefix_products(m: Tensor) -> Tensor:
        t, n = m.shape[2], m.shape[-1]
        eye = torch.eye(n, device=m.device, dtype=m.dtype)
        eye = eye.expand(m.shape[0], m.shape[1], 1, n, n)
        shift = 1
        while shift < t:
            shifted = torch.cat((eye.expand(-1, -1, shift, -1, -1), m[:, :, :-shift]), 2)
            m = torch.matmul(m, shifted)
            shift *= 2
        return m

    def _one_direction(self, x, dir_i):
        b, t, _ = x.shape
        h, n, r = self.n_heads, self.n, self.rank
        u = self.u_proj(x).float().view(b, t, self.n_dir, h, n, r)[:, :, dir_i]
        v = self.v_proj(x).float().view(b, t, self.n_dir, h, n, r)[:, :, dir_i]
        m = torch.eye(n, device=x.device, dtype=torch.float32) + torch.einsum(
            "bthnr,bthmr->bthnm", u, v
        ) / math.sqrt(r)
        p = self._prefix_products(m.permute(0, 2, 1, 3, 4))
        bx = self.b_proj(x).float().view(b, t, self.n_dir, h, n)[:, :, dir_i]
        return torch.einsum("bhtnm,bthm->bthn", p, bx).reshape(b, t, h * n)

    def forward(self, x, mask=None):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            if mask is not None:
                x = x * mask.unsqueeze(-1).to(x.dtype)
            outs = [self._one_direction(x, 0)]
            if self.bidirectional:
                outs.append(
                    torch.flip(
                        self._one_direction(torch.flip(x, dims=(1,)), 1), dims=(1,)
                    )
                )
            return self.c_proj(torch.cat(outs, dim=-1))


def _make_mixer() -> nn.Module:
    if ARCH == "delta":
        return DeltaProductMixer(
            D_MODEL, n_heads=N_HEADS, head_dim=HEAD_DIM, n_h=N_H,
            eig_range=EIG, bidirectional=BIDIR,
        )
    if ARCH == "pdssm":
        return PDSSMMixer(
            D_MODEL, n_heads=N_HEADS, state_size=STATE, tau=TAU, ste=STE,
            bidirectional=BIDIR,
        )
    if ARCH == "matscan":
        return MatrixScanMixer(
            D_MODEL, n_heads=N_HEADS, state_size=STATE, bidirectional=BIDIR
        )
    raise ValueError(f"unknown P2_ARCH={ARCH}")


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n1 = RMSNorm(D_MODEL)
        self.mixer = _make_mixer()
        self.n2 = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x, mask):
        x = x + self.mixer(self.n1(x), mask)
        return x + self.down(F.gelu(self.up(self.n2(x))))


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        # The head is tied to the embedding, so nn.Embedding's default N(0,1) init
        # makes the initial logits std ~sqrt(D_MODEL) and the initial loss ~80
        # instead of ln(17)=2.83.  Measured: ~100 of 1500 steps are spent undoing
        # that.  At a tier-faithful Easy budget (14-18 steps, RESUME P3) it would
        # consume the whole run.  All three architectures share this scaffold, so
        # this does not favour any of them.
        nn.init.normal_(self.token_embedding.weight, std=EMB_STD)
        nn.init.normal_(self.position_embedding.weight, std=EMB_STD)
        self.register_buffer("_calls", torch.zeros((), dtype=torch.long), persistent=False)

    def forward(self, input_ids, attention_mask=None):
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        mask = None
        if attention_mask is not None and attention_mask.dim() == 2:
            mask = attention_mask.to(torch.bool)
        for _ in range(REPEAT):
            for blk in self.blocks:
                x = blk(x, mask)
        logits = self.head(self.final_norm(x))
        if DIAG_FILE is not None and self.training:
            self._diag(logits)
        return logits, None

    # ---- LAB-ONLY diagnostics; inert unless P2_DIAG_FILE is set ------------ #
    @torch.no_grad()
    def _diag(self, logits: Tensor) -> None:
        self._calls += 1
        n = int(self._calls.item())
        if n % DIAG_EVERY != 0:
            return
        probs = logits.float().softmax(-1)
        pred = logits.argmax(-1)
        vocab = logits.shape[-1]
        # collapse detector: distinct predicted tokens / vocab, and the mass of
        # the single most-predicted token.  A constant map reads ~1/vocab and 1.0.
        counts = torch.bincount(pred.reshape(-1), minlength=vocab).float()
        rec = {
            "call": n,
            "arch": ARCH, "n_h": N_H, "eig": EIG, "tau": TAU, "ste": STE,
            "repeat": REPEAT, "layers": N_LAYERS, "lr": LR,
            "out_diversity": float((counts > 0).float().mean()),
            "top_token_share": float(counts.max() / counts.sum()),
            "pred_entropy": float(-(probs * (probs + 1e-9).log()).sum(-1).mean()),
            "max_prob": float(probs.amax(-1).mean()),
        }
        mixer = self.blocks[0].mixer
        if hasattr(mixer, "transition_eigenvalues"):
            ev = mixer.transition_eigenvalues(self._last_h)
            rec.update(
                eig_min=float(ev.min()), eig_max=float(ev.max()),
                eig_frac_neg=float((ev < 0).float().mean()),
                eig_mean=float(ev.mean()),
            )
        elif hasattr(mixer, "transition_stats"):
            rec.update(mixer.transition_stats(self._last_h))
        with open(DIAG_FILE, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    if DIAG_FILE is not None:
        # capture the mixer input so diagnostics see real activations
        blk = model.blocks[0]
        orig = blk.mixer.forward

        def wrapped(x, mask=None, _orig=orig, _m=model):
            _m._last_h = x.detach().float()
            return _orig(x, mask)

        blk.mixer.forward = wrapped
        model._last_h = torch.zeros(1, 1, D_MODEL)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(
        torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            betas=(0.9, 0.95),
            weight_decay=WD,
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=BATCH,
    eval_batch_size=1024,
)
