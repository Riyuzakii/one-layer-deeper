#!/usr/bin/env python
"""PLAN2 §3.6 falsifier, measured first: is a serial non-linear RNN affordable?

PLAN2's falsifier for the sequential-RNN entry is "step time makes the achievable
update count non-competitive".  This script measures it directly and *relative to a
calibration model*, because this box is sm_107, not the H100 the competition scores on
(BRIEF.md §5: absolute wall clock does not transfer, ratios at fixed shape do).

Calibration anchor (BRIEF2 §4, real hosted Hard run): the `exp_axis` D=128 / 8-loop
recurrent transformer costs **38.6 ms/step at batch 512** on an H100, i.e. ~93,000
steps in the 3600 s Hard budget.  We re-time that exact model here, then quote every
RNN variant as `ratio x 38.6 ms` -> implied H100 steps in 3600 s.

COMPLIANCE: synthetic integer inputs only.  Nothing under data/generated/ is opened.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---- H100 anchor from the one hosted Hard run -------------------------------------
H100_REF_MS = 38.6          # ms/step, batch 512, exp_axis D=128 NUM_LOOPS=8
H100_STARTUP_S = 3.9        # import + construct + .to() + optimizer + first step
HARD_BUDGET_S = 3600.0

VOCAB = 17


# =====================================================================================
# Calibration model: the exact architecture that produced the 38.6 ms/step number.
# =====================================================================================
class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


class RefBlock(nn.Module):
    def __init__(self, d: int, heads: int = 4, ff: int = 4) -> None:
        super().__init__()
        self.d, self.heads = d, heads
        self.attention_norm = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.mixer_norm = RMSNorm(d)
        self.up = nn.Linear(d, ff * d)
        self.down = nn.Linear(ff * d, d)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        residual = x
        x = self.attention_norm(x)
        b, l, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, l, self.heads, -1).transpose(1, 2)
        k = k.view(b, l, self.heads, -1).transpose(1, 2)
        v = v.view(b, l, self.heads, -1).transpose(1, 2)
        m = mask[:, None, None, :].bool() if mask is not None else None
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        x = x.transpose(1, 2).contiguous().view(b, l, self.d)
        x = residual + self.out(x)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


class RefRecurrentTransformer(nn.Module):
    """`submissions/exp_axis/L8_d128_adamw_ce_lr0.001/submission.py`, verbatim shape."""

    def __init__(self, seq_len: int, d: int = 128, loops: int = 8) -> None:
        super().__init__()
        self.loops = loops
        self.token_embedding = nn.Embedding(VOCAB, d)
        self.position_embedding = nn.Embedding(seq_len, d)
        self.block = RefBlock(d)
        self.final_norm = RMSNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(self, ids: Tensor, mask: Tensor | None = None) -> Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(pos)
        for _ in range(self.loops):
            x = self.block(x, mask)
        return self.head(self.final_norm(x))


# =====================================================================================
# The candidate family: a sequential non-linear RNN.
#
# RECURRENCE AXIS (stated explicitly, per BRIEF2 §2a): the recurrence runs over the
# *token / digit-place axis of the prompt*, length <= 21.  It does NOT run over T.
# The decoder scans positions RIGHT-TO-LEFT so its hidden state travels in the carry
# direction (least-significant answer digit first).
# =====================================================================================
class NaiveLSTMScan(nn.Module):
    """One LSTM layer written as a Python loop with a per-step input projection.

    This is the "unfused" reference: 2 matmuls + ~10 pointwise kernels per timestep.
    """

    def __init__(self, d_in: int, d_h: int) -> None:
        super().__init__()
        self.d_h = d_h
        self.wi = nn.Linear(d_in, 4 * d_h)
        self.wh = nn.Linear(d_h, 4 * d_h, bias=False)

    def forward(self, x: Tensor) -> Tensor:  # x: (B, T, d_in)
        b, t, _ = x.shape
        h = x.new_zeros(b, self.d_h)
        c = x.new_zeros(b, self.d_h)
        outs = []
        for i in range(t):
            g = self.wi(x[:, i]) + self.wh(h)
            gi, gf, gg, go = g.chunk(4, dim=-1)
            i_, f_, g_, o_ = (
                torch.sigmoid(gi),
                torch.sigmoid(gf),
                torch.tanh(gg),
                torch.sigmoid(go),
            )
            c = f_ * c + i_ * g_
            h = o_ * torch.tanh(c)
            outs.append(h)
        return torch.stack(outs, dim=1)


class PreprojLSTMScan(nn.Module):
    """Same math, input projection hoisted out of the loop (one big matmul)."""

    def __init__(self, d_in: int, d_h: int) -> None:
        super().__init__()
        self.d_h = d_h
        self.wi = nn.Linear(d_in, 4 * d_h)
        self.wh = nn.Linear(d_h, 4 * d_h, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        xs = self.wi(x)  # (B,T,4H) -- one matmul for the whole sequence
        h = x.new_zeros(b, self.d_h)
        c = x.new_zeros(b, self.d_h)
        outs = []
        for i in range(t):
            g = xs[:, i] + self.wh(h)
            gi, gf, gg, go = g.chunk(4, dim=-1)
            c = torch.sigmoid(gf) * c + torch.sigmoid(gi) * torch.tanh(gg)
            h = torch.sigmoid(go) * torch.tanh(c)
            outs.append(h)
        return torch.stack(outs, dim=1)


class CudnnLSTM(nn.Module):
    """`nn.LSTM` -- cuDNN's fused multi-timestep kernel.  Standard torch, fully legal."""

    def __init__(self, d_in: int, d_h: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(d_in, d_h, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.lstm(x)[0]


class Candidate(nn.Module):
    """biLSTM encoder over the token axis + reverse-scan LSTM decoder (carry axis)."""

    def __init__(self, seq_len: int, d_h: int, kind: str = "cudnn", loops: int = 1):
        super().__init__()
        self.kind, self.loops = kind, loops
        d_in = d_h
        self.token_embedding = nn.Embedding(VOCAB, d_in)
        self.position_embedding = nn.Embedding(seq_len, d_in)
        if kind == "cudnn":
            self.enc = nn.LSTM(d_in, d_h, batch_first=True, bidirectional=True)
            self.dec = nn.LSTM(2 * d_h, d_h, batch_first=True)
        elif kind == "naive":
            self.enc_f = NaiveLSTMScan(d_in, d_h)
            self.enc_b = NaiveLSTMScan(d_in, d_h)
            self.dec = NaiveLSTMScan(2 * d_h, d_h)
        elif kind == "preproj":
            self.enc_f = PreprojLSTMScan(d_in, d_h)
            self.enc_b = PreprojLSTMScan(d_in, d_h)
            self.dec = PreprojLSTMScan(2 * d_h, d_h)
        else:
            raise ValueError(kind)
        self.head = nn.Linear(d_h, VOCAB)

    def _encode(self, x: Tensor) -> Tensor:
        if self.kind == "cudnn":
            return self.enc(x)[0]
        f = self.enc_f(x)
        b = self.enc_b(x.flip(1)).flip(1)
        return torch.cat([f, b], dim=-1)

    def _decode(self, e: Tensor) -> Tensor:
        if self.kind == "cudnn":
            return self.dec(e.flip(1))[0].flip(1)
        return self.dec(e.flip(1)).flip(1)

    def forward(self, ids: Tensor, mask: Tensor | None = None) -> Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(pos)
        for _ in range(self.loops):
            e = self._encode(x)
            d = self._decode(e)
            x = d
        return self.head(x)


# =====================================================================================
# timing
# =====================================================================================
def time_model(model, ids, mask, targets, steps=30, warmup=10, amp=True):
    model = model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, capturable=True)

    def one():
        opt.zero_grad(set_to_none=True)
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if amp
            else torch.autocast(device_type="cuda", enabled=False)
        )
        with ctx:
            logits = model(ids, mask)
            loss = F.cross_entropy(
                logits.float().reshape(-1, VOCAB), targets.reshape(-1)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        one()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / steps * 1e3
    n_par = sum(p.numel() for p in model.parameters())
    del opt, model
    torch.cuda.empty_cache()
    return ms, n_par


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=21, help="21 = m4/Hard-like")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    L, B = args.seq_len, args.batch
    ids = torch.randint(0, VOCAB, (B, L), device=dev)
    mask = torch.ones(B, L, dtype=torch.bool, device=dev)
    tgt = torch.randint(0, VOCAB, (B, L), device=dev)

    rows = []

    ref_ms, ref_par = time_model(
        RefRecurrentTransformer(L), ids, mask, tgt, steps=args.steps
    )
    rows.append(dict(name="REF exp_axis d128 L8", ms=ref_ms, params=ref_par, ratio=1.0))
    print(f"calibration: REF = {ref_ms:.2f} ms/step local  (== {H100_REF_MS} ms on H100)")

    for d_h in (16, 32, 64, 128):
        for kind in ("naive", "preproj", "cudnn"):
            m = Candidate(L, d_h, kind=kind)
            ms, par = time_model(m, ids, mask, tgt, steps=args.steps)
            rows.append(
                dict(name=f"rnn d{d_h} {kind}", ms=ms, params=par, ratio=ms / ref_ms)
            )
            print(f"  d_h={d_h:<4} {kind:<8} {ms:8.2f} ms  ratio {ms/ref_ms:6.3f}")

    # tied-depth sweep on the fused variant
    for loops in (2, 4):
        m = Candidate(L, 64, kind="cudnn", loops=loops)
        ms, par = time_model(m, ids, mask, tgt, steps=args.steps)
        rows.append(
            dict(name=f"rnn d64 cudnn x{loops}", ms=ms, params=par, ratio=ms / ref_ms)
        )
        print(f"  d_h=64   cudnn x{loops}  {ms:8.2f} ms  ratio {ms/ref_ms:6.3f}")

    if args.compile:
        torch._dynamo.config.cache_size_limit = 64
        m = Candidate(L, 64, kind="preproj")
        t0 = time.perf_counter()
        mc = torch.compile(m)
        ms, par = time_model(mc, ids, mask, tgt, steps=args.steps, warmup=3)
        comp_s = time.perf_counter() - t0
        rows.append(
            dict(
                name="rnn d64 preproj+compile",
                ms=ms,
                params=par,
                ratio=ms / ref_ms,
                compile_seconds=comp_s,
            )
        )
        print(f"  d_h=64   compile  {ms:8.2f} ms  ratio {ms/ref_ms:6.3f} "
              f"(compile {comp_s:.1f}s)")

    print()
    print(f"{'variant':<28} {'local ms':>9} {'ratio':>7} {'H100 ms':>9} "
          f"{'steps/3600s':>12}")
    for r in rows:
        h100 = r["ratio"] * H100_REF_MS
        steps = (HARD_BUDGET_S - H100_STARTUP_S) / (h100 / 1e3)
        r["h100_ms_est"] = h100
        r["hard_steps_est"] = steps
        print(f"{r['name']:<28} {r['ms']:9.2f} {r['ratio']:7.3f} {h100:9.2f} "
              f"{steps:12.0f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                dict(seq_len=L, batch=B, ref_local_ms=ref_ms, rows=rows), f, indent=2
            )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
