#!/usr/bin/env python
"""Probe a trained group-rotation model on SYNTHETIC prompts.

Compliance: this script never opens anything under `data/generated/`.  It builds
its own prompts from the *public tokenizer spec* in `data/squaring_mod.py`
(`[N] digits(N) [X] digits(x) [T] digits(T)`, most-significant digit first,
vocab 17) and analyses the trained model's own weights and activations.  The
modulus is passed on the command line; it is public configuration (it is in the
dataset's directory name and in lab/BRIEF.md), not data.

Probes
  1. exact-digit accuracy over *every* unit of Z*_N, per ladder rung T.
  2. is the pooled prompt representation a *value*?  R^2 of  u(x) ~ x  and of
     the multiplicative interaction  (u*v)(x) ~ x^2.
  3. are the learned phases group-structured?  R^2 of theta_k(x) ~ x^2 and
     whether the implied frequency w_k satisfies  w_k * N / 2pi ~ integer
     (i.e. whether "mod N" is free in the learned representation).
  4. spectral concentration of cos(theta_k(x)) over x (a Fourier solution is
     sparse in this basis) and singular values of the learned tables.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
DIGIT_OFFSET = 7
TOK_N, TOK_X, TOK_T = 2, 3, 4


def number_tokens(value: int) -> list[int]:
    return [DIGIT_OFFSET + int(c) for c in str(value)]


def build_prompt(modulus: int, x: int, time_steps: int) -> list[int]:
    return (
        [TOK_N]
        + number_tokens(modulus)
        + [TOK_X]
        + number_tokens(x)
        + [TOK_T]
        + number_tokens(time_steps)
    )


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("gr_probe_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_batch(prompts: list[list[int]], max_len: int, device):
    n = len(prompts)
    ids = torch.zeros(n, max_len, dtype=torch.long)
    mask = torch.zeros(n, max_len, dtype=torch.bool)
    for i, p in enumerate(prompts):
        ids[i, : len(p)] = torch.tensor(p, dtype=torch.long)
        mask[i, : len(p)] = True
    return ids.to(device), mask.to(device)


def r2(y: np.ndarray, design: np.ndarray) -> float:
    """R^2 of a least-squares fit of y on `design` (columns include a bias)."""
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    denom = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float((resid**2).sum()) / denom if denom > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--modulus", type=int, required=True, help="public config, e.g. 323")
    ap.add_argument("--p", type=int, default=None, help="optional factor, for phi")
    ap.add_argument("--ladder", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    module = load_module(Path(args.submission).resolve())
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    # max_seq_len is recoverable from the model's own table shapes -- no dataset.
    max_len = None
    for key, value in state.items():
        if key.endswith("position_embedding.weight"):
            max_len = value.shape[0]
        if "tables.0.weight" in key or key.endswith("attn.embed.weight"):
            max_len = value.shape[0] // 17
    if max_len is None:
        raise SystemExit("could not infer max_seq_len from checkpoint")

    spec = module.ModelSpec(
        vocab_size=17, max_seq_len=max_len, maximum_model_state_elements=500_000_000
    )
    model = module.build_model(spec)
    model.load_state_dict(state)
    device = torch.device(args.device)
    model.to(device).eval()

    modulus = args.modulus
    units = [x for x in range(1, modulus) if math.gcd(x, modulus) == 1]
    print(f"modulus={modulus} units={len(units)} max_seq_len={max_len}")

    captured: dict[str, torch.Tensor] = {}

    def grab(name):
        def hook(_m, _i, out):
            captured[name] = out.detach().float().cpu()

        return hook

    handles = []
    for name in ("pool", "to_theta"):
        sub = getattr(model, name, None)
        if sub is not None:
            handles.append(sub.register_forward_hook(grab(name)))

    # ---- probe 1: exact-digit accuracy over every unit, per rung ----
    print("\n[1] exact accuracy over ALL units (synthetic prompts), per rung T")
    theta_by_x = None
    pool_by_x = None
    for time_steps in args.ladder:
        prompts, answers = [], []
        for x in units:
            if args.p is not None:
                q = modulus // args.p
                phi = (args.p - 1) * (q - 1)
                y = pow(x, pow(2, time_steps, phi), modulus)
            else:
                y = x
                for _ in range(time_steps):
                    y = (y * y) % modulus
            prompts.append(build_prompt(modulus, x, time_steps))
            answers.append(y)
        width = max(len(p) for p in prompts)
        ids, mask = make_batch(prompts, max(width, max_len), device)
        with torch.no_grad():
            logits, _ = model(ids[:, :max_len] if ids.shape[1] > max_len else ids,
                              mask[:, :max_len] if mask.shape[1] > max_len else mask)
        pred = logits.argmax(-1).cpu()
        lengths = mask.sum(1).cpu()
        correct = 0
        for i, y in enumerate(answers):
            want = number_tokens(y)
            end = int(lengths[i])
            got = pred[i, end - len(want) : end].tolist()
            correct += int(got == want)
        print(f"    T={time_steps:>2}  exact={correct / len(units):.4f}  ({correct}/{len(units)})")
        if time_steps == args.ladder[0]:
            theta_by_x = captured.get("to_theta")
            pool_by_x = captured.get("pool")

    for h in handles:
        h.remove()

    xs = np.array(units, dtype=np.float64)
    design = np.stack([xs**2, xs, np.ones_like(xs)], axis=1)
    design_lin = np.stack([xs, np.ones_like(xs)], axis=1)

    # ---- probe 2: is the pooled prompt representation a value? ----
    if pool_by_x is not None:
        pooled = pool_by_x.numpy()  # (B, n_pool, W)
        print("\n[2] pooled prompt representation")
        u = pooled[:, 0]
        best_u = max(r2(u[:, c], design_lin) for c in range(u.shape[1]))
        print(f"    best channel  R^2(u ~ a*x + b)      = {best_u:.4f}")
        if pooled.shape[1] > 1:
            prod = pooled[:, 0] * pooled[:, 1]
            best_p = max(r2(prod[:, c], design) for c in range(prod.shape[1]))
            print(f"    best channel  R^2(u*v ~ x^2,x,1)   = {best_p:.4f}")

    # ---- probe 3 & 4: are the phases group-structured? ----
    if theta_by_x is not None:
        theta = theta_by_x.numpy()
        scale = state.get("theta_scale")
        if scale is not None:
            theta = theta * np.exp(scale.numpy())[None, :]
        print("\n[3] learned phases theta_k(x)")
        rows = []
        for k in range(theta.shape[1]):
            yk = theta[:, k]
            coef, *_ = np.linalg.lstsq(design, yk, rcond=None)
            resid = yk - design @ coef
            denom = float(((yk - yk.mean()) ** 2).sum())
            fit = 1.0 - float((resid**2).sum()) / denom if denom > 0 else 0.0
            w = coef[0]  # coefficient on x^2
            cycles = abs(w) * modulus / (2 * math.pi)
            rows.append((fit, w, cycles, abs(cycles - round(cycles))))
        rows.sort(key=lambda r: -r[0])
        good = [r for r in rows if r[0] > 0.99]
        print(f"    channels with R^2(theta_k ~ x^2) > 0.99 : {len(good)}/{len(rows)}")
        print("    top 8 channels (R^2, w, w*N/2pi, dist-to-integer):")
        for fit, w, cycles, dist in rows[:8]:
            print(f"       R2={fit:.4f}  w={w: .6g}  cycles={cycles: .4f}  d={dist:.4f}")
        if good:
            dists = np.array([r[3] for r in good])
            print(
                f"    among those, mean |w*N/2pi - nearest int| = {dists.mean():.4f} "
                f"(0 => 'mod N' is free in the representation; 0.25 => unrelated)"
            )

        print("\n[4] spectral concentration of cos(theta_k) over the unit group")
        full = np.zeros((modulus, theta.shape[1]))
        idx = np.array(units)
        full[idx] = np.cos(theta)
        spec = np.abs(np.fft.rfft(full - full.mean(0, keepdims=True), axis=0)) ** 2
        power = spec / np.maximum(spec.sum(0, keepdims=True), 1e-12)
        pr = 1.0 / np.maximum((power**2).sum(0), 1e-12)  # participation ratio
        print(
            f"    participation ratio over {spec.shape[0]} freqs: "
            f"median={np.median(pr):.1f} min={pr.min():.1f} "
            f"(small => sparse/Fourier; ~{spec.shape[0] / 3:.0f} => dense/dense-memorisation)"
        )

    print("\n[5] singular-value structure of the learned tables")
    for key, value in state.items():
        if value.ndim == 2 and min(value.shape) > 2:
            sv = torch.linalg.svdvals(value.float())
            energy = (sv**2 / (sv**2).sum()).cumsum(0)
            rank90 = int((energy < 0.9).sum()) + 1
            print(f"    {key:<40} shape={tuple(value.shape)} rank@90%={rank90}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
