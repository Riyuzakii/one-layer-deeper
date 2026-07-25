#!/usr/bin/env python
"""LAB ONLY offline study of the single squaring step, stripped of the prompt.

The crux of the group-rotation hypothesis is this question:

    can a phase-based, pairwise (outer-product) map learn
        digits(v)  ->  digits(v^2 mod N)
    from ~250 examples and get the remaining ~38 values of Z*_N right?

If yes, the mechanism is real and the remaining work is prompt handling and
iteration.  If no, the whole family is blocked at its first step and no amount
of T-conditioning helps.

Self-generated data only; nothing under data/generated/ is opened.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

INIT_NOISE = 0.0
from torch import nn


class PairPhaseStep(nn.Module):
    """theta = sum_{i<=j} Q[i,j] . (s_i (x) s_j)   ->   (cos, sin)  ->  per-slot digits.

    Exactly the structure of  v^2 = sum_{i,j} 10^(i+j) d_i d_j  carried in phase,
    so that reduction mod N is free.  All parameters are learned.
    """

    def __init__(self, slots: int, n_freq: int, n_harm: int, linear_only: bool = False):
        super().__init__()
        self.slots = slots
        self.linear_only = linear_only
        pairs = [(i, j) for i in range(slots) for j in range(i, slots)]
        self.pairs = pairs
        if linear_only:
            self.table = nn.Parameter(torch.randn(slots, 10, n_freq) * 0.5)
        else:
            self.table = nn.Parameter(torch.randn(len(pairs), 10, 10, n_freq) * 0.5)
        self.n_harm = n_harm
        feat = 2 * n_harm * n_freq
        self.readout = nn.Parameter(torch.randn(slots, feat, 10) * feat**-0.5)
        self.bias = nn.Parameter(torch.zeros(slots, 10))

    def forward(self, s):  # s: (B, slots, 10) soft digits
        if self.linear_only:
            theta = torch.einsum("bia,iaf->bf", s, self.table)
        else:
            theta = 0.0
            for n, (i, j) in enumerate(self.pairs):
                theta = theta + torch.einsum(
                    "ba,bc,acf->bf", s[:, i], s[:, j], self.table[n]
                )
        parts = []
        for h in range(1, self.n_harm + 1):
            parts.append(torch.cos(theta * h))
            parts.append(torch.sin(theta * h))
        feats = torch.cat(parts, dim=-1)
        return torch.einsum("bf,ifd->bid", feats, self.readout) + self.bias


class FactoredPhaseStep(nn.Module):
    """The same pairwise phase map, but the table is forced into its rank-1
    "place x digit x frequency" factorisation:

        u      = sum_i  c_i * e[d_i]          (a learned scalar-valued pool)
        theta_k = omega_k * u^2 + nu_k * u

    That is 10 + slots + 2K learnable numbers in the whole phase pathway, so the
    phase *cannot* memorise individual values -- it can only be a quadratic form
    in a learned linear digit code.  The exact solution (e[d]=d, c_i=10^i,
    omega_k = 2*pi*k/N) lives inside this class; nothing is hard-coded.
    """

    def __init__(self, slots: int, n_freq: int, n_harm: int):
        super().__init__()
        self.digit = nn.Parameter(torch.randn(10) * 0.5)
        self.place = nn.Parameter(torch.randn(slots) * 0.5)
        self.omega = nn.Parameter(torch.randn(n_freq) * 0.1)
        self.nu = nn.Parameter(torch.randn(n_freq) * 0.1)
        self.n_harm = n_harm
        feat = 2 * n_harm * n_freq
        self.readout = nn.Parameter(torch.randn(slots, feat, 10) * feat**-0.5)
        self.bias = nn.Parameter(torch.zeros(slots, 10))

    def forward(self, s):
        u = torch.einsum("bia,a,i->b", s, self.digit, self.place)
        theta = u[:, None] ** 2 * self.omega[None, :] + u[:, None] * self.nu[None, :]
        parts = []
        for h in range(1, self.n_harm + 1):
            parts.append(torch.cos(theta * h))
            parts.append(torch.sin(theta * h))
        feats = torch.cat(parts, dim=-1)
        return torch.einsum("bf,ifd->bid", feats, self.readout) + self.bias


class ModFreePhaseStep(nn.Module):
    """Scale-free additive characters, with the modulus read from the prompt.

    A learned linear digit code  code(v) = sum_i c_i e[d_i]  equals  s*v  for an
    unknown learned scale s.  Then

        code(x)^2 / (code(N) * code(1))  =  (s x)^2 / (s N * s)  =  x^2 / N

    -- the scale cancels exactly.  Wrapping that in cos/sin of INTEGER multiples
    of 2*pi makes periodicity mod N structural rather than something the
    optimiser has to discover to 1 part in 1e5.  No modulus, no mod operation and
    no exponent appears; c, e and the readout are all learned.
    """

    def __init__(self, slots: int, n_slots_mod: int, n_harm: int):
        super().__init__()
        self.digit = nn.Parameter(torch.arange(10.0) / 10.0 + torch.randn(10) * INIT_NOISE)
        self.place = nn.Parameter(
            torch.tensor([10.0**i for i in range(max(slots, n_slots_mod))]) / 10.0
            * (1.0 + torch.randn(max(slots, n_slots_mod)) * INIT_NOISE)
        )
        self.slots = slots
        self.n_harm = n_harm
        feat = 2 * n_harm
        self.readout = nn.Parameter(torch.randn(slots, feat, 10) * feat**-0.5)
        self.bias = nn.Parameter(torch.zeros(slots, 10))

    def code(self, s):
        n = s.shape[1]
        return torch.einsum("bia,a,i->b", s, self.digit, self.place[:n])

    def forward(self, s, s_mod):
        ux = self.code(s)
        un = self.code(s_mod)
        u1 = self.place[0] * self.digit[1]
        denom = un * u1
        base = ux * ux / torch.sign(denom).detach().clamp(min=-1) / denom.abs().clamp_min(1e-3)
        parts = []
        for h in range(1, self.n_harm + 1):
            ang = 2 * math.pi * h * base
            parts.append(torch.cos(ang))
            parts.append(torch.sin(ang))
        feats = torch.stack(parts, dim=-1)
        return torch.einsum("bf,ifd->bid", feats, self.readout) + self.bias


def digits_of(value: int, slots: int) -> list[int]:
    out = []
    for _ in range(slots):
        out.append(value % 10)
        value //= 10
    return out  # slot 0 = units


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulus", type=int, default=323)
    ap.add_argument("--train-x", type=int, default=250)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--freqs", type=int, default=32)
    ap.add_argument("--harm", type=int, default=8)
    ap.add_argument("--linear-only", action="store_true")
    ap.add_argument("--factored", action="store_true")
    ap.add_argument("--modfree", action="store_true")
    ap.add_argument("--init-noise", type=float, default=0.0)
    ap.add_argument("--code-lr", type=float, default=None,
                    help="separate (much smaller) lr for the digit/place code")
    ap.add_argument("--vinit", action="store_true",
                    help="structured init: monotone digit ramp, geometric place code,\nlog-uniform frequencies. All parameters stay trainable.")
    ap.add_argument(
        "--oracle",
        action="store_true",
        help="LAB DIAGNOSTIC ONLY: freeze the pair table at the exact additive-character "
        "phases of v^2 and train only the readout, to separate 'the representation "
        "cannot express/identify the map' from 'the optimizer will not find it'.",
    )
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--freq-jitter", type=float, default=0.0,
                    help="relative perturbation of the oracle frequencies; measures how\nwide the basin around the group solution is")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    globals()['INIT_NOISE'] = args.init_noise
    modulus = args.modulus
    units = [x for x in range(1, modulus) if math.gcd(x, modulus) == 1]
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(units), generator=g).tolist()
    train_x = [units[i] for i in perm[: args.train_x]]
    held_x = [units[i] for i in perm[args.train_x :]]

    device = torch.device(args.device)

    def tensors(xs):
        inp = torch.zeros(len(xs), args.slots, 10)
        tgt = torch.zeros(len(xs), args.slots, dtype=torch.long)
        for r, x in enumerate(xs):
            for i, d in enumerate(digits_of(x, args.slots)):
                inp[r, i, d] = 1.0
            for i, d in enumerate(digits_of((x * x) % modulus, args.slots)):
                tgt[r, i] = d
        return inp.to(device), tgt.to(device)

    xin, xt = tensors(train_x)
    hin, ht = tensors(held_x)

    if args.modfree:
        nslots = len(str(modulus))
        smod = torch.zeros(1, nslots, 10)
        for i, d in enumerate(digits_of(modulus, nslots)):
            smod[0, i, d] = 1.0
        smod = smod.to(device)
        model = ModFreePhaseStep(args.slots, nslots, args.harm).to(device)
        _fwd = model.forward
        model.forward = lambda s: _fwd(s, smod.expand(s.shape[0], -1, -1))
    elif args.factored:
        model = FactoredPhaseStep(args.slots, args.freqs, args.harm).to(device)
        if args.vinit:
            with torch.no_grad():
                model.digit.copy_(torch.arange(10.0) / 10.0)
                model.place.copy_(torch.tensor([10.0**i for i in range(args.slots)]) / 10.0)
                lo, hi = math.log(1e-3), math.log(10.0)
                model.omega.copy_(torch.exp(torch.linspace(lo, hi, args.freqs)))
                model.nu.zero_()
            print("[vinit] structured (still trainable) init applied")
    else:
        model = PairPhaseStep(args.slots, args.freqs, args.harm, args.linear_only).to(device)
    if args.oracle:
        with torch.no_grad():
            table = torch.zeros_like(model.table)
            ab = torch.outer(torch.arange(10.0), torch.arange(10.0))  # a*b
            ks = torch.arange(1.0, args.freqs + 1.0)
            for n, (i, j) in enumerate(model.pairs):
                mult = 1.0 if i == j else 2.0
                place = mult * (10**i) * (10**j)
                table[n] = (
                    (2 * math.pi / modulus)
                    * place
                    * ab[:, :, None]
                    * ks[None, None, :]
                ).to(table.device)
            model.table.copy_(table * (1.0 + args.freq_jitter))
        model.table.requires_grad_(False)
        print("[oracle] pair table frozen at the exact additive-character phases of v^2")
    print(
        f"modulus={modulus} units={len(units)} train={len(train_x)} held={len(held_x)} "
        f"params={sum(p.numel() for p in model.parameters()):,} "
        f"linear_only={args.linear_only}"
    )
    if args.code_lr is not None:
        code_names = {"digit", "place", "omega", "nu"}
        code = [p_ for n_, p_ in model.named_parameters() if n_ in code_names]
        rest = [p_ for n_, p_ in model.named_parameters() if n_ not in code_names]
        opt = torch.optim.AdamW(
            [{"params": code, "lr": args.code_lr, "weight_decay": 0.0},
             {"params": rest, "lr": args.lr, "weight_decay": args.wd}],
            betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.95))

    def acc(inp, tgt):
        with torch.no_grad():
            pred = model(inp).argmax(-1)
        return (pred == tgt).all(dim=1).float().mean().item()

    for step in range(1, args.steps + 1):
        logits = model(xin)
        loss = F.cross_entropy(logits.reshape(-1, 10), xt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0:
            print(
                f"step={step:>7} loss={loss.item():.5f} "
                f"train_exact={acc(xin, xt):.3f} held_exact={acc(hin, ht):.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
