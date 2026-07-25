#!/usr/bin/env python
"""LAB ONLY -- the oracle-equivalent of a DIGIT-COMPOSITIONAL readout.

group-rotation §7 measured the ceiling of a readout that is an arbitrary
function of the residue, by freezing the representation at the truth and
training only the readout:

    N=323 -> 1.000,  N=899 -> 0.614,  N=2021 -> 0.337,  N=10403 -> 0.031
    (closed form P[x^2 already seen]: 1.000 / 0.614 / 0.337 / 0.072)

This is the same measurement for a readout in which EVERY learned tensor is a
table indexed by a *digit tuple* with a small, modulus-independent alphabet.
Nothing is indexed by the residue, so the coverage argument cannot apply; the
question is what replaces it.

Three tiers are measured, mirroring how §7 separated representation from
optimisation:

  --construct   tables set to the exact solution (LAB DIAGNOSTIC, never in a
                submission).  Answers: is the ceiling of this readout family
                1.000 at every modulus?  This is the direct analogue of
                probe_step.py --oracle.
  (default)     tables learned from random init on the same 250 training x.
                Answers: is the family identifiable from e1-sized supervision?
  --freeze X    a middle tier: freeze some tables at the truth, learn the rest.

Structure of the forward pass (a shift, a scan order and a gate; all *values*
learned):

    lo_ij, hi_ij = Tmul[d_i, d_j]                      SHARED over places
    r = zero
    for k = K-1 .. 0:                                  Horner over places of x^2
        r = shift_up(r)                                structural, no params
        for (i,j) with i+j == k:  r = add_scan(r, [lo_ij, hi_ij])
        for _ in range(R):        r = cond_sub(r, digits(N))
    y = r[:S]

    add_scan  : slot scan LSB->MSB, Tadd[digit, addend, carry] -> digit, carry
    cond_sub  : slot scan LSB->MSB, Tsub[digit, n_digit, borrow] -> digit, borrow
                then a learned gate on the final borrow state chooses between
                the scanned result and the input.

Total parameters ~7k, every one of them indexed by (digit, digit, small state).
There is no tensor with an index that ranges over Z_N.

Self-generated values only; nothing under data/generated/ is opened.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F
from torch import nn

BIG = 30.0


def digits_le(value: int, slots: int) -> list[int]:
    out = []
    for _ in range(slots):
        out.append(value % 10)
        value //= 10
    return out


class DigitALU(nn.Module):
    def __init__(self, slots: int, n_carry: int = 2, n_borrow: int = 2,
                 reduce_steps: int = 11, tau: float = 1.0,
                 identity_init: float = 0.0, hard: bool = False):
        super().__init__()
        self.S = slots
        self.K = 2 * slots - 1
        self.W = slots + 1
        self.R = reduce_steps
        self.Ca, self.Cb = n_carry, n_borrow
        self.tau = tau
        self.hard = hard
        self.Tmul = nn.Parameter(torch.randn(10, 10, 20) * 0.5)
        self.Tadd = nn.Parameter(torch.randn(10, 10, n_carry, 10 + n_carry) * 0.5)
        self.Tsub = nn.Parameter(torch.randn(10, 10, n_borrow, 10 + n_borrow) * 0.5)
        self.zero = nn.Parameter(torch.randn(10) * 0.5)
        self.carry0 = nn.Parameter(torch.randn(n_carry) * 0.5)
        self.borrow0 = nn.Parameter(torch.randn(n_borrow) * 0.5)
        self.gate = nn.Linear(n_borrow, 1)
        # A 280-step soft chain from random init is badly conditioned: every
        # scan scrambles the register before any of them is right.  A learned
        # copy-through path makes each scan the IDENTITY at init, so the chain
        # starts well-conditioned and learning perturbs away from it.  This is
        # residual/identity initialisation, not a supplied arithmetic rule --
        # copy_scale is trainable and the tables can override it.
        self.copy_scale = nn.Parameter(torch.tensor(float(identity_init)))
        if identity_init > 0:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -4.0)  # reduction starts closed

    # ---------------- construction (LAB DIAGNOSTIC ONLY) ----------------
    @torch.no_grad()
    def construct(self, which=("mul", "add", "sub", "gate", "zero")):
        if "mul" in which:
            t = torch.full_like(self.Tmul, -BIG)
            for a in range(10):
                for b in range(10):
                    t[a, b, (a * b) % 10] = BIG
                    t[a, b, 10 + (a * b) // 10] = BIG
            self.Tmul.copy_(t)
        if "add" in which:
            assert self.Ca == 2
            t = torch.full_like(self.Tadd, -BIG)
            for u in range(10):
                for v in range(10):
                    for c in range(2):
                        s = u + v + c
                        t[u, v, c, s % 10] = BIG
                        t[u, v, c, 10 + s // 10] = BIG
            self.Tadd.copy_(t)
        if "sub" in which:
            assert self.Cb == 2
            t = torch.full_like(self.Tsub, -BIG)
            for u in range(10):
                for v in range(10):
                    for c in range(2):
                        s = u - v - c
                        t[u, v, c, s % 10] = BIG
                        t[u, v, c, 10 + (1 if s < 0 else 0)] = BIG
            self.Tsub.copy_(t)
        if "gate" in which:
            self.gate.weight.copy_(torch.tensor([[BIG, -BIG]]))
            self.gate.bias.zero_()
        if "zero" in which:
            z = torch.full((10,), -BIG)
            z[0] = BIG
            self.zero.copy_(z)
            c = torch.full((self.Ca,), -BIG)
            c[0] = BIG
            self.carry0.copy_(c)
            b = torch.full((self.Cb,), -BIG)
            b[0] = BIG
            self.borrow0.copy_(b)
        self.copy_scale.zero_()

    # ---------------- scans ----------------
    def _sm(self, logits):
        p = F.softmax(logits / self.tau, -1)
        if self.hard:  # straight-through: close the continuous side-channel
            h = F.one_hot(p.argmax(-1), p.shape[-1]).to(p.dtype)
            p = h + p - p.detach()
        return p

    def add_scan(self, r, addend):
        c = self._sm(self.carry0).expand(r.shape[0], self.Ca)
        outs = []
        for m in range(self.W):
            o = torch.einsum("bu,bv,bc,uvco->bo", r[:, m], addend[:, m], c, self.Tadd)
            outs.append(self._sm(o[:, :10] + self.copy_scale * r[:, m]))
            c = self._sm(o[:, 10:] + self.copy_scale * c)
        return torch.stack(outs, 1)

    def cond_sub(self, r, ndig):
        b = r.shape[0]
        c = self._sm(self.borrow0).expand(b, self.Cb)
        outs = []
        for m in range(self.W):
            o = torch.einsum("bu,bv,bc,uvco->bo", r[:, m], ndig[m].expand(b, 10),
                             c, self.Tsub)
            outs.append(self._sm(o[:, :10] + self.copy_scale * r[:, m]))
            c = self._sm(o[:, 10:] + self.copy_scale * c)
        t = torch.stack(outs, 1)
        g = torch.sigmoid(self.gate(c))[:, :, None]  # (B,1,1)
        return g * t + (1 - g) * r

    def forward(self, s, ndig):
        b = s.shape[0]
        z = self._sm(self.zero).expand(b, 10)
        prod = {}
        for i in range(self.S):
            for j in range(self.S):
                o = torch.einsum("bu,bv,uvo->bo", s[:, i], s[:, j], self.Tmul)
                prod[(i, j)] = (self._sm(o[:, :10]), self._sm(o[:, 10:]))
        r = z[:, None].expand(b, self.W, 10)
        for k in range(self.K - 1, -1, -1):
            r = torch.cat([z[:, None], r[:, : self.W - 1]], dim=1)  # x10
            for i in range(self.S):
                j = k - i
                if 0 <= j < self.S:
                    lo, hi = prod[(i, j)]
                    slots = [lo[:, None], hi[:, None]] + \
                            [z[:, None]] * (self.W - 2)
                    r = self.add_scan(r, torch.cat(slots, dim=1))
            for _ in range(self.R):
                r = self.cond_sub(r, ndig)
        return torch.log(r[:, : self.S] + 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulus", type=int, default=323)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--train-x", type=int, default=250)
    ap.add_argument("--reduce", type=int, default=11)
    ap.add_argument("--carry", type=int, default=2)
    ap.add_argument("--borrow", type=int, default=2)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--tau-final", type=float, default=None,
                    help="anneal the state temperature from --tau to this")
    ap.add_argument("--identity-init", type=float, default=0.0,
                    help="copy-through logit scale; >0 makes every scan the "
                         "identity at init (trainable)")
    ap.add_argument("--hard", action="store_true",
                    help="straight-through discrete states: closes the "
                         "continuous side-channel through the soft digits")
    ap.add_argument("--construct", action="store_true",
                    help="LAB DIAGNOSTIC: set every table to the exact solution")
    ap.add_argument("--freeze", nargs="*", default=[],
                    help="subset of mul/add/sub/gate/zero to construct+freeze")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    modulus, S = args.modulus, args.slots
    units = [x for x in range(1, modulus) if math.gcd(x, modulus) == 1]
    g = torch.Generator().manual_seed(args.split_seed)
    perm = torch.randperm(len(units), generator=g).tolist()
    train_x = [units[i] for i in perm[: args.train_x]]
    held_x = [units[i] for i in perm[args.train_x:]]
    device = torch.device(args.device)

    def tensors(xs):
        inp = torch.zeros(len(xs), S, 10)
        tgt = torch.zeros(len(xs), S, dtype=torch.long)
        for r, x in enumerate(xs):
            for i, d in enumerate(digits_le(x, S)):
                inp[r, i, d] = 1.0
            for i, d in enumerate(digits_le((x * x) % modulus, S)):
                tgt[r, i] = d
        return inp.to(device), tgt.to(device)

    xin, xt = tensors(train_x)
    hin, ht = tensors(held_x)
    W = S + 1
    ndig = torch.zeros(W, 10)
    for i, d in enumerate(digits_le(modulus, W)):
        ndig[i, d] = 1.0
    ndig = ndig.to(device)

    model = DigitALU(S, args.carry, args.borrow, args.reduce, args.tau,
                     args.identity_init, args.hard).to(device)
    frozen = []
    if args.construct:
        model.construct()
        frozen = ["Tmul", "Tadd", "Tsub", "gate", "zero", "carry0", "borrow0"]
    elif args.freeze:
        model.construct(tuple(args.freeze))
        name = {"mul": ["Tmul"], "add": ["Tadd"], "sub": ["Tsub"],
                "gate": ["gate"], "zero": ["zero", "carry0", "borrow0"]}
        for f in args.freeze:
            frozen += name[f]
    for n, p in model.named_parameters():
        if any(n.startswith(f) for f in frozen):
            p.requires_grad_(False)
    n_par = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.tag}] modulus={modulus} S={S} K={model.K} W={W} R={model.R} "
          f"units={len(units)} train={len(train_x)} held={len(held_x)} "
          f"params={n_par:,} trainable={n_tr:,} frozen={frozen}", flush=True)

    @torch.no_grad()
    def evaluate(inp, tgt, chunk=4096):
        model.eval()
        ok, ce = 0, 0.0
        for i in range(0, inp.shape[0], chunk):
            lg = model(inp[i:i + chunk], ndig)
            ok += (lg.argmax(-1) == tgt[i:i + chunk]).all(dim=1).sum().item()
            ce += F.cross_entropy(lg.reshape(-1, 10), tgt[i:i + chunk].reshape(-1),
                                  reduction="sum").item()
        model.train()
        return ok / inp.shape[0], ce / (inp.shape[0] * tgt.shape[1])

    if args.construct:
        tr, tr_ce = evaluate(xin, xt)
        he, he_ce = evaluate(hin, ht)
        print(f"[{args.tag}] CONSTRUCTED  train_exact={tr:.3f} held_exact={he:.3f} "
              f"train_ce={tr_ce:.4f} held_ce={he_ce:.4f}", flush=True)
        return 0

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if args.tau_final is not None:
            f = step / args.steps
            model.tau = math.exp((1 - f) * math.log(args.tau)
                                 + f * math.log(args.tau_final))
        logits = model(xin, ndig)
        loss = F.cross_entropy(logits.reshape(-1, 10), xt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            tr, tr_ce = evaluate(xin, xt)
            he, he_ce = evaluate(hin, ht)
            print(f"[{args.tag}] step={step:>6} loss={loss.item():.5f} "
                  f"train_exact={tr:.3f} held_exact={he:.3f} "
                  f"train_ce={tr_ce:.3f} held_ce={he_ce:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
