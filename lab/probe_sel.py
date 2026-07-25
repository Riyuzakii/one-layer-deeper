#!/usr/bin/env python
"""LAB ONLY: does the depth SELECTOR destroy the iteration gain, and what fixes it?

`lab/probe_iter.py` supervises every composition depth separately and gets held-out
0.254.  `GRIter` (the ported submission) instead has ONE target per row and a learned
soft selector that mixes the L step outputs -- and its selector diagnostic shows it never
learns `T -> step count`.  This probe reproduces GRIter's *training setting* offline
(one target per row, a selector that must read T) so the selector variants can be
screened in ~2 minutes instead of a 10-minute evaluator run.

Variants (`--selector`):
  soft     plain softmax mix                       (what GRIter does today)
  entropy  softmax mix + entropy penalty           (push it to one-hot)
  gumbel   straight-through hard sample            (discrete forward, soft backward)
  anneal   softmax with a temperature schedule
  oracle   selector FROZEN at one-hot(T-1)         -- LAB DIAGNOSTIC ONLY, the ceiling

`--rand-loops` additionally randomises the number of applications available per step,
so a selector is never trained against a single fixed depth budget.

Self-generated data only; nothing under data/generated/ is opened.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from torch import nn


class SquaringBlock(nn.Module):
    def __init__(self, slots: int, n_freq: int, n_harm: int):
        super().__init__()
        self.pairs = [(i, j) for i in range(slots) for j in range(i, slots)]
        self.table = nn.Parameter(torch.randn(len(self.pairs), 10, 10, n_freq) * 0.5)
        self.n_harm = n_harm
        feat = 2 * n_harm * n_freq
        self.readout = nn.Parameter(torch.randn(slots, feat, 10) * feat**-0.5)
        self.bias = nn.Parameter(torch.zeros(slots, 10))

    def forward(self, s):
        theta = None
        for n, (i, j) in enumerate(self.pairs):
            term = torch.einsum("ba,bc,acf->bf", s[:, i], s[:, j], self.table[n])
            theta = term if theta is None else theta + term
        parts = []
        for h in range(1, self.n_harm + 1):
            parts.append(torch.cos(theta * h))
            parts.append(torch.sin(theta * h))
        feats = torch.cat(parts, dim=-1)
        return torch.einsum("bf,ifd->bid", feats, self.readout) + self.bias


class Net(nn.Module):
    def __init__(self, slots, t_slots, n_freq, n_harm, loops, mode):
        super().__init__()
        self.block = SquaringBlock(slots, n_freq, n_harm)
        self.loops = loops
        self.mode = mode
        self.sel = nn.Sequential(
            nn.Linear(t_slots * 10, 128), nn.GELU(), nn.Linear(128, loops)
        )
        # "pointer": composition depth is an ORDERED quantity, so parameterise the
        # selector as a learned scalar location on the step axis plus a soft window,
        # instead of an unstructured L-way classifier.  Still entirely learned.
        self.ptr = nn.Sequential(
            nn.Linear(t_slots * 10, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.log_sigma = nn.Parameter(torch.zeros(()))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long), persistent=False)

    def forward(self, s0, t_digits, t_index=None, loops=None, temp=1.0):
        loops = loops or self.loops
        state = s0
        outs = []
        for _ in range(loops):
            step = self.block(state)
            outs.append(step)
            state = F.softmax(step, dim=-1)
        stacked = torch.stack(outs, dim=1)  # B, loops, slots, 10
        logits = self.sel(t_digits.flatten(1))[:, :loops]
        if self.mode == "oracle":
            weights = F.one_hot(t_index.clamp(max=loops - 1), loops).float()
        elif self.mode == "gumbel" and self.training:
            weights = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        elif self.mode == "pointer":
            loc = F.softplus(self.ptr(t_digits.flatten(1)))  # B,1  >= 0
            grid = torch.arange(loops, device=s0.device, dtype=loc.dtype)[None, :]
            sigma = (self.log_sigma.exp().clamp(0.05, 4.0) if temp is None
                     else torch.as_tensor(temp, device=loc.device, dtype=loc.dtype))
            weights = F.softmax(-((grid - loc) ** 2) / (2 * sigma**2), dim=-1)
            probs = weights
            mixed = torch.einsum("bt,btsd->bsd", weights, stacked)
            entropy = -(probs * (probs + 1e-9).log()).sum(-1).mean()
            return mixed, entropy, probs
        elif self.mode == "anneal":
            weights = F.softmax(logits / temp, dim=-1)
        else:
            weights = F.softmax(logits, dim=-1)
        mixed = torch.einsum("bt,btsd->bsd", weights, stacked)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-9).log()).sum(-1).mean()
        return mixed, entropy, probs


def digits_of(value, slots):
    out = []
    for _ in range(slots):
        out.append(value % 10)
        value //= 10
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulus", type=int, default=323)
    ap.add_argument("--p", type=int, default=17)
    ap.add_argument("--train-x", type=int, default=250)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--t-slots", type=int, default=2)
    ap.add_argument("--freqs", type=int, default=32)
    ap.add_argument("--harm", type=int, default=8)
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument(
        "--selector", default="soft",
        choices=("soft", "entropy", "gumbel", "anneal", "oracle", "pointer"),
    )
    ap.add_argument("--ent-coef", type=float, default=0.1)
    ap.add_argument("--rand-loops", action="store_true")
    ap.add_argument("--ptr-anneal", action="store_true",
                    help="schedule the pointer window from wide to sharp")
    ap.add_argument("--time-steps", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--eval-time-steps", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=2000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    modulus, p = args.modulus, args.p
    q = modulus // p
    phi = (p - 1) * (q - 1)
    units = [x for x in range(1, modulus) if math.gcd(x, modulus) == 1]
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(units), generator=g).tolist()
    train_x = [units[i] for i in perm[: args.train_x]]
    held_x = [units[i] for i in perm[args.train_x :]]
    device = torch.device(args.device)

    def pack(xs, ts):
        n = len(xs) * len(ts)
        s0 = torch.zeros(n, args.slots, 10)
        td = torch.zeros(n, args.t_slots, 10)
        ti = torch.zeros(n, dtype=torch.long)
        tg = torch.zeros(n, args.slots, dtype=torch.long)
        r = 0
        for t in ts:
            for x in xs:
                for i, d in enumerate(digits_of(x, args.slots)):
                    s0[r, i, d] = 1.0
                for i, d in enumerate(digits_of(t, args.t_slots)):
                    td[r, i, d] = 1.0
                ti[r] = t - 1
                y = pow(x, pow(2, t, phi), modulus)
                for i, d in enumerate(digits_of(y, args.slots)):
                    tg[r, i] = d
                r += 1
        return s0.to(device), td.to(device), ti.to(device), tg.to(device)

    tr = pack(train_x, args.time_steps)
    evals = {
        (kind, t): pack(xs, [t])
        for t in args.eval_time_steps
        for kind, xs in (("seen", train_x[:96]), ("held", held_x))
    }

    net = Net(args.slots, args.t_slots, args.freqs, args.harm, args.loops,
              args.selector).to(device)
    print(
        f"modulus={modulus} train_x={len(train_x)} held_x={len(held_x)} "
        f"selector={args.selector} loops={args.loops} rand_loops={args.rand_loops} "
        f"params={sum(p_.numel() for p_ in net.parameters()):,}"
    )
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.95))

    for step in range(1, args.steps + 1):
        net.train()
        frac = step / args.steps
        temp = max(0.05, 1.0 - 0.95 * frac)
        if not args.ptr_anneal and args.selector == 'pointer':
            temp = None
        loops = args.loops
        if args.rand_loops:
            loops = int(torch.randint(max(args.time_steps), args.loops + 1, (1,)).item())
        out, ent, _ = net(tr[0], tr[1], tr[2], loops=loops, temp=temp)
        loss = F.cross_entropy(out.reshape(-1, 10), tr[3].reshape(-1))
        if args.selector == "entropy":
            loss = loss + args.ent_coef * ent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0:
            net.eval()
            with torch.no_grad():
                parts = []
                for t in args.eval_time_steps:
                    row = []
                    for kind in ("seen", "held"):
                        s0, td, ti, tg = evals[(kind, t)]
                        o, _, _ = net(s0, td, ti, temp=temp)
                        row.append((o.argmax(-1) == tg).all(1).float().mean().item())
                    parts.append(f"T{t}:seen={row[0]:.3f},held={row[1]:.3f}")
                _, _, probs = net(tr[0], tr[1], tr[2], temp=temp)
                sel_ent = float(
                    -(probs * (probs + 1e-9).log()).sum(-1).mean()
                )
            print(
                f"step={step:>6} loss={loss.item():.5f} sel_entropy={sel_ent:.3f}"
                f"(max {math.log(args.loops):.3f}) " + " ".join(parts),
                flush=True,
            )

    net.eval()
    with torch.no_grad():
        rows = []
        for t in args.time_steps:
            s0, td, ti, tg = pack(train_x[:1], [t])
            _, _, probs = net(s0, td, ti)
            rows.append((t, [round(float(v), 3) for v in probs[0]]))
    print("[selector] T -> step weights:", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
