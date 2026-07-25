#!/usr/bin/env python
"""LAB ONLY -- does a DIGIT-COMPOSITIONAL readout escape the coverage ceiling?

`explore/group-rotation` §7: with a *perfect* representation of the residue and
the generator's 250 training x, the held-out ceiling of a readout that is an
arbitrary function of the residue is

    N=323 -> 1.000,  N=899 -> 0.614,  N=2021 -> 0.337,  N=10403 -> 0.031

because such a readout is unconstrained on any residue not seen in training.

This probe measures the same quantity for a readout in which **every learned
tensor is shared across digit places** -- so nothing in it is indexed by the
residue.  Architecture (all parameters learned from random init; no modular
arithmetic, multiplication or carry rule is written anywhere in the forward
pass -- only a shift, a scan order, and shared cells):

    stage A   f_k = sum_{i+j=k} s_i^T M s_j              M : (10,10,H), SHARED
              (the one structural fact used: a partial product from places i
               and j belongs at place i+j.  That is the digit-place algebra of
               a decimal string, not an arithmetic rule.)

    stage B   an outer weight-tied recurrence over places k = K-1 .. 0
                r <- shift_up(r)                          (structural, no params)
                h  = Emb(r) + Wn(onehot(N_slot)) + Wf(f_k) at slot 0
                h  = SlotBlock(h)  x L                    (SHARED across slots,
                                                           across L, across k)
                r  = softmax(Wout(h))                     (soft-digit round trip)

    SlotBlock = LSB->MSB scan + MSB->LSB scan with two shared recurrent cells,
                then a shared per-slot residual update.  This is the structure
                a ripple carry / borrow / comparison needs; what those cells
                compute is learned.

`--pair-oracle` freezes stage A at the true partial products -- the exact
analogue of group-rotation's `probe_step.py --oracle`, which froze its pair
table at the true additive-character phases.  Representation given, readout
learned; the only difference between the two experiments is whether the readout
is indexed by the residue or by digit-local quantities.

Self-generated values only; nothing under data/generated/ is opened.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F
from torch import nn


def digits_le(value: int, slots: int) -> list[int]:
    out = []
    for _ in range(slots):
        out.append(value % 10)
        value //= 10
    return out


class SlotBlock(nn.Module):
    """One weight-tied pass over the slot axis.  No per-slot parameters."""

    def __init__(self, d_model: int, d_carry: int):
        super().__init__()
        self.fwd = nn.GRUCell(d_model, d_carry)
        self.bwd = nn.GRUCell(d_model, d_carry)
        self.upd = nn.Sequential(
            nn.Linear(d_model + 2 * d_carry, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.d_carry = d_carry

    def forward(self, h):  # h: (B, W, D)
        b, w, _ = h.shape
        c = h.new_zeros(b, self.d_carry)
        fs = []
        for m in range(w):  # LSB -> MSB
            c = self.fwd(h[:, m], c)
            fs.append(c)
        U = torch.stack(fs, dim=1)
        c = h.new_zeros(b, self.d_carry)
        bs = []
        for m in range(w - 1, -1, -1):  # MSB -> LSB
            c = self.bwd(h[:, m], c)
            bs.append(c)
        V = torch.stack(bs[::-1], dim=1)
        return self.norm(h + self.upd(torch.cat([h, U, V], dim=-1)))


class DigitCarryNet(nn.Module):
    def __init__(self, slots, n_pair, d_model, d_carry, n_block, n_place=None,
                 pair_oracle=False, share_pairs=True):
        super().__init__()
        self.S = slots
        self.K = 2 * slots - 1 if n_place is None else n_place
        self.W = slots + 1
        self.share_pairs = share_pairs
        self.pair_oracle = pair_oracle
        if pair_oracle:
            n_pair = 82  # one-hot over the 82 possible values of a*b
            tbl = torch.zeros(10, 10, 82)
            for a in range(10):
                for b in range(10):
                    tbl[a, b, a * b] = 1.0
            self.register_buffer("pair", tbl)
        elif share_pairs:
            self.pair = nn.Parameter(torch.randn(10, 10, n_pair) / math.sqrt(n_pair))
        else:  # control: a separate table per (i,j) -- NOT place-shared
            self.pair = nn.Parameter(
                torch.randn(slots, slots, 10, 10, n_pair) / math.sqrt(n_pair))
        self.emb = nn.Embedding(10, d_model)          # soft-digit embedding
        self.nemb = nn.Linear(10, d_model, bias=False)  # N's digit at this slot
        self.finj = nn.Linear(n_pair, d_model)        # inject f_k at slot 0
        self.zero = nn.Parameter(torch.zeros(10))     # learned "empty slot" digit
        self.block = SlotBlock(d_model, d_carry)
        self.n_block = n_block
        self.out = nn.Linear(d_model, 10)

    def pair_features(self, s):  # s: (B, S, 10) soft digits -> (B, K, H)
        b = s.shape[0]
        h = self.pair.shape[-1]
        f = s.new_zeros(b, self.K, h)
        for i in range(self.S):
            for j in range(self.S):
                k = i + j
                if k >= self.K:
                    continue
                tbl = self.pair if self.share_pairs or self.pair_oracle else self.pair[i, j]
                f[:, k] = f[:, k] + torch.einsum("ba,bc,ach->bh", s[:, i], s[:, j], tbl)
        return f

    def forward(self, s, ndig):
        """s: (B,S,10) soft digits of x.  ndig: (W,10) one-hot digits of N."""
        b = s.shape[0]
        f = self.pair_features(s)
        nvec = self.nemb(ndig)[None]                      # (1, W, D)
        r = F.softmax(self.zero, dim=-1)[None, None].expand(b, self.W, 10)
        emb_w = self.emb.weight                            # (10, D)
        for k in range(self.K - 1, -1, -1):
            r = torch.cat(
                [F.softmax(self.zero, dim=-1)[None, None].expand(b, 1, 10),
                 r[:, : self.W - 1]], dim=1)                # shift up (x10)
            h = r @ emb_w + nvec
            inj = self.finj(f[:, k])
            h = torch.cat([h[:, :1] + inj[:, None], h[:, 1:]], dim=1)
            for _ in range(self.n_block):
                h = self.block(h)
            r = F.softmax(self.out(h), dim=-1)
        return torch.log(r[:, : self.S] + 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulus", type=int, default=323)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--train-x", type=int, default=250)
    ap.add_argument("--pair", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--d-carry", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--pair-oracle", action="store_true",
                    help="LAB DIAGNOSTIC: freeze stage A at the true partial "
                         "products, the analogue of probe_step.py --oracle")
    ap.add_argument("--no-share-pairs", action="store_true",
                    help="control: one pair table per (i,j) place pair")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0, help="0 = full batch")
    ap.add_argument("--log-every", type=int, default=500)
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

    model = DigitCarryNet(S, args.pair, args.d_model, args.d_carry, args.blocks,
                          pair_oracle=args.pair_oracle,
                          share_pairs=not args.no_share_pairs).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.tag}] modulus={modulus} S={S} units={len(units)} "
          f"train={len(train_x)} held={len(held_x)} params={n_par:,} "
          f"pair_oracle={args.pair_oracle} share_pairs={not args.no_share_pairs} "
          f"blocks={args.blocks} d={args.d_model}/{args.d_carry}", flush=True)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    @torch.no_grad()
    def evaluate(inp, tgt, chunk=2048):
        model.eval()
        ok = 0
        ce = 0.0
        for i in range(0, inp.shape[0], chunk):
            lg = model(inp[i:i + chunk], ndig)
            ok += (lg.argmax(-1) == tgt[i:i + chunk]).all(dim=1).sum().item()
            ce += F.cross_entropy(lg.reshape(-1, 10), tgt[i:i + chunk].reshape(-1),
                                  reduction="sum").item()
        model.train()
        return ok / inp.shape[0], ce / (inp.shape[0] * tgt.shape[1])

    t0 = time.time()
    for step in range(1, args.steps + 1):
        if args.batch and args.batch < xin.shape[0]:
            idx = torch.randint(0, xin.shape[0], (args.batch,), device=device)
            bi, bt = xin[idx], xt[idx]
        else:
            bi, bt = xin, xt
        logits = model(bi, ndig)
        loss = F.cross_entropy(logits.reshape(-1, 10), bt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
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
