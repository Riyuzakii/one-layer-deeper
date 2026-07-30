#!/usr/bin/env python
"""Legal-objective training probe for the digit-position monoid scan (PLAN2 §3.1).

Reports the metric ROW, not a cell (BRIEF2 §6.3):

    train_exact / held_exact           soft states
    train_exact_hard / held_exact_hard every state argmax-snapped  <-- headline
    held_div                           output diversity over held-out operands
                                       (a constant map reads ~1/n)
    tbl                                fraction of digit-table cells whose
                                       argmax matches the true solution
    repaired                           corrupted cells put back exactly
                                       (only with --corrupt; the conditioning
                                       measurement from lab/RESUME.md)

COMPLIANCE.  Operands are synthesised in-process from `--modulus` / `--bits`;
nothing under data/generated/ is opened.  `--construct` / `--corrupt` write the
true tables and are LAB DIAGNOSTICS (BRIEF §4 rule 2) — they can never appear in
a submission.  The training objective itself is always the plain end-to-end
label CE, i.e. LEGAL.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monoid import MonoidALU, digits_to_int, int_to_digits  # noqa: E402


# --------------------------------------------------------------------------- #
# operand synthesis (public generator semantics, re-derived locally)           #
# --------------------------------------------------------------------------- #

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _sample_prime(bits: int, rng: random.Random) -> int:
    while True:
        v = rng.randrange(1 << (bits - 1), 1 << bits) | 1
        if _is_prime(v):
            return v


def sample_semiprime(bits: int, rng: random.Random) -> int:
    half = bits // 2
    while True:
        p = _sample_prime(half, rng)
        q = _sample_prime(bits - half, rng)
        if p != q and (p * q).bit_length() == bits:
            return p * q


def build_operands(args) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    """Return (train, held) lists of (N, x) and the required digit-slot count."""
    rng = random.Random(args.split_seed)
    g = torch.Generator().manual_seed(args.split_seed)
    if args.modulus:
        # fixed modulus: held-out operands only, no held-out modulus pool
        mods_tr, mods_he = [args.modulus], []
    else:
        mods = []
        while len(mods) < args.n_mod_train + args.n_mod_held:
            m = sample_semiprime(args.bits, rng)
            if m not in mods:
                mods.append(m)
        mods_tr, mods_he = mods[: args.n_mod_train], mods[args.n_mod_train:]

    def units(m):
        return [x for x in range(1, m) if math.gcd(x, m) == 1]

    train: list[tuple[int, int]] = []
    held: list[tuple[int, int]] = []
    for m in mods_tr:
        u = units(m)
        perm = torch.randperm(len(u), generator=g).tolist()
        n_tr = min(args.train_x, len(u) - 1) if args.modulus else min(args.train_x, len(u))
        train += [(m, u[i]) for i in perm[:n_tr]]
        if args.modulus:
            held += [(m, u[i]) for i in perm[n_tr: n_tr + args.held_x]]
    for m in mods_he:
        u = units(m)
        perm = torch.randperm(len(u), generator=g).tolist()
        held += [(m, u[i]) for i in perm[: args.held_x]]
    slots = args.slots or max(len(str(m)) for m in mods_tr + mods_he)
    return train, held, slots


def to_tensors(pairs, slots, device, time_steps: int):
    L = 2 * slots
    xs = [x for _, x in pairs]
    ns = [n for n, _ in pairs]
    ys = []
    for n, x in pairs:
        v = x
        for _ in range(time_steps):
            v = (v * v) % n
        ys.append(v)
    xin = int_to_digits(xs, L, device)
    nin = int_to_digits(ns, L, device)
    tgt = torch.zeros(len(pairs), slots, dtype=torch.long, device=device)
    for r, y in enumerate(ys):
        for k in range(slots):
            tgt[r, k] = (y // 10**k) % 10
    return xin, nin, tgt


# --------------------------------------------------------------------------- #
# metrics                                                                     #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model, xin, nin, tgt, S, T, chunk=1024, hard=False):
    ok = 0
    ce = 0.0
    preds = []
    for i in range(0, xin.shape[0], chunk):
        y = xin[i:i + chunk]
        for _ in range(T):
            y = model(y, nin[i:i + chunk], hard=hard)
        lg = y[:, :S].clamp_min(1e-9).log()
        ok += (lg.argmax(-1) == tgt[i:i + chunk]).all(dim=1).sum().item()
        ce += F.cross_entropy(lg.reshape(-1, 10), tgt[i:i + chunk].reshape(-1),
                              reduction="sum").item()
        preds.append(lg.argmax(-1))
    pred = torch.cat(preds)
    # BRIEF2 6.3 collapse detector.  NOTE the reference value is NOT 1.0: the
    # squaring map on Z*_N is 4-to-1, so even the exact solution has diversity
    # |image| / n (0.221 for the 326-operand N=323 held set).  A constant map
    # reads 1/n.  Always compare against `div_ref` from the constructed run.
    div = len({tuple(r.tolist()) for r in pred}) / pred.shape[0]
    return ok / xin.shape[0], ce / (xin.shape[0] * S), div


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulus", type=int, default=323)
    ap.add_argument("--bits", type=int, default=11, help="sampled-modulus mode when --modulus 0")
    ap.add_argument("--n-mod-train", type=int, default=8)
    ap.add_argument("--n-mod-held", type=int, default=4)
    ap.add_argument("--slots", type=int, default=0)
    ap.add_argument("--train-x", type=int, default=250)
    ap.add_argument("--held-x", type=int, default=512)
    ap.add_argument("--time-steps", type=int, default=1)
    # architecture
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--family", default="colsoftmax",
                    choices=["colsoftmax", "sthard", "dsink", "orth", "dense"])
    ap.add_argument("--impl", default="scan", choices=["scan", "serial", "hop"])
    ap.add_argument("--init-scale", type=float, default=0.5)
    ap.add_argument("--untie-mul", action="store_true")
    # training
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--hard-train", action="store_true", help="straight-through discrete states")
    # diagnostics
    ap.add_argument("--construct", action="store_true",
                    help="LAB DIAGNOSTIC: write the exact tables, evaluate, exit")
    ap.add_argument("--corrupt", type=int, default=0,
                    help="LAB DIAGNOSTIC start point: construct then randomise k cells, "
                         "then train on the LEGAL objective and count exact repairs")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.modulus == 0:
        args.modulus = None
    train, held, S = build_operands(args)
    T = args.time_steps

    xin, nin, tgt = to_tensors(train, S, device, T)
    hin, hnin, htgt = to_tensors(held, S, device, T)

    model = MonoidALU(S, d=args.d, family=args.family, impl=args.impl,
                      init_scale=args.init_scale, tie_mul=not args.untie_mul).to(device)
    ref = model._reference()  # built ONCE; table_correct/repaired would rebuild it per row
    hit = None
    if args.construct or args.corrupt:
        model.construct_()
    if args.corrupt:
        g = torch.Generator().manual_seed(args.seed + 1000)
        hit = model.corrupt_(args.corrupt, g)

    n_par = sum(p.numel() for p in model.parameters())
    head = (f"[{args.tag}] mod={args.modulus or f'sampled{args.bits}'} S={S} L={model.L} "
            f"d={args.d} family={args.family} impl={args.impl} T={T} "
            f"train={len(train)} held={len(held)} params={n_par:,} "
            f"op_depth={model.op_depth} graph_depth={model.graph_depth}")
    print(head, flush=True)

    def row(step, loss):
        tr, tr_ce, tr_div = evaluate(model, xin, nin, tgt, S, T)
        he, he_ce, he_div = evaluate(model, hin, hnin, htgt, S, T)
        trh, _, _ = evaluate(model, xin, nin, tgt, S, T, hard=True)
        heh, _, hh_div = evaluate(model, hin, hnin, htgt, S, T, hard=True)
        tbl = model.table_correct(ref)
        rec = {
            "tag": args.tag, "step": step, "loss": loss,
            "train_exact": round(tr, 4), "held_exact": round(he, 4),
            "train_exact_hard": round(trh, 4), "held_exact_hard": round(heh, 4),
            "held_ce": round(he_ce, 4), "held_div": round(hh_div, 4),
            "tbl_all": round(tbl["_all"], 4),
            "tbl": {k: round(v, 3) for k, v in tbl.items() if not k.startswith("_")},
        }
        if hit is not None:
            ok, tot = model.repaired(hit, ref)
            rec["repaired"] = ok
            rec["corrupted"] = tot
        print(f"[{args.tag}] step={step:>6} loss={loss:.5f} "
              f"train_exact={tr:.3f} held_exact={he:.3f} "
              f"train_exact_hard={trh:.3f} held_exact_hard={heh:.3f} "
              f"held_div={hh_div:.3f} tbl={tbl['_all']:.3f}"
              + (f" repaired={rec['repaired']}/{rec['corrupted']}" if hit is not None else "")
              + f" ({time.time() - t0:.0f}s)", flush=True)
        return rec

    t0 = time.time()
    rows = []
    if args.construct:
        rows.append(row(0, float("nan")))
        if args.jsonl:
            with open(args.jsonl, "a") as f:
                for r in rows:
                    f.write(json.dumps({**r, "args": vars(args), "phase": "construct"}) + "\n")
        return 0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.95))
    rows.append(row(0, float("nan")))
    n = xin.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (min(args.batch, n),), generator=gen).to(device)
        y = xin[idx]
        model.ste = args.hard_train  # straight-through inside the training forward
        for _ in range(T):
            y = model(y, nin[idx], hard=args.hard_train)
        model.ste = False
        lg = y[:, :S].clamp_min(1e-9).log()
        loss = F.cross_entropy(lg.reshape(-1, 10), tgt[idx].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == args.steps:
            rows.append(row(step, loss.item()))

    if args.jsonl:
        with open(args.jsonl, "a") as f:
            for r in rows:
                f.write(json.dumps({**r, "args": vars(args), "phase": "train"}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
