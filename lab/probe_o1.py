#!/usr/bin/env python
"""Legal-objective probe for the O(1)-depth modular reduction (`lab/o1reduce.py`).

Reports the metric ROW, not a cell (BRIEF2 §6.3):

    train_exact / held_exact            soft states
    train_exact_hard / held_exact_hard  every state argmax-snapped   <-- headline
    held_div                            output diversity over held operands.
                                        The reference is NOT 1.0 -- squaring is
                                        4-to-1 on Z*_N -- so `--construct` on the
                                        same set prints `div_ref` to compare to.
    tbl                                 fraction of table cells whose argmax
                                        matches the true solution
    repaired / by_depth                 corrupted cells put back exactly, split
                                        by each table's learned-op depth from
                                        the loss.  This is the conditioning
                                        measurement (`lab/RESUME.md`, Round 3).

COMPLIANCE.  Operands are synthesised in-process from `--modulus` / `--bits`;
nothing under data/generated/ is opened.  `--construct`, `--corrupt` and
`--recip oracle` write or supply the truth and are LAB DIAGNOSTICS (BRIEF §4
rule 2).  The training objective is always the plain end-to-end label CE, i.e.
LEGAL.
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
from o1reduce import O1ReduceALU, int_to_digits, true_mu  # noqa: E402
from probe_monoid import build_operands  # noqa: E402


def to_tensors(pairs, slots, L, device, time_steps: int):
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
    return xin, nin, tgt, ns


@torch.no_grad()
def evaluate(model, xin, nin, tgt, mus, S, T, chunk=512, hard=False):
    ok, ce, preds = 0, 0.0, []
    for i in range(0, xin.shape[0], chunk):
        nb = nin[i:i + chunk]
        if model.recip_kind == "oracle":
            model.set_mu(mus[i:i + chunk])
        y = xin[i:i + chunk]
        for _ in range(T):
            y = model(y, nb, hard=hard)
        lg = y[:, :S].clamp_min(1e-9).log()
        ok += (lg.argmax(-1) == tgt[i:i + chunk]).all(dim=1).sum().item()
        ce += F.cross_entropy(lg.reshape(-1, 10), tgt[i:i + chunk].reshape(-1),
                              reduction="sum").item()
        preds.append(lg.argmax(-1))
    pred = torch.cat(preds)
    div = len({tuple(r.tolist()) for r in pred}) / pred.shape[0]
    return ok / xin.shape[0], ce / (xin.shape[0] * S), div


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulus", type=int, default=323)
    ap.add_argument("--bits", type=int, default=11)
    ap.add_argument("--bits-list", default="", help="comma list, e.g. 16,18,20 (hf1)")
    ap.add_argument("--n-mod-train", type=int, default=8)
    ap.add_argument("--n-mod-held", type=int, default=4)
    ap.add_argument("--slots", type=int, default=0)
    ap.add_argument("--train-x", type=int, default=250)
    ap.add_argument("--held-x", type=int, default=512)
    ap.add_argument("--time-steps", type=int, default=1)
    # architecture
    ap.add_argument("--recip", default="oracle", choices=["oracle", "head", "div"])
    ap.add_argument("--impl", default="serial", choices=["serial", "scan"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-corr", type=int, default=2)
    ap.add_argument("--init-scale", type=float, default=0.5)
    # training
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--hard-train", action="store_true")
    # diagnostics
    ap.add_argument("--construct", action="store_true",
                    help="LAB DIAGNOSTIC: write the exact tables, evaluate, exit")
    ap.add_argument("--corrupt", type=int, default=0,
                    help="LAB DIAGNOSTIC start point: construct then randomise k cells, "
                         "then train on the LEGAL objective and count exact repairs")
    ap.add_argument("--corrupt-scale", type=float, default=0.5)
    ap.add_argument("--corrupt-tables", default="logit", choices=["logit", "all"],
                    help="logit = only the +/-BIG tables, which are on exactly the scale "
                         "alu-relational and matrix-scan measured; all = include ColSum")
    ap.add_argument("--corrupt-mode", default="uniform", choices=["uniform", "per_table"],
                    help="per_table corrupts k cells of EVERY table, which is what "
                         "makes the depth-stratified repair read balanced")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-chunk", type=int, default=512)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.modulus == 0:
        args.modulus = None
    if args.bits_list:
        # hf1's shape: several modulus sizes, DISJOINT train/held pools
        rng = random.Random(args.split_seed)
        g = torch.Generator().manual_seed(args.split_seed)
        from probe_monoid import sample_semiprime
        mods_tr, mods_he = [], []
        for b in [int(v) for v in args.bits_list.split(",")]:
            seen = []
            while len(seen) < args.n_mod_train + args.n_mod_held:
                m = sample_semiprime(b, rng)
                if m not in seen:
                    seen.append(m)
            mods_tr += seen[: args.n_mod_train]
            mods_he += seen[args.n_mod_train:]
        train, held = [], []
        for pool, out, k in ((mods_tr, "tr", args.train_x), (mods_he, "he", args.held_x)):
            for m in pool:
                xs = []
                while len(xs) < k:
                    v = rng.randrange(2, m)
                    if math.gcd(v, m) == 1:
                        xs.append(v)
                (train if out == "tr" else held).extend((m, x) for x in xs)
        # held-out operands at *training* moduli too, so the split matches hf1's
        for m in mods_tr:
            xs = []
            while len(xs) < args.held_x // 4:
                v = rng.randrange(2, m)
                if math.gcd(v, m) == 1:
                    xs.append(v)
            held.extend((m, x) for x in xs)
        S = args.slots or max(len(str(m)) for m in mods_tr + mods_he)
    else:
        train, held, S = build_operands(args)
    T = args.time_steps

    model = O1ReduceALU(S, recip=args.recip, init_scale=args.init_scale,
                        impl=args.impl, hidden=args.hidden, n_corr=args.n_corr).to(device)
    L = model.L
    xin, nin, tgt, ns_tr = to_tensors(train, S, L, device, T)
    hin, hnin, htgt, ns_he = to_tensors(held, S, L, device, T)

    # mu is a function of N alone; precomputing it once removes a 1,500-iteration
    # Python loop from every optimizer step (measured: it dominated the cell).
    mu_tr = true_mu(ns_tr, S, model.Lmu, device)
    mu_he = true_mu(ns_he, S, model.Lmu, device)

    ref = model._reference()
    hit = pre = None
    if args.construct or args.corrupt:
        model.construct_()
    if args.corrupt:
        g = torch.Generator().manual_seed(args.seed + 1000)
        hit = model.corrupt_(args.corrupt, g, args.corrupt_scale, args.corrupt_mode,
                             args.corrupt_tables)
        pre = model.cell_correct(ref)   # baseline: which touched cells are wrong
        # a cell no example exercises cannot be repaired at any conditioning
        use = model.usage(xin, nin, mu_tr if args.recip == "oracle" else None)
        used = {n_: int((use[n_][i.to(device)] > 1e-6).sum()) for n_, i in hit.items()
                if n_ in use}
        print(f"[{args.tag}] corrupted cells exercised by the training set: "
              f"{sum(used.values())}/{sum(len(i) for i in hit.values())} "
              + " ".join(f"{k}:{v}/{len(hit[k])}" for k, v in sorted(used.items())),
              flush=True)
        # usage() is forward-only under no_grad, so the start point is untouched

    n_par = sum(p.numel() for p in model.parameters())
    n_cells = ref.table_correct(ref)["_n_cells"]
    print(f"[{args.tag}] mod={args.modulus or args.bits_list or f'sampled{args.bits}'} "
          f"S={S} L={L} recip={args.recip} T={T} train={len(train)} held={len(held)} "
          f"params={n_par:,} cells={n_cells} reduce_op_depth={model.reduce_op_depth} "
          f"op_depth={model.op_depth} full_op_depth={model.full_op_depth}", flush=True)

    t0 = time.time()

    def row(step, loss):
        tr, _, _ = evaluate(model, xin, nin, tgt, mu_tr, S, T, args.eval_chunk)
        he, he_ce, _ = evaluate(model, hin, hnin, htgt, mu_he, S, T, args.eval_chunk)
        trh, _, _ = evaluate(model, xin, nin, tgt, mu_tr, S, T, args.eval_chunk, hard=True)
        heh, _, hdiv = evaluate(model, hin, hnin, htgt, mu_he, S, T, args.eval_chunk, hard=True)
        tbl = model.table_correct(ref)
        rec = {"tag": args.tag, "step": step, "loss": loss,
               "train_exact": round(tr, 4), "held_exact": round(he, 4),
               "train_exact_hard": round(trh, 4), "held_exact_hard": round(heh, 4),
               "held_ce": round(he_ce, 4), "held_div": round(hdiv, 4),
               "tbl_all": round(tbl["_all"], 4),
               "tbl": {k: round(v, 3) for k, v in tbl.items() if not k.startswith("_")}}
        extra = ""
        if hit is not None:
            ok, tot, by_depth = model.repaired(hit, ref, pre)
            # `naive` uses every touched cell as the denominator, which is what
            # `plan2/matrix-scan` §5 reported -- kept so the rows are comparable.
            nok, ntot, _ = model.repaired(hit, ref, None)
            lok, ltot, lby = model.repaired(hit, ref, pre, use)
            rec["repaired"], rec["corrupted"] = ok, tot
            rec["repaired_naive"], rec["touched"] = nok, ntot
            rec["repaired_live"], rec["live"] = lok, ltot
            rec["by_depth_live"] = {str(k): list(v) for k, v in lby.items()}
            rec["by_depth"] = {str(k): list(v) for k, v in by_depth.items()}
            extra = (f" repaired={ok}/{tot} live={lok}/{ltot} naive={nok}/{ntot} by_depth="
                     + ",".join(f"{k}:{v[0]}/{v[1]}" for k, v in by_depth.items()))
        print(f"[{args.tag}] step={step:>6} loss={loss:.5f} train_exact={tr:.3f} "
              f"held_exact={he:.3f} train_exact_hard={trh:.3f} held_exact_hard={heh:.3f} "
              f"held_div={hdiv:.3f} tbl={tbl['_all']:.3f}{extra} "
              f"({time.time() - t0:.0f}s)", flush=True)
        return rec

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
        if model.recip_kind == "oracle":
            model.set_mu(mu_tr[idx])
        model.ste = args.hard_train
        y = xin[idx]
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
