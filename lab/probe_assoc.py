#!/usr/bin/env python
"""Is the learned multi-digit ADDER identifiable from generic algebraic laws
alone, by DIRECT DISCRETE SEARCH?

Why this probe exists.  `probe_rel.py --basin` measures how far from the exact
solution each objective can still see.  The end-of-chain LABEL objective is
flat beyond ~1-2 % cell corruption (`discrete-search` §2, reproduced here).
The ASSOCIATIVITY objective on the learned adder is not: it rises monotonically
with corruption all the way out to a uniformly random table, with a spread far
below its increments.  That is the first objective in this repo with a graded
landscape over the *whole* space.

Gradient descent on its soft relaxation still does not descend it
(`probe_rel.py --assoc`, block B/E).  So the question this probe answers is the
one that separates the two: **is the landscape benign and only the gradient
estimator bad?**  Here the objective is evaluated on the pure INTEGER
transducer -- no floating point, no relaxation -- and searched by greedy
coordinate descent, exactly as `discrete-search` did for the label objective.

The laws used are generic properties of a binary operation and supply no
values:
  assoc  (A + B) + C  ==  A + (B + C)
  comm    A + B       ==  B + A
  cancel  A + B       !=  A + C   whenever B != C
The constant adder satisfies assoc and comm and is excluded by cancel.

Self-generated register values only; nothing under data/generated/ is opened.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch


def add_scan_int(A, B, add_d, add_c, c0):
    """r = A (+) B, LSB->MSB, as an integer transducer.

    A, B : (P, n, W) int64 digits.  add_d, add_c : (P, 10, 10, Cc) int64.
    Returns (P, n, W) digits and (P, n) final carry state."""
    P, n, W = A.shape
    c = torch.full((P, n), c0, dtype=torch.long, device=A.device)
    outs = []
    pid = torch.arange(P, device=A.device)[:, None].expand(P, n)
    for m in range(W):
        u, v = A[:, :, m], B[:, :, m]
        o = add_d[pid, u, v, c]
        c = add_c[pid, u, v, c]
        outs.append(o)
    return torch.stack(outs, -1), c


def objective(add_d, add_c, c0, A, B, C, w_assoc, w_comm, w_cancel):
    """Mean per-slot violation rate of each law.  (P,) tensor."""
    P = add_d.shape[0]
    tot = torch.zeros(P, dtype=torch.float32, device=add_d.device)
    if w_assoc:
        l = add_scan_int(add_scan_int(A, B, add_d, add_c, c0)[0], C,
                         add_d, add_c, c0)[0]
        r = add_scan_int(A, add_scan_int(B, C, add_d, add_c, c0)[0],
                         add_d, add_c, c0)[0]
        tot = tot + w_assoc * (l != r).float().mean((1, 2))
    if w_comm:
        l = add_scan_int(A, B, add_d, add_c, c0)[0]
        r = add_scan_int(B, A, add_d, add_c, c0)[0]
        tot = tot + w_comm * (l != r).float().mean((1, 2))
    if w_cancel:
        l = add_scan_int(A, B, add_d, add_c, c0)[0]
        r = add_scan_int(A, C, add_d, add_c, c0)[0]
        diff = (B != C).any(-1).float()                      # (1,n) or (P,n)
        same_out = (l == r).all(-1).float()
        tot = tot + w_cancel * (diff * same_out).mean(-1)
    return tot


def structure(add_d):
    """Gauge-invariant: is each (v, c) column a cyclic shift of the identity,
    and how many DISTINCT shifts are realised?  The truth has 10 (a constant
    adder has 1) -- the analogue of alu-credit's `mul_gauge`."""
    u = torch.arange(10, device=add_d.device)
    k = n = 0
    shifts = set()
    for v in range(10):
        for c in range(add_d.shape[-1]):
            col = add_d[0, :, v, c]
            best, bs = -1, 0
            for s in range(10):
                hit = int((col == (u + s) % 10).sum())
                if hit > best:
                    best, bs = hit, s
            k += best
            n += 10
            shifts.add(bs)
    return round(k / n, 3), len(shifts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=6, help="register width W")
    ap.add_argument("--carry", type=int, default=2)
    ap.add_argument("--n", type=int, default=512, help="register triples")
    ap.add_argument("--sweeps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--assoc", type=float, default=1.0)
    ap.add_argument("--comm", type=float, default=1.0)
    ap.add_argument("--cancel", type=float, default=1.0)
    ap.add_argument("--start", default="random",
                    choices=["random", "true", "corrupt"])
    ap.add_argument("--corrupt", type=int, default=100)
    ap.add_argument("--resample", action="store_true",
                    help="draw fresh register triples every sweep (guards "
                         "against overfitting the evaluation set)")
    ap.add_argument("--hops", type=int, default=0,
                    help="basin hopping: after greedy converges, perturb "
                         "--hop-m cells at random and re-run greedy, keeping "
                         "the result only if it is better.  A strictly "
                         "stronger search than greedy, still cheap because one "
                         "objective evaluation is 2-4 register scans rather "
                         "than a 39-step chain.")
    ap.add_argument("--hop-m", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--jsonl", default="")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    W, Cc = args.slots, args.carry

    def draw(n):
        r = torch.randint(0, 10, (3, 1, n, W), device=dev)
        return r[0], r[1], r[2]

    A, B, C = draw(args.n)

    true_d = torch.zeros(1, 10, 10, Cc, dtype=torch.long, device=dev)
    true_c = torch.zeros(1, 10, 10, Cc, dtype=torch.long, device=dev)
    for u in range(10):
        for v in range(10):
            for c in range(Cc):
                s = u + v + (c if c < 2 else 0)
                true_d[0, u, v, c] = s % 10
                true_c[0, u, v, c] = min(s // 10, Cc - 1)

    if args.start == "true":
        add_d, add_c = true_d.clone(), true_c.clone()
    else:
        add_d = torch.randint(0, 10, (1, 10, 10, Cc), device=dev)
        add_c = torch.randint(0, Cc, (1, 10, 10, Cc), device=dev)
        if args.start == "corrupt":
            add_d, add_c = true_d.clone(), true_c.clone()
            n_cell = 100 * Cc
            pick = torch.randperm(2 * n_cell)[:args.corrupt]
            for p in pick.tolist():
                t, i = (add_d, p) if p < n_cell else (add_c, p - n_cell)
                hi = 10 if t is add_d else Cc
                t.view(-1)[i] = int(torch.randint(0, hi, (1,)))

    c0 = 0
    obj = objective(add_d, add_c, c0, A, B, C,
                    args.assoc, args.comm, args.cancel).item()
    st, ns = structure(add_d)
    print(f"[{args.tag}] W={W} Cc={Cc} n={args.n} start={args.start} "
          f"cells={2 * 100 * Cc} | obj0={obj:.4f} add_shift={st} "
          f"n_shifts={ns}", flush=True)
    # the reference value of the objective at the truth and at random
    ot = objective(true_d, true_c, c0, A, B, C,
                   args.assoc, args.comm, args.cancel).item()
    print(f"[{args.tag}] reference: obj(truth)={ot:.4f} "
          f"add_shift(truth)={structure(true_d)[0]} "
          f"n_shifts(truth)={structure(true_d)[1]}", flush=True)

    t0 = time.time()
    cells = [("d", u, v, c) for u in range(10) for v in range(10)
             for c in range(Cc)] + \
            [("c", u, v, c) for u in range(10) for v in range(10)
             for c in range(Cc)]

    def greedy(add_d, add_c, obj, sweeps, quiet=False):
        for sw in range(sweeps):
            order = torch.randperm(len(cells)).tolist()
            moved = 0
            for ci in order:
                kind, u, v, c = cells[ci]
                P = 10 if kind == "d" else Cc
                cand_d = add_d.expand(P, 10, 10, Cc).contiguous()
                cand_c = add_c.expand(P, 10, 10, Cc).contiguous()
                tgt = cand_d if kind == "d" else cand_c
                tgt[:, u, v, c] = torch.arange(P, device=dev)
                vals = objective(cand_d, cand_c, c0,
                                 A.expand(P, -1, -1), B.expand(P, -1, -1),
                                 C.expand(P, -1, -1),
                                 args.assoc, args.comm, args.cancel)
                best = int(vals.argmin())
                if vals[best].item() < obj - 1e-9:
                    obj = vals[best].item()
                    if kind == "d":
                        add_d[0, u, v, c] = best
                    else:
                        add_c[0, u, v, c] = best
                    moved += 1
            if moved == 0:
                break
        return obj

    hist = []
    for sweep in range(1, args.sweeps + 1):
        if args.resample:
            A, B, C = draw(args.n)
            obj = objective(add_d, add_c, c0, A, B, C,
                            args.assoc, args.comm, args.cancel).item()
        order = torch.randperm(len(cells)).tolist()
        moved = 0
        for ci in order:
            kind, u, v, c = cells[ci]
            P = 10 if kind == "d" else Cc
            cand_d = add_d.expand(P, 10, 10, Cc).contiguous()
            cand_c = add_c.expand(P, 10, 10, Cc).contiguous()
            tgt = cand_d if kind == "d" else cand_c
            tgt[:, u, v, c] = torch.arange(P, device=dev)
            vals = objective(cand_d, cand_c, c0,
                             A.expand(P, -1, -1), B.expand(P, -1, -1),
                             C.expand(P, -1, -1),
                             args.assoc, args.comm, args.cancel)
            best = int(vals.argmin())
            if vals[best].item() < obj - 1e-9:
                obj = vals[best].item()
                if kind == "d":
                    add_d[0, u, v, c] = best
                else:
                    add_c[0, u, v, c] = best
                moved += 1
        st, ns = structure(add_d)
        cell_agree = round(((add_d == true_d).float().mean().item()
                            + (add_c == true_c).float().mean().item()) / 2, 3)
        hist.append({"sweep": sweep, "obj": round(obj, 5), "moves": moved,
                     "add_shift": st, "n_shifts": ns, "cell": cell_agree})
        print(f"[{args.tag}] sweep={sweep:>3} obj={obj:.5f} moves={moved:>4} "
              f"add_shift={st} n_shifts={ns} cell_agree={cell_agree} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if moved == 0 and not args.resample:
            break

    if args.hops:
        best_d, best_c, best_o = add_d.clone(), add_c.clone(), obj
        for h in range(1, args.hops + 1):
            add_d, add_c = best_d.clone(), best_c.clone()
            n_cell = 100 * Cc
            pick = torch.randperm(2 * n_cell)[:args.hop_m]
            for p in pick.tolist():
                t, i = (add_d, p) if p < n_cell else (add_c, p - n_cell)
                hi = 10 if t is add_d else Cc
                t.view(-1)[i] = int(torch.randint(0, hi, (1,)))
            o = objective(add_d, add_c, c0, A, B, C,
                          args.assoc, args.comm, args.cancel).item()
            o = greedy(add_d, add_c, o, args.sweeps)
            if o < best_o - 1e-9:
                best_d, best_c, best_o = add_d.clone(), add_c.clone(), o
            if h % 10 == 0 or h == args.hops:
                st, ns = structure(best_d)
                ca = round(((best_d == true_d).float().mean().item()
                            + (best_c == true_c).float().mean().item()) / 2, 3)
                print(f"[{args.tag}] hop={h:>4} best_obj={best_o:.5f} "
                      f"add_shift={st} n_shifts={ns} cell_agree={ca} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        add_d, add_c, obj = best_d, best_c, best_o
        st, ns = structure(add_d)
        hist.append({"sweep": "hop", "obj": round(obj, 5), "moves": 0,
                     "add_shift": st, "n_shifts": ns,
                     "cell": round(((add_d == true_d).float().mean().item()
                                    + (add_c == true_c).float().mean().item())
                                   / 2, 3)})

    # held-out check: does the found table satisfy the laws on FRESH registers?
    A2, B2, C2 = draw(args.n)
    held = objective(add_d, add_c, c0, A2, B2, C2,
                     args.assoc, args.comm, args.cancel).item()
    print(f"[{args.tag}] FINAL obj={obj:.5f} held_obj={held:.5f} "
          f"add_shift={hist[-1]['add_shift']} n_shifts={hist[-1]['n_shifts']} "
          f"cell_agree={hist[-1]['cell']} ({time.time()-t0:.0f}s)", flush=True)
    if args.jsonl:
        with open(args.jsonl, "a") as fh:
            fh.write(json.dumps({"tag": args.tag, "argv": sys.argv[1:],
                                 "obj": obj, "held_obj": held,
                                 "obj_truth": ot, "hist": hist}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
