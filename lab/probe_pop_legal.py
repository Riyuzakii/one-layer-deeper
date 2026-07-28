#!/usr/bin/env python
"""LAB ONLY -- explore/alu-relational's LEGAL training signals, re-screened at
P = 32 replicas.

WHY THIS EXISTS.  `alu-relational` closed the legal-signal search after 44
training runs, each at P = 1 (one or two seeds).  This branch then measured
that the basin of the working (illegal) signal is selected by INITIALISATION,
with a per-replica hit rate of 0.26 -- so a single-seed screen sees one draw
from a distribution.  If any legal term produces even a small tail of replicas
toward the `local_ce` cliff (0.0050-0.0073), a P=1 screen would miss it and the
"closed" verdict would be an artifact of the screening design rather than a
fact about the signals.  This re-runs the candidates with 32 draws each.

THE TERMS, all LEGAL under alu-relational report §7 (its rule: a term is legal
if it asserts a GENERIC algebraic property of an operation the model already
performs, and illegal if it asserts the specific relationship between two
operations that constitutes the definition of the arithmetic):

  --dual fold|redall|horner   one set of tables computes one function however
                              it is composed.  §7 calls this the cleanest.
  --sym                       the learned add and multiply are commutative
  --inv                       the learned subtract inverts the learned add
  --assoc                     the learned multi-digit adder is associative
  --cancel                    B -> A (+) B is injective (non-degeneracy)
  --rel free                  SOME additive-increment structure exists, with
                              the increment map D free and learned

The loss-term implementations are ported verbatim in content from
`explore/alu-relational/lab/probe_rel.py` (read-only to this branch), with a
replica index added to every table read.  `--rel affine` (which that report
flags as on the wrong side of its own line) and `--rel true` (illegal) are NOT
implemented here.

`local_ce` is MEASURED with a constructed reference tape, exactly as
`alu-relational` and `alu-credit` do.  That is a measurement, not a training
signal -- no gradient flows from it.  The training itself is fully LEGAL.

Self-generated operands only; nothing under data/generated/ is opened.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_alu_depth import digits_le                       # noqa: E402
from probe_pop import PopALU, eval_pop, pop_clip_           # noqa: E402
import optim_extra                                        # noqa: E402


def sym_ce(p, q):
    """Symmetric cross-entropy: minimised at p = q = one-hot, so it prices
    agreement AND sharpness.  (alu-relational's `sym_ce`.)"""
    return (-(q.detach() * p.clamp_min(1e-9).log()).sum(-1).mean()
            - (p.detach() * q.clamp_min(1e-9).log()).sum(-1).mean())


def sym_kl(p, q):
    """Symmetric KL: zero whenever p = q at ANY entropy -- agreement only."""
    lp, lq = p.clamp_min(1e-9).log(), q.clamp_min(1e-9).log()
    return (((p - q.detach()) * (lp - lq.detach())).sum(-1).mean()
            + ((q - p.detach()) * (lq - lp.detach())).sum(-1).mean())


class PopFreeD(torch.nn.Module):
    """Per-replica free learned increment map D(x).  Conservative variant:
    asserts only that SOME additive-increment structure exists.  Discarded at
    eval."""

    def __init__(self, P, slots, hidden=256):
        super().__init__()
        self.P, self.S, self.H = P, slots, hidden
        d = slots * 10
        self.W1 = torch.nn.Parameter(torch.randn(P, d, hidden) / math.sqrt(d))
        self.b1 = torch.nn.Parameter(torch.zeros(P, hidden))
        self.W2 = torch.nn.Parameter(torch.randn(P, hidden, d)
                                     / math.sqrt(hidden))
        self.b2 = torch.nn.Parameter(torch.zeros(P, d))

    def forward(self, s):
        """s: (b, S, 10) shared -> (P, b, S, 10) per-replica."""
        x = s.reshape(s.shape[0], -1)[None].expand(self.P, *([-1] * 2))
        h = torch.tanh(torch.einsum("pbi,pih->pbh", x, self.W1)
                       + self.b1[:, None])
        o = torch.einsum("pbh,pho->pbo", h, self.W2) + self.b2[:, None]
        return F.softmax(o.view(self.P, s.shape[0], self.S, 10), -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--modulus", type=int, default=10403)
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--train-x", type=int, default=8000)
    ap.add_argument("--held-x", type=int, default=1024)
    ap.add_argument("--max-quot", type=int, default=10)
    ap.add_argument("--label-w", type=float, default=1.0)
    # family 2: dual-path agreement
    ap.add_argument("--dual", default="none",
                    choices=["none", "fold", "redall", "horner"])
    ap.add_argument("--dual-w", type=float, default=1.0)
    ap.add_argument("--dual-batch", type=int, default=0)
    ap.add_argument("--dual-label", type=float, default=1.0)
    # family 3: algebraic laws
    ap.add_argument("--sym", type=float, default=0.0)
    ap.add_argument("--inv", type=float, default=0.0)
    ap.add_argument("--assoc", type=float, default=0.0)
    ap.add_argument("--cancel", type=float, default=0.0)
    ap.add_argument("--law-batch", type=int, default=256)
    # family 1: the conservative relational variant
    ap.add_argument("--rel", default="none", choices=["none", "free"])
    ap.add_argument("--rel-w", type=float, default=1.0)
    ap.add_argument("--rel-nondeg", type=float, default=0.0)
    ap.add_argument("--rel-batch", type=int, default=128)
    ap.add_argument("--rel-hidden", type=int, default=256)
    ap.add_argument("--div", default="ce", choices=["ce", "kl"])
    ap.add_argument("--opt", default="adamw",
                    choices=["adamw", "soap", "ademamix"],
                    help="LEGAL: build_optimizer may return any "
                         "torch.optim.Optimizer; the evaluator still owns the "
                         "loop, the backward and the one-step-per-batch cadence")
    ap.add_argument("--soap-freq", type=int, default=10)
    ap.add_argument("--soap-max-dim", type=int, default=512)
    ap.add_argument("--soap-merge", type=int, default=1)
    ap.add_argument("--soap-beta", type=float, default=0.95)
    ap.add_argument("--soap-mspace", default="rot", choices=["rot", "orig"])
    ap.add_argument("--soap-warmup", type=int, default=0)
    ap.add_argument("--ade-alpha", type=float, default=8.0)
    ap.add_argument("--ade-beta2", type=float, default=0.999)
    ap.add_argument("--ade-beta3", type=float, default=0.9999)
    ap.add_argument("--ade-warmup", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--init-scale", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=400)
    ap.add_argument("--eval-n", type=int, default=1024)
    ap.add_argument("--eval-chunk", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--jsonl", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    N, S, P = args.modulus, args.slots, args.pop
    units = [x for x in range(1, N) if math.gcd(x, N) == 1]
    g = torch.Generator().manual_seed(args.split_seed)
    perm = torch.randperm(len(units), generator=g).tolist()
    tr_x = [units[i] for i in perm[: args.train_x]]
    he_x = [units[i] for i in perm[args.train_x: args.train_x + args.held_x]]
    dev = torch.device(args.device)

    def tens(xs):
        inp = torch.zeros(len(xs), S, 10)
        tgt = torch.zeros(len(xs), S, dtype=torch.long)
        for r, x in enumerate(xs):
            for i, d in enumerate(digits_le(x, S)):
                inp[r, i, d] = 1.0
            for i, d in enumerate(digits_le((x * x) % N, S)):
                tgt[r, i] = d
        return inp.to(dev), tgt.to(dev)

    xin, xt = tens(tr_x)
    hin, ht = tens(he_x)
    W = S + 1
    nd = torch.zeros(W, 10)
    for i, d in enumerate(digits_le(N, W)):
        nd[i, d] = 1.0
    nd = nd.to(dev)

    model = PopALU(P, S, 2, 2, args.max_quot, 1.0, False, False, False,
                   args.init_scale).to(dev)
    ref = PopALU(1, S, 2, 2, args.max_quot).to(dev)
    ref.construct()
    for p in ref.parameters():
        p.requires_grad_(False)

    freeD = PopFreeD(P, S, args.rel_hidden).to(dev) if args.rel == "free" \
        else None
    rel_c = torch.nn.Parameter(torch.randn(P, S, 10) * 0.5).to(dev) \
        if args.rel != "none" else None
    if rel_c is not None:
        rel_c = torch.nn.Parameter(rel_c.detach().clone())

    from benchmark import ModelSpec, assert_model_state
    n_state = assert_model_state(
        model, ModelSpec(vocab_size=17, max_seq_len=64,
                         maximum_model_state_elements=500_000_000))
    print(f"[{args.tag}] POP={P} N={N} S={S} opt={args.opt} "
          f"tree:quotient LEGAL "
          f"dual={args.dual} sym={args.sym} inv={args.inv} "
          f"assoc={args.assoc} cancel={args.cancel} rel={args.rel} "
          f"div={args.div} state={n_state:,}/5e8", flush=True)

    params = list(model.parameters())
    if freeD is not None:
        params += list(freeD.parameters())
    if rel_c is not None:
        params += [rel_c]
    okw = dict(lr=args.lr, wd=args.wd, betas=(0.9, 0.95), batch_dims=1,
               soap_freq=args.soap_freq, soap_max_dim=args.soap_max_dim,
               soap_merge=args.soap_merge, soap_beta=args.soap_beta,
               soap_mspace=args.soap_mspace, soap_warmup=args.soap_warmup,
               ade_alpha=args.ade_alpha, ade_beta2=args.ade_beta2,
               ade_beta3=args.ade_beta3, ade_warmup=args.ade_warmup)
    opt = optim_extra.build(args.opt, params, **okw)
    agree = sym_ce if args.div == "ce" else sym_kl
    idx = torch.arange(10, device=dev)

    @torch.no_grad()
    def diversity_vec(n=256):
        """Collapse detector, per replica: fraction of DISTINCT predicted
        answers.  A map collapsed to a constant reads ~1/n; the truth reads
        1.0.  (alu-relational's `out_diversity`, per replica.)  A low local_ce
        with low diversity is the known constant-map collapse, not progress."""
        was, model.hard = model.hard, True
        a = model(xin[:n], nd).argmax(-1)                  # (P,n,S)
        model.hard = was
        return torch.tensor(
            [len({tuple(r.tolist()) for r in a[q]}) / n for q in range(P)])

    @torch.no_grad()
    def local_ce_vec(n=512):
        """Per-replica local CE.  MEASUREMENT ONLY -- a constructed reference
        tape, no gradient, never added to the loss."""
        ref.mode, ref.tape = 'record', []
        ref(xin[:n], nd)
        ref.mode = None
        model.mode, model.tape = 'force', ref.tape
        model.tf_loss, model.tf_n, model.tf_p = \
            torch.zeros(P, device=dev), 0, 1.0
        model(xin[:n], nd)
        model.mode = None
        return (model.tf_loss / max(model.tf_n, 1)).cpu()

    # ---------------- the LEGAL penalties (ported from probe_rel.py) --------
    def sym_pen():
        p = ((model.Tmul_eff - model.Tmul_eff.transpose(1, 2)) ** 2).mean()
        return p + ((model.Tadd_eff - model.Tadd_eff.transpose(1, 2)) ** 2).mean()

    def inv_pen():
        a = F.softmax(model.Tadd_eff, -1)
        w = a[..., :10]                                  # (P,u,v,c,w)
        s = F.softmax(model.Tsub_eff, -1)                # (P,w,v,c,o)
        got = torch.einsum("puvcw,pwvco->puvco", w, s)
        tgt_d = F.one_hot(idx, 10).float().to(dev)[None, :, None, None, :] \
            .expand(P, 10, 10, model.Ca, 10)
        return -(tgt_d * got[..., :10].clamp_min(1e-9).log()).sum(-1).mean()

    def rand_regs(bs):
        r = torch.randint(0, 10, (3, bs, W), device=dev)
        A, B, C = [F.one_hot(r[i], 10).float()[None].expand(P, bs, W, 10)
                   for i in range(3)]
        return r, A, B, C

    def assoc_pen(bs):
        _, A, B, C = rand_regs(bs)
        l = model.add_scan(model.add_scan(A, B), C)
        rr = model.add_scan(A, model.add_scan(B, C))
        return agree(l, rr)

    def cancel_pen(bs):
        r, A, B, C = rand_regs(bs)
        same = (r[1] == r[2]).all(-1).float()[None, :, None]
        p, q = model.add_scan(A, B), model.add_scan(A, C)
        ov = (p * q).sum(-1)
        return ((1 - same) * ov).mean()

    t0 = time.time()
    for step in range(1, args.steps + 1):
        i0 = torch.randint(0, xin.shape[0], (args.batch,), device=dev)
        bi, bt = xin[i0], xt[i0]
        bie = bi[None].expand(P, *bi.shape)
        mults = model.multiples(nd)
        model._prep(mults)
        parts = {}
        loss = torch.zeros((), device=dev)
        r = None
        if args.label_w > 0 or args.dual != "none" or args.rel != "none":
            r = model.square(bie, mults)
        if args.label_w > 0:
            loss = args.label_w * F.cross_entropy(
                torch.log(r + 1e-9).reshape(-1, 10),
                bt[None].expand(P, *bt.shape).reshape(-1))

        if args.dual != "none":
            nb = args.dual_batch or args.batch
            rb = model.square(bie[:, :nb], mults, args.dual)
            d = agree(r[:, :nb], rb)
            parts["dual"] = d.item()
            loss = loss + args.dual_w * d
            if args.dual_label > 0:
                loss = loss + args.dual_label * F.cross_entropy(
                    torch.log(rb + 1e-9).reshape(-1, 10),
                    bt[:nb][None].expand(P, nb, S).reshape(-1))

        if args.rel != "none":
            nb = min(args.rel_batch, args.batch)
            sub = bie[:, :nb]
            cc = F.softmax(rel_c, -1)[:, None].expand(P, nb, S, 10)
            inc = model.addmod(sub, cc, mults)
            lhs = model.square(inc, mults)
            D = freeD(bi[:nb])
            rhs = model.addmod(r[:, :nb], D, mults)
            rl = agree(lhs, rhs)
            parts["rel"] = rl.item()
            loss = loss + args.rel_w * rl
            if args.rel_nondeg > 0:
                zc = F.softmax(model.zero, -1)[:, None].expand(P, S, 10)
                ov = (F.softmax(rel_c, -1) * zc).sum(-1).mean()
                loss = loss + args.rel_nondeg * ov
                parts["nondeg"] = ov.item()

        for nm, wgt, fn in (("sym", args.sym, sym_pen),
                            ("inv", args.inv, inv_pen)):
            if wgt > 0:
                p = fn()
                parts[nm] = p.item()
                loss = loss + wgt * p
        for nm, wgt, fn in (("assoc", args.assoc, assoc_pen),
                            ("cancel", args.cancel, cancel_pen)):
            if wgt > 0:
                p = fn(args.law_batch)
                parts[nm] = p.item()
                loss = loss + wgt * p

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip:
            pop_clip_(model, args.clip)
            aux = [p for p in params if p.dim() and p.grad is not None
                   and p is not None and not any(p is q for q in
                                                 model.replica_params())]
            if aux:
                torch.nn.utils.clip_grad_norm_(aux, args.clip * math.sqrt(P))
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            lce = local_ce_vec()
            ev = eval_pop(model, xin[:args.eval_n], xt[:args.eval_n], nd, True,
                          args.eval_chunk)
            print(f"[{args.tag}] step={step:>5} loss={loss.item():.4f} "
                  f"{ {k: round(v, 4) for k, v in parts.items()} } "
                  f"lce_min={lce.min():.3f} lce_med={lce.median():.3f} "
                  f"lce_max={lce.max():.3f} n_lce006={(lce < 0.006).sum()} "
                  f"n_lce1={(lce < 1.0).sum()} best={ev['best']:.3f} "
                  f"mix={ev['mix']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    lce = local_ce_vec()
    div = diversity_vec()
    tr = eval_pop(model, xin[:args.eval_n], xt[:args.eval_n], nd, True,
                  args.eval_chunk)
    he = eval_pop(model, hin, ht, nd, True, args.eval_chunk)
    q = torch.quantile(lce, torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 1.0]))
    out = {"tag": args.tag, "argv": sys.argv[1:], "pop": P,
           "legal": True, "opt": args.opt,
           "local_ce_min": round(float(lce.min()), 4),
           "local_ce_p10": round(float(q[1]), 4),
           "local_ce_med": round(float(q[3]), 4),
           "local_ce_max": round(float(lce.max()), 4),
           "local_ce_spread": round(float((lce.max() - lce.min())
                                          / lce.median()), 3),
           "local_ce_quantiles": [round(float(v), 4) for v in q],
           "local_ce_sorted": [round(float(v), 4)
                               for v in lce.sort().values[:12]],
           "n_lce_006": int((lce < 0.006).sum()),
           "n_lce_010": int((lce < 0.010).sum()),
           "n_lce_100": int((lce < 1.0).sum()),
           "n_lce_234": int((lce < 2.34).sum()),
           "div_min": round(float(div.min()), 3),
           "div_med": round(float(div.median()), 3),
           "div_max": round(float(div.max()), 3),
           "div_at_lce_min": round(float(div[int(lce.argmin())]), 3),
           "train_exact_hard_best": round(float(tr["per"].max()), 4),
           "train_exact_hard_mix": round(tr["mix"], 4),
           "train_exact_hard_argmax": round(tr["argmax"], 4),
           "held_exact_hard_best": round(float(he["per"].max()), 4),
           "held_exact_hard_mix": round(he["mix"], 4),
           "held_exact_hard_argmax": round(he["argmax"], 4),
           "n_basin": int((tr["per"] > 0.5).sum()),
           "secs": round(time.time() - t0, 1),
           "ms_per_step": round(1000 * (time.time() - t0) / args.steps, 1),
           "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
    print(f"[{args.tag}] FINAL " + " ".join(
        f"{k}={v}" for k, v in out.items() if k != "argv"), flush=True)
    if args.jsonl:
        with open(args.jsonl, "a") as fh:
            fh.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
