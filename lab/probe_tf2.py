#!/usr/bin/env python
"""LAB ONLY -- Stage 1: what is the per-step-signal CEILING under the best
conditions now available?

My earlier teacher-forcing number (`train_exact_hard` 0.064) was taken at e1
scale, on the 257-step Horner graph, with no weight ties.  `discrete-search`
closed direct discrete search and concluded the only thing that reopens the
`DigitALU` family is a legal source of per-step signal.  Before anyone builds a
legal version, this measures the *ideal illegal* version with three
improvements stacked:

  * m1-scale operands (N=10403, S=5, ~10,200 units) instead of 250;
  * `alu-depth`'s tree:quotient graph (39 sequential steps, ceiling 1.000
    soft AND hard);
  * weight tying -- Tmul and Tadd symmetric in their two digit arguments,
    Tsub's digit slice DERIVED from Tadd by the exact inverse relation
    (add(u,v,c) -> (w,c')  <=>  sub(w,v,c) -> (u,c')).

HOW THE TARGETS ARE MADE.  The tree:quotient call order is intricate (a
multiples prefix, a balanced product tree, then a long division with a batched
candidate scan), so rather than reimplement it I run a CONSTRUCTED copy of the
same model through the SAME code path in `record` mode and tape every scan
output in call order.  The learner then replays that tape in `force` mode.
Order alignment is guaranteed because it is literally the same code.

This uses the constructed tables to build the tape, so it is a LAB DIAGNOSTIC
and NOT a legal submission (rules 2 and 7).  It measures a ceiling, not a
recipe.

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
from probe_alu_depth import DigitALU, digits_le  # noqa: E402


class TFALU(DigitALU):
    def __init__(self, *a, tie=False, tie_sub=False, **kw):
        self.tie, self.tie_sub = tie, tie_sub
        self.mode = None            # None | 'record' | 'force'
        self.tape: list = []
        self.tpos = 0
        self.tf_loss = None
        self.tf_n = 0
        self.tf_p = 1.0
        super().__init__(*a, **kw)

    # ---- weight ties (a reparameterisation; the forward is unchanged) ----
    @property
    def Tmul_eff(self):
        return 0.5 * (self.Tmul + self.Tmul.transpose(0, 1)) if self.tie \
            else self.Tmul

    @property
    def Tadd_eff(self):
        return 0.5 * (self.Tadd + self.Tadd.transpose(0, 1)) if self.tie \
            else self.Tadd

    @property
    def Tsub_eff(self):
        if not self.tie_sub:
            return self.Tsub
        # add(u,v,c) -> (w, c')   <=>   sub(w,v,c) -> (u, c')
        dig = self.Tadd_eff[..., :10].permute(3, 1, 2, 0)      # (w,v,c,u)
        return torch.cat([dig, self.Tsub[..., 10:]], -1)

    # ---- the tap ----
    def _tap(self, x):
        if self.mode == 'record':
            self.tape.append(x.detach())
            return x
        if self.mode == 'force':
            truth = self.tape[self.tpos]
            self.tpos += 1
            lg = x.clamp_min(1e-9).log()
            self.tf_loss = self.tf_loss + F.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), truth.argmax(-1).reshape(-1))
            self.tf_n += 1
            if self.tf_p >= 1.0:
                return truth
            m = torch.rand(x.shape[0], *([1] * (x.dim() - 1)),
                           device=x.device) < self.tf_p
            return torch.where(m, truth, x)
        return x

    # ---- scans, re-expressed against the tied tensors and tapped ----
    def add_scan(self, r, addend):
        c = self._sm(self.carry0).expand(r.shape[0], self.Ca)
        outs = []
        for m in range(r.shape[1]):
            o = torch.einsum("bu,bv,bc,uvco->bo", r[:, m], addend[:, m], c,
                             self.Tadd_eff)
            outs.append(self._sm(o[:, :10] + self.copy_scale * r[:, m]))
            c = self._sm(o[:, 10:] + self.copy_scale * c)
        return self._tap(torch.stack(outs, 1))

    def sub_scan(self, r, sub):
        b = r.shape[0]
        c = self._sm(self.borrow0).expand(b, self.Cb)
        outs = []
        for m in range(r.shape[1]):
            o = torch.einsum("bu,bv,bc,uvco->bo", r[:, m], sub[:, m], c,
                             self.Tsub_eff)
            outs.append(self._sm(o[:, :10] + self.copy_scale * r[:, m]))
            c = self._sm(o[:, 10:] + self.copy_scale * c)
        return self._tap(torch.stack(outs, 1)), c

    def quot_reduce(self, r, mults):
        b, M = r.shape[0], mults.shape[0]
        rr = r[:, None].expand(b, M, self.W, 10).reshape(b * M, self.W, 10)
        ss = mults[None].expand(b, M, self.W, 10).reshape(b * M, self.W, 10)
        t, c = self.sub_scan(rr, ss)
        t = t.view(b, M, self.W, 10)
        c = c.view(b, M, self.Cb)
        pair = torch.cat([c[:, :-1], c[:, 1:]], dim=-1)
        w = self._sm(self.sel(pair).squeeze(-1))
        self.last_q = w
        return self._tap(torch.einsum("bm,bmwo->bwo", w, t[:, :-1]))

    def forward(self, s, ndig):
        b = s.shape[0]
        self.tpos = 0
        z = self._sm(self.zero).expand(b, 10)
        mults = self.multiples(ndig) if self.needed else None
        prod = {}
        for i in range(self.S):
            for j in range(self.S):
                o = torch.einsum("bu,bv,uvo->bo", s[:, i], s[:, j],
                                 self.Tmul_eff)
                prod[(i, j)] = (self._sm(o[:, :10]), self._sm(o[:, 10:]))
        return self.forward_tree(s, ndig, mults, prod, z)


@torch.no_grad()
def scores(model, ref, ndigits):
    """cell_agree (raw argmax vs the construction) + the gauge-invariant
    structure scores from lab/probe_credit.py."""
    out, tot, ok = {}, 0, 0
    for name in ("Tmul_eff", "Tadd_eff", "Tsub_eff"):
        g, w = getattr(model, name), getattr(ref, name)
        a = (g.argmax(-1) == w.argmax(-1)).float()
        out["cell_" + name[1:4]] = round(a.mean().item(), 3)
        ok += a.sum().item()
        tot += a.numel()
    out["cell_agree"] = round(ok / tot, 3)
    # gauge-corrected Tmul: recover the best relabelling pi of the output code
    # from the (a*b)%10 classes, then score under it.  The uncorrected version
    # has a hole -- a CONSTANT map scores 1.000 -- so it is reported as
    # `mul_fn` (is it a function of a*b at all) alongside `mul_gauge` (is that
    # function a bijection, i.e. an actual relabelled multiplication table).
    am = model.Tmul_eff[..., :10].argmax(-1)
    groups = {}
    for a in range(10):
        for b in range(10):
            groups.setdefault((a * b) % 10, []).append(int(am[a, b]))
    k = sum(v.count(max(set(v), key=v.count)) for v in groups.values())
    out["mul_fn"] = round(k / 100, 3)
    pi = {c: max(set(v), key=v.count) for c, v in groups.items()}
    out["mul_gauge"] = round(len(set(pi.values())) / len(pi), 3)
    for nm, T, cols in (("add_shift", model.Tadd_eff, range(10)),
                        ("sub_shift", model.Tsub_eff, sorted(set(ndigits)))):
        k = n = 0
        for v in cols:
            for c in range(T.shape[2]):
                a = T[:, v, c, :10].argmax(-1)
                k += max(sum(1 for u in range(10) if int(a[u]) == (u + s) % 10)
                         for s in range(10))
                n += 10
        out[nm] = round(k / n, 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulus", type=int, default=10403)
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--train-x", type=int, default=8000)
    ap.add_argument("--held-x", type=int, default=1024)
    ap.add_argument("--max-quot", type=int, default=10)
    ap.add_argument("--tie", action="store_true")
    ap.add_argument("--tie-sub", action="store_true")
    ap.add_argument("--teacher-force", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--construct", action="store_true")
    ap.add_argument("--jsonl", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    N, S = args.modulus, args.slots
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

    mk = lambda: TFALU(S, 2, 2, 11, 1.0, 0.0, False, "quotient",
                       args.max_quot, "tree", "serial",
                       tie=args.tie, tie_sub=args.tie_sub).to(dev)
    model = mk()
    ref = mk()
    ref.construct()
    for p in ref.parameters():
        p.requires_grad_(False)

    free = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] N={N} S={S} tree:quotient tie={args.tie} "
          f"tie_sub={args.tie_sub} params={free:,} "
          f"train={len(tr_x)} held={len(he_x)} units={len(units)}", flush=True)

    @torch.no_grad()
    def ev(m, inp, tgt, discrete=False, chunk=512):
        m.eval()
        was, m.hard = m.hard, True if discrete else m.hard
        ok = 0
        for i in range(0, inp.shape[0], chunk):
            lg = m(inp[i:i + chunk], nd)
            ok += (lg.argmax(-1) == tgt[i:i + chunk]).all(-1).sum().item()
        m.hard = was
        m.train()
        return ok / inp.shape[0]

    if args.construct:
        model.construct()
        print(f"[{args.tag}] CONSTRUCTED train_exact={ev(model, xin, xt):.3f} "
              f"train_exact_hard={ev(model, xin, xt, True):.3f} "
              f"held_exact_hard={ev(model, hin, ht, True):.3f}", flush=True)
        return 0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    t0 = time.time()
    nd_dig = digits_le(N, W)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, xin.shape[0], (args.batch,), device=dev)
        bi, bt = xin[idx], xt[idx]
        with torch.no_grad():                       # tape the true trace
            ref.mode, ref.tape = 'record', []
            ref(bi, nd)
            ref.mode = None
        model.mode, model.tape = 'force', ref.tape
        model.tf_loss, model.tf_n, model.tf_p = torch.zeros((), device=dev), 0, \
            args.teacher_force
        logits = model(bi, nd)
        model.mode = None
        loss = F.cross_entropy(logits.reshape(-1, 10), bt.reshape(-1)) \
            + model.tf_loss / max(model.tf_n, 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            sc = scores(model, ref, nd_dig)
            print(f"[{args.tag}] step={step:>5} loss={loss.item():.4f} "
                  f"train_exact={ev(model, xin[:1024], xt[:1024]):.3f} "
                  f"train_exact_hard={ev(model, xin[:1024], xt[:1024], True):.3f} "
                  f"held_exact_hard={ev(model, hin, ht, True):.3f} "
                  f"{sc} ({time.time()-t0:.0f}s)", flush=True)
    trh = ev(model, xin[:2048], xt[:2048], True)
    heh = ev(model, hin, ht, True)
    sc = scores(model, ref, nd_dig)
    # the decisive decomposition: how well does each op fit its own LOCAL task,
    # given true inputs?  If this is ~0 while train_exact_hard is ~0.1, every
    # table is locally right on the state distribution it sees and the residual
    # is composition, not learning.
    with torch.no_grad():
        ref.mode, ref.tape = 'record', []
        ref(xin[:512], nd)
        ref.mode = None
        model.mode, model.tape = 'force', ref.tape
        model.tf_loss, model.tf_n, model.tf_p = torch.zeros((), device=dev), 0, 1.0
        model(xin[:512], nd)
        model.mode = None
        sc["local_ce"] = round((model.tf_loss / max(model.tf_n, 1)).item(), 4)
    print(f"[{args.tag}] FINAL train_exact={ev(model, xin[:2048], xt[:2048]):.3f} "
          f"train_exact_hard={trh:.3f} held_exact={ev(model, hin, ht):.3f} "
          f"held_exact_hard={heh:.3f} {sc} ({time.time()-t0:.0f}s)", flush=True)
    if args.jsonl:
        with open(args.jsonl, "a") as fh:
            fh.write(json.dumps({"tag": args.tag, "argv": sys.argv[1:],
                                 "train_exact_hard": trh,
                                 "held_exact_hard": heh, **sc,
                                 "secs": round(time.time() - t0, 1)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
