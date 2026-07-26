#!/usr/bin/env python
"""LAB ONLY -- training-procedure search for `DigitALU` (lab/probe_alu.py).

The architecture is held FIXED (that is `alu-depth`'s lane).  Everything here
varies *how it is optimised*:

  * chain-length curriculum on R (the number of tied conditional subtractions
    executed per Horner place).  R is a graph-length knob, not a parameter
    count: the tables are shared across every cond_sub, so a table learned at
    R=3 is the same table at R=11.  Evaluation ALWAYS runs at full R, so the
    constructed ceiling of the reported configuration is unchanged.
  * cross-(modulus, slots) curriculum.  LAB ONLY -- a submission never sees a
    second modulus, so this is evidence about learnability, not a recipe.
  * init scale / temperature / gate-bias / copy-through, each separable (the
    report's `--identity-init` confounded copy-through with a closed gate).
  * Gumbel noise, entropy pressure toward one-hot tables, straight-through.
  * staged gradient release (mul -> add -> sub) from RANDOM init, which is a
    different object from the report's "freeze at truth".
  * label-free algebraic consistency regularisers (Tmul/Tadd commutativity,
    Tsub inverts Tadd).  These live in the loss, never in the forward pass.
  * optimiser / lr / decay / schedule.
  * deep supervision on the true intermediate Horner registers.  LAB ONLY.

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
from probe_alu import BIG, DigitALU, digits_le  # noqa: E402


@torch.no_grad()
def _spread(r):
    """How much of the input still reaches the register: mean over slots of the
    across-batch std of the digit distribution."""
    return r.std(dim=0).mean().item()


class CreditALU(DigitALU):
    """DigitALU with training-procedure hooks.  Forward semantics identical."""

    def __init__(self, *a, **kw):
        self.noise = 0.0
        self.trunc = 0          # detach the register every `trunc` Horner places
        self.collect = False
        self.trace = False
        self.gate_off = 0.0
        self.hard_gate = False
        super().__init__(*a, **kw)
        self.inter = []
        self.flow = []
        self.tf = None          # (B,P,W) true register trace, LAB ONLY
        self.tf_p = 0.0
        self._op = 0

    def _record(self, r):
        """Log the register, then optionally overwrite it with the truth
        (teacher forcing).  LAB ONLY -- it needs the true trace."""
        if self.collect:
            self.inter.append(r)
        if self.trace:
            self.flow.append(("op", _spread(r)))
        idx, self._op = self._op, self._op + 1
        if self.tf is not None and self.training:
            tgt = F.one_hot(self.tf[:r.shape[0], idx], 10).to(r.dtype)
            if self.tf_p >= 1.0:
                return tgt
            m = (torch.rand(r.shape[0], 1, 1, device=r.device) < self.tf_p)
            return torch.where(m, tgt, r)
        return r

    def _sm(self, logits):
        if self.noise > 0.0 and self.training:
            u = torch.rand_like(logits).clamp_(1e-9, 1 - 1e-9)
            logits = logits - self.noise * torch.log(-torch.log(u))
        p = F.softmax(logits / self.tau, -1)
        if self.hard:
            h = F.one_hot(p.argmax(-1), p.shape[-1]).to(p.dtype)
            p = h + p - p.detach()
        return p

    def cond_sub(self, r, ndig):
        """Identical to DigitALU.cond_sub except for `gate_off`, a SCHEDULED
        additive offset on the gate logit annealed to exactly 0 before training
        ends -- a temperature-style schedule, not a change to the model."""
        b = r.shape[0]
        c = self._sm(self.borrow0).expand(b, self.Cb)
        outs = []
        for m in range(self.W):
            o = torch.einsum("bu,bv,bc,uvco->bo", r[:, m], ndig[m].expand(b, 10),
                             c, self.Tsub)
            outs.append(self._sm(o[:, :10] + self.copy_scale * r[:, m]))
            c = self._sm(o[:, 10:] + self.copy_scale * c)
        t = torch.stack(outs, 1)
        g = torch.sigmoid(self.gate(c) + self.gate_off)
        if self.hard_gate:
            # straight-through on the gate too: without this the sigmoid blend
            # re-softens the register every cond_sub and the chain contracts.
            g = (g > 0.5).to(g.dtype) + g - g.detach()
        g = g[:, :, None]
        return g * t + (1 - g) * r

    # ---- the chain as an explicit op list, so a single op can be applied to an
    # ---- arbitrary register (needed by target propagation)
    def op_schedule(self):
        ops = []
        for k in range(self.K - 1, -1, -1):
            first = True
            for i in range(self.S):
                j = k - i
                if 0 <= j < self.S:
                    ops.append(("add", (i, j), first))
                    first = False
            for t in range(self.R):
                ops.append(("sub", None, first and t == 0))
        return ops

    def apply_op(self, r, op, prod, z, ndig):
        kind, arg, shift = op
        if shift:
            r = torch.cat([z[:, None], r[:, : self.W - 1]], dim=1)   # x10
        if kind == "add":
            lo, hi = prod[arg]
            slots = [lo[:, None], hi[:, None]] + [z[:, None]] * (self.W - 2)
            return self.add_scan(r, torch.cat(slots, dim=1))
        return self.cond_sub(r, ndig)

    def products(self, s):
        prod = {}
        for i in range(self.S):
            for j in range(self.S):
                o = torch.einsum("bu,bv,uvo->bo", s[:, i], s[:, j], self.Tmul)
                prod[(i, j)] = (self._sm(o[:, :10]), self._sm(o[:, 10:]))
        return prod

    def forward(self, s, ndig):
        b = s.shape[0]
        z = self._sm(self.zero).expand(b, 10)
        prod = self.products(s)
        r = z[:, None].expand(b, self.W, 10)
        self.inter, self.flow, self._op = [], [], 0
        place = 0
        for op in self.op_schedule():
            if op[2]:
                place += 1
                if self.trunc and place > 1 and (place - 1) % self.trunc == 0:
                    r = r.detach()
            r = self._record(self.apply_op(r, op, prod, z, ndig))
        return torch.log(r[:, : self.S] + 1e-9)


# ---------------------------------------------------------------- consistency
def sym_loss(model):
    """Commutativity of the two digit-indexed binary tables.  Label free."""
    a = model.Tmul - model.Tmul.transpose(0, 1)
    b = model.Tadd - model.Tadd.transpose(0, 1)
    return (a ** 2).mean() + (b ** 2).mean()


def inv_loss(model):
    """`Tsub` inverts `Tadd`.  Label free, value free.

    If add(u,v,c) -> (w, c') then sub(w,v,c) -> (u, c').  This pins the
    RELATION between two learned tables; it says nothing about what either of
    them computes.  It lives in the loss, not the forward pass.
    """
    tau = model.tau
    A = F.softmax(model.Tadd / tau, -1)            # (10,10,2,12)
    Sb = F.softmax(model.Tsub / tau, -1)           # (10,10,2,12)
    pw, pc = A[..., :10], A[..., 10:]              # (10,10,2,10), (...,2)
    # sub(w, v, c) marginalised over the add's output digit distribution
    out = torch.einsum("uvcw,wvco->uvco", pw, Sb)  # (10,10,2,12)
    dig = out[..., :10].clamp_min(1e-9)
    bor = out[..., 10:].clamp_min(1e-9)
    tgt = torch.eye(10, device=A.device)[:, None, None, :].expand_as(dig)
    l_dig = -(tgt * dig.log()).sum(-1).mean()
    l_bor = -(pc.detach() * bor.log()).sum(-1).mean()
    return l_dig + l_bor


@torch.no_grad()
def structure_scores(model, ndigits):
    """GAUGE-INVARIANT "did it learn the algorithm" scores.

    Raw argmax-vs-truth is meaningless here: `Tmul`'s output alphabet is
    consumed only by `Tadd`'s addend index, so any permutation pi of the digit
    symbols fixing the zero symbol can be applied to both without changing the
    function, and the two borrow states can be swapped.  These scores are
    invariant to all of that.

      mul_lo/mul_hi : is the Tmul argmax a well-defined FUNCTION of (a*b)%10
                      / (a*b)//10?  (any relabelling still passes)
      add_shift     : for each addend column, is u -> out a cyclic shift of the
                      identity?  addition by a constant is exactly that.
      sub_shift     : the same, restricted to the columns N's digits reach --
                      the only Tsub entries training can ever touch.

    Random-table baseline for the shift scores is ~0.27 (best of 10 shifts on
    10 cells); 1.000 means every column is exactly an add/subtract-a-constant.
    """
    out = {}
    for sl, name, fn in ((slice(0, 10), "mul_lo", lambda a, b: (a * b) % 10),
                         (slice(10, 20), "mul_hi", lambda a, b: (a * b) // 10)):
        am = model.Tmul[..., sl].argmax(-1)
        groups = {}
        for a in range(10):
            for b in range(10):
                groups.setdefault(fn(a, b), []).append(int(am[a, b]))
        ok = tot = 0
        for vs in groups.values():
            ok += vs.count(max(set(vs), key=vs.count))
            tot += len(vs)
        out[name] = round(ok / tot, 3)
    for name in ("add", "sub"):
        T = model.Tadd if name == "add" else model.Tsub
        cols = range(10) if name == "add" else sorted(set(ndigits))
        ok = tot = 0
        for v in cols:
            for c in range(T.shape[2]):
                am = T[:, v, c, :10].argmax(-1)
                ok += max(sum(1 for u in range(10)
                              if int(am[u]) == (u + s) % 10) for s in range(10))
                tot += 10
        out[name + "_shift"] = round(ok / tot, 3)
    return out


class LatentTrace(torch.nn.Module):
    """Amortised predictor of the register trace -- the LEGAL analogue of
    teacher forcing.

    Teacher forcing (below) needs the TRUE trace and is therefore a lab
    diagnostic.  Here the trace is a *learned latent* predicted from the same
    inputs the model already sees, trained jointly with the tables under
    (a) local consistency -- one op applied to latent t must reproduce latent
    t+1 -- and (b) two boundary conditions that use only the given label:
    the last latent is the answer, the first input is the model's own `zero`.
    Nothing about arithmetic is supplied.  This is method-of-auxiliary-
    coordinates / target propagation, and it is discarded at eval.
    """

    def __init__(self, slots, width, n_ops, hidden=128):
        super().__init__()
        self.W, self.P = width, n_ops
        self.pos = torch.nn.Parameter(torch.randn(n_ops, hidden) * 0.05)
        self.inp = torch.nn.Linear(slots * 10, hidden)
        self.out = torch.nn.Linear(hidden, width * 10)

    def forward(self, s):
        h = torch.tanh(self.inp(s.flatten(1))[:, None] + self.pos[None])
        return self.out(h).view(s.shape[0], self.P, self.W, 10)


def tprop_loss(model, latent, s, ndig, label, tau):
    ops = model.op_schedule()
    L = F.softmax(latent(s) / tau, -1)                 # (B,P,W,10)
    prod = model.products(s)
    z = model._sm(model.zero).expand(s.shape[0], 10)
    prev = z[:, None].expand(s.shape[0], model.W, 10)
    tot = 0.0
    for t, op in enumerate(ops):
        pred = model.apply_op(prev, op, prod, z, ndig).clamp_min(1e-9)
        tgt = L[:, t]
        tot = tot + -(tgt.detach() * pred.log()).sum(-1).mean() \
                  + -(pred.detach() * tgt.clamp_min(1e-9).log()).sum(-1).mean()
        prev = tgt
    tot = tot / len(ops)
    # boundary: the final latent IS the answer.  Slots >= S are zero because
    # the result is < N and N has S digits -- that is the digit count of the
    # modulus, which the model is given.
    fin = L[:, -1].clamp_min(1e-9).log()
    anchor = F.cross_entropy(fin[:, :model.S].reshape(-1, 10), label.reshape(-1))
    if model.W > model.S:
        hi = fin[:, model.S:].reshape(-1, 10)
        anchor = anchor + F.cross_entropy(
            hi, torch.zeros(hi.shape[0], dtype=torch.long, device=hi.device))
    return tot + anchor


def entropy_loss(model):
    tot = 0.0
    for t in (model.Tmul[..., :10], model.Tmul[..., 10:],
              model.Tadd[..., :10], model.Tadd[..., 10:],
              model.Tsub[..., :10], model.Tsub[..., 10:]):
        p = F.softmax(t / model.tau, -1)
        tot = tot + -(p * p.clamp_min(1e-9).log()).sum(-1).mean()
    return tot / 6.0


# ------------------------------------------------------------------ task data
def make_task(modulus, slots, train_x, split_seed, device):
    units = [x for x in range(1, modulus) if math.gcd(x, modulus) == 1]
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(units), generator=g).tolist()
    tr = [units[i] for i in perm[:train_x]]
    he = [units[i] for i in perm[train_x:]]

    def tensors(xs):
        inp = torch.zeros(len(xs), slots, 10)
        tgt = torch.zeros(len(xs), slots, dtype=torch.long)
        for r, x in enumerate(xs):
            for i, d in enumerate(digits_le(x, slots)):
                inp[r, i, d] = 1.0
            for i, d in enumerate(digits_le((x * x) % modulus, slots)):
                tgt[r, i] = d
        return inp.to(device), tgt.to(device)

    W = slots + 1
    nd = torch.zeros(W, 10)
    for i, d in enumerate(digits_le(modulus, W)):
        nd[i, d] = 1.0
    return tensors(tr), tensors(he), nd.to(device), tr, he


def horner_targets(xs, modulus, slots, reduce_steps, device):
    """True register after EVERY add_scan and EVERY cond_sub.  LAB ONLY.

    Mirrors CreditALU.forward exactly, so target j lines up with
    `model.inter[j]`.  This is the finest possible deep supervision: the
    gradient then only has to cross one 4-slot scan.
    """
    K, W = 2 * slots - 1, slots + 1
    rows = []
    for x in xs:
        d = digits_le(x, slots)
        seq, r = [], 0
        for k in range(K - 1, -1, -1):
            r = 10 * r
            for i in range(slots):
                j = k - i
                if 0 <= j < slots:
                    r += d[i] * d[j]
                    seq.append(r)
            for _ in range(reduce_steps):
                if r >= modulus:
                    r -= modulus
                seq.append(r)
        rows.append([digits_le(v, W) for v in seq])
    return torch.tensor(rows, dtype=torch.long, device=device)  # (B, P, W)


@torch.no_grad()
def table_accuracy(model, ndigits=None):
    """Fraction of table rows whose argmax equals the constructed truth.

    CAVEAT -- there is a gauge freedom.  `Tmul`'s output alphabet is consumed
    only by `Tadd`'s addend index, so any permutation pi of the digit symbols
    with pi(zero) = zero can be applied to Tmul's outputs and Tadd's `v` index
    together without changing the function.  So Tmul and Tadd accuracy are
    uninformative (chance is the expected reading for a correct model).

    `Tsub` has NO gauge freedom: its `u` index is the register, its `v` index
    is a digit of N supplied as a fixed one-hot, and its output is the
    register.  `Tsub_N` restricts it to the `v` columns that N actually uses --
    the only entries any amount of training can reach -- and is therefore the
    one honest "did it find the intended solution" number.
    """
    ref = CreditALU(model.S, model.Ca, model.Cb, model.R).to(model.Tmul.device)
    ref.construct()
    out = {}
    for name in ("Tmul", "Tadd", "Tsub"):
        got, want = getattr(model, name), getattr(ref, name)
        lo = (got[..., :10].argmax(-1) == want[..., :10].argmax(-1)).float().mean()
        hi = (got[..., 10:].argmax(-1) == want[..., 10:].argmax(-1)).float().mean()
        out[name] = (round(lo.item(), 3), round(hi.item(), 3))
    if ndigits:
        cols = sorted(set(ndigits))
        g, w = model.Tsub[:, cols], ref.Tsub[:, cols]
        lo = (g[..., :10].argmax(-1) == w[..., :10].argmax(-1)).float().mean()
        hi = (g[..., 10:].argmax(-1) == w[..., 10:].argmax(-1)).float().mean()
        out["Tsub_N"] = (round(lo.item(), 3), round(hi.item(), 3))
    return out


# ---------------------------------------------------------------------- train
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modulus", type=int, default=323)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--train-x", type=int, default=250)
    ap.add_argument("--reduce", type=int, default=11)
    ap.add_argument("--carry", type=int, default=2)
    ap.add_argument("--borrow", type=int, default=2)
    # --- relaxation
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--tau-final", type=float, default=None)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--noise-final", type=float, default=None)
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--hard-gate", action="store_true",
                    help="straight-through the cond_sub gate as well")
    ap.add_argument("--hard-at", type=float, default=None,
                    help="switch to straight-through after this fraction")
    # --- init
    ap.add_argument("--init-scale", type=float, default=1.0,
                    help="multiply all table logits at init (1.0 = randn*0.5)")
    ap.add_argument("--onehot-init", type=float, default=0.0,
                    help="init every table row as a RANDOM one-hot * this logit")
    ap.add_argument("--perm-init", type=float, default=0.0,
                    help="init Tadd/Tsub digit rows as a RANDOM PERMUTATION in "
                         "the register index (the truth is a permutation there "
                         "too, so this is right structure / wrong values)")
    ap.add_argument("--copy-scale", type=float, default=0.0)
    ap.add_argument("--gate-bias", type=float, default=None)
    ap.add_argument("--gate-off", type=float, default=0.0,
                    help="scheduled additive gate-logit offset (annealed to 0)")
    ap.add_argument("--gate-warm", type=float, default=0.5,
                    help="fraction of steps over which --gate-off reaches 0")
    # --- curricula
    ap.add_argument("--r-start", type=int, default=None,
                    help="start the cond_sub chain at this R and grow to --reduce")
    ap.add_argument("--r-warm", type=float, default=0.5,
                    help="fraction of steps over which R reaches --reduce")
    ap.add_argument("--curr", default="",
                    help="LAB ONLY cross-task curriculum 'N:S:steps,N:S:steps'")
    ap.add_argument("--xcurr", type=float, default=0.0,
                    help="LEGAL magnitude curriculum: weight training examples "
                         "by |x| <= a threshold that grows from 10 to N over "
                         "this fraction of the run.  Small x need no reduction "
                         "and only the last Horner place is non-trivial, so "
                         "this is a chain-length curriculum expressed purely as "
                         "a per-example loss weight on the GIVEN data.")
    ap.add_argument("--trunc", type=int, default=0,
                    help="detach the register every this many Horner places")
    ap.add_argument("--stage", default="",
                    help="staged gradient release from RANDOM init, "
                         "'mul:0,add:300,sub:600,gate:600,zero:0'")
    # --- loss
    ap.add_argument("--sym", type=float, default=0.0)
    ap.add_argument("--inv", type=float, default=0.0)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--tprop", type=float, default=0.0,
                    help="target-propagation weight (LEGAL: learned latents)")
    ap.add_argument("--tprop-hidden", type=int, default=128)
    ap.add_argument("--tprop-tau", type=float, default=1.0)
    ap.add_argument("--main-warm", type=float, default=0.0,
                    help="ramp the free-running CE in over this fraction")
    ap.add_argument("--deep-sup", type=float, default=0.0,
                    help="LAB ONLY: CE on the true intermediate Horner registers")
    ap.add_argument("--teacher-force", type=float, default=0.0,
                    help="LAB ONLY: probability of overwriting the register "
                         "with the true trace after every op")
    ap.add_argument("--tf-decay", type=float, default=None,
                    help="fraction of steps over which teacher forcing -> 0")
    ap.add_argument("--deep-sup-decay", type=float, default=None,
                    help="fraction of steps after which deep-sup weight is 0")
    # --- optimiser
    ap.add_argument("--opt", default="adamw",
                    choices=["adamw", "sgd", "sign", "lion", "rmsprop"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--lr-final", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=0, help="0 = full batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--construct", action="store_true",
                    help="LAB DIAGNOSTIC: ceiling of the reported configuration")
    ap.add_argument("--diag", action="store_true",
                    help="report per-module grad norms + table accuracy")
    ap.add_argument("--dump-tables", action="store_true",
                    help="print learned vs true argmax grids at the end")
    ap.add_argument("--flow", action="store_true",
                    help="print the across-batch register spread at every step")
    ap.add_argument("--jsonl", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    (xin, xt), (hin, ht), ndig, train_x, held_x = make_task(
        args.modulus, args.slots, args.train_x, args.split_seed, device)

    # copy_scale and gate-bias are set separately: `probe_alu --identity-init`
    # changed BOTH at once, so the report's identity-init rows confound them.
    model = CreditALU(args.slots, args.carry, args.borrow, args.reduce,
                      args.tau, 0.0, args.hard).to(device)
    with torch.no_grad():
        model.copy_scale.fill_(args.copy_scale)
    if args.init_scale != 1.0:
        with torch.no_grad():
            for t in (model.Tmul, model.Tadd, model.Tsub, model.zero,
                      model.carry0, model.borrow0):
                t.mul_(args.init_scale)
    if args.onehot_init > 0:
        with torch.no_grad():
            for t in (model.Tmul[..., :10], model.Tmul[..., 10:],
                      model.Tadd[..., :10], model.Tadd[..., 10:],
                      model.Tsub[..., :10], model.Tsub[..., 10:]):
                idx = torch.randint(0, t.shape[-1], t.shape[:-1], device=t.device)
                t.copy_(F.one_hot(idx, t.shape[-1]).to(t.dtype) * args.onehot_init)
    if args.perm_init > 0:
        with torch.no_grad():
            k = args.perm_init
            for t in (model.Tadd, model.Tsub):     # (10,10,C,10+C)
                for v in range(10):
                    for c in range(t.shape[2]):
                        pi = torch.randperm(10, device=t.device)
                        t[:, v, c, :10] = F.one_hot(pi, 10).to(t.dtype) * k
            for bdig in range(10):                 # Tmul (10,10,20)
                pi = torch.randperm(10, device=model.Tmul.device)
                model.Tmul[:, bdig, :10] = F.one_hot(pi, 10).to(
                    model.Tmul.dtype) * k
            # the carry/borrow slices and the three constants must be sharp too,
            # otherwise the einsum averages one-hot rows back into mush
            for t in (model.Tmul[..., 10:], model.Tadd[..., 10:],
                      model.Tsub[..., 10:]):
                idx = torch.randint(0, t.shape[-1], t.shape[:-1], device=t.device)
                t.copy_(F.one_hot(idx, t.shape[-1]).to(t.dtype) * k)
            for t in (model.zero, model.carry0, model.borrow0):
                idx = torch.randint(0, t.shape[-1], (), device=t.device)
                t.copy_(F.one_hot(idx, t.shape[-1]).to(t.dtype) * k)
    model.gate_off = args.gate_off
    model.hard_gate = args.hard_gate
    if args.gate_bias is not None:
        with torch.no_grad():
            model.gate.bias.fill_(args.gate_bias)
    model.trunc = args.trunc

    if args.construct:
        model.construct()
        model.eval()

        @torch.no_grad()
        def ev(inp, tgt):
            lg = model(inp, ndig)
            return (lg.argmax(-1) == tgt).all(dim=1).float().mean().item()
        print(f"[{args.tag}] CONSTRUCTED N={args.modulus} S={args.slots} "
              f"R={args.reduce} train_exact={ev(xin, xt):.3f} "
              f"held_exact={ev(hin, ht):.3f}", flush=True)
        return 0

    # staged gradient release
    stages = {}
    for part in filter(None, args.stage.split(",")):
        name, at = part.split(":")
        stages[name] = int(at)
    mod_params = {"mul": ["Tmul"], "add": ["Tadd"], "sub": ["Tsub"],
                  "gate": ["gate"], "zero": ["zero", "carry0", "borrow0"]}

    def apply_stages(step):
        if not stages:
            return
        for name, pref in mod_params.items():
            on = step >= stages.get(name, 0)
            for n, p in model.named_parameters():
                if any(n.startswith(q) for q in pref):
                    p.requires_grad_(on)

    apply_stages(0)

    latent = None
    if args.tprop > 0:
        latent = LatentTrace(args.slots, model.W, len(model.op_schedule()),
                             args.tprop_hidden).to(device)
    params = list(model.parameters()) + (
        list(latent.parameters()) if latent is not None else [])
    if args.opt == "adamw":
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd,
                                betas=(0.9, 0.95))
    elif args.opt == "sgd":
        opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9,
                              weight_decay=args.wd)
    elif args.opt == "rmsprop":
        opt = torch.optim.RMSprop(params, lr=args.lr, weight_decay=args.wd)
    else:                                    # sign / lion
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd,
                                betas=(0.9, 0.99))

    # LAB-ONLY curriculum stages over (modulus, slots)
    tasks = []
    for part in filter(None, args.curr.split(",")):
        n, s, st = part.split(":")
        n, s, st = int(n), int(s), int(st)
        tasks.append((n, s, st) + make_task(n, s, min(args.train_x,
                     len([x for x in range(1, n) if math.gcd(x, n) == 1]) - 8),
                     args.split_seed, device)[:3])
    tasks.append((args.modulus, args.slots, args.steps,
                  (xin, xt), (hin, ht), ndig))

    deep = None
    if args.deep_sup > 0 or args.teacher_force > 0:
        deep = horner_targets(train_x, args.modulus, args.slots,
                              args.reduce, device)
        if args.teacher_force > 0:
            model.tf, model.tf_p = deep, args.teacher_force

    if args.flow:
        model.trace = True
        with torch.no_grad():
            model(xin, ndig)
        prof = [f"{k}:{v:.4f}" for k, v in model.flow]
        print(f"[{args.tag}] flow " + " ".join(prof), flush=True)
        model.trace = False
        return 0

    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] N={args.modulus} S={args.slots} K={model.K} "
          f"W={model.W} R={args.reduce} train={len(train_x)} "
          f"held={len(held_x)} params={n_par:,} opt={args.opt} lr={args.lr}",
          flush=True)

    @torch.no_grad()
    def evaluate(inp, tgt, nd):
        model.eval()
        save = (model.R, model.tau, model.noise, model.S, model.K, model.W)
        model.R, model.noise = args.reduce, 0.0
        model.tau = args.tau_final if args.tau_final is not None else args.tau
        model.S, model.K, model.W = args.slots, 2 * args.slots - 1, args.slots + 1
        lg = model(inp, nd)
        ok = (lg.argmax(-1) == tgt).all(dim=1).float().mean().item()
        ce = F.cross_entropy(lg.reshape(-1, 10), tgt.reshape(-1)).item()
        (model.R, model.tau, model.noise,
         model.S, model.K, model.W) = save
        model.train()
        return ok, ce

    xmag = torch.tensor([float(x) for x in train_x], device=device)
    t0 = time.time()
    step = 0
    best = 0.0
    for (mod, slt, st_end, (ain, at_), (bin_, bt), nd) in tasks:
        if slt != model.S:            # rebuild the graph shape, keep the tables
            model.S, model.K, model.W = slt, 2 * slt - 1, slt + 1
        while step < st_end:
            step += 1
            f = step / args.steps
            if args.tau_final is not None:
                model.tau = math.exp((1 - f) * math.log(args.tau)
                                     + f * math.log(args.tau_final))
            if args.noise > 0:
                nf = args.noise_final if args.noise_final is not None else 0.0
                model.noise = args.noise + (nf - args.noise) * f
            if args.hard_at is not None:
                model.hard = f >= args.hard_at
            if args.gate_off != 0.0:
                model.gate_off = args.gate_off * max(
                    0.0, 1.0 - f / max(args.gate_warm, 1e-9))
            if args.r_start is not None:
                fr = min(1.0, f / max(args.r_warm, 1e-9))
                model.R = int(round(args.r_start
                                    + (args.reduce - args.r_start) * fr))
            else:
                model.R = args.reduce
            if args.lr_final is not None:
                for gp in opt.param_groups:
                    gp["lr"] = math.exp((1 - f) * math.log(args.lr)
                                        + f * math.log(args.lr_final))
            apply_stages(step)

            if args.batch and args.batch < ain.shape[0]:
                idx = torch.randint(0, ain.shape[0], (args.batch,), device=device)
                bi, bt_ = ain[idx], at_[idx]
            else:
                bi, bt_ = ain, at_
            if args.tf_decay is not None and model.tf is not None:
                model.tf_p = args.teacher_force * max(0.0, 1.0 - f / args.tf_decay)
                if model.tf_p <= 0.0:
                    model.tf = None
            model.collect = deep is not None and args.deep_sup > 0
            logits = model(bi, nd)
            if args.xcurr > 0 and xmag is not None and bi.shape[0] == xmag.shape[0]:
                thr = 10.0 * (args.modulus / 10.0) ** min(1.0, f / args.xcurr)
                w = (xmag <= thr).float()
                if w.sum() < 4:
                    w = torch.ones_like(w)
                per = F.cross_entropy(logits.reshape(-1, 10), bt_.reshape(-1),
                                      reduction="none").view(bi.shape[0], -1)
                loss = (per.mean(1) * w).sum() / w.sum()
            else:
                loss = F.cross_entropy(logits.reshape(-1, 10), bt_.reshape(-1))
            main_ce = loss.item()
            if latent is not None:
                if args.main_warm > 0:
                    loss = loss * min(1.0, f / args.main_warm)
                loss = loss + args.tprop * tprop_loss(
                    model, latent, bi, nd, bt_, args.tprop_tau)
            if model.collect and (args.batch == 0 or args.batch >= ain.shape[0]):
                w = args.deep_sup
                if args.deep_sup_decay is not None:
                    w = args.deep_sup * max(0.0, 1.0 - f / args.deep_sup_decay)
                if w > 0:
                    dl = 0.0
                    for idx_k, rk in enumerate(model.inter):
                        lg = torch.log(rk + 1e-9)
                        dl = dl + F.cross_entropy(lg.reshape(-1, 10),
                                                  deep[:, idx_k].reshape(-1))
                    loss = loss + w * dl / len(model.inter)
            if args.sym > 0:
                loss = loss + args.sym * sym_loss(model)
            if args.inv > 0:
                loss = loss + args.inv * inv_loss(model)
            if args.ent > 0:
                loss = loss + args.ent * entropy_loss(model)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.opt == "sign":
                for p in params:
                    if p.grad is not None:
                        p.grad = p.grad.sign()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.clip)
            if args.diag and (step == 1 or step % args.log_every == 0):
                gn = {n: (0.0 if p.grad is None else p.grad.norm().item())
                      for n, p in model.named_parameters()}
                print(f"[{args.tag}] grad step={step} " +
                      " ".join(f"{k}={v:.2e}" for k, v in gn.items()), flush=True)
            opt.step()

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                model.collect = False
                tr, tr_ce = evaluate(xin, xt, ndig)
                he, he_ce = evaluate(hin, ht, ndig)
                best = max(best, tr)
                print(f"[{args.tag}] step={step:>6} R={model.R} "
                      f"tau={model.tau:.3f} loss={main_ce:.4f} "
                      f"train_exact={tr:.3f} held_exact={he:.3f} "
                      f"train_ce={tr_ce:.3f} held_ce={he_ce:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    model.collect = False
    tr, tr_ce = evaluate(xin, xt, ndig)
    he, he_ce = evaluate(hin, ht, ndig)
    best = max(best, tr)
    nd_dig = digits_le(args.modulus, args.slots + 1)
    ta = table_accuracy(model, nd_dig)
    ss = structure_scores(model, nd_dig)
    print(f"[{args.tag}] FINAL train_exact={tr:.3f} held_exact={he:.3f} "
          f"best_train={best:.3f} struct[" +
          " ".join(f"{k}={v}" for k, v in ss.items()) +
          f"] ({time.time()-t0:.0f}s)", flush=True)
    if args.dump_tables:
        with torch.no_grad():
            print(f"[{args.tag}] Tmul lo argmax (row=a, col=b), true = a*b%10")
            print(model.Tmul[..., :10].argmax(-1).cpu().numpy())
            print(f"[{args.tag}] Tadd digit argmax at carry=0, true = (u+v)%10")
            print(model.Tadd[:, :, 0, :10].argmax(-1).cpu().numpy())
            print(f"[{args.tag}] Tsub digit argmax at borrow=0, true = (u-v)%10")
            print(model.Tsub[:, :, 0, :10].argmax(-1).cpu().numpy())
            print(f"[{args.tag}] zero={model.zero.argmax().item()} "
                  f"carry0={model.carry0.argmax().item()} "
                  f"borrow0={model.borrow0.argmax().item()} "
                  f"gate.w={model.gate.weight.flatten().tolist()} "
                  f"gate.b={model.gate.bias.tolist()}")
    if args.jsonl:
        with open(args.jsonl, "a") as fh:
            fh.write(json.dumps({"tag": args.tag, "argv": sys.argv[1:],
                                 "train_exact": tr, "held_exact": he,
                                 "best_train": best, "train_ce": tr_ce,
                                 "held_ce": he_ce, "table_acc": ta,
                                 "struct": ss,
                                 "secs": round(time.time() - t0, 1)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
