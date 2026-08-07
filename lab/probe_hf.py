#!/usr/bin/env python
"""LAB ONLY -- `DigitALU` at Hard-FAITHFUL scale, with every earned correction
applied at once, on a MODULUS-SPLIT dataset.

WHY THIS EXISTS.  Every `DigitALU` measurement in this project was taken on a
*single* modulus (or, in `alu-depth`, on a handful of sampled ones at eval).
The one hosted Hard run showed that h1 uses `split_group=modulus` with
`separate_ood_splits`, i.e. **train and test draw from disjoint modulus pools**.
That changes the *training distribution*, not just the test one: instead of one
N with many x, the model sees thousands of moduli.  For a modulus-independent
digit transducer that is strictly more constraint per parameter, and it has
never been screened.  This probe reproduces that structure exactly, offline.

WHAT IS REPRODUCED FROM `lab/gen_hard_faithful.sh` (generator source only --
nothing under data/generated/ is opened, and this file never imports a record):

  * ID modulus sizes [16,18,20]; OOD-N sizes [17,19,21].
  * `_enumerate_sampled_factor_pairs` for the ID sizes, partitioned 90/10 by
    COUNT into disjoint train / test modulus pools (the generator's
    `_partition_factor_pairs`).
  * moduli drawn from a pool with weight (p-1)(q-1), x a unit mod N
    (the generator's `_generate_records_from_factor_pool` / `_sample_unit`).
  * OOD-N moduli rejection-sampled at 17/19/21 bits.

WHAT IS MEASURED.  One squaring step, `x^2 mod N`, on 7 digit slots -- the one
open bottleneck (T-fold composition, parsing, the readout and the depth
controller are all solved elsewhere and are deliberately absent here so that
nothing else can absorb the result).

  train_exact_hard          headline (every inter-step state snapped to argmax)
  held_seen_N               unseen x, TRAINING-pool modulus
  held_unseen_N             unseen x, HELD-OUT modulus   <- hf1's `test`
  held_ood_n                unseen x, 17/19/21-bit modulus <- hf1's `ood_n_*`
  local_ce                  per-op CE against the construction (DIAGNOSTIC)
  diversity                 distinct answers / n, against a MEASURED reference
  mixture vs argmax + wmax  the replica selector, reported separately

CORRECTIONS APPLIED (each individually measured elsewhere):
  tree:quotient graph (`alu-depth`), UNTIED (`alu-credit`: ties help repair and
  hurt learning from random init), replica population with a differentiable
  selector (`alu-population`), a calibrated step budget, and `--lr 0` as a
  mandatory control (`alu-optimizer`).

COMPLIANCE.  `--construct` and `--tf` replay / install the exact solution and
are LAB DIAGNOSTICS (rules 2 and 7); every row they produce is labelled
DIAGNOSTIC and can never appear in a submission.  `--tf 0` with no
`--construct` is the LEGAL objective.  Self-generated moduli and operands only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from probe_pop import PopALU, eval_pop, struct_scores  # noqa: E402
from probe_alu_depth import digits_le  # noqa: E402
from data.squaring_mod import (  # noqa: E402  (generator SOURCE, not data)
    _enumerate_sampled_factor_pairs,
    _sample_rsa_factors,
)


# --------------------------------------------------------------- the dataset
class ModulusSplit:
    """hf1's modulus-split structure, rebuilt from the generator's own
    enumeration.  `train` and `test` modulus pools are disjoint by
    construction, exactly as `_partition_factor_pairs` makes them."""

    def __init__(self, id_bits=(16, 18, 20), oodn_bits=(17, 19, 21),
                 train_fraction=0.9, seed=45):
        self.rng = random.Random(seed)
        self.pools = {"train": [], "test": []}
        self.by_bits = {}
        for bits in id_bits:
            pairs = _enumerate_sampled_factor_pairs(bits)
            if pairs is None:
                raise ValueError(f"modulus_bits={bits} is not enumerable "
                                 "(the split_group=modulus 20-bit cap)")
            pairs = sorted(pairs)
            self.rng.shuffle(pairs)
            k = int(len(pairs) * train_fraction)
            # the generator assigns by COUNT, train first
            self.by_bits[bits] = {"train": pairs[:k], "test": pairs[k:]}
            self.pools["train"] += pairs[:k]
            self.pools["test"] += pairs[k:]
        self.oodn = []
        for bits in oodn_bits:
            seen = set()
            while len(seen) < 64:
                p, q = _sample_rsa_factors(modulus_bits=bits, rng=self.rng)
                seen.add((p, q))
            self.oodn += sorted(seen)
        self.weights = {k: [(p - 1) * (q - 1) for p, q in v]
                        for k, v in self.pools.items()}
        self.weights["oodn"] = [(p - 1) * (q - 1) for p, q in self.oodn]
        self.pools["oodn"] = self.oodn

    def summary(self):
        return {b: {k: len(v) for k, v in d.items()}
                for b, d in self.by_bits.items()} | \
            {"oodn": len(self.oodn)}

    def draw(self, pool, n_mod, per_mod, used=None, rng=None):
        """n_mod moduli from `pool`, each with `per_mod` fresh units.

        Returns (moduli, xs) with xs shaped (n_mod, per_mod).  `used` is a set
        of (N, x) pairs to avoid -- that is how the held-out sets are made
        disjoint from the training set at the *prompt* level, which is what the
        generator does with `seen_prompts`."""
        rng = rng or self.rng
        pairs, w = self.pools[pool], self.weights[pool]
        picked = rng.choices(range(len(pairs)), weights=w, k=n_mod)
        mods, xs = [], []
        for i in picked:
            p, q = pairs[i]
            N = p * q
            row = []
            while len(row) < per_mod:
                x = rng.randrange(1, N)
                if math.gcd(x, N) != 1:
                    continue
                if used is not None and (N, x) in used:
                    continue
                if used is not None:
                    used.add((N, x))
                row.append(x)
            mods.append(N)
            xs.append(row)
        return mods, xs


def encode(mods, xs, S, W, device):
    """(G moduli, G x per_mod operands) -> the tensors the ALU consumes."""
    G, n = len(mods), len(xs[0])
    sin = torch.zeros(G, n, S, 10)
    tgt = torch.zeros(G, n, S, dtype=torch.long)
    nd = torch.zeros(G, W, 10)
    for g, N in enumerate(mods):
        for i, d in enumerate(digits_le(N, W)):
            nd[g, i, d] = 1.0
        for j, x in enumerate(xs[g]):
            for i, d in enumerate(digits_le(x, S)):
                sin[g, j, i, d] = 1.0
            for i, d in enumerate(digits_le((x * x) % N, S)):
                tgt[g, j, i] = d
    return (sin.reshape(G * n, S, 10).to(device),
            tgt.reshape(G * n, S).to(device),
            nd.to(device))


# ------------------------------------------------------------------ the model
class PoolALU(PopALU):
    """`PopALU` (tree:quotient, replica dimension) with PER-EXAMPLE moduli.

    Only two things change and both are bookkeeping: the multiples prefix is
    computed for each distinct modulus in the batch instead of once, and the
    quotient reduction indexes its candidate multiples by the example's modulus
    group.  The graph, the alphabet, the depth and the parameter count are
    identical -- `mults` is data, not a parameter, so nothing about the
    hypothesis class moves."""

    def multiples_g(self, ndig):
        """ndig: (G, W, 10) -> (P, G, M, W, 10), built from digits(N) by the
        SAME learned Tadd (zero new parameters, as in PopALU)."""
        P, W = self.P, self.W
        G = ndig.shape[0]
        z = self._sm(self.zero)
        have = {0: z[:, None, None, :].expand(P, G, W, 10),
                1: ndig[None].expand(P, G, W, 10)}
        target = set(self.needed)
        while not target <= set(have):
            newly = []
            for t in sorted(target - set(have)):
                cand = [a for a in have if a <= t - a and (t - a) in have]
                if cand:
                    newly.append((t, max(cand), t - max(cand)))
            if not newly:
                mx = max(have)
                newly = [(2 * mx, mx, mx)]
            A = torch.cat([have[a] for _, a, _ in newly], 1)
            B = torch.cat([have[b] for _, _, b in newly], 1)
            out = self.add_scan(A, B)
            for i, (t, _, _) in enumerate(newly):
                have[t] = out[:, i * G:(i + 1) * G]
        return torch.stack([have[m] for m in self.needed], 2)   # (P,G,M,W,10)

    def quot_reduce_g(self, r, G):
        """r: (P, G*n, W, 10).  Uses the cached per-group `_Tmg`."""
        P, W, Cb = self.P, self.W, self.Cb
        n = r.shape[1] // G
        M = self._Tmg.shape[2]
        rg = r.view(P, G, n, W, 10)
        cs = self.copy_scale.view(P, 1, 1, 1, 1)
        c = self._sm(self.borrow0)[:, None, None, None].expand(P, G, n, M, Cb)
        outs = []
        for m in range(W):
            rw = rg[:, :, :, m]                              # (P,G,n,10)
            o = torch.einsum("pgnu,pgnmc,pgmuco->pgnmo", rw, c,
                             self._Tmg[:, :, :, m])
            outs.append(self._sm(o[..., :10] + cs * rw[:, :, :, None]))
            c = self._sm(o[..., 10:] + cs * c)
        t = self._tap(torch.stack(outs, 4).reshape(P, G * n, M, W, 10))
        c = c.reshape(P, G * n, M, Cb)
        pair = torch.cat([c[:, :, :-1], c[:, :, 1:]], dim=-1)
        w = self._sm(torch.einsum("pbmk,pk->pbm", pair, self.sel_w)
                     + self.sel_b[:, None, None])
        self.last_q = w
        return self._tap(torch.einsum("pbm,pbmwo->pbwo", w, t[:, :, :-1]))

    def forward(self, s, ndig):
        """s: (G*n, S, 10) shared across replicas.  ndig: (G, W, 10).
        returns log-probs (P, G*n, S, 10)."""
        P, S, W = self.P, self.S, self.W
        b = s.shape[0]
        G = ndig.shape[0]
        self.tpos = 0
        F2 = 2 * S
        z = self._sm(self.zero)[:, None].expand(P, b, 10)
        mults = self.multiples_g(ndig)                       # (P,G,M,W,10)
        self._Tmg = torch.einsum("pgmwv,puvco->pgmwuco", mults, self.Tsub_eff)
        se = s[None].expand(P, b, S, 10)
        Tm = self.Tmul_eff
        prod = {}
        for i in range(S):
            for j in range(S):
                o = torch.einsum("pbu,pbv,puvo->pbo", se[:, :, i],
                                 se[:, :, j], Tm)
                prod[(i, j)] = (self._sm(o[..., :10]), self._sm(o[..., 10:]))
        buckets = {}
        for (i, j), lh in prod.items():
            buckets.setdefault(i + j, []).append(lh)
        leaves = []
        for par in (0, 1):
            offs = [k for k in sorted(buckets) if k % 2 == par]
            if not offs:
                continue
            for t in range(max(len(buckets[k]) for k in offs)):
                cols = [z] * F2
                for k in offs:
                    if t < len(buckets[k]):
                        cols[k], cols[k + 1] = buckets[k][t]
                leaves.append(torch.stack(cols, 2))
        Pr = self.tree_sum(leaves)
        r = z[:, :, None].expand(P, b, W, 10)
        for t in range(F2 - 1, -1, -1):
            r = torch.cat([Pr[:, :, t:t + 1], r[:, :, :W - 1]], dim=2)
            if t <= S:
                r = self.quot_reduce_g(r, G)
        return torch.log(r[:, :, :S] + 1e-9)


def graph_depth(S, W):
    """Chained soft (softmax) steps on the x -> y critical path, same
    accounting as probe_alu_depth.DigitALU.depth()."""
    counts = [min(k + 1, S, 2 * S - 1 - k) for k in range(2 * S - 1)]
    n_leaf = max([counts[k] for k in range(0, 2 * S - 1, 2)] or [0]) + \
        max([counts[k] for k in range(1, 2 * S - 1, 2)] or [0])
    adds = math.ceil(math.log2(n_leaf)) * (2 * S) if n_leaf > 1 else 0
    red = (S + 1) * (W + 1)
    return 1 + adds + red


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, sets, discrete=True, chunk_g=16):
    """Per-split exact accuracy for the mixture, the argmax replica and the
    best replica.  Splits are stored as a list of (sin, tgt, nd, G) chunks."""
    model.eval()
    was, model.hard = model.hard, discrete
    P = model.P
    out = {}
    star = int(model.alpha.argmax().item())
    for name, chunks in sets.items():
        ok = torch.zeros(P, device=model.alpha.device)
        ok_mix = 0.0
        n = 0
        answers = set()
        for sin, tgt, nd in chunks:
            lg = model(sin, nd)
            ok += (lg.argmax(-1) == tgt[None]).all(-1).float().sum(1)
            mp = model.mix_probs(lg)
            pred = mp.argmax(-1)
            ok_mix += (pred == tgt).all(-1).float().sum().item()
            for row in pred.cpu().tolist():
                answers.add(tuple(row))
            n += tgt.shape[0]
        acc = ok / n
        out[name] = {"mix": ok_mix / n, "argmax": float(acc[star]),
                     "best": float(acc.max()), "diversity": len(answers) / n}
    model.hard = was
    model.train()
    return out


def fmt(res, keys):
    """argmax-replica / mixture / best-replica -- the row, never one cell.

    `argmax` is what a submission would commit to (the replica the learned
    selector picks); `mix` is the blend, which a mixture can win without any
    component being right; `best` is the population's oracle upper bound and is
    DIAGNOSTIC by itself."""
    return " ".join(
        f"{k}={res[k]['argmax']:.4f}/{res[k]['mix']:.4f}/{res[k]['best']:.4f}"
        for k in keys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=7,
                    help="digit slots for x; 7 covers a 21-bit modulus")
    ap.add_argument("--pop", type=int, default=8, help="replicas")
    ap.add_argument("--max-quot", type=int, default=10)
    ap.add_argument("--train-rows", type=int, default=81000,
                    help="fixed training prompts; hf1 has 81,000 per T setting")
    ap.add_argument("--n-train-mod", type=int, default=0,
                    help="distinct training moduli (0 = train_rows/per_mod, "
                         "hf1's own density).  Set small to reproduce the "
                         "PROMPT-split condition -- many x on few moduli -- at "
                         "the same row count, which is the controlled "
                         "modulus-split-vs-prompt-split comparison.")
    ap.add_argument("--held-rows", type=int, default=4096)
    ap.add_argument("--group", type=int, default=32,
                    help="distinct moduli per batch")
    ap.add_argument("--per-mod", type=int, default=16,
                    help="operands per modulus per batch (batch = G*per_mod)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--sel-lr", type=float, default=1e-1)
    ap.add_argument("--init-scale", type=float, default=0.5)
    ap.add_argument("--sel-hard", action="store_true")
    ap.add_argument("--tf", type=float, default=0.0,
                    help="DIAGNOSTIC teacher forcing probability (rules 2, 7)")
    ap.add_argument("--construct", action="store_true",
                    help="DIAGNOSTIC: install the exact solution and report "
                         "the ceiling")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=45)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--jsonl", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    S, W = args.slots, args.slots + 1

    split = ModulusSplit(seed=args.split_seed)
    depth = graph_depth(S, W)
    print(f"[{args.tag}] hf1-faithful modulus split {split.summary()} "
          f"S={S} W={W} tree:quotient depth={depth} P={args.pop}", flush=True)

    # ---- fixed training prompts, drawn the way the generator draws them ----
    used: set[tuple[int, int]] = set()
    rng = random.Random(args.split_seed + 1)
    n_tr_mod = args.n_train_mod or max(1, args.train_rows // args.per_mod)
    per_row = max(args.per_mod, args.train_rows // n_tr_mod)
    tr_mods, tr_xs = split.draw("train", n_tr_mod, per_row, used, rng)
    print(f"[{args.tag}] train prompts={n_tr_mod * per_row:,} over "
          f"{len(set(tr_mods)):,} distinct training moduli "
          f"({per_row} operands each)", flush=True)

    def build(pool, rows, per=16):
        g = max(1, rows // per)
        mods, xs = split.draw(pool, g, per, used, rng)
        chunks = []
        for i in range(0, g, 16):
            chunks.append(encode(mods[i:i + 16], xs[i:i + 16], S, W, dev))
        return chunks

    sets = {
        "held_seen_N": build("train", args.held_rows),
        "held_unseen_N": build("test", args.held_rows),
        "held_ood_n": build("oodn", args.held_rows // 2),
    }
    # a fixed slice of the training prompts, for train_exact / train_exact_hard
    tr_chunks = []
    n_tr_chunk = min(len(tr_mods), max(16, 2048 // args.per_mod))
    for i in range(0, n_tr_chunk, 16):
        tr_chunks.append(encode(tr_mods[i:i + 16],
                                [r[:args.per_mod] for r in tr_xs[i:i + 16]],
                                S, W, dev))
    sets["train"] = tr_chunks

    mk = lambda: PoolALU(args.pop, S, 2, 2, args.max_quot, 1.0, False,
                         False, False, args.init_scale, 0.0).to(dev)
    model = mk()
    ref = mk()
    ref.construct()
    for p in ref.parameters():
        p.requires_grad_(False)
    n_par = sum(p.numel() for p in model.parameters()) // args.pop
    print(f"[{args.tag}] params/replica={n_par:,} "
          f"params/train-row={n_par / (n_tr_mod * args.per_mod):.4f}",
          flush=True)

    order = ["train", "held_seen_N", "held_unseen_N", "held_ood_n"]

    if args.construct:
        model.construct()
        res = evaluate(model, sets)
        print(f"[{args.tag}] DIAGNOSTIC CONSTRUCTED (hard states) "
              f"{fmt(res, order)}", flush=True)
        print(f"[{args.tag}] DIAGNOSTIC diversity "
              + " ".join(f"{k}={res[k]['diversity']:.4f}" for k in order),
              flush=True)
        soft = evaluate(model, sets, discrete=False)
        print(f"[{args.tag}] DIAGNOSTIC CONSTRUCTED (soft states) "
              f"{fmt(soft, order)}", flush=True)
        return 0

    body = [p for p in model.parameters() if p is not model.alpha]
    opt = torch.optim.AdamW(
        [{"params": body, "weight_decay": 0.0},
         {"params": [model.alpha], "weight_decay": 0.0, "lr": args.sel_lr}],
        lr=args.lr, betas=(0.9, 0.95))
    model.sel_hard = args.sel_hard

    t0 = time.time()
    G, per = args.group, args.per_mod
    step_rng = random.Random(args.seed + 7)

    def batch():
        gs = step_rng.sample(range(len(tr_mods)), min(G, len(tr_mods)))
        mods = [tr_mods[g] for g in gs]
        xs = [step_rng.sample(tr_xs[g], per) if len(tr_xs[g]) > per
              else tr_xs[g] for g in gs]
        return encode(mods, xs, S, W, dev)

    hist = []
    for step in range(1, args.steps + 1):
        sin, tgt, nd = batch()
        model.tf_loss = torch.zeros(model.P, device=dev)
        model.tf_n = 0
        if args.tf > 0:
            with torch.no_grad():
                ref.mode, ref.tape = "record", []
                ref(sin, nd)
                ref.mode = None
            model.mode, model.tape = "force", ref.tape
            model.tf_p = args.tf
        logits = model(sin, nd)
        model.mode = None
        mix = model.mix_probs(logits).clamp_min(1e-9).log()
        loss = F.cross_entropy(mix.reshape(-1, 10), tgt.reshape(-1))
        if args.tf > 0:
            loss = loss + model.tf_loss.mean() / max(model.tf_n, 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), args.clip)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            res = evaluate(model, sets)
            soft = evaluate(model, sets, discrete=False)
            w = F.softmax(model.alpha / model.sel_tau, 0)
            print(f"[{args.tag}] step={step:>6} loss={loss.item():.4f} "
                  f"HARD {fmt(res, order)} "
                  f"soft_train={soft['train']['argmax']:.4f} "
                  f"wmax={w.max().item():.3f} "
                  f"div_unseenN={res['held_unseen_N']['diversity']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            hist.append({"step": step, "loss": round(loss.item(), 5),
                         **{f"{k}_hard": round(res[k]["argmax"], 5)
                            for k in order},
                         "train_soft": round(soft["train"]["argmax"], 5)})

    # ---- local_ce: DIAGNOSTIC per-op CE against the construction ----
    with torch.no_grad():
        sin, tgt, nd = batch()
        ref.mode, ref.tape = "record", []
        ref(sin, nd)
        ref.mode = None
        model.mode, model.tape = "force", ref.tape
        model.tf_loss, model.tf_n, model.tf_p = \
            torch.zeros(model.P, device=dev), 0, 1.0
        model(sin, nd)
        model.mode = None
        lce = (model.tf_loss / max(model.tf_n, 1)).cpu()
    res = evaluate(model, sets)
    soft = evaluate(model, sets, discrete=False)
    star = int(model.alpha.argmax().item())
    w = F.softmax(model.alpha / model.sel_tau, 0)
    sc = struct_scores(model, star, digits_le(tr_mods[0], W))
    label = "DIAGNOSTIC" if (args.tf > 0 or args.construct) else "LEGAL"
    print(f"[{args.tag}] FINAL [{label}] HARD {fmt(res, order)} "
          f"soft_train={soft['train']['argmax']:.4f} "
          f"local_ce(min/med)={lce.min():.4f}/{lce.median():.4f} "
          f"wmax={w.max().item():.3f} struct={sc} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if args.jsonl:
        with open(args.jsonl, "a") as fh:
            fh.write(json.dumps({
                "tag": args.tag, "label": label, "argv": sys.argv[1:],
                "depth": depth, "params": n_par,
                **{f"{k}_hard_argmax": round(res[k]["argmax"], 5)
                   for k in order},
                **{f"{k}_hard_mix": round(res[k]["mix"], 5) for k in order},
                **{f"{k}_hard_best": round(res[k]["best"], 5) for k in order},
                **{f"{k}_div": round(res[k]["diversity"], 5) for k in order},
                "train_soft": round(soft["train"]["argmax"], 5),
                "local_ce_min": round(float(lce.min()), 5),
                "local_ce_med": round(float(lce.median()), 5),
                "wmax": round(float(w.max()), 4), **sc,
                "curve": hist, "secs": round(time.time() - t0, 1),
            }) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
