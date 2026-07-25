#!/usr/bin/env python
"""LAB ONLY -- bottleneck (A): recovering place-valued digit slots from a
left-aligned prompt with variable-length fields.

group-rotation §9 measured this at ~0.25 of exact accuracy: the identical
architecture scores 0.237-0.263 on clean one-hot digit slots and 0.000-0.026 on
the real prompt.  Absolute position cannot fix place value because a field's
length varies; distance from the *end of the prompt* fixes it only while the T
field is one digit, so it breaks on exactly the T=16/32/64 rungs.

The fix measured here: index every digit by its offset from the marker token
that TERMINATES its own field, not from the end of the prompt.

    prompt = [N] d(N) [X] d(x) [T] d(T)

    * d(N) is terminated by the [X] marker  -> place p of N sits 1+p before [X]
    * d(x) is terminated by the [T] marker  -> place p of x sits 1+p before [T]
    * d(T) is terminated by the padding     -> place p of T sits p before the end

x's slots are therefore indexed off the [T] MARKER POSITION, which does not move
when T gains a digit.  That is the whole difference from `posmode=rev`.

`MarkerPointer` implements this as a learned differentiable pointer, entirely
inside the autograd graph:

    anchor_a[i]  = softmax_i(<learned probe_a, embed(tok_i)>)      a in {N,X,T}
    anchor_end   = last valid position of the attention mask
    rel[a,i,o]   = anchor_a[i+o]                     (soft relative offset)
    attn[s,i]    = softmax_i( sum_{a,o} rel[a,i,o] R[s,a,o] )      R learned
    slot_s       = sum_i attn[s,i] * embed(tok_i)

No Python control flow reads input_ids; no field boundary is computed with an
integer index; R is learned from random init and decides which offset each slot
uses.

`--construct-parse` sets the marker probes and R to the intended values (LAB
DIAGNOSTIC ONLY) to verify the *mechanism* at multi-digit T, separately from
whether it is learnable.

Prompts are synthesised from the public generator spec.  Nothing under
data/generated/ is opened.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from torch import nn

DIGIT_OFFSET = 7
TOK_PAD, TOK_BOS, TOK_N, TOK_X, TOK_T = 0, 1, 2, 3, 4
VOCAB = 17
BIG = 20.0


def number_tokens(v: int) -> list[int]:
    return [DIGIT_OFFSET + int(c) for c in str(v)]


def build_prompt(modulus: int, x: int, t: int) -> list[int]:
    return ([TOK_N] + number_tokens(modulus) + [TOK_X] + number_tokens(x)
            + [TOK_T] + number_tokens(t))


class MarkerPointer(nn.Module):
    """Learned differentiable field-relative pointer.  n_anchor = 4 (N, X, T,
    end-of-sequence).  Slot s reads position `offset` before anchor `a`, with
    (a, offset) chosen by the learned table R."""

    def __init__(self, d_model: int, n_slot: int, o_lo: int = -2, o_hi: int = 9):
        super().__init__()
        self.o_lo, self.o_hi = o_lo, o_hi
        self.n_off = o_hi - o_lo + 1
        self.probe = nn.Linear(d_model, 3, bias=False)      # N, X, T markers
        self.R = nn.Parameter(torch.randn(n_slot, 4, self.n_off) * 0.5)
        # A field has TWO boundaries.  The offset table alone indexes off the
        # terminating marker, so a slot above the field's actual width runs off
        # the front into the previous field (x has 1-3 digits but 3 slots).  G
        # scores the cumulative anchor mass -- "have we passed marker a yet" --
        # which is the differentiable form of the opening boundary.
        self.G = nn.Parameter(torch.randn(n_slot, 4) * 0.5)
        self.n_slot = n_slot

    @torch.no_grad()
    def construct(self, emb_weight, slot_spec):
        """slot_spec: list of (anchor, offset, open_anchor) -- DIAGNOSTIC ONLY."""
        w = torch.zeros_like(self.probe.weight)
        # a linear probe that fires on the marker token's embedding
        for k, tok in enumerate((TOK_N, TOK_X, TOK_T)):
            w[k] = emb_weight[tok] / (emb_weight[tok].norm() ** 2 + 1e-9) * BIG
        self.probe.weight.copy_(w)
        r = torch.zeros_like(self.R)
        g = torch.zeros_like(self.G)
        for s, (a, o, opn) in enumerate(slot_spec):
            r[s, a, o - self.o_lo] = BIG
            # the opening marker is the fallback: it wins exactly when the
            # intended position lies outside the field (a leading-zero sentinel)
            r[s, opn, 0 - self.o_lo] = 0.5 * BIG
            g[s, opn] = BIG
        self.R.copy_(r)
        self.G.copy_(g)

    def anchors(self, emb, mask):
        b, L, _ = emb.shape
        neg = torch.finfo(emb.dtype).min
        sc = self.probe(emb).transpose(1, 2)                 # (B,3,L)
        sc = sc.masked_fill(~mask[:, None, :], neg)
        a = F.softmax(sc, dim=-1)
        idx = mask.long().sum(-1) - 1                        # last valid position
        a_end = F.one_hot(idx, L).to(a.dtype)[:, None]
        return torch.cat([a, a_end], dim=1)                  # (B,4,L)

    def forward(self, emb, mask):
        b, L, _ = emb.shape
        a = self.anchors(emb, mask)
        rels = []
        for o in range(self.o_lo, self.o_hi + 1):
            if o >= 0:
                rels.append(F.pad(a[:, :, o:], (0, o)))
            else:
                rels.append(F.pad(a[:, :, :L + o], (-o, 0)))
        rel = torch.stack(rels, dim=-1)                      # (B,4,L,O)
        cum = a.cumsum(dim=-1)                               # (B,4,L)
        logit = (torch.einsum("balo,sao->bsl", rel, self.R)
                 + torch.einsum("bal,sa->bsl", cum, self.G))
        neg = torch.finfo(emb.dtype).min
        logit = logit.masked_fill(~mask[:, None, :], neg)
        attn = F.softmax(logit, dim=-1)                      # (B,n_slot,L)
        return torch.einsum("bsl,bld->bsd", attn, emb), attn


class AbsPointer(nn.Module):
    """Control: slots indexed by ABSOLUTE position (group-rotation posmode=abs)."""

    def __init__(self, d_model, n_slot, max_len):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_slot, max_len) * 0.5)

    def forward(self, emb, mask):
        neg = torch.finfo(emb.dtype).min
        L = emb.shape[1]
        logit = self.q[None, :, :L].expand(emb.shape[0], -1, -1)
        logit = logit.masked_fill(~mask[:, None, :], neg)
        attn = F.softmax(logit, dim=-1)
        return torch.einsum("bsl,bld->bsd", attn, emb), attn


class RevPointer(nn.Module):
    """Control: slots indexed by DISTANCE FROM THE END of the prompt
    (group-rotation posmode=rev).  This is the scheme that breaks when T gains
    a digit."""

    def __init__(self, d_model, n_slot, max_len):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_slot, max_len) * 0.5)

    def forward(self, emb, mask):
        b, L, _ = emb.shape
        neg = torch.finfo(emb.dtype).min
        last = mask.long().sum(-1) - 1
        pos = torch.arange(L, device=emb.device)[None, :]
        rev = (last[:, None] - pos).clamp(0, self.q.shape[1] - 1)  # (B,L)
        logit = self.q[None].expand(b, -1, -1).gather(
            2, rev[:, None, :].expand(-1, self.q.shape[0], -1))
        logit = logit.masked_fill(~mask[:, None, :], neg)
        attn = F.softmax(logit, dim=-1)
        return torch.einsum("bsl,bld->bsd", attn, emb), attn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", default="marker", choices=("marker", "abs", "rev"))
    ap.add_argument("--modulus", type=int, default=323)
    ap.add_argument("--slots", type=int, default=3, help="digit slots for x")
    ap.add_argument("--construct-parse", action="store_true")
    ap.add_argument("--eval-time-steps", type=int, nargs="+",
                    default=[1, 2, 3, 4, 8, 16, 32, 64])
    ap.add_argument("--x-samples", type=int, default=250)
    ap.add_argument("--sampled-n", action="store_true",
                    help="e5 regime: 10/11-bit sampled moduli, len(N) varies too")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    S = args.slots
    modulus = args.modulus
    if args.sampled_n:
        # the e5 regime: 10/11-bit sampled moduli, so len(N) varies too
        import random as _r
        from probe_alu_oodn import sample_semiprime
        rng = _r.Random(args.seed)
        pool = [sample_semiprime(b, rng) for b in (10, 11) for _ in range(6)]
        mods, xs = [], []
        for _ in range(args.x_samples):
            m = rng.choice(pool)
            mods.append(m)
            xs.append(rng.randrange(1, m))
        S = max(len(str(m)) for m in pool)
    else:
        units = [x for x in range(1, modulus) if math.gcd(x, modulus) == 1]
        xs = units[: args.x_samples]
        mods = [modulus] * len(xs)

    # one-hot token embedding: the parse question is about POSITION, not about
    # what a token means, so give the pointer the least helpful embedding there is
    d_model = VOCAB
    emb = torch.eye(VOCAB, device=device)

    n_slot = 3 * S  # x slots, N slots, T slots (T reuses S for headroom)
    if args.front == "marker":
        ptr = MarkerPointer(d_model, n_slot).to(device)
        if args.construct_parse:
            # (terminating anchor, offset, opening anchor);
            # anchors: 0=[N] 1=[X] 2=[T] 3=end-of-sequence
            spec = ([(2, 1 + p, 1) for p in range(S)]   # x: ends at [T], opens at [X]
                    + [(1, 1 + p, 0) for p in range(S)]  # N: ends at [X], opens at [N]
                    + [(3, p, 2) for p in range(S)])     # T: ends at EOS, opens at [T]
            ptr.construct(emb, spec)
    elif args.front == "abs":
        ptr = AbsPointer(d_model, n_slot, 64).to(device)
    else:
        ptr = RevPointer(d_model, n_slot, 64).to(device)
        if args.construct_parse:
            with torch.no_grad():
                q = torch.full_like(ptr.q, -BIG)
                # the best a distance-from-the-end scheme can do: it must commit
                # to ONE assumed width for the T field.  Training sees T=1,2,3,
                # so it commits to one digit.
                for p in range(S):
                    q[p, 1 + 1 + p] = BIG          # x slot p, assuming len(T)=1
                    q[S + p, 0] = BIG              # N slots: unreachable, park them
                    q[2 * S + p, p] = BIG          # T slot p
                ptr.q.copy_(q)

    print(f"front={args.front} construct={args.construct_parse} modulus={modulus} "
          f"S={S} params={sum(p.numel() for p in ptr.parameters()):,}\n")
    print(f"{'T':>4} {'len(T)':>6} {'L':>3} "
          f"{'x-parse':>8} {'N-parse':>8} {'T-parse':>8}")
    print("-" * 44)

    for t in args.eval_time_steps:
        rows = [build_prompt(m, x, t) for m, x in zip(mods, xs)]
        L = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), L, dtype=torch.long)
        mask = torch.zeros(len(rows), L, dtype=torch.bool)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = True
        ids, mask = ids.to(device), mask.to(device)
        with torch.no_grad():
            vec, attn = ptr(emb[ids], mask)
        # a slot is parsed correctly if its attention lands on the token that
        # carries the intended digit
        got = attn.argmax(-1)                                   # (B, n_slot)
        okx = okn = okt = 0
        for i, x in enumerate(xs):
            p, m = rows[i], mods[i]
            posT = len(p) - len(number_tokens(t)) - 1
            posX = 1 + len(number_tokens(m))
            # a slot ABOVE the field width must land on the field's opening
            # marker, which is a distinct token and so a usable leading zero
            xd = [posT - 1 - k if k < len(number_tokens(x)) else posX
                  for k in range(S)]
            nd = [posX - 1 - k if k < len(number_tokens(m)) else 0
                  for k in range(S)]
            td = [len(p) - 1 - k if k < len(number_tokens(t)) else posT
                  for k in range(S)]
            okx += all(got[i, k].item() == xd[k] for k in range(S))
            okn += all(got[i, S + k].item() == nd[k] for k in range(S))
            okt += all(got[i, 2 * S + k].item() == td[k] for k in range(S))
        n = len(xs)
        print(f"{t:>4} {len(str(t)):>6} {L:>3} "
              f"{okx/n:>8.3f} {okn/n:>8.3f} {okt/n:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
