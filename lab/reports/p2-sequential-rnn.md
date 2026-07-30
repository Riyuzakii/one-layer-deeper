# `plan2/sequential-rnn` — PLAN2 §3.6, the sequential non-linear RNN, aggressively fused

**Mandate.** Build PLAN2 §3.6 — "the highest-variance entry in the plan and the one most
likely to be under-explored by other competitors" — as a *candidate*, not as a
diagnostic. Own budget, fusion, hidden-state size and generalisation. Secondary,
timeboxed: PLAN2 §3.5, the Neural GPU / tied conv-GRU.

**Status:** _in progress — numbers below are filled in as runs land._

---

## 0. The recurrence axis, stated explicitly (BRIEF2 §2a)

The recurrence runs over the **token / digit-place axis of the prompt**, length 13
(e1–e5) / 15 (m1) / 21 (m4). It does **not** run over the task's composition depth
`T ≤ 64`. `T` enters as input digits, like `N` and `x`.

BRIEF2 §2b: T-fold composition is already solved (exactness composes 1.000 at every
rung; a learned controller reaches MAX_T=64) and the one open bottleneck is a single
squaring on unseen operands. So the recurrence is aimed at that. The decoder scans
prompt positions **right-to-left, anchored to each row's own last valid token**, so its
hidden state travels in the *carry* direction: least-significant answer digit first.
Targets are supervised at positions `input_len - target_len … input_len - 1`
(`data/squaring_mod.py:collate_squaring_mod`), so decoder step 0 lands exactly on the
least-significant answer digit. That is the classic propagate/generate shape for
multi-digit arithmetic, on the axis BRIEF2 §2c identifies as the open one.

Architecture (`submissions/p2-sequential-rnn/submission.py`):

```
tok_emb + pos_emb
  -> enc_fwd : nn.LSTM  left-to-right over the token axis
  -> enc_rev : nn.LSTM  right-to-left, started from each row's last valid token
  -> concat  -> optional learned L×L mixing in end-anchored (digit-place) coordinates
  -> dec     : nn.LSTM  the carry scan, LSB -> MSB
  -> LayerNorm -> Linear -> logits
```

`D_H` is the single dial the mandate cares about: it is simultaneously the width of the
carry channel and the memorisation capacity. Padding is handled by an index permutation
(`_end_anchored_reverse_index`) rather than `pack_padded_sequence`, so nothing leaves
the GPU and no host sync is introduced.

---

## 1. PLAN2's own falsifier, measured first — and it does not fire

PLAN2's falsifier for §3.6 is *"step time makes the achievable update count
non-competitive."* This box is `sm_107`, not the H100 the competition scores on, and
BRIEF.md §5 says absolute wall clock does not transfer. So every number below is quoted
as a **ratio against a calibration model whose H100 cost is known**: the
`exp_axis` D=128 / 8-loop recurrent transformer, measured at **38.6 ms/step at batch
512 → ~93,000 steps in a 3600 s Hard run** on the one hosted Hard run (BRIEF2 §4).

`lab/rnn_bench.py`, batch 512, forward + backward + grad-clip + AdamW step, bf16 amp,
30 timed steps after 10 warmup.

### L = 21 (m4 / Hard-like prompt length)

| variant | local ms | ratio vs REF | implied H100 ms | implied steps / 3600 s |
|---|---|---|---|---|
| REF `exp_axis` d128 ×8 | 4.43 | 1.000 | 38.60 | 93,163 |
| LSTM d16 naive loop | 16.70 | 3.766 | 145.4 | 24,735 |
| LSTM d16 pre-projected loop | 11.42 | 2.575 | 99.4 | 36,180 |
| **LSTM d16 `nn.LSTM` (cuDNN fused)** | **2.08** | **0.469** | **18.1** | **198,742** |
| LSTM d64 naive loop | 16.94 | 3.820 | 147.5 | 24,386 |
| LSTM d64 pre-projected loop | 13.76 | 3.102 | 119.7 | 30,032 |
| **LSTM d64 `nn.LSTM`** | **2.10** | **0.474** | **18.3** | **196,477** |
| LSTM d128 `nn.LSTM` | 2.29 | 0.517 | 20.0 | 180,090 |
| LSTM d64 fused, 2 tied loops | 3.11 | 0.702 | 27.1 | 132,651 |
| LSTM d64 fused, 4 tied loops | 5.19 | 1.169 | 45.1 | 79,667 |

### L = 13 (e1–e5 prompt length)

| variant | local ms | ratio vs REF | implied H100 ms | implied steps / 3600 s |
|---|---|---|---|---|
| REF `exp_axis` d128 ×8 | 6.07 | 1.000 | 38.60 | 93,163 |
| LSTM d64 naive loop | 11.07 | 1.823 | 70.4 | 51,103 |
| LSTM d64 `nn.LSTM` | 2.10 | 0.346 | 13.3 | 269,496 |
| LSTM d64 fused, 4 tied loops | 5.13 | 0.845 | 32.6 | 110,312 |

**Reading.**

1. **The falsifier does not fire in any configuration measured.** Even the *unfused*
   Python-loop LSTM affords ~24,000 H100 steps at Hard-like length — six times the
   ≳4,000 steps `DigitALU` needed, and above the 2,900/31,000 step ceilings that made
   Easy and Medium look tight in the previous session. Fused, the candidate affords
   ~196,000 steps, i.e. **2.1× the reference model's own budget**. The standard
   objection to non-linear RNNs — `O(T)` serial depth — is worth nothing here because
   `T` is the prompt, and the prompt is 13–21 tokens.
2. **Cost is flat in `D_H` from 16 to 128** (2.08 → 2.29 ms, +10% for 8× the width and
   64× the FLOPs). This is a direct confirmation of the previous session's finding that
   this model class is **kernel-launch bound, not FLOP bound**. Hidden-state width is
   therefore free on the wall clock, which means the width question below is purely a
   *generalisation* question, with no budget trade-off attached to it.
3. **Serial depth is what costs.** Going from 1 to 4 tied encode/decode passes costs
   2.5× (2.10 → 5.19 ms) while going from d=16 to d=128 costs 1.10×. Depth, not width,
   is the price.

### Timing of the actual submission file (`lab/time_submission.py`, L=21, batch 512)

Measured under GPU contention (the calibration model read 10.53 ms in this window
instead of 4.43 — which is exactly why the ratio, not the absolute, is the number):

| `D_H` | params | local ms | ratio vs REF | implied H100 ms | implied steps / 3600 s |
|---|---|---|---|---|---|
| 8 | 2,898 | 2.81 | 0.267 | 10.3 | 348,895 |
| 16 | 8,922 | 2.89 | 0.274 | 10.6 | 340,023 |
| 32 | 31,722 | 3.11 | 0.296 | 11.4 | 315,044 |
| 64 | 120,330 | 3.42 | 0.324 | 12.5 | 287,143 |
| 128 | 469,578 | 4.21 | 0.400 | 15.4 | 232,968 |

**The competition-budget answer: ~230,000–350,000 steps in a Hard run, against the
~93,000 the reference model gets.** The sequential RNN is not an expensive
architecture on this task; it is a cheap one.

---

## 2. Fusion: worth 8×, and it needs no Triton

`nn.LSTM` dispatches to cuDNN's fused multi-timestep kernel. Against the identical
math written as a Python loop:

| L | naive loop | pre-projected loop | `nn.LSTM` | fused speed-up vs naive |
|---|---|---|---|---|
| 21 | 16.94 ms | 13.76 ms | 2.10 ms | **8.1×** |
| 13 | 11.07 ms | 9.24 ms | 2.10 ms | **5.3×** |

Hoisting the input projection out of the loop — the cheap half of "fusing" — buys only
1.2–1.3×. The remaining 6× is the per-timestep pointwise kernel launches, which is
exactly the regime the mandate predicted fusing would pay in.

### Triton is importable on this box but cannot compile on it

BRIEF2 §5 records `triton` 3.7.1 as importable and concludes custom fused kernels are
on the table. Importable is not usable. `lab/triton_probe.py`, the smallest possible
elementwise kernel:

```
ptxas-blackwell fatal : Value 'sm_107a' is not defined for option 'gpu-name'
```

The same failure kills `torch.compile` (Inductor emits Triton), so **both routes to a
hand-fused recurrence are unavailable locally** — a hand-written Triton kernel could
not be validated here even if it were written. This is a correction to BRIEF2 §5 that
applies to every agent in the fleet, not just this branch.

It costs this branch nothing, because **cuDNN already provides the fusion**, is plain
`torch`, introduces no custom `autograd.Function`, and therefore raises none of the
"participant-controlled backward" questions of BRIEF.md §4.4 that a hand-written Triton
kernel would have raised. **Recommendation: do not spend fleet time on a Triton
recurrence kernel. The fused path is `nn.LSTM`, it is legal, and it is already 2× faster
than the reference model.**

---

## 3. Hidden-state width: train vs held-out

_pending_

---

## 4. Neural GPU (PLAN2 §3.5) — secondary

_pending_

---

## 5. Evaluator runs

_pending_

---

## 6. Verdict and recommendation

_pending_
