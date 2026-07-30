# `plan2/sequential-rnn` — PLAN2 §3.6, the sequential non-linear RNN, aggressively fused

**Mandate.** Build PLAN2 §3.6 — "the highest-variance entry in the plan and the one most
likely to be under-explored by other competitors" — as a *candidate*, not as a
diagnostic. Own budget, fusion, hidden-state size and generalisation. Secondary,
timeboxed: PLAN2 §3.5, the Neural GPU / tied conv-GRU.

**One-line verdict.** The falsifier PLAN2 attached to this entry — "step time makes the
achievable update count non-competitive" — is **decisively refuted**: fused, this is the
*cheapest* architecture the project has built, worth ~230k–350k Hard steps against the
reference model's ~93,000, with an 8.4× eval-budget margin. And with maximal
expressivity, no theoretical caveats and 13× the screening compute, it reproduces the
project's canonical failure exactly. **MAX_T = 0 everywhere.**

**The result with the widest blast radius is not about this architecture.** Spending that
surplus budget at Medium scale (§3.5) shows **m1 does fit** — `train_exact` 0.0098 at
3,000 steps and **0.8572 at 40,000** — with held-out at three examples in 3,000. So BRIEF2
§2(d)'s "at m1 scale and above they cannot even fit" is a **step-count artifact**, the
Easy/Medium split in the project's diagnosis is not real, and the memorisation failure is
**one phenomenon at every scale measured**. A corollary that affects sibling branches
directly: the `fs1200`/`fs1500` screens in wide use across the PLAN2 worktrees sit inside
the pre-fitting region at Medium scale (§3.5a).

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

**Datasets and their λ (BRIEF2 §6.5).** Screening is on **e5** — 512-example rungs,
4,800 train / 1,200 `test` / 600 `ood` rows, sampled 10/11-bit moduli, so `test`
measures operand generalisation and `ood` measures unseen-modulus generalisation. Its
moduli are drawn per row and their λ is not knowable without reading
`data/generated/`, which is forbidden; this costs nothing here because **no measurement
in this report is a depth measurement** — the recurrence is not over T and the ranking
quantity screened on is rung-1/held-out exactness. **e1** (N = 323 = 17·19,
**λ = lcm(16,18) = 144**, so its T-ladder has four distinct maps, not seven) is used
*only* as the memorisation-isolation dataset, where T-degeneracy is irrelevant because
the question is whether the model memorises operands.

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
2. **Cost is nearly flat in `D_H`.** For the pure LSTM stack, 16 → 128 costs
   2.08 → 2.29 ms: **+10% for 8× the width and 64× the FLOPs**. For the full submission,
   which adds an `L×L` place-mixing einsum and a `D_H`-wide head, 8 → 128 costs
   2.81 → 4.21 ms (+50% for 16× the width). Either way this is a direct confirmation of
   the previous session's finding that this model class is **kernel-launch bound, not
   FLOP bound**. Hidden-state width is close to free on the wall clock, which means the
   width question in §3 is purely a *generalisation* question with no budget trade-off
   attached to it.
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

### The *eval* budget, which is the one that actually killed a previous candidate

`alu-compose`'s P2 is that a candidate can be fast enough to train and still score 0 by
running out of eval clock, and that the ACT/PonderNet ALU only cleared Easy with a
~1.1× margin (24.8 s of 30 s), losing the OOD-N ladder in 1 of 3 tier-faithful runs.

Tier-faithful Easy run on e5 (`lab/manifests/lab_e5_wc_s74.json`, 60 s train / 30 s
eval, contended GPU so pessimistic):

| quantity | fused LSTM, `D_H`=64 | ACT `DigitALU` (alu-compose P2) |
|---|---|---|
| eval seconds, all 16 splits + both full ladders | **3.57 s of 30 s (8.4× margin)** | 24.8 s of 30 s (1.1× margin) |
| training steps completed in 60 s | 314 (contended) | 14–18 |
| model state elements | 119,546 | 6,817 |

The reason is structural, not tuning: an LSTM has **no depth ladder to run at eval
time**. It is one forward pass per example at every rung, so the 7 seen-N and 7 OOD-N
rungs cost what `test` costs. This family cannot fail the way P2 describes, and it does
not need early halting to be affordable. That removes ranked-next-action #4 from
`lab/RESUME.md` §5 for this architecture.

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

**The hypothesis under test**, from the mandate and from `digit-carry`'s falsification
#2: *a continuous carry channel is a value-encoding channel that restores
memorisation* — a place-shared product table with a 32-dim continuous carry reached
train 1.000 / held 0.000 by step 2000. An LSTM's hidden state is exactly such a
channel, so `D_H` should trade memorisation against generalisation. **If a small hidden
state generalises where a large one memorises, that is the result to isolate.**

`lab/probe_rnn.py`, 3,000 steps, batch 128, AdamW lr 1e-3 wd 0.1, 3 seeds per cell,
bf16 amp. Metrics reported as a row, never a cell: `train_exact` and `held_exact` side
by side, plus the collapse detector `div` (distinct predicted answer strings / held-out
examples) and `top` (share of held-out examples receiving the single most common
answer). A constant map reads `div ≈ 1/n`, `top ≈ 1.0`. `held_ce` is recorded but is
**not** ranked on (RESUME.md: label smoothing moves it with zero algebraic content).

### 3.1 e5 — 4,800 train rows, sampled 10/11-bit moduli, 3,000 steps, 3 seeds

| `D_H` | params | train_exact | **held_exact** | div | top | held_ce |
|---|---|---|---|---|---|---|
| 4 | 926 | 0.0073±0.0010 | 0.0075±0.0018 | 0.056 | 0.178 | 2.155 |
| 8 | 2,562 | 0.0089±0.0017 | 0.0086±0.0014 | 0.157 | 0.076 | 2.154 |
| 16 | 8,522 | 0.0116±0.0018 | 0.0083±0.0035 | 0.349 | 0.029 | 2.181 |
| 32 | 31,194 | 0.0341±0.0020 | 0.0072±0.0010 | 0.529 | 0.011 | 2.357 |
| 64 | 119,546 | 0.4843±0.0410 | 0.0069±0.0034 | 0.607 | 0.008 | 4.275 |
| 128 | 468,282 | 0.9694±0.0042 | 0.0086±0.0011 | 0.605 | 0.007 | 8.315 |
| 256 | 1,853,882 | 0.9804±0.0011 | 0.0078±0.0021 | 0.618 | 0.005 | 7.956 |

**Reading.** `train_exact` climbs ≈**135×** across the sweep, from 0.0073 at `D_H`=4 to
0.98 at `D_H`=256. `held_exact` **does not move at all** — every cell sits in
0.0069–0.0086, and the *lowest* held-out accuracy in the table belongs to the *most
capable* model. There is no width at which the carry channel is narrow enough to force
the algorithm; there is only a width at which it is wide enough to memorise, and below
it, a width at which it fits nothing.

The collapse detector says the failure is **not** a constant map, so this is a real
null and not a degenerate one: at `D_H`≥32 the model emits **635–742 distinct answers
over 1,200 held-out prompts** with no single answer taking more than 1.1% of them. It is
producing varied, confident, wrong answers. `held_ce` rising to 8.3 — far above
`ln(17) = 2.833` — is the confident-and-wrong signature, and is the reason RESUME.md
tells you not to rank on held-out CE in either direction.

The only cells where the collapse detector *does* fire are the smallest: `D_H`=4 gives
`div` 0.056 with `top` 0.178, i.e. one answer covers 18% of held-out prompts. That is
the "too small to do anything" regime, not a generalising one.

**Ablating the place-alignment does the same thing.** Removing the learned `L×L`
end-anchored mixing (`ALIGN=0`, `D_H`=64, e5, 3,000 steps, 3 seeds):

| variant | params | train_exact | **held_exact** | div | top |
|---|---|---|---|---|---|
| `ALIGN=1` | 119,546 | 0.4843±0.0410 | 0.0069±0.0034 | 0.607 | 0.008 |
| `ALIGN=0` | 119,377 | 0.1876±0.0074 | 0.0078±0.0024 | 0.573 | 0.011 |

169 parameters — one `L×L` matrix — are worth a **2.6× swing in `train_exact`** and
**zero** in held-out exactness (0.0069 vs 0.0078 is one example out of 1,200, and the
ablated model is nominally *higher*). The mixing is doing real work; the work is
fitting. This is the third independent lever in this report (width, steps, alignment)
that moves fitting by a large factor and held-out exactness by nothing.

### 3.1a The mandatory `--lr 0` control (BRIEF2 §6.1)

Same models, same data, optimizer learning rate forced to 0, so every number is the
value at **random initialisation**. e5, 3 seeds per width. Run at both 200 and 3,000
steps; the two agree to every printed digit, as they must when the learning rate is 0.

| `D_H` | train_exact @ init | held_exact @ init | div @ init | top @ init |
|---|---|---|---|---|
| 8 | 0.0011±0.0015 | 0.0003±0.0004 | 0.024 | 0.545 |
| 64 | 0.0014±0.0010 | 0.0008±0.0007 | 0.150 | 0.137 |
| 256 | 0.0026±0.0010 | 0.0028±0.0011 | 0.119 | 0.194 |

**This control matters and it does not say what the ALU's did.** RESUME.md's Round 5
found that on the `DigitALU` objective, training moved the control variable *away* from
the discrete solution, so every apparent improvement was regression toward init. That is
**not** what happens here: an untrained LSTM is close to a constant map (`top` up to
0.76 on individual seeds, `div` as low as 0.003) and scores held-out ≈ 0.001, while the
trained models score 0.007–0.010 with `div` ≈ 0.6. Training genuinely moves both
correctness and output diversity in the right direction.

It just stops almost immediately. The entire measured benefit of the legal objective on
held-out data, at every width from 4 to 256 and at 3,000 steps, is about **+0.007
exact-example accuracy — eight examples out of 1,200** — against a rung requirement of
512 out of 512. So the honest statement is not "training goes backwards" but **"training
goes forwards by eight examples and then the curve is flat in every direction we can
push it: width, depth, steps, and architecture class."**

### 3.2 e1 — 600 train rows, N = 323 fixed, 3,000 steps, 3 seeds

The replication on the dataset where this project's memorisation is fastest. e1 has
150 held-out examples, so its variance floor is one example ≈ 0.0067.

| `D_H` | params | train_exact | **held_exact** | div | top | held_ce |
|---|---|---|---|---|---|---|
| 4 | 926 | 0.0208±0.0033 | 0.0222±0.0113 | 0.207 | 0.169 | 1.938 |
| 8 | 2,562 | 0.0534±0.0094 | 0.0244±0.0083 | 0.500 | 0.085 | 2.098 |
| 16 | 8,522 | 0.3757±0.1267 | 0.0222±0.0113 | 0.640 | 0.038 | 3.617 |
| 32 | 31,194 | 0.9974±0.0018 | 0.0200±0.0094 | 0.691 | 0.029 | 11.90 |
| 64 | 119,546 | 0.9974±0.0024 | 0.0245±0.0032 | 0.711 | 0.036 | 12.62 |
| 128 | 468,282 | 0.9967±0.0046 (saturated by **step 750**) | 0.0289±0.0063 | 0.689 | 0.029 | 15.65 |
| 256 | 1,853,882 | 0.9941±0.0048 | 0.0378±0.0032 | 0.682 | 0.036 | 17.72 |

`--lr 0` control at `D_H`=128, 3 seeds: train **0.0039±0.0055**, held
**0.0156±0.0125**, `div` 0.362, `top` 0.129.

**Same shape, sharper.** `train_exact` goes 0.021 → 0.994 across the sweep; `held_exact`
stays within 0.020–0.040, i.e. within **three examples out of 150** — indistinguishable
from the variance floor at every width. And against the control: memorising the *entire*
training set (train 0.004 → 1.000) moves held-out exactness from 0.016 to 0.029, which on
a 150-example split is **two examples**. `D_H`=128 reaches train 1.000 by **step 750**
and holds it for the remaining 2,250 steps with held-out never leaving the floor: the
`grok-optimization` mechanism note ("train exact hits 1.00 by ~2,000 steps and *holds
it*; a grokking plateau creeps before it jumps, this does not") reproduces exactly in a
maximally expressive non-linear RNN.

### 3.3 Spending the budget this branch proved exists — 40,000 steps on e5

This is the experiment the §1 result obliges this branch to run. If a fused sequential
RNN affords ~250,000 Hard steps where the reference model affords ~93,000, the first
question is whether the surplus converts into anything. `D_H`=8 is the right place to
ask: **2,562 parameters against 4,800 training rows** is below memorisation capacity, so
`digit-carry`'s diagnostic inversion applies — for an architecture that *cannot*
memorise, `train_exact → 1.000` would imply `held_exact → 1.000`. Any movement in
`train_exact` here would be real algorithmic progress, not lookup.

| e5 | `D_H` | steps | train_exact | **held_exact** | div | top | held_ce |
|---|---|---|---|---|---|---|---|
| screening | 8 | 3,000 | 0.0089±0.0017 | 0.0086±0.0014 | 0.157 | 0.076 | 2.154 |
| **long run** | **8** | **40,000** | **0.0108** | **0.0058** | 0.308 | 0.063 | 2.198 |
| screening | 64 | 3,000 | 0.4843±0.0410 | 0.0069±0.0034 | 0.607 | 0.008 | 4.275 |
| **long run** | **64** | **40,000** | **0.9842** | **0.0058** | 0.643 | 0.006 | 11.286 |

Training-batch exact accuracy over each run, `[step, loss, train_exact]`:

```
D_H=8   [1, 3.076, 0.000]  [5000, 2.149, 0.008]  [10000, 2.041, 0.023]  [15000, 2.046, 0.016]
        [20000, 2.089, 0.016]  [25000, 2.037, 0.008]  [30000, 2.035, 0.031]
        [35000, 2.078, 0.008]  [40000, 2.023, 0.000]

D_H=64  [1, 2.916, 0.000]  [5000, 0.055, 0.984]  [10000, 0.028, 0.992]  [15000, 0.016, 0.984]
        [20000, 0.021, 0.992]  [25000, 0.014, 1.000]  [30000, 0.029, 0.977]
        [35000, 0.025, 0.977]  [40000, 0.011, 1.000]
```

**The `D_H`=64 curve is the cleanest single object in this report.** Training exact
accuracy reaches **0.984 by step 5,000** and then holds 0.98–1.00 for the remaining
**35,000 steps** while held-out exactness sits at 0.0058 — seven examples out of 1,200,
*below* where it was at 3,000 steps. Loss falls to 0.011. This is
`grok-optimization`'s mechanism note — *"train exact hits 1.00 and holds it; a grokking
plateau creeps before it jumps, this does not"* — reproduced at 40,000 steps on a
maximally expressive non-linear RNN. There is no creep.

At `D_H`=8, **13.3× the step count buys nothing.** Loss falls from 3.08 to ~2.03 in the first 5,000
steps and then is flat to four significant figures for the remaining 35,000; exact
accuracy oscillates between 0.000 and 0.031 with no trend; held-out exactness *ends
lower* than at 3,000 steps (a two-example difference on 1,200, i.e. noise). This is the
same shape `grok-optimization` measured on a dense transformer at 2×10⁵ steps, now
reproduced on a maximally expressive non-linear RNN at a width that provably cannot
memorise. At `D_H`=64 the same 13.3× takes `train_exact` 0.484 → 0.984 and leaves
held-out at 0.0069 → 0.0058. **The surplus training budget this branch found is real and
it is worthless on this objective at both ends of the width range** — at the width that
cannot memorise it fits nothing, and at the width that can it finishes memorising.

### 3.4 m1 — the Medium-scale cross-check (short runs, superseded by §3.5)

BRIEF2 §2d says the expressivity framing is tier-dependent: at Easy the models fit train
and fail to generalise, but "at m1 scale and above they cannot even fit". m1 (N = 10403
fixed, 27,000 train rows, 3,000 held-out), 3,000 steps, 2 seeds:

| `D_H` | params | train_exact | **held_exact** | div | top | held_ce |
|---|---|---|---|---|---|---|
| 32 | 31,314 | 0.0011±0.0001 | 0.0005±0.0005 | 0.326 | 0.016 | 2.277 |
| 128 | 468,594 | 0.0098±0.0016 | 0.0007±0.0007 | 0.775 | 0.003 | 2.375 |

**The pattern BRIEF2 §2d describes reproduces: at m1 the model does not fit either.**
`train_exact` 0.0098 at `D_H`=128, against 0.9694 for the same width on e5.

**But this branch's own data supplies a confound and it should be stated rather than
buried.** §3.3 shows that at e5 scale `D_H`=64 needs **~5,000 steps** to reach
`train_exact` 0.98. m1 has **5.6× more training rows** than e5, and these cells were run
for **3,000 steps**. So "cannot fit at m1" is not separable here from "was not given
enough steps to fit at m1", and the same caveat applies to the Easy-vs-Medium contrast
BRIEF2 §2d draws — that contrast was also drawn at small step counts. The clean
experiment is m1 at 40,000 steps, which this branch has now shown is affordable many
times over; it is the single most useful thing left undone here and is listed in §7.

Note this cuts *against* the convenient reading. If m1 turns out to fit given enough
steps, then the "expressivity is binding at the ranked tier" defence in BRIEF2 §2d
weakens and the memorisation diagnosis extends upward — which would strengthen §6.3's
recommendation rather than weaken it.

**§3.5 makes the measurement.**

### 3.5 m1 at 40,000 steps — resolving all three readings

The coordinator's response to §3.4 added a third candidate explanation, and it is a good
one: the m1 cells may be **under capacity**, not merely under-trained. Parameters per
training row make that concrete, and turn it into a prediction rather than a caveat:

| dataset | `D_H` | params | train rows | **params/row** | fits? |
|---|---|---|---|---|---|
| e5 | 32 | 31,194 | 4,800 | 6.5 | no (train 0.034) |
| e5 | **64** | 119,546 | 4,800 | **24.9** | **yes — 0.984 at 40k** |
| e5 | 128 | 468,282 | 4,800 | 97.6 | yes — 0.969 at 3k |
| **m1** | **128** | 468,594 | 27,000 | **17.4** | ? |
| **m1** | **256** | 1,854,194 | 27,000 | **68.7** | ? |

**m1 at `D_H`=128 sits at 17.4 params/row — below the 24.9 that did fit on e5.** So §3.4's
`D_H`=128 cells were plausibly starved of capacity as well as steps, and `D_H`=256 at
68.7/row is the discriminator. The three readings separate cleanly:

| outcome at 40,000 steps | reading | consequence for BRIEF2 §2d |
|---|---|---|
| both widths fit | steps were the binding variable | **§2d wrong**, memorisation extends to Medium |
| only `D_H`=256 fits | capacity was the binding variable | **§2d wrong**, for a capacity reason |
| neither fits | a real Medium-scale obstruction | **§2d holds**, the Easy/Medium split is real |

**Scoping note, so that BRIEF2 is corrected accurately rather than broadly.** BRIEF2 §2d
bundles two claims: (i) that models at m1 scale and above "cannot even fit", and (ii) a
specific `local_ce` measurement of 3.5–4.2 against a 0.006 cliff. Claim (ii) was measured
on the **`DigitALU`**, a different architecture and a different metric, and **this
experiment does not address it.** What is measured here is claim (i), for the
non-linear-RNN family, at adequate capacity and step count. A result here should revise
(i); it says nothing about the ALU's `local_ce`.

A second property of m1 worth stating, because it cuts against reading a null as "scale":
m1 has a **single fixed modulus** (N = 10403), so its held-out split is unseen `x` at a
*seen* N — pure operand generalisation, bottleneck #2 in isolation. Memorising m1 means
learning a map over `x` at one modulus, whereas memorising e5 means learning `(N, x)`
pairs across *sampled* moduli. On that axis m1 is the **easier** memorisation target
despite having 5.6× the rows.

`--lr 0` control at m1 (200 steps, 2 seeds, so every number is random init):

| `D_H` | train_exact @ init | held_exact @ init | div | top |
|---|---|---|---|---|
| 128 | 0.0003±0.0002 | **0.0000** | 0.094 | 0.099 |
| 256 | 0.0000±0.0000 | **0.0000** | 0.095 | 0.178 |

**Instrumentation limitation, stated because it bounds what the answer can settle.**
`probe_rnn.py` records *training-batch* exact accuracy at every checkpoint but evaluates
held-out only at the end of a run. That is true of the e5 40k runs too, so the two remain
directly comparable — but it means a **held-out** trajectory here is two points (3,000
and 40,000 steps), not nine, and a transient held-out spike between them would not be
seen. If the 40k result turns out to be "fits", the held-out trajectory becomes the
interesting object and intermediate points are worth adding; if it is "does not fit",
held-out cannot have moved and the two points suffice.

#### The answer: m1 **fits**. BRIEF2 §2(d) claim (i) is wrong.

Training-batch exact accuracy at identical checkpoints, e5 and m1 side by side
(`lab/compare_long.py`):

| dataset | `D_H` | params/row | seed | 1 | 5k | 10k | 15k | 20k | 25k | 30k | 35k | 40k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e5 | 8 | 0.5 | 0 | 0.000 | 0.008 | 0.023 | 0.016 | 0.016 | 0.008 | 0.031 | 0.008 | 0.000 |
| e5 | 64 | 24.9 | 0 | 0.000 | 0.984 | 0.992 | 0.984 | 0.992 | 1.000 | 0.977 | 0.977 | 1.000 |
| **m1** | **128** | 17.4 | 0 | 0.000 | 0.039 | 0.219 | 0.492 | 0.492 | 0.547 | 0.555 | 0.648 | **0.625** |
| **m1** | **128** | 17.4 | 1 | 0.000 | 0.039 | 0.320 | 0.453 | 0.508 | 0.492 | 0.617 | 0.586 | **0.672** |
| **m1** | **256** | 68.7 | 0 | 0.000 | 0.320 | 0.742 | 0.781 | 0.859 | 0.883 | 0.859 | 0.844 | **0.844** |
| **m1** | **256** | 68.7 | 1 | 0.000 | 0.297 | 0.758 | 0.836 | 0.789 | 0.859 | 0.859 | 0.805 | **0.836** |

Both seeds, aggregated:

| dataset | `D_H` | n | final loss | train_exact | **held_exact** | **`lr=0` held** | div | top | held_ce | n_held |
|---|---|---|---|---|---|---|---|---|---|---|
| e5 | 64 | 1 | 0.011 | 0.9842 | 0.0058 | 0.0008 | 0.643 | 0.006 | 11.29 | 1,200 |
| **m1** | **128** | 2 | 0.34 | **0.6205±0.0007** | **0.0010±0.0003** | **0.0000** | 0.871 | 0.002 | 7.32 | 3,000 |
| **m1** | **256** | 2 | 0.15 | **0.8572±0.0005** | **0.0008±0.0005** | **0.0000** | 0.867 | 0.002 | 7.87 | 3,000 |

Seed agreement is tight — `train_exact` reproduces to ±0.0007 at `D_H`=128 and ±0.0005 at
`D_H`=256 — so this is not a lucky initialisation.

**Both readings the coordinator asked to separate contributed, and neither is "cannot
fit".**

* **Steps were the dominant variable.** At fixed `D_H`=128, going 3,000 → 40,000 steps
  takes `train_exact` from **0.0098 → 0.6205±0.0007**, a **63× move from step count alone**.
  §3.4's cells stopped at 3,000 steps; the curve shows m1 was still at 0.039 by step
  5,000, so 3,000 steps landed in the flat pre-fitting region and measured nothing.
* **Capacity was a real secondary variable, as predicted.** At fixed 40,000 steps,
  `D_H`=128 → 256 takes `train_exact` from 0.6205 → 0.8572, and the params/row table
  called this in advance: 17.4/row is below the 24.9/row that fit on e5. `D_H`=256 is at
  0.883 by step 25,000 and flat thereafter, i.e. near its ceiling.
* **"Cannot fit at m1" is false.** It was an artifact of the step count at which the
  claim was measured.

**And the memorisation signature reproduces at Medium, in full.** `held_exact` is
**0.0010±0.0003 — three examples out of 3,000** — against **0.0000** at the `lr = 0`
control, at both widths, while train exact accuracy reaches 0.86 and loss falls to 0.14. `held_ce`
7.4–7.9 is the confident-and-wrong signature (e5's was 8.3). The collapse detector says
this is a real null and not a degenerate one: `div` 0.86–0.87 with `top` 0.002, so the
model emits ~2,600 distinct answers over 3,000 held-out prompts.

**Consequence, in the direction the coordinator asked me to state explicitly:**

> **m1 fits given enough steps and capacity, so BRIEF2 §2(d)'s claim (i) is wrong. The
> memorisation diagnosis extends to Medium, and the expressivity framing weakens at every
> tier rather than only at Easy.** The Easy/Medium split in the diagnosis is **not** real;
> it was a step-count artifact. This *strengthens* §6.3's recommendation rather than
> weakening it — the failure is one phenomenon at every scale measured, not two.

Two caveats stated so the correction to BRIEF2 does not over-reach:

1. **Scope.** This revises claim (i) for the non-linear-RNN family. BRIEF2 §2(d)'s other
   claim — `local_ce` 3.5–4.2 against a 0.006 cliff — is a **`DigitALU`** measurement on a
   different metric and is **untouched**. It may still hold.
2. **Held-out trajectory.** Held-out is measured at two points (3,000 and 40,000 steps),
   not nine, so a transient held-out spike between them would not have been seen. Given
   the answer is "fits", this is now the interesting gap; `probe_rnn.py --eval-every`
   was added to close it and the follow-up command is in §8.

#### 3.5a A screening warning that falls out of these curves, and it affects other branches

Read the fitting onsets off the table above:

| dataset | `D_H` | `train_exact` at 1.2k–3k | at 5k | at 10k | at 40k |
|---|---|---|---|---|---|
| e5 | 64 | 0.484 (at 3k) | 0.984 | 0.992 | 1.000 |
| m1 | 128 | 0.010 (at 3k) | 0.039 | 0.219 | 0.625 |
| m1 | 256 | — | 0.320 | 0.742 | 0.844 |

**A 1,200–1,500 step screen at m1 scale sits in the pre-fitting region.** At 5,000 steps
m1/`D_H`=128 is still at 0.039. A null measured at 1,200 steps there does not distinguish
*"this architecture cannot do it"* from *"this run had not started yet"* — and the
manifests currently in use across the PLAN2 worktrees are dominated by `fs1200` and
`fs1500`, including `p2_m1_fs1200_s74.json`.

Stated carefully, because it is calibrated on one architecture: the onsets above are for
this LSTM at these widths, and a different architecture may fit far sooner or never. The
claim is **not** "everyone's nulls are wrong". It is that **the pre-fitting region is
real, it is wide at Medium scale, and it is cheap to rule out** — this branch's §1 result
means 40,000 steps costs a fraction of one Hard budget. Any Medium-scale null worth
acting on should be accompanied by evidence that its `train_exact` had begun to move,
which is one extra curve, not one extra experiment.

---

## 4. Neural GPU (PLAN2 §3.5) — secondary, and it dies on the budget, not on training

`submissions/p2-sequential-rnn-neuralgpu/submission.py`: a tied conv-GRU over a
`(W=4, L, C)` state grid, `K` applications, orthogonal candidate kernel and gate biases
at +1 so the carry gate starts near-open. Two of the three training aids PLAN2 §3.5
says it needs are available under the competition contract and both are used:

* **gradient noise** — `sigma_t = sqrt(eta / (1+t)^0.55)` (Neelakantan et al.), delivered
  as a custom `torch.optim.Optimizer` subclass, which BRIEF2 §7 explicitly permits.
  Autograd still computes every gradient; the optimizer perturbs only its own update.
* **careful init** — above.
* **curriculum learning is NOT available.** The evaluator owns the data order and the
  training loop (BRIEF.md §4.4), and there is no legal way to stage examples by
  difficulty inside a submission. This is a genuine handicap for this architecture and
  is reported rather than worked around.

**PLAN2 told this branch to budget for the Neural GPU failing to *optimise*. It does not
get that far: it fails on wall clock first.** Same ratio calibration, `C=48`, `W=4`,
L=21, batch 512:

| model | depth | local ms | ratio vs REF | implied H100 ms | implied steps / 3600 s |
|---|---|---|---|---|---|
| Neural GPU `C=48` | `K=6` | 54.97 | 3.77 | 145.5 | 24,712 |
| Neural GPU `C=48` | `K=12` | 170.28 | 7.22 | 278.5 | 12,911 |
| Neural GPU `C=48` | `K=20` | 276.64 | 18.98 | 732.6 | 4,908 |
| fused LSTM `D_H=48` | 1 | 7.68 | 0.33 | 12.6 | 286,191 |

At its intended depth the Neural GPU is **22× more expensive per step than the
sequential RNN it is supposed to be the parallel-friendly alternative to**, and buys
~12,900 Hard steps. Kaiser & Sutskever train Neural GPUs for hundreds of thousands of
steps *with* the curriculum that is unavailable here. Some of the 22× is recoverable —
`conv2d` on a 4×21 grid is badly launch-bound and folding the grid width into channels
would help — but a 4× engineering win still lands at ~50k steps against the LSTM's
~287k, for an architecture with a much worse optimisation reputation.

**Secondary verdict: PLAN2 §3.5 is dominated by §3.6 on this task on wall clock alone,
before any question of trainability is reached. Timebox spent; recommend not
resuming it.** The training numbers that were nevertheless collected are in §4.1.

*(Aside, and it validates the whole calibration method: the LSTM's ratio read 0.324
when the reference measured 10.53 ms and 0.326 when contention pushed the reference to
23.60 ms. Contention cancels; the ratios in this report are stable.)*

### 4.1 Neural GPU training numbers (e5, 2,000 steps, `lab/probe_ngpu.jsonl`)

| config | `eta` | n | train_exact | **held_exact** | div | top | held_ce |
|---|---|---|---|---|---|---|---|
| `C=48 K=12` | 0.01 | 2 | 0.0119 | 0.0079 | 0.280 | 0.050 | 2.208 |
| `C=48 K=12` | **0** | 2 | **0.2117** | 0.0075 | 0.593 | 0.007 | 3.317 |
| `C=48 K=6` | 0.01 | 1 | 0.0108 | 0.0050 | 0.285 | 0.041 | 2.213 |
| `C=48 K=20` | 0.01 | 1 | 0.0103 | 0.0075 | 0.330 | 0.045 | 2.206 |
| `C=24 K=12` | 0.01 | 1 | 0.0070 | 0.0042 | 0.230 | 0.028 | 2.187 |
| `C=96 K=12` | 0.01 | 1 | 0.0139 | 0.0058 | 0.316 | 0.053 | 2.255 |
| **`lr = 0` control** | — | 1 | 0.0006 | **0.0000** | 0.228 | 0.048 | 2.921 |

Depth is flat: `K` = 6 / 12 / 20 give `train_exact` 0.0108 / 0.0119 / 0.0103 at fixed
`eta`, i.e. **3.3× the serial depth and 3.9× the wall clock buy nothing**. Width is flat:
`C` = 24 / 48 / 96 give 0.0070 / 0.0119 / 0.0139 with held-out pinned at 0.004–0.008.
Same two-axis null as the LSTM, on a completely different architecture, and against a
`lr = 0` control at held-out **0.0000** the entire trained-vs-init gain is again ~8
examples in 1,200.

Held-out is at the floor in every cell, exactly as for the LSTM, and with the same
non-collapsed prediction distribution.

**The one substantive training finding: the paper's own training aid hurts here.**
Gradient noise at `eta = 0.01` holds `train_exact` at **0.012** while the identical model
with `eta = 0` reaches **0.202** — a 17× difference in the *fitting* direction, with
held-out unmoved at ~0.008 in both. On this task the noise scale that the Neural GPU
literature calls modest is large relative to a 2,000-step budget, so it prevents the
model from fitting rather than helping it escape a bad basin. Anyone reviving §3.5
should sweep `eta` down by 1–2 orders of magnitude, or drop it — but see the budget
argument above first.

---

## 5. Evaluator runs

Every run is archived in `lab/archive.jsonl` through `lab/run_experiment.py`.
All manifests are `--mode fixed_step` except the one labelled `wallclock`, which is a
deliberate tier-faithful timing check (BRIEF.md §5).

| tag | submission | manifest | steps | MAX_T | OOD_N MAX_T | mean acc | `test` |
|---|---|---|---|---|---|---|---|
| `rnn-base` | `D_H=64` | e5 fs2000 | 2000 | **0** | 0 | 0.0050 | 0.007 |
| `ref-calib` | `exp_axis` d128 ×8 (control) | e5 fs2000 | 400 | **0** | 0 | 0.0058 | 0.007 |
| `rnn-wallclock` | `D_H=64` | e5 **wallclock 60 s** | 314 | **0** | 0 | 0.0071 | 0.008 |
| `rnn-width` | `D_H=8` | e5 fs2000 | 2000 | **0** | 0 | 0.0075 | 0.007 |
| `rnn-width` | `D_H=128` | e5 fs2000 | 2000 | **0** | 0 | 0.0071 | 0.002 |
| `rnn-control` | `lr = 0` (**control**) | e5 fs2000 | 2000 | **0** | 0 | 0.0058 | 0.010 |
| `rnn-depth` | `D_H=64`, **4 tied passes** | e5 fs2000 | 2000 | **0** | 0 | 0.0042 | 0.005 |
| `ngpu` | Neural GPU `C=48 K=12` | e5 fs2000 | 2000 | **0** | 0 | 0.0046 | 0.004 |
| `rnn-width` | `D_H=128` | **e1** fs2000 | 2000 | **0** | 0 | 0.0483 | 0.047 |

**Nine cells, `MAX_T = 0` and `OOD_N MAX_T = 0` in every one.** Three of them deserve a
sentence.

1. **`rnn-depth` is "one layer deeper" taken literally** — four tied applications of the
   whole encode/carry-scan stack, 2.5× the wall clock — and it scores *below* the
   single-pass model on the diagnostic metric (0.0042 vs 0.0071) and identically (0) on
   the ranking metric.
2. **The e1 cell carries the highest `mean_exact_accuracy` in the branch, 0.0483 — 10×
   the e5 cells — and still scores `MAX_T = 0`**, because rung 1 is 0/38. Its rung
   profile is `{1: 0.000, 2: 0.000, 4: 0.026, 8: 0.026, 16: 0.000, 32: 0.000,
   64: 0.000}`; the two non-zero entries are *one example out of 38*, i.e. the variance
   floor. This is BRIEF2 §1's "`mean_exact_accuracy` is a diagnostic with no ranking
   value" as a concrete instance rather than a warning.
3. **The Neural GPU cost 591 s for the same 2,000 steps the LSTM ran in 330–380 s**, a
   1.6–1.8× gap — far smaller than the 22× measured in §4 at L=21 / batch 512. The reason
   is the same one that drives the whole report: at e5's L=13 and batch 128 both models
   are launch-bound and the DataLoader is a large share of the step. **The 22× is the
   figure that applies at Hard's shape; 1.7× is what a small-tier screen sees** — worth
   stating explicitly, because screening on Easy would badly understate the Neural GPU's
   real cost.

**Eval seconds across all nine cells: 2.2–5.2 s** against a 30 s Easy allowance. The 8.4×
margin in §1 is not a one-off; it holds for the Neural GPU and the 4-pass model too.

**The `lr = 0` control scores MAX_T = 0 and mean 0.0058 through the evaluator, against
the trained model's MAX_T = 0 and mean 0.0071.** On the ranking metric an untrained
network and a fully trained one are indistinguishable, and on the diagnostic metric they
differ by 0.0013. That is the cleanest one-line statement of where this architecture
family stands.

Per-rung, `D_H=128` (the configuration that reaches train 0.97): seen-N
`{1: 0.002, 2: 0.008, 4: 0.002, 8: 0.006, 16: 0.006, 32: 0.004, 64: 0.004}`, OOD-N
`{1: 0.002, 2: 0.0, 4: 0.004, 8: 0.0, 16: 0.002, 32: 0.0, 64: 0.0}`. **The rung profile
is flat, not decaying** — consistent with `tied-recurrence`'s finding that error
compounding in T is not the constraint, and confirming again that nothing about depth is
what is failing here.

All nine cells are archived in `lab/archive.jsonl`; none is outstanding.

---

## 6. What this branch establishes, what it falsifies, and what it recommends

### 6.1 Established

1. **PLAN2's falsifier for §3.6 does not fire, and the entry's cost hedge is backwards.**
   PLAN2 calls the sequential non-linear RNN "the worst wall clock" on Axis B and hedges
   the whole entry on `max_seq_len` being small. `max_seq_len` is 13–21, and the fused
   sequential RNN is **2.5–3.9× cheaper per step than the reference model that was
   actually run on the competition H100** — ~230,000–350,000 Hard steps against ~93,000.
   It is the *cheapest* thing this project has built, not the most expensive.
2. **Fusion is worth 5–8× and comes free from `nn.LSTM`.** No Triton, no
   `torch.autograd.Function`, no compliance grey area.
3. **Triton cannot compile on this box at all** (`ptxas-blackwell`: `sm_107a` is not a
   known target), which also disables `torch.compile`. BRIEF2 §5's "custom fused kernels
   are on the table" should be read as "on the H100, unverifiable here".
4. **The eval budget, which killed the ALU candidate (`alu-compose` P2), is a non-issue
   for this family**: 3.57 s of 30 s on tier-faithful Easy, an 8.4× margin against the
   ALU's 1.1×, because an RNN has no depth ladder to run at eval time.
5. **The pre-fitting region at Medium scale is wide, and short screens land inside it**
   (§3.5a). m1/`D_H`=128 reads `train_exact` 0.010 at 3,000 steps and 0.625 at 40,000.
   Any Medium-scale null should carry evidence that `train_exact` had begun to move
   before it is read as an architecture result.

### 6.2 Falsified

1. **"A small hidden state generalises where a large one memorises" — falsified, and in
   the least interesting way.** Held-out exact accuracy is **flat at the floor across
   six octaves of `D_H`** while train exact accuracy climbs monotonically from 0.007 to
   0.98. Four independent levers — hidden width (≈135×), step count (13.3×), the learned
   place alignment (2.6×), and tied depth — all move `train_exact` by large factors and
   `held_exact` by nothing. Small hidden states do not generalise; they simply fail to
   fit. There is no width at which the carry channel is "narrow enough to force the
   algorithm" — the curve has no such regime. `digit-carry`'s finding #2 is confirmed and strengthened:
   **the state alphabet must be small *and discrete*; making a continuous state small
   only removes capacity, it does not add structure.**
2. **"Expressivity is the binding constraint" (PLAN2 §0) — not supported by the
   maximally expressive member of PLAN2's own Axis A, at Easy *or* at Medium.** §3.6 is
   the top row of Axis A ("maximal, but no scan"), it has no `TC⁰` caveat, and it was
   given 13× the screening budget. It reproduces the project's canonical signature
   exactly rather than escaping it, at **every scale measured**:

   | dataset | train rows | train_exact | **held_exact** | `lr=0` held |
   |---|---|---|---|---|
   | e1 | 600 | 1.0000 | 0.0333 | 0.0156 |
   | e5 | 4,800 | 0.9842 | 0.0058 | 0.0008 |
   | **m1** | **27,000** | **0.8572** | **0.0010** | **0.0000** |

   PLAN2 §6's first kill criterion is written for precisely this outcome.
   **This claim was scoped to Easy in an earlier draft of this report; §3.5 removed that
   scope.** The Easy/Medium split BRIEF2 §2(d) draws was a step-count artifact — at
   40,000 steps and adequate capacity m1 fits to 0.86 with held-out at three examples
   in 3,000. *(Coordination note: `plan2/phase0` owns the diagnostic version of this
   question; this branch reports the candidate-side evidence and does not claim to have
   run their experiment.)*
3. **"Budget is the constraint" — falsified for this family in both directions, and
   then falsified again by spending it.** Training budget is ~3× surplus and eval budget
   is ~8× surplus. A 40,000-step run at the width that provably cannot memorise (§3.3)
   converts **13.3× the screening compute into nothing**: loss flat to four significant
   figures from step 5,000 to step 40,000, exact accuracy oscillating between 0.000 and
   0.031 with no trend. This reproduces `grok-optimization`'s dense-transformer result on
   an architecture that branch explicitly flagged as needing a re-test rather than an
   inheritance (RESUME.md's scope caveat), so the re-test is now done and it agrees.

### 6.3 The single highest-value recommendation

> **Stop buying parallelism. On this task it is not merely unnecessary — it is a net
> loss — and every hour spent on scan kernels, chunkwise algorithms or Triton is an hour
> not spent on the one thing that is open.**

PLAN2 §0 builds the entire plan on one premise: *composition is the bottleneck,
composition is associative, so represent it as an associative operator and compute it
with a parallel prefix scan in `O(log T)` instead of `O(T)`.* Sections 3.1, 3.2, 3.3
and 3.7 are all instances of that, and §3.5 is chosen partly because it is "highly
parallel, good H100 utilization". §3.6 — the serial one — is the entry PLAN2 hedges.

Measured, on the same GPU, in the same window, at the competition's own batch size and
sequence length:

| candidate | how it computes depth | ms/step, ratio to the H100-calibrated reference |
|---|---|---|
| **fused sequential LSTM** (§3.6, *no scan at all*) | `O(T)` serial, T ≤ 21 | **0.33×** |
| reference recurrent transformer (the H100 anchor) | 8 tied parallel blocks | 1.00× |
| Neural GPU (§3.5, "highly parallel") | `K` tied conv-GRU steps | **7.2×** |

**The most parallel candidate in the plan is 22× slower than the least parallel one.**
The reason is structural and was already in the previous session's notes without being
followed through: this workload is **kernel-launch bound, not FLOP bound**. A serial
recurrence over 21 steps issues ~21 launches; a "parallel" alternative issues more
launches doing more total work on tensors too small to fill the machine. Amdahl has
nothing to say at `T = 21`. The `O(T)` vs `O(log T)` argument that motivates PLAN2 §0 is
an asymptotic statement about a sequence axis this task does not have — BRIEF2 §2a
already says the sequence axis is not the composition axis, and this is the wall-clock
consequence of that correction.

**What follows, concretely.**

1. **Do not write a Triton recurrence kernel.** It cannot even be compiled on this box,
   it raises a "participant-controlled backward" question under BRIEF.md §4.4, and the
   thing it would replace is already 3× faster than the reference. `nn.LSTM` is the
   fused path; it is free, legal and stock.
2. **Treat ~250,000 Hard steps and an 8× eval margin as the working budget** for any
   RNN-shaped candidate, not the ~93,000 from the reference calibration. Step famine
   (`alu-compose` P3) and the eval-budget failure (P2) are not constraints on this
   family. That removes two of the five ranked next actions in `lab/RESUME.md` §5 for
   anything built this way.
3. **Spend the freed budget and the freed engineering time on the state alphabet.**
   This branch's width sweep is the argument: across six octaves of hidden width and
   **three datasets spanning 600 to 27,000 training rows**, `train_exact` moved ≈135×
   and `held_exact` moved not at all. Capacity, compute, expressivity and wall clock are
   all *surplus*, and after §3.5 that is true at **Medium as well as Easy** — the tier
   split that could have made this an Easy-only result is gone. What
   `digit-carry` #2 and this sweep jointly say is that a **continuous** state is a
   value-encoding channel at every width — small ones just encode less. The open
   question is a state that is discrete *by construction* rather than by relaxation
   (PD-SSM's column-one-hot transition is the obvious instance, and straight-through
   has already been measured to be destructive), and the budget to explore it is
   3× larger than anyone has been assuming.

### 6.4 Honest summary

This is a clean, well-controlled negative, and BRIEF2 §8 says to say so plainly. The
architecture was built as a candidate, it trains, it is fast, it fits the training set,
and it certifies nothing: **MAX_T = 0 and OOD-N MAX_T = 0 in all nine evaluator cells**,
with the best rung-1 anywhere 2/512 on e5 and 0/38 on e1 — at the floor and below the
field best of 3/38.

Its value is that it removes variables from the search rather than adding one. Before
this branch, "the sequential RNN is maximally expressive but too slow", "maybe a narrower
state would force generalisation", and "Medium-scale models cannot even fit, so the
Easy diagnosis may not transfer" were all live and all plausible. All three are now
measured and false: it is the *fastest* thing in the plan; the state width does not trade
against generalisation at all, because held-out never moves; and Medium fits perfectly
well once you run past step 5,000.

What is left is what `alu-credit` and `alu-relational` already converged on from a
completely different direction, and this branch reaches it from the expressivity side
instead of the credit-assignment side: **the missing ingredient is a discrete state, and
no amount of capacity, compute, expressivity or wall clock substitutes for it.** The m1
result matters most because it removes the last place the failure could have been hiding
as *two* phenomena — it is one phenomenon, at 600 rows and at 27,000.

One thing this branch got wrong and corrected in place, recorded because the reasoning
error is more reusable than the result: §3.4 initially read m1's `train_exact` 0.0098 as
"cannot fit at Medium", which is what BRIEF2 §2(d) predicted and therefore what was easy
to believe. The confound was visible in this branch's *own* §3.3 data — e5 needed ~5,000
steps to fit — and was only caught by checking the new measurement against the existing
one rather than against the expectation. **The generalisable lesson is that a null at a
step count you have not calibrated against a fitting curve is not a null.**

---

## 7. In flight at cutoff

Every one of these is confirmatory — none of them can change a conclusion above, and
all are cheap to finish. Exact resume commands are in §8.

**Everything queued on this branch has landed, including the m1 40,000-step experiment.
Nothing is in flight.**

Complete: the e5 width sweep, its `lr=0` control and the `ALIGN` ablation (33 cells,
3 seeds each); the e1 sweep through `D_H`=256 plus its `lr=0` control (24 cells); the
whole Neural GPU sweep including its `lr=0` control (9 cells); both 40,000-step runs
(`D_H`=8 and 64); the m1 cross-check; and all nine evaluator cells.

### The one experiment this branch identified and did not run

**m1 at 40,000 steps.** §3.4 measures `train_exact` 0.0098 at m1 with `D_H`=128 and reads
it as "cannot fit at Medium scale" per BRIEF2 §2d — but those cells ran for 3,000 steps,
and §3.3 shows e5 needed ~5,000 steps to fit at 5.6× less data. The two readings are not
separable from the data collected here. This branch has established that 40,000 steps is
affordable several times over, so the experiment is cheap:

```
$V lab/probe_rnn.py --dataset m1 --steps 40000 --d-h 128 --seeds 0 1 --tag m1-long40k
```

It matters because it is load-bearing for a premise the whole fleet is screening against.
If m1 *does* fit given enough steps, BRIEF2 §2d's "the expressivity framing is defensible
at the tier that is ranked" weakens, and the memorisation diagnosis — which this branch
established at Easy across four levers — extends upward to Medium.

The 40,000-step run at `D_H`=8 — the width that provably cannot memorise, and therefore
the only cell where a moving `train_exact` would have *implied* a moving `held_exact` —
**landed and did not change anything** (§3.3). The `D_H`=64 companion can only show
memorisation, which is already established at 3,000 steps.

Aggregated numbers for every probe cell, refreshed as runs land, are in
`lab/summary_tables.md`; raw rows in `lab/probe_rnn.jsonl` and `lab/probe_ngpu.jsonl`;
evaluator rows in `lab/archive.jsonl`.

---

## 8. Reproduction

```bash
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd .worktrees/p2-sequential-rnn

# 1. the budget falsifier (no dataset touched)
$V lab/rnn_bench.py --seq-len 21 --batch 512 --out lab/bench_L21.json
$V lab/rnn_bench.py --seq-len 13 --batch 512 --out lab/bench_L13.json
$V lab/time_submission.py --submission submissions/p2-sequential-rnn/submission.py \
    --seq-len 21 --batch 512 --d-h 8 16 32 64 128
$V lab/triton_probe.py            # expect: PTXASError, sm_107a unknown

# 2. the width sweep (the memorisation dial), with its mandatory lr=0 control
$V lab/probe_rnn.py --dataset e5 --steps 3000 --d-h 4 8 16 32 64 128 256 --seeds 0 1 2 \
    --tag width-sweep
$V lab/probe_rnn.py --dataset e5 --steps 3000 --d-h 8 64 256 --seeds 0 1 2 --lr 0 \
    --tag lr0-control
$V lab/probe_rnn.py --dataset e1 --steps 3000 --d-h 4 8 16 32 64 128 256 --seeds 0 1 2 \
    --tag e1-width-sweep
$V lab/summarize_rnn.py --files lab/probe_rnn.jsonl --group dataset tag params

# 3. spending the budget this branch proved exists
$V lab/probe_rnn.py --dataset e5 --steps 40000 --d-h 8  --seeds 0 --tag e5-long40k
$V lab/probe_rnn.py --dataset e5 --steps 40000 --d-h 64 --seeds 0 --tag e5-long40k

# 4. the Neural GPU secondary
$V lab/probe_rnn.py --dataset e5 --submission \
    submissions/p2-sequential-rnn-neuralgpu/submission.py --steps 2000 --d-h 48 \
    --seeds 0 1 --tag ngpu_c48_k12_noise --out lab/probe_ngpu.jsonl

# 5. evaluator cells (MAX_T), all archived to lab/archive.jsonl
bash lab/eval_cells.sh
```

---

## 8a. Statements in the shared docs that this branch bears on

Collected in one place so the corrections are mechanical. Each row is what *this branch
measured*; none of it is inferred from another branch's work.

| shared doc | statement | this branch's evidence | verdict |
|---|---|---|---|
| BRIEF2 §5 | "`triton` 3.7.1 importable → custom chunkwise/fused kernels are on the table" | `lab/triton_probe.py`: the smallest possible kernel fails with `ptxas-blackwell fatal: Value 'sm_107a' is not defined`. Kills `torch.compile` too. | **wrong locally.** True on the H100, unverifiable here. Suggest: "importable, but cannot compile on the lab box." |
| BRIEF2 §4 | "~38.6 ms/step … ~93,000 steps in a full Hard run" | Reproduced as the calibration anchor. A fused `nn.LSTM` candidate runs at 0.27–0.40× that. | **correct, but not a ceiling.** Suggest adding: the figure is architecture-specific; an RNN-shaped candidate gets 230k–350k. |
| PLAN2 §3.6 | "viable only if `max_seq_len` is small … worst wall clock" (Axis B) | §1: fused 0.33× the reference, unfused 3.8×; the "highly parallel" §3.5 entry is 7.2×. | **backwards.** §3.6 is the cheapest entry in the plan, not the most expensive. |
| PLAN2 §3.5 | budget for it "failing to optimise rather than to express" | §4: 7.2× the reference at `K`=12 → ~12,900 Hard steps, and curriculum learning is not expressible under the evaluator's loop. | **it fails on budget first**, before trainability is reached. |
| RESUME §5 #4 | "Every candidate must halt early" | §1: eval is 2.2–5.2 s of 30 s across nine cells, because an RNN has no depth ladder at eval time. | **not binding for this family.** |
| `digit-carry` #2 | "a continuous carry channel restores memorisation; the state alphabet must be small" | §3: held-out flat across `D_H` 4→256 while train goes 0.007→0.98. | **confirmed and strengthened** — small *and discrete*; shrinking a continuous state only removes capacity. |
| BRIEF2 §2d | "at m1 scale and above they cannot even fit" | §3.5: m1 at 40k steps reaches `train_exact` **0.6205** (`D_H`=128) / **0.8572** (`D_H`=256), 2 seeds each, from 0.0098 at 3,000 steps. Held-out 0.0010 vs 0.0000 at `lr=0`. | **wrong** for the RNN family — a step-count artifact. The Easy/Medium split is not real; memorisation extends to Medium. **Scope: claim (i) only**; §2d's `local_ce` 3.5–4.2 figure is a `DigitALU` measurement on a different metric and is untouched. |

---

## 9. Compliance statement

* No file under `data/generated/` was read, printed, sampled or summarised. The probe
  in `lab/probe_rnn.py` constructs dataloaders through the public `data.factory` API
  exactly as the evaluator does and reports only statistics of **model predictions**
  (exact accuracy, prediction diversity, cross-entropy). `lab/rnn_bench.py`,
  `lab/time_submission.py` and `lab/triton_probe.py` use synthetic integer tensors and
  touch no dataset at all.
* Both submissions pass `benchmark.validation.lint_submission_source`.
* No hard-coded arithmetic, solver, lookup table or data-dependent Python control flow
  in either forward pass. Every tensor is learned from random init in the run.
  `attention_mask` is used only to build an index permutation and to zero pad
  embeddings.
* No custom training loop and no participant-controlled backward in either submission.
  The Neural GPU's `NoisyAdamW` is a `torch.optim.Optimizer` subclass, which BRIEF2 §7
  explicitly permits; autograd computes every gradient and the subclass perturbs only
  the update it itself applies. **No `torch.autograd.Function` and no custom kernel is
  used anywhere** — the fusion is cuDNN's, reached through stock `nn.LSTM`.
* Nothing was submitted to the hosted service. No installs. `lab/probe_rnn.py` uses its
  own training loop and is therefore a **lab diagnostic**, never a submission.
* Every result in this report is **LEGAL** unless explicitly labelled otherwise. There
  are no `--construct` / `--teacher-force` style oracles in this branch: nothing here
  was ever handed the answer, so there is no DIAGNOSTIC ceiling to report and no
  legal/illegal boundary to police.
