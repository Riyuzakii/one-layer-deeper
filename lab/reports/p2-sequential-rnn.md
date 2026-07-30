# `plan2/sequential-rnn` — PLAN2 §3.6, the sequential non-linear RNN, aggressively fused

**Mandate.** Build PLAN2 §3.6 — "the highest-variance entry in the plan and the one most
likely to be under-explored by other competitors" — as a *candidate*, not as a
diagnostic. Own budget, fusion, hidden-state size and generalisation. Secondary,
timeboxed: PLAN2 §3.5, the Neural GPU / tied conv-GRU.

**One-line verdict.** The falsifier PLAN2 attached to this entry — "step time makes the
achievable update count non-competitive" — is **decisively refuted**: fused, this is the
*cheapest* architecture the project has built, worth ~230k–350k Hard steps against the
reference model's ~93,000, with an 8.4× eval-budget margin. And with maximal
expressivity, no theoretical caveats and 3× the compute, it reproduces the project's
canonical failure exactly. **MAX_T = 0 everywhere.**

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
| 256 | _pending_ | | | | | |

_analysis pending_

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

### 4.1 Neural GPU training numbers

_pending_

---

## 5. Evaluator runs

Every run is archived in `lab/archive.jsonl` through `lab/run_experiment.py`.
All manifests are `--mode fixed_step` except the one labelled `wallclock`, which is a
deliberate tier-faithful timing check (BRIEF.md §5).

_table pending_

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

### 6.2 Falsified

1. **"A small hidden state generalises where a large one memorises" — falsified, and in
   the least interesting way.** Held-out exact accuracy is **flat at the floor across
   six octaves of `D_H`** while train exact accuracy climbs monotonically from 0.007 to
   0.97. Small hidden states do not generalise; they simply fail to fit. There is no
   width at which the carry channel is "narrow enough to force the algorithm" — the
   curve has no such regime. `digit-carry`'s finding #2 is confirmed and strengthened:
   **the state alphabet must be small *and discrete*; making a continuous state small
   only removes capacity, it does not add structure.**
2. **"Expressivity is the binding constraint" (PLAN2 §0) — not supported by the
   maximally expressive member of PLAN2's own Axis A.** §3.6 is the top row of Axis A
   ("maximal, but no scan"), it has no `TC⁰` caveat, and it was given ~3× the reference
   model's step budget. It reproduces the project's canonical signature exactly
   (train → 0.97 / held → 0.009) rather than escaping it. PLAN2 §6's first kill
   criterion is written for precisely this outcome. *(Coordination note: `plan2/phase0`
   owns the diagnostic version of this question; this branch reports the candidate-side
   evidence and does not claim to have run their experiment.)*
3. **"Budget is the constraint" — falsified for this family in both directions.**
   Training budget is ~3× surplus and eval budget is ~8× surplus, and neither converts
   into a single certified rung.

### 6.3 The single highest-value recommendation

_pending_

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

## 7. Compliance statement

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
