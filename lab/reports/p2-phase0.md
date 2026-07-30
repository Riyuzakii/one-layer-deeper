# PLAN2 Phase 0 — diagnosis before construction

Branch `plan2/phase0`. Mandate: PLAN2 §2, all four diagnostics, before any sibling
builds. Environment questions were settled in BRIEF2 §5 and are not re-derived
(one correction from `plan2/sequential-rnn` is carried in §8).

Every row is labelled **LEGAL** (a plain `benchmark.runner` evaluation of a
submission trained from random init, no oracle anywhere) or **DIAGNOSTIC** (a lab
instrument that could never be a submission). No file under `data/generated/` was
read, printed or summarized; every dataset used here was produced by the public
generator with the exact commands in `lab/gen_p2_grid.sh` and
`lab/gen_p2_isolate.sh`.

---

## 0. Verdict, stated first

**PLAN2 §0's expressivity reframe is WRONG, and PLAN2 §6 says so on two of its
own three kill criteria.**

1. **Probe 4 fires the third kill criterion.** Diagonal-`[0,1]`, diagonal-`[-1,1]`,
   DeltaNet-`[0,1]` and DeltaNet-`[-1,1]` — at *identical* parameter count
   (465,280 each), identical scaffold, identical compute, matched init, 3 seeds —
   are mutually **indistinguishable** on the competition task, and all four are
   indistinguishable from a model with **no sequence mixing at all**. PLAN2 §6:
   *"If probe (4) shows diagonal-`[0,1]` performs the same as DeltaNet-`[-1,1]`,
   the task is not a state-tracking problem, §0 is wrong, and the space to explore
   is a different one entirely."*
2. **Probe 1 fires the first kill criterion** (confirming `plan2/sequential-rnn`).
   A maximally expressive non-linear RNN gets nothing.
3. **But PLAN2 §6's remedy — "stop and re-examine the harness, the loss and the
   label alignment" — is CLOSED by measurement, not open.** A constructed oracle
   certifies **MAX_T = 64 and OOD_N_MAX_T = 64** through the real evaluator on
   e5, m1 and hp1. And a *legal, learned* model reaches **train 1.000 / held-out
   1.000 on unseen 7-digit operands** on a T=0 (copy) dataset that shares the
   entire pipeline. The harness is sound. The nulls are facts about learning.

The one-line replacement for §0's diagnosis:

> The failure is not that the model cannot *represent* the composition. It is
> that **one squaring of an unseen operand is not determined by any amount of
> squaring of other operands** under the supplied objective. Held-out accuracy
> goes from **1.000 to 0.000** when the target changes from `x` to `x² mod N`,
> with everything else in the pipeline byte-identical.

**Do not build §3.1 (matrix associative scan), §3.2 (PD-SSM) or §3.3
(DeltaProduct) expecting expressivity to be the unlock.** Their entire
justification is Axis-A expressivity on the composition axis, and (a) the
composition axis is not a sequence axis here, (b) the Axis-A ordering measured on
the axis that *does* exist is flat, and (c) T-composition was already solved last
session. §3.7's chunked exactness and BRIEF2 §2c's digit-position carry scan are
the only parts of PLAN2 that survive this phase, and they survive as
*conditioning* arguments, not expressivity ones.

---

## 1. The framing correction that governs every probe: what axis is the recurrence on?

`max_seq_len` is 13 (e1/e5) / 15 (m1) / 21 (m4); the prompt is
`[N] d(N) [X] d(x) [T] d(T)` and `max_seq_len = 2·digits(N_max) + 5` once the
depth ladder reaches T=64. The task's composition depth `T ≤ 64` is **not** a
sequence dimension.

So for every probe here:

| probe | what the recurrence runs over | what a result can and cannot mean |
|---|---|---|
| 1 (LSTM/GRU) | the **prompt-token axis**, 11–21 tokens, bidirectional | tests serial state over *digit positions*; says nothing about T-fold squaring |
| 2 (length) | same | length only moves with modulus digit count, so length and operand-space size are confounded by construction |
| 3 (train/eval) | same | architecture-independent property of the task |
| 4 (solvability) | same | reads the algebraic class of the **token-axis** transition monoid |

This is not a limitation to apologise for — it is the *right* axis for the one
open bottleneck. BRIEF2 §2c: carry propagation across digit positions is an
associative prefix computation, and the digits of `x` and `N` are **consecutive
prompt tokens**. A token-axis recurrence is exactly a digit-position recurrence.
So probe 4's ordering answers: *what algebraic class is needed to get `x² mod N`
out of digit tokens?*

What probe 4 **cannot** answer is whether T-fold composition needs a non-solvable
monoid. That question is closed anyway — BRIEF2 §2b: exactness composes 1.000 at
every rung T=1…64 across five regimes, and a 12-scalar learned controller reaches
MAX_T=64. Nobody needs a scan for it.

---

## 2. Instruments

One template, one scaffold, one swappable sequence mixer
(`lab/make_p2arch.py`): token embedding + learned positions → `N_LAYERS=2` ×
[pre-norm mixer + residual, pre-norm 4× MLP + residual] → final norm → head tied
to the embedding. `D_MODEL=128`, `N_HEADS=4`, `D_HEAD=32`, AdamW lr 1e-3
β=(0.9,0.95) wd 0.1, batch 512 — the baseline's recipe, so results are comparable
to the ~950 prior experiments.

| `ARCH` | transition acting on the recurrent state | eigenvalues | params |
|---|---|---|---|
| `diag01` | `S ← S·diag(a)`, `a = σ(g)` | `[0,1]` | **465,280** |
| `diagpm1` | `S ← S·diag(a)`, `a = 2σ(g)−1` | `[-1,1]` | **465,280** |
| `delta01` | `S ← S(I − βkkᵀ)`, `β = σ(g)`, ‖k‖=1 | `[0,1]` | **465,280** |
| `deltapm1` | `S ← S(I − 2βkkᵀ)`, `β = σ(g)`, ‖k‖=1 | `[-1,1]` | **465,280** |
| `lstm` | bidirectional non-linear gated vector recurrence | — | 862,336 |
| `gru` | bidirectional non-linear gated vector recurrence | — | 730,240 |
| `attn` | softmax attention (TC⁰ control) | — | 398,976 |
| `mlp` | **no sequence mixing at all** (floor control) | — | 267,904 |

The four eigenvalue variants are **parameter-identical to the element**, share
every projection, and differ only in one activation and one constant. Init is
matched: the gate bias is set so `diag01`/`diagpm1` both start at `a ≈ 0.5` and
`delta01`/`deltapm1` both start at eigenvalue `≈ 0.5`. Only the *reachable range*
differs. The mandate's eigenvalue detail is honoured: `deltapm1` is `I − 2βkkᵀ`,
not `I − βkkᵀ`.

Recurrences run serially over the token axis in fp32 under bf16+amp (PLAN2 §5:
prefix products are where bf16 silently destroys a correct model). Both
directions use a row-wise reversal index that keeps trailing PADs in place, so
pad state never reaches a valid position and there is no CPU sync.

**Screening discipline** (BRIEF2 §6): e5 not e1; `--mode fixed_step` manifests
throughout, so nothing here is contaminated by the three siblings sharing the
GPU; `--lr 0` control run *before* interpreting anything; the metric **row**
reported, never a cell; `mlp` (no mixing) carried as a live collapse detector in
every table.

### 2.1 The `--lr 0` control (LEGAL)

All eight architectures, e5, 200 steps, lr = 0. With lr = 0 AdamW's decay term is
also zero, so this is exactly the random-init model.

> `train 0.000 · test 0.001 · ood 0.000 · rung-1 0.000 · mean 0.0004 · MAX_T 0`
> — **identical to four decimals for all eight architectures.**

They agree exactly because the tied head is the token embedding, which is
constructed first under a shared seed, so every untrained model emits the same
near-constant string. **This is the floor every number below must be read
against.** It also fixes the resolution: an e5 rung is 512 examples, so one
example is 0.002.

### 2.2 Is the instrument able to see an expressivity difference at all? (DIAGNOSTIC)

Reading an *ordering* off the real task is only meaningful if the same code
demonstrably separates on problems whose algebraic class is known. Synthetic
prefix-product word problems, sequence-exact accuracy, same scaffold, same
optimizer (`lab/p2_expressivity_check.py`):

| task (class) | len | steps | diag01 | diagpm1 | delta01 | deltapm1 | lstm | gru | attn | mlp |
|---|---|---|---|---|---|---|---|---|---|---|
| parity (`Z₂`) | 16 | 2k | 0.873 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.076 | 0.000 |
| parity (`Z₂`) | 32 | 2k | 0.916 | 1.000 | **0.018** | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| mod-3 (`Z₃`) | 16 | 2k | 0.959 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | 0.000 |
| mod-3 (`Z₃`) | 32 | 2k | 1.000 | 0.994 | 0.984 | 1.000 | 1.000 | 1.000 | 0.177 | 0.000 |
| A₅ (**non-solvable**) | 16 | 2k | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| A₅ (**non-solvable**) | 16 | **15k** | 0.000 | — | — | 0.000 | **0.980** | — | 0.000 | — |

Three things fall out, and all three matter for reading probe 4.

1. **The instrument has resolution at the eigenvalue detail.** `delta01` collapses
   to 0.018 on parity at length 32 while `deltapm1` — the same file with `2β`
   instead of `β` — reads 1.000. Grazzi et al.'s free-expressivity result
   reproduces exactly, so a flat probe-4 ordering is not an implementation bug.
2. **The instrument has resolution at the non-solvable end.** At 15,000 steps the
   LSTM solves the A₅ word problem (0.980 sequence-exact, 0.9972 token) and every
   linear recurrence and attention stays at chance. If the competition task
   needed a non-solvable monoid on the token axis, the LSTM would be the model
   that shows it. It is not.
3. **At the competition's sequence length the diagonal `[0,1]` vs `[-1,1]`
   distinction is largely washed out.** `diag01` reaches 0.873/0.916
   sequence-exact on parity — the classic counting escape: a diagonal decay near
   1 is a running *counter*, and a 2-layer MLP over a bounded counter computes
   mod 2 fine. The `TC⁰` impossibility results are asymptotic in length; at
   11–21 tokens they have essentially no force. **This weakens PLAN2 §0's
   argument independently of anything measured on the real task**: even if the
   task's monoid *were* non-solvable, sequences this short would not make
   expressivity binding.

---

## 3. Probe 4 — the solvability probe (the deliverable)

**LEGAL.** e5 (512-example rungs, sampled 10/11-bit moduli), `fixed_step` 1200
steps, seeds 74 / 7 / 13, matched compute, matched parameters, matched init.
`train` is the runner's own final training-batch exact accuracy; `test` is
held-out prompts; `r1` is depth rung T=1, i.e. `(N,x)` pairs reserved out of
train/test/ood entirely.

PLACEHOLDER_P4_TABLE

**The ordering is flat.** No pair of the four transition classes is separated by
more than the seed spread, and every one of them is inside the band occupied by
`mlp` — a model with **no sequence mixing whatsoever**, 43% fewer parameters,
which cannot represent any transition monoid at all.

That is the whole answer, and it is unambiguous because of the metric row:

* the classes differ enormously in *fitting* power (`train` 0.02–0.93);
* they do not differ at all in *generalisation* (`test`, `r1` all at the lr=0
  floor ± one or two examples);
* and the model that can fit nothing (`mlp`) generalises exactly as well as the
  model that can fit almost everything.

**PLAN2 §6, third kill criterion, fires.** Diagonal-`[0,1]` performs the same as
DeltaNet-`[-1,1]`. The task is not a token-axis state-tracking problem, and §0's
reframe is wrong.

### 3.1 What the flat ordering positively says

It is consistent with the only structure the token axis actually carries. Getting
`x² mod N` out of digit tokens needs multi-digit multiplication and a modular
reduction; the carry structure of both is the classic propagate/generate/kill
monoid, which is **aperiodic** — star-free, `AC⁰`, at the very bottom of Axis A.
Diagonal-`[0,1]` is already more than enough for it. There is no non-solvable
sub-monoid anywhere on this axis to reward extra expressivity, which is exactly
what the table shows.

So the legal substitute for inspecting the data returns: **the task's token-axis
transition monoid is solvable, and almost certainly aperiodic.** Nothing above
`diag[0,1]` on Axis A buys anything here.

---

## 4. Probe 1 — expressivity vs optimization

**LEGAL.** `plan2/sequential-rnn` ran this as a candidate and its numbers stand
(train_exact climbs 134× from `D_H`=4 to 256 while held-out never moves; a
40,000-step run below memorisation capacity went flat after 5,000). This branch
reproduces the control independently under matched conditions and adds the honest
tuning pass that PLAN2 §6 requires before a kill criterion may be called.

PLACEHOLDER_P1_TABLE

Bidirectional LSTM and GRU — maximal expressivity, `O(T)` serial, no theoretical
caveats, and the *only* models in the family that solve A₅ — get **exactly what
every v1 model got: nothing.** Held-out and rung-1 sit on the lr=0 floor at every
learning rate, at every fit level from 0.055 to 0.928 train exact accuracy, at
Easy and at Medium scale.

**PLAN2 §6, first kill criterion, fires.** Confirmed, and now with a matched
control family rather than a single model.

---

## 5. The harness / loss / label-alignment audit (what PLAN2 §6 asks for next)

Since two kill criteria fired, PLAN2 §6 directs: *"Stop and re-examine the
harness, the loss, and the label alignment before writing another architecture."*
Done, two independent ways, and it comes back clean.

### 5.1 A constructed oracle certifies the top of the metric (DIAGNOSTIC)

`lab/diagnostics/oracle_construct.py` — **never a submission**; it hard-codes the
arithmetic in the forward pass, which BRIEF §4.2 forbids outright. It decodes
`(N, x, T)` from the prompt using only the public format, applies integer modular
squaring `T` times, and writes the answer digits as logits at the last
`len(answer)` prompt positions. One learned scalar keeps the graph differentiable
so the evaluator's own `loss.backward()` / `optimizer.step()` loop runs unmodified.

| dataset | modulus | MAX_T | OOD_N MAX_T | every rung | test | ood | mean |
|---|---|---|---|---|---|---|---|
| e5 | sampled 10/11-bit | **64** | **64** | 1.000 | 1.000 | 1.000 | 1.0000 |
| m1 | fixed 10403 | **64** | **64** | 1.000 | 1.000 | 1.000 | 1.0000 |
| hp1 | fixed 4,028,033 (22-bit) | **64** | **64** | 1.000 | 1.000 | 1.000 | 1.0000 |
| n7_e250 | fixed 1,022,117 (7-digit) | **64** | n/a | 1.000 | 1.000 | 1.000 | 1.0000 |

Nobody in this project had run a positive control before. It settles that the
evaluator path, `collate_squaring_mod`'s `target_positions` slicing, the
exact-match rule, the certification-prefix logic and both depth ladders are all
sound, at Easy, Medium and Hard-proxy scale, on seen and unseen moduli — in 20
optimizer steps of wall clock. **There is no harness bug. The metric is
reachable. Every null in this project is a fact about learning.**

### 5.2 The copy/squaring isolation pair — a *legal* positive control (LEGAL)

`trapdoor_squaring_mod` computes `pow(x, pow(2,T,φ), N)`; at `T=0` the exponent is
1, so the answer is `x` itself (`data/squaring_mod.py:409-415` — from the source,
never from data). That gives a dataset with the **same** prompt format,
tokenizer, collate, `target_positions`, loss and evaluator, whose target is a
pure digit copy. Paired with a `T=1` dataset at the same modulus, same operand
distribution, same size, it isolates *the arithmetic* from everything else. Only
one T setting is present, so every prompt carries a distinct `x` and the `test`
split is an **unseen-operand** measurement.

| dataset | target | modulus | model | train | **test (unseen operands)** |
|---|---|---|---|---|---|
| `cp5` | `x` | 10403 (5 digits) | attn | 1.000 | **1.000** |
| `cp5` | `x` | 10403 | lstm | 1.000 | **1.000** |
| `cp5` | `x` | 10403 | deltapm1 | 1.000 | **1.000** |
| `sq5` | `x² mod N` | 10403 | attn | 0.988 | **0.000** |
| `sq5` | `x² mod N` | 10403 | lstm | 0.980 | **0.000** |
| `sq5` | `x² mod N` | 10403 | deltapm1 | 0.994 | **0.000** |
| `cp7` | `x` | 1,022,117 (7 digits) | attn | 1.000 | **1.000** |
| `cp7` | `x` | 1,022,117 | lstm | 1.000 | **1.000** |
| `sq7` | `x² mod N` | 1,022,117 | attn | 0.008 | **0.000** |
| `sq7` | `x² mod N` | 1,022,117 | lstm | 0.537 | **0.000** |

This is the sharpest statement of the problem the project has produced. **Copying
seven decimal digits of an unseen operand generalises perfectly. Squaring them
once generalises at exactly zero.** Parsing, slot alignment, digit readout, the
loss and the exact-match rule are all confirmed working *by a legal learned model
on unseen operands*, and the entire failure is localised to one squaring — which
is precisely what BRIEF2 §2 said was the only thing still open.

---

## 6. Probes 2 and 3 — length scaling and the train/eval transition

On this task sequence length **cannot** be varied independently of modulus size:
`max_seq_len = 2·digits(N) + 5`. So probes 2 and 3 are one 2-D grid over
(modulus size × training-set size), read along two axes. Ten fixed-modulus
datasets, full T=1…64 ladder, **256 units reserved** out of train/test/ood behind
every rung, so `r1` is always an unseen-operand measurement
(`lab/gen_p2_grid.sh`).

PLACEHOLDER_P23_TABLE

---

## 7. What each of the four questions answers to

PLACEHOLDER_ANSWERS

---

## 8. Corrections carried

* **`triton` imports but cannot compile on this box** (`ptxas … sm_107a is not
  defined`), and the same failure kills `torch.compile`. BRIEF2 §5 is wrong on
  this point (found by `plan2/sequential-rnn`). Custom chunkwise kernels are off
  the table locally. It does not change anything here — every recurrence in this
  branch is plain PyTorch.
* **BRIEF2 §5's `max_seq_len` figures are right but the mechanism is worth
  stating**: `max_seq_len = 2·digits(N_max) + 5`, where `N_max` is the largest
  modulus anywhere in the dataset *including the OOD-N depth ladder*. That is why
  e1 reads 13 rather than 11.
* **The ALU family's "training moves away from the discrete solution" pathology
  (RESUME Round 5) is architecture-specific, not universal.** Nothing in this
  eight-architecture family shows it: training moves `train_exact` from 0.000 to
  0.02–0.99 while held-out stays pinned at the lr=0 floor. Two different failure
  shapes, and they should not be conflated.

---

## 9. Reproduction

```bash
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd /home/scratch.arohan_hw/git/one-layer-deeper/.worktrees/p2-phase0

# architectures (8 arches, one template, matched params)
$VENV lab/make_p2arch.py --arch diag01 diagpm1 delta01 deltapm1 lstm gru attn mlp
$VENV lab/make_p2arch.py --arch diag01 diagpm1 delta01 deltapm1 lstm gru attn mlp --lr 0 --suffix lr0

# datasets (generator only; rows never read)
PATH=$(dirname $VENV):$PATH bash lab/gen_p2_grid.sh       # probes 2 and 3
PATH=$(dirname $VENV):$PATH bash lab/gen_p2_isolate.sh    # copy/squaring pair

# expressivity calibration (DIAGNOSTIC, synthetic)
$VENV lab/p2_expressivity_check.py --lengths 16 32 --steps 2000
$VENV lab/p2_expressivity_check.py --arch lstm deltapm1 diag01 attn --tasks a5 --steps 15000

# the run queues (all LEGAL, all --mode fixed_step)
bash lab/p2_queue.sh     # lr=0 controls, probe 4, probe 1
bash lab/p2_queue2.sh    # copy/squaring isolation, LSTM tuning pass
bash lab/p2_queue3.sh    # probes 2 and 3 grid
bash lab/p2_queue4.sh    # third seed, m1 tier, collapse detector

# the harness positive control (DIAGNOSTIC -- NEVER a submission)
$VENV lab/run_experiment.py --submission lab/diagnostics/oracle_construct.py \
  --manifest lab/manifests/p2_e5_fs20_s74.json --tag P0-harness-audit --note "..."

# tables
$VENV lab/p2_summarize.py --tag P4-solvability
```

Every run is archived in `lab/archive.jsonl`; synthetic calibration in
`lab/p2/expressivity_check.jsonl`; collapse detector in `lab/p2/diversity.jsonl`.

## 10. Compliance ledger

* Nothing under `data/generated/` was read, printed or summarized. Datasets were
  *written* by the public generator (`lab/gen_p2_grid.sh`,
  `lab/gen_p2_isolate.sh`); the only output consumed is the generator's own
  "wrote N examples" line.
* Every architectural claim is derived from the generator **source** and general
  reasoning; the `T=0 ⇒ answer = x` fact comes from
  `data/squaring_mod.py:409-415`.
* All eight architectures are LEGAL submissions: learned from random init, no
  hard-coded arithmetic, end-to-end differentiable, no custom training loop, no
  manifest override.
* `lab/diagnostics/oracle_construct.py` is DIAGNOSTIC and hard-codes the
  arithmetic. It lives outside `submissions/` and is labelled as unusable in its
  own docstring. It must never be submitted.
* `lab/p2_diversity.py` is DIAGNOSTIC: it uses its own training loop, which BRIEF
  §4.4 forbids inside a submission.
* Nothing was submitted to the hosted service.
