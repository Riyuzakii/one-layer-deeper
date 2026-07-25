# algebraic-closure — manufacturing supervision from the composition law

Branch `explore/algebraic-closure`. Metric is the post-2026-07-24 one: **MAX_T**
= the largest rung `T ∈ {1,2,4,8,16,32,64}` whose rung *and every lower rung* are
100 % exact. Every comparison below uses a `--mode fixed_step` manifest, so step
counts are identical across runs and immune to GPU contention (four agents shared
one B300 throughout). The only wall-clock run is the feasibility check in §8 and
it is labelled as such.

47 runs, all in `lab/archive.jsonl` under tags `CL*`.

---

## 0. TL;DR

* **MAX_T = 0 for every configuration**, on `e1` and on `e5`, at every
  composition weight, on random and on batch-derived synthetic operands, with
  and without operand dropout, at 3 000 and at 20 000 steps.
* **The composition law is imposable and the model does satisfy it.** The
  measured consistency divergence
  `KL( M(N,x,2) ‖ M(N, M(N,x,1), 1) )` falls from **5.45 nats** (no term) to
  **0.06 nats** at λ=3 — a 90× reduction. The mechanism works exactly as
  designed. It buys **no exactness**.
* **It is satisfied trivially on operands the labels cannot reach, and I
  measured the collapse.** With the constraint applied to *uniform random* digit
  operands the model's one-squaring map becomes the **identity** on them:
  identity rate **0.96 / 0.99 / 0.99** at λ_rand = 0.1 / 0.3 / 1.0 (chance 0.10).
  `s(x) = x` satisfies `s∘s = s²` perfectly and costs nothing on the labelled
  data, because random operands are distinguishable from real ones.
* **Using operands that are *not* distinguishable removes the collapse and
  changes nothing else.** Manufacturing operands by permuting the batch's own
  digit places across rows (same per-place digit statistics, no row's
  trajectory) gives identity rate 0.11–0.14 — no collapse — with rung-1 still
  0–1 of 38.
* **The headline methodological result: held-out cross-entropy is not a valid
  progress signal for this task.** The brief (and the `tied-recurrence` report)
  treat held-out CE against ln 17 = 2.833 as a sensitive proxy for progress. A
  **label-smoothing control** — a pure confidence regulariser with zero
  algebraic content — takes held-out CE from **7.61 → 2.93**, *below* every
  composition-law configuration and essentially at the uniform reference, with
  train accuracy still 0.94 and rung-1 still 1 of 38 (§5). CE measures how
  loudly the model asserts, not what it knows. **Reduced confident-wrongness is
  free and means nothing here.**
* **Operand dropout is the first lever in this repository that visibly breaks
  the memorisation fit** — training exact-match falls from 1.00 to 0.20 at p=0.5
  — and held-out accuracy does not move (§6). "Make the lookup unrepresentable"
  was the last standing recommendation from `exact-arithmetic` §5.1; on this
  evidence it is not sufficient either.
* **The one nominal positive did not replicate.** λ_compose = 1.0 beats the
  matched baseline on `e1` at 3 seeds (rung-1 1/3/1 vs 0/0/0 of 38; `test`
  8/8/12 vs 2/7/5 of 150; exact permutation p = 1/20, the floor of a 3-seed
  design). On `e5`, whose rungs are **512 examples** instead of 38, the same
  comparison is flat (§7): rung-1 {1,3,2} vs {1,2,3}, i.e. the same numbers.
* **A structural finding nobody had recorded: on `e1`, 96 % of the `test`
  split's operands are training operands seen at a different T** (§1.2, derived
  from the public generator recipe). `test` and `mean_exact_accuracy` on `e1`/`e2`
  measure *T-transfer*, not operand generalisation — and T-transfer is exactly
  what a composition-law loss buys. At 20 000 steps the term takes `test` from 6
  to **16** of 150 while leaving rung-1 at 1 of 38 (§8.3). That is the branch in
  one line: the law does exactly what it says, and what it says is not what
  certification needs.
* **Why it cannot work, stated once:** the semigroup law constrains
  `M(N,·,T)` to be `h^T` for a *single* map `h`. It says nothing about the value
  of `h` at any particular operand. It therefore removes the T-dependence degree
  of freedom (a 3× reduction in what has to be fitted on `e1`) and adds **zero
  information at unlabelled operands** — the residual freedom is the whole
  function space `{h : h agrees with squaring on the 200 training units}`, and
  gradient descent picks the cheapest member of it. §9.

---

## 1. Hypothesis

`answer(x, T) = s^T(x)` with `s(y) = y² mod N`. Hence for every `x` and every
`a, b ≥ 0`

```
s^(a+b)(x) = s^b( s^a(x) )            (the semigroup / flow law)
s^0(x)     = x                        (its a = 0 case)
```

Both hold on **every** operand, labelled or not. The claim under test: a model
can be forced to satisfy the law on operands it has no labels for — including
operands it manufactures itself — and a per-example lookup table cannot satisfy
a law it was never trained on, whereas a genuine algorithm satisfies it for
free. This is the one class of pressure that separates the two, and no sibling
branch had tested it.

The diagnosis this branch starts from (established by `exact-arithmetic` §3.2
and `tied-recurrence` §4.1, not re-derived here): every model reaches ~100 %
exact-match on the **training** set with ~zero loss by step 300 while held-out
sits at chance. The bottleneck is a memorisation/generalisation gap.

### 1.1 What the law can and cannot reach — worked out before running anything

For `e1` (`N = 323 = 17·19`, φ = 288, 250 units used by train/test/ood, **38
units held out** for the ladder; pure number theory on the public generator
parameters, no generated data opened):

* Squaring is **4-to-1** on `Z*_N`, so exactly **72 of 288** units are quadratic
  residues.
* The composition law applied at a *labelled* base point propagates supervision
  to `x²`, `x⁴`, `x⁸`, … — all of which are **squares**.
* The held-out rung cohort is a uniformly chosen subset of all units, so in
  expectation only `38·72/288 ≈ 9.5` of the 38 evaluation points are quadratic
  residues.

So **composition-manufactured supervision provably cannot reach ~3/4 of the
rung-1 cohort on a fixed-N Easy set.** That is `exact-arithmetic` §3.5 applied
to this hypothesis family, and it was known going in. The reason to run the
experiments anyway is that a *function-space* constraint need not act
pointwise: forcing the law everywhere could in principle select the algorithmic
solution by inductive bias rather than by pointwise determination. §9 reports
what actually happened.

### 1.2 On `e1`, the `test` split is not an operand hold-out — 96 % of it is
### T-transfer, and that is precisely what this branch's loss buys

This was not in either sibling report and it changes how several numbers in this
repository should be read. From the **public** recipe
(`scripts/generate_datasets.sh` lines 19–28: `--examples_per_setting 250
--ood_examples_per_setting 100 --depth_evaluation_exhaustive_x true`) and
`data/squaring_mod.py:_generate_prompt_grouped_records`:

```
units of N=323                 : phi(323) = 288
required_per_time_step         : max(250, 100) = 250
reserve_count                  : 288 - 250   = 38        <- the ladder cohort
units available to every T     : 288 - 38    = 250
prompts drawn per T setting    : 250
```

Each of `T ∈ {1,2,3}` (and the `ood` setting `T = 6`) therefore draws its 250
prompts from **exactly the same 250 units**, and since it needs all of them,
**every T setting uses every one of those 250 units exactly once**. The
train/test partition (200/50) is an independent draw per T. So for a `test`
prompt `(x, T=1)`:

```
P( x appears in the TRAIN split at T=2 or T=3 ) = 1 - (50/250)^2 = 0.96
```

**96 % of `e1`'s `test` operands are training operands seen at a different T.**
`test` on `e1` (and by the same argument `e2`, and `mean_exact_accuracy` which is
built from `test` and `ood`) measures *T-transfer at a known operand*, not
generalisation to an unseen operand. The 38-unit depth cohort — reserved before
anything else is drawn, appearing in no train/test/ood row — is the only clean
operand hold-out on the dataset.

That matters enormously for this branch specifically, because **T-transfer at a
known operand is exactly what the composition law provides**: it is the identity
`M(N,x,2) = M(N, M(N,x,1), 1)` and it links the three T settings of one `x`. So
a composition term should be expected to raise `test` on `e1` *whether or not it
teaches the model anything about squaring*, and §4 and §8.3 show that it does.
On `e5` the moduli and operands are sampled from a much larger space, the
per-setting overlap is far lower, and the effect disappears (§7). Any conclusion
drawn from `e1`'s `test` split or from `mean_exact_accuracy` needs this caveat.

---

## 2. What was built

`lab/make_closure.py` emits one submission per configuration. The architecture
is held **fixed** across every run; configurations differ only in the auxiliary
loss and in operand-pathway regularisation, so "does the composition law change
what is learned?" stays a one-variable question.

**Architecture (all arms).** A canonical LSD-first *slot* layout computed inside
`forward`: the prompt `[N] d(N) [X] d(x) [T] d(T)` is re-laid into
`places(N) ‖ places(operand) ‖ 3 slots for T`, each slot carrying one decimal
place, zero-padded in the high places. `places` is a build-time constant
`(max_seq_len − 3) // 2` derived from `ModelSpec` — no `.item()`, no
data-dependent shape. 4 untied transformer blocks (d=128, 4 heads, FF×4,
~800 k parameters), RMSNorm, a 10-way digit head read off the operand slots and
scattered back onto sequence position `input_len − 1 − j` for place `j`. AdamW
lr 1e-3, wd 0.1 (1-D tensors excluded), `batch_size = 64`.

Weight tying and iteration count are deliberately **not** used: `tied-recurrence`
§6.1 showed `h ← core(h + base)` with a pre-norm readout makes the iteration
count a gauge freedom. Using untied fixed depth removes that confound entirely —
the composition structure lives in the loss, not in the architecture.

**The auxiliary term.** Inside `forward`, in training mode only, with the
modulus slots held fixed, write `L(op, t)` for the model's digit-logit readout
given operand embedding `op` and a T-value `t` chosen by me (0, 1 or 2 — a
compile-time constant, not decoded from the prompt):

| flag | term | what it asserts |
|---|---|---|
| `--lam-compose λ` | `KL( L(x,2) ‖ L(reembed(L(x,1)), 1) )` | `s∘s = s²` at the batch's own operands |
| `--lam-rand λ` | the same at manufactured operands `r` | `s∘s = s²` where no label exists |
| `--lam-ident λ` | `CE( L(x,0), x )` | `s⁰ = id` |
| `--rand-mode` | `uniform` = i.i.d. random digits; `permute` = each place taken from a different, independently shuffled row of the batch | how distinguishable the manufactured operand is from a real one |
| `--straight-through` | hard argmax through the digit bottleneck | |
| `--op-dropout p` | drop whole decimal places of the operand embedding | anti-memorisation pathway control |
| `--label-smoothing e` | CE label smoothing | **control**: confidence without content |

`reembed` sends a readout back through the digit bottleneck as a soft (or
straight-through) mixture of the 10 digit embeddings, so an intermediate must be
expressible as a digit string to survive. The entire term is computed in
`forward` and returned through `aux`; `training_loss(logits[valid],
labels[valid], aux)` only adds it to the cross entropy. The evaluator still owns
`.backward()` and the single `optimizer.step()`.

**Structural verification.** `lab/test_closure.py` builds prompts with the
**public tokenizer** and asserts: (1) the slot layout is exactly the LSD-first
decimal expansion of `N`, `x` and `T` for a mixed batch of 1-, 2- and 3-digit
`x` and 1- and 2-digit `T`; (2) every evaluator-scored position is written by
the place scatter; (3) answer place `j` is scored at `input_len − 1 − j`;
(4) the auxiliary scalar is a differentiable 0-dim tensor and the loss
backprops to all 46 parameter tensors; (5) the auxiliary term is reproducible
given the RNG state; (6) `validate_submission_source` passes.
All six checks pass.

**Diagnostics (lab-only, `--diag-every`).** Every N training forwards the model
prints one line of purely model-internal telemetry:

| symbol | meaning | trivial-solution signature |
|---|---|---|
| `kl_c` | composition KL at the batch's operands | → 0 when the law holds |
| `kl_r` | composition KL at manufactured operands | → 0 when the law holds |
| `ent` | mean entropy of the one-squaring readout | → 0 for a *constant* map |
| `idm` | fraction of places where `argmax L(r,1) == r` | → 1 for the **identity** map on manufactured operands (chance 0.10) |
| `agr` | agreement between `argmax L(r,1)` and `argmax L(r,2)` | → 1 when T is ignored |
| `mode` | batch mode share of the predicted digit | → 1 for a constant map |
| `idx` | same as `idm` but at the batch's own operands | → 1 for identity on real operands |

`idm`/`ent`/`agr`/`mode` are computed on operands drawn from `torch.randint`,
i.e. they involve no dataset value at all. `idx` compares a model output to a
model *input*; no label is involved (the evaluator's own `accuracy=` print is
the same category of telemetry). The shipped submission sets `DIAG_EVERY = 0`.

### 2.1 Compliance

* No file under `data/generated/` was read, printed, sampled or summarised.
  Every structural claim comes from `data/squaring_mod.py`,
  `scripts/generate_datasets.sh`, `benchmark/runner.py` or elementary number
  theory on public generator parameters. `lab/test_closure.py` constructs its
  prompts by calling the public tokenizer.
* **No arithmetic on digit values anywhere.** `_slots` does index arithmetic on
  token *positions* (a `cumsum` over the public marker token ids plus
  gather/scatter); it never adds, multiplies or reduces a number. There is no
  modular-exponentiation routine, no digit-wise multiplication, no lookup table,
  no `torch.load`. Every prediction comes from parameters trained from random
  init in the run.
* **The law shapes the loss only.** Every auxiliary target is either the model's
  own output or the operand digits already present in the input. The step map is
  learned.
* **No data-dependent Python control flow.** Layer count is a constant; the T
  values {0,1,2} used by the auxiliary term are constants I chose, not decoded
  from the prompt; `places` is derived from `ModelSpec.max_seq_len`, which is
  shape metadata the evaluator hands the submission.
* End-to-end differentiable, no custom training loop, no participant-controlled
  backward, no manifest override in the submission.
* Nothing was submitted to the hosted service; `one-layer login`/`submit` were
  never run and no network call was made.
* Every run, including all negatives and the smoke run, is in
  `lab/archive.jsonl`.

---

## 3. Method

```bash
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python

$V lab/make_manifest.py --dataset e1 --mode fixed_step --max-steps 3000 --seeds 74
$V lab/make_closure.py  --tag c2_c10 --diag-every 200 --lam-compose 1.0
$V lab/run_experiment.py \
   --submission submissions/exp_closure/c2_c10/submission.py \
   --manifest lab/manifests/lab_e1_fs3000_s74.json \
   --tag CL2-seeds --note c2_c10
```

`lab/closure_par.sh <manifest-stem> <tag> <parallelism> <cfg>...` runs a grid;
`lab/closure_table.py <tag>...` renders any tag as the tables below.
`lab/run_experiment.py` was extended (backwards-compatibly) to archive
`split_loss` (per-split held-out CE) and `diag_curve` alongside the existing
fields.

Screening budget: `e1`, fixed 3 000 steps, `batch_size = 64` (the free 5.4×
step lever from `tied-recurrence` §3). `e1` rungs are 38 examples (grid
1/38 = 0.026); `e5` rungs are **512** examples (grid 1/512 = 0.002) — 13× finer.
`e1`'s `test` split is 150 examples, `e5`'s is 300.

**Every table reports `train_exact_accuracy` next to held-out**, per the brief's
first-line filter.

### 3.1 Configuration index

| cfg | λ_compose | λ_rand | λ_ident | rand mode | op dropout | label smoothing |
|---|---|---|---|---|---|---|
| `c0_base` | – | – | – | – | – | – |
| `c1_c03` | 0.3 | – | – | – | – | – |
| `c2_c10` | 1.0 | – | – | – | – | – |
| `c3_c30` | 3.0 | – | – | – | – | – |
| `c4_c10r10` | 1.0 | 1.0 | – | uniform | – | – |
| `c5_r10` | – | 1.0 | – | uniform | – | – |
| `c6_i10` | – | – | 1.0 | – | – | – |
| `c7_all` | 1.0 | 1.0 | 1.0 | uniform | – | – |
| `c8_r01` / `c9_r03` | – | 0.1 / 0.3 | – | uniform | – | – |
| `c10_rp10` | – | 1.0 | – | **permute** | – | – |
| `c11_c10rp10` | 1.0 | 1.0 | – | **permute** | – | – |
| `d1_drop02` / `d2_drop05` | – | – | – | – | 0.2 / 0.5 | – |
| `d3_drop02_c10` | 1.0 | – | – | – | 0.2 | – |
| `d4_dig4` / `d5_dig4_c10` | – / 1.0 | – | – | – | – | – (digit embedding width 4 instead of 128) |
| `e0_ls01` / `e1_ls03` | – | – | – | – | – | 0.1 / 0.3 |
| `f1_best` | 1.0 | 1.0 | – | permute | 0.2 | – |

---

## 4. Does the composition term change what is learned? (`e1`, 3 seeds)

`lab_e1_fs3000_s{74,175,276}`, tag `CL2-seeds` / `CL5-control`. Rung-1 is
reported as **correct examples out of 38** and `test` as **correct out of 150**,
because that is the only honest resolution at this cohort size.

| config | rung-1 /38 (3 seeds) | `test` /150 (3 seeds) | held-out CE (3 seeds) | **train exact acc** | MAX_T |
|---|---|---|---|---|---|
| `c0_base` (λ = 0) | 0, 0, 0 | 2, 7, 5 | 7.62, 7.42, 7.89 | **0.98, 1.00, 1.00** | 0 |
| `c2_c10` (λ = 1) | **1, 3, 1** | **8, 8, 12** | 5.69, 5.81, 6.17 | **0.97, 1.00, 1.00** | 0 |
| `c3_c30` (λ = 3) | 1, 0, 1 | 2, 4, 8 | **4.73, 4.88, 4.50** | 0.84, 0.95, 0.98 | 0 |
| `e0_ls01` (label smoothing 0.1, **control**) | 1, 1, 0 | 7, 7, 5 | **3.53, 3.39, 3.48** | 1.00, 0.98, 1.00 | 0 |

Reference: uniform over the 17-token vocabulary is `ln 17 = 2.833`.

**Reading 1 — the first-line filter says this lever is on a saturated
objective.** Train exact-match is 0.97–1.00 at step 3 000 for every composition
weight up to λ=1, exactly as it is for the baseline and for every configuration
the two sibling branches tried. Only λ=3 dents it (0.84–0.98), and λ=3 is the
weight with the *worst* `test` accuracy in the table. The composition term does
not make memorisation harder; it is an extra term on top of an objective the
model has already solved.

**Reading 2 — held-out CE moves a lot and means nothing.** CE falls
monotonically with λ (7.6 → 5.9 → 4.7) with non-overlapping 3-seed ranges. That
is a real, replicated effect. It is also *smaller* than what a label-smoothing
control achieves (3.4–3.5 at ε=0.1, and 2.93 at ε=0.3 — §5), and the control has
no algebraic content whatsoever. §5 makes this the report's main methodological
claim.

**Reading 3 — the one nominal positive.** `c2_c10` beats the matched baseline
on both held-out measures with non-overlapping ranges:

* rung-1: {1,3,1} vs {0,0,0}. Exact one-sided permutation test over all
  C(6,3)=20 arrangements: **p = 1/20 = 0.05**, which is the *smallest p a
  3-seed design can produce*.
* `test`: {8,8,12} vs {2,7,5}, same test, **p = 1/20 = 0.05**.

And it separates from the label-smoothing control on `test` ({8,8,12} vs
{7,7,5}, p = 1/20) but **not** on rung-1 ({1,3,1} vs {1,1,0}, p = 6/20 = 0.30).
The magnitude is 1–3 correct examples of 38 against the 38/38 certification
requires. §7 tests whether it survives a 13× finer cohort. It does not.

**And the `test` half of that positive is explained without any algebra.** By
§1.2, 96 % of `e1`'s `test` operands are training operands at a different T, and
linking a fixed operand's answers across T is the literal content of the
composition law. A `test` gain on `e1` is therefore the *expected* consequence of
the term working as advertised, and carries no information about unseen
operands. Only the rung column speaks to the metric, and it moves by one to three
examples of 38.

---

## 5. The control that reframes held-out CE

This was run because §4's CE effect looked too good, and it is the most
transferable result on the branch.

```bash
$V lab/make_closure.py --tag e0_ls01 --diag-every 200 --label-smoothing 0.1
$V lab/make_closure.py --tag e1_ls03 --diag-every 200 --label-smoothing 0.3
```

`e1`, fixed 3 000 steps:

| config | held-out CE | train exact acc | rung-1 /38 | `test` /150 |
|---|---|---|---|---|
| `c0_base` | 7.62 | 0.98 | 0 | 2 |
| `c3_c30` (strongest composition term) | 4.73 | 0.84 | 1 | 2 |
| `d2_drop05` (operand dropout 0.5) | 4.21 | 0.20 | 1 | 5 |
| **`e0_ls01`** (label smoothing 0.1) | **3.53** | 1.00 | 1 | 7 |
| **`e1_ls03`** (label smoothing 0.3) | **2.93** | 0.94 | 1 | 6 |
| *uniform reference* `ln 17` | *2.833* | – | – | – |

**A pure confidence regulariser drives held-out CE from 7.6 to 2.93 — to the
uniform reference — while training exact-match stays at 0.94 and rung-1 stays at
1 of 38.** Held-out CE on this task is dominated by *how confidently* the model
asserts its memorised extrapolation, not by how much of the function it has
learned. Any intervention that caps output confidence buys most of the available
CE, and no intervention observed anywhere in this repository converts CE into
exactness.

Consequences for the team, stated plainly:

1. **"Held-out CE vs ln 17 is a more sensitive progress signal than rung
   accuracy" (my brief) is wrong**, and should not be used to rank ideas.
2. `tied-recurrence` §4.5 reports `reembed_st` reducing held-out CE from 5.88 to
   4.087 and calls it "the only mechanism tested that meaningfully reduces
   confident wrongness". That reduction is real but is **within reach of label
   smoothing**, so it is not evidence that on-manifold state does anything
   algebraic. (That branch drew the right conclusion anyway — "aimed at the wrong
   bottleneck" — but the intermediate result is weaker than it looks.)
3. The only signals worth ranking on are **train-vs-held-out exact accuracy** and
   the rung profile.

---

## 6. Anti-memorisation: bottlenecking the operand pathway

`exact-arithmetic` §5.1 ends with "architectures where the lookup is not
expressible is the least explored and most likely place a win is hiding", and
notes that cutting *parameters* 14× did not stop memorisation. Two pathway
interventions, `e1`, fixed 3 000 steps, seed 74 (tag `CL4-pathway`):

| config | train exact acc | rung-1 /38 | `test` /150 | held-out CE | `kl_c` |
|---|---|---|---|---|---|
| `c0_base` | 0.98 | 0 | 2 | 7.62 | 5.45 |
| `d4_dig4` (digit embedding 128 → **4**) | 0.98 | 1 | 3 | 7.91 | 5.60 |
| `d1_drop02` (drop 20 % of operand places) | **0.55** | 1 | 6 | 6.08 | 2.65 |
| `d2_drop05` (drop 50 % of operand places) | **0.20** | 1 | 5 | 4.21 | 0.50 |
| `d3_drop02_c10` (dropout 0.2 + λ_compose 1) | **0.61** | **2** | 7 | 4.84 | 0.56 |
| `f1_best` (dropout 0.2 + compose 1 + permute-rand 1) | **0.55** | 0 | 8 | 4.30 | 0.59 |

**Narrowing the input channel does nothing** — a 4-dimensional digit embedding
memorises `e1` exactly as fast as a 128-dimensional one (0.98 train accuracy,
`kl_c` unchanged at 5.6). That is the expected result: 4 places × log₂10 ≈ 13
bits is already more than enough to name one of 288 operands, so "lookup
capacity" is not in the embedding width. It confirms `exact-arithmetic` §3.7 in
a second, independent way: **parameter count and channel width are not the same
thing as lookup capacity, and neither is the binding variable.**

**Operand dropout is the first lever in this repository that visibly breaks the
memorisation fit.** Training exact-match falls from 1.00 to 0.55 (p=0.2) and to
0.20 (p=0.5). Held-out exact accuracy does not move: rung-1 is 1 of 38 in every
case, `test` is 5–8 of 150 against a baseline 2–7 of 150.

*Caveat, stated because it matters:* the runner's `accuracy=` is measured on the
training forward, so it is measured **with dropout active**, and part of the
fall is the noise itself. A memoriser that needs all `P = 4` places intact would
score `0.8⁴ = 0.41` at p=0.2 and `0.5⁴ = 0.063` at p=0.5; observed values are
0.55 and 0.20, i.e. *more* robust than that bound, so the fit is genuinely
partly distributed rather than purely destroyed. The unambiguous numbers are the
dropout-free held-out ones, and they are flat. Making the lookup harder to
represent does not start generalisation.

---

## 7. Confirmation on a 512-example rung (`e5`)

`e1` rungs are 38 examples and the seed-only floor is one example, so §4's
p = 0.05 needs a finer cohort. `e5` is the sampled-modulus Easy set with T ∈
{1,2,3} in training and **512-example rungs** (two modulus sizes × 256), a 13×
finer grid, and a 300-example `test` split.

`lab_e5_fs3000_s{74,175,276}`, tags `CL6-e5` / `CL9-e5conf`:

**The matched 3-seed comparison** (`c0_base` vs `c2_c10`, the configuration that
showed the `e1` effect), rung-1 as **correct of 512**, `test` as correct of 300:

| config | rung-1 /512 (s74, s175, s276) | `test` /300 | held-out CE | **train exact acc** | MAX_T |
|---|---|---|---|---|---|
| `c0_base` (λ = 0) | 1, 2, 3 | 2, 2, 1 | 4.58, 4.74, 4.54 | 0.41, 0.41, 0.45 | 0 |
| `c2_c10` (λ = 1) | 1, 3, 2 | 1, 2, 2 | 4.03, 3.99, 3.48 | 0.27, 0.33, 0.33 | 0 |

The two rung-1 distributions are the same set of numbers. The `e1` advantage is
gone. Note also that the composition term *slows* the training fit on `e5`
(0.27–0.33 vs 0.41–0.45) without buying anything held out.

**Single-seed screen of the other arms** (seed 74, tag `CL6-e5`):

| config | rung-1 /512 | `test` /300 | held-out CE | train exact acc | `kl_c` | `idm` | MAX_T |
|---|---|---|---|---|---|---|---|
| `c0_base` | 1 | 2 | 4.58 | 0.41 | 3.16 | 0.10 | 0 |
| `c2_c10` (λ_compose 1) | 1 | 1 | 4.03 | 0.27 | 0.29 | 0.36 | 0 |
| `c3_c30` (λ_compose 3) | 1 | 1 | 3.32 | 0.28 | 0.05 | 0.30 | 0 |
| `c11_c10rp10` (compose + permute-rand) | 1 | 2 | 3.99 | 0.31 | 0.11 | 0.07 | 0 |
| `d3_drop02_c10` (dropout 0.2 + compose) | 1 | 1 | 2.45 | 0.02 | 0.13 | 0.18 | 0 |
| `e1_ls03` (label smoothing 0.3, **control**) | 1 | 2 | **2.59** | 0.05 | 0.51 | 0.08 | 0 |

**Nothing separates.** Every configuration lands at 1–3 correct of 512 at rung 1
— the same value the untreated baseline reaches, and the same 0.2–0.6 % band the
previous session and both sibling branches report everywhere. The
λ_compose = 1.0 advantage measured on `e1` at p = 1/20 does not appear at 13× the
resolution: the two 3-seed rung-1 vectors are literally permutations of one
another. Held-out CE again falls with the composition weight and falls further
with label smoothing, reproducing §5 on a second dataset.

`e5` also runs a different fitting regime — 3 000 steps at batch 64 leaves train
exact-match at 0.41 (baseline) rather than 1.00 — so this is not a
"memorisation-saturated only" result: the composition term does nothing whether
the model has finished memorising or not.

**This is the check that decides the branch.** Per the brief's own instruction —
confirm anything promising on a 256-example rung before believing it — the one
nominal positive in §4 is not believed.

---

## 8. Everything else that was measured

### 8.1 The trivial-satisfaction check — the core result of the branch

`e1`, fixed 3 000 steps, seed 74 (tags `CL1-lambda`, `CL3-trivial`). `idm` is
the identity rate of the one-squaring readout on **manufactured** operands
(chance 0.10); `kl_r` is the composition KL at those operands.

| config | `kl_r` | **`idm`** | `agr` | `ent` | `idx` (real operands) | train acc | rung-1 /38 |
|---|---|---|---|---|---|---|---|
| `c0_base` (no term) | 7.27 | 0.09 | 0.36 | 0.32 | 0.13 | 0.98 | 0 |
| `c8_r01` (λ_rand 0.1, uniform) | 0.09 | **0.96** | 0.88 | 0.60 | 0.13 | 1.00 | 0 |
| `c9_r03` (λ_rand 0.3, uniform) | 0.02 | **0.99** | 0.98 | 0.30 | 0.24 | 0.98 | 0 |
| `c5_r10` (λ_rand 1.0, uniform) | 0.02 | **0.99** | 0.96 | 0.25 | – | 1.00 | 1 |
| `c7_all` (compose+rand+ident, uniform) | 0.12 | **0.73** | 0.87 | 0.84 | – | 0.95 | 0 |
| `c10_rp10` (λ_rand 1.0, **permute**) | 0.14 | 0.14 | 0.63 | 0.74 | 0.12 | 0.98 | 1 |
| `c11_c10rp10` (compose 1 + rand 1, **permute**) | 0.09 | 0.11 | 0.65 | 1.12 | 0.11 | 0.95 | 0 |

**The collapse is real, complete, and immediate.** Applying the composition law
to uniform-random digit operands drives the one-squaring map to the **identity**
on them (`idm` 0.96–0.99 against chance 0.10) at *every* weight tested, down to
λ_rand = 0.1. `s(x) = x` satisfies `s∘s = s²` exactly, so `kl_r` reaches 0.02 —
the constraint is *perfectly* satisfied and carries no information. Meanwhile
`idx` stays at ~0.1–0.24, i.e. the model is **not** the identity on real
operands, and train accuracy stays at 1.00: the network has simply partitioned
its domain, behaving as a memorised map where the labels are and as the identity
where only the consistency constraint is. This is precisely the failure mode the
brief asked me to measure rather than assume away.

It is *not* a constant collapse: `ent` stays at 0.25–1.12 nats and the batch
mode share stays at 0.15–0.43, so the model has not degenerated to emitting one
digit.

**Making the manufactured operands indistinguishable removes the collapse and
adds nothing.** `--rand-mode permute` builds operands from the batch's own digit
places, independently shuffled across rows: the same per-place digit statistics
as a real operand, on no row's trajectory. `idm` then sits at 0.11–0.14, i.e. no
identity collapse, and `kl_r` still reaches 0.09–0.14, i.e. the law is still
satisfied — by a non-trivial map. Rung-1 is 0–1 of 38, `test` 7 of 150. So the
law can be satisfied non-trivially, and satisfying it non-trivially buys
nothing.

### 8.2 The `s⁰ = id` law on its own

`c6_i10` (λ_ident = 1.0, no composition term): `kl_c` = 5.75 (unchanged from the
baseline's 5.45), train accuracy 0.98, rung-1 0 of 38, held-out CE 7.79. Teaching
the model that `T = 0` means "copy the operand" is learned easily and transfers
nothing to `T = 1`. Combined with the composition term (`c7_all`) it neither
helps nor prevents the identity collapse (`idm` 0.73).

### 8.3 Training 6.7× longer

`lab_e1_fs20000_s74`, tag `CL7-long`:

| config | train acc @20 k | rung-1 /38 | `test` /150 | held-out CE @3 k → @20 k | `kl_c` |
|---|---|---|---|---|---|
| `c0_base` | 1.00 | 1 | 6 | 7.62 → 7.61 | 6.57 |
| `c2_c10` | 1.00 | 1 | **16** | 5.69 → 5.76 | 0.05 |

**No grokking transition with the composition term either.** 20 000 steps at
batch 64 (~2 100 epochs over `e1`'s ~600 rows) leaves both models at train 1.00,
rung-1 at **one example of 38** for both, and held-out CE where it was at step
3 000. The composition KL is driven to 0.05 and stays there. This is a third
independent confirmation of `exact-arithmetic` §3.8 and `tied-recurrence` §4.2,
now with the algebraic constraint in place.

The one striking number is `c2_c10`'s `test` accuracy: **16 of 150 (10.7 %)
against the baseline's 6**, the highest `test` accuracy anywhere on this branch,
with rung-1 unchanged. **That is exactly the split §1.2 predicts the composition
law should move, and exactly the split that does not matter.** `test` on `e1` is
96 % T-transfer; the rung cohort is the operand hold-out; the term moves the
former by 2.7× and the latter not at all. It is the cleanest single
demonstration on the branch that the law does precisely what it says and that
what it says is not what certification needs.

### 8.4 Feasibility (deliberate wall-clock run, contended GPU)

`lab_e1_wc_s74` (the real Easy 60 s budget), shipped submission, tag `CL8-ship`:
**1 522 optimizer steps in 60.0 s** on a heavily contended shared GPU, and all
16 evaluation splits — `test`, `ood`, 7 seen-N rungs, 7 OOD-N rungs — completed
inside the 30 s evaluation window with no `not_completed` rung (evaluation took
3.4 s). Model state **795 786 elements** against the 500 M ceiling (0.16 %).

Cost of the auxiliary term: 3 000 fixed steps on `e1` take **76.9 s** with no
auxiliary term (`c0_nodiag`) and **148.2 s** with `LAM_COMPOSE = 1.0`, i.e.
**≈1.9×**, which is the price of the three extra core evaluations per training
step. Evaluation cost is *identical* to the plain baseline, because the
auxiliary term is skipped in `eval()` mode. Both timings are B300-and-contention
local; the ratio is the transferable part.

One useful side check: `c0_nodiag` (no auxiliary term, no diagnostics)
reproduces `c0_base` (no auxiliary term, diagnostics on) **bit for bit** —
identical rung profile, `test` CE 7.615, `ood` CE 8.447. The diagnostic block is
provably inert, so every arm in this report is compared against a genuinely
matched baseline.

---

## 9. What is falsified, and why

**The hypothesis is falsified, and the reason is structural rather than
empirical.**

1. **Composition-law self-consistency does not close the memorisation gap.**
   The constraint is imposable (KL 5.45 → 0.06, a 90× reduction) and it leaves
   train exact-match at 0.97–1.00 and rung-1 at 0–3 of 38. MAX_T = 0.
2. **On operands the labels cannot reach, it is satisfied by the identity.**
   Measured, not assumed: `idm` 0.99 at every λ_rand from 0.1 to 1.0 (§8.1).
3. **Removing the collapse does not help.** Manufacturing indistinguishable
   operands gives a non-trivial consistent map with the same held-out numbers
   (§8.1).
4. **The `s⁰ = id` law adds nothing** (§8.2).
5. **Anti-memorisation capacity control is not sufficient either.** Operand
   dropout breaks the training fit (1.00 → 0.20) and held-out accuracy is flat
   (§6). Narrowing the operand channel 32× does not even break the fit.
6. **Held-out CE is not a progress signal** — a label-smoothing control beats
   every algebraic configuration on it (§5).

### 9.1 The one-line reason

The semigroup law constrains the model's three-argument map `M(N, x, T)` to be
`h^T(x)` for a **single** map `h`. It says nothing about the value of `h` at any
particular operand. So it collapses a family of T-indexed maps into one map — on
`e1` that is a 3× reduction in what has to be fitted — and adds **zero
information at unlabelled operands**. The solution set after imposing the law
*and* the labels is

```
{ h : Z_N → Z_N  |  h(x) = x² mod N for the ~200 training units }
```

which still has `|Z_N|^(#unseen)` members. Every one of them satisfies the law
exactly. Gradient descent picks the cheapest — the identity where it can get
away with it, an arbitrary consistent map where it cannot. Consistency
constrains the *shape* of the hypothesis, never its *values*, and generalisation
in this task is entirely a question of values.

That is a statement about the law, not about my implementation, and it is why I
believe this closes the class rather than one instance of it. It also explains,
retrospectively, why `tied-recurrence`'s weight-tying argument (§1.2 of that
report: "composition covers the domain") did not deliver: weight tying is the
architectural version of exactly the same constraint, and it carries exactly the
same amount of information.

### 9.2 What that means for the competition

Combining this with the two sibling reports, the following have now been
measured and none of them is the binding constraint: input/output representation
(8 variants), depth, iteration count including an ideal-halting upper bound,
on-manifold state, error propagation, parameter count (14× down), data volume
(12× up), weight decay 10× at 20 000 steps, operand-channel width (32× down),
operand dropout, and the composition law in five forms. The binding constraint
is that ~3/4 of every fixed-N Easy rung cohort consists of operands that are
**neither labelled nor reachable by any structural constraint available to a
submission**, and the model must nevertheless be exact on all of them
simultaneously.

**My read: MAX_T ≥ 1 is not reachable by architecture or loss search inside this
budget.** The only remaining lever with a mechanism that could in principle
supply values at unreachable operands is an inductive bias that makes the *true*
solution the shortest description — the Fourier/rotation family the BRIEF §2
names — combined with an optimisation regime that actually finds it. That is
`grok-optimization`'s axis, and on the evidence of three branches it needs a
qualitatively different optimiser or initialisation story, not another loss term.

---

## 10. What I would run next, in priority order

1. **Stop using held-out CE to rank anything** (§5). Rank on train-vs-held-out
   exact accuracy and on the rung profile only. This is free and it invalidates
   at least one intermediate conclusion already in the repo.
2. **If anyone continues on losses, the only version with a mechanism left is a
   constraint whose solution set does not contain the identity.** Concretely:
   impose the law simultaneously at `T = 1` and at a *labelled* higher `T` on
   the same manufactured operand, so that the operand's own orbit is anchored.
   I did not build this — it needs manufactured operands whose label is derivable
   from the batch, which the composition law alone does not give you. If it
   cannot be built, that is itself the proof that the class is closed.
3. **Do not screen anything on `e1` again.** 38-example rungs turned a
   p = 1/20 effect into a non-effect on the first 13× finer cohort (§7). `e5`
   costs the same wall clock.
4. **`--rand-mode permute` is worth keeping as a general tool** for anyone who
   needs in-distribution synthetic operands: it is differentiable, needs no data
   inspection, and it demonstrably avoids the off-distribution shortcut that
   uniform-random operands open.

---

## 11. The submission

`submissions/algebraic-closure/submission.py` — the slot-layout model with
`LAM_COMPOSE = 1.0`, diagnostics off, `batch_size = 64`, everything else as
§2. Generated by
`lab/make_closure.py --tag algebraic-closure --lam-compose 1.0 --diag-every 0 --batch-size 64`.

**It does not beat the baseline on the metric. MAX_T = 0**, like every other
configuration in this report and like both sibling branches' best. I ship it
rather than the plain baseline for reasons that are measured, and with the
caveat stated:

* it is the branch's hypothesis in its strongest **non-collapsing** form —
  λ_compose = 1.0 imposes the law at the batch's own operands, where the
  identity solution is excluded by the labels (`idx` ≈ 0.12, chance);
* it is the only configuration with a replicated nominal held-out advantage over
  its matched baseline on `e1` (3 seeds, exact permutation p = 1/20, §4) —
  **which does not replicate on `e5`'s 512-example rungs (§7)**, so it should not
  be believed;
* the auxiliary term is training-only, so evaluation cost is identical to the
  baseline and the 16-split evaluation fits the Easy budget with margin (§8.4);
* the place-exact readout does not shift when `x` or `T` gains a digit, which
  happens at rungs 16/32/64.

Checks run: `lab/test_closure.py` (6 structural assertions incl. the source
lint), one fixed-step run and one deliberate wall-clock run through
`lab/run_experiment.py` (tag `CL8-ship`), which exercises
`benchmark/validation.py:lint_submission_source`.

---

## 12. Coverage and budget

47 archived runs (`CL*` tags) against a briefed budget of 25–35. The overrun is
entirely §5 (the label-smoothing control, 5 runs), §7 (the `e5` confirmation of
the one nominal positive, 10 runs), §8.3 (the 20 000-step check, 2 runs) and
§8.4 (3 feasibility/shipping runs) —
i.e. controls and replication rather than new configurations. I judged all three
necessary: without §5 this report would have claimed a large held-out CE
improvement as evidence of algebraic learning, and without §7 it would have
claimed a p = 0.05 rung effect that is not there.

Not run, and worth recording: the `tp1` (N = 143) proxy, the straight-through
variant of the digit bottleneck, and the symmetric (non-stop-gradient) form of
the composition KL. Given §8.1 — the constraint is *already* satisfied to 0.02
nats and the collapse is measured — the expected value of all three is low; they
change how well the law is enforced, and the law being enforced is not the
problem.
