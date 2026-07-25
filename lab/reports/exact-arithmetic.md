# exact-arithmetic — input/output representation for exact digit arithmetic

Branch `explore/exact-arithmetic`. Metric: **Max T** (largest rung whose own and
every lower rung's exact accuracy is 100%). Everything below reports **per-rung
exact accuracy**, because MAX_T is 0 almost everywhere and rung-1 is the real
progress signal.

## 1. Hypothesis

A plain transformer over the flat digit string cannot align digit *places*
between `N`, `x` and the answer, which is what carrying and modular reduction
require. The claim under test is that the ~1–5 % plateau found by the previous
session is a **representation** failure, not a capacity/optimisation failure.

### 1.1 The structural fact the family rests on (verified, not assumed)

`lab/test_repr.py::check_target_alignment` builds prompts with the **public
tokenizer** (`data/squaring_mod.tokenize_squaring_mod_with_result`, no generated
data touched) and checks the evaluator's own `target_positions`:

```
answer digit of place value 10^j  is read at sequence position  (input_len - 1 - j)
```

i.e. *distance-from-end j == answer place value j*, exactly. Meanwhile the
prompt is `[N] d(N) [X] d(x) [T] d(T)` with **no leading zeros**, so
`input_len` varies with the digit count of `x` (on `e1`: 8, 9 or 10 tokens) and
of `T` (1 or 2 digits — the ladder goes to T=64). Consequences for a model with
a single absolute-from-left position embedding:

* the position that carries answer place 0 is 9 for a 3-digit `x` and 8 for a
  2-digit `x` — the readout mapping is **not a function of absolute position**;
* `x`'s own place values sit at different absolute positions per example;
* place k of `x` and place k of the answer share no index at all.

That is a concrete, verifiable misalignment, and it is what the axes below fix.

### 1.2 What the public dataset recipes say about the T=1 rung

`scripts/generate_datasets.sh` (public, checked in) is the exact command list for
`e1`–`e5` / `m1`–`m5`. Two facts from it reframe the milestone, and neither is
in `lab/findings.md`:

| dataset | modulus | training T | depth cohort | is T=1 in training? |
|---------|---------|-----------|--------------|---------------------|
| e1 | fixed 323 | 1,2,3 | exhaustive held-out x → **38** | yes |
| e2 | fixed 899 | 1,2,4 | exhaustive held-out x → **40** | yes |
| e3 | sampled 10/11-bit | **2 only** | 256 | **no** |
| e4 | sampled 11/12-bit | **2 only** | 256 | **no** |
| e5 | sampled 10/11-bit | 1,2,3 | 256 | yes |
| m1,m2 | fixed 10403 / 38021 | 4,8,16 | 192 / 768 | **no** |
| m3 | sampled 11/13/15-bit | **2 only** | 256 | **no** |
| m4 | sampled 14/18/22-bit | **8 only** | 256 | **no** |
| m5 | sampled 12/14/16-bit | 2,4,8 | 256 | **no** |

1. **On every Medium dataset and on e3/e4, rung T=1 is out of distribution in
   T.** `max_certified_time_steps` is a *prefix* from T=1 up, so on those tiers
   MAX_T ≥ 1 is impossible for any model that memorises a T-conditioned map; it
   requires a model that genuinely applies its squaring step T times. That is a
   representation-independent structural constraint on the whole competition and
   it is the strongest argument yet for weight-tied, T-conditioned iteration.
2. **e1's rung is only 38 examples, and the accuracy scale there is 1/38 =
   0.026.** Everything the previous session measured on e1 (0.013–0.079) is
   0–3 correct examples. As a place to detect a representation effect, e1 has
   almost no signal — which is why the runs below also use `e5` (256-example
   rungs, T=1 in training) and two purpose-built probes with 256-example rungs.
3. e1's `--depth_evaluation_exhaustive_x true` means the depth cohort is
   *literally every unit of 323 not used elsewhere*: φ(323)=288, 250 go to
   train/test/ood, **38 are held out**. Certifying T=1 on e1 therefore means
   getting x² mod 323 right on 38 units whose value the training set never
   constrains except through composition. This is exact algorithmic
   generalisation with no partial credit, from 200 T=1 rows.

## 2. What was built

`lab/repr_template.py` + `lab/make_repr_submission.py` generate one submission
per configuration, differing in exactly one flag. Shared across every run:
weight-tied recurrent block, `D_MODEL=128`, 4 heads, `NUM_LOOPS=8`, FF mult 4,
AdamW lr 1e-3 wd 0.1, plain CE, manifest batch 512 — i.e. the previous
session's reference model, changed **only** in how tokens are embedded and
where the answer is read from.

| flag | meaning |
|------|---------|
| `USE_ABS` | learned absolute-from-left position embedding (baseline) |
| `USE_FIELD` | learned field id: digit-of-N / digit-of-x / digit-of-T / the marker token itself |
| `USE_PLACE` | learned place-value index **within its own field**, LSD = 0, one table shared by all fields (abacus / index-hint) |
| `USE_RPOS` | learned distance-from-end index — by §1.1 this *is* the answer's place value |
| `LAYOUT=slots_sep` | digits re-laid-out inside `forward` into place-aligned slots (N places, x places, T digits), LSD-first, answer read off the x-aligned slot of the same place |
| `LAYOUT=slots_sum` | one token per place value; N-digit and x-digit embeddings summed into that slot |

Slot layouts do their re-ordering with gathers inside `forward` and scatter the
per-place logits back onto the evaluator's own positions, so the
`(logits[B,L,17], aux)` contract is unchanged and the path stays differentiable
(`lab/test_repr.py` asserts every scored position is written by the scatter and
that the loss backprops to every parameter).

**Compliance note.** The field segmentation is a `cumsum` over the public marker
tokens and the place index is arithmetic on positions; no arithmetic on the
*numbers* is performed anywhere, there is no lookup table of answers, no Python
control flow is switched on data values (loop count is a constant), and every
prediction comes from parameters trained from random init in the run.

## 3. Results

### 3.0 Variance floor first (the previous session was burned by not doing this)

Same submission bytes, same fixed-step manifest, three model-init seeds
(74 / 175 / 276) on `e1`:

| config | rung T=1 per seed | mean_exact_accuracy per seed | σ(mean) |
|--------|-------------------|------------------------------|--------:|
| r0_base (flat) | 0.026, 0.000, 0.000 | 0.0250, 0.0333, 0.0450 | 0.008 |
| r6_slotsep (place-aligned) | 0.000, 0.000, 0.000 | 0.0433, 0.0100, 0.0300 | 0.014 |

**The floor: rung-1 for a fixed configuration moves by 1 example (0.026) purely
from the seed, and `mean_exact_accuracy` has σ ≈ 0.01.** The full spread of
rung-1 across all eight representations at a fixed seed is 0.000–0.053, i.e.
two examples. Nothing in §3.1 clears the floor, and I will not claim otherwise.

### 3.1 The e1 representation screen — every axis is inside the noise

```bash
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
$V lab/make_manifest.py --dataset e1 --mode fixed_step --max-steps 2000 --seeds 74
$V lab/make_repr_submission.py --tag r0_base       --layout flat
$V lab/make_repr_submission.py --tag r1_field      --layout flat --field
$V lab/make_repr_submission.py --tag r2_fieldplace --layout flat --field --place
$V lab/make_repr_submission.py --tag r3_rpos       --layout flat --rpos
$V lab/make_repr_submission.py --tag r4_all        --layout flat --field --place --rpos
$V lab/make_repr_submission.py --tag r5_norel      --layout flat --no-abs --field --place --rpos
$V lab/make_repr_submission.py --tag r6_slotsep    --layout slots_sep --no-abs
$V lab/make_repr_submission.py --tag r7_slotsum    --layout slots_sum --no-abs
bash lab/repr_par.sh lab_e1_fs2000_s74 repr-screen r0_base r1_field ... r7_slotsum
```

e1, fixed 2000 steps, seed 74, rung = 38 examples (so the grid is 1/38 = 0.026):

| config | representation | rung T=1 | mean acc | MAX_T |
|--------|----------------|---------:|---------:|------:|
| r0_base | absolute position only (baseline) | 0.026 | 0.025 | 0 |
| r1_field | + field id | 0.000 | 0.038 | 0 |
| r2_fieldplace | + field + place-in-field | 0.026 | 0.025 | 0 |
| r3_rpos | + distance-from-end | 0.000 | 0.008 | 0 |
| r4_all | + field + place + rpos | 0.000 | 0.010 | 0 |
| r5_norel | field + place + rpos, **no** absolute | 0.053 | 0.052 | 0 |
| r6_slotsep | place-aligned slots, LSD-first | 0.000 | 0.043 | 0 |
| r7_slotsum | one token per place value | 0.000 | 0.013 | 0 |

Rung-1 takes exactly three values across eight architectures: 0, 1 or 2 correct
out of 38. Nothing here is an effect. (The same baseline at 400 vs 2000 steps
gives 0.053 vs 0.026 — the *identical* submission moves by the same amount that
separates the best and worst representation.)

### 3.2 The reason, and it is not representation: **every model already fits the
training set perfectly**

`train_curve` from the archive, exact-match accuracy on the training batch:

| config | step 1 | step 300 | step 900 | step 1800 | final train loss |
|--------|-------:|---------:|---------:|----------:|-----------------:|
| r0_base | 0.00 | 0.99 | 1.00 | 0.99 | 0.0000 |
| r1_field | 0.00 | 0.99 | 1.00 | 1.00 | 0.0000 |
| r2_fieldplace | 0.00 | 0.99 | 1.00 | 1.00 | 0.0000 |
| r3_rpos | 0.02 | 1.00 | 1.00 | 1.00 | 0.0000 |
| r4_all | 0.00 | 1.00 | 1.00 | 1.00 | 0.0000 |
| r5_norel | 0.01 | 0.97 | 1.00 | 1.00 | 0.0000 |
| r6_slotsep | 0.00 | 0.99 | 1.00 | 0.98 | 0.0010 |
| r7_slotsum | 0.00 | 0.99 | 1.00 | 1.00 | 0.0060 |

**Every configuration reaches 100 % exact-match on the training data by ~step
300 and drives the loss to zero, while held-out rung-1 stays at 0–5 %.** A
**205 000-parameter** model memorises e1's ~600 training rows almost immediately
(all eight configs are within 2 % of that count, so this is not a capacity
difference between them either).

This falsifies the framing that both my brief and `lab/findings.md` were working
from. The plateau is **not** a capacity wall, **not** an optimisation wall, and
**not** an expressiveness wall — the network can represent and reach a
zero-loss solution in 300 steps under *every* representation tried. It is a
**pure generalisation gap**: the model chooses the lookup table over the
algorithm, and no embedding scheme I gave it changes that preference, because
the lookup table is exactly as reachable in the place-aligned representation as
in the flat one.

That is the single most important number in this report, and it should change
what the rest of the team optimises.

### 3.3 Is it the *modulus* or the *data*? A purpose-built probe

e1 cannot separate "the representation cannot express the algorithm" from
"there is not enough data to identify the algorithm", because fixed N=323 has
only φ(323)=288 input/output pairs in total. `lab/gen_repr_probe.sh` therefore
generates two datasets that hold the modulus fixed at **N = 101·103 = 10403**
(5 digits) with the same training `T ∈ {1,2,3}` and 256-example rungs, varying
**only** the number of prompts per setting:

```bash
bash lab/gen_repr_probe.sh          # writes rp_small (250/setting) and rp_big (3000/setting)
$V lab/make_manifest.py --dataset rps --mode fixed_step --max-steps 2000 --seeds 74
$V lab/make_manifest.py --dataset rpb --mode fixed_step --max-steps 2000 --seeds 74
bash lab/repr_par.sh lab_rps_fs2000_s74 repr-data r0_base r6_slotsep
bash lab/repr_par.sh lab_rpb_fs2000_s74 repr-data r0_base r6_slotsep
```

| dataset | train rows | config | train acc @2000 | `test` split | rung T=1 |
|---------|-----------:|--------|----------------:|-------------:|---------:|
| rp_small | 600 | flat | **1.00** | 0.000 | 0.000 |
| rp_small | 600 | slots_sep | **0.98** | 0.000 | 0.000 |
| rp_big | 7 200 | flat | 0.76 | 0.0006 | 0.000 |
| rp_big | 7 200 | slots_sep | 0.81 | 0.0011 | 0.000 |

Twelve times the data of exactly the same arithmetic buys **nothing**: held-out
accuracy is 0.0–0.1 % either way. What changes is only that the model can no
longer finish memorising 7 200 rows inside 2 000 steps. Generalisation never
starts; the curve is memorisation all the way up.

The one measurable representation effect in the whole study appears here, and it
is about **fitting**, not generalising — the place-aligned layout memorises
faster. Replicated over **3 seeds each** (tag `repr-fit`, raw per-seed values
shown; the two configs' ranges do not overlap at any of steps 800/1200/1600):

| step | flat train acc (3 seeds) | slots_sep train acc (3 seeds) |
|-----:|--------------------------|-------------------------------|
| 800 | 0.069 — {0.084, 0.062, 0.061} | **0.115** — {0.125, 0.107, 0.113} |
| 1200 | 0.426 — {0.428, 0.418, 0.432} | **0.587** — {0.617, 0.551, 0.594} |
| 1600 | 0.719 — {0.717, 0.748, 0.693} | **0.823** — {0.832, 0.824, 0.812} |
| 2000 | 0.793 — {0.762, 0.801, 0.816} | 0.814 — {0.807, 0.812, 0.824} |

That is ~35 % fewer steps to a given train accuracy, and it is the only claim in
this report that clears its own variance floor. Held-out accuracy over the same
six runs: flat `test` {0.0006, 0.0011, 0.000}, slots `test` {0.0011, 0.0011,
0.0006}; rung T=1 flat {0, 0, 0}, slots {1/256, 1/256, 0}. **The fitting speedup
does not convert into any generalisation.**

### 3.4 The Hard proxy `hp1` — both representations are at the floor

```bash
$V lab/make_manifest.py --dataset hp1 --mode fixed_step --max-steps 2000 --seeds 74
bash lab/repr_par.sh lab_hp1_fs2000_s74 repr-scale r0_base r6_slotsep
```

`hp1` is 22-bit fixed N (7 digits), 81 000 training rows, T ∈ {4,8,16}.

| config | train acc (steps 1→1800) | final train loss | `test` | every rung |
|--------|--------------------------|-----------------:|-------:|-----------:|
| flat | 0.00 … 0.00 | 2.175 | 0.000 | 0.000 |
| slots_sep | 0.00 … 0.00 | 2.173 | 0.000 | 0.000 |

Train loss 2.17 against `ln 10 = 2.303` — the model is barely above a uniform
digit prior and never gets a single 7-digit answer fully right, under either
representation. Place alignment changes nothing here either.

**The three regimes together:**

| regime | train rows | fits training data? | generalises? |
|--------|-----------:|--------------------|--------------|
| e1 / rp_small | 600 | yes, 100 % by step 300 | no (0–5 %) |
| rp_big | 7 200 | partly, 0.8 by step 2000 | no (0.1 %) |
| hp1 | 81 000 | no, ~uniform prior | no (0 %) |

There is no data volume at which this model class starts to generalise; it
simply moves from "memorises everything" to "fits nothing". Representation
shifts *where* on that curve you sit, never *whether* generalisation happens.

### 3.5 Why memorisation provably cannot certify T=1 on a fixed-N Easy set

Number theory on the *public* generator parameters only (`fixed_p 17,
fixed_q 19` for e1; nothing under `data/generated/` is opened):

```
e1: N=323  phi=288  units used by train/test/ood=250  held out=38  |QR|=72 (25% of units)
e2: N=899  phi=840  units used=800                    held out=40  |QR|=210 (25% of units)
```

Squaring is exactly 4-to-1 on `Z*_pq`, so **only a quarter of the units are
quadratic residues**. A training row can only constrain the squaring map `S` at
a point that is either an explicit `x` in the prompt or an *intermediate* of a
higher-T row — and every intermediate is a square. The held-out cohort is a
uniformly random subset of all units, so in expectation only 38·(72/288) ≈ **9.5
of e1's 38 evaluation points are quadratic residues**; the other ≈ 28 can never
occur as an intermediate of any training row.

So for ~3/4 of the rung-1 cohort, *no* amount of memorisation plus compositional
constraint-propagation determines the answer. Certifying T=1 on e1 or e2 requires
a model that has actually learned "square the digits and reduce". Combined with
§3.2 — the model reaches zero train loss in 300 steps by memorising — this says
the gap is not going to be closed by a better embedding, and it puts a hard floor
on what any purely-interpolating approach can score.

### 3.6 Sampled-N (`e5`, 256-example rungs, T=1 in training)

```bash
$V lab/make_manifest.py --dataset e5 --mode fixed_step --max-steps 2000 --seeds 74
bash lab/repr_par.sh lab_e5_fs2000_s74 repr-sampledn r0_base r6_slotsep
```

| config | train acc @1800 | `test` | rung T=1 (of 256) |
|--------|----------------:|-------:|------------------:|
| flat | 0.94 | 0.0033 | 0.0078 (2/256) |
| slots_sep | 0.93 | 0.0067 | 0.0000 (0/256) |

Same shape as everywhere else — 93–94 % of the training set memorised, ~0.5 % of
held-out prompts right, no representation difference. This is the finest grid in
the study (1/256 = 0.004) and it still shows nothing.

### 3.7 The grokking regime — does place alignment change *when* it groks?

The sharpened version of the hypothesis: if the model memorises because
memorisation is the cheapest solution, then under strong weight decay and long
training the representation that gives the *algorithm* the shortest description
should transition first. e1, wd = 1.0, **20 000 steps** (10× the screen):

## 4. What is falsified

**The hypothesis I own is falsified, and not narrowly.**

1. **Field-aware embeddings** — no effect (rung-1 0.000 vs baseline 0.026; both
   inside a 1-example floor).
2. **Digit-position-within-field embeddings** — no effect.
3. **LSD-first internal ordering** — no effect (`slots_sep`, `slots_sum`). One
   thing worth recording so nobody re-runs the trivial version: a *uniform*
   reversal of the whole sequence is a no-op for a bidirectional transformer
   with learned position embeddings, since permuting positions and permuting the
   position table together leave the function class unchanged. The content of
   "LSD-first" is therefore entirely in the *place indexing* and in making the
   layout canonical across examples of different digit counts — which is what
   the slot layouts implement and what was tested here.
4. **Abacus / shared place-value index** — no effect.
5. **Output head design** (place-aligned readout, per-place heads, dedicated
   answer slots) — no effect on generalisation.

And the diagnosis is stronger than "the effects were small": **§3.2 shows there
was no room for a representation effect to exist.** Every model already reaches
100 % training exact-match with zero loss in ~300 steps. The optimiser is not
struggling to fit; it fits perfectly and generalises at chance. Better alignment
between digit places cannot help a model that has already solved its training
objective — it only makes the memorisation cheaper to find (§3.3).

The one thing that *is* real and replicated: **place-aligned slots memorise
~35 % faster in steps** (rp_big, 3 seeds, non-overlapping ranges at steps
800/1200/1600). That is a per-step learning-efficiency win that transfers to
H100, and it is worth keeping — but it is not the bottleneck.

**Also falsified: "the plateau is a capacity/architecture wall"**
(`lab/findings.md`'s headline). It is not. Capacity is *excessive*: 205 k
parameters memorise e1's 600 rows in 300 steps. The previous session's width
sweep only went **up** (128 → 768, all worse), which is the wrong direction for
a memorisation problem.

## 5. What I would do next, and why

### 5.1 The one recommendation, if the team only takes one

**Re-frame the target from "fit better" to "make memorisation unavailable", and
put `train_exact_accuracy` next to eval accuracy in every comparison.** It is
already in `lab/archive.jsonl` under `train_curve` and nobody has been reading
it. Any lever that leaves the train curve at 1.00-by-step-300 and the eval curve
at chance is, by construction, a lever on the wrong variable — that includes
depth, width, optimiser, loss shaping, and (as this report shows) every
input/output representation. The variables that *can* move a memorisation gap
are: capacity **downward**, regularisation strength, training far past
convergence (grokking), and architectures in which a per-example lookup is not
expressible.

### 5.2 Stop screening on e1

e1's rung is 38 examples: the accuracy quantum is 1/38 = 0.026, the seed-only
spread is one example, and every number the previous session and this one
produced on e1 is 0–3 correct examples. It has essentially no statistical power
to rank architectures. `e5`, `e3`, `e4`, `m5` and the two probes here all have
**256-example rungs** — a 6.7× finer grid and far better SNR — and `e5` also has
T=1 inside its training range. Screening on e1 has already cost two sessions'
worth of conclusions drawn from single-example differences.

### 5.3 Which tiers are even winnable (structural, representation-independent)

From §1.2: rung T=1 is **out of distribution in T** on m1–m5 and e3/e4. Since
`max_certified_time_steps` is a prefix from T=1 upward, MAX_T ≥ 1 on those tiers
is unreachable for anything that memorises a T-conditioned map, no matter how
accurate it is at the T values it trained on. If Hard resembles Medium (the
brief says it sits above Medium on the same two knobs), **the Hard leaderboard is
gated on T-extrapolation *downward* to T=1**, and a weight-tied model that
literally iterates T times is the only shape of solution that gets there. That
argues for the `tied-recurrence` agent's axis being the load-bearing one, with
the caveat that it must be combined with something that fixes generalisation —
iterating a step you cannot compute exactly T times still gives 0.

### 5.4 Things I would *not* spend more time on

Input/output representation for this task. Not because the ideas are wrong —
place alignment is the correct way to encode the problem and it does measurably
speed up fitting — but because §3.2 shows the model is not representation-limited
at any point on the data-volume curve.

## 6. The submission

`submissions/exact-arithmetic/submission.py` — the place-aligned slot layout
(`LAYOUT="slots_sep"`, no absolute position embedding), D=128, 4 heads, 8
weight-tied loops, AdamW lr 1e-3 wd 0.1, plain CE.

**It does not beat the baseline on the metric.** MAX_T = 0, exactly like every
other configuration in this report and like the previous session's best. I am
shipping it rather than the flat baseline for three reasons that are measured,
not asserted:

1. it removes the readout misalignment verified in §1.1 (place-exact readout
   that does not shift when `x` or `T` gains a digit — the latter happens at
   rungs 16/32/64);
2. it memorises ~35 % faster in steps, replicated over 3 seeds with
   non-overlapping ranges (§3.3), which is hardware-independent and so transfers
   to H100;
3. shorter sequences (2·places + digits-of-T instead of the full prompt) make
   both training and the 14-rung evaluation cheaper, and eval budget is a real
   failure mode on the deeper tiers.

Checks run: source lint via `submission_validation.validate_submission_source`
(passes), and a **deliberate wall-clock feasibility run** on the real Easy 60 s
manifest (`lab/manifests/lab_e1_wc60_s74.json`, tag `repr-feasibility`) — 332
steps in 60.0 s on a heavily contended shared GPU, and all 14 rungs plus
`test`/`ood` completed inside the half-budget evaluation window with no
`not_completed` rung. Timing is B300-and-contention-local; the *fit-in-budget*
conclusion is what matters.

## 7. Compliance record

* Nothing under `data/generated/` was read, printed, sampled or summarised.
  Every structural claim comes from `data/squaring_mod.py`,
  `scripts/generate_datasets.sh`, `benchmark/runner.py` or elementary number
  theory on public generator parameters. `lab/test_repr.py` constructs its
  prompts by calling the public tokenizer, not by loading a split.
* Two datasets were generated with the public generator
  (`lab/gen_repr_probe.sh`, exact commands in that file and in §3.3) and, like
  every other dataset, never inspected.
* No arithmetic is implemented in any `forward`. The slot layouts do index
  arithmetic on token *positions* (a `cumsum` over the public marker tokens plus
  gather/scatter); they never operate on the digit *values*. Every prediction
  comes from parameters trained from random init in the run, and
  `lab/test_repr.py` asserts the loss backprops to all of them.
* No data-dependent Python control flow: the loop count is the constant
  `NUM_LOOPS`. The only `.item()` calls read the batch's maximum field length to
  size a tensor, which is shape metadata, not a value-dependent branch.
* No custom training loop, no participant-controlled backward, no manifest
  override in the submission.
* Nothing was submitted to the hosted service; `one-layer login`/`submit` were
  never run and no network call was made.
* Every run — including all the negative ones — is in `lab/archive.jsonl` under
  tags `repr-*`.

