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
| m3,m4 | sampled | 2 / 8 only | 256 | m3 no, m4 no |
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
   almost no signal — which is why the runs below add `e2` (40), `e5` (256) and
   two purpose-built probes with 256-example rungs.
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
| r6_slotsep | 0.00 | 0.99 | 1.00 | 0.98 | 0.0010 |
| r7_slotsum | 0.00 | 0.99 | 1.00 | 1.00 | 0.0060 |

**Every configuration reaches 100 % exact-match on the training data by ~step
300 and drives the loss to zero, while held-out rung-1 stays at 0–5 %.** A
0.25 M-parameter model memorises e1's ~600 training rows almost immediately.

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

