# `DigitALU` at Hard-faithful scale — and the reference floor for `hf1`

Branch `hard/digitalu-hf1`. Dataset `hf1` (`split_group=modulus`,
`separate_ood_splits`, ID bits [16,18,20], OOD-N [17,19,21], train T {4,8,16},
OOD T 32, 298,752 rows, `max_seq_len` 19).

Every row below is labelled **LEGAL** or **DIAGNOSTIC**. Nothing was submitted to
the hosted service. Nothing under `data/generated/` was read.

---

## 0. THE REFERENCE FLOOR — read this first

Three siblings are blocked on these numbers. They are all measured through the
**real evaluator** (`benchmark.runner`) on `hf1`, `--mode fixed_step` so they are
immune to GPU contention and comparable across agents.

### 0.1 Structural constants of `hf1` (measured, not assumed)

| quantity | value |
|---|---|
| train rows | **243,000** (27,000 per (bit size, T), 3 sizes × 3 T) |
| `test` / `ood_t` / `ood_n_t` | 27,000 / 9,000 / 9,000 |
| depth rungs, **each** | **768 examples** (256 per bit size × 3) — both ladders |
| **variance floor on a rung** | **1/768 = 0.0013** |
| train modulus pools | 133 / 488 / 1546 (16b / 18b / 20b) = **2,167** |
| test modulus pools (disjoint) | **15 / 55 / 172 = 242** |
| digit slots needed | S = 7 for x, W = 8 for N (21-bit OOD-N fits) |

A rung reading `0.001302` is **one example**. Treat anything below `0.003` on a
rung as the variance floor.

### 0.2 The `--lr 0` control — the floor itself

`submissions/hard-digitalu-hf1/ref/d128_L1_e0.02_lr0.0_b512_lr0` — the official
baseline architecture with `EMB_INIT=0.02`, AdamW at `lr=0` and `wd=0`, so the
parameters are bit-identical to `build_model`'s output at every step.

**LEGAL.** `lab_hf1_fs20_s74`, seed 74.

| | value |
|---|---|
| `MAX_T` / `OOD_N_MAX_T` | **0 / 0** |
| rungs, seen_n, T = 1…64 | **0.000** at every rung |
| rungs, ood_n, T = 1…64 | **0.000** at every rung |
| `test` / `ood_t` / `ood_n_t` | **0.000111 / 0.000 / 0.000111** |
| `mean_exact_accuracy` | 7.41e-05 |
| step-1 loss | **2.862** (`ln 17 = 2.833`) |
| model state elements | 202,752 |
| evaluation wall clock, all 16 splits | **3.1 s** |

> **Everything at initialisation is a hard zero on the ranked metric.** Any
> non-zero rung a sibling reports is above this floor only if it exceeds 2/768.

Note the step-1 loss: `EMB_INIT=0.02` removes the toll. The hosted Hard run's
`metric.jsonl` shows step-1 loss **79.936** for the same architecture with
`nn.Embedding`'s default `N(0,1)` and a tied head. `submissions/baseline_adamw`
still pays it.

### 0.3 The fitting curve — where the pre-fitting region ends on `hf1`

**LEGAL.** Reference-class transformer, `EMB_INIT=0.02`, batch 512, lr 1e-3,
AdamW(0.9, 0.95), wd 0.1, `lab_hf1_fs40000_s74`, seed 74, 40,000 steps.

`params/row` is against **243,000 train rows**. `e5` fit at 24.9 params/row;
`m1` at `D_H`=128 was 17.4 and did fit at 40k steps.

| `D_MODEL` | params | params/row | `train_exact` @ 40k | final loss | `MAX_T` |
|---|---|---|---|---|---|
| 128 | 202,752 | **0.83** | **0.000** (flat for all 40,000 steps) | 2.156 | 0 |
| 512 | 3,170,304 | 13.0 | (see §0.4) | | 0 |
| 1024 | 12,632,064 | 52.0 | (see §0.4) | | 0 |

**`D_MODEL`=128 on `hf1` never leaves the pre-fitting region.** Loss falls
2.862 → 2.16 in the first ~4,000 steps and then sits there for 36,000 more:
the model has learned the digit marginals and nothing else. Batch-level
`train_exact` is 0.000 at every one of the 400 logged points; the single
0.00195 at step 40,000 is one example in a batch of 512.

**Consequence for every sibling:** a null taken with a baseline-width model on
`hf1` is *uninterpretable* — it is a null in a region where the model has not
begun to fit anything. `hf1` is 9× more rows than m1 at the same width, and
0.83 params/row is 30× below the 24.9/row that fit e5.

### 0.4 Throughput calibration on this box (B300, contended)

| model | shape | ms/step | steps in a 1,800 s training half-budget |
|---|---|---|---|
| ref `D`=128, 1 block | batch 512, L=19 | **5.3** | ~340,000 |
| `DigitALU` tree:quotient, S=7, 16 loops | batch 128, L=19 | **12,800** | **~140** |

The hosted H100 reference is 38.6 ms/step for a `D`=128 × 8-loop stack, so the
1-block reference here is within ~10% of the H100 figure per unit of work; take
the ALU row as a **ratio**: the `DigitALU` is **~2,400×** the reference model's
step cost at Hard's shape. Evaluation is not the problem — 4.7 s per 2,048
prompts × 28 batches = **131 s** against an 1,800 s eval budget.

---

## 1. What this branch ran, and why

`DigitALU` was ranked #2 for one reason: it has a **provably exact solution in
its class**, and it had never been run on a modulus-split dataset with the
corrections this project earned applied together. Those corrections are:

| correction | source | applied here |
|---|---|---|
| `tree:quotient` graph | `alu-depth` | yes — 129 sequential soft steps at S=7 instead of 1,537 for the serial graph |
| **untied** tables | `alu-credit` (ties help repair, hurt learning from random init) | yes |
| `EMB_INIT=0.02` | `plan2/phase0` + the hosted run's step-1 loss 79.936 | yes — step-1 loss is `2.833 = ln 17` exactly |
| replica population + differentiable selector | `alu-population` | yes, P = 32 |
| the real step budget | hosted H100 calibration | measured, and it is the headline negative — see §5 |
| a calibrated fitting curve | BRIEF2 §2(e) | yes — §0.3 and §4 |
| `--lr 0` control | `alu-optimizer` | yes, on every result |

The probe is `lab/probe_hf.py`. It rebuilds `hf1`'s structure offline from the
**generator source** (`_enumerate_sampled_factor_pairs`, a 90/10 partition by
count, `(p-1)(q-1)`-weighted modulus draws, unit `x`) and never opens anything
under `data/generated/`. Its modulus pools come out at **133/488/1546 train**
and **15/55/172 test** — matching the "15/54/172 unseen moduli" recorded for
`hf1` itself, so the reconstruction is faithful.

It measures **one squaring step on 7 digit slots**, which is the only open
bottleneck; parsing, T-fold composition and the depth controller are solved
elsewhere and are deliberately absent so nothing else can absorb the result.

---

## 2. The constructed ceiling at Hard-faithful scale — DIAGNOSTIC

`lab/probe_hf.py --construct` (tables set to the truth; **never a submission**).

| split | soft states | hard (argmax-snapped) states |
|---|---|---|
| train (train-pool moduli) | 1.0000 | **1.0000** |
| held, unseen x, **train-pool** modulus | 1.0000 | **1.0000** |
| held, unseen x, **held-out** modulus (= `hf1`'s `test`) | 1.0000 | **1.0000** |
| held, **17/19/21-bit** modulus (= `hf1`'s `ood_n_*`) | 1.0000 | **1.0000** |

**DIAGNOSTIC.** The exact solution is in the class at S=7, on unseen moduli, at
unseen modulus *sizes*, soft and hard. The 39-step S=5 ceiling from `alu-depth`
survives the move to Hard-faithful shape.

**Measured collapse reference for this setting: output diversity of the exact
solution is 0.998–1.000.** (It is *not* 1.0 in general — `plan2/phase0` measured
0.7012 on an e5 rung, and `matrix-scan` computed 0.221 for the whole map. On
`hf1`-faithful cohorts every prompt carries a different (N, x) pair, so the
image is essentially injective and the reference is ~1.0. Use this number, not
the e5 one, for anything on `hf1`.)

---

## 3. The illegal ceiling under a per-op signal — DIAGNOSTIC, and it is 1.000

`--tf 1.0`, P = 32, 1,200 steps, 80,992 training prompts over 1,725 distinct
training moduli. Teacher forcing replays a tape recorded from a **constructed**
copy of the same model, so this is a **LAB DIAGNOSTIC** (rules 2 and 7) and can
never appear in a submission.

| metric (hard states) | train | held seen-N | held **unseen-N** | held OOD-N |
|---|---|---|---|---|
| **best replica** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| mixture over 32 | 0.2197 | 0.2441 | 0.2217 | 0.0439 |
| argmax(`alpha`) replica | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

`local_ce` min/median = **3e-05 / 0.0891** (the cliff is at ~0.006).
Structure: `mul_fn` 1.000, `mul_gauge` 1.000, `add_shift` 0.905,
`sub_shift` 1.000. Diversity 0.998–0.999.

Three things follow, and they matter more than the headline:

1. **The modulus split costs the ceiling nothing.** train = held-seen-N =
   held-unseen-N = held-OOD-N = 1.000. A `DigitALU` that gets its tables right
   generalises to moduli it has never seen, at bit sizes it has never seen,
   *exactly* as well as it fits. This is stronger than `alu-population`'s
   1.000/1.000, which was S=5 on a **single** modulus with 6,821 parameters
   against 8,000 rows; here it is 6,821 parameters against **80,992 rows over
   1,725 moduli** — 0.084 params/row, twelve times below any memorisation
   threshold.
2. **A NEW WAY THE `argmax` RULE LIES, and `wmax` catches it.** The argmax
   replica reads 0.0000 while the population contains a replica at 1.0000.
   Cause: under full teacher forcing every replica emits the forced trace during
   training, so the mixture CE has **no gradient with respect to `alpha`** —
   `wmax` sits at exactly **1/32 = 0.0312** from step 1 to step 1,200. With an
   uninformative selector, "commit to the mode" commits to an arbitrary replica.
   `RESUME`'s rule "check `wmax` first" fires here, and this is a cleaner
   instance of it than the one that produced the rule.
3. **The mixture is worse than its best component by 4.5x.** Blending 32
   replicas of which a few are exact dilutes the exact ones. Report
   argmax / mixture / best as a row; any one of the three is misleading alone.

The trajectory also peaks and decays: the mixture reads 0.670 at step 400,
0.394 at 600, 0.210 at 1,000. **Fitting onset under the working signal is
between step 200 and step 400.** That is the number the legal run has to be
calibrated against, and it is the number §5 kills.

---
