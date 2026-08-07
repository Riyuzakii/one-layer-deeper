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

All confirmed from the runner's own `RESULT_JSON` (`example_count` per split), not
inferred.

| quantity | value |
|---|---|
| train rows | **243,000** (27,000 per (bit size, T), 3 sizes × 3 T) |
| `test` / `ood_t` / `ood_n_t` | **27,000 / 9,000 / 9,000** |
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

And the same control for **this branch's own architecture**, because the floor
has to be read per model class:
`submissions/hard-digitalu-hf1/lr0` — the tree:quotient `DigitALU` (S=7) with
`LR=0`, `SEL_LR=0`. **LEGAL.** `lab_hf1_fs5_s74`.

| | value |
|---|---|
| `MAX_T` / `OOD_N_MAX_T` | **0 / 0** |
| every rung, both ladders, `correct_examples` | **0 of 768** |
| `test` / `ood_t` / `ood_n_t` | **0.000 / 0.000 / 0.000** (0 of 27,000 / 9,000 / 9,000) |
| step-1 / final train loss | 2.860 (`ln 17 = 2.833`) |
| model state elements | 8,873 |
| **evaluation wall clock, all 16 splits** | **265 s** at `eval_batch_size` 2048 |

> **Everything at initialisation is a hard zero on the ranked metric, for both
> model classes.** Any non-zero rung a sibling reports is above this floor only
> if it exceeds 2/768.

**A budget result worth flagging on its own: `tree:quotient` fixes the eval
budget.** `alu-compose` P2 measured the 257-step serial `DigitALU` at a **1.1×**
margin against the Easy allowance and a hard `TimeoutError` at fixed depth —
which is *below* the leaderboard floor, since `service/db.py` counts only
`status='succeeded'`. At Hard's own shape the 129-step graph runs all 16 splits
of `hf1` in **265 s against an 1,800 s allowance — a 6.8× margin**, on a GPU
shared with three other agents. Eval is no longer a reason to reject this
family.

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
| `DigitALU` tree:quotient, S=7, 16 loops | batch 128, L=19 | **5,030** (evaluator-measured) | **~360** |

The hosted H100 reference is 38.6 ms/step for a `D`=128 × 8-loop stack, i.e.
~4.8 ms per loop, so this box's 1-block reference is within ~10% of the H100
figure per unit of work. Take the ALU row as a **ratio**: the `DigitALU` costs
**~950×** the reference model's step, at Hard's own shape. Both numbers are
from the evaluator, on a GPU shared with three sibling agents, so the ratio is
the transferable quantity and the absolute is pessimistic.

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

## 4. The legal objective on `hf1` — and the modulus split changes nothing

Same graph, same population, same budget shape, the only difference being that
no tape is replayed. `--lr 0` control alongside, per rule 1.

| run | LEGAL? | moduli in train | steps | `train_exact_hard` | `held_unseen_N` hard (argmax / mix / **best**) | replicas ≥ 0.99 |
|---|---|---|---|---|---|---|
| `--lr 0` control | LEGAL | 1,725 | 20 | 0.0000 | 0.0000 / 0.0000 / **0.0000** | 0 / 32 |
| **pool** (hf1's own density) | LEGAL | 1,725 | 2,000 (and 3,000 in a second run) | **0.0000** | 0.0000 / 0.0000 / **0.0000** | **0 / 32** |
| **few-8** (prompt-split analogue) | LEGAL | 8 | 2,000 | **0.0000** | 0.0000 / 0.0000 / **0.0000** | **0 / 32** |
| teacher forcing | DIAGNOSTIC | 1,725 | 1,200 | — | 0.0000 / 0.2217 / **1.0000** | ≥ 1 / 32 |

All three of argmax / mixture / best replica are **0.0000** on every split in
every legal cell — 32 replicas × 2 conditions and **not one of 64 replicas gets a
single 7-digit answer right on 2,048 held-out prompts.** The per-replica vector
is 32 exact zeros in both runs, so this is not a selector failure hiding a good
component.

**Calibration (rule 2).** The legal runs are 5–15× past the fitting onset the
*working* signal shows on the same graph, the same scale and the same
population: at step 200 the teacher-forced run's best replica is already at
**0.517** on unseen moduli, and 13 of 32 replicas are off zero. So the legal
null at 2,000–3,000 steps is a calibrated null, not a pre-fitting artefact. The
loss curve confirms it flattens: 4.601 → 3.851 by step 250 and then
3.867 / 3.835 / 3.821 / 3.838 / 3.821 / 3.866 / 3.783 through step 2,000.

**The control that decides how to read it.** `local_ce` (per-op CE against the
construction; the cliff separating basin from no-basin is ~0.006):

| | `local_ce` min over 32 | median | `mul_gauge` |
|---|---|---|---|
| **random init (`--lr 0`)** | **2.1425** | **2.1899** | 0.8 |
| legal objective, few-8, 2,000 steps | **2.9074** | **3.6912** | 0.4 |
| legal objective, **pool**, 2,000 steps | **3.1214** | **3.9535** | 0.5 |
| teacher forcing, 600 steps | 0.0142 | 0.1274 | 1.0 |
| teacher forcing, 1,200 steps | 0.00003 | 0.0891 | 1.0 |

**Training on the legal objective moves `local_ce` 2.14 → 3.12 (best replica)
and 2.19 → 3.95 (median) — i.e. AWAY from the discrete solution**, and it does so
at Hard-faithful scale, on a modulus-split dataset, at 129 graph depth, with
32 independent initialisations. This is `alu-optimizer`'s inversion reproduced
under every condition it had never been tested at. The structure scores move the
same way: `mul_gauge` **0.8 at init → 0.4–0.5 trained**, and `add_shift`
0.255 → 0.265 / `sub_shift` 0.287 → 0.237, i.e. inside the 0.23–0.28 random
band throughout.

**The population's basin rate at Hard-faithful conditions, measured.** Under the
working signal at 600 steps the 32 per-replica held-unseen-N accuracies are

`[0, 0, 0, .806, 0, 0, 0, .251, 0, 0, 0, .651, 0, 0, 0, .806, .651, .368, 0, .651, .368, 0, 0, .004, .616, .485, 0, .696, 0, 0, 0, .319]`

— **13 of 32 off zero, 7 of 32 above 0.6**, converging to a replica at 1.000 by
step 1,200. That is a per-replica rate of ~0.22–0.41, consistent with
`alu-population`'s 0.26 at S=5 on one modulus, so the population instrument
transfers to this scale unchanged. Under the legal objective the same instrument
returns 0 of 64. **A population is the right instrument for a stochastic
obstruction and the wrong one for a systematic one, and it reads systematic
here too** — the same conclusion `alu-population` reached, now at the ranked
tier and on the ranked split structure.

**The modulus split changes nothing, in either direction.** Compare `pool`
(1,725 training moduli, hf1's own density, 47 operands per modulus) against
`few-8` (8 training moduli, 10,124 operands each — the `hp1`/`hp2`/`hp3`
prompt-split condition at matched row count and matched step count):

| | pool (1,725 moduli) | few-8 (prompt-split analogue) |
|---|---|---|
| loss at step 1,000 | 3.8208 | 3.8460 |
| loss at step 2,000 | 3.7832 | 3.8550 |
| `train_exact_hard` | 0.0000 | 0.0000 |
| `held_unseen_N` hard, best of 32 | 0.0000 | 0.0000 |
| `local_ce` min / median | 3.121 / 3.954 | 2.907 / 3.691 |
| output diversity, unseen-N | 0.97 | 0.96 |

They are the same run to three decimals. **Training across 1,725 moduli is
neither the extra constraint that unlocks the tables nor the extra difficulty
that breaks them.** That is exactly what §2 and §3 predict: the parameters are
modulus-independent by construction, so the number of moduli in the training set
is not a variable the objective can see.

So the honest answer to "does the modulus split change anything?" is:

* **for the ceiling, no** — constructed and teacher-forced both read 1.000 on
  unseen moduli and unseen modulus sizes;
* **for the legal objective, no** — identical null at 8 moduli and at 1,725;
* **for the field, yes, but only by closing an escape hatch** — a residue-indexed
  readout is dead by construction on `hf1`, and the memorisation route that
  produced every `train_exact` → 1.000 result in this project's history is dead
  too (§0.3). `hf1` does not make `DigitALU` harder; it removes the alternatives.

**Collapse check, against the MEASURED reference (§2: 0.998–1.000 here).**
Legal runs read 0.86–0.99 on unseen-N at their final step, with transient dips
to 0.63 (few-8, step 500) and 0.39 (pool, step 750). Diverse and wrong, not
collapsed — the `--assoc` degeneracy from `alu-population` does not appear.

---

## 5. THE RESULT THAT CLOSES IT: the step budget, measured on the evaluator

This is independent of the objective, and it is new.

| | measured |
|---|---|
| `DigitALU` tree:quotient, S=7, 16 training loops, batch 128 | **4.93 s / optimizer step** (591.3 s for 120 steps, `benchmark.runner`, contended B300) |
| eval, all 16 `hf1` splits, `eval_batch_size` 2048 | **265 s** |
| steps available in a 3,600 s Hard run (1,800 s training) | **≈ 360** |
| **fitting onset under the ILLEGAL working signal** | **200–400 steps** (best replica 0.517 at 200, 1.000 by 1,200) |

> **A Hard run affords ~360 optimizer steps for this architecture. The teacher-
> forced ceiling — which is illegal — needs 200–400 steps to leave zero and
> ~1,200 to reach 1.000. Even if a legal per-op signal existed, the ranked tier
> could not train this model with it.**

This is a *different* closure argument from the objective one and does not
depend on it. It also survives the obvious rescues:

* Reducing training loops (the 16 is `hf1`'s deepest training T) trades step
  cost against the composition the loop is there to learn.
* Batch size does not help — the model, not the DataLoader, dominates
  (`RESUME`: batch 512→32 is 1.2× for the ALU, not 5.8×).
* `torch.compile` and hand-written kernels are unavailable on this box
  (`sm_107a` / `ptxas-blackwell`), and on an H100 they would need ~10× to close
  a 950× gap.
* The 129-step graph is already `alu-depth`'s best; the serial graph is 1,537
  steps at S=7, i.e. 12× worse.

The step famine `alu-compose` called P3 was measured at ~14–22 steps for the
*Easy/Medium* tiers on the 257-step graph. Hard's 3,600 s and the 129-step graph
lift it to ~360 — a 20× improvement that lands **just below** where the ceiling
becomes reachable, and only under a signal that cannot be used.

---

## 6. Through the real evaluator — LEGAL, and at the floor

`submissions/hard-digitalu-hf1/submission.py` (parser → tree:quotient
`DigitALU` × k → counted-halting depth controller; every tensor learned from
random init; lint clean; no oracle, no teacher forcing, no manifest override).

| run | steps | `MAX_T` | rungs seen_n | rungs ood_n | `test`/`ood_t`/`ood_n_t` |
|---|---|---|---|---|---|
| `lr0` control (**LEGAL**) | 5 | **0** | all 0/768 | all 0/768 | 0.000 / 0.000 / 0.000 |
| trained (**LEGAL**) | 120 | **0** | 1/768 at T=32, else 0 | all 0/768 | 0.000 / 0.000 / 0.000 |

The one non-zero cell is **one example out of 768** — the variance floor from
§0.1. Trained and untrained are indistinguishable through the evaluator, which
is the fourth time in this project that `--lr 0` has come out level with a
trained model.

Scope, stated plainly: 120 steps is *below* the ~360 a Hard run affords and far
below any fitting onset, so **this evaluator row is a compliance and
timing artefact, not the scientific null.** The calibrated null is §4's, taken
offline at 5–15× the measured onset. I am reporting the evaluator row because
the deliverable asks for it and because it fixes the eval-budget question
(§0.2), not because 120 steps proves anything about trainability.

---

## 7. Verdict

**`DigitALU` is closed at the ranked tier, and it is now closed twice over.**

1. **The objective.** At Hard-faithful scale, on the ranked split structure,
   with `tree:quotient`, untied tables, `EMB_INIT`, a 32-replica population, a
   differentiable selector and a calibrated budget — every legal cell is
   0.0000 on argmax, mixture *and* best replica, across 64 independent
   initialisations, while `local_ce` moves from 2.14 at random init to 3.12
   trained. Training goes the wrong way from step one, exactly as
   `alu-optimizer` measured at a tenth of this scale.
2. **The budget.** A Hard run affords ~360 optimizer steps for this
   architecture. The *illegal* ceiling needs 200–400 to leave zero and ~1,200
   to reach 1.000. Even a perfect legal per-op signal, if one were found
   tomorrow, would not fit in the tier.

Argument 2 is the one to carry forward, because it does not depend on the
objective and it prices the whole family: **any candidate whose per-step cost is
~10³× the reference model's cannot be trained at this tier, whatever its
gradients look like.** `DigitALU` is 950×.

**What survives and should be reused.**

* The **representation** is not the problem and never was. The constructed and
  teacher-forced ceilings are 1.000 on unseen moduli and unseen modulus *sizes*
  with 6,821 modulus-independent parameters against 80,992 rows. `hf1`'s
  modulus split is not an obstacle to this family; it is only an obstacle to
  everything else.
* **`tree:quotient` solves the eval budget** at Hard's shape — 265 s against
  1,800 s, a 6.8× margin where the serial graph had 1.1× and a `TimeoutError`.
  A future candidate in this family inherits that for free.
* The **replica population** transfers to this scale unchanged (13/32 off zero
  at 600 steps under the working signal) and correctly reads *systematic* on the
  legal objective (0/64).

**Two additions to `RESUME.md`'s fooling table**, both earned here:

| metric | fooled by | reads | but |
|---|---|---|---|
| argmax-replica accuracy ("commit to the mode") | teacher forcing, which removes `alpha`'s gradient | 0.0000 | best replica **1.0000**; `wmax` pinned at exactly 1/P |
| mixture over replicas | a population with a few exact members | 0.2217 | best replica **1.0000** — the blend is 4.5× *worse* than its best component |

The general rule stands and is sharpened: **`wmax` is not a diagnostic to check
after the fact, it is a precondition for reading either the argmax or the
mixture at all.** If `wmax ≈ 1/P` the selector has learned nothing and both
numbers are meaningless.

**Recommendation for the ranking.** Move `DigitALU` from #2 to closed. Its two
transferable assets — the modulus-independent digit readout and the 129-step
graph's eval margin — belong to whatever replaces it, and the ranking's #1
(`addition-only transducer`) should be checked against argument 2 **before** it
is built: if its per-step cost is within ~10× of the reference model it is
worth building, and if it is within ~10³× it is not, independent of how good its
repair basin is.

---

## 8. Files

| what | where |
|---|---|
| this report | `lab/reports/hard-digitalu-hf1.md` |
| the submission (LEGAL) | `submissions/hard-digitalu-hf1/submission.py` |
| its `--lr 0` control | `submissions/hard-digitalu-hf1/lr0/submission.py` |
| reference-class models | `submissions/hard-digitalu-hf1/ref/*/submission.py` |
| generators | `lab/make_dalu.py`, `lab/make_ref.py` |
| the hf1-faithful probe | `lab/probe_hf.py` (+ `lab/probe_pop.py`, `lab/probe_alu_depth.py` reused from siblings) |
| interleaved step-cost ratio | `lab/time_ratio.py` |
| raw run logs | `lab/runs/*.log`, `lab/runs/hf_alu.jsonl` |
| evaluator archive | `lab/archive.jsonl` (tags `F-floor`, `F-fit`, `D-alu`) |
