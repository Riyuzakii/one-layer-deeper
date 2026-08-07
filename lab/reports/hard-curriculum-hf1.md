# `hard/curriculum-hf1` — a loss-side curriculum over modulus size and operand magnitude

**Verdict: NULL, and closed for a mechanical reason rather than a tuning one.**

Method #4 in `lab/RANKING.md`. Ranked below the architectural entries, and the
brief asked for a clean negative if that is what the evidence says. It is.

**The one-paragraph version.** A curriculum needs a *learnable source* and
*transfer from source to target*. Transfer is free here and I measured it: given
an illegal per-op signal restricted to **16-bit examples only**, the shared
digit tables come out **exactly right** and score **0.896 hard exact on 20-bit
operands, 0.878 on unseen moduli, 0.867 at unseen modulus sizes** — beating the
same signal spread over all three sizes (0.380) by 2.4x at matched steps
(§12). The source does not exist: under the legal
end-of-chain label, `train_exact_hard` is **0.000 at every modulus size**,
including a 10-bit, three-decimal-digit problem six bits below `hf1`'s floor
(§6). And the method's stated premise is inverted — with a fixed slot count a
small modulus does **not** shorten the chain, it just makes the example touch
21.5 of the 100 shared product cells instead of 28.2 and makes its answer *less*
sensitive to a wrong cell (§8). So the curriculum upweights the thinner, quieter
gradient toward a source that is not there. Every schedule, strength and axis
reads `train_exact_hard` **0.000** and held-out exact **0.000** (§7), the
strongest one does measurable damage, and the `--lr 0` control is the same 0.000
(§3, §9).

---

## 1. The hypothesis, and the one thing that makes it worth testing

`hf1` contains **three modulus sizes — 16, 18 and 20 bits — in one training
set**, and a `DigitALU`'s tables are modulus-independent by construction, so
*every* example trains the *same* ~6.8k parameters whatever its modulus. That
makes "upweight the 16-bit examples early and anneal toward 20-bit" a genuine
curriculum over difficulty on shared parameters — not a data-order trick, which
the evaluator's fixed loop would forbid. Operand magnitude within a modulus is a
second, finer axis.

The honest prior, stated in the brief: `explore/alu-credit` measured a magnitude
curriculum as null (`--xcurr 0.5`, `train_exact_hard` 0.012 against a 0.008
baseline), but at **e1 scale on a single 3-digit modulus**, where the
modulus-size axis does not exist at all. Whether the axis behaves differently on
a dataset that actually contains three sizes was untested.

---

## 2. What I built, and why it had to be new

`explore/alu-depth`'s `DigitALU` and `explore/alu-population`'s `PopALU` both
take **one modulus per forward** (`ndig` is a single `(W,10)` tensor). A
curriculum over modulus size needs several moduli in the same batch, so
`lab/probe_curric.py` carries a **per-example modulus**: `ndig` is `(b, W, 10)`,
`multiples` is `(b, M, W, 10)`, and the multiples-contracted subtract table is
formed once per forward. The graph is otherwise exactly
`probe_alu_depth.DigitALU(mul_mode='tree', reduce_mode='quotient',
scan_mode='serial')` — the corrected `tree:quotient` the ranking names — at
`S = 7` slots, which is what a 20-bit (and a 21-bit OOD-N) modulus needs.

**Base architecture, and why.** The corrected `DigitALU`: `tree:quotient`,
untied, 6,820 parameters, no index ranging over `Z_N`. It is the only
architecture in the project whose *representation* survives `hf1`'s
modulus split — test moduli never appear in training, so residue-indexed
readouts are dead by construction (`RESUME.md` Round 6). A curriculum over
modulus size is only meaningful on a readout whose parameters are shared across
moduli, which singles this family out.

`EMB_INIT = 0.02` is an embedding-init correction and has no embedding to act on
in the probe; it is applied in the shipped submission, which does have one.

**A correctness consequence of mixing sizes, and it is load-bearing.** With a
*fixed* slot count shared across modulus sizes, the published reduction schedule
("reduce only for `t <= S`") is **wrong for the small moduli**: after `2S-1-S`
unreduced places the register can exceed `10·N` for a 16-bit `N`, so the true
quotient leaves the `Q+1` alphabet. `probe_curric.py` therefore reduces at
**every** place, which keeps `r < 10·N` at every modulus size and costs 14
reductions instead of 8 (graph depth **183** sequential softmax steps instead of
~129, plus a 32-step shared `N`-multiples prefix). Anyone else running a
fixed-`S` ALU on a mixed-modulus dataset needs this.

---

## 3. Gates, run before any curriculum run

| gate | result |
|---|---|
| **constructed ceiling** (DIAGNOSTIC) | `train`/`held_x`/`held_n` **1.000 soft AND hard** at 16/18/20 bits, and **1.000** at OOD-N 17/19/21 |
| `assert_model_state` | 6,820 / 500,000,000 |
| true max quotient | 9, alphabet covers 0..10 |
| **`--lr 0` control** (LEGAL) | exact **0.000** everywhere; `dacc` 0.105; `local_ce` **2.201** |
| **`local_ce` cross-check** | 2.201 lands inside `alu-optimizer`'s measured random-init band **2.08–2.24**, at a different `S` and a different modulus mix |
| **measured collapse reference** | constructed-solution output diversity **0.992–0.997** on these cohorts (not 1.0, and not `matrix-scan`'s 0.221 — that is the right quantity for a fixed-modulus cohort, not this one) |
| **measured `dacc` trivial floors** | constant-**zero** predictor: **0.379 / 0.303 / 0.235** at 16/18/20 bits; per-slot majority 0.390 / 0.323 / 0.247 |
| **`--grad-equiv`** | per-row loss weight vs per-row gradient scale: relative gradient difference **6.9e-08**, cosine **1.00000000** |

The trivial-floor row is the important one. A 16-bit answer leaves the top two of
seven slots identically zero, so *predicting zero everywhere* already scores
0.379 on 16-bit examples and 0.235 on 20-bit ones. **A model that learns nothing
but "leading digits are zero" produces exactly the profile a successful
size-curriculum would produce.** Only *within-bucket, across-configuration*
comparisons of `dacc` mean anything; the cross-bucket ordering is an artifact.

---

## 4. A mechanical fact about `training_loss` on this dataset

`training_loss(logits, labels, aux)` receives `token_logits[valid]` and
`token_targets[valid]` — already flattened. On `squaring_mod`,
`tokenize_squaring_mod_with_result` emits `number_tokens(result)` with **no zero
padding**, so the number of supervised positions is the **decimal digit count of
the answer** and varies row by row (`data/squaring_mod.py:434`,
`collate_squaring_mod:90-106`). Nothing in `(logits, labels, aux)` recovers the
row boundaries, and the model cannot predict them — that count is a property of
the answer.

**So a per-example weight cannot be applied inside `training_loss` here.** The
equivalent operation is a per-row **gradient scale** in the forward, applied
*after* the head:

```python
logits = w.view(-1, 1, 1) * logits + (1 - w.view(-1, 1, 1)) * logits.detach()
```

forward value exactly `logits`, gradient to every upstream parameter scaled by
`w`. It is ordinary arithmetic — no custom autograd `Function`, no
participant-controlled backward — and `--grad-equiv` verifies it agrees with a
weighted loss to 6.9e-08. A useful side effect: the logged training loss stays
**unweighted**, so the curriculum cannot flatter the number the runner prints.

Applying it before the head is a bug: the head (and, with a tied embedding, the
embedding) would then receive *unweighted* gradients.

---

## 5. Calibration — the fitting curve, and exactly what the plateau is

**LEGAL.** No curriculum (`beta = 0`), batch 512, 49,152 rows over 48 training
moduli, `lab/logs/CAL-lr3e-2.log`.

| step | 1 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 |
|---|---|---|---|---|---|---|---|---|---|
| loss | 2.2904 | 1.7370 | 1.7340 | 1.7339 | 1.7417 | 1.7295 | 1.7337 | 1.7315 | 1.7335 |
| `train_exact` (soft) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `dacc` (soft) | 0.292 | 0.303 | 0.312 | 0.309 | 0.309 | 0.312 | 0.315 | 0.317 | 0.315 |
| `local_ce` | 2.195 | 3.673 | 4.044 | 4.635 | 5.127 | 5.526 | 5.729 | 5.727 | 5.766 |

Two learning rates on either side, to 2,000 steps, land in the same place:
`lr 1e-2` loss 1.7444 / `dacc` 0.3092, `lr 1e-1` loss 1.7405 / `dacc` 0.3114.
`local_ce` is the only thing that separates them, and it separates them in the
wrong direction: 4.36 / 5.13 / 7.70 at lr 1e-2 / 3e-2 / 1e-1.

**What the plateau is, exactly.** At step 3,500 the per-bucket soft digit
accuracies are

| bucket | 16-bit | 18-bit | 20-bit |
|---|---|---|---|
| trained model, step 3,500 | 0.3831 | 0.3159 | 0.2504 |
| **measured constant-zero predictor** | **0.3792** | **0.3031** | **0.2350** |
| excess | +0.004 | +0.013 | +0.015 |

**The model converges to the digit marginals and stops**, to within 0.4–1.5
points. Loss is flat from step 500 to step 4,000; `train_exact` is 0.000 at
every one of nine logged points. This is the same plateau the hosted Hard run
showed ("digit marginals learned, nothing more") and the same one
`hard/digitalu-hf1` measured for the reference transformer, flat across
**40,000** steps on `hf1` itself.

**So a 1,500-step screen here is post-plateau, not pre-fitting.** That
distinction is the whole point of `RESUME.md`'s rule, and it is why every cell
below is compared at step 1,500 against the baseline seed band read from the
same logged step.

Two further calibration facts:

* **Under hard (argmax-snapped) states nothing is learned at all.**
  `train_exact_hard` is **0.000** at every logged point, and `dacc_hard` wanders
  in a 0.07–0.23 band across training against a `--lr 0` floor of **0.112** and
  a chance level of 0.10 — it dips *below* its own initialisation at steps
  1,000–2,000 in both seeds (0.071 / 0.097) before drifting back up. I am
  reporting the band rather than an endpoint, because either endpoint alone
  supports a story the other contradicts. The one stable statement: the soft
  channel carries all 0.31 of the model's digit accuracy and the discrete
  channel carries none of it.
* **`local_ce` moves monotonically away from the cliff at every learning rate**
  — 2.195 (random init) → 5.77, against a basin cliff of 0.006. The legal
  objective's gradient points away from the discrete solution from step 1. This
  is `alu-optimizer`'s result, reproduced at S=7 with three modulus sizes.

---

## 6. The source-difficulty ladder — is there anything at the easy end?

Before sweeping schedules, "measure the basin before optimising it" (BRIEF2
§6.6). A curriculum needs something correct at the source. `alu-credit`'s
chain-length curriculum died on exactly this — *"the short chain fits the
degenerate solution faster; it does not find the algorithm"*.

Same graph (S=7, 183 sequential soft steps, one set of digit tables), varying
only the modulus size the model is trained on, from far below `hf1`'s floor.

**10-bit** — `LAD-b10`, 4 moduli, a **3-decimal-digit** problem, six bits below
`hf1`'s smallest modulus. Trivial `dacc` floor 0.6305. **LEGAL**, 3,000 steps.

| step | 1 | 500 | 1000 | 1500 | 2000 | 3000 (FINAL) |
|---|---|---|---|---|---|---|
| loss | 2.272 | 0.868 | 0.840 | 0.819 | 0.807 | 0.784 |
| `train_exact` (soft) | 0.0067 | 0.0075 | 0.0183 | 0.0150 | 0.0142 | 0.0183 |
| **`train_exact_hard`** | 0.000 | 0.000 | 0.0008 | 0.000 | 0.0008 | **0.000** |
| `dacc` (soft) | 0.630 | 0.666 | 0.680 | 0.691 | 0.697 | 0.704 |
| `local_ce` | 2.271 | 5.005 | 5.734 | 6.374 | 7.559 | **8.914** |

FINAL, all splits: **`held_x` exact 0.000, `held_n` exact 0.000**,
`ood_n` hard 0.0078 (**1 of 128**, the variance floor). Structure scores
`mul_fn` 0.27 / `add_shift` 0.275 / `sub_shift` 0.225 — the random baseline is
0.23–0.28, so **the tables have not moved off chance**. Output diversity 0.278
against a measured constructed reference of ~0.99.

**Even at 10 bits the easy end is not learnable.** The 1.8% soft train exact is
carried entirely by the continuous channel: it is 0.000 under hard states, 0.000
on held-out `x` at the *same* modulus, and the tables are at chance. `local_ce`
climbs from 2.27 to 8.91 — the objective walks away from the discrete solution
just as fast at 10 bits as at 20.

**The whole ladder, at a matched 1,500 steps.** Each cell trains on ONE bit size
with the same graph and the same tables. `dacc` is only comparable *down* a
column, never across rows — the trivial floor moves with the number of
structurally-zero slots — so the excess over each rung's own measured floor is
the column to read. **LEGAL.**

| rung | `Tmul` cells / example | trivial `dacc` floor | soft `dacc` | **excess** | soft `train_exact` | **`train_exact_hard`** | `held_n` exact |
|---|---|---|---|---|---|---|---|
| 10-bit (3 digits) | — | 0.6305 | 0.6798 | **+0.049** | 0.0183 | **0.0008** (1 ex.) | **0.000** |
| 12-bit | 15.58 | 0.5384 | 0.5563 | +0.018 | 0.0013 | **0.000** | **0.000** |
| 16-bit | 21.47 | 0.3860 | 0.3871 | +0.001 | 0.000 | **0.000** | **0.000** |
| 20-bit | 27.54 | 0.2475 | 0.2466 | −0.001 | 0.000 | **0.000** | **0.000** |

*(matched at step 1,000; the 10-bit rung also ran to 3,000 and ends at soft
`dacc` 0.704 / `train_exact` 0.0183 / `train_exact_hard` **0.000**.)*

The excess over the trivial predictor **does** grow monotonically as the modulus
shrinks — so the difficulty axis is real, and a curriculum has something to
climb. But it is climbing the wrong thing: at **every** rung, including a
3-digit one, `train_exact_hard` is 0.000, held-out exact is 0.000, and the
gauge-invariant structure scores sit on their random baseline. What gets easier
with a smaller modulus is *how much of the answer is structurally zero*, not
*how findable the algorithm is*.

**This is `alu-credit`'s chain-length result again, on a different axis**: the
easy end fits the degenerate solution faster and does not find the algorithm.
A curriculum needs something correct at the source; there isn't one at any
modulus size this task can present.

---

## 7. The sweep — schedule shape, strength, axis

All cells: 1,500 steps, lr 3e-2, batch 512, seed 0, same 48 training moduli and
49,152 rows, `--mode fixed_step`-equivalent so contention cannot corrupt the
comparison. **Compared at a matched logged step against the three-seed baseline
band read from that same step.** The weight ratio the slope buys, measured on a
worked batch: `beta` 0.5 / 1 / 2 / 4 → **3.1x / 9.4x / 87x / 7,664x** between a
16-bit and a 20-bit example at step 0, annealing to 1.0x.

### 7.1 The baseline band (LEGAL, `beta` = 0)

| seed | soft `dacc` 16/18/20 @1000 | @1500 | soft `train_exact` | **`train_exact_hard`** | `held_n` exact | `local_ce` @1000 |
|---|---|---|---|---|---|---|
| 0 | 0.3817 / 0.3107 / 0.2435 | 0.3797 / 0.3072 / 0.2387 | 0.000 | **0.000** | 0.000 | 4.044 |
| 1 | 0.3820 / 0.3110 / 0.2287 | 0.3783 / 0.3131 / 0.2398 | ≤0.0007 | **0.000** | 0.000 | 4.619 |
| 2 | 0.3766 / 0.3104 / 0.2352 | 0.3851 / 0.3112 / 0.2315 | 0.000 | **0.000** | 0.000 | 5.179 |
| **band @1000** | **0.377–0.382 / 0.310–0.311 / 0.229–0.244** | | 0.000–0.0007 | **0.000** | 0.000 | 4.04–5.18 |
| *trivial floor* | *0.3792 / 0.3031 / 0.2350* | | | | | *2.20 at `--lr 0`* |

Seed 0 reproduces `CAL-lr3e-2` to every decimal at both steps (loss 1.7339,
`local_ce` 4.63542 at 1,500) — the probe is deterministic given a seed, so a
difference between two cells is a real effect of the flag. The *within*-run
oscillation is not: `d20` moves by ~0.01 between consecutive logged steps of the
same run, which sets the resolution of this screen.

### 7.2 The cells, at a matched step 1,000

Every row **LEGAL**. Matched at **step 1,000** because that is the last step
every cell in the sweep has logged (the anneal completes at 750, so every cell is
post-anneal *and* post-plateau there); the longer runs' step-1,500 rows say the
same thing and are in `lab/logs/`. `train_exact_hard` is the headline.

| cell | `beta_n` | `beta_x` | schedule | soft `dacc` 16 / 18 / 20 | soft `train_exact` | **`train_exact_hard`** | `held_n` exact hard | `local_ce` |
|---|---|---|---|---|---|---|---|---|
| **band, `beta`=0, 3 seeds** | 0 | 0 | — | 0.377–0.382 / 0.310–0.311 / **0.229–0.244** | 0.000 | **0.000** | **0.000** | 4.04–5.18 |
| `C-N-linear-b1` | 1 | 0 | linear | 0.3809 / 0.3123 / 0.2350 | 0.000 | **0.000** | **0.000** | 3.942 |
| `C-N-linear-b4` | 4 | 0 | linear | 0.3786 / 0.3107 / 0.2409 | 0.000 | **0.000** | **0.000** | 3.893 |
| `C-N-linear-b1`, seed 1 | 1 | 0 | linear | 0.3856 / 0.3153 / 0.2335 | 0.000 | **0.000** | **0.000** | 4.496 |
| `C-N-linear-b1`, seed 2 | 1 | 0 | linear | 0.3800 / 0.3096 / 0.2424 | 0.000 | **0.000** | **0.000** | 5.513 |
| `C-N-step-b1` | 1 | 0 | **step** | 0.3817 / 0.3085 / 0.2350 | 0.000 | **0.000** | **0.000** | 3.979 |
| `C-N-exp-b2` | 2 | 0 | **exponential** | 0.3806 / 0.3061 / 0.2330 | 0.000 | **0.000** | **0.000** | 3.995 |
| `C-N-const-b1` | 1 | 0 | **constant, no anneal** | 0.3814 / 0.3083 / 0.2321 | 0.000 | **0.000** | **0.000** | 4.005 |
| `C-X-linear-b1` (magnitude axis only) | 0 | **1** | linear | 0.3744 / 0.3142 / 0.2478 | 0.000 | **0.000** | **0.000** | 4.196 |
| `C-NX-linear-b1` | 1 | **1** | linear | 0.3800 / 0.3115 / 0.2358 | 0.000 | **0.000** | **0.000** | 4.136 |
| `D-only16` (extreme) | ∞ | 0 | none | 0.3817 / **0.1507** / 0.2270 | 0.000 | **0.000** | **0.000** | 3.561 |
| `D-only20` (mirror) | −∞ | 0 | none | **0.3334** / **0.2167** / 0.2444 | 0.000 | **0.000** | **0.000** | 4.936 |
| *trivial floor* | | | | *0.3792 / 0.3031 / 0.2350* | | | | *2.20 at `--lr 0`* |

Three seeds of the canonical cell (`beta_n`=1, linear) read `d20` **0.2350 /
0.2335 / 0.2424** against the baseline band 0.229–0.244 — dead centre. **Ten
cells across four schedule shapes, four strengths, three axes and three seeds,
and not one leaves the band.**

**The parameter-level view agrees.** Gauge-invariant structure scores at the end
of each run, against a random baseline of **0.23–0.28**:

| cell | `mul_fn` | `mul_gauge` | `add_shift` | `sub_shift` |
|---|---|---|---|---|
| `B0` (3 seeds) | 0.26 / 0.32 / 0.32 | 0.5 / 0.5 / 0.7 | 0.300 / 0.265 / 0.275 | 0.278 / 0.217 / 0.294 |
| `C-N-linear-b1` | 0.30 | 0.5 | 0.290 | 0.272 |
| `C-N-linear-b4` | 0.34 | 0.5 | 0.265 | 0.278 |
| `C-N-exp-b2` | 0.27 | 0.4 | 0.275 | 0.294 |
| `C-NX-linear-b1` | 0.24 | 0.6 | 0.270 | 0.289 |
| `D-only16` | 0.29 | 0.5 | 0.270 | 0.289 |
| `D-only20` | 0.26 | 0.4 | 0.325 | 0.289 |
| **`TF-only16` (DIAGNOSTIC)** | **1.000** | **1.000** | **1.000** | **1.000** |

Every legal cell sits on the random baseline; the illegal one is exact. There is
no *partial* progress for a curriculum to accelerate.

### 7.3 Held out on UNSEEN MODULI, per modulus size — the number the brief asked for

`hf1`'s `test` split is *disjoint moduli*, so this is its offline analogue: 24
moduli (8 per bit size) that appear in no training row, with operands that appear
nowhere. FINAL at each cell's own endpoint (1,500 steps for every row here,
including the baseline `B0-s2`). **Never pooled.**

| cell | `held_n` **exact** 16 / 18 / 20 | `held_n` soft `dacc` 16 / 18 / 20 | `held_n` diversity |
|---|---|---|---|
| `B0-s2` (baseline) | **0.000 / 0.000 / 0.000** | 0.3803 / 0.3039 / 0.2327 | 0.089 |
| `C-N-linear-b1` | **0.000 / 0.000 / 0.000** | 0.3800 / 0.2938 / 0.2377 | 0.173 |
| `C-N-linear-b4` | **0.000 / 0.000 / 0.000** | 0.3680 / 0.2944 / 0.2347 | 0.132 |
| `C-N-exp-b2` | **0.000 / 0.000 / 0.000** | 0.3677 / 0.2972 / 0.2302 | 0.120 |
| `C-N-linear-b1`, seed 1 | **0.000 / 0.000 / 0.000** | 0.3890 / 0.2983 / 0.2274 | 0.150 |
| `C-N-linear-b1`, seed 2 | **0.000 / 0.000 / 0.000** | 0.3689 / 0.2955 / 0.2341 | 0.100 |
| `C-N-step-b1` | **0.000 / 0.000 / 0.000** | 0.3686 / 0.2988 / 0.2436 | 0.163 |
| `C-X-linear-b1` | **0.000 / 0.000 / 0.000** | 0.3842 / 0.3008 / 0.2411 | 0.113 |
| `C-NX-linear-b1` | **0.000 / 0.000 / 0.000** | 0.3761 / 0.2924 / 0.2383 | 0.230 |
| `D-only16` | **0.000 / 0.000 / 0.000** | 0.3823 / **0.2296** / 0.2162 | 0.368 |
| `D-only20` | **0.000 / 0.000 / 0.000** | 0.3624 / **0.2319** / 0.2076 | 0.613 |
| *trivial floor (held)* | | *0.3770 / 0.2991 / 0.2363* | |
| *exact solution* | *1.000 / 1.000 / 1.000* | *1.000 / 1.000 / 1.000* | *0.995* |
| *`--lr 0`* | *0.000 / 0.000 / 0.000* | *0.104 / 0.098 / 0.106* | *0.998 (hard) / 0.0007 (soft)* |

**Exact accuracy on unseen moduli is 0.000 in every bucket of every cell**, and
the soft digit accuracies are the held-out trivial floor to within ±0.01 —
except where the extremes push a down-weighted bucket *below* it. The diversity
column is read against a **measured** constructed reference of **0.995**: every
trained cell is well below it (0.09–0.61), i.e. the models are partially
collapsed relative to the exact solution, and the ones that look most "diverse"
here are the ones trained on one bucket.

**Reading.**

* **`train_exact_hard` is 0.000 in every cell, and so is held-out exact.** The
  ranked quantity does not move for any schedule shape (step / linear /
  exponential / constant), any strength (`beta` 1 → 4, a 9x → 7,664x weight
  ratio), or any axis (modulus size / operand magnitude / both). That is the
  headline and nothing below changes it.
* **No cell's per-bucket digit accuracy leaves the baseline band.** The
  three-seed band on `d20` at step 1,000 is 0.229–0.244; every curriculum cell
  lands inside it. **`d20` also oscillates by ~0.01 within a single run**
  (`C-N-linear-b1` reads 0.2321 / 0.2350 / 0.2461 at steps 750 / 1,000 / 1,500;
  `B0-s0` reads 0.2344 / 0.2435 / 0.2387) — larger than any difference between
  cells, which is the honest resolution limit of this screen.
* **`local_ce` looks like it improves, and that is the trap — twice over.** At
  matched seed 0 every curriculum cell reads *lower* `local_ce` than the seed-0
  baseline (3.89–4.20 against 4.04), and `D-only16` lowest of all at 3.56. Two
  reasons not to believe it. (i) The **same** cell across three seeds spans
  3.94 / 4.50 / 5.51, wider than the effect — `local_ce` here is a seed
  property, not a curriculum property; the baseline spans 4.04 / 4.62 / 5.18 and
  the two ranges overlap almost completely. (ii) Even if it were real, `--lr 0`
  reads **2.20**, so any leftward shift is **partial regression toward
  initialisation** — the last row of `RESUME.md`'s fooling table. The cells that
  "improve" `local_ce` most are the ones that train least.
* **Both extremes damage the buckets they down-weight, symmetrically.**
  `D-only16` leaves its own bucket exactly at its trivial floor (`d16` 0.3817
  against 0.3792 — the same +0.003 the baseline gets for free) while driving
  `d18` to **0.1507**, half its own 0.3031 floor. The mirror image `D-only20`
  drives `d16` to 0.3334 and `d18` to 0.2167. **This is the failure mode the
  brief asked me to check for, and it is the dominant effect of a strong
  curriculum**: the model fits the *digit marginals* of whichever bucket it is
  fed, and those marginals are wrong for every other bucket. Nothing
  modulus-independent is being learned to transfer.

---

## 8. Why it cannot work here — the premise, corrected and measured

### 8.0 A smaller modulus does NOT shorten the chain

The method as ranked says *"smaller moduli mean shorter carry chains and a
strictly easier instance of the same function."* The second half is true; **the
first half is false for any fixed-slot ALU**, and it is worth saying plainly
because it is the load-bearing half.

The model's slot count `S` is fixed at build time and has to be sized for the
largest modulus it will ever see. A 16-bit example is then a 20-bit-shaped
computation with leading zeros: **the same 183 sequential softmax steps, the
same 14 reductions, the same tables.** Nothing about the graph gets shorter, and
under this branch's `redall` schedule (§2) the small moduli actually need
*every* reduction, so if anything they exercise more of the reduction path, not
less.

What is genuinely easier is only the *values*: fewer significant digits. The two
measurements below ask what that buys, and the answer is: less signal, not
cheaper signal.

Both are DIAGNOSTIC (they start from the constructed solution or read the
inputs directly), and both are about the *premise* rather than the tuning.

### 8.1 A small-modulus example carries LESS signal, not the same signal cheaper

`--coverage`, measured on the training cohort. The `Tmul` table has 100 cells
and is shared by every example at every modulus size.

| | 16-bit | 18-bit | 20-bit |
|---|---|---|---|
| mean significant decimal digits of `x` | 4.39 | 4.91 | 5.33 |
| **mean distinct `Tmul` cells touched per example** | **21.47** | **25.13** | **28.19** |
| cells covered by the bucket as a whole | 100 | 100 | 100 |

Every bucket covers the whole table *eventually*, so this is not a coverage
ceiling. It is a **density** statement: a 16-bit example constrains 24% fewer
shared cells per gradient than a 20-bit one, because its top two slots are
structurally zero. Upweighting the easy end upweights the thinner gradient.

### 8.2 A small-modulus example is also LESS sensitive to a wrong cell

`--basin`, 3 reps, 768 examples, constructed solution with `k` of the 200
`Tmul` argmaxes flipped, hard states. This is `alu-relational`'s and
`matrix-scan`'s instrument asked a new question: *is the end-of-chain label a
stronger signal at the easy end?*

| k (of 200) | exact all | **16-bit** | **18-bit** | **20-bit** | `dacc` 16/18/20 |
|---|---|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 / 1.000 / 1.000 |
| 1 | 0.7444 | **0.7857** | 0.7302 | **0.7181** | 0.894 / 0.843 / 0.831 |
| 2 | 0.5048 | 0.5318 | 0.4760 | 0.5071 | 0.721 / 0.647 / 0.648 |
| 5 | 0.3498 | 0.3876 | 0.3735 | 0.2896 | 0.643 / 0.588 / 0.509 |
| 10 | 0.1293 | 0.1601 | 0.1349 | 0.0940 | 0.482 / 0.390 / 0.331 |
| 20 | 0.0035 | 0.0026 | 0.0065 | 0.0013 | 0.379 / 0.296 / 0.234 |
| 50 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.373 / 0.301 / 0.234 |
| 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.370 / 0.302 / 0.239 |
| 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.379 / 0.291 / 0.235 |

Two readings, and they point the same way as §8.1:

1. **The easy end is systematically *less* disturbed by a wrong product cell**
   (0.786 vs 0.718 at k=1, and the ordering holds at every k where anything is
   left). A 16-bit example is a *quieter* error signal, so the curriculum
   upweights the examples that complain least.
2. **The label's basin is dead by k = 20 of 200, at every modulus size, and by
   k = 20 the per-bucket digit accuracy has already fallen back to its trivial
   floor** (0.379 / 0.296 / 0.234 against 0.379 / 0.303 / 0.235). A random
   initialisation is **176.6 wrong cells of 200** (measured, 8 seeds) —
   **nine times outside the basin** — and no bucket's basin is meaningfully
   wider than another's. There
   is no modulus size at which the end-of-chain label can see the solution from
   random init.

**Scope of this instrument.** It measures how many wrong cells the label can
still *see* (forward sensitivity), not how many training *repairs* — that is
`matrix-scan`'s measurement and it is a different, strictly harder quantity.
Sensitivity is an upper bound on repair: an objective cannot fix what it cannot
distinguish. So "0.0035 exact at k=20" is the optimistic end, and it is already
nine times inside where random init sits. Re-weighting *which* examples supply
that label cannot move it: the weighting changes the mixture, not the
conditioning.

---

## 9. The evaluator cells on `hf1`

**These are a MECHANISM check, not the scientific test**, and `hard/digitalu-hf1`
§0.3 says why in one line: the reference-width transformer *never leaves the
pre-fitting region on `hf1`* — `train_exact` 0.000 flat across 40,000 steps at
0.83 params/row. Any accuracy comparison at this width on `hf1` is
uninterpretable. What these rows do establish is that the curriculum machinery
is legal, lints, runs inside the evaluator's own loop on real `hf1` prompts, and
breaks nothing.

`lab/manifests/lab_hf1_fs2000_s74.json`, seed 74, `--mode fixed_step`.

| submission | curriculum | lr | `MAX_T` | `OOD_N_MAX_T` | rungs (seen N) | `test`/`ood_t`/`ood_n_t` | steps |
|---|---|---|---|---|---|---|---|
| `hard-curriculum-hf1-lr0` | off | **0** | 0 | 0 | **all 0.000** | 0.000 / 0.000 / 0.000 | 2000 |
| `hard-curriculum-hf1-off` | off (`beta`=0) | 1e-3 | 0 | 0 | all 0.000 except T=4 at 0.001 (**1 of 768 = the variance floor**) | 0.000 / 0.000 / 0.000 | 2000 |
| `hard-curriculum-hf1` | **`beta`=1, linear** | 1e-3 | 0 | 0 | **all 0.000** | 0.000 / 0.000 / 0.000 | 2000 |
| `hard-curriculum-hf1-ids` | `beta`=1, linear, **`ids` signal** (FLAGGED) | 1e-3 | 0 | 0 | **all 0.000** | 0.000 / 0.000 / 0.000 | 2000 |
| `hard-curriculum-hf1-b4` | `beta`=4, linear | 1e-3 | 0 | 0 | **all 0.000** | 0.000 / 0.000 / 0.000 | 2000 |

**Through the evaluator all five submissions are indistinguishable**: `MAX_T` 0
and `OOD_N_MAX_T` 0 for every one, all rungs 0.000 on both ladders, and
`mean_exact_accuracy` spanning **0.0 to 3.70e-05** — one example in 27,000. The
`--lr 0` control, the no-curriculum control, the curriculum at `beta`=1 and
`beta`=4, and the prompt-decoding difficulty signal are all the same run as far
as the ranked metric is concerned. That is `plan2/sequential-rnn`'s finding
reproduced on `hf1`, now with three curriculum arms.

The `--lr 0` row is the mandatory control and it agrees with
`hard/digitalu-hf1`'s independently-measured floor for the same architecture
class (0.000 at every rung on both ladders). The single 0.001 in the `off` row
is **one example of 768** and sits inside that floor. A smoke run of the same
submission on `hf1s` at 30 steps also completed cleanly
(`lab/archive.jsonl`, tag `curric-smoke`), which is what establishes that the
forward-side gradient scale passes `lint_submission_source` and runs inside the
evaluator's own loop.

---

## 10. Compliance

| item | status |
|---|---|
| `data/generated/` | **never opened**. Every modulus and operand in the probe is self-generated from the *generator source* (`data/squaring_mod.py:1259`, `p_bits = bits//2`). Independent check: my enumeration finds **148 / 543 / 1718** balanced semiprimes at 16 / 18 / 20 bits, and `hard/digitalu-hf1` measured `hf1`'s own pools at 133+15 / 488+55 / 1546+172 — **the same three numbers**. |
| difficulty signal | derived at runtime from the input tensors (`log10` of the modulus and of the operand, from the one-hot digit slots), never from a dataset field |
| custom training loop / backward | none. The curriculum is arithmetic in the forward; the evaluator still does one forward, one `backward()`, one `step()` |
| step counter | **non-persistent buffer** (`benchmark/api.py:26-42` excludes it from the 5e8 ceiling) |
| `--construct`, `--basin`, `local_ce` | **DIAGNOSTIC** — they set tables to the truth or replay a constructed tape. Never in a submission. |
| `--only-bits` | **LEGAL** (a 0/1 weight computed from the input) but reported as the curriculum's extreme point, not as a candidate |
| `_CURRIC_SRC="ids"` | **FLAGGED compliance-uncertain** (it decodes the prompt format to build the difficulty scalar). Not the default; `"len"` uses only the input's length. |
| hosted service | nothing submitted |

---

## 11. Exact commands

```bash
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
D="--n-mod 16 --n-mod-held 8 --n-x 1024 --n-held-x 64"

# gates
$V lab/probe_curric.py --construct   $D --tag GATE-construct
$V lab/probe_curric.py --grad-equiv  $D --tag GATE-gradequiv
$V lab/probe_curric.py --basin       $D --basin-n 768 --basin-reps 3 --tag GATE-basin
$V lab/probe_curric.py --steps 100 --lr 0 $D --tag GATE-lr0

# calibration (fitting curve)
$V lab/probe_curric.py $D --steps 12000 --lr 3e-2 --log-every 500 --tag CAL-lr3e-2

# sweep and ladder
bash lab/curric_sweep2.sh a   # ... through g

# the curriculum's illegal ceiling (DIAGNOSTIC)
bash lab/curric_run.sh TF-only16 --steps 800 --lr 3e-2 --tf 1.0 --task-w 0 \
     --train-bits 16 --log-every 100 --seed 0
bash lab/curric_run.sh TF-all    --steps 800 --lr 3e-2 --tf 1.0 --task-w 0 \
     --log-every 100 --seed 0

# evaluator cells on hf1 (archived to lab/archive.jsonl)
bash lab/curric_eval.sh
```

**Files.** `lab/probe_curric.py` (the probe and every gate),
`lab/curric_run.sh` / `curric_sweep2.sh` / `curric_ladder.sh` (drivers),
`lab/curric_at.py` and `lab/curric_table.py` (matched-step renderers),
**`lab/runs/curric/`** — every run's log plus `curric.jsonl` (`lab/logs` is
gitignored repo-wide, so the archive copy lives here), `lab/archive.jsonl`
(evaluator rows), `submissions/hard-curriculum-hf1{,-off,-lr0,-ids,-b4}/`.

**One harness incident, recorded rather than hidden.** I patched
`curric_sweep2.sh` while four of its shells were mid-file; bash resumed at a
stale byte offset and each of them executed a spurious extra cell that
overwrote `C-N-step-b1.log` four times over. No *intended* cell was skipped.
The clobbered file is kept as `C-N-step-b1.CLOBBERED.log` and the cell was
re-run alone — **it reproduced the recorded numbers exactly** (soft `dacc`
0.3817 / 0.3085 / 0.2350, `local_ce` 3.97906 at step 1,000), which is what the
probe's determinism predicts. Do not edit a shell script that is currently
executing.

---

## 12. The curriculum's ILLEGAL ceiling — does easy-to-hard transfer work at all?

**DIAGNOSTIC (rules 2 and 7).** BRIEF2 §6.6: measure a signal's illegal ceiling
before building its legal version. A curriculum has two independent
preconditions — a *learnable source* and *transfer from source to target* — and
§6 kills the first. This isolates the second, by handing the model the one
signal that is known to work on `hf1`-faithful data (`hard/digitalu-hf1` §3:
teacher forcing on a constructed register tape reaches 1.000) and then
**restricting it to 16-bit examples only**.

`--tf 1.0 --task-w 0 --train-bits 16`, 800 steps, lr 3e-2, P=1, seed 0. The
18- and 20-bit rows below are examples the model **never trained on**.

| step | `local_ce` | soft exact **16-bit** | soft exact **18-bit** | soft exact **20-bit** |
|---|---|---|---|---|
| 200 | 0.0880 | 0.0354 | 0.0247 | 0.0000 |
| 300 | 0.0770 | 0.2790 | 0.0684 | 0.0719 |
| 400 | 0.0098 | 0.9096 | 0.8080 | **0.7824** |
| 500 | 0.0060 | 0.9332 | 0.8992 | **0.9002** |
| 600 | **0.0053** | 0.9450 | 0.8992 | **0.9002** |

FINAL at 800 steps, **argmax-snapped (hard) states**, and every split below
except `train` is operands the model never saw — `held_n` is *unseen moduli*
(`hf1`'s `test`), `ood_n` is *unseen modulus sizes* 17/19/21:

| split | all | 16-bit | 18-bit | **20-bit** |
|---|---|---|---|---|
| train (16-bit only) | **0.9076** | 0.9312 | 0.8954 | **0.8962** |
| held `x`, train moduli | **0.9121** | 0.9357 | 0.9085 | **0.8904** |
| **held `n` (unseen moduli)** | **0.8776** | 0.8867 | 0.9043 | **0.8418** |
| `ood_n` (17/19/21-bit) | **0.8672** | 0.8848 (17b) | 0.8613 (19b) | 0.8555 (21b) |

`local_ce` **0.00549** — below the 0.006 cliff. Structure scores
**`mul_fn` 1.000, `mul_gauge` 1.000, `add_shift` 1.000, `sub_shift` 1.000**: the
digit tables are *exactly right*. `state_sharpness` 0.9886, output diversity
0.9954 against the measured constructed reference of 0.9954.

**Taught from 16-bit examples alone, the tables come out exact and carry to
20-bit operands, to unseen moduli, and to modulus sizes never trained on.**

**And the restriction *helped*.** The control `TF-all` — identical signal,
identical budget, all three sizes in the batch — reads hard exact **0.3796**
train / **0.3789** held-`n`, against `TF-only16`'s 0.9076 / 0.8776. At matched
steps, **training on the easy end alone is 2.4x better on the hard metric than
training on the mixture, and generalises to the hard end anyway.** That is the
curriculum hypothesis, confirmed — under a signal that is illegal.

The mechanism is visible and it is the `(1-eps)^L` law at L = 183 chained ops:
*both* runs reach exactly correct tables by the structure scores
(`mul_fn`/`mul_gauge`/`add_shift`/`sub_shift` = 1.000 in both), and the entire
difference is residual **per-op** error — `local_ce` **0.00549** for
`TF-only16` (below the 0.006 cliff) against **0.02032** for `TF-all`. A 3.7x
lower per-op error over 183 steps is the whole 2.4x. So what concentrating on
the easy end buys is *rate of per-op error reduction per optimizer step*, which
is exactly the quantity a curriculum is supposed to buy.

(Caveat, stated rather than smoothed: matched in *steps*, not in per-bucket
examples — `TF-only16` spends every gradient on one bucket. The defensible claim
is the narrow one: restricting to the easy end cost nothing at the hard end and
converged faster per step.)

**So the curriculum's transfer step is not merely adequate, it is free, and its
concentration is a genuine win when the source is learnable.** Combined with §6,
the diagnosis is unambiguous —

> **The curriculum fails on the source, never on the transfer.** There is no
> modulus size at which the *legal* objective produces tables worth annealing
> away from, and no schedule can weight its way to one.

**This is the one place in this report that carries a forward-looking positive.**
If any future architecture makes the legal objective produce even a partly
correct source — `hard/add-only`'s adder-only transducer is the candidate, since
adder laws repair 50/400 cells against the label's ~0 — then a modulus-size
curriculum is worth **re-testing there**, because both of its preconditions
other than the source are measured good here.

---

## 13. Scope — what this does and does not cover

Stated up front so nobody over-reads the null.

1. **`T = 1`.** The probe measures **one squaring step**. `hf1` trains at
   `T ∈ {4,8,16}`. This follows `RANKING.md`'s organising fact ("the failure is
   one squaring on unseen operands"; composition is solved and certified) and
   every ALU-family probe before it. A curriculum could in principle act on `T`
   as a third axis — but `T` is *not* a difficulty axis for a weight-tied step,
   and `alu-compose` already showed downward generalisation in `T` works by
   construction.
2. **48 training moduli, not 2,167.** 16 per bit size out of the 148 / 543 /
   1718 that exist, and 49,152 rows against `hf1`'s 243,000. The modulus
   *universe* is reproduced exactly (§10); the *sample* is smaller. For a
   modulus-independent readout more moduli means more constraints, so this is
   the direction that would make the null *stronger*, not weaker.
3. **P = 1, no replica population.** `alu-population`'s instrument is available
   and legal, and I did not spend it here: it characterised itself as *"the
   right instrument for a stochastic obstruction and the wrong one for a
   systematic one"*, and §8 reads systematic — the basin is dead at k=20 of 200
   at **every** modulus size, so extra draws sample the same dead region.
4. **The evaluator rows are a mechanism check only** (§9), for the reason
   `hard/digitalu-hf1` §0.3 gives.

---

## 14. Verdict

**Null, and the evidence is calibrated.** Recommendation: **spend no further
budget on a curriculum over modulus size or operand magnitude for this task.**

The argument in four steps, each a measurement rather than an expectation:

1. **The mechanism is sound and it is cheap.** A per-example curriculum *is*
   expressible under the evaluator's fixed loop — not in `training_loss`, whose
   arguments arrive flattened by a ragged mask (§4), but as a per-row gradient
   scale in the forward, which agrees with a weighted loss to 6.9e-08. The
   schedule fits in a non-persistent buffer. Nothing about the contract blocks
   this idea.
2. **The transfer it assumes works perfectly, and the concentration even
   helps.** Under a per-op (illegal) signal restricted to 16-bit examples
   *only*, the tables come out **exactly right** (`mul_fn`/`add_shift`/
   `sub_shift` all 1.000) and score **0.896 hard exact on 20-bit** examples,
   **0.878 on unseen moduli** and **0.867 at unseen modulus sizes** — while the
   same signal spread over all three sizes reaches only 0.380 at matched steps
   (§12). Easy-to-hard transfer is not the problem; it is the one thing in this
   report that works.
3. **The source it assumes does not exist.** Under the legal end-of-chain label,
   `train_exact_hard` is **0.000 at every modulus size on the ladder**,
   including a 10-bit, 3-decimal-digit problem six bits below `hf1`'s floor
   (§6), with gauge-invariant structure scores on their random baseline and
   `local_ce` climbing *away* from the cliff at every rung. There is nothing
   correct at the easy end to anneal away from.
4. **And the premise is inverted.** With a fixed slot count a small modulus does
   not shorten the chain at all (§8.0); what it does is make the example touch
   **21.5 of 100** shared product cells instead of 28.2, and make its answer
   **less** sensitive to a wrong cell (0.786 vs 0.718 exact at one corrupted
   cell). Upweighting the easy end upweights the thinner, quieter gradient. The
   label's basin is dead by 20 wrong cells of 200 at *every* modulus size, and
   random init is ~177 wrong.

So the curriculum is not mistuned; its precondition never holds. **No schedule
between "16-bit only" and "uniform" can help, because the sweep's two endpoints
are measured and both are 0.000** — and the endpoint that concentrates hardest
on the easy set is the one that does measurable *damage* (`D-only16` drives
18-bit digit accuracy to **0.151**, half its own 0.303 trivial floor, and its
mirror `D-only20` does the same to 16- and 18-bit).

### What I would do with the budget instead

* **Rank 1 (`hard/add-only`) and rank 3 (`hard/o1-reduction`) both attack the
  quantity this branch measured as binding**: how many learned ops separate the
  objective from the tables. §8.2 adds a data point of a *related but distinct*
  kind to `matrix-scan`'s repair law — not "how many cells does the label
  repair" but "how many wrong cells can the label still *see*". At 183 learned
  ops the answer is **~10–20 of 200**, and it is the same at every modulus size.
  Random init is ~177 wrong. Nothing about *which examples* supply the objective
  changes that number; only shortening the chain does.
* **If anyone does revisit a curriculum**, the only version not closed by this
  report is one over a *graph* axis rather than a *value* axis — something that
  genuinely shortens the learned-op chain early and grows it, e.g.
  `alu-credit`'s `--r-start`/`--r-warm` on the reduction count, ported to
  `tree:quotient`. That was measured null at e1 scale and is untested here. I
  would still rank it below 1–3, for the reason in §8.2: it changes the number
  of ops, but the basin has to widen by roughly an order of magnitude, not a
  factor of two.
* **Do not port the replica population here.** §8.2 reads systematic across
  every modulus size, and `alu-population` characterised a population as the
  wrong instrument for a systematic obstruction.

### Two things worth carrying to other branches

1. **`training_loss` cannot express a per-example weight on this dataset** — the
   valid mask is ragged because the answer's digit count varies, and the row
   boundaries are unrecoverable from `(logits, labels, aux)`. Use the forward
   gradient-scale identity (§4), and put it *after* the head.
2. **A fixed-slot ALU on a mixed-modulus dataset must reduce at every place.**
   The published "reduce only for `t <= S`" schedule overflows the quotient
   alphabet for the smaller moduli; `redall` costs 14 reductions instead of 8
   and restores the 1.000 constructed ceiling at every bit size (§2).
