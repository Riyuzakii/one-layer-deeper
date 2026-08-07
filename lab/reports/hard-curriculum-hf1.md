# `hard/curriculum-hf1` — a loss-side curriculum over modulus size and operand magnitude

**Verdict: NULL, and closed for a mechanical reason rather than a tuning one.**
Draft — numbers are filled in as runs land. See §9 for the state of play.

Method #4 in `lab/RANKING.md`. Ranked below the architectural entries, and the
brief asked for a clean negative if that is what the evidence says. It is.

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

* **Under hard (argmax-snapped) states, training makes the model *worse* than
  its own initialisation.** `dacc_hard` goes 0.112 at `--lr 0` to **0.085–0.093**
  after training. The `RESUME.md` inversion ("the untrained model has scored
  higher than the trained one") reproduces here, at Hard-faithful scale, on a
  new metric.
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

*(rungs land as runs complete; b10 to step 2,000 below)*

**10-bit** (`LAD-b10`, 4 moduli, 3 decimal digits, trivial `dacc` floor 0.6305):

| step | 1 | 500 | 1000 | 1500 | 2000 |
|---|---|---|---|---|---|
| loss | 2.272 | 0.868 | 0.840 | 0.819 | 0.807 |
| `train_exact` (soft) | 0.0067 | 0.0075 | 0.0183 | 0.0150 | 0.0142 |
| **`train_exact_hard`** | 0.000 | 0.000 | 0.0008 | 0.000 | 0.0008 |
| `dacc` (soft) | 0.630 | 0.666 | 0.680 | 0.691 | 0.697 |
| `local_ce` | 2.271 | 5.005 | 5.734 | 6.374 | 7.559 |

Even at **10 bits — six bits below `hf1`'s smallest modulus, a 3-digit
problem** — soft digit accuracy sits 6 points above its trivial floor,
`train_exact` reaches 1.8% and `train_exact_hard` is **one example**, and
`local_ce` still climbs away from the cliff. There is no rung of this ladder at
which the graph finds the algorithm.

---

## 7. The sweep

*(filled in)*

---

## 8. Why it cannot work here — the two measurements that close it

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
   initialisation is ~180 wrong cells out of 200 — **nine times outside the
   basin** — and no bucket's basin is meaningfully wider than another's. There
   is no modulus size at which the end-of-chain label can see the solution from
   random init.

This is the same shape as `matrix-scan`'s closing argument (0/5 repaired at
k=20 of 337 on a 39-op graph) at 183 ops and three modulus sizes, and it is why
re-weighting *which* examples supply that label cannot help: the weighting
changes the mixture, not the conditioning.

---

## 9. State of play

*(filled in)*
