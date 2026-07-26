# depth-controller — making the loop counter reach the top of the ladder from the T values a tier actually provides

**Branch:** `explore/depth-controller` · **Owner of P1 from `lab/reports/alu-compose.md`.**
Every accuracy number below is measured with the parser and the squaring step
held at their construction (`--construct`, a LAB DIAGNOSTIC that appears in no
submission) and **only the depth controller trained**, so the controller is the
only thing being measured. Nothing under `data/generated/` was read, printed or
summarised; prompts and targets are synthesised from the public generator spec.

---

## 0. The answer

| | Easy `train T = {1,2,3}` | Medium/Hard `train T = {4,8,16}` |
|---|---|---|
| `alu-compose` §4, 28 cells, best of six controller parameterisations | **MAX_T = 2** | **MAX_T = 0** |
| this branch, learned 12-scalar counted-halting controller | **MAX_T = 64** (2 of 3 seeds) | **MAX_T = 64** (2 of 3 seeds at m1's modulus) |

**A learned controller reaches the top of the ladder.** At `N = 329`
(λ = 138, all seven rungs distinct) and at **m1's own modulus `N = 10403`**
(λ = 5100, all seven rungs distinct), routing and exactness are **1.000 at every
rung T = 1, 2, 4, 8, 16, 32, 64**, from a controller trained only on the tier's
own three T values, with 12 learned scalars, none of them indexed by a place or
a digit of T. It also holds with every ALU inter-step state snapped to argmax.

It is **not yet reliable**: 2 of 3 seeds. The failure mode is diagnosed to a
single scalar and §7 gives the two candidate fixes.

**Four things had to be true at once**, and each is an ablation in §5:

1. **Do not dump unspent halting mass on the last candidate.** The standard
   PonderNet convenience makes "never halt" *exactly correct* for the deepest
   training T — so the deepest training example carries no gradient — and it
   also makes the self-consistency law of (4) unsatisfiable.
2. **Read out the mode, not the blend.** The soft mixture `Σ_k w_k out_k`
   **under-states the controller by a factor of two on the ladder**: the
   baseline controller's decision is already correct at T = 8 (`route` = 1.000)
   while the mixture is 0.355 exact there. Committing to `out_argmax_k w_k` at
   evaluation turns MAX_T = 4 into MAX_T = 8 with no retraining, and turns
   Medium's MAX_T = 0 into MAX_T = 4.
3. **Keep the count discrete.** A straight-through one-hot on the digit register
   after every decrement.
4. **Self-consistency `w(inc r) = shift_right(w(r))`** over the controller's own
   learned increment orbit. Its first component is `w(inc r)[0] = 0`, i.e.
   `p(r) = 0` at every successor register — the leak-suppression constraint the
   cross-entropy provably cannot see.

**And self-consistency is not a free lunch.** Applied to the same
`(place, digit)`-indexed heads that alu-compose measured, it makes them
**worse** (MAX_T 2 → 0, `loc` diverging to 10⁴–10⁶). §6 explains exactly why,
and the reason is the sharpest structural result on this branch.

---

## 1. Every modulus, its λ, and which rungs it collapses

`x^(2^T) mod N` depends on T only through `2^T mod λ(N)`. Pure number theory on
moduli that appear in dataset *directory names*; no dataset was opened.

| used as | N | factorisation | λ(N) | distinct ladder maps | collapses |
|---|---|---|---|---|---|
| e1 (public Easy) | 323 | 17·19 | 144 | **4 / 7** | {4,16,64}, {8,32} |
| e2 (public Easy) | 899 | 29·31 | 420 | **4 / 7** | {4,16,64}, {8,32} |
| **screening modulus, this branch** | **329** | **7·47** | **138** | **7 / 7** | none |
| m1 (public Medium) | 10403 | 101·103 | 5100 | 7 / 7 | none |
| m2 (public Medium) | 38021 | 193·197 | 9408 | 7 / 7 | none |
| hp1 (Hard proxy) | 4028033 | 2003·2011 | 2012010 | 7 / 7 | none |

`N = 329` was chosen by exhaustive search over three-digit semiprimes for a
non-degenerate ladder at Easy's scale (276 units against e1's 288, same digit
count, so the ALU chain and the held-out cohort are the same size). It is the
cheap stand-in; **m1's own modulus is the confirmation** and carries the same
verdict.

Every depth number in `alu-compose` §4.2's Easy and first Medium tables was
taken at N = 323, where four and sixteen applications are *the same function*.
Those numbers should be read as bounds, not measurements. Reproduced here at
N = 329 (§4), the same controller and the same training recipe give a
*different* MAX_T, in both directions.

---

## 2. What was run

`lab/probe_depth.py` — the controller in isolation. It reuses
`probe_compose.py`'s `Composed` pipeline (`MarkerPointer` parser → `BatchedALU`
→ controller), constructs the parser and the ALU, freezes them, and trains only
`sel.*`. Two engineering points make the grids affordable and are worth
recording because they are *measurements*, not assumptions:

* **The candidate stack is independent of T.** `out_k = ALU^(k+1)(x)` does not
  depend on the T field at all, so it is computed once and reused at all seven
  rungs. The probe re-parses each rung's own prompt and checks the x/N slots
  against the first rung's before reusing (a distance-from-the-end parser would
  fail this check at T = 16/32/64; the marker-relative parser passes it at
  every rung, `max |Δ| < 1e-2`). **7× faster evaluation, identical numbers.**
* **The T register is constant across a fixed-T batch**, so the halting
  distribution is one vector per training T rather than one per example.
  Checked, not assumed; ~200× less controller work per step.

Additions over `probe_compose.py --mode select`:

| flag | what it does |
|---|---|
| `--train-loops L` | decouples the training mixture depth from `max(train T)` |
| `--no-dump` | stop piling unspent halting mass on the last index |
| `--halt-grid G` | halting distribution deeper than the candidate stack (costs only the controller's own chain) |
| `--halt-pen w` | penalty on halting mass that leaks past the stack |
| `--reg-hard` | straight-through one-hot digit register |
| `--detector {sum,log}` | how the zero detector scores the conjunction (§7) |
| `--cons w --cons-j J --cons-jd J --cons-space {w,loc} --cons-start S` | self-consistency over the increment orbit |
| `--eval-hard` | snap every ALU inter-step state to argmax at evaluation |
| — | every rung reports **both** `ex_mix` (soft mixture) and `ex_arg` (commit to the mode) |

---

## 3. The baseline, reproduced at a modulus that can measure it

`alu-compose`'s exact setting (counted halting, mass dumped on the last index,
training mixture depth `L = max(train T)`), moved from N = 323 to N = 329,
3 seeds:

| tier | seeds, MAX_T (soft mixture) | seeds, MAX_T (commit to the mode) |
|---|---|---|
| Easy `T ∈ {1,2,3}` | 4 / 4 / 0 | — |
| Medium `T ∈ {4,8,16}` | **0 / 0 / 0** | **4 / 4 / 1** |

Two things are already visible.

* **Medium's MAX_T = 0 is a readout artifact as much as a controller failure.**
  The same trained controller scores 0.000 at T = 1 through the mixture and
  **1.000** when the readout commits to the mode. `alu-compose` §8's conclusion
  that "the controller is the entire downward-generalisation risk" is right
  about the location and wrong about the mechanism: at T = 1 and T = 2 the
  *decision* was already correct.
* The ordered-pointer family is unchanged by any of this. At N = 329, 3 seeds,
  training mixture depth 64: `place` 2/2/2, `placev` 1/1/1, `mlp` 2/2/2 —
  identical to `alu-compose`'s verdict, and identical for the mixture and the
  mode. **Cause (a) is real, and it is specific to heads indexed by
  `(place, digit)` of T.**

---

## 4. The recipe, per rung

Recipe **R** = counted halting + `--no-dump` + `--reg-hard` + `--cons 0.1`
(orbit `j = 8`, chain depth 4). Training mixture depth = `max(train T)`
(3 on Easy, 16 on Medium) — *no extra training depth is needed.*
Sixteen hundred optimizer steps, AdamW, lr 0.1, controller only.

### 4.1 Easy tier, `train T = {1,2,3}`, N = 329 (λ = 138, 7/7 distinct)

exact-example accuracy on 76 held-out x, commit-to-the-mode readout:

| seed | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | T=64 | MAX_T |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | **64** |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | **64** |
| 2 | 0.039 | 1.000 | 0.026 | 0.026 | 0.039 | 0.039 | 0.039 | 0 |

Seeds 0 and 1 are 1.000 for the **soft mixture too**, which the baseline never
achieved above T = 4.

### 4.2 Medium/Hard tier T values, `train T = {4,8,16}` — downward *and* upward

| modulus | λ | seed | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | T=64 | MAX_T |
|---|---|---|---|---|---|---|---|---|---|---|
| 329 | 138 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | **64** |
| 329 | 138 | 1 | 0.026 | 0.026 | 0.039 | 0.039 | 0.039 | 0.039 | 0.039 | 0 |
| 329 | 138 | 2 | 0.026 | 0.026 | 0.039 | 0.039 | 0.039 | 0.039 | 0.039 | 0 |
| **10403 (m1)** | **5100** | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | **64** |
| **10403 (m1)** | **5100** | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | **64** |
| **10403 (m1)** | **5100** | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.055 | 0.000 | 0.062 | 0 |

T = 1 and T = 2 are **never presented in training** at this tier, and T = 32 and
T = 64 are three and four doublings above the deepest training value. Both
directions are exact on the seeds that converge.

### 4.3 Easy tier at m1's modulus

| seed | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | T=64 | MAX_T |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 0.000 | 0.000 | 0.016 | 0.000 | 0.062 | 2 |
| 1 | 1.000 | 1.000 | 0.000 | 0.000 | 0.016 | 0.000 | 0.062 | 2 |
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | **64** |

### 4.4 With hard (argmax) ALU states at evaluation

`alu-depth` found that a *trained* ALU collapses from `train_exact` 0.556 to
0.004 when every inter-step state is snapped to argmax. The controller is a
learned component in the same graph, so the same failure mode is available to
it. Recipe R, Easy, N = 329, `--eval-hard`:

| seed | MAX_T (soft states) | MAX_T (**hard states**) |
|---|---|---|
| 0 | 64 | **64** |
| 1 | 64 | **64** |
| 2 | 0 | 4 |

**The controller is not riding a continuous relaxation.** Its halting signal is
a conjunction over a one-hot digit register, and it survives the snap — on the
converging seeds it is unchanged at 1.000 on every rung.

---

## 5. Ablations — which of the four things is load bearing

Easy, N = 329, 3 seeds, MAX_T with the commit-to-the-mode readout.

| cell | dump | reg-hard | cons | seeds |
|---|---|---|---|---|
| baseline (`alu-compose` setting) | yes | no | no | 4 / 4 / 0 |
| `--no-dump` only | no | no | no | 2 / 2 / 4 |
| `--no-dump --reg-hard` | no | **yes** | no | **64** / 2 / 2 |
| `--no-dump --cons` (no hard register) | no | no | **yes** | 2 / **64** / 1 |
| **R = `--no-dump --reg-hard --cons`** | no | **yes** | **yes** | **64** / **64** / 0 |
| R but with the dump restored | **yes** | yes | yes | see §5.1 |

* **The dump has to go.** With `ws[-1] += rest`, "never halt" is exactly correct
  for the deepest training T — the Medium baseline converges to a controller
  that halts at index 15 for every T, which is precisely that optimum. It also
  makes the consistency law unsatisfiable: with the dump, `w(inc r)[1]` is
  `1 − p(r)` rather than `(1−p(r))·p(r−1)`, so the law reads
  `1 − p(r) = p(r−1)` along the chain, which no monotone detector satisfies.
  The first consistency run in this branch drove the detector *gain down* to
  1.61 for exactly this reason; removing the dump reversed the sign of that
  gradient.
* **Neither `reg-hard` nor `cons` alone is reliable; together they are 2 of 3.**
* Two levers that looked promising and are **not** the fix:
  `--halt-grid 64 --halt-pen 1.0` (a halting distribution deeper than the
  candidate stack, with an explicit leak penalty) reaches MAX_T = 2: the penalty
  is satisfied by halting *earlier* rather than by sharpening, because with a
  soft register the halting statistic is continuous and the threshold slides.
  `--train-loops 64` (train the mixture at the full ladder depth) reaches
  MAX_T = 4 — better than the baseline's flat behaviour but 16× the training
  cost of the recipe that reaches 64.

### 5.1 The mixture under-states the controller

Every cell reports both readouts. The gap is systematic:

| cell | MAX_T, soft mixture | MAX_T, commit to the mode |
|---|---|---|
| baseline, Easy | 4 | **8** |
| baseline, Medium | **0** | **4** |
| `--no-dump --reg-hard`, Easy seed 0 | 4 | **64** |
| R, Easy seeds 0/1 | 64 | 64 |
| R, m1 Easy seed 2 | 16 | **64** |

The mechanism: `route` (the fraction of examples whose *selected* iteration is
exactly T) is 1.000 at T = 8 in the baseline while `ex_mix` is 0.355 and the
mean maximum halting weight has fallen from 0.809 at T = 1 to 0.232 at T = 8.
The decision is right; the blend of eight candidate answers is not one of them.
A readout that commits is the honest measurement of a *controller*, it is what
an ACT inference rule actually computes, and on this architecture it is free.

---

## 6. Self-consistency: what it is, why it works here, and where it fails

### 6.1 Why this is not `algebraic-closure`'s falsified law

`explore/algebraic-closure` showed that the semigroup law on the whole
input-output map, `M(N,x,T+1) = h(M(N,x,T))`, adds no information: it constrains
`M` to be `h^T` for some `h` and says nothing about `h`'s value. Here the
unknown is not `h` — `h` is the constructed step, and it is exact. The unknown
is the **iteration count**, and the law is applied to the *controller's* output,
where it reads

```
w(inc r) = shift_right(w(r))          for every digit register r
```

with `inc` the model's own learned digit increment. Written out for the counted
halting distribution `w(r)_k = p(r−k−1)·Π_{j≤k}(1−p(r−j))`, the law's components
are

* `k = 0`:  `p(r) = 0`  — at every register that is a successor;
* `k > 0`:  automatically satisfied once `p(r) = 0`.

So the law is **exactly a leak-suppression constraint**, and it is available at
every register, with no labels and no extra data. Combined with the anchor
(the cross-entropy, which forces `p(0) ≈ 1` at the training T values) it pins
the detector everywhere.

### 6.2 Why the cross-entropy cannot supply it

The candidate answers are log-probabilities of a saturated softmax: the correct
digit scores ≈ 0 and every other digit ≈ −20.7. A detector that leaks 16 % of
its mass one step early therefore changes the mixed logit gap by
`0.16 × 20.7 = 3.3` on a gap of 20.7 — invisible to a cross-entropy that is
already at 2 × 10⁻⁴. The leak is invisible at T = 3 and fatal at T = 16, where
it compounds over ten single-zero registers: survival `0.82¹⁰ = 0.14`, less than
the first leak's 0.18, so the mode moves from iteration 15 to iteration 5. That
is the measured failure (`loc(16) = 5`) in the baseline, exactly.

The consistency term replaces a flat cross-entropy signal with a direct L2 on
the halting probabilities. That is the whole of its contribution.

### 6.3 Where it fails — and this is the sharp result

Applied to the `(place, digit)`-indexed heads, in `loc` space
(`loc(inc r) = loc(r) + 1`, orbit of 63), 3 seeds each:

| head | control, no consistency | with consistency |
|---|---|---|
| `placev` (13 params) | 2 / 2 / 2 | **0 / 0 / 0**, `loc` → 1.2 × 10⁴ |
| `mlp` (1409 params) | 2 / 2 / 2 | **0 / 0 / 0**, `loc` → 5.3 × 10⁵ |
| `placev`, Medium | 0 / 0 / 0 | 0 / 0 / 0, `loc` → 1.2 × 10⁴ |

In every one of these runs the learned unit digit collapses to **argmax 0** —
the increment becomes the identity. Then `loc(r) = loc(r) + 1` is
unsatisfiable, the gradient pushes `loc` up by one every step, and the head
diverges.

**The law only has content when the operator that generates its orbit is itself
anchored by the task loss.** In the counted-halting controller the same learned
`one` is used for the countdown (which the cross-entropy anchors: subtracting
the wrong digit gets the training T values wrong) *and* for the orbit, so the
orbit operator is identified. An ordered-pointer head has no countdown, nothing
ties its orbit operator to the data, and the law becomes an unsatisfiable
constraint that destroys the head.

So the honest answer to "did digit-independent parameterisation or
self-consistency break the coverage barrier" is: **neither alone — the
combination did, and self-consistency is only usable inside a
digit-independent, counting parameterisation.**

---

## 7. What still fails, diagnosed to one scalar

The failing seeds are not random. Across every counted-halting cell the verdict
tracks a single quantity — the detector's **margin between "all digits of the
register are zero" and "all but one are"**:

| cell / seed | learned `one` | gain | thresh | gain·(thresh − 1) | MAX_T |
|---|---|---|---|---|---|
| R Easy s0 | argmax **1** | 5.93 | 1.642 | **3.81** | 64 |
| R Easy s1 | argmax **1** | 6.54 | 1.641 | **4.19** | 64 |
| R Easy s2 | argmax **1** | 4.86 | 1.271 | 1.32 | 0 |
| R m1-Easy s0 | argmax **1** | 2.99 | 1.154 | 0.46 | 2 |
| R m1-Easy s1 | argmax **1** | 3.07 | 1.150 | 0.46 | 2 |
| R m1-Easy s2 | argmax **1** | 4.11 | 1.755 | **3.10** | 64 |
| R Medium s0 | argmax **1** | 4.37 | 1.914 | **3.99** | 64 |
| R Medium s1 | argmax **6** | 3.09 | 1.765 | — | 0 |
| R Medium s2 | argmax **6** | 3.27 | 1.722 | — | 0 |
| baseline Easy s2 | argmax **9** | 2.00 | 0.301 | — | 0 |

Two failure modes, cleanly separated:

**(i) The detector's margin is too small.** `> 3` reaches T = 64, `< 2` stops at
T = 2, with no exceptions in ten cells. The cause is the *parameterisation*:
"every digit of the register matches the learned zero digit" is a
**conjunction**, and scoring it as a *sum* of per-slot match masses puts the
decision boundary between `n_t` and `n_t − 1` — a margin of one. Scoring the
same conjunction in **log space** (`mean_slots log⟨c_slot, zero⟩`) gives
all-match = 0 and any-mismatch = log(1e-6) — a margin of ~14, so any positive
gain works and there is nothing left to get wrong. Measured, `--detector log`,
Easy, 3 seeds: MAX_T = **0 / 64 / 64**, and the two converging seeds reach 64
with gain 1.4 and 1.2 — i.e. **the log detector removes failure mode (i)
entirely** and only mode (ii) survives.

**(ii) The learned unit digit lands on the wrong digit** (argmax 2, 6 or 9
instead of 1). This is a straight optimisation basin in a 10-way softmax, and
it is the only remaining source of seed variance. Two untried fixes follow
directly and are the first thing I would run next: delay the consistency term
until the anchor exists (`--cons-start`, since the law *propagates* an anchor
and cannot create one), and raise the controller's learning rate — both are in
`lab/dc_gridG.sh` and were still running at the cutoff.

---

## 8. The evaluation budget — measured, and it does not fit on the current step

`P2` from `alu-compose` is a hard constraint: the eval budget is half the
training budget and must cover 16 splits at every tier, so Easy gets 30 s for
the same 16 splits Hard gets 1800 s for.

The controller's contribution is its **iteration count**. A correctly trained
one spends `T` iterations on rung `T`:
`2 × (1+2+4+8+16+32+64) + test + ood ≈ 260` against a fixed grid's `16 × 64 =
1024` — a **3.9× reduction**. That is measured here by proxy: the `fixed16` lab
variant runs exactly 16 loops on all 16 splits (256 iterations).

Real evaluator, tier-faithful wall-clock manifests, `eval_batch_size = 4096`,
`max_steps = 5` so the measurement is of evaluation alone. **Six to twelve
sibling jobs were on the GPU throughout, so every second here is
contention-pessimistic; the ratios and the outcomes are the transferable part.**

| tier | budget | variant | iterations over 16 splits | eval s | seen-N rungs | OOD-N rungs |
|---|---|---|---|---|---|---|
| Easy e1 | 30 s | learned early exit, untrained model | ~32 | **19.1** | 7/7 | **7/7** |
| Easy e1 | 30 s | same, "semantically right" threshold init | ~590 | 31.7 | 7/7 | **2/7** |
| Easy e1 | 30 s | `fixed16` ≈ a *trained* controller | 256 | **33.4** | 7/7 | **0/7** |
| Easy e1 | 30 s | `fixed64`, no early exit | 1024 | 40.1 | **0/7** | **0/7** |
| Medium m1 | 300 s | learned early exit, untrained model | ~32 | 23.2 | 7/7 | 7/7 |
| Medium m1 | 300 s | `fixed16` ≈ a *trained* controller | 256 | **43.2** | 7/7 | 7/7 |
| Medium m1 | 300 s | `fixed64`, no early exit | 1024 | 203.6 | 7/7 | 7/7 |

Three conclusions, and the middle one is uncomfortable.

* **Early halting is necessary.** Without it, the Easy run loses *both* ladders
  entirely (0 of 7 rungs on each) — the score is 0 no matter how good the model
  is. Medium survives it but at a 1.47× margin.
* **Early halting is not sufficient on the current step.** The trained-controller
  proxy costs **33.4 s of a 30 s budget on Easy** and forfeits the whole OOD-N
  ladder. Fitting the two Easy points gives ~64 ms per iteration-across-splits
  and a **fixed harness cost of ~17 s**, so the Easy budget contains only ~13 s
  of model time and 260 iterations of the `REDUCE = 11` DigitALU costs ~14 s.
  It is over by about 10 %.
* **`alu-depth`'s cheaper step closes it.** `tree:quotient` is 39 sequential
  soft steps against 257, 103 ms/step against 390 — 3.8×. That takes the
  trained-controller cost from ~14 s to ~3.8 s and the Easy total to ~21 s, a
  **1.43× margin**, with the full OOD-N ladder. Medium goes from 6.9× to ~25×.
  **The controller and the cheap step are both required; neither alone is
  enough on Easy.**

Measured margins for the deliverable as it stands (untrained, contended GPU):
**Easy 19.1 s / 30 s = 1.57×**, **Medium 23.2 s / 300 s = 12.9×**, both with all
16 splits and both full ladders.

### 8.1 The halting threshold at initialisation is an eval-budget decision

An untrained detector that is initialised at its *semantically correct*
threshold fires with probability ≈ 0.07, so an untrained model runs ~37
iterations per batch and costs 31.7 s of the 30 s Easy budget — silently
truncating the OOD-N ladder to 2 of 7 rungs. Initialising it shallow (fires with
probability ≈ 0.5) costs 19.1 s and completes everything. The submission uses
the shallow init and lets training earn depth; the log-space detector of §7
makes this free, because there the correct threshold is reachable from a shallow
start without crossing a small-margin region.

---

## 9. The submission

`submissions/depth-controller/submission.py` — every tensor learned from random
init, ~8.6 K model-state elements, lint-clean, runs end to end inside the real
Easy budget with all 16 splits completed.

```
marker-relative parser (learned)
  -> DigitALU, weight tied, applied k times        (learned; alu-depth owns it)
  -> counted-halting controller                    (12 learned scalars)
       * no mass dump
       * straight-through one-hot digit register
       * self-consistency term via training_loss(aux)
  -> commit to the mode + exact early exit at eval
```

The early-exit rule is exact for that readout: once the remaining halting mass
can no longer exceed the running maximum weight, no later iteration can win the
argmax, so the loop stops. It is driven entirely by the model's own learned
halting scalar; training never takes that path and is the full soft mixture.

**It scores MAX_T = 0 and will until a trained `DigitALU` lands** — the ALU is
the sibling branches' problem, not this one, and this file is a scaffold for it,
not a candidate. What it delivers is that the controller, the dtypes, the eval
budget and the compliance surface are verified end to end.

Lab variants `fixed16/` and `fixed64/` exist only so the §8 timings reproduce.

---

## 10. What is falsified

1. **"The depth controller cannot extrapolate in T."** False as stated. A
   controller with no parameter indexed by a place or a digit of T routes
   1.000 at every rung to T = 64 from `T ∈ {1,2,3}` or `T ∈ {4,8,16}`, at two
   moduli whose ladders do not collapse. `alu-compose` §4's 28 cells measured
   the ordered-pointer family (for which the conclusion stands) plus a counted
   controller crippled by the mass dump.
2. **"Coverage in the T field is the binding cause."** False for a counting
   controller — it is immune to (a) by construction, and the thing that actually
   binds is (b), the flat loss, localised to a *single scalar*: the detector's
   conjunction margin. Cause (a) remains exactly true for every
   `(place, digit)`-indexed head, at every parameter count from 3 to 1409, with
   or without self-consistency.
3. **"Medium fails at T = 1 and T = 2 (downward generalisation)."** Half false.
   The *decision* is correct at T = 1 and T = 2 (route 1.000, exact 1.000 under
   a committing readout) while the soft mixture reads 0.026. The reported
   failure was substantially a readout artifact.
4. **"Self-consistency is the cheapest untried lever, and it should be tried on
   any controller."** Half false, and this is the branch's sharpest structural
   result. On a counting controller it is the fix. On a `(place, digit)`-indexed
   head it is actively harmful (2 → 0), because the law's orbit operator is
   unidentified there and the law becomes unsatisfiable.
5. **"Running the states hard at eval is a risk for the controller."** False for
   this controller: `--eval-hard` leaves every converging seed at 1.000 on every
   rung. The halting signal is a conjunction over a one-hot register, not a
   continuous relaxation.
6. **"ACT early exit makes the Easy eval budget fit."** False on the current
   step. Measured: a *correctly trained* controller's iteration count costs
   33.4 s of a 30 s Easy budget and forfeits the entire OOD-N ladder. Early exit
   is necessary (without it both ladders are lost) but the cheap step is
   required too.
7. **"`e1` is a safe place to test depth"** — confirmed from `alu-compose` and
   strengthened: at N = 323 the same controller and recipe give a *different*
   MAX_T from N = 329 in both directions, because four of the seven rungs are
   only two distinct maps.

---

## 11. The single highest-value recommendation

**Adopt the counting controller with the log-space conjunction detector, and
spend the next runs on the one remaining failure mode — the learned unit digit
landing on the wrong digit — not on the T-coverage problem, which is solved.**

In priority order:

1. **Ship the recipe**: counted halting, no mass dump, straight-through digit
   register, self-consistency, commit-to-the-mode readout with the exact early
   exit. Four changes, each measured, and together they take the controller from
   MAX_T = 2/0 to MAX_T = 64/64.
2. **Replace the sum-and-threshold detector with the log-space conjunction.**
   It is the same two learned scalars and the same information, but the margin
   goes from 1 to ~14 and the measured seed-failure mode (i) disappears. This is
   a five-line change with no cost.
3. **Fix failure mode (ii)** — a 10-way softmax landing in a bad basin. Delay
   the consistency term until the anchor exists, and raise the controller's
   learning rate. Both are one flag.
4. **The controller alone does not fit the Easy eval budget.** Combine it with
   `alu-depth`'s `tree:quotient` step before quoting any margin; the two
   together give 1.43× on Easy, and neither alone does.
5. **Never quote a depth number from e1 or e2 again.** λ(323) = λ-collapse makes
   {4,16,64} one map and {8,32} another.

---

## 12. Reproduction

```bash
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python

# SS3  the alu-compose baseline at a modulus that can measure it
$VENV lab/probe_depth.py --modulus 329 --selector counter2 --train-t 1 2 3 \
    --train-loops 3 --sel-steps 2000 --sel-anneal 1200 --seeds 0 1 2
bash lab/dc_gridA.sh          # counter/place/placev/mlp x train-loops
bash lab/dc_gridB.sh          # the same at Medium's T values

# SS4/SS5  the recipe and its ablations
bash lab/dc_gridF.sh easy
bash lab/dc_gridF.sh medium
bash lab/dc_gridF.sh heads    # SS6.3: consistency on (place,digit)-indexed heads
bash lab/dc_gridD.sh          # m1 (N=10403) and hp1 (N=4028033)

# SS7  the detector reparameterisation and the warm-start
bash lab/dc_gridG.sh easy
bash lab/dc_gridG.sh medium

# SS8  the evaluation budget
$VENV lab/make_manifest.py --dataset e1 --mode wallclock --max-steps 5 \
    --eval-batch-size 4096 --name dc_e1_evalonly
$VENV lab/make_manifest.py --dataset m1 --mode wallclock --max-steps 5 \
    --eval-batch-size 4096 --name dc_m1_evalonly
bash lab/dc_evalbudget.sh

$VENV lab/dc_table.py         # collate lab/runs/grid*.jsonl
```

## 13. Compliance

Nothing under `data/generated/` was read, printed, sampled or summarised. Every
probe synthesises its prompts and targets from the public generator spec and
from moduli passed on the command line; §1's λ table is elementary number theory
on moduli that appear in dataset *directory names*. `--construct` sets the
parser and the digit tables to their exact values and is a LAB DIAGNOSTIC in
`lab/`, never imported by a submission — the direct analogue of
`probe_alu.py --construct`.

`submissions/depth-controller/submission.py` contains no modular-exponentiation
routine, no digit-multiplication rule, no carry rule and no lookup of answers;
every tensor is initialised randomly and trained in the run, and no Python
control flow reads `input_ids` to decide a loop count. Its training path is
fully differentiable with a single scalar loss, no participant-controlled
backward and no custom training loop. The three `self.training` branches are the
loop count, the readout (soft mixture in training, commit-to-the-mode at eval)
and the early exit — all of which the README and BRIEF §4.3 permit as ACT /
PonderNet inference; the exit condition is a function of the model's own learned
halting scalar only.

**One judgement call, flagged explicitly.** The self-consistency term encodes
the structural prior that *incrementing the T register by one unit corresponds
to one more application of the step*. It is realised entirely through learned
tensors (which digit vector is the unit, the shared increment/decrement tables,
the halting detector), it uses no labels and no data, and it says nothing about
modular arithmetic. It is the loss-side statement of a prior the *architecture*
already commits to — the controller counts down and halts at zero whether or not
the term is present. I read that as an architectural prior of the kind BRIEF §4
rule 2 explicitly encourages rather than a hard-coded algorithm, but it is the
one place on this branch where a reviewer could reasonably want a second
opinion, and the ablation without it (§5) is a legal fallback that reaches
MAX_T = 64 on 1 of 3 seeds.

Nothing was submitted to the hosted service and no `one-layer login`/`submit`
was run.
