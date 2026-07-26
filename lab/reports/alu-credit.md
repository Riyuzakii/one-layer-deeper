# alu-credit — making `DigitALU` trainable, and what the step budget does to that question

**Branch** `explore/alu-credit`, from `explore/digit-carry`.
**Mandate** hold the architecture fixed (`alu-depth` owns changing it), vary
everything about how it is optimised, get `train_exact` to 1.000.
**Tooling** `lab/probe_credit.py` (subclasses `DigitALU`; `--construct` still
reports `train_exact = held_exact = 1.000` after refactoring, so the hypothesis
class is unchanged), `lab/credit_sweep.sh`, ~90 runs in `lab/credit_runs.jsonl`
and `lab/logs/`.

---

## 0. Headline

**1. Nothing a submission can legally do trains this model, at any step count.**
Five initialisation families, four relaxation schedules, straight-through on the
states *and* on the gate, three curricula, four regularisers, five optimiser
settings — all land inside the 0.10–0.21 seed band of the untouched baseline at
1,000–3,000 steps, and inside the 0.004–0.036 band at the tier-faithful 20-step
budget. Their tables are at **chance** on a gauge-invariant structure score in
every single case.

**2. The one thing that works is teacher forcing on the true register trace, and
it is fast.** It reaches `train_exact` 1.000 free-running at eval, and its
*tables* reach 78% of the intended structure in **20 optimizer steps** — inside
the Easy budget. It needs the true intermediate registers, which are computed
arithmetic, so it is a **LAB DIAGNOSTIC, not a legal submission** (rule 2).

**3. The 4,000-step figure is the cost of the rollout, not of the tables.** This
is the finding that matters most under the coordinator's step famine. Given
per-step *inputs*, six thousand table cells are learned in tens of steps. Given
only the end-of-chain loss, they never leave chance at 600, 3,000 or 20,000
steps. **Step budget is not the binding constraint on this architecture;
rollout depth is.** If `alu-depth` gets the loss to within a few ops of every
learned table, 15–150 steps is plenty.

**4. Shortening the chain buys convergence *speed* but buys the wrong
solution.** At the 20-step budget, halving the chain (280 → 117 ops) moves
`train_exact` from 0.024 to 0.25 (mean of 3 seeds) — a 10× gain that compounds
with the cheaper step. But the structure score stays at chance the whole way,
held-out stays at 0.000, and held-out CE climbs monotonically 2.31 → 9.06. The
short chain makes the **degenerate** solution easier to fit, not the algorithm.
**This is a falsifiable prediction against the plan to shorten the chain: it
will raise `train_exact` and will not certify a rung.**

**5. Three falsifications, one of a premise in my own brief.**

* **"For `DigitALU`, `train_exact` → 1.000 implies `held_exact` → 1.000."**
  **False.** Teacher-forced: `train_exact` 1.000, held 0.63–0.79. Short chain:
  `train_exact` 0.75, held 0.000, held CE 9.06. `DigitALU` overfits.
* **"The 0.2 plateau is a partially-learned table."** **False.** It is a
  degenerate non-arithmetic solution — structure score 0.23–0.29 against a
  random-init baseline of 0.23–0.28.
* **"The forward pass destroying information is the binding constraint."**
  **False.** I found the mechanism and fixed it (register spread 0.0000 → 0.26
  across all 280 ops); training with the fix is *worse* (0.004–0.020).

---

## 1. What I re-verified first

| check | result |
|---|---|
| constructed ceiling, N=323 S=3 R=11 | `train_exact = held_exact = 1.000` |
| the same after refactoring `forward` into an explicit op list | **1.000** (regression test) |
| baseline from random init, 1,000 steps, seed 0 / seed 1 | 0.100 / 0.184 |

The seed spread on one recipe (0.100 vs 0.184) is the noise floor for everything
below. Constructed ceiling versus `R`, because `R` is the obvious chain-length
knob:

| R | 1 | 2 | 3 | 4 | 5 | 7 | 9 | 11 |
|---|---|---|---|---|---|---|---|---|
| constructed `train_exact` | 0.100 | 0.180 | 0.280 | 0.372 | 0.468 | 0.704 | **1.000** | **1.000** |

`R ≥ 9` is needed for the class to contain the solution, so `R` is only usable as
a *warm-up* knob that ends at 11 — which is how I used it.

## 2. Two diagnostics that located the obstruction

### 2.1 The gradient never reaches the front of the chain

`--diag`, step 1, N=323:

```
Tmul=3.27e-07  Tadd=4.57e-06  Tsub=3.41e-02  gate.weight=5.69e-05
```

`Tsub`, used by the *last* op, gets **10⁵×** the gradient of `Tmul`, used by the
first.

### 2.2 The register stops depending on the input after four subtractions

`--flow` reports the across-batch standard deviation of the register after every
op — how much of `x` still reaches the state:

| init | spread at op 1 | spread at op 280 |
|---|---|---|
| default | 8e-4 | **0.0000** (numerically zero by op 5) |
| `--gate-bias −6` (gate held closed) | 8e-4 | 9e-4 |
| `--perm-init 8` (tables ≈ random permutations, constants sharp) | 0.0075 | 0.0082 |
| **`--hard --hard-gate` (straight-through everywhere)** | **0.125** | **0.26** |

The mechanism is `cond_sub`: the gate computes `g·t + (1−g)·r` with `g ≈ 0.5` at
init, and `t` — the scan of a near-uniform `Tsub` — is nearly constant across the
batch, so **every conditional subtraction halves the deviation**. Eleven per
place is 2⁻¹¹, and the profile resets to 8e-4 at each new Horner place (a fresh
`Tmul` product is added) and dies again.

Straight-through on the gate is the piece `digit-carry` never tried, and it is
the piece that actually fixes this — a 300× improvement in surviving signal.
**It makes training strictly worse (§4.2).** That is the most useful negative
here: the signal is not absent, it is uninformative. The chain is a
near-permutation dynamical system; the gradient *direction* through 280 of its
steps is chaotic, not small.

## 3. The measurement that made the rest interpretable

Raw "does the learned table match the truth" accuracy is **meaningless** for this
architecture, and reading it naively would have made me report the exact
opposite of the truth. There is a gauge freedom:

* `Tmul`'s output alphabet is consumed *only* by `Tadd`'s addend index. Any
  permutation `π` of the ten digit symbols with `π(zero) = zero` applies to both
  with no change to the function — a 9!-element gauge group.
* The two borrow states can be swapped, and the two carry states with them.

`Tsub` is the one table with **no** gauge freedom (its register index and output
are the register; its second index is a digit of `N` given as a fixed one-hot),
but only the columns `N`'s digits reach are ever exercised.

`structure_scores()` is gauge-invariant by construction — is `Tmul`'s argmax a
well-defined *function* of `(a·b) mod 10`, and is each column of `Tadd`/`Tsub` a
*cyclic shift* of the identity (which "add/subtract a constant" is, whatever the
labelling)? Validated at both ends:

```
constructed : mul_lo=1.000 mul_hi=1.000 add_shift=1.000 sub_shift=1.000
random init : mul_lo=0.280 mul_hi=0.240 add_shift=0.275 sub_shift=0.233
```

**Every sibling working on this architecture should screen on this, not on
`train_exact`.** `train_exact` 0.75 is reachable with tables at chance.

## 4. Everything a submission could legally express — all negative

### 4.1 At the tier-faithful budget (20 optimizer steps, Easy), N=323 S=3

Baseline band across three seeds: **0.016 / 0.020 / 0.036.**

| procedure | `train_exact` @20 | `sub_shift` |
|---|---|---|
| baseline, lr 0.03 / 0.1 / 0.3 / 1.0 / 3.0 | 0.012 / 0.000 / 0.016 / 0.016 / 0.000 | 0.15–0.25 |
| `--onehot-init 4` | 0.024 | 0.25 |
| `--init-scale 4` | 0.012 | 0.233 |
| `--perm-init 8` | 0.008 | 0.30 |
| `--perm-init 8 --gate-off −6` (annealed to 0) | 0.008 | 0.30 |
| `--hard --hard-gate` | 0.004 | 0.283 |
| `--r-start 3 --r-warm 0.5` (chain grows 3→11) | 0.016 | 0.25 |
| `--inv 1.0` (`Tsub` inverts `Tadd`) | 0.016 | 0.30 |
| `--ent 0.05` (entropy pressure) | 0.016 | 0.267 |
| `--sym 1.0` (commutativity) | 0.016 | 0.25 |
| `--xcurr 0.5` (magnitude curriculum) | 0.016 | 0.283 |
| `--opt sign` | 0.012 | 0.233 |

**Not one of them is outside the seed band, and not one moves the structure score
off chance.** Chain length is the only thing that moves the number at all:

| chain | ops | `train_exact` @20 (3 seeds) | `sub_shift` |
|---|---|---|---|
| N=91, S=2 | ~117 | **0.367 / 0.317 / 0.067** | 0.217–0.25 |
| N=323, S=3 | ~280 | 0.016 / 0.020 / 0.036 | 0.233–0.283 |
| N=2021, S=4 | ~525 | 0.028 | 0.217 |

### 4.2 At 1,000–3,000 steps (out of budget, but the fair test of the procedure)

| procedure | steps | `train_exact` | `sub_shift` |
|---|---|---|---|
| baseline seed 0 / 1 | 1000 | 0.100 / 0.184 | ~0.25 |
| baseline | 600 / 3000 / 20000 | 0.048 / 0.132 (@2k) | 0.267 |
| `--tau 0.3 --init-scale 0.3` seed 0 / 1 | 1000 | 0.112 / 0.212 | — |
| `--tau 0.1 --init-scale 0.1` seed 0 / 1 | 1000 | 0.052 / 0.076 | — |
| `--tau 0.03 --init-scale 0.03` | 1000 | 0.008 | — |
| `--init-scale 4` / `--onehot-init 4` | 1000 | 0.180 / 0.204 | — |
| `--hard` (the report's config) | 1000 | 0.008 | 0.26 |
| **`--hard --hard-gate`** | 1000 | **0.020** | 0.25 |
| `--hard --hard-gate --tau` 0.1 / 0.3 / 3.0 | 1000 | 0.008 / 0.008 / 0.012 | — |
| `--perm-init 8` | 1000 | 0.020 | — |
| `--r-start 3 --r-warm 0.5` | 1500 | 0.164 | — |
| `--curr 91:2:500` → N=323 (LAB ONLY) | 1500 | 0.116 | — |
| `--inv 1.0` / `--ent 0.05` | 1000 | 0.096 / 0.096 | — |
| lr 0.01 / 0.1 | 3000 | 0.096 / 0.184 | 0.367 / 0.267 |
| `--opt sign` | 3000 | 0.060 | 0.367 |
| **`--xcurr 0.5`** (best legal number anywhere) | 3000 | **0.224** | **0.417** |
| N=91 S=2 (short chain) | 3000 | 0.750 | 0.45 |

The `tau`/init-scale rows were co-varied deliberately: the per-step softmax
Jacobian is `(1/tau)(diag p − p pᵀ)`, so holding `logits/tau` fixed while lowering
`tau` multiplies the per-step gain by `1/tau` without changing how sharp the
distributions are. It buys nothing. Separately, `digit-carry`'s `--identity-init`
confounded copy-through scale with a closed gate; `probe_credit` separates them
and neither helps alone.

The two best legal numbers (`--xcurr` 0.224/0.417, short chain 0.750/0.45) are
the only ones whose `sub_shift` is even arguably off chance, and both are far
from the 0.78–1.00 that teacher forcing reaches in 20 steps.

## 5. Shortening the chain: what it does and does not buy

`digit-carry` §2.4's strongest fact was that ~280 → ~117 ops took `train_exact`
from 0.20 to 0.72–0.78. Reproduced (0.683 at 1,000 steps), and it is real — and
it is **also** a 10× gain at the 20-step budget (§4.1), which under step famine
is the more valuable half. But run it out:

| N=91, S=2, ~117 ops | step 20 | 80 | 150 | 300 | 600 | 1000 | 2000 | 3000 | 3500 |
|---|---|---|---|---|---|---|---|---|---|
| `train_exact` | 0.367 | 0.483 | 0.517 | 0.483 | 0.533 | 0.683 | 0.750 | 0.750 | 0.633 |
| `held_exact` | 0.000 | 0.000 | 0.000 | 0.000 | 0.083 | 0.083 | 0.000 | 0.000 | 0.083 |
| `held_ce` | — | — | — | — | — | 7.27 | 8.58 | **9.06** | 8.37 |
| `sub_shift` | 0.217 | 0.267 | — | 0.333 | 0.383 | — | — | 0.45 | — |

**Held-out CE rises monotonically from 2.31 to 9.06 while train CE falls to
0.27**, in the architecture the previous branch described as unable to memorise.
The structure score crawls from 0.22 to 0.45 — off chance, but nowhere near the
solution. The short chain fits the *degenerate* solution faster; it does not find
the algorithm. This is why the cross-modulus curriculum (train the shared tables
at S=2, then move to S=3) transfers nothing: there is nothing correct at the
source to transfer.

## 6. What worked, why, and exactly why it is not a submission

### 6.1 Teacher forcing on the true register trace — LAB DIAGNOSTIC ONLY

`--teacher-force p` overwrites the register with the true value after every one
of the ~70 ops **during training only** (`model.eval()` disables it, so every
number below is measured free-running). `--deep-sup` puts a cross-entropy on each
op's output against the true next register. Both need the true trace,
`(10·r + Σ dᵢdⱼ) mod N`, computed by me in Python. **Rule 2. Not legal.**

**The convergence curve — the answer to the step-famine question.** N=323, S=3,
lr 0.3, teacher forcing:

| steps | 5 | 10 | 20 | 40 | 80 | 150¹ | 300 | 600 | 1000 |
|---|---|---|---|---|---|---|---|---|---|
| `train_exact` | 0.000 | 0.004 | 0.000 | 0.188 | 0.568 | 0.728 | 0.976 | 0.984 | **1.000** |
| `held_exact` | 0.000 | 0.026 | 0.000 | 0.105 | 0.421 | **0.632** | 0.579 | 0.447 | 0.789 |
| `mul_lo` | 0.41 | 0.62 | **0.77** | 0.75 | 0.87 | 0.83 | 0.83 | 0.75 | — |
| `add_shift` | 0.50 | 0.50 | 0.51 | 0.69 | 0.74 | 0.75 | 0.68 | 0.64 | — |
| `sub_shift` | 0.60 | 0.72 | **0.78** | 0.78 | 0.82 | 0.82 | 0.82 | 0.82 | — |

¹ lr 0.1; the rest lr 0.3.

**At five optimizer steps `sub_shift` is already 0.60 against a random baseline of
0.233. At twenty — the tier-faithful Easy budget — it is 0.78 and `mul_lo` is
0.77.** The six thousand table cells are most of the way learned inside the
budget. What takes another 300 steps is turning ~80%-correct tables into an
end-to-end exact 280-step rollout, which is an all-or-nothing composition
problem, not a learning problem.

Note also `held_exact` peaks at 80–150 steps (0.42–0.63) and *declines* to 0.447
by step 600 while `train_exact` climbs to 0.984 — overfitting again. Under step
famine that is good news: the budget-constrained regime sits near the
held-optimal point.

**Chain length does not hurt teacher forcing.** Because tf makes every op an
independent one-step problem, a longer chain is *more* supervision per step:

| chain | ops | steps | `train_exact` | `mul_lo` | `add_shift` | `sub_shift` |
|---|---|---|---|---|---|---|
| N=91, S=2 | ~117 | 20 | 0.250 | 0.61 | 0.37 | 0.967 |
| N=323, S=3 (3 seeds) | ~280 | 20 | 0.000/0.000/0.004 | 0.77/0.85/0.60 | 0.51/0.49/0.53 | 0.78/0.77/0.77 |
| N=2021, S=4 | ~525 | 20 | 0.000 | 0.71 | 0.56 | **0.967** |
| N=2021, S=4 | ~525 | 80 | 0.608 | 0.78 | 0.835 | **1.000** |

That is the cleanest statement of the mechanism in this report: **under teacher
forcing the chain is an asset; under free-running it is the entire problem.**

**And it holds after the crutch is removed.** With `--tf-decay 0.5`, teacher
forcing is exactly zero from step 1000 of 2000 onward, and `train_exact` stays at
1.000 with the training loss computed free-running. The learned automaton is
genuinely self-consistent.

### 6.2 Deep supervision alone is not enough — and the split is diagnostic

Giving the true *target* at every op but letting the register free-run:

| procedure | steps | `train_exact` | `mul_lo` | `add_shift` | `sub_shift` |
|---|---|---|---|---|---|
| deep sup, per Horner place (5 targets) | 1000 | 0.172 | — | — | — |
| deep sup, per op (~70 targets) | 20 / 80 / 300 | 0.032 / 0.048 / 0.108 | 0.54 / 0.42 / 0.44 | 0.22 / 0.30 / 0.27 | **0.217 / 0.217 / 0.217** |
| deep sup, per op | 1000 | 0.240 | — | — | — |
| **+ teacher forcing** | 1000 | **1.000** | 0.77 | 0.51 | 0.78 |

Deep supervision teaches **`Tmul` only** (`mul_lo` 0.28 → 0.54), and leaves
`Tadd`/`Tsub` at exactly chance. That is precisely what the mechanism predicts:
`Tmul` sits one op from its own target, so a correct target is usable; `Tadd` and
`Tsub` sit inside a register that has already drifted, so a correct target for a
wrong input is worthless. **Per-step targets are not the active ingredient.
Per-step *inputs* are.**

### 6.3 What the teacher-forced model actually learned

Read through the gauge of §3 (`--dump-tables`):

* `Tadd` at carry-index 0: column `v=3` is the identity `[0..9]`; column `v=0` is
  `[1,2,…,9,0] = (u+1) mod 10`; column `v=4` is `(u+2) mod 10` bar one cell. It
  **is** an addition table with the addend alphabet relabelled by `π`, exactly as
  §3 predicts (`π(3)=0`, `π(0)=1`, `π(4)=2`).
* `Tsub` at borrow-index 0, at the two columns `N=323` reaches:
  `[7,8,9,0,1,2,3,4,5,6] = (u−3) mod 10` for `v=2`, and
  `[6,7,8,9,0,1,2,3,4,5] = (u−4) mod 10` for `v=3` — both exactly
  `(u − v − 1) mod 10`. The model's `borrow0` argmax is **1**, not 0: it swapped
  the borrow states, so borrow-index 0 *is* "borrow set" and `u−v−1` is correct.
* `gate.weight = [17.6, −18.3]` — the same sign pattern the construction uses.

**Given per-step inputs, gradient descent recovers the constructed solution up to
the gauge.** The target is reachable in this parameterisation; the rollout is the
whole problem.

Held-out stops at 0.63–0.79 rather than 1.000. I tested the obvious coverage
explanation by varying the train split (150 / 250 / 280 of 288 units) and the
held-out numbers are 0.63 (n=8) and 0.79 (n=38) — **the held cohorts are too
small to separate coverage from noise, so I am not claiming a cause.** What is
solid is that it is not 1.000, which is what the metric needs.

### 6.4 Target propagation — the legal analogue

Teacher forcing works because it supplies per-step *inputs*. The legal way to get
those without computing them is to make them **learned latents**: `LatentTrace`
predicts the whole ~70-step register trace from the same inputs the model already
sees; tables and latents train jointly under (a) local consistency — one op
applied to latent `t` must reproduce latent `t+1` — and (b) two boundary
conditions using only given quantities (the last latent is the label; the first
input is the model's own learned `zero`). This is method-of-auxiliary-coordinates
/ target propagation. It supplies no arithmetic, it is discarded at eval, and it
*is* expressible under the evaluator's fixed loop.

It is also ~12× the cost per step, which under step famine is close to
disqualifying on its own. Result in `lab/credit_runs.jsonl` under `z_tprop*`.

## 7. Which procedures a submission could actually express

The brief asked for this explicitly, and the step budget sharpens it.

**Expressible under `build_model` / `build_optimizer` / `training_loss` with one
`optimizer.step()` per batch — all tested, all null:**

| procedure | how it fits | survives a 15–150-step budget? |
|---|---|---|
| temperature / noise / gate-offset annealing | step counter in a **non-persistent** buffer (excluded from the 5e8 ceiling, `api.py:26-42`) | yes, but a schedule over 15 steps is barely a schedule |
| straight-through, scheduled hardening | relaxation choice in `forward` | yes |
| **chain-length warm-up on `R`** | `R_eff = f(step)` in `forward`; eval at full `R`, so the constructed ceiling is unchanged | yes — and it makes early steps *cheaper*, which matters now |
| staged gradient release (`mul → add → sub`) | `T.detach()` under a step condition | no — needs stages |
| magnitude curriculum on \|x\| | per-example loss weight from `input_ids` | marginal — needs a ramp |
| entropy / commutativity / `Tsub ∘ Tadd = id` | pure `training_loss` terms | yes |
| every init family here | `build_model` | yes (free) |
| every optimiser / lr / wd / schedule | `build_optimizer` | yes |
| target propagation (§6.4) | latents in `aux`, terms in `training_loss` | **no** — ~12× cost per step |

**Not expressible — and these are exactly the ones that worked:**

| procedure | why not |
|---|---|
| `--construct` | sets tables to the truth (rule 7) |
| `--deep-sup` | targets are `(10·r + Σ dᵢdⱼ) mod N`, computed by me (rule 2) |
| `--teacher-force` | the same trace, used as *input* (rule 2) |
| cross-`(modulus, slots)` curriculum | a submission sees one modulus; manufacturing a second with known answers is doing the arithmetic |

I did **not** ship `submissions/alu-credit/submission.py`. No legal procedure
beat the baseline seed band, so a submission from this branch would be
`digit-carry`'s with a different schedule, and two branches have already
established that the evaluator cannot resolve anything while `train_exact` is
below ~1.0. Adding a `MAX_T = 0` row would have cost GPU and carried no
information. Every procedure above is one flag in `lab/probe_credit.py` if
someone wants to revisit that.

## 8. Recommendation

**Stop looking for a training trick, and re-plan around this: the step budget is
not what is stopping `DigitALU`. Rollout depth is, and it is stopping it by a
margin no schedule closes.**

1. **Per-step inputs make the tables learnable inside the Easy budget.** 20
   steps → `sub_shift` 0.78 at S=3 and 0.967 at S=4. Without them, chance at
   every step count I tried up to 20,000. The "≳4,000 steps" figure attached to
   `DigitALU` is a property of the free-running rollout, not of the 6,817
   parameters. **Any architecture that puts the loss within a few ops of each
   learned table inherits the 20-step number, not the 4,000-step one.**
2. **For `alu-depth`: my §5 is a falsifiable prediction against your plan.**
   A 2.4× depth cut (280 → 117 ops) raises `train_exact` at budget by 10× and
   leaves the structure score at chance, held-out at 0.000, and held-out CE
   climbing to 9.06. `digit-carry` §6's "one quotient digit + one subtraction,
   ~60 ops" is a 4.7× cut — better, same kind. I would design for **O(10)
   sequential learned ops** (make the per-place reduction a single table lookup
   rather than a `W`-slot scan) and I would not expect ~60 to certify a rung.
   If it does, my prediction is wrong and that is worth knowing quickly.
3. **Change the screen.** `structure_scores()` in `lab/probe_credit.py` is
   gauge-invariant, costs nothing, reads 1.000 on the construction and 0.23–0.28
   on random init, and would have caught every false positive in this report —
   including `digit-carry`'s headline short-chain result, whose 0.75
   `train_exact` has tables at chance. Screen struct → `train_exact` → held-out,
   in that order.
4. **Retire the rule "`train_exact` → 1.000 implies `held_exact` → 1.000."** It
   is false in both directions of evidence I have (§0.5). Whatever finally trains
   still has to be checked on held-out.
5. **If anyone wants one more shot from the training side**, target propagation
   (§6.4) is the only legal construction that supplies per-step inputs, which is
   the exact thing shown to be sufficient — but its ~12× per-step cost probably
   disqualifies it under the new budget, so I would spend the GPU on
   `alu-depth`'s lane instead.

*Noted for the record:* the coordinator's correction that `batch_size` 512→32 is
1.2× for this model rather than 5.8× is consistent with what I saw — these runs
are model-step-bound, not loader-bound.

## 9. Compliance

* Nothing under `data/generated/` was read, printed, sampled or summarised. All
  probes generate their own operands from `math.gcd` over `range(1, N)`.
* `--construct`, `--deep-sup` and `--teacher-force` are **lab diagnostics**. They
  set the true tables or use the true register trace and are never part of a
  submission (rules 2 and 7). Every table above says which side of the line a row
  is on.
* No hosted submission, no network call, no `one-layer` CLI invocation. No
  evaluator cells were run, for the reason in §7.
* Negative results are all here, including those that contradict my brief's
  premises and `digit-carry`'s screening rule.

## 10. Reproduction

```bash
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python

# the ceiling of whatever configuration you are about to compare against
$V lab/probe_credit.py --construct --tag ceil

# the two diagnostics
$V lab/probe_credit.py --diag --steps 1 --tag grad
$V lab/probe_credit.py --flow --tag flow
$V lab/probe_credit.py --flow --hard --hard-gate --tag flow_st

# the headline: tables learned inside the Easy step budget, LAB ONLY
$V lab/probe_credit.py --steps 20 --lr 0.3 --teacher-force 1.0 --deep-sup 1.0 \
   --tag tf20
# the legal control at the same budget
$V lab/probe_credit.py --steps 20 --lr 0.3 --tag legal20

# any sweep
bash lab/credit_sweep.sh lab/jobs_micro2.txt 6
```

Job files: `lab/jobs_{relax,curr,st,tf,b4,b5,b6,b7,micro,micro2,curve,curve2}.txt`.
