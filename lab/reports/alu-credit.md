# alu-credit — making `DigitALU` trainable, and what the step budget does to that question

**Branch** `explore/alu-credit`, from `explore/digit-carry`.
**Mandate** hold the architecture fixed (`alu-depth` owns changing it), vary
everything about how it is optimised, get `train_exact` to 1.000.
**Tooling** `lab/probe_credit.py` (subclasses `DigitALU`; `--construct` still
reports `train_exact = held_exact = 1.000` after refactoring, so the hypothesis
class is unchanged), `lab/credit_sweep.sh`, ~90 runs in `lab/credit_runs.jsonl`
and `lab/logs/`.

---

## 0a. The metric changed mid-branch — read this first

`alu-depth` falsified `train_exact` as a screen for this family: `DigitALU`
learns a **continuous relaxation riding the W×10 simplex**, and ranking on
`train_exact` actively selects for that failure mode. The honest number is
**`train_exact_hard`** — every inter-step state (register slots, carry, borrow,
and the `cond_sub` gate) snapped to its argmax.

I re-screened on it, adopted `alu-depth`'s cheap graph (`--mul-mode tree
--reduce-mode quotient`, 39 sequential steps, **constructed ceiling 1.000
including under snapping, state sharpness 1.000** — verified in my worktree),
and then spent the remaining budget on the direction the coordinator identified
as highest-value: **per-step pressure on state discreteness.**

**That direction is falsified, and cleanly.** Every lever controls sharpness
exactly as intended and none of them buys a single correct example:

| | `train_exact` | **`train_exact_hard`** | state sharpness |
|---|---|---|---|
| baseline, 3 seeds | 0.380 / 0.368 / 0.440 | 0.004 / 0.000 / 0.000 | 0.765–0.775 |
| state-entropy penalty, w=1.0, 3 seeds | 0.028 / 0.084 / 0.028 | 0.012 / 0.016 / 0.012 | **0.932–0.948** |
| temperature annealed to 0.01 | 0.016 | 0.000 | **0.975** |
| entropy + anneal together | 0.008–0.012 | 0.000 | 0.951–0.953 |
| sharpness hinge (no pressure once sharp) | 0.152–0.220 | 0.000–0.008 | 0.908–0.936 |
| Gumbel annealed to the discrete limit, 3 seeds | 0.164 / 0.296 / 0.172 | 0.004 / 0.004 / 0.008 | 0.831–0.856 |
| anneal *into* straight-through at 50% / 80% | 0.000 / 0.000 | 0.000 / 0.000 | 0.797 / 0.807 |

**Sharpness is fully controllable — 0.765 → 0.975, essentially one-hot — and
`train_exact_hard` never leaves the 0.000–0.016 band**, the same band
`alu-depth` measured across its entire depth ladder. Meanwhile `train_exact`
*collapses* from 0.38 to 0.01 as pressure rises.

The interpretation matters. The soft channel is **not** a crutch the model leans
on *in addition to* a nearly-correct discrete solution — if it were, pricing the
crutch would expose the solution underneath. It is the *entirety* of what the
model has. Sharpen it and there is nothing there. The relaxation is not loose;
it is pointing somewhere else. §9 gives the full grid.

**And I have to withdraw my own headline.** Teacher forcing on the true register
trace, which I reported at `train_exact` 1.000, reads **`train_exact_hard`
0.064** (§9.2). It rides the simplex too. What survives — and it is the one
result in ~130 runs that is off the floor at all — is that its *tables* reach
`mul_lo` 0.90 / `add_shift` 0.78 / `sub_shift` 0.817 against a random baseline of
0.23–0.28, and 78% of that within **20 optimizer steps**. So per-step inputs do
teach the tables; what they leave behind is a 10–20% residue of wrong cells that
the soft state channel absorbs. That makes the remaining problem **table
identification**, not credit assignment and not relaxation tightness — a much
narrower target, and the one I would aim the next branch at (§10).

---

## 0. Headline (as measured before the re-screen — see §9 for what survives)

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
per-step *inputs*, all 500 table cells are learned in tens of steps. Given
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
  **False.** Teacher-forced: `train_exact` 1.000, held 0.500 on a 138-example
  cohort (0.816 on the 38-example one). Short chain:
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

### 2.3 The loss punishes sharpening unless you are already nearly right

`--basin` takes the construction, corrupts `k` of its 500 table cells at random
(3 repeats) and measures what is left:

| corrupted cells `k` | 0 | 1 | 2 | 3 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| `train_exact` | 1.000 | 0.871 | 0.947 | 0.628 | 0.719 | 0.555 | 0.311 | 0.036 | 0.013 |
| `train_ce` | 0.000 | 2.08 | 0.80 | 6.06 | 3.59 | 6.87 | 10.65 | 16.62 | **17.13** |

(Non-monotonicity at small `k` is sampling noise — some cells are never
exercised by the 250 training operands.)

Two things follow, and together they explain the plateau better than anything
else I measured.

**There *is* a graded signal near the solution.** A model with ten wrong cells
still scores 0.555. So the landscape is not a cliff, and `digit-carry`'s "no
partial-credit path" is too strong as stated — it is true *far* from the
solution, not near it.

**But the cross-entropy saturates at ~17 for a sharp-and-wrong table, while a
soft random init sits at 2.25.** `ln(10) = 2.30`. So the optimiser starts at CE
2.25 in the maximum-entropy region, and *any* move toward a confident table it
has not already got right costs it up to 15 nats. The gradient therefore points
at staying soft. **This is why entropy pressure, sharp initialisation, permutation
initialisation and straight-through all make things worse rather than better
(§4.1, §4.2)** — they all force the model into the region the loss punishes,
without supplying the information needed to land in the small part of it that is
correct. It is also the same 18.7-vs-2.30 number `digit-carry` §2.4 saw when it
froze modules at the truth, now explained: a correct module inside a sharp-wrong
chain is exactly the `k` ≈ 50 row.

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
| `--loss linear` (`1 − p_correct`, bounded) | 0.016 | 0.183 |
| `--loss brier` (bounded) | 0.016 | 0.267 |

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
| `--wd 0.1` | 3000 | 0.044 | 0.417 |
| `--batch 32` (minibatch rather than full batch) | 3000 | 0.200 | 0.25 |
| `--perm-init 8 --gate-off −6` (annealed to 0) | 3000 | 0.056 | 0.30 |
| `--loss linear` / `--loss brier` | 1000 | 0.016 / 0.036 | 0.15 / 0.233 |
| `--loss linear`, lr 1.0 | 1000 | 0.016 | 0.15 |
| **`--xcurr 0.5`** (best legal number anywhere) | 3000 | **0.224** | **0.417** |
| baseline | 3000 | 0.132 | 0.317 |
| N=91 S=2 (short chain) | 3000 | 0.750 | 0.45 |
| N=91 S=2, `--loss linear` | 1000 | 0.167 | 0.333 |
| `--sym 1.0` (commutativity of `Tmul`/`Tadd`) | 3000 | 0.052 | — |
| `--trunc 1` / `--trunc 2` (truncated BPTT) | 1000 | 0.024 / 0.024 | 0.20 / 0.20 |
| `--trunc 1`, N=91 S=2 | 1000 | 0.283 | 0.283 |
| **target propagation, weight 1.0** (§6.5) | 1500 | 0.196 | **0.567** |

Truncated BPTT is worth its own line because it is the cheapest legal way to
shorten the *gradient* path without touching the forward pass (the constructed
ceiling is unchanged). Detaching the register at every Horner place cuts the
backward path from ~280 ops to ~14 and makes things slightly worse (0.024), for
a reason the §2.1 gradient profile predicts: `Tmul` sits upstream of every
detach, so truncation starves the one module that was getting a usable signal.
Shortening the gradient path is not a substitute for shortening the chain.

The bounded losses deserve a note because §2.3 predicted they would help and they
do not. If cross-entropy's 17-nat penalty for confident error is what pins the
model in the maximum-entropy region, a loss bounded by 1 should release it. It
does not — and at the short chain it makes things distinctly *worse* (0.167
against 0.750). So the CE saturation in §2.3 explains why sharpening levers fail,
but removing it does not supply the missing information; the model simply has no
gradient telling it *which* sharp table to pick.

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

### 5.1 The full chain-length ladder — and an honest complication

All three configurations have **constructed ceiling 1.000**, verified, so this is
a valid comparison. Legal training, 3,000 steps:

| chain | ops | constructed | `train_exact` @20 | `train_exact` @3000 | `sub_shift` @3000 |
|---|---|---|---|---|---|
| N=91, S=2 | ~117 | 1.000 | **0.25** (3-seed mean) | **0.750** | 0.45 |
| N=323, S=3 | ~280 | 1.000 | 0.024 (3-seed mean) | 0.132 | 0.317 |
| N=2021, S=4 | ~525 | 1.000 | 0.028 | 0.164 | **0.600** |

`train_exact` behaves as the depth story predicts — shorter is much better,
especially at budget. **`sub_shift` does not.** The *longest* chain has the
highest legal structure score of the three (0.600, and the second-highest of any
legal run in this report). I did not predict that and I am reporting it rather
than smoothing it over.

The most likely reading is that `Tsub` is the module *nearest* the loss (§2.1
measured its gradient at 10⁵× `Tmul`'s), and S=4 applies `cond_sub` far more
times per example — 7 Horner places × 11 subtractions × 5 slots against 3 × 11 ×
3 — so it accumulates far more gradient into that one table. More depth buys
`Tsub` signal and costs composition. That is consistent with everything else
here, but it is one configuration and I would want it replicated before anyone
plans on it.

The practical consequence is a warning: **`sub_shift` alone is not a sufficient
screen either.** S=4 reaches 0.600 while sitting at `train_exact` 0.164 and
`held_exact` 0.000. Screen on the structure scores *and* `train_exact` *and*
held-out — no single one of the three is safe on its own in this family.

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
0.77.** All 500 table cells (6,817 parameters) are most of the way learned
inside the budget. What takes another 300 steps is turning ~80%-correct tables into an
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

### 6.4 Held-out under teacher forcing, and the coverage question

The best held-out exactness anywhere in this repo, and still not 1.000:

| train operands (of 288 units) | held cohort | steps | `train_exact` | `held_exact` |
|---|---|---|---|---|
| 150 | **138** | 3000 | 1.000 | **0.500** |
| 250 | 38 | 1000 | 1.000 | 0.789 |
| 250 | 38 | 3000 | 1.000 | **0.816** |
| 280 | 8 | 2000 | 1.000 | 0.625 (n=8, not usable) |

The 138-example row is the only trustworthy held-out estimate here, and it says
**0.500**. Held-out improves with the number of training operands (150 → 250
takes it from 0.50 to 0.82 on their respective cohorts), which is *consistent
with* the coverage story — `Tsub`'s column for `N`'s leading zero digit is
exercised almost nowhere, and `digit-carry` §2.3's closed form put digit coverage
at 0.95–0.99 — but the two rows use different held cohorts, so I am **not**
claiming coverage is proven to be the cause. What is solid: even the procedure
that reaches `train_exact` 1.000 does not reach `held_exact` 1.000, and 1.000 is
what `MAX_T ≥ 1` requires.

### 6.5 Target propagation — the legal analogue

Teacher forcing works because it supplies per-step *inputs*. The legal way to get
those without computing them is to make them **learned latents**: `LatentTrace`
predicts the whole ~70-step register trace from the same inputs the model already
sees; tables and latents train jointly under (a) local consistency — one op
applied to latent `t` must reproduce latent `t+1` — and (b) two boundary
conditions using only given quantities (the last latent is the label; the first
input is the model's own learned `zero`). This is method-of-auxiliary-coordinates
/ target propagation. It supplies no arithmetic, it is discarded at eval, and it
*is* expressible under the evaluator's fixed loop.

**It is the only legal procedure in this report that moves the tables.**

| procedure | steps | `train_exact` | `mul_lo` | `add_shift` | `sub_shift` |
|---|---|---|---|---|---|
| random init (reference) | — | — | 0.28 | 0.275 | **0.233** |
| best other legal (`--xcurr 0.5`) | 3000 | 0.224 | 0.29 | 0.26 | 0.417 |
| best other legal (short chain, S=2) | 3000 | 0.750 | 0.29 | 0.305 | 0.45 |
| **target propagation, weight 1.0** | 1500 | 0.196 | 0.31 | **0.38** | **0.567** |
| teacher forcing (LAB ONLY), for scale | 20 | 0.000 | 0.77 | 0.51 | 0.78 |

`train_exact` 0.196 is squarely inside the baseline seed band, so **as a recipe it
is null** — it does not train the model. But `sub_shift` 0.567 against a random
baseline of 0.233 is the largest structural movement any legal procedure
produced, and it is produced by exactly the mechanism teacher forcing identified:
per-step *inputs*, here learned rather than computed. That is a real signal on a
real mechanism, from one untuned configuration.

Two caveats keep me from recommending it. It is **~12× the cost per step**, which
under step famine is close to disqualifying on its own. And it is still 0.567
against teacher forcing's 0.78-in-20-steps, so the learned latents are a much
weaker substitute for the true trace than the mechanism would suggest. Raw
results in `lab/credit_runs.jsonl` under `z_tprop*`.

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
| **truncated BPTT** (`--trunc`, detach the register every *k* Horner places) | `r.detach()` in `forward`; forward is unchanged, so the constructed ceiling is unchanged | yes, and it makes the backward *cheaper* |
| magnitude curriculum on \|x\| | per-example loss weight from `input_ids` | marginal — needs a ramp |
| entropy / commutativity / `Tsub ∘ Tadd = id` | pure `training_loss` terms | yes |
| every init family here | `build_model` | yes (free) |
| every optimiser / lr / wd / schedule | `build_optimizer` | yes |
| target propagation (§6.5) | latents in `aux`, terms in `training_loss` | **no** — ~12× cost per step |

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

> **Updated after the re-screen.** §8 was written when `train_exact` was the
> screen and depth looked like the binding variable. `alu-depth` removed 85% of
> the chain with no movement in `train_exact_hard`, and §9 shows state pressure
> does not move it either. The parts of §8 that were about *depth* are
> superseded by §10; the parts about *per-step inputs* are the ones that
> survived both metric changes and are restated there.

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
5. **The one legal thread I would not cut is target propagation (§6.5).** It is
   the only legal procedure that moved the tables at all — `sub_shift` 0.567
   against 0.233 random and 0.417/0.45 for the next best — from a single untuned
   configuration, and it moves them by exactly the mechanism teacher forcing
   identified. It did **not** train the model (`train_exact` 0.196, inside the
   baseline band) and it costs ~12× per step, so it is not a candidate under the
   current budget. But if `alu-depth` shortens the chain, the per-step cost falls
   with it and target propagation becomes affordable at the same time as it
   becomes easier — those two compound. That is the one place I would spend
   further training-side GPU, and I would spend it *after* the depth work, not
   before.

*Noted for the record:* the coordinator's correction that `batch_size` 512→32 is
1.2× for this model rather than 5.8× is consistent with what I saw — these runs
are model-step-bound, not loader-bound.

## 9. The re-screen: state-discreteness pressure on the cheap graph

Base: `alu-depth`'s `tree:quotient` graph, N=323 S=3, 39 sequential steps,
7,474 parameters, 2,000 steps, AdamW lr 3e-2, full batch. **Constructed ceiling
verified in my worktree: `train_exact` 1.000, `train_exact_hard` 1.000,
`held_exact_hard` 1.000, state sharpness 1.000.** So the discrete solution is in
the class and survives snapping; everything below is about reaching it.

| lever | `train_exact` | **`train_exact_hard`** | best hard | sharpness |
|---|---|---|---|---|
| baseline seed 0 / 1 / 2 | 0.380 / 0.368 / 0.440 | 0.004 / 0.000 / 0.000 | 0.004 | 0.765 / 0.775 / 0.775 |
| `--state-ent 0.03` | 0.244 | 0.008 | 0.008 | 0.865 |
| `--state-ent 0.1` | 0.108 | 0.004 | 0.004 | 0.931 |
| `--state-ent 0.1 --state-ent-warm 0.3` | 0.084 | 0.004 | 0.004 | 0.892 |
| `--state-ent 0.3` | 0.160 | 0.004 | 0.004 | 0.900 |
| `--state-ent 0.3 --state-ent-warm 0.3` | 0.168 | 0.000 | 0.004 | 0.889 |
| `--state-ent 1.0` seed 0 / 1 / 2 | 0.028 / 0.084 / 0.028 | 0.012 / **0.016** / 0.012 | 0.016 | 0.948 / 0.938 / 0.932 |
| `--state-ent 3.0` | 0.012 | 0.000 | 0.000 | 0.940 |
| `--sharp-target 0.9 --state-ent 1.0` | 0.220 | 0.004 | 0.008 | 0.913 |
| `--sharp-target 0.9` + warm | 0.200 | 0.008 | 0.008 | 0.908 |
| `--sharp-target 0.99 --state-ent 3.0` | 0.152 | 0.000 | 0.008 | 0.936 |
| `--tau-final 0.15` | 0.044 | 0.004 | 0.012 | 0.886 |
| `--tau-final 0.05` | 0.012 | 0.004 | 0.004 | 0.959 |
| `--tau-final 0.01` | 0.016 | 0.000 | **0.016** | **0.975** |
| `--gumbel 1.0 → 0` seed 0 / 1 / 2 | 0.164 / 0.296 / 0.172 | 0.004 / 0.004 / 0.008 | **0.020** | 0.856 / 0.842 / 0.831 |
| `--gumbel 1.0 → 0 --tau-final 0.05` | 0.024 | 0.008 | 0.008 | 0.935 |
| `--hard-at 0.5` / `0.8` (anneal *into* ST) | 0.000 / 0.000 | 0.000 / 0.000 | 0.004 | 0.797 / 0.807 |
| `--state-ent 0.3 --tau-final 0.05` | 0.008 | 0.000 | 0.000 | 0.951 |
| `--state-ent 1.0 --tau-final 0.05` | 0.012 | 0.000 | 0.008 | 0.953 |
| `--state-ent 1.0`, 20-step tier budget | 0.008 | 0.004 | 0.012 | 0.869 |
| baseline, 20-step tier budget | 0.028 | **0.016** | 0.016 | 0.746 |

**Nineteen configurations, three seeds on the three that mattered, and
`train_exact_hard` never exceeds 0.020 — against a trivial floor of ~0.004 and
a constructed ceiling of 1.000.**

Three readings worth separating:

1. **The levers work.** Sharpness is a controlled variable: 0.765 at baseline,
   0.93–0.95 under entropy pressure, 0.975 under a hard temperature anneal.
   These are not failures to apply pressure.
2. **Discreteness is not the missing ingredient.** A model at sharpness 0.975 —
   states essentially one-hot — gets 0.000 exact under snapping. If the soft
   channel were a *shortcut around* a nearly-found discrete solution, removing
   it would reveal the solution. It reveals nothing.
3. **Pressure destroys the soft solution without building a discrete one.**
   `train_exact` falls monotonically with pressure (0.38 → 0.16 → 0.03 → 0.01)
   while `train_exact_hard` stays flat at ~0. The two optima are not near each
   other, and the path between them is not downhill in the loss.

The one shape of pressure designed to avoid (3) — the `--sharp-target` hinge,
which applies no gradient once a state is already sharp, so it cannot
over-sharpen — does preserve `train_exact` better (0.152–0.220 against
0.012–0.028 for plain entropy at comparable sharpness) and still yields
`train_exact_hard` 0.000–0.008. Preserving the soft solution does not help
either.

**Long runs at 12,000 steps** (6× the sweep) confirm it is not a slow transition.
All four completed:

| 12,000 steps | `train_exact` | **`train_exact_hard`** | sharpness |
|---|---|---|---|
| baseline | **0.616** | **0.000** | 0.762 |
| `--state-ent 0.1 --state-ent-warm 0.3` | 0.460 | 0.008 | 0.875 |
| `--gumbel 1.0 → 0 --tau-final 0.15` | 0.028 | 0.004 | 0.926 |
| `--tau-final 0.05` | 0.004 | **0.020** | **0.967** |

Read down the table: as pressure rises, sharpness climbs monotonically
0.762 → 0.967 and `train_exact` collapses monotonically 0.616 → 0.004, while
`train_exact_hard` stays pinned in 0.000–0.020 throughout. Six times the sweep
length, the soft fit keeps improving, and the discrete fit never starts. Full
traces in `lab/logs/i_long_*.log`.

Two confirmatory re-screens (`k_tprop_hard`, `k_xcurr_hard`) were still running
at cutoff; their logs land in `lab/logs/`. Neither can change a conclusion here —
both configurations already read `train_exact` 0.196 and 0.224 soft, i.e. inside
the baseline band before snapping, so their hard numbers are bounded above by
that.

### 9.1 Which of my earlier conclusions survive the metric change

| conclusion | survives? |
|---|---|
| **Every legal training procedure is null.** | **Yes, and more strongly.** Ranked on `train_exact` the legal levers were inside the seed band; ranked on `train_exact_hard` they are all inside 0.000–0.020, i.e. at the floor. The metric change makes this conclusion safer, not weaker. |
| **The 0.2 plateau is a degenerate solution, not a partial table** (§0.2, gauge-invariant structure scores at chance) | **Yes** — and `alu-depth`'s finding is the mechanism I was missing. The structure score said the tables were not arithmetic; `train_exact_hard` says the *states* were carrying the answer instead. Two independent measurements of the same thing. |
| **`train_exact` → 1.000 does not imply `held_exact` → 1.000** (§0.5) | **Yes**, and the mechanism is now identified: the state simplex is the memorisation channel. This is the same falsification `alu-depth` made, reached independently from held-out CE blow-up. |
| **Sharpening levers (entropy, sharp init, permutation init, ST) make things worse**, explained by the CE basin (§2.3) | **Yes** — and §9 is the strongest version of it. The basin result predicted exactly this: sharpening into a wrong table costs up to 15 nats, and there is no gradient telling the model *which* sharp table to pick. |
| **Chain length: shortening buys `train_exact`, not the algorithm** (§0.4, §5) | **Superseded and confirmed in the stronger direction.** `alu-depth` measured `train_exact_hard` = 0.000–0.012 across 257→39 steps; my §5 said the short chain fits the degenerate solution faster. Same conclusion, theirs is the better-controlled experiment. My S=2-vs-S=3 comparison shares the confound they identified (operand width, output length, cohort and training-set size all move together) — **discount my §5 numbers in favour of theirs.** |
| **Teacher forcing reaches `train_exact` 1.000** (§6.1) | **No — corrected.** `train_exact_hard` is **0.064** (§9.2). It rides the simplex like everything else. It is still the only run in the family off the 0.000–0.020 floor, and its tables are 78–90% correct, but "reaches 1.000" was an artefact of the wrong metric and I have withdrawn it. |
| **Tables reach 78% structure in 20 optimizer steps under teacher forcing** (§0.3) | **Yes** — the structure scores are computed on the *parameters*, not the states, so snapping does not affect them. This is the finding I would still lead with. |
| **Target propagation is the only legal lever that moves the tables** (§6.5) | **Yes** on the same reasoning (parameter-level metric), but it never trained the model and is now less attractive still. |

### 9.2 Does teacher forcing survive snapping? No — and *how* it fails is the most useful number in this report

Re-ran the headline configuration with the discrete evaluation added
(`lab/logs/k_tf_hard.log`, N=323 S=3 R=11, 1,000 steps):

```
train_exact=1.000  held_exact=0.789
train_exact_hard=0.064  held_exact_hard=0.053
struct  mul_lo=0.90  mul_hi=0.62  add_shift=0.78  sub_shift=0.817
```

**`train_exact` 1.000 → `train_exact_hard` 0.064.** So the answer to §0.2 is no:
teacher forcing rides the simplex too, and my earlier headline overstated what it
achieved. I am correcting it rather than defending it.

But 0.064 is **16× the baseline's 0.004 and above the entire 0.000–0.020 band**
that every other run in this report and every rung of `alu-depth`'s depth ladder
sits in. It is the only number in the family that is off the floor at all. And
the structure scores say why: `mul_lo` 0.90, `add_shift` 0.78, `sub_shift` 0.817
against a random baseline of 0.23–0.28. **The parameters really are ~80–90% the
intended tables.**

Put those two together and the diagnosis sharpens into something quite different
from where this branch started:

> Per-step inputs **do** teach the tables — that conclusion survives both metric
> changes. What they do not do is produce a *discrete* transducer, because the
> remaining 10–20% of wrong table cells are papered over by the soft state
> channel. The model is not choosing the continuum *instead of* the algorithm
> here; it has most of the algorithm and is using the continuum to absorb the
> errors it has left.

That reframes the residual as a **table-identification** problem — which cells
are still wrong, and why the data does not pin them — rather than a credit
assignment or a relaxation problem. It is consistent with §6.4 (held-out rises
with the number of training operands) and with `digit-carry` §2.3's coverage
closed form, and it is a much narrower target than "make a 39–280 step rollout
trainable".

It is also, unfortunately, measured under a **lab-only** procedure (rule 2), so
it is a statement about what the hypothesis class can be driven to, not a recipe.

### 9.3 Every headline result re-screened

Original 257-step graph, `lab/logs/k_*.log`:

| run | `train_exact` | **`train_exact_hard`** | `held_exact` | `held_exact_hard` | `sub_shift` |
|---|---|---|---|---|---|
| baseline, 1,000 steps | 0.100 | 0.008 | 0.000 | 0.000 | 0.417 |
| **teacher forcing, 1,000 steps** (LAB ONLY) | 1.000 | **0.064** | 0.789 | **0.053** | **0.817** |
| **teacher forcing, 3,000 steps** (LAB ONLY) | 1.000 | **0.072** | 0.816 | 0.026 | **0.817** |
| **N=91 S=2 short chain, 3,000 steps** | 0.750 | **0.183** | 0.000 | 0.000 | 0.45 |

The 1,000-vs-3,000-step teacher-forcing pair is worth reading carefully, because
it is the cleanest single demonstration of what the metric change exposed.
Training 3× longer moves the soft metrics *up* (`held_exact` 0.789 → 0.816) and
the hard held metric *down* (`held_exact_hard` 0.053 → **0.026**), with
`train_exact_hard` essentially flat (0.064 → 0.072) and the structure scores
unchanged (`sub_shift` 0.817 both times). **The extra training does not buy more
algorithm; it buys more reliance on the soft channel.** Anyone ranking this
family on `train_exact` or `held_exact` would have read that as steady progress.

Two things I did not expect and am reporting straight.

**The short chain has the best `train_exact_hard` in the whole branch (0.183),
better than teacher forcing.** That cuts against `alu-depth`'s ladder, which read
0.000–0.012 at every depth. The two are reconcilable and theirs is the
better-controlled measurement: my S=2 point changes operand width, output length,
cohort and training-set size along with the chain — the exact confound they
identified — and its `held_exact_hard` is **0.000**, so whatever those 11 of 60
training examples are, they do not generalise. I would not build on this number,
but I would not hide it either: **someone should re-run S=2 against `alu-depth`'s
controlled ladder with `train_exact_hard` and settle it**, because if depth does
move the hard metric under a clean control, that changes the plan.

**Teacher forcing's held-out survives snapping in proportion** (0.789 → 0.053
train-side 1.000 → 0.064; held 0.789 → 0.053). The ratio is the same on both
splits, which is what you expect if the soft channel is doing a fixed fraction of
the work everywhere rather than memorising the training split specifically.

## 10. Updated recommendation

**The relaxation and the discrete target are different problems — and the fix is
not more pressure on the relaxation. It is to stop optimising a relaxation.**

That is a stronger claim than "state pressure did not work", and §9 is what
licenses it. Pressure is not a knob that was applied too weakly: sharpness
reaches 0.975, states are one-hot to three digits, and the model is still wrong
on 99.6% of *training* examples under snapping. There is no gradient anywhere in
this family that distinguishes the correct discrete transducer from the
incorrect one; the soft loss is minimised by a continuous object that has no
discrete neighbour.

Three things follow, in the order I would spend GPU on them.

1. **The one measurement that still points somewhere is per-step inputs.**
   Under teacher forcing the *tables* reach `sub_shift` 0.78 (S=3) and 0.967
   (S=4) in **20 optimizer steps** — a parameter-level metric that snapping does
   not touch, so the metric change leaves it standing (§9.1). Nothing else in
   ~110 runs moved the parameters at all. Whatever the eventual architecture, the
   thing that has to be arranged is that **each learned table gets a target it
   can be right or wrong about on its own**, rather than through a rollout.
2. **Search the discrete space directly rather than descending a relaxation.**
   The basin measurement (§2.3) says the neighbourhood of the solution is
   informative — 10 of 500 cells wrong still scores `train_exact` 0.555 — so a
   *discrete* local search has signal where the gradient does not. I did not run
   this and it is the obvious next probe: greedy coordinate descent over the 500
   argmax cells, evaluated with states snapped, on the 39-step graph (now ~30 ms
   per evaluation, so a full sweep of 500×12 candidates is minutes). It answers a
   question nobody has asked: *is the discrete landscape itself benign, and only
   the gradient estimator bad?* Either answer is decisive — a success names the
   legal surrogate to build (a score-function/ES estimator, which needs no
   Jacobian), and a failure closes the entire `DigitALU` family rather than just
   the training-procedure lane.
3. **Do not spend more on relaxation schedules.** Between `alu-depth`'s
   straight-through results at three depths and §11's nineteen configurations,
   temperature, entropy, Gumbel, hinge, anneal-into-ST and every combination of
   them are covered. `--sharp-target` (pressure only while soft) was the last
   untried *shape* of that idea and it behaves like the rest.

**Method note for whoever picks this up.** Report `train_exact_hard`, state
sharpness, *and* the gauge-invariant parameter scores together. §9 has
configurations with sharpness 0.975 and zero correctness, §5.1 has one with
`sub_shift` 0.600 and zero correctness, and §0.2 has plenty with `train_exact`
0.38 and chance-level tables. **Each of the three metrics has a configuration
that fools it. None of them is safe alone.**

## 13. Compliance

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

## 14. Reproduction

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

```bash
# the basin around the exact solution
$V lab/probe_credit.py --basin --tag basin
```

Job files: `lab/jobs_*.txt` (`relax`, `curr`, `st`, `tf`, `b4`–`b7a`, `all`,
`micro`, `micro2`, `curve`, `curve2`, `loss`, `trunc`). Per-run logs in
`lab/logs/<tag>.log`; one JSON line per run in `lab/credit_runs.jsonl` with the
full argv, `train_exact`, `held_exact`, the raw table accuracies and the
gauge-invariant `struct` scores.

No evaluator runs, so `lab/archive.jsonl` is untouched by this branch — see §7
for why.
