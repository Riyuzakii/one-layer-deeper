# discrete-search — is the discrete landscape benign, or is the objective empty?

**Branch** `explore/discrete-search`, from `explore/alu-credit`.
**Mandate** the decisive diagnostic for the whole `DigitALU` family: run a
*direct discrete local search* over the table cells, with states snapped hard,
on `alu-depth`'s `tree:quotient` graph. A success names the legal surrogate to
build; a failure closes the family.
**Tooling** `lab/probe_search.py` (new), `lab/search_runs.jsonl`, `lab/logs/`.

---

## 0. Verdict

**No. Discrete search does not find the tables, and the reason is not the
search — it is the objective.**

The headline number: from a uniformly random table assignment, greedy
coordinate descent over all 1,007 discrete cells converges in ~5 sweeps to
`train_exact_hard` **0.028–0.048** (5 seeds), `held_exact_hard` **0.000**, and
gauge-invariant structure scores **0.23–0.31** — i.e. *chance*. Parallel
simulated annealing with 2,560 independent chains and **1.5 × 10⁸ objective
evaluations** finishes at `train_exact_hard` **0.064**, `held_exact_hard`
**0.000**, structure **0.24–0.28** — the same place, with the table agreeing
with the truth on 0.262 of cells against a chance value of 0.258. That is the
same signature ~160 gradient procedures produce, reached by a completely
different optimiser.

Three measurements say *why*, and together they are stronger than the negative
result itself:

1. **The objective is informative only inside a Hamming ball of radius ≈10–20
   cells (1–2 % of the table).** Corrupt 20 of 1,007 cells of the exact solution
   and mean digit accuracy is 0.43; corrupt 100 and it is 0.14; corrupt 200 and
   it is 0.116 against a chance floor of ~0.105 — about **one standard error**.
   A random assignment is ~900 cells wrong. The search starts, and stays, on a
   plateau that is statistically indistinguishable from chance.
2. **Handing the search three of the four modules *exactly* does not help.**
   With `Tmul`, `Tadd`, the constants and the quotient scorer set to the truth
   and only `Tsub` (400 cells) random, greedy reaches `train_exact_hard` 0.008 /
   0.024 / 0.044 and `sub_shift` 0.233–0.283 — chance; annealing with 7.7 × 10⁷
   evaluations on that same 400-cell subproblem reaches 0.072 and `sub_shift`
   0.267 — still chance. Same for `Tadd` alone and `Tmul` alone. **This is not a
   credit-assignment problem, a chain-length problem, or a joint-search problem.
   A single 400-cell table is not identifiable from the end-of-chain label when
   everything else is perfect.**
3. **The information budget is exhausted.** The 250 training operands read 967
   of the 1,007 cells; describing those cells costs **2,318 bits**, and the
   labels supply **2,491 bits** (250 × 3 digits × log₂10). Ratio **1.075**. The
   solution is *just barely* uniquely determined and there is essentially **no
   redundancy** — which is exactly the regime in which an objective is a needle
   in a haystack with no gradient, discrete or continuous.

**Consequence — Stage 2 is moot, and not for legality reasons.** Every surrogate
in my brief (population-in-forward, score-function/ES, a custom optimizer
reading forward-written state) is a *different estimator of the same
objective*. My search already evaluates that objective 10⁸ times from within
what is, semantically, "one forward per candidate batch"; a tier-faithful Easy
run affords perhaps 10⁶. The objective, not the estimator and not the step
budget, is what is empty. **I did not ship a submission.**

**The one number that moved, for the record.** The best `train_exact_hard`
anywhere in this family is **0.080** — a float model trained 2,000 AdamW steps,
snapped to argmax, then discretely polished (§4.1) — against `alu-credit`'s best
legal gradient result of 0.020. It is 4× a floor and it is still a floor:
`held_exact_hard` 0.000, tables at chance, `cell_agree` 0.239 (*below* the 0.258
chance level).

**What this closes and what it does not.** It closes *learning `DigitALU` from
the end-of-chain label by any optimiser*. It does **not** touch the hypothesis
class: `--construct` is still 1.000 soft and hard. And it does not contradict
`alu-credit`'s one positive — per-step *inputs* still teach the tables — it
explains it: teacher forcing works because it replaces one 2,491-bit end-of-chain
signal with ~70 per-op signals, which is a different and far larger information
budget. The family reopens if and only if someone finds a **legal** source of
per-step signal. `alu-credit` §6.5's target propagation is the only candidate on
the table and it was null.

---

## 1. The tool, and why its correctness is load-bearing

With every inter-step state snapped to its argmax — the only honest evaluation
of this family (`alu-depth` §2.3) — `DigitALU`'s forward pass has no floating
point in it at all. Every `softmax` is an `argmax`; every
`einsum("bu,bv,bc,uvco->bo", ...)` against one-hot inputs is a table lookup.

`lab/probe_search.py` re-implements `alu-depth`'s `tree:quotient` graph as a
pure **integer transducer**, batched over a leading *population* dimension so
that thousands of candidate table assignments are evaluated in one pass.

**Correctness gate (`--verify`).** The integer sim is compared against the float
`DigitALU` with `hard=True`, on the construction and on three random inits,
matching `train_exact`, `train_digit` and `held_exact` to 1e-6:

```
[verify] seed=0 construct=True  float(exact=1.0000 digit=1.0000 held=1.0000) int(...) MATCH=True
[verify] seed=0 construct=False float(exact=0.0040 digit=0.1333 held=0.0000) int(...) MATCH=True
[verify] seed=1 construct=False float(exact=0.0000 digit=0.0987 held=0.0000) int(...) MATCH=True
[verify] seed=2 construct=False float(exact=0.0000 digit=0.0987 held=0.0000) int(...) MATCH=True
```

So every number below is `train_exact_hard` **by construction**, not by
approximation, and the search is exploring exactly the space the trained model
is snapped into.

### 1.1 The search space

| table | shape | candidates/cell | cells |
|---|---|---|---|
| `mul_lo` | (10,10) | 10 | 100 |
| `mul_hi` | (10,10) | 10 | 100 |
| `add_d` | (10,10,2) | 10 | 200 |
| `add_c` | (10,10,2) | 2 | 200 |
| `sub_d` | (10,10,2) | 10 | 200 |
| `sub_b` | (10,10,2) | 2 | 200 |
| `zero`, `carry0`, `borrow0` | — | 10/2/2 | 3 |
| `sel` (quotient scorer) | (2,2) score table | 2 | 4 |
| | | | **1,007** |

`sel` is the float model's 5-parameter `nn.Linear(2·Cb, 1)` in its exact
canonical form: with one-hot borrow states the layer is a function of the pair
`(borrow_m, borrow_{m+1})` only, so `score[a][b] = W[0,a] + W[0,Cb+b] + bias` is
a faithful reparameterisation. Ties are broken toward the smallest `m`.

### 1.2 The objective

`digit` = mean over the 250 training operands × 3 output digit positions of
"predicted digit == target digit" (graded; chance ≈ 0.10).
`exact` = `train_exact_hard`, the headline (all 3 digits right).
Search ranks on `digit`; every table also reports `exact`, held-out, the
gauge-invariant structure scores, and `cell_agree` (fraction of cells equal to
the construction — meaningful only in the repair experiments, where the
uncorrupted cells fix the gauge).

Cost: ~1 s per full 1,007-cell greedy sweep; ~60 ms per parallel-Metropolis step
regardless of population size from 320 to 2,560 chains (the workload is
kernel-launch bound, so **population is free** — the same fact that made the
"population inside one forward" surrogate attractive).

---

## 2. The basin: how far from the solution does the objective still see?

`--basin` corrupts `k` randomly chosen cells of the construction (5 repeats) and
measures what is left. `alu-credit` §2.3 ran this on the 500 digit-output cells
of the old graph; this is the same measurement over the full 1,007-cell space of
the graph that is actually being used.

**N = 323, S = 3, `tree:quotient`:**

| corrupted `k` | 0 | 1 | 2 | 3 | 5 | 10 | 20 | 50 | 100 | 200 | 500 | 1007 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `digit` | 1.000 | 0.940 | 0.846 | 0.921 | 0.860 | 0.783 | 0.434 | 0.221 | 0.142 | 0.115 | 0.096 | 0.111 |
| `exact` | 1.000 | 0.928 | 0.818 | 0.883 | 0.822 | 0.693 | 0.292 | 0.083 | 0.022 | 0.003 | 0.000 | 0.001 |

**N = 10403, S = 5 (m1's modulus, 83 sequential steps):**

| corrupted `k` | 0 | 1 | 2 | 3 | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `digit` | 1.000 | 0.985 | 0.851 | 0.832 | 0.702 | 0.587 | 0.286 | 0.194 | 0.126 | 0.116 | 0.099 |
| `exact` | 1.000 | 0.976 | 0.767 | 0.726 | 0.550 | 0.398 | 0.059 | 0.010 | 0.001 | 0.000 | 0.000 |

The shape is the same at both scales and is the central fact of this report.
The standard error of `digit` on 750 digits at chance is ≈ 0.011, so:

* at `k` = 100 (10 % of cells wrong) the signal is ≈ **3 σ** — spread over 900
  cells, i.e. ~0.003 σ per cell;
* at `k` = 200 (20 % wrong) it is ≈ **1 σ**;
* at `k` ≥ 500 it is **indistinguishable from chance**.

A uniformly random assignment sits at `k` ≈ 900. **There is nothing there to
descend.** This is a property of the loss surface, and it is identical for a
gradient, a coordinate sweep, an evolution strategy or a REINFORCE estimator —
they all read the same surface.

---

## 3. Repair: the search works, and works beautifully, near the solution

`--repair k` corrupts `k` cells and then runs the search. This is the cleanest
available measurement of the *radius of the basin of attraction*, and it is what
tells us the search implementation is not the weak link. 5 reps each,
block-greedy with prefix verification, N = 323.

| corrupted `k` | reps reaching **1.000** | mean `train_exact_hard` | mean `held_exact_hard` | mean `sub_shift` |
|---|---|---|---|---|
| 1 | **4/5** | 0.973 | 0.953 | 1.000 |
| 2 | 2/5 | 0.965 | 0.963 | 1.000 |
| 3 | 3/5 | 0.962 | 0.979 | 1.000 |
| 5 | 1/5 | 0.846 | 0.821 | 0.967 |
| 10 | 2/7 (7 reps) | 0.727 | 0.613 | 0.945 |
| 20 | 1/5 | 0.474 | 0.374 | 0.893 |
| 50 | 0/5 | 0.264 | 0.189 | 0.783 |
| 100 | 0/5 | 0.078 | 0.021 | 0.693 |
| 200 | 0/5 | 0.042 | 0.000 | 0.643 |
| 400 | 0/5 | 0.034 | 0.000 | 0.500 |

Two things worth stating plainly.

**The search reaches `held_exact_hard` = 1.000.** Every rep that recovers gets
*all 38* held-out operands exactly right, from a 250-operand training set, with
a 6,820-parameter digit-indexed model. The starting point is oracle-derived (a
corrupted construction), so this is **not** a claim that anything was learned —
but it is the first 1.000 held-out exact produced in this repo by an *optimiser*
rather than by `--construct` itself, and the best comparable number from a
procedure is teacher forcing's 0.816 (which is also illegal). It confirms —
from a third independent direction, after `--construct` and after `alu-credit`
§6.3's table dump — that the target is reachable, that the parameterisation is
right, and that nothing about generalisation is the obstacle. **Only the route
to it is missing.**

**The failures are not near-misses; they are departures.** At `k` = 10, rep 1
started 10 cells wrong and finished **103 cells wrong** (`cell_agree`
0.9921 → 0.8977) at `digit` 0.668. Every accepted move strictly increased the
objective — prefix verification guarantees that — so the search climbed *away*
from the truth into a different local optimum. That is the landscape being
rugged, not the search being greedy.

The 400-cell row is worth its own note: even with 40 % of the table randomised,
the search still ends at `cell_agree` 0.637 and structure scores 0.45–0.60 —
well off the 0.23–0.28 random baseline. There is *some* pull toward the
solution far out. It is just nowhere near enough.

---

## 4. From scratch — the decisive run

Uniformly random assignment, greedy block-coordinate search, 5 seeds, N = 323:

| seed | init `digit` | final `digit` | **`train_exact_hard`** | `held_exact_hard` | `cell_agree` | `mul_lo` | `add_shift` | `sub_shift` |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.111 | 0.424 | **0.048** | 0.000 | 0.265 | 0.28 | 0.275 | 0.25 |
| 1 | 0.099 | 0.393 | 0.032 | 0.000 | 0.289 | 0.24 | 0.310 | 0.283 |
| 2 | 0.103 | 0.368 | 0.040 | 0.000 | 0.277 | 0.27 | 0.265 | 0.267 |
| 3 | 0.089 | 0.352 | 0.028 | 0.000 | 0.268 | 0.25 | 0.275 | 0.267 |
| 4 | 0.108 | 0.380 | 0.036 | 0.000 | 0.249 | 0.28 | 0.250 | 0.25 |

Converged in ~5 sweeps every time. `cell_agree` ≈ 0.27 is **chance** (a random
table agrees with the truth on 0.258 of cells: 600 cells at 1/10 and 400 at
1/2), and so are all four structure scores against the 0.23–0.28 random
baseline.

Three comparisons make this the decisive number:

* **Against gradient descent.** `alu-credit` §9's best legal `train_exact_hard`
  over 19 configurations and ~130 runs is **0.020**. Discrete greedy gets
  **0.048** — 2.4× better, and *identical in kind*: both are at the floor with
  chance-level tables. Two optimisers with nothing in common find the same
  degenerate solutions.
* **Against the basin.** `digit` 0.40 is the value the construction has at
  `k` ≈ 20 wrong cells. The search's solution has ~735 wrong cells. **The same
  objective value is reached by assignments 700 cells apart** — the level sets
  are enormously degenerate, and the degenerate branch is overwhelmingly the
  larger one.
* **Against a stronger search** (§5, §6).

### 4.1 A gradient warm start does not put you in the basin either

The obvious hybrid: train the float model, snap it, and let the discrete search
finish the job. `--float-steps 2000` trains `tree:quotient` with the standard
AdamW lr 3e-2 full-batch cell, then snaps to argmax and searches. 3 seeds.

| seed | float `train_exact` (soft) | float `train_exact_hard` | **snapped** `digit` | after polish `digit` | after polish `train_exact_hard` | `held_exact_hard` | `cell_agree` |
|---|---|---|---|---|---|---|---|
| 0 | 0.312 | 0.004 | 0.155 | 0.453 | **0.056** | 0.000 | 0.223 |
| 1 | 0.424 | 0.008 | 0.167 | 0.424 | 0.052 | 0.000 | 0.253 |
| 2 | 0.372 | 0.000 | 0.140 | 0.449 | **0.080** | 0.000 | 0.239 |

(The float column independently reproduces `alu-depth` §2.3's `tree:quotient`
cell — soft 0.37–0.44, hard 0.000–0.008 — which is a useful cross-check that
this branch and that one are measuring the same object.)

Read the "snapped" column: **2,000 steps of gradient descent leave the table at
`digit` 0.14–0.17 against a random-table floor of 0.10.** After 2,000 steps the
discrete content of the model is barely distinguishable from a random draw, and
`cell_agree` after polishing (0.22–0.25) is *below* the 0.258 chance level.

The hybrid does produce the **best `train_exact_hard` anywhere in this family:
0.080**, against `alu-credit`'s best legal gradient result of 0.020 and this
branch's cold-start 0.048. It is still at the floor, `held_exact_hard` is still
0.000, and the tables are still at chance. Four times a floor is a floor.

---

## 5. Module-restricted search — the result that removes every alternative explanation

If the failure were about *depth*, *credit assignment through 39 steps*, or
*many interacting tables*, then giving the search everything except one table
should fix it. It does not. Everything outside `--modules` is set to the
construction and held there; only the named tables start random and are
searched. 3 seeds each, greedy, N = 323.

| searched (random) | frozen at truth | cells searched | **`train_exact_hard`** (mean/max) | `digit` | structure of the searched table |
|---|---|---|---|---|---|
| `Tsub` only | mul, add, consts, sel | 400 | **0.025 / 0.044** | 0.327 | `sub_shift` 0.233–0.283 (chance) |
| `Tadd` only | mul, sub, consts, sel | 400 | **0.015 / 0.032** | 0.312 | `add_shift` 0.27–0.29 (chance) |
| `Tmul` only | add, sub, consts, sel | 200 | **0.037 / 0.052** | 0.339 | `mul_lo` 0.24–0.35 (chance) |
| `Tmul`+`Tadd` | sub, consts, sel | 600 | 0.020 / 0.032 | 0.324 | chance |
| `Tadd`+`Tsub` | mul, consts, sel | 800 | 0.017 / 0.024 | 0.343 | chance |
| `Tmul`+`Tsub` | add, consts, sel | 600 | 0.029 / 0.040 | 0.359 | chance |
| all (§4) | — | 1,007 | 0.037 / 0.048 | 0.383 | chance |
| `Tsub` only, **annealed** (§6) | mul, add, consts, sel | 400 | 0.072 | 0.435 | `sub_shift` 0.267 (chance) |

**Read the right-hand column.** A 200-cell table — `Tmul`, the multiplication table,
sitting at the very front of the graph with a *perfect* adder, a *perfect*
subtractor, perfect constants and a perfect quotient selector downstream of it —
is not identifiable from the end-of-chain label. Its `mul_lo` finishes at 0.24–0.35
against a 0.28 random baseline.

Note also that searching *one* table is no easier than searching *all* of them
(0.025 vs 0.037): the search-space size is irrelevant, which is precisely what
you expect when the objective is flat.

This kills, in one table, every explanation of `DigitALU`'s failure that has been
proposed in this repo — depth (`alu-depth`), credit assignment through the chain
(`alu-credit`'s original mandate), relaxation tightness (`alu-credit` §9), and
the gradient estimator (this branch). What is left is the objective.

---

## 6. A serious global search — parallel simulated annealing

Greedy is a weak optimiser, so I ran a strong one. Each population member is an
*independent Metropolis chain* over the full genome; one cell is re-drawn per
chain per step and accepted on the digit objective; every chain advances by one
proposal per forward pass. Population is free (§1.2), so this is 2,560 chains at
~60 ms per generation.

**`sa_sub` — 400 cells, everything else exact, 2,560 chains, 2 × 15,000
proposals per chain = 7.7 × 10⁷ objective evaluations, then a greedy polish of
the three best chains.** This is the most favourable configuration I can
construct for a search: the smallest interesting subproblem, with a perfect
machine around it, and a global optimiser with a serious budget.

| | `digit` | `train_exact_hard` | `held_exact_hard` | `sub_shift` | `cell_agree` |
|---|---|---|---|---|---|
| random `Tsub`, rest exact | 0.089–0.112 | 0.000 | 0.000 | ~0.25 | 0.72 (chance) |
| greedy (§5) | 0.327 | 0.025 | 0.000 | 0.261 | 0.721 |
| **annealing + polish, top 3 chains** | **0.435** | **0.072** | **0.000** | **0.267** | 0.737 |
| the truth | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

`cell_agree` 0.737 is exactly the chance value: the 607 frozen cells are all
correct (0.603 of 1,007) and the 400 searched ones are right on 0.30 of
themselves, which is the random rate for 200 ten-way plus 200 two-way cells
(0.119 of 1,007). **Seventy-seven
million evaluations on a 400-cell table, with the rest of the transducer
perfect, leave that table at chance.** The reheat round never beat round 0
(best 0.4320 → 0.4333 over a second 15,000 steps), so this is a saturated
result, not a truncated one.

**`sa_big` — the full 1,007-cell space, 2,560 chains, 3 × 20,000 proposals per
chain = 1.5 × 10⁸ evaluations.**

| | `digit` | `train_exact_hard` | `held_exact_hard` | `cell_agree` | `mul_lo` | `add_shift` | `sub_shift` |
|---|---|---|---|---|---|---|---|
| random init | 0.089–0.111 | 0.000 | 0.000 | 0.26 | 0.28 | 0.275 | 0.233 |
| greedy, 5 seeds (§4) | 0.352–0.424 | 0.028–0.048 | 0.000 | 0.249–0.289 | 0.24–0.28 | 0.25–0.31 | 0.25–0.28 |
| **annealing + polish, top 4 chains** | **0.476–0.479** | **0.060–0.064** | **0.000** | 0.260–0.264 | 0.28 | 0.27 | 0.25–0.27 |
| the truth | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The three reheat rounds returned best-of-population 0.4600 → 0.4613 → 0.4627:
**saturated after the first**. `cell_agree` 0.262 is the 0.258 chance value, and
all four structure scores are inside the random baseline. Annealing buys
`digit` 0.42 → 0.48 and `train_exact_hard` 0.048 → 0.064 over greedy, and buys
**nothing at all** on the question of whether the transducer was found.

Note what `digit` 0.48 means by the §2 table: the *construction* has `digit`
0.43 at 20 corrupted cells. So the annealed solution scores *better* than a
table that is 20 cells from the truth, while being 743 cells from it. The
degenerate branch of the level set does not merely exist; it dominates.

For scale: a tier-faithful Easy run affords 65–80 optimizer steps. Even a
surrogate that evaluated 10⁴ candidates inside every one of those forwards
would reach ~10⁶ evaluations — **two orders of magnitude below what is
demonstrated here to be insufficient**, on an objective whose signal at 20 %
corruption is one standard error. The search budget is not the binding
constraint, and neither is the step budget. The objective is.

---

## 7. Why: the information budget

`--exercise` runs the *constructed* model with the gathers instrumented and
counts how many times each table cell is read, then compares the description
length of the read cells against the information in the labels.

| modulus | S | cells read (of 1,007) | table bits (read) | supervision bits | **ratio** |
|---|---|---|---|---|---|
| 323 (e1) | 3 | 967 | 2,318 | 2,491 | **1.075** |
| 899 (e2) | 3 | 999 | 2,388 | 2,491 | **1.044** |
| 2021 | 4 | 943 | 2,266 | 3,322 | 1.466 |
| 10403 (m1) | 5 | 999 | 2,388 | 4,152 | 1.739 |

(`zero`, `carry0`, `borrow0` are read as constants outside the instrumented
gathers and are not counted; that is 3 cells of 1,007.)

**At the Easy moduli the labels carry 4–8 % more information than the table
needs.** The problem is a constraint-satisfaction problem with as many
constraints as unknowns and no slack. That is why:

* the solution is nevertheless *unique* — which is why `--construct` and the
  repair experiments work, and why held-out hits 1.000 once you are inside the
  basin;
* and why there is no graded approach to it — with no redundancy, a partially
  correct table satisfies essentially none of the constraints. Each of the 750
  output digits is a composition of ~39 sequential lookups, so changing one cell
  either misses the path entirely (no signal) or scrambles the result (no
  signal). That is the definition of a needle-in-a-haystack objective, and it is
  exactly what the basin curve in §2 measures.

The ratio improves with modulus (1.04 → 1.74) because the table is
modulus-independent by design while the supervision grows with the digit count —
so **the problem is better posed at Medium and Hard than at Easy**, which is an
interesting inversion but nowhere near enough: the basin at N = 10403 is if
anything *narrower* in cells (§2), because the chain is longer.

---

## 8. The one legal lever that reshapes the hypothesis class — tested

Section 7 says the problem is short of information. There is exactly one
*clearly legal* way to fix that without touching the data: **make the hypothesis
class smaller by weight tying**, which is an architecture choice and supplies no
values.

`--tie sym,inv`:

* `sym` — `Tmul` and `Tadd` are symmetric in their two digit indices. Enforced
  by construction (the table is read through a canonicalising index map), not by
  a penalty. Commutativity is a structural property of the *operation*, not a
  statement about base 10.
* `inv` — `Tsub` is **derived** from `Tadd`: `sub_d[add_d[u,v,c], v, c] = u` and
  `sub_b[add_d[u,v,c], v, c] = add_c[u,v,c]`. This is exact for the
  construction (proof: with `u+v+c = w+10c'`, `w−v−c = u−10c'`, which is `u`
  mod 10 and borrows iff `c' = 1`). It is weight tying — "the subtract table is
  the add table read backwards" — and it supplies no arithmetic.

Both keep the target in the class: `--construct` still reads
`digit = exact = held_exact = 1.000` under the ties. Free cells fall
**1,007 → 337** (verified by the tool: `cells=337 candidates=2442`) and the
description length of the whole table **2,402 → ~850 bits** (computed from the
free-cell alphabets; `--exercise` counts untied cells), taking the
identifiability ratio of §7 from 1.075 to **~2.9**.

**It measurably improves the landscape near the solution:**

| corrupted `k` (of 337) | reps reaching 1.000 | mean `train_exact_hard` | mean `held_exact_hard` |
|---|---|---|---|
| 3 | **5/5** | **1.000** | **1.000** |
| 10 | 1/5 | 0.780 | 0.768 |
| 20 | 0/5 | 0.486 | 0.484 |
| 50 | 0/5 | 0.015 | 0.000 |

Compare the untied ladder in §3: at `k` = 3, 3/5 → **5/5**; at `k` = 10, mean
0.727 → 0.780 on a *harder* corruption (each tied cell moves 2–4 table entries).
This is a real, legal improvement to the conditioning of the problem and I would
keep it in any future attempt at this family.

**And it does not create a path from random init.**

| seed | final `digit` | `train_exact_hard` | `held_exact_hard` | `cell_agree` | `sub_shift` |
|---|---|---|---|---|---|
| 0–4 | 0.203–0.333 | **0.012–0.024** | 0.000 (one 0.026) | 0.251–0.287 | 0.233–0.317 |

Mean `train_exact_hard` **0.015** — if anything *worse* than untied greedy's
0.037, because tied moves are more disruptive. The tied basin is narrower in
cells (chance by `k` ≈ 50 of 337 = 15 %, versus `k` ≈ 100 of 1,007 = 10 %), i.e.
about the same *fraction*. Cutting the description length by 2.7× does not cut
the plateau.

---

## 9. Stage 2: why no legal surrogate is worth building

My brief asked me to design a legal surrogate if Stage 1 succeeded, and to
reason about legality myself. Stage 1 did not succeed, and the reason
generalises to every surrogate that was proposed. I am recording the analysis
because it is the part that redirects the calendar.

| proposed surrogate | legal? | does it help? |
|---|---|---|
| **Population evaluation inside one forward** (thousands of candidate assignments on a batch dim, combined by a loss-weighted softmax) | **Yes** — one forward, one backward, one `optimizer.step()`; the evaluator never sees inside `forward`. `runner.py:316-338` is satisfied. | **No.** This is exactly what §4–§6 do, at a scale the evaluator cannot approach. I ran ~1.5 × 10⁸ objective evaluations; a tier-faithful Easy run affords ~10⁶ (65–80 steps × ~10⁴ candidates). The objective is flat over that whole region. |
| **Score-function / ES estimators** | **Constructible, with one caveat I want on the record.** Draw candidate tables from a learned categorical, evaluate them in the forward, and have `training_loss` return `Σ_i (L_i.detach() − b) · log p_θ(cand_i)` — one finite differentiable scalar, no Jacobian through the discrete states, all input-dependent computation inside the graph. The caveat: the gradient reaches θ through `log p_θ`, i.e. through the *sampling distribution*, not through the prediction path itself. That is the standard REINFORCE structure and I think it is defensible under rule 8, but a strict reading of "unbroken gradient path from the loss to the parameters responsible for the prediction" could reject it. **Flagged as compliance-uncertain.** | **No.** REINFORCE/ES estimate `E[L]` under a distribution over the *same* discrete objective. §2 says that objective has ~1 σ of signal at 20 % corruption. A zeroth-order estimator of a flat function is a flat estimator. |
| **A custom optimizer reading state the model wrote in `forward`** | **Compliance-uncertain, and I would not ship it.** `optimizer.step()` receives no loss, so the pattern needs a non-persistent buffer written in `forward` and read in `step()`. That is a side channel around the "unbroken gradient path from the loss to the parameters responsible for the prediction". It is arguably within the letter of rule 8 (the prediction path is still differentiable) and clearly against its spirit. | **Moot** — same objective again. |
| **Weight tying (§8)** | **Yes, cleanly** — architecture only, no values supplied, construction stays in the class. | Improves conditioning near the solution (5/5 repair at k=3, versus 3/5); does not move from random init. **Keep it, do not rely on it.** |
| **Algebraic constraints strong enough to pin the tables** (mul-from-repeated-add, additive identity on the learned `zero`, base-case anchoring) | **No, in my judgement.** A constraint set that determines base-10 arithmetic up to relabelling *without any data* is implementing the arithmetic — rule 2 in the loss instead of the forward pass. `alu-credit` already measured the weak, generic forms (`--sym`, `--inv`) as soft penalties and they were null. | Not tested, deliberately. I do not think a result obtained this way would be a legitimate submission, and reporting it as a "win" would mislead the team. |

**The general statement.** Every legal procedure sees the training labels and
nothing else. §7 says those labels carry 2,491 bits against a 2,318-bit table,
and §2 says the resulting objective is at the chance floor everywhere except a
1–2 % ball around the answer. Changing the *estimator* of a flat objective does
not make it less flat. This is why I am returning a closed family rather than a
surrogate.

---

## 10. What would reopen it

Stated as falsifiable conditions, in the order I would try them.

1. **A legal source of per-step signal.** This is the only real one. `alu-credit`
   §6.1 measured that per-step *inputs* take the tables to `sub_shift` 0.78 in 20
   optimizer steps; §7 now explains why (a different, much larger information
   budget). The legal analogue is target propagation with learned latents
   (`alu-credit` §6.5), which reached `sub_shift` 0.567 — the largest legal
   movement anyone has produced — but never trained the model and costs ~12×
   per step. **If anyone spends more GPU on this family, spend it there and
   nowhere else.** Concretely: run the discrete search with the latents included
   in the search space and a local-consistency term in the objective; if that
   widens the basin from `k` ≈ 20 to `k` ≈ 500, the family is alive.
2. **A graph whose end-of-chain objective is not a random CSP.** The obstruction
   is that each output digit is a composition of ~39 lookups, so a single wrong
   cell is either invisible or catastrophic. A graph in which each learned table
   sits O(1) ops from an *observable* quantity would have a graded objective by
   construction. I do not know how to build one for modular squaring without
   supplying the intermediate values, which is rule 2.
3. **More labels per parameter.** The ratio is 1.04–1.07 at Easy and 1.74 at
   Medium (§7). Nothing in the evaluator lets a submission manufacture more
   labelled operands without doing the arithmetic. A larger training set is not
   a lever a submission controls.

**What I would not spend another run on:** relaxation schedules (`alu-credit`
§9, 19 configurations), chain depth (`alu-depth` §3.2, 257 → 21 steps),
optimiser/init/curriculum (`alu-credit` §4, ~130 runs), discrete search of any
flavour (this branch), or any zeroth-order surrogate for it.

---

## 11. Reproduction

```bash
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python

# correctness gate: the integer sim IS the float model with hard states
$V lab/probe_search.py --verify --verify-seeds 3

# why nothing can find it: the objective's basin
$V lab/probe_search.py --basin --tag basin
$V lab/probe_search.py --modulus 10403 --slots 5 --basin --tag basin10403

# the information budget
for cfg in "323 3" "899 3" "2021 4" "10403 5"; do set -- $cfg
  $V lab/probe_search.py --modulus $1 --slots $2 --exercise --tag ex$1; done

# the search DOES work near the solution (repair ladder)
for k in 1 2 3 5 10 20 50 100 200 400; do
  $V lab/probe_search.py --repair $k --repair-reps 5 --sweeps 25 --tag rep$k \
     --jsonl lab/search_runs.jsonl; done

# THE DECISIVE RUN: from a random table assignment
$V lab/probe_search.py --restarts 5 --sweeps 60 --tag scratch \
   --jsonl lab/search_runs.jsonl

# and it is not about interaction between tables
for m in mul add sub mul,add add,sub mul,sub; do
  $V lab/probe_search.py --modules $m --restarts 3 --sweeps 60 --tag mod_$m \
     --jsonl lab/search_runs.jsonl; done

# the gradient-warm-start hybrid (best train_exact_hard in the family: 0.080)
$V lab/probe_search.py --float-steps 2000 --restarts 3 --sweeps 60 \
   --tag warm2000 --jsonl lab/search_runs.jsonl

# a strong global search: 2,560 independent Metropolis chains
$V lab/probe_search.py --anneal 20000 --anneal-rounds 3 --block 256 \
   --polish 4 --sweeps 60 --t0 5e-3 --t1 1e-5 --seed 7 --tag sa_big \
   --jsonl lab/search_runs.jsonl
$V lab/probe_search.py --modules sub --anneal 15000 --anneal-rounds 2 \
   --block 256 --polish 3 --sweeps 60 --t0 5e-3 --t1 1e-5 --seed 11 \
   --tag sa_sub --jsonl lab/search_runs.jsonl

# the legal hypothesis-class ties
$V lab/probe_search.py --tie sym,inv --basin --tag basin_tie
$V lab/probe_search.py --tie sym,inv --restarts 5 --sweeps 80 --tag tie_scratch \
   --jsonl lab/search_runs.jsonl
for k in 3 10 20 50; do
  $V lab/probe_search.py --tie sym,inv --repair $k --repair-reps 5 --sweeps 40 \
     --tag tierep$k --jsonl lab/search_runs.jsonl; done
```

One JSON line per search in `lab/search_runs.jsonl` (tag, full argv,
`train_digit`, `train_exact_hard`, `held_digit`, `held_exact_hard`, the
gauge-invariant `struct` scores, `cell_agree`). Per-run logs in `lab/logs/`.

---

## 12. Compliance

* **Nothing under `data/generated/` was read, printed, sampled or summarised.**
  Every operand in this report is generated by `probe_search.py` from
  `math.gcd` over `range(1, N)` with the modulus given on the command line.
* `--construct`, `--basin`, `--repair`, `--modules` and `--exercise` are **LAB
  DIAGNOSTICS**. They set tables to the truth or measure distance to it, and are
  never part of a submission (rule 7: no hard-coded algorithm in the forward
  pass). Every table above says which side of that line a row is on.
  `--float-steps` (§4.1) is not in that category — it trains the float model
  from random init with the standard AdamW cell and touches no oracle — but the
  *discrete polish* that follows it is a participant-controlled search and is
  therefore lab-only too.
* 110 searches are archived in `lab/search_runs.jsonl`, one JSON line each with
  the full argv. `lab/logs/` is gitignored by repo convention, so the raw stdout
  does not survive the branch; every number quoted above is either in the JSONL
  or reproducible from §11 in minutes.
* The `--tie` transformations (§8) are **not** in that category: they are weight
  sharing, they supply no value, and the construction is not used to derive
  them. They would be legal in a submission. I did not ship one because there is
  nothing to ship (§0).
* **No submission was written and no evaluator run was made**, so
  `lab/archive.jsonl` is untouched on this branch. Consistent with `alu-depth`
  §4 and `alu-credit` §7: while `train_exact_hard` is at the floor the evaluator
  cannot resolve anything, and a `MAX_T = 0` row would cost GPU and carry no
  information.
* Nothing was submitted to the hosted service; no `one-layer login` or `submit`
  was run; no network call was made.
* Negative results are all here, including the ones that contradict my own
  brief's framing (the brief expected discrete search to have "usable signal
  exactly where the gradient has none"; it has usable signal only within 1–2 %
  of the answer) and including the one result that flatters the branch (the
  repair ladder reaching `held_exact_hard` 1.000).
