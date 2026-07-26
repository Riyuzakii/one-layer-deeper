# alu-credit — making `DigitALU` trainable: what the optimiser can and cannot do

**Branch** `explore/alu-credit`, from `explore/digit-carry`.
**Mandate** hold the architecture fixed (`alu-depth` owns changing it), vary
everything about how it is optimised, get `train_exact` to 1.000.
**Tooling** `lab/probe_credit.py` (subclasses `DigitALU`; `--construct` still
reports `train_exact = held_exact = 1.000`, so the hypothesis class is
unchanged), `lab/credit_sweep.sh`, results in `lab/credit_runs.jsonl` and
`lab/logs/`.

---

## 0. Headline

**Best `train_exact` = 1.000, and it is a LAB DIAGNOSTIC, not a legal recipe.**
Teacher forcing on the true register trace reaches `train_exact` 1.000
free-running at eval; held-out lifts to **0.71–0.79** — the first time anything
in this repo has moved held-out exactness off the floor for an unseen operand.
Annealing teacher forcing to exactly zero keeps `train_exact` at 1.000, so the
model *is* self-consistent without the crutch. It needs the true intermediate
registers, which are computed arithmetic, so it cannot go in a submission.

**Best legal `train_exact` = 0.24**, from fine-grained deep supervision — and
that is also lab-only. Of the procedures that a submission could actually
express, **none beat the 0.10–0.21 seed band of the untouched baseline.** I ran
relaxation schedules, five initialisation families, straight-through on the
states and on the gate, three curricula, four regularisers, five optimiser
settings and a magnitude curriculum. All of it is inside the noise.

**Three things were falsified, one of them a premise of my own brief:**

1. **"For `DigitALU`, `train_exact` → 1.000 implies `held_exact` → 1.000."**
   False. The teacher-forced model is at `train_exact` 1.000 with held-out
   0.71–0.79, and the S=2 short-chain model reaches `train_exact` 0.75 with
   `held_exact` 0.000 and held-out CE climbing monotonically from 2.3 to 9.1.
   `DigitALU` overfits. `train_exact` is still the right *screen*, but it is a
   necessary condition, not a sufficient one, and the eventual candidate has to
   be checked on held-out anyway.
2. **The plateau is not a partially-learned table.** Every configuration that
   plateaus around 0.2 has tables at *chance* on a gauge-invariant structure
   score (0.24–0.30 against a random-init baseline of 0.24–0.28). The plateau
   is a degenerate non-arithmetic solution, not 20% of the way to the answer.
3. **The forward pass being information-destroying is not the binding
   constraint.** I found and fixed it — the register's across-batch spread
   collapses from 8e-4 to 0.0000 within four `cond_sub`s, and straight-through
   everywhere holds it at 0.25 for all 280 steps, a 300× improvement. Training
   with that fix is *worse* (0.004–0.020). Information flow is necessary and
   nowhere near sufficient.

**The single highest-value recommendation is in §7:** the obstruction is not
the optimiser and it is not the relaxation. Stop looking for a training trick.

---

## 1. What I re-verified before starting

`DigitALU`: 6,817 parameters, Horner over `K = 2S−1` places, `R = 11` tied
conditional subtractions per place, ~280 sequential soft table lookups at
N=323/S=3.

| check | result |
|---|---|
| constructed ceiling, N=323 S=3 R=11 | `train_exact = held_exact = 1.000` |
| same after refactoring `forward` into an explicit op list | 1.000 (regression test) |
| baseline from random init, 1000 steps, seed 0 / seed 1 | 0.100 / 0.184 |

The seed spread (0.100 vs 0.184 on the same recipe) is the noise floor for
every number below. Nothing under ~0.25 is distinguishable from baseline.

Constructed ceiling as a function of `R`, which matters because `R` is the
obvious chain-length knob:

| R | 1 | 2 | 3 | 4 | 5 | 7 | 9 | 11 |
|---|---|---|---|---|---|---|---|---|
| constructed `train_exact` | 0.100 | 0.180 | 0.280 | 0.372 | 0.468 | 0.704 | **1.000** | **1.000** |

`R ≥ 9` is required for the class to contain the solution. Anything trained at
`R < 9` is being trained against an unreachable target, which is why I only used
`R` as a *warm-up* knob (always ending at `R = 11`).

## 2. Two diagnostics that located the obstruction

### 2.1 The gradient never reaches the front of the chain

`probe_credit.py --diag` prints per-module gradient norms. At step 1, N=323:

```
Tmul=3.27e-07  Tadd=4.57e-06  Tsub=3.41e-02  gate.weight=5.69e-05
```

`Tsub` — used by the *last* op in the chain — gets **10⁵ times** the gradient of
`Tmul`, which is used by the first. By step 250 the ratio has closed to ~6×, but
by then the model is already in the degenerate basin.

### 2.2 The register stops depending on the input after four subtractions

`--flow` prints the across-batch standard deviation of the register after every
op — how much of `x` still reaches the state. At the default init:

```
op:      1      2      3      4      5   ...  12  | 13     14  ...
base   0.0008 0.0004 0.0002 0.0001 0.0000 ... 0.0000 | 0.0009 0.0009 ...
```

It halves at every `cond_sub` and is numerically zero by the fifth, then resets
to 8e-4 at the start of the next Horner place (a fresh `Tmul` product is added)
and dies again. **`cond_sub` is a contraction**: the gate is `g·t + (1−g)·r`
with `g ≈ 0.5` at init and `t` — the scan of a near-uniform `Tsub` — nearly
constant across the batch, so each subtraction halves the deviation. Eleven per
place gives 2⁻¹¹.

This is a mechanism, and it is fixable. Holding the gate closed
(`--gate-bias −6`) makes the profile perfectly flat. Straight-through on the
softmaxes *and* on the gate holds it at **0.25 for all 280 ops**:

| init | spread at op 1 | spread at op 280 |
|---|---|---|
| default | 8e-4 | **0.0000** |
| `--gate-bias −6` (gate closed) | 8e-4 | 9e-4 |
| `--perm-init 8` (tables ≈ random permutations) | 0.0075 | 0.0082 |
| `--perm-init 8 --gate-off −6` | 0.0075 | 0.0082 |
| **`--hard --hard-gate` (straight-through everywhere)** | **0.125** | **0.26** |

Fixing it does not help (§4). That is the most useful negative in this report:
the failure is not that the signal is absent, it is that the signal is
uninformative — the chain is a near-permutation dynamical system and the
gradient direction through 280 of its steps is chaotic, not small.

## 3. The measurement that made the rest interpretable

Raw "does the learned table match the truth" accuracy is **meaningless** for
this architecture, and reading it naively would have made me report the exact
opposite of the truth. There is a gauge freedom:

* `Tmul`'s output alphabet is consumed *only* by `Tadd`'s addend index. Any
  permutation `π` of the ten digit symbols with `π(zero) = zero` can be applied
  to `Tmul`'s outputs and `Tadd`'s `v` index together with no change to the
  function. That is a 9!-element gauge group.
* The two borrow states can be swapped, and the two carry states with them.

`Tsub` is the one table with **no** gauge freedom — its register index and its
output are the register, and its second index is a digit of `N` supplied as a
fixed one-hot — but only the columns `N`'s digits actually reach are ever
exercised (three of ten at N=323).

`structure_scores()` is therefore gauge-invariant by construction: is `Tmul`'s
argmax a well-defined *function* of `(a·b) mod 10`, and is each column of
`Tadd`/`Tsub` a *cyclic shift* of the identity (which "add/subtract a constant"
is, whatever the labelling)? Validated at both ends:

```
constructed : mul_lo=1.000 mul_hi=1.000 add_shift=1.000 sub_shift=1.000
random init : mul_lo=0.280 mul_hi=0.240 add_shift=0.275 sub_shift=0.233
```

## 4. Everything I ran that a submission could legally express — all negative

All at N=323, S=3, R=11, AdamW, full batch of 250, unless stated. Baseline band
is **0.10–0.18** across two seeds.

### 4.1 Relaxation and initialisation (1000 steps)

| config | `train_exact` |
|---|---|
| baseline (`tau 1.0`, `randn*0.5`) seed 0 / seed 1 | 0.100 / 0.184 |
| `--tau 0.3 --init-scale 0.3` seed 0 / seed 1 | 0.112 / 0.212 |
| `--tau 0.1 --init-scale 0.1` seed 0 / seed 1 | 0.052 / 0.076 |
| `--tau 0.03 --init-scale 0.03` | 0.008 |
| `--tau 0.1 --init-scale 1.0` (sharp) | 0.028 |
| `--tau 1.0 --init-scale 0.1` (flat) | 0.096 |
| `--init-scale 4` (low-entropy random) | 0.180 |
| `--onehot-init 4` (random one-hot rows) | 0.204 |

`tau` and the init scale were co-varied deliberately: the per-step softmax
Jacobian is `(1/tau)(diag p − p pᵀ)`, so holding `logits/tau` fixed while
lowering `tau` multiplies the per-step gain by `1/tau` without changing how
sharp the distributions are. It buys nothing. Note also that the report's
`--identity-init` confounded two knobs (copy-through scale *and* a closed gate);
`probe_credit` separates them, and neither helps on its own.

### 4.2 Straight-through, including on the gate (1000 steps)

| config | `train_exact` | `add_shift` | `sub_shift` |
|---|---|---|---|
| `--hard` (report's config) | 0.008 | 0.29 | 0.26 |
| `--hard --hard-gate` | 0.020 | 0.28 | 0.25 |
| `--hard --hard-gate --tau 0.1` | 0.008 | — | — |
| `--hard --hard-gate --tau 0.3` | 0.008 | — | — |
| `--hard --hard-gate --tau 3.0` | 0.012 | — | — |
| `--hard --hard-gate --perm-init 4` | 0.004 | — | — |
| `--perm-init 8` (sharp, soft forward) | 0.020 | — | — |

Straight-through on the gate is the piece the report never tried, and it is the
piece that makes the forward pass genuinely information-preserving (§2.2). It
makes training *worse*. The bias of the straight-through estimator compounds
over 280 steps faster than the extra information helps.

### 4.3 Curricula

| config | `train_exact` |
|---|---|
| `--r-start 3 --r-warm 0.5` (chain grows 3→11 cond_subs), 1500 steps | 0.164 |
| `--curr 91:2:500` then N=323/S=3 (LAB ONLY, cross-modulus), 1500 steps | 0.116 |
| `--trunc 1` (detach the register at each Horner place) + deep sup | 0.072 |
| `--xcurr 0.5` (magnitude curriculum, LEGAL), 3000 steps | *(§4.6)* |

The `R` curriculum is worth singling out because it is the one chain-length
curriculum that **is** legal — `R_eff` is set from a step counter inside
`forward`, exactly like a temperature schedule, and evaluation always runs at
the full `R = 11`, so the constructed ceiling of the reported configuration is
1.000. It does nothing.

The cross-modulus curriculum (train the shared tables at N=91/S=2, ~117 ops,
then move to N=323/S=3, ~280 ops) is the coordinator's highest-prior idea. It
is **lab-only** — a submission never sees a second modulus, and it cannot
manufacture one without doing arithmetic. It also does not work, and §5 explains
why: the short-chain stage does not learn the tables either, it overfits.

### 4.4 Label-free regularisers

| config | `train_exact` |
|---|---|
| `--inv 1.0` (`Tsub` must invert `Tadd`; label-free, value-free) | 0.096 |
| `--ent 0.05` (entropy pressure toward one-hot tables) | 0.096 |
| `--sym` (commutativity of `Tmul`/`Tadd`) | *(§4.6)* |

`--inv` deserves a note because it is the most interesting of the legal
regularisers: if `add(u,v,c) → (w,c′)` then `sub(w,v,c) → (u,c′)` is an exact
identity that constrains the *relation* between two learned tables while saying
nothing about what either computes, and it lives in the loss, not the forward
pass. It transfers whatever the data teaches `Tadd` straight into `Tsub`, which
is precisely the "modules are only useful jointly" problem. It has no effect,
because `Tadd` never learns anything to transfer.

### 4.5 Optimiser and loss

*(§4.6 table)*

### 4.6 Results table for the remaining legal sweeps

*(filled in from `lab/credit_runs.jsonl`)*

## 5. The chain-length ladder, and why the short-chain result does not mean what it looked like

`digit-carry` §2.4's strongest measured fact was that shortening the chain from
~280 to ~117 ops took `train_exact` from 0.20 to 0.72–0.78. I reproduced it
(0.683 at 1000 steps, seed 0) and then ran it to 5000 steps with the
gauge-invariant structure scores and a held-out column:

| N=91, S=2, ~117 ops | step 1000 | 1500 | 2000 | 2500 | 3000 | 3500 |
|---|---|---|---|---|---|---|
| `train_exact` | 0.683 | 0.717 | 0.750 | 0.750 | 0.750 | 0.633 |
| `held_exact` | 0.083 | 0.000 | 0.000 | 0.000 | 0.000 | 0.083 |
| `held_ce` | 7.27 | 8.06 | 8.58 | 8.88 | **9.06** | 8.37 |

**Held-out CE rises monotonically from 2.31 to 9.06 while train CE falls to
0.27.** That is textbook overfitting, in the architecture the previous branch
described as unable to memorise. The short chain does not make the tables
learnable; it makes the *degenerate* solution easier to fit. This is the
result that kills the curriculum plan, and it is why the cross-modulus
curriculum in §4.3 transfers nothing: there is nothing correct at the source to
transfer.

## 6. What actually worked, and exactly why it is not a submission

### 6.1 Teacher forcing on the true register trace — LAB DIAGNOSTIC ONLY

`--teacher-force p` overwrites the register with the true value after every one
of the ~70 ops during training only (`model.eval()` disables it, so every number
below is measured **free-running**). `--deep-sup` puts a cross-entropy on each
op's output against the true next register. Both need the true trace, which is
`(10·r + Σ dᵢdⱼ) mod N` computed in Python — a hard-coded algorithm. **Rule 2.
Not legal. Diagnostic only.**

| config | `train_exact` | `held_exact` |
|---|---|---|
| deep supervision only, per Horner place (5 targets) | 0.172 | 0.000 |
| deep supervision only, per op (~70 targets) | 0.240 | 0.026 |
| **teacher forcing `p=1.0` + deep sup** | **1.000** | **0.789** |
| teacher forcing `p=0.5` + deep sup | 1.000 | 0.737 |
| teacher forcing annealed 1.0 → 0.0 over the first half | 1.000 | 0.579–0.605 |
| teacher forcing + straight-through | 0.008 | 0.000 |

Two things to read here.

**Deep supervision alone is not enough — teacher forcing is.** Giving the true
target at every op but letting the register free-run (deep sup) gets 0.24.
Giving the true *input* at every op gets 1.000. The difference is exactly credit
assignment: once the register has drifted, a correct target for it is useless.

**And it holds after the crutch is removed.** With `--tf-decay 0.5`, teacher
forcing is exactly zero from step 1000 of 2000 onward and `train_exact` stays at
1.000. So the learned automaton is genuinely self-consistent, not propped up.

### 6.2 What the teacher-forced model learned

The learned tables, read through the gauge of §3 (`--dump-tables`):

* `Tadd` at carry-index 0, column `v=3` is the identity `[0..9]`; column `v=0`
  is `[1,2,3,4,5,6,7,8,9,0] = (u+1) mod 10`; column `v=4` is `(u+2) mod 10`
  except at one cell. So `Tadd` **is** an addition table, with the addend
  alphabet relabelled by a permutation `π` — `π(3)=0`, `π(0)=1`, `π(4)=2` — as
  §3 predicts.
* `Tsub` at borrow-index 0, at the two columns `N=323` actually reaches, is
  `[7,8,9,0,1,2,3,4,5,6] = (u−3) mod 10` for `v=2` and
  `[6,7,8,9,0,1,2,3,4,5] = (u−4) mod 10` for `v=3` — both exactly
  `(u − v − 1) mod 10`. The model's `borrow0` argmax is **1**, not 0: it swapped
  the two borrow states, so borrow-index 0 *is* "borrow set" and `u−v−1` is
  correct.
* `gate.weight = [17.6, −18.3]`, the same sign pattern the construction uses.

**The optimiser, given per-step supervision, recovers the constructed solution
up to the gauge.** That is worth stating plainly, because it means the target is
reachable by gradient descent in this parameterisation and the entire problem is
the rollout.

Held-out stops at ~0.79 rather than 1.000 because of table *coverage*, not
optimisation: `Tsub`'s column `v=0` (the leading zero digit of `N`) is exercised
only in the top register slot, where the register is almost always zero, so most
of that column never receives a gradient. That is a data-coverage problem with a
known closed form (`digit-carry` §2.3 measured it at 0.947–0.99), and it is a
different problem from the one I own.

### 6.3 Target propagation — the legal analogue, and it does not substitute

Teacher forcing works because it supplies per-step targets. The legal way to get
per-step targets without computing them is to make them *learned latents*:
`LatentTrace` predicts the whole ~70-step register trace from the same inputs the
model already sees; the tables and the latents are trained jointly under (a) a
local consistency loss — one op applied to latent `t` must reproduce latent
`t+1` — and (b) two boundary conditions that use only given quantities: the last
latent is the label, the first input is the model's own learned `zero`. This is
method-of-auxiliary-coordinates / target propagation. It supplies no arithmetic,
it is discarded at eval, and it is expressible under the evaluator's fixed loop
(`forward` returns the latents in `aux`, `training_loss` combines the terms, one
`optimizer.step()` per batch).

*(result in §4.6)*

## 7. Recommendation

*(to be written)*

## 8. Compliance

* Nothing under `data/generated/` was read, printed, sampled or summarised. All
  probes generate their own operands from `math.gcd` over `range(1, N)`.
* `--construct`, `--deep-sup` and `--teacher-force` are **lab diagnostics**.
  They set or use the true tables / the true register trace and are never part
  of a submission (rule 7 / rule 2). Every table in this report says which side
  of that line a row is on.
* No hosted submission, no network call, no `one-layer` CLI invocation.
* Negative results are all here, including the ones that contradict my brief's
  premises and `digit-carry`'s screening rule.
