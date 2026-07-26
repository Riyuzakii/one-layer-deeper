# alu-compose — given a perfect single squaring step, what MAX_T does the pipeline certify?

**Branch:** `explore/alu-compose` · **No model was trained.** Every number below
is measured with the one unsolved component — the squaring transducer — replaced
by its constructed ceiling (`probe_alu.py --construct`, a LAB DIAGNOSTIC, never
in a submission). The question is what the *rest* of the system does when the
step is perfect.

---

## 0. The answer, in one table

| what is constructed | Easy (train T = 1,2,3) | Medium/Hard (train T = 4,8,16) |
|---|---|---|
| step + parser + **depth controller all constructed** | **MAX_T = 64** | **MAX_T = 64** |
| step + parser constructed, **depth controller trained on the tier's T values** | **MAX_T = 2** | **MAX_T = 0** |

**A perfect squaring step buys MAX_T = 2 on Easy and MAX_T = 0 on Medium.** The
arithmetic is not what stops the ladder. Two things do, and neither was on the
team's list:

* **(P1) The depth controller cannot extrapolate in T.** Its parameters are
  indexed by `(place of T, digit)`. Training presents `T ∈ {1,2,3}` or
  `{4,8,16}`; the ladder needs `T ∈ {1,2,4,8,16,32,64}`. The `(place, digit)`
  pairs the ladder needs but training never shows are unconstrained, so the
  controller is arbitrary there. Five controller parameterisations, two seeds
  each, all land on MAX_T ≤ 2 (§4).
* **(P2) The eval budget binds, and it binds as a hard run failure.** A
  fixed-grid 64-iteration readout on Easy raises `TimeoutError: evaluation
  exhausted its 30.0s time budget` inside the `test` split — before the depth
  ladder is reached — so the run's status is *failed*, not MAX_T = 0 (§5).
  ACT/PonderNet early exit removes this, and is the same design change that
  attacks P1.

The good news is equally definite: **exactness composes, perfectly, everywhere
tested** — 1.000 at every rung to T = 64, at five modulus regimes from 9 to 32
bits, in fp32 and under the manifests' bf16 + amp, with soft or discrete states
(§3). Downward generalisation to T = 1, 2 from a model trained at T ∈ {4,8,16}
is exact **by construction** in the weight-tied step, and fails **only** in the
depth controller (§7).

---

## 1. What was composed

`lab/probe_compose.py` builds one forward pass out of the three components that
had each been verified in isolation on other branches, and runs it on prompts
synthesised from the public generator spec:

```
[N] d(N) [X] d(x) [T] d(T)
        |
        |  MarkerPointer            (digit-carry §3, verified 1.000 at T=16/32/64)
        v
  x-slots (S), N-slots (S+1), T-slots (2)      each a soft 10-way digit
        |
        |  DigitALU applied `loops` times, weight tied    (digit-carry §2.2)
        v
  a stack of `loops` candidate answers
        |
        |  depth controller: which iteration is the answer
        v
  tail-aligned digit logits
```

`--construct` sets the pointer probes, the offset/opening tables `R`/`G`, the
slot→digit projection, the digit tables `Tmul`/`Tadd`/`Tsub`, the gate and the
depth controller to their exact values. `--mode compose` forces `loops = T` and
measures exactness; `--mode select` runs `loops = 64` and lets the controller
choose; `--mode time` measures throughput.

Two small fixes were needed to make the components compose at all, both worth
recording:

1. `lab/probe_parse.py` used `torch.finfo(x.dtype).min` as its attention-mask
   value, which raises `RuntimeError: value cannot be converted to type
   c10::BFloat16 without overflow` under the manifests' bf16 + amp. This is the
   bug digit-carry §4 recorded for the *submission*; the *probe* still had it,
   so the parser could not be measured in the evaluator's dtype. Patched to a
   finite `NEG = -1e4` (mathematically identical after softmax).
2. `MarkerPointer`'s offset range `o_hi = 9` is too small once `N` has ten
   digits (the 30/32-bit Hard regime needs offsets up to `S+1 = 11`). The probe
   and the submission now size it from `S`.

---

## 2. Method and compliance boundary

* Everything is synthesised: moduli are passed on the command line, `x` values
  are enumerated from the unit group (or sampled above 24 bits, where
  enumeration is intractable), and targets are computed as `pow(x, 1<<T, N)`.
  **Nothing under `data/generated/` was read, printed or summarised.**
* `--construct` is a diagnostic oracle, the direct analogue of `probe_alu.py
  --construct` and `probe_step.py --oracle`. It lives in `lab/`, is never
  imported by a submission, and no constructed tensor appears in
  `submissions/alu-compose/submission.py`.
* Screening used probes; the evaluator was used only for the eval-budget
  question, where it is the ground truth and nothing else is.
* Four to thirteen sibling jobs shared the GPU for the whole session. **Every
  accuracy number here is contention-immune** (fixed step counts, deterministic
  forward passes). **Every wall-clock number is contention-pessimistic**, and is
  reported both as an absolute and as a ratio; §5 says which is which.

---

## 3. Does exactness compose? Yes — 1.000 at every rung, everywhere tested

`--mode compose`, `loops` forced to `T`, held-out `x`, exact-example accuracy:

| regime | modulus | S | held-out n | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | T=64 |
|---|---|---|---|---|---|---|---|---|---|---|
| e1 | 323 (9 bit, fixed) | 3 | 38 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| e5 | 12 sampled 10/11-bit | 4 | 512 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| m1 | 10403 (14 bit, fixed) | 5 | 512 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| hp1 | 4028033 (22 bit, fixed) | 7 | 256 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| hp2 | 4 sampled 30/32-bit | 10 | 64 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |

and the numerical questions that a 64-fold composition raises:

| variant (e1) | T=1 … T=64 | min over examples of `max_d p(d)` at T=64 |
|---|---|---|
| fp32, soft states | all 1.000 | 1.00000 |
| **bf16 + amp** (the manifests' dtype), soft states | all 1.000 | 1.00000 |
| bf16 + amp, discrete (straight-through) states | all 1.000 | 1.00000 |

**There is no drift.** The soft state after 64 compositions is one-hot to the
printed precision — the ±30-logit tables keep every softmax saturated, so the
round trip `softmax(log p)` is a fixed point rather than a contraction toward
uniform. Discrete states are therefore *not needed* for exactness, which matters
because they cost ~40 % more wall clock (§5).

This is a real forward pass with the real parser, at every rung, including the
T = 16/32/64 rungs where the T field gains a digit and every distance-from-the-end
scheme breaks. **The three components compose without an interface mismatch.**

### 3.1 A number-theoretic fact the ladder hides

`x^(2^T) mod N` depends on `T` only through `2^T mod λ(N)`, and `2^T mod λ(N)`
is eventually periodic. Computed from the public moduli (pure number theory, no
dataset access):

| dataset | modulus | λ(N) | distinct exponent classes on the ladder |
|---|---|---|---|
| e1 | 323 = 17·19 | 144 | {1} {2} **{4,16,64}** **{8,32}** — four maps, not seven |
| e2 | 899 = 29·31 | 420 | {1} {2} **{4,16,64}** **{8,32}** |
| m1 | 10403 = 101·103 | 5100 | all seven distinct |
| m2 | 38021 = 191·199 | 18810 | all seven distinct |
| hp1 | 4028033 = 2003·2011 | 2012010 | all seven distinct |

Two consequences, both load bearing.

* **On e1/e2 the depth loss cannot identify the iteration count.** Applying the
  step 4, 16 or 64 times gives *the same function*, so a controller that routes
  T = 4 to 16 iterations has zero loss. This is a second gauge freedom on top of
  the one `tied-recurrence` §6 found, and it is why a controller trained on e1
  can report a perfectly fitted training loss while its `loc` map is wrong
  (§4.2). Any depth experiment run on e1 or e2 alone is measuring a degenerate
  objective; **run depth experiments at m1's modulus**.
* On Easy the ladder is genuinely easier than it looks: certifying T = 64 there
  requires only four distinct maps, not seven.

---

## 4. Does the depth controller route T? Constructed yes, trained no — this is P1

### 4.1 Constructed controllers reach the top of the ladder

`--mode select --eval-loops 64`, e1, held-out `x`, `route` = fraction of
examples whose selected iteration is exactly `T`:

| controller | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | T=64 | MAX_T |
|---|---|---|---|---|---|---|---|---|
| ordered pointer, constructed (`loc = Σ_p 10^p d_p − 1`) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | **64** |
| counted halting, constructed | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | **64** |

Routing and exactness are both 1.000 at T = 64. **The selector mechanism is not
the problem; obtaining it from the tier's training data is.**

### 4.2 Trained controllers do not extrapolate — five parameterisations, two seeds

The parser and the ALU are held at their construction and **only the depth
controller is trained**, on the tier's own T values, with a cross-entropy loss on
the answer digits and the window annealed wide→sharp. This is the most
favourable possible setting: a perfect step, a perfect parse, and a loss that is
exactly the competition's.

**Easy (`train T = {1,2,3}`, e1):**

| controller (params) | fitted `loc` at T=1/2/3 | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | T=64 | MAX_T |
|---|---|---|---|---|---|---|---|---|---|
| MLP `Lin(20,64)-GELU-Lin(64,1)` (1409) | −1.43 / 0.56 / 2.60 | 1.000 | 1.000 | 0.026 | 0.026 | 0.026 | 0.026 | 0.026 | **2** |
| linear `Lin(20,1)` (21) | −0.38 / 0.74 / 1.76 | 1.000 | 1.000 | 0.105 | 0.237 | 0.026 | 0.026 | 0.026 | **2** |
| factored `Σ_p c_p Σ_d v_d st[p,d] + b`, `v` learned (13) | −0.43 / 0.66 / 1.58 | 1.000 | 1.000 | 0.105 | 0.237 | 0.105 | 0.237 | 0.105 | **2** |
| factored, `v` fixed to the ordinal ramp (3) | −0.10 / 0.73 / 1.56 | 1.000 | 1.000 | 0.053 | 0.184 | 0.184 | 0.237 | 0.053 | **2** |
| counted halting, per-digit zero detector (22) | 0 / 1 / 2 (**exact**) | 1.000 | 1.000 | 0.026 | 0.026 | 0.026 | 0.237 | 0.026 | **2** |
| counted halting, zero detector shared with the ALU (12) | 0 / 1 / 2 (**exact**) | 1.000 | 1.000 | 0.026 | 0.026 | 0.974 | 0.237 | 0.921 | **2** |

Every cell reaches training loss ≈ 0 and every cell stops at T = 2. Seed 1
reproduces every row within one example.

The last row is the informative one. Counted halting is the only
parameterisation whose fitted `loc` is *exactly* `T−1` on the training values
rather than merely in the right bin — it has no continuous affine map to be
underdetermined. It is also the only one that gets T = 16 and T = 64 nearly
right (0.974, 0.921), and that is not luck: on e1 those rungs are the same
function as T = 4 (§3.1) and the controller halts at 4 iterations. It still
fails T = 4 itself, for the reason in §4.3(a).

**Medium (`train T = {4,8,16}`, e1 modulus):**

| controller | fitted `loc` at T=4/8/16 | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | T=64 | MAX_T |
|---|---|---|---|---|---|---|---|---|---|
| MLP | 30.1 / 12.5 / 34.8 | 0.026 | 0.079 | 0.184 | 0.395 | 0.447 | 0.447 | 0.447 | **0** |
| linear | 3.5 / 12.5 / 2.9 | 0.026 | 0.237 | 1.000 | 0.395 | 1.000 | 0.447 | 0.026 | **0** |
| factored, `v` learned | 25.6 / 12.5 / 27.1 | 0.026 | 0.105 | 0.184 | 0.184 | 1.000 | 0.026 | 0.447 | **0** |
| factored, `v` ordinal | 14.5 / 13.1 / 23.1 | 0.053 | 0.105 | 1.000 | 1.000 | 0.447 | 1.000 | 1.000 | **0** |
| counted halting, per-digit detector | 6 / 7 / 9 | 0.026 | 0.026 | 0.184 | 0.711 | 0.632 | 0.447 | 0.184 | **0** |
| counted halting, shared detector | 15 / 15 / 15 | 0.053 | 0.079 | 0.342 | 0.342 | 0.342 | 0.342 | 0.342 | **0** |

**Every Medium cell fails at T = 1 and T = 2.** That is the downward-
generalisation failure, and §7 shows it is entirely the controller's. Note the
`factored, v ordinal` row scoring 1.000 at four rungs while `loc(4) = 14.5`: it
is applying the step fifteen times when asked for four and still getting the
right answer, because on e1 that is the same function (§3.1). Read it as
evidence that **e1 cannot be used to evaluate a depth controller**, not as
evidence that the controller works.

**At a modulus whose ladder does not collapse (m1, N = 10403, `train T =
{4,8,16}`)** the same heads do not even fit the training values:

| controller | fitted `loc` at T=4/8/16 | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | T=64 | MAX_T |
|---|---|---|---|---|---|---|---|---|---|
| MLP | −236 / −231 / −103 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **1** |
| factored, `v` ordinal | 8.9 / 17.5 / 22.5 | 0.000 | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | **0** |
| counted halting, shared detector | 15 / 15 / 15 | 0.711 | 0.305 | 0.031 | 0.000 | 0.000 | 0.000 | 0.000 | **0** |

The MLP diverges to `loc ≈ −200`, so all mass sits on iteration 1 and it
"certifies" T = 1 by accident while scoring 0.000 at every other rung — a
reminder that a certified prefix of length 1 can be an artifact. The ordinal
head and the counter both fit `loc` values that are simply wrong on their own
training data. **The near-misses on e1 were the exponent degeneracy doing the
work, not the controller.**

### 4.3 Why — two independent mechanisms, both diagnosed

**(a) The training T values do not exercise the `(place, digit)` pairs the
ladder needs.** The ladder's T field contains the digits
`1, 2, 4, 8, (1,6), (3,2), (6,4)`. Easy training shows only `1, 2, 3` at place
0 and nothing at place 1 — so the tens place of `T` is *completely
unconstrained*, and T = 16, 32, 64 can never be routed. Medium training shows
`4, 8, (1,6)` — the digits `2` and `3` at place 0 never appear, so T = 2 and
T = 32 are unconstrained, and place 0 sees only three of ten digits. This is a
coverage argument of exactly the same shape as `group-rotation`'s residue
coverage bound, applied to the T field instead of `Z_N`, and it is why *more
structure in the head does not help*: the 3-parameter ordinal head fails for the
same reason as the 1409-parameter MLP.

The counted-halting controller was built to escape it — it has no parameter
indexed by a place of T; it subtracts a learned unit digit from T's digit
register using the `Tsub` table and borrow state **shared with the modular
reduction**, and halts when the register matches the ALU's own learned zero
digit (12 learned scalars in the sharing variant). It escapes the *place* half
of the problem — the constructed version routes T = 64 with a controller that
has never seen a two-digit T — but not the *digit* half: trained on
`T ∈ {1,2,3}`, the zero detector only ever sees register values `0,1,2`, so it
is free to fire on `3`, and it does: `loc(4) = 0`.

**(b) The loss constrains `loc` to the right *bin*, not to the right *value*.**
With the window annealed to σ = 0.1 the loss is flat inside `|loc − (T−1)| <
0.5`, so a whole interval of affine maps is optimal on the training T values and
only the exact corner extrapolates. Read the Easy ordinal row: the fit is
`loc ≈ 0.83 T − 0.92`, which lands in the right bin at T = 1,2,3 and is wrong by
1.4 iterations by T = 8. **Depth supervision through a soft window is
underdetermined by construction.**

And on e1/e2 there is a third, dataset-specific degeneracy on top: §3.1's
exponent collapse means 4, 16 and 64 iterations are literally the same function,
so even a perfectly-fitted training loss is compatible with a wrong iteration
count. The Medium `place` row shows it directly — `loc(4) = 14.5`, i.e. the
model applies the step 15 times when asked for 4, and scores 1.000.

---

## 5. Eval budget and throughput — this is P2, and it is a hard failure

### 5.1 The model is kernel-launch bound, so batch size is nearly free

One application of `DigitALU` at slot count `S` is
`S²·(S+1) + (2S−1)·R·(S+1)` sequential slot updates (S=3, R=11: 256; S=10: 3399),
each 3–4 kernel launches. At T = 64 that is ~57,000 launches per forward at
S = 3. FLOPs are negligible — so wall clock is set by the *number of batches*,
not by their size. Measured (`--mode time`, e1, bf16, 64 loops, heavy
contention, so read the ratios):

| eval batch | ms per forward | relative |
|---|---|---|
| 38 | 6332 | 1.00 |
| 512 (13.5× the examples) | 10426 | 1.65 |
| 4096 (108× the examples) | 10920 | **1.72** |

**8× the eval batch costs 5 % more wall clock.** Setting a large
`SUBMISSION.eval_batch_size` makes every scoring split a single batch and is the
single cheapest eval-budget lever available. `submissions/alu-compose` sets
`eval_batch_size = 4096`.

Discrete eval states cost real time and buy nothing (§3): `eval-hardsel`
(discrete states + a gather) is 140.9 ms/loop against 98.9 ms/loop for soft
states at batch 38 — **+42 %**. `HARD_EVAL_STATES = False` in the deliverable.

### 5.2 A fixed 64-iteration readout FAILS the Easy eval budget

Real evaluator, `lab/manifests/alu_e1_evalonly.json` (Easy 60 s wall clock,
`max_steps = 5` so the measurement is of eval alone), `eval_batch_size = 4096`:

| submission | eval budget | outcome |
|---|---|---|
| `alu-compose/mix64` — fixed 64-iteration mixture, no early exit | 30.0 s | **`TimeoutError: evaluation exhausted its 30.0s time budget`, run status *failed*** |
| `alu-compose` — ACT early exit | 30.0 s | completed all 16 splits, `evaluation_seconds = 24.8` |

Two things about the failure mode matter more than the number.

* It is raised at `runner.py:589`, in the loop over the **scoring** splits
  (`test`, `ood`) — *before* `_evaluate_depth_profile`. A rung that times out is
  gracefully recorded as `not_completed`; the `test` split timing out is an
  uncaught `TimeoutError` that propagates out of `run_submission_file`. The run
  is **failed**, not MAX_T = 0. `service/db.py` ranks over `status='succeeded'`
  runs only, so this is strictly worse than submitting nothing at all — and it
  costs one of the user's 1/day Hard attempts.
* The 24.8 s that *did* fit was measured with 13 sibling jobs on the GPU, so it
  is pessimistic — but the margin is 1.21×, not the ~5× that `tied-recurrence`
  §7 measured for a 0.4 M-parameter dense block. **That measurement does not
  transfer to this architecture and should not be relied on.**

### 5.3 What ACT early exit buys, and the per-tier arithmetic

At evaluation there is no backward pass, so the halting distribution may commit:
iterate until the remaining halting mass is spent (`rest < 1e-3`), capped at
`EVAL_LOOPS`. This is standard ACT/PonderNet inference, which BRIEF §4.3
explicitly permits, and the training path is untouched and fully differentiable.

A *correctly trained* model then spends `T` iterations on rung `T`:

| readout | iterations over the 16 scoring splits |
|---|---|
| fixed 64-iteration mixture | 16 × 64 = **1024** |
| ACT early exit, trained controller | (1+2+4+8+16+32+64) × 2 ladders + test + ood ≈ **260** |

a **3.9× reduction**, and it is what turns the Easy budget from a failure into a
1.9× margin (idle-GPU per-loop cost 18.3 ms at S = 3: 260 × 18.3 ms = 4.8 s of
model time plus ~11 s of DataLoader start-up across 16 splits).

Per-tier, using the idle-GPU per-loop costs (18.3 ms at S=3, 38.3 ms at S=4,
261 ms at S=5) and the analytic slot-step count for larger `S`:

| tier | dataset | S | eval budget | fixed 64-mixture | ACT early exit |
|---|---|---|---|---|---|
| Easy | e1 | 3 | 30 s | **measured: TimeoutError, run failed** | measured 24.8–31.1 s → **0.96–1.21×** |
| Medium | m1 | 5 | 300 s | ~267 s model time alone → fails | **measured 111.5 s → 2.7×** |
| Hard | hp1 proxy (22-bit N) | 7 | 1800 s | ~1000 s (est.) | **measured 274.5 s → 6.6×** |

**Easy is the binding tier, not Hard.** That inverts the assumption everyone has
been working under, and it is a direct consequence of the eval budget being half
the training budget while the number of scoring splits (16) is the same at every
tier — Easy gets 30 s for the same 16 splits that Hard gets 1800 s for.

### 5.4 Measured evaluator runs

All in `lab/archive.jsonl` under tags `C-evalbudget` and `D-tierbudget`. Every
one of these was taken with 6–13 sibling jobs on the GPU, so the seconds are
pessimistic; the *outcomes* are not.

| manifest | submission | status | steps | eval s | budget | note |
|---|---|---|---|---|---|---|
| `alu_e1_evalonly` (eval only) | fixed 64-iteration mixture | **failed** | 5 | — | 30 s | `TimeoutError` in the `test` split |
| `alu_e1_evalonly` (eval only) | ACT early exit | ok | 5 | 24.8 | 30 s | 16/16 splits |
| `alu_m1_evalonly` (eval only) | ACT early exit | ok | 5 | 111.5 | 300 s | 16/16 splits |
| `alu_hp1_evalonly` (eval only, 22-bit N, S=7) | ACT early exit | ok | 5 | 274.5 | 1800 s | 16/16 splits |
| **`alu_e1_wc60` (tier-faithful Easy)** | ACT, `batch_size=128` | ok | **14** | 27.5 | 30 s | 16/16 splits, MAX_T=0 |
| `alu_e1_wc60` | ACT, `batch_size=32` | ok | **18** | 27.0 | 30 s | 16/16 splits |
| `alu_e1_wc60` | ACT, `batch_size=512` | ok | **15** | **31.1** | 30 s | **eval budget exceeded**: the seen-N ladder stops after T=32 and the **entire OOD-N ladder is `not_completed`** |

The last row is the warning. Nothing about that submission is deeper than the
other two — `eval_batch_size` is 4096 in all three and the model is identical —
it simply landed on the wrong side of a 1.0× margin. **A model whose eval cost
is within noise of the budget silently forfeits the OOD-N tie-break**, which is
the second ranking key on the Hard leaderboard.

---

## 6. The training/eval asymmetry — what it actually buys

The README endorses branching on `self.training`, and there are three things
worth branching on here. Measured value of each:

| lever | training path | eval path | measured value |
|---|---|---|---|
| discrete states | soft (differentiable) | argmax one-hot | **negative**: +42 % wall clock, and §3 shows soft states are *already* exactly one-hot after 64 compositions under bf16 |
| iteration count | `TRAIN_LOOPS` = max training T (3 or 16) | `EVAL_LOOPS = 64` | **essential** — the ladder needs 64, training must not pay for it (§7) |
| halting | full soft mixture over `TRAIN_LOOPS`, no early exit | ACT early exit at spent halting mass | **essential** — 3.9× eval time, and it is what makes Easy fit (§5.3) |

The first is the surprising one: the standard "run it hard at eval" instinct is
a net loss for this architecture, because the constructed chain is already
saturated and the extra `argmax`/`one_hot` kernels are pure launch overhead on a
launch-bound model.

---

## 7. Training-time depth, per tier

Gradients only need to flow through the T values that appear in training, so
training depth is `max(train T)` applications, not 64.

Measured (`--mode time`, batch 128, bf16, under contention; the *ratio* is the
transferable part): a training step (forward + backward) costs **3.4×** an eval
forward at the same loop count.

The **measured** step counts, from tier-faithful evaluator runs (§5.4) and from
the Medium eval-only run's training segment:

| tier | train T | TRAIN_LOOPS | steps in the training budget (measured, contended) | idle-GPU estimate |
|---|---|---|---|---|
| Easy (60 s), N=323, S=3 | {1,2,3} | 3 | **14** at bs 128, **18** at bs 32, **15** at bs 512 | ~150 |
| Medium (600 s), N=10403, S=5 | {4,8,16} | 16 | 5 steps took 136.5 s → **~22** | ~200 |
| Hard (3600 s), 30/32-bit, S=10 | {4,8,16} | 16 | — | ~280 |

Even taking the optimistic idle-GPU column, set it against what `digit-carry`
§2.4 measured: `DigitALU` at S = 3 plateaus at train_exact 0.196 after ~800
steps and does not improve by 2,500; the short-chain variant needs 4,000 to
reach 0.78. **No tier affords even the step count at which the architecture is
already known to fail.** Easy affords ~150.

This is not a small gap to be closed by tuning. It is the difference between
~150 steps and the ≳4,000 the architecture demonstrably needs — a factor of 25
in throughput, or an equivalent factor in sample efficiency.

This is the third argument — after digit-carry's trainability measurement and
§5's eval budget — for the same fix: **shorten the chain**. `R = 11` weight-tied
conditional subtractions is 86 % of the 256 slot updates at S = 3; replacing them
with one learned quotient digit plus a single subtraction takes the chain to
~56 slot updates, a 4.6× cut in *both* the training step time and the eval time,
on top of digit-carry's measured 0.20 → 0.78 train_exact effect.

---

## 8. Downward generalisation to T = 1 and T = 2

Medium and Hard train on `T ∈ {4,8,16}` while certification is a prefix from
T = 1, so the model must generalise **downward** to T values it has never seen.

* **The weight-tied step generalises downward by construction, exactly.**
  §3 measures it: the same constructed block, applied once, is 1.000 at T = 1,
  and applied twice is 1.000 at T = 2, with no T = 1 or T = 2 supervision
  anywhere. There is no per-T parameter to be wrong. This confirms the premise
  the coordinator flagged: a weight-tied T-iterated model is the one
  architecture that can do this.
* **The depth controller does not.** Every Medium cell in §4.2 scores 0.026–0.132
  at T = 1 — the trivial floor — because `loc` is a function fitted on
  `{4,8,16}` and evaluated at 1. The controller is the entire downward-
  generalisation risk.

So the answer is: **downward generalisation works in the component that does the
arithmetic and fails in the component that counts.**

---

## 9. The submission scaffold

`submissions/alu-compose/submission.py` — 16 KiB, lint-clean, **every tensor
learned from random init**, ~8.5 K model-state elements.

```
marker-relative parser (learned pointer, learned R/G, learned probe)
  -> learned slot->digit projection (temperature DIGIT_TAU)
  -> DigitALU, weight tied, applied `loops` times
  -> counted-halting depth controller (PonderNet marginal)
  -> tail-aligned digit head
```

Decisions in it, and the measurement each rests on:

| decision | measurement |
|---|---|
| `batch_size = 128` | §9.1 — kept, but the lever is nearly gone for this model |
| `eval_batch_size = 4096` | §5.1 — 8× the batch costs 5 % |
| `HARD_EVAL_STATES = False` | §3 (soft is already exact) + §5.1 (+42 % if on) |
| ACT early exit at eval, no early exit in training | §5.2 — the difference between a failed run and a completed one |
| `sel_thresh` initialised *shallow* | an untrained halting controller never fires, so it runs the full `EVAL_LOOPS` on every split; initialising the threshold at 0 makes the untrained model ~8 iterations deep and training has to earn depth |
| `TRAIN_LOOPS` = 3 (Easy) / 16 (Medium+Hard), chosen from `OptimizerSpec.training_time_seconds` | §7; `training_time_seconds` is explicitly public, data-independent manifest information |
| counted halting rather than an ordered pointer | §4 — every `(place, digit)`-indexed head is arbitrary off the training T values |
| depth controller in its own optimizer param group at `lr = 0.1` | the halting threshold has to travel O(1)–O(10); at the body's `3e-3` that alone is thousands of steps out of the ~300 the Easy tier affords |

**It is expected to score MAX_T = 0 until a sibling's training procedure lands,
and it does: MAX_T = 0, OOD-N MAX_T = 0 on every evaluator run in §5.4.** It is
a scaffold, not a candidate; it is not better than the baseline and I am not
claiming it is. What it delivers is that the wiring, the dtypes, the budget and
the compliance surface are verified end to end — it lints, it trains, and it
completes all 16 eval splits inside the real 60 s Easy budget — so a trained
`DigitALU` can be dropped into `class DigitALU` and nothing else has to change.

### 9.1 The batch-size lever does **not** transfer to this model

`grok-optimization` measured `batch_size = 512 → 32` as 5.8× more optimizer
steps on Easy, from the evaluator's `num_workers=2` DataLoader with no
`persistent_workers` respawning its workers every epoch. Retested here on the
same tier-faithful Easy manifest, same 60 s, three runs (§5.4):

| `SUBMISSION.batch_size` | steps in 60 s | ratio to bs 512 |
|---|---|---|
| 32 | 18 | 1.20 |
| 128 | 14 | 0.93 |
| 512 | 15 | 1.00 |

**1.2×, not 5.8×.** The sibling's lever is real but it is *DataLoader* overhead,
and it only dominates when the model step is cheap (their block was ~19 ms at
bs 32). This model's step is ~3–4 s at TRAIN_LOOPS = 3 under contention, so the
worker respawn is a rounding error and the lever essentially disappears. The
scaffold keeps `batch_size = 128` because it costs nothing and the effect is
within noise, but **nobody should budget 5.8× more steps for a deep model.**
The transferable version of the finding is the *eval* one (§5.1), which is much
larger here.

---

## 10. What is falsified

1. **"A perfect single squaring step gives MAX_T = 64."** False. It gives
   MAX_T = 2 on Easy and MAX_T = 0 on Medium. The depth controller, not the
   arithmetic, is the binding constraint once the step is exact.
2. **"Iteration/depth in T is solved."** Too strong. *Iterating* is solved — the
   weight-tied block composes exactly to T = 64, in both directions. *Deciding
   how many times to iterate* is not, and is a coverage problem in the T field
   with the same structure as the residue-coverage bound.
3. **"Eval budget is not the constraint; margins are ~5× on Easy and ~200× on
   Hard."** True for a 0.4 M dense block, false here, and the failure is not
   graceful. Measured: a hard `TimeoutError` on Easy, in the `test` split, with
   run status *failed*.
4. **"Running the states hard at eval is free performance."** False for this
   architecture: +42 % wall clock for zero accuracy gain, because the soft chain
   is already exactly one-hot.
5. **"e1 is a safe place to test depth."** False. λ(323) = 144 makes 4, 16 and
   64 iterations the same function, so the depth objective is degenerate there.
   Depth experiments belong at m1's modulus.
6. **"An entropy penalty is the wrong fix because it makes the selector sharp
   but T-independent"** (the sibling's finding) — confirmed and generalised: the
   window-annealing schedule has the same defect for a different reason. The
   loss is flat inside the correct bin, so it constrains `loc` to an interval,
   not to a value (§4.3b).

---

## 11. The single highest-value recommendation

**Stop treating the depth controller as solved, and replace the shorten-the-chain
work's justification with a stronger one: it is now required three times over.**

Concretely, in priority order:

1. **Shorten the chain — `R = 11` → one learned quotient digit plus a single
   subtraction.** It is 86 % of the sequential depth. It is simultaneously
   (a) digit-carry's measured trainability lever (train_exact 0.20 → 0.78),
   (b) the fix for §5's eval-budget failure (4.6×), and (c) the fix for §7's
   step-count famine (4.6× more optimizer steps at every tier). Three
   independent bottlenecks, one change. This is `alu-depth`'s mandate and it
   should be the whole of it.
2. **Make the depth controller's parameters independent of the digits of T.**
   Counted halting with a detector shared with the ALU (§4.3) is the design that
   gets furthest; its remaining leak is that the zero detector is only exercised
   on register values reachable from the training T values. The fix that follows
   from the analysis: **train the depth controller on more T values than the
   dataset provides** — the model may iterate its own block any number of times
   inside one forward, so a self-supervised consistency term
   (`step(state, k+1) = step(step(state, k), 1)`) supervises the counter at every
   depth up to `TRAIN_LOOPS` for free, with no extra data. Nothing in the repo
   has tried this and it is the cheapest untried lever I found.
3. **Move every depth experiment off e1 and e2.** §3.1: their ladders have four
   distinct maps, not seven, so a depth result there is not a depth result.
4. **Set `eval_batch_size` large in every submission.** §5.1, free, applies to
   every candidate on the branch.

If I had one more session I would spend it on (2), because (1) is already owned
and (2) is the bottleneck that this branch discovered and nobody is working on.

---

## 12. Reproduction

```bash
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python

# §3  exactness composes, at five modulus regimes and both dtypes
$VENV lab/probe_compose.py --mode compose --modulus 323 --dtype fp32
$VENV lab/probe_compose.py --mode compose --modulus 323 --dtype bf16
$VENV lab/probe_compose.py --mode compose --modulus 323 --dtype bf16 --eval-hard
$VENV lab/probe_compose.py --mode compose --regime sampled --bits 10 11 --moduli 6 \
    --n-eval 512 --dtype bf16
$VENV lab/probe_compose.py --mode compose --modulus 10403   --n-eval 512 --dtype bf16
$VENV lab/probe_compose.py --mode compose --modulus 4028033 --n-eval 256 --dtype bf16
$VENV lab/probe_compose.py --mode compose --regime sampled --bits 30 32 --moduli 2 \
    --n-eval 64 --dtype bf16

# §4  the depth controller: constructed vs trained on the tier's T values
$VENV lab/probe_compose.py --mode select --selector construct --sel-steps 0 --eval-loops 64
$VENV lab/probe_compose.py --mode select --selector counterz  --sel-steps 0 --eval-loops 64
bash lab/sel_grid.sh          # mlp/linear/placev/place x {1,2,3} and {4,8,16} x 2 seeds
bash lab/sel_counter.sh       # counted halting, per-digit detector
bash lab/sel_counterz.sh      # counted halting, detector shared with the ALU
bash lab/sel_m1.sh            # the same at a modulus whose ladder does not collapse

# §5  throughput and the eval budget
$VENV lab/probe_compose.py --mode time --batch 38 512 4096 --loops 64 --train-loops-max 0
$VENV lab/probe_compose.py --mode time --batch 128 --loops 1 3 16 --train-loops-max 16
$VENV lab/make_manifest.py --dataset e1 --mode wallclock --max-steps 5 \
    --eval-batch-size 4096 --name alu_e1_evalonly
$VENV lab/run_experiment.py --submission submissions/alu-compose/mix64/submission.py \
    --manifest lab/manifests/alu_e1_evalonly.json --tag C-evalbudget     # FAILS: TimeoutError
$VENV lab/run_experiment.py --submission submissions/alu-compose/submission.py \
    --manifest lab/manifests/alu_e1_evalonly.json --tag C-evalbudget     # ok, 24.8s of 30s
```

## 13. Compliance

Nothing under `data/generated/` was read, printed, sampled or summarised; every
probe synthesises prompts and targets from the public generator spec and from
moduli passed on the command line, and §3.1's periodicity table is elementary
number theory on moduli that appear in dataset *directory names*.
`--construct` / `--construct-parse` set tables to exact values and are lab
diagnostics in `lab/`, never imported by a submission — the direct analogue of
`probe_step.py --oracle` and `probe_alu.py --construct`.
`submissions/alu-compose/submission.py` contains no modular-exponentiation
routine, no digit-multiplication rule, no carry rule and no lookup of answers;
every tensor is initialised randomly and trained in the run. Its training path
is fully differentiable with no participant-controlled backward and no custom
training loop; the only `self.training` branches are the loop count, the state
discretisation and the ACT early exit, all of which the README and BRIEF §4.3
permit. The eval-side early exit is data-dependent control flow driven by the
model's **own learned halting scalar**, not by any Python inspection of
`input_ids`; it is the standard ACT inference rule. Nothing was submitted to the
hosted service and no `one-layer login`/`submit` was run.
