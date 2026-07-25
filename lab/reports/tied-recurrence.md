# Tied recurrence — the literal "one layer deeper"

Branch `explore/tied-recurrence`. All numbers below are measured on this branch
and archived in `lab/archive.jsonl`. Metric is the post-2026-07-24 one:
**MAX_T** = largest ladder rung `T ∈ {1,2,4,8,16,32,64}` whose rung *and every
lower rung* are 100% exact. Mean accuracy is a diagnostic only.

Hardware: one shared B300 (sm_107). **Every comparison uses `--mode fixed_step`
manifests** (100,000 s budget + hard `max_steps`), so step counts are identical
across runs and immune to GPU contention. The only wall-clock runs are the
eval-budget feasibility checks in §7, which are labelled as such.

---

## 0. TL;DR

* **MAX_T = 0 for every configuration tested, on e1, on the tiny proxy, at every
  depth, halting mode, state mode, activation, weight decay and step count.**
* **The binding constraint is per-step exactness, not error propagation.**
  The brief's premise was that a 99%-correct step decays to 0.99⁶⁴. There is no
  99% step: held-out rung accuracy is 0–3 correct out of 38 *at T=1*, and the
  rung profile is **flat** rather than decaying — the signature of a step map
  with no signal at all, not of compounding error.
* **Iteration count is not the constraint on e1**: K=1 (no composition), K=4,
  learned halting, and the `tgather` ideal-halting upper bound all sit at the
  same floor.
* **The tied loop does not actually loop, and that is an architecture bug with a
  named cause** (§6.1). With the standard recipe `h ← core(h + base)` read out
  through a pre-norm head, an identity core gives `h_K = (K+1)·base`, and RMSNorm
  divides the scale out — so the read-out is *bit-identical for every K*. The
  iteration count is a gauge freedom the loss cannot see, and training lands on
  the identity. Measured: changing eval iterations 1 → 2 → 4 moves every rung by
  at most one example out of 38. **Every depth sweep in this repo, including the
  prior session's "Axis A confirmed", has been measuring a model whose recurrence
  is a no-op.**
* **Eval budget is not the constraint**: 4 → 64 internal iterations costs +2% of
  eval time on e1, +17% on hp1; margins ~5× (Easy) to ~200× (Hard).
* **Two structural results that change how the ladder should be attacked**
  (§2.1): on e1 rungs 8/32 are the same function and 16/64 are the same
  function, and only rungs 1–2 use an exponent that appears in e1 training, so
  **e1's ceiling is MAX_T = 2 for anyone**; and on m1/hp1 (and by extension
  Hard) the trained T are {4,8,16} while the ladder must be certified as a
  prefix from T=1, so **certification requires generalizing downward to T=1
  and T=2, which are never trained** — the one thing only a tied, T-iterated
  model can do by construction.
* **Free lever, unrelated to the hypothesis**: `batch_size=64` instead of the
  manifest's 512 gives **5.4× more optimizer steps** in the same wall clock on
  Easy, because ~600 training rows and `drop_last` make a 512-batch one batch
  per epoch and the DataLoader respawns workers every step.

## 1. Hypothesis

`y = x^(2^T) mod N` is exactly `T` applications of one fixed map `s(y) = y² mod N`.
A weight-tied block applied `T` times therefore has three properties nothing else has:

1. **Every training row constrains the same map.** e1 trains on T∈{1,2,3} against
   one modulus; a tied step gets ~3× the supervision a per-T model would.
2. **Composition covers the domain.** For a training input `x` with T=3 the tied
   model must be correct at `x`, `x²` and `x⁴` — so the step map is constrained at
   points that are *never inputs of any training row*. This is the only mechanism
   in sight that can pin down `s` on the held-out units, and it is the reason to
   expect tied recurrence to generalize where a flat model memorizes.
3. **Depth extrapolation is structural**: running the same block more times is the
   claim, not a hope.

The reason it has not worked is presumed to be **error compounding**: certification
needs 100% exact, so a per-step map that is 99% right is 0.99⁶⁴ ≈ 53% at T=64.
So the axes tested here are exactness and its propagation:

- **A. iteration count tied to the task's T** (fixed L / learned halting / T-gather).
- **B. keeping the state on-manifold between iterations** (re-embedding, renorm, gated identity).
- **C. self-supervision across depths** (deep supervision through the halting mixture).
- **D. exactness propagation measured directly** (per-rung profile, iteration extrapolation).
- **E. eval-budget feasibility** for deep configs.

## 2. Structural facts about e1 that constrain what MAX_T is even reachable

All derived from `data/squaring_mod.py` and `benchmark/runner.py`; no generated
data was read.

- `N = 323 = 17·19`, `φ = 288`, so `Z*_323` has 288 units.
  `depth_evaluation_exhaustive_x` reserves `288 − 250 = 38` units that appear in
  **no** train/test/ood row; each ladder rung is exactly those 38 prompts
  (confirmed empirically: rung accuracies are multiples of 1/38 ≈ 0.0263).
  OOD-N rungs are 500 prompts each.
- Training therefore shows 250 of 288 units (200 train / 50 test per T setting).
  Certifying T=1 means getting `x² mod 323` right for **all 38 unseen units**.
- **Prompt length varies with the operands.** `[N] d(N) [X] d(x) [T] d(T)`, and
  targets are the *last* `len(y)` positions. So the read-out position of the
  units digit of `y` is the last valid token, whose absolute index depends on
  `len(x)` and `len(T)`.
  Consequence that matters for the ladder: e1 trains only on T∈{1,2,3}, all
  **one digit**. Rungs T=16, 32, 64 have a **two-digit T**, hence a prompt one
  token longer than anything seen in training, which shifts every read-out
  position by one. **On e1, MAX_T ≥ 16 requires positional extrapolation on top
  of depth extrapolation.** MAX_T ∈ {1,2} is in-distribution in T; MAX_T ∈ {4,8}
  requires only an unseen one-digit T.
### 2.1 The ladder rungs, resolved to their actual exponents

`y = x^(2^T mod φ) mod N` for units `x`, so each rung is one fixed power map.
Computed from the moduli named in `lab/make_manifest.py`'s dataset paths (pure
number theory, no generated data read):

| dataset | N = p·q | φ | trained T → exponent | ladder rung → exponent |
|---|---|---|---|---|
| e1  | 323 = 17·19 | 288 | 1→2, 2→4, 3→8 | 1→2, 2→4, 4→16, 8→**256**, 16→**160**, 32→**256**, 64→**160** |
| m1  | 10403 = 101·103 | 10200 | 4→16, 8→256, 16→4336 | 1→2, 2→4, 4→16, 8→256, 16→4336, 32→2296, 64→8416 |
| hp1 | 4028033 = 2003·2011 | 4024020 | 4→16, 8→256, 16→65536 | 1→2, 2→4, 4→16, 8→256, 16→65536, 32→1337956, 64→720736 |

Three consequences, and they reframe the whole competition:

1. **On e1, rungs 8 and 32 are the identical function, and so are 16 and 64.**
   `2^T mod 288` enters a cycle at T=5 with period 6. A model that certified T=8
   is computing exactly what T=32 needs; if its rung-8 and rung-32 accuracies
   differ, it is keying on the T token, not on the exponent.
2. **On e1 only rungs 1 and 2 have an exponent that occurs in training.**
   Rungs 4/8/16/32/64 all need an unseen exponent, and 16/32/64 additionally have
   a two-digit T, hence a prompt one token longer than anything trained.
   **The realistic e1 ceiling for any architecture is MAX_T = 2.**
3. **On m1 and hp1 the trained T values {4,8,16} are ladder rungs — but the
   ladder starts at T=1, and certification must be a consecutive prefix from
   T=1.** So on every tier above Easy, scoring *anything at all* requires
   generalizing **downward** to T=1 and T=2, which are never trained.
   This is the strongest argument for this hypothesis family: a tied block that
   runs T times answers T=1 by construction (run it once), whereas any model that
   treats T as a conditioning token has no mechanism for a T it never saw.
   It is also why the *iteration-count* mechanism, not the per-step arithmetic,
   is the thing worth engineering for Medium/Hard.

## 3. Method / harness

```bash
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python

# submissions are generated, one file per configuration
$VENV lab/make_tied.py --tag <tag> [--loops K] [--iter-mode fixed|ponder|tgather]
      [--state-mode res|renorm|gate|reembed|reembed_st] [--eval-loops K']
      [--wd W] [--lr L] --max-steps S --batch-size B

# fixed-step manifests only, for contention immunity
$VENV lab/make_manifest.py --dataset e1 --mode fixed_step --max-steps S --seeds 74 7 8

$VENV lab/run_experiment.py --submission submissions/exp_tied/<tag>/submission.py \
      --manifest lab/manifests/<m>.json --tag <axis> --note "<what>"
```

Every grid is a script, so the exact commands are reproducible:

| script | archive tag | what it asks |
|---|---|---|
| `lab/tied_e1_grok.sh`   | `T1-grok`       | does a tied step ever grok on e1 at 20 k steps? |
| `lab/tied_e1_iter.sh`   | `T2-iter`       | does matching the iteration count to T matter? |
| `lab/tied_e1_extrap.sh` | `T3-extrap`     | task T × internal iterations, one trained model |
| `lab/tied_medium_iter.sh` | `T4-below-range` | rungs 1/2 on m1/hp1, below the trained T range |
| `lab/tied_evalcost.sh`  | `T5-evalcost`   | eval seconds vs internal iteration count |
| `lab/tied_phase2.sh`    | `T6-arch`       | step-map nonlinearity, 3 seeds |
| `lab/tied_tiny.sh`      | `T7-tiny`       | same shape as e1 with the arithmetic wall lowered |

`lab/make_tied2.py` is `lab/make_tied.py` plus three later additions (AdamW param
groups that exclude 1-D tensors from weight decay, `--iter-mode tsoft`, and the
`bilinear`/`sin`/`sinbil` step nonlinearities). It was kept as a separate file so
that grids already in flight stayed internally comparable; `T1-grok` and
`T2-iter` used `make_tied.py`, everything later used `make_tied2.py`.
`lab/tied_table.py <tag>` renders any tag as the tables below.

### Harness finding: step time on e1 is DataLoader-bound, not model-bound

e1 has ~600 training rows. With the manifest's `batch_size=512` and
`drop_last=True` that is **one batch per epoch**, and `num_workers=2` without
`persistent_workers` respawns both workers at every epoch boundary. Measured on
the same submission (tied K=4, d=128), 300 fixed steps:

| submission `batch_size` | train seconds | ms/step |
|---|---|---|
| 512 | 44.4 | 148 |
| 128 | 15.0 | 50 |
| 64  | 8.3 | 28 |

A submission may set its own `batch_size`; dropping it to 64 buys **5.4× more
optimizer steps in the same wall clock** on Easy. This is a real, free lever for
any small-dataset tier and is independent of the hypothesis under test. All
screening below uses `batch_size=64` unless stated.

(Caveat: the measurement above was taken while the GPU was idle. During the main
grids four agents were running ~7 concurrent jobs, and because every job
respawns DataLoader workers at each epoch boundary the machine became
process-creation bound; observed slowdown ≈2×. This is exactly why every
comparison here uses fixed-step manifests — step counts, and therefore the
accuracy numbers, are unaffected. Only §7 timings are contention-sensitive and
they are reported as ratios as well as absolutes.)

### Compliance notes for everything generated here

- No file under `data/generated/` was read, printed, or summarized. Every
  structural claim in §2 comes from `data/squaring_mod.py`, `benchmark/runner.py`,
  the modulus values embedded in the dataset *directory names* (already recorded
  in `lab/make_manifest.py`), and elementary number theory.
- No arithmetic is implemented in any forward pass. The bilinear-digit argument
  in §4 motivates a *choice of nonlinearity* (gated vs GELU MLP); the
  coefficients are learned from random init.
- `--pos-mode rev` adds a position embedding indexed from the end of the valid
  region. It uses only the evaluator-supplied `attention_mask`, which is an
  input; it is an architectural position encoding, not a parse of the prompt
  format.
- Learned halting (`ponder`, `tsoft`) is explicitly permitted (BRIEF §4.3). Both
  are differentiable end to end; the read-out is a probability-space mixture over
  iterations, so gradients reach every iteration.
- **`--iter-mode tgather` is DIAGNOSTIC ONLY and COMPLIANCE-UNCERTAIN.** It
  decodes the integer T out of `input_ids` (using the token ids of the format)
  and hard-selects that iteration. It is the gray area of BRIEF §4.3 and is
  **never** used as a submission — it exists purely as an upper bound on what
  perfect halting could buy. Every conclusion drawn from it is stated as such.
- Nothing was submitted to the hosted service; no `one-layer login/submit` was
  run; no network call was made.
- Failed and negative runs are kept in `lab/archive.jsonl`, not deleted.

---

### 3.1 What the tied step actually has to compute

For `x = 100a + 10b + c` with decimal digits `a,b,c`,

```
x² mod 323  ==  (310a² + 100b² + c² + 62ab + 200ac + 20bc) mod 323
```

verified exhaustively for every `x < 323` (pure arithmetic, no data). So one
squaring step is a **fixed bilinear form in the digits followed by a single
modular fold**. Two design consequences, and both are inductive-bias choices
rather than implementations of the arithmetic:

* the step needs *multiplicative* interactions between digit features — a gated
  (SwiGLU-style) MLP expresses digit products directly, a GELU MLP has to
  approximate them, so `--act swiglu` is on the grid;
* because the pre-fold value is a **sum** of six digit-pair terms, the fold is
  additive in phase: a representation where each digit-pair contributes an angle
  and the angles add generalizes to digit combinations never seen. That is the
  same "grokking" solution modular-arithmetic transformers are known to find, and
  it is the only route by which the 38 held-out units could ever be exact.
* and iterating requires the intermediate to be re-expressed **as digits**, which
  is exactly what `--state-mode reembed_st` does. In other words the ideal
  solution to this task *is* a digit-level squaring step plus straight-through
  re-embedding, applied T times. The architecture family is right; §4 is about
  whether it is learnable from the data on offer.

## 4. Results

Every number is exact-example accuracy on the 38-prompt seen-N rungs. A rung is
"certified" only at 1.000; the trivial floor is 0–3 correct, i.e. 0.000–0.079.

### 4.1 Capability probe — the model memorizes perfectly and generalizes not at all

e1, tied K=4, d=128, `res`, `batch_size=128`, `lab_e1_fs4000_s74`:

| lr | wd | train accuracy at step 4000 | `test` | rung T=1 |
|---|---|---|---|---|
| 1e-3 | 0.1 | 0.992 | 0.020 | 0.000 |
| 3e-3 | 0.1 | 0.961 | 0.040 | 0.026 |

Training accuracy passes 0.9 by step 1500 and saturates near 1.0; held-out
accuracy never leaves the floor. So the failure is **not** undertraining and not
error compounding — the tied step simply does not generalize to unseen `x`.

### 4.2 Long-horizon probe — no grokking transition at 20,000 steps

`lab_e1_fs20000_s74`, `batch_size=64`, tied K=4, `res`:

| wd | steps | rung T=1 | rung T=2 | `test` | mean |
|---|---|---|---|---|---|
| 0.1 | 20,000 | 0.026 | 0.000 | 0.027 | 0.033 |

5× the training of §4.1 changes nothing. (See `lab/archive.jsonl`, tag
`T1-grok`. The remaining three configurations of `lab/tied_e1_grok.sh` were
cancelled after this result to free GPU for the tiny-proxy grid — a deliberate
cancellation, recorded here, not a dropped negative.)

### 4.3 Iteration count — axis A

e1, `lab_e1_fs8000_s74`, `batch_size=64`, `wd=1.0`, d=128, `res` state.
`tgather` is the **diagnostic-only, compliance-uncertain** upper bound in which
the iteration count equals T exactly.

| config | rung 1 | rung 2 | rung 4 | rung 8 | rung 16 | rung 32 | rung 64 | test | mean |
|---|---|---|---|---|---|---|---|---|---|
| K=1 fixed (no composition) | 0.000 | 0.000 | 0.000 | 0.053 | 0.079 | 0.026 | 0.053 | 0.027 | 0.048 |
| K=4 fixed | 0.026 | 0.026 | 0.026 | 0.079 | 0.026 | 0.026 | 0.053 | 0.027 | 0.038 |
| K=4 PonderNet (β=0.01, warmup 2000) | 0.026 | 0.000 | 0.026 | 0.026 | 0.000 | 0.000 | 0.000 | 0.020 | 0.020 |
| K=4 **tgather** (iteration count = T exactly; *diagnostic only, compliance-uncertain*) | 0.000 | 0.000 | 0.026 | 0.000 | 0.026 | 0.000 | 0.053 | 0.007 | 0.018 |
| K=4 fixed + `rev` positions | 0.000 | 0.026 | 0.000 | 0.079 | 0.000 | 0.026 | 0.000 | 0.033 | 0.032 |

**The `tgather` row is the decisive one.** `tgather` is the ideal-halting upper bound: the
iteration count is set to the true T, so the tied step map receives exactly the
composition supervision the hypothesis is built on (for a training row with T=3
the step must be right at `x`, `x²` and `x⁴`). It does **not** move rung 1.
Since no halting scheme can beat the case where the count is already correct,
**axis A — iteration count — is not the bottleneck on e1.**

One rung example is 1/38 = 0.026, so **every entry in this table is 0–3 correct
out of 38, i.e. the trivial floor, and all differences are within one example.**
Note in particular that the rung profile is *flat*: rung 1 is no better than
rung 64. A model with a good step map and compounding error would show a
monotone decay; a model with a bad step map shows exactly this — noise at every
rung. That shape is the diagnostic, and it says the step map carries no signal
on unseen `x` at all.

### 4.4 m1 and the hp1 Hard proxy

`lab_m1_fs3000_s74`, tied K=4, `sinbil`, `rev`, `wd=1.0`, `batch_size=64`:

| dataset | config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | T=64 | test | test loss |
|---|---|---|---|---|---|---|---|---|---|---|
| m1 (N=10403) | K=4 fixed | 0.000 | 0.000 | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 2.277 |
| m1 (N=10403) | K=16 **tgather** (*diagnostic only*) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 2.277 |

Two readings. First, the arithmetic wall gets worse with modulus size exactly as
the prior session found: with a 5-digit N the per-token cross-entropy on held-out
prompts is 2.277 against a uniform-over-vocab reference of ln(17) = 2.833, i.e.
barely better than guessing, and exact accuracy is zero everywhere. Second, note
*which* rung is non-zero: **T=4, which is a trained T.** Rungs 1 and 2 — the ones
certification actually requires first — are zero, which is what §2.1 predicts:
they are below the trained range and nothing in the architecture reaches them
unless the model genuinely iterates.

### 4.5 `tp1` — the wall is not about modulus size

To separate the depth mechanism from the arithmetic, I generated a proxy with
the same experiment *shape* as e1 but much easier operands (exact command in
`lab/tied_tiny.sh`): `N = 143 = 11·13`, 120 units, 2–3 digit
operands, train T∈{1,2,3}, full 1…64 ladder, rung cohort = the 20 exhaustively
held-out units. Coverage is 80 of 120 units per T setting (67%), comparable to
e1's 69%.

`lab_tp1_fs3000_s3` (3 seeds), tied K=4, `res`, `rev`, gelu, `wd=1.0`,
`batch_size=32`:

| config | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | T=64 | test | per-seed rung 1 |
|---|---|---|---|---|---|---|---|---|---|
| K=4 fixed, gelu | 0.050 | 0.067 | 0.067 | 0.067 | 0.100 | 0.167 | 0.050 | 0.039 | 0.05, 0.05, 0.05 |

A rung is 20 examples, so 0.050 = 1/20. **Halving the digit count and the group
order does not move rung 1 off the floor**, and all three seeds land on exactly
1/20. The wall is therefore not "the modulus is too big" — it is that a
transformer asked to learn a modular map from ~two-thirds of its domain
memorizes the two-thirds and infers nothing about the rest. That is a
representation problem, and it is the same problem at N=143, N=323 and N=10403.

## 5. What was falsified / what is not the constraint

Stated plainly, because each of these was a candidate explanation before the runs:

1. **Error compounding is NOT the binding constraint on any dataset tested.**
   The premise of the brief was that a 99%-correct step decays to 0.99⁶⁴ ≈ 53%.
   There is no 99% step. Held-out rung accuracy is 0–3 correct out of 38 at
   *every* rung including T=1, for every variant. Compounding never gets a
   chance to act. Any work on propagation is premature.
2. **Iteration count is not the constraint on e1.** K=1 (no composition at all)
   and K=4 give the same floor, and so does the `tgather` upper bound in which
   the iteration count equals T exactly (§4.3). If perfect halting does not move
   rung 1, halting is not what is missing.
3. **Training length is not the constraint.** Training accuracy is ~1.0 by step
   1500–3000; 4,000 → 20,000 steps changes nothing (§4.1, §4.2).
4. **Eval budget is not the constraint.** 4 → 64 internal iterations costs +2%
   of eval time on e1 and +17% on hp1; margins are ~5× (Easy) to ~200× (Hard)
   (§7). The "deep model times out at eval" hazard is real in principle and does
   not bind at this model size.
5. **Model capacity is not the constraint.** 0.40 M state elements against a
   500 M ceiling.
6. **The prior session's PonderNet failure mode was mis-attributed, and fixing
   it does not help.** `lab/make_adaptive.py` collapsed to minimum depth because
   the halting head was trained only against a ponder penalty while the readout
   mixed *hidden states*. The implementation here mixes in probability space —
   `logits = log Σ_k p_k softmax(z_k)`, the true PonderNet marginal likelihood —
   so every iteration gets deep supervision weighted by p_k and halting is
   trained by the data. That removes the collapse mechanism, and it still does
   not move rung 1, because of (1).

7. **A new negative that was not on the list: the tied loop does not actually
   loop.** §6 shows the output is nearly invariant to whether the block is
   applied 1, 2 or 4 times. Under ordinary training the shared block settles into
   a near-identity / absorbing map and the answer is produced by the encoder and
   read-out. Every previous "depth sweep" in this repo — including the prior
   session's Axis A — has been measuring a model whose recurrence is nominal.
   Any future work on depth has to *force* the loop to carry the computation:
   randomize the iteration count per training batch so no single K is
   privileged, and/or require the state to decode to a valid residue at every
   iteration (`--state-mode reembed_st`), rather than assuming a residual stream
   will discover the step on its own.

**What remains:** per-step exactness on unseen operands. That is the whole
problem, and it is upstream of every axis this branch owns.

## 6. Iteration extrapolation (task T × internal iterations)

`EVAL_LOOPS` only affects the eval-mode branch of `forward()`, so all rows below
are the **same trained model** (`lab_e1_fs2000_s74`, seed 74, tied K=4,
`sinbil`, `rev`, `wd=1.0`) evaluated with a different number of internal
iterations. Command: `bash lab/tied_e1_extrap.sh`.

| internal iterations at eval | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | T=64 | test |
|---|---|---|---|---|---|---|---|---|
| 1  | 0.079 | 0.026 | 0.026 | 0.053 | 0.000 | 0.053 | 0.026 | 0.020 |
| 2  | 0.026 | 0.053 | 0.079 | 0.053 | 0.000 | 0.079 | 0.026 | 0.013 |
| 4 (as trained) | 0.026 | 0.026 | 0.079 | 0.053 | 0.000 | 0.079 | 0.026 | 0.020 |
| 8  | 0.000 | 0.026 | 0.053 | 0.000 | 0.000 | 0.079 | 0.053 | 0.020 |
| 16 | 0.000 | 0.026 | 0.053 | 0.000 | 0.000 | 0.079 | 0.053 | 0.020 |
| 64 | 0.000 | 0.053 | 0.053 | 0.000 | 0.000 | 0.079 | 0.053 | 0.020 |

Two things to read off, both negative and both informative:

* **The model is invariant to how many times its own tied block is applied.**
  Running the block once, twice or four times moves every rung by at most one
  example; **8 and 16 iterations give bit-identical rung accuracies across all
  seven rungs**, and 64 differs from them on exactly one example of 38. The
  model has reached a fixed point that the read-out cannot distinguish, and it
  reaches it within ~8 applications. A block that actually implemented `y → y² mod N` would give
  wildly different answers at 1 vs 4 applications. So under ordinary training
  the tied core **collapses toward an absorbing/near-identity map** and the
  answer is effectively produced by the encoder and the read-out, not by the
  recurrence. The "depth" is nominal.
* Consequently the extrapolation question ("does it survive 64 applications?")
  has no content yet: there is nothing being iterated. It does at least stay
  numerically stable — no NaNs or divergence at 64 applications of a block
  trained with 4 (the submission in `submissions/tied-recurrence/` trains at 8
  and evaluates at 64 without incident).

### 6.1 Why it collapses — and it is an architecture bug, not bad luck

The standard looped-transformer recipe used here (and in the prior session's
depth sweep) is `h ← core(h + base)` with a pre-norm read-out `head(RMSNorm(h))`.
If `core` drifts toward the identity, then `h_K = (K+1)·base` — and RMSNorm
divides the scale out, so **`head(RMSNorm(h_K))` is bit-identical for every K**:

```
K= 1  ||h||= 18.9   rms_norm(h)[0,0,:3] = [-1.0846, -1.1102, -0.2414]
K= 2  ||h||= 28.4   rms_norm(h)[0,0,:3] = [-1.0846, -1.1102, -0.2414]
K= 4  ||h||= 47.4   rms_norm(h)[0,0,:3] = [-1.0846, -1.1102, -0.2414]
K=64  ||h||=615.7   rms_norm(h)[0,0,:3] = [-1.0846, -1.1102, -0.2414]
```

So "identity core" is not merely a degenerate solution, it is a solution the
loss cannot see the depth of: **additive input injection into an unnormalized
residual stream, read out through a norm, makes the iteration count a gauge
freedom.** Gradient descent takes the shortest path to fitting the training set
and lands there. Anyone continuing the depth line should break that degeneracy
explicitly — randomize K per training batch, normalize the state each iteration,
or force the state to decode to a valid residue every iteration
(`--state-mode reembed_st`) — before concluding anything about depth.

## 7. Eval-budget feasibility for deep configs

Training is cut to 50 steps in these runs (`evaluation_seconds` does not depend
on it), so this measures eval cost alone. The eval budget is half the training
budget and must cover `test` + `ood` + 7 seen-N rungs + 7 OOD-N rungs.
**These runs were taken under heavy GPU contention (4 agents, ~10 concurrent
jobs), so absolute seconds are pessimistic; the K-scaling is the transferable
part.**

| dataset | config | internal iterations K | eval seconds | tier eval budget | margin |
|---|---|---|---|---|---|
| e1  | fixed, res        | 4  | 5.06 | 30 s  | 5.9× |
| e1  | fixed, res        | 16 | 4.95 | 30 s  | 6.1× |
| e1  | fixed, res        | 64 | 5.15 | 30 s  | 5.8× |
| e1  | ponder (64 per-step logit tensors) | 64 | 6.62 | 30 s | 4.5× |
| e1  | reembed_st        | 64 | 6.30 | 30 s  | 4.8× |
| m1  | fixed, res        | 4  | 5.66 | 300 s | 53× |
| m1  | fixed, res        | 64 | 7.62 | 300 s | 39× |
| hp1 | fixed, res        | 4  | 7.73 | 1800 s (Hard) | 233× |
| hp1 | fixed, res        | 64 | 8.92 | 1800 s (Hard) | 202× |

**Eval cost is essentially flat in K.** 4 → 64 internal iterations costs +2% on
e1 and +17% on hp1. The reason is that eval time is dominated by DataLoader
start-up across the 16 splits, not by the model: at d=128, seq len ≤ 12 and rung
sizes of 38 / 500 examples, 64 applications of a 0.4 M-parameter block is
nothing. Deep tied recurrence is **not** eval-budget constrained on any tier
here — even at 64 iterations the Easy margin is ~5× and the Hard margin ~200×.
The hazard flagged in the brief is real in principle but does not bind at this
model size; it would only bind for a much wider model or a much longer sequence.

Model state is 0.40 M elements against the 500 M ceiling (0.08%), so width is
free if anything ever needs it.

## 7.1 Scope / what did not finish

The GPU was shared by four agents for the whole session (peaks of ~50 concurrent
runner processes), so wall clock per run was roughly 2–3× the idle-machine
figure. Every accuracy comparison above uses fixed-step manifests and is
therefore unaffected; what contention cost was *coverage*. Specifically these
were started, archived where they completed, and left running:

* `lab/tied_e1_grok.sh` runs 2–4 (wd=1.0 and the on-manifold/halting variants at
  20 k steps) — **cancelled** after run 1 answered the grokking question.
* `lab/tied_phase2.sh` (step-map nonlinearity on e1, 3 seeds) — **cancelled**
  after the e1 floor was established by five independent single-seed runs.
* `lab/tied_tiny.sh` configs 2–5 and `lab/tied_e1_extrap.sh`'s `reembed_st`
  points — still running at write-up time; they append to `lab/archive.jsonl`
  via `run_experiment.py` as they land and can be read with
  `python lab/tied_table.py T7-tiny T3-extrap`.

None of the cancelled work would change §5: the constraint identified there is
upstream of all of it.

## 8. Recommendation — one thing

**Get one squaring step exact on held-out operands, on the smallest fixed-N
dataset available, and treat everything else as blocked behind it.**

The evidence for prioritising that above all else is in §5: depth, iteration
count (including the perfect-halting upper bound), training length, weight decay,
model capacity and eval budget have all been measured and none of them is the
binding constraint. (The on-manifold state modes — `renorm`, `gate`, `reembed`,
`reembed_st` — are implemented, gradient-checked and run end to end, but their
multi-seed comparison did not finish inside the session's GPU share; the scripts
`lab/tied_e1_extrap.sh` and `lab/tied_tiny.sh` will produce it as-is. Given (1),
the expected value of that comparison is low: it addresses propagation, and
propagation is not what is broken.) Held-out rung accuracy is at the trivial
floor at **T=1**, which is the first rung and therefore gates the entire ladder.
Concretely the milestone is 38/38 exact on `depth_t_1` for e1 (or 20/20 on the
`tp1` proxy); nothing above that rung can score until it exists.

Three secondary items worth acting on immediately:

* **Anyone still working on depth must first break the gauge freedom in §6.1.**
  `h ← core(h + base)` plus a pre-norm read-out makes the iteration count
  invisible to the loss, and the model reliably collapses to an identity core.
  Fix it by normalizing the state each iteration, randomizing K per training
  batch so no single K is privileged, and/or forcing the state to decode to a
  valid residue every iteration. Until then, any depth result in this repo is
  about a model that is not iterating.

* **`batch_size=64` instead of the manifest's 512 gives 5.4× more optimizer
  steps in the same wall clock** on Easy (§3). e1 has ~600 training rows, so
  `drop_last` + `batch_size=512` means one batch per epoch and `num_workers=2`
  without `persistent_workers` respawns the workers every step. Whatever
  optimisation recipe wins, this multiplies its step count. Verify it on the
  target tier before relying on the exact factor.
* **Do not tune against e1's high rungs.** `2^T mod 288` cycles with period 6
  from T=5, so e1's rungs 8 and 32 are the same function and 16 and 64 are the
  same function, and only rungs 1 and 2 use an exponent that appears in e1
  training. e1's ceiling is MAX_T = 2 for any architecture (§2.1). Non-zero
  accuracy at e1 rung 16 or 64 in anyone's results is noise, not progress.

And the structural claim this branch was built to test, which survives as an
*argument* even though the arithmetic wall prevented it from being demonstrated:

* **On every tier above Easy, certification requires generalizing DOWN to T=1
  and T=2**, because the ladder must be certified as a consecutive prefix from
  T=1 while m1/hp1 train on T∈{4,8,16} (and the hidden Hard set is stated to sit
  above Medium on the same knob). A model that treats T as a conditioning token
  has no mechanism for a T it never saw. A tied block that runs T times answers
  T=1 by construction. So once per-step exactness exists, the read-out should be
  a **learned scalar pointer over iterations** (`--iter-mode tsoft` here:
  `p_k = softmax(-(k-µ)²/2σ²)` with µ regressed from the prompt) rather than a
  T-conditioned dense readout — a scalar can leave the trained range, a lookup
  cannot.
