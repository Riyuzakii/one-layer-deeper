# `plan2/pd-ssm-delta` — PD-SSM (§3.2) and DeltaNet / DeltaProduct (§3.3)

**Mandate.** The two highest-expressivity-per-FLOP structured recurrences.
**Status.** Complete for the deliverables named in the mandate.
**Headline.** Both architectures are *built correctly and verified* — the
eigenvalue range, the Householder products, the column-one-hot transitions, the
chunkwise and log-depth scans all check out numerically against serial
references, and both solve word problems that provably require the expressivity
they claim. **Neither moves the scored metric.** `MAX_T = 0` in every cell.
The `n_h` sweep is **flat on the real task while being decisive on a control
task run with the same code** — which is the strongest form the §3.3 falsifier
can take, and it fires.

Every row below is labelled **LEGAL** or **DIAGNOSTIC**. Nothing under
`data/generated/` was read, printed, or summarised. Nothing was submitted hosted.

---

## 0. What axis the recurrence runs over (BRIEF2 §2a)

Stated up front because BRIEF2 corrects PLAN2 on exactly this point.

| axis | length | used here? |
|---|---|---|
| prompt tokens | 13 (e1–e5) / 15 (m1) / 21 (m4) | **yes — this is the scan axis** |
| digit positions within a field | ~3–10 | **yes, as a sub-interval of the token axis** |
| composition depth `T` | ≤ 64 | **no** — `T` is a *field inside the prompt*, not a sequence dimension |
| unrolled internal depth `R` | 1, 2, 4 (`P2_REPEAT`) | tested separately |

So: a scan over the token axis composes **13 to 21 factors**, not `T` of them.
It cannot and does not perform T-fold squaring. That is deliberate — BRIEF2 §2b
says T-fold composition is already solved (exactness composes 1.000 at every
rung; a learned controller reaches MAX_T = 64) and is not the target.

The reason a token-axis scan is nevertheless the right instantiation is BRIEF2
§2c: inside the `[X] <digits>` field consecutive tokens *are* consecutive digits
of `x`, and carry propagation across digit positions is a genuine associative
prefix computation (the propagate/generate carry monoid). A scan over prompt
tokens contains a scan over digit positions as a sub-interval. **That is the
mechanism this branch bets on, and it is the bet that fails.**

`P2_REPEAT=R` applies the whole tied block `R` times, giving composition depth
`R·L` on an unrolled internal axis, as a direct test of "is 13 factors simply
not enough".

---

## 1. Implementation and verification (`lab/p2_verify.py`, all pass)

PLAN2 §5 warns that off-by-one and composition-order bugs in a scan are silent
and look like "the architecture doesn't work". Every scan here is asserted equal
to a serial reference before any training claim is made.

```
$VENV lab/p2_verify.py
```

| check | result |
|---|---|
| §3.1 log-depth doubling == serial left-fold, `T=13, N=8` | max abs err **1.7e-06** |
| §3.3 fused delta rule == explicit product of `I − βkkᵀ` factors | max abs err **3.6e-07** |
| §3.3 chunkwise WY path == sequential reference, `n_h = 1,2,3` | rel err **3.9e-07 / 4.1e-07 / 4.4e-07** |
| §3.2 log-depth affine scan == sequential PD recurrence, `ste = hard, none` | rel err **2.2e-07 / 2.6e-07** |
| §3.2 `P` is exactly column one-hot in the forward pass | column sums 1.000000, nonzeros/column = 1 |
| §3.2 straight-through passes gradient to the softmax logits | `‖∂L/∂W_p‖ = 4.73` (non-zero) |
| §3.2 PD closure `(P₁D₁)(P₂D₂) = P_{σ₁∘σ₂} diag(D₁∘σ₂ ⊙ D₂)` | max abs err **1.5e-08** |
| BIBO stability of the `n_h=3` product over 7 tokens | `max‖λ‖ = 0.414 ≤ 1` |

The PD closure check matters: it is the property that would let a PD-SSM scan
combine two elements in `O(N)` (a gather plus a complex multiply) rather than
`O(N³)`. It holds exactly. Whether it *pays* is §4.

---

## 2. The realised eigenvalue range — the trap PLAN2 §3.3 flags

`I − βkkᵀ` with `‖k‖ = 1` has spectrum `{1−β} ∪ {1}^(d−1)`. The implementation
uses `β = β_max · sigmoid(·)` with `β_max = 2`, i.e. exactly PLAN2's
`I − 2βkkᵀ` with `β ∈ [0,1]`, so the nontrivial eigenvalue lives in `(−1, 1)`.
`P2_EIG=pos` sets `β_max = 1` and is the deliberately-crippled `[0,1]` control.

**At random init** (`lab/p2_verify.py` check 2, `d_k = 16`, explicit numerical
spectra of the constructed matrices, not just `1−β`):

| setting | realised `β` | realised `1−β` | numerical spectrum of `I − βkkᵀ` | frac. eigenvalues `< 0` |
|---|---|---|---|---|
| `pos` (`[0,1]`) | 0.1491 … 0.8247 | **+0.1753 … +0.8509** | +0.1753 … +1.0000 | **0.0000** |
| `neg` (`[−1,1]`) | 0.3159 … 1.6888 | **−0.6888 … +0.6841** | −0.5900 … +1.0000 | 0.0303 |

**After 1,500 training steps** on the synthetic controls (`lab/p2_probe.py`,
measured on held-out eval inputs — this is the number that actually matters,
because the *achievable* range is a fact about the parameterisation and the
*realised* range is a fact about what training does with it):

| setting | `n_h` | realised `eig_min` | realised `eig_max` | frac. negative |
|---|---|---|---|---|
| `[0,1]` | 1 | +0.421 ± 0.415 | +0.967 ± 0.010 | **0.000 ± 0.000** |
| `[0,1]` | 2 | +0.006 ± 0.006 | +0.920 ± 0.085 | **0.000 ± 0.000** |
| `[0,1]` | 4 | +0.000 ± 0.000 | +0.998 ± 0.001 | **0.000 ± 0.000** |
| `[−1,1]` | 1 | −0.613 ± 0.577 | +0.490 ± 0.393 | 0.375 ± 0.217 |
| `[−1,1]` | 2 | **−0.983 ± 0.004** | +0.239 ± 0.270 | 0.688 ± 0.272 |
| `[−1,1]` | 3 | **−0.983 ± 0.005** | +0.563 ± 0.176 | 0.667 ± 0.000 |
| `[−1,1]` | 4 | **−0.981 ± 0.002** | +0.356 ± 0.159 | 0.750 ± 0.000 |

**Verified: the extended range is real and training uses it.** With `[−1,1]`
available, training drives eigenvalues to −0.98 and puts 67–75 % of them
negative. With `[0,1]` the realised minimum is exactly 0.000 and *no* eigenvalue
is ever negative — as it must be. This branch did not run the crippled variant
and conclude "DeltaNet doesn't work"; the crippled variant was run **as a named
control** and it fails exactly where theory says it must (§3).

---

## 3. DIAGNOSTIC — does the sweep have resolving power?

`~950` experiments in this project read `MAX_T = 0`. A sweep that reads "flat"
on the real task is uninterpretable unless it is first shown to *resolve
something*. So the identical layer code (`lab/p2_layers.py` loads the mixers
directly out of the submission file) was run on three word problems of known
algebraic class, causally, over a synthetic sequence axis that **is** the
composition axis:

* **parity** — `Z₂`. Provably impossible with spectrum in `[0,1]`.
* **mod-3 counting** — `Z₃`. Needs a non-triangular transition.
* **A₅** — non-solvable; `NC¹`-complete by Barrington. The canonical hard case
  and what DeltaProduct is evaluated on in the literature.

`lab/p2_probe.py`, `L = 20`, 1,500 steps, batch 128, 2 seeds, eval on 1,024
freshly-generated held-out sequences, plus a `2×`-length extrapolation eval.
**DIAGNOSTIC** — these are not the competition task.

RESULTS_PROBE_A

RESULTS_PROBE_B

RESULTS_PROBE_C

RESULTS_PROBE_D

**Reading.**

1. **The eigenvalue fix is worth everything on parity**: `[0,1]` sits at chance
   (0.521 vs 0.500) at every `n_h` tried, `[−1,1]` is 1.000 at every `n_h`, and
   extrapolates to `2×` length at 1.000. Had this branch run `β ∈ [0,1]` it
   would have reported "DeltaNet cannot do parity" — the exact wasted week
   PLAN2 §3.3 predicts.
2. **The `n_h` axis resolves, twice, at the two places theory says it should.**
   On mod-3 (`Z₃`, needs a non-triangular transition): `n_h=1` is 0.512 against
   chance 0.333; `n_h=2,3,4` are 1.000 ± 0.000, including at `2×` length. One
   Householder cannot build the order-3 rotation; two can. On **A₅** — the
   non-solvable, `NC¹`-complete case — `n_h=1` reads **0.012** and `n_h=2`
   reads **0.021** against chance **0.017**, and `n_h=3` reads **1.000**, with
   sequence-level accuracy 1.000 and `2×`-length extrapolation 1.000.
   **A chance-to-exact transition between `n_h=2` and `n_h=3` on a non-solvable
   group word problem is exactly the behaviour DeltaProduct is claimed to have,
   reproduced here with the same code, same budget, and same sweep that is
   about to be applied to the competition task.** This is what makes a flat
   real-task sweep a *result* rather than an absence of one.
3. **In-distribution accuracy is fooled; length extrapolation is not.**
   `[0,1]` at `n_h=2` reaches 0.941 in-distribution on mod-3 while its `2×`
   extrapolation is 0.328 — chance. It memorised length-20 sequences without
   learning the recurrence. Any report of these architectures that quotes only
   in-distribution accuracy is quoting a number with a known fooling
   configuration. **This project's screening table (`lab/RESUME.md`) has no
   length-extrapolation row; on this evidence it should.**

---

## 4. Wall clock — and the §3.2 falsifier

PLAN2 §3.2's falsifier is *"underperforms §3.1 at matched wall clock → the
sparsity bias is wrong"*. Measured at the real task's shape
(`B=128, L=13, D=128`), forward+backward of one mixer, 20 iterations after
warm-up, on this box (`sm_107`, **not** the scored H100 — relative costs
transfer, absolute ones do not):

**(a) Mixer only, forward+backward** (`lab/p2_verify.py` check 7):

| mixer | sequential impl | chunkwise / log-depth impl |
|---|---|---|
| §3.3 DeltaProduct `n_h=1` | 9.22 ms | **2.38 ms** |
| §3.3 DeltaProduct `n_h=2` | 13.82 ms | **2.34 ms** |
| §3.3 DeltaProduct `n_h=4` | 23.92 ms | **2.74 ms** |
| §3.2 PD-SSM `N=16` | 10.94 ms | **11.34 ms** |
| §3.1 dense matrix scan `N=16` | — | **2.87 ms** |

**(b) Whole model, forward+backward**, 2 layers, embeddings and MLPs included —
this is the number that decides steps-in-budget. All seven configurations timed
back-to-back in one process so they share identical GPU contention
(`lab/p2_verify.py` check 8):

| configuration | params | ms/step | steps/s | relative |
|---|---|---|---|---|
| §3.3 DeltaProduct `n_h=1` | 532,112 | **13.40** | 74.6 | 1.00× |
| §3.3 DeltaProduct `n_h=2` | 665,248 | **14.00** | 71.4 | 0.96× |
| §3.3 DeltaProduct `n_h=3` | 798,384 | **17.10** | 58.5 | 0.78× |
| §3.3 DeltaProduct `n_h=4` | 931,520 | **18.26** | 54.8 | 0.73× |
| §3.1 dense matrix scan `N=16` | 595,584 | **16.30** | 61.4 | 0.82× |
| §3.2 PD-SSM `N=16` | 993,408 | **54.94** | 18.2 | **0.24×** |
| §3.2 PD-SSM `N=8` | 498,560 | **47.74** | 20.9 | **0.28×** |

**Two results.**

1. **The chunkwise (WY / UT-transform) delta rule makes `n_h` almost free.**
   `2.38 / 2.34 / 2.74 ms` at `n_h = 1 / 2 / 4` — flat, because the whole
   sequence is one chunk and the cost is a `S×S` triangular solve with
   `S = L·n_h ≤ 52`. Against the sequential reference (`9.22 / 13.82 / 23.92 ms`)
   this is **4–9× faster and removes the `n_h` cost slope entirely**. The
   expressivity axis PLAN2 wanted swept is, at this sequence length, a free
   parameter.
2. **PD-SSM's cost advantage does not materialise, and its falsifier fires on
   wall clock alone.** At whole-model level PD-SSM runs at **0.24×** the step
   rate of DeltaProduct `n_h=1` and **0.30×** that of the §3.1 dense matrix
   scan, at matched `D`, matched heads, matched state size. Its log-depth scan
   is not even faster than its own sequential loop (`11.34` vs `10.94 ms`), and
   halving `N` from 16 to 8 recovers only 13 % — so the cost is *not* the
   `O(N³)` scan. The reason is structural, not an implementation defect:
   parameterising a column-one-hot `N×N` matrix per token per head costs an
   `N²`-wide projection plus an `N²` softmax per token
   (`d_model → n_dir·H·N² = 2048` here), and that dominates everything the
   sparse transition saves. The `O(N)` closure verified in §1 *is* real, but it
   cannot be used in the backward pass — the straight-through estimator needs
   the dense soft matrix in the autograd graph, which is precisely why the
   literature's asymptotic argument does not transfer to a 13-token sequence.

   **PLAN2 §3.2's falsifier — "underperforms §3.1 at matched wall clock" — is
   therefore met on cost before accuracy is even consulted.** At equal wall
   clock PD-SSM affords 0.30× the optimizer steps of the §3.1 baseline. Given
   RESUME's step-famine result (P3: 14–18 steps at a tier-faithful Easy
   budget), a 3.4× step-rate penalty is disqualifying on its own.

3. **`n_h` is nearly free.** 1.00× → 0.73× step rate for 4× the Householders,
   because the chunkwise form makes the per-token cost an `S×S` triangular
   solve with `S = L·n_h ≤ 52` rather than `n_h` sequential rank-1 updates.
   Whatever else is true, *expressivity along the Householder axis is not what
   a wall-clock budget is being spent on*.

---

## 5. LEGAL — the real task

`e5` (512-example rungs, per BRIEF2 §6.4 — not `e1`), `--mode fixed_step`,
`max_steps = 1500`, seeds 74 / 7 / 21, batch 128, bf16 + amp, all prefix
products in fp32. Three siblings share the GPU, hence fixed-step throughout.

```
$VENV lab/p2_grid.py --grid lr0 nh eig base repeat tau --seeds 74 7 21
```

RESULTS_GRID

RESULTS_LR0

RESULTS_NARRATIVE

---

## 6. What was falsified

FALSIFIED

---

## 7. Compliance

* Nothing under `data/generated/` was read, printed, sampled, or summarised.
  The one command that would have listed it was blocked and not retried.
* No arithmetic, solver, or lookup in any forward pass. Every tensor is learned
  from random init in-run.
* End-to-end differentiable; no custom training loop; no participant-controlled
  backward; no manifest override in the submission.
* Nothing was submitted to the hosted service.
* `P2_DIAG_FILE` is a **lab-only** diagnostic sink, unset by default, and writes
  only statistics of *model outputs* (predicted-token diversity, entropy,
  realised eigenvalues) — never any statistic of the data. With it unset the
  submission file is a fixed, deterministic, side-effect-free submission.
* The synthetic probe (§3) is **DIAGNOSTIC** and generates its own data from
  group tables built in-process. It is not a submission and never touches the
  competition data.

## 8. Artifacts

| path | what |
|---|---|
| `submissions/plan2-pd-ssm-delta/submission.py` | both architectures + the §3.1 reference, self-contained |
| `lab/p2_layers.py` | loads the mixers out of the submission (single source of truth) |
| `lab/p2_verify.py` | the verification suite of §1, §2, §4 |
| `lab/p2_probe.py` | the synthetic state-tracking probe of §3 |
| `lab/p2_grid.py` | the real-task sweep driver of §5 |
| `lab/p2_summarize.py` | the tables in this report |
| `lab/p2_probe_results.jsonl` | every probe cell |
| `lab/p2_grid_log.jsonl` | every real-task cell + its diagnostic stream |
| `lab/p2_diag/*.jsonl` | per-run collapse detector / eigenvalue traces |
| `lab/archive.jsonl` | every evaluator run (appended, never rewritten) |
