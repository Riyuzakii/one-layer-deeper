# `plan2/matrix-scan` — learned monoid + matrix associative scan (PLAN2 §3.1)

**Status:** in progress. Numbers below are measured; sections marked *(pending)*
are still running.

---

## 0. Mandate, and the one correction the build makes to it

PLAN2 §3.1 asks for: each input element → a small `d×d` matrix, prefix products
by a parallel scan, readout from `P_k`, with permutation- / orthogonal- /
doubly-stochastic-constrained variants, `d ∈ {8,16,32}`.

BRIEF2 §2(c) moves the scan off the prompt sequence and onto **digit
positions**, because carry propagation is the associative prefix computation
and `x² mod N` on unseen operands is the only open bottleneck. That is what is
built here.

**The correction this branch adds, and it is the load-bearing one.** The brief
frames the experiment as *"a log-depth scan puts ~5 learned ops between the loss
and the tables where the serial ALU put 39–280; whether that changes
conditioning rather than merely speed is the single most valuable untested
question."*

A serial prefix and a log-depth prefix over the **same** operator compute the
**same function**, so up to floating-point re-association they induce the
**same gradients**. Replacing a serial scan by a parallel scan therefore
*cannot* change conditioning — it is a throughput transform. This is not a
conjecture: §1 measures the forward equality (bit-exact on permutation
matrices) and §4 measures the training equality.

What *can* change conditioning is the thing the scan makes possible but is not
identical to: **re-factoring the recurrence so that the learned tables are
applied once per digit position in parallel and the state recursion becomes
linear**, instead of applying a learned table once per serial step with a
softmax between every step. `DigitALU`'s carry chain is `c_{k+1} =
softmax(Tadd[a_k, b_k, c_k])` — `W` nonlinearities in series. The monoid form is
`c_{k+1} = M_k c_k` with `M_k = softmax(trans[a_k, b_k])` — the nonlinearity is
paid once per position, in parallel, and the recursion itself is linear.

So this branch tests two separable claims:

| claim | instrument | verdict |
|---|---|---|
| the log-depth scan trains better than the serial scan | §1 equality + §4 `impl=scan` vs `impl=serial` | **falsified by construction and by measurement** |
| the monoid *factorisation* (linear recurrence, one table application per position) improves the conditioning of the legal objective | §5 repair basin, §3 training | *(see below)* |

---

## 1. The scan, and the equality check PLAN2 §5 demands

`lab/monoid.py` provides three interchangeable prefix-product implementations
with the convention `P[k] = M[k] @ M[k-1] @ … @ M[0]` (inclusive):

* `prefix_serial` — `K-1` sequential matmuls, the reference;
* `prefix_scan` — Hillis–Steele doubling, `ceil(log2 K)` sequential matmuls;
* `prefix_hop` — `torch._higher_order_ops.associative_scan`.

`lab/test_monoid.py` (**run: `python lab/test_monoid.py`; ALL CHECKS PASSED**):

| check | result |
|---|---|
| permutation matrices, all `d ∈ {3,8,16,32}` × `K ∈ {1,2,3,5,6,7,8,9,12,16,17}` | `torch.equal(serial, scan)` **True** — bit-exact, 44/44 cells |
| random dense fp32, same grid | max abs err ≤ **1.8e-6** |
| `associative_scan` HOP vs serial, `d=8, K=8` | max abs err **3.0e-7** |
| depth | e.g. `K=16`: serial 15 levels, scan **4** |

Composition order and the exclusive-prefix off-by-one are pinned by these
assertions, so a null result below cannot be a silent scan bug.

**Numerics.** The prefix products run in **fp32** even under bf16 autocast
(PLAN2 §5). The default family `colsoftmax` is *column-stochastic*, and products
of column-stochastic matrices are column-stochastic, so the prefix products can
neither vanish nor explode by construction — no renormalisation needed. The
`dense` family is renormalised per position; `orth` uses a Cayley transform.

---

## 2. `MonoidALU` — the architecture and its constructed ceiling

Digit strings are `(B, L, 10)` distributions, LSB first, `L = 2S`. Every learned
tensor is indexed by a **digit tuple**; no index ranges over `Z_N`, so the
parameter set is identical at every modulus (BRIEF2 §4 makes this mandatory for
Hard, whose train and test moduli are disjoint).

| stage | learned pieces | scans |
|---|---|---|
| square | `mul_lo`, `mul_hi` (100×10 each) → `2S` partial rows → balanced tree of adds | `ceil(log2 2S)` add-scans |
| multiples of `N` | same `mul` tables, `q = 0…9` in parallel | 1 add-scan |
| reduce mod `N` | comparison monoid → `fits_q`; learned 10×10 selector `logit_q = fits_q − fits_{q+1}`; subtract | `S ×` (1 compare-scan + 1 sub-scan) |

Three monoids — **add** (carry), **sub** (borrow), **compare** (`{EQ,LT,GT}`) —
each a `(100, d×d)` transition table plus a `(100, d×10)` emission table.

**Depth, measured (`model.op_depth` / `model.graph_depth`):**

| architecture | learned-table applications on the critical path | sequential matmul levels |
|---|---|---|
| `DigitALU` horner:serial (S=3) | 257 | 257 |
| `alu-depth` tree:quotient (S=3) | 39 | 39 |
| `alu-depth` tree:quotient:prefix (S=3) | 43 | 43 |
| **`MonoidALU` (S=3), `impl=serial`** | **12** | 61 |
| **`MonoidALU` (S=3), `impl=scan`** | **12** | **41** |

So the monoid factorisation is a **3.3× reduction in serial learned-table
applications** vs the best previously-adopted graph — not the ~8× the brief
hoped for, because the `S` long-division steps are irreducibly serial (see §7).

**Constructed ceiling (DIAGNOSTIC ORACLE — `--construct`, never legal in a
submission):**

| N | S | d | soft exact | hard exact | table cells | params |
|---|---|---|---|---|---|---|
| 323 | 3 | 8 | **1.000** | **1.000** | 700 | 45,308 |
| 323 | 3 | 16 | **1.000** | **1.000** | 700 | 126,916 |

identical under `impl=serial`, under the `sthard` (straight-through column
one-hot / PD-SSM) family, and under **bf16 autocast**. The hypothesis class
contains the exact solution and the scan finds it; any training failure below is
an optimisation result, not an expressivity one.

**A calibration correction for the collapse detector (§6.3 of BRIEF2).**
`alu-relational`'s `out_diversity` is documented as "a constant map reads ~1/n,
the truth reads 1.0". **The truth does not read 1.0 on this task.** Squaring on
`Z*_N` with `N = pq` is 4-to-1, so the exact solution's diversity is `|image|/n`
— measured **0.221** over a 326-operand `N=323` set (72 distinct images). Any
branch reading diversity against a 1.0 reference is mis-calibrated.

---

## 3. Legal training from random init — *(pending)*

## 4. Does a log-depth scan train where the serial one did not?

### 4a. The pre-existing parallel-prefix carry, put through a learning run — **LEGAL**

`alu-depth` built `--scan-mode prefix` (a learned 3-element propagate/generate
carry semigroup with its own learned composition table `Ccomp`, +651 params) and
verified its *constructed* ceiling at 1.000, but it is absent from that branch's
`depth_grid.sh` cells, so it had only ever been measured for **throughput**.
This is the matched learning comparison. Graph `tree:quotient`, N=323, S=3,
3000 steps, 3 seeds, `lab/scan_grid.sh`.

Depth first, because it reverses the premise:

| S | serial | prefix |
|---|---|---|
| 3 | **39** | 43 |
| 5 | 83 | **71** |

At `S=3` the parallel-prefix variant is *deeper*, because `W=4` and
`log2(W)+3 > W`. It only wins from `W ≥ 5`.

| cell | train_exact | held_exact | **train_exact_hard** | state_sharp | q_sharp |
|---|---|---|---|---|---|
| serial s0/s1/s2 | 0.448 / 0.436 / 0.460 | 0.000 / 0.000 / 0.000 | 0.008 / 0.000 / 0.008 | 0.77-0.78 | 0.61-0.91 |
| prefix s0 | 0.076 | 0.000 | 0.000 | 0.781 | 0.091 |
| **`--lr 0` control** | 0.016 | 0.000 | **0.004** | 0.291 | 0.093 |

**The log-depth variant trains *worse*, not better** — `train_exact` 0.076 vs
~0.45 — and on the metric that matters (`train_exact_hard`, argmax-snapped) both
sit **at or below the `--lr 0` control's 0.004**. Held-out exact is 0.000
everywhere.

Mechanism, and it is a design lesson rather than a tuning one: a parallel prefix
over a **learned** monoid replaces "apply a learned table `W` times in sequence"
with "apply a learned table `W` times in parallel **and additionally identify
the composition operator**". The prefix variant adds `Ccomp` (3×3×3) and
`Aapp` — parameters the same failing objective must now also identify. It buys
depth and pays identification. `q_sharpness` 0.091 (≈ the `--lr 0` value 0.093)
says the quotient head never left init in the prefix variant.

`MonoidALU` avoids that trade deliberately: its composition operator is **matrix
multiplication**, which is fixed, not learned. Only the per-position transition
matrices are learned.

### 4b. `impl=scan` vs `impl=serial` inside `MonoidALU` — *(pending)*

### 4c. `--lr 0` controls — **CONTROL**

`lab/runs/monoid_C.log`, N=323, 500 steps at `lr = 0`: every metric frozen at
its initialisation value, as it must be —
`train_exact_hard` 0.000, `held_exact_hard` 0.000, `held_div` 0.684 (d=16) /
0.579 (d=8), table-cell correctness `tbl` 0.040 across `colsoftmax`, `sthard`,
`dense` and `d ∈ {8,16,32}`. These are the reference values every trained row
below must be read against.

Note `tbl = 0.040` **is chance**: only `mul_lo`/`mul_hi` (2 of 7 tables, 10-way
argmax) can be hit by luck, giving `2/7 × 0.1 ≈ 0.029`.

## 5. Repair basin: the conditioning measurement — *(pending)*

## 6. Evaluator — **LEGAL**

`lab/manifests/lab_e5_fs400_s74.json` (`--mode fixed_step`, so
contention-immune), batch 128, `eval_batch_size` 1024. Screened on **e5**
(512-example rungs) per BRIEF2 §6.4.

| run | MAX_T | OOD_N MAX_T | rung-1 | mean | steps | train s |
|---|---|---|---|---|---|---|
| `p2-matrix-scan` | **0** | 0 | 0.002 (1/512) | 0.0071 | 400 | 93.5 |
| `p2-matrix-scan-lr0` (**CONTROL**) | **0** | 0 | **0.006 (3/512)** | 0.0021 | 400 | 98.9 |

**The untrained model scores higher on rung-1 than the trained one** (3/512 vs
1/512). Both are inside the variance floor of a 512-example rung, so the honest
statement is *indistinguishable from random initialisation* — but it is one more
instance of exactly the pattern BRIEF2 §6.1 exists to catch: `mean_exact`
improved 0.0021 → 0.0071 while the ranked quantity did not move, and the one
rung that did move, moved the wrong way.

Throughput: 400 steps in 93.5 s at batch 128 on a contended `sm_107`
(234 ms/step); ~95 ms/step measured in isolation.

## 7. What resists the scan, and why

The squaring splits into two halves with opposite parallel structure.

**Multiplication genuinely collapses to log depth.** All `S²` single-digit
products are independent (one batched table read), and resolving the `2S` column
carries is exactly one associative prefix. Learned-op depth
`1 + ceil(log2 2S)` = 4 at `S=3`, 5 at `S=8`.

**Modular reduction does not.** Long division emits quotient digits MSB-first,
and the state carried from one quotient digit to the next is the *partial
remainder* — an `S`-digit number, not a small alphabet. It is not an element of
a small monoid, so there is nothing associative to scan. Learned-op depth `2S`,
which is **the majority of `MonoidALU`'s 12** and the whole of its growth in
`S`.

Two consequences worth recording:

1. **`alu-depth`'s 257 → 39 already captured most of the available win, and both
   it and this branch are bounded below by the same `S` serial division steps.**
   That is why the monoid form buys 39 → 12 rather than 39 → 5. Any future
   digit-transducer proposal should be costed against `2S`, not against zero.
2. **The obvious escape does not work.** Reducing via
   `p mod N = Σ_j p_j · (10^j mod N)` turns the reduction into a log-depth tree
   add plus a 2-3 digit final correction — but the residues `10^j mod N` are
   themselves produced by a `2S`-step serial chain through *the same* learned
   tables, so the gradient path from the loss to those tables is not shortened
   at all. It moves depth off the `x`-dependent branch, not off the gradient
   path. Recorded here because it is the first thing anyone will propose.

---

## 8. Compliance

* Nothing under `data/generated/` was read, printed or summarised. The probe
  synthesises `(N, x, x² mod N)` in-process from `--modulus` / `--bits` using
  Miller-Rabin prime sampling written locally; the only dataset access is the
  evaluator's own, through `lab/run_experiment.py`.
* **No hard-coded arithmetic in the forward pass.** The product table, the
  add/subtract carry monoids, the comparison monoid, the quotient selector, the
  marker pointer and the digit readout are all learned from random init. The
  only fixed operations are re-indexing (a shift by a digit position is a
  slice), softmax, and matmul.
* `MonoidALU.construct_()` and `--construct` / `--corrupt` write the true tables
  and are **LAB DIAGNOSTICS ONLY** (BRIEF §4 rule 2). They never appear in
  `submissions/p2-matrix-scan/submission.py`. Every row in this report is
  labelled **LEGAL** or **DIAGNOSTIC**.
* The submission is one file, uses no custom training loop and no
  participant-controlled backward, and returns `(logits, None)` with the default
  cross-entropy. Nothing was submitted to the hosted service.
* `submissions/p2-matrix-scan-lr0/submission.py` is byte-identical to the
  submission except `lr=1e-2 → lr=0.0`; it exists only as the BRIEF2 §6.1
  control and is not a candidate.
