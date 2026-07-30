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

## 4. `--lr 0` control and the scan/serial gradient-identity check — *(pending)*

## 5. Repair basin: the conditioning measurement — *(pending)*

## 6. Evaluator — *(pending)*

## 7. What resists the scan, and why — *(pending)*
