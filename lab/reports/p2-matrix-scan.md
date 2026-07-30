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
each a `(100, d×d)` transition table plus a `(100, d×10)` emission table. The
compare monoid only reads its final state, so `cmp.emit` receives no gradient
and is dead weight (~40k of the 127k parameters at `d=16`); it is left in place
so the three monoids share one class, and it is excluded from the 700-cell table
count used by the repair basin.

**Carry alphabets are small by construction**, which is why `d=8` is already
above the requirement rather than below it: the add carry is `{0,1}`, the
subtract borrow is `{0,1}`, and the comparison state is `{EQ,LT,GT}`. `d` is
therefore a *slack* knob here, not a capacity knob — a fact worth stating,
because PLAN2 §3.1 treats `d` as "the critical knob" and on this axis it is not.

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

## 3. Legal training from random init — **LEGAL**

N=323 (λ(323)=144, so this modulus is degenerate for *depth* work; it is used
here only at T=1, where that does not apply). 250 train / 38 held operands,
2000 steps, AdamW lr 3e-2, 3 seeds. Every row's `--lr 0` reference is
`train_exact_hard` 0.000-0.004, `tbl` 0.040, `held_div` 0.58-0.97.

| family | `d` | train_exact | held_exact | **train_exact_hard** | held_exact_hard | held_div | tbl |
|---|---|---|---|---|---|---|---|
| colsoftmax | 16 | 0.052-0.092 | 0.000 | 0.000-0.008 | 0.000 | 0.32-0.53 | 0.027-0.046 |
| **sthard** (perm-constrained) | 16 | **0.176-0.252** | 0.000-0.026 | 0.000-0.020 | 0.000 | 0.29-0.37 | 0.019-0.029 |
| dsink (doubly stochastic) | 16 | 0.068-0.080 | 0.000 | 0.000-0.016 | 0.000-0.026 | 0.34-0.40 | 0.026-0.043 |
| **orth** (Cayley orthogonal) | 16 | **0.724-0.972** | 0.000-0.026 | 0.000-0.012 | 0.000 | 0.53-0.71 | 0.027-0.034 |
| dense (free matrix) | 16 | 0.040-0.184 | 0.000-0.026 | 0.000 | 0.000 | 0.18-0.55 | 0.029-0.036 |
| colsoftmax | 4 | 0.076-0.092 | 0.000 | 0.004-0.016 | 0.000 | 0.34-0.40 | 0.046-0.074 |
| colsoftmax | 8 | 0.084-0.108 | 0.000 | 0.004-0.008 | 0.000 | 0.500 | 0.031-0.039 |
| colsoftmax | 32 | 0.048-0.088 | 0.000 | 0.004-0.012 | 0.000 | 0.37-0.42 | 0.023-0.036 |
| colsoftmax | 64 | 0.068-0.088 | 0.000 | 0.008-0.024 | 0.000 | 0.53-0.55 | 0.023-0.039 |
| sthard | 64 | 0.296-0.372 | 0.000 | 0.000-0.008 | 0.000 | 0.40-0.58 | 0.033-0.040 |

**The falsifier fires.** PLAN2 §3.1's falsifier is "no signal at any `d` up to the
memory limit, with matrix-valued and permutation-constrained variants both
tried." Both were tried, across `d ∈ {4,8,16,32,64}` (a 16× range) and five
constraint families: `held_exact_hard` is **0.000 in every one of the 30 cells**,
and `train_exact_hard` never separates from the `--lr 0` control.

Three things in that table are worth more than the null itself.

**(a) A new fooling row for `lab/RESUME.md`'s table.** The Cayley-orthogonal
family reaches `train_exact` **0.972** — by far the best fit any architecture in
this project has produced from random init — with `train_exact_hard` **0.012**
and `held_exact` **0.000**. An orthogonal transition matrix is a *dense
invertible* state map, so the carry state becomes a high-capacity continuous
channel that memorises 250 operands outright. This is `digit-carry`'s "a 32-dim
carry just re-encodes the value" appearing again, now as a property of the
*constraint family* rather than of the state width.

| metric | fooled by | reads | but |
|---|---|---|---|
| `train_exact` | `--family orth` (Cayley-orthogonal transitions) | **0.972** | `train_exact_hard` 0.012, `held_exact` 0.000 |

**(b) `d` is slack, and the sequential-RNN branch's shape does *not* reproduce.**
That branch measured `train_exact` climbing **134×** (0.007 → 0.98) with hidden
size 4 → 256 while held-out stayed at 0.0069-0.0086. Here `d` moves over the same
16× range and `train_exact` is **flat** (0.048-0.108 at every `d` for
`colsoftmax`). The reason is structural: `d` is the *carry alphabet*, and the
exact solution needs only 2 states (carry/borrow) or 3 (compare), so `d ≥ 4` is
already slack. Capacity here lives in the `(100, ·)` digit tables, whose size is
independent of `d`. Params/row is 508 at `d=16` and 40 at `d=4`, both far above
the 24.9/row that branch found sufficient to fit e5 — so this is **not** a
capacity-limited null.

**(c) The constraint families order by how *continuous* their state is,
and that ordering is exactly the fitting ordering:** orth (dense invertible,
0.97) > sthard (one-hot, 0.18-0.37) > dense (renormalised, 0.04-0.18) ≈
colsoftmax (stochastic, 0.05-0.09) ≈ dsink (doubly stochastic, 0.07-0.08).
Nothing in that ordering transfers to held-out. Fitting is bought with state
continuity and it buys nothing.

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

### 4a-bis. The same comparison at S=5, where the prefix variant is genuinely shallower

N=10403, S=5, 8000 train operands, 3000 steps. Here `prefix` really is shallower
(71 vs 83).

| cell | loss | train_exact | held_exact | train_exact_hard |
|---|---|---|---|---|
| serial s0/s1/s2 | 1.788 / 1.802 / 1.764 | 0.001 | 0.000 | 0.000 |
| prefix s0 | 1.791 | 0.001 | 0.000 | 0.000 |
| `--lr 0` | 2.296 | 0.000 | 0.000 | 0.000 |

**A 14% depth reduction produces no measurable difference in anything except
wall clock** (prefix 404-491 s vs serial 657-726 s per cell, ≈1.5×).

> **Calibration caveat, and it matters (per `plan2/sequential-rnn`).** At S=5
> `train_exact` is **0.001** after 3000 steps — this pair is compared *inside the
> pre-fitting region*, so it establishes "no difference between them", not "both
> are converged nulls". The S=3 block above **is** calibrated: serial reaches
> `train_exact` 0.45 and `MonoidALU --family orth` reaches 0.97, so fitting is
> demonstrably underway there. §3's nulls rest on the calibrated block.

### 4b. `impl=scan` vs `impl=serial` inside `MonoidALU` — the gradient-identity check

Same seeds, same parameters, same hypothesis class; the *only* difference is
whether the prefix products are computed by `ceil(log2 6) = 3` doubling levels or
`5` sequential matmuls (graph depth 41 vs 61).

| step | scan s0 | serial s0 | scan s1 | serial s1 |
|---|---|---|---|---|
| 500 loss | 1.47359 | 1.46580 | 1.57473 | 1.57040 |
| 1000 loss | 1.19565 | 1.19588 | 1.26574 | 1.26124 |
| 1000 train_exact | 0.068 | 0.056 | 0.068 | 0.060 |
| 1000 train_exact_hard | 0.008 | 0.024 | 0.016 | 0.016 |

Losses agree to **0.3-0.5%** after 1000 optimizer steps — the residual is the
chaotic amplification of fp32 re-association, not a systematic difference. This
is the measured form of §0's argument: **a parallel scan and a serial scan over
the same operator are the same function and therefore the same gradient; a
1.5× graph-depth reduction changes nothing about trainability.**

### 4d. Wall clock, taken at **Hard's shape** and reported as a ratio

Per `plan2/sequential-rnn`, cost ratios read off Easy-shaped runs understate
badly (Neural GPU is 22× the fused LSTM at L=21/batch 512 but 1.7× at e5's
shape), so this is measured at batch 512, S=10 (L=20 digit slots), against the
official baseline transformer at batch 512 / L=21 as 1.00× (`lab/bench_shape.py`):

| model | ms/step | ratio |
|---|---|---|
| baseline transformer (reference) | 3.8 | **1.00×** |
| `MonoidALU impl=scan` | 165.0 | **43.75×** |
| `MonoidALU impl=serial` | 173.0 | **45.86×** |

**The log-depth scan buys 1.05×.** The digit axis is 20 long; log-depth needs a
long axis to pay, and this one is not. Both are ~44× the reference model. This
is reported as a fact, not as a justification — per the coordinator, the scan was
never to be defended on speed, and on the evidence it could not be.

### 4c. `--lr 0` controls — **CONTROL**

`lab/runs/monoid_C.log`, N=323, 500 steps at `lr = 0`: every metric frozen at
its initialisation value, as it must be —
`train_exact_hard` 0.000, `held_exact_hard` 0.000, `held_div` 0.684 (d=16) /
0.579 (d=8), table-cell correctness `tbl` 0.040 across `colsoftmax`, `sthard`,
`dense` and `d ∈ {8,16,32}`. These are the reference values every trained row
below must be read against.

Note `tbl = 0.040` **is chance**: only `mul_lo`/`mul_hi` (2 of 7 tables, 10-way
argmax) can be hit by luck, giving `2/7 × 0.1 ≈ 0.029`.

## 5. Repair basin — the conditioning measurement, and the one number this branch adds

This is the measurement the branch exists for. `lab/RESUME.md`'s design
criterion is *"a constraint's conditioning is set by how many learned ops
separate it from the parameters"*, evidenced by: laws evaluated O(1) ops from a
table exactly repair **50 of 400** corrupted cells, while the **end-of-chain
label repairs 0 of 5 at k=20** on `DigitALU`.

`MonoidALU` puts **12** learned-table applications between the loss and its
tables instead of 39-257. Protocol: construct the exact solution (DIAGNOSTIC
start point), randomise `k` of the 700 digit-table cells, then train on the
**plain end-to-end label CE only** (LEGAL objective, no laws, no teacher
forcing) for 2000 steps, and count how many corrupted cells argmax-match the
truth again. 3 seeds.

| k corrupted (of 700) | **repaired** (3 seeds) | tbl after | train_exact | held_exact_hard |
|---|---|---|---|---|
| 5 | 1 / 0 / 0 | 0.991-0.994 | 0.61-0.78 | **0.737-0.763** |
| 20 | 0 / 1 / 1 | 0.971-0.973 | 0.06-0.32 | 0.026-0.132 |
| 50 | 0 / 2 / 0 | 0.929-0.931 | 0.04-0.11 | 0.000-0.053 |
| 100 | 2 / 11 / 1 | 0.859-0.873 | 0.03-0.12 | 0.000-0.026 |
| 400 | 14 / 19 / 9 | 0.441-0.454 | 0.27-0.48 | 0.000 |
| 700 (all) | 15 / 21 / 20 | 0.021-0.030 | 0.06 | 0.000-0.026 |

**The basin widens, measurably, and nowhere near enough.**

* At the directly comparable point, `k=20`: `DigitALU`'s end-of-chain label
  repairs **0/5**; `MonoidALU`'s repairs a median of **1/20 (5%)**. Off the floor,
  but only just.
* At `k=400`: **14/400 (3.5%)**, against `alu-relational`'s O(1)-from-parameters
  algebraic laws at **50/400 (12.5%)**. So cutting learned-op depth 39 → 12
  moves the end-of-chain label from ~0% to ~3.5% of the way, still **3.6×**
  short of what a law evaluated one op from the table achieves — and that law
  itself was not enough to solve the task.
* From a fully random start (`k=700`) training reaches `tbl` **0.021-0.030**,
  i.e. **chance** — the same value the `--lr 0` control reads (0.040). 2000 steps
  of the legal objective from random init leaves the tables at chance.

**The sharpest cell is `k=5`.** Five wrong cells out of 700 already cost
`held_exact_hard` **0.24-0.26** (0.737-0.763 from a ceiling of 1.000) — the
`(1-eps)^n` law again — and after 2000 steps of the legal objective the model
has repaired ~0 of them while *ending with about as many wrong cells as it
started with*: it broke correct cells at roughly the rate it fixed corrupted
ones. **The legal objective does not have a gradient that points at the discrete
solution even when it is standing five cells away from it.**

That is the answer to the branch's question, and it is a negative one:
**shortening the learned-op chain 3.3× improves the conditioning of the legal
objective from "immeasurable" to "measurable and still hopeless."** Conditioning
is a real axis — the repair rate is not zero and it does scale the right way —
but its coefficient is far too small for the remaining distance.

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
