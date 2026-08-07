# `hard/o1-reduction` — modular reduction at O(1) learned-op depth, on Hard-faithful `hf1`

**Status: complete.** Every number below was measured on this branch.
Runs: `lab/runs/*.log`, `lab/runs/*.jsonl`; evaluator rows in `lab/archive.jsonl`.

**Verdict in one line.** The reduction *was* taken to O(1) — learned-op depth
`2S` → **5**, constant in the modulus size, with the constructed ceiling
certified **1.000 soft AND argmax-hard on `hf1`'s own moduli** — and the
conditioning law's coefficient **did not become large enough to matter**: the
legal end-of-chain label repairs **0/5, 0/20 and (see §4) at k=400** where
`MonoidALU` at depth 12 repaired 1/20 and 14/400. Conditioning improved by a
factor of ~2.4 in depth and the basin did **not** widen with it. That breaks the
monotone dose-response the mandate was extrapolating.

---

## 0. Mandate, and the one thing it got wrong

RANKING.md #3, set by `plan2/matrix-scan` §9: *"modular reduction at O(1)
learned-op depth, modulus-independent … cost it against `2S`, not against zero,
and measure it with the repair-basin protocol before training anything."*

The measured law being exploited:

| architecture | learned-op depth | exact repair |
|---|---|---|
| `DigitALU` | 39 | 0/5 at k=20 |
| `MonoidALU` | 12 | 1/20, 14/400 |
| O(1) algebraic laws | 1 | 50/400 |

The framing correction I was given — *a parallel prefix and a serial prefix over
one operator are the same function with the same gradients, so reassociating a
product does not reduce learned-op depth* — is right, and this branch does not
touch scans. It reduces the number of **distinct learned applications** between
the loss and a table.

**What the mandate got wrong, and it is the load-bearing correction this branch
makes.** The table above is not a one-parameter family. Its bottom row is a
*different objective* (an algebraic law evaluated directly on a table) and its
top two rows are the *same objective* (the end-of-chain label) at two depths.
Reading it as "label repair scales with depth, extrapolate to 1" assumes the
label at depth 1 behaves like a law at depth 1. **This branch is the measurement
that separates them, and they do not agree.**

---

## 1. The architecture: what actually removes the `2S`

`plan2/matrix-scan` §7 located the obstruction exactly: long division emits
quotient digits MSB-first and the state carried between them is the **partial
remainder**, an S-digit number. It is not an element of a small monoid, so
there is nothing associative to scan; depth `2S`, and it is the whole of the
growth in `S`.

Two changes remove it.

### 1a. Barrett reduction — the serial quotient becomes two products

For `p = x² < 10^{2S}` and `mu = floor(10^{2S}/N)`:

```
q = trunc_{2S}(p * mu)        # floor(p/N) or one less
r = p - (q + c) * N           # c in {0,1} selected by the sign
```

`trunc` is a slice. The bound `q ∈ {floor(p/N)-1, floor(p/N)}`, hence
`r ∈ [0, 2N)`, is *verified* rather than assumed — over hf1's ID moduli
(16/18/20-bit) and OOD-N moduli (17/19/21-bit), 3,000 operands each, the largest
`floor(r/N)` is **1** (`lab/test_o1reduce.py`). That is why `n_corr = 2`
suffices, which matters for compliance (§7): with two candidates the extra
subtrahend is a 0/1 *gate*, not a multiplication by `c`.

### 1b. `ColSum` — a multi-row accumulation in one learned application

`MonoidALU` paid `1 + ceil(log2 2S)` for a multiply: one product-table read plus
a balanced tree of pairwise learned adds. Here the partial products of a whole
column are pooled into a **bag of digit-pair indicators** — a plain sum of
one-hots, which counts *which pairs occur*, not what they are worth — and a
single learned table `Wp: (role, 100, M)` maps the bag to a distribution over
that column's total. A second learned table (`CarryMonoid`) resolves carries
across columns as a linear monoid, so the nonlinearity is paid once per column
in parallel and the composition is fixed matmul (the accounting
`plan2/matrix-scan` used for `PairMonoid`).

Subtraction is a *role* inside the same bag, not another module. So
`p - (q+c)N` for **every** correction `c` at once is one `ColSum` plus one
`CarryMonoid`, and the carry monoid's **final state is the sign** — which
removes the compare-and-subtract loop entirely.

The exact solution is in the class: with column total `t`, the score
`LAM*(m*t) - LAM*m²/2` equals `-LAM/2 (m-t)²` up to a term constant in `m`, and
both pieces are linear in the bag plus a bias.

### 1c. Depth

| architecture | reduction | total | grows with S? |
|---|---|---|---|
| `DigitALU` `horner:serial` | — | 257 | yes |
| `alu-depth` `tree:quotient` | — | 39 | yes |
| `MonoidALU` (S=3 / S=7) | 8 / **16** | 12 / **21** | yes (`2 + 2S`) |
| **`O1ReduceALU` (any S)** | **5** | **7** | **no** |

`model.reduce_op_depth` and `model.op_depth` are asserted in the gate at S=3 and
S=7. At `hf1`'s S=7 that is **21 → 7 overall and 16 → 5 on the reduction**, and
the reduction figure is now a constant.

Depth is counted the way `plan2/matrix-scan` counts it: the number of learned
**table applications** on the critical path, inclusive of the table itself, with
prefix products of learned matrices counting once (the composition is fixed
matmul, so the nonlinearity is paid once however long the digit string is). So
`rs_carry` = 1, `rs_cols` = 2, `qm_carry` = 3, `qm_cols` = 4, `sq_carry` = 5,
`sq_cols` = 6, and the correction selector = 1.

### 1d. What is *not* O(1), and it is a result

`mu` is a division. In a digit representation the reduction's serial state is an
element of `Z_N`; representing it in a small monoid alphabet **is** the
residue-indexed readout, which is closed by the coverage bound and doubly dead
on modulus-split Hard. So an exact O(1) reduction needs an S-digit reciprocal of
N, which is itself a division. **Modular reduction cannot be made O(1) end to
end in this representation** — what can be made O(1) is the *x*-dependent path,
by hoisting the division onto an N-only branch with **private** parameters.

Three reciprocals are provided:

| `--recip` | what it is | learned-op depth | status |
|---|---|---|---|
| `div` | learned long division of `10^{2S}` by N, private tables | `3 * 2S` (42 at S=7) | LEGAL, exactly constructible |
| `head` | one shallow learned map from N's digits | 2 | LEGAL, ceiling not certifiable |
| `oracle` | the true `mu` supplied | 0 | **DIAGNOSTIC** |

`oracle` is what the basin measurement uses: it puts the reduction tables at
their O(1) depth with no deep path contributing gradient at all, i.e. **the most
favourable conditioning this task allows**. If the basin does not widen there,
it never will.

---

## 2. Correctness gates — run before believing any number

`lab/test_o1reduce.py`, **ALL CHECKS PASSED** (`lab/runs/gate2.log`).

| check | result |
|---|---|
| Barrett `r ∈ [0,2N)` — hf1 ID moduli (16/18/20 bit) | max `floor(r/N)` = **1** |
| Barrett `r ∈ [0,2N)` — hf1 OOD-N moduli (17/19/21 bit) | max `floor(r/N)` = **1** |
| learned long division reproduces `floor(10^{2S}/N)` | **15/15** hf1-scale moduli, exact |
| constructed ceiling, S=3 / S=5 | **1.000** soft and argmax-hard |
| **constructed ceiling, hf1 ID moduli (S=7)** | **1.000** soft and argmax-hard (180/180) |
| **constructed ceiling, hf1 OOD-N moduli (S=7)** | **1.000** soft and argmax-hard (120/120) |
| same, under **bf16 autocast** (the manifests' dtype) | **1.000** soft and hard |
| composition at T=4 | **1.000** (117/117) |
| `reduce_op_depth` / `op_depth` at S=3 and S=7 | 5 / 7, constant |

**Two bugs the gate caught before they became results**, both recorded because
they are the sort of thing that silently invalidates a branch:

1. A 4-digit modulus at S=3 breaks the Barrett bound (`max floor(r/N)` = **4**,
   not 1) — `p = x²` must fit in `2S` digits. My first modulus list mixed 3- and
   4-digit moduli at S=3 and the "architecture" read 277/399.
2. `monoid.int_to_digits` writes one GPU element per digit inside a Python loop.
   On a shared GPU that cost **144 s** for input encoding where the model itself
   takes 0.5 s, and it looked exactly like a model-side hang.

DIAGNOSTIC: `--construct` writes the true tables (BRIEF §4 rule 2). It certifies
that any training failure below is an optimisation result, not an expressivity
one. It appears nowhere in the submission.

---

## 3. The instrument, and three fixes it needed

Protocol is `plan2/matrix-scan` §5 verbatim: construct the exact solution
(DIAGNOSTIC start point), randomise `k` table cells, train on the **plain
end-to-end label CE only** (LEGAL objective, no laws, no teacher forcing) for
2,000 steps at AdamW lr 3e-2, N=323, 250 train / 38 held operands, 3 seeds —
so the rows compose with that branch's.

Three corrections were forced by the first cells, and each would have produced a
wrong number:

1. **A `ColSum` cell encodes a value, so it decodes as `argmax(W[row] + b)`.**
   Reading `argmax W[row]` alone sends every positive row to the top of the
   range and every negative row to the bottom.
2. **Repairs are counted against the post-corruption state.** A randomised
   `ColSum` row decodes to 0, and about half of the true `hi` rows *are* 0, so
   the naive count reports repairs the objective never made. At k=20 the first
   cell read `naive 4/20` against `repaired 0/16`. Both are reported.
3. **The basin is measured on the ±BIG tables only** (`carry_trans`,
   `carry_emit`, the correction selector). Those are ±12 logit tables of exactly
   the kind `alu-relational` (400 cells) and `matrix-scan` (700 cells) measured.
   `ColSum` rows are not: the constructed quadratic reaches ~3e4, so a corrupted
   row sits ~1e5 away from the truth and **no learning rate that trains the rest
   of the model can walk back to it in 2,000 steps**. Mixing the two would
   measure parameter scale, not conditioning. The corruption pool is 452 cells
   at S=3.

### 3a. The instrument is not blind — POSITIVE CONTROL

Every basin cell below reads **0**, so the instrument has to be shown capable of
reading a repair. `lab/test_o1reduce.py::test_basin_instrument`: corrupt 40
cells, hand **19** of them their true values back, and `repaired()` reports
exactly **19 of 40**, resolved correctly by depth (`d1: 7/15, d3: 8/17,
d5: 4/8`), and **0 of 40** before the hand-back. **PASS.** A zero below is a
real zero.

### 3b. How much of the table surface the data can reach

A cell no example exercises receives no gradient and cannot be repaired at any
conditioning, so the basin has to be read against this
(`lab/usage_report.py`, `lab/runs/usage.log`):

| table (rows) | S=3, N=323, 250 operands | hf1 S=7, 24 moduli x 100 operands |
|---|---|---|
| `sq_carry` | 34/52 (**0.65**) | 64/120 (**0.53**) |
| `qm_carry` | 31/103 (**0.30**) | 85/239 (**0.36**) |
| `rs_carry` | 32/70 (**0.46**) | 62/138 (**0.45**) |
| `sel` | 2/2 (1.00) | 2/2 (1.00) |

So an exercised-only denominator is roughly 40% of the quoted one. **It does not
change any conclusion here, because every numerator is zero** — but it is the
right caveat to carry, and it is a real cost of the wide column-total alphabet
that buys the O(1) depth. `probe_o1.py` reports the exercised-only count
(`live=`) for every run made after this was added.

---

## 4. THE HEADLINE — repair basin at O(1) depth

**LEGAL objective, DIAGNOSTIC start point** (constructed then corrupted) and a
DIAGNOSTIC reciprocal (`--recip oracle`, so the reduction tables sit at their
O(1) depth with nothing deep contributing gradient — the most favourable
conditioning this task allows). N=323, 250 train / 38 held, 2,000 steps,
AdamW lr 3e-2, 3 seeds, corruption pool 452 cells.

### 4a. Against the scaling the mandate was extrapolating

| architecture | learned-op depth | k=5 | k=20 | k=400 |
|---|---|---|---|---|
| `DigitALU` (matrix-scan) | 39 | — | **0/5** | — |
| `MonoidALU` (matrix-scan) | 12 | 1/0/0 | **0/1/1** | **14/19/9 of 400** |
| **`O1ReduceALU` (this branch)** | **7 (reduction 5)** | **0/0/0** | **0/0/0** | **0 of 400, 3/3 seeds** |

### 4b. Measured — LEGAL objective, 3 seeds, N=323

| k corrupted (of 452) | **repaired** (pooled, 3 seeds) | `tbl` step 0 → 2000 | `train_exact_hard` | `held_exact_hard` |
|---|---|---|---|---|
| 5 | **0 / 15** | 0.995 → 0.980-0.995 | 0.612-0.936 | 0.342-0.947 |
| 20 | **0 / 60** | 0.981 → 0.956-0.967 | 0.224-0.504 | 0.053-0.263 |
| 50 | **1 / 150** | 0.953 → 0.918-0.931 | 0.016-0.192 | 0.000-0.079 |
| 100 | **0 / 300** | 0.907 → 0.871-0.877 | 0.008-0.100 | 0.000-0.053 |
| 400 | **0 / 1200** | 0.627 → 0.610-0.620 | 0.000-0.004 | 0.000 |
| 4 per table (balanced) | **0 / 77** | 0.977 → 0.952-0.955 | 0.376-0.472 | 0.210-0.395 |

**One repair in 1,802 corrupted cells**, and `tbl` falls in every single row.
Against `MonoidALU` at depth 12: **14/400 (3.5%)** at the matched k. Against
`DigitALU` at depth 39: 0/5. **Cutting learned-op depth 12 → 7 took the basin
from 3.5% back to 0.0%.**

The balanced (`per_table`) read is the cleanest, because it gives each depth a
comparable number of cells rather than letting the largest table dominate:

| learned-op depth from the loss | corrupted | repaired |
|---|---|---|
| 1 (`rs_carry`, correction selector) | 29 | **0** |
| 3 (`qm_carry`) | 24 | **0** |
| 5 (`sq_carry`) | 24 | **0** |

**Zero at depth 1.** The instrument is validated (§3a) and the cells are
exercised (§3b), so this is a real zero.

### 4c. The finding: conditioning behaves exactly as the law says, with the wrong sign

The depth-resolved read is the point of this branch, and it is unambiguous. The
tables are **private per stage**, so each sits at one well-defined learned-op
depth from the loss. Cell correctness, step 0 → 2,000 (`lab/runs/basin.jsonl`):

| table | learned-op depth | k=20 s0 | k=20 s2 | k=50 s0 |
|---|---|---|---|---|
| `sq_carry_e` (the squaring's digit emission) | **5** | 1.000 → 0.981 | 0.962 → **0.962** | 0.904 → **0.904** |
| `qm_carry_e` (the quotient's digit emission) | **3** | 0.971 → **0.971** | 0.971 → **0.971** | 0.913 → **0.913** |
| `rs_carry_e` (the residual's digit emission) | **1** | 0.957 → **0.657** | 0.971 → **0.629** | 0.857 → **0.443** |

**At depth 3 and 5 the tables do not move at all. At depth 1 the table moves a
long way — and every step of it is in the wrong direction.** Zero corrupted
cells are repaired anywhere; 20-40% of the *correct* cells of the depth-1 table
are destroyed.

The accompanying metric row says the same thing twice:

| cell | `train_exact` (soft) | `train_exact_hard` | `held_exact_hard` |
|---|---|---|---|
| k=5 s1 | 0.648 → **0.736** | 0.648 → **0.612** | 0.579 → **0.342** |
| k=20 s0 | 0.232 → **0.460** | 0.232 → **0.224** | 0.132 → **0.053** |
| k=20 s2 | 0.516 → **0.644** | 0.552 → **0.452** | 0.500 → **0.184** |
| k=50 s0 | 0.056 → **0.320** | 0.060 → **0.016** | 0.053 → **0.000** |

**The soft metric rises by up to 5.7x while the argmax-hard metric falls and the
depth-1 table is dismantled.** That is `alu-optimizer`'s "training moves *away*
from the discrete solution" — but now *localised*: it is not diffuse, it happens
in the one table the gradient actually reaches, and it happens because the
gradient reaches it.

**k=0 — the control that makes the claim precise.** Start at the exact solution
with *nothing* corrupted and train the same legal label for 2,000 steps
(`--from-construct`, `lab/runs/k0.log`): loss **0.00000**, `train_exact` /
`held_exact` / `train_exact_hard` / `held_exact_hard` / `tbl` all **1.000** at
every checkpoint, 3 seeds. **The exact solution is a stable fixed point of the
legal objective.**

So the correct statement is *not* "the objective destroys correct tables". It is
sharper than that:

> The solution is stable when it is exact. The moment **any** cell is wrong the
> loss is non-zero, and the gradient that non-zero loss produces repairs none of
> the wrong cells and breaks correct ones — and it does so **entirely in the
> table one learned op from the loss**, because that is the only table it
> reaches.

**The one alternative explanation, and it is ruled out.** The carry-state index
is a *gauge*: permuting carry states consistently in `trans` and `emit` leaves
the function unchanged while breaking an argmax match against the reference.
Two things rule that out here. (i) A gauge move would break `carry_t` and
`carry_e` together; instead `rs_carry_t` moves 0.91 → 0.90 while `rs_carry_e`
moves 0.957 → 0.657. (ii) A gauge move is function-preserving, and
`train_exact_hard` and `held_exact_hard` both **fall** (to 0.016 and 0.000 at
k=50). The degradation is functional.

**So the conditioning law's coefficient is not too small. Its sign is negative.**
Reducing learned-op depth increases the gradient's reach, exactly as measured;
the extra reach is spent trading discrete correctness for soft mixture. Putting
the objective one op from the parameters does not make the label repair the
solution; it makes the label break it faster.

### 4d. Why `MonoidALU`'s 14/400 is not the middle of a trend

The three-row table this branch was sent to extrapolate is not a one-parameter
family, and the depth-resolved data above shows why:

* the **bottom** row (50/400 at depth 1) is a *different objective* — an
  algebraic law evaluated directly on an adder table, not the end-of-chain
  label. This branch measures the **label** at depth 1 and gets **0**;
* the **middle** row (`MonoidALU`, 14/400) has its `add`/`sub`/`cmp` tables
  **shared across all `S` reduce steps**, so "depth 12" is a *maximum*, not a
  uniform depth: the same table is also used one op from the loss. Its non-zero
  repair is therefore attributable to its shallowest uses, not to the depth of
  its deepest one — which is exactly what this branch measures directly by
  making the tables private.

The correct statement of the law after this branch is: **learned-op depth
predicts how much a table moves, and predicts nothing about which direction it
moves.**

*(The coordinator reached the first half of this independently from
`hard/add-only`: the 50/400 row belongs to an O(1) **algebraic objective**
evaluated next to a table, not to the label, and the number does not carry
across a change of architecture. The two derivations agree.)*

### 4e. SEPARABILITY is the variable, and it shows up in the one cell that repaired

`hard/add-only` measured that the label's exact-repair radius on an *arithmetic*
table is 5-10 cells, while a 10-cell **separable** selector is recovered exactly
every seed. This branch reproduces that on a different architecture — and it
was already visible before I went looking.

**Of the 1,802 corrupted cells in §4b, exactly one was repaired, and it is the
correction selector** (`o1-k50-s1`: `sel` 0.50 → **1.00**, while in the same run
`rs_carry_e` fell 0.90 → 0.53, `rs_carry_t` 0.914 → 0.829 and both `rs_cols`
tables fell). The single repair in the whole grid is the one separable table in
the architecture.

The controlled version, at **matched learned-op depth 1**, matched protocol,
10 seeds each (`lab/runs/sep.log`):

| corrupted table | separable? | depth | **repaired** | exercised-only |
|---|---|---|---|---|
| `sel` (Barrett correction selector) | **yes** — each row's answer is fixed by the label independently | 1 | **9 / 19** | **9 / 19** |
| `rs_carry_t` + `rs_carry_e` (carry/borrow + digit emission) | **no** — rows must agree across every carry state and interact through the carry chain | 1 | **0 / 18** | **0 / 8** |

Same depth, same objective, same optimiser, same number of steps: **9/19 versus
0/18.** In the `sel` arm the loss returns to 0.00000 and `held_exact_hard`
returns to 0.921-1.000 — a full recovery. In the arithmetic arm the loss also
returns to 0 and the model reaches `held_exact_hard` 0.974-1.000 **without
repairing the cell** — it routes around it instead.

**One qualification that matters, and it limits how far this can be pushed.**
The selector is repaired when it is *the only thing wrong*. In the balanced
`per_table` runs, where every table is corrupted at once, `sel` goes
0.50 → **0.00** (seed 0) or stays at 0.00 — it is not repaired. Separability
buys repair only when the residual error is attributable to the separable
parameter; it does not survive competing error elsewhere. So separability is a
real axis and it is **not** a route to training the whole transducer from random
init, where everything is wrong at once.

*(First attempt kept at `lab/runs/sep_k1.log`: at 1 corrupted row per arithmetic
table the exercised-only denominator came out 0 — a random carry row is
exercised only ~46% of the time (§3b) — so the arm was rerun at 4 rows. Recorded
because the first version would have compared 0/0 against 9/19 and looked like
the same result for the wrong reason.)*

---

## 5. Legal training from random init — **LEGAL**, and calibrated

`lab/runs/fit.log`, `lab/runs/hf1long.log`. Two scales: N=323 (where the class
provably converges, so the null is not a step-count artefact) and an `hf1`-shaped
synthetic set — `--bits-list 16,18,20`, 8 train + 4 held moduli per size,
**disjoint train/held modulus pools**, 6,000 train / 4,608 held operands, so
parameters must be modulus-independent exactly as on `hf1`.

### 5a. N=323 — fully converged, and this is the load-bearing null

| step | loss | `train_exact` | `held_exact` | **`train_exact_hard`** | **`held_exact_hard`** | `tbl` |
|---|---|---|---|---|---|---|
| 0 | — | 0.008 | 0.053 | 0.000 | 0.000 | 0.021 |
| 1000 | 0.040 | 0.976 | 0.000 | 0.028 | 0.000 | 0.026 |
| 2000 | 0.00048 | **1.000** | 0.000 | 0.032 | 0.000 | 0.023 |
| 3000 | 0.00006 | 1.000 | 0.000 | 0.008 | 0.026 | 0.020 |
| 8000 | 0.00000 | 0.996 | 0.026 | 0.044 | 0.000 | **0.019** |
| **`--lr 0` (CONTROL)** | 2.299 | 0.008 | **0.053** | 0.000 | 0.000 | 0.021 |

**`train_exact` reaches 1.000 and the loss reaches 0.00000 — there is no
pre-fitting excuse available — while `train_exact_hard` never exceeds 0.044 and
the tables sit at chance (0.019-0.026 against 0.021 at init).** Fully legal, no
oracle in the loss.

Two rows deserve their own line. **`tbl` ends *below* its own initialisation**
(0.019 vs 0.021): 8,000 steps of the legal objective moved the tables backwards,
which is `alu-optimizer`'s "regression toward init" with the sign made explicit.
And **`held_exact` at `lr = 0` is 0.053 (2/38), higher than any trained
checkpoint reaches** — `matrix-scan`'s inversion, reproduced on this
architecture at O(1) depth.

The same run with the reciprocal learned too (`--recip div`, nothing oracular
anywhere) is identical: `train_exact` 0.988 → 0.996, `train_exact_hard`
0.028-0.044, `held_exact_hard` 0.000, `tbl` **0.048 → 0.019**.

### 5b. `hf1` shape and scale

| step | loss | `train_exact` | `held_exact` | **`train_exact_hard`** | **`held_exact_hard`** | `tbl` | `held_div` |
|---|---|---|---|---|---|---|---|
| 0 | — | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.996 |
| 2000 | 1.031 | 0.047 | 0.000 | 0.000 | 0.000 | 0.004 | 0.990 |
| 8000 | 0.481 | **0.266** | 0.000 | 0.000 | 0.000 | 0.005 | 0.997 |
| **`--lr 0` (CONTROL)** | 2.303 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.996 |

**Fitting has demonstrably begun** — `train_exact` 0.000 → 0.266 and loss
2.30 → 0.48, against a frozen `lr = 0` control — so this is past the onset the
pre-fitting rule is about, though not converged (see §5c). **`held_exact_hard`
is 0.000 at every checkpoint**, and `tbl` does not leave chance.

**Best `train_exact_hard` / `held_exact_hard` on `hf1` from random init: 0.000 /
0.000**, against `--lr 0` at 0.000 / 0.000.

**Collapse detector, against a MEASURED reference** (BRIEF2 §6.5 — the exact
solution's diversity is not 1.0): running the *constructed* solution through the
same detector gives `held_div` **0.816** at N=323 (38 held operands) and
**0.988** at hf1 scale (4,608 held operands). The trained models read 0.79-0.87
and 0.985-0.997 respectively. **Diverse and wrong, not collapsed** — so the null
is a genuine null, not a constant map.

### 5c. Scope, stated plainly

`hard/curriculum-hf1` measured that a reference-width model has not left the
pre-fitting region on the *real* `hf1` (243k rows) at **40,000** steps. My
`hf1`-shaped set is 6,000 rows, which is why fitting starts within 8,000 steps
here; it is a *shape* replica (modulus-split, three sizes, S=6-7,
modulus-independent parameters), not a size replica, and fewer rows make
memorisation easier. The converged claim rests on §5a; §5b establishes that the
same behaviour appears at `hf1`'s split structure and modulus sizes.

## 6. Evaluator — **LEGAL**, and read as uncalibrated

`--mode fixed_step` so the clock never binds and the rows are contention-immune
(six jobs shared this GPU). `submissions/hard-o1-reduction/` (learned
long-division reciprocal, the variant whose ceiling is certified) and
`-head` (shallow reciprocal). Each `-lr0` twin is byte-identical except
`lr 1e-2 → 0.0`.

| run | manifest | MAX_T | OOD_N | rung-1 | mean | steps | train s |
|---|---|---|---|---|---|---|---|
| `hf1-o1-div400` | `lab_hf1_fs400_s74` | **0** | 0 | 0.000 | 0.0001 | 400 | 179 |
| `hf1-o1-head` | `lab_hf1_fs2000_s74` | **0** | 0 | 0.000 | 0.0001 | 2000 | 270 |
| `hf1-o1-head-lr0` (**CONTROL**) | `lab_hf1_fs2000_s74` | **0** | 0 | 0.000 | 0.0000 | 2000 | 139 |
| `smoke_o1` (plumbing) | `smoke_hf1s` | 0 | 0 | 0.000 | 0.0000 | 30 | 72 |

Trained and `lr=0` are **indistinguishable** — mean 0.0001 vs 0.0000, every rung
0.000, the two non-zero cells anywhere are one example out of ~1,000. The
submission lints and scores end to end through the real runner on both `hf1` and
`hf1s`, so the pipeline is sound.

**These rows are NOT a null and I am not reporting them as one.** Per
`hard/curriculum-hf1`, a reference-width model does not leave the pre-fitting
region on `hf1` at **40,000** steps (243k train rows, 768-example rungs);
400-2,000 steps is far inside it. They establish compliance, plumbing and cost,
nothing about learnability. The learnability claims in this report rest on the
basin measurement (§4), which starts *at* the solution and therefore needs no
fitting-curve calibration at all, and on §5.

**Cost, at hf1's shape (batch 128, S=7), as a ratio to the `-head` variant:**
the learned long-division reciprocal costs **~3.4x** per step (0.45 vs
0.135 s/step) because its 14 quotient positions are 14 sequential
`AccStage` applications over 10 candidates each — kernel-launch bound, exactly
`plan2/sequential-rnn`'s finding. Two 2,000-step `div` runs **exceeded the lab
harness's 1,200 s timeout** and were rerun shorter. This is the price of the one
part of the architecture that is *not* O(1), and it is another argument for
isolating the division onto an N-only branch rather than leaving it inline.

---

## 7. Compliance ruling

The mandate asked me to rule against my own method if the O(1) reduction amounts
to *writing* the reduction algorithm rather than learning it. I have gone
through it line by line and the ruling is **LEGAL, by exactly the standard this
project has already applied to `DigitALU` and `MonoidALU`** — with two calls
flagged, because a reader should be able to disagree with me on the evidence
rather than on my say-so.

**Learned from random init, in the submission** — the column-value tables
(`ColSum.Wp/Wd/b`: which value a digit *pair* contributes to a column, i.e. the
entire multiplication table), the carry monoids (`CarryMonoid.trans/emit`: carry
and borrow propagation and the output digit, i.e. the entire adder), the Barrett
correction selector, the reciprocal's long-division tables and quotient-digit
selector, the marker pointer, the digit readout and the depth selector.

**Fixed, and why each is structure rather than arithmetic:**

| fixed thing | ruling |
|---|---|
| `pair_plan` — the low half of `a_i * b_j` lands in column `i+j`, the high half in `i+j+1` | **Legal, and settled precedent.** `MonoidALU.multiply` uses `shift_up(lo[:,i], i)` / `shift_up(hi[:,i], i+1)` for the same thing and `matrix-scan` §8 rules it "re-indexing (a shift by a digit position is a slice)". `DigitALU` is identical. |
| `pair_bag` — pooling one-hot digit-pair *indicators* per column | **Legal.** The bag counts *which pairs occur*; what each is worth is `Wp`, which is learned and can be learned wrong. It does bake in commutativity and associativity of the accumulation — a generic algebraic property, which BRIEF §4.2 permits as a structural choice ("choosing an architecture whose structure suits modular arithmetic is allowed and encouraged"). |
| Barrett's truncations (`q2[:, 2S:]`, the `q3` slice) | **Legal.** A slice is division by a power of ten, the inverse of the accepted `shift_up`. |
| the Barrett *schedule* itself | **Legal, and this is the sharpest call.** It is an algorithm skeleton in the forward pass with every operation learned — which is precisely what `MonoidALU.reduce_mod` is (`for k in S-1..0: compare against shifted multiples, select a quotient digit, subtract`) and what `DigitALU` is. If Barrett is illegal here then so is long division there, and so is every result this project has. |
| `10^{2S}` as the dividend fed to `LongDivRecip` | **Legal but flagged.** It is a fixed *input constant* chosen because it is the right Barrett scaling for base 10 at `S` slots — analogous to `MonoidALU.multiples` enumerating the ten one-hot digits as an input, and to initialising a carry to zero. It is the closest thing in this branch to a number I chose because I knew the algorithm. |
| `n_corr = 2` | **Legal, and chosen for compliance.** With two correction candidates the extra subtrahend is a 0/1 *gate* on N's digit bag. `n_corr = 3` would need a `×2`, which is arithmetic I would have written; the gate in §2 verifies 2 is enough, so I never needed it. |

**Where I would rule against myself if I had done it, and did not:** giving
`ColSum` a fixed `φ(m) = m` feature so the learned part is a per-pair *scalar
value* and the "sum of values → one-hot of the sum" decoder is fixed. That
would have solved the parameter-scale problem in §3.3 at a stroke, and it is
writing the adder. It is not in this branch.

**DIAGNOSTIC-only, never in a submission:** `construct_`, `corrupt_`,
`--recip oracle`, and the `true_mu` helper. Every row in this report is labelled
LEGAL or DIAGNOSTIC. Nothing under `data/generated/` was read, printed or
summarised — probe operands are synthesised in-process by locally-written
Miller-Rabin prime sampling, and the only dataset access is the evaluator's own
through `lab/run_experiment.py`. Nothing was submitted to the hosted service.
The submission is one file, no custom training loop, no participant-controlled
backward, `(logits, None)` with the default cross-entropy.

---

## 8. Verdict
