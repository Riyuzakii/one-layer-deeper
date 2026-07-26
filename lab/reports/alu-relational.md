# alu-relational — legal training signals with intra-squaring content

**Branch** `explore/alu-relational`, from `explore/alu-credit`.
**Mandate** find a *legal* training signal with intra-squaring content for
`DigitALU`'s discrete tables. Three families: (1) relational constraints on the
learned map at self-generated operands, (2) dual-path agreement, (3) re-screen
the algebraic regularisers under Stage-1 conditions.
**Tooling** `lab/probe_rel.py` (new), `lab/rel_sweep.sh`, `lab/jobs_*.txt`,
runs in `lab/rel_runs.jsonl`, per-run logs in `lab/logs/`.

Every table below is labelled **LEGAL** or **DIAGNOSTIC**, as `alu-credit` §0C
does. Nothing under `data/generated/` was read; all operands are generated from
`math.gcd` over `range(1, N)`.

---

## 0. Verdict

*(filled in at the end — see §9)*

---

## 1. Conditions, and the correctness gate

All runs: **m1 scale** (N = 10403, S = 5, 10,200 units, 8,000 training operands,
1,024 held-out), `alu-depth`'s **`tree:quotient`** graph (39 sequential soft
steps), **untied** (the coordinator's mid-branch correction: ties help *repair*
from a near-solution and hurt *learning* from random init), AdamW lr 3e-2,
betas (0.9, 0.95), batch 512, grad-clip 1.0.

`probe_rel.py` adds to `alu-depth`'s `DigitALU`, using **the same tables and no
new arithmetic**:

* a general two-operand modular multiply `mulmod(s, t)` — the diagonal `s == t`
  is exactly the task, so every table it uses is anchored by the task loss;
* a modular add `addmod(s, t)` — `add_scan` then one learned `quot_reduce`;
* three alternate graph shapes for the same squaring: `fold` (leaf sum by a
  chained left fold instead of a balanced tree), `redall` (reduce after every
  product digit instead of only the last S+1), `horner` (the 257-step Horner
  graph).

**Correctness gate** (`--check-law`, DIAGNOSTIC — uses `--construct`):

```
CONSTRUCTED tree   soft=1.000 hard=1.000 held_hard=1.000
CONSTRUCTED fold   soft=1.000 hard=1.000
CONSTRUCTED redall soft=1.000 hard=1.000
CONSTRUCTED horner soft=1.000 hard=1.000
LAW CHECK (constructed, true constants): lhs==rhs exactly on 1.000 of 512 operands; sym_ce=0.0000
DUAL CHECK tree vs fold  : agree=1.000 sym_ce=0.0000
DUAL CHECK tree vs redall: agree=1.000 sym_ce=0.0000
DUAL CHECK tree vs horner: agree=1.000 sym_ce=0.0000
```

So the target is in the class for every path, and the relational law is exactly
satisfied at the solution. Any failure below is a failure of the *signal*, not
of the implementation.

## 2. The control that was missing: the LEGAL baseline at Stage-1 conditions

`alu-credit` reported the *teacher-forced* number at these conditions (0.951)
and the target-propagation number (0.000), but never the plain legal baseline.
It is a hard zero.

**LEGAL**

| run | `train_exact` | **`train_exact_hard`** | **`held_exact_hard`** | `local_ce` | `add_shift` | `sub_shift` |
|---|---|---|---|---|---|---|
| baseline seed 0 | 0.000 | **0.000** | 0.000 | 3.495 | 0.270 | 0.300 |
| baseline seed 1 | 0.001 | **0.000** | 0.000 | 3.674 | 0.315 | 0.362 |
| baseline seed 2 | 0.000 | **0.000** | 0.000 | 4.177 | 0.280 | 0.237 |
| tied seed 0 / 1 / 2 | 0.000 / 0.000 / 0.001 | 0.000 | 0.000 | 3.26 / 3.12 / 4.03 | ~0.26 | ~0.26 |

Random-init reference for the structure scores is 0.23–0.28, so the tables are
at chance. `local_ce` 3.5–4.2 against a cliff at **0.005–0.0073**: the baseline
is ~500× the per-op error rate that separates 0.951 from 0.000. Ties change
nothing at this end either.

Two conventions used throughout:

* **`out_div`** — the fraction of *distinct* predicted answers over 512
  operands with states snapped. The truth reads 1.000; a map collapsed to a
  constant reads ~0.002. This is a sharper collapse detector than `mul_gauge`
  because it looks at the composed map, not one table. The plain baseline
  oscillates in 0.1–0.9 (measured control, `t_basediv`), so a *low* `out_div`
  is only meaningful against that band.
* **`local_ce`** is now logged during training, not only at the end.

## 3. Family 1 — the relational increment law. Closed by its own illegal ceiling.

### 3.1 What was built

The template from my mandate: assert that the learned map has an
additive-increment structure — there exist a learned unary `inc`, a learned
composition `⊕` and a learned increment `D` with

```
f(inc(x)) = f(x) ⊕ D(x)      for every operand the model sees
```

Everything on the right is the model's *own* machinery: `⊕` is `addmod` (the
learned `Tadd` scan plus the learned quotient reduction), `inc(x)` is
`addmod(x, c)` for a learned register `c`, and in the `affine` form
`D(x) = addmod(mulmod(a, x), b)` for learned registers `a, b`. The diagonal of
`mulmod` *is* the task, so every table in the law is anchored by the task loss —
which is the condition `explore/depth-controller` identified as making a
consistency law non-vacuous.

Three forms of `D`, in increasing specificity:

| form | what it asserts | status |
|---|---|---|
| `free` | *some* additive-increment structure exists (`D` a free MLP) | **LEGAL** — the conservative variant |
| `affine` | the increment is affine in `x` under the model's own ops, i.e. the map is quadratic | **LEGAL-BUT-FLAGGED** — see §7 |
| `true` | `c, a, b` pinned to the digits of 1, 2, 1 — literally `(x+1)² = x² + 2x + 1` | **ILLEGAL / DIAGNOSTIC** (rules 2, 7) |

### 3.2 The ceiling was measured first, and it is the floor

`alu-credit` established the right methodology here: measure the *ideal illegal*
version of a signal before spending budget building a legal one. The legal
forms of this law have a strictly smaller hypothesis space for the constants
than `true` (which is handed them) and strictly more ways to be satisfied
trivially, so `true` upper-bounds the intended mechanism.

**DIAGNOSTIC — the true identity written into the loss. 2,000 steps, m1 scale.**

| run | `train_exact` | **`train_exact_hard`** | **`held_exact_hard`** | `local_ce` | `out_div` | `mul_gauge` |
|---|---|---|---|---|---|---|
| `--rel true` seed 0 | 0.000 | **0.000** | 0.000 | 3.329 | 0.203 | 0.50 |
| `--rel true` seed 1 | 0.000 | **0.000** | 0.000 | 3.222 | 0.426 | 0.50 |
| `--rel true` seed 2 | 0.001 | **0.000** | 0.000 | 3.702 | 0.566 | 0.50 |
| `--rel true` weight 3.0 | 0.000 | **0.000** | 0.000 | 3.355 | 0.418 | 0.50 |
| *baseline, for comparison* | 0.000 | 0.000 | 0.000 | 3.50–4.18 | 0.10–0.90 | 0.4–0.8 |

Every cell is inside the baseline band on every metric, including `local_ce` —
the sensitive one. The relational law is *satisfied at the solution* (§1 gate,
`sym_ce` exactly 0.0000) and *supplies nothing on the way there*, even when it
is handed the true coefficients.

**This closes the family.** No version with learned `inc`, `⊕` and `D` can beat
the version that is given them.

### 3.3 The legal forms, measured for the record

Run anyway, because the mandate asks for the flagged variant and the
conservative variant both to be reported.


## 7. Compliance — where I put the line, and why

My mandate asked me to flag the judgement call explicitly, keep a conservative
variant that omits it, and report both. `discrete-search` §9 had already ruled
on an adjacent case ("algebraic constraints strong enough to pin the tables …
**No, in my judgement**"). I adopt a line consistent with both, and state it as
a rule rather than case by case:

> A loss term is **legal** if it asserts a *generic algebraic property of an
> operation the model already performs* — a property shared by a large class of
> operations, which therefore does not single out base-10 arithmetic. It is
> **not legal** if it asserts *the specific relationship between two operations
> that constitutes the definition of the arithmetic*, because that is rule 2
> (no hard-coded algorithm) moved from the forward pass into the loss.

Applying it:

| term | what it asserts | verdict |
|---|---|---|
| `--sym` / `--tie sym` | the learned add and multiply are commutative | **LEGAL.** Commutativity holds for a huge class of operations. This is `discrete-search`'s own judgement on the tied form ("legal, cleanly"), and the soft-penalty form has the same content. |
| `--inv` / `--tie-sub` | the learned subtract inverts the learned add | **LEGAL.** Invertibility is a generic property; it says nothing about base 10. Again matches `discrete-search`. |
| `--assoc` | the learned multi-digit adder is associative | **LEGAL.** Same category. Note it is *not* sufficient to determine base-10 addition: the constant map, the projection, and digit-wise addition mod 10 with no carry are all associative. That it fails to pin the adder is evidence *for* its legality, not against. |
| `--cancel` | `B ↦ A ⊕ B` is injective | **LEGAL.** A generic non-degeneracy law (it is what upgrades a commutative semigroup to an abelian group). It supplies no values. Its only content is "the adder is not constant", which is a statement about degeneracy, not about arithmetic. |
| `--dual fold` / `redall` / `horner` | one set of tables computes one function regardless of the order it is composed in | **LEGAL, and the cleanest in this report.** No new parameters, no constants, no assertion about arithmetic at all — just that the model is a function. |
| `--rel free` | *some* additive-increment structure exists, with `inc`, `⊕` learned and `D` a free learned map | **LEGAL — the conservative variant.** It asserts nothing about the increment's form. Its weakness is exactly its legality: `D` free makes the law satisfiable by any `f` whatsoever, so it is close to vacuous by construction. |
| `--rel affine` | the increment is affine in `x` under the model's own operations, i.e. `f` is a quadratic | **LEGAL-BUT-FLAGGED, and I would not ship it.** No coefficient is supplied and every operator is learned from random init, so it is not "the arithmetic in the loss" in the way `--rel true` is. But "degree 2" is a fact about *this task*, not a generic property of an operation, and it is read off the generator source in a way that goes beyond BRIEF §2's licence to *choose an architecture*. Under my own rule above it is on the wrong side of the line: it asserts a relationship that is specific to squaring. **I flag it, I do not build a submission on it, and I report it separately from the conservative variant.** |
| `--rel true` | `c, a, b` = the digits of 1, 2, 1 | **ILLEGAL** (rules 2 and 7). Diagnostic only, run to measure the family's ceiling. |
| mul-from-repeated-add (`Tmul(a, b⊕1) = Tmul(a,b) ⊕ a`) | multiplication is iterated addition | **NOT LEGAL, and I did not run it.** This is the definition of multiplication. `discrete-search` §9 reached the same conclusion independently; I agree and record the agreement. It matters because §8 argues it is the *only* thing that would identify `Tmul`. |

**Everything I would recommend building on is in the top six rows**, all of
which measured null. The one term I flagged (`--rel affine`) is also null, so
the judgement call cost nothing.
