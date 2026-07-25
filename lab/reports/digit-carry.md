# digit-carry — a digit-compositional readout, and prompt→slot parsing

**Branch:** `explore/digit-carry` · **Two bottlenecks owned, two results, one new
bottleneck exposed.**

**The headline, in one table.** With the generator's 250 training `x`, the
held-out ceiling of a readout that is *an arbitrary function of the residue*
(`explore/group-rotation` §7) against the ceiling of a readout in which every
learned tensor is indexed by a *digit tuple*:

| dataset | modulus | **residue readout** | **digit readout** | digit-readout parameters |
|---|---|---|---|---|
| e1 | 323 | **1.000** | **1.000** | 6,817 |
| e2 | 899 | **0.614** | **1.000** | 6,817 |
| — | 2021 | **0.337** | **1.000** | 6,817 |
| m1 | 10403 | **0.031** | **1.000** | 6,817 |
| e5 regime | 12 *unseen* sampled 10/11-bit moduli | (needs a new table per `N`) | **1.000** (4800 operands) | 6,818, *one* vector for all 12 |

**The coverage ceiling does not apply to a digit-compositional readout, and the
escape is total, not marginal.** The parameter count is *identical at every
modulus* because no tensor has an index that ranges over `Z_N` — that is the
mechanism, stated exactly. The same 6,818 numbers are exact on twelve moduli
they were never fitted to, which is the OOD-N tie-break property obtained by
construction.

**But it is not learnable.** Trained from random init on the same 250 `x`, the
same architecture plateaus at train_exact 0.30 / held_exact 0.000. That is a
**fourth bottleneck** — *optimisability of the digit pipeline* — and it now gates
everything, because the other three are either solved or shown not to bind.

---

## 1. What I was given and what I did not re-derive

Settled by ~180 sibling runs and taken as given: the failure is a
memorisation/generalisation gap; eight representations, depth/iteration count,
on-manifold state, error compounding, capacity, data volume and the
Fourier/rotation route are falsified; iteration/depth in `T` is **solved** by a
weight-tied block plus an ordered depth selector. I ran nothing on those axes.

Two protocol corrections arrived mid-session from the coordinator and are
honoured throughout:

* **held-out CE is not used as a ranking signal anywhere in this report.** Every
  comparison is ranked on train-vs-held-out exact accuracy and the rung profile.
  (CE appears in raw logs only.)
* **e5 is the screening dataset**, e1 is reserved for the final attempt. All new
  evaluator runs are on e5.

## 2. Bottleneck (B): does a digit-compositional readout escape the coverage ceiling?

### 2.1 What "digit-compositional" has to mean for the bound not to apply

group-rotation §7's bound is about *what the learned tables are indexed by*. A
Fourier readout, a softmax over `Z_N`, an embedding table and a learned
permutation are all functions of the residue, so they are unconstrained on a
residue that never appeared in training, and the ceiling is exactly
`P[x² already seen]`.

To escape it, **no learned tensor may have an index that ranges over `Z_N`.**
That is a much stronger requirement than "the architecture looks compositional",
and most things that look compositional fail it. In particular a place-shared
partial-product table feeding an unconstrained slot network does *not* escape,
because the slot network's continuous state re-encodes the value (§2.4).

### 2.2 The architecture that does satisfy it (`lab/probe_alu.py`)

`DigitALU` computes `digits(x) → digits(x² mod N)` by a Horner recurrence over
the places of `x²`, with a shift, a scan order and a gate as the only structure:

```
lo_ij, hi_ij = Tmul[d_i, d_j]                     shared over all place pairs
r = zero
for k = K-1 .. 0:                                 K = 2S-1 places
    r = shift_up(r)                               structural, no parameters
    for (i,j) with i+j == k:  r = add_scan(r, [lo_ij, hi_ij])
    for _ in range(R):        r = cond_sub(r, digits(N))     R = 11, weight-tied
y = r[:S]

add_scan : slot scan LSB->MSB, Tadd[digit, addend, carry]  -> digit, carry
cond_sub : slot scan LSB->MSB, Tsub[digit, n_digit, borrow] -> digit, borrow,
           then a learned gate on the final borrow state chooses between the
           scanned result and the input
```

Every learned tensor is `Tmul (10,10,20)`, `Tadd (10,10,2,12)`,
`Tsub (10,10,2,12)`, a gate and three learned constants: **6,817 parameters,
every index a digit or a 2-state carry.** `N` enters only as *input digits*, so
the parameter count is independent of the modulus and of the slot count.

The `--construct` path (LAB DIAGNOSTIC, never in a submission — the exact
analogue of group-rotation's `probe_step.py --oracle`, which froze its pair
table at the true additive-character phases) sets those tables to the exact
solution and measures the resulting ceiling.

```bash
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
for cfg in "323 3" "899 3" "2021 4" "10403 5"; do set -- ${=cfg}
  $VENV lab/probe_alu.py --modulus $1 --slots $2 --construct
done
$VENV lab/probe_alu_oodn.py --moduli 6      # 12 unseen sampled moduli
```

Result: `train_exact = held_exact = 1.000`, `held_ce = 0.0000`, at **all four
moduli** and at **all twelve unseen sampled moduli** (4,800 operands). This is
the headline table above.

### 2.3 The closed-form counterpart (`lab/probe_coverage.py`)

§7 was believable because a pure combinatorial quantity predicted the measured
oracle exactly at three moduli. The same computation for a digit readout: over
the 250 training `x`, enumerate which *place-shared table entries* each `x`
exercises (pair products, add-with-carry, digit×multidigit, subtract-with-borrow,
comparator states), then ask what fraction of held-out `x` need only entries
already exercised. This is the strictest possible reading — a pure lookup
learner that cannot interpolate at all.

| N | residue: `P[x² already seen]` | digit: `P[every table entry already exercised]` |
|---|---|---|
| 323 | 1.000 | 0.947 |
| 899 | 0.614 | 0.966 |
| 2021 | 0.337 | 0.956 |
| 10403 | **0.072** | **0.977** |

Two things to read here. First, the digit quantity **rises with `N`** (0.947 →
0.977) where the residue quantity falls by 14×, because a bigger modulus means
more digit places per example and therefore *more* coverage of a fixed
alphabet. Second, 0.95–0.98 is a *lower* bound for anything that interpolates:
the alphabet is nearly saturated (e.g. at N=323, 110 of 111 needed add entries
and 100 of 100 pair entries are already seen), and the constructed model — which
does generalise across table entries — measures 1.000. Seeds 1 and 2 reproduce
the ordering (digit 0.79–0.99 vs residue 1.000/0.63/0.35/0.07).

**Consequence for the team's plan.** group-rotation's conclusion that "e1 is the
only public dataset where a readout can certify even T=1" is correct *for
value-indexed readouts only*. For a digit-compositional readout there is no such
restriction: e5, m1 and Hard are all reachable in principle, and the OOD-N
tie-break is reachable *with the same parameters*. The scale argument that made
Medium/Hard look hopeless is removed.

### 2.4 The negative that came with it: it does not train

`probe_alu.py` from random init, e1 modulus, same 250/38 split:

| variant | train_exact | held_exact |
|---|---|---|
| constructed (diagnostic ceiling) | **1.000** | **1.000** |
| learned, default | 0.196 (plateau by ~800 steps) | 0.000 |
| learned, freeze `zero` at truth | 0.296 | 0.000 |
| learned, freeze `zero`+`gate` | 0.016 | 0.000 |
| learned, freeze `zero`+`gate`+`sub` | 0.000 (loss 18.7) | 0.000 |
| learned, freeze all but `mul` | 0.000 (loss 15.6) | 0.000 |
| learned, straight-through discrete states | 0.012 | 0.000 |

**Freezing sub-modules at the truth makes it monotonically worse** — a correct
`Tsub` applied to a register that a random `Tmul`/`Tadd` has scrambled amplifies
the error rather than correcting it (loss 18.7 against `ln(10)=2.30`). The
modules are only individually useful once they are *jointly* nearly right, so
there is no partial-credit gradient path into the solution. Straight-through
discretisation, which closes the continuous side-channel, does not help either.

This is the same *shape* as group-rotation §5.5b ("the hypothesis class that
excludes memorisation also excludes anything the optimizer can descend into")
but with a crucial difference that changes what to do about it: **there, the
target was a measure-zero point in a flat continuous landscape and the class was
also not expressive enough to fit; here the exact solution is a discrete table
assignment that the class contains, provably reaches 1.000 at every modulus, and
is 6,817 parameters.** The obstruction is credit assignment through ~280
sequential soft table lookups, not identifiability and not capacity.

I also ran the intermediate design — place-shared pair table + a weight-tied
bidirectional slot scan with a continuous carry (`lab/probe_digit.py`,
`d_model` 8/16/32/64) — and it behaves like everything else in this repo: at
`d=32` it reaches train 1.000 / held 0.000 by step 2000, at `d≤16` it fails to
fit at all (train 0.06–0.42, held 0.000). **A continuous carry vector is a
value-encoding channel**: it is what separates this design from `DigitALU`, and
it is enough to restore memorisation. Compositional *structure* is not
sufficient; the state alphabet has to be small.

## 3. Bottleneck (A): prompt → place-valued digit slots

### 3.1 The fix

A digit's place value is defined by its offset from the end of **its own field**.
The prompt `[N] d(N) [X] d(x) [T] d(T)` is left aligned with variable-length
fields, so:

* absolute position cannot express place value (`x` starts at a `len(N)`-dependent
  offset, and under sampled `N` that offset varies row to row);
* distance from the **end of the prompt** expresses it only while the `T` field
  has a fixed digit count — it breaks on exactly the T=16/32/64 rungs.

The fix is to anchor each field on the marker that **terminates** it:
`d(N)` ends at `[X]`, `d(x)` ends at `[T]`, `d(T)` ends at the padding. **`x`'s
slots are anchored on the `[T]` marker position, which does not move when `T`
gains a digit.** That is the entire difference from `posmode=rev`.

`MarkerPointer` (`lab/probe_parse.py`, and in the submission) implements it as a
learned differentiable pointer, fully inside the autograd graph:

```
anchor_a[i] = softmax_i(<learned probe_a, embed(tok_i)>)      a in {N, X, T}
anchor_end  = last valid position of the attention mask
rel[a,i,o]  = anchor_a[i+o]                                   soft relative offset
attn[s,i]   = softmax_i( sum_ao rel[a,i,o] R[s,a,o] + sum_a cum_a[i] G[s,a] )
slot_s      = sum_i attn[s,i] * embed(tok_i)
```

`R` and `G` are learned tables over (slot, anchor, offset). No Python control
flow reads `input_ids`; no field boundary is computed with an integer index.

A field has **two** boundaries, and the offset table `R` only supplies the
closing one. Without the second term a slot above the field's actual width runs
off the front into the previous field — `x` has 1–3 digits but 3 slots, so a
1-digit `x` made slot 2 read the last digit of `N`. `G` scores the *cumulative*
anchor mass ("have we passed marker `a` yet"), which is the differentiable form
of the opening boundary, and makes the opening marker the fallback — a distinct
token, hence a usable leading-zero sentinel.

### 3.2 Verification at multi-digit T (the mandate's specific requirement)

`--construct-parse` sets the marker probes and `R`/`G` to their intended values
so the *mechanism* is measured separately from whether it is learnable. Metric:
fraction of prompts on which every one of a field's slots attends to the token
carrying that place's digit (and out-of-field slots attend to the opening
marker).

```bash
$VENV lab/probe_parse.py --front marker --construct-parse                # e1
$VENV lab/probe_parse.py --front marker --construct-parse --sampled-n    # e5 regime
$VENV lab/probe_parse.py --front rev    --construct-parse --sampled-n    # control
```

| T | len(T) | marker: x / N / T | distance-from-end: x / N / T |
|---|---|---|---|
| 1, 2, 3, 4, 8 | 1 | **1.000 / 1.000 / 1.000** | 1.000 / 0.000 / 1.000 |
| **16, 32, 64** | **2** | **1.000 / 1.000 / 1.000** | **0.000** / 0.000 / 1.000 |

and in the e5 regime (sampled 10/11-bit moduli, so `len(N)` varies too):

| T | len(T) | marker: x / N / T | distance-from-end: x / N / T |
|---|---|---|---|
| 1, 2, 3, 4, 8 | 1 | **1.000 / 1.000 / 1.000** | 0.864 / 0.000 / 0.000 |
| **16, 32, 64** | **2** | **1.000 / 1.000 / 1.000** | **0.000 / 0.000 / 0.000** |

**Parsing is solved.** The marker-relative scheme is exact at every rung of the
ladder and under variable-length `N`; the distance-from-the-end scheme collapses
to 0.000 on precisely the T=16/32/64 rungs, as group-rotation §9 predicted, and
can never parse `N` at all once `N`'s width varies.

### 3.3 …and solving it does not, by itself, move held-out accuracy

This is the part that revises the sibling's estimate. §9 attributed ~0.25 of
exact accuracy to parsing, on the evidence that the identical architecture
scored 0.237–0.263 on clean one-hot slots and 0.000–0.026 on the real prompt.
With the parse now exact, `lab/probe_learnability.py` on the real submission and
real prompts (3 seeds, e1 modulus, 250/38, 2000 steps):

| front end | train_exact | **held-out T=1** |
|---|---|---|
| marker-relative (parse verified 1.000) | 1.000 / 1.000 / 1.000 | 0.026 / 0.026 / 0.026 |
| distance-from-end (control) | 0.993 / 1.000 / 1.000 | 0.053 / 0.000 / 0.000 |
| clean one-hot slots (group-rotation `probe_sel`) | 1.000 | 0.237 – 0.263 |

**The marker front-end and the broken control are indistinguishable, and both are
at the one-example floor.** So the §9 gap was not caused by parsing alone.
What clean one-hot slots additionally supply is that the step's input is
*exactly one-hot*; a learned slot→digit projection is a continuous 10-vector, and
the step's pairwise table can index that continuum instead of a digit pair —
restoring the memorisation route that the parse was supposed to remove.
Sharpening the projection (`--digit-tau 0.2`) delays memorisation (train 0.84 at
step 200 instead of 1.000) and lifts held T=1 to 0.053 at 600 steps, but it is
gone again by 2000.

**Corrected accounting: bottleneck (A) is necessary but not sufficient, and it is
worth ~0, not ~0.25, until the state alphabet is also discrete.** That is the
same conclusion as §2.4 reached from the other end, and the two together are the
main content of this branch.

## 4. Evaluator runs

All `--mode fixed_step` (contention-immune), screening on **e5** per the
coordinator's correction (512-example rungs; e1 rungs are 38 examples, so the
variance floor there is one example).

RUNS_TABLE_PLACEHOLDER

## 5. What is falsified

1. **"The coverage ceiling bounds any readout."** False. It bounds any readout
   *indexed by the residue*. A readout whose every index is a digit tuple reaches
   1.000 at N = 323, 899, 2021 and 10403 with a modulus-independent parameter
   count, and on 12 unseen moduli with one parameter vector.
2. **"Prompt→slot parsing is worth ~0.25 on its own."** False. The parse is now
   exact at every rung and under sampled `N`, and held-out T=1 does not move off
   0.026. Parsing is necessary, not sufficient.
3. **"Compositional structure is enough to prevent memorisation."** False, and
   this is the sharpest of the three. A place-shared partial-product table with a
   *continuous* carry vector memorises exactly as fast as anything else
   (train 1.000 / held 0.000 by step 2000 at `d_model=32`). The state alphabet
   has to be small; a 32-dimensional carry re-encodes the value.
4. **Straight-through discrete states do not rescue the digit pipeline**
   (train 0.012). Nor does identity/copy-through initialisation of the scans, nor
   temperature annealing (§2.4 table).
5. **Freezing correct sub-modules does not help the digit pipeline, it hurts it**
   monotonically (loss 2.3 → 18.7 as more modules are frozen at the truth). There
   is no partial-credit path: the modules are useful only jointly.
6. **Shrinking capacity does not force the algorithm.** `d_model` 8/16 fails to
   fit (train 0.06–0.42) while 32/64 memorises — the same bottleneck-that-blocks-
   fitting-too pattern group-rotation reported at §5.5b, reproduced in a
   different architecture family.

## 6. What survives, and the single highest-value recommendation

**Recommendation: the bottleneck is now optimisation of a small discrete
transducer, not representation, not scale, and not parsing. Spend the next
session on making `DigitALU`-class models trainable — the payoff is no longer
capped, and that is new.**

Why this and not something else:

1. **The ceiling result changes the strategic picture more than anything else
   measured this session.** group-rotation's honest expected value was
   "`MAX_T = 1` on Easy with a fixed modulus, and nothing beyond". That was
   correct *conditional on a value-indexed readout*. The digit readout reaches
   1.000 at m1's modulus and on unseen moduli with the same 6,818 parameters, so
   Medium, Hard and the OOD-N tie-break are no longer excluded by scale. Any
   branch still choosing between representations should choose this one.
2. **The obstruction is now precisely located and is a different kind of problem
   from everything falsified so far.** Previous negatives were about
   identifiability (§5.3's 1e-5 basin), capacity, or data. This one is credit
   assignment through ~280 sequential soft table lookups where no module is
   useful until all of them are. That is a well-posed optimisation problem with
   known families of attack, none of which I had time to run:
   * **shorten the chain.** `R = 11` weight-tied conditional subtractions per
     Horner step is 80% of the depth and 10 of 11 are no-ops. Replacing them with
     a learned quotient digit plus one subtraction takes the chain from ~280 to
     ~60 steps. This is the first thing I would run.
   * **supervise the recurrence at more than one depth.** The `T` ladder already
     supplies `x²`, `x⁴`, `x⁸` for the same operand, which puts a loss on the
     shared tables at three different unroll depths for free. `probe_alu`
     currently trains T=1 only.
   * **operand dropout** (the coordinator's tip, and the first lever in this repo
     to dent memorisation) is worth a cell specifically *here*, because
     `DigitALU` is the first readout that could convert reduced memorisation into
     generalisation — it has nowhere to memorise to.
3. **Do not spend more runs on parsing.** It is solved, verified at multi-digit
   `T` and under sampled `N`, and reusable: `MarkerPointer` is ~40 lines and
   drops into any architecture. Its measured value on its own is zero, so it
   should be carried along, not optimised.
4. **Do not screen this family with the evaluator.** Every cell is pinned at the
   trivial floor by §2.4, exactly as group-rotation found; `probe_alu`'s
   train_exact is the informative number, and for `DigitALU` specifically
   **train_exact is the whole game** — it cannot memorise 250 residues in 6,817
   digit-indexed parameters, so train_exact → 1.000 implies held_exact → 1.000.
   That inverts the usual diagnostic and makes screening cheap.

### Honest expected value

`MAX_T = 1` was not reached; best rung-1 is 1/38, level with the field. The
deliverable submission is not better than the baseline and I am not claiming it
is. What this branch delivers is the removal of a bound that was being planned
around, a parser that is exact where the previous one was structurally wrong, and
a sharply-located replacement bottleneck with a concrete attack list.

## 7. Deliverables and reproduction

* `submissions/digit-carry/submission.py` — marker-relative parser + weight-tied
  step with an ordered pointer selector (`dcp_marker`). Variants under
  `submissions/digit-carry/dcp_{marker,rev,abs,marker_digit}/`.
* `lab/probe_alu.py`, `lab/probe_alu_oodn.py`, `lab/probe_coverage.py`,
  `lab/probe_digit.py`, `lab/probe_parse.py`, `lab/make_dcp.py`.
* Every evaluator run appended to `lab/archive.jsonl` via `run_experiment.py`.

```bash
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python

# THE HEADLINE: digit readout vs the coverage ceiling
for cfg in "323 3" "899 3" "2021 4" "10403 5"; do set -- ${=cfg}
  $VENV lab/probe_alu.py --modulus $1 --slots $2 --construct; done
$VENV lab/probe_alu_oodn.py --moduli 6          # 12 unseen sampled moduli
$VENV lab/probe_coverage.py                     # closed-form counterpart

# it does not train (and freezing sub-modules at the truth makes it worse)
$VENV lab/probe_alu.py --modulus 323 --slots 3 --steps 2500
$VENV lab/probe_alu.py --modulus 323 --slots 3 --steps 1000 --freeze zero gate sub
$VENV lab/probe_alu.py --modulus 323 --slots 3 --steps 1500 --hard

# a continuous carry restores memorisation
for cfg in "8 4 1" "16 8 2" "32 16 2"; do set -- ${=cfg}
  $VENV lab/probe_digit.py --pair-oracle --d-model $1 --d-carry $2 --blocks $3 \
      --steps 2000; done

# PARSING: exact at every rung, and the control that breaks at 2-digit T
$VENV lab/probe_parse.py --front marker --construct-parse
$VENV lab/probe_parse.py --front marker --construct-parse --sampled-n
$VENV lab/probe_parse.py --front rev    --construct-parse --sampled-n

# and solving it does not move held-out accuracy
for f in marker rev; do for s in 0 1 2; do
  $VENV lab/probe_learnability.py \
      --submission submissions/digit-carry/dcp_$f/submission.py \
      --steps 2000 --seed $s; done; done

# evaluator (screening on e5 per the coordinator's correction)
$VENV lab/make_manifest.py --dataset e5 --mode fixed_step --max-steps 2000 --seeds 74
$VENV lab/run_experiment.py --submission submissions/digit-carry/dcp_marker/submission.py \
    --manifest lab/manifests/lab_e5_fs2000_s74.json --tag dcp-front
```

## 8. Compliance

Nothing under `data/generated/` was read, printed, sampled or summarised; every
probe synthesises its own values from the public generator spec and from moduli
passed on the command line. No submission contains a modular-exponentiation
routine, a digit multiplication rule, a carry rule or a lookup of answers; the
`--construct` / `--construct-parse` paths that *do* set tables to exact values
are lab diagnostics in `lab/`, are never imported by a submission, and are the
direct analogue of group-rotation's `probe_step.py --oracle`. All submission
computation is differentiable and inside the autograd graph; no Python control
flow reads `input_ids`; there is no custom training loop and no
participant-controlled backward. Nothing was submitted to the hosted service and
no `one-layer login`/`submit` was run.
