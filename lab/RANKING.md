# Method ranking after ~1,100 experiments — and the Hard-faithful test plan

Read with `lab/RESUME.md` (full state) and `lab/BRIEF2.md` (screening discipline).
Everything below is ranked by **measured evidence**, not by appeal.

## The organising fact

The failure is **one squaring on unseen operands**. Everything else is solved and
verified end to end: parsing, T-fold composition, the depth controller (MAX_T=64 with a
learned controller), the readout representation, and the harness itself (a constructed
oracle certifies MAX_T=64 AND OOD_N=64 through the real runner; a *legal learned* model
reaches held-out **1.000** on unseen 7-digit operands at `T=0`, and **0.000** at `T=1`
on the same modulus with the same pipeline).

The sharpest sub-diagnosis is `alu-relational`'s: **`Tmul`'s ~200 cells are
unidentifiable** — not from the end-of-chain label, not from any generic algebraic law,
and only from the definition of multiplication, which is illegal as a loss term.
Meanwhile the **adder** laws have the widest repair basin measured anywhere in this
project: **50/400 cells, against the label's 0/5 at k=20**.

**That asymmetry is the whole basis of the ranking.**

Structural note: the task is a **time-lock puzzle**. `trapdoor_squaring_mod` computes
`pow(2, T, phi)` because the *generator* knows `p, q`. A model without the factorisation
must actually perform T squarings — by construction. So composition being solved does not
help; the difficulty is concentrated entirely in the single step.

## Ranking

| # | method | evidence | status |
|---|---|---|---|
| ~~1~~ | ~~Addition-only transducer~~ | **STRUCK — the ranking premise was a substitution error.** The 50/400 belonged to an **O(1) algebraic objective**, not to the adder as a table. Module-restricted with everything else at truth, the label leaves the **adder at chance** (0.008-0.016) while `Tmul` scores **higher** (0.028-0.052) — the adder is the *less* identifiable table. Constructed ceiling 36/36 at 1.000 soft+hard; basin advantage real at 4.2% corruption (8/25 vs 0/20, p=0.0056) but only in *rate*, not *radius* (5.9% both sides), and only radius is reachable from init. Best legal `train_exact_hard` 0.002 vs `--lr 0` 0.000 | **closed** |
| ~~1~~ | ~~`DigitALU` at hf1 scale, fully corrected** — `tree:quotient` (39 steps), `EMB_INIT=0.02`, ~93k-step budget, replica population, calibrated fitting curve | **CLOSED at the ranked tier.** All corrections at once (`tree:quotient`, `EMB_INIT=0.02`, S=7, 1,725 moduli, P=32): `train_exact_hard` 0.000, `local_ce` trains *away* 2.14 -> 3.12, 64 replicas and not one correct answer. Modulus-split changes nothing (1,725 moduli vs 8 agree to 3 dp). Cost does NOT close it (~2,100 steps available vs an illegal ceiling reached by ~1,200) | **closed** |
| ~~2~~ | ~~Modular reduction at O(1) learned-op depth~~ | **CLOSED — the scaling REVERSES.** Barrett reduction at total depth **7** (reduction depth a constant 5) repairs **0 of 1,802** corrupted cells, and **0/1200 at k=400** where depth-12 `MonoidALU` repaired 14/400. Instrument validated. Ceiling 1.000 soft+hard per modulus size at all six hf1 bit sizes. hf1 legal: `train_exact` 0.725 (past onset), `train_exact_hard` **0.000**, `--lr 0` *better* on held-out | **closed** |
| 4 | ~~Loss-side curriculum over modulus size / operand magnitude~~ | **CLOSED — null on hf1.** 10 cells x 4 schedule shapes x 4 strengths x 3 axes x 3 seeds: `train_exact_hard` 0.000 everywhere, held-out 0.000 per size; all six evaluator arms within **one example in 27,000** of the `--lr 0` control | **closed** |
| 5 | **Replica population** as a standard instrument on whichever of 1-3 shows any tail | validated: 1-in-7 -> 6-of-6, legal, +29% wall clock, eval at P=1 cost | instrument |

## Do NOT spend runs on these — closed with reasons

| method | why it is closed |
|---|---|
| generic algebraic laws (`sym`/`inv`/`assoc`/`cancel`) | null, and **closed by their own ILLEGAL ceiling** — handing the loss the true identity also does nothing |
| expressivity axis (PD-SSM, DeltaProduct, matrix scan, LSTM) | **refuted with positive controls**: same code goes chance -> 1.000 on A5 (NC1-complete) and parity, and nothing here. A model with **no sequence mixing at all** scores the same |
| residue-indexed readouts (Fourier, softmax over `Z_N`, tables, permutations) | capped by the closed-form coverage bound; on modulus-split Hard, **doubly dead** — test moduli never appear in training |
| target propagation, dual-path agreement | the **agreement-law dichotomy**: KL form is satisfied at init (no gradient), CE form starts at `2 ln 10` and does not descend |
| log-depth / parallel scan for conditioning | a parallel and a serial prefix over one operator are **the same function with the same gradients** (losses agree to 0.3-0.5%) |
| Neural GPU | 22x the cost of the RNN it replaces at Hard's shape; flat in both depth and width; its own gradient noise hurts |
| more compute / capacity / steps / optimizer family | all measured *not* to be the constraint, incl. SOAP and AdEMAMix past a diagnostic gate |

## Test plan on `hf1`

`hf1` = `split_group=modulus` + `separate_ood_splits`, ID bits [16,18,20], OOD-N
[17,19,21], train T {4,8,16}, OOD T 32, full 7-rung ladder both sides, 298,752 rows,
`max_seq_len` 19. `hf1s` is the same structure at 49,152 rows for fitting-curve
calibration only (fewer rows makes memorisation *easier* — not for accuracy claims).

**Mandatory for every result** (all earned the hard way, see `lab/BRIEF2.md` §6):
1. `--lr 0` control before interpreting any improvement.
2. A **`train_exact` fitting curve** showing training had begun to move — a 1,200-1,500
   step screen at Medium scale sits in the pre-fitting region, and hf1 is larger.
3. `train_exact_hard` (argmax-snapped) as the headline where a discrete state exists.
4. Report the metric **row**, not the cell; every metric here has a measured
   configuration that fools it.
5. Collapse detector with a **measured** reference — the exact solution's output
   diversity is **not** 1.0.
6. Label every row **LEGAL** or **DIAGNOSTIC**. Oracles and teacher forcing can never
   appear in a submission.

## Findings from the closed #4 branch that the live branches depend on

1. **A fixed-slot ALU on a MIXED-modulus dataset must reduce at every place.** `hf1` has
   three modulus sizes in one training set; the `t <= S` schedule **overflows the
   quotient alphabet for smaller moduli**, so the constructed ceiling is NOT 1.000 across
   sizes. `redall` restores **1.000 at all six bit sizes** (ID 16/18/20, OOD-N 17/19/21)
   for +54 sequential steps (183 vs ~129). **Verify constructed ceilings PER MODULUS
   SIZE, never pooled.**
2. **`training_loss` cannot express a per-example weight here.** The valid mask is ragged
   (supervised positions = the *answer's* digit count; `number_tokens(result)` is
   unpadded) and row boundaries are unrecoverable from `(logits, labels, aux)` after
   flattening. Working equivalent: a per-row gradient scale in the forward, after the
   head — `logits = w*logits + (1-w)*logits.detach()` — verified to 6.9e-08.
3. **Digit accuracy is fooled by leading zeros.** A constant-zero predictor scores
   **0.379 / 0.303 / 0.235** at 16/18/20 bits — exactly the profile a "successful"
   size-curriculum would show. Measure the constant-zero floor for your own setup.
4. **Correct tables transfer across modulus size nearly for free (DIAGNOSTIC).** Taught
   from **16-bit examples only** under an illegal per-op signal, the tables come out
   exactly right and score **0.896** hard exact at 20-bit, **0.878** on unseen moduli and
   **0.867** at unseen modulus *sizes* — 2.4x better than the same signal spread over all
   three sizes. **Transfer is not the obstacle; the source is.**
5. **A ranking premise of mine was wrong:** with a fixed slot count a smaller modulus does
   NOT shorten the chain (same 183 steps), touches fewer shared cells (21.5 vs 28.2 of
   100), and is *less* disturbed by a wrong cell. "Smaller = easier" does not hold here.
6. `hf1` reference scale: **243k train rows, 768-example rungs, variance floor 1/768**,
   and the reference-width model **never leaves the pre-fitting region at 40k steps**.

## The single most actionable result so far: SEPARABILITY, not size

`hard/add-only` measured that the legal label's exact-repair radius on an arithmetic
table is **5-10 wrong cells**, the same for `Tadd` and `Tmul` — but a **10-cell
*separable* table** (a selector whose parameters are determined by the label largely
independently of one another) is **recovered exactly, every seed**. That is a
qualitative difference in learnability, and it is the **first structural property found
in ~1,200 experiments that the legal objective can actually exploit.**

**Design implication:** prefer architectures whose learned components are *separable*,
not merely small or shallow. This is now the most promising untested axis, and it
supersedes "reduce learned-op depth" as the primary design criterion.

## A correction to this document's own reasoning

**A repair basin is a property of an (objective, architecture) PAIR, not of a table.**
The 0/5 @39 ops -> 14/400 @12 -> 50/400 @1 scaling is valid *for a fixed objective*. The
50/400 came from an O(1) algebraic objective evaluated next to a table. Carrying that
number to a different architecture by deleting a table is a **substitution error**, and
it is what put add-only at #1. Recorded so it is not repeated.

---

# FINAL STATE — all five ranked methods are closed

| # | method | outcome |
|---|---|---|
| 1 | addition-only transducer | **struck** — the ranking premise was a substitution error; the adder is the *less* identifiable table |
| 2 | `DigitALU` at hf1, fully corrected | **closed** — hard zero, gradient away from the solution, cost is not the obstacle |
| 3 | O(1)-depth modular reduction | **closed** — the conditioning scaling *reverses* at depth 7 |
| 4 | loss-side curriculum | **closed** — null; transfer works, the source does not exist |
| 5 | replica population (instrument) | works mechanically; nothing for it to find |

## Two design criteria of mine, both now refuted by measurement

1. **"Reduce learned-op depth"** — refuted. The scaling reverses at depth 7, and the
   middle data point was an artifact: `MonoidALU`'s tables are **shared across all S
   reduce steps**, so its "depth 12" was a *maximum*, not a uniform depth. The metric was
   mis-specified.
2. **"Prefer separable components"** — refined into uselessness for discovery. Separable
   9/19 vs non-separable 0/18 at matched depth is real, but the threshold is **<=2
   competing wrong rows**, decaying from 4 and **gone by 16**; random init is ~176 wrong.
   **Separability is a CONVERGENCE property, not a DISCOVERY property.**

## What is actually established, and it is one sentence

A k=0 control shows the exact solution is a **stable fixed point** (loss 0.00000). Any
wrong cell produces a gradient that **repairs none and breaks others**. The legal
objective does not transmit information about modular squaring to the parameters — at
any depth, any separability, any architecture, any optimizer, any scale, and on a
dataset that is structurally identical to the ranked tier.

## Engineering notes worth keeping

- **Barrett reduction sidesteps the mixed-modulus quotient-alphabet overflow entirely**
  (it never emits quotient digits); the fix is widening `mu`, costing **+0 sequential
  steps** versus `redall`'s +54.
- `tree:quotient` fixes the Hard eval budget: **265 s of 1,800 s (6.8x)** where the
  serial graph had 1.1x and threw `TimeoutError`.
- `EMB_INIT=0.02` puts step-1 loss at 2.86 instead of 79.936 (the official baseline pays
  the 79.936).
- **`hf1` never fits at baseline-class capacity within 40k steps** — calibrate on `hf1s`.
