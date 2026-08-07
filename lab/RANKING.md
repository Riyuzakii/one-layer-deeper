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
| 1 | **Addition-only transducer** — multiplication by double-and-add, so the only learned arithmetic primitive is an adder (+ comparator) | removes the provably-unidentifiable `Tmul`; adder laws repair 50/400 where `Tmul` repairs ~0 | **untested** |
| 2 | **`DigitALU` at hf1 scale, fully corrected** — `tree:quotient` (39 steps), `EMB_INIT=0.02`, ~93k-step budget, replica population, calibrated fitting curve | never run on a modulus-split dataset; applies every earned correction at once; supplies the reference and `--lr 0` floor | **untested at Hard-faithful** |
| 3 | **Modular reduction at O(1) learned-op depth** | `matrix-scan` named it the one well-posed remaining target; the conditioning law is measured and correct (0/5 @39 ops, 14/400 @12, 50/400 @1) — this is where its coefficient is largest | **untested** |
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
