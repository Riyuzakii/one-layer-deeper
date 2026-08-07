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
| 4 | **Loss-side curriculum over modulus size** (hf1 has [16,18,20]) and operand magnitude | legal, cheap, untested at Hard-faithful scale; the prior null was e1/`DigitALU` | weak |
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
