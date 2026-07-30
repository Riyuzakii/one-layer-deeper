# Exploration brief v2 — non-transformer architectures

You are one of several agents running the `PLAN2.md` exploration space. This file is
the shared ground truth and **supersedes `PLAN2.md` where they disagree** — PLAN2 was
written before the last session's results landed, and two of its premises need
correcting.

Read, in order: this file → `lab/RESUME.md` (the full state of the previous session) →
`PLAN2.md` (the architecture space) → `lab/BRIEF.md` (evaluator contract and compliance,
still authoritative).

---

## 1. The metric (unchanged)

Hard ranks by **Max T** — the largest `T ∈ {1,2,4,8,16,32,64}` whose rung *and every
lower rung* are **100% exact** — then OOD-N Max T, then earliest submission. No partial
credit. `mean_exact_accuracy` is a diagnostic with no ranking value.

~950 experiments across 15 branches have scored **MAX_T = 0**. Best rung-1 anywhere is
3/38 on e1 against a trivial-predictor floor of 1/38.

## 2. Corrections to PLAN2's premises — read before building

**(a) The sequence axis is NOT the composition axis.** `max_seq_len` is **13** (e1/e5),
**15** (m1), **21** (m4); vocab is 17. The prompt is `[N] digits [X] digits [T] digits`.
The task's composition depth is `T ≤ 64`, which is *not* a sequence dimension. A matrix
associative scan **over the sequence** composes prompt tokens; it does not give you
T-fold squaring.

**(b) T-fold composition is already solved, so it is not the target.** With a correct
single step, exactness composes **1.000 at every rung T=1…64** across five regimes
including 30/32-bit sampled moduli, in bf16+amp, with no drift after 64 compositions.
A learned depth controller (12 scalars) reaches **MAX_T = 64**. Do not spend runs here.

**What is actually open is ONE squaring on unseen operands**: `x² mod N` exactly, for
`x` (and on Hard, `N`) never seen in training. Everything else in the pipeline works.

**(c) Where a scan genuinely fits.** Carry propagation across digit positions in
multi-digit arithmetic *is* an associative prefix computation (the classic
propagate/generate carry monoid). That is the open bottleneck, it lives on the
digit-position axis (length ~5-10), and it has never been tried in log-depth form.

**(d) PLAN2's own Phase-0 item 3 partly contradicts its diagnosis.** On Easy, models
reach ~100% *train* exact accuracy with held-out at chance: a generalisation failure,
not an expressivity failure. This is established and robust — it now includes a 40,000-
step run on a maximally expressive non-linear RNN that hits train 0.984 by step 5,000
and holds 0.98-1.00 for the remaining 35,000 while held-out sits at 7/1200.

**The Medium claim was WRONG and is now settled — there is no Easy/Medium split.**
An earlier version of this brief said "at m1 scale and above they cannot even fit". A
40,000-step curve at m1 disproves it: `train_exact` at `D_H`=128 goes 0.0098 (3k steps)
-> 0.039 (5k) -> **0.62 (40k)**, and `D_H`=256 reaches **0.86**. The memorisation
signature is **one phenomenon at 600 rows and at 27,000** — train 0.86 with held-out at
**3 examples in 3,000**, against **0.0000** at `lr=0`, output diversity 0.87 (a genuine
null, not a collapse) and `held_ce` 7.9 (confident and wrong). This *strengthens* the
diagnosis rather than weakening it. Scope: this revises the claim for the RNN family;
the `DigitALU` `local_ce` 3.5-4.2 measurement is a different metric and stands.

**(e) THE PRE-FITTING REGION — read before you interpret any Medium-scale null.** At m1
scale a **1,200-1,500 step screen sits inside the pre-fitting region**, where
`train_exact` has not begun to move. The PLAN2 manifests are dominated by
`fs1200`/`fs1500`. Fitting onsets are architecture-specific, so this does not invalidate
results by itself — but **every Medium-scale null must carry a `train_exact` curve
showing it had begun to move.** That is one extra curve, not one extra experiment.
**A null at a step count you have not calibrated against a fitting curve is not a null.**
Capacity is a real secondary axis and **params/row predicts it**: m1 at `D_H`=128 is
17.4 params/row, below the 24.9/row that fit e5, and raising capacity moved train
0.62 -> 0.86.

## 3. What the last session established — do not re-derive

**Solved:** prompt→digit-slot parsing (marker-relative anchoring, 1.000 at T=16/32/64);
T-fold composition; the depth controller; and the *representation* of a
modulus-independent digit readout.

**Closed with reasons:**
- **Residue-indexed readouts** (Fourier, softmax over `Z_N`, embedding table, learned
  permutation) are capped by a closed-form coverage bound: oracle held-out accuracy is
  `P[x² already seen]` = 1.000 / 0.614 / 0.337 / 0.031 at N = 323 / 899 / 2021 / 10403.
- **`DigitALU`** (a digit-indexed transducer, ~6.8k params, exact solution provably in
  the class) cannot be trained legally: ~160 gradient procedures, 1.5x10^8 discrete-search
  evaluations, 544-replica screening at P=32-64, and three optimizer families including
  full-matrix preconditioning (SOAP) and slow-EMA (AdEMAMix). All null.
- **The killer measurement:** `local_ce` at **random init is 2.08-2.24**; the legal
  end-to-end objective *trains* it to **3.5-4.2**. **Training moves away from the
  discrete solution from the first step.** Every "improvement" ever recorded on that
  objective was partial regression toward init.

**The design criterion that came out of it, and the reason PLAN2 §3.1 is worth
running:** *a constraint's conditioning is set by how many learned ops separate it from
the parameters, not by how much information it carries.* Laws evaluated O(1) ops from a
table exactly repair 50/400 corrupted cells; the end-of-chain label repairs 0/5 at k=20.
The failed ALU put 39-280 **serial** soft steps between the loss and its tables. **A
log-depth scan puts ~5.** Whether that changes *conditioning* rather than merely speed
is the single most valuable untested question, and it is what this plan should attack.

## 4. Feedback from the one hosted Hard run

- **Hard uses `split_group="modulus"` with `separate_ood_splits=True`** (scoring splits
  `test`, `ood_t`, `ood_n_t`). Train and test draw from **disjoint modulus pools**, so
  even `test` demands **unseen-modulus** generalisation. Any per-modulus lookup or
  per-modulus table is dead on Hard by construction. Modulus-independent parameters are
  mandatory.
- Our `hp1`/`hp2`/`hp3` proxies used `--split_group prompt` and therefore model the
  wrong thing. Regenerate with `--split_group modulus --separate_ood_splits true` if you
  need a Hard proxy.
- **H100 calibration (real, from the hosted run):** 3.9 s startup (import + construct +
  `.to(device)` + optimizer + first step), **~38.6 ms/step** at batch 512 for a D=128
  recurrent stack → **~93,000 steps** in a full 3600 s Hard run. Far more generous than
  the ALU-based estimates.

## 5. Environment (already checked — do not re-verify)

- `triton` **3.7.1 imports but CANNOT COMPILE on this box** — `ptxas-blackwell: sm_107a
  is not defined`. The same failure **kills `torch.compile`**. **Do not write or budget
  for custom kernels.** Fusion still pays enormously via built-ins: cuDNN `nn.LSTM` beat
  a naive Python loop **8.1x**.
- `torch._higher_order_ops.associative_scan` **importable** (torch 2.12.1+cu130).
- `max_seq_len` 13 / 15 / 21 (e1-e5 / m1 / m4); `vocab_size` 17 everywhere.
- Venv: `/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python`. No installs.

## 6. Screening discipline — earned the hard way, do not skip

1. **Run `--lr 0` as a control before interpreting any "improvement."** The legal
   objective trains *away* from the solution; a leftward metric shift is usually
   regression toward init.
2. **Screen on `train_exact_hard`** (states snapped to argmax) where a discrete state
   exists, never on soft exact-match. Soft metrics are won by mixtures.
3. **Report the metric row, not the cell.** Every metric used last session has a
   measured configuration that fools it — see the table in `lab/RESUME.md`. Include a
   collapse detector (output diversity over held-out operands; a constant map reads
   ~1/n_answers).
4. **Screen on e5 (512-example rungs), not e1** (38-example rungs, variance floor of one
   example, and 96% of its `test` operands appear at another T). Use e1 only for a final
   MAX_T attempt.
5. **e1 and e2 are degenerate for depth work**: λ(323)=144 collapses T=4/16/64 to one
   map and T=8/32 to another — four distinct rungs, not seven. State the λ of any
   modulus you use.
6. **Measure a signal's illegal ceiling and its basin before building its legal
   version.** Four runs closed the strongest family last session.
7. **Cost ratios do not transfer across tier shapes.** The Neural GPU is 22x the fused
   LSTM at Hard's shape (L=21, batch 512) but only 1.7x at e5's (L=13, batch 128) — at
   small shapes both are launch-bound and the DataLoader dominates. Screen *accuracy* on
   e5, but take every wall-clock comparison at Hard's shape, as a ratio to a fixed
   reference model (absolutes do not transfer off sm_107).
8. **Parallelism is a net loss at this scale.** Fused serial LSTM 0.33x, reference 1.00x,
   Neural GPU 7.22x — the most parallel candidate is 22x slower than the least. Judge
   scan-based candidates on *conditioning*, never on speed.
9. **Eval budget does not constrain PLAN2**: 2.2-5.2 s against a 30 s Easy allowance
   across every family tested (~8x margin), unlike the ALU family's 1.1x.
10. Use `--mode fixed_step` manifests for cross-agent comparisons; several agents share
   one GPU and wall-clock step counts are not comparable.

## 7. Compliance (BRIEF.md §4 in full — unchanged, and it binds)

Never read/print/summarize anything under `data/generated/`. No hard-coded arithmetic,
solver, or lookup of answers in the forward pass — choose structure, learn the function.
End-to-end differentiable during training; no custom training loop, no
participant-controlled backward (a custom `torch.optim.Optimizer` is fine). Never submit
to the hosted service. `--construct`/`--teacher-force`-style oracles are **lab
diagnostics only** and can never appear in a submission — label every result **LEGAL**
or **DIAGNOSTIC**.

The compliance line adopted last session, which two branches derived independently:
> A loss term is **legal** if it asserts a *generic algebraic property of an operation
> the model already performs*. It is **not legal** if it asserts *the specific
> relationship that constitutes the definition of the arithmetic* — that is a
> hard-coded algorithm moved from the forward pass into the loss.

## 8. Deliverables

Work only on your own branch. `lab/reports/<branch>.md`, any submission under
`submissions/<branch>/`, every run archived. Commit as you go. Negative results carry
the same weight as positive ones — and given the state of this project, a clean,
well-controlled negative is the most likely useful outcome. Say so plainly if you get
one.
