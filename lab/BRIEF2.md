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

**The Medium claim is NOT established — treat it as open.** An earlier version of this
brief said "at m1 scale and above they cannot even fit". That is confounded with step
count: e5 needed ~5,000 steps to fit at `D_H`=64, and m1 has **5.6x more rows** but was
only run for 3,000. `plan2/sequential-rnn` flagged this against its own conclusion and
is running the clean experiment (`--dataset m1 --steps 40000`, with a capacity arm to
separate "cannot fit" from "too few steps" from "too little capacity"). Until it lands,
do not design around an Easy/Medium split in the diagnosis. Note the direction cuts
against convenience: if m1 *does* fit given enough steps, the memorisation diagnosis
extends to Medium and the expressivity framing weakens at every tier.

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

- `triton` **3.7.1 importable** → custom chunkwise/fused kernels are on the table.
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
7. Use `--mode fixed_step` manifests for cross-agent comparisons; several agents share
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
