# Hard-candidate queue (ranked)

Rate limits: Hard = 1 accepted attempt/UTC day. Never spend one on an untested change;
never bundle two untested changes. Promote only effects that hold on Easy across e1–e5
above the variance floor, then survive Medium. **Nothing submitted hosted yet** (needs
user GitHub auth + burns attempts — deferred).

## Current belief / emerging recipe
Weight-tied recurrent Transformer block (the "one layer deeper" primitive), looped to a
depth matched to the tier's max composition T, optimized for step-count efficiency. Depth
buys OOD-T generalization up to a knee (~L=8 for e1's T≤6); beyond the knee, spend budget
on updates, not depth. Depth is nearly free until the block becomes compute-bound.

## Candidates (to validate before any Hard submit)
1. **Deep tied recurrence, L≈max_T.** Axis A CONFIRMED on e1 with a controlled fixed-step
   multi-seed grid: OOD L2=0.003±0.005 vs L8=0.023±0.005 (gap ≈4× std); knee ≤8. Single 60s
   runs were too noisy to show it — USE FIXED-STEP + MULTI-SEED. Re-confirm on Medium (OOD T=32,
   3000-ex eval → high SNR); pick L≈max_T per tier. Status: e1 ✓ (rigorous), Medium pending.
2. **+ Muon (hybrid) at the chosen depth.** Faster per-step convergence => more effective
   updates in-budget. Status: running (Phase 2B, equal-step vs AdamW).
3. **+ exact-match-aligned loss (focal).** Aligns training with all-or-none scoring. Status:
   generator ready, untested.
4. **+ LR schedule / more steps.** Constant LR likely suboptimal; needs a horizon. Untested.
5. **Adaptive computation (PonderNet-lite)** — BUILT (`lab/make_adaptive.py`), tested on e1.
   Halting collapses to min-depth under any ponder penalty (accuracy signal too weak); β=0 ≈
   fixed deep-L, no win (e1 is single-T). Needs: mixed-T data + working arithmetic + ponder
   warmup/KL-prior. **Gated behind the arithmetic wall — not a near-term Hard candidate.**
6. **Depth curriculum** (shallow early -> deep late) via non-persistent step-counter buffer
   (free vs 500M ceiling). Untested. Hazard: eval must run at final depth within half-budget.

## The real blocker (re-prioritized after this session)
Every tier beyond tiny-N e1 is at the **arithmetic-generalization wall** (m1 N=10403 → ~0
everywhere). Depth/adaptive/optimizer all address iteration or convergence, NOT the ability to
compute x²mod N and generalize. **The #1 next-session problem is per-step arithmetic:** digit-wise
/ place-value architectures, or exploiting that N is fixed per Easy/Medium dataset (learn the
squaring permutation table for that N, then compose). Until that moves, no lever reaches Hard.

## Blocking unknowns before Hard
- Variance floor (e1–e5 spread) — needed to trust small effects. Script ready.
- B300->H100 calibration (one hosted Easy run) — needed to translate step counts. Needs user OK.
- Learnability ceiling — does the recipe reach high accuracy at all? (Phase 2B answers.)
