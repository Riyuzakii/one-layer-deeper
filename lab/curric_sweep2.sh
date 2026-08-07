#!/usr/bin/env bash
# The curriculum sweep, at a CALIBRATED 1,500 steps.
#
# Why 1,500 and not more: the no-curriculum fitting curve (lab/logs/CAL-lr*.log,
# three learning rates) reaches its plateau by step 500 and holds it --
# loss 2.290 -> 1.737 (step 500) -> 1.7315 (step 3,500), `train_exact` 0.000 at
# every logged point, soft digit accuracy equal to the measured constant-zero
# predictor.  Step 1,500 is statistically indistinguishable from step 3,500, so
# this is a post-plateau screen, not the pre-fitting region BRIEF2 warns about.
# The anneal completes at step 750, leaving 750 plain steps after it -- more
# than one full plateau time.
#
# Every cell is compared at step 1,500 against the baseline seed band read off
# the SAME logged step of B0-s0/s1/s2.
set -uo pipefail
R=/home/scratch.arohan_hw/git/one-layer-deeper/.worktrees/hf-curriculum-hf1
C="--steps 1500 --lr 3e-2 --log-every 250 --seed 0"
run() { bash "$R/lab/curric_run.sh" "$@"; }

case "$1" in
  a)
    run C-N-linear-b1  $C --beta-n 1.0 --sched linear --anneal-frac 0.5
    run D-only16       $C --only-bits 16
    run C-NX-linear-b1 $C --beta-n 1.0 --beta-x 1.0 --sched linear --anneal-frac 0.5
    ;;
  b)
    run C-N-step-b1    $C --beta-n 1.0 --sched step --anneal-frac 0.5
    run C-N-linear-b4  $C --beta-n 4.0 --sched linear --anneal-frac 0.5
    run C-X-linear-b1  $C --beta-x 1.0 --sched linear --anneal-frac 0.5
    ;;
  c)
    run B0-s2          $C --seed 2
    run D-only20       $C --only-bits 20
    run C-N-const-b1   $C --beta-n 1.0 --sched const
    ;;
  d)
    run LAD-b12 $C --id-bits 12 --ood-bits 13 --n-mod 8  --n-mod-held 4 --n-x 800  --n-held-x 64
    run LAD-b16 $C --id-bits 16 --ood-bits 17 --n-mod 16 --n-mod-held 8 --n-x 1024 --n-held-x 64
    ;;
  e)
    run LAD-b20 $C --id-bits 20 --ood-bits 21 --n-mod 16 --n-mod-held 8 --n-x 1024 --n-held-x 64
    run C-N-exp-b2 $C --beta-n 2.0 --sched exp --anneal-frac 0.25
    ;;
  f)
    run C-N-linear-b1  $C --beta-n 1.0 --sched linear --anneal-frac 0.5
    run C-N-step-b1    $C --beta-n 1.0 --sched step --anneal-frac 0.5
    ;;
  g)
    run C-N-linear-b4  $C --beta-n 4.0 --sched linear --anneal-frac 0.5
    run C-X-linear-b1  $C --beta-x 1.0 --sched linear --anneal-frac 0.5
    ;;
  *) echo "usage: $0 a|b|c|d|e|f|g"; exit 2;;
esac
