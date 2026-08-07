#!/usr/bin/env bash
# Sequential stream of probe_curric.py cells.  Usage: curric_sweep.sh <stream>
# Three streams are meant to run concurrently; `--mode fixed_step`-equivalent
# (a fixed --steps) so results are contention-immune.
set -uo pipefail
R=/home/scratch.arohan_hw/git/one-layer-deeper/.worktrees/hf-curriculum-hf1
# 3,000 steps is a CALIBRATED budget, not a guess: the no-curriculum fitting
# curve (lab/logs/CAL-lr*.log) reaches its plateau -- soft digit accuracy equal
# to the measured constant-zero predictor, to three decimals -- by step 500-1000
# at every learning rate, and holds it.  The anneal completes at step 1,500,
# leaving 1,500 plain steps after it.
STEPS=3000
LR=3e-2
COMMON="--steps $STEPS --lr $LR --log-every 500"

run() { bash "$R/lab/curric_run.sh" "$@"; }

case "$1" in
  a)
    run B0-s0   $COMMON --seed 0
    run C-N-linear-b1 $COMMON --seed 0 --beta-n 1.0 --sched linear --anneal-frac 0.5
    run C-N-linear-b4 $COMMON --seed 0 --beta-n 4.0 --sched linear --anneal-frac 0.5
    run C-NX-linear-b1 $COMMON --seed 0 --beta-n 1.0 --beta-x 1.0 --sched linear --anneal-frac 0.5
    run D-only16 $COMMON --seed 0 --only-bits 16
    ;;
  b)
    run B0-s1   $COMMON --seed 1
    run C-N-step-b1 $COMMON --seed 0 --beta-n 1.0 --sched step --anneal-frac 0.5
    run C-N-linear-b05 $COMMON --seed 0 --beta-n 0.5 --sched linear --anneal-frac 0.5
    run C-X-linear-b1 $COMMON --seed 0 --beta-x 1.0 --sched linear --anneal-frac 0.5
    run D-only20 $COMMON --seed 0 --only-bits 20
    ;;
  c)
    run B0-s2   $COMMON --seed 2
    run C-N-exp-b1 $COMMON --seed 0 --beta-n 1.0 --sched exp --anneal-frac 0.25
    run C-N-linear-b2 $COMMON --seed 0 --beta-n 2.0 --sched linear --anneal-frac 0.5
    run C-N-const-b1 $COMMON --seed 0 --beta-n 1.0 --sched const
    run C-N-linear-b2-f025 $COMMON --seed 0 --beta-n 2.0 --sched linear --anneal-frac 0.25
    ;;
  *) echo "usage: $0 a|b|c"; exit 2;;
esac
