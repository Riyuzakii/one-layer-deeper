#!/usr/bin/env bash
# Phase T1 -- "does a tied step generalize, or only memorize?"
# e1, fixed-step manifest (contention-immune), single seed 74, long horizon.
# The 4k-step probe showed 99% TRAIN accuracy with 2-4% test: pure memorization.
# This asks whether weight decay + a long horizon produces the grokking
# transition, and whether the on-manifold / halting mechanisms change that.
set -u
cd "$(dirname "$0")/.."
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
MAN=lab/manifests/lab_e1_fs20000_s74.json
S=20000
BS=64

run () {  # tag  extra-args...
  local tag=$1; shift
  $VENV lab/make_tied.py --tag "$tag" --max-steps $S --batch-size $BS "$@" >/dev/null
  echo "##### $tag"
  $VENV lab/run_experiment.py --submission "submissions/exp_tied/$tag/submission.py" \
    --manifest "$MAN" --tag T1-grok --note "e1 20k-step grokking probe: $tag" --timeout 3000
}

run g_res_wd01   --loops 4 --state-mode res        --wd 0.1
run g_res_wd1    --loops 4 --state-mode res        --wd 1.0
run g_rest_wd1   --loops 4 --state-mode reembed_st --wd 1.0
run g_pond_wd1   --loops 4 --iter-mode ponder --state-mode res --wd 1.0 \
                 --beta 0.01 --warmup 4000 --ramp 4000
