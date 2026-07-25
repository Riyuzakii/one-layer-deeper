#!/usr/bin/env bash
# Phase T3 -- iteration extrapolation:  task T  x  internal iterations.
#
# EVAL_LOOPS only affects the eval-mode branch of forward(), and nothing else in
# the config differs, so every run below trains the SAME model (same seed, same
# steps) and differs only in how many times the tied block is applied at
# evaluation time.  The resulting grid is the depth-extrapolation claim under
# direct test: train at K=4, run at 1/2/4/8/16/64, and read the per-rung profile.
#
# It also checks stability: 64 applications of a block trained with 4 is where a
# residual stream would blow up if it were drifting off-manifold.
set -u
cd "$(dirname "$0")/.."
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
MAN=${MAN:-lab/manifests/lab_e1_fs2000_s74.json}
S=${S:-2000}
BS=${BS:-64}
WD=${WD:-1.0}

run () {
  local tag=$1; shift
  $VENV lab/make_tied2.py --tag "$tag" --max-steps $S --batch-size $BS --wd $WD "$@" >/dev/null
  echo "##### $tag"
  $VENV lab/run_experiment.py --submission "submissions/exp_tied/$tag/submission.py" \
    --manifest "$MAN" --tag T3-extrap --note "iteration extrapolation: $tag" --timeout 9000
}

for E in 1 2 4 8 16 64; do
  run x_K4e${E} --loops 4 --eval-loops $E --state-mode res --pos-mode rev --act sinbil
done
for E in 4 64; do
  run x_reK4e${E} --loops 4 --eval-loops $E --state-mode reembed_st --pos-mode rev --act sinbil
done
