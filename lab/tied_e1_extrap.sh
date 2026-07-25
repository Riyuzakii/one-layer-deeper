#!/usr/bin/env bash
# Phase T3 -- iteration extrapolation:  task T  x  internal iterations.
#
# EVAL_LOOPS only affects the eval-mode branch of forward(), so every run below
# trains the SAME model (same seed, same steps, same config) and differs only in
# how many times the tied block is applied at evaluation time.  The resulting
# grid is the depth-extrapolation claim under direct test: can a block trained
# at K iterations be run at 8K or 16K and still decode anything exact?
#
# Also included: state-mode variants at fixed training K, to see whether keeping
# the inter-iteration state ON the token manifold (re-embedding) changes the
# SHAPE of the per-rung profile (clean cliff = one bad step; slow decay = drift).
set -u
cd "$(dirname "$0")/.."
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
MAN=${MAN:-lab/manifests/lab_e1_fs4000_s74.json}
S=${S:-4000}
BS=${BS:-64}
WD=${WD:-1.0}
MODE=${MODE:-fixed}

run () {
  local tag=$1; shift
  $VENV lab/make_tied.py --tag "$tag" --max-steps $S --batch-size $BS --wd $WD "$@" >/dev/null
  echo "##### $tag"
  $VENV lab/run_experiment.py --submission "submissions/exp_tied/$tag/submission.py" \
    --manifest "$MAN" --tag T3-extrap --note "iteration extrapolation: $tag" --timeout 9000
}

# trained at K=4, evaluated at 1/2/4/8/16/64 internal iterations
for E in 1 2 4 8 16 64; do
  run x_K4e${E} --loops 4 --eval-loops $E --state-mode res --pos-mode rev
done

# state handling between iterations, all at train K=4 / eval K=16
for SM in res renorm gate reembed reembed_st; do
  run x_sm_${SM} --loops 4 --eval-loops 16 --state-mode $SM --pos-mode rev \
      $( [ "$SM" = gate ] && echo "--gate-init -1.0" )
done
