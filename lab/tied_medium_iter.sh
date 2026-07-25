#!/usr/bin/env bash
# Phase T4 -- the rung ladder always starts at T=1, but m1/hp1 TRAIN on
# T in {4,8,16}.  So on every tier above Easy, certifying anything at all
# requires generalizing DOWNWARD to T=1 and T=2, which were never trained.
#
# A tied model whose iteration count follows T is the only architecture in the
# search that can do that structurally: it just runs the block once.  This
# script measures rung-1 / rung-2 exact accuracy on m1 and the hp1 Hard proxy
# for a fixed-K model (cannot adapt), a learned-halting model, and the
# T-gather upper bound (DIAGNOSTIC ONLY, COMPLIANCE-UNCERTAIN -- parses T out
# of input_ids; never a submission).
set -u
cd "$(dirname "$0")/.."
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
S=${S:-8000}
BS=${BS:-64}
WD=${WD:-1.0}

run () {  # tag manifest extra...
  local tag=$1 man=$2; shift 2
  $VENV lab/make_tied.py --tag "$tag" --max-steps $S --batch-size $BS --wd $WD "$@" >/dev/null
  echo "##### $tag on $man"
  $VENV lab/run_experiment.py --submission "submissions/exp_tied/$tag/submission.py" \
    --manifest "lab/manifests/$man.json" --tag T4-below-range \
    --note "rung 1/2 are BELOW the trained T range: $tag" --timeout 9000
}

for DS in m1 hp1; do
  run m_fixK4_$DS   lab_${DS}_fs8000_s74 --loops 4  --state-mode res --pos-mode rev
  run m_pondK16_$DS lab_${DS}_fs8000_s74 --loops 16 --iter-mode ponder --state-mode res \
                    --pos-mode rev --beta 0.01 --warmup 2000 --ramp 2000
  run m_tgatK16_$DS lab_${DS}_fs8000_s74 --loops 16 --iter-mode tgather --state-mode res \
                    --pos-mode rev
done
