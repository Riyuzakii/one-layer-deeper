#!/usr/bin/env bash
# Phase T4 -- the rung ladder always starts at T=1, but m1/hp1 TRAIN on
# T in {4,8,16}.  Certification must be a consecutive prefix from T=1, so on
# every tier above Easy, scoring anything at all requires generalizing DOWNWARD
# to T=1 and T=2, which are never trained.
#
# A tied model whose iteration count follows T is the only architecture in the
# search that can do that structurally: it just runs the block once.  This
# measures rung-1 / rung-2 exact accuracy on m1 and the hp1 Hard proxy for
#   fixed K   -- cannot adapt at all
#   tsoft     -- learned scalar read-out pointer (the real candidate; a scalar
#                can leave the trained range, which is what T=1,2 needs)
#   tgather   -- *** DIAGNOSTIC ONLY, COMPLIANCE-UNCERTAIN *** parses T out of
#                input_ids; the upper bound on what perfect halting could buy.
set -u
cd "$(dirname "$0")/.."
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
S=${S:-3000}
BS=${BS:-64}
WD=${WD:-1.0}

run () {  # tag manifest extra...
  local tag=$1 man=$2; shift 2
  $VENV lab/make_tied2.py --tag "$tag" --max-steps $S --batch-size $BS --wd $WD "$@" >/dev/null
  echo "##### $tag on $man"
  $VENV lab/run_experiment.py --submission "submissions/exp_tied/$tag/submission.py" \
    --manifest "lab/manifests/$man.json" --tag T4-below-range \
    --note "rungs 1/2 are BELOW the trained T range: $tag" --timeout 12000
}

for DS in m1 hp1; do
  run m_fixK4_$DS   lab_${DS}_fs3000_s74 --loops 4  --state-mode res --pos-mode rev --act sinbil
  run m_tgatK16_$DS lab_${DS}_fs3000_s74 --loops 16 --eval-loops 64 --iter-mode tgather \
                    --state-mode res --pos-mode rev --act sinbil
done
