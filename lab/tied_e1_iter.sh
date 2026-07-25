#!/usr/bin/env bash
# Phase T2 -- does matching the iteration count to the task's T matter?
#
# The decisive comparison for this hypothesis family.  If composition
# supervision is what fills in the held-out units, then a model whose iteration
# count equals T (tgather = the ideal-halting upper bound) should have HIGHER
# rung-1 exact accuracy than a fixed-K model, because its step map is
# additionally constrained at x^2 and x^4 for every training x.
#
#   K=1 fixed   -> no composition at all (control)
#   K=4 fixed   -> composition, but the count never matches T
#   ponder K=4  -> learned halting (the real candidate)
#   tgather K=4 -> iteration count = T exactly.
#                  *** DIAGNOSTIC ONLY, COMPLIANCE-UNCERTAIN ***
#                  it parses T out of input_ids (BRIEF.md section 4.3 gray area).
#                  Never a submission; it exists to upper-bound halting.
#
# `rev` variants add a position embedding indexed from the END of the valid
# region.  Targets are right-aligned to the prompt end and the prompt length
# varies with len(x)/len(T), so this makes the read-out slot identical across
# prompt lengths -- including the two-digit-T rungs (16/32/64), whose prompts
# are one token longer than anything in e1 training.
set -u
cd "$(dirname "$0")/.."
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
MAN=${MAN:-lab/manifests/lab_e1_fs8000_s74.json}
S=${S:-8000}
BS=${BS:-64}
WD=${WD:-1.0}

run () {
  local tag=$1; shift
  $VENV lab/make_tied.py --tag "$tag" --max-steps $S --batch-size $BS --wd $WD "$@" >/dev/null
  echo "##### $tag"
  $VENV lab/run_experiment.py --submission "submissions/exp_tied/$tag/submission.py" \
    --manifest "$MAN" --tag T2-iter --note "e1 iteration-count axis: $tag" --timeout 9000
}

run i_fixK1       --loops 1 --state-mode res
run i_fixK4       --loops 4 --state-mode res
run i_pondK4      --loops 4 --iter-mode ponder --state-mode res --beta 0.01 --warmup 2000 --ramp 2000
run i_tgatK4      --loops 4 --iter-mode tgather --state-mode res
run i_fixK4_rev   --loops 4 --state-mode res --pos-mode rev
run i_tgatK4_rev  --loops 4 --iter-mode tgather --state-mode res --pos-mode rev
