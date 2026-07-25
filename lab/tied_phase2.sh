#!/usr/bin/env bash
# Phase T6 -- can ANY tied step get off the trivial floor on held-out x?
#
# Everything measured so far sits at 0-3 correct out of 38 on every rung, i.e.
# the per-step map does not generalize at all, so error compounding never gets a
# chance to be the constraint.  This grid attacks the step map itself, with
# knobs that follow from the structure of what one step has to compute:
#
#   x^2 mod 323 == (310a^2+100b^2+c^2+62ab+200ac+20bc) mod 323   (verified)
#
#   --act bilinear : the digit products are literally a product of two linear
#                    maps, so a gated MLP with NO activation expresses them.
#   --act sin      : the modular fold is periodic; a sinusoidal MLP makes
#                    "wrap around N" representable rather than approximated.
#   --act sinbil   : both.
#   reembed_st     : the intermediate residue is re-expressed as digits between
#                    iterations (straight-through, exact forward, live gradient).
#   tsoft          : learned scalar read-out pointer (extrapolates in T).
#
# 3 seeds; rungs are 38 examples so single-seed differences are meaningless.
# Uses lab/make_tied2.py (= make_tied.py + AdamW param groups that exclude 1-D
# tensors from weight decay + the tsoft/bilinear/sin additions).
set -u
cd "$(dirname "$0")/.."
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
MAN=${MAN:-lab/manifests/lab_e1_fs4000_s3.json}
S=${S:-4000}
BS=${BS:-64}
WD=${WD:-1.0}

run () {
  local tag=$1; shift
  $VENV lab/make_tied2.py --tag "$tag" --max-steps $S --batch-size $BS --wd $WD "$@" >/dev/null
  echo "##### $tag"
  $VENV lab/run_experiment.py --submission "submissions/exp_tied/$tag/submission.py" \
    --manifest "$MAN" --tag T6-arch --note "per-step exactness axis (3 seeds): $tag" --timeout 12000
}

run a_gelu     --loops 4 --state-mode res --pos-mode rev --act gelu
run a_bilin    --loops 4 --state-mode res --pos-mode rev --act bilinear
run a_sin      --loops 4 --state-mode res --pos-mode rev --act sin
run a_sinbil   --loops 4 --state-mode res --pos-mode rev --act sinbil
run a_sinb_re  --loops 4 --state-mode reembed_st --pos-mode rev --act sinbil
run a_sinb_ts  --loops 4 --eval-loops 8 --iter-mode tsoft --state-mode res \
               --pos-mode rev --act sinbil --beta 0.02 --warmup 1000 --ramp 1000
