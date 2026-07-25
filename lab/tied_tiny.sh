#!/usr/bin/env bash
# Phase T7 -- isolate the DEPTH mechanism from the ARITHMETIC wall.
#
# On e1 every tied variant sits at 0-3 correct out of 38 on every rung, so the
# per-step map does not generalize and error compounding never becomes the
# constraint.  `tp1` is the same experiment shape with the arithmetic wall
# lowered: N=143=11*13 (120 units, 2-3 digit operands) instead of N=323, train
# T in {1,2,3}, full 1..64 ladder, x held out exhaustively for the rungs.
# Generated with (documented per BRIEF section 5):
#
#   python -m data.squaring_mod \
#     --output_dir data/generated/proxy_tiny_fixed_n_143_t123 \
#     --fixed_p 11 --fixed_q 13 --time_steps '[1,2,3]' --ood_time_steps '[4]' \
#     --examples_per_setting 100 --ood_examples_per_setting 20 \
#     --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
#     --depth_evaluation_exhaustive_x true \
#     --ood_n_depth_evaluation_modulus_bits '[9,10]' \
#     --ood_n_depth_evaluation_examples_per_setting 64 \
#     --train_fraction 0.8 --test_fraction 0.2 --split_group prompt \
#     --seed 45 --separate_input_output true
#
# If a tied step cannot certify T=1 even here, the family is falsified for the
# right reason.  If it can, this is where the iteration-count and on-manifold
# mechanisms can actually be compared.
set -u
cd "$(dirname "$0")/.."
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
MAN=${MAN:-lab/manifests/lab_tp1_fs4000_s3.json}
S=${S:-4000}
BS=${BS:-64}
WD=${WD:-1.0}

run () {
  local tag=$1; shift
  $VENV lab/make_tied2.py --tag "$tag" --max-steps $S --batch-size $BS --wd $WD "$@" >/dev/null
  echo "##### $tag"
  $VENV lab/run_experiment.py --submission "submissions/exp_tied/$tag/submission.py" \
    --manifest "$MAN" --tag T7-tiny --note "tiny-N proxy, depth mechanism isolated: $tag" \
    --timeout 12000
}

run t_fixK1     --loops 1 --state-mode res --pos-mode rev --act sinbil
run t_fixK4     --loops 4 --state-mode res --pos-mode rev --act sinbil
run t_fixK4_g   --loops 4 --state-mode res --pos-mode rev --act gelu
run t_reK4      --loops 4 --state-mode reembed_st --pos-mode rev --act sinbil
run t_tsoftK4   --loops 4 --eval-loops 8 --iter-mode tsoft --state-mode res \
                --pos-mode rev --act sinbil --beta 0.02 --warmup 1000 --ramp 1000
run t_tgatK4    --loops 4 --eval-loops 8 --iter-mode tgather --state-mode res \
                --pos-mode rev --act sinbil
