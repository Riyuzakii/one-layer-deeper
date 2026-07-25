#!/usr/bin/env bash
# Data-volume probe datasets for the representation hypothesis.
#
# e1 is fixed N=323, so the squaring map has only phi(323)=288 points in total and
# a training split can never contain more than ~250 of them.  That makes it
# impossible to tell "the representation cannot express the algorithm" apart from
# "there is not enough data to identify the algorithm".  These two datasets hold
# the modulus fixed (N = 101*103 = 10403, 5 digits, phi = 10200) and vary ONLY the
# number of training prompts, with T=1 inside the training range so rung-1 needs
# no T-extrapolation.
#
#   rp_small : 250 prompts/setting  (e1-like data volume)
#   rp_big   : 3000 prompts/setting (12x more of exactly the same arithmetic)
#
# COMPLIANCE: runs the public generator only.  Generated rows are never read,
# printed or summarized.
set -euo pipefail
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python

for n in 250 3000; do
  case $n in
    250)  name=rp_small; ood=250 ;;
    3000) name=rp_big;   ood=1000 ;;
  esac
  "$V" -m data.squaring_mod \
    --output_dir "data/generated/repr_probe_${name}_n10403_t123" \
    --fixed_p 101 --fixed_q 103 \
    --time_steps '[1,2,3]' --ood_time_steps '[6]' \
    --examples_per_setting "$n" --ood_examples_per_setting "$ood" \
    --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
    --depth_evaluation_examples_per_setting 256 \
    --ood_n_depth_evaluation_modulus_bits '[16,18]' \
    --ood_n_depth_evaluation_examples_per_setting 256 \
    --train_fraction 0.8 --test_fraction 0.2 \
    --split_group prompt --seed 45 --separate_input_output true
done
