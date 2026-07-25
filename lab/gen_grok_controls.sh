#!/usr/bin/env bash
# Diagnostic controls for the grok-optimization axis.
#
# e1 asks the model to learn a UNARY map on a fixed modulus: for T=1 the target
# depends only on x, and the whole universe is phi(323)=288 (x, x^2 mod 323)
# facts of which ~200 are in train and 38 are the held-out rung-1 cohort.  When
# no training recipe moves rung-1 off zero, the natural question is whether the
# blocker is (a) the amount of arithmetic per fact (digits of N) or (b) the
# number of facts available to force an algorithm rather than a lookup table.
#
# These two controls vary both in opposite directions while keeping the e1
# recipe (fixed N, T in {1,2,3}, OOD T=6, exhaustive-x depth ladder, 80/20)
# otherwise identical:
#
#   gsmall : p=7,  q=11 -> N=77   (2-digit), 60 units,  ~40 train facts per T
#   e1     : p=17, q=19 -> N=323  (3-digit), 288 units, ~200 train facts per T
#   gbig   : p=31, q=37 -> N=1147 (4-digit), 1080 units,~800 train facts per T
#
# COMPLIANCE: runs the public generator only.  Generated rows are never read,
# printed or summarized.
set -euo pipefail
cd /home/scratch.arohan_hw/git/one-layer-deeper/.worktrees/grok-optimization
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python

$V -m data.squaring_mod \
  --output_dir data/generated/grokctl_small_n77_t123 \
  --fixed_p 7 --fixed_q 11 \
  --time_steps '[1,2,3]' --ood_time_steps '[6]' \
  --examples_per_setting 50 --ood_examples_per_setting 20 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_exhaustive_x true \
  --ood_n_depth_evaluation_modulus_bits '[9,10]' \
  --ood_n_depth_evaluation_examples_per_setting 128 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

$V -m data.squaring_mod \
  --output_dir data/generated/grokctl_big_n1147_t123 \
  --fixed_p 31 --fixed_q 37 \
  --time_steps '[1,2,3]' --ood_time_steps '[6]' \
  --examples_per_setting 1000 --ood_examples_per_setting 300 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_exhaustive_x true \
  --ood_n_depth_evaluation_modulus_bits '[13,14]' \
  --ood_n_depth_evaluation_examples_per_setting 256 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true
