#!/usr/bin/env bash
# PLAN2 Phase-0 probes 2 and 3: a 2-D grid over (modulus size, training-set size).
#
# Probe 2 (length scaling) and probe 3 (train-vs-eval transition) are the same
# grid read along two axes.  On this task sequence length CANNOT be varied
# independently of modulus size -- the prompt is [N] d(N) [X] d(x) [T] d(T), so
# max_seq_len = 2*digits(N) + 5 once the depth ladder reaches T=64 (2 digits).
# The length ladder therefore also moves the operand-space size; the data-size
# ladder at fixed N is what separates the two.
#
# Every dataset carries the full T = 1,2,4,8,16,32,64 certification ladder with
# 256 RESERVED units per rung.  `depth_evaluation_exhaustive_x false` +
# fixed_p/fixed_q means those 256 x values are excluded from train/test/ood, so
# rung-1 accuracy is a clean *unseen-operand* measurement, while the `test`
# split is held-out prompts whose operands may recur at another T.  The OOD-N
# ladder is deliberately omitted so that max_seq_len is set by the ID modulus
# alone.
#
# COMPLIANCE: this only runs the public generator.  Generated rows are never
# read, printed, or summarized -- data/generated/ is write-only for us.
#
# Usage:  PATH=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin:$PATH \
#           bash lab/gen_p2_grid.sh
set -euo pipefail

gen () {  # gen <tag> <p> <q> <examples_per_setting>
  local tag=$1 p=$2 q=$3 eps=$4
  local out="data/generated/p2grid_${tag}"
  if [ -f "${out}/dataset_config.json" ]; then echo "skip ${tag} (exists)"; return; fi
  echo "=== generating ${tag}: p=${p} q=${q} eps=${eps} ==="
  python -m data.squaring_mod \
    --output_dir "${out}" \
    --fixed_p "${p}" --fixed_q "${q}" \
    --time_steps '[1,2,3]' --ood_time_steps '[6]' \
    --examples_per_setting "${eps}" --ood_examples_per_setting 100 \
    --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
    --depth_evaluation_examples_per_setting 256 \
    --depth_evaluation_exhaustive_x false \
    --train_fraction 0.8 --test_fraction 0.2 \
    --split_group prompt --seed 45 --separate_input_output true
}

# --- length ladder: examples_per_setting fixed at 250 (600 train rows) --------
# tag   p     q      N          digits  phi(N)     max_seq_len
gen n3_e250   23   29   250   # N=667      3  616        11
gen n4_e250   31   37   250   # N=1147     4  1080       13
gen n5_e250  101  103   250   # N=10403    5  10200      15
gen n6_e250  331  337   250   # N=111547   6  110880     17
gen n7_e250 1009 1013   250   # N=1022117  7  1020096    19

# --- data-size ladder at N=10403 (5 digits, phi=10200) -----------------------
gen n5_e1000  101 103  1000
gen n5_e4000  101 103  4000
gen n5_e9000  101 103  9000

# --- data-size ladder at N=1022117 (7 digits, phi=1020096) -------------------
gen n7_e1000  1009 1013  1000
gen n7_e16000 1009 1013 16000

echo "=== p2 grid complete ==="
