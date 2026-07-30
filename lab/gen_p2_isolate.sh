#!/usr/bin/env bash
# PLAN2 Phase-0: the harness / loss / label-alignment control that s6 demands.
#
# `trapdoor_squaring_mod` computes pow(x, pow(2, T, phi), N).  At T=0 the
# exponent is pow(2,0,phi) = 1, so the answer is x itself: the SAME prompt
# format, the SAME tokenizer, the SAME collate, the SAME target_positions
# slicing and the SAME loss -- but the target is a pure digit copy.  This is
# derived from the generator source (data/squaring_mod.py:409-415), not from any
# dataset row.
#
#   cp*  T=0   answer = x                 -> parsing + copy only
#   sq*  T=1   answer = x^2 mod N         -> exactly one squaring, same operands
#
# The pair isolates "the arithmetic" from everything else in the pipeline.  If a
# model reaches ~1.0 held-out on cp* and ~0 on sq*, the harness, the loss and the
# label alignment are all fine and PLAN2 s6's "stop and re-examine" branch is
# closed by measurement.
#
# Only one T setting is present, so every prompt carries a distinct x: the
# `test` split is therefore an UNSEEN-OPERAND measurement, and so are the 256
# reserved units behind the depth ladder.
#
# COMPLIANCE: runs the public generator only; generated rows are never read.
set -euo pipefail

gen () {  # gen <tag> <p> <q> <T> <examples_per_setting>
  local tag=$1 p=$2 q=$3 t=$4 eps=$5
  local out="data/generated/p2iso_${tag}"
  if [ -f "${out}/dataset_config.json" ]; then echo "skip ${tag} (exists)"; return; fi
  echo "=== generating ${tag}: p=${p} q=${q} T=${t} eps=${eps} ==="
  python -m data.squaring_mod \
    --output_dir "${out}" \
    --fixed_p "${p}" --fixed_q "${q}" \
    --time_steps "[${t}]" --ood_time_steps '[3]' \
    --examples_per_setting "${eps}" --ood_examples_per_setting 100 \
    --depth_evaluation_time_steps '[1,2]' \
    --depth_evaluation_examples_per_setting 256 \
    --depth_evaluation_exhaustive_x false \
    --train_fraction 0.8 --test_fraction 0.2 \
    --split_group prompt --seed 45 --separate_input_output true
}

gen cp3  23   29  0  250     # N=667,   copy, 3 digits
gen sq3  23   29  1  250     # N=667,   one squaring, 3 digits
gen cp5 101  103  0 1000     # N=10403, copy, 5 digits
gen sq5 101  103  1 1000     # N=10403, one squaring, 5 digits
gen cp7 1009 1013 0 4000     # N=1022117, copy, 7 digits
gen sq7 1009 1013 1 4000     # N=1022117, one squaring, 7 digits

echo "=== p2 isolate pair complete ==="
