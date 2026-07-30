#!/usr/bin/env bash
# PLAN2 Phase-0 probes 2 (length scaling) and 3 (train-vs-eval transition).
# Two fixed models: deltapm1 (the best fitter in the family) and lstm (maximal
# expressivity).  attn is excluded from the grid because at 1200 steps it does
# not fit even e5 (train 0.049), which would confound "cannot fit" with "not
# enough steps".
set -uo pipefail
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.."

run () {
  echo "### $(date +%H:%M:%S)  $1  $2  [$3]"
  $VENV lab/run_experiment.py \
    --submission "submissions/plan2-phase0/$1/submission.py" \
    --manifest "lab/manifests/$2.json" --tag "$3" --note "$4" --timeout 7200 \
    || echo "!!! failed: $1 $2"
}

# length ladder (examples_per_setting fixed at 250) -- probe 2
for d in n3_e250 n4_e250 n5_e250 n6_e250 n7_e250; do
  run deltapm1_d128_L2 "p2_${d}_fs1200_s74" P23-grid "grid ${d}, deltapm1"
done
for d in n3_e250 n4_e250 n5_e250 n6_e250 n7_e250; do
  run lstm_d128_L2 "p2_${d}_fs1200_s74" P23-grid "grid ${d}, lstm"
done
# data-size ladders at fixed modulus -- probe 3
for d in n5_e1000 n5_e4000 n5_e9000 n7_e1000 n7_e16000; do
  run deltapm1_d128_L2 "p2_${d}_fs1200_s74" P23-grid "grid ${d}, deltapm1"
  run lstm_d128_L2 "p2_${d}_fs1200_s74" P23-grid "grid ${d}, lstm"
done
echo "=== p2 queue3 complete $(date +%H:%M:%S) ==="
