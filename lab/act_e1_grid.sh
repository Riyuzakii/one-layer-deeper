#!/usr/bin/env bash
# Adaptive (PonderNet-lite) vs fixed-L on e1, fixed 400 steps x seeds{74,1,2}.
set -u
cd /home/scratch.arohan_hw/git/one-layer-deeper
export PATH="$HOME/.local/bin:$PATH"
declare -A MAN=( [74]="lab/manifests/lab_e1_long.json" [1]="lab/manifests/lab_e1_long_s1.json" [2]="lab/manifests/lab_e1_long_s2.json" )
for SUB in ACT_ML32_d128_b0.01_lr0.001 ACT_ML32_d128_b0_lr0.001; do
  for S in 74 1 2; do
    echo "=== $SUB seed=$S ==="
    uv run python lab/run_experiment.py \
      --submission "submissions/exp_adaptive/${SUB}/submission.py" \
      --manifest "${MAN[$S]}" --tag F-adaptive \
      --note "${SUB} seed=${S} fixed400 e1 (adaptive vs fixed-L)" 2>&1 | tail -1
  done
done
echo "=== ACT_GRID_DONE ==="
