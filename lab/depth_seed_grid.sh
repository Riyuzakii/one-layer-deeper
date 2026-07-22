#!/usr/bin/env bash
# Rigorous Axis A: L in {2,8,32} x seeds {74,1,2}, FIXED 400 steps (no wall-clock jitter).
# Multi-seed error bars answer: does depth's OOD benefit survive the noise floor?
set -u
cd /home/scratch.arohan_hw/git/one-layer-deeper
export PATH="$HOME/.local/bin:$PATH"
declare -A MAN=( [74]="lab/manifests/lab_e1_long.json" [1]="lab/manifests/lab_e1_long_s1.json" [2]="lab/manifests/lab_e1_long_s2.json" )
for L in 2 8 32; do
  for S in 74 1 2; do
    echo "=== L=$L seed=$S ==="
    uv run python lab/run_experiment.py \
      --submission "submissions/exp_axis/L${L}_d128_adamw_ce_lr0.001/submission.py" \
      --manifest "${MAN[$S]}" --tag A-grid \
      --note "L=${L} seed=${S} fixed400 (Axis A multi-seed)" 2>&1 | tail -1
  done
done
echo "=== GRID_DONE ==="
