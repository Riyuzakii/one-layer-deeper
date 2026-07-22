#!/usr/bin/env bash
# Medium Axis A confirmation on m1 (fixed N=10403, ID T{4,8,16}, OOD T=32, 3000-ex eval -> high SNR).
# Fixed 800 steps x seeds{74,1}, L in {4,16,64}. Tests whether deep-T OOD needs deep L, cleanly.
set -u
cd /home/scratch.arohan_hw/git/one-layer-deeper
export PATH="$HOME/.local/bin:$PATH"
declare -A MAN=( [74]="lab/manifests/lab_m1_long_s74.json" [1]="lab/manifests/lab_m1_long_s1.json" )
for L in 4 16 64; do
  for S in 74 1; do
    echo "=== L=$L seed=$S ==="
    uv run python lab/run_experiment.py \
      --submission "submissions/exp_axis/L${L}_d128_adamw_ce_lr0.001/submission.py" \
      --manifest "${MAN[$S]}" --tag A-medium \
      --note "m1 L=${L} seed=${S} fixed800 (Medium Axis A, OOD T=32)" 2>&1 | tail -1
  done
done
echo "=== MEDIUM_GRID_DONE ==="
