#!/usr/bin/env bash
# Variance floor: L=8 d128 AdamW.
#  (a) e1 x3 repeats -> pure noise floor (wall-clock jitter + GPU nondeterminism;
#      OOD eval is ~100 examples so accuracy is quantized in 0.01 steps).
#  (b) e1..e5 x1 -> cross-dataset spread.
# Any effect (e.g. the depth->OOD trend) smaller than (a) is not real.
set -u
cd /home/scratch.arohan_hw/git/one-layer-deeper
export PATH="$HOME/.local/bin:$PATH"
SUB="submissions/exp_recur/L8_d128/submission.py"
for i in 1 2 3; do
  echo "=== e1 repeat $i ==="
  uv run python lab/run_experiment.py --submission "$SUB" --manifest h100_easy_e1 \
    --tag var-e1 --note "L8 d128 e1 repeat $i (noise floor)" 2>&1 | tail -1
done
for E in e2 e3 e4 e5; do
  echo "=== $E ==="
  uv run python lab/run_experiment.py --submission "$SUB" --manifest "h100_easy_$E" \
    --tag var-cross --note "L8 d128 $E (cross-dataset)" 2>&1 | tail -1
done
echo "=== VARFLOOR_DONE ==="
