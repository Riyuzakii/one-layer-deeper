#!/usr/bin/env bash
# Phase 2A depth sweep: weight-tied recurrence L in {1,2,4,8,16,32} on real e1 (60s).
# Serves Phase 1 (step-time vs depth) and Phase 2A (accuracy vs depth, incl OOD-T).
set -u
cd /home/scratch.arohan_hw/git/one-layer-deeper
export PATH="$HOME/.local/bin:$PATH"
MANIFEST="${MANIFEST:-h100_easy_e1}"
for L in 1 2 4 8 16 32; do
  echo "=== L=$L on $MANIFEST ==="
  uv run python lab/run_experiment.py \
    --submission "submissions/exp_recur/L${L}_d128/submission.py" \
    --manifest "$MANIFEST" \
    --tag "A-depth" \
    --note "weight-tied recurrence L=${L} d=128, 60s wall-clock on B300 (~2x H100 steps)" \
    2>&1 | tail -2
done
echo "=== SWEEP_DONE ==="
