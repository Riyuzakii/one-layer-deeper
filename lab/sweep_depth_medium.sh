#!/usr/bin/env bash
# Phase 2A on Medium m1 (T{4,8,16}, OOD T=32): if depth->OOD holds, the knee should
# move much deeper than Easy's L~8. Medium budget = 600s (10x Easy) -> more steps.
# LOOPS and MANIFEST overridable; default a focused subset to bound GPU time.
set -u
cd /home/scratch.arohan_hw/git/one-layer-deeper
export PATH="$HOME/.local/bin:$PATH"
MANIFEST="${MANIFEST:-h100_medium_m1}"
LOOPS="${LOOPS:-8 16 32 64}"
for L in $LOOPS; do
  echo "=== L=$L on $MANIFEST ==="
  uv run python lab/run_experiment.py \
    --submission "submissions/exp_recur/L${L}_d128/submission.py" \
    --manifest "$MANIFEST" --tag A-depth-med \
    --note "recurrence L=${L} d=128 on ${MANIFEST} (deeper OOD test)" 2>&1 | tail -2
done
echo "=== SWEEP_MED_DONE ==="
