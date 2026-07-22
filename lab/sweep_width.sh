#!/usr/bin/env bash
# Capability push: does width break the D=128 plateau? L=8 AdamW, e1@60s (overhead-bound
# so still ~400 steps even wide; GPU is 96% idle -> width is ~free on wall-clock).
set -u
cd /home/scratch.arohan_hw/git/one-layer-deeper
export PATH="$HOME/.local/bin:$PATH"
for D in 256 512 768; do
  echo "=== D=$D ==="
  uv run python lab/run_experiment.py \
    --submission "submissions/exp_axis/L8_d${D}_adamw_ce_lr0.001/submission.py" \
    --manifest h100_easy_e1 --tag H-width \
    --note "L8 d=${D} adamw ce, width screen e1@60s" 2>&1 | tail -2
done
echo "=== SWEEP_WIDTH_DONE ==="
