#!/usr/bin/env bash
# Phase 2B: learnability ceiling + optimizer, L=8, equal steps (2000) on 600s manifest.
set -u
cd /home/scratch.arohan_hw/git/one-layer-deeper
export PATH="$HOME/.local/bin:$PATH"
M="lab/manifests/lab_e1_long.json"
echo "=== L8 AdamW ce, 2000 steps ==="
uv run python lab/run_experiment.py --submission submissions/exp_axis/L8_d128_adamw_ce_lr0.001/submission.py --manifest "$M" --tag B-learn --note "L8 adamw ce 2000 steps: learnability ceiling" 2>&1 | tail -2
echo "=== L8 Muon ce, 2000 steps ==="
uv run python lab/run_experiment.py --submission submissions/exp_axis/L8_d128_muon_ce_lr0.02/submission.py --manifest "$M" --tag C-opt --note "L8 muon ce 2000 steps: vs adamw equal steps" 2>&1 | tail -2
echo "=== PHASE2B_DONE ==="
