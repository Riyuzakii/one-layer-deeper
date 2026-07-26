#!/usr/bin/env bash
# Eval-budget re-measurement after the log-space detector landed, plus the two
# tier-faithful full runs (real training budget, not eval-only).
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
S=submissions/depth-controller
$V lab/run_experiment.py --submission $S/fixed16/submission.py \
   --manifest lab/manifests/dc_e1_evalonly.json --tag A-evalbudget \
   --note "log detector, fixed16 = 256 iterations ~ a correctly trained controller (Easy)" 2>&1 | tail -5
$V lab/run_experiment.py --submission $S/submission.py \
   --manifest lab/manifests/dc_m1_evalonly.json --tag A-evalbudget \
   --note "log detector, learned early exit (Medium eval-only)" 2>&1 | tail -5
$V lab/run_experiment.py --submission $S/fixed16/submission.py \
   --manifest lab/manifests/dc_m1_evalonly.json --tag A-evalbudget \
   --note "log detector, fixed16 = 256 iterations (Medium)" 2>&1 | tail -5
$V lab/run_experiment.py --submission $S/submission.py \
   --manifest lab/manifests/dc_e1_wc60.json --tag B-tierfaithful --timeout 1200 \
   --note "TIER-FAITHFUL Easy 60s: full training budget then the 16 scoring splits" 2>&1 | tail -5
$V lab/run_experiment.py --submission $S/submission.py \
   --manifest lab/manifests/dc_m1_wc600.json --tag B-tierfaithful --timeout 2400 \
   --note "TIER-FAITHFUL Medium 600s" 2>&1 | tail -5
echo ALL_DONE_EVALBUDGET2
