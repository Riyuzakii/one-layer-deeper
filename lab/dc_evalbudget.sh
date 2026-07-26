#!/usr/bin/env bash
# Eval-budget measurements for the depth controller.
#   main    : learned early exit -- stops as soon as the remaining halting mass
#             can no longer beat the running maximum.  At random init that is
#             ~2 iterations per example, which is what an evaluator actually
#             sees today.
#   fixed16 : LAB VARIANT, 16 fixed loops and no early exit -- 16x16 = 256
#             iterations over the 16 scoring splits, which is the count a
#             CORRECTLY TRAINED controller spends (2 x (1+2+4+8+16+32+64)
#             + test + ood ~ 260).
#   fixed64 : LAB VARIANT, the fixed-grid readout the early exit replaces.
# Every number here is taken with sibling jobs on the GPU and is therefore
# CONTENTION-PESSIMISTIC; the ratios are the transferable part.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
S=submissions/depth-controller
for cell in "$S/fixed64/submission.py dc_e1_evalonly fixed-64-loop readout, no early exit (Easy)" \
            "$S/submission.py dc_m1_evalonly learned early exit (Medium)" \
            "$S/fixed16/submission.py dc_m1_evalonly 256-iteration trained-controller proxy (Medium)" \
            "$S/fixed64/submission.py dc_m1_evalonly fixed-64-loop readout (Medium)"; do
  set -- $cell; sub=$1; man=$2; shift 2
  echo "=== $sub on $man"
  $V lab/run_experiment.py --submission "$sub" --manifest "lab/manifests/$man.json" \
     --tag A-evalbudget --timeout 2400 --note "$*" 2>&1 | tail -6
done
echo ALL_DONE_EVALBUDGET
