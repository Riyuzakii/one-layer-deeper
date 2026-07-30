#!/usr/bin/env bash
# PLAN2 Phase-0 audit fold-in: the tied-head embedding-init bug.
#
# nn.Embedding's default init is N(0,1); tying it to the head makes step-1 CE
# ~80 instead of ln(17)=2.83, so the first ~100 optimizer steps are spent
# unlearning the init.  The hosted Hard run shows step-1 loss 79.936, so this is
# in every result this project has ever produced.  EMB_INIT=0.02 fixes it.
# Question: does it change anything that matters?
set -uo pipefail
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.."

run () {
  echo "### $(date +%H:%M:%S)  $1  $2  [$3]"
  $VENV lab/run_experiment.py \
    --submission "submissions/plan2-phase0/$1/submission.py" \
    --manifest "lab/manifests/$2.json" --tag "$3" --note "$4" --timeout 7200 \
    || echo "!!! failed: $1 $2"
}

for a in deltapm1 lstm attn diag01; do
  run "${a}_d128_L2_emb02" p2_e5_fs1200_s74 P0-initfix "emb init 0.02, e5 seed 74, ${a}"
done
run lstm_d128_L2_emb02 p2_sq5_fs1200_s74 P0-initfix "emb init 0.02, sq5, lstm"
run lstm_d128_L2_emb02 p2_e5_fs8000_s74  P0-initfix "emb init 0.02, e5 fs8000, lstm"
# fitting-curve calibration: a null at an uncalibrated step count is not a null
run lstm_d128_L2 p2_m1_fs8000_s74 P1-expressivity "m1 fs8000 seed 74, lstm -- fitting curve"
run deltapm1_d128_L2 p2_m1_fs8000_s74 P1-expressivity "m1 fs8000 seed 74, deltapm1 -- fitting curve"
echo "=== p2 queue5 complete $(date +%H:%M:%S) ==="
