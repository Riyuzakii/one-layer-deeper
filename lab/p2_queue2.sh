#!/usr/bin/env bash
# PLAN2 Phase-0, second queue: the harness/label-alignment control (PLAN2 s6)
# and the honest tuning pass that the kill criterion requires before a null on
# the non-linear RNN can be called.
set -uo pipefail

VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.."

run () {  # run <arch-dir> <manifest> <tag> <note>
  echo "### $(date +%H:%M:%S)  $1  $2  [$3]"
  $VENV lab/run_experiment.py \
    --submission "submissions/plan2-phase0/$1/submission.py" \
    --manifest "lab/manifests/$2.json" \
    --tag "$3" --note "$4" --timeout 7200 || echo "!!! failed: $1 $2"
}

# ---------------------------------------------------------------- batch G ----
# cp<d> (T=0, answer = x, a pure digit copy) vs sq<d> (T=1, one squaring) at the
# same modulus, same operands, same pipeline.  Isolates "the arithmetic".
for d in cp3 sq3 cp5 sq5 cp7 sq7; do
  for a in attn lstm; do
    run "${a}_d128_L2" "p2_${d}_fs1200_s74" P0-harness "isolate ${d}, ${a}"
  done
done
for d in cp5 sq5; do
  run deltapm1_d128_L2 "p2_${d}_fs1200_s74" P0-harness "isolate ${d}, deltapm1"
done

# ---------------------------------------------------------------- batch F ----
# One honest tuning pass on the kill-criterion model before calling it null.
for tag in lstm_lr3e4 lstm_lr3e3 gru_lr3e4 deltapm1_lr3e4; do
  run "${tag}" p2_e5_fs1200_s74 P1-expressivity "tuning pass, e5, ${tag}"
done

echo "=== p2 queue2 complete $(date +%H:%M:%S) ==="
