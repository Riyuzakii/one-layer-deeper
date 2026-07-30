#!/usr/bin/env bash
# PLAN2 Phase-0: probe-4 third seed, the m1 tier, and the collapse detector.
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

for a in diag01 diagpm1 delta01 deltapm1 lstm gru attn mlp; do
  run "${a}_d128_L2" p2_e5_fs1200_s13 P4-solvability "e5 fs1200 seed 13, ${a}"
done

for a in lstm deltapm1 diag01 attn; do
  run "${a}_d128_L2" p2_m1_fs1200_s74 P1-expressivity "m1 fs1200 seed 74, ${a}"
done

# collapse detector (DIAGNOSTIC): output diversity over held-out cohorts
for a in diag01 deltapm1 lstm attn mlp; do
  $VENV lab/p2_diversity.py \
    --submission "submissions/plan2-phase0/${a}_d128_L2/submission.py" \
    --manifest lab/manifests/p2_e5_fs1200_s74.json --steps 1200 --seed 74 \
    --note "collapse detector, e5, ${a}" >/dev/null || echo "!!! div failed ${a}"
done

echo "=== p2 queue4 complete $(date +%H:%M:%S) ==="
