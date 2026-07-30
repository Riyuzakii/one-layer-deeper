#!/usr/bin/env bash
# PLAN2 Phase-0 run queue.  Ordered by importance: if it is killed part-way, the
# results that matter are already in lab/archive.jsonl.
#
# Every cell is LEGAL (a plain evaluator run of a from-random-init submission on
# a fixed_step manifest).  The lr=0 batch is the BRIEF2 s6.1 control.
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

ARCHES="diag01 diagpm1 delta01 deltapm1 lstm gru attn mlp"

# ---------------------------------------------------------------- batch B ----
# lr=0 controls.  With lr=0 AdamW's decay term is also zero, so 200 steps is
# exactly the random-init model; this is the reference every "improvement" must
# be read against (BRIEF2 s6.1).
for a in $ARCHES; do
  run "${a}_d128_L2_lr0" p2_e5_fs200_s74 P0-lr0 "lr=0 control, e5, ${a}"
done

# ---------------------------------------------------------------- batch A ----
# Probe 4 (solvability ordering) + probe 1 (expressivity vs optimization).
# e5 = 512-example rungs, sampled 10/11-bit moduli.  3 seeds each.
for s in 74 7 13; do
  for a in $ARCHES; do
    run "${a}_d128_L2" "p2_e5_fs1200_s${s}" P4-solvability "e5 fs1200 seed ${s}, ${a}"
  done
done

# ---------------------------------------------------------------- batch E ----
# Probes 2 (length scaling) and 3 (train-vs-eval transition), fixed model.
for d in n3_e250 n4_e250 n5_e250 n6_e250 n7_e250 n5_e1000 n5_e4000 n5_e9000 n7_e1000 n7_e16000; do
  run attn_d128_L2 "p2_${d}_fs1200_s74" P23-grid "grid ${d}, attn"
done
for d in n3_e250 n4_e250 n5_e250 n6_e250 n7_e250; do
  run lstm_d128_L2 "p2_${d}_fs1200_s74" P23-grid "grid ${d}, lstm"
done

# ---------------------------------------------------------------- batch C ----
# m1 (14-bit fixed N=10403, T in {4,8,16}) -- the tier where BRIEF2 s2d says
# models cannot even fit.
for a in lstm deltapm1 attn; do
  run "${a}_d128_L2" p2_m1_fs1200_s74 P1-expressivity "m1 fs1200 seed 74, ${a}"
done

# ---------------------------------------------------------------- batch D ----
# Step-count insurance: is 1200 steps simply too few?
for a in lstm deltapm1; do
  run "${a}_d128_L2" p2_e5_fs8000_s74 P1-expressivity "e5 fs8000 seed 74, ${a}"
done

echo "=== p2 queue complete $(date +%H:%M:%S) ==="
