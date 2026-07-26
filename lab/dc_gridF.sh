#!/usr/bin/env bash
# Grid F -- the recipe that reaches the top of the ladder, and its ablations.
# $1 = easy | medium | heads
#
# RECIPE R = counted halting
#            + --no-dump   unspent halting mass is NOT piled on the last index.
#                          With the dump, "never halt" is exactly correct for
#                          the deepest training T (no gradient), and the dump
#                          also makes the consistency law UNSATISFIABLE: it
#                          turns the law into 1-p(r) = p(r-1) along the chain.
#            + --reg-hard  straight-through one-hot digit register.
#            + --cons 0.1  self-consistency w(inc r) = shift_right(w(r)) over
#                          the model's own learned increment orbit.  Its first
#                          component is w(inc r)[0] = 0, i.e. p(r) = 0 at every
#                          successor register -- the leak-suppression constraint
#                          the cross-entropy cannot see.
#
# N = 329 = 7*47, lambda = lcm(6,46) = 138, 7/7 distinct ladder classes.
# (e1: N=323, lambda=144, collapses {4,16,64} and {8,32} -> 4 distinct maps.)
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
mkdir -p lab/logs lab/runs
OUT=lab/runs/gridF.jsonl
base="--modulus 329 --sel-steps 1500 --sel-anneal 900 --sel-log 1500 --seeds 0 1 2 --eval-loops 64 --n-eval 76 --out $OUT"
CONS="--cons 0.1 --cons-space w --cons-j 8 --cons-loops 4"
R="--no-dump --reg-hard $CONS"

run () {
  tag=$1; shift
  echo "=== $tag"
  $V lab/probe_depth.py $base --tag "$tag" "$@" > "lab/logs/F_$tag.log" 2>&1
  grep -E "fitted loc" "lab/logs/F_$tag.log"
  sed -n '/=== summary ===/,$p' "lab/logs/F_$tag.log"
}

case "${1:-easy}" in
easy)
  run F1_easy_R           --selector counter2 --train-t 1 2 3 --train-loops 3 $R
  run F2_easy_nocons      --selector counter2 --train-t 1 2 3 --train-loops 3 --no-dump --reg-hard
  run F3_easy_nohard      --selector counter2 --train-t 1 2 3 --train-loops 3 --no-dump $CONS
  run F4_easy_R_evalhard  --selector counter2 --train-t 1 2 3 --train-loops 3 $R --eval-hard
  run F5_easy_R_dump      --selector counter2 --train-t 1 2 3 --train-loops 3 --reg-hard $CONS
  run F12_easy_R_thresh0  --selector counter2 --train-t 1 2 3 --train-loops 3 $R --thresh-init 0.0
  echo ALL_DONE_F_EASY ;;
medium)
  run F6_med_R            --selector counter2 --train-t 4 8 16 --train-loops 16 $R --cons-jd 3
  run F7_med_nocons       --selector counter2 --train-t 4 8 16 --train-loops 16 --no-dump --reg-hard
  run F8_med_R_evalhard   --selector counter2 --train-t 4 8 16 --train-loops 16 $R --cons-jd 3 --eval-hard
  run F13_med_R_thresh0   --selector counter2 --train-t 4 8 16 --train-loops 16 $R --cons-jd 3 --thresh-init 0.0
  echo ALL_DONE_F_MEDIUM ;;
heads)
  # does self-consistency rescue a head whose parameters ARE indexed by
  # (place of T, digit)?  cause (a) says no amount of anchoring can fix the
  # unseen (place, digit) pairs; the consistency chain says it can.
  run F9_easy_placev_cons  --selector placev --train-t 1 2 3 --train-loops 3 --cons 0.1 --cons-space loc --cons-j 63
  run F10_easy_mlp_cons    --selector mlp    --train-t 1 2 3 --train-loops 3 --cons 0.1 --cons-space loc --cons-j 63
  run F11_med_placev_cons  --selector placev --train-t 4 8 16 --train-loops 16 --cons 0.1 --cons-space loc --cons-j 48 --cons-jd 3
  run F14_easy_placev_ctrl --selector placev --train-t 1 2 3 --train-loops 3
  echo ALL_DONE_F_HEADS ;;
esac
