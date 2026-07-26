#!/usr/bin/env bash
# Grid G -- stabilising the recipe.  Grid F reaches MAX_T = 64 on 2 of 3 seeds;
# the failing seeds are diagnosed exactly:
#   * `sum` detector: the ladder rides on the single scalar gain*(thresh-1).
#     Measured across 10 cells: >3 -> T=64, <2 -> T=2.  The `log` detector
#     scores the same conjunction with margin ~14 instead of 1.
#   * `one` converging to the wrong digit (argmax 2 or 6 instead of 1), i.e.
#     the countdown step itself is wrong.  The consistency term PROPAGATES an
#     anchor, so applying it before the anchor exists can lock in a bad basin;
#     --cons-start delays it.
# $1 = easy | medium
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
mkdir -p lab/logs lab/runs
OUT=lab/runs/gridG.jsonl
base="--modulus 329 --sel-log 1500 --seeds 0 1 2 3 4 --eval-loops 64 --n-eval 76 --out $OUT"
CONS="--cons 0.1 --cons-space w --cons-j 8 --cons-loops 4"

run () {
  tag=$1; shift
  echo "=== $tag"
  $V lab/probe_depth.py $base --tag "$tag" "$@" > "lab/logs/G_$tag.log" 2>&1
  grep -E "fitted loc" "lab/logs/G_$tag.log"
  sed -n '/=== summary ===/,$p' "lab/logs/G_$tag.log"
}

E="--selector counter2 --train-t 1 2 3 --train-loops 3 --no-dump --reg-hard"
M="--selector counter2 --train-t 4 8 16 --train-loops 16 --no-dump --reg-hard"
case "${1:-easy}" in
easy)
  run G1_easy_log_warm  $E --detector log $CONS --cons-start 400 --sel-steps 2000 --sel-anneal 900
  run G2_easy_log_lr03  $E --detector log $CONS --cons-start 400 --sel-steps 2000 --sel-anneal 900 --sel-lr 0.3
  run G3_easy_sum_warm  $E --detector sum $CONS --cons-start 400 --sel-steps 2000 --sel-anneal 900
  run G4_easy_log_nocons $E --detector log --sel-steps 2000 --sel-anneal 900
  echo ALL_DONE_G_EASY ;;
medium)
  run G5_med_log_warm   $M --detector log $CONS --cons-jd 3 --cons-start 400 --sel-steps 2000 --sel-anneal 900
  run G6_med_log_lr03   $M --detector log $CONS --cons-jd 3 --cons-start 400 --sel-steps 2000 --sel-anneal 900 --sel-lr 0.3
  run G7_med_log_nocons $M --detector log --sel-steps 2000 --sel-anneal 900
  echo ALL_DONE_G_MEDIUM ;;
esac
