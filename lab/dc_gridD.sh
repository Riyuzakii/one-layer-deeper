#!/usr/bin/env bash
# Grid D -- the recipe at m1's real modulus and at a 22-bit Hard-proxy modulus.
#   m1  : N = 10403 = 101*103, lambda = lcm(100,102) = 5100, 7/7 distinct.
#   hp1 : N = 4028033 = 2003*2011, lambda = 2012010, 7/7 distinct (S = 7).
# Both tiers' T values are run at each, because certification is a prefix from
# T = 1 and Medium/Hard never see T = 1 or T = 2 in training.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
mkdir -p lab/logs lab/runs
OUT=lab/runs/gridD.jsonl
CONS="--cons 0.1 --cons-space w --cons-j 8 --cons-loops 4"
R="--no-dump --reg-hard $CONS"
base="--sel-steps 1500 --sel-anneal 900 --sel-log 1500 --seeds 0 1 2 --eval-loops 64 --out $OUT"

run () {
  tag=$1; shift
  echo "=== $tag"
  $V lab/probe_depth.py $base --tag "$tag" "$@" > "lab/logs/D_$tag.log" 2>&1
  grep -E "fitted loc|lambda" "lab/logs/D_$tag.log" | head -4
  sed -n '/=== summary ===/,$p' "lab/logs/D_$tag.log"
}

M1="--modulus 10403 --n-train 250 --n-eval 128"
run D1_m1_easy_R      $M1 --selector counter2 --train-t 1 2 3  --train-loops 3  $R
run D2_m1_med_R       $M1 --selector counter2 --train-t 4 8 16 --train-loops 16 $R --cons-jd 3
run D3_m1_med_nocons  $M1 --selector counter2 --train-t 4 8 16 --train-loops 16 --no-dump --reg-hard
run D4_m1_med_R_hard  $M1 --selector counter2 --train-t 4 8 16 --train-loops 16 $R --cons-jd 3 --eval-hard

HP="--modulus 4028033 --n-train 200 --n-eval 64"
run D5_hp1_med_R      $HP --selector counter2 --train-t 4 8 16 --train-loops 16 $R --cons-jd 3
echo ALL_DONE_GRIDD
