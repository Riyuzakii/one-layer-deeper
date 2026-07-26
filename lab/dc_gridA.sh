#!/usr/bin/env bash
# Grid A -- Easy tier (train T = {1,2,3}) at N=329 (lambda=138, 7/7 distinct
# ladder classes, so no rung collapses; e1/e2 have only 4 distinct maps and
# cannot measure a depth controller at all).
# Question: is the MAX_T = 2 ceiling caused by the T-digit coverage bound, or by
# training the mixture at depth L = max(train T), where "never halt" is the
# exact optimum for the deepest training T and therefore carries no gradient?
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
mkdir -p lab/logs lab/runs
OUT=lab/runs/gridA.jsonl
common="--modulus 329 --train-t 1 2 3 --sel-steps 2000 --sel-anneal 1200 --sel-log 1000 --seeds 0 1 2 --eval-loops 64 --n-eval 76 --out $OUT"

run () {
  tag=$1; shift
  echo "=== $tag"
  $V lab/probe_depth.py $common --tag "$tag" "$@" > "lab/logs/A_$tag.log" 2>&1
  grep -E "MAX_T =|fitted loc" "lab/logs/A_$tag.log"
  sed -n '/=== summary ===/,$p' "lab/logs/A_$tag.log"
}

run A1_counter2_L3  --selector counter2 --train-loops 3
run A2_counter2_L64 --selector counter2 --train-loops 64
run A3_place_L3     --selector place    --train-loops 3
run A4_place_L64    --selector place    --train-loops 64
run A5_placev_L64   --selector placev   --train-loops 64
run A6_mlp_L64      --selector mlp      --train-loops 64
echo ALL_DONE_GRIDA
