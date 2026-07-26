#!/usr/bin/env bash
# Grid D -- confirm the winners at m1's real modulus, N=10403 = 101*103,
# lambda = lcm(100,102) = 5100, 7/7 distinct ladder classes (no rung collapses).
# S = 5, so the ALU chain is ~14x the N=329 screen; 2 seeds, 128 held-out x.
# $1 selects the tier: easy (train T = 1,2,3) or medium (train T = 4,8,16).
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
mkdir -p lab/logs lab/runs
TIER=${1:-easy}
if [ "$TIER" = easy ]; then TT="1 2 3"; LMAX=3; else TT="4 8 16"; LMAX=16; fi
OUT=lab/runs/gridD.jsonl
common="--modulus 10403 --train-t $TT --sel-steps 2000 --sel-anneal 1200 --sel-log 1000 --seeds 0 1 --eval-loops 64 --n-train 250 --n-eval 128 --out $OUT"

run () {
  tag=$1; shift
  echo "=== $tag"
  $V lab/probe_depth.py $common --tag "$tag" "$@" > "lab/logs/D_$tag.log" 2>&1
  grep -E "MAX_T =|fitted loc" "lab/logs/D_$tag.log"
  sed -n '/=== summary ===/,$p' "lab/logs/D_$tag.log"
}

run D_${TIER}_counter2_Lmax    --selector counter2 --train-loops $LMAX
run D_${TIER}_counter2_nodump  --selector counter2 --train-loops $LMAX --no-dump
run D_${TIER}_counter2_L64     --selector counter2 --train-loops 64
run D_${TIER}_placev_cons      --selector placev --train-loops $LMAX --cons 1.0 --cons-space loc --cons-j 63 --cons-jd 3
echo ALL_DONE_GRIDD_$TIER
