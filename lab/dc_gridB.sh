#!/usr/bin/env bash
# Grid B -- Medium/Hard tier T values (train T = {4,8,16}) at N=329
# (lambda=138, 7/7 distinct ladder classes).  The tier's *T values* are what the
# controller sees, so the controller question is answerable at a cheap modulus;
# grid D re-runs the winners at m1's real modulus.
# Certification is a prefix from T=1, so this grid is the DOWNWARD
# generalisation test as well as the upward one.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
mkdir -p lab/logs lab/runs
OUT=lab/runs/gridB.jsonl
common="--modulus 329 --train-t 4 8 16 --sel-steps 2000 --sel-anneal 1200 --sel-log 1000 --seeds 0 1 2 --eval-loops 64 --n-eval 76 --out $OUT"

run () {
  tag=$1; shift
  echo "=== $tag"
  $V lab/probe_depth.py $common --tag "$tag" "$@" > "lab/logs/B_$tag.log" 2>&1
  grep -E "MAX_T =|fitted loc" "lab/logs/B_$tag.log"
  sed -n '/=== summary ===/,$p' "lab/logs/B_$tag.log"
}

run B1_counter2_L16  --selector counter2 --train-loops 16
run B2_counter2_L64  --selector counter2 --train-loops 64
run B3_place_L16     --selector place    --train-loops 16
run B4_place_L64     --selector place    --train-loops 64
run B5_mlp_L64       --selector mlp      --train-loops 64
echo ALL_DONE_GRIDB
