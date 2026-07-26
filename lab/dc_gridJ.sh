#!/usr/bin/env bash
# Grid J -- the final recipe confirmed at the real moduli.
#   RECIPE: counted halting, log-space conjunction detector, no mass dump,
#           straight-through one-hot digit register, self-consistency with a
#           warm start, commit-to-the-mode readout.
#   Easy   uses --thresh-init -3.0 ; Medium/Hard uses --thresh-init 0.0.
#   (Measured: at Medium's T values a -3.0 start puts the untrained detector
#    below every reachable score and it never recovers -- 0/5; a 0.0 start is
#    4/5.  At Easy's T values the -3.0 start is 5/5.)
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
mkdir -p lab/logs lab/runs
OUT=lab/runs/gridJ.jsonl
base="--sel-log 2000 --seeds 0 1 2 3 4 --eval-loops 64 --out $OUT --sel-steps 2000 --sel-anneal 900 --detector log --no-dump --reg-hard --selector counter2"
CONS="--cons 0.1 --cons-space w --cons-j 8 --cons-loops 4 --cons-start 400"

run () {
  tag=$1; shift
  echo "=== $tag"
  $V lab/probe_depth.py $base --tag "$tag" "$@" > "lab/logs/J_$tag.log" 2>&1
  grep -E "fitted loc|lambda" "lab/logs/J_$tag.log" | head -6
  sed -n '/=== summary ===/,$p' "lab/logs/J_$tag.log"
}

M1="--modulus 10403 --n-train 250 --n-eval 128"
run J1_m1_med   $M1 --train-t 4 8 16 --train-loops 16 --thresh-init 0.0 $CONS --cons-jd 3
run J2_m1_easy  $M1 --train-t 1 2 3  --train-loops 3  --thresh-init -3.0 $CONS
run J3_329_med_evalhard --modulus 329 --n-eval 76 --train-t 4 8 16 --train-loops 16 --thresh-init 0.0 $CONS --cons-jd 3 --eval-hard
run J4_hp1_med  --modulus 4028033 --n-train 200 --n-eval 64 --train-t 4 8 16 --train-loops 16 --thresh-init 0.0 $CONS --cons-jd 3
echo ALL_DONE_GRIDJ
