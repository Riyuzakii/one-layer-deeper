#!/usr/bin/env bash
# Grid I -- the Medium/Hard tier T values, which is where the controller still
# fails.  The diagnosed failure is NOT the detector and NOT coverage: it is the
# learned unit digit landing on argmax 6 or 8 instead of 1.  `one` is a 10-way
# softmax initialised from randn*0.5, so it starts with a random preference;
# `--one-init 0.0` starts it exactly uniform and lets the gradient choose.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
mkdir -p lab/logs lab/runs
OUT=lab/runs/gridI.jsonl
base="--modulus 329 --sel-log 2000 --seeds 0 1 2 3 4 --eval-loops 64 --n-eval 76 --out $OUT --sel-steps 2000 --sel-anneal 900"
CONS="--cons 0.1 --cons-space w --cons-j 8 --cons-loops 4 --cons-start 400 --cons-jd 3"
M="--selector counter2 --train-t 4 8 16 --train-loops 16 --no-dump --reg-hard"

run () {
  tag=$1; shift
  echo "=== $tag"
  $V lab/probe_depth.py $base --tag "$tag" "$@" > "lab/logs/I_$tag.log" 2>&1
  grep -E "fitted loc" "lab/logs/I_$tag.log"
  sed -n '/=== summary ===/,$p' "lab/logs/I_$tag.log"
}

run I1_med_sum_uniform      $M --detector sum --one-init 0.0 $CONS
run I2_med_sum_uniform_lr03 $M --detector sum --one-init 0.0 $CONS --sel-lr 0.3
run I3_med_log_uniform      $M --detector log --one-init 0.0 $CONS
run I4_med_sum_uniform_noc  $M --detector sum --one-init 0.0
echo ALL_DONE_GRIDI
