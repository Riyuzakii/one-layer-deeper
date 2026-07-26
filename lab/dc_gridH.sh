#!/usr/bin/env bash
# Grid H -- the two remaining questions.
#  H1/H2: is the log detector's advantage real, or is it starting near the right
#         answer?  In the PROBE `alu.zero` is constructed, so `--detector log`
#         with thresh init -3 begins at a nearly correct detector.  Re-run from
#         thresh init 0.0, which is a *shallow* start (fires with p = 0.5 when
#         the register is zero) and is what the eval budget wants anyway.
#  H3/H4: the last failure mode is the learned unit digit landing on argmax 6
#         instead of 1 on Medium.  Does it escape with more steps, or is the
#         basin absorbing?
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
mkdir -p lab/logs lab/runs
OUT=lab/runs/gridH.jsonl
base="--modulus 329 --sel-log 2000 --seeds 0 1 2 3 4 --eval-loops 64 --n-eval 76 --out $OUT"
CONS="--cons 0.1 --cons-space w --cons-j 8 --cons-loops 4 --cons-start 400"
E="--selector counter2 --train-t 1 2 3 --train-loops 3 --no-dump --reg-hard"
M="--selector counter2 --train-t 4 8 16 --train-loops 16 --no-dump --reg-hard"

run () {
  tag=$1; shift
  echo "=== $tag"
  $V lab/probe_depth.py $base --tag "$tag" "$@" > "lab/logs/H_$tag.log" 2>&1
  grep -E "fitted loc" "lab/logs/H_$tag.log"
  sed -n '/=== summary ===/,$p' "lab/logs/H_$tag.log"
}

run H1_easy_log_t0  $E --detector log --thresh-init 0.0 $CONS --sel-steps 2000 --sel-anneal 900
run H2_med_log_t0   $M --detector log --thresh-init 0.0 $CONS --cons-jd 3 --sel-steps 2000 --sel-anneal 900
run H3_med_log_4k   $M --detector log --thresh-init 0.0 $CONS --cons-jd 3 --sel-steps 4000 --sel-anneal 900
run H4_med_log_lr03 $M --detector log --thresh-init 0.0 $CONS --cons-jd 3 --sel-steps 2000 --sel-anneal 900 --sel-lr 0.3
echo ALL_DONE_GRIDH
