#!/usr/bin/env bash
# Grid C -- the two named directions, Easy tier (train T = {1,2,3}), N=329
# (lambda=138, 7/7 distinct ladder classes).
#
#  * "no-dump": the cheapest possible version of the depth fix.  The halting
#    distribution is computed over the SAME short grid, but unspent mass is not
#    piled on the last index -- it is simply lost from the mixture, which the
#    cross-entropy then penalises.  Costs no extra ALU depth at all.
#  * self-consistency: w(inc(r)) == shift_right(w(r)) / loc(inc(r)) == loc(r)+1,
#    where inc is the model's OWN learned digit increment (Tadd/carry0/zero
#    shared with the arithmetic) applied to T's digit register.  No labels, no
#    extra data.  This supervises the controller at T values the tier never
#    provides, which is exactly the coverage hole.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
mkdir -p lab/logs lab/runs
OUT=lab/runs/gridC.jsonl
common="--modulus 329 --train-t 1 2 3 --sel-steps 2000 --sel-anneal 1200 --sel-log 1000 --seeds 0 1 2 --eval-loops 64 --n-eval 76 --out $OUT"

run () {
  tag=$1; shift
  echo "=== $tag"
  $V lab/probe_depth.py $common --tag "$tag" "$@" > "lab/logs/C_$tag.log" 2>&1
  grep -E "MAX_T =|fitted loc" "lab/logs/C_$tag.log"
  sed -n '/=== summary ===/,$p' "lab/logs/C_$tag.log"
}

run C1_counter2_L3_nodump  --selector counter2 --train-loops 3  --no-dump
run C2_counter2_L4_nodump  --selector counter2 --train-loops 4  --no-dump
run C3_counter2_L64_init3  --selector counter2 --train-loops 64 --one-init 3.0
run C4_placev_cons         --selector placev --train-loops 3 --cons 1.0 --cons-space loc --cons-j 63
run C5_mlp_cons            --selector mlp    --train-loops 3 --cons 1.0 --cons-space loc --cons-j 63
run C6_mlp_L3_ctrl         --selector mlp    --train-loops 3
run C7_placev_L3_ctrl      --selector placev --train-loops 3
run C8_counter2_L3_reghard --selector counter2 --train-loops 3 --reg-hard
run C9_counter2_L3_nd_rh   --selector counter2 --train-loops 3 --no-dump --reg-hard
echo ALL_DONE_GRIDC
