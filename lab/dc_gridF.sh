#!/usr/bin/env bash
# Grid F -- the recipe that reaches the top of the ladder, and its ablations.
#
# RECIPE R = counted halting
#            + --no-dump   (unspent halting mass is NOT piled on the last index;
#                           with the dump, "never halt" is exactly correct for
#                           the deepest training T, and the dump also makes the
#                           consistency law unsatisfiable)
#            + --reg-hard  (straight-through one-hot digit register)
#            + --cons 0.1  (self-consistency w(inc r) = shift_right(w(r)) over
#                           the model's own learned increment orbit).
#
# The consistency law's FIRST component is w(inc r)[0] = 0, i.e. the halting
# probability must be zero at every register that is a successor.  That is the
# leak-suppression constraint the cross-entropy cannot see, because CE through
# saturated log-probabilities is flat once the argmax is right.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
mkdir -p lab/logs lab/runs
OUT=lab/runs/gridF.jsonl
base="--modulus 329 --sel-steps 2000 --sel-anneal 1200 --sel-log 2000 --seeds 0 1 2 --eval-loops 64 --n-eval 76 --out $OUT"
R="--no-dump --reg-hard --cons 0.1 --cons-space w --cons-j 8 --cons-loops 8"

run () {
  tag=$1; shift
  echo "=== $tag"
  $V lab/probe_depth.py $base --tag "$tag" "$@" > "lab/logs/F_$tag.log" 2>&1
  grep -E "MAX_T =|MAX_T\(|fitted loc" "lab/logs/F_$tag.log"
  sed -n '/=== summary ===/,$p' "lab/logs/F_$tag.log"
}

# --- Easy tier, train T = {1,2,3} ---
run F1_easy_R          --selector counter2 --train-t 1 2 3 --train-loops 3 $R
run F2_easy_nocons     --selector counter2 --train-t 1 2 3 --train-loops 3 --no-dump --reg-hard
run F3_easy_nohard     --selector counter2 --train-t 1 2 3 --train-loops 3 --no-dump --cons 0.1 --cons-space w --cons-j 8 --cons-loops 8
run F4_easy_R_evalhard --selector counter2 --train-t 1 2 3 --train-loops 3 $R --eval-hard
run F5_easy_R_dump     --selector counter2 --train-t 1 2 3 --train-loops 3 --reg-hard --cons 0.1 --cons-space w --cons-j 8 --cons-loops 8

# --- Medium/Hard tier T values, train T = {4,8,16}: downward generalisation ---
run F6_med_R           --selector counter2 --train-t 4 8 16 --train-loops 16 $R
run F7_med_nocons      --selector counter2 --train-t 4 8 16 --train-loops 16 --no-dump --reg-hard
run F8_med_R_evalhard  --selector counter2 --train-t 4 8 16 --train-loops 16 $R --eval-hard

# --- does self-consistency rescue a (place,digit)-indexed head? ---
run F9_easy_placev_R   --selector placev --train-t 1 2 3 --train-loops 3 --cons 0.1 --cons-space loc --cons-j 63
run F10_easy_mlp_R     --selector mlp    --train-t 1 2 3 --train-loops 3 --cons 0.1 --cons-space loc --cons-j 63
run F11_med_placev_R   --selector placev --train-t 4 8 16 --train-loops 16 --cons 0.1 --cons-space loc --cons-j 48 --cons-jd 3
echo ALL_DONE_GRIDF
