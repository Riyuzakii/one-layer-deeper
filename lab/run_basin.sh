#!/usr/bin/env bash
# Repair-basin ladder: ADD-ONLY vs DigitALU, matched conditions.
# LAB DIAGNOSTIC (starts from a corrupted construction); measures how far the
# LEGAL end-of-chain label objective can exactly repair a discrete transducer.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
KS=1,2,3,5,10,20,50,100,200
cd "$(dirname "$0")/.."
mkdir -p lab/logs

# 1. ADD-ONLY variant A (learned pick), sym+inv ties (237 cells) -- the direct
#    analogue of discrete-search section 8's 337-cell DigitALU ladder.
$V lab/probe_addsearch.py --modulus 323 --slots 3 --train-x 250 \
  --tie sym,inv --repair-ks $KS --repair-reps 5 --seed 11 \
  --tag addA_tied --jsonl lab/basin_runs.jsonl > lab/logs/addA_tied.log 2>&1

# 2. ADD-ONLY variant A, untied (817 cells)
$V lab/probe_addsearch.py --modulus 323 --slots 3 --train-x 250 \
  --repair-ks $KS,500 --repair-reps 5 --seed 11 \
  --tag addA_untied --jsonl lab/basin_runs.jsonl > lab/logs/addA_untied.log 2>&1

# 3. ADD-ONLY variant B (fixed pick, BORDERLINE), sym+inv ties (227 cells)
$V lab/probe_addsearch.py --modulus 323 --slots 3 --train-x 250 --pick fixed \
  --tie sym,inv --repair-ks $KS --repair-reps 5 --seed 11 \
  --tag addB_tied --jsonl lab/basin_runs.jsonl > lab/logs/addB_tied.log 2>&1

# 4. CONTROL: DigitALU, sym+inv ties (337 cells) -- reproduces the published
#    5/5 at k=3 and 0/5 at k=20 in this worktree, with this searcher.
: > lab/logs/dalu_tied.log
for k in 1 2 3 5 10 20 50 100 200; do
  $V lab/probe_search.py --modulus 323 --slots 3 --train-x 250 \
    --tie sym,inv --repair $k --repair-reps 5 --seed 11 \
    --tag dalu_tied_k$k --jsonl lab/basin_runs_dalu.jsonl \
    >> lab/logs/dalu_tied.log 2>&1
done
echo DONE
