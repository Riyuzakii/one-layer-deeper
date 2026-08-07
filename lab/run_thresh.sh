#!/usr/bin/env bash
# IDENTIFIABILITY THRESHOLD.  Section 5 found that the legal end-of-chain label
# recovers a 10-cell table exactly and a 400-cell table not at all.  This walks
# the ladder in between, CONFINED TO ONE MODULE (everything outside it stays at
# the construction), so the answer is "how many wrong cells of one table can the
# legal label repair" -- a number the project has never had.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.."
mkdir -p lab/logs
KS=5,10,20,40,80,160,400

$V lab/probe_addsearch.py --modulus 323 --slots 3 --train-x 250 \
  --modules add --repair-ks $KS --repair-reps 5 --seed 77 \
  --tag thr_add_addonly --jsonl lab/thresh_runs.jsonl \
  > lab/logs/thr_add_addonly.log 2>&1

: > lab/logs/thr_dalu.log
for k in 5 10 20 40 80 160; do
  $V lab/probe_search.py --modulus 323 --slots 3 --train-x 250 \
    --modules mul --repair $k --repair-reps 5 --seed 77 \
    --tag thrD_mul_k$k --jsonl lab/thresh_runs_dalu.jsonl >> lab/logs/thr_dalu.log 2>&1
done
for k in 5 10 20 40 80 160 400; do
  $V lab/probe_search.py --modulus 323 --slots 3 --train-x 250 \
    --modules add --repair $k --repair-reps 5 --seed 77 \
    --tag thrD_add_k$k --jsonl lab/thresh_runs_dalu.jsonl >> lab/logs/thr_dalu.log 2>&1
done
echo DONE
