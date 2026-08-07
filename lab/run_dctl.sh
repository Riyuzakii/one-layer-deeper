#!/usr/bin/env bash
# Targeted 20-rep DigitALU control at the two decisive k values, so the
# add-only comparison rests on 20 reps on BOTH sides, not 5.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.."
mkdir -p lab/logs
: > lab/logs/dctl.log
for k in 10 20 14; do
  $V lab/probe_search.py --modulus 323 --slots 3 --train-x 250 \
    --tie sym,inv --repair $k --repair-reps 20 --seed 501 \
    --tag hiD_tied_k$k --jsonl lab/basin2_runs_dalu.jsonl >> lab/logs/dctl.log 2>&1
done
echo DONE
