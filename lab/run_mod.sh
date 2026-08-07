#!/usr/bin/env bash
# MODULE-RESTRICTED identification (LAB DIAGNOSTIC: every table outside
# --modules is set to the construction).  The question the branch exists to
# answer: with everything else exactly right, does the LEGAL end-of-chain label
# identify the ADDER?  In DigitALU the same question about Tmul answered no.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.."
mkdir -p lab/logs
: > lab/logs/mod2_addA.log
for m in add sub pick add,sub; do
  for s in 0 1 2; do
    $V lab/probe_addsearch.py --modulus 323 --slots 3 --train-x 250 \
      --modules $m --repair-ks 0 --seed $s --tag mod2_${m//,/+}_s$s \
      --jsonl lab/mod2_runs.jsonl >> lab/logs/mod2_addA.log 2>&1
  done
done
: > lab/logs/mod2_dalu.log
for m in mul add; do
  for s in 0 1 2; do
    $V lab/probe_search.py --modulus 323 --slots 3 --train-x 250 \
      --modules $m --seed $s --tag mod2D_${m}_s$s \
      --jsonl lab/mod2_runs_dalu.jsonl >> lab/logs/mod2_dalu.log 2>&1
  done
done
echo DONE
