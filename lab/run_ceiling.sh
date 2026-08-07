#!/usr/bin/env bash
# Constructed ceiling for the ADD-ONLY transducer at hf1's modulus sizes.
# LAB DIAGNOSTIC (--construct sets the learned tables to truth; never a
# submission).  hf1 is ID bits [16,18,20], OOD-N bits [17,19,21]; parameters are
# modulus-independent so ONE vector must cover all six.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.."
mkdir -p lab/logs
: > lab/logs/ceiling.log
for pick in learned fixed; do
  for b in 16 17 18 19 20 21; do
    for ms in 7 23 91; do
      $V lab/probe_add.py --construct --pop 1 --modulus-bits $b \
        --modulus-seed $ms --slots 7 --train-x 3000 --held-x 1000 \
        --eval-n 1024 --eval-chunk 128 --pick $pick \
        --tag ceil_${pick}_b${b}_m${ms} --jsonl lab/ceiling_runs.jsonl \
        >> lab/logs/ceiling.log 2>&1
    done
  done
done
echo DONE
