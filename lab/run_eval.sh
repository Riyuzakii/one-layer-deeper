#!/usr/bin/env bash
# Evaluator runs on hf1 (the Hard-faithful, modulus-split proxy).
# --mode fixed_step manifests only, so the shared GPU cannot corrupt the
# comparison.  The --lr 0 control runs beside every trained cell.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.."
mkdir -p lab/logs

$V lab/run_experiment.py \
  --submission submissions/hard-add-only/submission.py \
  --manifest lab/manifests/lab_hf1_fs2000_s74.json \
  --tag add-only-hf1 --note "add-only transducer, LEGAL, 2000 steps on hf1" \
  > lab/logs/eval_hf1_trained.log 2>&1

$V lab/run_experiment.py \
  --submission submissions/hard-add-only-lr0/submission.py \
  --manifest lab/manifests/lab_hf1_fs2000_s74.json \
  --tag add-only-hf1-lr0 --note "CONTROL: identical model, lr=0, no learning" \
  > lab/logs/eval_hf1_lr0.log 2>&1
echo DONE
