#!/usr/bin/env bash
# Evaluator runs on hf1 (the Hard-faithful, modulus-split proxy).
# --mode fixed_step manifests only, so the shared GPU cannot corrupt the
# comparison.  The --lr 0 control runs beside the trained cell, same manifest,
# same step count, same batch size.
#
# COST NOTE (recorded because it is itself a result): at the manifest's
# batch_size 512 the add-only transducer did not finish 2,000 steps inside
# run_experiment.py's 1,200 s default and the run was killed (status failed,
# archived).  SUBMISSION.batch_size = 128 plus --timeout 5400 fits.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.."
mkdir -p lab/logs

$V lab/run_experiment.py --timeout 5400 \
  --submission submissions/hard-add-only/submission.py \
  --manifest lab/manifests/lab_hf1_fs2000_s74.json \
  --tag add-only-hf1 --note "add-only transducer, LEGAL, 2000 steps on hf1, bs128" \
  > lab/logs/eval_hf1_trained.log 2>&1

$V lab/run_experiment.py --timeout 5400 \
  --submission submissions/hard-add-only-lr0/submission.py \
  --manifest lab/manifests/lab_hf1_fs2000_s74.json \
  --tag add-only-hf1-lr0 --note "CONTROL: identical model, lr=0, no learning" \
  > lab/logs/eval_hf1_lr0.log 2>&1
echo DONE
