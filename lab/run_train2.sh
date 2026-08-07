#!/usr/bin/env bash
# ADD-ONLY variants, all LEGAL unless labelled.  Runs beside lab/run_train.sh.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.."
mkdir -p lab/logs
J=lab/train_runs.jsonl
BIG="--modulus-bits 20 --modulus-seed 7 --slots 7 --train-x 8000 --held-x 1024 --batch 256 --eval-n 512 --eval-chunk 64"
SML="--modulus 323 --slots 3 --train-x 250 --held-x 38 --batch 128 --eval-n 250 --eval-chunk 250"

# D. Can the class FIT at all?  e1-scale sanity, where DigitALU reached
#    train_exact 0.20-0.78 -- plus its own --lr 0 control.
$V lab/probe_add.py --pop 32 $SML --tf 0 --steps 6000 --lr 3e-2 \
  --log-every 500 --tag D_small_legal --jsonl $J > lab/logs/D_small_legal.log 2>&1
$V lab/probe_add.py --pop 32 $SML --tf 0 --steps 1200 --lr 0 \
  --log-every 400 --tag D_small_lr0 --jsonl $J > lab/logs/D_small_lr0.log 2>&1
# and the DIAGNOSTIC gate at the same small scale
$V lab/probe_add.py --pop 32 $SML --tf 1.0 --sel 0 --steps 2000 --lr 3e-2 \
  --log-every 400 --tag D_small_gate --jsonl $J > lab/logs/D_small_gate.log 2>&1

# E. learning-rate sweep on the LEGAL objective at hf1 scale
for lr in 1e-2 1e-1; do
  $V lab/probe_add.py --pop 32 $BIG --tf 0 --steps 3000 --lr $lr \
    --log-every 500 --tag E_legal_lr$lr --jsonl $J \
    > lab/logs/E_legal_lr$lr.log 2>&1
done

# F. BORDERLINE variant: fixed pick (digit one-hot indexes the chain directly)
$V lab/probe_add.py --pop 32 $BIG --pick fixed --tf 0 --steps 6000 --lr 3e-2 \
  --log-every 500 --tag F_fixedpick_legal --jsonl $J \
  > lab/logs/F_fixedpick_legal.log 2>&1
$V lab/probe_add.py --pop 32 $BIG --pick fixed --tf 0 --steps 1200 --lr 0 \
  --log-every 400 --tag F_fixedpick_lr0 --jsonl $J \
  > lab/logs/F_fixedpick_lr0.log 2>&1

# G. hypothesis-class ties and hard states -- one cell each
$V lab/probe_add.py --pop 32 $BIG --tie --tie-sub --tf 0 --steps 3000 \
  --lr 3e-2 --log-every 500 --tag G_tied_legal --jsonl $J \
  > lab/logs/G_tied_legal.log 2>&1
$V lab/probe_add.py --pop 32 $BIG --sthard --tf 0 --steps 3000 --lr 3e-2 \
  --log-every 500 --tag G_sthard_legal --jsonl $J \
  > lab/logs/G_sthard_legal.log 2>&1
echo DONE
