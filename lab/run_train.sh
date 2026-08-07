#!/usr/bin/env bash
# ADD-ONLY training block at hf1's arithmetic scale (20-bit sampled modulus,
# S=7, the same operand size the hf1 manifest presents).
#
# ORDER MATTERS and follows BRIEF2 section 6:
#   A  DIAGNOSTIC GATE first (teacher forcing off a constructed tape).  If the
#      instrument cannot reach the ceiling, a legal null is uninterpretable.
#   B  --lr 0 CONTROL before any legal number is interpreted.
#   C  LEGAL objective with a long enough FITTING CURVE to leave the
#      pre-fitting region.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.."
mkdir -p lab/logs
J=lab/train_runs.jsonl
COMMON="--modulus-bits 20 --modulus-seed 7 --slots 7 --train-x 8000 --held-x 1024 --batch 256 --eval-n 512 --eval-chunk 64"

# A. DIAGNOSTIC GATE -- teacher-forced local CE.  NEVER a submission (rules 2,7).
$V lab/probe_add.py --pop 32 $COMMON --tf 1.0 --sel 0 --steps 2000 \
  --lr 3e-2 --log-every 200 --tag A_gate_tf_p32 --jsonl $J \
  > lab/logs/A_gate_tf_p32.log 2>&1

# B. --lr 0 CONTROL (LEGAL objective, no learning).  Every legal claim below is
#    read against this row.
$V lab/probe_add.py --pop 32 $COMMON --tf 0 --steps 1200 --lr 0 \
  --log-every 300 --tag B_lr0_p32 --jsonl $J > lab/logs/B_lr0_p32.log 2>&1

# C. LEGAL, long fitting curve.
$V lab/probe_add.py --pop 32 $COMMON --tf 0 --steps 12000 --lr 3e-2 \
  --log-every 500 --tag C_legal_p32_12k --jsonl $J \
  > lab/logs/C_legal_p32_12k.log 2>&1
echo DONE
