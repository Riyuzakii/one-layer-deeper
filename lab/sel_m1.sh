#!/usr/bin/env bash
# The Medium tier measured at a modulus whose ladder does NOT collapse.
# On e1 (N=323) lambda=144 and 2^T mod 144 has period 6 from T=4, so the rungs
# {4,16,64} and {8,32} are the SAME function -- the depth loss cannot identify
# the iteration count there.  N=10403 (m1) has seven distinct exponent classes
# on the ladder, so this is the honest downward-generalisation test.
set -u
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
mkdir -p lab/logs
for cell in "mlp 0" "place 0" "counterz 0" "counterz 1"; do
  set -- $cell; k=$1; s=$2
  f=lab/logs/sel_m1_${k}_s${s}.log
  $VENV lab/probe_compose.py --mode select --modulus 10403 --dtype bf16 \
    --n-eval 128 --selector "$k" --sel-steps 2000 --sel-lr 0.1 \
    --sel-anneal 1000 --sel-log 2000 --train-t 4 8 16 --eval-loops 64 \
    --seed $s --tag "sel-m1-${k}-s${s}" > "$f" 2>&1
  echo "== m1 $k seed$s"; tail -13 "$f"
done
