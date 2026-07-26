#!/usr/bin/env bash
# The counter depth controller, TRAINED (22 parameters) on each tier's T values
# with the arithmetic held at its construction.  Does counting escape the
# T-digit-coverage failure of the ordered-pointer heads?
set -u
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
mkdir -p lab/logs
for tier in easy medium; do
  if [ "$tier" = easy ]; then TT="1 2 3"; else TT="4 8 16"; fi
  for s in 0 1; do
    f=lab/logs/sel_${tier}_counter_s${s}.log
    $VENV lab/probe_compose.py --mode select --modulus 323 --dtype bf16 \
      --selector counter --sel-steps 2000 --sel-lr 0.1 --sel-anneal 1000 \
      --sel-log 2000 --train-t $TT --eval-loops 64 --seed $s \
      --tag "sel-${tier}-counter-s${s}" > "$f" 2>&1
    echo "== $tier counter seed$s"; tail -13 "$f"
  done
done
