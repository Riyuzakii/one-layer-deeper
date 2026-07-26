#!/usr/bin/env bash
# `counterz`: the same countdown controller, but the zero detector is a match
# against the ALU's OWN learned zero digit (2 scalars) instead of a per-digit
# linear head.  It therefore has no parameter indexed by a digit of T at all.
set -u
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
mkdir -p lab/logs
for tier in easy medium; do
  if [ "$tier" = easy ]; then TT="1 2 3"; else TT="4 8 16"; fi
  for s in 0 1; do
    f=lab/logs/sel_${tier}_counterz_s${s}.log
    $VENV lab/probe_compose.py --mode select --modulus 323 --dtype bf16 \
      --selector counterz --sel-steps 2000 --sel-lr 0.1 --sel-anneal 1000 \
      --sel-log 2000 --train-t $TT --eval-loops 64 --seed $s \
      --tag "sel-${tier}-counterz-s${s}" > "$f" 2>&1
    echo "== $tier counterz seed$s"; tail -13 "$f"
  done
done
