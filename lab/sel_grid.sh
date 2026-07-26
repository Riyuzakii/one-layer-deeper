#!/usr/bin/env bash
# Depth-selector grid: with the step CONSTRUCTED (a perfect single squaring
# step), does the ordered-pointer selector route T across the whole ladder when
# it has only ever seen the tier's training T values?
#   Easy   trains T in {1,2,3}   -> must extrapolate UP   to 4..64
#   Medium trains T in {4,8,16}  -> must generalise DOWN  to 1,2 and UP to 32,64
set -u
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
OUT=lab/logs
mkdir -p $OUT

for tier in easy medium; do
  if [ "$tier" = easy ]; then TT="1 2 3"; else TT="4 8 16"; fi
  for k in mlp linear placev place; do
    for s in 0 1; do
      f=$OUT/sel_${tier}_${k}_s${s}.log
      echo "== $tier $k seed$s -> $f"
      $VENV lab/probe_compose.py --mode select --modulus 323 --dtype bf16 \
        --selector "$k" --sel-steps 1500 --sel-anneal 800 --sel-log 1500 \
        --train-t $TT --eval-loops 64 --seed $s \
        --tag "sel-${tier}-${k}-s${s}" > "$f" 2>&1
      tail -13 "$f"
    done
  done
done
