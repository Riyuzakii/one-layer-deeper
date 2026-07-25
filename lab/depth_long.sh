#!/usr/bin/env bash
# explore/alu-depth: the same depth ladder run to convergence rather than to a
# fixed 2000 steps.  The shallow cells are 5-7x cheaper per step, so equal
# wall-clock gives them far more steps; this cell equalises STEPS instead.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
OUT=${OUT:-lab/logs/long}
mkdir -p "$OUT"
STEPS=${STEPS:-20000}
MOD=${MOD:-323}
SLOTS=${SLOTS:-3}
SEEDS=${SEEDS:-"0 1 2"}
CELLS=${CELLS:-"tree:quotient"}
EXTRA=${EXTRA:-}
SUF=${SUF:-}
for cell in $CELLS; do
  mm=${cell%%:*}; rm=${cell##*:}
  for s in $SEEDS; do
    tag="${MOD}_S${SLOTS}_${mm}_${rm}${SUF}_n${STEPS}_s${s}"
    f="$OUT/$tag.log"
    [ -s "$f" ] && { echo "skip $tag"; continue; }
    echo "=== $tag ==="
    $V lab/probe_alu.py --modulus "$MOD" --slots "$SLOTS" --mul-mode "$mm" \
       --reduce-mode "$rm" --steps "$STEPS" --seed "$s" --log-every 1000 \
       --tag "$tag" $EXTRA > "$f" 2>&1
    tail -1 "$f"
  done
done
