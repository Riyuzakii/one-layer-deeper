#!/usr/bin/env bash
# explore/alu-depth: sweep the sequential depth of the DigitALU graph at a fixed
# target.  CELLS are "<mul-mode>:<reduce-mode>"; everything else is identical to
# digit-carry's default cell, so the only variable is the chain length.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
OUT=${OUT:-lab/logs/depth}
mkdir -p "$OUT"
STEPS=${STEPS:-2000}
MOD=${MOD:-323}
SLOTS=${SLOTS:-3}
SEEDS=${SEEDS:-"0 1 2"}
CELLS=${CELLS:-"tree:quotient horner:quotient tree:binary horner:binary tree:serial"}
EXTRA=${EXTRA:-}
SUF=${SUF:-}
for cell in $CELLS; do
  mm=${cell%%:*}; rm=${cell##*:}
  for s in $SEEDS; do
    tag="${MOD}_S${SLOTS}_${mm}_${rm}${SUF}_s${s}"
    f="$OUT/$tag.log"
    [ -s "$f" ] && { echo "skip $tag"; continue; }
    echo "=== $tag ==="
    $V lab/probe_alu.py --modulus "$MOD" --slots "$SLOTS" --mul-mode "$mm" \
       --reduce-mode "$rm" --steps "$STEPS" --seed "$s" --log-every 250 \
       --tag "$tag" $EXTRA > "$f" 2>&1
    tail -1 "$f"
  done
done
