#!/usr/bin/env bash
# LAB ONLY.  ms/step vs replica count, min of R repeats (the GPU is shared, so
# only the minimum is meaningful; the mean tracks whoever else is running).
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
R="${R:-3}"
STEPS="${STEPS:-40}"
BATCH="${BATCH:-512}"
EXTRA="${EXTRA:-}"
for P in $@; do
  best=999999; mem=""
  for r in $(seq 1 "$R"); do
    line=$(timeout 900 "$V" "$ROOT/lab/probe_pop.py" --pop "$P" --steps "$STEPS" \
             --sel 0 --timing-only --batch "$BATCH" $EXTRA \
             --tag "tm_p${P}_r${r}" 2>&1 | grep TIMING)
    m=$(echo "$line" | sed 's/.*ms_per_step=\([0-9.]*\).*/\1/')
    mem=$(echo "$line" | sed 's/.*peak_mem=\([0-9.]*\).*/\1/')
    [ -n "${m:-}" ] && best=$(python3 -c "print(min($best,$m))")
  done
  echo "P=$P batch=$BATCH extra='$EXTRA' min_ms=$best peak_GiB=$mem"
done
