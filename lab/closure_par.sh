#!/usr/bin/env bash
# usage: bash lab/closure_par.sh <manifest-stem> <tag> <par> <cfg> [<cfg> ...]
# Runs each generated submission under submissions/exp_closure/<cfg>/ against the
# same fixed-step manifest, <par> at a time, archiving every result.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$1"; shift
TAG="$1"; shift
PAR="$1"; shift
cd "$REPO"
i=0
for cfg in "$@"; do
  $V lab/run_experiment.py \
     --submission "submissions/exp_closure/${cfg}/submission.py" \
     --manifest "lab/manifests/${MANIFEST}.json" \
     --tag "$TAG" --note "$cfg" --timeout 5400 \
     > "/tmp/closure_${TAG}_${cfg}_${MANIFEST}.log" 2>&1 &
  i=$((i+1))
  if [ $((i % PAR)) -eq 0 ]; then wait; fi
done
wait
for cfg in "$@"; do
  echo "===== $cfg ====="
  tail -8 "/tmp/closure_${TAG}_${cfg}_${MANIFEST}.log"
done
