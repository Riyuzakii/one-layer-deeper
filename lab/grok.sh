#!/usr/bin/env bash
# grok-optimization helper: generate a submission from knobs and run it.
#   lab/grok.sh <name> <steps> <seed> "<note>" <extra make_grok args...>
set -euo pipefail
REPO=/home/scratch.arohan_hw/git/one-layer-deeper/.worktrees/grok-optimization
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
NAME=$1; STEPS=$2; SEED=$3; NOTE=$4; shift 4
cd "$REPO"
$V lab/make_grok.py --name "$NAME" --max-steps "$STEPS" "$@" >/dev/null
$V lab/run_experiment.py \
  --submission "submissions/grok-optimization/$NAME/submission.py" \
  --manifest "lab/manifests/lab_e1_fs${STEPS}_s${SEED}.json" \
  --tag "${TAG:-grok}" --note "$NOTE" --timeout "${TMO:-14400}"
