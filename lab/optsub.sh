#!/usr/bin/env bash
# grok-optimization helper: generate a submission from knobs and run it.
#   lab/grok.sh <name> <steps> <seed> "<note>" <extra make_grok args...>
set -euo pipefail
REPO=/home/scratch.arohan_hw/git/one-layer-deeper/.worktrees/alu-optimizer
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
NAME=$1; STEPS=$2; SEED=$3; NOTE=$4; shift 4
cd "$REPO"
$V lab/make_optsub.py --name "$NAME" --max-steps "$STEPS" "$@" >/dev/null
$V lab/run_experiment.py \
  --submission "submissions/alu-optimizer/$NAME/submission.py" \
  --manifest "lab/manifests/lab_${DS:-e1}_fs${STEPS}_s${SEED}.json" \
  --tag "${TAG:-optsub}" --note "$NOTE" --timeout "${TMO:-14400}"
