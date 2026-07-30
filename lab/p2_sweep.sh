#!/usr/bin/env bash
# Drive submissions/plan2-pd-ssm-delta/submission.py over a P2_* configuration.
#
#   lab/p2_sweep.sh <manifest-name> <tag> <note>   [env P2_* already exported]
#
# Every run goes through lab/run_experiment.py so it lands in lab/archive.jsonl.
set -euo pipefail
ROOT=/home/scratch.arohan_hw/git/one-layer-deeper/.worktrees/p2-pd-ssm-delta
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
MANIFEST=$1; TAG=$2; NOTE=$3
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-/home/scratch.arohan_hw/.nv_cache}
cd "$ROOT"
$VENV lab/run_experiment.py \
  --submission submissions/plan2-pd-ssm-delta/submission.py \
  --manifest "lab/manifests/${MANIFEST}.json" \
  --tag "$TAG" --note "$NOTE" --timeout "${P2_TIMEOUT:-2400}"
