#!/usr/bin/env bash
# Run one probe_curric.py cell.  Usage: curric_run.sh <tag> <args...>
set -euo pipefail
ROOT=/home/scratch.arohan_hw/git/one-layer-deeper/.worktrees/hf-curriculum-hf1
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
TAG=$1; shift
DATA="--n-mod 16 --n-mod-held 8 --n-x 1024 --n-held-x 64"
cd "$ROOT"
mkdir -p lab/logs
exec $V lab/probe_curric.py $DATA --jsonl lab/logs/curric.jsonl \
  --tag "$TAG" "$@" > "lab/logs/${TAG}.log" 2>&1
