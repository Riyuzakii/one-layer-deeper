#!/usr/bin/env bash
# Representation screen. $1 = manifest stem, $2 = tag, rest = submission tags.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
REPO=/home/scratch.arohan_hw/git/one-layer-deeper/.worktrees/exact-arithmetic
MAN="$1"; shift
TAG="$1"; shift
for name in "$@"; do
  echo "=== $name @ $MAN ==="
  "$V" "$REPO/lab/run_experiment.py" \
    --submission "$REPO/submissions/exp_repr/$name/submission.py" \
    --manifest "$REPO/lab/manifests/${MAN}.json" \
    --tag "$TAG" --note "$name @ $MAN" --timeout 3000
done
