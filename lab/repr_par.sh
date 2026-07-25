#!/usr/bin/env bash
# Parallel representation screen.  Safe because every manifest is fixed_step:
# the clock never binds, so contention changes wall time but not the result.
# Usage: bash lab/repr_par.sh <manifest-stem> <tag> <sub-tag>...
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
REPO=/home/scratch.arohan_hw/git/one-layer-deeper/.worktrees/exact-arithmetic
MAN="$1"; shift
TAG="$1"; shift
pids=()
for name in "$@"; do
  "$V" "$REPO/lab/run_experiment.py" \
    --submission "$REPO/submissions/exp_repr/$name/submission.py" \
    --manifest "$REPO/lab/manifests/${MAN}.json" \
    --tag "$TAG" --note "$name @ $MAN" --timeout 6000 \
    > "/tmp/repr_logs/${TAG}_${name}_${MAN}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "done (fail=$fail)"
