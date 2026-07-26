#!/usr/bin/env bash
# LAB ONLY.  Run a file of `<tag> <args...>` lines through lab/probe_credit.py
# with a concurrency cap.  Results append to lab/credit_runs.jsonl.
#   usage: bash lab/credit_sweep.sh jobs.txt [parallel]
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="$1"; P="${2:-5}"
mkdir -p "$ROOT/lab/logs"
run_one() {
  local tag="$1"; shift
  "$V" "$ROOT/lab/probe_credit.py" --tag "$tag" \
      --jsonl "$ROOT/lab/credit_runs.jsonl" "$@" \
      > "$ROOT/lab/logs/$tag.log" 2>&1
  tail -1 "$ROOT/lab/logs/$tag.log"
}
export -f run_one
export V ROOT
grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' | \
  xargs -P "$P" -I{} bash -c 'run_one {}'
