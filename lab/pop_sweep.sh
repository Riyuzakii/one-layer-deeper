#!/usr/bin/env bash
# LAB ONLY.  Run a file of `<tag> <args...>` lines through lab/probe_pop.py
# with a concurrency cap.  Results append to lab/pop_runs.jsonl.
#   usage: bash lab/pop_sweep.sh jobs.txt [parallel]
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="$1"; P="${2:-3}"
mkdir -p "$ROOT/lab/logs"
run_one() {
  local tag="$1"; shift
  "$V" "$ROOT/lab/probe_pop.py" --tag "$tag" \
      --jsonl "$ROOT/lab/pop_runs.jsonl" "$@" \
      > "$ROOT/lab/logs/$tag.log" 2>&1
  grep -h "FINAL" "$ROOT/lab/logs/$tag.log" | tail -1
}
export -f run_one
export V ROOT
grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' | \
  xargs -P "$P" -I{} bash -c 'run_one {}'
