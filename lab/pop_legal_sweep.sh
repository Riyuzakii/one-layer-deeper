#!/usr/bin/env bash
# LAB ONLY.  Run `<tag> <args...>` lines through lab/probe_pop_legal.py.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="$1"; P="${2:-3}"
mkdir -p "$ROOT/lab/logs"
run_one() {
  local tag="$1"; shift
  "$V" "$ROOT/lab/probe_pop_legal.py" --tag "$tag" \
      --jsonl "$ROOT/lab/opt_legal_runs.jsonl" "$@" \
      > "$ROOT/lab/logs/$tag.log" 2>&1
  grep -h FINAL "$ROOT/lab/logs/$tag.log" | tail -1 | cut -c1-160
}
export -f run_one
export V ROOT
grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' | \
  xargs -P "$P" -I{} bash -c 'run_one {}'
