#!/usr/bin/env bash
# Assemble lab/reports/alu-relational.md from the section drafts in /tmp and
# the live tables in lab/rel_runs.jsonl / lab/assoc_runs.jsonl.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/lab/reports/alu-relational.md"
cat /tmp/rep_head.md \
    /tmp/rep_fam1.md \
    /tmp/rep_dual_pre.md /tmp/rep_dual_tab.md \
    /tmp/rep_mid.md \
    /tmp/rep_basin.md \
    /tmp/rep_compliance.md \
    /tmp/rep_tail.md \
    /tmp/rep_repro.md > "$OUT"
wc -l "$OUT"
