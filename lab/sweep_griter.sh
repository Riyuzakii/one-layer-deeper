#!/usr/bin/env bash
# GRIter: does the one construction that worked offline (iterating ONE shared
# pairwise-phase block, with a learned prompt-conditioned depth selector) survive
# contact with the real prompt format?  LOOPS is the only axis that mattered in
# lab/probe_iter.py, so it is the axis swept here; LOOPS=1 is the control that
# removes the round-trip constraint entirely.
set -u
cd "$(dirname "$0")/.."
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
MAN=lab/manifests/lab_e1_nw0_fs20000_s74.json

run() {
  local name="$1"; shift
  local note="$1"; shift
  $VENV lab/make_griter.py --name "$name" "$@" >/dev/null || return 1
  $VENV lab/run_experiment.py \
    --submission "submissions/group-rotation/$name/submission.py" \
    --manifest "$MAN" --tag griter --note "$note" --timeout 5400
}

run gi_l1 "GRIter LOOPS=1 (control: no round-trip constraint)" --loops 1 --slots 4
run gi_l2 "GRIter LOOPS=2"                                     --loops 2 --slots 4
run gi_l4 "GRIter LOOPS=4"                                     --loops 4 --slots 4
run gi_l8 "GRIter LOOPS=8"                                     --loops 8 --slots 4
run gi_l4_k64 "GRIter LOOPS=4 K=64"                            --loops 4 --slots 4 --freqs 64
echo "GRITER DONE"
