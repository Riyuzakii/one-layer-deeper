#!/usr/bin/env bash
# Long fixed-step GRNet grid on e1 (grokking needs many post-memorisation steps).
# Every run is fixed_step so it is immune to GPU contention from other agents.
set -u
cd "$(dirname "$0")/.."
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
MAN=lab/manifests/lab_e1_nw0_fs20000_s74.json

run() {  # name, note, extra args...
  local name="$1"; shift
  local note="$1"; shift
  $VENV lab/make_grnet.py --name "$name" "$@" >/dev/null || return 1
  $VENV lab/run_experiment.py \
    --submission "submissions/group-rotation/$name/submission.py" \
    --manifest "$MAN" --tag grnet-long --note "$note" --timeout 3600
}

run gl_wd01   "GRNet w2 K256 wd0.1 lr0.01 20k"        --width 2 --freqs 256 --wd 0.1 --lr 0.01
run gl_wd1    "GRNet w2 K256 wd1.0 lr0.01 20k"        --width 2 --freqs 256 --wd 1.0 --lr 0.01
run gl_wd3    "GRNet w2 K256 wd3.0 lr0.01 20k"        --width 2 --freqs 256 --wd 3.0 --lr 0.01
run gl_noquad "GRNet w2 K256 wd1.0 QUAD=0 (control)"  --width 2 --freqs 256 --wd 1.0 --lr 0.01 --quad 0
run gl_rev    "GRNet w2 K256 wd1.0 posmode=rev"       --width 2 --freqs 256 --wd 1.0 --lr 0.01 --posmode rev
run gl_abs    "GRNet w2 K256 wd1.0 posmode=abs"       --width 2 --freqs 256 --wd 1.0 --lr 0.01 --posmode abs
run gl_k64    "GRNet w2 K64  wd1.0"                   --width 2 --freqs 64  --wd 1.0 --lr 0.01
run gl_w16    "GRNet w16 K256 wd1.0"                  --width 16 --freqs 256 --wd 1.0 --lr 0.01
echo "SWEEP DONE"
