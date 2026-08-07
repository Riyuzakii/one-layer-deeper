#!/usr/bin/env bash
# Evaluator cells on hf1 (fixed-step, contention-immune).  These are the
# MECHANISM check plus the mandatory --lr 0 control, on the cheap reference
# architecture -- not the scientific test of the curriculum, which is
# lab/probe_curric.py.  The evaluator reports pooled metrics only, so it cannot
# resolve a per-modulus-size effect; that is why the probe exists.
set -uo pipefail
R=/home/scratch.arohan_hw/git/one-layer-deeper/.worktrees/hf-curriculum-hf1
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
M=lab/manifests/lab_hf1_fs2000_s74.json
cd "$R"
for v in off lr0 "" ids b4; do
  name="hard-curriculum-hf1${v:+-$v}"
  $V lab/run_experiment.py --submission "submissions/$name/submission.py" \
    --manifest "$M" --tag curric-eval --timeout 3600 \
    --note "hf1 fs2000 :: $name" 2>&1 | tee "lab/logs/eval_${name}.log"
done
