#!/usr/bin/env bash
# Phase T5 -- eval-budget feasibility for DEEP configs.
#
# The eval budget is HALF the training budget and must cover test + ood +
# 7 seen-N rungs + 7 OOD-N rungs.  A model that loops 64 times per forward can
# train fine and still score 0 by running out of eval clock.  Training is
# deliberately cut to 50 steps here: `evaluation_seconds` does not depend on it,
# so this measures eval cost cheaply.
#
# Budgets to compare against: Easy 60s train -> 30s eval; Medium 600 -> 300;
# Hard 3600 -> 1800.  (B300 numbers; H100 will differ, treat as an order of
# magnitude plus the K-scaling, which is the transferable part.)
set -u
cd "$(dirname "$0")/.."
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python

measure () {  # tag manifest extra...
  local tag=$1 man=$2; shift 2
  $VENV lab/make_tied.py --tag "$tag" --max-steps 50 --batch-size 64 "$@" >/dev/null
  echo "##### $tag on $man"
  $VENV lab/run_experiment.py --submission "submissions/exp_tied/$tag/submission.py" \
    --manifest "lab/manifests/$man.json" --tag T5-evalcost \
    --note "eval-seconds vs internal iteration count: $tag" --timeout 3000 >/dev/null
  $VENV - <<'PY'
import json
r=[json.loads(l) for l in open('lab/archive.jsonl')][-1]
print("   eval_s=%s  train_s=%s  steps=%s" % (
    [round(x,2) for x in (r.get('evaluation_seconds') or [])],
    [round(x,2) for x in (r.get('training_seconds') or [])],
    r.get('completed_training_steps')))
PY
}

for K in 4 16 64; do
  measure ec_fix_e1_K$K  lab_e1_fs8000_s74  --loops $K --state-mode res
done
measure ec_pond_e1_K64 lab_e1_fs8000_s74 --loops 64 --iter-mode ponder --state-mode res
measure ec_reem_e1_K64 lab_e1_fs8000_s74 --loops 64 --state-mode reembed_st
for K in 4 64; do
  measure ec_fix_m1_K$K  lab_m1_fs8000_s74  --loops $K --state-mode res
  measure ec_fix_hp1_K$K lab_hp1_fs8000_s74 --loops $K --state-mode res
done
