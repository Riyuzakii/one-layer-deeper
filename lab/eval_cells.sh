#!/usr/bin/env bash
# Evaluator cells for plan2/sequential-rnn.  Every row lands in lab/archive.jsonl.
# All fixed_step manifests (BRIEF.md §5: several agents share one GPU).
set -u
cd "$(dirname "$0")/.." || exit 1
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
export CUDA_CACHE_PATH=/home/scratch.arohan_hw/.nv_cache

run () {  # run <submission> <manifest> <tag> <note>
  $V lab/run_experiment.py --submission "$1" --manifest "$2" --tag "$3" --note "$4" \
    --timeout 3600
}

S=submissions/p2-sequential-rnn
M=lab/manifests

run $S/d8/submission.py     $M/lab_e5_fs2000_s74.json rnn-width \
  "LEGAL: fused LSTM D_H=8 (2.9k params, below e5 memorisation capacity), e5, 2000 steps"
run $S/d128/submission.py   $M/lab_e5_fs2000_s74.json rnn-width \
  "LEGAL: fused LSTM D_H=128 (470k params, memorises e5 train), e5, 2000 steps"
run $S/lr0/submission.py    $M/lab_e5_fs2000_s74.json rnn-control \
  "CONTROL (BRIEF2 6.1): lr=0. Any metric at or below this is regression toward init."
run $S/d64_x4/submission.py $M/lab_e5_fs2000_s74.json rnn-depth \
  "LEGAL: fused LSTM D_H=64 with 4 tied encode/carry-scan passes -- depth on the axis the competition is named for"
run submissions/p2-sequential-rnn-neuralgpu/submission.py $M/lab_e5_fs2000_s74.json ngpu \
  "LEGAL: PLAN2 3.5 Neural GPU C=48 W=4 K=12 with Neelakantan gradient noise, e5, 2000 steps"
run $S/d128/submission.py   $M/lab_e1_fs2000_s74.json rnn-width \
  "LEGAL: fused LSTM D_H=128 on e1 -- the dataset where this family reaches train 1.000 by step 750"

echo EVAL_CELLS_DONE
