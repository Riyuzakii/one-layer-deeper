#!/usr/bin/env bash
# HIGH-RESOLUTION repair basin at the decisive k values, 20 reps each, so the
# ADD-ONLY vs DigitALU comparison is not read off 5-rep binomial noise.
# Includes FRACTION-MATCHED cells (AddOnly-tied has 237 free cells, DigitALU-
# tied has 337, so equal k is NOT equal corruption).
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.."
mkdir -p lab/logs
R=20

# --- ADD-ONLY variant A (learned pick) ---
$V lab/probe_addsearch.py --modulus 323 --slots 3 --train-x 250 \
  --tie sym,inv --repair-ks 10,14,20,28,50 --repair-reps $R --seed 501 \
  --tag hiA_tied --jsonl lab/basin2_runs.jsonl > lab/logs/hiA_tied.log 2>&1
$V lab/probe_addsearch.py --modulus 323 --slots 3 --train-x 250 \
  --repair-ks 10,20,50,68 --repair-reps $R --seed 501 \
  --tag hiA_untied --jsonl lab/basin2_runs.jsonl > lab/logs/hiA_untied.log 2>&1
# --- ADD-ONLY variant B (fixed pick, BORDERLINE) ---
$V lab/probe_addsearch.py --modulus 323 --slots 3 --train-x 250 --pick fixed \
  --tie sym,inv --repair-ks 10,20,50 --repair-reps $R --seed 501 \
  --tag hiB_tied --jsonl lab/basin2_runs.jsonl > lab/logs/hiB_tied.log 2>&1

# --- CONTROL: DigitALU, same searcher, same GPU, same seeds ---
: > lab/logs/hiD_tied.log
for k in 10 14 20 28 50; do
  $V lab/probe_search.py --modulus 323 --slots 3 --train-x 250 \
    --tie sym,inv --repair $k --repair-reps $R --seed 501 \
    --tag hiD_tied_k$k --jsonl lab/basin2_runs_dalu.jsonl \
    >> lab/logs/hiD_tied.log 2>&1
done
: > lab/logs/hiD_untied.log
for k in 10 20 50 68; do
  $V lab/probe_search.py --modulus 323 --slots 3 --train-x 250 \
    --repair $k --repair-reps $R --seed 501 \
    --tag hiD_untied_k$k --jsonl lab/basin2_runs_dalu.jsonl \
    >> lab/logs/hiD_untied.log 2>&1
done
echo DONE
