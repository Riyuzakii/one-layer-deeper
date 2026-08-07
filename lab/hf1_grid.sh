#!/usr/bin/env bash
# hf1-scale work: the repair basin at S=7, and LEGAL training from random init
# with the fitting curve and --lr 0 control BRIEF2 §6 requires.
#
# `--bits-list 16,18,20` reproduces hf1's structure offline: several modulus
# sizes with DISJOINT train/held modulus pools, so parameters must be
# modulus-independent.  S = 7, set by the widest (20-bit) modulus.
set -u
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
export CUDA_CACHE_PATH=/home/scratch.arohan_hw/.cuda_cache
OUT=lab/runs
mkdir -p "$OUT"
HF="--bits-list 16,18,20 --n-mod-train 8 --n-mod-held 4 --train-x 250 --held-x 256"

run() { local tag=$1; shift
  echo "=== $tag ==="
  $VENV -u lab/probe_o1.py --jsonl "$OUT/hf1.jsonl" --tag "$tag" "$@" 2>&1
}

case "${1:-fit}" in
ceiling)   # DIAGNOSTIC: the class ceiling on hf1's own shape, soft and hard
  run hf1-construct-div   $HF --recip div   --construct
  run hf1-construct-orac  $HF --recip oracle --construct ;;
basin)     # the headline, at hf1 scale
  for k in 20 400; do
    for s in 0 1 2; do
      run "hf1-k$k-s$s" $HF --recip oracle --corrupt "$k" --seed "$s" \
          --steps 2000 --log-every 2000
    done
  done
  for s in 0 1 2; do
    run "hf1-strat4-s$s" $HF --recip oracle --corrupt 4 --corrupt-mode per_table \
        --seed "$s" --steps 2000 --log-every 2000
  done
  run "hf1-k20-lr0"  $HF --recip oracle --corrupt 20  --seed 0 --lr 0 --steps 2000 --log-every 2000
  run "hf1-k400-lr0" $HF --recip oracle --corrupt 400 --seed 0 --lr 0 --steps 2000 --log-every 2000 ;;
fit)       # LEGAL, random init, with the fitting curve BRIEF2 §6(e) demands
  run hf1-legal-div-s0    $HF --recip div    --seed 0 --steps 8000 --log-every 1000
  run hf1-legal-orac-s0   $HF --recip oracle --seed 0 --steps 8000 --log-every 1000
  run hf1-legal-head-s0   $HF --recip head   --seed 0 --steps 8000 --log-every 1000
  run hf1-legal-div-lr0   $HF --recip div    --seed 0 --steps 2000 --log-every 1000 --lr 0
  run hf1-legal-orac-lr0  $HF --recip oracle --seed 0 --steps 2000 --log-every 1000 --lr 0 ;;
fit2)      # seeds + the small-modulus (fits faster) calibration point
  run hf1-legal-orac-s1   $HF --recip oracle --seed 1 --steps 8000 --log-every 1000
  run hf1-legal-orac-s2   $HF --recip oracle --seed 2 --steps 8000 --log-every 1000 ;;
esac
