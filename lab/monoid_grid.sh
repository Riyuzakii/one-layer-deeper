#!/usr/bin/env bash
# Experiments B-E on lab/probe_monoid.py.  usage: lab/monoid_grid.sh <stage>
#
#   B  legal training from random init: constraint family x d x seed
#   C  --lr 0 controls (BRIEF2 6.1) + impl=serial vs scan gradient-identity check
#   D  repair basin: construct, corrupt k cells, train on the LEGAL objective,
#      count exact repairs.  This is the conditioning measurement -- lab/RESUME
#      records 0/5 repaired at k=20 for DigitALU's end-of-chain label.
#   E  modulus-independence: sampled 11-bit moduli, disjoint train/held pools
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-/home/scratch.arohan_hw/.nv_cache}
STAGE=${1:-B}
STEPS=${STEPS:-2000}
J=lab/runs/monoid_${STAGE}.jsonl
LOG=lab/runs/monoid_${STAGE}.log
mkdir -p lab/runs

run() { local tag=$1; shift
  echo "### $tag :: $*" >> "$LOG"
  $V lab/probe_monoid.py --jsonl "$J" --tag "$tag" --log-every 500 "$@" >> "$LOG" 2>&1
}

case "$STAGE" in
B)  for fam in colsoftmax sthard dsink orth dense; do
      for seed in 0 1 2; do
        run "B-323-d16-$fam-s$seed" --modulus 323 --d 16 --family "$fam" \
            --seed "$seed" --steps "$STEPS"
      done
    done
    for d in 8 32; do for seed in 0 1 2; do
        run "B-323-d$d-colsoftmax-s$seed" --modulus 323 --d "$d" \
            --family colsoftmax --seed "$seed" --steps "$STEPS"
    done; done ;;
C)  for fam in colsoftmax sthard dense; do
      run "C-lr0-323-d16-$fam" --modulus 323 --d 16 --family "$fam" --lr 0 --steps 500
    done
    for d in 8 16 32; do
      run "C-lr0-323-d$d" --modulus 323 --d "$d" --family colsoftmax --lr 0 --steps 500
    done
    # scan vs serial prefix: same function, so this must reproduce to fp noise
    for impl in scan serial; do for seed in 0 1; do
      run "C-impl-$impl-s$seed" --modulus 323 --d 16 --family colsoftmax \
          --impl "$impl" --seed "$seed" --steps 1000
    done; done ;;
D)  for k in 5 20 50 100 200 400 700; do
      for seed in 0 1 2; do
        run "D-corrupt$k-s$seed" --modulus 323 --d 16 --family colsoftmax \
            --corrupt "$k" --seed "$seed" --steps "$STEPS" --lr 1e-2
      done
    done ;;
E)  for seed in 0 1 2; do
      run "E-samp11-d16-s$seed" --modulus 0 --bits 11 --n-mod-train 8 --n-mod-held 4 \
          --train-x 400 --held-x 256 --slots 4 --d 16 --family colsoftmax \
          --seed "$seed" --steps "$STEPS"
    done
    run "E-samp11-lr0" --modulus 0 --bits 11 --n-mod-train 8 --n-mod-held 4 \
        --train-x 400 --held-x 256 --slots 4 --d 16 --family colsoftmax --lr 0 --steps 500 ;;
*)  echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "done -> $J"
