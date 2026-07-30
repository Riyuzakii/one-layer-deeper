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
G)  # plan2/pd-ssm-delta reports that in structured recurrences hard
    # straight-through FITS WORSE and GENERALISES BETTER than soft (train 0.430
    # vs 0.141, held 0.002 vs 0.012).  `--hard-train` snaps EVERY state --
    # transition matrices, carry vectors and digit outputs -- so this is the
    # matched pair: identical parameters, differing only in discretisation.
    for fam in colsoftmax sthard; do for seed in 0 1 2; do
      run "G-hard-$fam-s$seed" --modulus 323 --d 16 --family "$fam" \
          --hard-train --seed "$seed" --steps "$STEPS"
    done; done
    # B cells lost when the parent shell was killed
    run "B-323-d8-colsoftmax-s2"  --modulus 323 --d 8  --family colsoftmax --seed 2 --steps "$STEPS"
    for seed in 0 1 2; do
      run "B-323-d32-colsoftmax-s$seed" --modulus 323 --d 32 --family colsoftmax \
          --seed "$seed" --steps "$STEPS"
    done ;;
D)  for k in 5 20 50 100 400 700; do
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
F)  # Coordinator's request from plan2/sequential-rnn: that branch measured
    # train_exact climbing 134x with hidden size (D_H 4->256) while held-out
    # never moved -- "small states don't generalise, they just fail to fit".
    # `d` here is the monoid STATE ALPHABET, and the constructed solution needs
    # only 2 (carry/borrow) or 3 (compare), so d is slack, not capacity.  Widen
    # the sweep to a 16x range and check whether the same shape appears.
    for d in 4 64; do for seed in 0 1 2; do
      run "F-323-d$d-colsoftmax-s$seed" --modulus 323 --d "$d" \
          --family colsoftmax --seed "$seed" --steps "$STEPS"
    done; done
    # continuous vs discrete state at the widest d: sthard is a straight-through
    # column one-hot (PD-SSM), i.e. the discrete-state arm of the same sweep
    for seed in 0 1 2; do
      run "F-323-d64-sthard-s$seed" --modulus 323 --d 64 --family sthard \
          --seed "$seed" --steps "$STEPS"
    done
    run "F-lr0-323-d4"  --modulus 323 --d 4  --family colsoftmax --lr 0 --steps 200
    run "F-lr0-323-d64" --modulus 323 --d 64 --family colsoftmax --lr 0 --steps 200 ;;
*)  echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "done -> $J"
