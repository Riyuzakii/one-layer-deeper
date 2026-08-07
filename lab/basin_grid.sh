#!/usr/bin/env bash
# Repair basin of the LEGAL end-of-chain label, by learned-op depth.
#
# Protocol is `plan2/matrix-scan` §5 verbatim: construct the exact solution
# (DIAGNOSTIC start point), randomise k table cells, train on the plain
# end-to-end label CE only (LEGAL objective, no laws, no teacher forcing), then
# count how many corrupted cells argmax-match the truth again.  Same modulus
# (323), same operand split (250 train / 38 held), same optimiser (AdamW
# lr 3e-2), same 2000 steps, same 3 seeds -- so the rows compose with that
# branch's 0/5 @ depth 39 and 14/400 @ depth 12.
set -u
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.." || exit 1
export CUDA_CACHE_PATH=/home/scratch.arohan_hw/.cuda_cache
OUT=lab/runs
mkdir -p "$OUT"
JSONL=$OUT/basin.jsonl
STEPS=${STEPS:-2000}

run() {  # run <tag> <args...>
  local tag=$1; shift
  echo "=== $tag ==="
  $VENV -u lab/probe_o1.py --jsonl "$JSONL" --tag "$tag" --log-every "$STEPS" \
        --steps "$STEPS" "$@" 2>&1 | grep -v '^\[.*step=     0'
}

case "${1:-basin}" in
basin)   # uniform corruption, the directly comparable protocol
  for k in 5 20 50 100 400; do
    for s in 0 1 2; do
      run "o1-k$k-s$s" --modulus 323 --recip oracle --corrupt "$k" --seed "$s"
    done
  done ;;
strat)   # k cells of EVERY table -> repair rate resolved by learned-op depth
  for k in 4 16; do
    for s in 0 1 2; do
      run "o1-strat$k-s$s" --modulus 323 --recip oracle --corrupt "$k" \
          --corrupt-mode per_table --seed "$s"
    done
  done ;;
div)     # same, with the reciprocal learned too (no oracle anywhere)
  for k in 20 400; do
    for s in 0 1 2; do
      run "div-k$k-s$s" --modulus 323 --recip div --corrupt "$k" --seed "$s"
    done
  done
  for s in 0 1 2; do
    run "div-strat4-s$s" --modulus 323 --recip div --corrupt 4 \
        --corrupt-mode per_table --seed "$s"
  done ;;
lr0)     # BRIEF2 6.1: the control every claim needs
  for k in 20 400; do
    run "o1-k$k-lr0" --modulus 323 --recip oracle --corrupt "$k" --seed 0 --lr 0
  done
  run "o1-strat4-lr0" --modulus 323 --recip oracle --corrupt 4 \
      --corrupt-mode per_table --seed 0 --lr 0
  run "o1-strat16-lr0" --modulus 323 --recip oracle --corrupt 16 \
      --corrupt-mode per_table --seed 0 --lr 0 ;;
mono)    # matched control: MonoidALU (learned-op depth 12) on this box
  for k in 20 400; do
    for s in 0 1 2; do
      echo "=== mono-k$k-s$s ==="
      $VENV -u lab/probe_monoid.py --jsonl "$OUT/basin_mono.jsonl" \
            --tag "mono-k$k-s$s" --modulus 323 --corrupt "$k" --seed "$s" \
            --steps "$STEPS" --log-every "$STEPS" 2>&1 | grep -v 'step=     0'
    done
  done ;;
esac
