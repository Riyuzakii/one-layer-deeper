#!/usr/bin/env bash
# Experiment A — does the ALREADY-BUILT parallel-prefix carry TRAIN better?
#
# `alu-depth` built `--scan-mode prefix` (a learned 3-element propagate/generate
# carry semigroup composed by Hillis-Steele) and verified its CONSTRUCTED
# ceiling at 1.000, but it is absent from that branch's `depth_grid.sh` cells,
# so it was never put through a learning run — only construct-accuracy and
# throughput.  This script runs the matched learning comparison, plus the
# `--lr 0` control that BRIEF2 §6.1 requires before interpreting any shift.
#
# usage: lab/scan_grid.sh [OUT]
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-/home/scratch.arohan_hw/.nv_cache}
OUT=${1:-lab/runs/scan_grid.log}
STEPS=${STEPS:-3000}
mkdir -p "$(dirname "$OUT")"

run() {  # run <tag> <extra args...>
  local tag=$1; shift
  echo "### $tag :: $*" >> "$OUT"
  $V lab/probe_alu_depth.py --mul-mode tree --reduce-mode quotient \
      --eval-hard --steps "$STEPS" --log-every 500 --tag "$tag" "$@" >> "$OUT" 2>&1
}

for mod_slots in "323 3" "10403 5"; do
  set -- $mod_slots; MOD=$1; S=$2
  TX=250; [ "$MOD" = 10403 ] && TX=8000
  for sc in serial prefix; do
    for seed in 0 1 2; do
      run "A-${MOD}-${sc}-s${seed}" --modulus "$MOD" --slots "$S" --train-x "$TX" \
          --scan-mode "$sc" --seed "$seed"
    done
    # --lr 0 control: what the metric row reads at random init
    run "A-${MOD}-${sc}-lr0" --modulus "$MOD" --slots "$S" --train-x "$TX" \
        --scan-mode "$sc" --seed 0 --lr 0 --steps 1 --log-every 1
  done
done
echo "done -> $OUT"
