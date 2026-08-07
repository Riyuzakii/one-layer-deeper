#!/usr/bin/env bash
# THE SOURCE-DIFFICULTY LADDER -- "measure a signal's basin before optimising
# it" (BRIEF2 §6.6).
#
# A curriculum can only work if the easy end is actually learnable: it needs
# something correct at the source to transfer.  `alu-credit` found the opposite
# for its chain-length curriculum ("the short chain fits the DEGENERATE solution
# faster; it does not find the algorithm"), and that is what killed it.
#
# This ladder holds the GRAPH FIXED (S=7, 183 sequential soft steps, one set of
# digit tables) and varies only the modulus SIZE the model is trained on, from
# far below hf1's floor (10 bits, 3 decimal digits) up to hf1's ceiling (20).
# If nothing on the ladder learns, no weighting between its rungs can help --
# which closes the method without needing to sweep schedules at all.
set -uo pipefail
R=/home/scratch.arohan_hw/git/one-layer-deeper/.worktrees/hf-curriculum-hf1
S="--steps 3000 --lr 3e-2 --log-every 500 --seed 0"
run() { bash "$R/lab/curric_run.sh" "$@"; }

run LAD-b10 $S --id-bits 10 --ood-bits 11 --n-mod 4  --n-mod-held 2 --n-x 300  --n-held-x 64
run LAD-b12 $S --id-bits 12 --ood-bits 13 --n-mod 8  --n-mod-held 4 --n-x 800  --n-held-x 64
run LAD-b14 $S --id-bits 14 --ood-bits 15 --n-mod 16 --n-mod-held 8 --n-x 1024 --n-held-x 64
run LAD-b16 $S --id-bits 16 --ood-bits 17 --n-mod 16 --n-mod-held 8 --n-x 1024 --n-held-x 64
run LAD-b20 $S --id-bits 20 --ood-bits 21 --n-mod 16 --n-mod-held 8 --n-x 1024 --n-held-x 64
