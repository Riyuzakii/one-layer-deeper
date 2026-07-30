#!/usr/bin/env bash
# Remainder of the real-task grid, priority-ordered.
#
# Budget note: train exact accuracy on e5 does not lift until ~800 steps, so a
# shorter budget would make every cell vacuously zero -- 1500 steps is the
# minimum informative budget.  PD-SSM runs at 0.24x DeltaProduct's step rate
# (p2_verify check 8), so its real-task sweep is deliberately narrower; the FULL
# 5-tau x 3-task x 2-seed temperature sweep lives on the synthetic probe, where
# it has measured resolving power.
set -u
cd /home/scratch.arohan_hw/git/one-layer-deeper/.worktrees/p2-pd-ssm-delta
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
export CUDA_CACHE_PATH=/home/scratch.arohan_hw/.nv_cache
S=1500
$V -u lab/p2_grid.py --grid nh     --seeds 74 7 --steps $S                    # headline
$V -u lab/p2_grid.py --grid base   --seeds 74 7 --steps $S                    # 3.1 ref
$V -u lab/p2_grid.py --grid eig    --seeds 74 7 --steps $S --nh 1 2 4         # kill crit 6.3
# the seed-74 tau group (0.1 / 1.0 / 10.0 + soft control) runs separately
$V -u lab/p2_grid.py --grid repeat --seeds 74   --steps $S                    # unrolled depth
$V -u lab/p2_grid.py --grid small  --seeds 74   --steps $S                    # carry granularity
$V -u lab/p2_grid.py --grid nh     --seeds 21   --steps $S                    # 3rd seed if time
echo "REST COMPLETE"
