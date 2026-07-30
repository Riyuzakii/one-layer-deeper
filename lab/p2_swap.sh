#!/usr/bin/env bash
# Wait for the lr0 control group (3 cells) to finish, then replace the old
# 3-seed plan with the priority-ordered reduced plan.
set -u
cd /home/scratch.arohan_hw/git/one-layer-deeper/.worktrees/p2-pd-ssm-delta
while [ "$(wc -l < lab/p2_grid_log.jsonl 2>/dev/null || echo 0)" -lt 3 ]; do sleep 20; done
echo "lr0 group complete; swapping plans"
# kill only the p2_grid.py driver and its shell wrapper (this branch's own PIDs)
for p in $(ps -eo pid,args | grep "[p]2_grid.py" | awk '{print $1}'); do kill "$p" 2>/dev/null; done
sleep 5
for p in $(ps -eo pid,args | grep "[r]un_experiment.py" | grep "plan2-pd-ssm-delta" | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
sleep 3
exec bash lab/p2_rest.sh
