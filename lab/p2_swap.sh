#!/usr/bin/env bash
# Wait for the lr0 control group (3 cells) and the seed-74 tau group (4 cells)
# to finish, then run the priority-ordered remainder.
set -u
cd /home/scratch.arohan_hw/git/one-layer-deeper/.worktrees/p2-pd-ssm-delta
count() { grep -c "\"tag\": \"$1\"" lab/p2_grid_log.jsonl 2>/dev/null || true; }
while :; do
  a=$(count S5-lr0); b=$(count S3-tau)
  a=${a:-0}; b=${b:-0}
  if [ "$a" -ge 3 ] && [ "$b" -ge 4 ]; then break; fi
  sleep 20
done
echo "lr0 ($a) + tau ($b) complete; starting remainder"
exec bash lab/p2_rest.sh
