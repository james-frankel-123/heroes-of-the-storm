#!/usr/bin/env bash
# Run the six stale-1yr MCTS jobs on GPU 3, keeping at most 2 rebuild GPU
# processes alive (owner cap, 2026-09-30).
cd /home/max/heroes-of-the-storm/training
i=0
while IFS= read -r cmd; do
  while [ "$(python3 -c 'import sys; sys.path.insert(0,"drift_rebuild"); import run_queue as q; print(sum(q.rebuild_gpu_load().values()))')" -ge 2 ]; do sleep 60; done
  echo "== start $(date)" > drift_rebuild/logs/r6_s1oof_era_s$i.log
  nohup bash -c "$cmd" >> drift_rebuild/logs/r6_s1oof_era_s$i.log 2>&1 &
  i=$((i+1)); sleep 120
done < drift_rebuild/logs/s1_cmds.txt
wait
