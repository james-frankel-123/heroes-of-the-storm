#!/usr/bin/env bash
# Sequential GPU-3 runner (owner cap: at most 2 rebuild GPU processes).
# Input lines: "<name>\t<command>"; log -> drift_rebuild/logs/r6_<name>.log
cd /home/max/heroes-of-the-storm/training
while IFS=$'\t' read -r name cmd; do
  while [ "$(python3 -c 'import sys; sys.path.insert(0,"drift_rebuild"); import run_queue as q; print(sum(q.rebuild_gpu_load().values()))')" -ge 2 ]; do sleep 60; done
  echo "== start $(date)" > drift_rebuild/logs/r6_$name.log
  nohup bash -c "$cmd" >> drift_rebuild/logs/r6_$name.log 2>&1 &
  sleep 120
done < drift_rebuild/logs/next_cmds.txt
wait
