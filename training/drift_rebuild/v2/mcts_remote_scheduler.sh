#!/usr/bin/env bash
# Keep one v2 MCTS agent running on <host> (one agent saturates a 3080), in
# drift_v2/mcts_remote_order.txt order. Never launches while any job on the
# host is paused (the owner's pause wins); agents already known to the job
# manager on any worker are not relaunched (resume_remote.sh handles paused ones).
# Usage: mcts_remote_scheduler.sh <host>   e.g. max-windows-3090, 3080-gaming-desktop
set -u
H=${1:?usage: mcts_remote_scheduler.sh <host>}
cd /home/max/heroes-of-the-storm
while true; do
  ST=$(training/remote_workers/jobs_remote.sh $H 2>/dev/null)
  if echo "$ST" | grep -qE "^\S+ +paused "; then sleep 300; continue; fi
  if ! echo "$ST" | grep -qE "^v2_\S+ +running +mcts"; then
    # skip agents known on any worker, so schedulers on two hosts never double-launch
    ST_ALL=$(training/remote_workers/jobs_remote.sh all 2>/dev/null)
    NEXT=""
    while read -r n; do
      echo "$ST_ALL" | grep -qE "^$n " || { NEXT=$n; break; }
    done < training/drift_v2/mcts_remote_order.txt
    [ -z "$NEXT" ] && { echo "all remote agents launched"; exit 0; }
    GDP=$(python3 -c "import json;print({j['name']:j for j in json.load(open('training/drift_v2/mcts_jobs_remote.json'))}['$NEXT']['env']['MCTS_GD_PATH'])")
    if ! ssh $H "wsl -e bash -s" <<< "test -f $GDP"; then sleep 300; continue; fi
    echo "$(date) launch $NEXT"
    training/drift_rebuild/v2/launch_mcts_remote.sh "$H" "$NEXT"
  fi
  sleep 300
done
