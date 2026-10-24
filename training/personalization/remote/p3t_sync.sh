#!/bin/bash
# P3 tournament: sync code plus the tournament's small config files to a worker.
# Usage: training/personalization/remote/p3t_sync.sh <host>
set -euo pipefail
HOST=${1:?host}
cd "$(dirname "$0")/../../.."
source training/remote_workers/_remote.sh
training/remote_workers/sync.sh "$HOST" code | tail -1
nice -n 19 rsync -t --rsync-path="$(remote_rsync "$HOST")" training/personalization/p3t_vintages.json training/personalization/p3t_explore.json training/personalization/p3t_excluded_replays.json \
  "$HOST:/home/max/hots/repo/training/personalization/"
