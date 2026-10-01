#!/bin/bash
# Pause every HotS job on <host|all>: checkpoint + clean exit (see README).
# Hosts are paused in parallel; each finishes within ~10 min (hotsjob timeout 540 s).
# Usage: training/remote_workers/pause_remote.sh <host|all> [job-name|all]
set -uo pipefail
source "$(dirname "$0")/_remote.sh"
TARGET=${1:?usage: pause_remote.sh <host|all> [job]}; JOB=${2:-all}
rc=0; pids=()
for h in $(hosts_for "$TARGET"); do
  ( hotsjob "$h" pause "$JOB" 2>&1 | sed "s/^/[$h] /" ; exit "${PIPESTATUS[0]}" ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc = 0 ] && echo "ALL PAUSED" || echo "PAUSE INCOMPLETE (see above)"
exit $rc
