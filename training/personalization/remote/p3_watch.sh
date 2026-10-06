#!/bin/bash
# Stream a P3 job's stage events (start/done/failed lines of p3_pipe) and its
# final state from a worker; exits when the job is no longer running.
# Usage: p3_watch.sh <host> <job> [poll seconds]
HOST=${1:?host}; JOB=${2:?job}; POLL=${3:-120}
SEEN=0
while true; do
  OUT=$(timeout 90 ssh -o ConnectTimeout=30 "$HOST" "wsl -e bash -s" <<REMOTE 2>/dev/null
source ~/hots/env.sh
echo "STATE \$(python ~/hots/repo/training/remote_workers/hotsjob.py state $JOB 2>/dev/null)"
grep -E "^\[p3_pipe\]|Traceback|Error|MemoryError|Killed" ~/hots/logs/$JOB.log 2>/dev/null
REMOTE
)
  if [ -z "$OUT" ]; then echo "WATCH: $HOST unreachable"; sleep "$POLL"; continue; fi
  LINES=$(echo "$OUT" | grep -v "^STATE")
  N=$(echo "$LINES" | grep -c .)
  if [ "$N" -gt "$SEEN" ]; then echo "$LINES" | tail -n +$((SEEN + 1)) | grep -E "done|failed|complete|Traceback|Error|Killed|signal"; SEEN=$N; fi
  ST=$(echo "$OUT" | grep "^STATE" | awk '{print $2}')
  case "$ST" in running|pausing|starting|"") ;; *) echo "JOB $JOB $ST"; exit 0;; esac
  sleep "$POLL"
done
