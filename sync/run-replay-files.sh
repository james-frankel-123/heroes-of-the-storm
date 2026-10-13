#!/usr/bin/env bash
# Cron wrapper for sync/fetch-replay-files.py (replay_download pool, separate from the
# replay_data pool the backfills use, so it does not share their lock).
#
# Crontab entries:
#   10 */2 * * * flock -n /home/max/heroes-of-the-storm/sync/.replay-files.lock \
#     /home/max/heroes-of-the-storm/sync/run-replay-files.sh run >> /home/max/heroes-of-the-storm/sync/logs/replay-files.log 2>&1
#   0 6 * * 0 flock -w 7200 /home/max/heroes-of-the-storm/sync/.replay-files.lock \
#     /home/max/heroes-of-the-storm/sync/run-replay-files.sh backup >> /home/max/heroes-of-the-storm/sync/logs/replay-files.log 2>&1
#
# Light IO and network only: nice 19, idle IO class, HotS cores 48-63.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
set -a
. ./.env
set +a
PY=/home/linuxbrew/.linuxbrew/bin/python3
RUN="nice -n 19 ionice -c3 taskset -c 48-63 $PY sync/fetch-replay-files.py"

case "${1:-run}" in
  run)
    echo "=== replay-files run started $(date -Iseconds) ==="
    timeout --signal=TERM --kill-after=2m 115m $RUN run --cron
    echo "=== replay-files run exited $? at $(date -Iseconds) ==="
    ;;
  backup)
    echo "=== replay-files backup started $(date -Iseconds) ==="
    $RUN backup
    $RUN probe-retention
    echo "=== replay-files backup exited $? at $(date -Iseconds) ==="
    ;;
  *) echo "usage: $0 run|backup"; exit 2 ;;
esac
