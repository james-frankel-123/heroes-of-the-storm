#!/bin/bash
# Parse a cohort's replays on a remote worker's CPU and bring the timelines back.
# Default host is the 3090, whose cores 24-35 are the HotS share (tennis has 0-23).
#
#   training/replays/remote_parse.sh setup  [host]            one-time venv (heroprotocol, mpyq)
#   training/replays/remote_parse.sh push   [host] [cohort]   code + that cohort's files
#   training/replays/remote_parse.sh run    [host]            detached parse (setsid, log)
#   training/replays/remote_parse.sh status [host]
#   training/replays/remote_parse.sh pull   [host]            timelines back to the RAID
#
# Sealed files never leave this box: the push list comes from the manifest with the
# sealed cohort and sealed IDs removed, and the parser refuses them again on the worker.
set -euo pipefail
CMD=${1:?usage: remote_parse.sh setup|push|run|status|pull [host] [cohort]}
HOST=${2:-max-windows-3090}
COHORT=${3:-stratified_b}
CORES=${HOTS_PARSE_CORES:-24-35}
WORKERS=${HOTS_PARSE_WORKERS:-12}
ROOT=/home/max/hots-replays
cd "$(dirname "$0")/../.."
source training/remote_workers/_remote.sh
R=(nice -n 19 rsync -rlt --rsync-path="$(remote_rsync "$HOST")")
RD=/home/max/hots/replays          # on the worker: code/, data/, venv/, parse.log

remote() { ssh "$HOST" "$(remote_shell "$HOST")"; }

case "$CMD" in
  setup)
    remote <<'EOF'
set -e
mkdir -p ~/hots/replays/code ~/hots/replays/data
[ -x ~/hots/replays/venv/bin/python ] || ~/hots/bin/uv venv -q ~/hots/replays/venv
VIRTUAL_ENV=~/hots/replays/venv ~/hots/bin/uv pip install -q heroprotocol==2.55.4.91756 mpyq==0.2.5
~/hots/replays/venv/bin/python -c "import sys,types; sys.modules['imp']=types.ModuleType('imp'); import heroprotocol, mpyq; print('replay venv ok')"
EOF
    ;;
  push)
    "${R[@]}" --relative training/replays/./parse_replay.py training/replays/./run_parse.py \
      training/replays/./ref/heroes_talents_ref.json training/replays/./cohorts/sealed_ids.txt \
      "$HOST:$RD/code/"
    LIST=$(mktemp)
    python3 - "$ROOT/manifest.jsonl" "$COHORT" training/replays/cohorts/sealed_ids.txt > "$LIST" <<'EOF'
import json, sys
sealed = {int(x) for x in open(sys.argv[3]) if x.strip()}
seen = set()
for line in open(sys.argv[1]):
    r = json.loads(line)
    if (r["status"] in ("ok", "linked") and r["cohort"] == sys.argv[2] and r["cohort"] != "sealed"
            and r["replay_id"] not in sealed and not r["path"].startswith("sealed/")
            and r["path"] not in seen):
        seen.add(r["path"]); print(r["path"])
EOF
    echo "pushing $(wc -l < "$LIST") files of cohort $COHORT"
    "${R[@]}" --files-from="$LIST" --stats "$ROOT/" "$HOST:$RD/data/" | grep -E "files transferred|Total transferred"
    "${R[@]}" "$LIST" "$HOST:$RD/data/list_$COHORT.txt"
    rm -f "$LIST"
    ;;
  run)
    remote <<EOF
set -e
cd $RD/code
for L in $RD/data/list_*.txt; do
  setsid nohup taskset -c $CORES nice -n 19 $RD/venv/bin/python run_parse.py --root $RD/data \
    --list "\$L" --workers $WORKERS >> $RD/parse.log 2>&1 < /dev/null &
  wait \$! 2>/dev/null || true
done &
disown
echo "launched on cores $CORES with $WORKERS workers; log $RD/parse.log"
EOF
    ;;
  status)
    remote <<EOF
tail -3 $RD/parse.log; pgrep -fc run_parse.py || true
EOF
    ;;
  pull)
    "${R[@]}" --stats "$HOST:$RD/data/timelines/" "$ROOT/timelines/" | grep -E "files transferred|Total transferred"
    ;;
  *) echo "unknown command $CMD"; exit 2 ;;
esac
