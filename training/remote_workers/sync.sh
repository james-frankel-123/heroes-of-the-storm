#!/bin/bash
# Sync code (and optionally data) from this box to a remote worker's WSL
# ~/hots/repo. rsync runs in WSL through `--rsync-path="wsl rsync"`.
# Usage: training/remote_workers/sync.sh <host> [code|data|all]   (default: code)
set -euo pipefail
HOST=${1:?usage: sync.sh <host> [code|data|all]}
WHAT=${2:-code}
cd "$(dirname "$0")/../.."
DEST=/home/max/hots/repo/
source training/remote_workers/_remote.sh
R=(nice -n 19 rsync -rlt --rsync-path="$(remote_rsync "$HOST")" -z --compress-choice=zstd --compress-level=3)
if [ "$WHAT" = code ] || [ "$WHAT" = all ]; then
  "${R[@]}" --max-size=2m --prune-empty-dirs --filter="merge training/remote_workers/code.filter" \
    --stats ./ "$HOST:$DEST" | grep -E "files transferred|Total transferred"
fi
if [ "$WHAT" = data ] || [ "$WHAT" = all ]; then
  LIST=$(mktemp)
  training/remote_workers/data_manifest.sh > "$LIST"
  "${R[@]}" --files-from="$LIST" --partial --stats ./ "$HOST:$DEST" | grep -E "files transferred|Total transferred"
  rm -f "$LIST"
fi
