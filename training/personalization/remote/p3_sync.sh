#!/bin/bash
# Sync P3 code and/or data to a remote worker's ~/hots/repo.
# Usage: training/personalization/remote/p3_sync.sh <host> [code|data|all]
set -euo pipefail
HOST=${1:?usage: p3_sync.sh <host> [code|data|all]}
WHAT=${2:-code}
cd "$(dirname "$0")/../../.."
if [ "$WHAT" = code ] || [ "$WHAT" = all ]; then
  training/remote_workers/sync.sh "$HOST" code
fi
if [ "$WHAT" = data ] || [ "$WHAT" = all ]; then
  LIST=$(mktemp)
  training/personalization/remote/p3_data_manifest.sh > "$LIST"
  nice -n 19 rsync -rlt --rsync-path="wsl rsync" -z --compress-choice=zstd --compress-level=3 \
    --files-from="$LIST" --partial --stats ./ "$HOST:/home/max/hots/repo/" | grep -E "files transferred|Total transferred"
  rm -f "$LIST"
fi
