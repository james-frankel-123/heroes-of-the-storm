#!/bin/bash
# Daily NAS backup of the paper sources and drafts, which are kept out of git.
# Each run writes a dated snapshot under /archive-backup/papers/; unchanged files are
# hard-linked to the previous snapshot, so a snapshot costs only what changed.
#   sync/backup_papers.sh            take today's snapshot (cron, daily)
# Sources: heroes-of-the-storm/paper/ and the edd-writer repo (narrative docs, review
# comments, fact ledgers; the paper 2 draft itself is a symlink into paper/paper 2/draft).
set -euo pipefail
DEST=/archive-backup/papers
SRC_PAPER=/home/max/heroes-of-the-storm/paper/
SRC_EDDW=/home/max/edd-writer/
mountpoint -q /archive-backup || { echo "NAS not mounted at /archive-backup" >&2; exit 1; }
mkdir -p "$DEST"
TODAY=$(date +%F)
PREV=$(ls -1d "$DEST"/20??-??-?? 2>/dev/null | grep -v "/$TODAY\$" | tail -1 || true)
for pair in "paper:$SRC_PAPER" "edd-writer:$SRC_EDDW"; do
  name=${pair%%:*}; src=${pair#*:}
  link=(); [ -n "$PREV" ] && [ -d "$PREV/$name" ] && link=(--link-dest="$PREV/$name")
  mkdir -p "$DEST/$TODAY/$name"
  rsync -a --delete "${link[@]}" --exclude '__pycache__/' --exclude '*.aux' --exclude '*.log' \
    --exclude '*.out' "$src" "$DEST/$TODAY/$name/"
done
echo "[$(date -Is)] papers snapshot $DEST/$TODAY ($(du -sh "$DEST/$TODAY" | cut -f1) apparent, prev ${PREV:-none})"
