#!/bin/bash
# Read-only export of the non-test expert ratings from Neon, plus sha256.
#   bash training/paper1_revision/expert_analysis/export_ratings.sh <out_dir>
# Uses the hots-pg container's psql (no psql on the host); the session is set
# read-only before the copy. Writes ratings.csv and ratings.csv.sha256.
set -euo pipefail
OUT=${1:?out dir}
mkdir -p "$OUT"
cd "$(dirname "$0")/../../.."
set -a; source .env; set +a
docker exec -i hots-pg psql "$NEON_DATABASE_URL" -v ON_ERROR_STOP=1 -q -X \
  -c "set default_transaction_read_only = on" \
  -c "\copy (select id, rater, item_id, p_team_a, better_team, confidence, ms_taken, team_a_is_team0, is_test, to_char(created_at, 'YYYY-MM-DD\"T\"HH24:MI:SS.US') as created_at, block from draft_ratings where not is_test order by rater, item_id) to stdout with csv header" \
  > "$OUT/ratings.csv"
( cd "$OUT" && sha256sum ratings.csv > ratings.csv.sha256 )
echo "exported $(($(wc -l < "$OUT/ratings.csv") - 1)) rows -> $OUT/ratings.csv ($(cut -c1-16 "$OUT/ratings.csv.sha256")…)"
