#!/bin/bash
# Full expert-study analysis on a frozen export.
#   bash run_all.sh <out_dir> [ratings.csv]
# Without ratings.csv it first makes a read-only export from Neon into
# <out_dir>. Runs the primary (strict partials) analysis in <out_dir>/primary
# and the lenient-partials sensitivity in <out_dir>/partials_lenient.
# Light: nice 19, cores 48-63 (R in docker with 2 cores).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
OUT=$(mkdir -p "$1" && cd "$1" && pwd)
CSV=${2:-}
if [ -z "$CSV" ]; then bash "$HERE/export_ratings.sh" "$OUT"; CSV="$OUT/ratings.csv"; fi
CSV=$(cd "$(dirname "$CSV")" && pwd)/$(basename "$CSV")
( cd "$REPO" && sha256sum data/rating-items.json oss-export/docs/prereg_expert_study_v3.md "$CSV" ) > "$OUT/inputs.sha256"
( cd "$REPO" && nice -n 19 taskset -c 48-63 npx tsx "$HERE/dump_sequences.ts" data/rating-items.json "$HERE/config.json" "$OUT/sequences.json" )
for rule in strict lenient; do
  D="$OUT/$([ $rule = strict ] && echo primary || echo partials_lenient)"
  nice -n 19 taskset -c 48-63 python3 "$HERE/prepare.py" "$CSV" "$OUT/sequences.json" "$D" --partials $rule
  docker run --rm --cpuset-cpus=48-49 --user "$(id -u):$(id -g)" -v "$HERE:/code:ro" -v "$D:/data" \
    hots-r-lme4:4.4.1 Rscript /code/models.R /data /data/models.json /code/config.json
  nice -n 19 taskset -c 48-63 python3 "$HERE/post.py" "$D"
done
echo "done: $OUT"
