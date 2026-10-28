#!/bin/bash
# Check every answer file against its walk-query file before a replay pass (review C2/C11).
# Usage (worker, from ~/hots/repo/training, P3_CACHE/P3_RESULTS/P3_RCF_ROWS of the RECORD pass):
#   bash personalization/remote/p3t_check_all.sh <query dir> <answer dir>
# Fails unless the answer directory holds exactly the query files and every check passes.
set -euo pipefail
Q=${1:?query dir}; A=${2:?answer dir}
qs=$(cd "$Q" && ls *.npz | sort)
as=$(cd "$A" && ls *.npz | sort)
[ -n "$qs" ] || { echo "no query files in $Q"; exit 1; }
if [ "$qs" != "$as" ]; then
  echo "answer files differ from query files"; diff <(echo "$qs") <(echo "$as") || true; exit 1
fi
n=0
for f in $qs; do
  echo "=== check ${f%.npz}"
  python -u personalization/p3t_check_answers.py "$Q" "$A" "${f%.npz}"
  python -u - "$A/$f" <<'PY'
import sys, numpy as np
z = np.load(sys.argv[1])
assert "sd_display" in z.files, "answer file lacks sd_display (old format)"
for k in ("s", "sd", "n"):
    assert np.isfinite(z[k]).all(), f"non-finite {k}"
assert (z["sd"] >= 0).all()
print("fields ok:", sorted(z.files))
PY
  n=$((n + 1))
done
echo "ALL ANSWERS CHECKED: $n files"
