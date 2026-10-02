#!/bin/bash
# Run a command in the oct2026 CPU environment (oct2026_refresh.base_env):
#   bash paper1_revision/oct2026_env.sh python3 -u <script> [args]
# Works on the main box and on the remote workers (paths are resolved there).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$(command -v python3)}
eval "$("$PY" paper1_revision/oct2026_refresh.py --print-env)"
exec "$@"
