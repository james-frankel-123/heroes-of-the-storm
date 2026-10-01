#!/bin/bash
# Show job manifests (status, last checkpoint, log paths) on <host|all>.
# Usage: training/remote_workers/jobs_remote.sh <host|all> [job-name]
set -uo pipefail
source "$(dirname "$0")/_remote.sh"
for h in $(hosts_for "${1:-all}"); do echo "== $h"; hotsjob "$h" status ${2:-all}; done
