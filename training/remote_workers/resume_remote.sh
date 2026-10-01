#!/bin/bash
# Relaunch every paused HotS job on <host|all> from its latest checkpoint.
# Usage: training/remote_workers/resume_remote.sh <host|all> [job-name|all]
set -uo pipefail
source "$(dirname "$0")/_remote.sh"
TARGET=${1:?usage: resume_remote.sh <host|all> [job]}; JOB=${2:-all}
for h in $(hosts_for "$TARGET"); do hotsjob "$h" resume "$JOB" 2>&1 | sed "s/^/[$h] /"; done
