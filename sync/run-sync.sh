#!/usr/bin/env bash
# HotS Fever data sync — cron wrapper
# Runs the TypeScript sync job with proper environment and logging.
#
# Crontab entry (every 4 hours):
#   0 */4 * * * /home/max/heroes-of-the-storm/sync/run-sync.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/sync/logs"
LOG_FILE="$LOG_DIR/sync-$(date +%Y-%m-%d_%H%M%S).log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Load nvm / node if needed (common on dev machines)
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" 2>/dev/null || true

# Use the project's .env file
export $(grep -v '^#' "$PROJECT_DIR/.env" 2>/dev/null | xargs) 2>/dev/null || true

cd "$PROJECT_DIR"

echo "=== HotS Fever sync started at $(date -Iseconds) ===" | tee "$LOG_FILE"

# Run sync, capturing output to log file
npx tsx sync/index.ts 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

echo "=== Sync finished at $(date -Iseconds) with exit code $EXIT_CODE ===" | tee -a "$LOG_FILE"

# Rotate logs: keep last 50 log files
cd "$LOG_DIR"
ls -1t sync-*.log 2>/dev/null | tail -n +51 | xargs -r rm --

exit $EXIT_CODE
