#!/usr/bin/env bash
# Cheap pipeline status (~2s even on a cold Neon endpoint). Never scans
# replay_players. Usage: bash sync/status.sh
set -euo pipefail
cd /home/max/heroes-of-the-storm
set -a; . ./.env; set +a
npx tsx sync/status.ts
