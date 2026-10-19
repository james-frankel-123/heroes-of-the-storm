#!/bin/bash
# Weekly offline build and publish of the player cards and name index
# (docs/personalization-site-design.md). Run from the repo root with .env sourced.
# Light: about 4 minutes on 4 cores; reads the research store and the local DB,
# writes the local table personal_accounts, then publish-site.ts pushes the
# site tables (only the ones that changed) to Neon.
set -euo pipefail
cd "$(dirname "$0")/.."
PYDB=/home/linuxbrew/.linuxbrew/bin/python3     # numpy + psycopg2
PYREF=/usr/bin/python3                          # numpy + numba (research reference)
STORE=${SITE_STORE:-full_1017}
echo "[$(date -Is)] site publish: store $STORE"
$PYDB personalization/p3_site_publish.py build --store "$STORE"
$PYREF personalization/p3_site_publish.py check --store "$STORE"
$PYDB personalization/p3_site_publish.py load --min-recent "${SITE_MIN_RECENT:-3}"
cd ..
npx tsx sync/publish-site.ts
echo "[$(date -Is)] site publish done"
