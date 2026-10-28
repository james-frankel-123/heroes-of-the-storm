#!/bin/bash
# Weekly offline build and publish of the player cards and name index
# (docs/personalization-site-design.md). Run from the repo root with .env sourced.
# About 5 minutes on 4 cores and ~17 GB of memory. Reads the research store, the
# definitive build's caches (the chosen model's state writer) and the local DB,
# writes the local table personal_accounts, then publish-site.ts pushes the site
# tables (only the ones that changed) to Neon.
#
# SITE_STORE (merged history store) and SITE_DEF_TAG (definitive build of the
# model) are fixed snapshots: until they are rebuilt, a weekly run refreshes
# names, MMR, game counts and the recent-games filter, not the skill estimates.
set -euo pipefail
cd "$(dirname "$0")/.."
PYDB=/home/linuxbrew/.linuxbrew/bin/python3     # numpy + psycopg2
PYREF=/usr/bin/python3                          # numpy + numba (state writer)
STORE=${SITE_STORE:-def1027}
DEF=${SITE_DEF_TAG:-def1027}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
echo "[$(date -Is)] site publish: store $STORE, model chosen ($DEF)"
$PYDB personalization/p3_site_publish.py build --store "$STORE"
(
  . personalization/remote/def_env.sh "$PWD" "$DEF"
  # the state writer must still reproduce the chosen arm's stored predictions (guards code drift)
  $PYREF personalization/p3_rcf_states.py selfcheck --halves 0 --no-filtercheck
  $PYREF personalization/p3_site_publish.py states --def-tag "$DEF"
  $PYREF personalization/p3_site_publish.py encode
  $PYREF personalization/p3_site_publish.py check --def-tag "$DEF"
)
$PYDB personalization/p3_site_publish.py load --min-recent "${SITE_MIN_RECENT:-3}"
cd ..
npx tsx sync/publish-site.ts
echo "[$(date -Is)] site publish done"
