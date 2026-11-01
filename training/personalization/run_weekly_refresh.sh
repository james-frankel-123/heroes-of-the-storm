#!/bin/bash
# Weekly refresh of the player cards (results/P3_WEEKLY.md): the def1027 chosen
# model with every coefficient frozen (no refit), walked over all games through
# yesterday, so the site's skill estimates stay current.
#   1. post-cap games (2.57, from 2026-09-28, sealed replays excluded) appended to
#      the frozen build's pinned inputs: WP, exports, history store capped today
#   2. A1 caches (pf_build.sh), the frozen chosen arm (pf_score.sh)
#   3. regression: the walk reproduces def1027 on pre-cap days (arm and states)
#   4. cards (p3_site_publish build/states/encode/check, anchored band) in a
#      staging dir; refit check (reports only)
#   5. with --publish: load the local table and push to Neon (publish-site.ts)
# Each stage runs once per build (markers in cache_def_<D>/.wk_done); runtime and
# peak RAM per stage go to logs/weekly_<D>.times.
# Usage (repo root, .env sourced): training/personalization/run_weekly_refresh.sh [--publish]
set -euo pipefail
cd "$(dirname "$0")/.."                                   # training/
T=$PWD
PUBLISH=${1:-}
CAP=${WEEKLY_CAP:-$(date +%F)}                            # games before CAP: through yesterday
D=${WEEKLY_TAG:-wk$(date -d "$CAP" +%y%m%d)}
B=${WEEKLY_BASE:-def1027}
PYDB=/home/linuxbrew/.linuxbrew/bin/python3              # numpy + psycopg2 + torch
PYREF=/usr/bin/python3                                    # numpy + numba (model, state writer)
# BLAS single-threaded: multi-threaded OpenBLAS corrupted memory on this box (2026-10-28)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=4
SHIM=$(mktemp -d); ln -s $PYREF $SHIM/python; export PATH=$SHIM:$PATH; trap 'rm -rf $SHIM' EXIT
DC=$T/personalization/cache_def_$D
RD=$T/personalization/results/def_$D
export SITE_OUT=$T/personalization/cache/site_$D
TIMES=$T/personalization/logs/weekly_$D.times
mkdir -p $DC/.wk_done $T/personalization/logs
step() {
  local n=$1; shift
  if [ -e $DC/.wk_done/$n ]; then echo "skip $n (done)"; return 0; fi
  echo "=== $n $(date -Is)"
  /usr/bin/time -f "$n MAXRSS_KB=%M SECS=%e" -a -o $TIMES "$@"
  touch $DC/.wk_done/$n
}
echo "weekly refresh $D: cap $CAP, frozen coefficients of $B"
step fetch env P3_PF_CAP=$CAP P3_PF_OUT=$DC $PYDB personalization/p3_x_fetch_post.py
step wp env P3_PF_OUT=$DC $PYDB personalization/p3_x_wp_post.py
step export bash -c "set -e; for w in stamps games scoreboard; do $PYDB personalization/p3_export.py \$w --tag $D --pf-cap $CAP --base-tag $B; done"
step store env P3_CAP_DATE=$CAP $PYREF personalization/p3_full_history.py build --tag $D
step a1 bash personalization/remote/pf_build.sh $D $B
step score env PF_ARMS="today chosen" bash personalization/remote/pf_score.sh $D $B
step changed_keys $PYREF personalization/p3_weekly.py changed-keys --base $B --out $DC/changed_keys.npy
step regress_arm $PYREF personalization/p3_rcf_frozen_check.py $T/personalization/results/def_$B/rcf/p3c/chosen__merged.npz \
  $RD/rcf/p3c/chosen__merged.npz --windows V1,V2,OOT --dup-stamps $DC/mmr_at_game.npz --skip-keys $DC/changed_keys.npy
step regress_states $PYREF personalization/p3_weekly.py regress-states --tag $D --base $B --skip-keys $DC/changed_keys.npy
(
  . personalization/remote/def_env.sh "$T" "$D"
  export P3_BAND=anchored_v1
  step site_build $PYDB personalization/p3_site_publish.py build --store "$D"
  step site_selfcheck $PYREF personalization/p3_rcf_states.py selfcheck --halves 0 --no-filtercheck
  step site_states $PYREF personalization/p3_site_publish.py states --def-tag "$D"
  step site_encode $PYREF personalization/p3_site_publish.py encode
  step site_check $PYREF personalization/p3_site_publish.py check --def-tag "$D"
  step refit_check $PYREF personalization/p3_weekly.py refit-check --tag $D --base $B
)
if [ "$PUBLISH" = "--publish" ]; then
  $PYDB personalization/p3_site_publish.py load --min-recent "${SITE_MIN_RECENT:-3}"
  (cd .. && npx tsx sync/publish-site.ts)
  echo "$D" > $T/personalization/cache/site_published_tag
  # keep the two newest weekly builds (caches ~45 GB each); older ones are rebuilt from scratch if needed
  for old in $(ls -d $T/personalization/cache_def_wk[0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null | sort | head -n -2); do
    t=$(basename $old); t=${t#cache_def_}
    echo "pruning weekly build $t"
    rm -rf "$old" "$T/personalization/cache/export/$t" "$T/personalization/cache/full_history/$t" "$T/personalization/cache/site_$t"
  done
fi
echo "WEEKLY REFRESH DONE $D $(date -Is)"
cat $TIMES
