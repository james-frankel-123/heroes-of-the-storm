#!/bin/bash
# P3 definitive rerun, orchestrated from the main box (never the 3090).
#   bash personalization/remote/p3_def.sh <D> <stage>
# Stages, in order (each one is idempotent and adds its outputs to the
# pinned manifest results/def_<D>/manifest.json):
#   export    main box: frozen DB exports stamps, games, scoreboard (read-only research role)
#   store     main box: merged as-of history store (HP shards and panels hold battletags and never leave this box)
#   upstream  main box: pin the window-cache inputs the chain links (sha256 checked on the 3080)
#   ship_a1   main box -> 3080: code, exports, history.npz, upstream inputs
#   a1        3080 job: def_build_a1.sh (A1 cache, MMR at game, merged rows, card, caches, fit inputs)
#   fit       3080 -> main -> 5090; 5090 job: def_fit.sh; params back to the 3080
#   walks     3080 job: def_walks.sh (baselines, chosen model on both halves, calibration, band)
#   collect   pull results and hash every definitive file into the manifest
set -euo pipefail
D=${1:?build tag}; STAGE=${2:?stage}
TR=$(cd "$(dirname "$0")/../.." && pwd)
cd "$TR"
W3=3080-gaming-desktop; W5=windows-5090-wsl
R3=/home/max/hots/repo/training; R5=hots/repo/training
MAN=personalization/results/def_$D/manifest.json
mkdir -p personalization/results/def_$D
MANPY="/usr/bin/python3 personalization/p3_manifest.py"
LOCAL="nice -n 19 taskset -c 48-51"
UPSTREAM="wp_drift.npz players_2024q2.npz x_wp_post.npz x_post_players.npz x_post_games.json.gz x_tiers.npz
  pickorder_2024q2.npz x_bans.npz drafts_post.json.gz x_side_games.npz x_side_players.npz"
source remote_workers/_remote.sh
wait_job() {  # host name: block until the hotsjob ends, fail if it did not complete
  until [ "$(hotsjob $1 state $2 2>/dev/null | tail -1)" != "running" ]; do sleep 60; done
  st=$(hotsjob $1 state $2 | tail -1); echo "job $2 on $1: $st"; [ "$st" = "completed" ]
}
case $STAGE in
export)
  set -a; . ../.env; set +a
  for w in stamps games scoreboard; do $LOCAL python3 personalization/p3_export.py $w --tag $D; done
  $MANPY add $MAN personalization/cache/export/$D/{stamps,games,scoreboard}.csv.gz \
    --role personalization --stage export --host main --format csv.gz ;;
store)
  $LOCAL /usr/bin/python3 personalization/p3_full_history.py build --tag $D
  $LOCAL /usr/bin/python3 personalization/p3_full_history.py stats --tag $D
  $MANPY add $MAN personalization/cache/full_history/$D/history.npz personalization/cache/full_history/$D/meta.json \
    --role personalization --stage store --host main ;;
upstream)
  $MANPY add $MAN $(for f in $UPSTREAM; do echo personalization/cache/$f; done) --role upstream --stage window-cache \
    --host main --note "window-cache input linked by p3_c_history (not rebuilt)" ;;
ship_a1)
  rsync -t --rsync-path="wsl rsync" personalization/*.py $W3:$R3/personalization/
  rsync -t --rsync-path="wsl rsync" personalization/tests/*.py $W3:$R3/personalization/tests/
  rsync -t --rsync-path="wsl rsync" personalization/remote/*.sh $W3:$R3/personalization/remote/
  rsync -t --rsync-path="wsl rsync" shared.py $W3:$R3/
  ssh $W3 "wsl -e bash -s" <<< "mkdir -p $R3/personalization/cache/export/$D $R3/personalization/cache/full_history/$D $R3/personalization/results/def_$D"
  rsync -t --rsync-path="wsl rsync" personalization/cache/export/$D/*.csv.gz $W3:$R3/personalization/cache/export/$D/
  rsync -t --rsync-path="wsl rsync" personalization/cache/full_history/$D/history.npz personalization/cache/full_history/$D/meta.json \
    $W3:$R3/personalization/cache/full_history/$D/
  for f in $UPSTREAM; do rsync -tL --rsync-path="wsl rsync" personalization/cache/$f $W3:$R3/personalization/cache/; done
  rsync -t --rsync-path="wsl rsync" $MAN $W3:$R3/$MAN
  ssh $W3 "wsl -e bash -s" <<< "source ~/hots/env.sh; cd $R3 && python personalization/p3_manifest.py verify $MAN" ;;
a1)
  RUN_NAME=def_${D}_a1${RUNSUF:-} HOTSJOB_FLAGS="--mem-max 20G" remote_workers/run_remote.sh $W3 \
    "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 nice -n 19 taskset -c 0-3 bash personalization/remote/def_build_a1.sh $D"
  wait_job $W3 def_${D}_a1${RUNSUF:-} ;;
fit)
  T=/tmp/p3def_$D; mkdir -p $T
  rsync -t --rsync-path="wsl rsync" $W3:$R3/personalization/cache_def_$D/rcf/fit_merged.npz \
    $W3:$R3/personalization/cache_def_$D/rcf/fitm_merged_a_shrink.npz $T/
  ssh $W5 "mkdir -p ~/$R5/personalization/cache_def_$D/rcf ~/$R5/personalization/results/def_$D/rcf/params ~/$R5/personalization/remote"
  rsync -t personalization/*.py $W5:$R5/personalization/
  rsync -t personalization/remote/*.sh $W5:$R5/personalization/remote/
  rsync -t $T/fit_merged.npz $T/fitm_merged_a_shrink.npz $W5:$R5/personalization/cache_def_$D/rcf/
  RUN_NAME=def_${D}_fit${RUNSUF:-} HOTSJOB_FLAGS="--mem-max 11500M" remote_workers/run_remote.sh $W5 \
    "OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 nice -n 19 bash personalization/remote/def_fit.sh $D"
  wait_job $W5 def_${D}_fit${RUNSUF:-}
  rsync -t "$W5:$R5/personalization/results/def_$D/rcf/params/full_ms*" $T/
  ssh $W3 "wsl -e bash -s" <<< "mkdir -p $R3/personalization/results/def_$D/rcf/params"
  rsync -t --rsync-path="wsl rsync" $T/full_ms* $W3:$R3/personalization/results/def_$D/rcf/params/
  rm -f $T/fit_merged.npz $T/fitm_merged_a_shrink.npz ;;
walks)
  rsync -t --rsync-path="wsl rsync" personalization/*.py $W3:$R3/personalization/
  rsync -t --rsync-path="wsl rsync" personalization/remote/*.sh $W3:$R3/personalization/remote/
  RUN_NAME=def_${D}_walks${RUNSUF:-} HOTSJOB_FLAGS="--mem-max 20G" remote_workers/run_remote.sh $W3 \
    "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 nice -n 19 taskset -c 0-3 bash personalization/remote/def_walks.sh $D"
  wait_job $W3 def_${D}_walks${RUNSUF:-} ;;
collect)
  # hash on the 3080, where the files live, then pull the manifest back
  rsync -t --rsync-path="wsl rsync" $MAN $W3:$R3/$MAN
  ssh $W3 "wsl -e bash -s" <<EOF
source ~/hots/env.sh; cd $R3
. personalization/remote/def_env.sh \$PWD $D
M="python personalization/p3_manifest.py add $MAN --role personalization --host 3080-gaming-desktop"
\$M personalization/cache/export/$D/stamps.npz --stage a1-parse
\$M \$DEF_CACHE/{players_2024q2,hero_mmr_2024q2,gametime_2024q2,mmr_stamps_db,hs_slots,hs_kernels,x_slots_ext,mmr_at_game}.npz --stage a1-cache
\$M \$P3_RCF_ROWS --stage merged-rows
\$M \$DEF_CACHE/rcf/sb_a_shrink.npz \$DEF_CACHE/rcf/sb_share6.npz --stage card
\$M ../src/lib/data/hero-roles.ts --stage role-map
\$M \$DEF_CACHE/rcf/fit_merged.npz \$DEF_CACHE/rcf/fitm_merged_a_shrink.npz \$DEF_CACHE/rcf/aug_merged.npz \
   \$DEF_CACHE/rcf/aug_merged_mmr.npz \$DEF_CACHE/rcf/zaug_a_shrink_*.npz \$DEF_CACHE/rcf/zaug_share6_*.npz \$DEF_CACHE/rcf/mmr_coef.npy --stage model-cache
\$M \$DEF_RESULTS/rcf/params/full_ms.npz \$DEF_RESULTS/rcf/params/full_ms.json \$DEF_RESULTS/rcf/params/full_ms_flip.npz \
   \$DEF_RESULTS/rcf/params/full_ms_flip.json --stage fit --host windows-5090-wsl
\$M \$DEF_CACHE/rcf/parts_full_ms.npz \$DEF_CACHE/rcf/parts_chosen.npz \$DEF_CACHE/rcf/parts_chosen_flip.npz --stage walk-parts
\$M \$DEF_RESULTS/rcf/p3c/{today,full_ms,full_ms_layer,chosen,chosen_flip}__merged.npz --stage arms
\$M \$DEF_RESULTS/rcf/calib_def.json \$DEF_RESULTS/rcf/band_def_chosen.json \$DEF_RESULTS/p3_c_tests_def.json --stage reports
\$M \$DEF_CACHE/rcf/parts_chosen_layer.npz \$DEF_CACHE/rcf/parts_chosen_flip_layer.npz --stage layer-inputs
python personalization/p3_manifest.py verify $MAN
EOF
  rsync -t --rsync-path="wsl rsync" $W3:$R3/$MAN $MAN
  rsync -t --rsync-path="wsl rsync" "$W3:$R3/personalization/results/def_$D/rcf/*.json" personalization/results/def_$D/
  rsync -t --rsync-path="wsl rsync" "$W3:$R3/personalization/results/fig_shrink_def.png" personalization/results/def_$D/ || true
  $MANPY show $MAN ;;
pin_tournament)
  # carry the tournament's pinned fixed inputs over from the dry-run manifest (same files, same hashes)
  /usr/bin/python3 - "$MAN" personalization/results/def_${PIN_FROM:-dry1022}/manifest.json <<'PY'
import json, sys
dst, src = sys.argv[1], sys.argv[2]
M = json.load(open(dst)); S = json.load(open(src))
have = {e["path"] for e in M["files"]}
add = [e for e in S["files"] if e["role"] == "tournament" and e["path"] not in have]
M["files"] += add
json.dump(M, open(dst, "w"), indent=1)
print(f"added {len(add)} tournament entries from {src}")
PY
  for h in $W3 $W5; do
    if [ $h = $W5 ]; then rsync -t $MAN $h:$R5/$MAN && ssh $h "source ~/hots/env.sh; cd ~/$R5 && python personalization/p3_manifest.py verify $MAN --role tournament"
    else rsync -t --rsync-path="wsl rsync" $MAN $h:$R3/$MAN && ssh $h "wsl -e bash -s" <<< "source ~/hots/env.sh; cd $R3 && python personalization/p3_manifest.py verify $MAN --role tournament"; fi
  done ;;
answer)
  # answer a tournament walk-query directory on the 3080: p3_def.sh <D> answer <query dir> <answer dir>
  Q=${3:?query dir on the 3080}; A=${4:?answer dir on the 3080}
  rsync -t --rsync-path="wsl rsync" personalization/p3_rcf_states.py $W3:$R3/personalization/
  RUN_NAME=def_${D}_answer${RUNSUF:-} HOTSJOB_FLAGS="--mem-max 20G" remote_workers/run_remote.sh $W3 \
    "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 nice -n 19 taskset -c 0-3 bash -c '. personalization/remote/def_env.sh \$PWD $D && python -u personalization/p3_rcf_states.py selfcheck && python -u personalization/p3_rcf_states.py answer $Q $A'"
  wait_job $W3 def_${D}_answer${RUNSUF:-} ;;
report)
  rsync -t --rsync-path="wsl rsync" $W3:$R3/personalization/results/def_$D/p3_c_tests_def.json personalization/results/def_$D/
  /usr/bin/python3 personalization/p3_def_report.py $D ;;
*) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "STAGE $STAGE DONE"
