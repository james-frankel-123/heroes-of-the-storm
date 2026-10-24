#!/bin/bash
# P3 tournament, definitive run end to end (review C2/R02, 2026-10-24): both model passes and the
# handoff between them, resumable. Run from the repo root on the main box (light orchestration):
#   nohup nice -n 19 taskset -c 48-51 training/personalization/remote/p3t_definitive.sh <D> \
#     > training/personalization/logs/p3t_definitive_<D>.log 2>&1 &
# Environment: P3T_RCF_ROWS (the build's merged-rows file, absolute worker path), P3T_MANIFEST
# (relative to training/), optional P3T_SMOKE=1 (small sizes; the dry run), P3T_ANSWER_CMD (default:
# the personalization lane's `bash personalization/remote/p3_def.sh <D> answer <Q> <A>`).
# Stages (a stage that finished is skipped on a rerun: marker logs/p3t_def_<D>/<stage>.done):
#   pass1    record pass, old full_ms states (sensitivity): p3t_pipeline.sh, namespace <D>_oldmodel;
#            every walk's queries land in cache_def_<D>_oldmodel/p3t_walk_queries (CPU host)
#   answer   the personalization lane answers them (the chosen model) into .../p3t_walk_answers
#   check    every answer file against its query file (p3t_check_all.sh, CPU host)
#   pass2    replay pass, the chosen model's states (headline): namespace <D>_headline
#   done     both passes' results in results/p3t_<D>_oldmodel/ and results/p3t_<D>_headline/
# The passes never share a directory (namespaces), so nothing is moved between them and no draft,
# score, walk file or vintage table of one pass can be picked up by the other.
set -euo pipefail
D=${1:?build tag}
cd "$(dirname "$0")/../../.."
source training/remote_workers/_remote.sh
: "${P3T_RCF_ROWS:?set P3T_RCF_ROWS to the build's merged-rows file}"
: "${P3T_MANIFEST:?set P3T_MANIFEST to the build's manifest (relative to training/)}"
CPU=${P3T_CPU_HOST:-3080-gaming-desktop}
B=/home/max/hots/repo/training/personalization
OLD=$B/cache_def_${D}_oldmodel
Q=$OLD/p3t_walk_queries
A=$OLD/p3t_walk_answers
MK=training/personalization/logs/p3t_def_$D
mkdir -p "$MK"
JOB=training/personalization/remote/p3_job.sh
PIPE=training/personalization/remote/p3t_pipeline.sh
STAMP=$(date +%m%d%H%M)

stage() {   # name command...: run once, mark done
  local n=$1; shift
  if [ -e "$MK/$n.done" ]; then echo "$(date -Is) stage $n: done earlier, skipped"; return 0; fi
  echo "$(date -Is) stage $n: start"
  "$@"
  date -Is > "$MK/$n.done"
  echo "$(date -Is) stage $n: done"
}
wait_job() {   # the pipeline's tolerant poll (unknown status is retried up to P3T_POLL_UNKNOWN_MAX s)
  local h=$1 n=$2 st since="" umax=${P3T_POLL_UNKNOWN_MAX:-3600}
  while true; do
    st=$(hotsjob "$h" state "$n" 2>/dev/null | tail -1) || st=""
    case "$st" in
      completed) return 0 ;;
      failed|died|paused) echo "$n: $st"; return 1 ;;
      running|queued|starting) since="" ;;
      *) [ -n "$since" ] || since=$(date +%s)
         [ $(( $(date +%s) - since )) -le "$umax" ] || { echo "$n: status unknown too long"; return 1; } ;;
    esac
    sleep 60
  done
}

pass1() { P3T_BUILD=$D P3T_PASS=oldmodel P3T_STATES=record bash $PIPE; }
answer() {
  # the queries must exist and be non-empty before the lane is asked
  local nq
  nq=$(ssh "$CPU" "$(remote_shell "$CPU")" <<<"ls $Q/*.npz 2>/dev/null | wc -l")
  [ "${nq:-0}" -gt 0 ] || { echo "no walk queries in $Q"; return 1; }
  echo "answering $nq query files"
  if [ -n "${P3T_ANSWER_CMD:-}" ]; then
    (cd training && eval "$P3T_ANSWER_CMD")
  else
    (cd training && RUNSUF=_$STAMP bash personalization/remote/p3_def.sh "$D" answer "$Q" "$A")
  fi
}
check() {
  local n=p3t_def_check_$STAMP
  P3_CORES=16-19 P3_MEM=20G P3_THREADS=4 $JOB $CPU $n env P3_CACHE=$OLD P3_RESULTS=$B/results/def_${D}_oldmodel \
    P3_RCF_ROWS=$P3T_RCF_ROWS P3_RCF_KERNELS=$OLD/hs_kernels.npz bash personalization/remote/p3t_check_all.sh $Q $A
  wait_job $CPU $n
}
pass2() { P3T_BUILD=$D P3T_PASS=headline P3T_STATES=replay P3T_WALK_ANSWERS=$A bash $PIPE; }

stage pass1 pass1
stage answer answer
stage check check
stage pass2 pass2
echo "P3T DEFINITIVE DONE $D $(date -Is): results/p3t_${D}_oldmodel (sensitivity), results/p3t_${D}_headline (headline)"
