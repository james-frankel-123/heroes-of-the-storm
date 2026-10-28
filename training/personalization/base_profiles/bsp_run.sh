#!/bin/bash
# BSP job runner on the 5090: nice 19, cores 12-15, 4 threads, memory watchdog.
# usage: bsp_run.sh <name> <cmd...>   (log ~/hots/base_profiles/logs/<name>.log)
# The watchdog stops the job (its own PID and that PID's children, by PID) when
# the job's RSS passes 7.6 GB or the host's available memory drops below 16 GB.
name=$1; shift
source ~/hots/env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export BSP=$HOME/hots/base_profiles P1_TIERS=site PYTHONPATH=$HOME/hots/repo/training CUDA_VISIBLE_DEVICES=
LOG=$BSP/logs/$name.log
cd $BSP/code
nice -n 19 taskset -c 12-15 "$@" > $LOG 2>&1 &
pid=$!
echo $pid > $BSP/logs/$name.pid
while kill -0 $pid 2>/dev/null; do
  rss=$(ps -o rss= --ppid $pid -p $pid | awk '{s+=$1} END{print s+0}')
  avail=$(free -m | awk '/Mem:/{print $7}')
  if [ "$rss" -gt 7600000 ] || [ "$avail" -lt 16000 ]; then
    echo "WATCHDOG stop: rss_kb=$rss avail_mb=$avail" >> $LOG
    for c in $(ps -o pid= --ppid $pid); do kill -TERM $c; done
    kill -TERM $pid
  fi
  sleep 5
done
wait $pid
echo "EXIT $?" >> $LOG
