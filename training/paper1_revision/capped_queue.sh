#!/bin/bash
# Resource-capped follow-on queue (owner cap 2026-09-30): at most 2 concurrent GPU
# processes for this lane, new ones on GPU 3 only, nice 19, cores 48-63, <=6 CPU threads.
cd /home/max/heroes-of-the-storm/training
R=paper1_revision
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 P1R_NPROC=4
L="nice -n 19 taskset -c 48-63"
ngpu() {  # my GPU processes: training workers under paper1_revision + benchmarks
  local n=0
  for p in $(pgrep -f "paper1_revision/train_mcts.py --child"); do n=$((n+1)); done
  for p in $(pgrep -f "paper1_revision/bench_mcts.py"); do n=$((n+1)); done
  echo $n
}
done_run() { grep -q "Complete\." $R/logs/mcts_$1.log 2>/dev/null; }
benched() { [ -f $R/results/mcts_bench/new__$1__T0.json ]; }
TASKS="bench:J_oof_s2 bench:J_oof_s3 resume:J:4 bench:J_oof_s4 resume:J:0 resume:J:1 bench:J_oof_s0 bench:J_oof_s1"
for t in $TASKS; do
  kind=${t%%:*}; rest=${t#*:}
  if [ $kind = bench ]; then
    until done_run $rest; do sleep 120; done
    benched $rest && continue
  fi
  until [ $(ngpu) -lt 2 ]; do sleep 120; done
  if [ $kind = bench ]; then
    CUDA_VISIBLE_DEVICES=3 $L python3 $R/bench_mcts.py new:$rest --temps 1,0 >> $R/logs/bench_new.log 2>&1 &
  else
    cfg=${rest%%:*}; s=${rest#*:}
    $L python3 $R/train_mcts.py $cfg $s --gpu 3 --resume > /dev/null 2>&1 &
  fi
  echo "$(date) launched $t" >> $R/logs/capped_queue.log
  sleep 180
done
wait
