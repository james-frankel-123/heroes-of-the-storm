#!/bin/bash
# Deferred revision jobs (owner request 2026-09-30), run after capped_queue.sh
# (the 800-sim J_oof seeds) finishes. One job at a time, so this lane never has
# more than 2 GPU processes; GPU jobs on GPU 3; nice 19, cores 48-63, <=6 CPU
# threads (OMP 1, pools of 4, CQL DataLoader 4 workers). Each job writes
# results/deferred/<name>.json and is skipped if that file exists.
# Order = value to the revision.
cd /home/max/heroes-of-the-storm/training
R=paper1_revision
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 P1R_NPROC=4
L="nice -n 19 taskset -c 48-63"
LOG=$R/logs/capped_queue2.log
while pgrep -f "paper1_revision/capped_queue.sh" > /dev/null; do sleep 300; done
ngpu() { echo $(( $(pgrep -f "paper1_revision/train_mcts.py --child" | wc -l) + $(pgrep -f "paper1_revision/bench_mcts.py" | wc -l) )); }
run() {  # name  output-json  gpu(0/1)
  local job=$1 out=$R/results/deferred/$2.json gpu=$3
  if [ -f $out ]; then echo "$(date) skip $job (done)" >> $LOG; return; fi
  if [ $gpu = 1 ]; then until [ $(ngpu) -lt 2 ]; do sleep 120; done; export CUDA_VISIBLE_DEVICES=3; else export CUDA_VISIBLE_DEVICES=""; fi
  echo "$(date) start $job" >> $LOG
  $L python3 -u $R/deferred.py $job > $R/logs/deferred_$job.log 2>&1
  echo "$(date) end $job rc=$?" >> $LOG
}
run scope            scope_sweep             1
run ngs_quartiles    ngs_quartiles           0
run rebench_ag       rebench_ag              1
run concentration    concentration           0
run ensemble         ensemble_uncertainty    1
run cql_build        cql_enriched_build      0
run cql_train_a2.0   cql_enriched_train_a2.0 1
run cql_train_a0.5   cql_enriched_train_a0.5 1
run cql_eval         cql_enriched_eval       0
echo "$(date) queue finished" >> $LOG
