#!/bin/bash
# Follow-on launcher for the paper-1 revision. Every 5 minutes:
#  * if no tournament pool is running and runnable pairs remain, start one;
#  * benchmark every finished leak-free MCTS run not yet benchmarked (GPU with most free memory);
#  * once the tournament is complete, score it once.
cd /home/max/heroes-of-the-storm/training
R=paper1_revision
while true; do
  if ! pgrep -f "run_tournament.py" > /dev/null; then
    n=$(python3 $R/tournament.py --list-todo | wc -l)
    if [ "$n" -gt 0 ]; then
      python3 - <<'EOF' > /tmp/claude-1000/-home-max-heroes-of-the-storm/a49813e0-7f18-4a49-90f1-0426c355d609/scratchpad/p1r_runnable.txt
import sys; sys.path.insert(0, '.')
from paper1_revision.run_tournament import ready
from paper1_revision.tournament import todo
print(sum(all(ready(x) for x in p.split("__")) for p in todo()))
EOF
      r=$(cat /tmp/claude-1000/-home-max-heroes-of-the-storm/a49813e0-7f18-4a49-90f1-0426c355d609/scratchpad/p1r_runnable.txt)
      if [ "$r" -gt 0 ]; then
        nohup python3 $R/run_tournament.py --procs 12 >> $R/logs/run_tournament_orch.log 2>&1 &
      fi
    elif [ ! -f $R/results/tournament_standings.json ]; then
      nohup python3 $R/score_tournament.py > $R/logs/score_tournament.log 2>&1 &
    fi
  fi
  if ! pgrep -f "bench_mcts.py new:" > /dev/null; then
    runs=""
    for cfg in B F J; do for s in 0 1 2 3 4; do
      log=$R/logs/mcts_${cfg}_oof_s${s}.log
      if [ -f $log ] && grep -q "Complete." $log && [ ! -f $R/results/mcts_bench/new__${cfg}_oof_s${s}__T0.json ]; then
        runs="$runs,new:${cfg}_oof_s${s}"
      fi
    done; done
    if [ -n "$runs" ]; then
      gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
      CUDA_VISIBLE_DEVICES=$gpu nohup python3 $R/bench_mcts.py ${runs#,} --temps 1,0 >> $R/logs/bench_new.log 2>&1 &
    fi
  fi
  sleep 300
done
