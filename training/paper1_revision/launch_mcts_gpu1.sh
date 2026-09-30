#!/bin/bash
# Remaining leak-free MCTS runs on GPU 1 (policy relaxed 2026-09-29: any GPU with free memory)
cd /home/max/heroes-of-the-storm/training
for run in "F 1" "F 2" "F 3" "F 4" "J 0" "J 1" "J 2" "J 3" "J 4"; do
  set -- $run
  if [ ! -f paper1_revision/logs/mcts_${1}_oof_s${2}.log ]; then
    nohup python3 paper1_revision/train_mcts.py $1 $2 --gpu 1 > /dev/null 2>&1 &
    sleep 60
  fi
done
