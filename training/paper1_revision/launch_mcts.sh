#!/bin/bash
# Launch the leak-free MCTS campaign on GPU 0 (staggered to avoid concurrent snapshot loads)
cd /home/max/heroes-of-the-storm/training
for cfg in B F J; do
  for s in 0 1 2 3 4; do
    if [ ! -f paper1_revision/mcts_runs/${cfg}_oof_s${s}/draft_policy.pt ] || ! grep -q "Complete." paper1_revision/logs/mcts_${cfg}_oof_s${s}.log 2>/dev/null; then
      nohup python3 paper1_revision/train_mcts.py $cfg $s --gpu 0 > /dev/null 2>&1 &
      sleep 45
    fi
  done
done
