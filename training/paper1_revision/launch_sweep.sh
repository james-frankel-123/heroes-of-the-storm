#!/bin/bash
# Fractional-factorial feature sweep (128 configs, OOF features), 2 shards on GPU 0,
# started after the main WP queue finishes.
cd /home/max/heroes-of-the-storm/training
while pgrep -f "train_wp.py naive,herostrength" > /dev/null; do sleep 30; done
nohup python3 paper1_revision/train_wp.py $(cat paper1_revision/cache/sweep_shard0.txt) > paper1_revision/logs/sweep0.log 2>&1 &
nohup python3 paper1_revision/train_wp.py $(cat paper1_revision/cache/sweep_shard1.txt) > paper1_revision/logs/sweep1.log 2>&1 &
wait
