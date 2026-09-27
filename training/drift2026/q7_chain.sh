#!/bin/bash
# Q7 decayed-aggregates 2x2: 3 new cells x 3 seeds on ONE GPU (W7 owns the
# rest; caller must ensure the W7 gate is open before launching).
# cumulative cell = existing d2c_cumprev (not retrained).
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=2

for feat in decayed90_prev decayed365_prev decayed90k100_prev; do
  cell="q7_${feat%_prev}"
  for seed in 42 123 777; do
    name="${cell}_s${seed}"
    if [ -f "drift2026/results/q7/${name}.json" ]; then
      echo "exists, skipping: ${name}"
      continue
    fi
    python3 -u drift2026/train_drift_wp.py \
      --name "$name" --features "$feat" --regime all --seed "$seed" \
      --results-subdir q7
  done
done
echo "Q7 chain done"
