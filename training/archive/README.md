# Archived Training Scripts

Superseded scripts moved here for reference. These were part of earlier
experimental iterations and have been replaced by newer approaches.

## Pre-CUDA MCTS (Python-based MCTS)
- `gpu_batch_server.py` — GPU inference server for batched Python MCTS
- `gpu_inference_server.py` — Standalone GPU inference server
- `test_gpu_server.py` — Tests for GPU server
- `mcts_batched.py` — Batched Python MCTS (replaced by CUDA kernel)
- `test_mcts_batched.py` — Tests for batched MCTS
- `train_draft_policy_worker.py` — Python MCTS training worker (replaced by train_mcts_worker.py)

## Superseded Model Architectures
- `train_draft_policy_transformer.py` — Transformer-based draft policy (abandoned)
- `train_gd_variant.py` — GD model variant training (one-off)

## Completed Experiments (results captured, scripts preserved)
- `experiment_wp_size.py` — WP model size sweep
- `experiment_stats_weights.py` — Draft stats weight optimization
- `experiment_composition.py` — Composition scoring experiments
- `ablation_wp.py` — WP model ablation study
- `hero_capabilities.py` — Hero capability feature extraction
- `experiment_value_function_quality.py` — Value function quality analysis
- `experiment_independent_baseline.py` — Independent baseline comparison
- `experiment_synthetic_ablation2.py` — Synthetic data ablation (v2)
- `benchmark_draft.py` — Original draft benchmark
- `benchmark_stats_vs_policy.py` — Stats vs policy comparison benchmark
