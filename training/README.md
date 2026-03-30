# Training & Experiments

## Directory Structure

```
training/
├── shared.py                        # Constants (heroes, maps, tiers), data loading, encoding
├── train_generic_draft.py           # Generic Draft model (behavioral cloning from replays)
├── train_win_probability.py         # Base WP model (197 dims)
├── sweep_enriched_wp.py             # Enriched WP model (283 dims), StatsCache, feature extraction
├── test_wp_sanity.py                # WP model sanity tests (composition safety)
│
├── train_draft_policy.py            # AlphaZeroDraftNet definition, DraftState, Python MCTS
├── train_mcts_worker.py             # CUDA MCTS training worker (300K+ ep/s pipeline)
│
├── cuda_mcts/                       # CUDA kernel for MCTS self-play
│   ├── mcts_kernel.cu               #   Full episode kernel (1 block = 1 episode)
│   ├── device_forward.cuh           #   Policy backbone + heads (device functions)
│   ├── enriched_features.cuh        #   WP model with enriched features on GPU
│   ├── kernel_bindings.cpp          #   pybind11 Python interface
│   ├── extract_weights.py           #   Weight extraction (PyTorch → flat arrays)
│   └── setup.py                     #   Build configuration
│
├── ── Experiment Runners ──
├── run_all_seeds.py                 # 15 seeds × E/G variants (300K episodes)
├── run_overnight_seeds.py           # 10 extra seeds per variant
├── run_1m_seeds.py                  # 1M episode extended training
├── run_large_model.py               # Large policy backbone (30M params)
├── run_policy_head_experiment.py    # Deep/step-conditioned policy heads
├── launch_parallel_mcts.py          # Original 4-GPU parallel launcher
│
├── ── Evaluation & Benchmarks ──
├── experiment_draft_quality.py      # Draft quality metrics (counter/synergy/resilience)
├── experiment_rich_evaluation.py    # Multi-strategy evaluation harness
├── experiment_hybrid_draft.py       # Hybrid MCTS early + greedy late
├── validate_wp_sensitivity.py       # WP model counter/synergy sensitivity
├── validate_search_budget.py        # MCTS sim count sweep (CUDA kernel)
├── benchmark_stats_with_search.py   # Stats mode + single-step search
├── benchmark_stats_minimax.py       # Full minimax with alpha-beta
│
├── ── WP Model Experiments ──
├── experiment_synthetic_augmentation.py  # Synthetic data generation & augmented WP
├── retrain_augmented_wp_v2.py           # Pairwise-adjusted synthetic WR
├── train_partial_wp.py                  # Partial-state WP model (Approach 2a)
│
├── ── CQL/BEAR Experiments ──
├── experiment_cql_draft.py          # Conservative Q-Learning for draft
├── experiment_cql_enriched.py       # CQL with enriched features
├── experiment_bear_draft.py         # BEAR offline RL
├── experiment_stage2_greedy.py      # Greedy WP draft benchmark
│
├── experiment_results/              # JSON results from experiments (committed)
│   ├── draft_quality/               #   Draft quality benchmark results
│   ├── diagnostics/                 #   WP sensitivity, search budget sweeps
│   ├── leaf_eval_fix/               #   Hybrid/rollout approach results
│   ├── stats_search/                #   Stats+search and minimax results
│   ├── cql/                         #   CQL experiment results
│   └── synthetic_augmentation/      #   Synthetic data experiment results
│
├── mcts_runs/                       # MCTS training checkpoints (gitignored, 6GB+)
│   ├── E_seed{0-14}/               #   Enriched WP, 300K episodes
│   ├── G_seed{0-14}/               #   Augmented WP, 300K episodes
│   ├── E1M_seed{0-3}/              #   Enriched WP, 1M episodes
│   ├── A_deep_s{0-14}/             #   Deep policy head, 300K episodes
│   ├── B_step_s{0-14}/             #   Step-conditioned head, 300K episodes
│   ├── C_dstep_s{0-14}/            #   Deep+step head, 300K episodes
│   ├── rollout_s{0-2}/             #   Rollout leaf evaluation
│   └── bench/                       #   Benchmark checkpoints
│
└── archive/                         # Superseded scripts (see archive/README.md)
```

## Key Models

| Model | Architecture | Input | Training | File |
|-------|-------------|-------|----------|------|
| Generic Draft (GD) | 289→256→128→90 MLP | Draft state | BCE on replay picks | `generic_draft_{0-4}.pt` |
| Win Probability (base) | 197→1024→512→512→128→1 | Multi-hot teams+map+tier | BCE on game outcomes | `win_probability.pt` |
| Win Probability (enriched) | 283→256→128→1 | Base + 86 enriched features | BCE on game outcomes | `wp_experiment_enriched.pt` |
| Win Probability (augmented) | 283→512→256→128→1 | Base + enriched + synthetic | BCE on game outcomes | `wp_enriched_winner.pt` |
| Draft Policy (MCTS) | 290→768→3×ResBlock→256→90 | Full draft state | AlphaZero self-play | `mcts_runs/*/draft_policy.pt` |

## Running Experiments

```bash
# Prerequisites
set -a && source .env && set +a

# Build CUDA kernel
cd training/cuda_mcts && CXX=g++ CC=gcc python3 setup.py build_ext --inplace

# Train MCTS policy (single GPU)
CUDA_VISIBLE_DEVICES=0 MCTS_WP_MODEL=enriched MCTS_NUM_EPISODES=300000 \
  python3 -u train_mcts_worker.py

# Multi-seed runs (4 GPUs)
python3 -u run_all_seeds.py

# Evaluate draft quality
python3 -u experiment_draft_quality.py --drafts 200

# WP sensitivity diagnostic
python3 -u validate_wp_sensitivity.py --replays 500
```
