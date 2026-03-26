# MCTS Policy Training Run History

## Architecture (all runs)

- **Network**: AlphaZero-style residual CNN (4.4M parameters)
  - Input: 290 dims (3x90 multi-hot + 14 map + 3 tier + 2 step + 1 our_team)
  - Backbone: 289 dims (no our_team) -> 768 -> 3 residual blocks -> 512 -> 256
  - Policy head: 256 -> 90 (perspective-invariant, same board = same recommendations)
  - Value head: 257 (backbone + our_team) -> 128 -> 64 -> 1 (perspective-aware)
- **Search**: 200 MCTS simulations per move, c_puct=2.0
- **Opponents**: 5 Generic Draft models (behavioral cloning, 275K replays, different seeds)
- **Training**: Self-play with MCTS, replay buffer (150K transitions), cosine LR schedule
- **Value symmetrization**: Both WP evaluation and value head run from both perspectives, averaged to enforce P(t0,t1) + P(t1,t0) = 1.0
- **Evaluation**: 50 games vs GD opponent pool every 5000 episodes

## Prior Runs (invalidated)

All runs before March 26 used non-symmetrized WP evaluation, which had significant team-order bias (up to 24% asymmetry on mirror matchups). These results are **not comparable** to the current runs and should not be cited in the paper.

Historical reference only:

| Run | WP Model | Episodes | Final WR | Final WP | Notes |
|-----|----------|----------|----------|----------|-------|
| A (old) | Base (197d) | 79K | 78.0% | 0.546 | Non-symmetrized, CPU, 1.6 ep/s |
| C (old) | Enriched (283d) | 167K | 89.0% | 0.598 | Non-symmetrized |
| E (old) | Enriched (300K) | 300K | 89.5% | 0.602 | Non-symmetrized |
| F (old) | + comp_wr | 300K | 89.5% | 0.598 | Non-symmetrized |
| G (old) | Augmented | 239K | 92.5% | 0.626 | Crashed at quantization, non-symmetrized |

## Current Runs (symmetrized, in progress)

All current runs use symmetrized WP evaluation AND symmetrized value head. Trained on the full CUDA kernel engine at ~48 ep/s per GPU.

| Run | GPU | WP Model | Episodes | Status | Notes |
|-----|-----|----------|----------|--------|-------|
| run_A_base | 1 | Base (197d, 256->128) | 300K | Running | Fair baseline: what does base WP achieve with full training? |
| run_E_enriched | 2 | Enriched (283d, 256->128, no augmentation) | 300K | Running | Enriched features without synthetic data |
| run_G_augmented | 0 | Augmented (283d, 512->256->128, synthetic) | 300K | Running | Best WP model from augmentation experiments |
| run_G2_augmented_fresh | 3 | Augmented (same as G, different seed) | 300K | Running | Second seed for confidence |

**ETA**: All 4 complete in ~100 minutes from start (~2h wall time including pretraining).

## Training Infrastructure Evolution

| Version | Date | Throughput | Bottleneck |
|---------|------|-----------|------------|
| Python sequential MCTS | Mar 18-24 | 2-5 ep/s (all workers) | PyTorch CPU forward passes |
| Persistent pool + shared memory | Mar 25 | 5.5 ep/s (64 workers) | Pool creation, weight pickling |
| Python batched MCTS (virtual loss) | Mar 25 | 0.67x (slower) | Numpy buffer overhead > GPU benefit |
| C++ MCTS + fused CUDA kernels (host-launched) | Mar 25 | 5.1 ep/s (single thread) | cudaStreamSynchronize per forward pass (2.4ms) |
| C++ MCTS + batched CUDA (K=32, host-launched) | Mar 25 | 10.9x vs Python | Amortized sync, 85μs/sample |
| Full CUDA kernel (one block = one episode) | Mar 26 | 26 ep/s per GPU | Zero launch overhead, all in-kernel |
| + ring buffer + pipelining + GPU training | Mar 26 | **48 ep/s per GPU** | Approaching hardware ceiling |

**Total speedup**: 48 ep/s per GPU vs ~1.2 ep/s per run (old Python 4×16 config) = **40x per GPU, 160x total** across 4 GPUs.

**Key insight**: The bottleneck was never the tree traversal or the neural network compute. It was the **overhead between forward passes**: Python dispatch, cudaStreamSynchronize, numpy allocation, pickle serialization. Putting the entire MCTS loop inside a single CUDA kernel eliminated all of it.

## For the Paper

The MCTS results will be updated once the current runs complete. Key claims that will be verifiable:

1. **Value function quality drives MCTS policy quality** — comparing Run A (base WP) vs Run E (enriched) vs Run G (augmented) at 300K episodes each, all with symmetrized evaluation.

2. **Augmented WP produces higher average WP in MCTS** — if Run G achieves higher avg WP than Run E at the same episode count, the synthetic augmentation benefit extends beyond greedy search.

3. **MCTS with augmented WP achieves both safety and context-awareness** — pending rich evaluation metrics (counter, synergy, diversity) on the final policies.

4. **Reproducibility** — Run G vs Run G2 (same config, different seed) provides confidence intervals.
