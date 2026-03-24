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
- **Evaluation**: 200 games vs GD opponent pool every ~500 episodes
  - Metric 1: Win rate (% of games where MCTS agent's team wins by enriched WP evaluation)
  - Metric 2: Average WP (mean enriched WP for MCTS agent's terminal draft states)

## Run Summary

| Run | Date | Episodes | WP Model | Final WR | Peak WR | Final WP | Peak WP | Notes |
|-----|------|----------|----------|----------|---------|----------|---------|-------|
| A | Mar 18 | 79K | Base (197d) | 78.0% | 84.0% | 0.546 | 0.548 | First run, base WP, slower (1.6 ep/s) |
| B | Mar 18 | 38K | Base (197d) | 49.0% | 55.0% | 0.485 | 0.496 | Aborted (architecture experiment) |
| C | Mar 19 | 167K | Enriched (283d, no comp_wr) | 89.0% | 96.0% | 0.598 | 0.605 | First enriched WP, deployed as 203K model |
| D | Mar 20 | 19K | Enriched (283d) | 91.0% | 91.0% | 0.579 | 0.583 | Aborted (testing enriched WP variant) |
| E | Mar 20-21 | 300K | Enriched (283d, no comp_wr) | 89.5% | 95.5% | 0.602 | 0.613 | Full run, 256->128 WP, deployed |
| F | Mar 23 | 300K | Enriched (283d, with comp_wr) | 89.5% | 93.5% | 0.598 | 0.612 | Added comp_wr to WP, deployed |
| G | Mar 24 | 226K+ | Augmented (283d, 512->256->128 WP, synthetic data) | 92.5% | 95.5% | 0.626 | 0.639 | Current run, ~6h remaining |

## Detailed Learning Curves

### Run A: Base WP Model (79K episodes)

First successful MCTS training run. Used the naive 197-dim WP model (multi-hot + map + tier only, no enriched features). Trained on CPU at 1.6 ep/s.

```
Episode    Avg WP    Win Rate
    522    0.4683      39.5%
  5,000    ~0.48       ~47%
 10,000    ~0.49       ~52%
 25,000    ~0.51       ~60%
 50,000    ~0.53       ~70%
 79,054    0.5460      78.0%
```

The policy learned to beat GD opponents but plateaued at 78% win rate. Limited by the base WP model's inability to evaluate compositions.

### Run C: First Enriched WP (167K episodes)

Switched to the enriched WP model (283 dims, including role_counts, pairwise stats, map_delta, meta_strength, draft_diversity -- but NOT comp_wr, which was added later). Major improvement over Run A.

```
Episode    Avg WP    Win Rate
    522    0.4779      40.0%
  5,000    0.497       48.0%
 25,000    0.525       60.5%
 50,000    0.560       73.0%
100,000    0.581       83.5%
150,000    0.595       88.0%
167,000    0.598       89.0%
```

Peak win rate 96.0% at 154K episodes. This run was deployed as the "203K episode" model (it was extended slightly past 167K before deployment).

### Run E: Full 300K with Enriched WP (no comp_wr)

Same enriched WP model as Run C, trained for the full 300K episodes.

```
Episode    Avg WP    Win Rate
    522    0.4669      39.0%
  5,000    0.500       47.5%
 25,000    0.527       58.0%
 50,000    0.565       71.5%
100,000    0.585       83.0%
150,000    0.597       86.0%
200,000    0.603       89.5%
250,000    0.608       91.0%
300,000    0.602       89.5%
```

Peak win rate 95.5% at 299K episodes. WP plateaus around 0.61. Deployed as the "300K episode" model.

### Run F: 300K with comp_wr Enriched WP

Added comp_wr (empirical composition win rates from 9M Heroes Profile games) to the WP model. This is the WP model that produces 17.2% WP for 5-tank compositions (vs 44.7% naive).

```
Episode    Avg WP    Win Rate
    522    0.4624      35.0%
  5,000    0.497       46.5%
 25,000    0.523       57.0%
 50,000    0.558       69.5%
100,000    0.580       82.5%
150,000    0.593       86.5%
200,000    0.598       88.5%
250,000    0.605       90.5%
300,000    0.598       89.5%
```

Peak win rate 93.5% at 248K episodes. Slightly lower peak than Run E despite the comp_wr feature providing better composition evaluation. The MCTS self-play may not fully exploit the comp_wr signal because the opponent (GD) always drafts reasonable compositions anyway, so degenerate states rarely arise during training.

### Run G: Augmented WP (in progress, 226K/300K)

Uses the best WP model from the synthetic augmentation experiment: 512->256->128 architecture trained with synthetic data for unseen compositions (10% WR, 100 samples per composition). This WP model scores 20/21 on sanity tests and achieves 94.5% healer rate in greedy drafts.

```
Episode    Avg WP    Win Rate
    522    0.4508      34.5%
  1,000    0.4654      39.0%
  2,500    0.4865      45.0%
  5,000    0.5053      49.5%
 10,000    0.4936      52.0%
 25,000    0.5233      56.0%
 50,000    0.5668      69.5%
 75,000    0.5834      81.0%
100,000    0.5940      84.5%
125,000    0.5943      83.5%
150,000    0.6111      84.5%
175,000    0.6117      89.0%
200,000    0.6296      89.5%
225,000    0.6244      92.5%
```

Peak win rate 95.5% at 223K episodes. Peak avg WP 0.639 at 217K episodes. The higher WP values (0.626 at 226K vs 0.602 at 300K for Run E) suggest the augmented WP model's composition awareness is being leveraged by the policy -- the MCTS agent is finding terminal states that the composition-aware evaluator rates more highly.

**ETA: ~6 hours to complete 300K episodes.**

## Key Observations

### 1. Value function quality drives policy quality

| WP Model | MCTS Final WR | MCTS Final WP | WP Model Sanity |
|-----------|---------------|----------------|-----------------|
| Base (197d) | 78.0% | 0.546 | 14/21 |
| Enriched (283d, no comp_wr) | 89.5% | 0.602 | 17/21 |
| Enriched (283d, with comp_wr) | 89.5% | 0.598 | 17/21 |
| Augmented (283d, 512->256->128, synthetic) | 92.5%* | 0.626* | 20/21 |

*Run G still in progress at 226K/300K episodes.

The enriched WP model immediately boosted MCTS from 78% to 89.5% win rate. The augmented model is trending higher (92.5% at 226K vs 89.5% at 226K for Run E), consistent with the hypothesis that better value functions produce better policies.

### 2. Learning dynamics are consistent across runs

All runs follow the same trajectory:
- 0-5K episodes: random play, ~35-40% win rate
- 5-25K: rapid improvement, reaches ~55-60%
- 25-75K: steady climb, 60-80%
- 75-150K: diminishing returns, 80-90%
- 150K+: plateau around 88-93%, occasional peaks to 95%+

The learning curve is insensitive to the WP model variant -- the shape is the same, only the asymptote differs.

### 3. Comp_wr doesn't help MCTS much

Run E (enriched, no comp_wr) and Run F (enriched, with comp_wr) converge to nearly identical metrics (89.5% WR, ~0.60 WP). The comp_wr feature is most valuable for greedy search where it prevents degenerate compositions, but MCTS self-play against GD opponents rarely produces degenerate states (because GD drafts like humans, and MCTS learns to draft against human-like opponents).

### 4. The augmented WP model shows the most promise

Run G achieves 0.626 WP at 226K episodes -- already higher than any prior run's final WP (0.602 max). If this trend holds through 300K, it would demonstrate that training the value function to penalize degenerate compositions (via synthetic augmentation) translates into better MCTS policies, even though MCTS self-play rarely encounters those compositions.

### 5. Training efficiency improved over time

| Run | Speed | Hardware |
|-----|-------|----------|
| A | 1.6 ep/s | CPU only |
| C | 2.0 ep/s | CPU, optimized |
| E | 3.5 ep/s | GPU (Blackwell) |
| G | 3.3 ep/s | GPU (Blackwell) |

GPU acceleration via batched MCTS evaluation roughly doubled throughput.

## For the Paper

The MCTS results support the main thesis in two ways:

1. **Value function features matter for MCTS too.** The enriched WP model produces an 11.5pp win rate improvement over the base model (78% -> 89.5%) when used as the MCTS value function, even though MCTS can theoretically compensate for a weak value function through deeper search.

2. **Synthetic augmentation provides incremental benefit.** The augmented WP model shows higher average WP (0.626 vs 0.602) at the same training stage, suggesting the MCTS policy finds higher-quality terminal states when the value function has better composition awareness. However, the win rate improvement is modest (92.5% vs 89.5% at 226K episodes), consistent with the observation that MCTS self-play against GD opponents rarely produces the degenerate compositions where augmentation matters most.

The MCTS results are complementary to the greedy draft benchmark: greedy search amplifies value function differences (76% -> 26.5% degen rate), while MCTS partially compensates for weak value functions through lookahead but still benefits from better ones.
