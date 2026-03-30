# MCTS Leaf Evaluation Fix: Three Approaches

## The Problem We Just Identified

The WP model was trained on complete 5v5 drafts but MCTS evaluates partial states (2-3 heroes per team at mid-draft nodes). This is out-of-distribution evaluation — the model extrapolates rather than interpolates. This explains why:
- WP counter signal is r=0.05 at early picks, r=0.32 at late picks (partial states degrade pairwise signal)
- MCTS learns resilience/sequencing (broad patterns survive noisy evaluation) but not counter/synergy (fine-grained pairwise signal doesn't survive)
- Greedy search achieves positive counter/synergy because it rolls out to completion before evaluating

We're fixing this three ways. Run all three overnight.

---

## Approach 1: Complete-State Leaf Evaluation (Rollout to Terminal)

### Concept
Instead of evaluating partial states with the WP model, roll out every MCTS leaf to a complete draft using GD models, THEN evaluate the complete state with the enriched WP model. This is what greedy search already does — we're bringing it into MCTS.

### Implementation
When MCTS expands a leaf node at step t:
1. Take the current partial draft state
2. Roll out remaining (16 - t) steps using GD models (sample from the 5-variant pool)
3. Evaluate the complete 5v5 terminal state with the enriched WP model (symmetrized)
4. Use that WP score as the leaf value for MCTS backup

This replaces the current `_evaluate_wp(partial_state)` call with `_rollout_then_evaluate_wp(partial_state, gd_models)`.

### Cost
Each simulation now requires ~8 GD forward passes (average remaining steps) + 1 WP forward pass instead of just 1 WP forward pass. With 200 sims per move, that's ~1600 GD forward passes per move decision. At 16 moves per episode, ~25,600 GD forward passes per episode.

GD models are small MLPs (289→256→128→90). At your CUDA speeds this should be feasible but will be slower. Estimate: 3-5x slowdown per episode. If 300K episodes currently takes 18 minutes, expect 60-90 minutes for 300K. Still very feasible for overnight runs.

### Rollout strategy
- Sample a different GD model per rollout step (not per rollout — per step within the rollout)
- For each leaf, do 1 rollout (not 3). The noise from single rollouts will average out across MCTS simulations.
- The GD model plays BOTH sides in the rollout (our remaining picks + opponent remaining picks), using temperature sampling for diversity

### CUDA considerations
The main change is that leaf evaluation now involves a variable-length sequential rollout instead of a single forward pass. Options:
- **Simplest:** do rollouts in Python, call CUDA only for the GD/WP forward passes. This is slower but straightforward.
- **Faster:** batch rollouts across multiple simultaneous MCTS simulations. If you can run 32 simulations in parallel, batch all 32 leaf rollouts together — each step is a batched GD forward pass of size 32.
- **Fastest:** implement GD rollout in the CUDA kernel. The GD model is tiny enough to fit alongside the policy/value network.

Start with the simplest approach. If it's too slow for 300K episodes overnight, reduce to 100K episodes (still informative).

### Training config
- 15 seeds, 300K episodes (or 100K if too slow), enriched WP model, 200 sims
- Compare against E baseline on all draft quality metrics

---

## Approach 2: Partial-State WP Model (Online Learning)

### Concept
Train a WP model that is explicitly calibrated for partial draft states. Instead of training only on complete drafts → game outcome, train on (state at step t) → game outcome for all t.

### Two sub-approaches:

### 2a: Pre-trained partial-state WP model
Train before MCTS, use as a fixed leaf evaluator during MCTS training.

**Data extraction:**
From each of the 275K replays, extract the draft state at every pick step (steps 4-15, skipping bans):
```python
for replay in replays:
    state = initial_state(replay)
    for step_idx, step in enumerate(replay["draft_order"]):
        state.apply_action(step)
        if step["type"] == "pick":  # skip bans
            features = extract_enriched_features(state, stats_cache)
            outcome = replay["winner"]  # 0 or 1
            partial_samples.append((features, outcome, step_idx))
```

This gives ~275K × 10 pick steps × 2 perspectives = ~5.5M training samples. Each sample includes the draft step index, which the model can use to calibrate predictions per stage.

**Architecture:**
Same enriched WP architecture (283 dims) plus a step embedding:
```python
class PartialStateWP(nn.Module):
    def __init__(self, input_dim=283, step_embed_dim=8, hidden=[256, 128]):
        self.step_embed = nn.Embedding(16, step_embed_dim)
        self.net = MLP(input_dim + step_embed_dim, hidden, output=1)
    
    def forward(self, features, step_idx):
        step_emb = self.step_embed(step_idx)
        x = torch.cat([features, step_emb], dim=-1)
        return torch.sigmoid(self.net(x))
```

The step embedding lets the model learn that predictions at step 4 should be closer to 0.5 (low confidence) while predictions at step 14 can be more extreme.

**Expected accuracy:** Much lower than 57% at early steps (step 4 might be ~52-53% — barely above chance). That's fine. Calibrated weak signal is better than miscalibrated strong signal from a model evaluated OOD.

**Training:**
- Standard BCE loss, weighted by step (optional: upweight later steps where signal is stronger)
- Same team-swap augmentation
- Same enriched features, computed at each partial state
- Train for convergence, save checkpoint
- Use this checkpoint as the fixed WP model for MCTS training

### 2b: Online partial-state model (evolves during MCTS training)
Start with the pre-trained partial-state WP from 2a. During MCTS self-play, collect (partial_state, eventual_WP_score) pairs from each episode and periodically fine-tune the partial-state WP model.

**Data collection during MCTS:**
Each self-play episode generates ~10 pick states. After the episode completes, you know the terminal WP score (from the complete-state enriched WP model). Use this as the label for all partial states in that episode:
```python
# After episode completes:
terminal_wp = evaluate_wp_symmetrized(enriched_wp, terminal_state)
for step_state in episode_pick_states:
    online_buffer.append((step_state.features, terminal_wp, step_state.step_idx))
```

Every N episodes (e.g., every 10K), fine-tune the partial-state WP model on the accumulated buffer for a few epochs, then resume MCTS with the updated model.

**Why this helps beyond 2a:** The pre-trained model learns from human replays, which concentrate on standard compositions. During MCTS self-play, the policy explores compositions that humans don't play. The online model adapts to these — learning that certain unusual compositions (which the pre-trained model would rate at ~50%) actually score well or poorly according to the enriched WP evaluator.

### Opponent diversity (2a enhancement)
Add self-play to the opponent pool. Currently MCTS plays against 5 GD models. Add the MCTS policy itself as a 6th opponent, sampled 20% of the time:
```python
def sample_opponent(gd_models, mcts_policy, step):
    if random.random() < 0.2:
        return mcts_policy  # self-play
    else:
        return random.choice(gd_models)  # GD opponent
```

This forces the policy to develop resilience against strategic opponents, not just behavioral cloning opponents. The GD models don't counter-pick; a self-play opponent will, creating training signal for resilience.

**Caution:** self-play opponents early in training are terrible. Start self-play mixing only after 100K episodes (when the policy is decent). Before that, use GD only.

### Training config
- **2a (fixed partial WP):** Train partial WP model first (~30 min). Then 15 seeds, 300K episodes, 200 sims, partial WP for leaf evaluation.
- **2b (online partial WP):** Same as 2a but fine-tune every 10K episodes. Likely only feasible for 5 seeds overnight due to complexity.
- Both use enriched WP features, symmetrized evaluation.

---

## Approach 3: Hybrid (MCTS Early + Greedy Late)

### Concept
Use MCTS for early draft decisions (bans and first 2-3 picks, steps 0-8) where strategic sequencing matters and the WP model's pairwise signal is weak anyway. Switch to greedy enriched search for late picks (steps 9-15) where counter/synergy signal is strong and greedy directly exploits it.

### Implementation
No training needed — this combines existing models at inference time.

```python
def hybrid_draft(state, mcts_policy, wp_model, gd_models, switch_step=9):
    for step in range(16):
        if step < switch_step:
            # MCTS for early decisions (sequencing, resilience)
            action = mcts_pick(state, mcts_policy, sims=200)
        else:
            # Greedy for late decisions (counter, synergy)
            action = greedy_pick(state, wp_model, gd_models, rollouts=3)
        state.apply_action(action)
    return state
```

### Switch point sweep
Test different switch points to find optimal transition:
- Switch at step 7 (after first 2 picks): MCTS handles bans + first picks only
- Switch at step 9 (after mid-draft bans): MCTS handles all bans + first 3 picks
- Switch at step 11 (late switch): MCTS handles most of draft, greedy only for last 2-3 picks
- Switch at step 13 (very late): MCTS handles everything except final 2 picks

### Which MCTS policy to use
Use the best E baseline checkpoint (E_seed0 or best of 15 seeds). No need to retrain.

### Which WP model for greedy
Use the enriched WP model (not augmented) — it has the strongest pairwise signal (r=0.65 synergy, r=0.32 counter). The greedy evaluation rolls out to completion with GD models, so OOD isn't an issue.

### Evaluation
Run 200 draft configs for each switch point. Full draft quality metrics. Compare against:
- Pure MCTS E (baseline)
- Pure greedy enriched
- Pure greedy enriched+augmented

### This is the fastest experiment
No training. Just inference with existing models. Should run in minutes per switch point. Do this FIRST as a quick win while the training runs are going.

---

## Execution Plan for 24 Hours

### Hour 0-1: Hybrid experiment (Approach 3)
Run immediately — no training needed.
- 5 switch points × 200 drafts × full metrics
- Should complete in under an hour
- This gives immediate signal on whether the hybrid concept works

### Hour 0-1 (parallel): Train partial-state WP model (Approach 2a prerequisite)
- Extract partial-state training data from 275K replays
- Train partial-state WP model (~30 min)
- Validate: check accuracy at each step, check counter/synergy sensitivity at partial states

### Hour 1-3: Launch Approach 1 (rollout MCTS) — 3 seeds
- Start with 3 seeds to gauge timing and early results
- If 300K episodes takes <90 min per seed, queue remaining 12 seeds
- If >90 min, run 100K episodes for all 15 seeds instead

### Hour 1-3: Launch Approach 2a (partial WP MCTS) — 3 seeds
- Use the trained partial-state WP from above
- 300K episodes, 200 sims, 3 initial seeds
- If results look promising, queue remaining 12 seeds

### Hour 3-8: Based on early results, prioritize
- If Approach 1 (rollout) shows dramatic counter/synergy improvement in 3 seeds → queue remaining 12 seeds
- If Approach 2a (partial WP) shows improvement → queue remaining seeds + start 2b (online) for 3-5 seeds
- If hybrid (Approach 3) already achieves everything → you might not need the training runs at all

### Hour 8-24: Full seed sweeps for the winning approach
- 15 seeds of whichever approach looks best
- Full draft quality evaluation on all completed seeds
- Compare against all baselines

### Self-play opponent mixing (Approach 2a enhancement)
Add to whichever approach wins. After initial results stabilize:
- Re-run best approach with 20% self-play opponents (starting at episode 100K)
- 5 seeds, 300K episodes
- Check if resilience gradient improves (it should — self-play opponents counter-pick)

---

## Evaluation for All Approaches

Same metrics across the board:
```
           | Counter | Ctr Early | Ctr Late | Synergy | R.Grad | Hlr% | Deg% | Div | WR% | Avg WP
E baseline |  -0.082 |   -0.069  |  -0.114  |  0.503  | -0.578 | 86%  | 26%  | 23  | 91% | 0.553
Approach 1 |    ???  |     ???   |    ???   |   ???   |   ???  | ??%  | ??%  |  ?? | ??% | 0.???
Approach 2a|    ???  |     ???   |    ???   |   ???   |   ???  | ??%  | ??%  |  ?? | ??% | 0.???
Approach 3 |    ???  |     ???   |    ???   |   ???   |   ???  | ??%  | ??%  |  ?? | ??% | 0.???
Greedy enr.|  +0.081 |     ???   |  +0.340  |  1.690  | +0.420 | 77%  | 42%  |  86 |  —  |   —
```

### Success criteria
A method that achieves:
- Counter avg ≥ 0 (not negative — actively counter-picking)
- Synergy ≥ 0.5 (at least matching E baseline)
- Resilience gradient negative (strategic sequencing preserved)
- Healer ≥ 85%, Degenerate ≤ 25%
- Hero diversity ≥ 20

If ANY approach achieves all five, that's the paper's resolution.

## CLI

```bash
set -a && source .env && set +a

# Approach 3 (hybrid) — run first, fastest
python3 -u training/experiment_hybrid_draft.py --configs 200 --switch-points 7,9,11,13

# Approach 2a prerequisite — train partial-state WP
python3 -u training/train_partial_wp.py --replays all --steps pick_only

# Approach 1 (rollout MCTS) — 3 initial seeds
python3 -u training/train_draft_policy.py --config rollout_leaf --seeds 3 --episodes 300000

# Approach 2a (partial WP MCTS) — 3 initial seeds
python3 -u training/train_draft_policy.py --config partial_wp --seeds 3 --episodes 300000

# Evaluate completed seeds
python3 -u training/evaluate_draft_quality.py --checkpoint <path> --configs 200 --sims 200
```

Save all results to `training/experiment_results/leaf_eval_fix/`.
