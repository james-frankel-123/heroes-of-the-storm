# Task: Author Full Paper for IEEE Transactions on Games

## Overview

Write a full research paper (~8-12 pages, IEEE two-column format) for IEEE Transactions on Games.

This is a NEW paper, not an extended version of the CoG 2026 auxiliary paper. The thesis has shifted substantially. The CoG paper identifies the composition gap (the problem). This paper's central contribution is showing that CQL collapses to behavioral cloning and that composition safety and context-awareness are orthogonal axes of draft quality that no single existing method optimizes both. Cite the CoG paper as prior work: "Our earlier work [self-cite] identified the composition gap and proposed feature engineering and synthetic augmentation as mitigations. Here we show that algorithmic pessimism (CQL), while appearing to solve the problem, actually collapses to behavioral cloning — and that the hard problem was never composition safety but context-awareness."

**Title:** "Safety Without Understanding: Why Conservative Value Functions Collapse to Behavioral Cloning in Combinatorial Selection"

**Thesis:** Behavioral cloning and offline RL achieve composition safety by staying near human behavior but cannot discover context-aware drafting. Value function search discovers context-aware drafting but produces degenerate compositions due to OOD overestimation. Domain-structured features plus targeted OOD augmentation resolve this tension, achieving both safety and context-awareness within a standard supervised framework. MCTS with the augmented value function further improves both axes through deeper search.

## Formatting

- IEEE Transactions two-column format: \documentclass[journal]{IEEEtran}
- Target 10-12 pages including references and author bios
- Double-anonymous (no author names/affiliations in submission version)
- Use IEEEtran.bst for bibliography
- Include the bar chart figure from the CoG paper (updated if needed)

## Paper Structure and Content

### Abstract (~200 words)

The abstract should tell the full story:
- Value functions with similar accuracy (56.5-57.9%) produce policies disagreeing on 90%+ of decisions in combinatorial team selection
- Feature engineering + synthetic augmentation reduce degenerate compositions from 76% to 26.5% while producing context-aware drafts (positive counter/synergy responsiveness, 90 distinct heroes)
- CQL appears to solve composition safety (9% degenerate) but rich evaluation reveals it collapses to behavioral cloning: negative counter/synergy deltas, half the hero diversity, 62% agreement with the behavior policy
- The GD behavioral cloning baseline already achieves 99.5% healer and 1% degenerate — composition safety was never the hard problem
- MCTS with the augmented value function achieves the best overall performance (0.626 avg WP, 92.5% win rate vs GD opponents), confirming that deeper search with a well-calibrated value function further improves both safety and context-awareness
- The contribution: composition safety and context-awareness are orthogonal capabilities requiring fundamentally different solutions; algorithmic pessimism provides the former by mimicking human behavior, while domain-structured features with OOD augmentation provide both

### 1. Introduction (1 page)

Lead with the general problem: when learned value functions guide search in combinatorial selection, aggregate accuracy fails to predict policy quality. Two capabilities matter — composition safety (avoiding structurally broken selections) and context-awareness (selecting elements that interact well given the specific situation) — and existing methods trade one for the other.

Key points:
- The accuracy-policy gap: similar accuracy ≠ similar policy quality
- Heroes of the Storm draft selection as testbed: 90 heroes, ~10^13 matchups, training data concentrates on reasonable compositions
- Preview the arc: (1) identify the composition gap, (2) feature engineering + augmentation partially close it while maintaining context-awareness, (3) CQL appears to solve it completely but actually collapses to behavioral cloning — achieving safety by mimicking humans rather than understanding composition, (4) MCTS with augmented value function achieves the best of both
- State contributions as a numbered list:
  1. Demonstrate that value functions with 1.4pp accuracy difference disagree on 90%+ of decisions
  2. Show feature engineering + synthetic augmentation produce both safe AND context-aware drafts
  3. Reveal that CQL collapses to behavioral cloning — first empirical demonstration in combinatorial selection
  4. Decompose draft quality into orthogonal axes (safety vs context-awareness) with quantitative metrics
  5. Show MCTS with augmented value function achieves best overall performance across both axes

### 2. Background and Related Work (1.5 pages)

#### 2a. MOBA Draft Prediction
- Chen et al. [RecSys 2018]: MCTS + win predictor for Dota 2. Multi-hot features, evaluated by same model. Didn't examine OOD behavior or composition quality.
- Gourdeau & Archambault [IEEE T-Games 2021]: Discriminative NN for HotS/Dota 2 professional drafts. Behavioral cloning approach. We reproduce their architecture and show it exhibits OOD overestimation (52.8% WP for 5-tank teams).
- Summerville et al. [2016]: LSTM behavioral cloning for HotS professional drafts.
- DraftRec [WWW 2022]: Personalized recommendation with distributional constraint. Avoids OOD by design but cannot discover novel strategies.
- Note the gap: no prior work examines how value function features affect policy quality, measures the composition gap, or tests whether CQL provides genuine compositional reasoning vs behavioral cloning.

#### 2b. Offline RL and OOD Overestimation
- CQL [Kumar et al., NeurIPS 2020]: Conservative Q-learning. We test it directly and show it collapses to behavioral cloning in this domain.
- BEAR [Kumar et al., NeurIPS 2019]: Policy constraint approach. DraftRec implicitly does this.
- MCQ [Lyu et al., 2022]: Mildly conservative Q-learning — relevant to the "too conservative" finding.
- Frame our contribution: first empirical demonstration that CQL's pessimism eliminates context-awareness in combinatorial selection, achieving safety by the same mechanism as behavioral cloning.

#### 2c. Value Function Design for Planning
- AlphaGo/AlphaZero [Silver et al.]: Value + policy with MCTS.
- Robotics [Kalashnikov et al.]: Learned value functions for manipulation.
- Our work: controlled experiment showing near-identical accuracy yields 90%+ decision disagreement, with quantitative decomposition into safety and context-awareness.

### 3. Problem Setup (1 page)

#### 3a. Draft Formalization
- Sequential two-player game, 16 actions (6 bans, 10 picks), fixed alternation order
- State: s_t ∈ R^289 (multi-hot picks/bans + map + tier + step)
- Action: discrete hero selection from ~84-90 valid heroes
- Terminal reward: game outcome
- Behavior policy π_β: 275K Storm League replays

#### 3b. Value Function Variants
Table I: naive (197d, 56.5%), hero strength (209d, 57.0%), enriched (283d, 57.9%)
Expand enriched feature description:
- Role counts: 9 fine-grained roles × 2 teams (18 dims)
- Pairwise counters/synergies: normalized deltas (76 dims)
- Composition WR: Heroes Profile 9M-game database (4 dims), encoding (w-50)/10 and log(1+g)/15, 33% pessimistic default for unseen compositions
- Coverage: 121-164 of 252 possible compositions per tier, 99.9%+ of real games

#### 3c. Evaluation Framework

**Composition safety metrics:**
- Healer rate, frontline rate, ranged damage rate, degenerate rate (missing any of these or 3+ role stacking)

**Context-awareness metrics:**
- Counter responsiveness: average normalized counter delta vs opponent. Positive = favorable matchups.
- Synergy exploitation: average normalized synergy delta within team. Positive = good internal synergy.
- Hero diversity: distinct heroes, Shannon entropy, top-10 concentration
- GD similarity: agreement rate with behavioral cloning model

**Cross-evaluation:** Each draft scored by multiple WP model variants.

#### 3d. Search Methods
- Greedy search: evaluate all ~84 candidates via batched GD rollouts + V_θ terminal evaluation
- MCTS: AlphaZero-style with policy + value network, 200 simulations per move, GD opponent sampling
- GD behavioral cloning: argmax of behavioral cloning model (the control)
- CQL: argmax of conservatively-learned Q-function (no rollout needed)

#### 3e. Sanity Tests
21 hand-crafted scenarios: 8 absurd, 5 trap, 5 normal, 3 symmetry.

### 4. The Composition Gap (1.5 pages)

Results from the CoG paper, expanded with more detail.

#### 4a. Cross-Evaluation
Table II. Diagonal dominance. Enriched sacrifices 6.9pp on naive evaluator for compositional quality invisible to single-model metrics.

#### 4b. Composition Quality
Table III. Healer: 49% → 39% → 66.5%. Hero strength WORSE than naive on healers. Statistical tests with χ², p-values, CIs. Phase divergence: 9.5% ban, 9.0% early, 4.8% late.

#### 4c. The Accuracy-Policy Disconnect
4,100-config sweep. Pairwise features dominate accuracy (+4.5%). comp_wr contributes +0.2% to accuracy but makes the model score 5-tanks at 17.2% vs 44.7%. Disagreement analysis: 41/50 no-healer, 48/50 high-WR-bad-comp.

#### 4d. Independent Baseline
Gourdeau's HotS drafter reproduced: 55.3% accuracy, 13/21 sanity, 5-tanks at 52.8% WP (rated as favored). Pathology is architectural.

### 5. Closing the Gap (1.5 pages)

#### 5a. Residual Gap Analysis
512→256→128 architecture, still 52% degenerate. Three factors: search horizon, training concentration, feature granularity.

#### 5b. Synthetic Augmentation
~100 unseen compositions per tier from 252 possible. 30K synthetic records at 10% WR. 27-config sweep: Tier 2 (unseen) is dominant, Tier 1 (sparse real) had no effect. Lower WR (10%) better than 20-30%.

#### 5c. Augmentation Results
Table IV: healer 64% → 94.5%, degenerate 52% → 26.5%, accuracy cost 0.8pp. Architecture alone only moves 58% → 52%. The mechanism: model learns to trust comp_wr because training includes examples where low comp_wr correlates with losses. Residual 26.5% from search horizon + genuine ambiguity (no-healer wins 43% in practice).

### 6. CQL: Safety Without Understanding (2 pages)

The central new contribution.

#### 6a. CQL Setup
- Draft as offline RL: 275K replays → ~8.8M transitions (16 steps × 2 perspectives)
- Q-network: state → 512→256→128→90
- Monte Carlo returns with γ=1.0 (terminal-only reward, fixed episode length)
- This makes CQL = supervised Q-regression + pessimism penalty on unseen actions
- Sweep: α ∈ {0.1, 0.5, 1.0, 2.0, 5.0}, also CQL with enriched features (375d input) at α ∈ {0.5, 2.0}

#### 6b. The Misleading Result
CQL composition metrics: α=1.0 gets 9% degenerate, 93.5% healer with naive features. Appears to completely subsume feature engineering. Present this as it initially appeared — surprisingly strong.

#### 6c. Rich Evaluation Reveals the Truth
THE KEY TABLE:

```
Method              | Healer% | Degen% | Counter | Synergy | Distinct | Entropy | Top10% | GD Sim%
GD baseline         | 99.5    | 1.0    | -0.09   | -0.05   | 40       | 4.24    | 75.3   | 83.2
CQL naive α=1.0     | 93.5    | 11.0   | -0.10   | -0.14   | 44       | 4.42    | 70.7   | 61.5
CQL enriched α=0.5  | 98.0    | 5.5    | -0.10   | -0.09   | 38       | 4.24    | 72.3   | —
CQL enriched α=2.0  | 98.5    | 3.0    | -0.11   | -0.04   | 38       | 4.30    | 72.7   | —
Enriched WP greedy  | 77.0    | 42.5   | +0.14   | +1.03   | 86       | 5.96    | 32.0   | 4.4
Enriched+aug greedy | 96.0    | 13.5   | +0.06   | +0.80   | 90       | 5.97    | 32.8   | 4.5
```

Walk through each column systematically:
- Counter responsiveness: CQL/GD negative, enriched positive
- Synergy: CQL/GD negative, enriched strongly positive (+1.03)
- Diversity: CQL 38-44 heroes, enriched 86-90. Top-10: CQL 70-75%, enriched 32%
- GD similarity: CQL 57-62%, enriched 4.4%. CQL makes similar decisions to behavioral cloning.

#### 6d. CQL + Enriched Features
Adding enriched features to CQL improves safety (3-5.5% degen) but NOT context-awareness. Counter/synergy/diversity indistinguishable from naive CQL. The features help CQL be more precise about "what human drafting looks like" but don't enable matchup-aware decisions. CQL's pessimism dominates regardless of input.

#### 6e. The GD Baseline Revelation
GD achieves 99.5% healer, 1% degenerate — better than any other method on safety. Composition safety was never the hard problem. Humans solve it; copying humans preserves the solution. The composition gap only emerges through optimization against a learned value function. CQL "solves" it by approximately reproducing behavioral cloning, not by understanding composition.

#### 6f. 100% Ranged as a Red Flag
CQL α=0.5 produces 100% ranged selection — never deviating from the standard template. This is overly conservative: 4-bruiser+healer comps are genuinely strong in certain matchups. CQL penalizes ALL uncommon actions, not just bad ones, collapsing the hero pool to a safe meta subset. This is the known tradeoff with pessimistic offline RL in domains with concentrated behavior policies.

### 7. MCTS with Augmented Value Function (1 page)

#### 7a. MCTS Training Setup
- AlphaZero-style: policy head + value head, shared residual backbone
- 200-400 MCTS simulations per move during self-play
- Opponent: sampled from pool of 5 GD variants
- Five training runs with different WP models as value function

#### 7b. MCTS Training Results

```
Run | WP Model        | Episodes | Final WR | Final WP | Peak WR
A   | Base (197d)     | 79K      | 78.0%    | 0.546    | 84.0%
C   | Enriched (283d) | 167K     | 89.0%    | 0.598    | 96.0%
E   | Enriched (300K) | 300K     | 89.5%    | 0.602    | 95.5%
F   | + comp_wr       | 300K     | 89.5%    | 0.598    | 93.5%
G   | Augmented       | 226K*    | 92.5%    | 0.626    | 95.5%
```

*Run G still in progress at time of writing — report final numbers.

Key findings:
- Enriched value function (Run C/E) dramatically improves over base (Run A): 89.5% vs 78.0% win rate
- Augmented value function (Run G) achieves highest average WP ever recorded (0.626 vs 0.602 for enriched)
- The win rate gain from augmentation is more modest (92.5% vs 89.5%) because MCTS self-play against GD opponents rarely produces the degenerate states where augmentation matters most — the policy head learns to avoid them through the MCTS training loop
- comp_wr alone (Run F) doesn't improve over enriched (Run E) in MCTS — the MCTS policy head already learns composition from the enriched features during training; comp_wr's value is primarily in the greedy setting where there's no learned policy

#### 7c. MCTS vs Greedy vs CQL
If rich evaluation metrics are available for the MCTS policy, include them here. The prediction: MCTS with augmented value function should show positive counter/synergy (context-awareness from the enriched value function), high hero diversity (from MCTS exploration), AND composition safety (from the augmented value function's OOD calibration + the policy head learning from self-play).

This would make MCTS+augmented the only method that achieves all properties simultaneously:
- GD/CQL: safety ✓, context-awareness ✗
- Greedy enriched: context-awareness ✓, safety ✗ (partially fixed by augmentation)
- MCTS+augmented: safety ✓, context-awareness ✓ (predicted — verify with data)

If rich metrics aren't available for MCTS, note this as future work and present only the training curves and WP/WR numbers.

#### 7d. Search Depth Matters
The residual 26.5% degenerate rate from greedy search is partially a depth-1 limitation. MCTS with 200 simulations provides multi-step lookahead, which should allow the agent to plan for future healer availability rather than greedily evaluating one pick at a time. The augmented WP model's 0.626 avg WP (vs 0.602 for non-augmented) suggests the augmentation helps even with deeper search.

### 8. Discussion (0.75 pages)

#### 8a. Two Orthogonal Axes
- Composition safety: "does this team have the right roles?" Trivially solved by copying human behavior.
- Context-awareness: "does this team counter the opponent and synergize internally?" Requires value function search with domain features.
- No single method optimizes both without additional engineering (features + augmentation).

#### 8b. Why CQL Collapses
CQL's pessimism penalizes ALL uncommon actions, not just bad ones. In draft, the behavior policy is concentrated (humans pick from ~40 heroes regularly), so CQL's penalty suppresses the other 50 heroes regardless of matchup context. With enriched features, CQL becomes more precise about "what's in-distribution" but still can't distinguish "uncommon and bad" from "uncommon and good for this matchup." This is a structural limitation of pessimistic offline RL in domains with concentrated behavior policies.

#### 8c. Practical Implications
- For draft recommenders: value function search with enriched features + augmentation, not CQL or behavioral cloning
- For value function evaluation: use policy-level metrics (counter responsiveness, synergy, diversity), not just accuracy
- For offline RL: measure action diversity alongside reward; CQL may collapse diversity in domains with concentrated behavior policies
- For MCTS systems: the value function's OOD calibration matters even with deep search; augmentation helps MCTS training produce higher-WP policies

### 9. Conclusion (0.5 pages)

The composition gap reveals two orthogonal axes of quality in combinatorial selection:
1. Composition safety — trivially achieved by behavioral cloning (99.5% healer, 1% degenerate), approximately achieved by CQL (93-98% healer, 3-11% degenerate), but at the cost of zero context-awareness
2. Context-awareness — achieved only by value function search with domain features (positive counter/synergy, 90 distinct heroes, 4.4% GD similarity), but requiring feature engineering and OOD augmentation for composition safety

CQL's apparent success (9% degenerate) is misleading: it collapses to behavioral cloning as shown by negative counter/synergy deltas, half the hero diversity, and 62% GD agreement. Adding enriched features to CQL does not recover context-awareness.

The solution that achieves both axes: domain-structured features (compositional reasoning) + synthetic augmentation (OOD calibration) + value function search (context-aware action selection). MCTS with the augmented value function achieves the highest overall performance (0.626 avg WP), combining the benefits of deeper search with a well-calibrated value function.

**Limitations:** Single game domain (discontinued game with shrinking player base), fixed GD opponent model, greedy search for most comparisons. The CQL collapse may be specific to domains with highly concentrated behavior policies. MCTS results are preliminary (Run G still in progress).

**Future work:**
- Full rich evaluation metrics for MCTS policies (counter/synergy/diversity)
- Offline RL methods that preserve action diversity while maintaining conservatism
- Testing the safety/context-awareness decomposition in other combinatorial domains
- In-game win probability prediction using temporal replay data

### References (~20-25)

Expand from the CoG paper's 9 references. Include:
- Self-cite: CoG 2026 auxiliary paper
- Chen et al. RecSys 2018 (MCTS Dota 2 drafting)
- Gourdeau & Archambault IEEE T-Games 2021 (HotS/Dota 2 discriminative NN)
- Lee et al. WWW 2022 (DraftRec)
- Summerville et al. 2016 (HotS draft prediction)
- Kumar et al. NeurIPS 2020 (CQL)
- Kumar et al. NeurIPS 2019 (BEAR)
- Levine et al. 2020 (offline RL tutorial)
- Lyu et al. 2022 (Mildly Conservative Q-Learning — relevant to "too conservative")
- Silver et al. Nature 2016 (AlphaGo)
- Kalashnikov et al. CoRL 2018 (robotics value functions)
- Semenov et al. 2016 (ML for Dota 2 prediction — early work)
- Hanke & Chaimowicz 2017 (association rules for hero recommendations)
- Pobiedina et al. 2013 (team formation in MOBAs)
- BPCoach if relevant (visual analytics for professional drafting)
- Any additional works on distributional shift, combinatorial optimization, or value function design for planning

## Data Sources

All experimental results are in:
- `training/experiment_results/` — main experiment (cross-eval, composition quality)
- `training/experiment_results/synthetic_augmentation/` — augmentation sweep + Stage 2
- `training/experiment_results/rich_evaluation/` — counter/synergy/diversity/GD-similarity metrics
- `training/experiment_results/cql/` — CQL training and evaluation (naive + enriched features)
- `training/win_prob_sweep_results.csv` — 4,100-config feature ablation
- wandb logs for MCTS training runs (Runs A, C, E, F, G)

The CoG paper source is in `paper/draft.tex`. This is a separate paper — reuse data and some methodology descriptions but do NOT copy-paste sections. The framing, narrative arc, and conclusions are different.

## Narrative Arc

The paper should read as a detective story with a twist:

1. **Setup** (Sections 1-3): Learned value functions guide draft search. We have models with similar accuracy. What could go wrong?

2. **The Problem** (Section 4): Everything. Models disagree on 90%+ of decisions. The naive model recommends 5 tanks. The enriched model is better but still produces 58% degenerate compositions. An independent baseline confirms this is architectural.

3. **A Partial Fix** (Section 5): Synthetic augmentation for unseen compositions teaches the model to trust its own features. Healer rate goes from 64% to 94.5%, degenerate from 52% to 26.5%. Good progress, but the residual suggests the problem isn't fully solved.

4. **The Apparent Complete Fix** (Section 6b): CQL enters. 9% degenerate rate with just multi-hot features! No feature engineering needed! It seems to subsume everything we just did...

5. **The Twist** (Section 6c-6e): Rich evaluation reveals CQL is behavioral cloning in disguise. Negative counter/synergy, half the hero diversity, 62% GD agreement. The GD baseline already gets 1% degenerate. CQL solved a problem that was never the hard one. The hard problem — context-awareness — is untouched by CQL.

6. **The Resolution** (Section 7): MCTS with the augmented value function achieves both safety and context-awareness through deeper search with a well-calibrated value function, achieving the highest overall performance.

7. **The Insight** (Section 8): Safety and context-awareness are orthogonal. Different methods solve different axes. Only domain-structured features with OOD augmentation plus search achieve both.

## Tone and Framing

- Frame as general ML contribution (learned value functions for combinatorial selection) with HotS as testbed
- The game is the experimental apparatus, not the research goal
- Lead with the CQL finding as the central surprise — it's what makes this paper different from the CoG paper
- Be honest about limitations (single domain, greedy search baseline, discontinued game)
- The narrative twist (CQL collapses to behavioral cloning) should land as a genuine surprise — build up the CQL composition results before revealing the rich evaluation metrics
- The GD baseline achieving 99.5% healer / 1% degenerate is the paper's most surprising single number

## What NOT to Do

- Don't present this as an "extended version" of the CoG paper — it's a new paper with a different thesis
- Don't lead with HotS game mechanics — lead with the ML problem
- Don't include every ablation config — summarize sweeps, highlight key findings
- Don't present CQL results without the rich evaluation context — the naive composition metrics are misleading in isolation
- Don't overstate — this is one domain with one search method
- Don't hide honest findings (26.5% residual degenerate, 0.8pp accuracy cost, MCTS Run G still in progress)
- Don't pad with game background — CoG/T-Games reviewers know MOBAs
- Don't use em-dashes or other overdone LLM writing tropes
