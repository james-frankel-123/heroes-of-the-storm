# Draft Quality Metrics: Resilience, Counter Play, and Compositional Synergy

## Motivation

Current evaluation metrics capture composition safety (healer%, degenerate%) and aggregate context-awareness (average counter/synergy deltas). But they miss the temporal and strategic dimensions of drafting:

- **When** you pick a hero matters as much as **which** hero you pick
- Early picks should be resilient (hard to counter); late picks should exploit information (counter the opponent)
- A good drafter sequences picks strategically, not just individually

This proposal defines three draft quality axes with concrete, data-backed metrics using HeroesProfile pairwise statistics.

## Data Foundation

All metrics use the existing StatsCache:
- `get_counter(hero_a, hero_b, tier)` → win rate of hero_a vs hero_b
- `get_synergy(hero_a, hero_b, tier)` → win rate of team containing both
- `get_hero_wr(hero, tier)` → overall win rate of hero

The key derived quantity is the **normalized delta**: how much does the pairwise outcome deviate from what individual win rates predict?

```python
def counter_delta(hero_a, hero_b, stats, tier):
    """How much does hero_a over/underperform vs hero_b relative to expectation?"""
    raw = stats.get_counter(hero_a, hero_b, tier)
    if raw is None:
        return None
    wr_a = stats.get_hero_wr(hero_a, tier)
    wr_b = stats.get_hero_wr(hero_b, tier)
    expected = wr_a + (100 - wr_b) - 50
    return raw - expected

def synergy_delta(hero_a, hero_b, stats, tier):
    """How much does pairing hero_a and hero_b over/underperform?"""
    raw = stats.get_synergy(hero_a, hero_b, tier)
    if raw is None:
        return None
    wr_a = stats.get_hero_wr(hero_a, tier)
    wr_b = stats.get_hero_wr(hero_b, tier)
    expected = 50 + (wr_a - 50) + (wr_b - 50)
    return raw - expected
```

---

## Axis 1: Resilience (not getting countered)

### Concept

A resilient draft minimizes the opponent's ability to counter your picks. Early picks are most vulnerable because the opponent has 4-5 remaining picks to target them. A good drafter picks resilient heroes early and saves counter-vulnerable heroes for late.

### Per-Hero Counter Vulnerability Profile

For each hero, precompute their **counter vulnerability** from the full pairwise stats:

```python
def hero_vulnerability_profile(hero, all_heroes, stats, tier):
    """
    How counterable is this hero? Compute distribution of counter deltas
    from all opponents' perspectives.

    Returns:
        worst_counters: list of (opponent, delta) sorted by how badly they counter this hero
        avg_vulnerability: average counter delta AGAINST this hero (negative = hero is easily countered)
        vulnerability_variance: how spread the counter deltas are (high = polarized matchups)
        n_hard_counters: count of opponents with counter delta > +3.0 against this hero
        n_soft_counters: count with delta > +1.5
    """
    deltas = []
    for opp in all_heroes:
        if opp == hero:
            continue
        # From opponent's perspective: how well does opp do vs hero?
        d = counter_delta(opp, hero, stats, tier)
        if d is not None:
            deltas.append((opp, d))

    deltas_sorted = sorted(deltas, key=lambda x: -x[1])  # worst counters first
    values = [d for _, d in deltas]

    return {
        "worst_counters": deltas_sorted[:10],
        "avg_vulnerability": np.mean(values),
        "vulnerability_variance": np.var(values),
        "n_hard_counters": sum(1 for v in values if v > 3.0),
        "n_soft_counters": sum(1 for v in values if v > 1.5),
    }
```

### Draft-Level Resilience Metric

For each pick in the draft, compute how exposed it is to the opponent's **remaining** picks:

```python
def draft_resilience(our_picks_ordered, opp_picks_ordered, pick_steps, stats, tier, all_heroes):
    """
    For each of our picks, compute how badly the opponent's SUBSEQUENT picks counter it.

    Args:
        our_picks_ordered: our heroes in draft order
        opp_picks_ordered: opponent heroes in draft order
        pick_steps: list of (hero, team, step_number) for the full draft sequence
        stats: StatsCache
        tier: skill tier

    Returns:
        per_pick_exposure: for each of our picks, the average counter delta of opponent picks made AFTER it
        weighted_resilience: sum of per_pick_exposure weighted by how early the pick was
        early_pick_resilience: average exposure of our first 2 picks only
        late_pick_resilience: average exposure of our last 2 picks only
    """
    # Reconstruct which opponent picks happened AFTER each of our picks
    our_pick_steps = [(h, step) for h, team, step in pick_steps if team == "ours"]
    opp_pick_steps = [(h, step) for h, team, step in pick_steps if team == "theirs"]

    per_pick = []
    for our_hero, our_step in our_pick_steps:
        # Opponent picks that came after this pick
        subsequent_opp = [h for h, s in opp_pick_steps if s > our_step]
        if not subsequent_opp:
            per_pick.append({"hero": our_hero, "exposure": 0.0, "n_subsequent": 0})
            continue

        # How well do those subsequent opponents counter our hero?
        exposure_deltas = []
        for opp_h in subsequent_opp:
            d = counter_delta(opp_h, our_hero, stats, tier)
            if d is not None:
                exposure_deltas.append(d)

        avg_exposure = np.mean(exposure_deltas) if exposure_deltas else 0.0
        per_pick.append({
            "hero": our_hero,
            "exposure": avg_exposure,
            "n_subsequent": len(subsequent_opp),
        })

    exposures = [p["exposure"] for p in per_pick]

    return {
        "per_pick": per_pick,
        "avg_resilience": -np.mean(exposures),  # negate so positive = more resilient
        "early_pick_resilience": -np.mean(exposures[:2]) if len(exposures) >= 2 else 0,
        "late_pick_resilience": -np.mean(exposures[-2:]) if len(exposures) >= 2 else 0,
        "resilience_gradient": (
            np.mean(exposures[-2:]) - np.mean(exposures[:2])
        ) if len(exposures) >= 4 else 0,
        # Positive gradient = later picks are more exposed, which is WRONG
        # Negative gradient = later picks are less exposed, which is CORRECT
        # (early picks should be the exposed ones, since opponent had more chances to counter)
    }
```

### What Good Drafting Looks Like

- **Rehgar early**: low vulnerability (few hard counters), safe early pick. Resilience metric should show low exposure.
- **Malthael late**: high vulnerability to Brightwing/Tychus, but strong into beefy frontline. If picked late after seeing the opponent has no Brightwing, the exposure is low despite the hero being intrinsically vulnerable.
- **Resilience gradient should be negative**: early picks are inherently more exposed (more opponent picks follow), but a smart drafter picks LOW-vulnerability heroes early, partially offsetting this structural exposure.

### Hero-Level Validation

Before using in draft evaluation, validate against HeroesProfile pick position data:

```python
def validate_resilience_vs_pick_position(all_heroes, stats, tier, replay_data):
    """
    Do humans already pick vulnerable heroes later? If yes, our metric
    captures a real drafting signal. If not, the metric may not be meaningful.
    """
    for hero in all_heroes:
        profile = hero_vulnerability_profile(hero, all_heroes, stats, tier)
        avg_pick_position = average_pick_position_in_replays(hero, replay_data)
        # Scatter plot: vulnerability vs average pick position
        # Expect positive correlation: more vulnerable heroes picked later
```

---

## Axis 2: Counter Play (countering the opponent)

### Concept

A good drafter picks heroes that are strong against what the opponent has already picked. Unlike resilience (which is about YOUR vulnerability to FUTURE opponent picks), counter play is about YOUR strength against PAST opponent picks.

### Per-Pick Counter Quality

```python
def draft_counter_quality(our_picks_ordered, opp_picks_ordered, pick_steps, stats, tier):
    """
    For each of our picks, compute how well it counters the opponent's PRIOR picks.

    Returns:
        per_pick_counter: for each of our picks, the average counter delta vs opponent's prior picks
        weighted_counter: weighted by pick order (later picks should counter more)
        counter_gradient: do we counter more as draft progresses? (should be positive)
        counter_opportunities_taken: fraction of picks where a strong counter was available AND taken
    """
    our_pick_steps = [(h, step) for h, team, step in pick_steps if team == "ours"]
    opp_pick_steps = [(h, step) for h, team, step in pick_steps if team == "theirs"]

    per_pick = []
    for our_hero, our_step in our_pick_steps:
        # Opponent picks that were visible when we picked
        prior_opp = [h for h, s in opp_pick_steps if s < our_step]
        if not prior_opp:
            per_pick.append({"hero": our_hero, "counter_delta": 0.0, "n_prior_opp": 0})
            continue

        deltas = []
        for opp_h in prior_opp:
            d = counter_delta(our_hero, opp_h, stats, tier)
            if d is not None:
                deltas.append(d)

        avg_delta = np.mean(deltas) if deltas else 0.0
        per_pick.append({
            "hero": our_hero,
            "counter_delta": avg_delta,
            "n_prior_opp": len(prior_opp),
        })

    deltas_by_position = [p["counter_delta"] for p in per_pick]

    return {
        "per_pick": per_pick,
        "avg_counter": np.mean(deltas_by_position),
        "early_counter": np.mean(deltas_by_position[:2]) if len(deltas_by_position) >= 2 else 0,
        "late_counter": np.mean(deltas_by_position[-2:]) if len(deltas_by_position) >= 2 else 0,
        "counter_gradient": (
            np.mean(deltas_by_position[-2:]) - np.mean(deltas_by_position[:2])
        ) if len(deltas_by_position) >= 4 else 0,
        # Positive gradient = later picks counter harder, which is CORRECT
    }
```

### Counter Opportunity Analysis

Beyond just measuring whether picks counter, measure whether the drafter RECOGNIZED counter opportunities:

```python
def counter_opportunity_analysis(our_picks_ordered, opp_picks_ordered, pick_steps,
                                  stats, tier, all_heroes, valid_heroes_at_step):
    """
    At each pick step, what was the BEST available counter, and did the drafter take it?

    Returns:
        opportunities: list of (step, best_counter_hero, best_counter_delta, chosen_hero, chosen_delta)
        capture_rate: fraction of steps where chosen hero was within top-5 counters
        missed_counter_magnitude: average delta between best available and chosen
    """
    opportunities = []
    for our_hero, our_step in [(h, s) for h, t, s in pick_steps if t == "ours"]:
        prior_opp = [h for h, s in opp_pick_steps if s < our_step]
        if not prior_opp:
            continue

        # Score all valid heroes at this step by counter quality
        valid = valid_heroes_at_step[our_step]
        scored = []
        for candidate in valid:
            deltas = [counter_delta(candidate, opp_h, stats, tier) for opp_h in prior_opp]
            deltas = [d for d in deltas if d is not None]
            if deltas:
                scored.append((candidate, np.mean(deltas)))

        scored.sort(key=lambda x: -x[1])
        if not scored:
            continue

        best_hero, best_delta = scored[0]
        chosen_delta = next((d for h, d in scored if h == our_hero), 0)
        rank = next((i for i, (h, _) in enumerate(scored) if h == our_hero), len(scored))

        opportunities.append({
            "step": our_step,
            "best_counter": best_hero,
            "best_delta": best_delta,
            "chosen": our_hero,
            "chosen_delta": chosen_delta,
            "rank_of_chosen": rank,
            "delta_gap": best_delta - chosen_delta,
        })

    capture_rate = np.mean([1 if o["rank_of_chosen"] < 5 else 0 for o in opportunities])

    return {
        "opportunities": opportunities,
        "capture_rate": capture_rate,
        "avg_delta_gap": np.mean([o["delta_gap"] for o in opportunities]),
    }
```

---

## Axis 3: Compositional Synergy (team coherence)

### Concept

A good draft builds a team where heroes amplify each other. Synergy is not just about individual pairwise interactions — it's about the full 5-hero team working together.

### Incremental Synergy (team-building quality)

```python
def incremental_synergy(our_picks_ordered, pick_steps, stats, tier):
    """
    For each pick after the first, compute synergy with ALL prior teammates.
    A good drafter builds synergy incrementally — each pick should fit the existing team.

    Returns:
        per_pick_synergy: for each pick, average synergy delta with all prior teammates
        synergy_trajectory: does synergy improve, decline, or stay flat across picks?
        team_synergy: final full-team synergy (all 10 pairs)
    """
    per_pick = []
    teammates_so_far = []

    for our_hero, our_step in [(h, s) for h, t, s in pick_steps if t == "ours"]:
        if not teammates_so_far:
            per_pick.append({"hero": our_hero, "synergy_with_team": 0.0, "n_teammates": 0})
            teammates_so_far.append(our_hero)
            continue

        deltas = []
        for teammate in teammates_so_far:
            d = synergy_delta(our_hero, teammate, stats, tier)
            if d is not None:
                deltas.append(d)

        avg_syn = np.mean(deltas) if deltas else 0.0
        per_pick.append({
            "hero": our_hero,
            "synergy_with_team": avg_syn,
            "n_teammates": len(teammates_so_far),
        })
        teammates_so_far.append(our_hero)

    # Full team synergy: all C(5,2) = 10 pairs
    all_pairs = []
    for i, h1 in enumerate(our_picks_ordered):
        for h2 in our_picks_ordered[i+1:]:
            d = synergy_delta(h1, h2, stats, tier)
            if d is not None:
                all_pairs.append(d)

    return {
        "per_pick": per_pick,
        "avg_incremental_synergy": np.mean([p["synergy_with_team"] for p in per_pick[1:]]),
        "team_synergy": np.mean(all_pairs) if all_pairs else 0.0,
        "synergy_trajectory": [p["synergy_with_team"] for p in per_pick],
    }
```

---

## Combined Draft Quality Score

```python
def full_draft_quality(our_picks, opp_picks, pick_steps, stats, tier, all_heroes, valid_at_step):
    """Compute all three axes for one draft."""
    resilience = draft_resilience(our_picks, opp_picks, pick_steps, stats, tier, all_heroes)
    counters = draft_counter_quality(our_picks, opp_picks, pick_steps, stats, tier)
    synergy = incremental_synergy(our_picks, pick_steps, stats, tier)
    opportunities = counter_opportunity_analysis(
        our_picks, opp_picks, pick_steps, stats, tier, all_heroes, valid_at_step
    )

    return {
        # Resilience
        "resilience_avg": resilience["avg_resilience"],
        "resilience_early": resilience["early_pick_resilience"],
        "resilience_gradient": resilience["resilience_gradient"],

        # Counter play
        "counter_avg": counters["avg_counter"],
        "counter_late": counters["late_counter"],
        "counter_gradient": counters["counter_gradient"],
        "counter_capture_rate": opportunities["capture_rate"],
        "counter_delta_gap": opportunities["avg_delta_gap"],

        # Synergy
        "team_synergy": synergy["team_synergy"],
        "incremental_synergy": synergy["avg_incremental_synergy"],
    }
```

---

## Validation: Do These Metrics Predict Wins?

Before using these to evaluate strategies, validate that they actually correlate with game outcomes in the 275K replay dataset:

```python
def validate_metrics_vs_outcomes(replay_data, stats, tier):
    """
    For each replay, compute all three axes for both teams.
    Then check: do winning teams have better resilience/counter/synergy?

    Key questions:
    1. Does resilience correlate with winning? (early picks not countered)
    2. Does counter quality correlate with winning? (picked into opponent)
    3. Does team synergy correlate with winning?
    4. Which axis has the strongest predictive signal?
    5. Are the axes independent or correlated?
    """
    records = []
    for replay in replay_data:
        for team in ["team1", "team2"]:
            quality = full_draft_quality(
                replay[team]["picks"],
                replay[other_team]["picks"],
                replay["pick_sequence"],
                stats, tier, ALL_HEROES, None  # valid_at_step not needed for replays
            )
            quality["won"] = replay[team]["won"]
            records.append(quality)

    df = pd.DataFrame(records)

    # Correlation matrix
    print(df[["resilience_avg", "counter_avg", "team_synergy", "won"]].corr())

    # Per-axis win rate split
    for metric in ["resilience_avg", "counter_avg", "team_synergy"]:
        above_median = df[df[metric] > df[metric].median()]
        below_median = df[df[metric] <= df[metric].median()]
        print(f"{metric}: above median WR = {above_median['won'].mean():.3f}, "
              f"below = {below_median['won'].mean():.3f}")
```

---

## Validation: Does the WP Model Reward These?

```python
def validate_wp_model_sensitivity(wp_model, stats, tier, device):
    """
    Generate controlled draft pairs that differ ONLY on one axis,
    and check whether the WP model assigns higher value to the better draft.

    Tests:
    1. Resilience: same heroes, but early-vulnerable vs early-resilient ordering
       (this only matters if WP model sees pick order — if it only sees final comp, skip)
    2. Counter play: same team, but opponent has/doesn't have counters to our picks
    3. Synergy: two teams with same roles but different pairwise synergy

    Since current WP models only see final composition (not pick order),
    tests 1 is not applicable. Focus on 2 and 3.
    """

    # Test 2: Counter play
    # Pick a hero with known strong counters (e.g., Malthael)
    # Construct opponent team WITH and WITHOUT the counter (e.g., with/without Brightwing)
    # WP model should assign lower win probability when opponent has the counter

    # Test 3: Synergy
    # Construct two teams with identical role composition but different synergy scores
    # WP model should assign higher WP to the higher-synergy team

    pass  # Implementation depends on specific model interface
```

---

## Evaluating Strategies

Run these on the same 200-draft benchmark configs used for Table VII:

```
                     | Resil. | Resil. | Counter | Counter | Ctr    | Team   | Incr.
Strategy             | Avg    | Grad.  | Avg     | Late    | Capt%  | Syn.   | Syn.
─────────────────────┼────────┼────────┼─────────┼─────────┼────────┼────────┼───────
GD baseline          |        |        |         |         |        |        |
CQL α=1.0 (naive)    |        |        |         |         |        |        |
CQL α=2.0 (enriched) |        |        |         |         |        |        |
Enriched WP (greedy) |        |        |         |         |        |        |
Enr.+aug (greedy)    |        |        |         |         |        |        |
MCTS Run E           |        |        |         |         |        |        |
MCTS Run G           |        |        |         |         |        |        |
```

### Expected Patterns

**GD baseline**: Moderate resilience (humans intuitively pick safe heroes early), low counter play (humans follow meta over matchup), moderate synergy (humans pick standard combos). This is the behavioral baseline.

**CQL**: Similar to GD on all axes (since it mimics GD). Possibly slightly worse resilience due to Q-value noise causing some early-pick vulnerable heroes.

**Enriched greedy**: High counter play (the value function explicitly uses pairwise counter stats), high synergy (pairwise synergy features), but possibly low resilience (greedy search doesn't reason about future opponent picks — it can't protect its early picks from being countered later).

**Enriched+aug greedy**: Similar to enriched but with better composition safety constraining the hero pool.

**MCTS**: This is the interesting one. MCTS has multi-step lookahead, so it CAN reason about future opponent responses. If MCTS shows higher resilience than greedy, that's evidence that search depth enables strategic sequencing. If MCTS shows similar resilience to greedy despite deeper search, the policy head may not have learned to exploit this capability.

---

## Implementation Notes

### Pick Order Reconstruction
The greedy benchmark saves per-step records. Extract our_picks and opp_picks in draft order, with step numbers, to compute temporal metrics. For MCTS, the self-play logs should contain the full pick sequence.

### Normalization
Counter deltas and synergy deltas are on different scales depending on the hero pool and tier. Normalize within-tier for cross-tier comparisons, but report raw deltas within-tier for interpretability.

### Statistical Significance
With 200 drafts per strategy, per-pick metrics give ~1000 pick-level observations (200 drafts × 5 picks). Use bootstrap confidence intervals on strategy-level averages.

### What If Resilience Doesn't Vary?
If all strategies show similar resilience, it may be because:
1. The greedy search horizon prevents strategic sequencing (all strategies pick greedily regardless of future opponent picks)
2. The WP model doesn't encode pick order, so there's no signal to optimize resilience
3. Resilience only matters in adversarial opponent modeling, which our GD opponents don't do (they draft independently of our picks)

Point 3 is important: our GD opponents draft independently — they don't try to counter our picks. So resilience may not be rewarded in the current evaluation setup. This is a limitation worth noting, and an argument for adversarial opponent modeling in future work.

## CLI

```bash
set -a && source .env && set +a
python3 -u training/experiment_draft_quality.py --drafts 200
```

Save results to `training/experiment_results/draft_quality/`.
