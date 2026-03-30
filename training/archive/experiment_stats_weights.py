"""
Experiment: optimize Stats Mode heuristic weights via grid search.

Tests different weight combinations for hero WR, counters, synergies,
map-specific bonus, and sample-size scaling. Measures avg WP against
GD opponents (same setup as benchmark_stats_vs_policy.py).

Usage:
    set -a && source .env && set +a
    python3 -u training/experiment_stats_weights.py
"""
import os
import sys
import random
import itertools
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from shared import NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS
from train_draft_policy import (
    DraftState, DRAFT_ORDER, _evaluate_wp, load_pretrained_models,
)
from train_generic_draft import GenericDraftModel
from benchmark_stats_vs_policy import (
    load_stats_data, get_hero_wr, gd_pick,
)


def stats_pick_weighted(
    state: DraftState,
    hero_wr, hero_map_wr, synergies, counters,
    weights: dict,
) -> int:
    """Parameterized stats heuristic with configurable weights."""
    valid = state.valid_mask_np()
    team, action_type = DRAFT_ORDER[state.step]

    t0_heroes = [HEROES[i] for i in range(NUM_HEROES) if state.team0_picks[i] > 0]
    t1_heroes = [HEROES[i] for i in range(NUM_HEROES) if state.team1_picks[i] > 0]
    tier = state.skill_tier
    game_map = state.game_map

    if action_type == 'ban':
        return _ban_weighted(valid, team, t0_heroes, t1_heroes, tier, game_map,
                             hero_wr, hero_map_wr, synergies, counters, weights)

    our_picks = t0_heroes if team == 0 else t1_heroes
    enemy_picks = t1_heroes if team == 0 else t0_heroes

    best_hero = -1
    best_score = -999.0

    w_wr = weights.get('hero_wr', 1.0)
    w_counter = weights.get('counter', 1.0)
    w_synergy = weights.get('synergy', 1.0)
    w_map_bonus = weights.get('map_bonus', 0.0)  # extra weight for map-specific WR
    min_games_counter = weights.get('min_games_counter', 30)
    min_games_synergy = weights.get('min_games_synergy', 30)
    use_confidence = weights.get('use_confidence', False)
    wr_power = weights.get('wr_power', 1.0)  # exponentiate WR delta

    tier_synergies = synergies.get(tier, {})
    tier_counters = counters.get(tier, {})

    for i in range(NUM_HEROES):
        if valid[i] <= 0:
            continue

        hero = HEROES[i]
        score = 0.0

        # Hero base WR
        wr = get_hero_wr(hero, tier, game_map, hero_wr, hero_map_wr)
        # Check if map-specific data is available for bonus
        map_data = hero_map_wr.get(tier, {}).get(game_map, {}).get(hero)
        is_map_specific = map_data and map_data["games"] >= 50
        wr_delta = wr - 50.0

        # Optionally exponentiate (preserving sign) to amplify strong heroes
        if wr_power != 1.0:
            wr_delta = np.sign(wr_delta) * abs(wr_delta) ** wr_power

        score += w_wr * wr_delta
        if is_map_specific:
            score += w_map_bonus * wr_delta

        # Counter-picks vs enemy
        hero_wr_val = wr
        counter_sum = 0.0
        counter_weight_sum = 0.0
        for enemy in enemy_picks:
            d = tier_counters.get(hero, {}).get(enemy)
            if not d or d["games"] < min_games_counter:
                continue
            enemy_wr_val = get_hero_wr(enemy, tier, game_map, hero_wr, hero_map_wr)
            expected = hero_wr_val + (100 - enemy_wr_val) - 50
            delta = d["win_rate"] - expected
            if use_confidence:
                # Weight by sqrt(games) for confidence scaling
                conf = min(1.0, np.sqrt(d["games"] / 200.0))
                counter_sum += delta * conf
                counter_weight_sum += conf
            else:
                counter_sum += delta
                counter_weight_sum += 1.0
        if counter_weight_sum > 0:
            score += w_counter * (counter_sum / counter_weight_sum)

        # Synergies with allies
        syn_sum = 0.0
        syn_weight_sum = 0.0
        for ally in our_picks:
            d = tier_synergies.get(hero, {}).get(ally)
            if not d or d["games"] < min_games_synergy:
                continue
            ally_wr_val = get_hero_wr(ally, tier, game_map, hero_wr, hero_map_wr)
            expected = 50 + (hero_wr_val - 50) + (ally_wr_val - 50)
            delta = d["win_rate"] - expected
            if use_confidence:
                conf = min(1.0, np.sqrt(d["games"] / 200.0))
                syn_sum += delta * conf
                syn_weight_sum += conf
            else:
                syn_sum += delta
                syn_weight_sum += 1.0
        if syn_weight_sum > 0:
            score += w_synergy * (syn_sum / syn_weight_sum)

        if score > best_score:
            best_score = score
            best_hero = i

    return best_hero


def _ban_weighted(valid, team, t0_heroes, t1_heroes, tier, game_map,
                  hero_wr, hero_map_wr, synergies, counters, weights):
    """Ban scoring with weights."""
    our_picks = t0_heroes if team == 0 else t1_heroes
    opp_picks = t1_heroes if team == 0 else t0_heroes
    w_ban_wr = weights.get('ban_wr', 1.0)
    w_ban_counter = weights.get('ban_counter', 1.0)
    w_ban_synergy = weights.get('ban_synergy', 1.0)

    tier_counters = counters.get(tier, {})
    tier_synergies = synergies.get(tier, {})

    best_hero = -1
    best_score = -999.0

    for i in range(NUM_HEROES):
        if valid[i] <= 0:
            continue
        hero = HEROES[i]
        wr = get_hero_wr(hero, tier, game_map, hero_wr, hero_map_wr)
        score = w_ban_wr * (wr - 50.0)

        for ally in our_picks:
            d = tier_counters.get(hero, {}).get(ally)
            if not d or d["games"] < 30:
                continue
            ally_wr = get_hero_wr(ally, tier, game_map, hero_wr, hero_map_wr)
            expected = wr + (100 - ally_wr) - 50
            if d["win_rate"] >= expected + 3:
                score += w_ban_counter * (d["win_rate"] - expected)

        for enemy in opp_picks:
            d = tier_synergies.get(hero, {}).get(enemy)
            if not d or d["games"] < 30:
                continue
            enemy_wr = get_hero_wr(enemy, tier, game_map, hero_wr, hero_map_wr)
            expected = 50 + (wr - 50) + (enemy_wr - 50)
            delta = d["win_rate"] - expected
            if delta >= 2:
                score += w_ban_synergy * delta

        if score > best_score:
            best_score = score
            best_hero = i

    return best_hero


def simulate_draft(weights, wp_model, gd_models, device,
                   game_map, skill_tier,
                   hero_wr, hero_map_wr, synergies, counters) -> float:
    state = DraftState(game_map, skill_tier)
    gd_temp = random.choice([0.8, 1.0, 1.2])

    while not state.is_terminal():
        team, action_type = DRAFT_ORDER[state.step]
        if team == 0:
            action = stats_pick_weighted(state, hero_wr, hero_map_wr, synergies, counters, weights)
            state.apply_action(action, team, action_type)
        else:
            action = gd_pick(state, gd_models, device, temperature=gd_temp)
            state.apply_action(action, team, action_type)

    return _evaluate_wp(wp_model, state, device)


def evaluate_weights(weights, wp_model, gd_models, device,
                     hero_wr, hero_map_wr, synergies, counters,
                     test_configs, label=""):
    wps = []
    for game_map, tier in test_configs:
        wp = simulate_draft(weights, wp_model, gd_models, device,
                           game_map, tier, hero_wr, hero_map_wr, synergies, counters)
        wps.append(wp)
    avg = np.mean(wps)
    std = np.std(wps)
    wr = np.mean([1 if w > 0.5 else 0 for w in wps])
    if label:
        print(f"  {label}: avg_wp={avg:.4f} +/- {std:.4f} win_rate={wr:.1%}")
    return avg, std, wr


def main():
    device = torch.device("cpu")
    print("Loading models and data...")
    wp_model, gd_models = load_pretrained_models(device)
    hero_wr, hero_map_wr, synergies, counters = load_stats_data()

    NUM_DRAFTS = 300
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    test_configs = [(random.choice(MAPS), random.choice(SKILL_TIERS)) for _ in range(NUM_DRAFTS)]

    # Baseline: current weights (all 1.0, no extras)
    baseline = {
        'hero_wr': 1.0, 'counter': 1.0, 'synergy': 1.0,
        'map_bonus': 0.0, 'min_games_counter': 30, 'min_games_synergy': 30,
        'use_confidence': False, 'wr_power': 1.0,
        'ban_wr': 1.0, 'ban_counter': 1.0, 'ban_synergy': 1.0,
    }
    print(f"\n{'='*70}")
    print("BASELINE")
    b_avg, _, _ = evaluate_weights(baseline, wp_model, gd_models, device,
                                    hero_wr, hero_map_wr, synergies, counters,
                                    test_configs, "baseline")

    # ── Phase 1: Individual factor scaling ──
    print(f"\n{'='*70}")
    print("PHASE 1: Individual factor scaling")
    results = {}

    experiments = {
        # Hero WR weight
        'hero_wr=0.5': {**baseline, 'hero_wr': 0.5},
        'hero_wr=1.5': {**baseline, 'hero_wr': 1.5},
        'hero_wr=2.0': {**baseline, 'hero_wr': 2.0},
        # Counter weight
        'counter=0.5': {**baseline, 'counter': 0.5},
        'counter=1.5': {**baseline, 'counter': 1.5},
        'counter=2.0': {**baseline, 'counter': 2.0},
        'counter=3.0': {**baseline, 'counter': 3.0},
        # Synergy weight
        'synergy=0.5': {**baseline, 'synergy': 0.5},
        'synergy=1.5': {**baseline, 'synergy': 1.5},
        'synergy=2.0': {**baseline, 'synergy': 2.0},
        'synergy=3.0': {**baseline, 'synergy': 3.0},
        # Map bonus
        'map_bonus=0.5': {**baseline, 'map_bonus': 0.5},
        'map_bonus=1.0': {**baseline, 'map_bonus': 1.0},
        'map_bonus=2.0': {**baseline, 'map_bonus': 2.0},
        # Confidence weighting
        'confidence=True': {**baseline, 'use_confidence': True},
        # Min games threshold
        'min_games=10': {**baseline, 'min_games_counter': 10, 'min_games_synergy': 10},
        'min_games=50': {**baseline, 'min_games_counter': 50, 'min_games_synergy': 50},
        'min_games=100': {**baseline, 'min_games_counter': 100, 'min_games_synergy': 100},
        # WR power (amplify strong heroes)
        'wr_power=1.5': {**baseline, 'wr_power': 1.5},
        'wr_power=0.7': {**baseline, 'wr_power': 0.7},
        # Ban weights
        'ban_wr=1.5': {**baseline, 'ban_wr': 1.5},
        'ban_counter=2.0': {**baseline, 'ban_counter': 2.0},
        'ban_synergy=2.0': {**baseline, 'ban_synergy': 2.0},
    }

    for name, weights in experiments.items():
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        avg, std, wr = evaluate_weights(weights, wp_model, gd_models, device,
                                         hero_wr, hero_map_wr, synergies, counters,
                                         test_configs, name)
        results[name] = (avg, std, wr, weights)

    # Sort by avg WP
    print(f"\n{'='*70}")
    print("PHASE 1 RESULTS (sorted by avg WP)")
    print(f"{'Name':<25} {'Avg WP':>8} {'WinRate':>8} {'vs Baseline':>12}")
    print(f"{'-'*53}")
    print(f"{'baseline':<25} {b_avg:>8.4f} {'':>8} {'':>12}")
    for name, (avg, std, wr, _) in sorted(results.items(), key=lambda x: -x[1][0]):
        diff = avg - b_avg
        print(f"{name:<25} {avg:>8.4f} {wr:>7.1%} {diff:>+11.4f}")

    # ── Phase 2: Combine best individual tweaks ──
    print(f"\n{'='*70}")
    print("PHASE 2: Combining top improvements")

    # Find top 5 improvements
    top = sorted(results.items(), key=lambda x: -x[1][0])[:5]
    print(f"Top 5: {[t[0] for t in top]}")

    # Build combined weights from the top improvements
    combined = dict(baseline)
    for name, (avg, std, wr, w) in top:
        if avg > b_avg:
            for k, v in w.items():
                if v != baseline[k]:
                    combined[k] = v

    print(f"Combined config: { {k: v for k, v in combined.items() if v != baseline.get(k)} }")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    c_avg, c_std, c_wr = evaluate_weights(combined, wp_model, gd_models, device,
                                           hero_wr, hero_map_wr, synergies, counters,
                                           test_configs, "combined")
    print(f"  Combined vs baseline: {c_avg - b_avg:+.4f}")

    # ── Phase 3: Fine-tune the combined config ──
    print(f"\n{'='*70}")
    print("PHASE 3: Fine-tuning combined config")

    best_config = dict(combined)
    best_avg = c_avg

    # Tweak each param up/down by 20%
    for param in ['hero_wr', 'counter', 'synergy', 'map_bonus', 'ban_wr', 'ban_counter', 'ban_synergy']:
        val = best_config[param]
        if val == 0:
            candidates = [0.3, 0.5, 1.0]
        else:
            candidates = [val * 0.8, val * 1.2, val * 1.5]
        for c in candidates:
            test = dict(best_config)
            test[param] = round(c, 2)
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            avg, _, _ = evaluate_weights(test, wp_model, gd_models, device,
                                         hero_wr, hero_map_wr, synergies, counters,
                                         test_configs, f"{param}={c:.2f}")
            if avg > best_avg:
                best_avg = avg
                best_config = test
                print(f"  ** New best: {param}={c:.2f} → {avg:.4f}")

    print(f"\n{'='*70}")
    print("FINAL BEST CONFIG")
    print(f"  avg_wp={best_avg:.4f} (baseline was {b_avg:.4f}, delta={best_avg-b_avg:+.4f})")
    for k, v in best_config.items():
        marker = " ← changed" if v != baseline.get(k) else ""
        print(f"  {k}: {v}{marker}")

    # Final validation with more drafts
    print(f"\n{'='*70}")
    print("VALIDATION (500 drafts)")
    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    val_configs = [(random.choice(MAPS), random.choice(SKILL_TIERS)) for _ in range(500)]

    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    evaluate_weights(baseline, wp_model, gd_models, device,
                     hero_wr, hero_map_wr, synergies, counters,
                     val_configs, "baseline (validation)")

    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    evaluate_weights(best_config, wp_model, gd_models, device,
                     hero_wr, hero_map_wr, synergies, counters,
                     val_configs, "optimized (validation)")


if __name__ == "__main__":
    main()
