"""
Experiment: optimize composition scoring + role heuristics.

Tests:
- Composition weight scaling
- No-healer/no-tank penalties
- Role heuristic bonuses
- Scaling factor curves

Usage:
    set -a && source .env && set +a
    python3 -u training/experiment_composition.py
"""
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from shared import NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS
from train_draft_policy import (
    DraftState, DRAFT_ORDER, _evaluate_wp, load_pretrained_models,
)
from benchmark_stats_vs_policy import (
    load_stats_data, get_hero_wr, gd_pick,
)

# Load hero roles
HERO_ROLES = {}
roles_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'lib', 'data', 'hero-roles.ts')
with open(roles_path) as f:
    for line in f:
        line = line.strip()
        if "'" in line and ':' in line and '//' not in line[:5]:
            parts = line.split("'")
            if len(parts) >= 4:
                hero = parts[1]
                role = parts[3]
                HERO_ROLES[hero] = role

# Load composition data
comp_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'lib', 'data', 'compositions.json')
with open(comp_path) as f:
    COMP_DATA = json.load(f)


def get_team_roles(state, team):
    """Get roles of heroes picked by a team."""
    picks = state.team0_picks if team == 0 else state.team1_picks
    roles = []
    for i in range(NUM_HEROES):
        if picks[i] > 0:
            role = HERO_ROLES.get(HEROES[i])
            if role:
                roles.append(role)
    return roles


def stats_pick_with_comp(
    state: DraftState,
    hero_wr, hero_map_wr, synergies, counters,
    weights: dict,
) -> int:
    valid = state.valid_mask_np()
    team, action_type = DRAFT_ORDER[state.step]

    if action_type == 'ban':
        # Use same ban logic as baseline
        return _ban_pick(state, valid, team, hero_wr, hero_map_wr, synergies, counters, weights)

    our_picks_roles = get_team_roles(state, team)
    num_picks = len(our_picks_roles)
    enemy_picks_heroes = [HEROES[i] for i in range(NUM_HEROES)
                          if (state.team1_picks if team == 0 else state.team0_picks)[i] > 0]
    our_picks_heroes = [HEROES[i] for i in range(NUM_HEROES)
                        if (state.team0_picks if team == 0 else state.team1_picks)[i] > 0]

    tier = state.skill_tier
    game_map = state.game_map

    w_wr = weights.get('hero_wr', 1.8)
    w_counter = weights.get('counter', 0.8)
    w_synergy = weights.get('synergy', 1.2)
    w_map_bonus = weights.get('map_bonus', 0.75)
    w_comp = weights.get('comp', 1.0)
    w_no_healer = weights.get('no_healer_penalty', 0.0)
    w_no_tank = weights.get('no_tank_penalty', 0.0)
    w_healer_bonus = weights.get('healer_bonus', 0.0)
    w_tank_bonus = weights.get('tank_bonus', 0.0)
    comp_scale_power = weights.get('comp_scale_power', 1.0)

    tier_synergies = synergies.get(tier, {})
    tier_counters = counters.get(tier, {})
    tier_comps = COMP_DATA.get(tier, [])

    has_healer = 'Healer' in our_picks_roles
    has_tank = 'Tank' in our_picks_roles
    picks_remaining = 5 - num_picks - 1  # after this pick

    best_hero = -1
    best_score = -999.0

    for i in range(NUM_HEROES):
        if valid[i] <= 0:
            continue

        hero = HEROES[i]
        hero_role = HERO_ROLES.get(hero)
        score = 0.0

        # Hero base WR
        wr = get_hero_wr(hero, tier, game_map, hero_wr, hero_map_wr)
        map_data = hero_map_wr.get(tier, {}).get(game_map, {}).get(hero)
        is_map = map_data and map_data["games"] >= 50
        wr_delta = wr - 50.0
        score += w_wr * wr_delta
        if is_map:
            score += w_map_bonus * wr_delta

        # Counter-picks
        counter_sum = 0.0
        counter_count = 0
        for enemy in enemy_picks_heroes:
            d = tier_counters.get(hero, {}).get(enemy)
            if not d or d["games"] < 30:
                continue
            enemy_wr_val = get_hero_wr(enemy, tier, game_map, hero_wr, hero_map_wr)
            expected = wr + (100 - enemy_wr_val) - 50
            counter_sum += d["win_rate"] - expected
            counter_count += 1
        if counter_count > 0:
            score += w_counter * (counter_sum / counter_count)

        # Synergies
        syn_sum = 0.0
        syn_count = 0
        for ally in our_picks_heroes:
            d = tier_synergies.get(hero, {}).get(ally)
            if not d or d["games"] < 30:
                continue
            ally_wr_val = get_hero_wr(ally, tier, game_map, hero_wr, hero_map_wr)
            expected = 50 + (wr - 50) + (ally_wr_val - 50)
            syn_sum += d["win_rate"] - expected
            syn_count += 1
        if syn_count > 0:
            score += w_synergy * (syn_sum / syn_count)

        # Composition scoring
        if hero_role and tier_comps:
            combined = sorted(our_picks_roles + [hero_role])
            achievable = [c for c in tier_comps if c["games"] >= 100
                         and is_subset(combined, c["roles"])]

            # Scale factor with configurable power
            scale = min(num_picks / 4, 1) ** comp_scale_power

            if achievable:
                best_comp_wr = max(conf_adj_wr(c["winRate"], c["games"]) for c in achievable)
                baseline_wr = 50.0  # simplified
                comp_delta = (best_comp_wr - baseline_wr) * scale
                score += w_comp * comp_delta
            else:
                # No achievable comp penalty
                score += w_comp * (-15 * scale)

        # Role diversity heuristic (subjective game knowledge)
        w_role_div = weights.get('role_diversity', 0.0)
        if hero_role and w_role_div > 0:
            roles_after = our_picks_roles + [hero_role]
            role_counts = {}
            for r in roles_after:
                role_counts[r] = role_counts.get(r, 0) + 1

            div_score = 0.0

            # Reward: having exactly 1 healer (essential)
            if role_counts.get('Healer', 0) == 1:
                div_score += 2.0
            elif role_counts.get('Healer', 0) == 0 and num_picks >= 2:
                div_score -= 1.0  # no healer yet and we should have one

            # Reward: having exactly 1 tank
            if role_counts.get('Tank', 0) == 1:
                div_score += 1.5
            elif role_counts.get('Tank', 0) == 0 and num_picks >= 2:
                div_score -= 0.5

            # Penalize: 3+ of same role (except maybe Ranged Assassin)
            for role, count in role_counts.items():
                if count >= 3 and role != 'Ranged Assassin':
                    div_score -= 2.0
                elif count >= 3:
                    div_score -= 1.0  # 3 ranged is bad but not as bad

            # Reward: at least 1 melee damage dealer (Bruiser or Melee Assassin)
            has_melee_dmg = role_counts.get('Bruiser', 0) + role_counts.get('Melee Assassin', 0) > 0
            if has_melee_dmg:
                div_score += 0.5

            # Penalize: double support (Support != Healer, e.g. Medivh, Zarya)
            if role_counts.get('Support', 0) >= 2:
                div_score -= 1.5

            # Penalize: no ranged damage at all
            if role_counts.get('Ranged Assassin', 0) == 0 and num_picks >= 3:
                div_score -= 1.0

            score += w_role_div * div_score

        # Role heuristics
        if hero_role:
            # Healer bonus: if we don't have a healer and picks are running out
            if hero_role == 'Healer' and not has_healer:
                urgency = max(0, 1 - picks_remaining / 3)  # 0→0, 1→0.33, 2→0.67, 3→1
                score += w_healer_bonus * urgency
            # Tank bonus
            if hero_role == 'Tank' and not has_tank:
                urgency = max(0, 1 - picks_remaining / 3)
                score += w_tank_bonus * urgency
            # No healer penalty: if this hero ISN'T a healer and we still need one
            if hero_role != 'Healer' and not has_healer and picks_remaining <= 1:
                score += w_no_healer
            if hero_role != 'Tank' and not has_tank and picks_remaining <= 1:
                score += w_no_tank

        if score > best_score:
            best_score = score
            best_hero = i

    return best_hero


def _ban_pick(state, valid, team, hero_wr, hero_map_wr, synergies, counters, weights):
    """Ban with optimized weights."""
    t0_heroes = [HEROES[i] for i in range(NUM_HEROES) if state.team0_picks[i] > 0]
    t1_heroes = [HEROES[i] for i in range(NUM_HEROES) if state.team1_picks[i] > 0]
    our_picks = t0_heroes if team == 0 else t1_heroes
    opp_picks = t1_heroes if team == 0 else t0_heroes
    tier = state.skill_tier
    game_map = state.game_map
    tier_counters = counters.get(tier, {})
    tier_synergies = synergies.get(tier, {})

    best_hero = -1
    best_score = -999.0
    for i in range(NUM_HEROES):
        if valid[i] <= 0:
            continue
        hero = HEROES[i]
        wr = get_hero_wr(hero, tier, game_map, hero_wr, hero_map_wr)
        score = 1.5 * (wr - 50.0)
        for ally in our_picks:
            d = tier_counters.get(hero, {}).get(ally)
            if not d or d["games"] < 30:
                continue
            ally_wr = get_hero_wr(ally, tier, game_map, hero_wr, hero_map_wr)
            expected = wr + (100 - ally_wr) - 50
            if d["win_rate"] >= expected + 3:
                score += 0.8 * (d["win_rate"] - expected)
        for enemy in opp_picks:
            d = tier_synergies.get(hero, {}).get(enemy)
            if not d or d["games"] < 30:
                continue
            enemy_wr = get_hero_wr(enemy, tier, game_map, hero_wr, hero_map_wr)
            expected = 50 + (wr - 50) + (enemy_wr - 50)
            delta = d["win_rate"] - expected
            if delta >= 2:
                score += delta
        if score > best_score:
            best_score = score
            best_hero = i
    return best_hero


def is_subset(subset, superset):
    counts = {}
    for r in superset:
        counts[r] = counts.get(r, 0) + 1
    for r in subset:
        if counts.get(r, 0) <= 0:
            return False
        counts[r] -= 1
    return True


def conf_adj_wr(wr, games, threshold=200):
    if games >= threshold:
        return wr
    w = games / threshold
    return wr * w + 50 * (1 - w)


def simulate_draft(weights, wp_model, gd_models, device,
                   game_map, skill_tier,
                   hero_wr, hero_map_wr, synergies, counters):
    state = DraftState(game_map, skill_tier)
    gd_temp = random.choice([0.8, 1.0, 1.2])
    while not state.is_terminal():
        team, action_type = DRAFT_ORDER[state.step]
        if team == 0:
            action = stats_pick_with_comp(state, hero_wr, hero_map_wr, synergies, counters, weights)
            state.apply_action(action, team, action_type)
        else:
            action = gd_pick(state, gd_models, device, temperature=gd_temp)
            state.apply_action(action, team, action_type)
    return _evaluate_wp(wp_model, state, device)


def evaluate(weights, wp_model, gd_models, device,
             hero_wr, hero_map_wr, synergies, counters,
             configs, label=""):
    wps = []
    for m, t in configs:
        wps.append(simulate_draft(weights, wp_model, gd_models, device,
                                  m, t, hero_wr, hero_map_wr, synergies, counters))
    avg = np.mean(wps)
    wr = np.mean([1 if w > 0.5 else 0 for w in wps])
    if label:
        print(f"  {label}: avg_wp={avg:.4f} win_rate={wr:.1%}")
    return avg


def main():
    device = torch.device("cpu")
    print("Loading...")
    wp_model, gd_models = load_pretrained_models(device)
    hero_wr, hero_map_wr, synergies, counters = load_stats_data()

    N = 300
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    configs = [(random.choice(MAPS), random.choice(SKILL_TIERS)) for _ in range(N)]

    # Baseline: optimized weights from Phase 1, no comp/role tweaks
    baseline = {
        'hero_wr': 1.8, 'counter': 0.8, 'synergy': 1.2, 'map_bonus': 0.75,
        'comp': 1.0, 'no_healer_penalty': 0.0, 'no_tank_penalty': 0.0,
        'healer_bonus': 0.0, 'tank_bonus': 0.0, 'comp_scale_power': 1.0,
    }

    print(f"\n{'='*70}")
    print("BASELINE (optimized stats weights, default comp)")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    b_avg = evaluate(baseline, wp_model, gd_models, device,
                     hero_wr, hero_map_wr, synergies, counters, configs, "baseline")

    print(f"\n{'='*70}")
    print("COMPOSITION EXPERIMENTS")

    experiments = {
        # Comp weight
        'comp=0': {**baseline, 'comp': 0.0},
        'comp=0.5': {**baseline, 'comp': 0.5},
        'comp=1.5': {**baseline, 'comp': 1.5},
        'comp=2.0': {**baseline, 'comp': 2.0},
        'comp=3.0': {**baseline, 'comp': 3.0},

        # Scale power (how fast comp importance ramps up)
        'scale_pow=0.5': {**baseline, 'comp_scale_power': 0.5},  # faster ramp
        'scale_pow=2.0': {**baseline, 'comp_scale_power': 2.0},  # slower ramp

        # Healer bonus (urgency-scaled)
        'healer_bonus=3': {**baseline, 'healer_bonus': 3.0},
        'healer_bonus=5': {**baseline, 'healer_bonus': 5.0},
        'healer_bonus=8': {**baseline, 'healer_bonus': 8.0},
        'healer_bonus=12': {**baseline, 'healer_bonus': 12.0},
        'healer_bonus=15': {**baseline, 'healer_bonus': 15.0},

        # Tank bonus
        'tank_bonus=3': {**baseline, 'tank_bonus': 3.0},
        'tank_bonus=5': {**baseline, 'tank_bonus': 5.0},
        'tank_bonus=8': {**baseline, 'tank_bonus': 8.0},

        # No-healer penalty (applied when picking non-healer with ≤1 pick left)
        'no_healer=-5': {**baseline, 'no_healer_penalty': -5.0},
        'no_healer=-10': {**baseline, 'no_healer_penalty': -10.0},
        'no_healer=-15': {**baseline, 'no_healer_penalty': -15.0},
        'no_healer=-20': {**baseline, 'no_healer_penalty': -20.0},

        # No-tank penalty
        'no_tank=-5': {**baseline, 'no_tank_penalty': -5.0},
        'no_tank=-10': {**baseline, 'no_tank_penalty': -10.0},

        # Combined healer + tank
        'healer5+tank3': {**baseline, 'healer_bonus': 5.0, 'tank_bonus': 3.0},
        'healer8+tank5': {**baseline, 'healer_bonus': 8.0, 'tank_bonus': 5.0},
        'healer12+tank5+noH-10': {**baseline, 'healer_bonus': 12.0, 'tank_bonus': 5.0, 'no_healer_penalty': -10.0},
        'healer8+noH-15+comp2': {**baseline, 'healer_bonus': 8.0, 'no_healer_penalty': -15.0, 'comp': 2.0},

        # Heuristic role diversity (subjective game knowledge)
        'role_diverse_light': {**baseline, 'role_diversity': 3.0},
        'role_diverse_medium': {**baseline, 'role_diversity': 5.0},
        'role_diverse_heavy': {**baseline, 'role_diversity': 8.0},

        # Full kitchen sink
        'kitchen_sink': {**baseline, 'healer_bonus': 8.0, 'tank_bonus': 5.0,
                        'no_healer_penalty': -10.0, 'no_tank_penalty': -5.0,
                        'role_diversity': 5.0, 'comp': 1.5},
    }

    results = {}
    for name, w in experiments.items():
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        avg = evaluate(w, wp_model, gd_models, device,
                      hero_wr, hero_map_wr, synergies, counters, configs, name)
        results[name] = (avg, w)

    print(f"\n{'='*70}")
    print("RESULTS (sorted)")
    print(f"{'Name':<35} {'Avg WP':>8} {'vs Base':>10}")
    print(f"{'-'*53}")
    print(f"{'baseline':<35} {b_avg:>8.4f}")
    for name, (avg, _) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"{name:<35} {avg:>8.4f} {avg-b_avg:>+9.4f}")

    # Combine top improvements
    print(f"\n{'='*70}")
    print("COMBINING TOP IMPROVEMENTS")
    top = sorted(results.items(), key=lambda x: -x[1][0])[:3]
    combined = dict(baseline)
    for name, (avg, w) in top:
        if avg > b_avg:
            for k, v in w.items():
                if v != baseline.get(k):
                    combined[k] = v

    print(f"Config: { {k: v for k, v in combined.items() if v != baseline.get(k)} }")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    c_avg = evaluate(combined, wp_model, gd_models, device,
                     hero_wr, hero_map_wr, synergies, counters, configs, "combined")
    print(f"  Combined vs baseline: {c_avg - b_avg:+.4f}")

    # Validate
    print(f"\n{'='*70}")
    print("VALIDATION (500 drafts)")
    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    val = [(random.choice(MAPS), random.choice(SKILL_TIERS)) for _ in range(500)]

    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    evaluate(baseline, wp_model, gd_models, device,
             hero_wr, hero_map_wr, synergies, counters, val, "baseline")
    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    evaluate(combined, wp_model, gd_models, device,
             hero_wr, hero_map_wr, synergies, counters, val, "optimized")


if __name__ == "__main__":
    main()
