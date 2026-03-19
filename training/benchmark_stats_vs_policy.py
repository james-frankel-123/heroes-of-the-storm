"""
Benchmark: Stats Mode heuristic vs Draft Policy vs Generic Draft baseline.

All agents play as team 0 against Generic Draft opponent pool (team 1).
Win probability evaluated by the WP model at the end of each draft.

The Stats Mode heuristic mirrors the TypeScript engine: for each hero,
compute heroWR - 50 + avg(counter deltas) + avg(synergy deltas), pick top.
For bans: heroWR - 50 + counter strength vs our picks.

Usage:
    set -a && source .env && set +a
    python3 training/benchmark_stats_vs_policy.py [--drafts 200]
"""
import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS,
    map_to_one_hot, tier_to_one_hot,
)
from train_draft_policy import (
    AlphaZeroDraftNet, DraftState, DRAFT_ORDER, _evaluate_wp,
    load_pretrained_models, STATE_DIM,
)
from train_generic_draft import GenericDraftModel


# ── Load stats data from DB ────────────────────────────────────────

def load_stats_data():
    """Load hero win rates, synergies, and counters from the DB."""
    import psycopg2

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        # Try reading from .env file
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            for line in open(env_path):
                if line.startswith("DATABASE_URL="):
                    db_url = line.split("=", 1)[1].strip().strip('"')
                    break
    if not db_url:
        raise ValueError("DATABASE_URL not set")

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # Hero win rates by tier
    # {tier: {hero: {"win_rate": float, "games": int}}}
    hero_wr = {}
    cur.execute("SELECT hero, win_rate, games, skill_tier FROM hero_stats_aggregate")
    for hero, wr, games, tier in cur.fetchall():
        hero_wr.setdefault(tier, {})[hero] = {"win_rate": wr, "games": games}

    # Hero-map win rates by tier
    # {tier: {map: {hero: {"win_rate": float, "games": int}}}}
    hero_map_wr = {}
    cur.execute("SELECT hero, map, win_rate, games, skill_tier FROM hero_map_stats_aggregate")
    for hero, map_name, wr, games, tier in cur.fetchall():
        hero_map_wr.setdefault(tier, {}).setdefault(map_name, {})[hero] = {
            "win_rate": wr, "games": games,
        }

    # Pairwise stats by tier
    # synergies: {tier: {heroA: {heroB: {"win_rate": float, "games": int}}}}
    # counters:  {tier: {heroA: {heroB: {"win_rate": float, "games": int}}}}
    synergies = {}
    counters = {}
    cur.execute("SELECT hero_a, hero_b, relationship, win_rate, games, skill_tier FROM hero_pairwise_stats")
    for ha, hb, rel, wr, games, tier in cur.fetchall():
        if rel == "with":
            synergies.setdefault(tier, {}).setdefault(ha, {})[hb] = {"win_rate": wr, "games": games}
        else:
            counters.setdefault(tier, {}).setdefault(ha, {})[hb] = {"win_rate": wr, "games": games}

    cur.close()
    conn.close()

    return hero_wr, hero_map_wr, synergies, counters


def get_hero_wr(hero, tier, game_map, hero_wr, hero_map_wr):
    """Get hero win rate, preferring map-specific data."""
    map_data = hero_map_wr.get(tier, {}).get(game_map, {}).get(hero)
    if map_data and map_data["games"] >= 50:
        return map_data["win_rate"]
    tier_data = hero_wr.get(tier, {}).get(hero)
    if tier_data:
        return tier_data["win_rate"]
    return 50.0


# ── Stats Mode heuristic strategy ──────────────────────────────────

def stats_pick(
    state: DraftState,
    hero_wr, hero_map_wr, synergies, counters,
) -> int:
    """Pick using the Stats Mode heuristic (mirrors the TypeScript engine)."""
    valid = state.valid_mask_np()
    team, action_type = DRAFT_ORDER[state.step]

    # Reconstruct picks for each team
    t0_heroes = [HEROES[i] for i in range(NUM_HEROES) if state.team0_picks[i] > 0]
    t1_heroes = [HEROES[i] for i in range(NUM_HEROES) if state.team1_picks[i] > 0]
    tier = state.skill_tier
    game_map = state.game_map

    if action_type == 'ban':
        return _stats_ban(valid, team, t0_heroes, t1_heroes, tier, game_map,
                          hero_wr, hero_map_wr, synergies, counters)
    else:
        our_picks = t0_heroes if team == 0 else t1_heroes
        enemy_picks = t1_heroes if team == 0 else t0_heroes
        return _stats_our_pick(valid, our_picks, enemy_picks, tier, game_map,
                               hero_wr, hero_map_wr, synergies, counters)


def _stats_our_pick(valid, our_picks, enemy_picks, tier, game_map,
                    hero_wr, hero_map_wr, synergies, counters):
    """Score each hero by heroWR + counters + synergies (like the TS engine)."""
    best_hero = -1
    best_score = -999.0

    tier_synergies = synergies.get(tier, {})
    tier_counters = counters.get(tier, {})

    for i in range(NUM_HEROES):
        if valid[i] <= 0:
            continue

        hero = HEROES[i]
        score = 0.0

        # 1. Hero base WR
        wr = get_hero_wr(hero, tier, game_map, hero_wr, hero_map_wr)
        score += wr - 50.0

        # 2. Counter-picks vs enemy
        hero_wr_val = wr
        counter_sum = 0.0
        counter_count = 0
        for enemy in enemy_picks:
            d = tier_counters.get(hero, {}).get(enemy)
            if not d or d["games"] < 30:
                continue
            enemy_wr_val = get_hero_wr(enemy, tier, game_map, hero_wr, hero_map_wr)
            expected = hero_wr_val + (100 - enemy_wr_val) - 50
            delta = d["win_rate"] - expected
            counter_sum += delta
            counter_count += 1
        if counter_count > 0:
            score += counter_sum / counter_count

        # 3. Synergies with allies
        syn_sum = 0.0
        syn_count = 0
        for ally in our_picks:
            d = tier_synergies.get(hero, {}).get(ally)
            if not d or d["games"] < 30:
                continue
            ally_wr_val = get_hero_wr(ally, tier, game_map, hero_wr, hero_map_wr)
            expected = 50 + (hero_wr_val - 50) + (ally_wr_val - 50)
            delta = d["win_rate"] - expected
            syn_sum += delta
            syn_count += 1
        if syn_count > 0:
            score += syn_sum / syn_count

        if score > best_score:
            best_score = score
            best_hero = i

    return best_hero


def _stats_ban(valid, team, t0_heroes, t1_heroes, tier, game_map,
               hero_wr, hero_map_wr, synergies, counters):
    """Score ban candidates: high WR + strong vs our picks."""
    our_picks = t0_heroes if team == 0 else t1_heroes
    opp_picks = t1_heroes if team == 0 else t0_heroes

    tier_counters = counters.get(tier, {})
    tier_synergies = synergies.get(tier, {})

    best_hero = -1
    best_score = -999.0

    for i in range(NUM_HEROES):
        if valid[i] <= 0:
            continue

        hero = HEROES[i]
        wr = get_hero_wr(hero, tier, game_map, hero_wr, hero_map_wr)
        score = wr - 50.0  # High WR = good ban target

        # Strong against our picks → ban to protect
        for ally in our_picks:
            d = tier_counters.get(hero, {}).get(ally)
            if not d or d["games"] < 30:
                continue
            ally_wr = get_hero_wr(ally, tier, game_map, hero_wr, hero_map_wr)
            expected = wr + (100 - ally_wr) - 50
            if d["win_rate"] >= expected + 3:
                score += d["win_rate"] - expected

        # Synergizes with opponent → ban to deny
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


# ── Other strategies (from benchmark_draft.py) ─────────────────────

def gd_pick(state: DraftState, gd_models, device, temperature=1.0) -> int:
    gd = random.choice(gd_models)
    x = state.to_tensor_gd(device)  # 289 dims for GD model
    mask = state.valid_mask(device)
    with torch.no_grad():
        logits = gd(x, mask)
        probs = F.softmax(logits / temperature, dim=1)
        return torch.multinomial(probs, 1).item()


def policy_pick(state: DraftState, network, device) -> int:
    """Pick using just the policy network head (argmax), no MCTS."""
    priors, _ = network.predict(state, device)
    valid = state.valid_mask_np()
    priors = priors * valid
    return int(np.argmax(priors))


# ── Draft simulation ───────────────────────────────────────────────

def simulate_draft(
    team0_strategy: str,
    network, wp_model, gd_models, device,
    game_map: str, skill_tier: str,
    hero_wr=None, hero_map_wr=None, synergies_data=None, counters_data=None,
) -> float:
    """
    Simulate a full draft. team0 uses the given strategy; team1 always uses GD.
    Returns win probability for team 0.
    """
    state = DraftState(game_map, skill_tier)
    gd_temp = random.choice([0.8, 1.0, 1.2])

    while not state.is_terminal():
        team, action_type = DRAFT_ORDER[state.step]

        if team == 0:
            if team0_strategy == "policy":
                action = policy_pick(state, network, device)
            elif team0_strategy == "stats":
                action = stats_pick(state, hero_wr, hero_map_wr, synergies_data, counters_data)
            elif team0_strategy == "gd":
                action = gd_pick(state, gd_models, device, temperature=1.0)
            else:
                raise ValueError(f"Unknown: {team0_strategy}")
            state.apply_action(action, team, action_type)
        else:
            action = gd_pick(state, gd_models, device, temperature=gd_temp)
            state.apply_action(action, team, action_type)

    return _evaluate_wp(wp_model, state, device)


def run_benchmark(strategies, num_drafts):
    device = torch.device("cpu")
    print(f"Device: {device}")

    print("Loading models...")
    wp_model, gd_models = load_pretrained_models(device)

    network = AlphaZeroDraftNet().to(device)
    policy_path = os.path.join(os.path.dirname(__file__), "draft_policy.pt")
    if os.path.exists(policy_path):
        network.load_state_dict(torch.load(policy_path, weights_only=True, map_location=device))
        network.eval()
        print("Loaded draft_policy.pt")

    hero_wr, hero_map_wr, synergies_data, counters_data = None, None, None, None
    if "stats" in strategies:
        print("Loading stats data from DB...")
        hero_wr, hero_map_wr, synergies_data, counters_data = load_stats_data()
        print(f"  Hero WR tiers: {list(hero_wr.keys())}")
        n_syn = sum(len(v) for t in synergies_data.values() for v in t.values())
        n_cnt = sum(len(v) for t in counters_data.values() for v in t.values())
        print(f"  Synergies: {n_syn}, Counters: {n_cnt}")

    # Use same map/tier configs for fair comparison
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    test_configs = [(random.choice(MAPS), random.choice(SKILL_TIERS)) for _ in range(num_drafts)]

    print(f"\nBenchmarking {num_drafts} drafts per strategy")
    print(f"Strategies: {strategies}")
    print(f"{'='*70}")

    results = {}
    for strategy in strategies:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        print(f"\n--- {strategy} ---")
        wps = []
        for i, (game_map, tier) in enumerate(test_configs):
            wp = simulate_draft(
                strategy, network, wp_model, gd_models, device,
                game_map, tier,
                hero_wr, hero_map_wr, synergies_data, counters_data,
            )
            wps.append(wp)
            if (i + 1) % 50 == 0 or i == num_drafts - 1:
                print(f"  [{i+1}/{num_drafts}] avg_wp={np.mean(wps):.4f} +/- {np.std(wps):.4f}")

        results[strategy] = {
            "mean": np.mean(wps),
            "std": np.std(wps),
            "median": np.median(wps),
            "win_rate": np.mean([1 if wp > 0.5 else 0 for wp in wps]),
        }

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Strategy':<15} {'Avg WP':>8} {'StdDev':>8} {'Median':>8} {'WinRate':>8}")
    print(f"{'-'*47}")
    for strategy in strategies:
        r = results[strategy]
        print(f"{strategy:<15} {r['mean']:>8.4f} {r['std']:>8.4f} {r['median']:>8.4f} {r['win_rate']:>7.1%}")

    if "policy" in results and "stats" in results:
        diff = results["policy"]["mean"] - results["stats"]["mean"]
        print(f"\nPolicy vs Stats: {diff:+.4f} WP ({diff*100:+.1f} percentage points)")
    if "stats" in results and "gd" in results:
        diff = results["stats"]["mean"] - results["gd"]["mean"]
        print(f"Stats vs GD:     {diff:+.4f} WP ({diff*100:+.1f} percentage points)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", type=int, default=200)
    parser.add_argument("--strategies", nargs="+", default=["policy", "stats", "gd"])
    args = parser.parse_args()

    run_benchmark(args.strategies, args.drafts)
