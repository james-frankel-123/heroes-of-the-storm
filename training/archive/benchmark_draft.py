"""
Benchmark: Draft Policy (MCTS) vs Greedy Heuristic vs Random
All agents play as team 0 against Generic Draft opponent pool.
Win probability is evaluated by the Win Probability model at the end of each draft.

Usage:
    python training/benchmark_draft.py [--drafts 500] [--mcts-sims 200]
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
    NUM_HEROES, HEROES, MAPS, SKILL_TIERS,
    map_to_one_hot, tier_to_one_hot,
)
from train_draft_policy import (
    AlphaZeroDraftNet, DraftState, DRAFT_ORDER, _evaluate_wp, mcts_search,
    load_pretrained_models, STATE_DIM,
)
from train_generic_draft import GenericDraftModel


def greedy_pick(state: DraftState, wp_model, gd_models, device) -> int:
    """
    Greedy heuristic: try each valid hero, simulate opponent moves with GD,
    pick the hero that maximizes expected WP at the end.
    For bans: pick the hero that minimizes opponent's best WP.
    """
    valid = state.valid_mask_np()
    valid_heroes = [i for i in range(NUM_HEROES) if valid[i] > 0]
    team, action_type = DRAFT_ORDER[state.step]

    best_hero = valid_heroes[0]
    best_score = -1.0 if action_type == 'pick' else 2.0

    for hero_idx in valid_heroes:
        # Simulate this choice
        sim = state.clone()
        sim.apply_action(hero_idx, team, action_type)

        # Roll out the rest greedily with GD for both sides
        score = _rollout_wp(sim, wp_model, gd_models, device, n_rollouts=3)

        if action_type == 'pick':
            if score > best_score:
                best_score = score
                best_hero = hero_idx
        else:
            # For bans, minimize opponent advantage
            # (We want to ban the hero that would give opponent highest WP if picked)
            # So we want the hero whose banning leads to lowest opponent WP = highest our WP
            if score > best_score:
                best_score = score
                best_hero = hero_idx

    return best_hero


def _rollout_wp(state: DraftState, wp_model, gd_models, device, n_rollouts=3) -> float:
    """Roll out remaining draft steps with GD, return average WP."""
    wps = []
    for _ in range(n_rollouts):
        sim = state.clone()
        while not sim.is_terminal():
            gd = random.choice(gd_models)
            x = sim.to_tensor_gd(device)
            mask = sim.valid_mask(device)
            with torch.no_grad():
                logits = gd(x, mask)
                probs = F.softmax(logits / 1.0, dim=1)
                action = torch.multinomial(probs, 1).item()
            team, action_type = DRAFT_ORDER[sim.step]
            sim.apply_action(action, team, action_type)
        wps.append(_evaluate_wp(wp_model, sim, device))
    return np.mean(wps)


def gd_pick(state: DraftState, gd_models, device, temperature=1.0) -> int:
    """Pick using a Generic Draft model (baseline / random-ish)."""
    gd = random.choice(gd_models)
    x = state.to_tensor_gd(device)
    mask = state.valid_mask(device)
    with torch.no_grad():
        logits = gd(x, mask)
        probs = F.softmax(logits / temperature, dim=1)
        return torch.multinomial(probs, 1).item()


def policy_pick(state: DraftState, network, wp_model, gd_models, device, num_sims=200) -> int:
    """Pick using the AlphaZero MCTS policy network."""
    visit_dist = mcts_search(
        state, network, wp_model, gd_models,
        gd_temperature=1.0, device=device, num_simulations=num_sims,
    )
    return np.argmax(visit_dist)


def policy_pick_no_mcts(state: DraftState, network, device) -> int:
    """Pick using just the policy network (no MCTS), for speed comparison."""
    priors, _ = network.predict(state, device)
    valid = state.valid_mask_np()
    priors = priors * valid
    return np.argmax(priors)


def simulate_draft(
    team0_strategy: str,
    network, wp_model, gd_models, device,
    game_map: str, skill_tier: str,
    mcts_sims: int = 200,
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
            if team0_strategy == 'policy_mcts':
                action = policy_pick(state, network, wp_model, gd_models, device, mcts_sims)
            elif team0_strategy == 'policy_raw':
                action = policy_pick_no_mcts(state, network, device)
            elif team0_strategy == 'greedy':
                action = greedy_pick(state, wp_model, gd_models, device)
            elif team0_strategy == 'gd':
                action = gd_pick(state, gd_models, device, temperature=1.0)
            else:
                raise ValueError(f"Unknown strategy: {team0_strategy}")
            state.apply_action(action, team, action_type)
        else:
            # Opponent always uses GD
            action = gd_pick(state, gd_models, device, temperature=gd_temp)
            state.apply_action(action, team, action_type)

    return _evaluate_wp(wp_model, state, device)


def run_benchmark(strategies, num_drafts, mcts_sims):
    device = torch.device("cpu")
    print(f"Device: {device}")

    print("Loading models...")
    wp_model, gd_models = load_pretrained_models(device)

    # Load policy network
    policy_path = os.path.join(os.path.dirname(__file__), "draft_policy.pt")
    network = AlphaZeroDraftNet().to(device)
    if os.path.exists(policy_path):
        network.load_state_dict(torch.load(policy_path, weights_only=True, map_location=device))
        network.eval()
        print("Loaded draft_policy.pt")
    else:
        print("WARNING: No draft_policy.pt found — policy strategies will use untrained network")

    print(f"\nBenchmarking {num_drafts} drafts per strategy, MCTS sims={mcts_sims}")
    print(f"Strategies: {strategies}")
    print(f"{'='*70}")

    results = {}
    for strategy in strategies:
        print(f"\n--- {strategy} ---")
        wps = []
        for i in range(num_drafts):
            game_map = random.choice(MAPS)
            tier = random.choice(SKILL_TIERS)
            wp = simulate_draft(strategy, network, wp_model, gd_models, device, game_map, tier, mcts_sims)
            wps.append(wp)
            if (i + 1) % 50 == 0 or i == num_drafts - 1:
                print(f"  [{i+1}/{num_drafts}] avg_wp={np.mean(wps):.4f} +/- {np.std(wps):.4f}")

        results[strategy] = {
            'mean': np.mean(wps),
            'std': np.std(wps),
            'median': np.median(wps),
            'min': np.min(wps),
            'max': np.max(wps),
            'win_rate': np.mean([1 if wp > 0.5 else 0 for wp in wps]),
            'all': wps,
        }

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Strategy':<20} {'Avg WP':>8} {'StdDev':>8} {'Median':>8} {'WinRate':>8} {'Min':>8} {'Max':>8}")
    print(f"{'-'*70}")
    for strategy in strategies:
        r = results[strategy]
        print(f"{strategy:<20} {r['mean']:>8.4f} {r['std']:>8.4f} {r['median']:>8.4f} "
              f"{r['win_rate']:>7.1%} {r['min']:>8.4f} {r['max']:>8.4f}")

    # Head-to-head comparison
    if 'policy_mcts' in results and 'greedy' in results:
        diff = results['policy_mcts']['mean'] - results['greedy']['mean']
        print(f"\nPolicy MCTS vs Greedy: {diff:+.4f} WP ({diff*100:+.1f} percentage points)")
    if 'policy_raw' in results and 'greedy' in results:
        diff = results['policy_raw']['mean'] - results['greedy']['mean']
        print(f"Policy Raw vs Greedy:  {diff:+.4f} WP ({diff*100:+.1f} percentage points)")
    if 'policy_mcts' in results and 'gd' in results:
        diff = results['policy_mcts']['mean'] - results['gd']['mean']
        print(f"Policy MCTS vs GD:     {diff:+.4f} WP ({diff*100:+.1f} percentage points)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark draft strategies")
    parser.add_argument("--drafts", type=int, default=200, help="Number of drafts per strategy")
    parser.add_argument("--mcts-sims", type=int, default=200, help="MCTS simulations per move")
    parser.add_argument("--strategies", nargs="+",
                        default=["policy_mcts", "policy_raw", "greedy", "gd"],
                        help="Strategies to benchmark")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    run_benchmark(args.strategies, args.drafts, args.mcts_sims)
