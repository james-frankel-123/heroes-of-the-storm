#!/usr/bin/env python3
"""
Draft Quality Metrics: Resilience, Counter Play, and Compositional Synergy.

Evaluates multiple strategies on temporal draft quality axes using
HeroesProfile pairwise statistics.

Usage:
    set -a && source .env && set +a
    python3 -u training/experiment_draft_quality.py --drafts 200
"""

import os, sys, json, random, argparse, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import (NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS,
                    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot, HERO_ROLE_FINE)
from sweep_enriched_wp import (StatsCache, WinProbEnrichedModel, FEATURE_GROUPS,
                                compute_group_indices, extract_features, FEATURE_GROUP_DIMS)
from train_draft_policy import AlphaZeroDraftNet, DraftState, DRAFT_ORDER
from train_generic_draft import GenericDraftModel


# ── Core metric functions ───────────────────────────────────────────

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


def draft_resilience(pick_steps, stats, tier):
    """
    For each of our picks, compute how badly the opponent's SUBSEQUENT picks counter it.
    Returns dict with avg_resilience, early/late resilience, and gradient.
    """
    our_picks = [(h, s) for h, team, s in pick_steps if team == "ours"]
    opp_picks = [(h, s) for h, team, s in pick_steps if team == "theirs"]

    exposures = []
    for our_hero, our_step in our_picks:
        subsequent_opp = [h for h, s in opp_picks if s > our_step]
        if not subsequent_opp:
            exposures.append(0.0)
            continue
        deltas = [counter_delta(opp_h, our_hero, stats, tier)
                  for opp_h in subsequent_opp]
        deltas = [d for d in deltas if d is not None]
        exposures.append(np.mean(deltas) if deltas else 0.0)

    n = len(exposures)
    return {
        "avg_resilience": -np.mean(exposures) if exposures else 0.0,
        "early_pick_resilience": -np.mean(exposures[:2]) if n >= 2 else 0.0,
        "late_pick_resilience": -np.mean(exposures[-2:]) if n >= 2 else 0.0,
        "resilience_gradient": (
            np.mean(exposures[-2:]) - np.mean(exposures[:2])
        ) if n >= 4 else 0.0,
    }


def draft_counter_quality(pick_steps, stats, tier):
    """
    For each of our picks, compute how well it counters opponent's PRIOR picks.
    """
    our_picks = [(h, s) for h, team, s in pick_steps if team == "ours"]
    opp_picks = [(h, s) for h, team, s in pick_steps if team == "theirs"]

    deltas_by_pos = []
    for our_hero, our_step in our_picks:
        prior_opp = [h for h, s in opp_picks if s < our_step]
        if not prior_opp:
            deltas_by_pos.append(0.0)
            continue
        deltas = [counter_delta(our_hero, opp_h, stats, tier) for opp_h in prior_opp]
        deltas = [d for d in deltas if d is not None]
        deltas_by_pos.append(np.mean(deltas) if deltas else 0.0)

    n = len(deltas_by_pos)
    return {
        "avg_counter": np.mean(deltas_by_pos) if deltas_by_pos else 0.0,
        "early_counter": np.mean(deltas_by_pos[:2]) if n >= 2 else 0.0,
        "late_counter": np.mean(deltas_by_pos[-2:]) if n >= 2 else 0.0,
        "counter_gradient": (
            np.mean(deltas_by_pos[-2:]) - np.mean(deltas_by_pos[:2])
        ) if n >= 4 else 0.0,
    }


def counter_opportunity_analysis(pick_steps, stats, tier, valid_heroes_at_step):
    """
    At each of our pick steps, what was the best available counter,
    and how close was the chosen hero to it?
    """
    our_picks = [(h, s) for h, team, s in pick_steps if team == "ours"]
    opp_picks = [(h, s) for h, team, s in pick_steps if team == "theirs"]

    in_top5 = 0
    delta_gaps = []
    total = 0

    for our_hero, our_step in our_picks:
        prior_opp = [h for h, s in opp_picks if s < our_step]
        if not prior_opp:
            continue

        valid = valid_heroes_at_step.get(our_step, [])
        if not valid:
            continue

        scored = []
        for candidate in valid:
            deltas = [counter_delta(candidate, opp_h, stats, tier) for opp_h in prior_opp]
            deltas = [d for d in deltas if d is not None]
            if deltas:
                scored.append((candidate, np.mean(deltas)))
        scored.sort(key=lambda x: -x[1])

        if not scored:
            continue

        best_delta = scored[0][1]
        chosen_delta = next((d for h, d in scored if h == our_hero), 0.0)
        rank = next((i for i, (h, _) in enumerate(scored) if h == our_hero), len(scored))

        if rank < 5:
            in_top5 += 1
        delta_gaps.append(best_delta - chosen_delta)
        total += 1

    return {
        "capture_rate": in_top5 / total if total > 0 else 0.0,
        "avg_delta_gap": np.mean(delta_gaps) if delta_gaps else 0.0,
    }


def incremental_synergy(pick_steps, stats, tier):
    """
    For each pick after the first, compute synergy with all prior teammates.
    """
    our_picks = [(h, s) for h, team, s in pick_steps if team == "ours"]
    our_heroes = [h for h, _ in our_picks]

    per_pick_syn = []
    teammates_so_far = []
    for our_hero, _ in our_picks:
        if not teammates_so_far:
            per_pick_syn.append(0.0)
            teammates_so_far.append(our_hero)
            continue
        deltas = [synergy_delta(our_hero, tm, stats, tier) for tm in teammates_so_far]
        deltas = [d for d in deltas if d is not None]
        per_pick_syn.append(np.mean(deltas) if deltas else 0.0)
        teammates_so_far.append(our_hero)

    # Full team synergy: all C(5,2) pairs
    all_pairs = []
    for i, h1 in enumerate(our_heroes):
        for h2 in our_heroes[i+1:]:
            d = synergy_delta(h1, h2, stats, tier)
            if d is not None:
                all_pairs.append(d)

    return {
        "team_synergy": np.mean(all_pairs) if all_pairs else 0.0,
        "avg_incremental_synergy": np.mean(per_pick_syn[1:]) if len(per_pick_syn) > 1 else 0.0,
    }


def full_draft_quality(pick_steps, stats, tier, valid_heroes_at_step):
    """Compute all three axes for one draft."""
    res = draft_resilience(pick_steps, stats, tier)
    ctr = draft_counter_quality(pick_steps, stats, tier)
    opp = counter_opportunity_analysis(pick_steps, stats, tier, valid_heroes_at_step)
    syn = incremental_synergy(pick_steps, stats, tier)

    return {
        "resilience_avg": res["avg_resilience"],
        "resilience_gradient": res["resilience_gradient"],
        "counter_avg": ctr["avg_counter"],
        "counter_late": ctr["late_counter"],
        "counter_gradient": ctr["counter_gradient"],
        "counter_capture_rate": opp["capture_rate"],
        "counter_delta_gap": opp["avg_delta_gap"],
        "team_synergy": syn["team_synergy"],
        "incremental_synergy": syn["avg_incremental_synergy"],
    }


# ── Draft simulation infrastructure ─────────────────────────────────

def simulate_draft(strategy_fn, opp_fn, game_map, tier, our_team, stats):
    """
    Simulate a full draft, tracking pick order for temporal metrics.

    strategy_fn(state, valid_mask, step_type) -> hero_idx
    opp_fn(state, valid_mask, step_type) -> hero_idx

    Returns:
        pick_steps: list of (hero_name, "ours"/"theirs", step_number) for PICKS only
        valid_heroes_at_step: dict[step] -> list of valid hero names at our pick steps
    """
    state = DraftState(game_map, tier, our_team=our_team)
    pick_steps = []
    valid_at_step = {}

    while not state.is_terminal():
        step_team, step_type = DRAFT_ORDER[state.step]
        mask = state.valid_mask_np()
        valid_heroes = [HEROES[i] for i in range(NUM_HEROES) if mask[i] > 0.5]
        step_num = state.step

        if step_team == our_team:
            if step_type == "pick":
                valid_at_step[step_num] = valid_heroes
            hero_idx = strategy_fn(state, mask, step_type, game_map, tier, our_team)
        else:
            hero_idx = opp_fn(state, mask, step_type, game_map, tier)

        hero_name = HEROES[hero_idx]
        if step_type == "pick":
            team_label = "ours" if step_team == our_team else "theirs"
            pick_steps.append((hero_name, team_label, step_num))

        state.apply_action(hero_idx, step_team, step_type)

    return pick_steps, valid_at_step


def make_state_tensor(state, game_map, tier, our_team=None):
    """Build the 289 or 290 dim state tensor from DraftState."""
    step_team, step_type = DRAFT_ORDER[state.step] if state.step < len(DRAFT_ORDER) else (0, "pick")
    s = np.concatenate([
        state.team0_picks, state.team1_picks, state.bans,
        map_to_one_hot(game_map), tier_to_one_hot(tier),
        [state.step / 15.0, 0.0 if step_type == "ban" else 1.0]
    ])
    if our_team is not None:
        s = np.concatenate([s, [float(our_team)]])
    return s


# ── Strategy factories ──────────────────────────────────────────────

def make_gd_strategy(gd_model, sample=True):
    """GD model strategy (sample or argmax)."""
    def strategy(state, mask, step_type, game_map, tier, our_team):
        s = make_state_tensor(state, game_map, tier)
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        m_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = gd_model(s_t, m_t)
            if sample:
                probs = F.softmax(logits, dim=1)
                return torch.multinomial(probs, 1).item()
            else:
                return logits.argmax(dim=1).item()
    return strategy


def make_gd_opponent(gd_model):
    """GD model as opponent (always samples)."""
    def opp(state, mask, step_type, game_map, tier):
        s = make_state_tensor(state, game_map, tier)
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        m_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = gd_model(s_t, m_t)
            probs = F.softmax(logits, dim=1)
            return torch.multinomial(probs, 1).item()
    return opp


def make_policy_strategy(net):
    """MCTS policy network strategy (argmax)."""
    def strategy(state, mask, step_type, game_map, tier, our_team):
        s = make_state_tensor(state, game_map, tier, our_team)
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        m_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = net(s_t, m_t)
            return logits.argmax(dim=1).item()
    return strategy


def make_wp_greedy_strategy(wp_model, stats, wp_cols, all_mask):
    """Greedy WP maximization strategy."""
    def strategy(state, mask, step_type, game_map, tier, our_team):
        best_idx = -1
        best_wp = -1.0

        for i in range(NUM_HEROES):
            if mask[i] < 0.5:
                continue
            # Simulate picking hero i
            hero = HEROES[i]
            t0 = [HEROES[j] for j in range(NUM_HEROES) if state.team0_picks[j] > 0.5]
            t1 = [HEROES[j] for j in range(NUM_HEROES) if state.team1_picks[j] > 0.5]

            step_team, _ = DRAFT_ORDER[state.step]
            if step_type == "ban":
                # For bans, pick the hero that minimizes opponent's best option
                # Simplified: just use GD-like behavior for bans
                if best_idx < 0:
                    best_idx = i
                continue

            if step_team == our_team:
                if our_team == 0:
                    t0_try = t0 + [hero]
                    t1_try = t1
                else:
                    t0_try = t0
                    t1_try = t1 + [hero]
            else:
                continue

            # Evaluate WP
            def _run_wp(t0h, t1h):
                d = {'team0_heroes': t0h, 'team1_heroes': t1h,
                     'game_map': game_map, 'skill_tier': tier, 'winner': 0}
                base, enriched = extract_features(d, stats, all_mask)
                x = np.concatenate([base, enriched[wp_cols]])
                with torch.no_grad():
                    return wp_model(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()

            wp_n = _run_wp(t0_try, t1_try)
            wp_s = _run_wp(t1_try, t0_try)
            wp_t0 = (wp_n + (1.0 - wp_s)) / 2.0
            wp_ours = wp_t0 if our_team == 0 else 1.0 - wp_t0

            if wp_ours > best_wp:
                best_wp = wp_ours
                best_idx = i

        # Fallback for bans or if nothing scored
        if best_idx < 0:
            for i in range(NUM_HEROES):
                if mask[i] > 0.5:
                    best_idx = i
                    break
        return best_idx
    return strategy


# ── Main evaluation ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    N = args.drafts
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Draft Quality Evaluation: {N} drafts")
    print("=" * 80)

    # Load stats
    stats = StatsCache()

    # Load WP setup
    gi = compute_group_indices()
    WP_GROUPS = ['role_counts', 'team_avg_wr', 'map_delta', 'pairwise_counters',
                 'pairwise_synergies', 'counter_detail', 'meta_strength',
                 'draft_diversity', 'comp_wr']
    wp_cols = []
    for g in WP_GROUPS:
        s, e = gi[g]
        wp_cols.extend(range(s, e))
    all_mask = [True] * len(FEATURE_GROUPS)
    enriched_dim = sum(FEATURE_GROUP_DIMS[g] for g in WP_GROUPS)
    wp_input_dim = 197 + enriched_dim

    # Load GD
    gd = GenericDraftModel()
    gd_path = os.path.join(os.path.dirname(__file__), "generic_draft_0.pt")
    gd.load_state_dict(torch.load(gd_path, weights_only=True, map_location="cpu"))
    gd.eval()
    print("GD model loaded")

    # Load WP models
    wp_enr = WinProbEnrichedModel(wp_input_dim, [256, 128], dropout=0.3)
    wp_enr.load_state_dict(torch.load(
        os.path.join(os.path.dirname(__file__), "wp_experiment_enriched.pt"),
        weights_only=True, map_location="cpu"))
    wp_enr.eval()

    wp_aug = WinProbEnrichedModel(wp_input_dim, [512, 256, 128], dropout=0.3)
    wp_aug.load_state_dict(torch.load(
        os.path.join(os.path.dirname(__file__), "wp_enriched_winner.pt"),
        weights_only=True, map_location="cpu"))
    wp_aug.eval()
    print("WP models loaded")

    # Load MCTS policies (remap old res_block1/2/3 → res_blocks.0/1/2)
    def load_policy(path):
        net = AlphaZeroDraftNet()
        sd = torch.load(path, weights_only=True, map_location="cpu")
        # Remap old naming if needed
        if any(k.startswith("res_block1.") for k in sd):
            new_sd = {}
            for k, v in sd.items():
                nk = k
                for old_i, new_i in [("res_block1.", "res_blocks.0."),
                                      ("res_block2.", "res_blocks.1."),
                                      ("res_block3.", "res_blocks.2.")]:
                    nk = nk.replace(old_i, new_i)
                new_sd[nk] = v
            sd = new_sd
        net.load_state_dict(sd)
        net.eval()
        return net

    mcts_models = {}
    for name, path in [
        ("MCTS E (seed0)", "mcts_runs/E_seed0/draft_policy.pt"),
        ("MCTS E (seed3)", "mcts_runs/E_seed3/draft_policy.pt"),
        ("MCTS G (seed4)", "mcts_runs/G_seed4/draft_policy.pt"),
    ]:
        fpath = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(fpath):
            mcts_models[name] = load_policy(fpath)
            print(f"  {name}: loaded")

    # Build strategies
    strategies = {}
    strategies["GD baseline"] = make_gd_strategy(gd, sample=False)
    strategies["Enriched greedy"] = make_wp_greedy_strategy(wp_enr, stats, wp_cols, all_mask)
    strategies["Augmented greedy"] = make_wp_greedy_strategy(wp_aug, stats, wp_cols, all_mask)
    for name, net in mcts_models.items():
        strategies[name] = make_policy_strategy(net)

    gd_opp = make_gd_opponent(gd)

    # Generate configs
    configs = [(random.choice(MAPS), random.choice(SKILL_TIERS), i % 2) for i in range(N)]

    # Run evaluation
    results = {}
    for strat_name, strat_fn in strategies.items():
        t0 = time.time()
        all_metrics = []

        for game_map, tier, our_team in configs:
            pick_steps, valid_at_step = simulate_draft(
                strat_fn, gd_opp, game_map, tier, our_team, stats)
            metrics = full_draft_quality(pick_steps, stats, tier, valid_at_step)
            all_metrics.append(metrics)

        elapsed = time.time() - t0

        # Aggregate
        agg = {}
        for key in all_metrics[0].keys():
            vals = [m[key] for m in all_metrics]
            agg[key] = np.mean(vals)
            agg[key + "_std"] = np.std(vals)

        results[strat_name] = agg
        print(f"\n{strat_name} ({elapsed:.1f}s):")
        print(f"  Resilience: avg={agg['resilience_avg']:.3f} grad={agg['resilience_gradient']:.3f}")
        print(f"  Counter:    avg={agg['counter_avg']:.3f} late={agg['counter_late']:.3f} "
              f"grad={agg['counter_gradient']:.3f} capt={agg['counter_capture_rate']:.1%} "
              f"gap={agg['counter_delta_gap']:.3f}")
        print(f"  Synergy:    team={agg['team_synergy']:.3f} incr={agg['incremental_synergy']:.3f}")

    # Print summary table
    print("\n\n" + "=" * 110)
    header = (f"{'Strategy':<22} {'Resil':>6} {'R.Grad':>7} {'Ctr':>6} {'CtrLate':>8} "
              f"{'C.Grad':>7} {'Capt%':>6} {'Gap':>6} {'T.Syn':>6} {'I.Syn':>6}")
    print(header)
    print("-" * 110)
    for name in ["GD baseline", "Enriched greedy", "Augmented greedy",
                 "MCTS E (seed0)", "MCTS E (seed3)", "MCTS G (seed4)"]:
        if name not in results:
            continue
        r = results[name]
        print(f"{name:<22} {r['resilience_avg']:>6.3f} {r['resilience_gradient']:>7.3f} "
              f"{r['counter_avg']:>6.3f} {r['counter_late']:>8.3f} "
              f"{r['counter_gradient']:>7.3f} {r['counter_capture_rate']:>5.0%} "
              f"{r['counter_delta_gap']:>6.3f} {r['team_synergy']:>6.3f} "
              f"{r['incremental_synergy']:>6.3f}")
    print("=" * 110)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "experiment_results", "draft_quality")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "draft_quality_results.json")

    # Convert for JSON
    json_results = {}
    for name, agg in results.items():
        json_results[name] = {k: float(v) for k, v in agg.items()}
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
