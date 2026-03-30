#!/usr/bin/env python3
"""
Approach 3: Hybrid MCTS early + greedy late drafting.

Uses Python MCTS (with E_seed0 policy prior + enriched WP leaf eval) for early
draft steps, then switches to greedy WP-maximizing search for late steps.
Tests switch points at steps 0, 7, 9, 11, 13, 16.

For greedy PICKS: iterate all valid heroes, for each do 3 GD rollouts to
terminal, evaluate with symmetrized enriched WP, average, pick best.
For greedy BANS after the switch: use GD model (bans don't need WP optimization).
Opponent always uses GD sampling regardless of switch point.

Usage:
    set -a && source .env && set +a
    python3 -u training/experiment_hybrid_draft.py --drafts 200
"""

import os, sys, json, random, argparse, time, math
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import (NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS,
                    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot, HERO_ROLE_FINE)
from sweep_enriched_wp import (StatsCache, WinProbEnrichedModel, FEATURE_GROUPS,
                                compute_group_indices, extract_features, FEATURE_GROUP_DIMS)
from train_draft_policy import (AlphaZeroDraftNet, DraftState, DRAFT_ORDER,
                                 MCTSNode, NUM_HEROES as _NH)
from train_generic_draft import GenericDraftModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "experiment_results", "leaf_eval_fix")


# ── WP setup ─────────────────────────────────────────────────────────

WP_GROUPS = ['role_counts', 'team_avg_wr', 'map_delta', 'pairwise_counters',
             'pairwise_synergies', 'counter_detail', 'meta_strength',
             'draft_diversity', 'comp_wr']


def setup_wp():
    """Load enriched WP model and return (wp_model, wp_cols, all_mask)."""
    gi = compute_group_indices()
    wp_cols = []
    for g in WP_GROUPS:
        s, e = gi[g]
        wp_cols.extend(range(s, e))
    enriched_dim = sum(FEATURE_GROUP_DIMS[g] for g in WP_GROUPS)
    wp_input_dim = 197 + enriched_dim

    wp_model = WinProbEnrichedModel(wp_input_dim, [256, 128], dropout=0.3)
    wp_model.load_state_dict(torch.load(
        os.path.join(os.path.dirname(__file__), "wp_experiment_enriched.pt"),
        weights_only=True, map_location="cpu"))
    wp_model.eval()
    all_mask = [True] * len(FEATURE_GROUPS)
    return wp_model, wp_cols, all_mask, gi


def evaluate_wp_sym(wp_model, state, stats, wp_cols, all_mask):
    """Symmetrized WP from state.our_team's perspective."""
    t0_heroes = [HEROES[i] for i in range(NUM_HEROES) if state.team0_picks[i] > 0.5]
    t1_heroes = [HEROES[i] for i in range(NUM_HEROES) if state.team1_picks[i] > 0.5]

    def _run(t0h, t1h):
        d = {'team0_heroes': t0h, 'team1_heroes': t1h,
             'game_map': state.game_map, 'skill_tier': state.skill_tier, 'winner': 0}
        base, enriched = extract_features(d, stats, all_mask)
        x = np.concatenate([base, enriched[wp_cols]])
        with torch.no_grad():
            return wp_model(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()

    wp_n = _run(t0_heroes, t1_heroes)
    wp_s = _run(t1_heroes, t0_heroes)
    wp_t0 = (wp_n + (1.0 - wp_s)) / 2.0
    return wp_t0 if state.our_team == 0 else 1.0 - wp_t0


# ── Policy loading ───────────────────────────────────────────────────

def load_policy(path):
    """Load AlphaZeroDraftNet, remapping old weight names if needed."""
    net = AlphaZeroDraftNet()
    sd = torch.load(path, weights_only=True, map_location="cpu")
    if any(k.startswith("res_block1.") for k in sd):
        new_sd = {}
        for k, v in sd.items():
            nk = k.replace("res_block1.", "res_blocks.0.") \
                  .replace("res_block2.", "res_blocks.1.") \
                  .replace("res_block3.", "res_blocks.2.")
            new_sd[nk] = v
        sd = new_sd
    net.load_state_dict(sd)
    net.eval()
    return net


def predict_fn(network, state, device="cpu"):
    """Get (priors, value) from the policy network. Value is symmetrized."""
    x = state.to_tensor(device)
    mask = state.valid_mask(device)
    x_swap = x.clone()
    x_swap[:, -1] = 1.0 - x_swap[:, -1]
    with torch.no_grad():
        logits, value = network(x, mask)
        _, value_swap = network(x_swap, mask)
        priors = F.softmax(logits, dim=1).cpu().numpy()[0]
        sym_value = (value.item() + (1.0 - value_swap.item())) / 2.0
    return priors, sym_value


# ── Python MCTS ──────────────────────────────────────────────────────

def mcts_search(root_state, network, wp_model, gd_models, stats,
                wp_cols, all_mask, num_sims=200, c_puct=2.0, device="cpu"):
    """
    Run MCTS from root_state for root_state.our_team's decision.
    Returns visit count distribution over actions (normalized).
    """
    our_team = root_state.our_team
    root = MCTSNode(root_state)

    # Expand root
    priors, _ = predict_fn(network, root_state, device)
    valid = root_state.valid_mask_np()
    priors = priors * valid
    ps = priors.sum()
    if ps > 0:
        priors /= ps
    root.is_expanded = True
    for a in range(NUM_HEROES):
        if valid[a] > 0:
            root.children[a] = MCTSNode(None, parent=root, action=a, prior=priors[a])

    for _ in range(num_sims):
        node = root
        scratch = root_state.clone()

        # Selection
        while node.is_expanded and not scratch.is_terminal():
            if scratch.current_team() == our_team:
                best_score = -float('inf')
                best_child = None
                for child in node.children.values():
                    score = child.ucb_score(c_puct)
                    if score > best_score:
                        best_score = score
                        best_child = child
                if best_child is None:
                    break
                team, atype = DRAFT_ORDER[scratch.step]
                scratch.apply_action(best_child.action, team, atype)
                node = best_child
            else:
                # Opponent: sample from random GD model
                gd = random.choice(gd_models)
                x = scratch.to_tensor_gd(device)
                m = scratch.valid_mask(device)
                with torch.no_grad():
                    logits = gd(x, m)
                    probs = F.softmax(logits, dim=1)
                    opp_act = torch.multinomial(probs, 1).item()
                team, atype = DRAFT_ORDER[scratch.step]
                scratch.apply_action(opp_act, team, atype)

        if scratch.is_terminal():
            value = evaluate_wp_sym(wp_model, scratch, stats, wp_cols, all_mask)
        else:
            if not node.is_expanded and scratch.current_team() == our_team:
                priors_leaf, value = predict_fn(network, scratch, device)
                valid_leaf = scratch.valid_mask_np()
                priors_leaf = priors_leaf * valid_leaf
                ps2 = priors_leaf.sum()
                if ps2 > 0:
                    priors_leaf /= ps2
                node.state = scratch
                node.is_expanded = True
                for a in range(NUM_HEROES):
                    if valid_leaf[a] > 0:
                        node.children[a] = MCTSNode(None, parent=node, action=a, prior=priors_leaf[a])
            else:
                _, value = predict_fn(network, scratch, device)

        # Backprop
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    visits = np.zeros(NUM_HEROES, dtype=np.float32)
    for action, child in root.children.items():
        visits[action] = child.visit_count
    vs = visits.sum()
    if vs > 0:
        visits /= vs
    return visits


# ── GD sampling helper ───────────────────────────────────────────────

def gd_sample(gd_model, state, device="cpu"):
    """Sample one action from GD model."""
    x = state.to_tensor_gd(device)
    m = state.valid_mask(device)
    with torch.no_grad():
        logits = gd_model(x, m)
        probs = F.softmax(logits, dim=1)
        return torch.multinomial(probs, 1).item()


# ── Greedy WP strategy ──────────────────────────────────────────────

def greedy_pick(state, gd_models, wp_model, stats, wp_cols, all_mask,
                n_rollouts=3, device="cpu"):
    """
    For a PICK step on our team's turn: try all valid heroes, for each do
    n_rollouts GD rollouts to terminal, evaluate with symmetrized WP, pick best.
    """
    our_team = state.our_team
    mask = state.valid_mask_np()
    team, atype = DRAFT_ORDER[state.step]

    best_idx = -1
    best_wp = -float('inf')

    candidates = [i for i in range(NUM_HEROES) if mask[i] > 0.5]

    for hero_idx in candidates:
        wp_sum = 0.0
        for r in range(n_rollouts):
            s = state.clone()
            s.apply_action(hero_idx, team, atype)
            # Roll out remaining steps with GD
            gd = gd_models[r % len(gd_models)]
            while not s.is_terminal():
                act = gd_sample(gd, s, device)
                st, at = DRAFT_ORDER[s.step]
                s.apply_action(act, st, at)
            wp_sum += evaluate_wp_sym(wp_model, s, stats, wp_cols, all_mask)
        avg_wp = wp_sum / n_rollouts
        if avg_wp > best_wp:
            best_wp = avg_wp
            best_idx = hero_idx

    return best_idx


# ── Metric functions (from experiment_draft_quality.py / validate_search_budget.py) ──

def counter_delta(hero_a, hero_b, stats, tier):
    raw = stats.get_counter(hero_a, hero_b, tier)
    if raw is None:
        return None
    return raw - (stats.get_hero_wr(hero_a, tier) + (100 - stats.get_hero_wr(hero_b, tier)) - 50)


def synergy_delta(hero_a, hero_b, stats, tier):
    raw = stats.get_synergy(hero_a, hero_b, tier)
    if raw is None:
        return None
    return raw - (50 + (stats.get_hero_wr(hero_a, tier) - 50) + (stats.get_hero_wr(hero_b, tier) - 50))


def compute_draft_metrics(pick_steps, stats, tier):
    """Compute full draft quality metrics for one draft."""
    our = [(h, s) for h, team, s in pick_steps if team == 'ours']
    opp = [(h, s) for h, team, s in pick_steps if team == 'theirs']
    our_heroes = [h for h, _ in our]

    # Counter: how well our picks counter opponent's prior picks
    ctr_all = []
    ctr_early = []  # first 2 picks
    ctr_late = []   # last 2 picks
    for idx, (our_hero, our_step) in enumerate(our):
        prior_opp = [h for h, s in opp if s < our_step]
        if not prior_opp:
            ctr_all.append(0.0)
            if idx < 2: ctr_early.append(0.0)
            if idx >= len(our) - 2: ctr_late.append(0.0)
            continue
        deltas = [d for d in (counter_delta(our_hero, oh, stats, tier) for oh in prior_opp) if d is not None]
        val = np.mean(deltas) if deltas else 0.0
        ctr_all.append(val)
        if idx < 2:
            ctr_early.append(val)
        if idx >= len(our) - 2:
            ctr_late.append(val)

    counter_avg = np.mean(ctr_all) if ctr_all else 0.0
    counter_early = np.mean(ctr_early) if ctr_early else 0.0
    counter_late = np.mean(ctr_late) if ctr_late else 0.0

    # Resilience gradient
    exposures = []
    for our_hero, our_step in our:
        subsequent_opp = [h for h, s in opp if s > our_step]
        if not subsequent_opp:
            exposures.append(0.0)
            continue
        deltas = [d for d in (counter_delta(oh, our_hero, stats, tier) for oh in subsequent_opp) if d is not None]
        exposures.append(np.mean(deltas) if deltas else 0.0)
    resil_grad = (np.mean(exposures[-2:]) - np.mean(exposures[:2])) if len(exposures) >= 4 else 0.0

    # Synergy
    syn_pairs = []
    for i, h1 in enumerate(our_heroes):
        for h2 in our_heroes[i + 1:]:
            d = synergy_delta(h1, h2, stats, tier)
            if d is not None:
                syn_pairs.append(d)
    team_syn = np.mean(syn_pairs) if syn_pairs else 0.0

    # Composition checks
    healer_heroes = set(h for h, r in HERO_ROLE_FINE.items() if r == 'healer')
    frontline = set(h for h, r in HERO_ROLE_FINE.items() if r in ('tank', 'bruiser'))
    ranged = set(h for h, r in HERO_ROLE_FINE.items() if r in ('ranged_aa', 'ranged_mage', 'pusher'))
    has_healer = any(h in healer_heroes for h in our_heroes)
    has_front = any(h in frontline for h in our_heroes)
    has_ranged = any(h in ranged for h in our_heroes)
    roles = {}
    for h in our_heroes:
        r = HERO_ROLE_FINE.get(h, 'unknown')
        roles[r] = roles.get(r, 0) + 1
    degen = not has_healer or not has_front or not has_ranged or any(c >= 3 for c in roles.values())

    return {
        'counter_avg': counter_avg,
        'counter_early': counter_early,
        'counter_late': counter_late,
        'team_synergy': team_syn,
        'resilience_gradient': resil_grad,
        'has_healer': has_healer,
        'degen': degen,
        'heroes': our_heroes,
    }


# ── Hybrid draft simulation ─────────────────────────────────────────

def simulate_hybrid_draft(switch_point, policy_net, gd_models, wp_model,
                          stats, wp_cols, all_mask, game_map, tier, our_team,
                          num_sims=200, n_rollouts=3, device="cpu"):
    """
    Simulate a full draft with hybrid strategy.

    Steps < switch_point: MCTS (policy prior + WP leaf eval)
    Steps >= switch_point:
      - Picks: greedy WP maximization (try all, 3 GD rollouts each)
      - Bans: GD model sampling
    Opponent always uses GD sampling.

    Returns pick_steps for metric evaluation.
    """
    state = DraftState(game_map, tier, our_team=our_team)
    pick_steps = []

    while not state.is_terminal():
        step_team, step_type = DRAFT_ORDER[state.step]
        current_step = state.step

        if step_team == our_team:
            # Our turn
            if current_step < switch_point:
                # MCTS mode
                visits = mcts_search(state, policy_net, wp_model, gd_models,
                                     stats, wp_cols, all_mask,
                                     num_sims=num_sims, device=device)
                hero_idx = np.argmax(visits)
            else:
                # Greedy mode
                if step_type == 'pick':
                    hero_idx = greedy_pick(state, gd_models, wp_model, stats,
                                           wp_cols, all_mask,
                                           n_rollouts=n_rollouts, device=device)
                else:
                    # Ban: use GD model
                    gd = random.choice(gd_models)
                    hero_idx = gd_sample(gd, state, device)
        else:
            # Opponent: always GD sampling
            gd = random.choice(gd_models)
            hero_idx = gd_sample(gd, state, device)

        hero_name = HEROES[hero_idx]
        if step_type == 'pick':
            team_label = "ours" if step_team == our_team else "theirs"
            pick_steps.append((hero_name, team_label, current_step))

        state.apply_action(hero_idx, step_team, step_type)

    return pick_steps


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", type=int, default=200)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    N = args.drafts
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Hybrid MCTS+Greedy Draft Experiment: {N} drafts, {args.sims} sims, {args.rollouts} rollouts")
    print("=" * 100)

    # Load stats
    stats = StatsCache()
    print("Stats loaded")

    # Load WP
    wp_model, wp_cols, all_mask, gi = setup_wp()
    print("WP model loaded")

    # Load GD models (cycle through 5 opponents)
    gd_models = []
    for i in range(5):
        gd_path = os.path.join(os.path.dirname(__file__), f"generic_draft_{i}.pt")
        if os.path.exists(gd_path):
            gd = GenericDraftModel()
            gd.load_state_dict(torch.load(gd_path, weights_only=True, map_location="cpu"))
            gd.eval()
            gd_models.append(gd)
    print(f"Loaded {len(gd_models)} GD models")

    # Load MCTS policy (E_seed0)
    policy_path = os.path.join(os.path.dirname(__file__), "mcts_runs", "E_seed0", "draft_policy.pt")
    policy_net = load_policy(policy_path)
    print("MCTS policy (E_seed0) loaded")

    # Generate fixed configs
    configs = [(random.choice(MAPS), random.choice(SKILL_TIERS), i % 2) for i in range(N)]

    # Switch points: 0 = pure greedy, 7/9/11/13 = hybrid, 16 = pure MCTS
    switch_points = [0, 7, 9, 11, 13, 16]
    all_results = {}

    for sp in switch_points:
        t0 = time.time()
        label = f"switch={sp}"
        if sp == 0:
            label += " (pure greedy)"
        elif sp == 16:
            label += " (pure MCTS)"
        print(f"\n--- {label} ---")

        all_metrics = []
        hero_counter = Counter()

        for di, (game_map, tier, our_team) in enumerate(configs):
            if (di + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  draft {di+1}/{N} ({elapsed:.1f}s)")

            pick_steps = simulate_hybrid_draft(
                switch_point=sp,
                policy_net=policy_net,
                gd_models=gd_models,
                wp_model=wp_model,
                stats=stats,
                wp_cols=wp_cols,
                all_mask=all_mask,
                game_map=game_map,
                tier=tier,
                our_team=our_team,
                num_sims=args.sims,
                n_rollouts=args.rollouts,
            )

            our_heroes = [h for h, team, _ in pick_steps if team == 'ours']
            if len(our_heroes) != 5:
                continue  # incomplete draft

            metrics = compute_draft_metrics(pick_steps, stats, tier)
            all_metrics.append(metrics)
            for h in metrics['heroes']:
                hero_counter[h] += 1

        elapsed = time.time() - t0

        if not all_metrics:
            print(f"  No valid drafts for switch={sp}")
            continue

        agg = {
            'counter_avg': np.mean([m['counter_avg'] for m in all_metrics]),
            'counter_early': np.mean([m['counter_early'] for m in all_metrics]),
            'counter_late': np.mean([m['counter_late'] for m in all_metrics]),
            'team_synergy': np.mean([m['team_synergy'] for m in all_metrics]),
            'resilience_gradient': np.mean([m['resilience_gradient'] for m in all_metrics]),
            'healer_pct': np.mean([m['has_healer'] for m in all_metrics]) * 100,
            'degen_pct': np.mean([m['degen'] for m in all_metrics]) * 100,
            'distinct': len(hero_counter),
            'n_drafts': len(all_metrics),
            'elapsed': elapsed,
        }

        all_results[sp] = agg
        print(f"  {label}: ctr={agg['counter_avg']:+.3f} ctr_e={agg['counter_early']:+.3f} "
              f"ctr_l={agg['counter_late']:+.3f} syn={agg['team_synergy']:.3f} "
              f"rg={agg['resilience_gradient']:.3f} hlr={agg['healer_pct']:.0f}% "
              f"deg={agg['degen_pct']:.0f}% div={agg['distinct']} ({elapsed:.1f}s)")

    # Summary table
    print("\n\n" + "=" * 100)
    print(f"{'Switch':>6} | {'Counter':>8} {'CtrEarly':>9} {'CtrLate':>8} | "
          f"{'Synergy':>8} {'R.Grad':>7} | {'Hlr%':>5} {'Deg%':>5} {'Div':>4}")
    print("-" * 100)
    for sp in switch_points:
        if sp not in all_results:
            continue
        r = all_results[sp]
        tag = ""
        if sp == 0:
            tag = " (greedy)"
        elif sp == 16:
            tag = " (MCTS)"
        print(f"{sp:>3}{tag:<7s} | {r['counter_avg']:>+8.3f} {r['counter_early']:>+9.3f} "
              f"{r['counter_late']:>+8.3f} | {r['team_synergy']:>8.3f} "
              f"{r['resilience_gradient']:>7.3f} | {r['healer_pct']:>4.0f}% "
              f"{r['degen_pct']:>4.0f}% {r['distinct']:>4}")
    print("=" * 100)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "hybrid_results.json")
    json_results = {}
    for sp, agg in all_results.items():
        json_results[str(sp)] = {k: float(v) for k, v in agg.items()}
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
