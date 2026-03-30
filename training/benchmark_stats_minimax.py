#!/usr/bin/env python3
"""
Benchmark: Stats Mode with Minimax Search.

Full minimax over stats-scored candidates with alpha-beta pruning.
At each node:
  - Our turns: pick best of top-K candidates (by stats score)
  - Opponent turns: pick best of top-K candidates (by opponent stats score)
  - Below search depth: GD rollout to terminal, evaluate with enriched WP

Sweeps depth 2, 4, 6, 8 with breadth 3.
Also tests composition enforcement fixes (healer bonus, stronger comp weight).

500 drafts × random tier/map/first-pick.

Usage:
    set -a && source .env && set +a
    python3 -u training/benchmark_stats_minimax.py
"""

import os, sys, json, random, time
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import (NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, MAP_TO_IDX,
                    SKILL_TIERS, TIER_TO_IDX,
                    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
                    HERO_ROLE_FINE)
from sweep_enriched_wp import (StatsCache, WinProbEnrichedModel, FEATURE_GROUPS,
                                compute_group_indices, extract_features, FEATURE_GROUP_DIMS)
from train_draft_policy import DraftState, DRAFT_ORDER
from train_generic_draft import GenericDraftModel


# ── Constants ────────────────────────────────────────────────────────

DRAFT_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
DRAFT_IS_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]

ROLE_MAP = {
    "tank": "Tank", "bruiser": "Bruiser", "healer": "Healer",
    "ranged_aa": "Ranged Assassin", "ranged_mage": "Ranged Assassin",
    "melee_assassin": "Melee Assassin", "support_utility": "Support",
    "varian": "Bruiser", "pusher": "Ranged Assassin",
}
HERO_TO_BLIZZ_ROLE = {h: ROLE_MAP.get(fr, "Ranged Assassin")
                       for h, fr in HERO_ROLE_FINE.items()}


# ── Load models ──────────────────────────────────────────────────────

print("Loading models...")
stats_cache = StatsCache()
gi = compute_group_indices()
WP_GROUPS = ['role_counts', 'team_avg_wr', 'map_delta', 'pairwise_counters',
             'pairwise_synergies', 'counter_detail', 'meta_strength',
             'draft_diversity', 'comp_wr']
wp_cols = []
for g in WP_GROUPS:
    s, e = gi[g]
    wp_cols.extend(range(s, e))
all_mask = [True] * len(FEATURE_GROUPS)
wp_input_dim = 197 + sum(FEATURE_GROUP_DIMS[g] for g in WP_GROUPS)

wp_model = WinProbEnrichedModel(wp_input_dim, [256, 128], dropout=0.3)
wp_model.load_state_dict(torch.load(
    os.path.join(os.path.dirname(__file__), "wp_experiment_enriched.pt"),
    weights_only=True, map_location="cpu"))
wp_model.eval()

gd_models = []
for i in range(5):
    gd = GenericDraftModel()
    gd.load_state_dict(torch.load(
        os.path.join(os.path.dirname(__file__), f"generic_draft_{i}.pt"),
        weights_only=True, map_location="cpu"))
    gd.eval()
    gd_models.append(gd)

# Composition data
comp_path = os.path.join(os.path.dirname(__file__), "..", "src", "lib", "data", "compositions.json")
COMP_DATA = {}
if os.path.exists(comp_path):
    for tier_name, comps in json.load(open(comp_path)).items():
        tier_comps = {}
        for c in comps:
            key = ",".join(sorted(c["roles"]))
            if key not in tier_comps or c["games"] > tier_comps[key]["games"]:
                tier_comps[key] = {"winRate": c["winRate"], "games": c["games"]}
        COMP_DATA[tier_name] = tier_comps

print(f"  Loaded WP + {len(gd_models)} GD + comp data for {len(COMP_DATA)} tiers")


# ── Scoring functions ────────────────────────────────────────────────

def get_hero_wr(hero, game_map, tier):
    """Resolve hero WR: prefer map-specific, fallback overall."""
    map_data = stats_cache.hero_map_wr.get(tier, {}).get(game_map, {}).get(hero)
    if map_data and map_data[1] >= 50:
        return map_data[0]
    return stats_cache.get_hero_wr(hero, tier)


def score_pick(hero, our_picks, opp_picks, game_map, tier, healer_bonus=3.0, comp_weight=2.0):
    """Score a hero for picking. Matches engine.ts factors + enhanced comp enforcement."""
    hwr = get_hero_wr(hero, game_map, tier)

    # 1. Hero WR delta
    wr_delta = hwr - 50.0

    # 2. Counter vs opponents (normalized)
    ctr_sum, ctr_n = 0.0, 0
    for opp in opp_picks:
        raw = stats_cache.get_counter(hero, opp, tier)
        if raw is None:
            continue
        owr = get_hero_wr(opp, game_map, tier)
        ctr_sum += raw - (hwr + (100 - owr) - 50)
        ctr_n += 1
    ctr_avg = ctr_sum / ctr_n if ctr_n > 0 else 0.0

    # 3. Synergy with allies (normalized)
    syn_sum, syn_n = 0.0, 0
    for ally in our_picks:
        raw = stats_cache.get_synergy(hero, ally, tier)
        if raw is None:
            continue
        awr = get_hero_wr(ally, game_map, tier)
        syn_sum += raw - (50 + (hwr - 50) + (awr - 50))
        syn_n += 1
    syn_avg = syn_sum / syn_n if syn_n > 0 else 0.0

    # 4. Composition scoring
    comp_delta = 0.0
    if tier in COMP_DATA:
        current_roles = [HERO_TO_BLIZZ_ROLE.get(h, "Ranged Assassin") for h in our_picks]
        cand_role = HERO_TO_BLIZZ_ROLE.get(hero, "Ranged Assassin")
        roles_with = sorted(current_roles + [cand_role])
        best_comp = None
        for comp_key, info in COMP_DATA[tier].items():
            if info["games"] < 100:
                continue
            comp_roles = comp_key.split(",")
            remaining = list(comp_roles)
            ok = True
            for r in roles_with:
                if r in remaining:
                    remaining.remove(r)
                else:
                    ok = False
                    break
            if ok:
                if best_comp is None or info["winRate"] > best_comp:
                    best_comp = info["winRate"]
        if best_comp is not None:
            comp_delta = (best_comp - 50.0) * comp_weight
        else:
            # No valid composition achievable — heavy penalty
            comp_delta = -5.0 * comp_weight

    # 5. Healer urgency
    healer_delta = 0.0
    healer_set = set(h for h, r in HERO_ROLE_FINE.items() if r == 'healer')
    has_healer = any(h in healer_set for h in our_picks)
    hero_is_healer = hero in healer_set
    if not has_healer and healer_bonus > 0:
        picks_remaining = 5 - len(our_picks) - 1
        if picks_remaining <= 1 and hero_is_healer:
            healer_delta = healer_bonus
        elif picks_remaining <= 1 and not hero_is_healer:
            healer_delta = -healer_bonus  # penalize non-healer when healer needed

    return wr_delta + ctr_avg + syn_avg + comp_delta + healer_delta


def score_opponent_pick(hero, their_picks, our_picks, game_map, tier):
    """Score from opponent's perspective (what they'd want to pick)."""
    hwr = get_hero_wr(hero, game_map, tier)
    wr_delta = hwr - 50.0

    # Counter vs our picks
    ctr_sum, ctr_n = 0.0, 0
    for our_h in our_picks:
        raw = stats_cache.get_counter(hero, our_h, tier)
        if raw is None:
            continue
        owr = get_hero_wr(our_h, game_map, tier)
        ctr_sum += raw - (hwr + (100 - owr) - 50)
        ctr_n += 1

    # Synergy with their team
    syn_sum, syn_n = 0.0, 0
    for ally in their_picks:
        raw = stats_cache.get_synergy(hero, ally, tier)
        if raw is None:
            continue
        awr = get_hero_wr(ally, game_map, tier)
        syn_sum += raw - (50 + (hwr - 50) + (awr - 50))
        syn_n += 1

    return wr_delta + (ctr_sum / ctr_n if ctr_n > 0 else 0.0) + (syn_sum / syn_n if syn_n > 0 else 0.0)


# ── WP evaluation ────────────────────────────────────────────────────

def evaluate_wp_sym(t0_heroes, t1_heroes, game_map, tier):
    def _run(t0, t1):
        d = {'team0_heroes': t0, 'team1_heroes': t1,
             'game_map': game_map, 'skill_tier': tier, 'winner': 0}
        base, enriched = extract_features(d, stats_cache, all_mask)
        x = np.concatenate([base, enriched[wp_cols]])
        with torch.no_grad():
            return wp_model(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()
    n = _run(t0_heroes, t1_heroes)
    s = _run(t1_heroes, t0_heroes)
    return (n + (1.0 - s)) / 2.0


# ── GD rollout ───────────────────────────────────────────────────────

def gd_rollout_to_terminal(state, game_map, tier, gd_temp=1.0):
    """Roll out remaining steps with GD sampling, return terminal state."""
    while not state.is_terminal():
        step_team, step_type = DRAFT_ORDER[state.step]
        s = np.concatenate([state.team0_picks, state.team1_picks, state.bans,
                            map_to_one_hot(game_map), tier_to_one_hot(tier),
                            [state.step / 15.0, 0.0 if step_type == 'ban' else 1.0]])
        mask = state.valid_mask_np()
        gd = random.choice(gd_models)
        with torch.no_grad():
            logits = gd(torch.tensor(s, dtype=torch.float32).unsqueeze(0),
                         torch.tensor(mask, dtype=torch.float32).unsqueeze(0))
            if gd_temp != 1.0:
                logits = logits / gd_temp
            probs = F.softmax(logits, dim=1)
            action = torch.multinomial(probs, 1).item()
        state.apply_action(action, step_team, step_type)
    return state


def evaluate_terminal(state, game_map, tier, our_team):
    """Evaluate completed draft from our_team's perspective."""
    t0h = [HEROES[j] for j in range(NUM_HEROES) if state.team0_picks[j] > 0.5]
    t1h = [HEROES[j] for j in range(NUM_HEROES) if state.team1_picks[j] > 0.5]
    if len(t0h) != 5 or len(t1h) != 5:
        return 0.5
    wp_t0 = evaluate_wp_sym(t0h, t1h, game_map, tier)
    return wp_t0 if our_team == 0 else 1.0 - wp_t0


# ── Minimax with alpha-beta ─────────────────────────────────────────

def get_top_candidates(state, team, our_team, game_map, tier, breadth,
                        healer_bonus, comp_weight):
    """Get top-K hero candidates for a team at current state."""
    mask = state.valid_mask_np()
    valid = [HEROES[i] for i in range(NUM_HEROES) if mask[i] > 0.5]

    our_picks = [HEROES[i] for i in range(NUM_HEROES) if
                 (state.team0_picks if our_team == 0 else state.team1_picks)[i] > 0.5]
    opp_picks = [HEROES[i] for i in range(NUM_HEROES) if
                 (state.team1_picks if our_team == 0 else state.team0_picks)[i] > 0.5]

    if team == our_team:
        scored = [(h, score_pick(h, our_picks, opp_picks, game_map, tier,
                                  healer_bonus, comp_weight)) for h in valid]
    else:
        # Opponent's perspective: their picks are what we see as opp_picks
        their_picks = opp_picks  # from our POV, opp is "them"
        scored = [(h, score_opponent_pick(h, their_picks, our_picks, game_map, tier))
                  for h in valid]

    scored.sort(key=lambda x: -x[1])
    return [(h, HERO_TO_IDX[h]) for h, _ in scored[:breadth]]


def minimax(state, our_team, game_map, tier, depth, breadth,
            alpha, beta, healer_bonus, comp_weight, gd_temp, node_count):
    """
    Minimax with alpha-beta pruning.
    Maximizing for our_team, minimizing for opponent.
    At depth 0 or terminal: GD rollout + WP eval.
    """
    if state.is_terminal():
        node_count[0] += 1
        return evaluate_terminal(state, game_map, tier, our_team)

    step_team, step_type = DRAFT_ORDER[state.step]

    # At depth 0: rollout with GD to terminal
    if depth <= 0 or step_type == 'ban':
        node_count[0] += 1
        rollout = state.clone()
        # For bans at depth>0, use stats scoring for the ban then continue search
        if step_type == 'ban' and depth > 0:
            mask = rollout.valid_mask_np()
            valid = [HEROES[i] for i in range(NUM_HEROES) if mask[i] > 0.5]
            our_picks = [HEROES[i] for i in range(NUM_HEROES) if
                         (rollout.team0_picks if our_team == 0 else rollout.team1_picks)[i] > 0.5]
            opp_picks = [HEROES[i] for i in range(NUM_HEROES) if
                         (rollout.team1_picks if our_team == 0 else rollout.team0_picks)[i] > 0.5]
            # Ban highest WR hero
            ban_scores = [(h, get_hero_wr(h, game_map, tier)) for h in valid]
            ban_scores.sort(key=lambda x: -x[1])
            ban_hero = HERO_TO_IDX[ban_scores[0][0]]
            rollout.apply_action(ban_hero, step_team, step_type)
            return minimax(rollout, our_team, game_map, tier, depth, breadth,
                           alpha, beta, healer_bonus, comp_weight, gd_temp, node_count)

        rollout = gd_rollout_to_terminal(rollout, game_map, tier, gd_temp)
        return evaluate_terminal(rollout, game_map, tier, our_team)

    is_maximizing = (step_team == our_team)
    candidates = get_top_candidates(state, step_team, our_team, game_map, tier,
                                     breadth, healer_bonus, comp_weight)

    if is_maximizing:
        value = -1e9
        for hero_name, hero_idx in candidates:
            child = state.clone()
            child.apply_action(hero_idx, step_team, step_type)
            v = minimax(child, our_team, game_map, tier, depth - 1, breadth,
                        alpha, beta, healer_bonus, comp_weight, gd_temp, node_count)
            value = max(value, v)
            alpha = max(alpha, v)
            if beta <= alpha:
                break
        return value
    else:
        value = 1e9
        for hero_name, hero_idx in candidates:
            child = state.clone()
            child.apply_action(hero_idx, step_team, step_type)
            v = minimax(child, our_team, game_map, tier, depth - 1, breadth,
                        alpha, beta, healer_bonus, comp_weight, gd_temp, node_count)
            value = min(value, v)
            beta = min(beta, v)
            if beta <= alpha:
                break
        return value


def minimax_pick(state, our_team, game_map, tier, depth, breadth,
                  healer_bonus, comp_weight, gd_temp):
    """Select the best pick using minimax search."""
    step_team, step_type = DRAFT_ORDER[state.step]
    candidates = get_top_candidates(state, step_team, our_team, game_map, tier,
                                     breadth, healer_bonus, comp_weight)
    best_hero_idx = candidates[0][1]
    best_value = -1e9
    node_count = [0]

    for hero_name, hero_idx in candidates:
        child = state.clone()
        child.apply_action(hero_idx, step_team, step_type)
        v = minimax(child, our_team, game_map, tier, depth - 1, breadth,
                    -1e9, 1e9, healer_bonus, comp_weight, gd_temp, node_count)
        if v > best_value:
            best_value = v
            best_hero_idx = hero_idx

    return best_hero_idx, node_count[0]


# ── Draft simulation ─────────────────────────────────────────────────

def run_draft(game_map, tier, our_team, depth, breadth, healer_bonus,
              comp_weight, gd_temp):
    state = DraftState(game_map, tier, our_team=our_team)
    pick_steps = []
    total_nodes = 0

    while not state.is_terminal():
        step_team, step_type = DRAFT_ORDER[state.step]
        step_num = state.step

        if step_team == our_team:
            if step_type == 'ban':
                mask = state.valid_mask_np()
                valid = [HEROES[i] for i in range(NUM_HEROES) if mask[i] > 0.5]
                ban_scores = [(h, get_hero_wr(h, game_map, tier)) for h in valid]
                ban_scores.sort(key=lambda x: -x[1])
                hero_idx = HERO_TO_IDX[ban_scores[0][0]]
            else:
                hero_idx, nodes = minimax_pick(state, our_team, game_map, tier,
                                                depth, breadth, healer_bonus,
                                                comp_weight, gd_temp)
                total_nodes += nodes
        else:
            hero_idx = gd_sample(state, game_map, tier, gd_temp)

        if step_type == 'pick':
            team_label = 'ours' if step_team == our_team else 'theirs'
            pick_steps.append((HEROES[hero_idx], team_label, step_num))

        state.apply_action(hero_idx, step_team, step_type)

    return pick_steps, total_nodes


def gd_sample(state, game_map, tier, temperature=1.0):
    step_team, step_type = DRAFT_ORDER[state.step]
    s = np.concatenate([state.team0_picks, state.team1_picks, state.bans,
                        map_to_one_hot(game_map), tier_to_one_hot(tier),
                        [state.step / 15.0, 0.0 if step_type == 'ban' else 1.0]])
    mask = state.valid_mask_np()
    gd = random.choice(gd_models)
    with torch.no_grad():
        logits = gd(torch.tensor(s, dtype=torch.float32).unsqueeze(0),
                     torch.tensor(mask, dtype=torch.float32).unsqueeze(0))
        if temperature != 1.0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=1)
        return torch.multinomial(probs, 1).item()


# ── Metrics ──────────────────────────────────────────────────────────

def cd(ha, hb, tier):
    raw = stats_cache.get_counter(ha, hb, tier)
    if raw is None: return None
    return raw - (stats_cache.get_hero_wr(ha, tier) + (100 - stats_cache.get_hero_wr(hb, tier)) - 50)

def sd(ha, hb, tier):
    raw = stats_cache.get_synergy(ha, hb, tier)
    if raw is None: return None
    return raw - (50 + (stats_cache.get_hero_wr(ha, tier) - 50) + (stats_cache.get_hero_wr(hb, tier) - 50))

def compute_metrics(pick_steps, tier):
    our = [(h, s) for h, tm, s in pick_steps if tm == 'ours']
    opp = [(h, s) for h, tm, s in pick_steps if tm == 'theirs']
    oh = [h for h, _ in our]
    exp = []
    for h, os_ in our:
        sub = [x for x, s in opp if s > os_]
        if not sub: exp.append(0.0); continue
        ds = [d for d in (cd(x, h, tier) for x in sub) if d is not None]
        exp.append(np.mean(ds) if ds else 0.0)
    rg = (np.mean(exp[-2:]) - np.mean(exp[:2])) if len(exp) >= 4 else 0.0
    ct = []
    for h, os_ in our:
        pr = [x for x, s in opp if s < os_]
        if not pr: ct.append(0.0); continue
        ds = [d for d in (cd(h, x, tier) for x in pr) if d is not None]
        ct.append(np.mean(ds) if ds else 0.0)
    ca = np.mean(ct) if ct else 0.0
    cl = np.mean(ct[-2:]) if len(ct) >= 2 else 0.0
    sy = []
    for i, h1 in enumerate(oh):
        for h2 in oh[i + 1:]:
            d = sd(h1, h2, tier)
            if d is not None: sy.append(d)
    ts = np.mean(sy) if sy else 0.0
    hs = set(h for h, r in HERO_ROLE_FINE.items() if r == 'healer')
    fs = set(h for h, r in HERO_ROLE_FINE.items() if r in ('tank', 'bruiser'))
    rs = set(h for h, r in HERO_ROLE_FINE.items() if r in ('ranged_aa', 'ranged_mage', 'pusher'))
    hh = any(h in hs for h in oh)
    hf = any(h in fs for h in oh)
    hr = any(h in rs for h in oh)
    roles = {}
    for h in oh: r = HERO_ROLE_FINE.get(h, 'x'); roles[r] = roles.get(r, 0) + 1
    from shared import is_degenerate
    dg = is_degenerate(oh)
    return {'counter': ca, 'counter_late': cl, 'synergy': ts, 'resil_grad': rg,
            'healer': hh, 'degen': dg, 'heroes': oh}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    N = 500
    random.seed(42)
    np.random.seed(42)

    configs = [(random.choice(MAPS), random.choice(SKILL_TIERS), random.randint(0, 1))
               for _ in range(N)]

    # (label, depth, breadth, healer_bonus, comp_weight, gd_temp)
    sweep = [
        ("d0_stats_only",    0, 3, 3.0, 2.0, 1.0),
        ("d2_b3",            2, 3, 3.0, 2.0, 1.0),
        ("d4_b3",            4, 3, 3.0, 2.0, 1.0),
        ("d6_b3",            6, 3, 3.0, 2.0, 1.0),
        ("d8_b3",            8, 3, 3.0, 2.0, 1.0),
        ("d6_b3_t2.0",       6, 3, 3.0, 2.0, 2.0),
        ("d6_b5",            6, 5, 3.0, 2.0, 1.0),
    ]

    print(f"\nMinimax Benchmark: {N} drafts × {len(sweep)} configs")
    print("=" * 115)

    all_results = {}
    for label, depth, breadth, hb, cw, gd_temp in sweep:
        t0 = time.time()
        all_m = []
        hero_counter = Counter()
        total_nodes_all = 0

        for ci, (game_map, tier, our_team) in enumerate(configs):
            pick_steps, nodes = run_draft(game_map, tier, our_team, depth, breadth,
                                           hb, cw, gd_temp)
            total_nodes_all += nodes
            m = compute_metrics(pick_steps, tier)
            all_m.append(m)
            for h in m['heroes']:
                hero_counter[h] += 1

            if (ci + 1) % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (ci + 1) * (N - ci - 1)
                print(f"  {label}: {ci+1}/{N} ({eta:.0f}s left, {total_nodes_all/(ci+1):.0f} nodes/draft)")

        elapsed = time.time() - t0
        agg = {k: np.mean([m[k] for m in all_m]) for k in ['counter', 'counter_late', 'synergy', 'resil_grad']}
        agg['healer'] = np.mean([m['healer'] for m in all_m]) * 100
        agg['degen'] = np.mean([m['degen'] for m in all_m]) * 100
        agg['distinct'] = len(hero_counter)
        agg['time'] = elapsed
        agg['avg_nodes'] = total_nodes_all / N
        all_results[label] = agg

        print(f"  {label:<20} ctr={agg['counter']:+.3f} ctrL={agg['counter_late']:+.3f} "
              f"syn={agg['synergy']:.3f} rG={agg['resil_grad']:+.3f} "
              f"hlr={agg['healer']:.0f}% deg={agg['degen']:.0f}% div={agg['distinct']} "
              f"nodes={agg['avg_nodes']:.0f} ({elapsed:.0f}s)\n")

    # Summary
    print("\n" + "=" * 115)
    print(f"{'Config':<20} {'Ctr':>7} {'CtrL':>7} {'Syn':>7} {'R.Grad':>7} "
          f"{'Hlr%':>5} {'Deg%':>5} {'Div':>4} {'Nodes':>7} {'Time':>6}")
    print("-" * 115)
    for label, agg in all_results.items():
        print(f"{label:<20} {agg['counter']:>+7.3f} {agg['counter_late']:>+7.3f} "
              f"{agg['synergy']:>7.3f} {agg['resil_grad']:>+7.3f} "
              f"{agg['healer']:>5.0f} {agg['degen']:>5.0f} {agg['distinct']:>4} "
              f"{agg['avg_nodes']:>7.0f} {agg['time']:>5.0f}s")
    print("-" * 115)
    print(f"{'E baseline (MCTS)':<20} {-0.082:>+7.3f} {-0.114:>+7.3f} {0.503:>7.3f} {-0.578:>+7.3f} {'86':>5} {'26':>5} {'23':>4}")
    print(f"{'Hybrid sw=9':<20} {+0.137:>+7.3f} {+0.368:>+7.3f} {1.071:>7.3f} {-0.619:>+7.3f} {'64':>5} {'64':>5} {'66':>4}")
    print(f"{'Greedy enriched':<20} {+0.305:>+7.3f} {+0.364:>+7.3f} {1.171:>7.3f} {+0.119:>+7.3f} {'74':>5} {'55':>5} {'83':>4}")
    print("=" * 115)

    out_dir = os.path.join(os.path.dirname(__file__), "experiment_results", "stats_search")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "minimax_benchmark.json"), "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_results.items()}, f, indent=2)
    print(f"\nSaved to {out_dir}/minimax_benchmark.json")


if __name__ == "__main__":
    main()
