#!/usr/bin/env python3
"""
Benchmark: Stats Mode with Search.

Reimplements the Stats Mode engine (engine.ts) in Python, adds lookahead
search (rollout to terminal with GD, evaluate with enriched WP), and
benchmarks against the same draft quality metrics used for AI evaluation.

Parameters swept:
  - GD opponent temperature: 0.5, 1.0, 2.0
  - Search breadth: 0 (no search, pure stats), 3, 5, 10 candidates
  - Counterability penalty: on/off

500 drafts × random tier/map/first-pick per configuration.

Usage:
    set -a && source .env && set +a
    python3 -u training/benchmark_stats_with_search.py
"""

import os, sys, json, random, time, math
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

DRAFT_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
DRAFT_IS_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]


# ── Load data ────────────────────────────────────────────────────────

print("Loading models and data...")
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
print(f"  Loaded enriched WP + {len(gd_models)} GD models")

# ── Composition data (mirrors engine.ts) ─────────────────────────────

comp_path = os.path.join(os.path.dirname(__file__), "..", "src", "lib", "data", "compositions.json")
COMP_DATA = {}
if os.path.exists(comp_path):
    raw_comp = json.load(open(comp_path))
    for tier_name, comps in raw_comp.items():
        tier_comps = {}
        for c in comps:
            key = ",".join(sorted(c["roles"]))
            tier_comps[key] = {"winRate": c["winRate"], "games": c["games"],
                                "popularity": c.get("popularity", c["games"])}
        COMP_DATA[tier_name] = tier_comps

HERO_TO_ROLE = {}
ROLE_MAP = {
    "tank": "Tank", "bruiser": "Bruiser", "healer": "Healer",
    "ranged_aa": "Ranged Assassin", "ranged_mage": "Ranged Assassin",
    "melee_assassin": "Melee Assassin", "support_utility": "Support",
    "varian": "Bruiser", "pusher": "Ranged Assassin",
}
for h, fr in HERO_ROLE_FINE.items():
    HERO_TO_ROLE[h] = ROLE_MAP.get(fr, "Ranged Assassin")


# ── Counterability precomputation ────────────────────────────────────

def precompute_counterability(tier):
    """Count hard counters for each hero at this tier."""
    result = {}
    for hero in HEROES:
        hwr = stats_cache.get_hero_wr(hero, tier)
        hard = 0
        for opp in HEROES:
            if opp == hero:
                continue
            raw = stats_cache.get_counter(opp, hero, tier)
            if raw is None:
                continue
            owr = stats_cache.get_hero_wr(opp, tier)
            expected = owr + (100 - hwr) - 50
            if raw - expected > 3.0:
                hard += 1
        result[hero] = hard
    return result


# ── Stats Mode scoring (faithful to engine.ts) ──────────────────────

def score_hero_for_pick(hero, our_picks, opp_picks, game_map, tier, data_cache,
                         counterability_map=None, our_pick_count=0):
    """Score a hero for our pick, matching engine.ts logic."""
    # 1. Hero WR
    wr = stats_cache.get_hero_wr(hero, tier)
    map_wr = stats_cache.hero_map_wr.get(tier, {}).get(game_map, {}).get(hero)
    if map_wr and map_wr[1] >= 50:
        use_wr = map_wr[0]
    else:
        use_wr = wr
    wr_delta = use_wr - 50.0

    # 2. Counter vs enemies (normalized)
    ctr_sum, ctr_n = 0.0, 0
    for opp in opp_picks:
        raw = stats_cache.get_counter(hero, opp, tier)
        if raw is None:
            continue
        owr = stats_cache.get_hero_wr(opp, tier)
        expected = use_wr + (100 - owr) - 50
        ctr_sum += raw - expected
        ctr_n += 1
    ctr_avg = ctr_sum / ctr_n if ctr_n > 0 else 0.0

    # 3. Synergy with allies (normalized)
    syn_sum, syn_n = 0.0, 0
    for ally in our_picks:
        raw = stats_cache.get_synergy(hero, ally, tier)
        if raw is None:
            continue
        awr = stats_cache.get_hero_wr(ally, tier)
        expected = 50 + (use_wr - 50) + (awr - 50)
        syn_sum += raw - expected
        syn_n += 1
    syn_avg = syn_sum / syn_n if syn_n > 0 else 0.0

    # 5. Composition scoring (data-driven from compositions.json)
    comp_delta = 0.0
    if COMP_DATA and tier in COMP_DATA:
        current_roles = [HERO_TO_ROLE.get(h, "Ranged Assassin") for h in our_picks]
        candidate_role = HERO_TO_ROLE.get(hero, "Ranged Assassin")
        roles_with = current_roles + [candidate_role]
        # Find best achievable 5-role comp containing these roles
        tier_comps = COMP_DATA[tier]
        roles_key_partial = sorted(roles_with)
        best_comp_wr = None
        for comp_key, comp_info in tier_comps.items():
            if comp_info["games"] < 100:
                continue
            comp_roles = comp_key.split(",")
            # Check if our roles are a subset of this comp
            remaining = list(comp_roles)
            ok = True
            for r in roles_key_partial:
                if r in remaining:
                    remaining.remove(r)
                else:
                    ok = False
                    break
            if ok:
                cwr = comp_info["winRate"]
                if best_comp_wr is None or cwr > best_comp_wr:
                    best_comp_wr = cwr
        if best_comp_wr is not None:
            # Scale by picks made (more picks = more weight)
            scale = min(1.0, len(our_picks) / 3.0)
            comp_delta = (best_comp_wr - 50.0) * scale * 0.5  # moderate weight

    # Counterability penalty (optional)
    risk_delta = 0.0
    if counterability_map is not None:
        hc = counterability_map.get(hero, 0)
        pos_scale = max(0, 1 - our_pick_count / 3)
        if hc >= 3 and pos_scale > 0:
            risk_delta = hc * -0.5 * pos_scale

    return wr_delta + ctr_avg + syn_avg + comp_delta + risk_delta


def score_hero_for_ban(hero, picks_to_protect, opp_picks, game_map, tier):
    """Score a ban candidate, matching engine.ts logic."""
    wr = stats_cache.get_hero_wr(hero, tier)
    wr_delta = wr - 50.0

    # Counter protection: ban heroes strong vs our picks
    for ally in picks_to_protect:
        raw = stats_cache.get_counter(hero, ally, tier)
        if raw is None:
            continue
        awr = stats_cache.get_hero_wr(ally, tier)
        expected = wr + (100 - awr) - 50
        if raw >= expected + 3:
            wr_delta += raw - expected

    return wr_delta


# ── WP evaluation ────────────────────────────────────────────────────

def evaluate_wp_sym(t0_heroes, t1_heroes, game_map, tier):
    """Symmetrized WP evaluation."""
    def _run(t0, t1):
        d = {'team0_heroes': t0, 'team1_heroes': t1,
             'game_map': game_map, 'skill_tier': tier, 'winner': 0}
        base, enriched = extract_features(d, stats_cache, all_mask)
        x = np.concatenate([base, enriched[wp_cols]])
        with torch.no_grad():
            return wp_model(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()
    wp_n = _run(t0_heroes, t1_heroes)
    wp_s = _run(t1_heroes, t0_heroes)
    return (wp_n + (1.0 - wp_s)) / 2.0


# ── GD opponent with temperature ─────────────────────────────────────

def gd_sample(state, game_map, tier, temperature=1.0):
    """Sample opponent action using GD with temperature."""
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


# ── Search: rollout to terminal with GD, evaluate with WP ───────────

def search_pick(state, our_team, game_map, tier, breadth, gd_temp,
                counterability_map=None, rollouts_per_candidate=1):
    """
    Score top-K candidates from stats mode, then for each:
    rollout remaining draft with GD sampling, evaluate terminal with WP.
    Pick the candidate with the best average WP.
    """
    mask = state.valid_mask_np()
    valid = [HEROES[i] for i in range(NUM_HEROES) if mask[i] > 0.5]
    our_picks = [HEROES[i] for i in range(NUM_HEROES) if
                 (state.team0_picks if our_team == 0 else state.team1_picks)[i] > 0.5]
    opp_picks = [HEROES[i] for i in range(NUM_HEROES) if
                 (state.team1_picks if our_team == 0 else state.team0_picks)[i] > 0.5]

    # Score all candidates with stats mode
    scores = []
    for hero in valid:
        s = score_hero_for_pick(hero, our_picks, opp_picks, game_map, tier,
                                 None, counterability_map, len(our_picks))
        scores.append((hero, s))
    scores.sort(key=lambda x: -x[1])

    if breadth == 0:
        # No search — just pick top stats recommendation
        return HERO_TO_IDX[scores[0][0]]

    # Take top-K candidates
    candidates = scores[:breadth]

    best_hero = candidates[0][0]
    best_wp = -1.0

    step_team, step_type = DRAFT_ORDER[state.step]

    for hero, _ in candidates:
        hero_idx = HERO_TO_IDX[hero]
        total_wp = 0.0

        for _ in range(rollouts_per_candidate):
            # Clone state, apply this hero
            rollout = state.clone()
            rollout.apply_action(hero_idx, step_team, step_type)

            # Roll out remaining steps with GD
            while not rollout.is_terminal():
                action = gd_sample(rollout, game_map, tier, temperature=gd_temp)
                rt, rtype = DRAFT_ORDER[rollout.step]
                rollout.apply_action(action, rt, rtype)

            # Evaluate terminal
            t0h = [HEROES[j] for j in range(NUM_HEROES) if rollout.team0_picks[j] > 0.5]
            t1h = [HEROES[j] for j in range(NUM_HEROES) if rollout.team1_picks[j] > 0.5]
            wp = evaluate_wp_sym(t0h, t1h, game_map, tier)
            # Normalize to our perspective
            total_wp += wp if our_team == 0 else 1.0 - wp

        avg_wp = total_wp / rollouts_per_candidate
        if avg_wp > best_wp:
            best_wp = avg_wp
            best_hero = hero

    return HERO_TO_IDX[best_hero]


# ── Draft quality metrics ────────────────────────────────────────────

def counter_delta(ha, hb, tier):
    raw = stats_cache.get_counter(ha, hb, tier)
    if raw is None:
        return None
    return raw - (stats_cache.get_hero_wr(ha, tier) + (100 - stats_cache.get_hero_wr(hb, tier)) - 50)


def synergy_delta(ha, hb, tier):
    raw = stats_cache.get_synergy(ha, hb, tier)
    if raw is None:
        return None
    return raw - (50 + (stats_cache.get_hero_wr(ha, tier) - 50) + (stats_cache.get_hero_wr(hb, tier) - 50))


def compute_metrics(pick_steps, tier):
    our = [(h, s) for h, tm, s in pick_steps if tm == 'ours']
    opp = [(h, s) for h, tm, s in pick_steps if tm == 'theirs']
    oh = [h for h, _ in our]

    # Resilience
    exp = []
    for h, os_ in our:
        sub = [x for x, s in opp if s > os_]
        if not sub:
            exp.append(0.0)
            continue
        ds = [d for d in (counter_delta(x, h, tier) for x in sub) if d is not None]
        exp.append(np.mean(ds) if ds else 0.0)
    re = -np.mean(exp[:2]) if len(exp) >= 2 else 0.0
    rl = -np.mean(exp[-2:]) if len(exp) >= 2 else 0.0

    # Counter
    ct = []
    for h, os_ in our:
        pr = [x for x, s in opp if s < os_]
        if not pr:
            ct.append(0.0)
            continue
        ds = [d for d in (counter_delta(h, x, tier) for x in pr) if d is not None]
        ct.append(np.mean(ds) if ds else 0.0)
    ca = np.mean(ct) if ct else 0.0
    cl = np.mean(ct[-2:]) if len(ct) >= 2 else 0.0

    # Synergy
    sy = []
    for i, h1 in enumerate(oh):
        for h2 in oh[i + 1:]:
            d = synergy_delta(h1, h2, tier)
            if d is not None:
                sy.append(d)
    ts = np.mean(sy) if sy else 0.0

    # Composition
    healer_set = set(h for h, r in HERO_ROLE_FINE.items() if r == 'healer')
    front_set = set(h for h, r in HERO_ROLE_FINE.items() if r in ('tank', 'bruiser'))
    ranged_set = set(h for h, r in HERO_ROLE_FINE.items() if r in ('ranged_aa', 'ranged_mage', 'pusher'))
    has_healer = any(h in healer_set for h in oh)
    has_front = any(h in front_set for h in oh)
    has_ranged = any(h in ranged_set for h in oh)
    roles = {}
    for h in oh:
        r = HERO_ROLE_FINE.get(h, 'x')
        roles[r] = roles.get(r, 0) + 1
    from shared import is_degenerate
    degen = is_degenerate(oh)

    return {'counter': ca, 'counter_late': cl, 'synergy': ts, 'resil_early': re, 'resil_late': rl,
            'healer': has_healer, 'degen': degen, 'heroes': oh}


# ── Run full draft simulation ────────────────────────────────────────

def run_draft(game_map, tier, our_team, breadth, gd_temp, counterability_map, rollouts=1):
    state = DraftState(game_map, tier, our_team=our_team)
    pick_steps = []

    while not state.is_terminal():
        step_team, step_type = DRAFT_ORDER[state.step]
        step_num = state.step

        if step_team == our_team:
            if step_type == 'ban':
                # Ban: use stats scoring
                mask = state.valid_mask_np()
                valid = [HEROES[i] for i in range(NUM_HEROES) if mask[i] > 0.5]
                our_picks = [HEROES[i] for i in range(NUM_HEROES) if
                             (state.team0_picks if our_team == 0 else state.team1_picks)[i] > 0.5]
                opp_picks = [HEROES[i] for i in range(NUM_HEROES) if
                             (state.team1_picks if our_team == 0 else state.team0_picks)[i] > 0.5]
                ban_scores = [(h, score_hero_for_ban(h, our_picks, opp_picks, game_map, tier))
                              for h in valid]
                ban_scores.sort(key=lambda x: -x[1])
                hero_idx = HERO_TO_IDX[ban_scores[0][0]]
            else:
                # Pick: stats + search
                hero_idx = search_pick(state, our_team, game_map, tier, breadth,
                                        gd_temp, counterability_map, rollouts)
        else:
            # Opponent: GD with temperature
            hero_idx = gd_sample(state, game_map, tier, temperature=gd_temp)

        if step_type == 'pick':
            team_label = 'ours' if step_team == our_team else 'theirs'
            pick_steps.append((HEROES[hero_idx], team_label, step_num))

        state.apply_action(hero_idx, step_team, step_type)

    return pick_steps


# ── Main benchmark ───────────────────────────────────────────────────

def main():
    N = 500
    random.seed(42)
    np.random.seed(42)

    configs = [(random.choice(MAPS), random.choice(SKILL_TIERS), random.randint(0, 1))
               for _ in range(N)]

    # Precompute counterability for all tiers
    ca_by_tier = {t: precompute_counterability(t) for t in SKILL_TIERS}
    print("  Counterability precomputed")

    # Configurations to sweep
    sweep = [
        # (label, breadth, gd_temp, use_counterability, rollouts_per_candidate)
        ("stats_only",           0, 1.0, False, 1),
        ("stats+ca",             0, 1.0, True,  1),
        ("search_b3_t1.0",      3, 1.0, False, 1),
        ("search_b3_t1.0+ca",   3, 1.0, True,  1),
        ("search_b5_t1.0",      5, 1.0, False, 1),
        ("search_b5_t0.5",      5, 0.5, False, 1),
        ("search_b5_t2.0",      5, 2.0, False, 1),
        ("search_b10_t1.0",    10, 1.0, False, 1),
        ("search_b10_t1.0+ca", 10, 1.0, True,  1),
        ("search_b5_t1.0_r3",   5, 1.0, False, 3),  # 3 rollouts per candidate
    ]

    print(f"\nBenchmark: {N} drafts × {len(sweep)} configs")
    print("=" * 110)

    all_results = {}
    for label, breadth, gd_temp, use_ca, rollouts in sweep:
        t0 = time.time()
        all_m = []
        hero_counter = Counter()

        for ci, (game_map, tier, our_team) in enumerate(configs):
            ca_map = ca_by_tier[tier] if use_ca else None
            pick_steps = run_draft(game_map, tier, our_team, breadth, gd_temp, ca_map, rollouts)
            m = compute_metrics(pick_steps, tier)
            all_m.append(m)
            for h in m['heroes']:
                hero_counter[h] += 1

            if (ci + 1) % 100 == 0 and breadth >= 5:
                elapsed = time.time() - t0
                eta = elapsed / (ci + 1) * (N - ci - 1)
                print(f"  {label}: {ci+1}/{N} ({eta:.0f}s remaining)")

        elapsed = time.time() - t0
        agg = {k: np.mean([m[k] for m in all_m]) for k in ['counter', 'counter_late', 'synergy', 'resil_early', 'resil_late']}
        agg['healer'] = np.mean([m['healer'] for m in all_m]) * 100
        agg['degen'] = np.mean([m['degen'] for m in all_m]) * 100
        agg['distinct'] = len(hero_counter)
        agg['time'] = elapsed
        all_results[label] = agg

        print(f"  {label:<25} ctr={agg['counter']:+.3f} ctrL={agg['counter_late']:+.3f} "
              f"syn={agg['synergy']:.3f} rE={agg['resil_early']:.3f} rL={agg['resil_late']:.3f} "
              f"hlr={agg['healer']:.0f}% deg={agg['degen']:.0f}% div={agg['distinct']} "
              f"({elapsed:.0f}s)")

    # Summary table
    print("\n\n" + "=" * 110)
    print(f"{'Config':<25} {'Ctr':>7} {'CtrL':>7} {'Syn':>7} {'R.Erly':>7} {'R.Late':>7} {'Hlr%':>5} {'Deg%':>5} {'Div':>4} {'Time':>6}")
    print("-" * 110)
    for label, agg in all_results.items():
        print(f"{label:<25} {agg['counter']:>+7.3f} {agg['counter_late']:>+7.3f} "
              f"{agg['synergy']:>7.3f} {agg['resil_early']:>7.3f} {agg['resil_late']:>7.3f} "
              f"{agg['healer']:>5.0f} {agg['degen']:>5.0f} {agg['distinct']:>4} "
              f"{agg['time']:>5.0f}s")
    print("-" * 110)
    # Reference baselines
    print(f"{'E baseline (MCTS)':25} {-0.082:>+7.3f} {-0.114:>+7.3f} {0.503:>7.3f} {0.026:>7.3f} {0.058:>7.3f} {'86':>5} {'26':>5} {'23':>4}")
    print(f"{'Hybrid sw=9':25} {+0.137:>+7.3f} {+0.368:>+7.3f} {1.071:>7.3f} {0.081:>7.3f} {0.063:>7.3f} {'64':>5} {'64':>5} {'66':>4}")
    print(f"{'Greedy enriched':25} {+0.305:>+7.3f} {+0.364:>+7.3f} {1.171:>7.3f} {-0.101:>7.3f} {-0.020:>7.3f} {'74':>5} {'55':>5} {'83':>4}")
    print("=" * 110)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "experiment_results", "stats_search")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "stats_search_benchmark.json")
    with open(out_path, "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_results.items()}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
