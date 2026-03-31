#!/usr/bin/env python3
"""
Expectimax trade space exploration.

Profiles tree size, timing, and draft quality across width/depth configurations.
Runs 50 drafts per config (enough to see trends, not full 500).

Reports: leaves/draft, seconds/draft, counter, synergy, healer%, degen%, avg WP.

Usage:
    set -a && source .env && set +a
    python3 -u training/benchmark_expectimax_tradespace.py
"""

import os, sys, json, random, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import (NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, MAP_TO_IDX,
                    SKILL_TIERS, TIER_TO_IDX, HERO_ROLE_FINE,
                    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
                    is_degenerate)
from sweep_enriched_wp import (StatsCache, WinProbEnrichedModel, FEATURE_GROUPS,
                                compute_group_indices, extract_features, FEATURE_GROUP_DIMS)
from train_draft_policy import DraftState, DRAFT_ORDER
from train_generic_draft import GenericDraftModel
from experiment_draft_quality import (draft_resilience, draft_counter_quality,
                                       incremental_synergy)

# ── Load models ──
print("Loading models...")
stats = StatsCache()
gi = compute_group_indices()
WP_GROUPS = ['role_counts', 'team_avg_wr', 'map_delta', 'pairwise_counters',
             'pairwise_synergies', 'counter_detail', 'meta_strength',
             'draft_diversity', 'comp_wr']
wp_cols = []
for g in WP_GROUPS:
    s, e = gi[g]; wp_cols.extend(range(s, e))
all_mask = [True] * len(FEATURE_GROUPS)
wp_dim = 197 + sum(FEATURE_GROUP_DIMS[g] for g in WP_GROUPS)

wp_model = WinProbEnrichedModel(wp_dim, [256, 128], dropout=0.3)
wp_model.load_state_dict(torch.load(
    os.path.join(os.path.dirname(__file__), "wp_experiment_enriched.pt"),
    weights_only=True, map_location="cpu"))
wp_model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
wp_model = wp_model.to(device)

gd_models = []
for i in range(5):
    gd = GenericDraftModel()
    gd.load_state_dict(torch.load(
        os.path.join(os.path.dirname(__file__), f"generic_draft_{i}.pt"),
        weights_only=True, map_location="cpu"))
    gd.eval()
    gd_models.append(gd)

# Composition data
HERO_TO_BLIZZ = {}
_RM = {"tank": "Tank", "bruiser": "Bruiser", "healer": "Healer",
       "ranged_aa": "Ranged Assassin", "ranged_mage": "Ranged Assassin",
       "melee_assassin": "Melee Assassin", "support_utility": "Support",
       "varian": "Bruiser", "pusher": "Ranged Assassin"}
for h, fr in HERO_ROLE_FINE.items():
    HERO_TO_BLIZZ[h] = _RM.get(fr, "Ranged Assassin")

comp_path = os.path.join(os.path.dirname(__file__), "..", "src", "lib", "data", "compositions.json")
COMP_DATA, BASELINE_WR = {}, {}
if os.path.exists(comp_path):
    for tn, comps in json.load(open(comp_path)).items():
        parsed = [{"roles": sorted(c["roles"]), "winRate": c["winRate"],
                    "games": c["games"], "popularity": c.get("popularity", c["games"])} for c in comps]
        COMP_DATA[tn] = parsed
        tw = sum(c["popularity"] for c in parsed)
        BASELINE_WR[tn] = sum(c["winRate"] * c["popularity"] for c in parsed) / tw if tw > 0 else 50.0

print(f"  All models loaded, device={device}")


# ── Scoring ──

def _subset(sub, sup):
    counts = {}
    for r in sup: counts[r] = counts.get(r, 0) + 1
    for r in sub:
        if counts.get(r, 0) <= 0: return False
        counts[r] -= 1
    return True

def score_pick(hero, our_picks, opp_picks, game_map, tier):
    wr = stats.get_hero_wr(hero, tier)
    md = stats.hero_map_wr.get(tier, {}).get(game_map, {}).get(hero)
    use_wr = md[0] if md and md[1] >= 50 else wr
    score = use_wr - 50.0
    cs, cn = 0.0, 0
    for opp in opp_picks:
        raw = stats.get_counter(hero, opp, tier)
        if raw is None: continue
        cs += raw - (use_wr + (100 - stats.get_hero_wr(opp, tier)) - 50); cn += 1
    if cn > 0: score += cs / cn
    ss, sn = 0.0, 0
    for ally in our_picks:
        raw = stats.get_synergy(hero, ally, tier)
        if raw is None: continue
        ss += raw - (50 + (use_wr - 50) + (stats.get_hero_wr(ally, tier) - 50)); sn += 1
    if sn > 0: score += ss / sn
    comps = COMP_DATA.get(tier, [])
    bl = BASELINE_WR.get(tier, 50.0)
    cr = HERO_TO_BLIZZ.get(hero)
    if cr and comps:
        combined = sorted([HERO_TO_BLIZZ.get(h, "Ranged Assassin") for h in our_picks if HERO_TO_BLIZZ.get(h)] + [cr])
        achievable = [c for c in comps if c["games"] >= 100 and _subset(combined, c["roles"])]
        sf = min(len(our_picks) / 4, 1.0)
        if not achievable:
            score += round(-15 * sf * 10) / 10
        else:
            bwr = max(min(c["winRate"], 50 + (c["winRate"]-50)*(min(c["games"],200)/200)) for c in achievable)
            score += round((bwr - bl) * sf * 10) / 10
    return score


def gd_predict_topn(state, game_map, tier, top_n):
    st, sty = DRAFT_ORDER[state.step]
    s = np.concatenate([state.team0_picks, state.team1_picks, state.bans,
                        map_to_one_hot(game_map), tier_to_one_hot(tier),
                        [state.step / 15.0, 0.0 if sty == 'ban' else 1.0]])
    mask = state.valid_mask_np()
    gd = random.choice(gd_models)
    with torch.no_grad():
        logits = gd(torch.tensor(s, dtype=torch.float32).unsqueeze(0),
                     torch.tensor(mask, dtype=torch.float32).unsqueeze(0))
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0).numpy()
    scored = [(HEROES[i], float(probs[i])) for i in range(NUM_HEROES) if mask[i] > 0.5 and probs[i] > 0.001]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]

def gd_sample(state, game_map, tier):
    st, sty = DRAFT_ORDER[state.step]
    s = np.concatenate([state.team0_picks, state.team1_picks, state.bans,
                        map_to_one_hot(game_map), tier_to_one_hot(tier),
                        [state.step / 15.0, 0.0 if sty == 'ban' else 1.0]])
    mask = state.valid_mask_np()
    gd = random.choice(gd_models)
    with torch.no_grad():
        logits = gd(torch.tensor(s, dtype=torch.float32).unsqueeze(0),
                     torch.tensor(mask, dtype=torch.float32).unsqueeze(0))
        probs = torch.nn.functional.softmax(logits, dim=1)
        return torch.multinomial(probs, 1).item()


def evaluate_wp_batch(drafts, tier):
    feats_n, feats_s = [], []
    for t0h, t1h, gm in drafts:
        dn = {'team0_heroes': t0h, 'team1_heroes': t1h, 'game_map': gm, 'skill_tier': tier, 'winner': 0}
        ds = {'team0_heroes': t1h, 'team1_heroes': t0h, 'game_map': gm, 'skill_tier': tier, 'winner': 0}
        bn, en = extract_features(dn, stats, all_mask)
        bs, es = extract_features(ds, stats, all_mask)
        feats_n.append(np.concatenate([bn, en[wp_cols]]))
        feats_s.append(np.concatenate([bs, es[wp_cols]]))
    xn = torch.tensor(np.array(feats_n), dtype=torch.float32).to(device)
    xs = torch.tensor(np.array(feats_s), dtype=torch.float32).to(device)
    with torch.no_grad():
        wn = wp_model(xn).cpu().numpy()
        ws = wp_model(xs).cpu().numpy()
    return (wn + (1.0 - ws)) / 2.0


# ── Expectimax with configurable widths ──

def expectimax_pick(state, our_team, game_map, tier, cfg):
    """Full expectimax at root: evaluate all root candidates."""
    step_team, step_type = DRAFT_ORDER[state.step]
    is_ban = step_type == 'ban'
    root_width = cfg['our_ban'] if is_ban else cfg['our_pick']

    mask = state.valid_mask_np()
    valid = [HEROES[i] for i in range(NUM_HEROES) if mask[i] > 0.5]
    our = [HEROES[i] for i in range(NUM_HEROES) if (state.team0_picks if our_team == 0 else state.team1_picks)[i] > 0.5]
    opp = [HEROES[i] for i in range(NUM_HEROES) if (state.team1_picks if our_team == 0 else state.team0_picks)[i] > 0.5]
    scored = [(h, score_pick(h, our, opp, game_map, tier)) for h in valid]
    scored.sort(key=lambda x: -x[1])
    candidates = [h for h, _ in scored[:root_width]]

    best_hero = candidates[0]
    best_value = -1e9
    total_leaves = 0

    for hero in candidates:
        child = state.clone()
        child.apply_action(HERO_TO_IDX[hero], step_team, step_type)
        leaves = []
        _collect_leaves(child, our_team, game_map, tier, cfg['depth'] - 1, 1.0, leaves, cfg)
        total_leaves += len(leaves)

        if not leaves:
            continue

        drafts = []
        for ls, _ in leaves:
            t0h = [HEROES[j] for j in range(NUM_HEROES) if ls.team0_picks[j] > 0.5]
            t1h = [HEROES[j] for j in range(NUM_HEROES) if ls.team1_picks[j] > 0.5]
            drafts.append((t0h, t1h, game_map))

        wp_batch = evaluate_wp_batch(drafts, tier)
        value = sum(w * (float(wp_batch[i]) if our_team == 0 else 1.0 - float(wp_batch[i]))
                    for i, (_, w) in enumerate(leaves)) / sum(w for _, w in leaves)

        if value > best_value:
            best_value = value
            best_hero = hero

    return HERO_TO_IDX[best_hero], total_leaves


def _collect_leaves(state, our_team, game_map, tier, depth, weight, leaves, cfg):
    if state.is_terminal() or depth <= 0:
        leaves.append((state, weight))
        return

    step_team, step_type = DRAFT_ORDER[state.step]
    is_ours = step_team == our_team
    is_ban = step_type == 'ban'

    if is_ours:
        # MAX: expand top-K candidates (not just 1)
        w = cfg['our_ban'] if is_ban else cfg['our_pick']
        mask = state.valid_mask_np()
        valid = [HEROES[i] for i in range(NUM_HEROES) if mask[i] > 0.5]
        our = [HEROES[i] for i in range(NUM_HEROES) if (state.team0_picks if our_team == 0 else state.team1_picks)[i] > 0.5]
        opp = [HEROES[i] for i in range(NUM_HEROES) if (state.team1_picks if our_team == 0 else state.team0_picks)[i] > 0.5]
        scored = [(h, score_pick(h, our, opp, game_map, tier)) for h in valid]
        scored.sort(key=lambda x: -x[1])
        # At inner MAX nodes, follow top candidate only to bound tree
        # At root, we already expand all candidates in expectimax_pick
        for h, _ in scored[:1]:
            child = state.clone()
            child.apply_action(HERO_TO_IDX[h], step_team, step_type)
            _collect_leaves(child, our_team, game_map, tier, depth - 1, weight, leaves, cfg)
    else:
        # CHANCE: branch on GD top-N
        w = cfg['opp_ban'] if is_ban else cfg['opp_pick']
        preds = gd_predict_topn(state, game_map, tier, w)
        preds = [(h, p) for h, p in preds if state.valid_mask_np()[HERO_TO_IDX[h]] > 0.5]
        if not preds:
            leaves.append((state, weight))
            return
        tp = sum(p for _, p in preds)
        for hero, prob in preds:
            child = state.clone()
            child.apply_action(HERO_TO_IDX[hero], step_team, step_type)
            _collect_leaves(child, our_team, game_map, tier, depth - 1,
                           weight * (prob / tp), leaves, cfg)


def simulate_draft(game_map, tier, our_team, strategy, cfg=None):
    state = DraftState(game_map, tier, our_team=our_team)
    pick_steps = []
    total_leaves = 0
    while not state.is_terminal():
        step_team, step_type = DRAFT_ORDER[state.step]
        if step_team == our_team:
            if step_type == 'ban':
                mask = state.valid_mask_np()
                valid = [HEROES[i] for i in range(NUM_HEROES) if mask[i] > 0.5]
                hero_idx = HERO_TO_IDX[max(valid, key=lambda h: stats.get_hero_wr(h, tier))]
            elif strategy == 'search':
                hero_idx, nl = expectimax_pick(state, our_team, game_map, tier, cfg)
                total_leaves += nl
            else:
                hero_idx = HERO_TO_IDX[max(
                    [HEROES[i] for i in range(NUM_HEROES) if state.valid_mask_np()[i] > 0.5],
                    key=lambda h: score_pick(h,
                        [HEROES[i] for i in range(NUM_HEROES) if (state.team0_picks if our_team==0 else state.team1_picks)[i]>0.5],
                        [HEROES[i] for i in range(NUM_HEROES) if (state.team1_picks if our_team==0 else state.team0_picks)[i]>0.5],
                        game_map, tier))]
        else:
            hero_idx = gd_sample(state, game_map, tier)
        if step_type == 'pick':
            pick_steps.append((HEROES[hero_idx], 'ours' if step_team == our_team else 'theirs', state.step))
        state.apply_action(hero_idx, step_team, step_type)
    return pick_steps, total_leaves


def run_config(label, strategy, cfg, configs, tier):
    t0 = time.time()
    all_m, all_wp_in, total_leaves = [], [], 0
    for game_map, _, our_team in configs:
        ps, nl = simulate_draft(game_map, tier, our_team, strategy, cfg)
        total_leaves += nl
        res = draft_resilience(ps, stats, tier)
        ctr = draft_counter_quality(ps, stats, tier)
        syn = incremental_synergy(ps, stats, tier)
        oh = [h for h, tm, _ in ps if tm == 'ours']
        oph = [h for h, tm, _ in ps if tm == 'theirs']
        all_m.append({
            'counter': ctr['avg_counter'], 'counter_late': ctr['late_counter'],
            'synergy': syn['team_synergy'],
            're': res['early_pick_resilience'], 'rl': res['late_pick_resilience'],
            'healer': any(HERO_ROLE_FINE.get(h) == 'healer' for h in oh),
            'degen': is_degenerate(oh), 'oh': oh, 'oph': oph,
        })
        t0h = oh if our_team == 0 else oph
        t1h = oph if our_team == 0 else oh
        all_wp_in.append((t0h, t1h, game_map, our_team))

    wps = evaluate_wp_batch([(a,b,c) for a,b,c,_ in all_wp_in], tier)
    wp_ours = [float(wps[i]) if ot==0 else 1.0-float(wps[i]) for i,(_, _, _, ot) in enumerate(all_wp_in)]
    elapsed = time.time() - t0
    n = len(configs)

    return {
        'label': label,
        'counter': np.mean([m['counter'] for m in all_m]),
        'counter_late': np.mean([m['counter_late'] for m in all_m]),
        'synergy': np.mean([m['synergy'] for m in all_m]),
        're': np.mean([m['re'] for m in all_m]),
        'rl': np.mean([m['rl'] for m in all_m]),
        'healer': np.mean([m['healer'] for m in all_m]) * 100,
        'degen': np.mean([m['degen'] for m in all_m]) * 100,
        'div': len(set(h for m in all_m for h in m['oh'])),
        'avg_wp': np.mean(wp_ours),
        'win_rate': np.mean([1 if w > 0.5 else 0 for w in wp_ours]) * 100,
        'leaves_per_draft': total_leaves / n if n > 0 else 0,
        'sec_per_draft': elapsed / n,
        'total_time': elapsed,
    }


def main():
    N = 50
    TIER = 'mid'
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    configs = [(random.choice(MAPS), TIER, random.randint(0, 1)) for _ in range(N)]

    print(f"\nExpectimax Trade Space: {N} drafts, tier={TIER}")
    print("=" * 130)

    # Configurations to test
    sweeps = [
        ("greedy",              'greedy', None),
        ("d4 opp3",             'search', {'depth': 4, 'our_pick': 8, 'our_ban': 4, 'opp_pick': 3, 'opp_ban': 3}),
        ("d4 opp5",             'search', {'depth': 4, 'our_pick': 8, 'our_ban': 4, 'opp_pick': 5, 'opp_ban': 3}),
        ("d4 opp8",             'search', {'depth': 4, 'our_pick': 8, 'our_ban': 4, 'opp_pick': 8, 'opp_ban': 4}),
        ("d6 opp3",             'search', {'depth': 6, 'our_pick': 8, 'our_ban': 4, 'opp_pick': 3, 'opp_ban': 3}),
        ("d6 opp5",             'search', {'depth': 6, 'our_pick': 8, 'our_ban': 4, 'opp_pick': 5, 'opp_ban': 3}),
        ("d6 opp8",             'search', {'depth': 6, 'our_pick': 8, 'our_ban': 4, 'opp_pick': 8, 'opp_ban': 4}),
        ("d8 opp3",             'search', {'depth': 8, 'our_pick': 8, 'our_ban': 4, 'opp_pick': 3, 'opp_ban': 3}),
        ("d8 opp5",             'search', {'depth': 8, 'our_pick': 8, 'our_ban': 4, 'opp_pick': 5, 'opp_ban': 3}),
        ("d6 opp5 our10",       'search', {'depth': 6, 'our_pick': 10, 'our_ban': 5, 'opp_pick': 5, 'opp_ban': 3}),
    ]

    results = []
    for label, strategy, cfg in sweeps:
        print(f"  Running: {label}...", end=" ", flush=True)
        r = run_config(label, strategy, cfg, configs, TIER)
        results.append(r)
        print(f"ctr={r['counter']:+.3f} syn={r['synergy']:.3f} "
              f"hlr={r['healer']:.0f}% deg={r['degen']:.0f}% "
              f"wp={r['avg_wp']:.4f} wr={r['win_rate']:.0f}% "
              f"leaves={r['leaves_per_draft']:.0f} {r['sec_per_draft']:.1f}s/draft")

    # Summary table
    print("\n\n" + "=" * 140)
    print(f"{'Config':<20} {'Ctr':>7} {'CtrL':>7} {'Syn':>7} {'R.E':>6} {'R.L':>6} "
          f"{'Hlr%':>5} {'Deg%':>5} {'Div':>4} {'AvgWP':>7} {'WR%':>5} "
          f"{'Leaves':>7} {'s/draft':>8}")
    print("-" * 140)
    for r in results:
        print(f"{r['label']:<20} {r['counter']:>+7.3f} {r['counter_late']:>+7.3f} "
              f"{r['synergy']:>7.3f} {r['re']:>6.3f} {r['rl']:>6.3f} "
              f"{r['healer']:>5.0f} {r['degen']:>5.0f} {r['div']:>4} "
              f"{r['avg_wp']:>7.4f} {r['win_rate']:>5.0f} "
              f"{r['leaves_per_draft']:>7.0f} {r['sec_per_draft']:>7.1f}s")
    print("=" * 140)


if __name__ == "__main__":
    main()
