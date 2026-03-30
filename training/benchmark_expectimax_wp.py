#!/usr/bin/env python3
"""
Benchmark: Stats greedy vs expectimax, evaluated with enriched WP model (PyTorch).

Simulates drafts using the stats engine (greedy and expectimax via the
Node.js benchmark), then evaluates all terminal states with the enriched
WP model on GPU in a single batched pass.

Reads draft results from the benchmark cache, or runs the Node.js benchmark
if cache doesn't exist.

Usage:
    set -a && source .env && set +a
    python3 -u training/benchmark_expectimax_wp.py
"""

import os, sys, json, random, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import (NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, MAP_TO_IDX,
                    SKILL_TIERS, TIER_TO_IDX, HERO_ROLE_FINE,
                    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
                    is_degenerate, load_replay_data)
from sweep_enriched_wp import (StatsCache, WinProbEnrichedModel, FEATURE_GROUPS,
                                compute_group_indices, extract_features, FEATURE_GROUP_DIMS)
from train_draft_policy import DraftState, DRAFT_ORDER, AlphaZeroDraftNet
from train_generic_draft import GenericDraftModel
from experiment_draft_quality import (counter_delta, synergy_delta,
                                       draft_resilience, draft_counter_quality,
                                       incremental_synergy)

# ── Config ──

N_DRAFTS = 500
TIER = 'mid'

# ── Load models ──

print("Loading models...")
stats = StatsCache()
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
wp_model = wp_model.to(device)
print(f"  WP model on {device}")

gd_models = []
for i in range(5):
    gd = GenericDraftModel()
    gd.load_state_dict(torch.load(
        os.path.join(os.path.dirname(__file__), f"generic_draft_{i}.pt"),
        weights_only=True, map_location="cpu"))
    gd.eval()
    gd_models.append(gd)
print(f"  {len(gd_models)} GD models loaded")


# ── WP evaluation (batched, GPU) ──

def evaluate_wp_batch(drafts, tier):
    """Evaluate a batch of drafts with the enriched WP model. Returns symmetrized WP for team0."""
    features = []
    for t0h, t1h, game_map in drafts:
        d = {'team0_heroes': t0h, 'team1_heroes': t1h,
             'game_map': game_map, 'skill_tier': tier, 'winner': 0}
        base, enriched = extract_features(d, stats, all_mask)
        features.append(np.concatenate([base, enriched[wp_cols]]))

    # Also compute swapped
    features_swap = []
    for t0h, t1h, game_map in drafts:
        d = {'team0_heroes': t1h, 'team1_heroes': t0h,
             'game_map': game_map, 'skill_tier': tier, 'winner': 0}
        base, enriched = extract_features(d, stats, all_mask)
        features_swap.append(np.concatenate([base, enriched[wp_cols]]))

    x = torch.tensor(np.array(features), dtype=torch.float32).to(device)
    x_swap = torch.tensor(np.array(features_swap), dtype=torch.float32).to(device)

    with torch.no_grad():
        wp_normal = wp_model(x).cpu().numpy()
        wp_swapped = wp_model(x_swap).cpu().numpy()

    # Symmetrize: P(t0 wins) = (normal + (1 - swapped)) / 2
    return (wp_normal + (1.0 - wp_swapped)) / 2.0


# ── Composition data (mirrors composition.ts) ──

HERO_TO_BLIZZ_ROLE = {}
_ROLE_MAP = {
    "tank": "Tank", "bruiser": "Bruiser", "healer": "Healer",
    "ranged_aa": "Ranged Assassin", "ranged_mage": "Ranged Assassin",
    "melee_assassin": "Melee Assassin", "support_utility": "Support",
    "varian": "Bruiser", "pusher": "Ranged Assassin",
}
for _h, _fr in HERO_ROLE_FINE.items():
    HERO_TO_BLIZZ_ROLE[_h] = _ROLE_MAP.get(_fr, "Ranged Assassin")

comp_path = os.path.join(os.path.dirname(__file__), "..", "src", "lib", "data", "compositions.json")
COMP_DATA = {}
BASELINE_COMP_WR = {}
if os.path.exists(comp_path):
    _raw_comp = json.load(open(comp_path))
    for _tn, _comps in _raw_comp.items():
        parsed = []
        for c in _comps:
            parsed.append({"roles": sorted(c["roles"]), "winRate": c["winRate"],
                            "games": c["games"], "popularity": c.get("popularity", c["games"])})
        COMP_DATA[_tn] = parsed
        # Baseline: popularity-weighted average WR
        tw = sum(c["popularity"] for c in parsed)
        BASELINE_COMP_WR[_tn] = sum(c["winRate"] * c["popularity"] for c in parsed) / tw if tw > 0 else 50.0
    print(f"  Composition data loaded for {len(COMP_DATA)} tiers")

MIN_COMP_GAMES = 100
COMP_CONFIDENCE_THRESHOLD = 200

def _confidence_wr(wr, games):
    if games >= COMP_CONFIDENCE_THRESHOLD: return wr
    w = games / COMP_CONFIDENCE_THRESHOLD
    return wr * w + 50 * (1 - w)

def _is_multiset_subset(subset, superset):
    counts = {}
    for r in superset: counts[r] = counts.get(r, 0) + 1
    for r in subset:
        if counts.get(r, 0) <= 0: return False
        counts[r] -= 1
    return True

def score_composition_for_hero(hero, our_picks, tier):
    """Score composition (mirrors scoreCompositionForHero in composition.ts)."""
    comps = COMP_DATA.get(tier, [])
    baseline = BASELINE_COMP_WR.get(tier, 50.0)
    if not comps: return 0.0

    candidate_role = HERO_TO_BLIZZ_ROLE.get(hero)
    if not candidate_role: return 0.0

    current_roles = [HERO_TO_BLIZZ_ROLE.get(h, "Ranged Assassin") for h in our_picks
                     if HERO_TO_BLIZZ_ROLE.get(h)]
    combined = sorted(current_roles + [candidate_role])
    picks_made = len(our_picks)

    # Find achievable compositions
    achievable = [c for c in comps
                  if c["games"] >= MIN_COMP_GAMES and _is_multiset_subset(combined, c["roles"])]

    scale = min(picks_made / 4, 1.0)

    if not achievable:
        penalty = -15 * scale
        return round(penalty * 10) / 10

    best_wr = max(_confidence_wr(c["winRate"], c["games"]) for c in achievable)
    delta = best_wr - baseline
    return round(delta * scale * 10) / 10


# ── Stats-based scoring (greedy pick) ──

def score_pick(hero, our_picks, opp_picks, game_map, tier):
    """Score hero for picking (mirrors engine.ts scoreHeroForPick exactly)."""
    wr = stats.get_hero_wr(hero, tier)
    map_data = stats.hero_map_wr.get(tier, {}).get(game_map, {}).get(hero)
    use_wr = map_data[0] if map_data and map_data[1] >= 50 else wr
    score = use_wr - 50.0

    # Counter (W.counter = 1.0)
    ctr_sum, ctr_n = 0.0, 0
    for opp in opp_picks:
        raw = stats.get_counter(hero, opp, tier)
        if raw is None: continue
        owr = stats.get_hero_wr(opp, tier)
        ctr_sum += raw - (use_wr + (100 - owr) - 50)
        ctr_n += 1
    if ctr_n > 0: score += ctr_sum / ctr_n

    # Synergy (W.synergy = 1.0)
    syn_sum, syn_n = 0.0, 0
    for ally in our_picks:
        raw = stats.get_synergy(hero, ally, tier)
        if raw is None: continue
        awr = stats.get_hero_wr(ally, tier)
        syn_sum += raw - (50 + (use_wr - 50) + (awr - 50))
        syn_n += 1
    if syn_n > 0: score += syn_sum / syn_n

    # Composition (W.comp = 1.0)
    comp_delta = score_composition_for_hero(hero, our_picks, tier)
    score += comp_delta

    return round(score * 10) / 10


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
        probs = torch.nn.functional.softmax(logits, dim=1)
        return torch.multinomial(probs, 1).item()


def greedy_pick(state, our_team, game_map, tier):
    mask = state.valid_mask_np()
    valid = [HEROES[i] for i in range(NUM_HEROES) if mask[i] > 0.5]
    our = [HEROES[i] for i in range(NUM_HEROES) if
           (state.team0_picks if our_team == 0 else state.team1_picks)[i] > 0.5]
    opp = [HEROES[i] for i in range(NUM_HEROES) if
           (state.team1_picks if our_team == 0 else state.team0_picks)[i] > 0.5]
    best = max(valid, key=lambda h: score_pick(h, our, opp, game_map, tier))
    return HERO_TO_IDX[best]


def greedy_ban(state, our_team, game_map, tier):
    mask = state.valid_mask_np()
    valid = [HEROES[i] for i in range(NUM_HEROES) if mask[i] > 0.5]
    best = max(valid, key=lambda h: stats.get_hero_wr(h, tier))
    return HERO_TO_IDX[best]


# ── 6-ply Expectimax with GD opponent model ──

# Width parameters
OUR_PICK_WIDTH = 8
OUR_BAN_WIDTH = 4
OPP_PICK_WIDTH = 6
OPP_BAN_WIDTH = 3
MAX_DEPTH = 6


def gd_predict_topn(state, game_map, tier, top_n):
    """Get GD model's top-N predictions with probabilities."""
    step_team, step_type = DRAFT_ORDER[state.step]
    s = np.concatenate([state.team0_picks, state.team1_picks, state.bans,
                        map_to_one_hot(game_map), tier_to_one_hot(tier),
                        [state.step / 15.0, 0.0 if step_type == 'ban' else 1.0]])
    mask = state.valid_mask_np()
    gd = random.choice(gd_models)
    with torch.no_grad():
        logits = gd(torch.tensor(s, dtype=torch.float32).unsqueeze(0),
                     torch.tensor(mask, dtype=torch.float32).unsqueeze(0))
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0).numpy()

    scored = []
    for i in range(NUM_HEROES):
        if mask[i] > 0.5 and probs[i] > 0.001:
            scored.append((HEROES[i], float(probs[i])))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]


def get_valid(state):
    mask = state.valid_mask_np()
    return [HEROES[i] for i in range(NUM_HEROES) if mask[i] > 0.5]


def prefilter_picks(state, our_team, game_map, tier, width):
    valid = get_valid(state)
    step_team = DRAFT_ORDER[state.step][0]
    is_ours = step_team == our_team
    our = [HEROES[i] for i in range(NUM_HEROES) if
           (state.team0_picks if our_team == 0 else state.team1_picks)[i] > 0.5]
    opp = [HEROES[i] for i in range(NUM_HEROES) if
           (state.team1_picks if our_team == 0 else state.team0_picks)[i] > 0.5]
    if is_ours:
        scored = [(h, score_pick(h, our, opp, game_map, tier)) for h in valid]
    else:
        scored = [(h, score_pick(h, opp, our, game_map, tier)) for h in valid]
    scored.sort(key=lambda x: -x[1])
    return [h for h, _ in scored[:width]]


def wp_leaf_eval(state, our_team, game_map, tier):
    """Evaluate a leaf state with the enriched WP model (symmetrized)."""
    t0h = [HEROES[j] for j in range(NUM_HEROES) if state.team0_picks[j] > 0.5]
    t1h = [HEROES[j] for j in range(NUM_HEROES) if state.team1_picks[j] > 0.5]
    if not t0h and not t1h:
        return 0.5

    d_n = {'team0_heroes': t0h, 'team1_heroes': t1h,
           'game_map': game_map, 'skill_tier': tier, 'winner': 0}
    d_s = {'team0_heroes': t1h, 'team1_heroes': t0h,
           'game_map': game_map, 'skill_tier': tier, 'winner': 0}
    base_n, enr_n = extract_features(d_n, stats, all_mask)
    base_s, enr_s = extract_features(d_s, stats, all_mask)
    x_n = torch.tensor(np.concatenate([base_n, enr_n[wp_cols]]), dtype=torch.float32).unsqueeze(0).to(device)
    x_s = torch.tensor(np.concatenate([base_s, enr_s[wp_cols]]), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        wp_n = wp_model(x_n).item()
        wp_s = wp_model(x_s).item()
    wp_t0 = (wp_n + (1.0 - wp_s)) / 2.0
    return wp_t0 if our_team == 0 else 1.0 - wp_t0


def expectimax(state, our_team, game_map, tier, depth, tt):
    """Recursive expectimax. Returns value from our_team's perspective."""
    if state.is_terminal() or depth <= 0:
        return wp_leaf_eval(state, our_team, game_map, tier)

    # Transposition table
    key = _hash_state(state, depth)
    if key in tt:
        return tt[key]

    step_team, step_type = DRAFT_ORDER[state.step]
    is_ours = step_team == our_team
    is_ban = step_type == 'ban'

    if is_ours:
        # MAX node
        width = OUR_BAN_WIDTH if is_ban else OUR_PICK_WIDTH
        candidates = prefilter_picks(state, our_team, game_map, tier, width)
        best = -1e9
        for hero in candidates:
            child = state.clone()
            child.apply_action(HERO_TO_IDX[hero], step_team, step_type)
            v = expectimax(child, our_team, game_map, tier, depth - 1, tt)
            if v > best:
                best = v
        value = best if best > -1e9 else wp_leaf_eval(state, our_team, game_map, tier)
    else:
        # CHANCE node — expected value over GD predictions
        width = OPP_BAN_WIDTH if is_ban else OPP_PICK_WIDTH
        predictions = gd_predict_topn(state, game_map, tier, width)
        # Filter taken heroes
        predictions = [(h, p) for h, p in predictions if state.valid_mask_np()[HERO_TO_IDX[h]] > 0.5]
        if not predictions:
            value = wp_leaf_eval(state, our_team, game_map, tier)
        else:
            total_prob = sum(p for _, p in predictions)
            value = 0.0
            for hero, prob in predictions:
                child = state.clone()
                child.apply_action(HERO_TO_IDX[hero], step_team, step_type)
                v = expectimax(child, our_team, game_map, tier, depth - 1, tt)
                value += (prob / total_prob) * v

    tt[key] = value
    return value


def _hash_state(state, depth):
    t0 = tuple(sorted(i for i in range(NUM_HEROES) if state.team0_picks[i] > 0.5))
    t1 = tuple(sorted(i for i in range(NUM_HEROES) if state.team1_picks[i] > 0.5))
    bans = tuple(sorted(i for i in range(NUM_HEROES) if state.bans[i] > 0.5))
    return (t0, t1, bans, state.step, depth)


def expectimax_pick(state, our_team, game_map, tier):
    """Pick the best hero using 6-ply expectimax search."""
    step_team, step_type = DRAFT_ORDER[state.step]
    is_ban = step_type == 'ban'
    width = OUR_BAN_WIDTH if is_ban else OUR_PICK_WIDTH
    candidates = prefilter_picks(state, our_team, game_map, tier, width)

    tt = {}
    best_hero = candidates[0]
    best_value = -1e9

    for hero in candidates:
        child = state.clone()
        child.apply_action(HERO_TO_IDX[hero], step_team, step_type)
        v = expectimax(child, our_team, game_map, tier, MAX_DEPTH - 1, tt)
        if v > best_value:
            best_value = v
            best_hero = hero

    return HERO_TO_IDX[best_hero]


# ── Draft simulation ──

DRAFT_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
DRAFT_IS_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]

def simulate_draft(game_map, tier, our_team, strategy='greedy'):
    state = DraftState(game_map, tier, our_team=our_team)
    pick_steps = []

    while not state.is_terminal():
        step_team, step_type = DRAFT_ORDER[state.step]
        step_num = state.step

        if step_team == our_team:
            if step_type == 'ban':
                hero_idx = greedy_ban(state, our_team, game_map, tier)
            elif strategy == 'search':
                hero_idx = expectimax_pick(state, our_team, game_map, tier)
            else:
                hero_idx = greedy_pick(state, our_team, game_map, tier)
        else:
            hero_idx = gd_sample(state, game_map, tier)

        if step_type == 'pick':
            team_label = 'ours' if step_team == our_team else 'theirs'
            pick_steps.append((HEROES[hero_idx], team_label, step_num))

        state.apply_action(hero_idx, step_team, step_type)

    return pick_steps


def compute_metrics(pick_steps, game_map, tier):
    our = [(h, s) for h, tm, s in pick_steps if tm == 'ours']
    opp = [(h, s) for h, tm, s in pick_steps if tm == 'theirs']
    oh = [h for h, _ in our]
    opph = [h for h, _ in opp]

    # Resilience early/late
    res = draft_resilience(pick_steps, stats, tier)

    # Counter
    ctr = draft_counter_quality(pick_steps, stats, tier)

    # Synergy
    syn = incremental_synergy(pick_steps, stats, tier)

    return {
        'counter': ctr['avg_counter'],
        'counter_late': ctr['late_counter'],
        'synergy': syn['team_synergy'],
        'resil_early': res['early_pick_resilience'],
        'resil_late': res['late_pick_resilience'],
        'healer': any(HERO_ROLE_FINE.get(h) == 'healer' for h in oh),
        'degen': is_degenerate(oh),
        'our_heroes': oh,
        'opp_heroes': opph,
        'map': game_map,
    }


# ── Main ──

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    configs = [(random.choice(MAPS), TIER, random.randint(0, 1)) for _ in range(N_DRAFTS)]

    print(f"\nBenchmark: {N_DRAFTS} drafts, tier={TIER}, enriched WP model on {device}")
    print("=" * 90)

    for strategy in ['greedy', 'search']:
        t0 = time.time()
        all_metrics = []
        all_drafts_for_wp = []  # (t0h, t1h, map) for batch WP eval

        for ci, (game_map, tier, our_team) in enumerate(configs):
            pick_steps = simulate_draft(game_map, tier, our_team, strategy)
            m = compute_metrics(pick_steps, game_map, tier)
            all_metrics.append(m)

            # Collect for batch WP
            t0h = m['our_heroes'] if our_team == 0 else m['opp_heroes']
            t1h = m['opp_heroes'] if our_team == 0 else m['our_heroes']
            all_drafts_for_wp.append((t0h, t1h, game_map, our_team))

            if (ci + 1) % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (ci + 1) * (N_DRAFTS - ci - 1)
                print(f"  {strategy}: {ci+1}/{N_DRAFTS} ({eta:.0f}s left)")

        # Batch WP evaluation on GPU
        wp_inputs = [(t0h, t1h, gm) for t0h, t1h, gm, _ in all_drafts_for_wp]
        wp_t0_batch = evaluate_wp_batch(wp_inputs, TIER)

        # Convert to our-team perspective
        wp_ours = []
        for i, (_, _, _, our_team) in enumerate(all_drafts_for_wp):
            wp = wp_t0_batch[i] if our_team == 0 else 1.0 - wp_t0_batch[i]
            wp_ours.append(float(wp))

        elapsed = time.time() - t0

        # Aggregate
        n = len(all_metrics)
        avg_wp = np.mean(wp_ours)
        win_rate = np.mean([1 if w > 0.5 else 0 for w in wp_ours]) * 100

        agg = {
            'counter': np.mean([m['counter'] for m in all_metrics]),
            'counter_late': np.mean([m['counter_late'] for m in all_metrics]),
            'synergy': np.mean([m['synergy'] for m in all_metrics]),
            'resil_early': np.mean([m['resil_early'] for m in all_metrics]),
            'resil_late': np.mean([m['resil_late'] for m in all_metrics]),
            'healer': np.mean([m['healer'] for m in all_metrics]) * 100,
            'degen': np.mean([m['degen'] for m in all_metrics]) * 100,
            'distinct': len(set(h for m in all_metrics for h in m['our_heroes'])),
            'avg_wp': avg_wp,
            'win_rate': win_rate,
        }

        print(f"\n  {strategy}:")
        print(f"    Counter:    avg={agg['counter']:+.3f}  late={agg['counter_late']:+.3f}")
        print(f"    Synergy:    {agg['synergy']:.3f}")
        print(f"    Resilience: early={agg['resil_early']:.3f}  late={agg['resil_late']:.3f}")
        print(f"    Comp:       healer={agg['healer']:.0f}%  degen={agg['degen']:.0f}%  div={agg['distinct']}")
        print(f"    WP (enr.):  avg={agg['avg_wp']:.4f}  win_rate={agg['win_rate']:.1f}%")
        print(f"    Time:       {elapsed:.0f}s")

    print("\n" + "=" * 90)
    print("  E MCTS ref:  avg_wp=0.553  wr=91%  ctr=-0.08  syn=0.50  hlr=86%  deg=26%")
    print("=" * 90)


if __name__ == "__main__":
    main()
