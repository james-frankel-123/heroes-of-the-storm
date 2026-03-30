#!/usr/bin/env python3
"""
Step 1: WP Model Counter/Synergy Sensitivity.

For partial draft states sampled from real replays, does the WP model
assign higher scores to heroes that counter the opponent and synergize
with teammates?

Usage:
    set -a && source .env && set +a
    python3 -u training/validate_wp_sensitivity.py --replays 500
"""

import os, sys, json, random, argparse, time
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import (NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, MAP_TO_IDX,
                    SKILL_TIERS, TIER_TO_IDX,
                    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
                    load_replay_data)
from sweep_enriched_wp import (StatsCache, WinProbEnrichedModel, FEATURE_GROUPS,
                                compute_group_indices, extract_features, FEATURE_GROUP_DIMS)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "experiment_results", "diagnostics")

# Draft sequence: team assignments (0-indexed)
DRAFT_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
DRAFT_IS_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]


def counter_delta(hero_a, hero_b, stats, tier):
    raw = stats.get_counter(hero_a, hero_b, tier)
    if raw is None:
        return None
    wr_a = stats.get_hero_wr(hero_a, tier)
    wr_b = stats.get_hero_wr(hero_b, tier)
    return raw - (wr_a + (100 - wr_b) - 50)


def synergy_delta(hero_a, hero_b, stats, tier):
    raw = stats.get_synergy(hero_a, hero_b, tier)
    if raw is None:
        return None
    wr_a = stats.get_hero_wr(hero_a, tier)
    wr_b = stats.get_hero_wr(hero_b, tier)
    return raw - (50 + (wr_a - 50) + (wr_b - 50))


def pearsonr(x, y):
    """Simple Pearson r without scipy."""
    x = np.array(x)
    y = np.array(y)
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    denom = np.sqrt((dx**2).sum() * (dy**2).sum())
    if denom == 0:
        return 0.0, 1.0
    r = (dx * dy).sum() / denom
    n = len(x)
    if n <= 2:
        return r, 1.0
    t = r * np.sqrt((n - 2) / (1 - r**2 + 1e-12))
    # Approximate p-value (two-tailed, using normal for large n)
    p = 2 * np.exp(-0.5 * t**2) / np.sqrt(2 * np.pi) if abs(t) < 30 else 0.0
    return r, p


def evaluate_wp_sym(wp_model, t0_heroes, t1_heroes, game_map, tier, stats, wp_cols, all_mask):
    """Evaluate WP model symmetrized."""
    def _run(t0, t1):
        d = {'team0_heroes': t0, 'team1_heroes': t1,
             'game_map': game_map, 'skill_tier': tier, 'winner': 0}
        base, enriched = extract_features(d, stats, all_mask)
        x = np.concatenate([base, enriched[wp_cols]])
        with torch.no_grad():
            return wp_model(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()
    wp_n = _run(t0_heroes, t1_heroes)
    wp_s = _run(t1_heroes, t0_heroes)
    return (wp_n + (1.0 - wp_s)) / 2.0


def run_sensitivity(wp_model, wp_name, stats, wp_cols, all_mask, replays, n_replays):
    """Run WP sensitivity analysis on sampled replay states."""
    sampled = random.sample(replays, min(n_replays, len(replays)))

    # Results keyed by pick step index
    counter_results = defaultdict(list)  # step -> list of (r, n_candidates)
    synergy_results = defaultdict(list)

    for ri, replay in enumerate(sampled):
        draft_order = replay['draft_order']
        if len(draft_order) != 16:
            continue
        game_map = replay['game_map']
        tier = replay['skill_tier']
        t0_set = set(replay['team0_heroes'])
        t1_set = set(replay['team1_heroes'])

        # Walk through draft, tracking state
        t0_picks = []
        t1_picks = []
        bans = set()
        taken = set()

        for step_idx, step in enumerate(draft_order):
            hero = step['hero']
            is_pick = step['type'] == '1'

            if not is_pick:
                bans.add(hero)
                taken.add(hero)
                continue

            # Determine which team this hero belongs to
            if hero in t0_set:
                pick_team = 0
            elif hero in t1_set:
                pick_team = 1
            else:
                taken.add(hero)
                continue

            # Analyze this state from team 0's perspective at pick steps
            # Only when team 0 is picking and there are opponent picks to counter
            if pick_team == 0 and len(t1_picks) > 0:
                valid_heroes = [h for h in HEROES if h not in taken]

                # Score each candidate
                wp_scores = []
                c_deltas = []
                s_deltas = []
                heroes_with_counter = []
                heroes_with_synergy = []

                for cand in valid_heroes:
                    # WP with this candidate added to team 0
                    t0_try = t0_picks + [cand]
                    wp = evaluate_wp_sym(wp_model, t0_try, t1_picks,
                                         game_map, tier, stats, wp_cols, all_mask)

                    # Counter delta vs current opponents
                    cds = [counter_delta(cand, opp, stats, tier) for opp in t1_picks]
                    cds = [d for d in cds if d is not None]
                    avg_cd = np.mean(cds) if cds else None

                    # Synergy delta with current teammates
                    sds = [synergy_delta(cand, tm, stats, tier) for tm in t0_picks]
                    sds = [d for d in sds if d is not None]
                    avg_sd = np.mean(sds) if sds else None

                    wp_scores.append(wp)
                    if avg_cd is not None:
                        c_deltas.append(avg_cd)
                        heroes_with_counter.append(len(wp_scores) - 1)
                    if avg_sd is not None:
                        s_deltas.append(avg_sd)
                        heroes_with_synergy.append(len(wp_scores) - 1)

                # Correlations
                if len(heroes_with_counter) > 10:
                    wps_c = [wp_scores[i] for i in heroes_with_counter]
                    r, p = pearsonr(wps_c, c_deltas)
                    counter_results[step_idx].append((r, p, len(heroes_with_counter)))

                if len(heroes_with_synergy) > 10 and len(t0_picks) > 0:
                    wps_s = [wp_scores[i] for i in heroes_with_synergy]
                    r, p = pearsonr(wps_s, s_deltas)
                    synergy_results[step_idx].append((r, p, len(heroes_with_synergy)))

            # Advance state
            if pick_team == 0:
                t0_picks.append(hero)
            else:
                t1_picks.append(hero)
            taken.add(hero)

        if (ri + 1) % 100 == 0:
            print(f"    {wp_name}: {ri+1}/{len(sampled)} replays")

    return counter_results, synergy_results


def print_results(counter_results, synergy_results, wp_name):
    """Print table of results by pick step."""
    all_pick_steps = sorted(set(list(counter_results.keys()) + list(synergy_results.keys())))

    print(f"\n{'='*90}")
    print(f"  {wp_name}")
    print(f"{'='*90}")
    print(f"{'Step':>5} | {'N States':>8} | {'Avg r(WP,Ctr)':>14} | {'% Sig(Ctr)':>10} | "
          f"{'Avg r(WP,Syn)':>14} | {'% Sig(Syn)':>10}")
    print("-" * 90)

    # Also collect overall
    all_r_counter = []
    all_r_synergy = []

    for step in all_pick_steps:
        cr = counter_results.get(step, [])
        sr = synergy_results.get(step, [])

        n_c = len(cr)
        n_s = len(sr)
        n = max(n_c, n_s)

        avg_r_c = np.mean([r for r, p, nc in cr]) if cr else float('nan')
        pct_sig_c = np.mean([1 if p < 0.05 else 0 for r, p, nc in cr]) * 100 if cr else float('nan')

        avg_r_s = np.mean([r for r, p, nc in sr]) if sr else float('nan')
        pct_sig_s = np.mean([1 if p < 0.05 else 0 for r, p, nc in sr]) * 100 if sr else float('nan')

        if cr:
            all_r_counter.extend([r for r, p, nc in cr])
        if sr:
            all_r_synergy.extend([r for r, p, nc in sr])

        print(f"{step:>5} | {n:>8} | {avg_r_c:>+14.4f} | {pct_sig_c:>9.0f}% | "
              f"{avg_r_s:>+14.4f} | {pct_sig_s:>9.0f}%")

    print("-" * 90)
    overall_c = np.mean(all_r_counter) if all_r_counter else 0.0
    overall_s = np.mean(all_r_synergy) if all_r_synergy else 0.0
    sig_c = np.mean([1 if abs(r) > 0.15 else 0 for r in all_r_counter]) * 100 if all_r_counter else 0
    sig_s = np.mean([1 if abs(r) > 0.15 else 0 for r in all_r_synergy]) * 100 if all_r_synergy else 0
    print(f"{'ALL':>5} | {len(all_r_counter):>8} | {overall_c:>+14.4f} | {sig_c:>9.0f}% | "
          f"{overall_s:>+14.4f} | {sig_s:>9.0f}%")

    return overall_c, overall_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", type=int, default=500)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    replays = load_replay_data()
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

    # Load both WP models
    models = {}

    wp_enr = WinProbEnrichedModel(wp_input_dim, [256, 128], dropout=0.3)
    wp_enr.load_state_dict(torch.load(
        os.path.join(os.path.dirname(__file__), "wp_experiment_enriched.pt"),
        weights_only=True, map_location="cpu"))
    wp_enr.eval()
    models["Enriched WP (256→128)"] = wp_enr

    wp_aug = WinProbEnrichedModel(wp_input_dim, [512, 256, 128], dropout=0.3)
    wp_aug.load_state_dict(torch.load(
        os.path.join(os.path.dirname(__file__), "wp_enriched_winner.pt"),
        weights_only=True, map_location="cpu"))
    wp_aug.eval()
    models["Augmented WP (512→256→128)"] = wp_aug

    print(f"Running sensitivity analysis on {args.replays} replays...")
    t0 = time.time()

    all_results = {}
    for wp_name, wp_model in models.items():
        print(f"\n  Testing: {wp_name}")
        cr, sr = run_sensitivity(wp_model, wp_name, stats, wp_cols, all_mask,
                                  replays, args.replays)
        avg_c, avg_s = print_results(cr, sr, wp_name)
        all_results[wp_name] = {"avg_counter_r": avg_c, "avg_synergy_r": avg_s}

    elapsed = time.time() - t0
    print(f"\nComplete in {elapsed / 60:.1f} minutes")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    for name, res in all_results.items():
        c = res["avg_counter_r"]
        s = res["avg_synergy_r"]
        print(f"\n{name}:")
        if c > 0.15:
            print(f"  Counter: r={c:+.3f} — STRONG signal. WP model differentiates counter-picks.")
        elif c > 0.05:
            print(f"  Counter: r={c:+.3f} — WEAK signal. WP model has some counter sensitivity.")
        else:
            print(f"  Counter: r={c:+.3f} — NO signal. WP model doesn't differentiate counters.")

        if s > 0.15:
            print(f"  Synergy: r={s:+.3f} — STRONG signal. WP model differentiates synergy picks.")
        elif s > 0.05:
            print(f"  Synergy: r={s:+.3f} — WEAK signal. WP model has some synergy sensitivity.")
        else:
            print(f"  Synergy: r={s:+.3f} — NO signal. WP model doesn't differentiate synergy.")

    # Save
    out_path = os.path.join(RESULTS_DIR, "wp_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
