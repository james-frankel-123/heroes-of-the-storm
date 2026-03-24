"""
Rich Draft Quality Evaluation Metrics.

Runs 200 drafts with 5 strategies and computes:
1. Counter-pick responsiveness
2. Synergy exploitation
3. Draft diversity (distinct heroes, entropy, top-10 concentration)
4. GD similarity (agreement with behavioral cloning)
5. Cross-model WP evaluation
6. Map-specific adaptation

Usage:
    set -a && source .env && set +a
    python3 -u training/experiment_rich_evaluation.py --drafts 200
"""
import os
import sys
import json
import random
import argparse
import time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS,
    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
    load_replay_data, split_data,
    HERO_ROLE_FINE,
)
from sweep_enriched_wp import (
    StatsCache, WinProbEnrichedModel, FEATURE_GROUPS,
    compute_group_indices, extract_features,
)
from train_draft_policy import DraftState, DRAFT_ORDER
from train_generic_draft import GenericDraftModel
from experiment_cql_draft import CQLDraftAgent, CQLDataset, replay_to_transitions
from experiment_synthetic_augmentation import ENRICHED_GROUPS, generate_synthetic_data
from experiment_synthetic_ablation2 import train_with_options

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "experiment_results", "rich_evaluation")


# ── Metric Functions ──

def counter_responsiveness(our_heroes, opp_heroes, stats, tier):
    deltas = []
    for our_h in our_heroes:
        our_wr = stats.get_hero_wr(our_h, tier)
        for opp_h in opp_heroes:
            raw = stats.get_counter(our_h, opp_h, tier)
            if raw is None:
                continue
            opp_wr = stats.get_hero_wr(opp_h, tier)
            expected = our_wr + (100 - opp_wr) - 50
            deltas.append(raw - expected)
    return np.mean(deltas) if deltas else 0.0


def synergy_exploitation(our_heroes, stats, tier):
    deltas = []
    for i, h1 in enumerate(our_heroes):
        wr1 = stats.get_hero_wr(h1, tier)
        for h2 in our_heroes[i+1:]:
            raw = stats.get_synergy(h1, h2, tier)
            if raw is None:
                continue
            wr2 = stats.get_hero_wr(h2, tier)
            expected = 50 + (wr1 - 50) + (wr2 - 50)
            deltas.append(raw - expected)
    return np.mean(deltas) if deltas else 0.0


def draft_diversity(all_drafts):
    hero_counts = Counter()
    total_picks = 0
    for draft in all_drafts:
        for hero in draft["our_picks"]:
            hero_counts[hero] += 1
            total_picks += 1
    num_distinct = len(hero_counts)
    probs = np.array(list(hero_counts.values())) / total_picks
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    top10 = sum(c for _, c in hero_counts.most_common(10))
    top10_pct = top10 / total_picks * 100
    return {
        "distinct_heroes": num_distinct,
        "entropy": round(entropy, 2),
        "top10_concentration": round(top10_pct, 1),
    }


def map_adaptation(all_drafts):
    map_hero_prefs = {
        "Braxis Holdout": ["Sonya", "Dehaka", "Malthael", "Leoric", "Thrall", "Yrel"],
        "Battlefield of Eternity": ["Valla", "Greymane", "Raynor", "Tychus", "Lunara"],
        "Cursed Hollow": ["Dehaka", "Falstad", "Brightwing", "Abathur"],
    }
    results = {}
    for game_map, pref_heroes in map_hero_prefs.items():
        pref_set = set(pref_heroes)
        map_drafts = [d for d in all_drafts if d["game_map"] == game_map]
        other_drafts = [d for d in all_drafts if d["game_map"] != game_map]
        if not map_drafts:
            continue
        map_rate = np.mean([any(h in pref_set for h in d["our_picks"]) for d in map_drafts])
        other_rate = np.mean([any(h in pref_set for h in d["our_picks"]) for d in other_drafts]) if other_drafts else 0
        results[game_map] = {
            "map_rate": round(map_rate * 100, 1),
            "other_rate": round(other_rate * 100, 1),
            "delta": round((map_rate - other_rate) * 100, 1),
        }
    return results


# ── Draft Runner ──

def run_drafts_with_strategy(strategy_fn, draft_configs, gd_models, stats, device):
    """
    Run drafts and record full step-by-step data for metrics.
    strategy_fn(state, step_team, step_type, game_map, tier, gd_models, device) -> hero_idx
    """
    healer_heroes = set(h for h, r in HERO_ROLE_FINE.items() if r == "healer")
    tank_heroes = set(h for h, r in HERO_ROLE_FINE.items() if r == "tank")
    bruiser_heroes = set(h for h, r in HERO_ROLE_FINE.items() if r == "bruiser")
    ranged_heroes = set(h for h, r in HERO_ROLE_FINE.items()
                        if r in ("ranged_aa", "ranged_mage", "pusher"))
    frontline = tank_heroes | bruiser_heroes

    all_drafts = []
    for di, (_, game_map, tier, our_team) in enumerate(draft_configs):
        state = DraftState(game_map, tier, our_team=our_team)
        steps = []

        while not state.is_terminal():
            step_team, step_type = DRAFT_ORDER[state.step]

            # Save state before action for GD similarity
            state_vec = np.concatenate([
                state.team0_picks.copy(), state.team1_picks.copy(), state.bans.copy(),
                map_to_one_hot(game_map), tier_to_one_hot(tier),
                [state.step / 15.0, 0.0 if step_type == "ban" else 1.0],
            ])
            mask_vec = state.valid_mask_np()

            if step_team == our_team:
                hero_idx = strategy_fn(state, step_team, step_type, game_map, tier, gd_models, device)
                steps.append({
                    "step": state.step, "team": step_team, "type": step_type,
                    "hero_idx": hero_idx, "state": state_vec, "mask": mask_vec,
                    "is_ours": True,
                })
                state.apply_action(hero_idx, step_team, step_type)
            else:
                # Opponent: GD sample
                gd = random.choice(gd_models)
                x = state.to_tensor_gd(torch.device("cpu"))
                mask = state.valid_mask(torch.device("cpu"))
                with torch.no_grad():
                    logits = gd(x, mask)
                    probs = F.softmax(logits / 1.0, dim=1)
                    hero_idx = torch.multinomial(probs, 1).item()
                steps.append({
                    "step": state.step, "team": step_team, "type": step_type,
                    "hero_idx": hero_idx, "state": state_vec, "mask": mask_vec,
                    "is_ours": False,
                })
                state.apply_action(hero_idx, step_team, step_type)

        # Extract our picks
        our_vec = state.team0_picks if our_team == 0 else state.team1_picks
        opp_vec = state.team1_picks if our_team == 0 else state.team0_picks
        our_picks = [HEROES[i] for i in range(NUM_HEROES) if our_vec[i] > 0]
        opp_picks = [HEROES[i] for i in range(NUM_HEROES) if opp_vec[i] > 0]

        has_healer = any(h in healer_heroes for h in our_picks)
        has_frontline = any(h in frontline for h in our_picks)
        has_ranged = any(h in ranged_heroes for h in our_picks)
        role_counts = {}
        for h in our_picks:
            r = HERO_ROLE_FINE.get(h, "unknown")
            role_counts[r] = role_counts.get(r, 0) + 1
        has_stacking = any(c >= 3 for c in role_counts.values())
        is_degen = not has_healer or not has_frontline or not has_ranged or has_stacking

        all_drafts.append({
            "game_map": game_map, "tier": tier, "our_team": our_team,
            "our_picks": our_picks, "opp_picks": opp_picks,
            "steps": steps,
            "has_healer": has_healer, "is_degen": is_degen,
            "terminal_t0": state.team0_picks.copy(),
            "terminal_t1": state.team1_picks.copy(),
            "terminal_bans": state.bans.copy(),
        })

        if (di + 1) % 50 == 0:
            n = di + 1
            hr = sum(1 for d in all_drafts if d["has_healer"]) / n * 100
            dr = sum(1 for d in all_drafts if d["is_degen"]) / n * 100
            print(f"    {n}/{len(draft_configs)}: healer={hr:.1f}% degen={dr:.1f}%")

    return all_drafts


# ── Strategy Functions ──

def make_gd_strategy():
    """Pure behavioral cloning (GD argmax)."""
    def strategy(state, team, step_type, game_map, tier, gd_models, device):
        gd = random.choice(gd_models)
        x = state.to_tensor_gd(torch.device("cpu"))
        mask = state.valid_mask(torch.device("cpu"))
        with torch.no_grad():
            logits = gd(x, mask)
            return logits.argmax(dim=1).item()
    return strategy


def make_cql_strategy(model_path, device):
    """CQL argmax Q."""
    model = CQLDraftAgent().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

    def strategy(state, team, step_type, game_map, tier, gd_models, dev):
        s = np.concatenate([
            state.team0_picks, state.team1_picks, state.bans,
            map_to_one_hot(game_map), tier_to_one_hot(tier),
            [state.step / 15.0, 0.0 if step_type == "ban" else 1.0],
        ])
        mask = state.valid_mask_np()
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
        m_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q = model(s_t, m_t).squeeze(0).cpu()
            return q.argmax().item()
    return strategy


def make_wp_greedy_strategy(wp_model, wp_groups, stats, group_indices, device):
    """WP greedy: rollout all candidates, pick best terminal WP."""
    all_mask = [True] * len(FEATURE_GROUPS)
    cols = []
    for g in wp_groups:
        s, e = group_indices[g]
        cols.extend(range(s, e))

    def eval_terminal(t0h, t1h, game_map, tier):
        d = {"team0_heroes": t0h, "team1_heroes": t1h,
             "game_map": game_map, "skill_tier": tier, "winner": 0}
        base, enriched = extract_features(d, stats, all_mask)
        enriched_sel = enriched[cols] if cols else np.array([], dtype=np.float32)
        x = np.concatenate([base, enriched_sel]) if len(enriched_sel) > 0 else base
        with torch.no_grad():
            return wp_model(torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)).item()

    def strategy(state, team, step_type, game_map, tier, gd_models, dev):
        mask = state.valid_mask_np()
        valid_idxs = [i for i in range(NUM_HEROES) if mask[i] > 0]
        best_hero = None
        best_wp = -1

        for hero_idx in valid_idxs:
            test_state = state.clone()
            test_state.apply_action(hero_idx, team, step_type)
            # Complete with GD
            s = test_state.clone()
            remaining_steps = DRAFT_ORDER[test_state.step:]
            for rs_team, rs_type in remaining_steps:
                gd = random.choice(gd_models)
                x = s.to_tensor_gd(torch.device("cpu"))
                mask = s.valid_mask(torch.device("cpu"))
                with torch.no_grad():
                    logits = gd(x, mask)
                    h = logits.argmax(dim=1).item()
                s.apply_action(h, rs_team, rs_type)

            t0h = [HEROES[i] for i in range(NUM_HEROES) if s.team0_picks[i] > 0]
            t1h = [HEROES[i] for i in range(NUM_HEROES) if s.team1_picks[i] > 0]
            wp = eval_terminal(t0h, t1h, game_map, tier)
            if state.our_team == 1:
                wp = 1 - wp

            if wp > best_wp:
                best_wp = wp
                best_hero = hero_idx

        return best_hero
    return strategy


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    data = load_replay_data()
    train_data, test_data = split_data(data)
    stats = StatsCache()
    group_indices = compute_group_indices()

    # Load GD models
    gd_models = []
    for i in range(5):
        gd_path = os.path.join(os.path.dirname(__file__), f"generic_draft_{i}.pt")
        if not os.path.exists(gd_path):
            gd_path = os.path.join(os.path.dirname(__file__), "generic_draft.pt")
        gd = GenericDraftModel()
        gd.load_state_dict(torch.load(gd_path, weights_only=True, map_location="cpu"))
        gd.cpu().eval()
        gd_models.append(gd)

    # Draft configs (same for all strategies)
    random.seed(42)
    draft_configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
                     for i in range(args.drafts)]

    # ═══════════════════════════════════════════════════════════════
    # Build strategies
    # ═══════════════════════════════════════════════════════════════
    strategies = {}

    # 1. GD baseline
    strategies["GD baseline"] = make_gd_strategy()

    # 2-3. CQL alpha=0.5 and 1.0
    for alpha in [0.5, 1.0]:
        path = os.path.join(os.path.dirname(__file__), "experiment_results", "cql",
                            f"_cql_temp_a{alpha}.pt")
        if os.path.exists(path):
            strategies[f"CQL a={alpha}"] = make_cql_strategy(path, device)
        else:
            print(f"  WARNING: {path} not found, skipping CQL a={alpha}")

    # 4. Enriched WP (256->128, no augmentation)
    enriched_path = os.path.join(os.path.dirname(__file__), "wp_experiment_enriched.pt")
    if os.path.exists(enriched_path):
        cols = []
        for g in ENRICHED_GROUPS:
            s, e = group_indices[g]
            cols.extend(range(s, e))
        dim = 197 + len(cols)
        enriched_model = WinProbEnrichedModel(dim, [256, 128], dropout=0.3).to(device)
        enriched_model.load_state_dict(torch.load(enriched_path, weights_only=True, map_location=device))
        enriched_model.eval()
        strategies["Enriched WP"] = make_wp_greedy_strategy(
            enriched_model, ENRICHED_GROUPS, stats, group_indices, device)

    # 5. Enriched + augmented (512->256->128)
    aug_path = os.path.join(os.path.dirname(__file__), "wp_enriched_winner.pt")
    if os.path.exists(aug_path):
        cols = []
        for g in ENRICHED_GROUPS:
            s, e = group_indices[g]
            cols.extend(range(s, e))
        dim = 197 + len(cols)
        aug_model = WinProbEnrichedModel(dim, [512, 256, 128], dropout=0.3).to(device)
        aug_model.load_state_dict(torch.load(aug_path, weights_only=True, map_location=device))
        aug_model.eval()
        strategies["Enriched+aug"] = make_wp_greedy_strategy(
            aug_model, ENRICHED_GROUPS, stats, group_indices, device)

    # ═══════════════════════════════════════════════════════════════
    # Run all strategies
    # ═══════════════════════════════════════════════════════════════
    all_results = {}
    t0 = time.time()

    for name, strategy_fn in strategies.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        random.seed(42)  # Reset seed for opponent consistency
        drafts = run_drafts_with_strategy(strategy_fn, draft_configs, gd_models, stats, device)
        all_results[name] = drafts

    elapsed = time.time() - t0
    print(f"\nAll drafts complete in {elapsed/60:.1f} minutes")

    # ═══════════════════════════════════════════════════════════════
    # Compute metrics
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("COMPUTING METRICS")
    print(f"{'='*70}")

    metrics = {}
    for name, drafts in all_results.items():
        print(f"\n  {name}...")
        n = len(drafts)
        healer_rate = sum(1 for d in drafts if d["has_healer"]) / n * 100
        degen_rate = sum(1 for d in drafts if d["is_degen"]) / n * 100

        # Counter responsiveness
        counters = [counter_responsiveness(d["our_picks"], d["opp_picks"], stats, d["tier"])
                    for d in drafts]
        avg_counter = np.mean(counters)

        # Synergy
        synergies = [synergy_exploitation(d["our_picks"], stats, d["tier"])
                     for d in drafts]
        avg_synergy = np.mean(synergies)

        # Diversity
        div = draft_diversity(drafts)

        # GD similarity
        gd_agree = 0
        gd_total = 0
        for draft in drafts:
            for step in draft["steps"]:
                if not step["is_ours"] or step["type"] == "ban":
                    continue
                s_t = torch.tensor(step["state"], dtype=torch.float32).unsqueeze(0)
                m_t = torch.tensor(step["mask"], dtype=torch.float32).unsqueeze(0)
                # Consensus of all 5 GD models
                votes = Counter()
                for gd in gd_models:
                    with torch.no_grad():
                        logits = gd(s_t, m_t)
                        votes[logits.argmax(dim=1).item()] += 1
                gd_consensus = votes.most_common(1)[0][0]
                if step["hero_idx"] == gd_consensus:
                    gd_agree += 1
                gd_total += 1
        gd_sim = gd_agree / gd_total * 100 if gd_total > 0 else 0

        # Map adaptation
        map_adapt = map_adaptation(drafts)

        # Cross-model WP (use enriched+aug model if available)
        if "Enriched+aug" in strategies and name != "Enriched+aug":
            wps = []
            for draft in drafts:
                t0h = [HEROES[i] for i in range(NUM_HEROES) if draft["terminal_t0"][i] > 0]
                t1h = [HEROES[i] for i in range(NUM_HEROES) if draft["terminal_t1"][i] > 0]
                d = {"team0_heroes": t0h, "team1_heroes": t1h,
                     "game_map": draft["game_map"], "skill_tier": draft["tier"], "winner": 0}
                all_mask = [True] * len(FEATURE_GROUPS)
                base, enriched = extract_features(d, stats, all_mask)
                cols = []
                for g in ENRICHED_GROUPS:
                    s, e = group_indices[g]
                    cols.extend(range(s, e))
                x = np.concatenate([base, enriched[cols]])
                with torch.no_grad():
                    wp = aug_model(torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)).item()
                if draft["our_team"] == 1:
                    wp = 1 - wp
                wps.append(wp)
            avg_aug_wp = np.mean(wps)
        else:
            avg_aug_wp = None

        metrics[name] = {
            "healer_rate": round(healer_rate, 1),
            "degen_rate": round(degen_rate, 1),
            "counter": round(avg_counter, 2),
            "synergy": round(avg_synergy, 2),
            **div,
            "gd_similarity": round(gd_sim, 1),
            "aug_wp": round(avg_aug_wp, 3) if avg_aug_wp is not None else "self",
            "map_adapt": map_adapt,
        }

    # ═══════════════════════════════════════════════════════════════
    # Print summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print("COMPREHENSIVE EVALUATION")
    print(f"{'='*100}")
    print(f"{'Strategy':<22} {'Heal%':>6} {'Deg%':>6} {'Counter':>8} {'Synergy':>8} "
          f"{'Distinct':>8} {'Entropy':>8} {'Top10%':>7} {'GD Sim%':>8} {'AugWP':>7}")
    print("-" * 100)
    for name in strategies:
        m = metrics[name]
        aug_wp_str = f"{m['aug_wp']:.3f}" if isinstance(m['aug_wp'], float) else m['aug_wp']
        print(f"{name:<22} {m['healer_rate']:>6.1f} {m['degen_rate']:>6.1f} "
              f"{m['counter']:>8.2f} {m['synergy']:>8.2f} "
              f"{m['distinct_heroes']:>8} {m['entropy']:>8.2f} "
              f"{m['top10_concentration']:>7.1f} {m['gd_similarity']:>8.1f} {aug_wp_str:>7}")

    # Map adaptation table
    print(f"\n{'='*80}")
    print("MAP ADAPTATION (% drafts with map-preferred hero, map vs other maps)")
    print(f"{'='*80}")
    maps_to_show = ["Braxis Holdout", "Battlefield of Eternity", "Cursed Hollow"]
    header = f"{'Strategy':<22}" + "".join(f"{m[:12]:>16}" for m in maps_to_show)
    print(header)
    print("-" * len(header))
    for name in strategies:
        vals = []
        for m in maps_to_show:
            adapt = metrics[name]["map_adapt"].get(m)
            if adapt:
                vals.append(f"{adapt['map_rate']:>5.1f}/{adapt['other_rate']:>4.1f} ({adapt['delta']:>+5.1f})")
            else:
                vals.append(f"{'N/A':>16}")
        print(f"{name:<22}" + "".join(f"{v:>16}" for v in vals))

    # Save
    save_path = os.path.join(RESULTS_DIR, "rich_evaluation_results.json")
    # Convert numpy types for JSON serialization
    def convert(o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2, default=convert)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
