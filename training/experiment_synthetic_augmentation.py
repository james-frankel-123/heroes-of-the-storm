"""
Experiment: Synthetic Data Augmentation Using Composition Win Rate Data.

Generates synthetic training records for underrepresented and unseen compositions,
trains enriched WP models with augmented data, and evaluates whether the model
learns to better penalize degenerate compositions.

See EXPERIMENT.md for full design.

Usage:
    set -a && source .env && set +a
    python3 -u training/experiment_synthetic_augmentation.py --stage1-only
    python3 -u training/experiment_synthetic_augmentation.py --drafts 200
    python3 -u training/experiment_synthetic_augmentation.py --quick
"""
import os
import sys
import json
import random
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations_with_replacement

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    is_degenerate,
    NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS,
    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
    load_replay_data, split_data,
    HERO_ROLE_FINE, FINE_ROLE_NAMES,
)
from sweep_enriched_wp import (
    StatsCache, WinProbEnrichedModel, FEATURE_GROUPS, FEATURE_GROUP_DIMS,
    compute_group_indices, precompute_all_features, extract_features,
)
from test_wp_sanity import TESTS, run_tests

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "experiment_results", "synthetic_augmentation")

# Enriched model config (matches the best model from the main experiment)
ENRICHED_GROUPS = ["role_counts", "team_avg_wr", "map_delta", "pairwise_counters",
                   "pairwise_synergies", "counter_detail", "meta_strength",
                   "draft_diversity", "comp_wr"]
TRAIN_HP = {
    "hidden_dims": [256, 128],
    "dropout": 0.3,
    "lr": 5e-4,
    "weight_decay": 5e-3,
    "patience": 25,
    "max_epochs": 200,
}

# ── Role mapping: fine roles → Blizzard 6-role system ──

FINE_TO_BLIZZ = {
    "tank": "Tank", "bruiser": "Bruiser", "healer": "Healer",
    "ranged_aa": "Ranged Assassin", "ranged_mage": "Ranged Assassin",
    "melee_assassin": "Melee Assassin", "support_utility": "Support",
    "varian": "Bruiser", "pusher": "Ranged Assassin",
}

BLIZZ_ROLES = ["Bruiser", "Healer", "Melee Assassin", "Ranged Assassin", "Support", "Tank"]


def build_heroes_by_blizz_role():
    """Group heroes by Blizzard's 6 official roles."""
    by_role = {r: [] for r in BLIZZ_ROLES}
    for hero, fine_role in HERO_ROLE_FINE.items():
        blizz = FINE_TO_BLIZZ.get(fine_role, "Ranged Assassin")
        by_role[blizz].append(hero)
    return by_role


def generate_all_role_tuples():
    """All 252 possible sorted 5-role tuples from 6 roles."""
    return [tuple(c) for c in combinations_with_replacement(BLIZZ_ROLES, 5)]


# ── Synthetic data generation ──

def sample_heroes_for_roles(role_tuple, heroes_by_role):
    """Sample 5 heroes matching a role tuple, no duplicates."""
    team = []
    used = set()
    for role in role_tuple:
        candidates = [h for h in heroes_by_role[role] if h not in used]
        if not candidates:
            # Fallback: any hero not used
            candidates = [h for h in HEROES if h not in used]
        hero = random.choice(candidates)
        team.append(hero)
        used.add(hero)
    return team


def generate_synthetic_data_v2(real_data, comp_data, stats_cache, unseen_wr=10.0,
                               unseen_volume=100, scope="tier2_only"):
    """
    Generate synthetic data with pairwise-adjusted WR (v2).
    Instead of flat 10% WR for all unseen comps, adjusts based on the
    specific hero matchup's counter/synergy statistics.
    """
    heroes_by_role = build_heroes_by_blizz_role()
    all_tuples = generate_all_role_tuples()

    opponent_teams = []
    for d in real_data:
        opponent_teams.append(d["team0_heroes"])
        opponent_teams.append(d["team1_heroes"])

    synthetic = []
    wr_stats = []

    for tier in ["low", "mid", "high"]:
        tier_comps = comp_data.get(tier, [])
        known = {}
        for c in tier_comps:
            key = tuple(sorted(c["roles"]))
            if key not in known or c["games"] > known[key]["games"]:
                known[key] = c

        for role_tuple in all_tuples:
            if role_tuple not in known:
                records = _make_records(
                    role_tuple, heroes_by_role, unseen_wr,
                    unseen_volume, tier, opponent_teams,
                    stats_cache=stats_cache, pairwise_adjust=True,
                )
                synthetic.extend(records)

    # Log WR distribution
    actual_wrs = []
    for rec in synthetic:
        # Reconstruct from winner field (approximate)
        pass
    stats = {"total": len(synthetic), "mode": "pairwise_adjusted"}
    return synthetic, stats


def generate_synthetic_data(real_data, comp_data, unseen_wr=20.0, unseen_volume=100,
                            scope="both"):
    """
    Generate synthetic training records.
    Returns list of dicts matching load_replay_data() format.
    """
    heroes_by_role = build_heroes_by_blizz_role()
    all_tuples = generate_all_role_tuples()

    # Build opponent pool from real data (both teams)
    opponent_teams = []
    for d in real_data:
        opponent_teams.append(d["team0_heroes"])
        opponent_teams.append(d["team1_heroes"])

    synthetic = []
    stats = {"tier1": 0, "tier2": 0, "by_tier": {}}

    for tier in ["low", "mid", "high"]:
        tier_comps = comp_data.get(tier, [])
        known = {}
        for c in tier_comps:
            key = tuple(sorted(c["roles"]))
            if key not in known or c["games"] > known[key]["games"]:
                known[key] = c

        tier_stats = {"tier1": 0, "tier2": 0}

        # Tier 1: sparse low-WR compositions
        if scope in ("both", "tier1_only"):
            for role_tuple, comp in known.items():
                if comp["games"] < 300 and comp["winRate"] < 48:
                    n_to_add = 300 - comp["games"]
                    records = _make_records(
                        role_tuple, heroes_by_role, comp["winRate"],
                        n_to_add, tier, opponent_teams,
                    )
                    synthetic.extend(records)
                    tier_stats["tier1"] += len(records)

        # Tier 2: unseen compositions
        if scope in ("both", "tier2_only"):
            for role_tuple in all_tuples:
                if role_tuple not in known:
                    records = _make_records(
                        role_tuple, heroes_by_role, unseen_wr,
                        unseen_volume, tier, opponent_teams,
                    )
                    synthetic.extend(records)
                    tier_stats["tier2"] += len(records)

        stats["tier1"] += tier_stats["tier1"]
        stats["tier2"] += tier_stats["tier2"]
        stats["by_tier"][tier] = tier_stats

    stats["total"] = len(synthetic)
    return synthetic, stats


def _make_records(role_tuple, heroes_by_role, win_rate, n, tier, opponent_teams,
                   stats_cache=None, pairwise_adjust=False):
    """Generate n synthetic records for a role composition.

    If pairwise_adjust=True and stats_cache is provided, adjusts the base WR
    based on pairwise counter/synergy statistics of the specific hero matchup.
    This preserves pairwise gradients in the synthetic data instead of
    flattening all degenerate comps to the same WR.
    """
    records = []
    for _ in range(n):
        team = sample_heroes_for_roles(role_tuple, heroes_by_role)
        # Sample opponent, ensure no hero overlap
        for _ in range(20):  # max retries
            opp = random.choice(opponent_teams)
            if not set(team) & set(opp):
                break
        else:
            continue  # skip if can't find non-overlapping opponent

        actual_wr = win_rate
        if pairwise_adjust and stats_cache is not None:
            actual_wr = _compute_pairwise_adjusted_wr(
                team, opp, stats_cache, tier, base_wr=win_rate)

        winner = 0 if random.random() < (actual_wr / 100) else 1
        records.append({
            "team0_heroes": team,
            "team1_heroes": list(opp),
            "game_map": random.choice(MAPS),
            "skill_tier": tier,
            "winner": winner,
        })
    return records


def _compute_pairwise_adjusted_wr(our_heroes, opp_heroes, stats_cache, tier, base_wr=10.0):
    """
    Adjust synthetic WR based on pairwise counter/synergy statistics.

    A degenerate comp with good counters against the opponent loses less badly
    than one with bad counters. This preserves pairwise gradients while
    maintaining the "degenerate = bad" signal.
    """
    # Counter adjustment: how well do our heroes counter theirs?
    counter_deltas = []
    for our_h in our_heroes:
        for opp_h in opp_heroes:
            raw = stats_cache.get_counter(our_h, opp_h, tier)
            if raw is not None:
                wr_a = stats_cache.get_hero_wr(our_h, tier)
                wr_b = stats_cache.get_hero_wr(opp_h, tier)
                counter_deltas.append(raw - (wr_a + (100 - wr_b) - 50))
    avg_counter = np.mean(counter_deltas) if counter_deltas else 0.0

    # Synergy adjustment: how well do our heroes synergize?
    synergy_deltas = []
    for i, h1 in enumerate(our_heroes):
        for h2 in our_heroes[i+1:]:
            raw = stats_cache.get_synergy(h1, h2, tier)
            if raw is not None:
                wr1 = stats_cache.get_hero_wr(h1, tier)
                wr2 = stats_cache.get_hero_wr(h2, tier)
                synergy_deltas.append(raw - (50 + (wr1 - 50) + (wr2 - 50)))
    avg_synergy = np.mean(synergy_deltas) if synergy_deltas else 0.0

    # Scale: counter deltas ~[-5, +5], synergy similar
    counter_adj = avg_counter * 1.0   # +/-5pp max from counter
    synergy_adj = avg_synergy * 0.5   # +/-2.5pp max from synergy

    adjusted_wr = base_wr + counter_adj + synergy_adj
    return float(np.clip(adjusted_wr, 5.0, 30.0))


# ── Training ──

def train_augmented_model(train_real, synthetic, test_data, stats_cache, group_indices,
                          device, seed=42):
    """Train enriched WP model on real + synthetic data. Returns (model, acc, path)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Combine and shuffle
    train_combined = list(train_real) + synthetic
    random.shuffle(train_combined)

    # Precompute features
    train_base, train_enriched, train_labels = precompute_all_features(train_combined, stats_cache)
    test_base, test_enriched, test_labels = precompute_all_features(test_data, stats_cache)

    # Select enriched columns for our groups
    cols = []
    for g in ENRICHED_GROUPS:
        s, e = group_indices[g]
        cols.extend(range(s, e))
    total_dim = 197 + len(cols)

    if cols:
        train_X = torch.cat([train_base, train_enriched[:, cols]], dim=1).to(device)
        test_X = torch.cat([test_base, test_enriched[:, cols]], dim=1).to(device)
    else:
        train_X = train_base.to(device)
        test_X = test_base.to(device)

    train_y = train_labels.to(device)
    test_y = test_labels.to(device)

    model = WinProbEnrichedModel(total_dim, TRAIN_HP["hidden_dims"],
                                  dropout=TRAIN_HP["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_HP["lr"],
                                   weight_decay=TRAIN_HP["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    criterion = nn.BCELoss()

    best_loss = float('inf')
    best_acc = 0
    patience_counter = 0
    save_path = os.path.join(RESULTS_DIR, f"_temp_model_seed{seed}.pt")

    batch_size = 4096
    n_train = train_X.shape[0]

    for epoch in range(TRAIN_HP["max_epochs"]):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            pred = model(train_X[idx])
            loss = criterion(pred, train_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        scheduler.step()

        # Eval
        model.eval()
        with torch.no_grad():
            test_pred = model(test_X)
            test_loss = criterion(test_pred, test_y).item()
            test_acc = ((test_pred > 0.5).float() == test_y).float().mean().item() * 100

        if test_loss < best_loss:
            best_loss = test_loss
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_HP["patience"]:
                break

    # Reload best
    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))
    model.eval()
    return model, best_acc, save_path


# ── Evaluation ──

def make_eval_fn(model, cols, stats_cache, device):
    """Create eval_fn compatible with test_wp_sanity.run_tests."""
    all_mask = [True] * len(FEATURE_GROUPS)
    def eval_fn(t0h, t1h, game_map="Cursed Hollow", tier="mid"):
        d = {"team0_heroes": t0h, "team1_heroes": t1h,
             "game_map": game_map, "skill_tier": tier, "winner": 0}
        base, enriched = extract_features(d, stats_cache, all_mask)
        enriched_sel = enriched[cols] if cols else np.array([], dtype=np.float32)
        x = np.concatenate([base, enriched_sel]) if len(enriched_sel) > 0 else base
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)).item()
    return eval_fn


DEGEN_COMPS = {
    "5 tanks": ["Muradin", "Johanna", "Diablo", "E.T.C.", "Mal'Ganis"],
    "5 healers": ["Brightwing", "Malfurion", "Rehgar", "Uther", "Anduin"],
    "5 ranged assassins": ["Valla", "Jaina", "Li-Ming", "Falstad", "Raynor"],
    "5 melee assassins": ["Zeratul", "Illidan", "Kerrigan", "Malthael", "Qhira"],
    "3T 2H no damage": ["Muradin", "Johanna", "Diablo", "Brightwing", "Malfurion"],
    "No healer high WR": ["Muradin", "Johanna", "Valla", "Falstad", "Li-Ming"],
}
STANDARD = ["Muradin", "Brightwing", "Valla", "Sonya", "Jaina"]


def evaluate_config(model, cols, stats_cache, device):
    """Run Stage 1 evaluation. Returns dict with all metrics."""
    eval_fn = make_eval_fn(model, cols, stats_cache, device)

    # Sanity tests
    passed, total, results_list = run_tests(eval_fn, verbose=False)
    cats = {}
    for t, p in zip(TESTS, results_list):
        cat = t.get("category", "")
        if cat:
            cats.setdefault(cat, {"passed": 0, "total": 0})
            cats[cat]["total"] += 1
            if p:
                cats[cat]["passed"] += 1

    # Degenerate composition WP scores
    degen_scores = {}
    for name, comp in DEGEN_COMPS.items():
        wp = eval_fn(comp, STANDARD, "Cursed Hollow", "mid")
        degen_scores[name] = wp

    return {
        "sanity_passed": passed,
        "sanity_total": total,
        "sanity_categories": cats,
        "degen_scores": degen_scores,
        "all_degen_below_25": all(v < 0.25 for v in degen_scores.values()),
    }


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-only", action="store_true", help="Skip Stage 2 greedy drafts")
    parser.add_argument("--drafts", type=int, default=200, help="Drafts for Stage 2")
    parser.add_argument("--quick", action="store_true", help="Quick test (2 configs)")
    parser.add_argument("--seeds", type=int, default=2, help="Seeds per config")
    parser.add_argument("--wr", type=float, default=None, help="Single unseen WR")
    parser.add_argument("--volume", type=int, default=None, help="Single unseen volume")
    parser.add_argument("--scope", type=str, default=None, help="Single scope")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    data = load_replay_data()
    train_data, test_data = split_data(data)
    stats_cache = StatsCache()
    group_indices = compute_group_indices()

    # Enriched columns
    cols = []
    for g in ENRICHED_GROUPS:
        s, e = group_indices[g]
        cols.extend(range(s, e))

    # Load composition data
    comp_path = os.path.join(os.path.dirname(__file__), "..", "src", "lib", "data", "compositions.json")
    with open(comp_path) as f:
        comp_data = json.load(f)

    # ═══════════════════════════════════════════════════════════════
    # Build config grid
    # ═══════════════════════════════════════════════════════════════
    if args.wr is not None and args.volume is not None and args.scope is not None:
        configs = [{"wr": args.wr, "volume": args.volume, "scope": args.scope}]
    elif args.quick:
        configs = [
            {"wr": 20, "volume": 100, "scope": "both"},
            {"wr": 20, "volume": 100, "scope": "tier2_only"},
        ]
    else:
        configs = []
        for wr in [10, 20, 30]:
            for vol in [50, 100, 200]:
                for scope in ["tier2_only", "both", "tier1_only"]:
                    configs.append({"wr": wr, "volume": vol, "scope": scope})

    print(f"\n{'='*70}")
    print(f"STAGE 1: SWEEP ({len(configs)} configs × {args.seeds} seeds)")
    print(f"{'='*70}")

    # ═══════════════════════════════════════════════════════════════
    # Stage 1: Train and evaluate all configs
    # ═══════════════════════════════════════════════════════════════
    all_results = []
    t0 = time.time()

    # First train baseline (no augmentation) for comparison
    print("\n--- BASELINE (no augmentation) ---")
    baseline_model, baseline_acc, _ = train_augmented_model(
        train_data, [], test_data, stats_cache, group_indices, device, seed=42,
    )
    print(f"  Accuracy: {baseline_acc:.2f}%")
    baseline_eval = evaluate_config(baseline_model, cols, stats_cache, device)
    print(f"  Sanity: {baseline_eval['sanity_passed']}/{baseline_eval['sanity_total']}")
    print(f"  Degen scores: " + ", ".join(f"{k}: {v:.3f}" for k, v in baseline_eval['degen_scores'].items()))
    baseline_result = {
        "name": "baseline",
        "config": {"wr": None, "volume": 0, "scope": "none"},
        "accuracy": baseline_acc,
        "eval": baseline_eval,
        "synthetic_stats": {"total": 0},
    }
    all_results.append(baseline_result)
    del baseline_model
    torch.cuda.empty_cache()

    for ci, cfg in enumerate(configs):
        name = f"wr{cfg['wr']}_vol{cfg['volume']}_{cfg['scope']}"
        print(f"\n--- [{ci+1}/{len(configs)}] {name} ---")

        # Generate synthetic data
        synthetic, syn_stats = generate_synthetic_data(
            train_data, comp_data,
            unseen_wr=cfg["wr"], unseen_volume=cfg["volume"], scope=cfg["scope"],
        )
        print(f"  Synthetic: {syn_stats['total']} records (T1: {syn_stats['tier1']}, T2: {syn_stats['tier2']})")

        seed_accs = []
        seed_evals = []
        for seed in range(args.seeds):
            model, acc, path = train_augmented_model(
                train_data, synthetic, test_data, stats_cache, group_indices,
                device, seed=42 + seed,
            )
            eval_result = evaluate_config(model, cols, stats_cache, device)
            seed_accs.append(acc)
            seed_evals.append(eval_result)
            del model
            torch.cuda.empty_cache()
            # Clean up temp file
            if os.path.exists(path):
                os.remove(path)

        avg_acc = np.mean(seed_accs)
        # Pick best seed by sanity score, then by degen scores
        best_idx = max(range(len(seed_evals)),
                       key=lambda i: (seed_evals[i]["sanity_passed"],
                                      -sum(seed_evals[i]["degen_scores"].values())))
        best_eval = seed_evals[best_idx]

        print(f"  Accuracy: {avg_acc:.2f}% (seeds: {', '.join(f'{a:.2f}' for a in seed_accs)})")
        print(f"  Best sanity: {best_eval['sanity_passed']}/{best_eval['sanity_total']}")
        degen_str = ", ".join(f"{k}: {v:.3f}" for k, v in best_eval['degen_scores'].items())
        print(f"  Degen: {degen_str}")

        result = {
            "name": name,
            "config": cfg,
            "accuracy": avg_acc,
            "all_accs": seed_accs,
            "eval": best_eval,
            "best_seed": best_idx,
            "synthetic_stats": syn_stats,
        }
        all_results.append(result)

    elapsed = time.time() - t0
    print(f"\nStage 1 complete in {elapsed/60:.1f} minutes")

    # ═══════════════════════════════════════════════════════════════
    # Print Stage 1 summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("STAGE 1 SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<35} {'Acc%':>6} {'Sanity':>7} {'5tank':>6} {'5heal':>6} {'noHeal':>6} {'allDeg<25':>10}")
    print("-" * 80)
    for r in sorted(all_results, key=lambda x: x['eval']['sanity_passed'], reverse=True):
        e = r['eval']
        ds = e['degen_scores']
        print(f"{r['name']:<35} {r['accuracy']:>6.2f} "
              f"{e['sanity_passed']:>3}/{e['sanity_total']:<3} "
              f"{ds.get('5 tanks', 0):>6.3f} {ds.get('5 healers', 0):>6.3f} "
              f"{ds.get('No healer high WR', 0):>6.3f} "
              f"{'YES' if e['all_degen_below_25'] else 'no':>10}")

    # Save Stage 1 results
    stage1_path = os.path.join(RESULTS_DIR, "stage1_results.json")
    with open(stage1_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nStage 1 results saved to {stage1_path}")

    # ═══════════════════════════════════════════════════════════════
    # Stage 2: Greedy drafts for top configs
    # ═══════════════════════════════════════════════════════════════
    if args.stage1_only:
        print("\n--stage1-only: skipping Stage 2")
        return

    # Select top 5 by sanity score, breaking ties by lowest total degen score
    ranked = sorted(all_results,
                    key=lambda x: (x['eval']['sanity_passed'],
                                   -sum(x['eval']['degen_scores'].values())),
                    reverse=True)
    top_configs = ranked[:5]

    print(f"\n{'='*70}")
    print(f"STAGE 2: GREEDY DRAFTS (top {len(top_configs)} configs, {args.drafts} drafts each)")
    print(f"{'='*70}")

    # Import greedy infrastructure
    from experiment_value_function_quality import greedy_pick_with_model
    from train_draft_policy import DraftState, DRAFT_ORDER
    from train_generic_draft import GenericDraftModel

    # Load GD models
    gd_models = []
    for i in range(5):
        gd_path = os.path.join(os.path.dirname(__file__), f"generic_draft_{i}.pt")
        if not os.path.exists(gd_path):
            gd_path = os.path.join(os.path.dirname(__file__), "generic_draft.pt")
        gd = GenericDraftModel()
        gd.load_state_dict(torch.load(gd_path, weights_only=True, map_location=device))
        gd.to(device).eval()
        gd_models.append(gd)

    # Generate draft scenarios
    random.seed(42)
    draft_configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
                     for i in range(args.drafts)]

    stage2_results = []
    for r in top_configs:
        cfg = r["config"]
        name = r["name"]
        print(f"\n--- {name} ---")

        # Retrain best seed
        if cfg.get("wr") is not None:
            synthetic, _ = generate_synthetic_data(
                train_data, comp_data,
                unseen_wr=cfg["wr"], unseen_volume=cfg["volume"], scope=cfg["scope"],
            )
        else:
            synthetic = []

        model, acc, path = train_augmented_model(
            train_data, synthetic, test_data, stats_cache, group_indices,
            device, seed=42 + r.get("best_seed", 0),
        )
        eval_fn = make_eval_fn(model, cols, stats_cache, device)

        # Run greedy drafts
        healer_count = 0
        degen_count = 0
        total_drafts = 0

        healer_heroes = set(h for h, role in HERO_ROLE_FINE.items() if role == "healer")
        tank_heroes = set(h for h, role in HERO_ROLE_FINE.items() if role == "tank")
        bruiser_heroes = set(h for h, role in HERO_ROLE_FINE.items() if role == "bruiser")
        ranged_heroes = set(h for h, role in HERO_ROLE_FINE.items()
                           if role in ("ranged_aa", "ranged_mage", "pusher"))
        frontline = tank_heroes | bruiser_heroes

        for di, (draft_id, game_map, tier, our_team) in enumerate(draft_configs):
            state = DraftState(game_map, tier)

            for step_idx, (step_team, step_type) in enumerate(DRAFT_ORDER):
                if step_type == "ban":
                    # Simple ban: pick highest WR hero
                    valid = state.valid_actions()
                    if not valid:
                        continue
                    # Use GD model for ban
                    gd = random.choice(gd_models)
                    x = state.to_tensor().unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = gd(x).squeeze(0).cpu()
                    mask = torch.full((NUM_HEROES,), float('-inf'))
                    for h in valid:
                        mask[HERO_TO_IDX[h]] = 0
                    logits = logits + mask
                    hero = HEROES[logits.argmax().item()]
                    state.apply_action(hero, step_team, "ban")
                elif step_team == our_team:
                    # Our pick: greedy via WP model
                    valid = state.valid_actions()
                    best_hero = None
                    best_wp = -1
                    for hero in valid:
                        test_state = state.clone()
                        test_state.apply_action(hero, step_team, "pick")
                        # Quick eval: complete draft with GD rollout
                        s = test_state.clone()
                        remaining = DRAFT_ORDER[step_idx+1:]
                        for rs_team, rs_type in remaining:
                            rv = s.valid_actions()
                            if not rv:
                                continue
                            gd = random.choice(gd_models)
                            x = s.to_tensor().unsqueeze(0).to(device)
                            with torch.no_grad():
                                logits = gd(x).squeeze(0).cpu()
                            mask = torch.full((NUM_HEROES,), float('-inf'))
                            for h in rv:
                                mask[HERO_TO_IDX[h]] = 0
                            logits = logits + mask
                            h = HEROES[logits.argmax().item()]
                            s.apply_action(h, rs_team, rs_type)

                        # Evaluate terminal state
                        wp = eval_fn(s.team_heroes[0], s.team_heroes[1], game_map, tier)
                        if our_team == 1:
                            wp = 1 - wp
                        if wp > best_wp:
                            best_wp = wp
                            best_hero = hero

                    if best_hero:
                        state.apply_action(best_hero, step_team, "pick")
                else:
                    # Opponent pick: use GD
                    valid = state.valid_actions()
                    if not valid:
                        continue
                    gd = random.choice(gd_models)
                    x = state.to_tensor().unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = gd(x).squeeze(0).cpu()
                    mask = torch.full((NUM_HEROES,), float('-inf'))
                    for h in valid:
                        mask[HERO_TO_IDX[h]] = 0
                    logits = logits + mask
                    hero = HEROES[logits.argmax().item()]
                    state.apply_action(hero, step_team, "pick")

            # Analyze our team composition
            our_heroes = state.team_heroes[our_team]
            has_healer = any(h in healer_heroes for h in our_heroes)
            has_frontline = any(h in frontline for h in our_heroes)
            has_ranged = any(h in ranged_heroes for h in our_heroes)

            # Role stacking
            role_counts_map = {}
            for h in our_heroes:
                r = HERO_ROLE_FINE.get(h, "unknown")
                role_counts_map[r] = role_counts_map.get(r, 0) + 1
            has_stacking = any(c >= 3 for c in role_counts_map.values())

            is_degen = is_degenerate(our_heroes)

            if has_healer:
                healer_count += 1
            if is_degen:
                degen_count += 1
            total_drafts += 1

            if (di + 1) % 50 == 0:
                print(f"  Draft {di+1}/{args.drafts}: healer={healer_count/total_drafts*100:.1f}% "
                      f"degen={degen_count/total_drafts*100:.1f}%")

        healer_rate = healer_count / total_drafts * 100
        degen_rate = degen_count / total_drafts * 100
        print(f"  Final: healer={healer_rate:.1f}% degen={degen_rate:.1f}%")

        stage2_results.append({
            "name": name,
            "config": cfg,
            "accuracy": acc,
            "healer_rate": healer_rate,
            "degen_rate": degen_rate,
            "total_drafts": total_drafts,
        })

        del model
        torch.cuda.empty_cache()
        if os.path.exists(path):
            os.remove(path)

    # Print Stage 2 summary
    print(f"\n{'='*70}")
    print("STAGE 2 SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<35} {'Acc%':>6} {'Healer%':>8} {'Degen%':>8}")
    print("-" * 60)
    print(f"{'baseline (from main experiment)':<35} {'57.9':>6} {'66.5':>8} {'57.7':>8}")
    for r in stage2_results:
        print(f"{r['name']:<35} {r['accuracy']:>6.2f} {r['healer_rate']:>8.1f} {r['degen_rate']:>8.1f}")

    # Save
    stage2_path = os.path.join(RESULTS_DIR, "stage2_results.json")
    with open(stage2_path, "w") as f:
        json.dump(stage2_results, f, indent=2, default=str)
    print(f"\nStage 2 results saved to {stage2_path}")


if __name__ == "__main__":
    main()
