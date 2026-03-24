"""
Stage 2: Greedy Draft Benchmark on Augmented Models.

Runs 200 greedy drafts with 4 model configs to measure whether synthetic
augmentation improves composition quality (healer rate, degenerate rate).

Usage:
    set -a && source .env && set +a
    python3 -u training/experiment_stage2_greedy.py
"""
import os
import sys
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS,
    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
    load_replay_data, split_data,
    HERO_ROLE_FINE, FINE_ROLE_NAMES,
)
from sweep_enriched_wp import (
    StatsCache, WinProbEnrichedModel, FEATURE_GROUPS, FEATURE_GROUP_DIMS,
    compute_group_indices, precompute_all_features, extract_features,
)
from experiment_value_function_quality import (
    run_single_draft, greedy_pick_with_model, analyze_composition,
    TRAIN_HP,
)
from experiment_synthetic_augmentation import generate_synthetic_data, ENRICHED_GROUPS
from experiment_synthetic_ablation2 import train_with_options
from train_draft_policy import DraftState, DRAFT_ORDER
from train_generic_draft import GenericDraftModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "experiment_results", "synthetic_augmentation")

NUM_DRAFTS = 200


def train_model(name, train_data, synthetic, test_data, stats_cache, group_indices,
                device, hidden_dims, seed=42):
    """Train and return (model, accuracy, state_dict)."""
    model, acc, path, params = train_with_options(
        train_data, synthetic, test_data, stats_cache, group_indices,
        device, hidden_dims=hidden_dims, seed=seed,
    )
    sd = model.state_dict()
    if os.path.exists(path):
        os.remove(path)
    print(f"  {name}: {acc:.2f}% acc, {params:,} params")
    return sd, acc, hidden_dims


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    data = load_replay_data()
    train_data, test_data = split_data(data)
    stats_cache = StatsCache()
    group_indices = compute_group_indices()

    comp_path = os.path.join(os.path.dirname(__file__), "..", "src", "lib", "data", "compositions.json")
    with open(comp_path) as f:
        comp_data = json.load(f)

    # Generate synthetic data for augmented configs
    synthetic_100, _ = generate_synthetic_data(
        train_data, comp_data, unseen_wr=10, unseen_volume=100, scope="tier2_only")
    synthetic_25, _ = generate_synthetic_data(
        train_data, comp_data, unseen_wr=10, unseen_volume=25, scope="tier2_only")
    print(f"Synthetic data: v100={len(synthetic_100)}, v25={len(synthetic_25)}")

    # ═══════════════════════════════════════════════════════════════
    # Train 4 models
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("TRAINING 4 MODELS")
    print(f"{'='*70}")

    model_configs = {
        "baseline_256_128": {"syn": [], "hidden": [256, 128]},
        "baseline_512_256_128": {"syn": [], "hidden": [512, 256, 128]},
        "wr10_v100_512": {"syn": synthetic_100, "hidden": [512, 256, 128]},
        "wr10_v25_512": {"syn": synthetic_25, "hidden": [512, 256, 128]},
    }

    model_sds = {}
    model_accs = {}
    model_hidden = {}
    for name, cfg in model_configs.items():
        sd, acc, hd = train_model(name, train_data, cfg["syn"], test_data,
                                   stats_cache, group_indices, device, cfg["hidden"])
        model_sds[name] = sd
        model_accs[name] = acc
        model_hidden[name] = hd

    # Load GD models
    print("\nLoading GD opponent models...")
    gd_models = []
    for i in range(5):
        gd_path = os.path.join(os.path.dirname(__file__), f"generic_draft_{i}.pt")
        if not os.path.exists(gd_path):
            gd_path = os.path.join(os.path.dirname(__file__), "generic_draft.pt")
        gd = GenericDraftModel()
        gd.load_state_dict(torch.load(gd_path, weights_only=True, map_location="cpu"))
        gd.cpu().eval()  # run_single_draft uses CPU for GD inference
        gd_models.append(gd)
    print(f"  Loaded {len(gd_models)} GD models")

    # Generate draft scenarios (same for all models)
    random.seed(42)
    draft_configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
                     for i in range(NUM_DRAFTS)]

    # ═══════════════════════════════════════════════════════════════
    # Run greedy drafts
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"GREEDY DRAFT BENCHMARK ({NUM_DRAFTS} drafts × {len(model_configs)} models)")
    print(f"{'='*70}")

    # Build all WP models for cross-evaluation
    all_wp_models = {}
    all_wp_groups = {}
    for name, sd in model_sds.items():
        cols = []
        for g in ENRICHED_GROUPS:
            s, e = group_indices[g]
            cols.extend(range(s, e))
        dim = 197 + len(cols)
        model = WinProbEnrichedModel(dim, model_hidden[name],
                                      dropout=TRAIN_HP["dropout"]).to(device)
        model.load_state_dict(sd)
        model.eval()
        all_wp_models[name] = model
        all_wp_groups[name] = ENRICHED_GROUPS

    all_records = []
    t0 = time.time()

    for model_name in model_configs:
        print(f"\n--- {model_name} ---")
        wp_model = all_wp_models[model_name]
        wp_groups = ENRICHED_GROUPS

        for di, (config_idx, game_map, skill_tier, our_team) in enumerate(draft_configs):
            record = run_single_draft(
                config_idx, game_map, skill_tier, our_team,
                model_name, wp_model, wp_groups,
                all_wp_models, all_wp_groups,
                gd_models, stats_cache, group_indices, device,
            )
            all_records.append(record)

            if (di + 1) % 50 == 0:
                # Quick stats so far
                model_recs = [r for r in all_records if r["wp_model"] == model_name]
                healer = sum(1 for r in model_recs if r["comp_has_healer"]) / len(model_recs) * 100
                degen = sum(1 for r in model_recs if r["comp_is_absurd"]) / len(model_recs) * 100
                print(f"  {di+1}/{NUM_DRAFTS}: healer={healer:.1f}% degen={degen:.1f}%")

    elapsed = time.time() - t0
    print(f"\nBenchmark complete in {elapsed/60:.1f} minutes")

    # ═══════════════════════════════════════════════════════════════
    # Analysis
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("STAGE 2 RESULTS")
    print(f"{'='*80}")

    model_names = list(model_configs.keys())
    by_model = {m: [r for r in all_records if r["wp_model"] == m] for m in model_names}

    print(f"\n{'Config':<25} {'Acc%':>6} {'Healer%':>8} {'Front%':>7} {'Ranged%':>8} "
          f"{'Degen%':>7} {'Roles':>6} {'SelfWP':>7}")
    print("-" * 80)

    summary = []
    for name in model_names:
        recs = by_model[name]
        n = len(recs)
        healer = sum(1 for r in recs if r["comp_has_healer"]) / n * 100
        front = sum(1 for r in recs if r["comp_has_frontline"]) / n * 100
        ranged = sum(1 for r in recs if r["comp_has_ranged_damage"]) / n * 100
        degen = sum(1 for r in recs if r["comp_is_absurd"]) / n * 100
        roles = np.mean([r["comp_num_distinct_roles"] for r in recs])
        self_wp = np.mean([r.get(f"wp_by_{name}", r.get("wp_by_enriched", 0.5)) for r in recs])

        print(f"{name:<25} {model_accs[name]:>6.2f} {healer:>8.1f} {front:>7.1f} "
              f"{ranged:>8.1f} {degen:>7.1f} {roles:>6.1f} {self_wp:>7.3f}")

        summary.append({
            "name": name,
            "accuracy": model_accs[name],
            "healer_rate": healer,
            "frontline_rate": front,
            "ranged_rate": ranged,
            "degen_rate": degen,
            "avg_roles": roles,
            "self_wp": self_wp,
            "n_drafts": n,
        })

    # Cross-evaluation matrix
    print(f"\n{'='*80}")
    print("CROSS-EVALUATION MATRIX")
    print(f"{'='*80}")
    header = f"{'Drafter':<25}" + "".join(f"{n:>18}" for n in model_names)
    print(header)
    print("-" * len(header))
    for drafter in model_names:
        recs = by_model[drafter]
        vals = []
        for evaluator in model_names:
            key = f"wp_by_{evaluator}"
            if key in recs[0]:
                avg = np.mean([r[key] for r in recs])
                vals.append(f"{avg:>18.3f}")
            else:
                vals.append(f"{'N/A':>18}")
        print(f"{drafter:<25}" + "".join(vals))

    # Key comparisons
    print(f"\n{'='*80}")
    print("KEY COMPARISONS")
    print(f"{'='*80}")

    def get_stats(name):
        recs = by_model[name]
        n = len(recs)
        return {
            "healer": sum(1 for r in recs if r["comp_has_healer"]) / n * 100,
            "degen": sum(1 for r in recs if r["comp_is_absurd"]) / n * 100,
        }

    s = {name: get_stats(name) for name in model_names}

    print(f"\n1. Architecture effect (no synthetic data):")
    print(f"   256→128:      healer={s['baseline_256_128']['healer']:.1f}%, degen={s['baseline_256_128']['degen']:.1f}%")
    print(f"   512→256→128:  healer={s['baseline_512_256_128']['healer']:.1f}%, degen={s['baseline_512_256_128']['degen']:.1f}%")
    d_h = s['baseline_512_256_128']['healer'] - s['baseline_256_128']['healer']
    d_d = s['baseline_512_256_128']['degen'] - s['baseline_256_128']['degen']
    print(f"   Delta: healer {d_h:+.1f}pp, degen {d_d:+.1f}pp")

    print(f"\n2. Synthetic data effect (architecture held constant at 512→256→128):")
    print(f"   No synthetic:  healer={s['baseline_512_256_128']['healer']:.1f}%, degen={s['baseline_512_256_128']['degen']:.1f}%")
    print(f"   v100 synthetic: healer={s['wr10_v100_512']['healer']:.1f}%, degen={s['wr10_v100_512']['degen']:.1f}%")
    d_h = s['wr10_v100_512']['healer'] - s['baseline_512_256_128']['healer']
    d_d = s['wr10_v100_512']['degen'] - s['baseline_512_256_128']['degen']
    print(f"   Delta: healer {d_h:+.1f}pp, degen {d_d:+.1f}pp")

    print(f"\n3. Synthetic volume effect (v25 vs v100, same architecture):")
    print(f"   v25:  healer={s['wr10_v25_512']['healer']:.1f}%, degen={s['wr10_v25_512']['degen']:.1f}%")
    print(f"   v100: healer={s['wr10_v100_512']['healer']:.1f}%, degen={s['wr10_v100_512']['degen']:.1f}%")
    d_h = s['wr10_v100_512']['healer'] - s['wr10_v25_512']['healer']
    d_d = s['wr10_v100_512']['degen'] - s['wr10_v25_512']['degen']
    print(f"   Delta: healer {d_h:+.1f}pp, degen {d_d:+.1f}pp")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, "stage2_results.json")
    with open(save_path, "w") as f:
        json.dump({
            "summary": summary,
            "records": all_records,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
