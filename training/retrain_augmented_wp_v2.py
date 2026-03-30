#!/usr/bin/env python3
"""
Retrain augmented WP model with pairwise-adjusted synthetic data (v2).

Compares:
  - Original flat 10% WR augmented model
  - v2 model with pairwise-adjusted WR (5-30% based on counter/synergy)
  - v2 with reduced volume (50 per comp instead of 100)

Usage:
    set -a && source .env && set +a
    python3 -u training/retrain_augmented_wp_v2.py
"""

import os, sys, json, random, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import load_replay_data, split_data, HEROES
from sweep_enriched_wp import (
    StatsCache, WinProbEnrichedModel, FEATURE_GROUPS, FEATURE_GROUP_DIMS,
    compute_group_indices, precompute_all_features,
)
from experiment_synthetic_augmentation import (
    generate_synthetic_data, generate_synthetic_data_v2,
    train_augmented_model, ENRICHED_GROUPS, TRAIN_HP,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "experiment_results", "diagnostics")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading replay data...")
    data = load_replay_data()
    train_data, test_data = split_data(data, test_frac=0.15)
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    stats_cache = StatsCache()
    gi = compute_group_indices()

    # Load composition data
    comp_path = os.path.join(os.path.dirname(__file__), "..", "src", "lib", "data", "compositions.json")
    import json as _json
    comp_data = _json.load(open(comp_path))

    # Generate synthetic data variants
    print("\nGenerating synthetic data...")

    # Original flat 10% (for comparison)
    syn_flat, stats_flat = generate_synthetic_data(
        train_data, comp_data, unseen_wr=10, unseen_volume=100, scope="tier2_only")
    print(f"  Flat 10%: {stats_flat['total']} records")

    # v2: pairwise-adjusted
    syn_pw100, stats_pw100 = generate_synthetic_data_v2(
        train_data, comp_data, stats_cache, unseen_wr=10.0, unseen_volume=100)
    print(f"  Pairwise v2 (100/comp): {stats_pw100['total']} records")

    # v2 reduced volume
    syn_pw50, stats_pw50 = generate_synthetic_data_v2(
        train_data, comp_data, stats_cache, unseen_wr=10.0, unseen_volume=50)
    print(f"  Pairwise v2 (50/comp): {stats_pw50['total']} records")

    # Train models
    configs = [
        ("flat_10pct", syn_flat, [512, 256, 128]),       # original augmented architecture
        ("pairwise_v2_100", syn_pw100, [512, 256, 128]),  # same arch, smarter data
        ("pairwise_v2_50", syn_pw50, [512, 256, 128]),    # less volume
        ("pairwise_v2_256", syn_pw100, [256, 128]),       # enriched architecture
    ]

    results = {}
    for name, synthetic, hidden_dims in configs:
        print(f"\nTraining: {name} (hidden={hidden_dims}, {len(synthetic)} synthetic records)")
        t0 = time.time()

        # Override TRAIN_HP with custom hidden dims
        orig_dims = TRAIN_HP["hidden_dims"]
        TRAIN_HP["hidden_dims"] = hidden_dims

        model, acc, path = train_augmented_model(
            train_data, synthetic, test_data, stats_cache, gi, device)

        TRAIN_HP["hidden_dims"] = orig_dims
        elapsed = time.time() - t0

        # Save model
        save_path = os.path.join(RESULTS_DIR, f"wp_{name}.pt")
        torch.save(model.state_dict(), save_path)

        results[name] = {"accuracy": acc, "elapsed": elapsed, "path": save_path}
        print(f"  {name}: acc={acc:.2f}%, time={elapsed:.0f}s")

    # Run WP sensitivity validation on each model
    print("\n\nRunning WP sensitivity validation...")
    from validate_wp_sensitivity import run_sensitivity, print_results

    replays = load_replay_data()
    wp_cols = []
    for g in ENRICHED_GROUPS:
        s, e = gi[g]
        wp_cols.extend(range(s, e))
    all_mask = [True] * len(FEATURE_GROUPS)

    for name in results:
        path = results[name]["path"]
        hidden = [512, 256, 128] if "256" not in name else [256, 128]
        wp_dim = 197 + sum(FEATURE_GROUP_DIMS[g] for g in ENRICHED_GROUPS)
        model = WinProbEnrichedModel(wp_dim, hidden, dropout=0.3)
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        model.eval()

        cr, sr = run_sensitivity(model, name, stats_cache, wp_cols, all_mask, replays, 300)
        avg_c, avg_s = print_results(cr, sr, name)
        results[name]["counter_r"] = avg_c
        results[name]["synergy_r"] = avg_s

    # Summary
    print("\n\n" + "=" * 75)
    print(f"{'Config':<25} {'Acc%':>6} {'r(Ctr)':>8} {'r(Syn)':>8} {'Time':>6}")
    print("-" * 75)
    for name, r in results.items():
        print(f"{name:<25} {r['accuracy']:>6.2f} {r.get('counter_r',0):>+8.4f} "
              f"{r.get('synergy_r',0):>+8.4f} {r['elapsed']:>5.0f}s")
    print("=" * 75)

    # Save
    out_path = os.path.join(RESULTS_DIR, "augmented_v2_results.json")
    json_r = {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else str(vv)
                   for kk, vv in v.items()} for k, v in results.items()}
    with open(out_path, "w") as f:
        json.dump(json_r, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
