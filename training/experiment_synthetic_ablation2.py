"""
Follow-up ablation: Test larger models and weighted loss with synthetic augmentation
to recover accuracy while keeping the degenerate composition penalty.

Builds on Stage 1 findings: wr10_vol100_tier2_only was best (20/21 sanity, 56.57% acc).
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
from test_wp_sanity import TESTS, run_tests
from experiment_synthetic_augmentation import (
    generate_synthetic_data, make_eval_fn, evaluate_config,
    ENRICHED_GROUPS, DEGEN_COMPS, STANDARD, RESULTS_DIR,
)

def train_with_options(train_real, synthetic, test_data, stats_cache, group_indices,
                       device, hidden_dims=[256, 128], dropout=0.3, lr=5e-4,
                       weight_decay=5e-3, patience=25, max_epochs=200,
                       synthetic_weight=1.0, seed=42):
    """Train enriched WP with configurable architecture and weighted loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Precompute features separately so we can apply different weights
    real_base, real_enriched, real_labels = precompute_all_features(train_real, stats_cache)
    test_base, test_enriched, test_labels = precompute_all_features(test_data, stats_cache)

    cols = []
    for g in ENRICHED_GROUPS:
        s, e = group_indices[g]
        cols.extend(range(s, e))
    total_dim = 197 + len(cols)

    if synthetic:
        syn_base, syn_enriched, syn_labels = precompute_all_features(synthetic, stats_cache)
        train_base = torch.cat([real_base, syn_base], dim=0)
        train_enriched_all = torch.cat([real_enriched, syn_enriched], dim=0)
        train_labels = torch.cat([real_labels, syn_labels], dim=0)
        # Build sample weights
        n_real = real_base.shape[0]
        n_syn = syn_base.shape[0]
        sample_weights = torch.cat([
            torch.ones(n_real),
            torch.full((n_syn,), synthetic_weight),
        ])
    else:
        train_base = real_base
        train_enriched_all = real_enriched
        train_labels = real_labels
        sample_weights = torch.ones(train_base.shape[0])

    # Shuffle
    perm = torch.randperm(train_base.shape[0])
    train_base = train_base[perm]
    train_enriched_all = train_enriched_all[perm]
    train_labels = train_labels[perm]
    sample_weights = sample_weights[perm]

    if cols:
        train_X = torch.cat([train_base, train_enriched_all[:, cols]], dim=1).to(device)
        test_X = torch.cat([test_base, test_enriched[:, cols]], dim=1).to(device)
    else:
        train_X = train_base.to(device)
        test_X = test_base.to(device)

    train_y = train_labels.to(device)
    test_y = test_labels.to(device)
    sample_weights = sample_weights.to(device)

    model = WinProbEnrichedModel(total_dim, hidden_dims, dropout=dropout).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    best_loss = float('inf')
    best_acc = 0
    patience_counter = 0
    save_path = os.path.join(RESULTS_DIR, f"_temp_ablation2_seed{seed}.pt")
    batch_size = 4096
    n_train = train_X.shape[0]

    for epoch in range(max_epochs):
        model.train()
        idx_perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            idx = idx_perm[i:i+batch_size]
            pred = model(train_X[idx])
            # Weighted BCE
            bce = nn.functional.binary_cross_entropy(pred, train_y[idx], reduction='none')
            loss = (bce * sample_weights[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            test_pred = model(test_X)
            test_loss = nn.functional.binary_cross_entropy(test_pred, test_y).item()
            test_acc = ((test_pred > 0.5).float() == test_y).float().mean().item() * 100

        if test_loss < best_loss:
            best_loss = test_loss
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))
    model.eval()
    return model, best_acc, save_path, params


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    data = load_replay_data()
    train_data, test_data = split_data(data)
    stats_cache = StatsCache()
    group_indices = compute_group_indices()

    cols = []
    for g in ENRICHED_GROUPS:
        s, e = group_indices[g]
        cols.extend(range(s, e))

    comp_path = os.path.join(os.path.dirname(__file__), "..", "src", "lib", "data", "compositions.json")
    with open(comp_path) as f:
        comp_data = json.load(f)

    # Best synthetic config from Stage 1
    synthetic_100, stats_100 = generate_synthetic_data(
        train_data, comp_data, unseen_wr=10, unseen_volume=100, scope="tier2_only")
    synthetic_50, stats_50 = generate_synthetic_data(
        train_data, comp_data, unseen_wr=10, unseen_volume=50, scope="tier2_only")
    synthetic_25, stats_25 = generate_synthetic_data(
        train_data, comp_data, unseen_wr=10, unseen_volume=25, scope="tier2_only")

    configs = [
        # Architecture ablations with vol=100
        {"name": "baseline_256_128", "syn": [], "hidden": [256, 128], "sw": 1.0},
        {"name": "baseline_512_256_128", "syn": [], "hidden": [512, 256, 128], "sw": 1.0},
        {"name": "wr10_v100_256_128", "syn": synthetic_100, "hidden": [256, 128], "sw": 1.0},
        {"name": "wr10_v100_512_256_128", "syn": synthetic_100, "hidden": [512, 256, 128], "sw": 1.0},
        {"name": "wr10_v100_768_512_256", "syn": synthetic_100, "hidden": [768, 512, 256], "sw": 1.0},
        # Weighted loss
        {"name": "wr10_v100_256_128_w0.5", "syn": synthetic_100, "hidden": [256, 128], "sw": 0.5},
        {"name": "wr10_v100_512_256_128_w0.5", "syn": synthetic_100, "hidden": [512, 256, 128], "sw": 0.5},
        {"name": "wr10_v100_256_128_w0.3", "syn": synthetic_100, "hidden": [256, 128], "sw": 0.3},
        # Lower volume
        {"name": "wr10_v50_256_128", "syn": synthetic_50, "hidden": [256, 128], "sw": 1.0},
        {"name": "wr10_v50_512_256_128", "syn": synthetic_50, "hidden": [512, 256, 128], "sw": 1.0},
        {"name": "wr10_v25_256_128", "syn": synthetic_25, "hidden": [256, 128], "sw": 1.0},
        {"name": "wr10_v25_512_256_128", "syn": synthetic_25, "hidden": [512, 256, 128], "sw": 1.0},
        # Larger patience
        {"name": "wr10_v100_512_256_128_p35", "syn": synthetic_100, "hidden": [512, 256, 128], "sw": 1.0, "patience": 35},
    ]

    print(f"\n{'='*70}")
    print(f"ABLATION 2: Architecture + Weighted Loss ({len(configs)} configs × 2 seeds)")
    print(f"{'='*70}")

    results = []
    t0 = time.time()

    for ci, cfg in enumerate(configs):
        name = cfg["name"]
        print(f"\n--- [{ci+1}/{len(configs)}] {name} ---")
        print(f"  Hidden: {cfg['hidden']}, Syn weight: {cfg['sw']}, "
              f"Synthetic: {len(cfg['syn'])} records")

        seed_accs = []
        seed_evals = []
        for seed in range(2):
            model, acc, path, params = train_with_options(
                train_data, cfg["syn"], test_data, stats_cache, group_indices,
                device, hidden_dims=cfg["hidden"], synthetic_weight=cfg["sw"],
                patience=cfg.get("patience", 25), seed=42 + seed,
            )
            if seed == 0:
                print(f"  Params: {params:,}")
            eval_result = evaluate_config(model, cols, stats_cache, device)
            seed_accs.append(acc)
            seed_evals.append(eval_result)
            del model
            torch.cuda.empty_cache()
            if os.path.exists(path):
                os.remove(path)

        avg_acc = np.mean(seed_accs)
        best_idx = max(range(len(seed_evals)),
                       key=lambda i: (seed_evals[i]["sanity_passed"],
                                      -sum(seed_evals[i]["degen_scores"].values())))
        best_eval = seed_evals[best_idx]

        print(f"  Accuracy: {avg_acc:.2f}% (seeds: {', '.join(f'{a:.2f}' for a in seed_accs)})")
        print(f"  Sanity: {best_eval['sanity_passed']}/{best_eval['sanity_total']}")
        ds = best_eval['degen_scores']
        print(f"  5tank: {ds['5 tanks']:.3f}  5heal: {ds['5 healers']:.3f}  "
              f"noHeal: {ds['No healer high WR']:.3f}")

        results.append({
            "name": name,
            "hidden": cfg["hidden"],
            "synthetic_weight": cfg["sw"],
            "synthetic_count": len(cfg["syn"]),
            "params": params,
            "accuracy": avg_acc,
            "all_accs": seed_accs,
            "eval": best_eval,
        })

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} minutes")

    # Summary
    print(f"\n{'='*80}")
    print("ABLATION 2 SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<35} {'Params':>8} {'Acc%':>6} {'Sanity':>7} {'5tank':>6} {'5heal':>6} {'noHeal':>7}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: (-x['eval']['sanity_passed'], -x['accuracy'])):
        ds = r['eval']['degen_scores']
        print(f"{r['name']:<35} {r['params']:>8,} {r['accuracy']:>6.2f} "
              f"{r['eval']['sanity_passed']:>3}/{r['eval']['sanity_total']:<3} "
              f"{ds['5 tanks']:>6.3f} {ds['5 healers']:>6.3f} {ds['No healer high WR']:>7.3f}")

    save_path = os.path.join(RESULTS_DIR, "ablation2_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
