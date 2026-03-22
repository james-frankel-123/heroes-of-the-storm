"""
Focused ablation tests for enriched WP model with capabilities.

Tests specific combinations rather than exhaustive sweep.
Runs on all GPUs in parallel.

Usage:
    set -a && source .env && set +a
    python3 -u training/ablation_wp.py
"""
import os
import sys
import csv
import json
import time
import subprocess
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from sweep_enriched_wp import (
    StatsCache, WinProbEnrichedModel,
    FEATURE_GROUPS, FEATURE_GROUP_DIMS, compute_group_indices,
    precompute_all_features,
)
from shared import load_replay_data, split_data

# The current best groups (from enriched sweep)
CURRENT_BEST = ['role_counts', 'team_avg_wr', 'map_delta',
                'pairwise_counters', 'pairwise_synergies', 'counter_detail']


def build_ablations():
    """Build focused ablation configs."""
    configs = []

    # 1. Baseline: no enriched features
    configs.append(("base_only", []))

    # 2. Current best (reference)
    configs.append(("current_best", CURRENT_BEST))

    # 3. Current best + capabilities
    configs.append(("best+caps", CURRENT_BEST + ['capabilities']))

    # 4. Current best + capabilities + map_type
    configs.append(("best+caps+map", CURRENT_BEST + ['capabilities', 'map_type']))

    # 5. Everything
    configs.append(("all_features", list(FEATURE_GROUPS)))

    # 6. Drop each group from current best (essentialness test)
    for drop in CURRENT_BEST:
        remaining = [g for g in CURRENT_BEST if g != drop]
        configs.append((f"best-{drop}", remaining))

    # 7. Capabilities alone
    configs.append(("caps_only", ['capabilities']))

    # 8. Capabilities + role_counts (composition pair)
    configs.append(("caps+roles", ['capabilities', 'role_counts']))

    # 9. Capabilities + role_counts + pairwise
    configs.append(("caps+roles+pw", ['capabilities', 'role_counts', 'pairwise_counters', 'pairwise_synergies']))

    # 10. Capabilities + current best - counter_detail (counter_detail is 50 features, maybe redundant with caps)
    configs.append(("best+caps-detail", [g for g in CURRENT_BEST if g != 'counter_detail'] + ['capabilities']))

    # 11. Role_counts only
    configs.append(("roles_only", ['role_counts']))

    # 12. map_type + role_counts (previous sanity test winner at 18/21)
    configs.append(("map+roles", ['map_type', 'role_counts']))

    # 13. map_type + role_counts + capabilities
    configs.append(("map+roles+caps", ['map_type', 'role_counts', 'capabilities']))

    # 14. Current best + meta_strength
    configs.append(("best+meta", CURRENT_BEST + ['meta_strength']))

    # 15. Current best + draft_diversity
    configs.append(("best+diversity", CURRENT_BEST + ['draft_diversity']))

    # 16. Current best + meta + diversity
    configs.append(("best+meta+div", CURRENT_BEST + ['meta_strength', 'draft_diversity']))

    # 17. Current best + meta + diversity + caps
    configs.append(("best+meta+div+caps", CURRENT_BEST + ['meta_strength', 'draft_diversity', 'capabilities']))

    # 18. Meta alone
    configs.append(("meta_only", ['meta_strength']))

    # 19. Diversity alone
    configs.append(("diversity_only", ['draft_diversity']))

    # 20. Kitchen sink: everything
    configs.append(("everything", list(FEATURE_GROUPS)))

    return configs


def train_config(name, groups, train_base, train_enriched, train_labels,
                 test_base, test_enriched, test_labels, group_indices, device,
                 hidden_dims=[256, 128], lr=5e-4, save_path=None):
    """Train one config. Returns (name, best_test_acc, best_epoch)."""
    cols = []
    for g in groups:
        s, e = group_indices[g]
        cols.extend(range(s, e))

    extra_dim = len(cols)
    total_dim = 197 + extra_dim

    if cols:
        trX = torch.cat([train_base, train_enriched[:, cols]], dim=1).to(device)
        teX = torch.cat([test_base, test_enriched[:, cols]], dim=1).to(device)
    else:
        trX = train_base.to(device)
        teX = test_base.to(device)

    trY = train_labels.to(device)
    teY = test_labels.to(device)

    # Train 3 seeds and average for stability
    accs = []
    for seed in [42, 123, 777]:
        torch.manual_seed(seed)
        model = WinProbEnrichedModel(total_dim, hidden_dims, dropout=0.15).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
        crit = nn.BCELoss()

        best_loss = float('inf')
        best_acc = 0
        patience = 0

        for epoch in range(200):
            model.train()
            perm = torch.randperm(len(trX), device=device)
            for i in range(0, len(trX), 1024):
                idx = perm[i:i+1024]
                pred = model(trX[idx])
                loss = crit(pred, trY[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                te_pred = model(teX)
                te_loss = crit(te_pred, teY).item()
                te_acc = ((te_pred > 0.5).float() == teY).float().mean().item() * 100

            if te_loss < best_loss:
                best_loss = te_loss
                best_acc = te_acc
                patience = 0
                if save_path and seed == 42:
                    torch.save(model.state_dict(), save_path)
            else:
                patience += 1
                if patience >= 15:
                    break

        accs.append(best_acc)

    avg_acc = np.mean(accs)
    std_acc = np.std(accs)
    return name, groups, avg_acc, std_acc, total_dim


def run_on_gpu(gpu_id, config_batch, train_base, train_enriched, train_labels,
               test_base, test_enriched, test_labels, group_indices, result_queue):
    """Run a batch of configs on a specific GPU."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda')

    for i, (name, groups) in config_batch:
        save_path = None
        if name in ('best+caps', 'current_best', 'all_features', 'best+caps+map'):
            save_path = f'training/wp_ablation_{name.replace("+","_")}.pt'

        result = train_config(
            name, groups,
            train_base, train_enriched, train_labels,
            test_base, test_enriched, test_labels,
            group_indices, device,
            save_path=save_path,
        )
        name_r, groups_r, avg_acc, std_acc, dim = result
        g_str = "+".join(g[:8] for g in groups_r) if groups_r else "base"
        print(f"  [GPU {gpu_id}] {name_r:<30} {avg_acc:.2f}% +/-{std_acc:.2f}  dim={dim}  ({g_str})")
        result_queue.put(result)


def main():
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        num_gpus = 1
    print(f"GPUs: {num_gpus}")

    print("Loading data...")
    data = load_replay_data()
    train_data, test_data = split_data(data)
    stats = StatsCache()

    print("Precomputing features...")
    t0 = time.time()
    train_base, train_enriched, train_labels = precompute_all_features(train_data, stats)
    test_base, test_enriched, test_labels = precompute_all_features(test_data, stats)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Enriched dims: {train_enriched.shape[1]}")

    # Share tensors
    train_base.share_memory_()
    train_enriched.share_memory_()
    train_labels.share_memory_()
    test_base.share_memory_()
    test_enriched.share_memory_()
    test_labels.share_memory_()

    group_indices = compute_group_indices()
    configs = build_ablations()

    # Round-robin assign to GPUs
    gpu_batches = [[] for _ in range(num_gpus)]
    for i, cfg in enumerate(configs):
        gpu_batches[i % num_gpus].append((i, cfg))

    print(f"\nRunning {len(configs)} ablation configs (3 seeds each) across {num_gpus} GPUs...")
    for g in range(num_gpus):
        print(f"  GPU {g}: {len(gpu_batches[g])} configs")
    print(f"{'='*70}")

    result_queue = mp.Queue()
    processes = []
    for gpu_id in range(num_gpus):
        if gpu_batches[gpu_id]:
            p = mp.Process(target=run_on_gpu, args=(
                gpu_id, gpu_batches[gpu_id],
                train_base, train_enriched, train_labels,
                test_base, test_enriched, test_labels,
                group_indices, result_queue,
            ))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Sort and print leaderboard
    results.sort(key=lambda r: -r[2])
    print(f"\n{'='*70}")
    print("LEADERBOARD (sorted by avg accuracy, 3 seeds)")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Config':<30} {'Acc':>8} {'Std':>6} {'Dim':>5}")
    print(f"{'-'*55}")
    for i, (name, groups, avg, std, dim) in enumerate(results):
        marker = " ***" if 'caps' in name else ""
        print(f"{i+1:<5} {name:<30} {avg:>7.2f}% {std:>5.2f} {dim:>5}{marker}")

    # Specific comparisons
    print(f"\n{'='*70}")
    print("KEY COMPARISONS")
    res = {r[0]: r for r in results}

    if 'current_best' in res and 'best+caps' in res:
        diff = res['best+caps'][2] - res['current_best'][2]
        print(f"  Adding capabilities: {diff:+.2f}% ({res['current_best'][2]:.2f}% → {res['best+caps'][2]:.2f}%)")

    if 'base_only' in res and 'caps_only' in res:
        diff = res['caps_only'][2] - res['base_only'][2]
        print(f"  Capabilities alone vs base: {diff:+.2f}%")

    if 'roles_only' in res and 'caps+roles' in res:
        diff = res['caps+roles'][2] - res['roles_only'][2]
        print(f"  Adding caps to roles: {diff:+.2f}%")

    # Drop analysis
    print(f"\n  Feature essentialness (drop from current best):")
    if 'current_best' in res:
        base = res['current_best'][2]
        for name, _, avg, _, _ in results:
            if name.startswith('best-'):
                dropped = name.replace('best-', '')
                diff = avg - base
                print(f"    Drop {dropped:<25} → {diff:+.2f}% (essential={diff < -0.1})")


if __name__ == "__main__":
    main()
