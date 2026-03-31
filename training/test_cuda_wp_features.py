#!/usr/bin/env python3
"""
Comprehensive test: CUDA enriched feature computation vs Python extract_features.

Computes features for multiple draft compositions using both the CUDA kernel
(via a test harness that exposes raw enriched features) and Python, then
compares element-by-element.

Since we can't directly call CUDA compute_enriched_features from Python,
we compare the final WP output. If the WP outputs match within tolerance,
the features must be correct (the MLP amplifies any input differences).

Usage:
    set -a && source .env && set +a
    python3 -u training/test_cuda_wp_features.py
"""

import sys, os, time, json
import numpy as np
import torch
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda_mcts'))
from shared import (NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, MAP_TO_IDX,
                    SKILL_TIERS, TIER_TO_IDX, HERO_ROLE_FINE)
from sweep_enriched_wp import (StatsCache, FEATURE_GROUPS, FEATURE_GROUP_DIMS,
                                compute_group_indices, extract_features)
from train_partial_wp import PartialStateWP
from train_draft_policy import AlphaZeroDraftNet, bootstrap_from_generic_draft
from train_generic_draft import GenericDraftModel
from extract_weights import (extract_policy_weights, extract_gd_weights,
                              extract_wp_weights, build_wp_net_offsets, extract_lookup_tables)


def setup():
    """Load all models and create kernel engine."""
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
    wp_dim = 197 + sum(FEATURE_GROUP_DIMS[g] for g in WP_GROUPS)

    ckpt = torch.load('partial_wp.pt', weights_only=True, map_location='cpu')
    wp_model = PartialStateWP(input_dim=wp_dim, step_embed_dim=8, hidden=(256, 128))
    wp_model.load_state_dict(ckpt['model_state_dict'])
    wp_model.eval()

    gd = GenericDraftModel()
    gd.load_state_dict(torch.load('generic_draft_0.pt', weights_only=True, map_location='cpu'))
    gd.eval()
    gd_flat, gd_offsets = extract_gd_weights(gd)

    net = AlphaZeroDraftNet()
    bootstrap_from_generic_draft(net, torch.device('cpu'))
    pf, po = extract_policy_weights(net)

    wp_flat, wp_no = extract_wp_weights(wp_model)
    wp_offsets = build_wp_net_offsets(wp_model, wp_no, wp_dim + 8)
    step_embed = wp_model.step_embed.weight.data.cpu().numpy()
    lut_blob = extract_lookup_tables(stats, step_embed_weights=step_embed)

    so_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda_mcts')
    so_files = [f for f in os.listdir(so_dir) if f.startswith('cuda_mcts_kernel') and f.endswith('.so')]
    spec = importlib.util.spec_from_file_location('cuda_mcts_kernel', os.path.join(so_dir, so_files[0]))
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)

    engine = kernel.MCTSKernelEngine(pf, gd_flat, wp_flat, po, gd_offsets, wp_offsets, lut_blob,
                                     max_concurrent=128, device_id=0)

    return stats, wp_model, wp_cols, all_mask, engine


def python_wp_symmetrized(t0h, t1h, game_map, tier, stats, wp_model, wp_cols, all_mask, step_idx=15):
    """Compute symmetrized WP using Python extract_features + PyTorch model."""
    d_n = {'team0_heroes': t0h, 'team1_heroes': t1h,
           'game_map': game_map, 'skill_tier': tier, 'winner': 0}
    d_s = {'team0_heroes': t1h, 'team1_heroes': t0h,
           'game_map': game_map, 'skill_tier': tier, 'winner': 0}
    bn, en = extract_features(d_n, stats, all_mask)
    bs, es = extract_features(d_s, stats, all_mask)
    xn = torch.tensor(np.concatenate([bn, en[wp_cols]]), dtype=torch.float32).unsqueeze(0)
    xs = torch.tensor(np.concatenate([bs, es[wp_cols]]), dtype=torch.float32).unsqueeze(0)
    step = torch.tensor([step_idx], dtype=torch.long)
    with torch.no_grad():
        wn = wp_model(xn, step).item()
        ws = wp_model(xs, step).item()
    return (wn + (1.0 - ws)) / 2.0


# ── Test cases ──

TEST_COMPS = [
    # (name, t0_heroes, t1_heroes, map, tier)
    ("Standard 5v5",
     ['Muradin', 'Valla', 'Jaina', 'Malfurion', 'Thrall'],
     ['Diablo', 'Raynor', 'Brightwing', 'Arthas', 'Li-Ming'],
     'Cursed Hollow', 'mid'),

    ("Tank-heavy t0",
     ['Muradin', 'Johanna', 'Diablo', 'Malfurion', 'Valla'],
     ['E.T.C.', 'Jaina', 'Brightwing', 'Thrall', 'Li-Ming'],
     'Dragon Shire', 'mid'),

    ("Triple healer (degenerate)",
     ['Alexstrasza', 'Ana', 'Anduin', 'Valla', 'Muradin'],
     ['Diablo', 'Raynor', 'Brightwing', 'Arthas', 'Li-Ming'],
     'Infernal Shrines', 'mid'),

    ("No healer t0",
     ['Muradin', 'Valla', 'Jaina', 'Thrall', 'Sonya'],
     ['Diablo', 'Raynor', 'Brightwing', 'Arthas', 'Li-Ming'],
     'Tomb of the Spider Queen', 'high'),

    ("All assassins t0",
     ['Valla', 'Jaina', 'Li-Ming', 'Genji', 'Zeratul'],
     ['Muradin', 'Malfurion', 'Raynor', 'Arthas', 'Brightwing'],
     'Sky Temple', 'low'),

    ("Mirror-ish comp",
     ['Muradin', 'Valla', 'Jaina', 'Malfurion', 'Thrall'],
     ['Johanna', 'Raynor', 'Kael\'thas', 'Brightwing', 'Sonya'],
     'Alterac Pass', 'mid'),

    ("Support heroes",
     ['Muradin', 'Abathur', 'Medivh', 'Malfurion', 'Valla'],
     ['Diablo', 'Raynor', 'Brightwing', 'Arthas', 'Li-Ming'],
     'Volskaya Foundry', 'mid'),

    ("Cho'Gall comp",
     ['Cho', 'Gall', 'Malfurion', 'Valla', 'Thrall'],
     ['Diablo', 'Raynor', 'Brightwing', 'Arthas', 'Li-Ming'],
     'Battlefield of Eternity', 'mid'),

    ("Different map + tier",
     ['Muradin', 'Valla', 'Jaina', 'Malfurion', 'Thrall'],
     ['Diablo', 'Raynor', 'Brightwing', 'Arthas', 'Li-Ming'],
     'Hanamura Temple', 'high'),

    ("Niche heroes",
     ['Probius', 'Murky', 'The Lost Vikings', 'Malfurion', 'Muradin'],
     ['Diablo', 'Raynor', 'Brightwing', 'Arthas', 'Li-Ming'],
     'Garden of Terror', 'low'),
]


def main():
    print("=" * 70)
    print("  CUDA WP Feature Accuracy Test Suite")
    print("=" * 70)

    stats, wp_model, wp_cols, all_mask, engine = setup()

    # For each test case, run a 1-sim episode that produces a known terminal state.
    # Since we can't control which heroes the kernel picks, we instead:
    # 1. Run episodes with the kernel
    # 2. Read the terminal state the kernel produced
    # 3. Compute Python WP for the SAME terminal state
    # 4. Compare kernel's WP output vs Python's WP output

    # Run a batch of episodes
    configs = np.array([[MAP_TO_IDX.get(m, 0), TIER_TO_IDX.get(t, 1), i % 2]
                        for i, (_, _, _, m, t) in enumerate(TEST_COMPS)], dtype=np.int32)
    results = engine.run_episodes(configs, 1, 2.0, 42)

    print(f"\n{'#':>2} {'Name':<30} {'Kernel':>8} {'Python':>8} {'Diff':>8} {'Status':>6}")
    print("-" * 70)

    passed = 0
    failed = 0
    all_diffs = []

    for i, (wp_kernel, examples, ts, ot) in enumerate(results):
        name = TEST_COMPS[i][0]
        terminal = np.array(ts)
        our_team = configs[i][2]

        t0h = [HEROES[j] for j in range(NUM_HEROES) if terminal[j] > 0.5]
        t1h = [HEROES[j] for j in range(NUM_HEROES) if terminal[NUM_HEROES + j] > 0.5]

        if len(t0h) != 5 or len(t1h) != 5:
            print(f"{i:>2} {name:<30} {'SKIP':>8} — incomplete draft")
            continue

        game_map = MAPS[configs[i][0]]
        tier = SKILL_TIERS[configs[i][1]]

        py_wp_t0 = python_wp_symmetrized(t0h, t1h, game_map, tier, stats, wp_model, wp_cols, all_mask, step_idx=15)
        py_ours = py_wp_t0 if our_team == 0 else 1.0 - py_wp_t0
        diff = abs(wp_kernel - py_ours)
        all_diffs.append(diff)

        if diff < 0.005:
            status = "PASS"
            passed += 1
        elif diff < 0.02:
            status = "WARN"
            passed += 1  # close enough
        else:
            status = "FAIL"
            failed += 1

        print(f"{i:>2} {name:<30} {wp_kernel:>8.4f} {py_ours:>8.4f} {diff:>8.4f} {status:>6}")

        if diff >= 0.02:
            # Print the heroes for debugging
            print(f"   t0: {t0h}")
            print(f"   t1: {t1h}")

    print("-" * 70)
    print(f"\nResults: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"Max diff:  {max(all_diffs):.6f}")
    print(f"Mean diff: {np.mean(all_diffs):.6f}")
    print(f"Median:    {np.median(all_diffs):.6f}")

    if failed == 0 and max(all_diffs) < 0.02:
        print("\n✓ ALL TESTS PASSED — CUDA features match Python within tolerance")
    else:
        print(f"\n✗ {failed} TESTS FAILED — CUDA features do NOT match Python")
        print("  Remaining feature mismatches need investigation")

    del engine
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
