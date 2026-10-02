"""
Kernel/trainer parity for the production (v2) MCTS kernel: the kernel's
in-kernel symmetrized WP of each finished self-play draft must equal the
Python WP of the same draft (extract_features + the trained model, serving
stats). Episodes cycle every map (Haunted Mines included) and tier, so the
91-hero and 15-map encodings, the lookup tables and the enriched features are
all exercised. Called by refresh.py (phase kparity); needs a GPU.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import torch

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TRAINING_DIR, "cuda_mcts"))

TOL = 2e-3          # fast-math kernel vs float32 torch
N_EPISODES = 256
SIMS = 50


def load_kernel():
    from shared import HERO_SET
    so_dir = os.path.join(TRAINING_DIR, "cuda_mcts", "h91" if HERO_SET == "v2" else "")
    so = [f for f in os.listdir(so_dir) if f.startswith("cuda_mcts_kernel.") and f.endswith(".so")]
    spec = importlib.util.spec_from_file_location("cuda_mcts_kernel", os.path.join(so_dir, so[0]))
    k = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k)
    return k


def run(wp_pt, gd_pt, wp_dim, policy_pt=None, seed=20261002):
    from shared import NUM_HEROES, NUM_MAPS, NUM_TIERS, HEROES, MAPS, SKILL_TIERS, HERO_TO_IDX
    from sweep_enriched_wp import StatsCache, WinProbEnrichedModel, compute_group_indices
    from experiment_synthetic_augmentation import ENRICHED_GROUPS, make_eval_fn
    from train_generic_draft import GenericDraftModel
    from train_draft_policy import AlphaZeroDraftNet
    from extract_weights import (extract_policy_weights, extract_gd_weights, extract_wp_weights,
                                 build_wp_net_offsets, extract_lookup_tables)
    kernel = load_kernel()
    gd = GenericDraftModel()
    gd.load_state_dict(torch.load(gd_pt, weights_only=True, map_location="cpu"))
    gd.eval()
    gd_flat, gd_offsets = extract_gd_weights(gd)
    wp = WinProbEnrichedModel(wp_dim, [256, 128], dropout=0.3)
    wp.load_state_dict(torch.load(wp_pt, weights_only=True, map_location="cpu"))
    wp.eval()
    wp_flat, wp_names = extract_wp_weights(wp)
    wp_offsets = build_wp_net_offsets(wp, wp_names, wp_dim)
    st = StatsCache()          # WP_STATS_PATH (refresh.py pins the serving stats)
    lut = extract_lookup_tables(st, step_embed_weights=None)
    net = AlphaZeroDraftNet(size="base", policy_head_type="linear")
    if policy_pt:
        net.load_state_dict(torch.load(policy_pt, weights_only=True, map_location="cpu"))
    net.eval()
    pol_flat, pol_offsets = extract_policy_weights(net)
    engine = kernel.MCTSKernelEngine(pol_flat, gd_flat, wp_flat, pol_offsets, gd_offsets,
                                     wp_offsets, lut, max_concurrent=128, device_id=0)
    cfg = np.array([[i % NUM_MAPS, (i // NUM_MAPS) % NUM_TIERS, i % 2]
                    for i in range(N_EPISODES)], dtype=np.int32)
    out = []
    for b in range(0, N_EPISODES, 128):
        out += list(engine.run_episodes(cfg[b:b + 128], SIMS, 2.0, seed + b))

    gi = compute_group_indices()
    cols = [c for g in ENRICHED_GROUPS for c in range(*gi[g])]
    f = make_eval_fn(wp, cols, st, torch.device("cpu"))
    H, M = NUM_HEROES, NUM_MAPS
    diffs, n_new_hero, n_new_map, n_bad = [], 0, 0, 0
    new_hero = HERO_TO_IDX.get("Xal'atath")
    for (kwp, _, ts, our), c in zip(out, cfg):
        ts = np.asarray(ts)
        t0 = [HEROES[i] for i in range(H) if ts[i] > 0.5]
        t1 = [HEROES[i] for i in range(H) if ts[H + i] > 0.5]
        mi = int(np.argmax(ts[3 * H:3 * H + M]))
        ti = int(np.argmax(ts[3 * H + M:3 * H + M + NUM_TIERS]))
        if len(t0) != 5 or len(t1) != 5 or mi != c[0] or ti != c[1]:
            n_bad += 1
            continue
        gmap, tier = MAPS[mi], SKILL_TIERS[ti]
        p0 = 0.5 * (f(t0, t1, gmap, tier) + 1 - f(t1, t0, gmap, tier))
        py = p0 if our == 0 else 1 - p0
        diffs.append(abs(float(kwp) - py))
        n_new_hero += new_hero is not None and (ts[new_hero] > 0.5 or ts[H + new_hero] > 0.5)
        n_new_map += mi == M - 1
    d = np.array(diffs)
    res = {"episodes": len(out), "compared": len(d), "malformed": n_bad,
           "max_abs_diff": float(d.max()) if len(d) else None,
           "mean_abs_diff": float(d.mean()) if len(d) else None,
           "with_new_hero": int(n_new_hero), "on_last_map": int(n_new_map),
           "num_heroes": H, "num_maps": M, "tol": TOL}
    res["pass"] = bool(len(d) and n_bad == 0 and d.max() <= TOL)
    return res


if __name__ == "__main__":
    print(json.dumps(run(*sys.argv[1:4]), indent=1))
