"""
Kernel/trainer parity of the research MCTS kernel (cuda_mcts, v1) on the
site-tier setup that paper1_revision/train_mcts.py trains with
(P1_TIERS=site): leaf WP models/site/{enriched,enriched_leak}_s<sel>.pt,
lookup tables from the own deploy statistics and own composition table, GD
opponent of rerun2026 namespace p1site. The kernel's in-kernel symmetrized WP
of each finished self-play draft must equal the Python WP of the same draft
(extract_features + the model, same StatsCache). Episodes cycle maps, all
three tiers and both sides; drafts where either team's role composition is
missing from its own tier's table but present in another (the composition
fallback) are reported separately.

Usage (from training/, GPU): P1_TIERS=site python3 paper1_revision/verify_kernel_site.py
Appends to paper1_revision/site/results/verify_kernel_site.json
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, os.path.join(TRAINING_DIR, "cuda_mcts"))
from paper1_revision import core, train_wp  # noqa: E402

os.environ["WP_STATS_PATH"] = core.stats_path("deploy")   # read by sweep_enriched_wp at import

TOL = 2e-3
N_EPISODES = 768
SIMS = 50
HP_ROLE = {"tank": "Tank", "bruiser": "Bruiser", "healer": "Healer", "ranged_aa": "Ranged Assassin",
           "ranged_mage": "Ranged Assassin", "melee_assassin": "Melee Assassin",
           "support_utility": "Support", "varian": "Bruiser", "pusher": "Ranged Assassin",
           "unknown": "Ranged Assassin"}


def load_kernel():
    d = os.path.join(TRAINING_DIR, "cuda_mcts")
    so = [f for f in os.listdir(d) if f.startswith("cuda_mcts_kernel.") and f.endswith(".so")]
    spec = importlib.util.spec_from_file_location("cuda_mcts_kernel", os.path.join(d, so[0]))
    k = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k)
    return k


def stats_cache():
    import sweep_enriched_wp as swp
    comp = core.comp_path("deploy")

    def _load_compositions(self):   # as train_mcts.worker()
        raw = json.load(open(comp))
        self.comp_data = {t: {",".join(sorted(c["roles"])): (c["winRate"], c["games"])
                              for c in cs} for t, cs in raw.items()}
    swp.StatsCache._load_compositions = _load_compositions
    return swp.StatsCache()


def comp_key(team):
    from sweep_enriched_wp import HERO_ROLE_FINE
    return ",".join(sorted(HP_ROLE.get(HERO_ROLE_FINE.get(h, "unknown"), "Ranged Assassin")
                           for h in team))


def fallback_kind(team, tier, comp_data):
    """None if the own tier has the composition; else the tier the Python
    order (mid, high, low) takes and the tier the index order takes."""
    key = comp_key(team)
    if key in comp_data.get(tier, {}):
        return None
    py = next((t for t in ("mid", "high", "low") if t != tier and key in comp_data.get(t, {})), None)
    ix = next((t for t in ("low", "mid", "high") if t != tier and key in comp_data.get(t, {})), None)
    return (py, ix)


def run(spec, kernel, st, seed=20261003):
    from shared import NUM_HEROES, NUM_MAPS, NUM_TIERS, HEROES, MAPS, SKILL_TIERS
    from sweep_enriched_wp import WinProbEnrichedModel
    from sweep_enriched_wp import extract_features, FEATURE_GROUPS
    from train_generic_draft import GenericDraftModel
    from train_draft_policy import AlphaZeroDraftNet
    from extract_weights import (extract_policy_weights, extract_gd_weights, extract_wp_weights,
                                 build_wp_net_offsets, extract_lookup_tables)
    meta = json.load(open(os.path.join(core.MODEL_DIR, f"{spec}.json")))
    wp_pt = os.path.join(core.MODEL_DIR, f"{spec}_s{meta['selected_seed']}.pt")
    _, cols = train_wp.load(spec)
    dim = len(cols)
    assert dim == meta["input_dim"] == 283, (dim, meta["input_dim"])
    gd = GenericDraftModel()
    gd.load_state_dict(torch.load(os.path.join(TRAINING_DIR, "rerun2026", "ns", "p1site", "models",
                                               "generic_draft_0.pt"),
                                  weights_only=True, map_location="cpu"))
    gd.eval()
    gd_flat, gd_offsets = extract_gd_weights(gd)
    wp = WinProbEnrichedModel(dim, [256, 128], dropout=0.3)
    wp.load_state_dict(torch.load(wp_pt, weights_only=True, map_location="cpu"))
    wp.eval()
    wp_flat, wp_names = extract_wp_weights(wp)
    wp_offsets = build_wp_net_offsets(wp, wp_names, dim)
    lut = extract_lookup_tables(st, step_embed_weights=None)
    net = AlphaZeroDraftNet(size="base", policy_head_type="linear")
    net.eval()
    pol_flat, pol_offsets = extract_policy_weights(net)
    engine = kernel.MCTSKernelEngine(pol_flat, gd_flat, wp_flat, pol_offsets, gd_offsets,
                                     wp_offsets, lut, max_concurrent=128, device_id=0)
    cfg = np.array([[i % NUM_MAPS, (i // NUM_MAPS) % NUM_TIERS, (i // (NUM_MAPS * NUM_TIERS)) % 2]
                    for i in range(N_EPISODES)], dtype=np.int32)
    out = []
    for b in range(0, N_EPISODES, 128):
        # c_puct 2, root temperature 2 (varied teams), chance search, pw 1.0/0.5
        # (train_mcts_worker defaults)
        out += list(engine.run_episodes(cfg[b:b + 128], SIMS, 2.0, seed + b, 2.0, 0.3, 0.0,
                                        kernel.SEARCH_CHANCE, 1.0, 0.5))
    mask = [True] * len(FEATURE_GROUPS)

    def f(t0, t1, gmap, tier):
        b_, e_ = extract_features({"team0_heroes": t0, "team1_heroes": t1, "game_map": gmap,
                                   "skill_tier": tier, "winner": 0}, st, mask)
        x = np.concatenate([b_, e_])[cols].astype(np.float32)
        with torch.no_grad():
            return wp(torch.tensor(x).unsqueeze(0)).item()
    H, M = NUM_HEROES, NUM_MAPS
    rows, n_bad = [], 0
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
        fk = [fallback_kind(t, tier, st.comp_data) for t in (t0, t1)]
        rows.append({"d": abs(float(kwp) - py), "tier": tier,
                     "fb": any(k is not None and k[0] is not None for k in fk),
                     "order_differs": any(k is not None and k[0] != k[1] for k in fk)})
    d = np.array([r["d"] for r in rows])
    fb = np.array([r["fb"] for r in rows])
    od = np.array([r["order_differs"] for r in rows])
    res = {"wp": spec, "wp_path": wp_pt, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "kernel_comp_fallback_order": getattr(kernel, "COMP_FALLBACK_ORDER", "low_mid_high (pre-attribute build)"),
           "episodes": len(out), "compared": len(d), "malformed": n_bad,
           "per_tier": {t: int(sum(r["tier"] == t for r in rows)) for t in SKILL_TIERS},
           "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
           "fallback_drafts": int(fb.sum()), "order_differs_drafts": int(od.sum()),
           "order_differs_share": float(od.mean()),
           "max_abs_diff_order_differs": float(d[od].max()) if od.any() else None,
           "mean_abs_diff_order_differs": float(d[od].mean()) if od.any() else None,
           "max_abs_diff_rest": float(d[~od].max()) if (~od).any() else None,
           "n_over_tol": int((d > TOL).sum()), "tol": TOL}
    res["pass"] = bool(n_bad == 0 and d.max() <= TOL)
    return res


def main():
    kernel = load_kernel()
    st = stats_cache()
    allres = [run(spec, kernel, st) for spec in ("enriched", "enriched_leak")]
    print(json.dumps(allres, indent=1))
    out = os.path.join(core.RESULTS, "verify_kernel_site.json")
    hist = json.load(open(out)) if os.path.exists(out) else []
    json.dump(hist + allres, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
