"""
W2b job runner: POLICY-LEVEL evaluation of one wave-1 drift-regime value
function on the held-out future period. Paper-1-style greedy drafting
(experiment_rich_evaluation machinery, same one phase3_benchmarks reuses):
the VF drafts greedily with GD-argmax rollouts, opponents are the rerun2026
GD pool (sampled), N drafts over random map/tier/side configs.

Drift-specific wiring:
  - The DRAFTER extracts enriched features with the arm's DEPLOYMENT-TIME
    stats (what that regime would ship with during the future period):
        d2b_* (features=cutoff)  cumulative-through-TRAIN_CUTOFF_BUILD
        d2c_cumprev              cumulative through the last completed build
                                 (patch-boundary refresh, deploy_stats conv.)
  - The METRICS (counter responsiveness / synergy exploitation) are scored
    against FUTURE-PERIOD GROUND TRUTH stats: per-build counts of all builds
    AFTER the cutoff merged (patch_counts.pkl.gz), i.e. what was actually
    true in the held-out period. Same truth object for every arm.
  - healer/degen rates and diversity need no stats.

Key question (paper-1 thesis applied to drift): do the small future-accuracy
gaps between regimes hide large policy-level gaps?

Usage (one cell x seed per process, launched by phase_w2.py):
    python drift2026/eval_policy_w2b.py --cell d2b_allhist --seed 42 --drafts 500
    python drift2026/eval_policy_w2b.py --cell gd --seed 42 --drafts 1500
"""
import os
import sys
import json
import gzip
import time
import pickle
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

common.setup()

import numpy as np
import torch

from shared import MAPS, SKILL_TIERS
from sweep_enriched_wp import compute_group_indices, WinProbEnrichedModel
from experiment_synthetic_augmentation import ENRICHED_GROUPS
from experiment_rich_evaluation import (
    run_drafts_with_strategy, make_wp_greedy_strategy, make_gd_strategy,
    counter_responsiveness, synergy_exploitation, draft_diversity,
    map_adaptation)
from drift2026.train_drift_wp import WPWithPatchEmbed
from drift2026.build_patch_stats import _new_cell, merge_cell

RERUN_MODELS = os.path.join(common.TRAINING_DIR, "rerun2026", "models")

# cell -> feature sourcing of the wave-1 arm (drives deployment stats choice)
CELLS = {
    "d2b_allhist": "cutoff",
    "d2b_win3": "cutoff",     # W2B_FILL: Table II missing cells
    "d2b_win6": "cutoff",
    "d2b_win12": "cutoff",    # W2B_FILL
    "d2b_decay90": "cutoff",
    "d2b_decay365": "cutoff",  # W2B_FILL
    "d2b_embed": "cutoff",
    "d2c_cumprev": "cumulative_prev",
    "q7_decayed90": "decayed90",          # thesis-figure check (2026-07-14)
    "q7_decayed90k100": "decayed90k100",  # thesis-figure check
    "w3d_hybrid": "hybrid",             # W3(d) per-signal hybrid sourcing
    "w3e_embedrefresh": "cumulative_prev",  # W3(e) embed regime + refresh
    "gd": None,   # GD-argmax reference (no VF)
}


class FixedEmbed(torch.nn.Module):
    """Embed-regime model pinned to the cutoff build's embedding (the only
    causal deployment choice) so it exposes a plain forward(x)."""

    def __init__(self, m, p):
        super().__init__()
        self.m, self.p = m, p

    def forward(self, x):
        return self.m(x, torch.full((len(x),), self.p, dtype=torch.long,
                                    device=x.device))


def load_cell_model(cell, seed, device):
    ck = torch.load(os.path.join(common.MODELS_DIR, f"{cell}_s{seed}.pt"),
                    weights_only=True, map_location="cpu")
    if ck.get("embed"):
        # cfg input_dim is the feature dim (283); the embed dim is appended
        # inside WPWithPatchEmbed.
        m = WPWithPatchEmbed(ck["input_dim"], ck["n_patches"],
                             ck["embed_dim"], arch=tuple(ck["arch"]),
                             dropout=ck["dropout"])
        m.load_state_dict(ck["state_dict"])
        model = FixedEmbed(m, ck["n_patches"] - 1)
    else:
        model = WinProbEnrichedModel(ck["input_dim"], list(ck["arch"]),
                                     ck["dropout"])
        model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def deployment_stats(features):
    """Stats the arm's drafter would use during the future period (same
    convention as train_drift_wp.deploy_stats)."""
    builds_255 = [b for b in common.load_patch_index()["builds"]
                  if b.startswith("2.55")]
    if features == "cumulative_prev":
        return common.load_patch_stats("cumulative", builds_255[-2])
    if features == "hybrid":
        return common.load_patch_stats("hybrid", builds_255[-1])
    if features in ("decayed90", "decayed90k100"):
        # Decayed-aggregate deployment: refreshed to the last completed
        # build, same convention as cumulative_prev.
        return common.load_patch_stats(features, builds_255[-2])
    return common.load_patch_stats("cumulative", common.TRAIN_CUTOFF_BUILD)


def future_truth_stats():
    """Ground-truth stats of the held-out period: merged per-build counts of
    every build strictly after the cutoff."""
    with gzip.open(common.COUNTS_PKL, "rb") as f:
        data = pickle.load(f)
    cut = data["builds"].index(common.TRAIN_CUTOFF_BUILD)
    merged = {}
    n_builds = 0
    for bidx, by_tier in data["per_build"].items():
        if bidx <= cut:
            continue
        n_builds += 1
        for tier, plain in by_tier.items():
            cell = merged.get(tier)
            if cell is None:
                cell = merged[tier] = _new_cell()
            merge_cell(cell, plain)
    games = sum(c["games"] for c in merged.values())
    print(f"future-truth stats: {n_builds} builds, {games:,} games")
    return common.stats_from_counts(merged, pair_min=10)


def load_gd_models(pool="frozen"):
    """Opponent/completion pool. 'frozen' = rerun2026 full-snapshot GD x5
    (has seen the future period); 'cutoff' = W5 GD_cutoff x2 (trained on
    replays <= TRAIN_CUTOFF_BUILD only — what was deployable at the cutoff)."""
    from train_generic_draft import GenericDraftModel
    if pool == "cutoff":
        paths = [os.path.join(common.MODELS_DIR, "gd_cutoff",
                              f"generic_draft_{i}.pt") for i in range(2)]
    else:
        paths = [os.path.join(RERUN_MODELS, f"generic_draft_{i}.pt")
                 for i in range(5)]
    models = []
    for p in paths:
        gd = GenericDraftModel()
        gd.load_state_dict(torch.load(p, weights_only=True, map_location="cpu"))
        gd.eval()
        models.append(gd)
    return models


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drafts", type=int, default=500)
    ap.add_argument("--gd-pool", default="frozen", choices=["frozen", "cutoff"],
                    help="opponent/completion GD pool (W5 GD-drift arm)")
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gd_models = load_gd_models(args.gd_pool)
    truth = future_truth_stats()

    if args.cell == "gd":
        label, strategy = "GD baseline", make_gd_strategy()
    else:
        features = CELLS[args.cell]
        stats = deployment_stats(features)
        model = load_cell_model(args.cell, args.seed, device)
        strategy = make_wp_greedy_strategy(model, ENRICHED_GROUPS, stats,
                                           compute_group_indices(), device)
        label = f"{args.cell} greedy"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    draft_configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
                     for i in range(args.drafts)]
    drafts = run_drafts_with_strategy(strategy, draft_configs, gd_models,
                                      truth, device)

    n = len(drafts)
    counters = [counter_responsiveness(d["our_picks"], d["opp_picks"], truth,
                                       d["tier"]) for d in drafts]
    synergies = [synergy_exploitation(d["our_picks"], truth, d["tier"])
                 for d in drafts]
    result = {
        "cell": args.cell, "label": label, "seed": args.seed, "drafts": n,
        "gd_pool": args.gd_pool,
        "healer_rate": round(sum(d["has_healer"] for d in drafts) / n * 100, 2),
        "degen_rate": round(sum(d["is_degen"] for d in drafts) / n * 100, 2),
        "counter_future": round(float(np.mean(counters)), 3),
        "synergy_future": round(float(np.mean(synergies)), 3),
        **draft_diversity(drafts),
        "map_adapt": map_adaptation(drafts),
        "truth": "merged per-build counts of all post-cutoff builds",
        "minutes": round((time.time() - t0) / 60, 1),
    }
    suffix = "" if args.gd_pool == "frozen" else f"_gd{args.gd_pool}"
    out = os.path.join(common.RESULTS_DIR, "w2b",
                       f"{args.cell}_s{args.seed}{suffix}.json")
    common.write_json(out, result)
    print(f"{args.cell}_s{args.seed}: healer={result['healer_rate']}% "
          f"degen={result['degen_rate']}% ctr={result['counter_future']:+.2f} "
          f"syn={result['synergy_future']:+.2f} ({result['minutes']} min)")


if __name__ == "__main__":
    main()
