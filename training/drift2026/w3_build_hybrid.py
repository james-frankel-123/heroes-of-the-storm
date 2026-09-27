"""
W3(d) feature prerequisite: HYBRID PER-SIGNAL stats + feature pass.

Per-signal-class sourcing informed by W3(a) recovery + D2a decay:
  hero-WR family (hero WR / pick+ban rate / hero-map WR)
        <- RECENT WINDOW: the smallest set of builds strictly before the
           row's build whose pooled games reach the W3(a) recovery target
           (results/w3_recovery.json hybrid_window_games; hero WR is the
           fastest-decaying signal, so recency > volume once estimates are
           stable)
  role-comp WR      <- ALL HISTORY (cumulative through the previous build;
                       essentially static per D2a)
  pairwise with/against <- CUMULATIVE through the previous build (the
                       interaction term is weak-but-stable; volume wins)

Strictly causal like cumulative_prev: every row's stats use only games from
builds before its own; build 0 gets empty stats.

Writes patch_stats/hybrid/<build>.json.gz (hero family from window,
pair/comp from cumulative) for every build after the first, then runs the
standard feature pass -> feature_cache/features_hybrid.npz via
build_drift_features (pass name "hybrid").

Usage:
    python drift2026/w3_build_hybrid.py [--window-games N] [--force]
"""
import os
import sys
import json
import gzip
import pickle
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common
from drift2026.build_patch_stats import _new_cell, merge_cell

DEFAULT_WINDOW_GAMES = 30000


def window_games_target():
    p = os.path.join(common.RESULTS_DIR, "w3_recovery.json")
    if os.path.exists(p):
        with open(p) as f:
            return int(json.load(f)["hybrid_window_games"])
    print(f"WARNING: {p} missing; using default {DEFAULT_WINDOW_GAMES}")
    return DEFAULT_WINDOW_GAMES


def derive_hybrid_file(build, window_by_tier, cum_by_tier, meta):
    """hero_stats/hero_map_stats from window counts; pairwise/comp from
    cumulative counts. Thresholds match build_patch_stats.derive_stats_file
    (hero >= 20, map/comp >= 5, pairwise >= 10)."""
    hero_stats, hero_map_stats, pairwise_stats, comp_stats = [], [], [], []
    for tier, cell in window_by_tier.items():
        total = cell["games"]
        if total == 0:
            continue
        for h, (g, w) in cell["hero"].items():
            if g < 20:
                continue
            hero_stats.append({
                "hero": h, "tier": tier, "games": g,
                "win_rate": round(100.0 * w / g, 3),
                "pick_rate": round(100.0 * g / total, 3),
                "ban_rate": round(100.0 * cell["bans"].get(h, 0) / total, 3)})
        for (m, h), (g, w) in cell["hmap"].items():
            if g < 5:
                continue
            hero_map_stats.append({
                "hero": h, "map": m, "tier": tier, "games": g,
                "win_rate": round(100.0 * w / g, 3)})
    for tier, cell in cum_by_tier.items():
        for (a, b), (g, w) in cell["with"].items():
            if g < 10:
                continue
            wr = round(100.0 * w / g, 3)
            pairwise_stats.append({"hero_a": a, "hero_b": b, "tier": tier,
                                   "relationship": "with", "win_rate": wr,
                                   "games": g})
            pairwise_stats.append({"hero_a": b, "hero_b": a, "tier": tier,
                                   "relationship": "with", "win_rate": wr,
                                   "games": g})
        for (a, b), (g, wa) in cell["against"].items():
            if g < 10:
                continue
            wr_a = round(100.0 * wa / g, 3)
            pairwise_stats.append({"hero_a": a, "hero_b": b, "tier": tier,
                                   "relationship": "against",
                                   "win_rate": wr_a, "games": g})
            pairwise_stats.append({"hero_a": b, "hero_b": a, "tier": tier,
                                   "relationship": "against",
                                   "win_rate": round(100.0 - wr_a, 3),
                                   "games": g})
        for ck, (g, w) in cell["comp"].items():
            if g < 5:
                continue
            comp_stats.append({"roles": ck, "tier": tier, "games": g,
                               "win_rate": round(100.0 * w / g, 3)})

    path = common.stats_path("hybrid", build)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump({"_meta": meta, "hero_stats": hero_stats,
                   "hero_map_stats": hero_map_stats,
                   "pairwise_stats": pairwise_stats,
                   "comp_stats": comp_stats}, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-games", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()
    common.setup()

    target = args.window_games or window_games_target()
    print(f"hybrid window target: {target:,} games (hero-WR family)")

    with gzip.open(common.COUNTS_PKL, "rb") as f:
        data = pickle.load(f)
    builds, per_build = data["builds"], data["per_build"]
    present = sorted(per_build)

    games_of = {bi: sum(c["games"] for c in per_build[bi].values())
                for bi in present}

    cum = {}   # tier -> cell, cumulative through the PREVIOUS build
    for pos, bi in enumerate(present):
        build = builds[bi]
        if pos > 0:
            # window: newest previous builds until >= target games
            win_pos, acc = [], 0
            for q in range(pos - 1, -1, -1):
                win_pos.append(q)
                acc += games_of[present[q]]
                if acc >= target:
                    break
            window = {}
            for q in win_pos:
                for tier, cell in per_build[present[q]].items():
                    if tier not in window:
                        window[tier] = _new_cell()
                    merge_cell(window[tier], cell)
            meta = {"build": build, "kind": "hybrid",
                    "window_games_target": target,
                    "window_builds": [builds[present[q]] for q in win_pos],
                    "window_games": acc,
                    "cumulative_through": builds[present[pos - 1]],
                    "source": "pinned snapshot 2026-05-22 (own corpus)"}
            # always (re)write: cheap, and the window target may have changed
            derive_hybrid_file(build, window, cum, meta)
            print(f"  {build}: window={len(win_pos)} builds/{acc:,} games,"
                  f" cum through {meta['cumulative_through']}", flush=True)
        # extend cumulative AFTER deriving (cum must lag by one build)
        for tier, cell in per_build[bi].items():
            if tier not in cum:
                cum[tier] = _new_cell()
            merge_cell(cum[tier], cell)

    if args.stats_only:
        return
    # feature pass (build_drift_features knows the "hybrid" pass)
    import subprocess
    cmd = [sys.executable,
           os.path.join(common.DRIFT_DIR, "build_drift_features.py"),
           "--only", "hybrid"] + (["--force"] if args.force else [])
    print("Running:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
