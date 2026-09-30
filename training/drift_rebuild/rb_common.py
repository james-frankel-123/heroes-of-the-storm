"""Shared paths/helpers for the drift-paper rebuild (2026-09-29).

Everything the rebuild writes lives under training/drift_rebuild/. The
drift2026 suite and every pre-registration-frozen file are imported, never
edited.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

CACHE_DIR = os.path.join(HERE, "feature_cache")
MODELS_DIR = os.path.join(HERE, "models")
RESULTS_DIR = os.path.join(HERE, "results")
LOGS_DIR = os.path.join(HERE, "logs")
RUNS_DIR = os.path.join(HERE, "mcts_runs")
for _d in (CACHE_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, RUNS_DIR):
    os.makedirs(_d, exist_ok=True)

K_FOLDS = 5


def fold_of(replay_id, k=K_FOLDS):
    """Deterministic hash fold (splitmix64 finalizer), independent of time."""
    z = (int(replay_id) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z % k


def stats_from_counts_rounded(counts_by_tier, pair_min=10):
    """drift2026.common.stats_from_counts with the exact rounding and
    thresholds of build_patch_stats.derive_stats_file (3-decimal WRs), so a
    stats object derived from the full cutoff counts reproduces the
    cumulative stats FILE bit-for-bit."""
    from drift2026 import common
    common._bind_statscache_methods()
    hero_wr, hero_meta, hero_map_wr, pairwise, comp_data = {}, {}, {}, {}, {}
    for tier, cell in counts_by_tier.items():
        total = cell["games"]
        if not total:
            continue
        hw = hero_wr.setdefault(tier, {})
        hm = hero_meta.setdefault(tier, {})
        for h, (g, w) in cell["hero"].items():
            if g < 20:
                continue
            hw[h] = round(100.0 * w / g, 3)
            hm[h] = (round(100.0 * g / total, 3),
                     round(100.0 * cell["bans"].get(h, 0) / total, 3))
        hmap = hero_map_wr.setdefault(tier, {})
        for (m, h), (g, w) in cell["hmap"].items():
            if g >= 5:
                hmap.setdefault(m, {})[h] = (round(100.0 * w / g, 3), g)
        pw = pairwise.setdefault(tier, {})
        for (a, b), (g, w) in cell["with"].items():
            if g < pair_min:
                continue
            wr = round(100.0 * w / g, 3)
            pw.setdefault("with", {}).setdefault(a, {})[b] = (wr, g)
            pw.setdefault("with", {}).setdefault(b, {})[a] = (wr, g)
        for (a, b), (g, wa) in cell["against"].items():
            if g < pair_min:
                continue
            wr_a = round(100.0 * wa / g, 3)
            pw.setdefault("against", {}).setdefault(a, {})[b] = (wr_a, g)
            pw.setdefault("against", {}).setdefault(b, {})[a] = \
                (round(100.0 - wr_a, 3), g)
        cd = comp_data.setdefault(tier, {})
        for ck, (g, w) in cell["comp"].items():
            if g >= 5:
                cd[ck] = (round(100.0 * w / g, 3), g)
    return common.PatchStats(hero_wr, hero_meta, hero_map_wr, pairwise,
                             comp_data)
