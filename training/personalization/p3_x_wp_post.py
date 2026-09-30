"""
P3 extensions, task 1: causal population WP for the post-snapshot games
(build 2.55.16.97039 after the snapshot, and 2.55.17.*).

Same model and convention as p3_wp_scores.py (paper 2's d2c_cumprev, three
seeds, swap-symmetrized), with features from cumulative statistics through
the PREVIOUS build:

  2.55.16.97039   drift2026 cumulative file through 2.55.16.96881 (exactly
                  what the snapshot scores used)
  2.55.17.97605   snapshot counts of builds 0..96881 + every DB game of 97039
  2.55.17.97650   ... + every DB game of 97605
  2.55.17.97771   ... + 97650
  2.55.17.98025   ... + 97771

No statistic includes a game of its own build, so no scored game is ever in
its own features. "Every DB game of the previous build" follows the paper-2
convention (complete previous build); a few of those games were uploaded
after the next build started.

Stats are derived from counts with the same thresholds and 3-decimal
rounding as drift2026/build_patch_stats.derive_stats_file, in memory (no
file under drift2026/ is written). Check: the 97039 games inside the
snapshot must reproduce cache/wp_drift.npz.

Usage (from training/): python3 personalization/p3_x_wp_post.py
Output: personalization/cache/x_wp_post.npz
  replay_ids, date_days, ts, version (index into versions), versions, y,
  wp0, wp0_seeds, in_snapshot
"""
import os
import sys
import gzip
import json
import pickle
import time
import multiprocessing as mp

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING)
from drift2026 import common

common.setup()
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GAMES = os.path.join(HERE, "cache", "x_post_games.json.gz")
OUT = os.path.join(HERE, "cache", "x_wp_post.npz")
SEEDS = [42, 123, 777]
BASE_BUILD = "2.55.16.96881"

_STATS = None


def stats_from_counts_rounded(cum):
    """PatchStats from {tier: cell}, identical to writing a cumulative file
    with derive_stats_file(pair_min=10) and loading it with load_patch_stats."""
    common._bind_statscache_methods()
    r3 = lambda x: round(x, 3)
    hero_wr, hero_meta, hero_map_wr, pairwise, comp_data = {}, {}, {}, {}, {}
    for tier, cell in cum.items():
        total = cell["games"]
        if total == 0:
            continue
        for h, (g, w) in cell["hero"].items():
            if g < 20:
                continue
            hero_wr.setdefault(tier, {})[h] = r3(100.0 * w / g)
            hero_meta.setdefault(tier, {})[h] = (r3(100.0 * g / total),
                                                 r3(100.0 * cell["bans"].get(h, 0) / total))
        for (m, h), (g, w) in cell["hmap"].items():
            if g >= 5:
                hero_map_wr.setdefault(tier, {}).setdefault(m, {})[h] = (r3(100.0 * w / g), g)
        for (a, b), (g, w) in cell["with"].items():
            if g < 10:
                continue
            wr = r3(100.0 * w / g)
            pw = pairwise.setdefault(tier, {}).setdefault("with", {})
            pw.setdefault(a, {})[b] = (wr, g)
            pw.setdefault(b, {})[a] = (wr, g)
        for (a, b), (g, wa) in cell["against"].items():
            if g < 10:
                continue
            wr_a = r3(100.0 * wa / g)
            pw = pairwise.setdefault(tier, {}).setdefault("against", {})
            pw.setdefault(a, {})[b] = (wr_a, g)
            pw.setdefault(b, {})[a] = (r3(100.0 - wr_a), g)
        for ck, (g, w) in cell["comp"].items():
            if g >= 5:
                comp_data.setdefault(tier, {})[ck] = (r3(100.0 * w / g), g)
    return common.PatchStats(hero_wr, hero_meta, hero_map_wr, pairwise, comp_data)


def to_cells(per_tier):
    from drift2026.build_patch_stats import _new_cell, merge_cell
    out = {}
    for tier, cell in per_tier.items():
        dst = out.setdefault(tier, _new_cell())
        src = {"games": cell["games"], "bans": cell["bans"]}
        for k in ("hero", "hmap", "with", "against", "comp"):
            src[k] = cell[k]
        merge_cell(dst, src)
    return out


def _feat(rows):
    from sweep_enriched_wp import extract_features, _swap_features, FEATURE_GROUPS
    mask = [True] * len(FEATURE_GROUPS)
    B, E, rid = [], [], []
    for d in rows:
        try:
            b, e = extract_features(d, _STATS, mask)
        except Exception:
            continue
        bs, es = _swap_features(b, e)
        B += [b, bs]
        E += [e, es]
        rid += [d["replay_id"]] * 2
    return (np.asarray(B, np.float32), np.asarray(E, np.float32), np.asarray(rid, np.int64))


def features(rows, stats):
    global _STATS
    _STATS = stats
    chunks = [rows[i:i + 2000] for i in range(0, len(rows), 2000)]
    with mp.get_context("fork").Pool(4) as pool:
        parts = pool.map(_feat, chunks)
    parts = [p for p in parts if len(p[0])]
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


def main():
    import torch
    torch.set_num_threads(4)
    from sweep_enriched_wp import WinProbEnrichedModel
    from drift2026.train_drift_wp import enriched_cols
    from drift2026.build_patch_stats import count_chunk, merge_cell, _new_cell

    t0 = time.time()
    games = json.load(gzip.open(GAMES, "rt"))
    games = [g for g in games if len(g["team0_heroes"]) == 5 and len(g["team1_heroes"]) == 5
             and g["winner"] in (0, 1)]
    versions = sorted({g["game_version"] for g in games}, key=common.build_sort_key)
    print(f"{len(games):,} games; versions {versions}", flush=True)
    by_v = {v: [g for g in games if g["game_version"] == v] for v in versions}

    # running cumulative counts through BASE_BUILD from the snapshot
    with gzip.open(common.COUNTS_PKL, "rb") as f:
        pc = pickle.load(f)
    builds = pc["builds"]
    base_i = builds.index(BASE_BUILD)
    cum = {}
    for bi, by_tier in pc["per_build"].items():
        if bi > base_i:
            continue
        for tier, cell in to_cells(by_tier).items():
            merge_cell(cum.setdefault(tier, _new_cell()), cell)

    models = []
    for s in SEEDS:
        ck = torch.load(os.path.join(common.MODELS_DIR, f"d2c_cumprev_s{s}.pt"),
                        map_location="cpu", weights_only=False)
        m = WinProbEnrichedModel(ck["input_dim"], ck["arch"], dropout=ck["dropout"])
        m.load_state_dict(ck["state_dict"])
        models.append(m.eval())
    ecols = enriched_cols()

    out = {k: [] for k in ("replay_ids", "wp0_seeds", "version")}
    prev_stats = common.load_patch_stats("cumulative", BASE_BUILD)
    # check: our rounded-counts object reproduces the file for BASE_BUILD
    mine = stats_from_counts_rounded(cum)
    assert mine.hero_wr == prev_stats.hero_wr, "hero_wr mismatch vs cumulative file"
    assert mine.comp_data == prev_stats.comp_data, "comp mismatch vs cumulative file"
    print("rounded-count stats reproduce the 96881 cumulative file", flush=True)
    stats = prev_stats
    for vi, v in enumerate(versions):
        rows = [{"team0_heroes": g["team0_heroes"], "team1_heroes": g["team1_heroes"],
                 "game_map": g["game_map"], "skill_tier": g["skill_tier"],
                 "winner": g["winner"], "avg_mmr": g["avg_mmr"],
                 "replay_id": g["replay_id"]} for g in by_v[v]]
        B, E, rid = features(rows, stats)
        X = torch.tensor(np.concatenate([B, E[:, ecols]], axis=1))
        ps = []
        with torch.no_grad():
            for m in models:
                p = np.concatenate([m(X[i:i + 131072]).numpy() for i in range(0, len(X), 131072)])
                ps.append((p[0::2] + 1.0 - p[1::2]) / 2.0)
        out["replay_ids"].append(rid[0::2])
        out["wp0_seeds"].append(np.stack(ps))
        out["version"].append(np.full(len(rid) // 2, vi, np.int16))
        print(f"{v}: {len(rows):,} games scored ({time.time() - t0:.0f}s)", flush=True)
        # add this build's games to the cumulative counts for the next build
        slim = [(0, g["skill_tier"], g["game_map"], tuple(g["team0_heroes"]),
                 tuple(g["team1_heroes"]), tuple(g["team0_bans"] or []) + tuple(g["team1_bans"] or []),
                 g["winner"]) for g in by_v[v]]
        for i in range(0, len(slim), 20000):
            for (_, tier), cell in count_chunk(slim[i:i + 20000]).items():
                merge_cell(cum.setdefault(tier, _new_cell()), cell)
        stats = stats_from_counts_rounded(cum)

    rid = np.concatenate(out["replay_ids"])
    seeds = np.concatenate(out["wp0_seeds"], axis=1)
    ver = np.concatenate(out["version"])
    gl = {g["replay_id"]: g for g in games}
    ts = np.array([gl[r]["ts"] for r in rid], np.int64)
    y = np.array([1.0 if gl[r]["winner"] == 0 else 0.0 for r in rid], np.float32)
    wp0 = seeds.mean(axis=0)
    w = np.load(os.path.join(HERE, "cache", "wp_drift.npz"))
    o = np.argsort(w["replay_ids"])
    wr = w["replay_ids"][o]
    j = np.searchsorted(wr, rid)
    ins = (j < len(wr)) & (wr[np.minimum(j, len(wr) - 1)] == rid)
    diff = np.abs(w["wp0"][o][j[ins]] - wp0[ins])
    print(f"check vs wp_drift on {ins.sum():,} snapshot games: max |dwp| {diff.max():.2e}, "
          f"mean {diff.mean():.2e}", flush=True)
    np.savez(OUT, replay_ids=rid, date_days=(ts // 86400).astype(np.int32), ts=ts,
             version=ver, versions=np.array(versions), y=y, wp0=wp0.astype(np.float32),
             wp0_seeds=seeds.astype(np.float32), in_snapshot=ins)
    oos = ~ins
    for vi, v in enumerate(versions):
        m = oos & (ver == vi)
        if m.any():
            print(f"  {v}: {m.sum():,} new games, acc {((wp0[m] > 0.5) == (y[m] == 1)).mean():.4f}, "
                  f"ll {-np.mean(y[m]*np.log(wp0[m]) + (1-y[m])*np.log(1-wp0[m])):.5f}")
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
