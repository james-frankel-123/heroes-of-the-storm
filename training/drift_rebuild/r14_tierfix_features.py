"""
R14 — tier-label sensitivity check (2026-09-30 data-labeling bug).

The replay daemon stored league_tier one level too high and labeled Master
games (league_tier NULL) as "mid", so the pinned snapshot's research tiers
are low = Bronze and below, mid = Silver + Gold + Master, high = Platinum +
Diamond. This pass rebuilds the maintained recipe's feature cache
(cumulative statistics through each row's previous build, the
`cumulative_prev` convention of drift2026/build_drift_features.py) with
corrected tiers in the site's scheme:
    real = league_tier - 1 (NULL -> Master)
    low  = Bronze + Silver (real <= 2)
    mid  = Gold + Platinum (real 3-4)
    high = Diamond + Master (real 5, NULL)
Both the per-tier aggregate statistics and the tier one-hot use the
corrected tier. Row order matches features_cumulative_prev.npz.

Usage (6 cores): nice -n 19 taskset -c 48-63 python3 drift_rebuild/r14_tierfix_features.py
Output: drift_rebuild/feature_cache/features_tierfix_cumprev.npz
"""
import os
import sys
import time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_common as rb  # noqa: E402

from drift2026 import common  # noqa: E402
from drift2026.build_patch_stats import count_chunk, merge_cell, _new_cell  # noqa: E402
import numpy as np  # noqa: E402

NUM_WORKERS = 6
_STATS = {}


def fixed_tier(league_tier):
    if league_tier is None:
        return "high"            # Master (and above)
    real = int(league_tier) - 1
    if real <= 2:
        return "low"
    if real <= 4:
        return "mid"
    return "high"


def feat_chunk(args):
    from sweep_enriched_wp import extract_features, _swap_features, FEATURE_GROUPS
    rows, key = args
    st = _STATS[key]
    mask = [True] * len(FEATURE_GROUPS)
    out = [[] for _ in range(6)]
    for d in rows:
        b, e = extract_features(d, st, mask)
        y = float(d["winner"] == 0)
        bs, es = _swap_features(b, e)
        out[0] += [b, bs]
        out[1] += [e, es]
        out[2] += [y, 1.0 - y]
        out[3] += [d["build_idx"]] * 2
        out[4] += [d["date_days"]] * 2
        out[5] += [d["replay_id"]] * 2
    return (np.asarray(out[0], np.float32), np.asarray(out[1], np.float32),
            np.asarray(out[2], np.float32), np.asarray(out[3], np.int16),
            np.asarray(out[4], np.int32), np.asarray(out[5], np.int64))


def main():
    common.setup()
    t0 = time.time()
    out_path = os.path.join(rb.CACHE_DIR, "features_tierfix_cumprev.npz")
    rows, builds = common.load_data_with_patches()
    keys = ("team0_heroes", "team1_heroes", "game_map", "winner", "avg_mmr",
            "replay_id", "build_idx", "date_days")
    by_build, cnt = {}, []
    moved = 0
    for r in rows:
        t = fixed_tier(r.get("league_tier"))
        moved += t != r["skill_tier"]
        d = {k: r[k] for k in keys}
        d["skill_tier"] = t
        by_build.setdefault(r["build_idx"], []).append(d)
        cnt.append((r["build_idx"], t, r["game_map"], tuple(r["team0_heroes"]),
                    tuple(r["team1_heroes"]),
                    tuple(r["team0_bans"]) + tuple(r["team1_bans"]), r["winner"]))
    del rows
    print(f"{len(cnt):,} replays; {moved:,} change tier ({time.time()-t0:.0f}s)", flush=True)
    per_build = {}
    for i in range(0, len(cnt), 50000):
        for (bi, tier), cell in count_chunk(cnt[i:i + 50000]).items():
            dst = per_build.setdefault(bi, {}).get(tier)
            if dst is None:
                dst = per_build[bi][tier] = _new_cell()
            merge_cell(dst, cell)
    del cnt
    present = sorted(by_build)
    cum = {}
    for pos, bi in enumerate(present):
        # stats for build bi = cumulative through the previous present build
        _STATS[bi] = (rb.stats_from_counts_rounded(cum, pair_min=10) if pos
                      else common.empty_stats())
        for tier, cell in per_build[bi].items():
            if tier not in cum:
                cum[tier] = _new_cell()
            merge_cell(cum[tier], cell)
    print(f"stats built ({time.time()-t0:.0f}s)", flush=True)
    tasks = [(by_build[bi][i:i + 4000], bi) for bi in present
             for i in range(0, len(by_build[bi]), 4000)]
    parts = []
    with mp.get_context("fork").Pool(NUM_WORKERS) as pool:
        for i, res in enumerate(pool.imap(feat_chunk, tasks)):
            parts.append(res)
            if (i + 1) % 100 == 0:
                print(f"  chunk {i+1}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    a = [np.concatenate([p[k] for p in parts]) for k in range(6)]
    np.savez(out_path, bases=a[0], enricheds=a[1], labels=a[2], build_idx=a[3],
             date_days=a[4], replay_ids=a[5])
    print(f"wrote {out_path}: {len(a[0]):,} rows ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
