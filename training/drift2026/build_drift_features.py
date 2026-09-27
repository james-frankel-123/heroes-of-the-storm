"""
PHASE D2 feature caches: full-draft WP features (base 197d + enriched 160d,
all 15 groups, team-swap augmented) for the WHOLE 2.55 corpus, under four
different sourcings of the aggregate statistics feeding the enriched groups:

  cutoff          every row uses cumulative stats through TRAIN_CUTOFF_BUILD.
                  -> D2(b) regime comparison: NOTHING after the cutoff touches
                  any feature, for train OR test rows.
  local           every row uses the per-patch stats of its OWN build
                  (deployment analogue: condition features on current-patch
                  aggregates; includes same-patch games by construction).
  cumulative_prev every row uses cumulative stats through the PREVIOUS build
                  (strictly causal: only data available before the row's patch
                  began; build 0 gets empty/default stats).
  frozen          paper-1 control: frozen_stats_2026-05-19 + HP
                  compositions.json for every row (end-of-time; leaks future
                  aggregates into past rows — that is the point of the arm).

Each pass writes feature_cache/features_<pass>.npz with row-aligned arrays:
  bases (N,197) f32, enricheds (N,160) f32, labels (N,) f32,
  build_idx (N,) i16, date_days (N,) i32, replay_ids (N,) i64
(2 rows per replay: original + team-swap, contiguous.)

Usage:
    python drift2026/build_drift_features.py            # all passes
    python drift2026/build_drift_features.py --only cutoff
"""
import os
import sys
import time
import argparse
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

import numpy as np

NUM_WORKERS = min(mp.cpu_count(), 56)
PASSES = ["cutoff", "local", "cumulative_prev", "frozen", "hybrid",
          # Q7: decayed-aggregate sourcing (build_decayed_stats.py), same
          # strictly-causal prev-build convention as cumulative_prev
          "decayed90_prev", "decayed365_prev", "decayed90k100_prev",
          # Q12: matched-clock per-signal decay (build_matched_clock_stats.py)
          "decayedmc_prev", "decayedmcrev_prev"]

_STATS_CACHE = {}   # per-worker: (kind, key) -> stats object


def _get_stats(kind, key):
    ck = (kind, key)
    st = _STATS_CACHE.get(ck)
    if st is None:
        if kind == "frozen":
            from sweep_enriched_wp import StatsCache
            st = StatsCache()   # frozen_stats_2026-05-19.json + compositions.json
        elif kind == "empty":
            st = common.empty_stats()
        else:   # 'per_patch' | 'cumulative'
            st = common.load_patch_stats(kind, key)
        _STATS_CACHE[ck] = st
        if len(_STATS_CACHE) > 6:   # workers see a rolling window of builds
            _STATS_CACHE.pop(next(iter(_STATS_CACHE)))
    return st


def feat_chunk(args):
    """(slim_rows, stats_kind, stats_key) -> aligned arrays (2 rows/replay)."""
    from sweep_enriched_wp import extract_features, _swap_features, FEATURE_GROUPS
    rows, kind, key = args
    stats = _get_stats(kind, key)
    all_mask = [True] * len(FEATURE_GROUPS)
    bases, enr, labels, bidx, days, rids = [], [], [], [], [], []
    failed = 0
    for d in rows:
        try:
            b, e = extract_features(d, stats, all_mask)
        except Exception:
            failed += 1
            continue
        y = float(d["winner"] == 0)
        bs, es = _swap_features(b, e)
        bases += [b, bs]
        enr += [e, es]
        labels += [y, 1.0 - y]
        bidx += [d["build_idx"]] * 2
        days += [d["date_days"]] * 2
        rids += [d["replay_id"]] * 2
    return (np.asarray(bases, dtype=np.float32),
            np.asarray(enr, dtype=np.float32),
            np.asarray(labels, dtype=np.float32),
            np.asarray(bidx, dtype=np.int16),
            np.asarray(days, dtype=np.int32),
            np.asarray(rids, dtype=np.int64),
            failed)


def out_path(pass_name):
    return os.path.join(common.CACHE_DIR, f"features_{pass_name}.npz")


def build_pass(pass_name, by_build, present, builds, force):
    path = out_path(pass_name)
    if os.path.exists(path) and not force:
        print(f"exists, skipping: {path}")
        return
    print(f"\n[{pass_name}] extracting features -> {path}")
    t0 = time.time()

    tasks = []
    for pos, bi in enumerate(present):
        rows = by_build[bi]
        if pass_name == "cutoff":
            kind, key = "cumulative", common.TRAIN_CUTOFF_BUILD
        elif pass_name.startswith("cutoff_"):
            # W2c policy (i): stats frozen at an arbitrary build for ALL rows
            kind, key = "cumulative", pass_name[len("cutoff_"):]
        elif pass_name == "local":
            kind, key = "per_patch", builds[bi]
        elif pass_name == "cumulative_prev":
            if pos == 0:
                kind, key = "empty", ""
            else:
                kind, key = "cumulative", builds[present[pos - 1]]
        elif pass_name.startswith("decayed") and pass_name.endswith("_prev"):
            # Q7: stats decayed through the PREVIOUS build (strictly causal,
            # exactly the cumulative_prev convention)
            if pos == 0:
                kind, key = "empty", ""
            else:
                kind, key = pass_name[:-len("_prev")], builds[present[pos - 1]]
        elif pass_name == "hybrid":
            # W3(d): per-signal sourcing (hero family from recent window,
            # comp/pair cumulative), all strictly before the row's build;
            # stats files written by w3_build_hybrid.py.
            if pos == 0:
                kind, key = "empty", ""
            else:
                kind, key = "hybrid", builds[bi]
        else:   # frozen
            kind, key = "frozen", ""
        for i in range(0, len(rows), 4000):
            tasks.append((rows[i:i + 4000], kind, key))

    parts = []
    failed = 0
    with mp.Pool(NUM_WORKERS) as pool:
        for i, res in enumerate(pool.imap(feat_chunk, tasks)):
            parts.append(res[:6])
            failed += res[6]
            if (i + 1) % 50 == 0:
                print(f"  {pass_name}: chunk {i+1}/{len(tasks)} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    arrays = [np.concatenate([p[k] for p in parts if len(p[k])]) for k in range(6)]
    np.savez(path, bases=arrays[0], enricheds=arrays[1], labels=arrays[2],
             build_idx=arrays[3], date_days=arrays[4], replay_ids=arrays[5])
    print(f"  {pass_name}: {len(arrays[0]):,} rows ({failed} replays failed) "
          f"in {(time.time()-t0)/60:.1f} min -> "
          f"{os.path.getsize(path)/1e9:.2f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="substring filter on pass name")
    ap.add_argument("--cutoff-at", default=None,
                    help="extra pass 'cutoff_<build>': cumulative stats frozen "
                         "at <build> for every row (W2c never-retrain arm)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    common.setup()

    all_passes = PASSES + ([f"cutoff_{args.cutoff_at}"] if args.cutoff_at else [])
    passes = [p for p in all_passes if not args.only or args.only in p]
    todo = [p for p in passes if args.force or not os.path.exists(out_path(p))]
    if not todo:
        print("all requested passes exist")
        return

    rows, builds = common.load_data_with_patches()
    slim_keys = ("team0_heroes", "team1_heroes", "game_map", "skill_tier",
                 "winner", "avg_mmr", "replay_id", "build_idx", "date_days")
    by_build = {}
    for r in rows:
        by_build.setdefault(r["build_idx"], []).append({k: r[k] for k in slim_keys})
    del rows
    present = sorted(by_build)
    print(f"{sum(len(v) for v in by_build.values()):,} replays, "
          f"{len(present)} builds; passes: {todo}")

    for p in todo:
        build_pass(p, by_build, present, builds, args.force)


if __name__ == "__main__":
    main()
