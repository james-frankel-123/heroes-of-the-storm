"""
R1 — leak-free ("out-of-fold") version of the drift paper's cutoff-frozen
feature passes (audit A1).

The original `cutoff` / `cutoff_<build>` passes of
drift2026/build_drift_features.py give EVERY row, training rows included,
cumulative statistics through the cutoff build. Those statistics contain each
training row's own game, so the row's label leaks into its features.

Here each replay gets a deterministic hash fold (K=5). For rows at or before
the cutoff (the training period, including its validation slice), the
statistics are the cutoff-cumulative counts MINUS the counts of the row's own
fold: they contain no game from that fold, hence never the row's own outcome.
Rows after the cutoff (the deployment period) get the unmodified cutoff
statistics, exactly as before (deployment-time statistics stay frozen at the
cutoff; they can contain no deployment-period game by construction).

Row order, dtype and array names are identical to drift2026's feature caches
(verified against features_cutoff*.npz replay_ids), so
drift2026/train_drift_wp.py consumes the output unchanged.

Usage:
  python3 drift_rebuild/r1_oof_features.py --cutoff 2.55.14.95918 [--verify]
  python3 drift_rebuild/r1_oof_features.py --cutoff 2.55.9.93613
Output: drift_rebuild/feature_cache/features_oof_<cutoff>.npz
"""
import os
import sys
import time
import argparse
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_common as rb  # noqa: E402

from drift2026 import common  # noqa: E402
from drift2026.build_patch_stats import count_chunk, merge_cell, _new_cell  # noqa: E402

import numpy as np  # noqa: E402

NUM_WORKERS = 48
_STATS = {}   # filled before fork: "full" | ("oof", f) -> stats object


def feat_chunk(args):
    from sweep_enriched_wp import extract_features, _swap_features, FEATURE_GROUPS
    rows, oof = args
    all_mask = [True] * len(FEATURE_GROUPS)
    out = [[] for _ in range(6)]
    failed = 0
    for d in rows:
        st = _STATS[("oof", rb.fold_of(d["replay_id"]))] if oof else _STATS["full"]
        try:
            b, e = extract_features(d, st, all_mask)
        except Exception:
            failed += 1
            continue
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
            np.asarray(out[4], np.int32), np.asarray(out[5], np.int64), failed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--verify", action="store_true",
                    help="also check full-count stats reproduce the stats file")
    args = ap.parse_args()
    common.setup()
    out_path = os.path.join(rb.CACHE_DIR, f"features_oof_{args.cutoff}.npz")

    t0 = time.time()
    rows, builds = common.load_data_with_patches()
    cut_bidx = builds.index(args.cutoff)
    slim_keys = ("team0_heroes", "team1_heroes", "game_map", "skill_tier",
                 "winner", "avg_mmr", "replay_id", "build_idx", "date_days")
    by_build = {}
    count_rows = []
    for r in rows:
        by_build.setdefault(r["build_idx"], []).append({k: r[k] for k in slim_keys})
        if r["build_idx"] <= cut_bidx:
            # count_chunk keys cells by its first field: use the fold id
            count_rows.append((rb.fold_of(r["replay_id"]), r["skill_tier"],
                               r["game_map"], tuple(r["team0_heroes"]),
                               tuple(r["team1_heroes"]),
                               tuple(r["team0_bans"]) + tuple(r["team1_bans"]),
                               r["winner"]))
    del rows
    present = sorted(by_build)
    print(f"{sum(len(v) for v in by_build.values()):,} replays; "
          f"{len(count_rows):,} at or before cutoff {args.cutoff} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # per-(fold, tier) counts over the training period
    size = max(1, len(count_rows) // (NUM_WORKERS * 4))
    chunks = [count_rows[i:i + size] for i in range(0, len(count_rows), size)]
    per_fold = {}   # fold -> tier -> cell
    ctx = mp.get_context("fork")
    with ctx.Pool(NUM_WORKERS) as pool:
        for res in pool.imap_unordered(count_chunk, chunks):
            for (f, tier), cell in res.items():
                dst = per_fold.setdefault(f, {}).get(tier)
                if dst is None:
                    dst = per_fold[f][tier] = _new_cell()
                merge_cell(dst, cell)
    del count_rows, chunks
    print(f"counted {len(per_fold)} folds ({time.time()-t0:.0f}s)", flush=True)

    def merged(folds):
        out = {}
        for f in folds:
            for tier, cell in per_fold[f].items():
                dst = out.get(tier)
                if dst is None:
                    dst = out[tier] = _new_cell()
                merge_cell(dst, cell)
        return out

    _STATS["full"] = common.load_patch_stats("cumulative", args.cutoff)
    fold_games = {}
    for f in range(rb.K_FOLDS):
        _STATS[("oof", f)] = rb.stats_from_counts_rounded(
            merged([g for g in range(rb.K_FOLDS) if g != f]), pair_min=10)
        fold_games[f] = sum(c["games"] for c in per_fold[f].values())
    print(f"fold sizes (games): {fold_games}", flush=True)

    if args.verify:
        full_from_counts = rb.stats_from_counts_rounded(
            merged(range(rb.K_FOLDS)), pair_min=10)
        ref = _STATS["full"]
        for attr in ("hero_wr", "hero_meta", "hero_map_wr", "pairwise",
                     "comp_data"):
            a, b = getattr(full_from_counts, attr), getattr(ref, attr)

            def norm(x):
                if isinstance(x, dict):
                    return {k: norm(v) for k, v in x.items()}
                if isinstance(x, (list, tuple)):
                    return tuple(norm(v) for v in x)
                return x
            assert norm(a) == norm(b), f"stats mismatch in {attr}"
        print("VERIFY: full-count stats reproduce the cumulative stats file "
              "exactly", flush=True)

    tasks = []
    for bi in present:
        rs = by_build[bi]
        for i in range(0, len(rs), 4000):
            tasks.append((rs[i:i + 4000], bi <= cut_bidx))
    parts, failed = [], 0
    with ctx.Pool(NUM_WORKERS) as pool:
        for i, res in enumerate(pool.imap(feat_chunk, tasks)):
            parts.append(res[:6])
            failed += res[6]
            if (i + 1) % 100 == 0:
                print(f"  chunk {i+1}/{len(tasks)} ({time.time()-t0:.0f}s)",
                      flush=True)
    arrays = [np.concatenate([p[k] for p in parts if len(p[k])]) for k in range(6)]
    np.savez(out_path, bases=arrays[0], enricheds=arrays[1], labels=arrays[2],
             build_idx=arrays[3], date_days=arrays[4], replay_ids=arrays[5])
    print(f"wrote {out_path}: {len(arrays[0]):,} rows, {failed} failed, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
