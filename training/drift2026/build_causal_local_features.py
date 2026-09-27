"""
W2a feature passes: WITHIN-PATCH-CAUSAL "local" enrichment stats.

Fixes the d2c_local self-leakage flagged in results/D2_SUMMARY.md: there,
per-patch stats contained the predicted games. Here, for EVERY row, the
enriched-feature stats are

    blend( rolling stats from games STRICTLY EARLIER (previous calendar days)
           in the row's own build,
           cumulative-to-cutoff prior stats )

with count-weighted shrinkage: for each statistic with g_local local games,
local winrate wr_l and prior winrate wr_p (prior weight kw),

    wr = (g_local * wr_l + kw * wr_p) / (g_local + kw)

Three variants of the prior weight kw:
    causal_merge   kw = prior's true game count (pure count merge — the
                   "refresh aggregates daily" deployment; local data gets its
                   natural weight, i.e. very little vs ~1.8M prior games)
    causal_k100    kw = 100  (local-tilted shrinkage)
    causal_k1000   kw = 1000 (conservative shrinkage)

Design notes:
  - Prior = cumulative-through-TRAIN_CUTOFF_BUILD for ALL rows (exactly the
    d2b "cutoff" sourcing), so train and test rows share one feature
    distribution and NOTHING after the cutoff enters any feature. Test rows
    are strictly deployable: prior <= cutoff + same-build earlier days only.
  - Rows on the first day of a build get pure prior stats (== features_cutoff).
  - Rows within a day use stats through the END OF THE PREVIOUS day (daily
    update cadence; also removes any intra-day ordering ambiguity and the
    row's own game from its features).
  - The `games` field stored for thresholded stats (pairwise/map/comp) is
    g_local + g_prior, so consumer reliability thresholds (>=30 pair /
    >=50 map) gate availability the same way the cumulative arms do; the
    variants differ in the VALUE, not the availability, of a statistic.

Outputs feature_cache/features_<pass>.npz, row-aligned with the wave-1 passes
(2 rows per replay: original + team-swap; within a build rows are date-sorted).

Usage:
    python drift2026/build_causal_local_features.py            # all 3 passes
    python drift2026/build_causal_local_features.py --only k100
"""
import os
import sys
import json
import gzip
import time
import argparse
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common
from drift2026.build_patch_stats import _new_cell, comp_key

import numpy as np

NUM_WORKERS = min(mp.cpu_count(), 48)

PASSES = {  # pass name -> prior weight mode ("merge" or pseudo-count k)
    "causal_merge": "merge",
    "causal_k100": 100.0,
    "causal_k1000": 1000.0,
}

_PRIOR = None   # per-worker cache


def load_prior_raw():
    """Cumulative-through-cutoff stats WITH game counts (load_patch_stats
    drops the counts we need for count-weighted blending)."""
    with gzip.open(common.stats_path("cumulative", common.TRAIN_CUTOFF_BUILD),
                   "rt") as f:
        raw = json.load(f)
    prior = {"total": raw["_meta"]["games"], "hero": {}, "meta": {},
             "hmap": {}, "pair": {}, "comp": {}}
    for r in raw["hero_stats"]:
        prior["hero"].setdefault(r["tier"], {})[r["hero"]] = (r["win_rate"], r["games"])
        prior["meta"].setdefault(r["tier"], {})[r["hero"]] = \
            (r["pick_rate"], r["ban_rate"])
    for r in raw["hero_map_stats"]:
        prior["hmap"].setdefault(r["tier"], {}).setdefault(r["map"], {})[r["hero"]] = \
            (r["win_rate"], r["games"])
    for r in raw["pairwise_stats"]:
        prior["pair"].setdefault(r["tier"], {}).setdefault(r["relationship"], {}) \
            .setdefault(r["hero_a"], {})[r["hero_b"]] = (r["win_rate"], r["games"])
    for r in raw["comp_stats"]:
        prior["comp"].setdefault(r["tier"], {})[r["roles"]] = (r["win_rate"], r["games"])
    return prior


def update_counts(cells, rows):
    """Add one day's rows to the rolling per-tier count cells
    (same counting logic as build_patch_stats.count_chunk)."""
    for d in rows:
        cell = cells.get(d["skill_tier"])
        if cell is None:
            cell = cells[d["skill_tier"]] = _new_cell()
        cell["games"] += 1
        gmap, winner = d["game_map"], d["winner"]
        for ti, team in ((0, d["team0_heroes"]), (1, d["team1_heroes"])):
            won = 1 if winner == ti else 0
            for h in team:
                e = cell["hero"][h]
                e[0] += 1
                e[1] += won
                e2 = cell["hmap"][(gmap, h)]
                e2[0] += 1
                e2[1] += won
            for i in range(len(team)):
                for j in range(i + 1, len(team)):
                    a, b = team[i], team[j]
                    if a > b:
                        a, b = b, a
                    e = cell["with"][(a, b)]
                    e[0] += 1
                    e[1] += won
            e = cell["comp"][comp_key(team)]
            e[0] += 1
            e[1] += won
        for h in set(d["bans"]):
            cell["bans"][h] += 1
        t0won = 1 if winner == 0 else 0
        for ha in d["team0_heroes"]:
            for hb in d["team1_heroes"]:
                if ha < hb:
                    e = cell["against"][(ha, hb)]
                    e[0] += 1
                    e[1] += t0won
                else:
                    e = cell["against"][(hb, ha)]
                    e[0] += 1
                    e[1] += 1 - t0won


def blended_stats(prior, cells, mode):
    """PatchStats = prior overridden by count-weighted blends on every key the
    rolling local counts have touched. mode: "merge" (kw = prior games) or a
    float pseudo-count kw."""
    hero_wr, hero_meta, hero_map_wr, pairwise, comp_data = {}, {}, {}, {}, {}
    for tier in set(prior["hero"]) | set(cells):
        ph = prior["hero"].get(tier, {})
        pm = prior["meta"].get(tier, {})
        pmap = prior["hmap"].get(tier, {})
        ppair = prior["pair"].get(tier, {})
        pcomp = prior["comp"].get(tier, {})
        total_p = prior["total"].get(tier, 0)

        hw = {h: wr for h, (wr, g) in ph.items()}
        hm = dict(pm)
        hmap_t = dict(pmap)                              # copy-on-write per map
        pw = {rel: dict(ad) for rel, ad in ppair.items()}  # copy-on-write per hero_a
        cd = dict(pcomp)

        cell = cells.get(tier)
        if cell and cell["games"]:
            total_l = cell["games"]
            for h, (g, w) in cell["hero"].items():
                wr_p, g_p = ph.get(h, (50.0, 0))
                kw = g_p if mode == "merge" else mode
                if (g_p or g >= 20) and g + kw > 0:
                    hw[h] = (g * (100.0 * w / g) + kw * wr_p) / (g + kw)
            kw_meta = total_p if mode == "merge" else mode
            for h in set(cell["hero"]) | set(cell["bans"]):
                g = cell["hero"].get(h, (0, 0))[0]
                pr_p, br_p = pm.get(h, (0.0, 0.0))
                if h not in pm and g < 20:
                    continue
                pr_l = 100.0 * g / total_l
                br_l = 100.0 * cell["bans"].get(h, 0) / total_l
                den = total_l + kw_meta
                hm[h] = ((total_l * pr_l + kw_meta * pr_p) / den,
                         (total_l * br_l + kw_meta * br_p) / den)
            touched_maps = {}
            for (m, h), (g, w) in cell["hmap"].items():
                wr_p, g_p = pmap.get(m, {}).get(h, (50.0, 0))
                if not (g_p or g >= 5):
                    continue
                kw = g_p if mode == "merge" else mode
                md = touched_maps.get(m)
                if md is None:
                    md = touched_maps[m] = dict(pmap.get(m, {}))
                md[h] = ((g * (100.0 * w / g) + kw * wr_p) / (g + kw), g + g_p)
            hmap_t.update(touched_maps)
            for rel, directional in (("with", False), ("against", True)):
                pp = ppair.get(rel, {})
                pr_out = pw.setdefault(rel, {})
                touched = {}
                for (a, b), (g, w) in cell[rel].items():
                    wr_p, g_p = pp.get(a, {}).get(b, (50.0, 0))
                    if not (g_p or g >= 5):
                        continue
                    kw = g_p if mode == "merge" else mode
                    wr_ab = (g * (100.0 * w / g) + kw * wr_p) / (g + kw)
                    ga = g + g_p
                    da = touched.get(a)
                    if da is None:
                        da = touched[a] = dict(pp.get(a, {}))
                    da[b] = (wr_ab, ga)
                    db = touched.get(b)
                    if db is None:
                        db = touched[b] = dict(pp.get(b, {}))
                    db[a] = ((100.0 - wr_ab) if directional else wr_ab, ga)
                pr_out.update(touched)
            for ck, (g, w) in cell["comp"].items():
                wr_p, g_p = pcomp.get(ck, (50.0, 0))
                if not (g_p or g >= 5):
                    continue
                kw = g_p if mode == "merge" else mode
                cd[ck] = ((g * (100.0 * w / g) + kw * wr_p) / (g + kw), g + g_p)

        hero_wr[tier] = hw
        hero_meta[tier] = hm
        hero_map_wr[tier] = hmap_t
        pairwise[tier] = pw
        comp_data[tier] = cd
    return common.PatchStats(hero_wr, hero_meta, hero_map_wr, pairwise, comp_data)


def run_build(args):
    """One (pass, build) task: walk the build's rows day by day; each day's
    rows get stats through the end of the previous day, then the day is added
    to the rolling counts. Rows must arrive date-sorted."""
    pass_name, mode, bidx, rows = args
    from sweep_enriched_wp import extract_features, _swap_features, FEATURE_GROUPS
    global _PRIOR
    if _PRIOR is None:
        _PRIOR = load_prior_raw()
    common._bind_statscache_methods()
    all_mask = [True] * len(FEATURE_GROUPS)

    cells = {}   # tier -> rolling count cell (games from earlier days only)
    bases, enr, labels, bidxs, days, rids = [], [], [], [], [], []
    failed, i, n = 0, 0, len(rows)
    while i < n:
        day = rows[i]["date_days"]
        j = i
        while j < n and rows[j]["date_days"] == day:
            j += 1
        stats = blended_stats(_PRIOR, cells, mode)
        for d in rows[i:j]:
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
            bidxs += [d["build_idx"]] * 2
            days += [d["date_days"]] * 2
            rids += [d["replay_id"]] * 2
        update_counts(cells, rows[i:j])
        i = j
    return (pass_name, bidx,
            np.asarray(bases, dtype=np.float32),
            np.asarray(enr, dtype=np.float32),
            np.asarray(labels, dtype=np.float32),
            np.asarray(bidxs, dtype=np.int16),
            np.asarray(days, dtype=np.int32),
            np.asarray(rids, dtype=np.int64),
            failed)


def out_path(pass_name):
    return os.path.join(common.CACHE_DIR, f"features_{pass_name}.npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="substring filter on pass name")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = ap.parse_args()
    common.setup()

    todo = [p for p in PASSES
            if (not args.only or args.only in p)
            and (args.force or not os.path.exists(out_path(p)))]
    if not todo:
        print("all requested passes exist")
        return

    rows, builds = common.load_data_with_patches()
    slim_keys = ("team0_heroes", "team1_heroes", "game_map", "skill_tier",
                 "winner", "avg_mmr", "replay_id", "build_idx", "date_days")
    by_build = {}
    for r in rows:
        s = {k: r[k] for k in slim_keys}
        s["bans"] = tuple(r["team0_bans"]) + tuple(r["team1_bans"])
        by_build.setdefault(r["build_idx"], []).append(s)
    del rows
    for bi in by_build:
        by_build[bi].sort(key=lambda d: d["date_days"])
    present = sorted(by_build)
    print(f"{sum(len(v) for v in by_build.values()):,} replays, "
          f"{len(present)} builds; passes: {todo}")

    # one task per (pass, build), largest builds first for balance
    tasks = [(p, PASSES[p], bi, by_build[bi]) for p in todo for bi in present]
    tasks.sort(key=lambda t: -len(t[3]))

    parts = {p: {} for p in todo}   # pass -> bidx -> arrays
    failed = {p: 0 for p in todo}
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_build, tasks, chunksize=1)):
            pname, bidx = res[0], res[1]
            parts[pname][bidx] = res[2:8]
            failed[pname] += res[8]
            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                print(f"  {i+1}/{len(tasks)} build-tasks done "
                      f"({time.time()-t0:.0f}s)", flush=True)

    for pname in todo:
        arrays = [np.concatenate([parts[pname][bi][k] for bi in present
                                  if len(parts[pname][bi][k])])
                  for k in range(6)]
        path = out_path(pname)
        np.savez(path, bases=arrays[0], enricheds=arrays[1], labels=arrays[2],
                 build_idx=arrays[3], date_days=arrays[4], replay_ids=arrays[5])
        print(f"  {pname}: {len(arrays[0]):,} rows ({failed[pname]} replays "
              f"failed) -> {os.path.getsize(path)/1e9:.2f} GB")
    print(f"All passes done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
