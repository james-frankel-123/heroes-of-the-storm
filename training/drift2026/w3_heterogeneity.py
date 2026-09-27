"""
W3(f): TIER + MAP HETEROGENEITY of drift and recovery.

Recomputes the D2a cross-build decay and the W3(a) within-build recovery
split by skill tier and by map:
  - by tier: hero WR / pick rate / comp WR cross-build r_adj at lags 1/3/8
    (signal_decay machinery on single-tier count cells) + games to 1pp
    hero-WR recovery within a build (from the W3(a) daily cache).
  - by map: hero-per-map WR cross-build r_adj at lags 1/3/8 (pooled tiers,
    >= MIN_MAP games per (map, hero) per build) + within-build recovery of
    map-local hero WR. Reference row = pooled (map-agnostic) hero WR.

Questions: does the meta drift faster at high tier? Are map-specific stats
more or less stable than pooled stats?

Inputs: patch_stats/patch_counts.pkl.gz, daily_hero_counts.pkl.gz.
Outputs: results/w3_heterogeneity.json + results/W3_HETEROGENEITY.md.
"""
import os
import sys
import gzip
import pickle
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common
from drift2026.signal_decay import (signals_for, weighted_corr, reliability,
                                    MIN_RELIABILITY)
from drift2026.w3_recovery import DAILY_CACHE

import numpy as np

MIN_MAP = 100      # min games per (map, hero) per build for map-WR decay
LAGS = (1, 3, 8)
TIERS = ["low", "mid", "high"]
REC_X = 1.0        # recovery threshold (pp) for hero WR
REC_MIN_FINAL = {"tier": 100, "map": 50}
REC_MIN_ROLL = 30


def decay_at_lags(vec_by_build, order):
    """vec_by_build: {bi: {key: (val, games)}}. Games-weighted r and r_adj
    averaged over all sizable pairs at each build-lag."""
    out = {}
    for lag in LAGS:
        rs, radjs = [], []
        for i in range(len(order) - lag):
            si = vec_by_build.get(order[i])
            sj = vec_by_build.get(order[i + lag])
            if not si or not sj:
                continue
            keys = sorted(set(si) & set(sj))
            if len(keys) < 8:
                continue
            x = np.array([si[k][0] for k in keys])
            y = np.array([sj[k][0] for k in keys])
            gx = np.array([si[k][1] for k in keys], dtype=np.float64)
            gy = np.array([sj[k][1] for k in keys], dtype=np.float64)
            w = np.minimum(gx, gy)
            r = weighted_corr(x, y, w)
            rs.append(r)
            ri, rj = reliability(x, gx), reliability(y, gy)
            if min(ri, rj) >= MIN_RELIABILITY:
                radjs.append(float(np.clip(r / np.sqrt(ri * rj), -1, 1)))
        out[lag] = {"r": round(float(np.mean(rs)), 3) if rs else None,
                    "r_adj": round(float(np.mean(radjs)), 3) if radjs else None,
                    "n_pairs": len(rs)}
    return out


def recovery_from_daily(per_build_daily, slice_kind, slice_val, min_final):
    """Games (in the slice) to hero-WR MAE <= REC_X vs the slice's final
    values, per build; day-boundary resolution, log-interpolated."""
    crossings = []
    for bi, daily in per_build_daily.items():
        days = sorted(daily)
        # final totals for the slice
        final = defaultdict(lambda: [0, 0])
        for d in days:
            src = (daily[d]["tier_hero"].get(slice_val, {})
                   if slice_kind == "tier"
                   else daily[d]["map_hero"].get(slice_val, {}))
            for h, (g, w) in src.items():
                final[h][0] += g
                final[h][1] += w
        keys = {h for h, (g, w) in final.items() if g >= min_final}
        if len(keys) < 8:
            continue
        roll = defaultdict(lambda: [0, 0])
        n_slice = 0.0
        curve = []
        for d in days:
            src = (daily[d]["tier_hero"].get(slice_val, {})
                   if slice_kind == "tier"
                   else daily[d]["map_hero"].get(slice_val, {}))
            for h, (g, w) in src.items():
                roll[h][0] += g
                roll[h][1] += w
                n_slice += g / 10.0
            diffs, wts = [], []
            for h in keys:
                gr, wr_ = roll[h]
                gf, wf = final[h]
                if gr < REC_MIN_ROLL:
                    continue
                diffs.append(abs(100.0 * wr_ / gr - 100.0 * wf / gf))
                wts.append(gf)
            if diffs and sum(wts):
                mae = float(np.sum(np.array(diffs) * np.array(wts))
                            / np.sum(wts))
                curve.append((n_slice, mae))
        # first crossing, log-interpolated
        x = None
        for i, (g, m) in enumerate(curve):
            if m <= REC_X:
                if i == 0:
                    x = g
                else:
                    g0, m0 = curve[i - 1]
                    f = (m0 - REC_X) / (m0 - m) if m0 != m else 1.0
                    x = float(np.exp(np.log(g0) + f * (np.log(g) - np.log(g0))))
                break
        crossings.append(x)
    reached = [c for c in crossings if c is not None]
    return {"median_games": int(np.median(reached)) if reached else None,
            "n_reached": len(reached), "n_builds": len(crossings)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()
    common.setup()

    with gzip.open(common.COUNTS_PKL, "rb") as f:
        data = pickle.load(f)
    builds, per_build = data["builds"], data["per_build"]

    sizable = [bi for bi in sorted(per_build)
               if sum(c["games"] for c in per_build[bi].values())
               >= common.SIZABLE_GAMES]

    with gzip.open(DAILY_CACHE, "rb") as f:
        cache = pickle.load(f)
    daily_sizable = {bi: cache["per_build"][bi] for bi in sizable}

    # ── by tier ──
    tier_rows = {}
    for tier in TIERS:
        sigs = {}
        games_t = []
        for bi in sizable:
            cell = per_build[bi].get(tier)
            if not cell:
                continue
            games_t.append(cell["games"])
            s = signals_for(cell)
            sigs[bi] = s
        row = {"games_per_build_median": int(np.median(games_t))}
        for sig in ("hero_wr", "hero_pickrate", "comp_wr", "pair_with_raw"):
            row[sig] = decay_at_lags(
                {bi: s[sig] for bi, s in sigs.items()}, sizable)
        row["recovery_1pp"] = recovery_from_daily(
            daily_sizable, "tier", tier, REC_MIN_FINAL["tier"])
        tier_rows[tier] = row

    # pooled reference (matches D2a)
    from drift2026.signal_decay import pool_tiers
    pooled_sigs = {bi: signals_for(pool_tiers(per_build[bi])) for bi in sizable}
    pooled_row = {sig: decay_at_lags({bi: s[sig] for bi, s in pooled_sigs.items()},
                                     sizable)
                  for sig in ("hero_wr", "hero_pickrate", "comp_wr")}

    # ── by map ──
    maps = sorted({m for bi in sizable for cell in per_build[bi].values()
                   for (m, h) in cell["hmap"]})
    map_rows = {}
    for m in maps:
        vecs = {}
        games_m = []
        for bi in sizable:
            agg = defaultdict(lambda: [0, 0])
            for cell in per_build[bi].values():
                for (mm, h), (g, w) in cell["hmap"].items():
                    if mm == m:
                        agg[h][0] += g
                        agg[h][1] += w
            vec = {h: (100.0 * w / g, g) for h, (g, w) in agg.items()
                   if g >= MIN_MAP}
            if vec:
                vecs[bi] = vec
                games_m.append(sum(g for _, g in vec.values()) / 10.0)
        map_rows[m] = {
            "games_per_build_median": int(np.median(games_m)) if games_m else 0,
            "map_hero_wr": decay_at_lags(vecs, sizable),
            "recovery_1pp": recovery_from_daily(
                daily_sizable, "map", m, REC_MIN_FINAL["map"]),
        }

    out = {"config": {"min_map_games": MIN_MAP, "lags": list(LAGS),
                      "recovery_threshold_pp": REC_X},
           "by_tier": tier_rows, "pooled_reference": pooled_row,
           "by_map": map_rows}
    common.write_json(os.path.join(common.RESULTS_DIR, "w3_heterogeneity.json"),
                      out)

    def lag_cells(d):
        return " / ".join(
            (f"{d[l]['r_adj']:.3f}" if d[l]["r_adj"] is not None
             else (f"({d[l]['r']:.3f})" if d[l]["r"] is not None else "-"))
            for l in LAGS)

    lines = [
        "# W3(f) — Drift and recovery heterogeneity by skill tier and map",
        "",
        "Cross-build stability (r_adj, disattenuated; raw r in parens where "
        "reliability is too low) at build-lags 1/3/8 over the 28 sizable "
        "builds, plus within-build games to 1pp hero-WR recovery (vs final, "
        "median across builds; day-boundary resolution).",
        "",
        "Reading note: recovery is measured in SLICE-local games against the "
        "slice's own final values, over the slice's own eligible key set — "
        "compare recovery numbers within a table, not against the pooled "
        "W3_RECOVERY row (fewer keys pass the games minimum in a slice, and "
        "a fixed game count is a larger fraction of a slice, which flatters "
        "the vs-final metric mechanically).",
        "",
        "## By skill tier",
        "",
        "| tier | games/build (med) | hero WR @1/3/8 | pick rate @1/3/8 | "
        "comp WR @1/3/8 | pair-with @1/3/8 | recovery to 1pp (games) |",
        "|---|---|---|---|---|---|---|",
    ]
    for tier in TIERS:
        r = tier_rows[tier]
        rec = r["recovery_1pp"]
        rec_s = (f"{rec['median_games']:,} ({rec['n_reached']}/{rec['n_builds']})"
                 if rec["median_games"] else f"never ({rec['n_reached']}/"
                 f"{rec['n_builds']})")
        lines.append(f"| {tier} | {r['games_per_build_median']:,} | "
                     f"{lag_cells(r['hero_wr'])} | "
                     f"{lag_cells(r['hero_pickrate'])} | "
                     f"{lag_cells(r['comp_wr'])} | "
                     f"{lag_cells(r['pair_with_raw'])} | {rec_s} |")
    lines.append(f"| (pooled) | - | {lag_cells(pooled_row['hero_wr'])} | "
                 f"{lag_cells(pooled_row['hero_pickrate'])} | "
                 f"{lag_cells(pooled_row['comp_wr'])} | - | - |")

    lines += ["", "## By map (hero-per-map WR)", "",
              "| map | slice games/build (med) | map-hero WR @1/3/8 | "
              "recovery to 1pp (games) |",
              "|---|---|---|---|"]
    for m in maps:
        r = map_rows[m]
        rec = r["recovery_1pp"]
        rec_s = (f"{rec['median_games']:,} ({rec['n_reached']}/{rec['n_builds']})"
                 if rec["median_games"] else
                 f"never ({rec['n_reached']}/{rec['n_builds']})")
        lines.append(f"| {m} | {r['games_per_build_median']:,} | "
                     f"{lag_cells(r['map_hero_wr'])} | {rec_s} |")
    lines.append(f"| (pooled hero WR ref) | - | "
                 f"{lag_cells(pooled_row['hero_wr'])} | see W3_RECOVERY |")

    md = os.path.join(common.RESULTS_DIR, "W3_HETEROGENEITY.md")
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
