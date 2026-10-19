"""
Paper 2 narrative facts, question 2 (game side): how well a hero's win rate
predicts its own win rate 3, 6, 12 and 24 months later.

Data: the pinned snapshot (training/snapshots/replay_snapshot_2026-05-22_1956753.json,
filtered to patch 2.55 like rerun2026.common), game dates from the drift2026
patch sidecar. Real league tiers: stored league_tier - 1, NULL = Master.

Method (the drift paper's D2a method on calendar windows instead of builds):
  * Tile time into windows of W days (default 91), one window start every 30 days.
  * For each pair of windows whose starts are h days apart, take the cells
    (hero, hero x tier group, or hero x map) that pass a filter in BOTH
    windows (all heroes with 30+ games; pick rate 1%+ and 100+ games; pick
    rate 2%+ and 300+ games; pick rate = share of the group's games with the
    hero), and and compute the games-weighted Pearson r of the two win rates
    (weight = min of the two game counts).
  * Disattenuate: each window's reliability = (observed weighted variance -
    weighted mean binomial variance p(1-p)/n) / observed variance;
    r_adj = r / sqrt(rel_a rel_b). Pairs with a reliability under 0.15 are dropped.
  * Report the mean r and r_adj over window pairs at each horizon.
  * W = 30 days is close to one balance build (the drift paper's unit);
    W = 91 is a quarter. With W = 182 the 3-month pairs overlap, so that
    cell is not a clean lag and is not reported.

Usage (from training/): nice -n 19 taskset -c 48-51 python3 personalization/p2f_game_clock.py
Output: results/p2f/game_clock.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
SNAP = os.path.join(TRAINING, "snapshots", "replay_snapshot_2026-05-22_1956753.json")
EXCL = os.path.join(TRAINING, "rerun2026", "pre255_exclude_ids.json")
SIDECAR = os.path.join(TRAINING, "drift2026", "patch_sidecar.npz")
CACHE = os.path.join(HERE, "cache", "p2f_snapshot_games.npz")
OUT = os.path.join(HERE, "results", "p2f")

HORIZONS = {"3 months": 91, "6 months": 182, "1 year": 365, "2 years": 730}
TIER_GROUPS = {"Bronze-Silver": (1, 2), "Gold-Platinum": (3, 4), "Diamond-Master": (5, 6)}
MIN_REL = 0.15


def extract():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=False)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    excl = set(json.load(open(EXCL)))
    rows = json.load(open(SNAP))
    print(f"loaded {len(rows):,} rows ({time.time() - t0:.0f}s)", flush=True)
    sc = np.load(SIDECAR)
    day_of = dict(zip(sc["replay_ids"].tolist(), sc["date_days"].tolist()))
    heroes = sorted({h for r in rows for h in r["team0_heroes"] + r["team1_heroes"]})
    maps = sorted({r["game_map"] for r in rows})
    hi = {h: i for i, h in enumerate(heroes)}
    mi = {m: i for i, m in enumerate(maps)}
    keep = [r for r in rows if r["replay_id"] not in excl and r["replay_id"] in day_of
            and len(r["team0_heroes"]) == 5 and len(r["team1_heroes"]) == 5]
    n = len(keep)
    out = {
        "replay_id": np.array([r["replay_id"] for r in keep], np.int64),
        "day": np.array([day_of[r["replay_id"]] for r in keep], np.int32),
        "map": np.array([mi[r["game_map"]] for r in keep], np.int16),
        "tier": np.array([6 if r.get("league_tier") is None else int(r["league_tier"]) - 1 for r in keep], np.int8),
        "winner": np.array([int(r["winner"]) for r in keep], np.int8),
        "heroes": np.array([[hi[h] for h in r["team0_heroes"] + r["team1_heroes"]] for r in keep], np.int16),
        "hero_names": np.array(heroes), "map_names": np.array(maps),
    }
    del rows, keep
    np.savez(CACHE, **out)
    print(f"cached {n:,} games ({time.time() - t0:.0f}s)", flush=True)
    return out


def wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx, vy = np.average((x - mx) ** 2, weights=w), np.average((y - my) ** 2, weights=w)
    return cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan


def rel(p, n):
    m = np.average(p, weights=n)
    vo = np.average((p - m) ** 2, weights=n)
    vn = np.average(p * (1 - p) / n, weights=n)
    return max(0.0, (vo - vn) / vo) if vo > 0 else 0.0


FILTERS = {"all heroes, 30+ games": (30, 0.0),
           "pick rate 1%+ and 100+ games": (100, 0.01),
           "pick rate 2%+ and 300+ games": (300, 0.02)}


def cell_counts(g, cell_of_slot, ncell, lo, hi_, game_group):
    """games and wins per cell, and games per group, in day window [lo, hi_)."""
    m = (g["day"] >= lo) & (g["day"] < hi_)
    c = cell_of_slot[m].ravel()
    w = g["slot_win"][m].ravel()
    ok = c >= 0
    gg = game_group[m]
    return (np.bincount(c[ok], minlength=ncell).astype(np.float64),
            np.bincount(c[ok], weights=w[ok], minlength=ncell),
            np.bincount(gg[gg >= 0], minlength=int(game_group.max()) + 1).astype(np.float64))


def clock(g, cell_of_slot, ncell, W, game_group, cell_group, step=30, filters=FILTERS):
    """Pick rate of a cell = its games / games of its group (all games, the
    tier group, or the map) in the window; a hero appears at most once per game."""
    d0, d1 = int(g["day"].min()), int(g["day"].max()) + 1
    starts = np.arange(d0, d1 - W + 1, step)
    cache = {}

    def cc(s):
        if s not in cache:
            cache[s] = cell_counts(g, cell_of_slot, ncell, s, s + W, game_group)
        return cache[s]

    res = {}
    for fname, (min_games, min_pick) in filters.items():
        res[fname] = {}
        for name, h in HORIZONS.items():
            rs, ra, ncells = [], [], []
            for s in starts:
                if s + h + W > d1:
                    break
                na, wa, ga = cc(s)
                nb, wb, gb = cc(s + h)
                pra = na / np.maximum(ga[cell_group], 1)
                prb = nb / np.maximum(gb[cell_group], 1)
                k = (na >= min_games) & (nb >= min_games) & (pra >= min_pick) & (prb >= min_pick)
                if k.sum() < 8:
                    continue
                pa, pb = wa[k] / na[k], wb[k] / nb[k]
                r = wcorr(pa, pb, np.minimum(na[k], nb[k]))
                ra_, rb_ = rel(pa, na[k]), rel(pb, nb[k])
                rs.append(r)
                ncells.append(int(k.sum()))
                if min(ra_, rb_) >= MIN_REL:
                    ra.append(float(np.clip(r / np.sqrt(ra_ * rb_), -1, 1)))
            res[fname][name] = {"window_pairs": len(rs), "r": float(np.mean(rs)) if rs else None,
                                "r_adj": float(np.mean(ra)) if ra else None,
                                "r_adj_sd_over_pairs": float(np.std(ra)) if len(ra) > 1 else None,
                                "pairs_with_rel": len(ra), "median_cells": float(np.median(ncells)) if ncells else None}
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    g = extract()
    H = len(g["hero_names"])
    # one row per game, 10 slots: hero and whether that slot's team won
    win0 = (g["winner"] == 0)
    g["slot_win"] = np.concatenate([np.repeat(win0[:, None], 5, 1), np.repeat(~win0[:, None], 5, 1)], 1).astype(np.float64)
    hero = g["heroes"].astype(np.int64)
    d0, d1 = int(g["day"].min()), int(g["day"].max())
    out = {"games": int(len(g["day"])), "first_day": str(np.datetime64(d0, "D")), "last_day": str(np.datetime64(d1, "D")),
           "horizons_days": HORIZONS, "method": __doc__.split("Method")[1].split("Usage")[0].strip()}
    t0 = time.time()
    allg = np.zeros(len(g["day"]), np.int64)
    zH = np.zeros(H, np.int64)
    for W in (30, 91, 182):
        key = f"W{W}"
        out[key] = {}
        out[key]["hero"] = clock(g, hero, H, W, allg, zH)
        # hero x tier group: correlate within each tier group
        tg = {}
        for tn, (a, b) in TIER_GROUPS.items():
            inn = (g["tier"] >= a) & (g["tier"] <= b)
            cell = np.where(inn[:, None], hero, -1)
            tg[tn] = clock(g, cell, H, W, np.where(inn, 0, -1), zH)
        out[key]["hero_x_tier_group"] = tg
        # hero x map (cells pooled across maps in one correlation; pick rate within the map)
        nm = len(g["map_names"])
        cell = g["map"].astype(np.int64)[:, None] * H + hero
        out[key]["hero_x_map"] = clock(g, cell, H * nm, W, g["map"].astype(np.int64), np.repeat(np.arange(nm), H))
        print(f"W{W} done ({time.time() - t0:.0f}s)", flush=True)
    # by region (region known for snapshot games from 2024-04-01 on, x_side_games.npz)
    sg = np.load(os.path.join(HERE, "cache", "x_side_games.npz"))
    o = np.argsort(sg["replay_ids"])
    j = np.minimum(np.searchsorted(sg["replay_ids"][o], g["replay_id"]), len(o) - 1)
    greg = np.where(sg["replay_ids"][o][j] == g["replay_id"], sg["region"][o][j], -1)
    out["W91_by_region"] = {"note": "games from 2024-04-01 on only (region known); 2 years not reachable"}
    for rv, rn in {1: "Americas", 2: "Europe", 3: "Korea"}.items():
        cell = np.where((greg == rv)[:, None], hero, -1)
        out["W91_by_region"][rn] = {"games": int((greg == rv).sum()),
                                    "hero": clock(g, cell, H, 91, np.where(greg == rv, 0, -1), zH)}
    cell = np.where((greg > 0)[:, None], hero, -1)
    out["W91_by_region"]["all regions, same period"] = {"games": int((greg > 0).sum()),
                                                        "hero": clock(g, cell, H, 91, np.where(greg > 0, 0, -1), zH)}
    json.dump(out, open(os.path.join(OUT, "game_clock.json"), "w"), indent=1)
    for rn, v in out["W91_by_region"].items():
        if rn != "note":
            for f, vv in v["hero"].items():
                print("region", rn, v["games"], f, {k: (round(x["r_adj"], 3) if x["r_adj"] is not None else None, x["window_pairs"])
                                                    for k, x in vv.items()})
    f2 = lambda v: round(v, 3) if v is not None else None
    for W in ("W30", "W91", "W182"):
        for lvl in ("hero", "hero_x_map"):
            for f, vv in out[W][lvl].items():
                print(W, lvl, f, {k: (f2(x["r"]), f2(x["r_adj"]), x["median_cells"]) for k, x in vv.items()})
        for tn, v in out[W]["hero_x_tier_group"].items():
            for f, vv in v.items():
                print(W, "tier", tn, f, {k: (f2(x["r"]), f2(x["r_adj"])) for k, x in vv.items()})

if __name__ == "__main__":
    main()
