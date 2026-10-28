"""
P3 rich CF, data stage: per-game performance metrics from the scoreboard,
standardized, aligned to the extended slot table (x_slots_ext.npz).

Input: cache/export/<tag>/scoreboard.csv.gz (p3_export.py scoreboard; one row
per player-game from 2024-04-01, no battletags).

Per metric m and player-game:
  rate      value per minute of game length (time_spent_dead: share of the
            game; highest_kill_streak: raw)
  t         asinh(rate / s_m), s_m the median positive rate of m
  z         (t - mean) / sd within (hero, map, length bin, outcome) cells
            fit on in-sample (E) rows only; cells with < 50 E rows fall back
            to (hero, length bin, outcome). Outcome is in the cell, so z is
            performance relative to what players on that hero, map and game
            length do in a won (or lost) game: it carries no information
            about the game's own result. Then a (hero, calendar quarter)
            centering from E rows (later quarters use the last E quarter).
  NaN       metric not recorded, or not informative for the hero (fewer
            than 5% nonzero E rows on the hero).

Only the scored game's OWN row carries its metrics; the online state adds
rows on earlier days only (p3_rcf.py), so a game's metrics never enter its
own prediction.

Usage (from training/):
  nice -n 19 taskset -c 44-47 python3 personalization/p3_rcf_data.py build --tag rcf1016
  ... reliability       per-metric reliability per game (E rows)
Output: cache/rcf/sb_z.npz (Z float16 n_slots x M, metric names, has_sb).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C

RCF = os.path.join(C.CACHE, "rcf")
METRICS = ["hero_damage", "siege_damage", "structure_damage", "healing", "self_healing",
           "protection_allies", "damage_taken", "deaths", "kills", "assists",
           "experience_contribution", "merc_camp_captures",
           "outnumbered_deaths", "escapes", "regen_globes"]
LEN_EDGES = np.array([0, 14, 17, 20, 24], np.float64)   # minutes
MIN_CELL = 50


def sb_path():
    return os.path.join(RCF, "sb_z.npz")


def load_csv(tag):
    import pandas as pd
    p = os.path.join(C.CACHE, "export", tag, "scoreboard.csv.gz")
    cols = ["replay_id", "blizz_id", "game_map", "game_length"] + METRICS
    t0 = time.time()
    df = pd.read_csv(p, usecols=cols, dtype={m: np.float32 for m in METRICS})
    print(f"read {len(df):,} scoreboard rows ({time.time() - t0:.0f}s)", flush=True)
    return df


def build(tag):
    from p3_keys import key_blizz
    import p3_x_common as X
    t0 = time.time()
    df = load_csv(tag)
    z = np.load(X.EXT)
    rid = z["replay_id"].astype(np.int64)
    pid = z["pid"]
    bl = key_blizz(z["player_keys"])[pid].astype(np.int64)
    hero = z["hero"].astype(np.int64)
    y = z["y"]
    day = z["day"]
    ins = z["in_sample"] & (day >= C.day_of(C.E_START))
    n = len(rid)
    # join (replay_id, blizz_id) -> slot row
    comp_s = (rid << 40) | bl
    comp_c = (df["replay_id"].to_numpy(np.int64) << 40) | df["blizz_id"].to_numpy(np.int64)
    o = np.argsort(comp_c)
    cs = comp_c[o]
    j = np.minimum(np.searchsorted(cs, comp_s), len(cs) - 1)
    hit = cs[j] == comp_s
    src = o[j]
    print(f"slots {n:,}: with a scoreboard row {hit.sum():,} ({hit.mean():.3f})", flush=True)
    glen = df["game_length"].to_numpy(np.float64)[src] / 60.0
    gmap_codes, gmap = np.unique(df["game_map"].fillna("?").to_numpy(str), return_inverse=True)
    gmap = gmap[src]
    ok = hit & (glen > 3)
    lb = np.searchsorted(LEN_EDGES, glen, side="right") - 1
    win = (y > 0.5).astype(np.int64)
    H = len(z["hero_names"])
    nmap = len(gmap_codes)
    cell_a = ((hero * nmap + gmap) * len(LEN_EDGES) + lb) * 2 + win
    cell_b = (hero * len(LEN_EDGES) + lb) * 2 + win
    q = C.day_of  # noqa
    quarter = day.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) // 3
    q_last = int(quarter[ins].max())
    q0 = int(quarter.min())
    nq = int(quarter.max()) - q0 + 1
    Z = np.full((n, len(METRICS)), np.nan, np.float32)
    info = {}
    for k, m in enumerate(METRICS):
        v = df[m].to_numpy(np.float64)[src]
        miss = ~ok | np.isnan(v)
        v = np.where(np.isnan(v), 0.0, v)
        if m == "highest_kill_streak":
            rate = v
        else:
            rate = v / np.maximum(glen, 1.0)
        pos = rate[ins & ~miss & (rate > 0)]
        s = float(np.median(pos)) if len(pos) else 1.0
        t = np.arcsinh(rate / s)
        fit = ins & ~miss
        # cell moments
        za = np.zeros(n)
        cnt_a = np.bincount(cell_a[fit], minlength=H * nmap * len(LEN_EDGES) * 2)
        s1a = np.bincount(cell_a[fit], weights=t[fit], minlength=len(cnt_a))
        s2a = np.bincount(cell_a[fit], weights=t[fit] ** 2, minlength=len(cnt_a))
        cnt_b = np.bincount(cell_b[fit], minlength=H * len(LEN_EDGES) * 2)
        s1b = np.bincount(cell_b[fit], weights=t[fit], minlength=len(cnt_b))
        s2b = np.bincount(cell_b[fit], weights=t[fit] ** 2, minlength=len(cnt_b))
        mb = s1b / np.maximum(cnt_b, 1)
        sb = np.sqrt(np.maximum(s2b / np.maximum(cnt_b, 1) - mb ** 2, 0))
        ma = s1a / np.maximum(cnt_a, 1)
        sa = np.sqrt(np.maximum(s2a / np.maximum(cnt_a, 1) - ma ** 2, 0))
        use_a = cnt_a[cell_a] >= MIN_CELL
        mu = np.where(use_a, ma[cell_a], mb[cell_b])
        sd = np.where(use_a, sa[cell_a], sb[cell_b])
        za = (t - mu) / np.maximum(sd, 1e-3)
        # informative per hero: >= 5% nonzero E rows and a usable spread
        nz_h = np.bincount(hero[fit], weights=(rate[fit] > 0), minlength=H) / np.maximum(
            np.bincount(hero[fit], minlength=H), 1)
        sd_h = np.bincount(hero[fit], weights=sd[fit], minlength=H) / np.maximum(np.bincount(hero[fit], minlength=H), 1)
        inf_h = (nz_h >= 0.05) & (sd_h > 0.02)
        bad = miss | ~inf_h[hero] | (sd < 1e-3) | ((cnt_b[cell_b] < MIN_CELL) & ~use_a)
        za = np.clip(za, -5, 5)
        # (hero, quarter) centering from E rows; later quarters use the last E quarter
        good = fit & ~bad
        hq = hero * nq + (quarter - q0)
        cq = np.bincount(hq[good], minlength=H * nq)
        sq = np.bincount(hq[good], weights=za[good], minlength=H * nq)
        mq = (sq / np.maximum(cq, 1)).reshape(H, nq)
        qq = np.minimum(quarter, q_last) - q0
        za = za - mq[hero, qq]
        za[bad] = np.nan
        Z[:, k] = za
        info[m] = {"scale": s, "informative_heroes": int(inf_h.sum()),
                   "rows_with_value": int((~np.isnan(za)).sum())}
        print(f"  {m:26s} heroes {inf_h.sum():2d}  rows {(~np.isnan(za)).sum():,}  "
              f"E sd {np.nanstd(za[ins]):.3f}", flush=True)
    os.makedirs(RCF, exist_ok=True)
    np.savez(sb_path(), Z=Z.astype(np.float16), metrics=np.array(METRICS), has_sb=hit,
             replay_id=rid, info=np.array(json.dumps(info)))
    print(f"wrote {sb_path()} ({time.time() - t0:.0f}s)")


def reliability():
    """Per-metric reliability of a single game (E rows): the share of a
    game's variance that is a stable player x hero effect (split-half by
    odd and even games in play order), and the same for the player-level
    effect (pooled over heroes). The win residual r = y - WP is the
    reference row."""
    import p3_x_common as X
    z = np.load(X.EXT)
    sb = np.load(sb_path())
    Z = sb["Z"].astype(np.float32)
    day = z["day"]
    ins = z["in_sample"] & (day >= C.day_of(C.E_START))
    pid, hero = z["pid"].astype(np.int64), z["hero"].astype(np.int64)
    r = (z["y"] - z["wp"]).astype(np.float32)
    H = 90
    out = {}
    cols = [("win residual (y - WP)", r)] + [(m, Z[:, k]) for k, m in enumerate(sb["metrics"])]
    idx = np.flatnonzero(ins)
    key = pid[idx] * H + hero[idx]
    o = np.lexsort((z["replay_id"][idx], day[idx], key))
    idx, key = idx[o], key[o]
    brk = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    rank = np.arange(len(idx)) - np.repeat(brk, np.diff(np.r_[brk, len(idx)]))
    half = rank % 2
    for name, v in cols:
        x = v[idx]
        okx = ~np.isnan(x)
        res = {}
        for lvl, kk in (("player x hero", key), ("player", pid[idx])):
            u, inv = np.unique(kk[okx], return_inverse=True)
            h = half[okx]
            xs = x[okx].astype(np.float64)
            s0 = np.bincount(inv, weights=np.where(h == 0, xs, 0), minlength=len(u))
            s1 = np.bincount(inv, weights=np.where(h == 1, xs, 0), minlength=len(u))
            n0 = np.bincount(inv, weights=(h == 0), minlength=len(u))
            n1 = np.bincount(inv, weights=(h == 1), minlength=len(u))
            g = (n0 >= 1) & (n1 >= 1)
            m0, m1 = s0[g] / n0[g], s1[g] / n1[g]
            w = np.minimum(n0[g], n1[g])
            mu = np.average(np.r_[m0, m1], weights=np.r_[w, w])
            cov = np.average((m0 - mu) * (m1 - mu), weights=w)
            res[lvl] = float(cov / xs.var())
        out[name] = {"per_game_var": float(np.nanvar(x)), "reliability_player_hero": res["player x hero"],
                     "reliability_player": res["player"], "rows": int(okx.sum())}
        print(f"  {name:26s} rel(player x hero) {res['player x hero']:.4f}  rel(player) {res['player']:.4f}",
              flush=True)
    os.makedirs(os.path.join(C.RESULTS, "rcf"), exist_ok=True)
    with open(os.path.join(C.RESULTS, "rcf", "reliability.json"), "w") as f:
        json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["build", "reliability"])
    ap.add_argument("--tag", default="rcf1016")
    a = ap.parse_args()
    build(a.tag) if a.what == "build" else reliability()


if __name__ == "__main__":
    main()
