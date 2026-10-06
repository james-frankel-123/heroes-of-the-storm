"""
P3 phase 2, item 0: is lagged MMR leaky?

player_mmr / hero_mmr are stamped when Heroes Profile parses a replay. If an
earlier game (by date) was uploaded AFTER the current game, its stamped MMR
may already include the current game's result. Replay ids roughly track
upload order, so for each slot and its lagged source game (latest game of
the same player at least G days earlier) we compare:

  src_after   source replay_id > current replay_id (uploaded later)
  src_before  source replay_id < current replay_id

within the same calendar gap between the two games. The statistic is the
slope and per-slot log-loss gain of r = y_team - wp_team on the lobby-
centered lagged MMR (fit within each stratum; slope reported per 100 MMR).
A leak shows up as a larger slope for src_after at the same gap. As a
positive control we also use the MMR stamped on the NEXT game of the same
player (after the current one), which certainly contains this result.

Usage (from training/): python3 personalization/p3_sd_mmrleak.py
Output: results/p3_sd_mmrleak.json
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from p3_heroes import HKEY
import p3_hs_core as C


def lag_source(d, group_key, lag, values):
    """Row index of the latest same-group row with non-NaN value and
    day <= day - lag (or -1). lag < 0 means the next row (any day >= today,
    later in time order)."""
    n = len(group_key)
    o = np.lexsort((d["replay_id"], d["day"], group_key))
    gk = group_key[o]
    ds = d["day"][o].astype(np.int64)
    vs = values[o]
    brk = np.flatnonzero(np.r_[True, gk[1:] != gk[:-1]])
    ends = np.r_[brk[1:], n]
    grp = np.repeat(np.arange(len(brk)), ends - brk)
    gstart = brk[grp]
    src = np.full(n, -1, np.int64)
    if lag < 0:
        nxt = np.arange(n) + 1
        ok = (nxt < n)
        ok[ok] &= grp[nxt[ok]] == grp[ok]
        src[ok] = nxt[ok]
    else:
        idx = np.where(~np.isnan(vs), np.arange(n), -1)
        runmax = np.maximum.accumulate(idx)
        runmax = np.where(runmax >= gstart, runmax, -1)
        comp = grp.astype(np.int64) * 100000 + ds
        pos = np.searchsorted(comp, comp - lag, side="right") - 1
        okp = (pos >= 0) & (pos >= gstart)
        j = np.where(okp, runmax[np.maximum(pos, 0)], -1)
        src = np.where(okp & (j >= gstart), j, -1)
    out = np.full(n, -1, np.int64)
    out[o] = np.where(src >= 0, o[np.maximum(src, 0)], -1)
    return out


def stratum(r, wp, x):
    x = x - x.mean()
    b = np.cov(r, x)[0, 1] / x.var()
    a = r.mean()
    pred = a + b * x
    y = r + wp
    p1 = np.clip(wp + pred, 1e-3, 1 - 1e-3)
    ll0 = -(y * np.log(wp) + (1 - y) * np.log(1 - wp))
    ll1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    se = np.sqrt(np.mean((r - pred) ** 2) / (len(x) * x.var()))
    return {"n": int(len(r)), "slope_per100": float(100 * b), "slope_se": float(100 * se),
            "corr": float(np.corrcoef(r, x)[0, 1]),
            "ll_gain_x1000": float(1000 * (ll0 - ll1).mean())}


def main():
    d = C.load_slots()
    n_games = int(d["g"].max()) + 1
    out = {}
    for col in ("player_mmr", "hero_mmr"):
        vals = d[col].astype(np.float64)
        key = d["pid"] if col == "player_mmr" else d["pid"] * HKEY + d["hero"]
        out[col] = {}
        for lag in (-1, 1, 7, 30):
            src = lag_source(d, key, lag, vals)
            has = src >= 0
            x = np.where(has, vals[np.maximum(src, 0)], np.nan)
            # lobby-center among slots with a value
            ok = ~np.isnan(x)
            lob_s = np.bincount(d["g"][ok], weights=x[ok], minlength=n_games)
            lob_n = np.bincount(d["g"][ok], minlength=n_games)
            xc = np.where(ok, x - lob_s[d["g"]] / np.maximum(lob_n[d["g"]], 1), np.nan)
            full = lob_n[d["g"]] == 10
            gap = np.where(has, d["day"] - d["day"][np.maximum(src, 0)], -999)
            after = has & (d["replay_id"][np.maximum(src, 0)] > d["replay_id"])
            ridgap = np.where(has, np.abs(d["replay_id"][np.maximum(src, 0)].astype(np.int64)
                                          - d["replay_id"]), -1)
            res = {"share_src_after": float(after[full].mean())}
            gaps = [(0, 0)] if lag < 0 else [(lag, lag), (lag + 1, lag + 2), (lag + 3, lag + 9)]
            if lag < 0:
                gaps = [(0, 10 ** 6)]
            for g0, g1 in gaps:
                gm = full & (gap >= g0) & (gap <= g1)
                for nm, sm in (("src_after", after), ("src_before", ~after)):
                    s = gm & sm
                    if s.sum() > 2000:
                        res[f"gap {g0}-{g1} {nm}"] = stratum(d["r"][s], d["wp"][s], xc[s])
                # replay-id distance tertiles among src_before
                s = gm & ~after
                if s.sum() > 6000:
                    q = np.percentile(ridgap[s], [33.3, 66.7])
                    for i, (lo, hi) in enumerate([(0, q[0]), (q[0], q[1]), (q[1], 1e18)]):
                        ss = s & (ridgap > lo) & (ridgap <= hi)
                        res[f"gap {g0}-{g1} src_before ridgap tertile {i + 1} "
                            f"(<= {hi:.0f})"] = stratum(d["r"][ss], d["wp"][ss], xc[ss])
            out[col][f"lag {lag}" if lag >= 0 else "next game (positive control)"] = res
            print(col, lag, json.dumps({k: (v if not isinstance(v, dict) else
                                            (v["n"], round(v["slope_per100"], 3),
                                             round(v["slope_se"], 3), round(v["ll_gain_x1000"], 3)))
                                        for k, v in res.items()}), flush=True)
    with open(os.path.join(C.RESULTS, "p3_sd_mmrleak.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
