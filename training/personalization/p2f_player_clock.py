"""
Paper 2 narrative facts, question 2 (player side): how well a player's
record on a hero predicts their record on the same hero 3, 6, 12 and 24
months later, restricted to heroes the player actually plays.

Two sources:
  history  full Heroes Profile Storm League histories of the 42,249 pulled
           players (store full_1017, 2019 on). Residual r = win - proxy, where
           proxy is the hero's win rate in that calendar quarter over every
           store row (shrunk to the hero's overall rate with 200 pseudo-games).
           Per-game noise variance = proxy (1 - proxy).
  slots    our corpus slots (x_slots_ext.npz, 2024-04-01 to 2026-09-27, every
           player). Residual r = y - WP (the draft- and lobby-aware win
           probability), noise variance wp (1 - wp).

Method:
  * Windows of W days (default 91), one start every 30 days. In each window,
    each (player, hero) cell has n games and mean residual x = sum r / n.
  * "Whole cell": x. "Hero-specific": x minus the player's mean residual
    over all their games in the same window (their own average).
  * For each pair of windows whose starts are h days apart, keep cells with
    at least k games in BOTH windows (k = 2: every hero played at all;
    k = 20 and 50: heroes the player plays a meaningful amount) and
    accumulate, after centering within the pair, the cross-window covariance
    and each window's split-half covariance.
  * Noise correction (disattenuation) by split halves: within each window a
    cell's games split into its earlier half and its later half, and the
    covariance of the two half means across cells estimates the variance of
    the stable part of one half (true variance of the full-window mean is
    about the same, since game noise is independent across halves). This
    needs no model of game noise. r_adj = cov_ab / sqrt(c_a c_b), with c the
    split-half covariance in each window. For hero-specific values the
    player's own average is taken over the same half.
    Contiguous halves, not alternating games: matchmaking makes a player's
    consecutive results anti-correlated (a win raises MMR and the next lobby
    is harder), which would bias an alternating split. The halves sit about
    W/2 days apart, so r_adj is the correlation at lag h relative to that at
    about W/2. A model-based version (noise = sum of p(1-p) / n^2) is also
    reported; it is fragile here because one game's noise is about 150 times
    the skill variance and the matchmaking anti-correlation makes real noise
    smaller than p(1-p) / n.
  * Game noise is independent across the two windows, so cov_ab needs no
    correction.
  * Uncertainty: delete-a-group jackknife over 20 player groups.

Usage (from training/): nice -n 19 taskset -c 48-51 python3 personalization/p2f_player_clock.py [--src history|slots|both]
Output: results/p2f/player_clock_<src>.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results", "p2f")
HIST = os.path.join(HERE, "cache", "full_history", "full_1017", "history.npz")
SLOTS = os.path.join(HERE, "cache", "x_slots_ext.npz")

HORIZONS = {"3 months": 91, "6 months": 182, "1 year": 365, "2 years": 730}
KS = (2, 20, 50)
NG = 20
DAY = 86400
PRIOR = 200.0


def load_history():
    z = np.load(HIST)
    keys, ptr = z["keys"], z["ptr"]
    kidx = np.repeat(np.arange(len(keys), dtype=np.int64), np.diff(ptr))
    hero = z["hero"].astype(np.int64)
    win = z["win"].astype(np.float64)
    day = (z["end_ts"] // DAY).astype(np.int64)
    ok = hero >= 0
    # proxy: hero x calendar-quarter win rate over every store row
    H = int(hero.max()) + 1
    q = day.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) // 3
    q0 = int(q.min())
    nq = int(q.max()) - q0 + 1
    hs = np.bincount(hero[ok], weights=win[ok], minlength=H)
    hc = np.bincount(hero[ok], minlength=H)
    base = (hs + 0.5 * PRIOR) / (hc + PRIOR)
    cell = hero * nq + (q - q0)
    s = np.bincount(cell[ok], weights=win[ok], minlength=H * nq).reshape(H, nq)
    c = np.bincount(cell[ok], minlength=H * nq).reshape(H, nq)
    tab = np.clip((s + PRIOR * base[:, None]) / (c + PRIOR), 0.2, 0.8)
    pulled = np.isin(keys, z["pulled_keys"])
    m = ok & pulled[kidx]
    p = tab[hero[m], (q - q0)[m]]
    return {"player": kidx[m], "hero": hero[m], "day": day[m], "r": win[m] - p, "v": p * (1 - p),
            "H": H, "n_players": int(pulled.sum())}


def load_slots():
    z = np.load(SLOTS)
    wp = z["wp"].astype(np.float64)
    return {"player": z["pid"].astype(np.int64), "hero": z["hero"].astype(np.int64),
            "day": z["day"].astype(np.int64), "r": z["y"].astype(np.float64) - wp, "v": wp * (1 - wp),
            "H": int(len(z["hero_names"])), "n_players": int(z["n_players"])}


def window_stats(d, lo, hi):
    """Per active (player, hero) cell in [lo, hi): cell id, n, S, V, the
    split-half sums (S1, n1, S2, n2) and the player's totals (all and per half)."""
    m = np.flatnonzero((d["day"] >= lo) & (d["day"] < hi))
    pl, he = d["player"][m], d["hero"][m]
    cid = pl * d["H"] + he
    o = np.lexsort((d["day"][m], cid))            # time order within cell
    m, cid, pl = m[o], cid[o], pl[o]
    u, start, inv = np.unique(cid, return_index=True, return_inverse=True)
    rank = np.arange(len(cid)) - start[inv]
    h2 = rank >= np.bincount(inv)[inv] / 2.0      # contiguous halves: earlier games vs later games
    r, v = d["r"][m], d["v"][m]
    n = np.bincount(inv).astype(np.float64)
    out = {"cid": u, "n": n, "S": np.bincount(inv, weights=r), "V": np.bincount(inv, weights=v),
           "n1": np.bincount(inv, weights=~h2).astype(np.float64), "S1": np.bincount(inv, weights=np.where(h2, 0, r)),
           "n2": np.bincount(inv, weights=h2).astype(np.float64), "S2": np.bincount(inv, weights=np.where(h2, r, 0))}
    pu, pinv = np.unique(pl, return_inverse=True)
    cp = np.searchsorted(pu, u // d["H"])
    for nm, w in (("N", None), ("R", r), ("Vp", v), ("N1", (~h2).astype(np.float64)), ("R1", np.where(h2, 0, r)),
                  ("N2", h2.astype(np.float64)), ("R2", np.where(h2, r, 0))):
        tot = np.bincount(pinv, weights=w).astype(np.float64)
        out[nm] = tot[cp]
    return out


def measures(w, idx):
    """(value, model noise, half-1 value, half-2 value) for whole cell and hero-specific."""
    g = {k: w[k][idx] for k in w if k != "cid"}
    x = g["S"] / g["n"]
    e = g["V"] / g["n"] ** 2
    x1, x2 = g["S1"] / np.maximum(g["n1"], 1), g["S2"] / np.maximum(g["n2"], 1)
    xs = x - g["R"] / g["N"]
    es = g["V"] / g["n"] ** 2 + g["Vp"] / g["N"] ** 2 - 2 * g["V"] / (g["n"] * g["N"])
    xs1 = x1 - g["R1"] / np.maximum(g["N1"], 1)
    xs2 = x2 - g["R2"] / np.maximum(g["N2"], 1)
    return {"cell": (x, e, x1, x2), "hero": (xs, es, xs1, xs2)}


def run(d, W, step=30):
    d0, d1 = int(d["day"].min()), int(d["day"].max()) + 1
    starts = np.arange(d0, d1 - W + 1, step)
    H = d["H"]
    cache = {}

    def ws(s):
        if s not in cache:
            cache[s] = window_stats(d, s, s + W)
        return cache[s]

    res = {}
    for hname, h in HORIZONS.items():
        # per (k, measure) per jackknife group: cov_ab, va, vb, model noise a, b, split-half cov a, b, cells
        acc = {(k, meas): np.zeros((NG, 8)) for k in KS for meas in ("cell", "hero")}
        pairs = 0
        for s in starts:
            if s + h + W > d1:
                break
            for old in [t for t in cache if t < s]:
                cache.pop(old)
            a, b = ws(s), ws(s + h)
            common, ia, ib = np.intersect1d(a["cid"], b["cid"], assume_unique=True, return_indices=True)
            if len(common) < 100:
                continue
            pairs += 1
            Ma, Mb = measures(a, ia), measures(b, ib)
            nmin = np.minimum(a["n"][ia], b["n"][ib])
            grp = ((common // H) * 2654435761 % 2 ** 32) % NG
            for k in KS:
                sel = nmin >= k
                if sel.sum() < 30:
                    continue
                g = grp[sel]
                for meas in ("cell", "hero"):
                    u, eu, u1, u2 = (z[sel] for z in Ma[meas])
                    v, ev, v1, v2 = (z[sel] for z in Mb[meas])
                    c = lambda z: z - z.mean()
                    cu, cv, cu1, cu2, cv1, cv2 = c(u), c(v), c(u1), c(u2), c(v1), c(v2)
                    A = acc[(k, meas)]
                    for j, wv in enumerate((cu * cv, cu * cu, cv * cv, eu, ev, cu1 * cu2, cv1 * cv2, np.ones(len(g)))):
                        A[:, j] += np.bincount(g, weights=wv, minlength=NG)
        out = {"window_pairs": pairs}
        for (k, meas), A in acc.items():
            if A[:, 7].sum() < 30:
                continue

            def stat(T):
                cov, va, vb, na, nb, sa, sb, c = T
                raw = cov / np.sqrt(va * vb)
                adj = cov / np.sqrt(sa * sb) if sa > 0 and sb > 0 else np.nan
                ta, tb = va - na, vb - nb
                adjm = cov / np.sqrt(ta * tb) if ta > 0 and tb > 0 else np.nan
                return raw, adj, adjm, np.sqrt(max(sa, 0) / c), np.sqrt(sa / va * sb / vb) if sa > 0 and sb > 0 else np.nan, c

            full = stat(A.sum(0))
            jk = np.array([stat(A.sum(0) - A[i])[:3] for i in range(NG)])
            se = np.sqrt((NG - 1) / NG * ((jk - np.nanmean(jk, 0)) ** 2).sum(0))
            out[f"{meas} k>={k}"] = {"r_raw": float(full[0]), "r_adj": float(full[1]), "se_adj": float(se[1]),
                                     "r_adj_noise_model": float(full[2]), "se_adj_noise_model": float(se[2]),
                                     "true_sd_half_pp": float(100 * full[3]), "split_half_rel_geo": float(full[4]),
                                     "cell_pairs": int(full[5])}
        res[hname] = out
        print(f"  W{W} {hname}: pairs {pairs}; " + "; ".join(
            f"{kk} {v['r_adj']:.2f}±{v['se_adj']:.2f} [model {v['r_adj_noise_model']:.2f}] (n {v['cell_pairs']:,})"
            for kk, v in out.items() if kk != "window_pairs"), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="both", choices=["history", "slots", "both"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"player_clock_{a.src}.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    out["method"] = __doc__.split("Method:")[1].split("Usage")[0].strip()
    t0 = time.time()
    for src in (["history", "slots"] if a.src == "both" else [a.src]):
        d = load_history() if src == "history" else load_slots()
        print(f"{src}: {len(d['r']):,} rows, {d['n_players']:,} players, days "
              f"{np.datetime64(int(d['day'].min()), 'D')} to {np.datetime64(int(d['day'].max()), 'D')} "
              f"({time.time() - t0:.0f}s)", flush=True)
        out[src] = {"rows": int(len(d["r"])), "players": d["n_players"],
                    "first_day": str(np.datetime64(int(d["day"].min()), "D")),
                    "last_day": str(np.datetime64(int(d["day"].max()), "D"))}
        for W in (91, 182):
            out[src][f"W{W}"] = run(d, W)
            json.dump(out, open(path, "w"), indent=1)
        del d
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
