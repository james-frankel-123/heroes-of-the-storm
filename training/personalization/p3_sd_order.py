"""
P3 phase 2 check: does the same-day (session) result survive ordering games
by play time instead of replay_id (upload order)?

game_date behaves as the game's END time (consecutive games of one player
overlap in 0.2% of pairs under that reading, 7% if it were the start), so
start = game_date - game_length. Upload order reverses play order in 29.5%
of a player's consecutive same-day game pairs.

Orderings of a player's games within a day (days still come from the WP
cache; the state is only propagated between days):
  rid     replay_id order (what p3_sd_kalman / p3_sd_session used)
  time    game_date order, replay_id as tiebreak
  strict  time order, and an earlier same-day game is visible to game g
          only if it ended before g started AND has a lower replay_id

For each ordering: (a) refit the fast player component (and the fast hero
component) on the same random-40k and heavy-6k E samples as p3_sd_kalman,
other parameters fixed at the rid fit, using the predictive likelihood
under that visibility rule; (b) predict V1/V2 slots with same-day games
visible under the rule, static kernel vs drift model, combiner fit on V1
and scored on V2 (paired replay bootstrap). Lag-1-day predictions do not
depend on within-day order and are the common reference.

Usage (from training/): python3 personalization/p3_sd_order.py
Output: results/p3_sd_order.json
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from numba import njit, prange
import p3_hs_core as C
import p3_sd_kalman as K
from p3_hs_fit import prepare
from p3_hs_eval import team_diff, slot_stats

GT = os.path.join(C.CACHE, "gametime_2024q2.npz")


@njit(parallel=True, cache=True)
def kalman_vis(starts, ends, tval, day, hero, y, v, ts_end, ts_start, rid, mode,
               lidx, lw, s2, lam, out_m, out_s2):
    """Rows of each player sorted by (day, within-day order). Prediction for
    row r uses the day-start state plus same-day rows q before r in order
    that are visible: mode 0 -> all earlier rows; mode 1 -> ended before r
    started and lower replay_id; mode 2 -> none (lag 1 day). After the day
    the state is updated with every row of the day in order."""
    d = s2.shape[0]
    L = lidx.shape[1]
    for p in prange(starts.shape[0]):
        a, b = starts[p], ends[p]
        x = np.zeros(d)
        P = np.zeros((d, d))
        for i in range(d):
            P[i, i] = s2[i]
        f = np.empty(d)
        pz = np.empty(d)
        xs = np.empty(d)
        Ps = np.empty((d, d))
        t_prev = tval[a]
        i = a
        while i < b:
            j = i
            while j < b and day[j] == day[i]:
                j += 1
            dt = tval[i] - t_prev
            if dt > 0:
                for k in range(d):
                    f[k] = np.exp(-lam[k] * dt)
                for k in range(d):
                    x[k] *= f[k]
                    for l in range(d):
                        P[k, l] *= f[k] * f[l]
                    P[k, k] += s2[k] * (1.0 - f[k] * f[k])
                t_prev = tval[i]
            for r in range(i, j):
                xs[:] = x
                Ps[:, :] = P
                if mode != 2:
                    for q in range(i, r):
                        if mode == 1 and not (ts_end[q] <= ts_start[r] and rid[q] < rid[r]):
                            continue
                        h = hero[q]
                        for k in range(d):
                            acc = 0.0
                            for q1 in range(L):
                                acc += Ps[k, lidx[h, q1]] * lw[h, q1]
                            pz[k] = acc
                        m = 0.0
                        S = v[q]
                        for q1 in range(L):
                            m += lw[h, q1] * xs[lidx[h, q1]]
                            S += lw[h, q1] * pz[lidx[h, q1]]
                        e = y[q] - m
                        for k in range(d):
                            xs[k] += pz[k] * e / S
                        for k in range(d):
                            c = pz[k] / S
                            for l in range(d):
                                Ps[k, l] -= c * pz[l]
                h = hero[r]
                m = 0.0
                s = 0.0
                for q1 in range(L):
                    m += lw[h, q1] * xs[lidx[h, q1]]
                    for q2 in range(L):
                        s += lw[h, q1] * lw[h, q2] * Ps[lidx[h, q1], lidx[h, q2]]
                out_m[r] = m
                out_s2[r] = s
            # end-of-day update with all rows in order
            for r in range(i, j):
                h = hero[r]
                for k in range(d):
                    acc = 0.0
                    for q1 in range(L):
                        acc += P[k, lidx[h, q1]] * lw[h, q1]
                    pz[k] = acc
                m = 0.0
                S = v[r]
                for q1 in range(L):
                    m += lw[h, q1] * x[lidx[h, q1]]
                    S += lw[h, q1] * pz[lidx[h, q1]]
                e = y[r] - m
                for k in range(d):
                    x[k] += pz[k] * e / S
                for k in range(d):
                    c = pz[k] / S
                    for l in range(d):
                        P[k, l] -= c * pz[l]
            i = j


class VisFilter:
    def __init__(self, d, meta, V, r_adj, ts_end, ts_start, order, rows_mask=None):
        sel = np.ones(len(d["pid"]), bool) if rows_mask is None else rows_mask
        idx = np.flatnonzero(sel)
        second = d["replay_id"][idx] if order == "rid" else ts_end[idx]
        o = np.lexsort((d["replay_id"][idx], second, d["day"][idx], d["pid"][idx]))
        self.rows = idx[o]
        pid = d["pid"][self.rows]
        brk = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
        self.starts = brk.astype(np.int64)
        self.ends = np.r_[brk[1:], len(pid)].astype(np.int64)
        self.day = d["day"][self.rows].astype(np.int64)
        self.tval = self.day.astype(np.float64)
        self.hero = d["hero"][self.rows].astype(np.int64)
        self.y = r_adj[self.rows].astype(np.float64)
        self.v = d["v"][self.rows].astype(np.float64)
        self.te = ts_end[self.rows].astype(np.int64)
        self.tsb = ts_start[self.rows].astype(np.int64)
        self.rid = d["replay_id"][self.rows].astype(np.int64)
        self.lidx, self.lw = K.loadings(meta, V)
        self.blizz = meta["blizz"]
        self.lam_state = K.Filter.lam_state.__get__(self)

    def run(self, p, mode):
        sd, lam, br = K.split(p)
        s2 = np.array([sd.get(K.COMPS[c], 0.0) ** 2 for c in K.state_comp()])
        n = len(self.rows)
        m = np.zeros(n)
        s = np.zeros(n)
        kalman_vis(self.starts, self.ends, self.tval, self.day, self.hero, self.y, self.v,
                   self.te, self.tsb, self.rid, mode, self.lidx, self.lw, s2,
                   self.lam_state(lam, br), m, s)
        return m, s

    def ll(self, p, mode):
        m, s = self.run(p, mode)
        S = s + self.v
        return float((-0.5 * (np.log(2 * np.pi * S) + (self.y - m) ** 2 / S)).sum())


VARIANTS = {"rid": ("rid", 0), "time": ("time", 0), "strict": ("time", 1)}


def main():
    t0 = time.time()
    d = C.load_slots()
    meta = C.hero_meta(d["hero_names"])
    kz = np.load(K.KOUT)
    e_mask, n_p, n_ph, table, r_adj = prepare(d)
    V = kz["V_cf2"]
    z = np.load(GT)
    o = np.argsort(z["replay_ids"])
    zr = z["replay_ids"][o]
    gi = np.searchsorted(zr, d["replay_id"])
    assert np.all(zr[gi] == d["replay_id"])
    ts_end = z["ts"][o][gi]
    ts_start = ts_end - z["game_length"][o][gi]
    with open(os.path.join(C.RESULTS, "p3_sd_fit.json")) as f:
        fits = json.load(f)
    rng = np.random.RandomState(11)
    fitp = kz["fit_players"]
    nE = np.bincount(d["pid"][e_mask], minlength=int(d["n_players"]))
    samp = np.zeros(int(d["n_players"]), bool)
    samp[rng.choice(np.flatnonzero(fitp & (nE > 0)), 40000, replace=False)] = True
    heavy = np.zeros(int(d["n_players"]), bool)
    cand = np.flatnonzero(fitp & (nE >= 300))
    heavy[rng.choice(cand, min(6000, len(cand)), replace=False)] = True
    out = {"fits": {}, "game": {}, "slot": {}}
    # share of visible earlier same-day games under the strict rule
    B = {"sd_player_f": (1e-4, 0.15), "lam_player_f": (0.02, 5.0),
         "sd_hero_f": (1e-4, 0.15), "lam_hero_f": (2e-4, 0.5)}
    refit = {}
    for sname, smask, key in (("random 40k", samp, "random 40k"),
                              ("heavy 6k (>=300 games)", heavy, "heavy 6k (>=300 games)")):
        base = fits[key]["fits"]["drift days"]["params"]
        static = fits[key]["fits"]["static (phase-1 kernel)"]["params"]
        rows = e_mask & smask[d["pid"]]
        for vname, (order, mode) in VARIANTS.items():
            vf = VisFilter(d, meta, V, r_adj, ts_end, ts_start, order, rows)
            ll_static = vf.ll(static, mode)
            ll_base = vf.ll(base, mode)
            p, ll = K.coord_fit(lambda q: vf.ll(q, mode), base,
                                ["sd_player_f", "lam_player_f", "sd_hero_f", "lam_hero_f"], B,
                                sweeps=1, label=f"{sname} {vname}")
            p_nof = dict(p)
            p_nof["sd_player_f"] = 0.0
            ll_nof = vf.ll(p_nof, mode)
            lamf = p["lam_player_f"]
            out["fits"][f"{sname} | {vname}"] = {
                "params": p, "ll": ll, "ll_rid_params": ll_base, "ll_static": ll_static,
                "ll_without_fast_player": ll_nof,
                "fast_player_sd_pp": 100 * p["sd_player_f"],
                "fast_player_halflife_days": float(np.log(2) / lamf),
                "fast_player_halflife_hours": float(24 * np.log(2) / lamf),
                "ll_gain_fast_player": ll - ll_nof}
            if sname == "random 40k":
                refit[vname] = p
            print(f"[{sname} | {vname}] fast player sd {100 * p['sd_player_f']:.2f}pp, half-life "
                  f"{24 * np.log(2) / lamf:.1f}h, ll gain from fast part {ll - ll_nof:+.1f}; "
                  f"fast hero sd {100 * p['sd_hero_f']:.2f}pp ({time.time() - t0:.0f}s)",
                  flush=True)

    # ---------- prediction on V1/V2 with same-day visibility
    mu = table[C.exp_bins(n_p, n_ph)]
    days = d["day"]
    post = ~d["in_sample"]
    _, first_row = np.unique(d["g"][post], return_index=True)
    med = np.median(days[post][first_row])
    v1r, v2r = post & (days < med), post & (days >= med)
    static = fits["random 40k"]["fits"]["static (phase-1 kernel)"]["params"]
    preds = {}
    for vname, (order, mode) in VARIANTS.items():
        vf = VisFilter(d, meta, V, r_adj, ts_end, ts_start, order)
        for mname, p in (("static", static), ("drift", refit[vname])):
            m, _ = vf.run(p, mode)
            pm = np.empty(len(d["pid"]))
            pm[vf.rows] = m
            preds[f"{mname} | {vname}"] = mu + pm
        if vname == "rid":
            m, _ = vf.run(static, 2)
            pm = np.empty(len(d["pid"]))
            pm[vf.rows] = m
            preds["static | lag 1 day"] = mu + pm
        print(f"predictions {vname}: {time.time() - t0:.0f}s", flush=True)
    n_games = int(d["g"].max()) + 1
    yg = np.zeros(n_games)
    wg = np.full(n_games, 0.5)
    t0m = d["team"] == 0
    yg[d["g"][t0m]] = d["y"][t0m]
    wg[d["g"][t0m]] = d["wp"][t0m]
    lo = np.log(wg / (1 - wg))
    full = np.bincount(d["g"][post], minlength=n_games) == 10
    gv1 = np.zeros(n_games, bool)
    gv1[d["g"][v1r]] = True
    gv2 = np.zeros(n_games, bool)
    gv2[d["g"][v2r]] = True
    fit_m, test_m = gv1 & full, gv2 & full
    rng = np.random.RandomState(0)
    idx = np.flatnonzero(test_m)
    boots = [rng.choice(idx, len(idx)) for _ in range(200)]
    X0 = lo[:, None]
    p0 = C.predict(C.fit_logistic(X0[fit_m], yg[fit_m]), X0)
    ll0 = -(yg * np.log(p0) + (1 - yg) * np.log(1 - p0))
    lls = {}
    for nm, pr in preds.items():
        X = np.column_stack([lo, team_diff(pr, d["g"], d["team"], n_games)])
        p = C.predict(C.fit_logistic(X[fit_m], yg[fit_m]), X)
        lls[nm] = -(yg * np.log(p) + (1 - yg) * np.log(1 - p))
        out["game"][nm] = C.game_metrics(p[test_m], yg[test_m])
        out["game"][nm]["gain_vs_M0"] = float((ll0 - lls[nm])[idx].mean())
        out["slot"][nm] = slot_stats(d["r"][v2r], d["wp"][v2r], pr[v2r])

    def paired(a, b):
        dd = lls[b] - lls[a]
        bs = [dd[bb].mean() for bb in boots]
        return [float(dd[idx].mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    out["paired"] = {}
    for vname in VARIANTS:
        out["paired"][f"drift vs static | {vname}"] = paired(f"drift | {vname}", f"static | {vname}")
        out["paired"][f"static same-day vs lag 1 day | {vname}"] = paired(f"static | {vname}",
                                                                          "static | lag 1 day")
        out["paired"][f"drift same-day vs lag 1 day | {vname}"] = paired(f"drift | {vname}",
                                                                         "static | lag 1 day")
    for k, v in out["paired"].items():
        print(f"  {k:45s} {v[0]:+.5f} [{v[1]:+.5f}, {v[2]:+.5f}]", flush=True)
    for nm, g in out["game"].items():
        print(f"  {nm:22s} acc {g['acc']:.4f} ll {g['logloss']:.5f} gain vs M0 {g['gain_vs_M0']:+.5f}",
              flush=True)
    with open(os.path.join(C.RESULTS, "p3_sd_order.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
