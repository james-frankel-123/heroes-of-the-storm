"""
P3 phase 2: head-to-head of drift models, momentum/volume arms,
uncertainty in the combiner, and upload-order-safe lagged MMR.

Windows as in phase 1 (E estimation, V1 fit, V2 test; 70,974 / 72,474
games). All per-slot features use games from days < game day only.

Skill models (per-slot posterior mean m and variance s2):
  EB static kernel online     phase-1 winner (+CF rank 2, no decay); equals
                              the Kalman filter with zero drift
  EB decay hl=365             phase-1 exponential down-weighting
  SS days                     state-space, calendar clock, tied rates
  SS days, hero rate by role  hero-specific rate split by Blizzard role
  SS games                    state-space, player's game-count clock
Momentum / volume arms (proxies; Max's exact formula pending):
  vol365 (all / hero)         games in the previous 365 days, log1p
  EWMA WR hl = 10/30/100 games over the player's games (y - 0.5)
  EWMA residual hl = 10/30/100 games (surprise-weighted: y - wp)
  EWMA hero WR hl = 10 games on the hero
  form10 / form20             mean residual over the last 10 / 20 games
Uncertainty (item 3): posterior sd sum, estimate x reliability, lower
bound m - 1.28 s, and calibration within strata of total uncertainty.
Lagged MMR, upload-order-safe: latest value from a game >= G days earlier
whose replay_id is below the current replay_id minus B (B = ids uploaded in
about 2 days), versus the naive lagged value.

Usage (from training/): python3 personalization/p3_sd_eval.py
Output: results/p3_sd_eval.json
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
from p3_hs_fit import prepare, KERNELS
from p3_hs_eval import nbin, N_LABELS, team_diff, slot_stats

DSL_EDGES = [-1, 0, 8, 31, 91, 181, 366, 10 ** 6]
DSL_LABELS = ["never", "1-7d", "8-30d", "31-90d", "91-180d", "181-365d", "366d+"]
Z80 = 1.2815516


def dsl_bin(x):
    # x = -1 for never played
    return np.where(x < 0, 0, np.searchsorted(DSL_EDGES[1:], x, side="right"))


@njit(parallel=True, cache=True)
def _momentum(starts, ends, day, hero, y, r, feat):
    """feat columns: 0-2 EWMA WR hl 10/30/100 games; 3-5 EWMA resid;
    6 hero EWMA WR hl 10; 7 form10; 8 form20; 9 days since hero last played
    (-1 never); 10 days since player's last game (-1 never)."""
    hls = np.array([10.0, 30.0, 100.0])
    al = 1.0 - np.exp(-np.log(2.0) / hls)
    ah = 1.0 - np.exp(-np.log(2.0) / 10.0)
    for p in prange(starts.shape[0]):
        a, b = starts[p], ends[p]
        ew = np.zeros(3)
        er = np.zeros(3)
        wsum = np.zeros(3)
        hew = np.zeros(90)
        hw = np.zeros(90)
        last_h = np.full(90, -1)
        buf = np.zeros(20)
        nb = 0
        last_any = -1
        i = a
        while i < b:
            j = i
            while j < b and day[j] == day[i]:
                j += 1
            for q in range(i, j):
                for k in range(3):
                    feat[q, k] = ew[k] / wsum[k] if wsum[k] > 0 else 0.0
                    feat[q, 3 + k] = er[k] / wsum[k] if wsum[k] > 0 else 0.0
                h = hero[q]
                feat[q, 6] = hew[h] / hw[h] if hw[h] > 0 else 0.0
                m10 = 0.0
                m20 = 0.0
                c10 = min(nb, 10)
                c20 = min(nb, 20)
                for t in range(c20):
                    val = buf[(nb - 1 - t) % 20]
                    m20 += val
                    if t < c10:
                        m10 += val
                feat[q, 7] = m10 / c10 if c10 > 0 else 0.0
                feat[q, 8] = m20 / c20 if c20 > 0 else 0.0
                feat[q, 9] = day[q] - last_h[h] if last_h[h] >= 0 else -1.0
                feat[q, 10] = day[q] - last_any if last_any >= 0 else -1.0
            for q in range(i, j):
                h = hero[q]
                for k in range(3):
                    ew[k] = (1 - al[k]) * ew[k] + al[k] * (y[q] - 0.5)
                    er[k] = (1 - al[k]) * er[k] + al[k] * r[q]
                    wsum[k] = (1 - al[k]) * wsum[k] + al[k]
                hew[h] = (1 - ah) * hew[h] + ah * (y[q] - 0.5)
                hw[h] = (1 - ah) * hw[h] + ah
                buf[nb % 20] = r[q]
                nb += 1
                last_h[h] = day[q]
                last_any = day[q]
            i = j


def momentum_features(d):
    o = np.lexsort((d["replay_id"], d["day"], d["pid"]))
    pid = d["pid"][o]
    brk = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    ends = np.r_[brk[1:], len(pid)]
    feat = np.zeros((len(o), 11))
    _momentum(brk.astype(np.int64), ends.astype(np.int64), d["day"][o].astype(np.int64),
              d["hero"][o].astype(np.int64), d["y"][o].astype(np.float64),
              d["r"][o].astype(np.float64), feat)
    out = np.empty_like(feat)
    out[o] = feat
    names = ["EWMA WR hl10", "EWMA WR hl30", "EWMA WR hl100", "EWMA resid hl10",
             "EWMA resid hl30", "EWMA resid hl100", "EWMA hero WR hl10", "form10", "form20",
             "days since hero", "days since any"]
    res = {n: out[:, i] for i, n in enumerate(names)}
    # 365-day volumes (games on days t-365 .. t-1)
    for nm, key in (("vol365 all", d["pid"]), ("vol365 hero", d["pid"] * 128 + d["hero"])):
        o2 = np.lexsort((d["day"], key))
        comp = np.empty(len(o2), np.int64)
        kk = key[o2]
        gid = np.cumsum(np.r_[True, kk[1:] != kk[:-1]]) - 1
        comp = gid.astype(np.int64) * 1000000 + d["day"][o2]
        cnt = np.searchsorted(comp, comp, side="left") - np.searchsorted(comp, comp - 365, side="left")
        v = np.empty(len(o2))
        v[o2] = cnt
        res[nm] = np.log1p(v)
    return res


@njit(parallel=True, cache=True)
def _safe_lag(starts, ends, day, rid, val, lag, margin, out):
    for p in prange(starts.shape[0]):
        a, b = starts[p], ends[p]
        for i in range(a, b):
            out[i] = np.nan
            j = i - 1
            steps = 0
            while j >= a and steps < 400:
                if day[j] <= day[i] - lag and rid[j] < rid[i] - margin and not np.isnan(val[j]):
                    out[i] = val[j]
                    break
                j -= 1
                steps += 1


def safe_lagged(d, values, key, lag, margin):
    o = np.lexsort((d["replay_id"], d["day"], key))
    kk = key[o]
    brk = np.flatnonzero(np.r_[True, kk[1:] != kk[:-1]])
    ends = np.r_[brk[1:], len(kk)]
    out = np.empty(len(o))
    _safe_lag(brk.astype(np.int64), ends.astype(np.int64), d["day"][o].astype(np.int64),
              d["replay_id"][o].astype(np.int64), values[o].astype(np.float64), lag, margin, out)
    res = np.empty(len(o))
    res[o] = out
    return res


def game_fit(specs, D, lo, y, fit_m, test_m, rng, base_name, extra_base=None):
    eps = 1e-7
    idx = np.flatnonzero(test_m)
    boots = [rng.choice(idx, len(idx)) for _ in range(200)]
    lls, res = {}, {}
    for name, cols in specs.items():
        X = np.column_stack([lo] + [D[c] for c in cols])
        w = C.fit_logistic(X[fit_m], y[fit_m])
        p = C.predict(w, X)
        lls[name] = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        r = C.game_metrics(p[test_m], y[test_m])
        r["coef"] = w.tolist()
        res[name] = r
        res[name]["_p"] = p
    for name in specs:
        for ref in [base_name] + ([] if extra_base is None else [extra_base]):
            dd = lls[ref] - lls[name]
            bs = [dd[b].mean() for b in boots]
            res[name][f"gain_vs[{ref}]"] = [float(dd[idx].mean()), float(np.percentile(bs, 2.5)),
                                            float(np.percentile(bs, 97.5))]
    return res


def main():
    t0 = time.time()
    d = C.load_slots()
    H = len(d["hero_names"])
    meta = C.hero_meta(d["hero_names"])
    kz = np.load(K.KOUT)
    e_mask, n_p, n_ph, table, r_adj = prepare(d)
    V = kz["V_cf2"]
    days = d["day"]
    post = ~d["in_sample"]
    _, first_row = np.unique(d["g"][post], return_index=True)
    med = np.median(days[post][first_row])
    v1r, v2r = post & (days < med), post & (days >= med)
    with open(K.PARAMS) as f:
        prm = json.load(f)
    th = kz["theta_+CF rank 2"]
    ex = lambda i: float(np.sqrt(np.exp(th[i])))
    p_static = {"sd_player_s": ex(0), "sd_role": ex(1), "sd_fine": ex(2), "sd_melee": ex(3),
                "sd_hero_s": ex(4), "sd_cf": ex(5)}

    def sd_lam(filt, p):
        sd, lam, br = K.split(p)
        return sd, filt.lam_state(lam, br)
    out = {"slot": {}, "slot_by_dsl": {}, "var_ratio_by_dsl": {}, "game": {}, "coverage": {}}

    # ---------- skill model predictions for every row (full-data filters)
    filt_d = K.Filter(d, meta, V, r_adj)
    filt_g = K.Filter(d, meta, V, r_adj, clock="games", n_p=n_p)
    rows = filt_d.rows
    assert np.array_equal(rows, filt_g.rows)
    preds = {}

    def run(filt, sd, lam, name, upd=None):
        m, s2, _ = filt.run(sd, lam, upd)
        pm = np.empty(len(d["pid"]))
        ps = np.empty(len(d["pid"]))
        pm[filt.rows] = m
        ps[filt.rows] = s2
        return pm, ps

    mu = table[C.exp_bins(n_p, n_ph)]
    preds["EB static kernel online"] = run(filt_d, *sd_lam(filt_d, p_static), "eb")
    specs_ss = {"SS days": ("drift days", filt_d),
                "SS days, hero rate by role": ("drift days, hero rates by role", filt_d),
                "SS games": ("drift games", filt_g)}
    for nm, (key, filt) in specs_ss.items():
        preds[nm] = run(filt, *sd_lam(filt, prm[key]), nm)
        print(f"{nm}: {time.time() - t0:.0f}s", flush=True)
    # EB decay (phase-1 GP code)
    bases = {k[6:]: kz[k] for k in kz.files if k.startswith("basis_")}
    Kcf = C.kernel_from([bases[b] for b in KERNELS["+CF rank 2"]], th)
    qidx, S, P, _, _, _, _ = C.online_state(d, np.ones_like(post), post, H, r_adj, half_life=365)
    m, s2 = C.gp_query(S, P, d["hero"][qidx], Kcf)
    pm = np.full(len(d["pid"]), np.nan)
    ps = np.full(len(d["pid"]), np.nan)
    pm[qidx], ps[qidx] = m, s2
    preds["EB decay hl=365"] = (pm, ps)
    del S, P

    # ---------- momentum features and days since last played
    mf = momentum_features(d)
    dsl = mf["days since hero"]
    q = np.flatnonzero(post)
    r_q, wp_q, v_q = d["r"][q], d["wp"][q], d["v"][q]
    in_v2 = v2r[q]
    nh_q = n_ph[q]
    db = dsl_bin(dsl[q])
    out["slot_share_by_dsl"] = {l: float((db[in_v2] == i).mean()) for i, l in enumerate(DSL_LABELS)}
    for nm, (pm, ps) in preds.items():
        pr = mu[q] + pm[q]
        sd_ = np.sqrt(ps[q])
        o = {"all": slot_stats(r_q[in_v2], wp_q[in_v2], pr[in_v2], sd_[in_v2])}
        for i, l in enumerate(N_LABELS):
            s = in_v2 & (nbin(nh_q) == i)
            o[l] = slot_stats(r_q[s], wp_q[s], pr[s], sd_[s])
        out["slot"][nm] = o
        ob, vr = {}, {}
        for i, l in enumerate(DSL_LABELS):
            s = in_v2 & (db == i)
            if s.sum() < 500:
                continue
            ob[l] = slot_stats(r_q[s], wp_q[s], pr[s], sd_[s])
            e2 = (r_q[s] - pr[s]) ** 2 - v_q[s]
            # bootstrap CI of the variance ratio
            rng = np.random.RandomState(1)
            ii = np.flatnonzero(s)
            bs = []
            for _ in range(100):
                jj = rng.choice(len(ii), len(ii))
                bs.append(e2[jj].mean() / ps[q][s][jj].mean())
            vr[l] = [float(e2.mean() / ps[q][s].mean()), float(np.percentile(bs, 2.5)),
                     float(np.percentile(bs, 97.5))]
        out["slot_by_dsl"][nm] = ob
        out["var_ratio_by_dsl"][nm] = vr
        print(f"slot {nm:28s} all {o['all']['ll_gain_x1000']:+.3f} | by dsl " +
              " ".join(f"{l}:{v['ll_gain_x1000']:+.2f}" for l, v in ob.items()) +
              " | var ratio " + " ".join(f"{l}:{v[0]:.2f}" for l, v in vr.items()), flush=True)

    # ---------- game-level
    n_games = int(d["g"].max()) + 1
    y = np.zeros(n_games)
    wg = np.full(n_games, 0.5)
    t0m = d["team"] == 0
    y[d["g"][t0m]] = d["y"][t0m]
    wg[d["g"][t0m]] = d["wp"][t0m]
    lo = np.log(wg / (1 - wg))
    full = np.bincount(d["g"][post], minlength=n_games) == 10
    gv1 = np.zeros(n_games, bool)
    gv1[d["g"][v1r]] = True
    gv2 = np.zeros(n_games, bool)
    gv2[d["g"][v2r]] = True
    fit_m, test_m = gv1 & full, gv2 & full
    D = {}
    for nm, (pm, ps) in preds.items():
        pmf = np.where(np.isnan(pm), 0, pm)
        psf = np.where(np.isnan(ps), 0, ps)
        D[nm] = team_diff(mu + pmf, d["g"], d["team"], n_games)
        D[nm + " |sd"] = team_diff(np.sqrt(psf), d["g"], d["team"], n_games)
        D[nm + " |lower80"] = team_diff(mu + pmf - Z80 * np.sqrt(psf), d["g"], d["team"], n_games)
        # reliability-weighted: estimate x (1 - s2 / prior variance at n = 0)
        prior = np.median(psf[(n_ph == 0) & post])
        D[nm + " |m*rel"] = team_diff((mu + pmf) * np.clip(1 - psf / prior, 0, 1), d["g"], d["team"],
                                      n_games)
    D["experience only"] = team_diff(mu, d["g"], d["team"], n_games)
    for nm, x in mf.items():
        if nm.startswith("days since"):
            continue
        D[nm] = team_diff(x, d["g"], d["team"], n_games)
    # lagged MMR, naive and upload-order-safe
    ids_per_day = float(np.median(np.diff(np.array(
        [np.median(d["replay_id"][days == t]) for t in np.unique(days[post])]))))
    margin = int(2 * ids_per_day)
    out["mmr_safe_margin_replay_ids"] = margin
    role_of = meta["blizz"][d["hero"]]
    for G in (1, 7, 30):
        for col, key, nm in (("player_mmr", d["pid"], "player"),
                             ("role_mmr", d["pid"] * 8 + role_of, "role"),
                             ("hero_mmr", d["pid"] * 128 + d["hero"], "hero")):
            vals = d[col].astype(np.float64)
            for kind, mg in (("naive", -10 ** 12), ("safe", margin)):
                x = safe_lagged(d, vals, key, G, mg)
                if nm == "hero" or nm == "role":
                    x = np.where(np.isnan(x), D.get(f"_fill_{kind}_{G}", x), x)
                if nm == "player":
                    D[f"_fill_{kind}_{G}"] = x
                gm = np.nanmean(x)
                D[f"lag {nm} MMR {kind} G={G}"] = team_diff(np.where(np.isnan(x), gm, x) / 100.0,
                                                            d["g"], d["team"], n_games)
        print(f"MMR G={G}: {time.time() - t0:.0f}s", flush=True)

    rng = np.random.RandomState(0)
    best = "SS days"
    base = "M0 population WP"
    specs = {base: []}
    for nm in preds:
        specs[nm] = [nm]
    specs["experience only"] = ["experience only"]
    # item 2: momentum/volume arms alone and on top of the best skill model
    arms = ["vol365 all", "vol365 hero", "EWMA WR hl10", "EWMA WR hl30", "EWMA WR hl100",
            "EWMA resid hl10", "EWMA resid hl30", "EWMA resid hl100", "EWMA hero WR hl10",
            "form10", "form20"]
    for a in arms:
        specs[f"{a} alone"] = [a]
        specs[f"{best} + {a}"] = [best, a]
    specs["Max-style: vol365 all + EWMA WR hl30"] = ["vol365 all", "EWMA WR hl30"]
    specs["Max-style: vol365 all+hero + EWMA WR hl10/30/100"] = [
        "vol365 all", "vol365 hero", "EWMA WR hl10", "EWMA WR hl30", "EWMA WR hl100"]
    specs[f"{best} + all momentum/volume arms"] = [best] + arms
    specs["EB static kernel online + all momentum/volume arms"] = ["EB static kernel online"] + arms
    # item 3: uncertainty
    for nm in ("EB static kernel online", best):
        specs[f"{nm} + sd sum"] = [nm, nm + " |sd"]
        specs[f"{nm} lower80 only"] = [nm + " |lower80"]
        specs[f"{nm} + lower80"] = [nm, nm + " |lower80"]
        specs[f"{nm} m*rel only"] = [nm + " |m*rel"]
        specs[f"{nm} + m*rel"] = [nm, nm + " |m*rel"]
    # item 0: MMR
    for G in (1, 7, 30):
        for kind in ("naive", "safe"):
            cols = [f"lag {x} MMR {kind} G={G}" for x in ("player", "role", "hero")]
            specs[f"lag MMR {kind} G={G}"] = cols
            specs[f"{best} + lag MMR {kind} G={G}"] = [best] + cols
    specs[f"{best} + all arms + lag MMR safe G=7"] = [best] + arms + \
        [f"lag {x} MMR safe G=7" for x in ("player", "role", "hero")]
    res = game_fit(specs, D, lo, y, fit_m, test_m, rng, base, extra_base=best)
    for nm, r in res.items():
        p = r.pop("_p")
        out["game"][nm] = r
        g0 = r[f"gain_vs[{base}]"]
        g1 = r[f"gain_vs[{best}]"]
        print(f"  {nm:62s} acc {r['acc']:.4f} ll {r['logloss']:.5f} gain {g0[0]:+.5f} "
              f"[{g0[1]:+.5f},{g0[2]:+.5f}] vs {best} {g1[0]:+.5f} [{g1[1]:+.5f},{g1[2]:+.5f}]",
              flush=True)
    # item 3b: calibration within strata of total uncertainty
    pm, ps = preds[best]
    tot_unc = np.bincount(d["g"], weights=np.where(np.isnan(ps), 0, ps), minlength=n_games)
    X0 = np.column_stack([lo])
    X1 = np.column_stack([lo, D[best]])
    w0 = C.fit_logistic(X0[fit_m], y[fit_m])
    w1 = C.fit_logistic(X1[fit_m], y[fit_m])
    p0, p1 = C.predict(w0, X0), C.predict(w1, X1)
    qs = np.percentile(tot_unc[test_m], [20, 40, 60, 80])
    strata = {}
    for i in range(5):
        lo_, hi_ = ([-1] + list(qs))[i], (list(qs) + [1e9])[i]
        s = test_m & (tot_unc > lo_) & (tot_unc <= hi_)
        strata[f"quintile {i + 1}"] = {
            "games": int(s.sum()), "mean_slot_sd_pp": float(100 * np.sqrt(tot_unc[s].mean() / 10)),
            "M0": C.game_metrics(p0[s], y[s]), best: C.game_metrics(p1[s], y[s])}
        a, b = strata[f"quintile {i + 1}"]["M0"], strata[f"quintile {i + 1}"][best]
        print(f"  unc q{i + 1} sd {strata[f'quintile {i + 1}']['mean_slot_sd_pp']:.2f}pp: M0 acc "
              f"{a['acc']:.4f} ece {a['ece']:.4f} slope {a['cal_slope']:.3f} | {best} acc "
              f"{b['acc']:.4f} ece {b['ece']:.4f} slope {b['cal_slope']:.3f}", flush=True)
    out["calibration_by_uncertainty"] = strata

    # ---------- cell coverage frozen at the V1/V2 split, by days since last played
    upd = days[rows] < med
    for nm, p_, filt in (("EB static kernel online", p_static, filt_d),
                         ("SS days", prm["drift days"], filt_d),
                         ("SS days, hero rate by role", prm["drift days, hero rates by role"],
                          filt_d)):
        sd, lam = sd_lam(filt, p_)
        pm, ps = run(filt, sd, lam, nm, upd)
        # cell = (player, hero) played in V2; prediction = mean over its V2 slots
        s = v2r
        cell = d["pid"][s] * 128 + d["hero"][s]
        u, inv, nf = np.unique(cell, return_inverse=True, return_counts=True)
        rbar = np.bincount(inv, weights=r_adj[s]) / nf
        noise = np.bincount(inv, weights=d["v"][s]) / nf ** 2
        mbar = np.bincount(inv, weights=pm[s]) / nf
        s2bar = np.bincount(inv, weights=ps[s]) / nf
        # days since hero last played as of the split
        first = np.zeros(len(u), np.int64)
        first[inv[::-1]] = np.flatnonzero(s)[::-1]
        # first V2 game's previous play of the hero is before the split
        last_day = days[first] - dsl[first]
        dsl_split = np.where(dsl[first] >= 0, med - last_day, -1)
        bz = dsl_bin(dsl_split)
        z = np.abs(rbar - mbar) / np.sqrt(s2bar + noise)
        cv = {}
        for lab, sel in [("all", np.ones(len(u), bool))] + \
                [(l, bz == i) for i, l in enumerate(DSL_LABELS)]:
            ss = sel & (nf >= 10)
            if ss.sum() < 50:
                continue
            err = (rbar[ss] - mbar[ss]) ** 2 - noise[ss]
            cv[lab] = {"cells": int(ss.sum()), "cover80": float((z[ss] < Z80).mean()),
                       "var_ratio": float(err.mean() / s2bar[ss].mean()),
                       "claimed_sd_pp": float(100 * np.sqrt(s2bar[ss].mean()))}
        out["coverage"][nm] = cv
        print(f"  coverage {nm:28s} " + " ".join(
            f"{k}:{v['cover80']:.2f}/{v['var_ratio']:.2f}/{v['claimed_sd_pp']:.1f}/{v['cells']}"
            for k, v in cv.items()), flush=True)
    with open(os.path.join(C.RESULTS, "p3_sd_eval.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
