"""
P3 hero strength, stage 2: strictly-future evaluation.

Test window: every snapshot game after the WP training cutoff (2026-02-10 ..
2026-05-22), split at the median day into V1 (combiner fit, half-life
choice) and V2 (test). For every player slot the strength estimate uses
only games from days <= game day - lag (online, causal), or only E games
(static, as of the cutoff).

Per method and per history size n (games by this player on this hero seen
before the game) we report on V2 slots:
  ll_gain   per-slot log-loss gain (x1000) of p = wp_team + prediction
            over p = wp_team (no fitting: the estimate is used as-is)
  mse_gain  E[r^2] - E[(r - prediction)^2] in pp^2 (1e-4)
  slope     OLS slope of r on the prediction (1 = shrinkage is right)
  sd        median posterior sd (pp)
Game level: logistic combiner of logit(WP) and team differences, fit on V1,
scored on V2 with replay bootstrap CIs.
Coverage: posterior as of a date vs the mean residual of the same
player x hero cell afterwards; 80% predictive interval
m +- 1.2816 sqrt(s^2 + noise/n_future).

Usage (from training/): python3 personalization/p3_hs_eval.py
Outputs: results/p3_hs_eval.json, results/p3_hs_eval.txt (log)
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C
from p3_hs_fit import KERNELS, KOUT, main_kernels

N_EDGES = [0, 1, 3, 6, 11, 21, 51, 101, 201, 10 ** 9]
N_LABELS = ["0", "1-2", "3-5", "6-10", "11-20", "21-50", "51-100", "101-200", "201+"]
Z80 = 1.2815516


def nbin(n):
    return np.searchsorted(N_EDGES, n, side="right") - 1


def slot_stats(r, wp, pred, sd=None):
    eps = 1e-3
    p1 = np.clip(wp + pred, eps, 1 - eps)
    y = r + wp
    ll0 = -(y * np.log(wp) + (1 - y) * np.log(1 - wp))
    ll1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    out = {"n": int(len(r)),
           "ll_gain_x1000": float(1000 * (ll0 - ll1).mean()),
           "mse_gain_pp2": float(1e4 * (r ** 2 - (r - pred) ** 2).mean())}
    if pred.std() > 1e-9:
        out["slope"] = float(np.cov(r, pred)[0, 1] / pred.var())
    if sd is not None:
        out["sd_median_pp"] = float(100 * np.median(sd))
        out["sd_mean_pp"] = float(100 * sd.mean())
    return out


def by_n(r, wp, pred, n, sd=None):
    b = nbin(n)
    res = {"all": slot_stats(r, wp, pred, sd)}
    for i, lab in enumerate(N_LABELS):
        m = b == i
        if m.sum() > 100:
            res[lab] = slot_stats(r[m], wp[m], pred[m], None if sd is None else sd[m])
    return res


def team_diff(values, g, team, n_games):
    sign = np.where(team == 0, 1.0, -1.0)
    return np.bincount(g, weights=sign * values, minlength=n_games)


def decayed_state(d, mask, t0, half_life, H, r_adj):
    """Static state as of day t0 with exponential down-weighting."""
    w = np.ones(len(d["day"])) if half_life is None else \
        np.exp(-np.log(2) / half_life * (t0 - d["day"]))
    dd = dict(d)
    dd["v"] = d["v"] / w
    return C.static_state(dd, mask, d["n_players"], H, r_adj)


def coverage(d, K, S, P, fut_mask, r_adj, H, hist_n):
    """Cells (player, hero) played in fut_mask: posterior from (S, P) vs the
    cell's future mean detrended residual."""
    key = d["pid"][fut_mask] * H + d["hero"][fut_mask]
    u, inv, nf = np.unique(key, return_inverse=True, return_counts=True)
    sr = np.bincount(inv, weights=r_adj[fut_mask])
    sv = np.bincount(inv, weights=d["v"][fut_mask])
    rbar = sr / nf
    noise = sv / nf ** 2
    pid, h = u // H, u % H
    m, s2 = C.gp_query(np.ascontiguousarray(S[pid]), np.ascontiguousarray(P[pid]), h, K)
    nh = hist_n[pid, h]
    z = (rbar - m) / np.sqrt(s2 + noise)
    res = {}
    for lab, sel in [("all", np.ones(len(u), bool))] + \
            [(N_LABELS[i], nbin(nh) == i) for i in range(len(N_LABELS))]:
        for fl, fm in (("nf>=1", nf >= 1), ("nf>=10", nf >= 10), ("nf>=30", nf >= 30)):
            s = sel & fm
            if s.sum() < 50:
                continue
            err2 = (rbar[s] - m[s]) ** 2
            res[f"{lab}|{fl}"] = {
                "cells": int(s.sum()),
                "cover80": float((np.abs(z[s]) < Z80).mean()),
                "cover80_latent_only": float((np.abs(rbar[s] - m[s]) < Z80 * np.sqrt(s2[s])).mean()),
                "claimed_sd_pp": float(100 * np.sqrt(s2[s].mean())),
                "realized_sd_pp": float(100 * np.sqrt(max(err2.mean() - noise[s].mean(), 0))),
                "var_ratio": float((err2.mean() - noise[s].mean()) / s2[s].mean()),
            }
    return res


def main():
    t_start = time.time()
    d = C.load_slots()
    MAIN, BEST = main_kernels()
    H = len(d["hero_names"])
    meta = C.hero_meta(d["hero_names"])
    kz = np.load(KOUT)
    table = kz["experience_table"]
    bases = {k[6:]: kz[k] for k in kz.files if k.startswith("basis_")}
    Ks = {name: C.kernel_from([bases[b] for b in bl], kz["theta_" + name])
          for name, bl in KERNELS.items()}
    n_p, n_ph = C.experience_counts(d)
    r_adj = d["r"] - table[C.exp_bins(n_p, n_ph)]
    days = d["day"]
    e_mask = d["in_sample"] & (days >= C.day_of(C.E_START))
    post = ~d["in_sample"]
    _, first_row = np.unique(d["g"][post], return_index=True)
    med = np.median(days[post][first_row])  # median game day, as in p3_nested_lift
    v1_row, v2_row = post & (days < med), post & (days >= med)
    n_games = int(d["g"].max()) + 1
    out = {"windows": {"E_slots": int(e_mask.sum()), "V1_slots": int(v1_row.sum()),
                       "V2_slots": int(v2_row.sum()), "split_day": float(med)},
           "slot": {}, "game": {}, "coverage": {}}
    print(json.dumps(out["windows"]), flush=True)

    # ---------------- online / static states and GP predictions
    configs = {"online hl=inf": dict(add=np.ones_like(post), hl=None, lag=1),
               "static@cutoff": dict(add=e_mask, hl=None, lag=1),
               "online hl=365": dict(add=np.ones_like(post), hl=365, lag=1),
               "online hl=180": dict(add=np.ones_like(post), hl=180, lag=1),
               "online hl=90": dict(add=np.ones_like(post), hl=90, lag=1),
               "online hl=inf lag=7": dict(add=np.ones_like(post), hl=None, lag=7),
               "online hl=inf lag=30": dict(add=np.ones_like(post), hl=None, lag=30)}
    preds = {}
    qidx = None
    for cname, cfg in configs.items():
        t0 = time.time()
        qi, S, P, Nh, Np, Wh, Rh = C.online_state(d, cfg["add"], post, H, r_adj,
                                                  half_life=cfg["hl"], lag=cfg["lag"])
        if qidx is None:
            qidx = qi
            q_nh = Nh.astype(np.int64)
            q_np = Np.astype(np.int64)
            mu = table[C.exp_bins(q_np, q_nh)]
            preds["experience only"] = (mu, None)
            preds["raw WR"] = (np.where(Nh > 0, Wh / np.maximum(Nh, 1) - 0.5, 0.0), None)
            preds["raw mean residual"] = (np.where(Nh > 0, Rh / np.maximum(Nh, 1), 0.0), None)
        else:
            assert np.array_equal(qi, qidx)
        hq = d["hero"][qidx]
        for kname, K in Ks.items():
            if cname != "online hl=inf" and kname not in MAIN:
                continue
            if "lag=" in cname and kname not in ("player+hero", BEST):
                continue
            m, s2 = C.gp_query(S, P, hq, K)
            preds[f"{kname} [{cname}]"] = (mu + m, np.sqrt(s2))
        print(f"{cname}: {time.time() - t0:.0f}s", flush=True)
        del S, P
    r_q = d["r"][qidx]
    wp_q = d["wp"][qidx]
    in_v1 = v1_row[qidx]
    in_v2 = v2_row[qidx]

    # ---------------- lagged MMR (player, role, hero), same causal lag
    role_of = meta["blizz"][d["hero"]]
    mmr_feats = {}
    for G in (1, 7, 30):
        lp = C.lagged_value(d, d["player_mmr"].astype(np.float32), d["pid"], G)
        lr = C.lagged_value(d, d["role_mmr"].astype(np.float32), d["pid"] * 8 + role_of, G)
        lh = C.lagged_value(d, d["hero_mmr"].astype(np.float32), d["pid"] * 128 + d["hero"], G)
        mmr_feats[G] = (lp, lr, lh)
    # MMR slot predictors: centered on the lobby, linear map fit on V1
    lobby_cnt = np.bincount(d["g"], minlength=n_games)

    def centered(x):
        gm = np.nanmean(x)
        xf = np.where(np.isnan(x), gm, x)
        lob = np.bincount(d["g"], weights=xf, minlength=n_games) / np.maximum(lobby_cnt, 1)
        return (xf - lob[d["g"]]) / 100.0

    for G in (1, 7, 30):
        lp, lr, lh = mmr_feats[G]
        hero_fill = np.where(np.isnan(lh), np.where(np.isnan(lr), lp, lr), lh)
        for nm, x in (("lag player MMR", lp), ("lag hero MMR", hero_fill),
                      ("lag role MMR", np.where(np.isnan(lr), lp, lr))):
            xc = centered(x)[qidx]
            Xv = np.column_stack([xc])
            b = np.linalg.lstsq(np.column_stack([np.ones(in_v1.sum()), Xv[in_v1]]),
                                r_q[in_v1], rcond=None)[0]
            preds[f"{nm} (G={G}d, V1 linear map)"] = (b[0] + Xv @ b[1:], None)
            mmr_feats[(G, nm)] = x

    # ---------------- slot-level tables (V1 and V2)
    for name, (pr, sd) in preds.items():
        out["slot"][name] = {
            "V1": slot_stats(r_q[in_v1], wp_q[in_v1], pr[in_v1]),
            "V2": by_n(r_q[in_v2], wp_q[in_v2], pr[in_v2], q_nh[in_v2],
                       None if sd is None else sd[in_v2])}
    print("slot-level (V2 all): ll_gain_x1000 / mse_gain_pp2 / slope / sd", flush=True)
    for name in preds:
        a = out["slot"][name]["V2"]["all"]
        print(f"  {name:55s} {a['ll_gain_x1000']:+.3f} {a['mse_gain_pp2']:+.2f} "
              f"{a.get('slope', float('nan')):.3f} {a.get('sd_median_pp', float('nan')):.2f}"
              f" | V1 {out['slot'][name]['V1']['ll_gain_x1000']:+.3f}", flush=True)
    # distribution of V2 slots over n
    b = nbin(q_nh[in_v2])
    out["slot_share_by_n"] = {lab: float((b == i).mean()) for i, lab in enumerate(N_LABELS)}

    # ---------------- game level
    y = np.zeros(n_games)
    wp0 = np.full(n_games, 0.5)
    t0_mask = d["team"] == 0
    # game outcome y (team 0 won) and wp0 from team-0 rows
    y[d["g"][t0_mask]] = d["y"][t0_mask]
    wp0[d["g"][t0_mask]] = d["wp"][t0_mask]
    lo = np.log(wp0 / (1 - wp0))
    cnt_post = np.bincount(d["g"][post], minlength=n_games)
    full = cnt_post == 10
    gv1 = np.zeros(n_games, bool)
    gv1[d["g"][v1_row]] = True
    gv2 = np.zeros(n_games, bool)
    gv2[d["g"][v2_row]] = True
    fit_m, test_m = gv1 & full, gv2 & full
    g_q, team_q = d["g"][qidx], d["team"][qidx]
    D = {name: team_diff(pr, g_q, team_q, n_games) for name, (pr, _) in preds.items()}
    for G in (1, 7, 30):
        for nm in ("lag player MMR", "lag hero MMR", "lag role MMR"):
            x = mmr_feats[(G, nm)]
            gm = np.nanmean(x)
            D[f"{nm} raw G={G}"] = team_diff(np.where(np.isnan(x), gm, x) / 100.0,
                                             d["g"], d["team"], n_games)
    best_gp = f"{BEST} [online hl=inf]"
    specs = {"M0 population WP": []}
    for name in preds:
        if "linear map" in name:
            continue
        specs[name] = [name]
    for G in (1, 7, 30):
        specs[f"lag player MMR G={G}"] = [f"lag player MMR raw G={G}"]
        specs[f"lag hero MMR G={G}"] = [f"lag hero MMR raw G={G}"]
        specs[f"lag role MMR G={G}"] = [f"lag role MMR raw G={G}"]
        specs[f"lag player+role+hero MMR G={G}"] = [f"lag player MMR raw G={G}",
                                                    f"lag role MMR raw G={G}",
                                                    f"lag hero MMR raw G={G}"]
    for G in (1, 7, 30):
        for gpn in (best_gp, f"{BEST} [online hl=180]", "player+hero [online hl=inf]"):
            specs[f"{gpn} + lag player MMR G={G}"] = [gpn, f"lag player MMR raw G={G}"]
            specs[f"{gpn} + lag all MMR G={G}"] = [gpn, f"lag player MMR raw G={G}",
                                                   f"lag role MMR raw G={G}",
                                                   f"lag hero MMR raw G={G}"]
    specs[f"{best_gp} + raw WR"] = [best_gp, "raw WR"]
    for G in (7, 30):
        gl = f"{BEST} [online hl=inf lag={G}]"
        specs[f"{gl} + lag player MMR G={G}"] = [gl, f"lag player MMR raw G={G}"]
        specs[f"{gl} + lag all MMR G={G}"] = [gl, f"lag player MMR raw G={G}",
                                              f"lag role MMR raw G={G}", f"lag hero MMR raw G={G}"]
    rng = np.random.RandomState(0)
    idx_all = np.flatnonzero(test_m)
    boots = [rng.choice(idx_all, len(idx_all)) for _ in range(200)]
    eps = 1e-7
    base_ll = None
    print("game-level (V2):", flush=True)
    for name, cols in specs.items():
        X = np.column_stack([lo] + [D[c] for c in cols])
        w = C.fit_logistic(X[fit_m], y[fit_m])
        p = C.predict(w, X)
        ll = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        if base_ll is None:
            base_ll = ll
        diffs = [float((base_ll[bb] - ll[bb]).mean()) for bb in boots]
        res = C.game_metrics(p[test_m], y[test_m])
        res["coef"] = w.tolist()
        res["gain_vs_M0"] = float((base_ll[test_m] - ll[test_m]).mean())
        res["gain_ci95"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        out["game"][name] = res
        print(f"  {name:62s} acc {res['acc']:.4f} ll {res['logloss']:.5f} "
              f"gain {res['gain_vs_M0']:+.5f} [{res['gain_ci95'][0]:+.5f},{res['gain_ci95'][1]:+.5f}]"
              f" ece {res['ece']:.4f} slope {res['cal_slope']:.3f}", flush=True)
    out["n_games"] = {"V1": int(fit_m.sum()), "V2": int(test_m.sum())}

    # ---------------- coverage
    cut_day = int(days[post].min())
    for label, t0, hist_mask, fut in (
            ("static@cutoff -> all post games", cut_day, e_mask, post),
            ("hl=inf @V2 start -> V2 games", int(med), ~v2_row, v2_row),
            ("hl=180 @V2 start -> V2 games", int(med), ~v2_row, v2_row)):
        hl = 180 if "hl=180" in label else None
        S, P, N, _, _ = decayed_state(d, hist_mask, t0, hl, H, r_adj)
        out["coverage"][label] = {}
        for kname in MAIN:
            cv = coverage(d, Ks[kname], S, P, fut, r_adj, H, N)
            out["coverage"][label][kname] = cv
            a = cv["all|nf>=10"]
            print(f"  cov {label:36s} {kname:26s} cells {a['cells']} cover80 {a['cover80']:.3f} "
                  f"var_ratio {a['var_ratio']:.2f}", flush=True)
        del S, P, N

    with open(os.path.join(C.RESULTS, "p3_hs_eval.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
