"""
Paper 2 narrative facts, section 11: is the carry-forward of a player's record
on a hero (P2_NARRATIVE_FACTS section 3: 24/30/43/57% by games bin) right?

Builds the exact record of section 3 (p3_c_merged.merged_predict's augmented
rows: corpus slot rows with r = y - WP, Heroes Profile rows with
r = y - hero-quarter win rate; record = sum of r over the player's games on
the hero on EARLIER days / their number), on the full_1017 panel, and:

  real       the section 3 estimator: per games bin, OLS slope of the scored
             slot's r on the record x, V2 + OOT slots with 30+ games on the hero
             (checks the record reproduces arm A2S's s_rec)
  relative   the same with the record taken relative to the player's own
             average over all heroes (slope on x_hero - x_player, controlling
             for x_player)
  window     "later" = all of the cell's V2 + OOT games: record as of the V2
             start, outcome = mean r over the cell's games from then on,
             slope weighted by the number of later games
  recency    records over the last 25 / 50 / 100 games on the hero, and
             exponentially weighted records (half-life 180 / 365 / 730 days),
             same bins (by the full count)
  simulate   outcomes replaced by draws around each row's own baseline
             p0 = y - r: y ~ Bernoulli(p0 + theta_cell), theta_cell ~ N(0, tau^2),
             tau = 3, 4, 5 pp, theta 100% persistent; and a drifting version
             (an Ornstein-Uhlenbeck path per cell in calendar days, correlation
             0.74 at one year, tau 4 pp). The record and the outcome are
             rebuilt from the simulated outcomes with the same code, so the
             estimator sees exactly the real structure of counts, timing and
             baselines. Two seeds each.

Runs on a worker: P3_CACHE=<cache_full> python personalization/p2f_carry.py
Output: $P3_CACHE/p2f_carry.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_hs_core as C
import p3_x_common as X
import p3_c_merged as M
import p3_c_tests as T
from p3_heroes import HKEY

BINS = [(30, 59), (60, 119), (120, 249), (250, 499), (500, 799), (800, 10 ** 9), (250, 10 ** 9)]
LABS = ["30-59", "60-119", "120-249", "250-499", "500-799", "800+", "250+ (all)"]
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)


d = X.load_ext()
ex, _ = M.load_rows(os.path.join(C.CACHE, "merged_rows_full_1017.npz"), d)
n = len(d["pid"])
w, med = T.windows(d)
A = M.merged_predict(d, ex, None, np.zeros(n, bool), exp_counts="slots", aug_only=True)
aug, o = A["aug"], A["order"]
N = len(o)
pos = np.empty(N, np.int64)
pos[o] = np.arange(N)
qrows = pos[np.flatnonzero(w >= 2)]          # V2 + OOT slot rows in aug order
qwin = w[w >= 2]
day, pid, hero = aug["day"].astype(np.int64), aug["pid"].astype(np.int64), aug["hero"].astype(np.int64)
key = pid * HKEY + hero
y, r = aug["y"].astype(np.float64), aug["r"].astype(np.float64)
p0 = y - r
log(f"aug rows {N:,}, queries {len(qrows):,}")

# per-cell order for recency and OU
oc = np.lexsort((day, key))
kc, dc = key[oc], day[oc]
gstart = np.maximum.accumulate(np.where(np.r_[True, kc[1:] != kc[:-1]], np.arange(N), 0))
bstart = np.maximum.accumulate(np.where(np.r_[True, (kc[1:] != kc[:-1]) | (dc[1:] != dc[:-1])], np.arange(N), 0))
inv_oc = np.empty(N, np.int64)
inv_oc[oc] = np.arange(N)


def lagsum(vals):
    """sum over the same cell on earlier days (lag-1 rule), per aug row."""
    cs = np.r_[0.0, np.cumsum(vals[oc])]
    out = np.empty(N)
    out[oc] = cs[bstart] - cs[gstart]
    return out


def lastk(vals, k):
    cs = np.r_[0.0, np.cumsum(vals[oc])]
    lo = np.maximum(gstart, bstart - k)
    out = np.empty(N)
    out[oc] = cs[bstart] - cs[lo]
    return out


def lagsum_player(vals):
    return M.lag1_key_sums(day, pid, vals)


nh = lagsum(np.ones(N))
qn = nh[qrows]


def slope(xx, rr, ww=None):
    if ww is None:
        ww = np.ones(len(xx))
    xm, rm = np.average(xx, weights=ww), np.average(rr, weights=ww)
    return float(np.sum(ww * (xx - xm) * (rr - rm)) / np.sum(ww * (xx - xm) ** 2))


def by_bin(x, rr, nn, extra=None):
    out = {}
    grp = (pid[qrows] * 2654435761 % 2 ** 32) % 20
    for (lo, hi), lb in zip(BINS, LABS):
        m = (nn >= lo) & (nn <= hi) & np.isfinite(x)
        b = slope(x[m], rr[m])
        jk = np.array([slope(x[m & (grp != gg)], rr[m & (grp != gg)]) for gg in range(20)])
        se = float(np.sqrt(19 / 20 * ((jk - jk.mean()) ** 2).sum()))
        noise = 0.245 / nn[m]
        rel = float(1 - noise.mean() / x[m].var())
        out[lb] = {"slots": int(m.sum()), "median_n": float(np.median(nn[m])), "slope": b, "se": se,
                   "reliability_of_record": rel, "record_sd_pp": float(100 * x[m].std()),
                   "carried_cov_pp2": float(1e4 * np.cov(x[m], rr[m])[0, 1])}
    return out


res = {"queries": int(len(qrows))}
rec = lagsum(r)
x = np.where(nh > 0, rec / np.maximum(nh, 1), np.nan)
res["real"] = by_bin(x[qrows], r[qrows], qn)
log("real " + json.dumps({k: round(v["slope"], 3) for k, v in res["real"].items()}))

# relative to the player's own average: slope on (x_h - x_p) controlling for x_p
npl = lagsum_player(np.ones(N))
xp = np.where(npl > 0, lagsum_player(r) / np.maximum(npl, 1), np.nan)
rel = {}
for (lo, hi), lb in zip(BINS, LABS):
    m = (qn >= lo) & (qn <= hi)
    xh, xpp, rr = x[qrows][m], xp[qrows][m], r[qrows][m]
    Xm = np.column_stack([np.ones(m.sum()), xh - xpp, xpp])
    coef = np.linalg.lstsq(Xm, rr, rcond=None)[0]
    rel[lb] = {"slope_relative": float(coef[1]), "slope_player_average": float(coef[2]),
               "relative_sd_pp": float(100 * (xh - xpp).std())}
res["relative"] = rel
log("relative " + json.dumps({k: round(v["slope_relative"], 3) for k, v in rel.items()}))

# window definition: record as of the V2 start; outcome = mean of later V2+OOT games on the cell
v2start = int(d["day"][(w == 2)].min())
slot_aug = pos[np.arange(n)]
later = np.zeros(N, bool)
later[slot_aug[w >= 2]] = True
pre = day < v2start
ck = np.unique(key[later])
ks, ki = np.unique(key, return_inverse=True)
Sp = np.bincount(ki, weights=np.where(pre, r, 0), minlength=len(ks))
Np_ = np.bincount(ki, weights=pre.astype(float), minlength=len(ks))
Sf = np.bincount(ki, weights=np.where(later, r, 0), minlength=len(ks))
Nf = np.bincount(ki, weights=later.astype(float), minlength=len(ks))
win = {}
for (lo, hi), lb in zip(BINS, LABS):
    m = (Np_ >= lo) & (Np_ <= hi) & (Nf >= 1)
    xx, rr = Sp[m] / Np_[m], Sf[m] / Nf[m]
    win[lb] = {"cells": int(m.sum()), "slope_weighted_by_later_games": slope(xx, rr, Nf[m]),
               "median_later_games": float(np.median(Nf[m]))}
res["window"] = win
log("window " + json.dumps({k: round(v["slope_weighted_by_later_games"], 3) for k, v in win.items()}))

# recency
recy = {}
for k in (25, 50, 100):
    nk = lastk(np.ones(N), k)
    xk = np.where(nk > 0, lastk(r, k) / np.maximum(nk, 1), np.nan)
    recy[f"last {k} games"] = by_bin(xk[qrows], r[qrows], qn)
for hl in (180, 365, 730):
    lam = np.log(2) / hl
    d0 = day.min()
    wgt = np.exp(lam * (day - d0))
    S = lagsum(r * wgt)
    W = lagsum(wgt)
    xe = np.where(W > 0, S / np.maximum(W, 1e-300), np.nan)
    recy[f"half-life {hl} days"] = by_bin(xe[qrows], r[qrows], qn)
res["recency"] = recy
for k, v in recy.items():
    log(f"recency {k} " + json.dumps({b: round(x_["slope"], 3) for b, x_ in v.items()}))

# simulation
try:
    from numba import njit

    @njit(cache=False)
    def ou_path(kc, dc, eps, rho_day, tau):
        th = np.empty(len(kc))
        for i in range(len(kc)):
            if i == 0 or kc[i] != kc[i - 1]:
                th[i] = tau * eps[i]
            else:
                a = rho_day ** (dc[i] - dc[i - 1])
                th[i] = a * th[i - 1] + tau * np.sqrt(max(1 - a * a, 0.0)) * eps[i]
        return th
except Exception:  # pragma: no cover
    ou_path = None

sims = {}
for scen, tau, drift in (("tau 3pp, persistent", 0.03, False), ("tau 4pp, persistent", 0.04, False),
                         ("tau 5pp, persistent", 0.05, False), ("tau 4pp, drift (0.74 at 1 year)", 0.04, True)):
    out = []
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        if drift:
            if ou_path is None:
                continue
            th_c = ou_path(kc, dc, rng.standard_normal(N), 0.74 ** (1 / 365.0), tau)
            th = th_c[inv_oc]
        else:
            uk, ui = np.unique(key, return_inverse=True)
            th = (tau * rng.standard_normal(len(uk)))[ui]
        ys = (rng.random(N) < np.clip(p0 + th, 0.02, 0.98)).astype(float)
        rs = ys - p0
        xs = np.where(nh > 0, lagsum(rs) / np.maximum(nh, 1), np.nan)
        out.append(by_bin(xs[qrows], rs[qrows], qn))
    if out:
        sims[scen] = {lb: {"slope": float(np.mean([o_[lb]["slope"] for o_ in out])),
                           "reliability_of_record": float(np.mean([o_[lb]["reliability_of_record"] for o_ in out])),
                           "record_sd_pp": float(np.mean([o_[lb]["record_sd_pp"] for o_ in out])),
                           "tau_mom_pp": float(100 * np.sqrt(max(np.mean([(o_[lb]["record_sd_pp"] / 100) ** 2 for o_ in out])
                                                                  - np.mean(0.245 / qn[(qn >= lo) & (qn <= hi)]), 0)))}
                      for (lo, hi), lb in zip(BINS, LABS)}
        log(f"sim {scen} " + json.dumps({k: round(v["slope"], 3) for k, v in sims[scen].items()}))
res["simulate"] = sims
json.dump(res, open(os.path.join(C.CACHE, "p2f_carry.json"), "w"), indent=1)
log("done")
