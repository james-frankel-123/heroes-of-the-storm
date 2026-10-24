"""
P3 play-by-play study (results/P3_PLAY_BY_PLAY.md): do play-by-play features
from a player's PAST replays predict the player's FUTURE results beyond the
scoreboard-based player model?

Inputs (gitignored caches, player keys): cache/pbp/targets.npz and
cache/pbp/fullhist.npz (p3_pbp_targets.py), cache/pbp/rows.npz
(p3_pbp_features.py), cache/hr/state_0928.npz (p3_hr_state.py).

Design
- Targets: every non-sealed 2.57 Storm League game with ten DB rows. Outcome:
  team 0 wins. Every history feature for a game on UTC day D uses rows from
  days < D only (asserted by a rewrite test in `leakcheck`).
- Three target windows, fixed before any result was seen:
    fit      2026-10-01 .. 10-10   (coefficients for the tuning evaluation)
    tune     2026-10-11 .. 10-15   (every selection: features, families, L2)
    holdout  2026-10-16 .. end     (reported once; coefficients refit on
                                    fit + tune, nothing chosen here)
  Normalization statistics (population means and sds, ability mixes) come
  from rows before 2026-10-16.
- Baseline columns, all team differences except draft WP:
    wp       logit population draft WP (d2c_cumprev ensemble); 0 when the
             map or a hero is unknown to it, with a flag x hero WR
    hero_wr  the hero's 2.57 win rate on earlier days (logit, shrunk)
    model    the main model (full_ms posterior + experience offset) frozen
             at 2026-09-28 for the player on the hero
    winrec   the player's 2.57 record on earlier days (residual vs hero WR)
    fh_*     full-history scoreboard direct channel as of 2026-09-28: 15
             a_shrink metric means on the hero and 15 over all heroes
    sb_*     2.57 DB scoreboard on earlier days: the same 15 metrics, on
             the hero and over all heroes
  Arms: A0 = wp + hero_wr; A1 = A0 + model + winrec (main model without the
  2.57 scoreboard channel); A2 = A1 + fh + sb (the chosen model's closest
  runnable version: main model + direct scoreboard channel); A3 = A2 + DB
  scoreboard extras (CC, stun, root, silence, teamfight, minion, creep, time
  dead, team shares, consistency, talent choices).
- Replay features: shrunk as-of player means of normalized per-game values
  from replay games on earlier days (k = (1 - r) / r pseudo-games, r the
  player-level per-game reliability from pre-holdout rows).

Stages (from training/): python3 personalization/p3_pbp_eval.py <stage>
  prep        normalize, as-of features -> cache/pbp/design.npz
  reliability split-half reliability and scoreboard redundancy
  tune        selection on the tune window -> results/pbp/tune.json
  holdout     final numbers -> results/pbp/holdout.json
  scaling     lift vs replays per player (last-k history truncation)
  coverage    share of lobbies with enough past replays
  leakcheck   rewriting rows on or after a day leaves earlier features unchanged
"""
import datetime
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import p3_pbp_features as PF  # noqa: E402
import p3_pbp_targets as PT  # noqa: E402
from p3_keys import player_keys  # noqa: E402

CACHE = os.path.join(HERE, "cache", "pbp")
RES = os.path.join(HERE, "results", "pbp")
DESIGN = os.path.join(CACHE, "design.npz")
STATE = os.path.join(HERE, "cache", "hr", "state_0928.npz")
DAY = 86400.0


def dnum(s):
    return int(datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc).timestamp() // DAY)


FIT0, TUNE0, HOLD0 = dnum("2026-10-01"), dnum("2026-10-11"), dnum("2026-10-16")
SEALED = PT.sealed_ids

RATE_PBP = {"d_isolated", "d_unpaid", "d_preobj", "d_late", "outnum_exposure", "x_globes", "x_lasthits",
            "x_soak_early", "x_soak_late", "x_struct_kills", "abil_pm", "rclick_pm", "p_move_pm",
            "apm_cmd", "apm_retarget", "apm_key", "apm_camera"}
SBX_DB = ["stunning_enemies", "rooting_enemies", "silencing_enemies", "teamfight_hero_damage",
          "teamfight_damage_taken", "teamfight_healing", "minion_damage", "creep_damage", "time_spent_dead"]
SHARE = ["hero_damage", "damage_taken", "healing", "siege_damage", "experience_contribution", "takedowns"]
CONS = ["hero_damage", "damage_taken", "deaths", "experience_contribution", "kills", "assists"]


# ---------------------------------------------------------------- helpers
def normalize(X, hero, mp, win, asinh_cols, fit_mask, k0=100.0):
    """asinh (for count/rate columns), z within hero x map x outcome shrunk to hero x outcome
    (k0 pseudo-games), clip +-5, mask heroes where a column is uninformative. Statistics from
    fit_mask rows only (as p3_hr_tl_analyze.normalize)."""
    n, F = X.shape
    uh, hi = np.unique(hero, return_inverse=True)
    um, mi = np.unique(mp, return_inverse=True)
    y = win.astype(np.int64)
    Z = np.full_like(X, np.nan, dtype=np.float64)
    for k in range(F):
        v = X[:, k].astype(np.float64).copy()
        if k in asinh_cols:
            pos = v[fit_mask & (v > 0) & np.isfinite(v)]
            s = np.median(pos) if len(pos) else 1.0
            v = np.arcsinh(v / s)
        ok = np.isfinite(v) & fit_mask
        c_h = hi * 2 + y
        nh = np.bincount(c_h[ok], minlength=len(uh) * 2)
        sh = np.bincount(c_h[ok], weights=v[ok], minlength=len(uh) * 2)
        qh = np.bincount(c_h[ok], weights=v[ok] ** 2, minlength=len(uh) * 2)
        mh = sh / np.maximum(nh, 1)
        vh = np.maximum(qh / np.maximum(nh, 1) - mh ** 2, 0)
        c_m = (hi * len(um) + mi) * 2 + y
        nm = np.bincount(c_m[ok], minlength=len(uh) * len(um) * 2)
        sm = np.bincount(c_m[ok], weights=v[ok], minlength=len(uh) * len(um) * 2)
        qm = np.bincount(c_m[ok], weights=v[ok] ** 2, minlength=len(uh) * len(um) * 2)
        par = (np.arange(len(nm)) // (2 * len(um))) * 2 + np.arange(len(nm)) % 2
        mm = (sm + k0 * mh[par]) / (nm + k0)
        e2 = (qm + k0 * (vh[par] + mh[par] ** 2)) / (nm + k0)
        sd = np.sqrt(np.maximum(e2 - mm ** 2, 0))
        with np.errstate(invalid="ignore", divide="ignore"):
            zz = (v - mm[c_m]) / np.where(sd[c_m] > 0.02, sd[c_m], np.nan)
        nz = np.bincount(hi[ok], weights=(v[ok] != 0), minlength=len(uh)) / np.maximum(
            np.bincount(hi[ok], minlength=len(uh)), 1)
        bad = (nz < 0.05) | (np.sqrt(vh.reshape(-1, 2).mean(1)) <= 0.02) | (np.bincount(hi[ok], minlength=len(uh)) < 30)
        zz[bad[hi]] = np.nan
        Z[:, k] = np.clip(zz, -5, 5)
    return Z


def asof(hk, hday, Zh, qk, qday, k, last=None):
    """Shrunk means sum / (n + k) over history rows of key hk on days < qday, per query.
    last: only the most recent `last` such rows (by day, then input order).
    Returns (means, rows used, per-column counts, (sums, counts))."""
    o = np.lexsort((hday, hk))
    hk, hday, Zh = hk[o], hday[o], Zh[o]
    F = Zh.shape[1]
    ok = np.isfinite(Zh)
    cs = np.vstack([np.zeros((1, F)), np.cumsum(np.where(ok, Zh, 0.0), 0)])
    cn = np.vstack([np.zeros((1, F)), np.cumsum(ok, 0)])
    uk, first = np.unique(hk, return_index=True)
    rank = np.cumsum(np.r_[True, hk[1:] != hk[:-1]]) - 1
    big = rank.astype(np.float64) * 1e6 + hday
    pos = np.minimum(np.searchsorted(uk, qk), len(uk) - 1)
    hit = uk[pos] == qk
    start = np.where(hit, first[pos], 0)
    end = np.where(hit, np.searchsorted(big, pos.astype(np.float64) * 1e6 + qday, "left"), 0)
    if last is not None:
        start = np.maximum(start, end - last)
    end = np.maximum(end, start)
    S = cs[end] - cs[start]
    N = cn[end] - cn[start]
    return S / (N + k[None, :]), (end - start), N, (S, N)


def fit_l2(X, y, lam, off=None, iters=60):
    n, p = X.shape
    Xb = np.column_stack([np.ones(n), X])
    w = np.zeros(p + 1)
    off = np.zeros(n) if off is None else off
    R = lam * np.eye(p + 1)
    R[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(off + Xb @ w, -30, 30)
        pr = 1 / (1 + np.exp(-eta))
        g = Xb.T @ (y - pr) - R @ w
        Hm = (Xb * (pr * (1 - pr))[:, None]).T @ Xb + R
        st = np.linalg.solve(Hm, g)
        w += st
        if np.abs(st).max() < 1e-9:
            break
    return w


def predict(w, X, off=None):
    eta = (0 if off is None else off) + np.column_stack([np.ones(len(X)), X]) @ w
    return 1 / (1 + np.exp(-np.clip(eta, -30, 30)))


def ll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def split_half(x, key, order, pk, boot=0, seed=0):
    """Per-game reliability: odd vs even games in play order within key (as p3_rcf_data.reliability)."""
    okx = np.isfinite(x)
    idx = np.flatnonzero(okx)
    kk = key[idx]
    o = np.lexsort((order[idx], kk))
    idx, kk = idx[o], kk[o]
    brk = np.flatnonzero(np.r_[True, kk[1:] != kk[:-1]])
    rank = np.arange(len(idx)) - np.repeat(brk, np.diff(np.r_[brk, len(idx)]))
    half = rank % 2
    xs = x[idx]
    u, inv = np.unique(kk, return_inverse=True)
    s0 = np.bincount(inv, weights=np.where(half == 0, xs, 0), minlength=len(u))
    s1 = np.bincount(inv, weights=np.where(half == 1, xs, 0), minlength=len(u))
    n0 = np.bincount(inv, weights=(half == 0), minlength=len(u))
    n1 = np.bincount(inv, weights=(half == 1), minlength=len(u))
    g = (n0 >= 1) & (n1 >= 1)
    m0, m1 = s0[g] / n0[g], s1[g] / n1[g]
    w = np.minimum(n0[g], n1[g])
    var = xs.var()

    if g.sum() < 20:
        return {"r": float("nan"), "cells": int(g.sum())}

    def est(ww):
        mu = np.average(np.r_[m0, m1], weights=np.r_[ww, ww])
        return float(np.average((m0 - mu) * (m1 - mu), weights=ww) / var)
    out = {"r": est(w), "cells": int(g.sum())}
    if boot:
        cp = np.zeros(len(u), np.int64)
        cp[inv] = pk[idx]
        cp = cp[g]
        up, pinv = np.unique(cp, return_inverse=True)
        rng = np.random.RandomState(seed)
        bs = [est(w * rng.multinomial(len(up), np.ones(len(up)) / len(up))[pinv]) for _ in range(boot)]
        out["ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


# ---------------------------------------------------------------- prep
def load_rows():
    z = dict(np.load(os.path.join(CACHE, "rows.npz")))
    assert int(z["sealed_overlap"]) == 0
    assert not np.isin(z["replay_id"], np.fromiter(SEALED(), np.int64)).any()
    return z


def prep():
    t0 = time.time()
    t = dict(np.load(PT.TARGETS))
    z = load_rows()
    n = len(t["win"])
    tday = np.floor(t["unix"] / DAY).astype(np.int64)
    tk = t["pkey"].astype(np.int64)
    # replay rows: key from the matching DB slot when there is one
    m = {(int(r), int(b)): i for i, (r, b) in enumerate(zip(t["replay_id"], t["toon_id"]))}
    ti = np.array([m.get((int(r), int(b)), -1) for r, b in zip(z["replay_id"], z["toon_id"])])
    rk = player_keys(np.maximum(z["region"], 0), z["toon_id"], label="pbp rows")
    rk = np.where(ti >= 0, tk[np.maximum(ti, 0)], rk)
    rday = np.floor(z["unix"] / DAY).astype(np.int64)
    print(f"replay rows {len(rk):,}, matched to DB slots {np.mean(ti >= 0):.3f}", flush=True)
    feats = list(PF.ALLF)
    Xr = np.column_stack([z[f] for f in feats])
    emb_p = os.path.join(CACHE, "emb.npz")
    if os.path.exists(emb_p):
        Em = np.load(emb_p)["E"]
        assert len(Em) == len(rk)
        feats += [f"emb_{j}" for j in range(Em.shape[1])]
        Xr = np.column_stack([Xr, Em])
        print(f"embedding: {Em.shape[1]} dims", flush=True)
    np.savez(os.path.join(CACHE, "rows_keys.npz"), pkey=rk, day=rday, hold0=np.int64(HOLD0))
    rate = {j for j, f in enumerate(feats) if f in RATE_PBP}
    # rates per minute are already per minute in the feature builder; asinh for heavy tails
    Zr = normalize(Xr, z["hero"], z["map"], z["win"], rate, rday < HOLD0)
    # DB scoreboard (15 model metrics + extras), per minute, asinh, z
    sb15 = PT.SB15
    Xs = np.column_stack([t["sb_" + f] / t["minutes"] for f in sb15 + SBX_DB])
    Zs = normalize(Xs, t["hero"], t["map"], t["win"], set(range(Xs.shape[1])), tday < HOLD0)
    # team shares (DB)
    gi = np.unique(t["replay_id"], return_inverse=True)[1]
    gt = gi * 2 + t["team"]
    Xsh = []
    for f in SHARE:
        v = t["sb_" + f] if f != "healing" else np.nan_to_num(t["sb_healing"]) + np.nan_to_num(t["sb_protection_allies"])
        tot = np.bincount(gt, weights=np.nan_to_num(v), minlength=gt.max() + 1)
        Xsh.append(np.where(tot[gt] > 0, np.nan_to_num(v) / np.maximum(tot[gt], 1e-9), np.nan))
    Xsh = np.column_stack(Xsh)
    Zsh = normalize(Xsh, t["hero"], t["map"], t["win"], set(), tday < HOLD0)
    # talents: as-of popularity (-log share) and as-of payoff (mean win residual) of the chosen talent
    tal_dev, tal_pay = talent_feats(t, tday)
    np.savez(DESIGN, tday=tday, tk=tk, rk=rk, rday=rday, ti=ti, Zr=Zr.astype(np.float32),
             Zs=Zs.astype(np.float32), Zsh=Zsh.astype(np.float32), tal_dev=tal_dev, tal_pay=tal_pay,
             feats=np.array(feats))
    print(f"prep done {time.time() - t0:.0f}s", flush=True)


def talent_feats(t, tday):
    """Per slot: mean over tiers 4-20 of (a) -log as-of pick share of the chosen talent on the hero,
    (b) as-of payoff of the chosen talent: mean (win - hero WR) of earlier-day picks, shrunk (k=200)."""
    n = len(t["win"])
    hero = t["hero"]
    dev = np.full((n, 6), np.nan)
    pay = np.full((n, 6), np.nan)
    days = np.unique(tday)
    # hero WR as of each day
    uh, hi = np.unique(hero, return_inverse=True)
    for j, tier in enumerate(PT.TIERS[1:]):
        tal = t["tal_" + tier]
        key = np.char.add(np.char.add(hero.astype(str), "|"), tal.astype(str))
        uk, ki = np.unique(key, return_inverse=True)
        has = tal != ""
        cnt = np.zeros(len(uk))
        wsum = np.zeros(len(uk))
        hcnt = np.zeros(len(uh))
        hw = np.zeros(len(uh))
        for dd in days:
            cur = tday == dd
            c = cur & has
            share = cnt[ki[c]] / np.maximum(hcnt[hi[c]], 1)
            ok = hcnt[hi[c]] >= 50
            dev[np.flatnonzero(c)[ok], j] = -np.log((cnt[ki[c]][ok] + 1) / (hcnt[hi[c]][ok] + 10))
            hwr = (hw[hi[c]] + 100) / (hcnt[hi[c]] + 200)
            res_sum = wsum[ki[c]] - cnt[ki[c]] * hwr
            pay[np.flatnonzero(c)[ok], j] = (res_sum / (cnt[ki[c]] + 200))[ok]
            _ = share
            cnt += np.bincount(ki[c], minlength=len(uk))
            wsum += np.bincount(ki[c], weights=t["win"][c], minlength=len(uk))
            hcnt += np.bincount(hi[c], minlength=len(uh))
            hw += np.bincount(hi[c], weights=t["win"][c], minlength=len(uh))
    with np.errstate(invalid="ignore"):
        return np.nanmean(dev, 1), np.nanmean(pay, 1)


# ---------------------------------------------------------------- columns
_COLS = {}


def columns(rel=None, last=None, hero_level=False, hist_mask=None):
    """Game-level design: dict name -> per-game column (team 0 minus team 1), plus game arrays."""
    t = dict(np.load(PT.TARGETS))
    D = dict(np.load(DESIGN))
    n = len(t["win"])
    tday, tk, rk, rday = D["tday"], D["tk"], D["rk"], D["rday"]
    feats = [str(f) for f in D["feats"]]
    ug, gi = np.unique(t["replay_id"], return_inverse=True)
    G = len(ug)
    gday = np.zeros(G, np.int64)
    gday[gi] = tday
    y0 = np.zeros(G)
    m0 = t["team"] == 0
    y0[gi[m0]] = t["win"][m0]
    sgn = np.where(t["team"] == 0, 1.0, -1.0)

    def tdiff(v):
        return np.bincount(gi, weights=sgn * np.nan_to_num(v), minlength=G)
    C = {}
    wp = t["wp_pop"]
    wp_ok = np.isfinite(wp)
    wl = np.where(wp_ok, np.log(np.clip(wp, 1e-6, 1) / np.clip(1 - wp, 1e-6, 1)), 0.0)
    gw = np.zeros(G)
    gw[gi[m0]] = wl[m0]
    gmiss = np.zeros(G)
    gmiss[gi[m0]] = ~wp_ok[m0]
    C["wp"] = gw
    # hero 2.57 WR as of earlier days
    uh, hi = np.unique(t["hero"], return_inverse=True)
    hw = np.zeros(n)
    for dd in np.unique(tday):
        prev = tday < dd
        ws = np.bincount(hi[prev], weights=t["win"][prev], minlength=len(uh))
        ns = np.bincount(hi[prev], minlength=len(uh))
        cur = tday == dd
        p = (ws + 100) / (ns + 200)
        hw[cur] = np.log(p / (1 - p))[hi[cur]]
    C["hero_wr"] = tdiff(hw)
    C["hero_wr_x_wpmiss"] = C["hero_wr"] * gmiss
    C["wpmiss"] = gmiss
    # main model state at 09-28
    s = np.load(STATE, allow_pickle=True)
    sk = player_keys(s["region"], s["blizz_id"])
    hn = [str(h) for h in s["hero_names"]]
    hix_ = {h: i for i, h in enumerate(hn)}
    hix = np.array([hix_.get(h, -1) for h in t["hero"]])
    os_ = np.argsort(sk)
    j = np.minimum(np.searchsorted(sk[os_], tk), len(sk) - 1)
    hit = (sk[os_][j] == tk) & (hix >= 0)
    est = np.zeros(n)
    rr = os_[j[hit]]
    est[hit] = s["mean_all"][rr, hix[hit]] + s["offset_all"][rr, hix[hit]]
    C["model"] = tdiff(est)
    # 2.57 win record
    res = t["win"] - 1 / (1 + np.exp(-hw))
    wr, nwr, _, _ = asof(tk, tday, res[:, None], tk, tday, np.array([74.0]))
    C["winrec"] = tdiff(wr[:, 0])
    # full-history direct channel (a_shrink z) as of 09-28
    fh = np.load(PT.FULLHIST)
    ck = fh["cell_key"]
    chero = fh["cell_hero"].astype(np.int64)
    kfh = 5.0
    cell = ck * 128 + chero
    oc = np.argsort(cell)
    q = tk * 128 + np.maximum(hix, 0)
    jj = np.minimum(np.searchsorted(cell[oc], q), len(oc) - 1)
    hh = (cell[oc][jj] == q) & (hix >= 0)
    Sfh, Nfh = fh["S"], fh["N"]
    up, pinv = np.unique(ck, return_inverse=True)
    SP = np.zeros((len(up), Sfh.shape[1]))
    NP = np.zeros((len(up), Sfh.shape[1]))
    np.add.at(SP, pinv, Sfh)
    np.add.at(NP, pinv, Nfh)
    jp = np.minimum(np.searchsorted(up, tk), len(up) - 1)
    hp = up[jp] == tk
    for m_, met in enumerate([str(x) for x in fh["metrics"]]):
        v = np.zeros(n)
        r_ = oc[jj[hh]]
        v[hh] = Sfh[r_, m_] / (Nfh[r_, m_] + kfh)
        C["fh_h_" + met] = tdiff(v)
        v = np.zeros(n)
        v[hp] = SP[jp[hp], m_] / (NP[jp[hp], m_] + kfh)
        C["fh_p_" + met] = tdiff(v)
    # 2.57 DB scoreboard as of earlier days (player and player x hero), extras, shares, consistency, talents
    Zs, Zsh = D["Zs"].astype(np.float64), D["Zsh"].astype(np.float64)
    ksb = np.full(Zs.shape[1], 4.0)
    Mp, _, _, (Sp_, Np_) = asof(tk, tday, Zs, tk, tday, ksb)
    hkey = tk * 128 + hi
    Mh, _, _, _ = asof(hkey, tday, Zs, hkey, tday, ksb)
    names = PT.SB15 + SBX_DB
    for m_, met in enumerate(names):
        pre = "sb_" if met in PT.SB15 else "sbx_"
        C[pre + "p_" + met] = tdiff(Mp[:, m_])
        C[pre + "h_" + met] = tdiff(Mh[:, m_])
    Msh, _, _, _ = asof(tk, tday, Zsh, tk, tday, np.full(Zsh.shape[1], 4.0))
    for m_, met in enumerate(SHARE):
        C["share_" + met] = tdiff(Msh[:, m_])
    ci = [names.index(c) for c in CONS]
    Z2 = Zs[:, ci] ** 2
    M2, n2, N2, _ = asof(tk, tday, Z2, tk, tday, np.zeros(len(ci)))
    M1, _, N1, _ = asof(tk, tday, Zs[:, ci], tk, tday, np.zeros(len(ci)))
    with np.errstate(invalid="ignore"):
        var = np.where(N2 >= 3, np.maximum(M2 - M1 ** 2, 0), np.nan)
    sdv = np.sqrt(var) - 0.9
    sdv = np.where(np.isfinite(sdv), sdv * N2 / (N2 + 5), 0)
    for m_, met in enumerate(CONS):
        C["cons_" + met] = tdiff(sdv[:, m_])
    for nm, v in (("tal_dev", D["tal_dev"]), ("tal_pay", D["tal_pay"])):
        vv = np.where(np.isfinite(v), v, np.nan)
        mu = np.nanmean(vv[tday < HOLD0])
        sd = np.nanstd(vv[tday < HOLD0])
        Mt, _, _, _ = asof(tk, tday, ((vv - mu) / sd)[:, None], tk, tday, np.array([5.0]))
        C[nm] = tdiff(Mt[:, 0])
    # replay features
    Zr = D["Zr"].astype(np.float64)
    hm = np.ones(len(rk), bool) if hist_mask is None else hist_mask
    if rel is None:
        kr = np.full(Zr.shape[1], 5.0)
    else:
        kr = np.array([np.clip((1 - rel[f]) / max(rel[f], 1e-3), 0.2, 200) for f in feats])
    if hero_level:
        rows = dict(np.load(os.path.join(CACHE, "rows.npz")))
        rhi = np.searchsorted(uh, rows["hero"])
        rhi = np.where(uh[np.minimum(rhi, len(uh) - 1)] == rows["hero"], rhi, 127)
        Mr, nr, _, _ = asof((rk * 128 + rhi)[hm], rday[hm], Zr[hm], tk * 128 + hi, tday, kr, last=last)
    else:
        Mr, nr, _, _ = asof(rk[hm], rday[hm], Zr[hm], tk, tday, kr, last=last)
    for m_, f in enumerate(feats):
        C["r_" + f] = tdiff(Mr[:, m_])
    # the replay games' own scoreboard is not used: the DB has every game's scoreboard
    meta = {"G": G, "gi": gi, "gday": gday, "y0": y0, "sgn": sgn, "n_hist": nr, "tday": tday, "tk": tk,
            "replay_id": t["replay_id"], "ug": ug, "has_state": hit, "Mr": Mr, "feats": feats, "wp_ok": wp_ok,
            "hw": hw}
    return C, meta


def groups(feats):
    base_a0 = ["wp", "hero_wr", "hero_wr_x_wpmiss", "wpmiss"]
    a1 = base_a0 + ["model", "winrec"]
    fh = [f"fh_h_{m}" for m in PT.SB15] + [f"fh_p_{m}" for m in PT.SB15]
    sb = [f"sb_h_{m}" for m in PT.SB15] + [f"sb_p_{m}" for m in PT.SB15]
    a2 = a1 + fh + sb
    ext = ([f"sbx_p_{m}" for m in SBX_DB] + [f"sbx_h_{m}" for m in SBX_DB] + [f"share_{m}" for m in SHARE]
           + [f"cons_{m}" for m in CONS] + ["tal_dev", "tal_pay"])
    a3 = a2 + ext
    fam = {k: ["r_" + f for f in v] for k, v in PF.FAMILIES.items()}
    if any(f.startswith("emb_") for f in feats):
        fam["emb"] = ["r_" + f for f in feats if f.startswith("emb_")]
    return {"A0": base_a0, "A1": a1, "A2": a2, "A3": a3, "ext": ext,
            "ext_sbx": [f"sbx_p_{m}" for m in SBX_DB] + [f"sbx_h_{m}" for m in SBX_DB],
            "ext_share": [f"share_{m}" for m in SHARE], "ext_cons": [f"cons_{m}" for m in CONS],
            "ext_tal": ["tal_dev", "tal_pay"], "fam": fam,
            "replay_all": [c for k, v in fam.items() if k != "emb" for c in v]}


def design(C, cols):
    return np.column_stack([C[c] for c in cols])


def fit_eval(C, cols, tr, te, y0, lam):
    X = design(C, cols)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    X = (X - mu) / sd
    w = fit_l2(X[tr], y0[tr], lam)
    return predict(w, X[te]), w, (mu, sd)


def cv_eval(C, cols, mask, y0, lam, fold, nf=5):
    X = design(C, cols)
    p = np.full(len(y0), np.nan)
    for k in range(nf):
        tr = mask & (fold != k)
        te = mask & (fold == k)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        w = fit_l2((X[tr] - mu) / sd, y0[tr], lam)
        p[te] = predict(w, (X[te] - mu) / sd)
    return p


def paired(l_new, l_base, days, B=200, seed=0):
    d = l_base - l_new
    out = {"gain_x1000": 1000 * float(d.mean()), "se_game_x1000": 1000 * float(d.std(ddof=1) / np.sqrt(len(d)))}
    ud, di = np.unique(days, return_inverse=True)
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(B):
        w_ = rng.multinomial(len(ud), np.ones(len(ud)) / len(ud))[di]
        if w_.sum():
            bs.append(np.average(d, weights=w_))
    out["se_day_x1000"] = 1000 * float(np.std(bs))
    out["games"] = int(len(d))
    return out


def rel_dict():
    p = os.path.join(RES, "reliability.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))["features"]
    return {f: max(r[f]["player"]["r"], 1e-3) for f in r}


# ---------------------------------------------------------------- stages
def reliability(boot=200):
    t0 = time.time()
    D = dict(np.load(DESIGN))
    z = load_rows()
    feats = [str(f) for f in D["feats"]]
    Zr = D["Zr"].astype(np.float64)
    pre = D["rday"] < HOLD0
    rk = D["rk"]
    uh, hi = np.unique(z["hero"], return_inverse=True)
    out = {"rows_pre_holdout": int(pre.sum()), "features": {}}
    order = z["unix"]
    for j, f in enumerate(feats):
        x = np.where(pre, Zr[:, j], np.nan)
        out["features"][f] = {
            "coverage": float(np.isfinite(Zr[pre, j]).mean()),
            "player_hero": split_half(x, rk * 128 + hi, order, rk, boot=boot),
            "player": split_half(x, rk, order, rk, boot=0)}
        print(f"  {f:20s} ph {out['features'][f]['player_hero']['r']:.3f} p {out['features'][f]['player']['r']:.3f}",
              flush=True)
    # win residual for scale
    out["win_minus_hero_rate"] = split_half(np.where(pre, z["win"] - np.array(
        [np.mean(z["win"][pre & (z["hero"] == h)]) for h in z["hero"]]), np.nan), rk * 128 + hi, order, rk)
    # redundancy: DB scoreboard means (even games) predict the feature's odd-game mean
    out["redundancy"] = redundancy(D, z, feats, pre)
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "reliability.json"), "w"), indent=1)
    print(f"reliability done {time.time() - t0:.0f}s", flush=True)


def redundancy(D, z, feats, pre, min_games=6):
    """Share of each feature's stable player-level part predicted by the player's scoreboard
    profile (15 metrics + extras from the replay games' DB rows), 5-fold ridge across players,
    even games -> odd games."""
    t = dict(np.load(PT.TARGETS))
    ti = D["ti"]
    Zs = D["Zs"].astype(np.float64)
    Zr = D["Zr"].astype(np.float64)
    ok = pre & (ti >= 0)
    rk = D["rk"][ok]
    order = z["unix"][ok]
    o = np.lexsort((order, rk))
    idx = np.flatnonzero(ok)[o]
    kk = rk[o]
    brk = np.flatnonzero(np.r_[True, kk[1:] != kk[:-1]])
    rank = np.arange(len(idx)) - np.repeat(brk, np.diff(np.r_[brk, len(idx)]))
    cnt = np.diff(np.r_[brk, len(idx)])
    keep_p = np.repeat(cnt >= min_games, cnt)
    idx, kk, rank = idx[keep_p], kk[keep_p], rank[keep_p]
    u, inv = np.unique(kk, return_inverse=True)

    def mean_by(X, sel):
        okm = np.isfinite(X) & sel[:, None]
        S = np.zeros((len(u), X.shape[1]))
        N = np.zeros((len(u), X.shape[1]))
        np.add.at(S, inv, np.where(okm, X, 0))
        np.add.at(N, inv, okm)
        return S / np.maximum(N, 1), N
    even = rank % 2 == 0
    SBm, _ = mean_by(Zs[ti[idx]], even)
    FEm, FN = mean_by(Zr[idx], ~even)
    FEe, _ = mean_by(Zr[idx], even)
    rng = np.random.RandomState(0)
    fold = rng.randint(0, 5, len(u))
    out = {"players": int(len(u))}
    for j, f in enumerate(feats):
        y = FEm[:, j]
        g = FN[:, j] >= 2
        if g.sum() < 200:
            continue
        X = np.column_stack([SBm[g], np.ones(g.sum())])
        yy = y[g]
        pred = np.zeros(g.sum())
        for k in range(5):
            tr, te = fold[g] != k, fold[g] == k
            A = X[tr].T @ X[tr] + 10 * np.eye(X.shape[1])
            pred[te] = X[te] @ np.linalg.solve(A, X[tr].T @ yy[tr])
        r2 = 1 - np.mean((yy - pred) ** 2) / np.var(yy)
        # reliability of the odd-half mean: correlation of even and odd means
        rel = np.corrcoef(FEe[g, j], yy)[0, 1]
        out[f] = {"r2": float(r2), "half_rel": float(rel), "share": float(r2 / rel) if rel > 0.02 else None}
    return out


def tune(lam_grid=(3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)):
    t0 = time.time()
    rel = rel_dict()
    C, M = columns(rel)
    Gp = groups(M["feats"])
    gday, y0 = M["gday"], M["y0"]
    tr = (gday >= FIT0) & (gday < TUNE0)
    te = (gday >= TUNE0) & (gday < HOLD0)
    has = np.bincount(M["gi"], weights=(M["n_hist"] > 0), minlength=M["G"])
    out = {"fit_games": int(tr.sum()), "tune_games": int(te.sum()),
           "tune_games_3plus_hist": int((te & (has >= 3)).sum())}
    # L2 for the big arms
    lls = {}
    for lam in lam_grid:
        p, _, _ = fit_eval(C, Gp["A3"] + Gp["replay_all"], tr, te, y0, lam)
        lls[lam] = float(ll(p, y0[te]).mean())
    lam = min(lls, key=lls.get)
    out["lam_grid"] = lls
    out["lam"] = lam
    L = {}

    def run(name, cols):
        p, _, _ = fit_eval(C, cols, tr, te, y0, lam)
        L[name] = ll(p, y0[te])
    for a in ("A0", "A1", "A2", "A3"):
        run(a, Gp[a])
    run("A2+ext_sbx", Gp["A2"] + Gp["ext_sbx"])
    run("A2+ext_share", Gp["A2"] + Gp["ext_share"])
    run("A2+ext_cons", Gp["A2"] + Gp["ext_cons"])
    run("A2+ext_tal", Gp["A2"] + Gp["ext_tal"])
    for base in ("A1", "A2", "A3"):
        run(f"{base}+replay", Gp[base] + Gp["replay_all"])
        for fam, cols in Gp["fam"].items():
            run(f"{base}+{fam}", Gp[base] + cols)
    for c in Gp["replay_all"]:
        run(f"A2+{c}", Gp["A2"] + [c])
    days = gday[te]
    sub = has[te] >= 3
    res = {}
    for k in L:
        if k in ("A0",):
            continue
        base = {"A1": "A0", "A2": "A1", "A3": "A2"}.get(k) or k.split("+")[0]
        r = paired(L[k], L[base], days)
        r_sub = paired(L[k][sub], L[base][sub], days[sub])
        res[f"{k} vs {base}"] = {"all": r, "3plus_hist": r_sub}
    out["results"] = res
    # selection rule (frozen before the full data were in): single replay features whose tune gain over
    # A2 exceeds two game-level SEs; families and DB extras whose gain exceeds one SE
    single = {c: res[f"A2+{c} vs A2"]["all"] for c in Gp["replay_all"]}
    sel = [c for c, r in single.items() if r["gain_x1000"] > 2 * r["se_game_x1000"]]
    fam_sel = [f for f in Gp["fam"] if res[f"A2+{f} vs A2"]["all"]["gain_x1000"] > res[f"A2+{f} vs A2"]["all"]["se_game_x1000"]]
    ext_sel = [e for e in ("ext_sbx", "ext_share", "ext_cons", "ext_tal")
               if res[f"A2+{e} vs A2"]["all"]["gain_x1000"] > res[f"A2+{e} vs A2"]["all"]["se_game_x1000"]]
    out["selected_features"] = sel
    out["selected_families"] = fam_sel
    out["selected_ext"] = ext_sel
    json.dump(out, open(os.path.join(RES, "tune.json"), "w"), indent=1)
    print(json.dumps({k: (round(v["all"]["gain_x1000"], 2), round(v["all"]["se_game_x1000"], 2),
                          round(v["3plus_hist"]["gain_x1000"], 2)) for k, v in res.items()}, indent=0), flush=True)
    print("lam", lam, "selected", sel, fam_sel, ext_sel, f"{time.time() - t0:.0f}s", flush=True)


def holdout(B=200):
    t0 = time.time()
    rel = rel_dict()
    tn = json.load(open(os.path.join(RES, "tune.json")))
    lam = tn["lam"]
    C, M = columns(rel)
    Gp = groups(M["feats"])
    gday, y0, gi, G = M["gday"], M["y0"], M["gi"], M["G"]
    tr = (gday >= FIT0) & (gday < HOLD0)
    te = gday >= HOLD0
    has = np.bincount(gi, weights=(M["n_hist"] > 0), minlength=G)
    has3 = np.bincount(gi, weights=(M["n_hist"] >= 3), minlength=G)
    rows = dict(np.load(os.path.join(CACHE, "rows.npz")))
    inrep = np.isin(M["ug"], rows["replay_id"])
    sel = tn["selected_features"]
    selfam = [c for f in tn["selected_families"] for c in Gp["fam"][f]]
    selext = [c for e in tn["selected_ext"] for c in Gp[e]]
    arms = {"A0": Gp["A0"], "A1": Gp["A1"], "A2": Gp["A2"], "A3": Gp["A3"],
            "A1+replay": Gp["A1"] + Gp["replay_all"], "A2+replay": Gp["A2"] + Gp["replay_all"],
            "A3+replay": Gp["A3"] + Gp["replay_all"],
            "A2+selected": Gp["A2"] + sel, "A3+selected": Gp["A3"] + sel,
            "A2+selected_families": Gp["A2"] + selfam,
            "A2+selected_ext": Gp["A2"] + selext}
    for e in ("ext_sbx", "ext_share", "ext_cons", "ext_tal"):
        arms[f"A2+{e}"] = Gp["A2"] + Gp[e]
    if "emb" in Gp["fam"]:
        arms["A2+replay+emb"] = Gp["A2"] + Gp["replay_all"] + Gp["fam"]["emb"]
    for fam, cols in Gp["fam"].items():
        arms[f"A2+{fam}"] = Gp["A2"] + cols
        arms[f"A1+{fam}"] = Gp["A1"] + cols
        if fam != "emb":
            arms[f"A2+all_but_{fam}"] = Gp["A2"] + [c for c in Gp["replay_all"] if c not in cols]
    L, P, W = {}, {}, {}
    for k, cols in arms.items():
        p, w, sc = fit_eval(C, cols, tr, te, y0, lam)
        L[k], P[k], W[k] = ll(p, y0[te]), p, (cols, w, sc)
    pairs = [("A1", "A0"), ("A2", "A1"), ("A3", "A2"), ("A1+replay", "A1"), ("A2+replay", "A2"),
             ("A3+replay", "A3"), ("A2+selected", "A2"), ("A3+selected", "A3"),
             ("A2+selected_families", "A2"), ("A2+selected_ext", "A2")]
    pairs += [(f"A2+{e}", "A2") for e in ("ext_sbx", "ext_share", "ext_cons", "ext_tal")]
    pairs += [(f"A2+{f}", "A2") for f in Gp["fam"]] + [(f"A1+{f}", "A1") for f in Gp["fam"]]
    pairs += [("A2+replay", f"A2+all_but_{f}") for f in Gp["fam"] if f != "emb"]
    if "emb" in Gp["fam"]:
        pairs += [("A2+replay+emb", "A2+replay")]
    days = gday[te]
    subsets = {"all": np.ones(te.sum(), bool), "1plus_hist": has[te] >= 1, "3plus_hist": has[te] >= 3,
               "5plus_hist": has[te] >= 5, "3plus_with_3plus_replays": has3[te] >= 3,
               "deploy_no_replay_file_3plus_hist": (~inrep[te]) & (has[te] >= 3),
               "replay_file_3plus_hist": inrep[te] & (has[te] >= 3)}
    out = {"lam": lam, "fit_games": int(tr.sum()), "holdout_games": int(te.sum()),
           "holdout_days": [int(gday[te].min()), int(gday[te].max())],
           "subset_games": {k: int(v.sum()) for k, v in subsets.items()},
           "base_ll": {k: float(L["A2"][v].mean()) for k, v in subsets.items()},
           "acc": {k: float(np.mean((P[k] > 0.5) == (y0[te] > 0.5))) for k in ("A0", "A1", "A2", "A3", "A2+replay",
                                                                               "A3+replay", "A2+selected")},
           "results": {}}
    for a, b in pairs:
        out["results"][f"{a} vs {b}"] = {sn: paired(L[a][sm], L[b][sm], days[sm], B) for sn, sm in subsets.items()}
    # per-hero variant: replay means on the player x hero cell instead of the player
    Ch, Mh_ = columns(rel, hero_level=True)
    for k, cols in (("A2+replay_hero", Gp["A2"] + Gp["replay_all"]), ("A2+selected_hero", Gp["A2"] + sel)):
        p, w, sc = fit_eval(Ch, cols, tr, te, y0, lam)
        L[k], P[k] = ll(p, y0[te]), p
    for a, b in (("A2+replay_hero", "A2"), ("A2+selected_hero", "A2")):
        out["results"][f"{a} vs {b}"] = {sn: paired(L[a][sm], L[b][sm], days[sm], B) for sn, sm in subsets.items()}
    hh = np.bincount(gi, weights=(Mh_["n_hist"] > 0), minlength=G)
    out["subset_games"]["hero_level_3plus_hist"] = int((hh[te] >= 3).sum())
    # edge units: slot-level replay score and realized residual
    out["edge"] = edge(C, M, W, te, P)
    if sel:
        out["edge_selected"] = edge(C, M, W, te, P, arm="A2+selected")
    out["coef_A2_selected"] = dict(zip(W["A2+selected"][0], np.round(W["A2+selected"][1][1:], 4).tolist()))
    out["coef_A2_replay"] = {c: v for c, v in zip(W["A2+replay"][0], np.round(W["A2+replay"][1][1:], 4).tolist())
                             if c.startswith("r_")}
    json.dump(out, open(os.path.join(RES, "holdout.json"), "w"), indent=1)
    for k, v in out["results"].items():
        print(f"{k:42s}", {s: (round(x["gain_x1000"], 2), round(x["se_game_x1000"], 2), round(x["se_day_x1000"], 2))
                           for s, x in v.items() if s in ("all", "3plus_hist", "deploy_no_replay_file_3plus_hist")},
              flush=True)
    print(json.dumps(out["edge"], indent=0)[:3000])
    print(f"holdout done {time.time() - t0:.0f}s", flush=True)


def edge(C, M, W, te, P, arm="A2+replay", base="A2"):
    """Edge units (points of win probability). For each holdout slot: the player's own replay
    score s = w . (player's standardized replay means), the team residual y - p_base. Report
    top vs bottom third of s among slots with replay history, and the regression slope of the
    residual on the predicted shift."""
    cols, w, (mu, sd) = W[arm]
    gi, sgn, tday = M["gi"], M["sgn"], M["tday"]
    feats = M["feats"]
    Mr = M["Mr"]
    wmap = dict(zip(cols, w[1:] / sd))
    coef = np.array([wmap.get("r_" + f, 0.0) for f in feats])
    s = np.nan_to_num(Mr) @ coef              # logit contribution of this player (own team sign)
    gte = np.flatnonzero(te)
    pos = -np.ones(M["G"], np.int64)
    pos[gte] = np.arange(len(gte))
    slot_te = pos[gi] >= 0
    pb = P[base][pos[gi[slot_te]]]
    y = M["y0"][gi[slot_te]]
    sg = sgn[slot_te]
    yt = np.where(sg > 0, y, 1 - y)
    pt = np.where(sg > 0, pb, 1 - pb)
    r = yt - pt
    ss = s[slot_te]
    nh = M["n_hist"][slot_te]
    h = nh > 0
    out = {"slots_with_history": int(h.sum())}
    q = np.quantile(ss[h], [1 / 3, 2 / 3])
    top, bot = h & (ss >= q[1]), h & (ss <= q[0])
    pk = M["tk"][slot_te]
    rng = np.random.RandomState(1)
    up, pinv = np.unique(pk, return_inverse=True)
    def diff(wts):
        return np.average(r[top], weights=wts[top]) - np.average(r[bot], weights=wts[bot])
    d0 = diff(np.ones(len(r)))
    bs = [diff(rng.multinomial(len(up), np.ones(len(up)) / len(up))[pinv].astype(float)) for _ in range(200)]
    shift = pt * (1 - pt) * ss     # first-order change in win probability from the player's replay score
    out["top_minus_bottom_third_pp"] = [100 * float(d0), 100 * float(np.std(bs))]
    out["top_third_pp"] = 100 * float(r[top].mean())
    out["bottom_third_pp"] = 100 * float(r[bot].mean())
    out["sd_shift_pp"] = 100 * float(shift[h].std())
    b = np.polyfit(shift[h], r[h], 1)[0]
    out["slope_residual_on_shift"] = float(b)
    for lo, hi_ in ((1, 1), (2, 3), (4, 7), (8, 15), (16, 10 ** 6)):
        m = (nh >= lo) & (nh <= hi_)
        if m.sum() > 500:
            qq = np.quantile(ss[m], [1 / 3, 2 / 3])
            t_, b_ = m & (ss >= qq[1]), m & (ss <= qq[0])
            out[f"by_hist_{lo}_{hi_}"] = {"slots": int(m.sum()), "top_minus_bottom_pp": 100 * float(r[t_].mean() - r[b_].mean()),
                                         "se_pp": 100 * float(np.sqrt(r[t_].var() / t_.sum() + r[b_].var() / b_.sum())),
                                         "sd_shift_pp": 100 * float(shift[m].std())}
    return out


def scaling(ks=(1, 2, 4, 8, 16)):
    """Lift vs past replays per player: features from only the most recent k replays, on
    holdout games whose players with history all have 16+ past replays... relaxed to slots:
    games with 3+ players having 16+ replays; each k refit on fit+tune."""
    rel = rel_dict()
    tn = json.load(open(os.path.join(RES, "tune.json")))
    lam = tn["lam"]
    out = {}
    base_C, M = columns(rel)
    Gp = groups(M["feats"])
    gday, y0, gi, G = M["gday"], M["y0"], M["gi"], M["G"]
    tr = (gday >= FIT0) & (gday < HOLD0)
    te = gday >= HOLD0
    n16 = np.bincount(gi, weights=(M["n_hist"] >= 16), minlength=G)
    sub = n16[te] >= 3
    out["games"] = int(sub.sum())
    p0, _, _ = fit_eval(base_C, Gp["A2"], tr, te, y0, lam)
    l0 = ll(p0, y0[te])
    for sname, cols_fn in (("all", lambda: Gp["replay_all"]), ("selected", lambda: tn["selected_features"])):
        out[sname] = {}
        for k in ks + (None,):
            C, _ = columns(rel, last=k)
            p, _, _ = fit_eval(C, Gp["A2"] + cols_fn(), tr, te, y0, lam)
            l1 = ll(p, y0[te])
            out[sname][str(k)] = paired(l1[sub], l0[sub], gday[te][sub])
            print(sname, k, round(out[sname][str(k)]["gain_x1000"], 2), round(out[sname][str(k)]["se_game_x1000"], 2),
                  flush=True)
    json.dump(out, open(os.path.join(RES, "scaling.json"), "w"), indent=1)


def coverage():
    t = dict(np.load(PT.TARGETS))
    D = dict(np.load(DESIGN))
    tday, tk = D["tday"], D["tk"]
    _, nr, _, _ = asof(D["rk"], D["rday"], np.ones((len(D["rk"]), 1)), tk, tday, np.zeros(1))
    gi = np.unique(t["replay_id"], return_inverse=True)[1]
    G = gi.max() + 1
    gday = np.zeros(G, np.int64)
    gday[gi] = tday
    out = {}
    for d0 in sorted(set(gday[gday >= HOLD0].tolist())):
        m = gday == d0
        r = {}
        for n_min in (1, 5, 10, 20):
            c = np.bincount(gi, weights=(nr >= n_min), minlength=G)[m]
            r[f"players_with_{n_min}+"] = {"mean_per_game": float(c.mean()),
                                           "games_with_3+": float((c >= 3).mean()),
                                           "games_with_6+": float((c >= 6).mean())}
        out[str(datetime.date(1970, 1, 1) + datetime.timedelta(days=int(d0)))] = r
    # players: distribution of past replays on the last day
    json.dump(out, open(os.path.join(RES, "coverage.json"), "w"), indent=1)
    print(json.dumps(out, indent=0)[:4000])


def leakcheck():
    """Rewrite every replay row and DB row on or after a cut day; features of target games before
    the cut must not change."""
    rel = rel_dict()
    C1, M = columns(rel)
    cut = TUNE0
    D = dict(np.load(DESIGN))
    Dm = dict(D)
    rng = np.random.RandomState(0)
    late = D["rday"] >= cut
    Dm["Zr"] = D["Zr"].copy()
    Dm["Zr"][late] = rng.normal(size=(late.sum(), D["Zr"].shape[1])).astype(np.float32)
    lt = D["tday"] >= cut
    Dm["Zs"] = D["Zs"].copy()
    Dm["Zs"][lt] = rng.normal(size=(lt.sum(), D["Zs"].shape[1])).astype(np.float32)
    bak = DESIGN + ".bak.npz"
    os.replace(DESIGN, bak)
    try:
        np.savez(DESIGN, **Dm)
        C2, _ = columns(rel)
    finally:
        os.replace(bak, DESIGN)
    early = M["gday"] < cut
    for k in C1:
        if k.startswith(("r_", "sb_", "sbx_")):
            dv = np.abs(C1[k][early] - C2[k][early]).max()
            if dv > 1e-9:
                print("  changed before cut:", k, float(dv))
    worst = max(float(np.abs(C1[k][early] - C2[k][early]).max()) for k in C1 if k.startswith(("r_", "sb_", "sbx_")))
    later = max(float(np.abs(C1[k][~early] - C2[k][~early]).max()) for k in C1 if k.startswith("r_"))
    print(f"leakcheck: max change before cut {worst:.3g} (must be 0 up to float rounding); after cut {later:.3g} "
          f"(should be > 0)")
    assert worst < 1e-9


if __name__ == "__main__":
    {"prep": prep, "reliability": reliability, "tune": tune, "holdout": holdout, "scaling": scaling,
     "coverage": coverage, "leakcheck": leakcheck}[sys.argv[1]]()
