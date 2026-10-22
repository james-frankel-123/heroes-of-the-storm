"""
P3 hero recs and replays, part B2: normalization, stable share (split-half
reliability) and the prospective incremental-value test of replay timeline
features (cache/hr/tl_rows.npz from p3_hr_tl_features.py).

Stages (from training/):
  python3 personalization/p3_hr_tl_analyze.py reliability   -> results/hr/tl_reliability.json
  python3 personalization/p3_hr_tl_analyze.py targets       -> cache/hr/tl_targets.npz (local research DB, read-only)
  python3 personalization/p3_hr_tl_analyze.py prospective   -> results/hr/tl_prospective.json

Normalization (in the spirit of p3_rcf_data.py, a_shrink):
  1. per minute for totals (score-screen totals, death counts over the game);
     fixed-window counts (deaths before 10:00, before the first objective),
     shares, APM, movement, first-death time and team leads are used as they are;
  2. asinh(x / s) with s the median positive value, for counts, rates and APM;
  3. z within hero x outcome; hero x map x outcome cells shrunk toward the
     hero x outcome cell with 100 pseudo-games (mean and variance);
  4. clip at +-5; masked for a hero where under 5% of rows are nonzero or the
     cell sd is <= 0.02.
Statistics come from every timeline game (population means and sds only; the
prospective test never uses a game's own row as history, and the
outcome-conditioned z carries no information about its own game's result).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_hr_tl_features as F  # noqa: E402

ROWS = os.path.join(HERE, "cache", "hr", "tl_rows.npz")
TARGETS = os.path.join(HERE, "cache", "hr", "tl_targets.npz")
STATE = os.path.join(HERE, "cache", "hr", "state_0928.npz")
RES = os.path.join(HERE, "results", "hr")
SB = [nm for _, nm in F.SB]
EXTRA = [nm for _, nm in F.EXTRA]
TLF = list(F.TLF)
RATE = set(SB + EXTRA + ["death_enemy_half", "death_isolated", "death_overext"])
ASINH = RATE | {"deaths_pre10", "deaths_pre_obj", "apm_cmd", "apm_key", "apm_camera"}
FEATS = SB + EXTRA + TLF
LABEL = {
    "deaths_pre10": "deaths before 10:00", "deaths_pre_obj": "deaths before the first map objective",
    "death_enemy_half": "deaths in the enemy half (per min)",
    "death_isolated": "deaths with no ally sampled within 10 units (per min)",
    "death_overext": "overextended deaths: enemy half and no ally near (per min)",
    "early_kp": "early takedown participation (share of team takedowns before 10:00)",
    "obj_kp": "takedown participation within 30 s of a map objective",
    "enemy_half_share": "share of position samples in the enemy half",
    "move_per_min": "distance moved per minute (position samples)",
    "merc_part": "present at own team's merc captures (share)",
    "apm_cmd": "APM: commands per minute", "apm_key": "key presses per minute",
    "apm_camera": "camera moves per minute", "time_dead_share": "share of the game spent dead",
    "first_death_min": "minutes to first death", "xp_lead10": "team XP lead at 10:00 (team)",
    "level_lead10": "team level lead at 10:00 (team)", "first_merc_lead": "team first-merc lead, min (team)",
    "cc_time": "time CCing enemy heroes (per min)", "stun_time": "time stunning (per min)",
    "root_time": "time rooting (per min)", "silence_time": "time silencing (per min)",
    "tf_hero_damage": "teamfight hero damage (per min)", "tf_damage_taken": "teamfight damage taken (per min)",
    "tf_healing": "teamfight healing (per min)", "minion_damage": "minion damage (per min)",
    "creep_damage": "creep damage (per min)",
}


def pkey(region, toon):
    return region.astype(np.int64) * 10 ** 10 + toon.astype(np.int64)


def raw_matrix(z):
    X = np.full((len(z["win"]), len(FEATS)), np.nan)
    for k, f in enumerate(FEATS):
        v = z[f].astype(np.float64)
        if f in RATE:
            v = v / z["minutes"]
        X[:, k] = v
    return X


def normalize(z, X=None, fit_mask=None):
    """z-scores (n, F) per the module doc. fit_mask: rows whose population
    statistics are used (default all)."""
    if X is None:
        X = raw_matrix(z)
    n = X.shape[0]
    fit_mask = np.ones(n, bool) if fit_mask is None else fit_mask
    hero = z["hero"]
    uh, hi = np.unique(hero, return_inverse=True)
    um, mi = np.unique(z["map"], return_inverse=True)
    y = z["win"].astype(np.int64)
    Z = np.full_like(X, np.nan)
    for k, f in enumerate(FEATS):
        v = X[:, k].copy()
        if f in ASINH:
            pos = v[fit_mask & (v > 0)]
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
        k0 = 100.0
        mm = (sm + k0 * mh[par]) / (nm + k0)
        e2 = (qm + k0 * (vh[par] + mh[par] ** 2)) / (nm + k0)
        vm = np.maximum(e2 - mm ** 2, 0)
        sd = np.sqrt(vm)
        zz = (v - mm[c_m]) / np.where(sd[c_m] > 0.02, sd[c_m], np.nan)
        # mask heroes where the metric is uninformative
        nz = np.bincount(hi[ok], weights=(v[ok] != 0), minlength=len(uh)) / np.maximum(
            np.bincount(hi[ok], minlength=len(uh)), 1)
        bad = (nz < 0.05) | (np.sqrt(vh.reshape(-1, 2).mean(1)) <= 0.02)
        zz[bad[hi]] = np.nan
        Z[:, k] = np.clip(zz, -5, 5)
    return Z


def split_half(x, key, order_keys, pkeys=None, boot=0, seed=0):
    """Per-game reliability as p3_rcf_data.reliability (odd/even games in play
    order within key), with a player-cluster bootstrap if boot > 0."""
    okx = np.isfinite(x)
    idx = np.flatnonzero(okx)
    kk = key[idx]
    o = np.lexsort(tuple(k[idx] for k in order_keys[::-1]) + (kk,))
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

    def est(ww):
        mu = np.average(np.r_[m0, m1], weights=np.r_[ww, ww])
        return float(np.average((m0 - mu) * (m1 - mu), weights=ww) / var)
    r = est(w)
    out = {"r": r, "cells": int(g.sum()), "rows_in_cells": int((n0 + n1)[g].sum())}
    if boot:
        # cluster = player: the cell's player
        first = np.r_[brk][np.flatnonzero(g)] if False else None  # noqa: F841
        cell_player = np.zeros(len(u), np.int64)
        cell_player[inv] = pkeys[idx]
        cp = cell_player[g]
        up, pinv = np.unique(cp, return_inverse=True)
        rng = np.random.RandomState(seed)
        bs = []
        for _ in range(boot):
            mult = rng.multinomial(len(up), np.ones(len(up)) / len(up))
            ww = w * mult[pinv]
            if ww.sum() > 0:
                bs.append(est(ww))
        out["ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out["se"] = float(np.std(bs))
    return out


def reliability(boot=200):
    z = dict(np.load(ROWS))
    Z = normalize(z)
    pk = pkey(z["region"], z["toon_id"])
    hero_ids = np.unique(z["hero"], return_inverse=True)[1].astype(np.int64)
    cell = pk * 100 + hero_ids
    y = z["win"].astype(np.float64)
    hw = np.bincount(hero_ids, weights=y) / np.bincount(hero_ids)
    resid = y - hw[hero_ids]
    order = [z["unix"], z["replay_id"]]
    E = json.load(open(os.path.join(HERE, "results", "rcf", "reliability.json")))
    out = {"n_rows": int(len(y)), "n_games": int(len(np.unique(z["replay_id"]))),
           "sealed_overlap": int(z["sealed_overlap"]), "features": {}}
    cols = [("win residual (y - hero 2.57 win rate)", resid)] + [(f, Z[:, k]) for k, f in enumerate(FEATS)]
    for name, v in cols:
        ph = split_half(v, cell, order, pk, boot=boot)
        pl = split_half(v, pk, order, pk, boot=boot)
        rec = {"player_hero": ph, "player": pl, "coverage": float(np.isfinite(v).mean()),
               "group": ("reference" if name.startswith("win") else "scoreboard" if name in SB else
                         "score screen, not in the model's export" if name in EXTRA else
                         "timeline (team-level)" if name in F.TEAM_FEATS else "timeline"),
               "label": LABEL.get(name, name)}
        if name in SB:
            rec["E_table_player_hero"] = E.get(name, {}).get("reliability_player_hero")
            rec["E_table_player"] = E.get(name, {}).get("reliability_player")
        r = ph["r"]
        rec["games_until_half_real"] = float((1 - r) / r) if r > 0 else None
        out["features"][name] = rec
        print(f"  {name:40s} ph {ph['r']:.3f} [{ph['ci'][0]:.3f},{ph['ci'][1]:.3f}] cells {ph['cells']:,}  "
              f"pl {pl['r']:.3f}  cov {rec['coverage']:.3f}", flush=True)
    out["features"]["win residual (y - hero 2.57 win rate)"]["E_table_player_hero"] = \
        E["win residual (y - WP)"]["reliability_player_hero"]
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "tl_reliability.json"), "w") as f:
        json.dump(out, f, indent=1)


def redundancy(min_games=6):
    """How much of each timeline feature's player-level mean is predictable
    from the same player's scoreboard means (same games): split-half, so the
    target is the player's mean over odd games and the predictors are from
    even games (no shared game noise). Out-of-sample R2 by 5-fold CV over players,
    divided by the target's split-half reliability (share of the stable part)."""
    z = dict(np.load(ROWS))
    Z = normalize(z)
    pk = pkey(z["region"], z["toon_id"])
    o = np.lexsort((z["replay_id"], z["unix"], pk))
    pk_s = pk[o]
    brk = np.flatnonzero(np.r_[True, pk_s[1:] != pk_s[:-1]])
    rank = np.arange(len(o)) - np.repeat(brk, np.diff(np.r_[brk, len(o)]))
    half = np.empty(len(o), np.int64)
    half[o] = rank % 2
    up, inv = np.unique(pk, return_inverse=True)
    n = np.bincount(inv, minlength=len(up))
    keep = n >= min_games

    def means(h):
        m = h == half
        S = np.zeros((len(up), Z.shape[1]))
        N = np.zeros((len(up), Z.shape[1]))
        ok = np.isfinite(Z) & m[:, None]
        for k in range(Z.shape[1]):
            S[:, k] = np.bincount(inv[ok[:, k]], weights=Z[ok[:, k], k], minlength=len(up))
            N[:, k] = np.bincount(inv[ok[:, k]], minlength=len(up))
        return np.where(N > 0, S / np.maximum(N, 1), np.nan)
    A0, A1 = means(0)[keep], means(1)[keep]
    sbx = [FEATS.index(f) for f in SB + EXTRA]
    Xp = np.nan_to_num(A0[:, sbx])
    out = {"players": int(keep.sum()), "min_games": min_games, "features": {}}
    rng = np.random.RandomState(1)
    fold = rng.randint(0, 5, len(Xp))
    for f in TLF:
        k = FEATS.index(f)
        yv = A1[:, k]
        ok = np.isfinite(yv)
        y1 = yv[ok]
        X1 = np.column_stack([np.ones(ok.sum()), Xp[ok]])
        pred = np.zeros(ok.sum())
        for j in range(5):
            tr = fold[ok] != j
            w = np.linalg.solve(X1[tr].T @ X1[tr] + 1.0 * np.eye(X1.shape[1]), X1[tr].T @ y1[tr])
            pred[~tr] = X1[~tr] @ w
        r2 = 1 - np.mean((y1 - pred) ** 2) / np.var(y1)
        # reliability of the target half-mean: corr of odd and even halves
        a0 = A0[ok, k]
        g = np.isfinite(a0)
        rel = float(np.corrcoef(a0[g], y1[g])[0, 1])
        out["features"][f] = {"r2_from_scoreboard": float(r2), "half_mean_reliability": rel,
                              "share_of_stable_part_explained": float(r2 / rel) if rel > 0.02 else None}
        print(f"  {f:20s} R2 {r2:.3f}  half-mean rel {rel:.3f}  share {r2 / rel if rel > 0.02 else float('nan'):.2f}",
              flush=True)
    with open(os.path.join(RES, "tl_redundancy.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "reliability":
        reliability()
    elif what == "redundancy":
        redundancy()
    elif what == "targets":
        import p3_hr_tl_prospective as PR
        PR.targets()
    elif what == "prospective":
        import p3_hr_tl_prospective as PR
        PR.prospective()
        PR.prospective(hist_cohorts=(3,), tag="_panelonly")
