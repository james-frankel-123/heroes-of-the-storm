"""
P3 H2: forced roles. Do players lose more than the WP plus their own skill
predict when they play a role they rarely play?

Role familiarity, causal, from each player's earlier games in play-time
order (game_date, replay_id tiebreak; counts only, no outcomes):
  share_b  share of earlier games in this hero's Blizzard role
  share_f  same for the fine role (HERO_ROLE_FINE)
  n_p, n_ph (time-ordered) as experience controls.
"Off-role": share < 10% among players with >= 50 earlier games.

Analyses
 A. Raw residual r = y - wp by role-share bin, within hero-familiarity bins
    (0, 1-4, 5-19, 20+ earlier games on the hero): role effect vs hero
    unfamiliarity.
 B. The same for e = r - (experience offset + skill posterior mean), the
    phase-1 static kernel run online through the previous day (it already
    has per-player role, fine-role and hero terms and the n_p x n_ph
    experience table). Does anything survive?
 C. By within-team pick position (1st..5th pick of the team; draft_order).
 D. Team level: off-role count per team and role-share features in the
    game combiner on top of M0 + skill model (fit V1, test V2).

Run (from training/, 4 threads):
  NUMBA_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_ph_role.py
Output: results/p3_ph_role.json
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from p3_heroes import HKEY
import p3_hs_core as C
import p3_sd_kalman as K
from p3_hs_fit import prepare
from p3_hs_eval import team_diff

GT = os.path.join(C.CACHE, "gametime_2024q2.npz")
PO = os.path.join(C.CACHE, "pickorder_2024q2.npz")
PRED = os.path.join(C.CACHE, "sd_pred_static_lag1.npz")
SHARE_EDGES = [0, 0.05, 0.10, 0.20, 0.35, 0.5, 1.01]
SHARE_LABELS = ["<5%", "5-10%", "10-20%", "20-35%", "35-50%", "50%+"]
NPH_EDGES = [0, 1, 5, 20, 10 ** 9]
NPH_LABELS = ["0", "1-4", "5-19", "20+"]


def prior_counts(d, key, order_second):
    """Number of earlier rows (by pid, time) with the same key value."""
    n = len(key)
    o = np.lexsort((d["replay_id"], order_second, key))
    k = key[o]
    brk = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    c = np.arange(n) - np.repeat(brk, np.diff(np.r_[brk, n]))
    out = np.empty(n, np.int64)
    out[o] = c
    return out


def static_predictions(d, meta, kz, r_adj):
    if os.path.exists(PRED):
        return np.load(PRED)["m"]
    th = kz["theta_+CF rank 2"]
    ex = lambda i: float(np.sqrt(np.exp(th[i])))
    p = {"sd_player_s": ex(0), "sd_role": ex(1), "sd_fine": ex(2), "sd_melee": ex(3),
         "sd_hero_s": ex(4), "sd_cf": ex(5)}
    f = K.Filter(d, meta, kz["V_cf2"], r_adj)
    sd, lam, br = K.split(p)
    m, s2, _ = f.run(sd, f.lam_state(lam, br))
    pm = np.empty(len(d["pid"]))
    pm[f.rows] = m
    np.savez(PRED, m=pm)
    return pm


def binned(x, edges):
    return np.searchsorted(edges, x, side="right") - 1


def table_by(val, share, nph, sel):
    sb = binned(share, SHARE_EDGES)
    hb = binned(nph, NPH_EDGES)
    out = {}
    for i, hl in enumerate(NPH_LABELS):
        row = {}
        for j, sl in enumerate(SHARE_LABELS):
            m = sel & (hb == i) & (sb == j)
            if m.sum() >= 2000:
                row[sl] = [float(100 * val[m].mean()), float(100 * val[m].std() / np.sqrt(m.sum())),
                           int(m.sum())]
        out[hl] = row
    return out


def penalty(val, share, nph, sel, lo=0.10, hi=0.35):
    """Off-role (< lo) minus on-role (>= hi) mean, averaged over hero-
    familiarity bins weighted by off-role counts; bootstrap SE by row."""
    hb = binned(nph, NPH_EDGES)
    num, den, var = 0.0, 0, 0.0
    for i in range(len(NPH_LABELS)):
        off = sel & (hb == i) & (share < lo)
        on = sel & (hb == i) & (share >= hi)
        if off.sum() < 500 or on.sum() < 500:
            continue
        diff = val[off].mean() - val[on].mean()
        w = off.sum()
        num += w * diff
        den += w
        var += w ** 2 * (val[off].var() / off.sum() + val[on].var() / on.sum())
    return [float(100 * num / den), float(100 * np.sqrt(var) / den)] if den else None


def main():
    t0 = time.time()
    d = C.load_slots()
    meta = C.hero_meta(d["hero_names"])
    kz = np.load(K.KOUT)
    e_mask, n_p_rid, n_ph_rid, table, r_adj = prepare(d)
    z = np.load(GT)
    o = np.argsort(z["replay_ids"])
    gi = np.searchsorted(z["replay_ids"][o], d["replay_id"])
    ts = z["ts"][o][gi]
    blizz = meta["blizz"][d["hero"]]
    fine = meta["fine"][d["hero"]]
    # time-ordered causal counts (key = player, or player x role/hero)
    n_p = prior_counts(d, d["pid"], ts)
    n_pb = prior_counts(d, d["pid"] * 8 + blizz, ts)
    n_pf = prior_counts(d, d["pid"] * 16 + fine, ts)
    n_ph = prior_counts(d, d["pid"] * HKEY + d["hero"], ts)
    share_b = n_pb / np.maximum(n_p, 1)
    share_f = n_pf / np.maximum(n_p, 1)
    est = n_p >= 50
    out = {"n_slots_established": int(est.sum()),
           "off_role_share_blizz": float((share_b[est] < 0.10).mean()),
           "off_role_share_fine": float((share_f[est] < 0.10).mean()),
           "off_role_share_blizz_by_role": {
               C.BLIZZ_ROLES[k]: float((share_b[est & (blizz == k)] < 0.10).mean()) for k in range(6)}}
    print(json.dumps(out), flush=True)
    r = d["r"]
    pm = static_predictions(d, meta, kz, r_adj)
    mu = table[C.exp_bins(n_p_rid, n_ph_rid)]
    e = r - (mu + pm)
    post = ~d["in_sample"]
    late = d["day"] >= C.day_of("2025-01-01")
    out["A_raw_residual_blizz"] = table_by(r, share_b, n_ph, est & late)
    out["A_raw_residual_fine"] = table_by(r, share_f, n_ph, est & late)
    out["B_model_residual_blizz"] = table_by(e, share_b, n_ph, est & late)
    out["B_model_residual_fine"] = table_by(e, share_f, n_ph, est & late)
    pen = {}
    for nm, val in (("raw", r), ("after model", e)):
        for rn, sh in (("blizz", share_b), ("fine", share_f)):
            for wn, w in (("2025-01+", est & late), ("post-cutoff", est & post)):
                pen[f"{nm} | {rn} | {wn}"] = penalty(val, sh, n_ph, w)
    # penalty by the role being played (Blizzard), after model
    for k in range(6):
        pen[f"after model | blizz | 2025-01+ | playing {C.BLIZZ_ROLES[k]}"] = penalty(
            e, share_b, n_ph, est & late & (blizz == k))
        pen[f"raw | blizz | 2025-01+ | playing {C.BLIZZ_ROLES[k]}"] = penalty(
            r, share_b, n_ph, est & late & (blizz == k))
    out["penalty_pp_offrole_minus_onrole"] = pen
    for k, v in pen.items():
        print(f"  penalty {k}: {v}", flush=True)
    # ---------- C: pick position
    if os.path.exists(PO):
        po = np.load(PO)
        names = {str(n): i for i, n in enumerate(d["hero_names"])}
        ph = np.array([names.get(str(h), -1) for h in po["hero"]])
        key_po = po["replay_ids"] * HKEY + ph
        key_d = d["replay_id"] * HKEY + d["hero"]
        so = np.argsort(key_po)
        j = np.searchsorted(key_po[so], key_d)
        j = np.minimum(j, len(so) - 1)
        okp = key_po[so][j] == key_d
        rank = np.where(okp, po["pick_rank"][so][j], -1).astype(np.int64)
        # within-team pick index
        tk = d["g"] * 2 + d["team"]
        o2 = np.lexsort((rank, tk))
        tks = tk[o2]
        brk = np.flatnonzero(np.r_[True, tks[1:] != tks[:-1]])
        wi = np.arange(len(o2)) - np.repeat(brk, np.diff(np.r_[brk, len(o2)]))
        within = np.empty(len(o2), np.int64)
        within[o2] = wi
        full_team = np.bincount(tk, weights=okp, minlength=tk.max() + 1)[tk] == 5
        within = np.where(okp & full_team, within, -1)
        out["pick_matched_share"] = float((within >= 0).mean())
        byp = {}
        for q in range(5):
            s = est & late & (within == q)
            byp[f"team pick {q + 1}"] = {
                "off_role_share": float((share_b[s] < 0.10).mean()),
                "penalty_raw": penalty(r, share_b, n_ph, s),
                "penalty_after_model": penalty(e, share_b, n_ph, s),
                "mean_after_model_resid_pp": float(100 * e[s].mean())}
            print(f"  pick {q + 1}: {byp[f'team pick {q + 1}']}", flush=True)
        out["C_pick_position"] = byp
    # ---------- D: game level
    n_games = int(d["g"].max()) + 1
    y = np.zeros(n_games)
    wg = np.full(n_games, 0.5)
    t0m = d["team"] == 0
    y[d["g"][t0m]] = d["y"][t0m]
    wg[d["g"][t0m]] = d["wp"][t0m]
    lo_ = np.log(wg / (1 - wg))
    days = d["day"]
    _, first_row = np.unique(d["g"][post], return_index=True)
    med = np.median(days[post][first_row])
    full = np.bincount(d["g"][post], minlength=n_games) == 10
    gv1 = np.zeros(n_games, bool)
    gv1[d["g"][post & (days < med)]] = True
    gv2 = np.zeros(n_games, bool)
    gv2[d["g"][post & (days >= med)]] = True
    fit_m, test_m = gv1 & full, gv2 & full
    base_rate_b = np.bincount(blizz, minlength=6) / len(blizz)
    base_rate_f = np.bincount(fine, minlength=9) / len(fine)
    sm_b = (n_pb + 10 * base_rate_b[blizz]) / (n_p + 10)
    sm_f = (n_pf + 10 * base_rate_f[fine]) / (n_p + 10)
    feats = {"skill": mu + pm,
             "off-role blizz": (est & (share_b < 0.10)).astype(float),
             "off-role fine": (est & (share_f < 0.10)).astype(float),
             "log role share blizz": np.log(sm_b),
             "log role share fine": np.log(sm_f)}
    D = {k: team_diff(v, d["g"], d["team"], n_games) for k, v in feats.items()}
    specs = {"M0": [], "skill": ["skill"],
             "skill + off-role count (blizz)": ["skill", "off-role blizz"],
             "skill + off-role count (fine)": ["skill", "off-role fine"],
             "skill + log role share (blizz)": ["skill", "log role share blizz"],
             "skill + log role share (fine)": ["skill", "log role share fine"],
             "skill + all role terms": ["skill", "off-role blizz", "off-role fine",
                                        "log role share blizz", "log role share fine"],
             "off-role count (blizz) alone": ["off-role blizz"],
             "log role share (blizz) alone": ["log role share blizz"]}
    rng = np.random.RandomState(0)
    idx = np.flatnonzero(test_m)
    boots = [rng.choice(idx, len(idx)) for _ in range(200)]
    lls = {}
    out["D_game"] = {}
    for nm, cols in specs.items():
        X = np.column_stack([lo_] + [D[c] for c in cols])
        w = C.fit_logistic(X[fit_m], y[fit_m])
        p = C.predict(w, X)
        lls[nm] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        res = C.game_metrics(p[test_m], y[test_m])
        res["coef"] = w.tolist()
        for ref in ("M0", "skill"):
            if ref in lls:
                dd = lls[ref] - lls[nm]
                bs = [dd[b].mean() for b in boots]
                res[f"gain_vs_{ref}"] = [float(dd[idx].mean()), float(np.percentile(bs, 2.5)),
                                         float(np.percentile(bs, 97.5))]
        out["D_game"][nm] = res
        print(f"  {nm:36s} acc {res['acc']:.4f} ll {res['logloss']:.5f} "
              f"{ {k: [round(x, 5) for x in v] for k, v in res.items() if k.startswith('gain')} } "
              f"coef {np.round(w[2:], 4) if len(w) > 2 else ''}", flush=True)
    # team off-role count distribution (V2, established players only counted)
    cnt = np.bincount(d["g"] * 2 + d["team"], weights=feats["off-role blizz"],
                      minlength=2 * n_games).reshape(-1, 2)
    out["team_offrole_count_distribution_V2"] = {
        str(k): float((cnt[test_m].ravel() == k).mean()) for k in range(4)}
    with open(os.path.join(C.RESULTS, "p3_ph_role.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
