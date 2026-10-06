"""
P3 extensions, task 6: does personal skill depend on the map? (proposal #5)

Variance decomposition of the detrended residual r_adj = y - WP_drift -
experience offset (snapshot games 2024-04 .. 2026-05) into player, player x
hero, player x map and player x hero x map components, with no game-noise
model. Cells are (player, hero, map). Covariances of cell means between
DIFFERENT cells of the same player share no games, so noise drops out:

  C_same   odd-vs-even split-half covariance inside a cell
           = s2_p + s2_ph + s2_pm + s2_phm
  C_hero   same hero, different map   = s2_p + s2_ph
  C_map    same map, different hero   = s2_p + s2_pm
  C_none   different hero and map     = s2_p

so s2_phm = C_same - C_hero - C_map + C_none, s2_pm = C_map - C_none,
s2_ph = C_hero - C_none, s2_p = C_none. Cells are weighted by
w = n / (1 + n / 50). Hero x map effects are in the population WP (it has
hero-map win rates), so they are not a personal component. CIs: player
bootstrap. Everything is also done on e = r - (offset + CF posterior mean,
lag 1 day), i.e. what is left after the skill model.

Predictive check: a causal player x map (and player x hero x map) mean of e
over earlier days, shrunk with k = noise / s2 from the decomposition, added
to the V1-fit combiner; V2 log-loss gain.

Usage (from training/): python3 personalization/p3_x_map.py
Output: results/p3_x_map.json
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from p3_heroes import HKEY
import p3_hs_core as C
import p3_x_common as X
import p3_x_side as XS

PRED = os.path.join(C.CACHE, "x_predall.npz")


def group_terms(pid, cell_h, cell_m, S1, S2, N1, N2, W):
    """Per-group sums needed for the pair covariances, computed once.
    Returns dict of (terms, player index of each group) per grouping."""
    x = (S1 + S2) / (N1 + N2)
    ok2 = (N1 > 0) & (N2 > 0)
    xa = np.where(N1 > 0, S1 / np.maximum(N1, 1), 0)
    xb = np.where(N2 > 0, S2 / np.maximum(N2, 1), 0)
    wx = W * x
    Q = W * W * x * x
    out = {"same": (W * xa * xb * ok2, W * ok2, pid)}
    for nm, key in (("p", pid), ("ph", pid * HKEY + cell_h), ("pm", pid * 64 + cell_m)):
        u, inv = np.unique(key, return_inverse=True)
        A = np.bincount(inv, weights=wx)
        B = np.bincount(inv, weights=W)
        Qg = np.bincount(inv, weights=Q)
        W2 = np.bincount(inv, weights=W * W)
        gp = u if nm == "p" else (u // HKEY if nm == "ph" else u // 64)
        out[nm] = (A ** 2 - Qg, B ** 2 - W2, gp)
    return out


def components(G, mult=None):
    def tot(nm):
        a, b, gp = G[nm]
        m = 1.0 if mult is None else mult[gp]
        return (m * a).sum(), (m * b).sum()
    ns, ds = tot("same")
    nh, dh = tot("ph")
    nm_, dm = tot("pm")
    na, da = tot("p")
    c_same, c_hero, c_map = ns / ds, nh / dh, nm_ / dm
    c_none = (na - nh - nm_) / (da - dh - dm)
    return {"C_same": c_same, "C_hero": c_hero, "C_map": c_map, "C_none": c_none,
            "s2_p": c_none, "s2_ph": c_hero - c_none, "s2_pm": c_map - c_none,
            "s2_phm": c_same - c_hero - c_map + c_none}


def decompose(d, val, mapi, rng, label):
    pid = d["pid"]
    key = (pid * HKEY + d["hero"]) * 64 + mapi
    o = np.lexsort((d["replay_id"], d["day"], key))
    ks = key[o]
    st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    pos = np.arange(len(ks)) - np.repeat(st, np.diff(np.r_[st, len(ks)]))
    odd = np.empty(len(ks), bool)
    odd[o] = pos % 2 == 0
    u, inv = np.unique(key, return_inverse=True)
    S1 = np.bincount(inv, weights=val * odd)
    S2 = np.bincount(inv, weights=val * ~odd)
    N1 = np.bincount(inv, weights=odd.astype(float))
    N2 = np.bincount(inv, weights=(~odd).astype(float))
    n = N1 + N2
    W = n / (1 + n / 50.0)
    cp = u // (HKEY * 64)
    ch = (u // 64) % HKEY
    cm = u % 64
    G = group_terms(cp, ch, cm, S1, S2, N1, N2, W)
    est = components(G)
    npl = int(d["n_players"])
    boots = []
    for b in range(200):
        pb = np.bincount(rng.randint(0, npl, npl), minlength=npl).astype(float)
        boots.append(components(G, pb))
    res = {}
    for k in est:
        vals = np.array([bb[k] for bb in boots])
        res[k] = {"pp2": float(1e4 * est[k]), "ci_pp2": [float(1e4 * np.percentile(vals, 2.5)),
                                                         float(1e4 * np.percentile(vals, 97.5))]}
    comps = ["s2_p", "s2_ph", "s2_pm", "s2_phm"]
    tot = sum(max(est[k], 0) for k in comps)
    res["shares"] = {k: float(max(est[k], 0) / tot) for k in comps}
    res["sd_pp"] = {k: float(100 * np.sqrt(max(est[k], 0))) for k in comps}
    print(label, {k: round(v["pp2"], 2) for k, v in res.items() if isinstance(v, dict) and "pp2" in v},
          res["shares"], flush=True)
    return res


def causal_group_mean(d, val, gkey):
    """Per row: sum and count of val over the same group's rows from earlier days."""
    o = np.lexsort((d["replay_id"], d["day"], gkey))
    gk = gkey[o]
    dy = d["day"][o].astype(np.int64)
    cs = np.r_[0.0, np.cumsum(val[o])]
    comp_first_day = np.searchsorted(gk * 100000 + dy, gk * 100000 + dy, side="left")
    first_g = np.searchsorted(gk, gk, side="left")
    s = cs[comp_first_day] - cs[first_g]
    c = (comp_first_day - first_g).astype(float)
    S = np.empty(len(val))
    Cn = np.empty(len(val))
    S[o] = s
    Cn[o] = c
    return S, Cn


def main():
    t0 = time.time()
    d = C.load_slots()
    P = np.load(PRED)
    sg = np.load(os.path.join(C.CACHE, "x_side_games.npz"))
    gi = XS.game_lookup(d, sg["replay_ids"])
    ok = gi >= 0
    print(f"slots with map: {ok.mean():.4f}", flush=True)
    mapi = np.where(ok, sg["map"][np.maximum(gi, 0)], 63).astype(np.int64)
    rng = np.random.RandomState(0)
    r_adj = P["r_adj"].astype(np.float64)
    e = d["r"] - (P["mu"] + P["m_cf2"]).astype(np.float64)
    keep = ok
    dd = {k: (v[keep] if np.ndim(v) > 0 and len(v) == len(ok) else v) for k, v in d.items()}
    out = {"maps": [str(m) for m in sg["map_names"]], "slots": int(keep.sum())}
    out["r_adj (after experience offset)"] = decompose(dd, r_adj[keep], mapi[keep], rng, "r_adj")
    out["e (after skill model)"] = decompose(dd, e[keep], mapi[keep], rng, "e")
    # heavy players only (300+ games), where map cells have data
    heavy = np.bincount(d["pid"], minlength=int(d["n_players"])) >= 300
    hk = keep & heavy[d["pid"]]
    dh = {k: (v[hk] if np.ndim(v) > 0 and len(v) == len(ok) else v) for k, v in d.items()}
    out["r_adj, players with 300+ games"] = decompose(dh, r_adj[hk], mapi[hk], rng, "heavy r_adj")

    # predictive check: shrinkage from variance components estimated on the E
    # window only (the full-sample decomposition above includes V games)
    ke = keep & d["in_sample"] & (d["day"] >= C.day_of(C.E_START))
    de_ = {k: (v[ke] if np.ndim(v) > 0 and len(v) == len(ok) else v) for k, v in d.items()}
    out["e (after skill model), E window only"] = decompose(de_, e[ke], mapi[ke], rng, "e E-only")
    comp = out["e (after skill model), E window only"]
    noise = float(np.mean(d["v"]))
    post = ~d["in_sample"]
    _, fr = np.unique(d["g"][post], return_index=True)
    med = np.median(d["day"][post][fr])
    n_games, y, wp0, cnt, gday = X.game_arrays(d)
    cntp = np.bincount(d["g"][post], minlength=n_games)
    lo = np.log(wp0 / (1 - wp0))
    fit_m = (cntp == 10) & (gday < med)
    test_m = (cntp == 10) & (gday >= med)
    base_s = (P["mu"] + P["m_cf2"]).astype(np.float64)
    D0 = X.team_diff(base_s, d["g"], d["team"], n_games)
    feats = {}
    for nm, gkey, s2k in (("player x map", d["pid"] * 64 + mapi, "s2_pm"),
                          ("player x hero x map", (d["pid"] * HKEY + d["hero"]) * 64 + mapi, "s2_phm")):
        s2 = max(comp[s2k]["pp2"] / 1e4, 1e-7)
        S, Cn = causal_group_mean(d, np.where(ok, e, 0.0), gkey)
        est = S / (Cn + noise / s2)
        feats[nm] = X.team_diff(np.where(ok, est, 0.0), d["g"], d["team"], n_games)
        out.setdefault("shrinkage_k_games", {})[nm] = float(noise / s2)
    w0 = C.fit_logistic(np.column_stack([lo, D0])[fit_m], y[fit_m])
    l0 = X.logloss(C.predict(w0, np.column_stack([lo, D0])), y)
    idx = np.flatnonzero(test_m)
    gl = {}
    for nm, cols in (("+ player x map", [feats["player x map"]]),
                     ("+ player x hero x map", [feats["player x hero x map"]]),
                     ("+ both", [feats["player x map"], feats["player x hero x map"]])):
        Xm = np.column_stack([lo, D0] + cols)
        w = C.fit_logistic(Xm[fit_m], y[fit_m])
        l1 = X.logloss(C.predict(w, Xm), y)
        g, ci = X.boot_gain(l0, l1, idx)
        gl[nm] = {"gain_over_skill_model": g, "ci": ci, "coef": w.tolist()}
    out["game_level_V2"] = gl
    print(json.dumps(gl), flush=True)
    with open(os.path.join(C.RESULTS, "p3_x_map.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
