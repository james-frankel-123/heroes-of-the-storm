"""
P3 hero strength: paired comparisons with player-clustered bootstrap CIs.

Online (hl = inf, lag 1 day) predictions on V2 slots for: no pooling (cell
shrunk to 0 alone), player+hero, player+role+melee+hero, +CF rank 2. For
each pair, the per-slot log-loss gain difference (x1000) overall and by
games on the hero, with 95% CIs from 200 bootstrap resamples of players.
Game level: paired replay bootstrap of the combiner log-loss difference
(logit WP + team difference, fit on V1).

Usage (from training/): python3 personalization/p3_hs_paired.py
Output: results/p3_hs_paired.json
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C
from p3_hs_fit import KERNELS, KOUT, prepare
from p3_hs_eval import nbin, N_LABELS, team_diff


def main():
    d = C.load_slots()
    H = len(d["hero_names"])
    kz = np.load(KOUT)
    bases = {k[6:]: kz[k] for k in kz.files if k.startswith("basis_")}
    e_mask, _, _, table, r_adj = prepare(d)
    with open(os.path.join(C.RESULTS, "p3_hs_nopool.json")) as f:
        t_np = json.load(f)["hero_sd_pp"] / 100
    Ks = {"no pooling": t_np ** 2 * np.eye(H)}
    for n in ("player+hero", "player+role+melee+hero", "+CF rank 2"):
        Ks[n] = C.kernel_from([bases[b] for b in KERNELS[n]], kz["theta_" + n])
    days = d["day"]
    post = ~d["in_sample"]
    _, first_row = np.unique(d["g"][post], return_index=True)
    med = np.median(days[post][first_row])
    qidx, S, P, Nh, Np, _, _ = C.online_state(d, np.ones_like(post), post, H, r_adj)
    mu = table[C.exp_bins(Np.astype(np.int64), Nh.astype(np.int64))]
    hq = d["hero"][qidx]
    r, wp = d["r"][qidx], d["wp"][qidx]
    y = r + wp
    ll0 = -(y * np.log(wp) + (1 - y) * np.log(1 - wp))
    preds = {"experience only": mu}
    for n, K in Ks.items():
        preds[n] = mu + C.gp_query(S, P, hq, K)[0]
    del S, P
    gain = {}
    for n, pr in preds.items():
        p1 = np.clip(wp + pr, 1e-3, 1 - 1e-3)
        gain[n] = 1000 * (ll0 + (y * np.log(p1) + (1 - y) * np.log(1 - p1)))
    v2 = days[qidx] >= med
    v1 = ~v2
    pid = d["pid"][qidx]
    b = nbin(Nh.astype(np.int64))
    # player-cluster bootstrap: resample players, weight slots by multiplicity
    rng = np.random.RandomState(0)
    upid, pinv = np.unique(pid[v2], return_inverse=True)
    W = [np.bincount(rng.randint(0, len(upid), len(upid)), minlength=len(upid))[pinv]
         for _ in range(200)]
    pairs = [("player+hero", "no pooling"), ("player+role+melee+hero", "player+hero"),
             ("+CF rank 2", "player+hero"), ("+CF rank 2", "no pooling"),
             ("no pooling", "experience only")]
    out = {"slot_pairs": {}, "game_pairs": {}}
    for a, c in pairs:
        diff = (gain[a] - gain[c])[v2]
        res = {}
        for lab, sel in [("all", np.ones(len(diff), bool))] + \
                [(N_LABELS[i], b[v2] == i) for i in range(len(N_LABELS))]:
            est = diff[sel].mean()
            bs = [np.average(diff[sel], weights=w[sel]) for w in W]
            res[lab] = [float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out["slot_pairs"][f"{a} - {c}"] = res
        print(f"{a} - {c}: " + " ".join(f"{k}:{v[0]:+.3f}[{v[1]:+.3f},{v[2]:+.3f}]"
                                         for k, v in res.items()), flush=True)
    # game level
    n_games = int(d["g"].max()) + 1
    yg = np.zeros(n_games)
    wg = np.full(n_games, 0.5)
    t0 = d["team"] == 0
    yg[d["g"][t0]] = d["y"][t0]
    wg[d["g"][t0]] = d["wp"][t0]
    lo = np.log(wg / (1 - wg))
    full = np.bincount(d["g"][post], minlength=n_games) == 10
    gv1 = np.zeros(n_games, bool)
    gv1[d["g"][qidx][v1]] = True
    gv2 = np.zeros(n_games, bool)
    gv2[d["g"][qidx][v2]] = True
    fit_m, test_m = gv1 & full, gv2 & full
    llg = {}
    for n, pr in preds.items():
        X = np.column_stack([lo, team_diff(pr, d["g"][qidx], d["team"][qidx], n_games)])
        w = C.fit_logistic(X[fit_m], yg[fit_m])
        p = C.predict(w, X)
        llg[n] = -(yg * np.log(p) + (1 - yg) * np.log(1 - p))
    idx = np.flatnonzero(test_m)
    boots = [rng.choice(idx, len(idx)) for _ in range(200)]
    for a, c in pairs:
        dd = llg[c] - llg[a]
        bs = [dd[bb].mean() for bb in boots]
        out["game_pairs"][f"{a} - {c}"] = [float(dd[idx].mean()), float(np.percentile(bs, 2.5)),
                                           float(np.percentile(bs, 97.5))]
        print(f"game {a} - {c}: {out['game_pairs'][f'{a} - {c}']}", flush=True)
    with open(os.path.join(C.RESULTS, "p3_hs_paired.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
