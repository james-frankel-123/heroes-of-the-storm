"""
P3 item 3: validity checks for the skill estimates.

A. New accounts and smurfs.
   New account: first seen on or after 2024-07-01 (three months after the
   window opens) with median hero_level <= 5 over its first 10 games (hero
   levels are low on a fresh account). Residual r = y - WP_drift by the
   account's game index. Smurf-like: a new account whose first 20 games
   have z = sum(r) / sqrt(sum wp(1 - wp)) > 2.33 (1% one-sided under no
   skill signal). Opponent bias: opponents' mean residual in games where a
   smurf-like account is in its first 20 games, vs games with other new
   accounts in their first 20. Effect on skill estimates: the online skill
   model (phase-1 static kernel, lag 1 day) rerun with every game that
   contains a smurf-like account's first 20 games removed from the
   estimation history; V2 game-level log loss and the change in slot
   estimates. The flag uses each account's own first 20 games, which all
   precede the later games it affects.
B. Premade parties: party id != 0, size = members of the same party on the
   team. Slot residual after the skill model, e = r - (experience offset +
   posterior mean), by party size; team level: party structure added to the
   V2 combiner on top of the skill model.

Run (from training/):
  OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_val_validity.py
Output: results/p3_val_validity.json
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from p3_heroes import NUM_HEROES
import p3_hs_core as C
import p3_hero_level_causal as HL
from p3_hs_fit import prepare, KERNELS
from p3_hs_eval import team_diff

GT = os.path.join(C.CACHE, "gametime_2024q2.npz")
PRED = os.path.join(C.CACHE, "sd_pred_static_lag1.npz")


def ci(x, rng, n=1000, cluster=None):
    x = np.asarray(x, float)
    if cluster is None:
        bs = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(n)]
    else:
        u, inv = np.unique(cluster, return_inverse=True)
        s = np.bincount(inv, weights=x)
        c = np.bincount(inv).astype(float)
        bs = []
        for _ in range(n):
            w = np.bincount(rng.randint(0, len(u), len(u)), minlength=len(u))
            bs.append((w * s).sum() / (w * c).sum())
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def game_level(d, preds_by_name, extra, rng):
    n_games = int(d["g"].max()) + 1
    y = np.zeros(n_games)
    wg = np.full(n_games, 0.5)
    t0 = d["team"] == 0
    y[d["g"][t0]] = d["y"][t0]
    wg[d["g"][t0]] = d["wp"][t0]
    lo = np.log(wg / (1 - wg))
    days = d["day"]
    post = ~d["in_sample"]
    _, first_row = np.unique(d["g"][post], return_index=True)
    med = np.median(days[post][first_row])
    full = np.bincount(d["g"][post], minlength=n_games) == 10
    gv1 = np.zeros(n_games, bool)
    gv1[d["g"][post & (days < med)]] = True
    gv2 = np.zeros(n_games, bool)
    gv2[d["g"][post & (days >= med)]] = True
    fit_m, test_m = gv1 & full, gv2 & full
    idx = np.flatnonzero(test_m)
    boots = [rng.choice(idx, len(idx)) for _ in range(200)]
    Dm = {k: team_diff(v, d["g"], d["team"], n_games) for k, v in preds_by_name.items()}
    Dm.update(extra)
    res, lls = {}, {}
    specs = {"M0": []}
    specs.update({k: [k] for k in preds_by_name})
    return Dm, lo, y, fit_m, test_m, boots, specs


def fit_eval(specs, Dm, lo, y, fit_m, test_m, boots, ref):
    res, lls = {}, {}
    for nm, cols in specs.items():
        X = np.column_stack([lo] + [Dm[c] for c in cols])
        w = C.fit_logistic(X[fit_m], y[fit_m])
        p = C.predict(w, X)
        lls[nm] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        r = C.game_metrics(p[test_m], y[test_m])
        r["coef"] = w.tolist()
        res[nm] = r
    idx = np.flatnonzero(test_m)
    for nm in specs:
        for rf in ("M0", ref):
            if rf in lls:
                dd = lls[rf] - lls[nm]
                bs = [dd[b].mean() for b in boots]
                res[nm][f"gain_vs_{rf}"] = [float(dd[idx].mean()), float(np.percentile(bs, 2.5)),
                                            float(np.percentile(bs, 97.5))]
    return res


def main():
    rng = np.random.RandomState(0)
    d = C.load_slots()
    H = NUM_HEROES
    e_mask, n_p, n_ph, table, r_adj = prepare(d)
    z = np.load(GT)
    o = np.argsort(z["replay_ids"])
    ts = z["ts"][o][np.searchsorted(z["replay_ids"][o], d["replay_id"])]
    npl = int(d["n_players"])
    # time-ordered game index per player
    srt = np.lexsort((d["replay_id"], ts, d["pid"]))
    pid_s = d["pid"][srt]
    brk = np.flatnonzero(np.r_[True, pid_s[1:] != pid_s[:-1]])
    gidx_s = np.arange(len(srt)) - np.repeat(brk, np.diff(np.r_[brk, len(srt)]))
    gidx = np.empty(len(srt), np.int64)
    gidx[srt] = gidx_s
    first_day = np.full(npl, 10 ** 6)
    np.minimum.at(first_day, d["pid"], d["day"])
    hl = HL.load(d)["hero_lo"]  # level stamped before each game started
    f10 = gidx < 10
    # median hero level over first 10 games
    med_hl = np.full(npl, np.nan)
    o10 = np.flatnonzero(f10)
    oo = o10[np.argsort(d["pid"][o10], kind="stable")]
    pp = d["pid"][oo]
    b2 = np.flatnonzero(np.r_[True, pp[1:] != pp[:-1]])
    for s_, e_ in zip(b2, np.r_[b2[1:], len(oo)]):
        v = hl[oo[s_:e_]]
        if (~np.isnan(v)).any():
            med_hl[pp[s_]] = np.nanmedian(v)
    new_acc = (first_day >= C.day_of("2024-07-01")) & (med_hl <= 5)
    total = np.bincount(d["pid"], minlength=npl)
    out = {"A": {}, "B": {}}
    A = out["A"]
    A["new_accounts"] = int(new_acc.sum())
    A["new_accounts_with_20_plus_games"] = int((new_acc & (total >= 20)).sum())
    r = d["r"]
    isnew = new_acc[d["pid"]]
    by = {}
    for lo_, hi_ in ((0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 300), (300, 10 ** 6)):
        m = isnew & (gidx >= lo_) & (gidx < hi_)
        if m.sum() > 1000:
            by[f"games {lo_ + 1}-{hi_ if hi_ < 10 ** 6 else ''}"] = {"slots": int(m.sum()),
                                                                     "mean_resid_pp": ci(100 * r[m], rng, 300)}
    A["new_account_residual_by_game_index"] = by
    old = ~isnew & (gidx >= 300)
    A["established_residual_pp"] = float(100 * r[old].mean())
    # smurf-like flag from first 20 games
    f20 = isnew & (gidx < 20)
    sr = np.bincount(d["pid"][f20], weights=r[f20], minlength=npl)
    sv = np.bincount(d["pid"][f20], weights=d["v"][f20], minlength=npl)
    cnt = np.bincount(d["pid"][f20], minlength=npl)
    zsc = np.where(cnt == 20, sr / np.sqrt(np.maximum(sv, 1e-9)), np.nan)
    elig = new_acc & (cnt == 20)
    smurf = elig & (zsc > 2.326)
    A["eligible_new_accounts_20_games"] = int(elig.sum())
    A["smurf_like_flagged"] = int(smurf.sum())
    A["expected_by_chance_1pct"] = float(0.01 * elig.sum())
    A["share_flagged"] = float(smurf.sum() / max(elig.sum(), 1))
    wr20 = np.bincount(d["pid"][f20], weights=d["y"][f20], minlength=npl) / np.maximum(cnt, 1)
    A["flagged_first20_winrate"] = float(wr20[smurf].mean())
    A["flagged_first20_mean_resid_pp"] = float(100 * (sr[smurf] / 20).mean())
    # later games of flagged accounts: do they stay strong? (regression to mean check)
    later = smurf[d["pid"]] & (gidx >= 20) & (gidx < 100)
    A["flagged_games_21_100_mean_resid_pp"] = ci(100 * r[later], rng, 300) if later.sum() > 100 else None
    # opponent bias
    sm_slot = smurf[d["pid"]] & (gidx < 20)
    nonsm_new = elig[d["pid"]] & ~smurf[d["pid"]] & (gidx < 20)
    n_games = int(d["g"].max()) + 1
    gteam = d["g"] * 2 + d["team"]
    has_sm = np.bincount(gteam[sm_slot], minlength=2 * n_games) > 0
    has_new = np.bincount(gteam[nonsm_new], minlength=2 * n_games) > 0
    opp = gteam ^ 1
    opp_of_sm = has_sm[opp] & ~sm_slot
    opp_of_new = has_new[opp] & ~has_sm[opp]
    mate_of_sm = has_sm[gteam] & ~sm_slot
    A["opponents_of_smurf_like_resid_pp"] = ci(100 * r[opp_of_sm], rng, 300, cluster=d["g"][opp_of_sm])
    A["opponents_of_other_new_accounts_resid_pp"] = ci(100 * r[opp_of_new], rng, 300, cluster=d["g"][opp_of_new])
    A["teammates_of_smurf_like_resid_pp"] = ci(100 * r[mate_of_sm], rng, 300, cluster=d["g"][mate_of_sm])
    A["games_with_smurf_like_first20"] = int(len(np.unique(d["g"][sm_slot])))
    A["share_of_all_games"] = float(len(np.unique(d["g"][sm_slot])) / n_games)
    print(json.dumps(A, default=float), flush=True)
    # skill estimates with those games removed from the history
    bad_game = np.zeros(n_games, bool)
    bad_game[d["g"][sm_slot]] = True
    keep = ~bad_game[d["g"]]
    post = ~d["in_sample"]
    kz = np.load(os.path.join(C.CACHE, "hs_kernels.npz"))
    bases = {k[6:]: kz[k] for k in kz.files if k.startswith("basis_")}
    Kc = C.kernel_from([bases[b] for b in KERNELS["+CF rank 2"]], kz["theta_+CF rank 2"])
    base_pm = np.load(PRED)["m"]
    qidx, S, P, _, _, _, _ = C.online_state(d, keep, post, H, r_adj)
    m2, _ = C.gp_query(S, P, d["hero"][qidx], Kc)
    del S, P
    pm2 = base_pm.copy()
    pm2[qidx] = m2
    mu = table[C.exp_bins(n_p, n_ph)]
    A["estimate_change"] = {"corr_post_slots": float(np.corrcoef(base_pm[qidx], m2)[0, 1]),
                            "mean_abs_change_pp": float(100 * np.abs(base_pm[qidx] - m2).mean()),
                            "share_slots_changed_over_0.5pp": float((np.abs(base_pm[qidx] - m2) > 0.005).mean())}
    Dm, lo, y, fit_m, test_m, boots, specs = game_level(
        d, {"skill (all history)": mu + base_pm, "skill (smurf games removed)": mu + pm2}, {}, rng)
    A["game_level"] = fit_eval(specs, Dm, lo, y, fit_m, test_m, boots, "skill (all history)")
    print(json.dumps({k: A[k] for k in ("estimate_change",)}),
          {k: (v["logloss"], v.get("gain_vs_M0")) for k, v in A["game_level"].items()}, flush=True)

    # ---------- B. parties
    B = out["B"]
    party = d["party"]
    pk = np.where(party != 0, (gteam.astype(np.int64) << 32) ^ (party % 2147483647), -1)
    sz = np.ones(len(party), np.int64)
    nz = party != 0
    u, inv, c = np.unique(pk[nz], return_inverse=True, return_counts=True)
    sz[nz] = c[inv]
    e = r - (mu + base_pm)
    B["slot_share_by_party_size"] = {str(k): float((sz == k).mean()) for k in range(1, 6)}
    for nm, val in (("raw residual r (vs WP)", r), ("after skill model e", e)):
        B[nm] = {}
        for win, m0 in (("all 2024-04+", np.ones(len(r), bool)), ("post-cutoff", post)):
            B[nm][win] = {f"party size {k}": ci(100 * val[m0 & (sz == k)], rng, 300,
                                                cluster=d["g"][m0 & (sz == k)])
                          for k in range(1, 6) if (m0 & (sz == k)).sum() > 1000}
    print(json.dumps(B, default=float), flush=True)
    # team level: partied players count and largest party, on top of the skill model
    partied = (sz >= 2).astype(float)
    big = np.zeros(2 * n_games)
    np.maximum.at(big, gteam, sz.astype(float))
    sign = np.where(d["team"] == 0, 1.0, -1.0)
    Dparty = np.bincount(d["g"], weights=sign * partied, minlength=n_games)
    t0big, t1big = big[0::2], big[1::2]
    extra = {"partied players (diff)": Dparty, "largest party (diff)": t0big - t1big,
             "full 5-stack (diff)": (t0big == 5).astype(float) - (t1big == 5).astype(float)}
    Dm, lo, y, fit_m, test_m, boots, _ = game_level(d, {"skill": mu + base_pm}, extra, rng)
    specs = {"M0": [], "skill": ["skill"], "skill + partied players": ["skill", "partied players (diff)"],
             "skill + largest party": ["skill", "largest party (diff)"],
             "skill + all party terms": ["skill", "partied players (diff)", "largest party (diff)",
                                         "full 5-stack (diff)"]}
    B["game_level"] = fit_eval(specs, Dm, lo, y, fit_m, test_m, boots, "skill")
    for k, v in B["game_level"].items():
        print(k, round(v["logloss"], 5), v.get("gain_vs_skill"), np.round(v["coef"][2:], 4), flush=True)
    with open(os.path.join(C.RESULTS, "p3_val_validity.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
