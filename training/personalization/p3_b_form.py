"""
P3 review item 11 (REVIEW section 3.4): is "form" skill or matchmaking?

If the matchmaker under-reacts to a run of results, a player on a winning run
is rated above his long-run level and gets harder lobbies, and recent
residuals predict the next one. Test: does the form term shrink once the
causal MMR terms are in?

Per-slot values (all from cache/b13_joint_feats.npz, built by p3_b_joint.py,
or rebuilt here with its functions):
  s         headline personal term (lag-1 offset + "+CF rank 2" GP mean)
  (a) form  EWMA residual (y - WP) over earlier-day games, half-lives 100
            and 10 games (p3_sd_eval.momentum_features); second definition:
            the strict same-day form term of p3_b_joint (sd_mean, sd_logk)
  (b) gap   long-run causal player MMR minus current causal player MMR, /100.
            current = causal at-game MMR (cache/mmr_at_game.npz: latest
            stamp parsed before the game started); long-run = EWMA (half-life
            100 games) of the player's current values at his games on
            earlier days. Variant "day-first": current = the causal MMR at
            the player's first game of the day, so no same-day result enters.
  (c) lobby team-sum difference of current causal player MMR, /100 (missing
            values take the player's long-run value, then the mean)
Slot level: OLS of e = r - s on (a), (b) and own-team minus opposing-team
sum, V1+V2 slots and OOT slots, player-cluster and game-cluster SEs.
Game level (p3_b_joint protocol: logistic on logit WP + D[s] + terms, fit on
V1, scored on V2 and on OOT before the sealed build 2.55.17.98025; --final
adds it): gains of form alone, MMR terms alone and both over the headline,
the form coefficient with and without the MMR terms, and the share of the
form gain explained = 1 - gain(form | headline + MMR) / gain(form | headline).

Returning players (REVIEW 2.5 item 2): slots whose previous game (play
order) was 30 to 180 days earlier, first game back and games 2-3 back, among
players with 100+ earlier-day games, from 2024-07-01. For each, mean causal
player MMR of opponents minus teammates (self excluded), and own team sum
minus opposing sum, against active controls (previous game < 3 days ago,
10+ games since any 30-day break), raw and within own-MMR deciles;
player-cluster CIs. Also r = y - WP for the same slots and how much of it a
slot-level fit of r on the lobby difference predicts.

Usage (from training/): python personalization/p3_b_form.py [--sample 0.1] [--final]
Output: results/p3_b_form.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from numba import njit

import p3_hs_core as C
import p3_x_common as X
import p3_fix_counts as F
import p3_b_joint as J

log = J.log


@njit(cache=True)
def _ewma_lag1(starts, ends, day, val, hl, out):
    """EWMA (games clock) of val over the player's games on earlier days,
    NaN values skipped; NaN if no earlier valid value."""
    a = 1.0 - np.exp(-np.log(2.0) / hl)
    for p in range(starts.shape[0]):
        e = 0.0
        w = 0.0
        i = starts[p]
        while i < ends[p]:
            j = i
            while j < ends[p] and day[j] == day[i]:
                j += 1
            for q in range(i, j):
                out[q] = e / w if w > 0 else np.nan
            for q in range(i, j):
                if not np.isnan(val[q]):
                    e = (1 - a) * e + a * val[q]
                    w = (1 - a) * w + a
            i = j


def long_run(dv, cur, hl=100.0):
    o, st, en = J.player_blocks(dv, dv["replay_id"])
    out = np.empty(len(o))
    _ewma_lag1(st, en, dv["day"][o].astype(np.int64), cur[o].astype(np.float64), hl, out)
    res = np.empty(len(o))
    res[o] = out
    return res


def day_first(d, cur, t_start):
    """Causal MMR at the player's first game of the day (play order)."""
    o = np.lexsort((d["replay_id"], t_start, d["day"], d["pid"]))
    k = d["pid"][o].astype(np.int64) * 100000 + d["day"][o]
    first = np.r_[True, k[1:] != k[:-1]]
    idx = np.maximum.accumulate(np.where(first, np.arange(len(o)), 0))
    out = np.empty(len(o))
    out[o] = cur[o][idx]
    return out


def team_sums(d, x, n_games):
    gt = d["g"].astype(np.int64) * 2 + d["team"]
    s = np.bincount(gt, weights=x, minlength=2 * n_games)
    c = np.bincount(gt, minlength=2 * n_games)
    return s, c


def ols_cluster(Xm, yv, clusters):
    """OLS with cluster-robust SEs for each clustering given."""
    X1 = np.column_stack([np.ones(len(Xm)), Xm])
    XtX = X1.T @ X1
    b = np.linalg.solve(XtX, X1.T @ yv)
    e = yv - X1 @ b
    inv = np.linalg.inv(XtX)
    ses = {}
    for nm, cl in clusters.items():
        _, ci = np.unique(cl, return_inverse=True)
        G = np.zeros((ci.max() + 1, X1.shape[1]))
        np.add.at(G, ci, X1 * e[:, None])
        V = inv @ (G.T @ G) @ inv
        ses[nm] = np.sqrt(np.diag(V))
    return b, ses


def slot_level(d, Fz, rows, lobby_own, gap, tag):
    out = {}
    cols = {"EWMA resid hl100": Fz["ewma100"], "EWMA resid hl10": Fz["_ewma10"],
            "MMR gap (long-run - current) /100": gap, "lobby: own team sum - opp sum /100": lobby_own}
    for win, m in (("V1+V2", rows["V1"] | rows["V2"]), ("OOT", rows["OOT"])):
        e = d["r"] - Fz["s"]
        ok = m & np.isfinite(e)
        for spec, names in (("form only", ["EWMA resid hl100", "EWMA resid hl10"]),
                            ("MMR only", ["MMR gap (long-run - current) /100", "lobby: own team sum - opp sum /100"]),
                            ("both", list(cols))):
            Xm = np.column_stack([cols[c][ok] for c in names])
            b, ses = ols_cluster(Xm, e[ok], {"player": d["pid"][ok], "game": d["g"][ok]})
            out[f"{win} | {spec}"] = {
                nm: {"coef": float(b[i + 1]),
                     "ci_player_cluster": [float(b[i + 1] - 1.96 * ses["player"][i + 1]),
                                           float(b[i + 1] + 1.96 * ses["player"][i + 1])],
                     "ci_game_cluster": [float(b[i + 1] - 1.96 * ses["game"][i + 1]),
                                         float(b[i + 1] + 1.96 * ses["game"][i + 1])]}
                for i, nm in enumerate(names)}
            out[f"{win} | {spec}"]["slots"] = int(ok.sum())
        log(f"slot {tag} {win}:", {k: {n: round(v["coef"], 5) for n, v in x.items() if isinstance(v, dict)}
                                   for k, x in out.items() if k.startswith(win)})
    return out


def game_level(d, Fz, games, gd, D_extra, tag):
    """Specs on top of the headline; gains, coefficients and the share of
    the form gain explained by the MMR terms."""
    des = J.Design(d, Fz, games, gd)
    for k, v in D_extra.items():
        des.D[k] = v
        des.sd[k] = float(v[games["V1"]].std()) or 1.0
    y = des.y
    fit = games["V1"]
    tests = [k for k in games if k != "V1"]
    clus = J.clusters_first_player(d, gd[0])
    week = (gd[4] // 7).astype(np.int64)
    boots = {k: J.Boot(np.flatnonzero(games[k]), clus, week) for k in tests}
    form_ewma = ["ewma100", "_ewma10"]
    form_sd = ["sd_mean", "sd_logk"]
    mmr = ["gap", "imb"]
    specs = {"headline": [], "form EWMA": form_ewma, "form EWMA-100 only": ["ewma100"],
             "MMR gap + imbalance": mmr, "MMR gap only": ["gap"], "lobby imbalance only": ["imb"],
             "form EWMA + MMR": form_ewma + mmr, "form EWMA-100 + MMR": ["ewma100"] + mmr,
             "form same-day strict": form_sd, "MMR (day-first current)": ["gap_df", "imb_df"],
             "form same-day strict + MMR": form_sd + mmr,
             "form EWMA + MMR (day-first current)": form_ewma + ["gap_df", "imb_df"],
             "form same-day strict + MMR (day-first current)": form_sd + ["gap_df", "imb_df"]}
    lls, fits = {}, {}
    for nm, cols in specs.items():
        Xm = np.column_stack([des.lo, des.D["s"]] + [des.D[c] for c in cols])
        w, Hm = J.fit_ridge(Xm[fit], y[fit], np.zeros(Xm.shape[1]))
        lls[nm] = X.logloss(J.predict(w, Xm), y)
        se = np.sqrt(np.diag(np.linalg.inv(Hm)))
        fits[nm] = {c: {"coef": float(w[i + 3]), "ci_x1.3": [float(w[i + 3] - 1.96 * 1.3 * se[i + 3]),
                                                              float(w[i + 3] + 1.96 * 1.3 * se[i + 3])]}
                    for i, c in enumerate(cols)}
        fits[nm]["s"] = float(w[2])
        fits[nm]["logit WP"] = float(w[1])
    out = {"coef_V1": fits, "gains": {}, "share_of_form_gain_explained": {}}
    for nm in specs:
        if nm == "headline":
            continue
        out["gains"][nm] = {t: b.gain(lls["headline"], lls[nm]) for t, b in boots.items()}
        log(f"  {tag} {nm:48s} " + "  ".join(f"{t} {J.fmt(v)}" for t, v in out["gains"][nm].items()))
    for fname, mname, both in (("form EWMA", "MMR gap + imbalance", "form EWMA + MMR"),
                               ("form EWMA-100 only", "MMR gap + imbalance", "form EWMA-100 + MMR"),
                               ("form same-day strict", "MMR gap + imbalance", "form same-day strict + MMR"),
                               ("form EWMA", "MMR (day-first current)", "form EWMA + MMR (day-first current)"),
                               ("form same-day strict", "MMR (day-first current)",
                                "form same-day strict + MMR (day-first current)")):
        r = {}
        for t, b in boots.items():
            rt = b.ratio(lls[mname], lls[both], lls["headline"], lls[fname])
            r[t] = {"form gain alone": b.gain(lls["headline"], lls[fname]),
                    "form gain given MMR": b.gain(lls[mname], lls[both]),
                    "share explained": 1 - rt["ratio"],
                    "share explained ci_replay": [1 - rt["ci_replay"][1], 1 - rt["ci_replay"][0]]}
        out["share_of_form_gain_explained"][f"{fname} | {mname}"] = r
        log(f"  {tag} share explained {fname} | {mname}: " +
            "  ".join(f"{t} {v['share explained']:.2f} [{v['share explained ci_replay'][0]:.2f},"
                      f"{v['share explained ci_replay'][1]:.2f}]" for t, v in r.items()))
    return out


def returning(d, M, cur, lr, t_end, t_start, n_games, s_vals):
    """Lobby tilt for players back from a 30-180 day break."""
    o = np.lexsort((d["replay_id"], t_start, d["pid"]))
    pid = d["pid"][o]
    newp = np.r_[True, pid[1:] != pid[:-1]]
    gap = np.r_[np.nan, (t_start[o][1:] - t_end[o][:-1]) / 86400.0]
    gap[newp] = np.nan
    brk = ~newp & (gap >= 30)
    idx = np.arange(len(o))
    last_brk = np.maximum.accumulate(np.where(brk, idx, -1))
    pstart = np.maximum.accumulate(np.where(newp, idx, 0))
    since = np.where(last_brk >= pstart, idx - last_brk, -1)
    brk_len = np.where(last_brk >= 0, gap[np.maximum(last_brk, 0)], np.nan)
    G = {k: np.empty(len(o)) for k in ("gap", "since", "brk_len")}
    for k, v in (("gap", gap), ("since", since), ("brk_len", brk_len)):
        G[k][o] = v
    n_p = F.lag1_key_counts(d, d["pid"])
    est = (n_p >= 100) & (d["day"] >= C.day_of("2024-07-01"))
    x = np.where(np.isfinite(cur), cur, np.nan)
    ok = np.isfinite(x)
    gt = d["g"].astype(np.int64) * 2 + d["team"]
    s_t = np.bincount(gt, weights=np.where(ok, x, 0), minlength=2 * n_games)
    c_t = np.bincount(gt, weights=ok, minlength=2 * n_games)
    own_s, own_c = s_t[gt], c_t[gt]
    opp_s, opp_c = s_t[gt ^ 1], c_t[gt ^ 1]
    mates = (own_s - np.where(ok, x, 0)) / np.maximum(own_c - ok, 1)
    opps = opp_s / np.maximum(opp_c, 1)
    full = (own_c - ok >= 3) & (opp_c >= 4)
    tilt = np.where(full, opps - mates, np.nan)
    lobby_own = np.where(full & ok, (own_s / np.maximum(own_c, 1) - opps) * 5, np.nan)
    in_brk = (G["brk_len"] >= 30) & (G["brk_len"] <= 180)
    groups = {"first game back (30-180 d break)": est & in_brk & (G["since"] == 0),
              "games 2-3 back": est & in_brk & (G["since"] >= 1) & (G["since"] <= 2),
              "games 4-10 back": est & in_brk & (G["since"] >= 3) & (G["since"] <= 9),
              "control: active (prev game < 3 d, 10+ games since any break)":
                  est & (G["gap"] < 3) & ((G["since"] < 0) | (G["since"] >= 10))}
    ctrl = groups["control: active (prev game < 3 d, 10+ games since any break)"]
    rng = np.random.RandomState(0)

    def cl_ci(v, cl, B=200):
        u, inv = np.unique(cl, return_inverse=True)
        sm = np.bincount(inv, weights=v)
        cn = np.bincount(inv).astype(float)
        bs = []
        for _ in range(B):
            w = np.bincount(rng.randint(0, len(u), len(u)), minlength=len(u))
            bs.append((w * sm).sum() / (w * cn).sum())
        return [float(x) for x in np.percentile(bs, [2.5, 97.5])]
    dec_edges = np.nanpercentile(x[ctrl], np.arange(10, 100, 10))
    dec = np.searchsorted(dec_edges, np.where(ok, x, 0))
    out = {}
    # slot-level slope of r on the lobby difference (own - opp team sum /100)
    sl = ctrl & np.isfinite(lobby_own)
    beta = float(np.cov(d["r"][sl], lobby_own[sl] / 100)[0, 1] / np.var(lobby_own[sl] / 100))
    out["slope_r_per_100_lobby_mmr (controls)"] = beta
    for nm, m in groups.items():
        mm = m & np.isfinite(tilt) & ok
        res = {"slots": int(mm.sum()), "players": int(len(np.unique(d["pid"][mm])))}
        if mm.sum() < 200:
            out[nm] = res
            continue
        res["own causal MMR mean"] = float(x[mm].mean())
        res["own causal MMR minus own long-run"] = float(np.nanmean(x[mm] - lr[mm]))
        res["own causal MMR stamp age, median days"] = float(np.median(M["player_age_s"][mm & (M["player_age_s"] >= 0)]) / 86400)
        res["opponents minus teammates, mean MMR"] = float(tilt[mm].mean())
        res["opponents minus teammates ci_player"] = cl_ci(tilt[mm], d["pid"][mm])
        res["own team sum minus opp sum"] = float(lobby_own[mm].mean())
        res["own team sum minus opp sum ci_player"] = cl_ci(lobby_own[mm], d["pid"][mm])
        res["r = y - WP, pp"] = 100 * float(d["r"][mm].mean())
        res["r ci_player, pp"] = [100 * v for v in cl_ci(d["r"][mm], d["pid"][mm])]
        res["r predicted from the lobby difference, pp"] = 100 * beta * float(lobby_own[mm].mean()) / 100
        hs = mm & np.isfinite(s_vals)
        if hs.sum() > 200:
            res["e = r - s (V/OOT rows only), pp"] = 100 * float((d["r"][hs] - s_vals[hs]).mean())
            res["e slots"] = int(hs.sum())
        if nm != "control: active (prev game < 3 d, 10+ games since any break)":
            cm = ctrl & np.isfinite(tilt) & ok
            diff, wsum = 0.0, 0
            for k_ in range(10):
                a, b = mm & (dec == k_), cm & (dec == k_)
                if a.sum() and b.sum():
                    diff += a.sum() * (tilt[a].mean() - tilt[b].mean())
                    wsum += a.sum()
            res["tilt minus control, within own-MMR deciles"] = diff / max(wsum, 1)
            res["tilt minus control, raw"] = float(tilt[mm].mean() - tilt[cm].mean())
        out[nm] = res
        log(f"returning {nm}: " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                               for k, v in res.items() if not isinstance(v, list)}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=float, default=None)
    ap.add_argument("--final", action="store_true", help="also score the sealed build 2.55.17.98025")
    a = ap.parse_args()
    t0 = time.time()
    d = X.load_ext()
    med, rows, games, gd = J.windows(d, a.sample, a.final)
    n_games = gd[0]
    Fz = J.load_features(d, a.sample)
    if Fz is None:
        log("no cached p3_b_joint features for this sample; building them")
        Fz, _ = J.build_features(d, rows, a.sample)
    J.add_nowcast(d, Fz, rows)  # the joint design carries every group
    t_end, t_start, _ = J.game_times(d)
    M = J.causal_mmr(d)
    cur = M["player"]
    cur_df = day_first(d, cur, t_start)
    n = len(d["pid"])
    lr = np.full(n, np.nan)
    for view, qm in (("snapshot", rows["V1"] | rows["V2"]), ("all", rows["OOT"])):
        vm = ~d["post"] if view == "snapshot" else np.ones(n, bool)
        dv = J.subset(d, vm)
        dest = np.flatnonzero(vm)[qm[vm]]
        lr[dest] = long_run(dv, cur[vm])[qm[vm]]
    # long-run for every row (history view = all rows), for the returning check
    lr_all = long_run(d, cur)
    gm = np.nanmean(cur)
    out = {"split_day": med, "sample": a.sample, "final": a.final,
           "coverage_query_rows": {"current causal player MMR": float(np.isfinite(cur[rows["V2"]]).mean()),
                                   "long-run": float(np.isfinite(lr[rows["V2"]]).mean())},
           "current_source_same_day_share_V2": float(np.mean(
               (M["player_age_s"][rows["V2"]] >= 0) &
               (M["player_age_s"][rows["V2"]] < (t_start[rows["V2"]] % 86400))))}
    D_extra = {}
    slot = {}
    for tag, cu in (("current", cur), ("day-first", cur_df)):
        c_f = np.where(np.isfinite(cu), cu, np.where(np.isfinite(lr), lr, gm))
        l_f = np.where(np.isfinite(lr), lr, c_f)
        gap = (l_f - c_f) / 100.0
        s_t, _ = team_sums(d, c_f / 100.0, n_games)
        gt = d["g"].astype(np.int64) * 2 + d["team"]
        lobby_own = s_t[gt] - s_t[gt ^ 1]
        sfx = "" if tag == "current" else "_df"
        D_extra["gap" + sfx] = X.team_diff(gap, d["g"], d["team"], n_games)
        D_extra["imb" + sfx] = s_t[0::2] - s_t[1::2]
        D_extra["_ewma10"] = X.team_diff(np.nan_to_num(Fz["_ewma10"]), d["g"], d["team"], n_games)
        slot[tag] = slot_level(d, Fz, rows, lobby_own, gap, tag)
    out["slot_level_e_on_terms"] = slot
    out["game_level"] = game_level(d, Fz, games, gd, D_extra, "game")
    out["returning_players"] = returning(d, M, cur, lr_all, t_end, t_start, n_games, Fz["s"])
    name = "p3_b_form" + ("_final" if a.final else "") + ("" if a.sample is None else f"_sample{a.sample}") + ".json"
    with open(os.path.join(C.RESULTS, name), "w") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {os.path.join(C.RESULTS, name)}; done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
