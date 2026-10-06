"""
P3 follow-up to items 3 and 7: tune the (build, hero[, tier]) nowcast
shrinkage on V1 only, then check its overlap with the one-trick C term.

Nowcast (p3_b_demean.causal_cell_demean): shrunken mean of r_adj over the
same cell on earlier days of the same build (lag 1), k = noise / (m tau2_E),
tau2_E the moment estimate on E for that cell key, m on a grid.

Selection uses V1 only: V1 games split at their median day; the combiner
(logit WP, team difference of the phase-1 slot estimate, team difference of
the nowcast) is fit on early V1 and scored on late V1. The chosen (key, m)
and the current setting (m = 1, both keys) are then refit on all of V1 and
scored on V2 and OOT (sealed build 2.55.17.98025 excluded unless --final).

Overlap: on the chosen nowcast, a joint combiner with skill (phase-1 kernel
or the C concentration-mean variant of p3_b_onetrick), nowcast, the
concentration term (main share x off-main) and the forced term (main share x
main unavailable before the pick), fit on V1; each term's marginal
contribution is the V2 / OOT log-loss change when it is dropped and the
combiner refit. Slot estimates come from cache/b7_onetrick_preds.npz (written
by p3_b_onetrick.py, full run), so no GP pass is needed.

Usage (from training/): python personalization/p3_b_nowcast.py [--final]
Output: C.RESULTS/p3_b_nowcast[_final].json
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from p3_heroes import NUM_HEROES
import p3_hs_core as C
import p3_x_common as X
import p3_b_demean as DM

MULTS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--nboot", type=int, default=200)
    a = ap.parse_args()
    t0 = time.time()
    H = NUM_HEROES
    d = X.load_ext()
    days, post = d["day"], d["post"]
    e_mask = d["in_sample"] & (days >= C.day_of(C.E_START))
    n_p, n_ph = C.experience_counts(d)
    table = C.fit_experience(d["r"], n_p, n_ph, e_mask)
    r_adj = d["r"] - table[C.exp_bins(n_p, n_ph)]
    build, bnames = DM.row_builds(d)
    tier = DM.row_tiers(d)
    keys = {"build x hero": build * H + d["hero"],
            "build x hero x tier": (build * H + d["hero"]) * DM.NTIER + tier}
    noise = float(np.mean(d["v"]))
    tau = {k: DM.estimate_tau2(r_adj, d["v"], kk, e_mask)[0] for k, kk in keys.items()}

    P = np.load(os.path.join(C.CACHE, "b7_onetrick_preds.npz"))
    names = [str(x) for x in P["names"]]
    qidx = P["qidx"]
    s_of = {nm: P["mu"].astype(np.float64) + P["m_" + str(names.index(nm))] for nm in
            ("phase-1 kernel", "C concentration mean")}
    offmain = (d["hero"][qidx] != P["main"]) & (P["main"] >= 0)
    conc = P["share"] * offmain
    forced = np.where(P["tot"] >= 30, P["share"], 0.0) * P["unavailable"]

    snap_post = (~post) & (~d["in_sample"])
    _, fr = np.unique(d["g"][snap_post], return_index=True)
    med = np.median(days[snap_post][fr])
    v1, v2 = snap_post & (days < med), snap_post & (days >= med)
    sealed = post & (build == (bnames.index(DM.SEALED_BUILD) if DM.SEALED_BUILD in bnames else -1))
    oot = post & (days > X.SNAP_LAST_DAY) & (a.final | ~sealed)
    n_games, y, wp0, cnt, gday = X.game_arrays(d)
    lo = np.log(wp0 / (1 - wp0))
    qc = np.bincount(d["g"][qidx], minlength=n_games)
    gv = {}
    for wn, mw in (("V1", v1), ("V2", v2), ("OOT", oot)):
        gm = np.zeros(n_games, bool)
        gm[d["g"][mw]] = True
        gv[wn] = gm & (cnt == 10) & (qc == 10)
    v1med = np.median(gday[gv["V1"]])
    gv["V1 early"] = gv["V1"] & (gday < v1med)
    gv["V1 late"] = gv["V1"] & (gday >= v1med)
    g_q, t_q = d["g"][qidx], d["team"][qidx]
    TD = lambda x: X.team_diff(x, g_q, t_q, n_games)
    D = {"skill phase-1": TD(s_of["phase-1 kernel"]), "skill C": TD(s_of["C concentration mean"]),
         "concentration": TD(conc), "forced": TD(forced)}

    def fit_score(cols, fit, test):
        Xm = np.column_stack([lo] + cols)
        w = C.fit_logistic(Xm[fit], y[fit])
        return X.logloss(C.predict(w, Xm), y), w

    out = {"tau2_E_pp2": {k: 1e4 * v for k, v in tau.items()}, "v1_split_day": str(np.datetime64(int(v1med), "D")),
           "oot_scope": "all OOT (final)" if a.final else f"OOT without {DM.SEALED_BUILD}", "grid": {}}
    l_base, _ = fit_score([D["skill phase-1"]], gv["V1 early"], gv["V1 late"])
    idx_late = np.flatnonzero(gv["V1 late"])
    nowD = {}
    for kn, kk in keys.items():
        for m in MULTS:
            _, now, _ = DM.causal_cell_demean(d, r_adj, kk, m * tau[kn], lag=1, noise=noise)
            Dn = TD(now[qidx])
            l1, w = fit_score([D["skill phase-1"], Dn], gv["V1 early"], gv["V1 late"])
            gain = float((l_base[idx_late] - l1[idx_late]).mean())
            out["grid"][f"{kn} | m={m:g}"] = {"late_V1_gain": gain, "k_games": noise / (m * tau[kn]),
                                             "nowcast_coef": float(w[3])}
            nowD[(kn, m)] = Dn
            print(f"  {kn:20s} m={m:<5g} k={noise / (m * tau[kn]):7.0f} late-V1 gain {gain:+.5f} coef {w[3]:.2f}",
                  flush=True)
    best = max(out["grid"], key=lambda s: out["grid"][s]["late_V1_gain"])
    bk, bm = best.split(" | m=")
    bm = float(bm)
    out["chosen"] = best
    print(f"chosen: {best} ({time.time() - t0:.0f}s)", flush=True)

    # V2 / OOT at the chosen and the current settings (combiner fit on all V1)
    l0, _ = fit_score([], gv["V1"], None)
    lsk, _ = fit_score([D["skill phase-1"]], gv["V1"], None)
    out["eval"] = {}
    for lab, kk in ((f"chosen: {best}", (bk, bm)), ("current: build x hero x tier | m=1", ("build x hero x tier", 1.0)),
                    ("current: build x hero | m=1", ("build x hero", 1.0))):
        l1, w = fit_score([D["skill phase-1"], nowD[kk]], gv["V1"], None)
        r = {"coef": w.tolist()}
        for wn in ("V2", "OOT"):
            idx = np.flatnonzero(gv[wn])
            g, ci = X.boot_gain(l0, l1, idx, n=a.nboot)
            dg, dci = X.boot_gain(lsk, l1, idx, n=a.nboot)
            r[wn] = {"gain_vs_WP": g, "ci": ci, "nowcast_gain_over_skill": dg, "ci_nowcast": dci}
        out["eval"][lab] = r
        print(f"  {lab:40s} V2 +now {r['V2']['nowcast_gain_over_skill']:+.5f} {r['V2']['ci_nowcast']}  "
              f"OOT +now {r['OOT']['nowcast_gain_over_skill']:+.5f} {r['OOT']['ci_nowcast']}", flush=True)

    # overlap: joint combiners and leave-one-out
    Dnow = nowD[(bk, bm)]
    out["joint"] = {}
    for sk in ("skill phase-1", "skill C"):
        terms = {"skill": D[sk], "nowcast": Dnow, "concentration": D["concentration"], "forced": D["forced"]}
        lf, wf = fit_score(list(terms.values()), gv["V1"], None)
        res = {"coef": dict(zip(["const", "logit WP"] + list(terms), wf.tolist()))}
        for wn in ("V2", "OOT"):
            idx = np.flatnonzero(gv[wn])
            g, ci = X.boot_gain(l0, lf, idx, n=a.nboot)
            res[wn] = {"full_gain_vs_WP": g, "ci": ci, "marginal": {}}
            for t in terms:
                ld, _ = fit_score([v for k, v in terms.items() if k != t], gv["V1"], None)
                mg, mci = X.boot_gain(ld, lf, idx, n=a.nboot)
                res[wn]["marginal"][t] = {"gain": mg, "ci": mci}
        out["joint"][sk] = res
        print(f"  joint ({sk}): V2 {res['V2']['full_gain_vs_WP']:+.5f} OOT {res['OOT']['full_gain_vs_WP']:+.5f} | "
              + " ".join(f"{t} {res['V2']['marginal'][t]['gain']:+.5f}/{res['OOT']['marginal'][t]['gain']:+.5f}"
                         for t in terms), flush=True)
    fn = os.path.join(C.RESULTS, "p3_b_nowcast" + ("_final" if a.final else "") + ".json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"wrote {fn} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
