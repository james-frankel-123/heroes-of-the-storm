"""
P3 H1 follow-up: a jump only where the hero's aggregate win rate moved a
lot at the boundary (|dWR| >= 2pp or >= 4pp between the capped pre and post
segments, all players), against a placebo of the same number of
(boundary, hero) pairs drawn from heroes whose aggregate moved < 1pp.
Descriptive likelihood test on the random-40k E sample (dWR uses both
sides of the boundary, so this is not a causal predictor).

Run (from training/, 4 threads):
  NUMBA_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_ph_patch_big.py
Output: results/p3_ph_patch_big.json
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C
import p3_sd_kalman as K
import p3_ph_patch as PP
from p3_hs_fit import prepare


def main():
    d = C.load_slots()
    meta = C.hero_meta(d["hero_names"])
    names = [str(x) for x in d["hero_names"]]
    kz = np.load(K.KOUT)
    e_mask, n_p, n_ph, table, r_adj = prepare(d)
    build = PP.slot_builds(d)
    bnds, builds = PP.boundaries(names)
    nb = len(builds)
    days = d["day"]
    dwr = np.zeros((len(bnds), 90))
    for k, b in enumerate(bnds):
        lo = bnds[k - 1]["pos"] if k > 0 else 0
        hi = bnds[k + 1]["pos"] if k + 1 < len(bnds) else 10 ** 6
        pre = (build >= lo) & (build < b["pos"])
        post = (build >= b["pos"]) & (build < hi)
        if not pre.any() or not post.any():
            continue
        t_b = days[post].min()
        pre &= days >= t_b - PP.CAP_DAYS
        post &= days < t_b + PP.CAP_DAYS
        w0 = np.bincount(d["hero"][pre], weights=d["y"][pre], minlength=90) / np.maximum(
            np.bincount(d["hero"][pre], minlength=90), 1)
        w1 = np.bincount(d["hero"][post], weights=d["y"][post], minlength=90) / np.maximum(
            np.bincount(d["hero"][post], minlength=90), 1)
        dwr[k] = w1 - w0
    rng = np.random.RandomState(3)
    fits = json.load(open(os.path.join(C.RESULTS, "p3_sd_fit.json")))
    rs = np.random.RandomState(11)
    fitp = kz["fit_players"]
    nE = np.bincount(d["pid"][e_mask], minlength=int(d["n_players"]))
    samp = np.zeros(int(d["n_players"]), bool)
    samp[rs.choice(np.flatnonzero(fitp & (nE > 0)), 40000, replace=False)] = True
    p = fits["random 40k"]["fits"]["drift days"]["params"]
    jf = PP.JumpFilter(d, meta, kz["V_cf2"], r_adj, build, e_mask & samp[d["pid"]])
    base = jf.run_jump(p, np.zeros((nb + 1, 90)))[2].sum()
    out = {}
    small = np.argwhere(np.abs(dwr) < 0.01)
    for thr in (0.02, 0.04):
        big = np.argwhere(np.abs(dwr) >= thr)
        pick = small[rng.choice(len(small), len(big), replace=False)]
        for lab, pairs in (("big shift", big), ("placebo (<1pp, same count)", pick)):
            for sdpp in (1.0, 2.0, 3.0, 5.0):
                t = np.zeros((nb + 1, 90))
                for k, h in pairs:
                    t[bnds[k]["pos"], h] = (sdpp / 100) ** 2
                ll = jf.run_jump(p, t)[2].sum() - base
                out[f"|dWR|>={100 * thr:.0f}pp | {lab} | J sd {sdpp}pp"] = {"pairs": int(len(pairs)),
                                                                            "dLL": float(ll)}
                print(f"|dWR|>={100 * thr:.0f}pp {lab} ({len(pairs)} pairs) J {sdpp}pp: dLL {ll:+.1f}",
                      flush=True)
    with open(os.path.join(C.RESULTS, "p3_ph_patch_big.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
