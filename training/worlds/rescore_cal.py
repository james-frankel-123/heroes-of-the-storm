"""
Paper 2 rescoring under f-cal (player.PlayerWorldsCal: the outcome-only player term from records and
scoreboard aggregates, no drafter state or model), exactly as rescore.py stage p2 does for f-out / f-cov:
the headline and tune drafts of the selected design and the support arms, family a's hero-generic truth
plus the f-cal player term under three assignments (random, human-like, best), 1,000 worlds.

  python worlds/rescore_cal.py p2 [--lsets headline,tune] [--n 1000] [--kcal K --tag T]
      -> site/cache/rescore_p2_<lset>_fcal<T>.npz, site/results/rescore_p2_cal.json (one entry per tag)
New files only; rescore.py's outputs are never touched.
"""
import os
import sys
import json
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import data as WD  # noqa: E402
import model as M  # noqa: E402
import world as W  # noqa: E402
import rescore as R  # noqa: E402

OUT_JSON = os.path.join(WD.RESULTS, "rescore_p2_cal.json")
_G = {}


def _world(args):
    wid, lset, kcal = args
    import player as PL
    D, keys = _G["D2"], _G["keys2"]
    Ws = _G.setdefault("W", W.Worlds())
    PW = _G.setdefault("PW", PL.PlayerWorldsCal(kcal))
    zg = Ws.logit("a", D, wid)
    A = R.assign_terms(PW, D, wid, np.random.default_rng([wid, 3]))
    res = []
    for nm in ("random", "human", "best"):
        v = R.p2_values(M.sig(zg + A[nm]), D, lset)
        res.append([v.get(k, np.nan) for k in keys])
    return np.array(res, np.float32)


def stage_p2(n, lsets, kcal, tag):
    from multiprocessing import get_context
    import player as PL
    kc = PL.PlayerWorldsCal(kcal).kcal
    for lset in lsets:
        out = os.path.join(WD.CACHE, f"rescore_p2_{lset}_fcal{tag}.npz")
        if os.path.exists(out):
            continue
        D = R.p2_drafts(lset)
        keys = sorted({(a, o) for a in [R.SEL] + R.P2_ARMS for o in R.P2_OPP[lset]
                       if ((D["S0"] == a) & (D["S1"] == o)).any()})
        _G.clear()
        _G.update(D2=D, keys2=keys)
        W.Worlds().base_draws([0])
        t0 = time.time()
        cells = [{"assign": x} for x in ("random", "human", "best")]
        with get_context("fork").Pool(R.NPROC) as pool:
            V = np.stack(pool.map(_world, [(w, lset, kc) for w in range(n)], chunksize=4))
        np.savez(out, V=V, cells=json.dumps(cells), keys=json.dumps(keys), kcal=kc)
        print(f"p2 {lset} family fcal{tag} (kcal {kc}): {V.shape} in {time.time() - t0:.0f}s", flush=True)


def summarize():
    res = {}
    for lset in ("headline", "tune"):
        for f in sorted(os.listdir(WD.CACHE)):
            if not (f.startswith(f"rescore_p2_{lset}_fcal") and f.endswith(".npz")):
                continue
            fam = f[len(f"rescore_p2_{lset}_"):-4]
            z = np.load(os.path.join(WD.CACHE, f))
            V, cells, keys = z["V"], json.loads(str(z["cells"])), [tuple(k) for k in json.loads(str(z["keys"]))]
            F = {"worlds": int(V.shape[0]), "kcal": float(z["kcal"]), "cells": cells, "arms": {}}
            for ci, cell in enumerate(cells):
                T = {}
                for a in [R.SEL] + R.P2_ARMS:
                    opps = [o for o in R.P2_OPP[lset] if (a, o) in keys and (R.SEL, o) in keys and o != a]
                    if not opps:
                        continue
                    va = np.mean([V[:, ci, keys.index((a, o))] for o in opps], 0)
                    vs = np.mean([V[:, ci, keys.index((R.SEL, o))] for o in opps], 0)
                    T[a] = {"value_pts": float(100 * va.mean()), "opponents": opps}
                    if a != R.SEL:
                        T[a]["minus_sel"] = R.contrast(va - vs)
                F["arms"][json.dumps(cell, sort_keys=True)] = T
            res.setdefault(lset, {})[fam] = F
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    for lset, fams in res.items():
        for fam, F in fams.items():
            for cell, T in F["arms"].items():
                for a in ("s-floor25", "s-pcal-floor25"):
                    if a in T:
                        c = T[a]["minus_sel"]
                        print(f"{lset} {fam} {cell} {a} - sel: {c['mean']:+.2f} [{c['lo95']:+.2f}, {c['hi95']:+.2f}]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["p2", "summary"])
    ap.add_argument("--lsets", default="headline,tune")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--kcal", type=float, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.stage == "p2":
        stage_p2(a.n, a.lsets.split(","), a.kcal, a.tag)
    summarize()
