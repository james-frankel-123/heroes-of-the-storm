"""
Fit the world model (model.py) on the paper-1 training split. Stages (from training/, p1site env):
  hp       base-model hyperparameters on the val split: a grid over (sS = sK, r, sE), one process
           per config (WORLDS_THREADS threads each) -> site/results/fit_hp.json
  base     the chosen config: MAP, Gauss-Newton precision, Cholesky -> site/cache/base.npz,
           base_chol.npy; summary -> site/results/fit_base.json
  layers   each higher-order layer conditional on the base MAP, prior SD by diag-Laplace evidence
           -> site/cache/layer_<name>.npz; summary -> site/results/fit_layers.json
  time     quick timing on a 200k-game subsample
"""
import os
import sys
import json
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as WD  # noqa: E402
import model as M  # noqa: E402

HP_GRID = [dict(sP=sP, r=r, sE=sE) for sP in (0.03, 0.05, 0.08) for r, sE in ((0, 0.1), (8, 0.25), (8, 0.4))]
LAYER_GRID = {
    "triple": [0.005, 0.01, 0.02, 0.03, 0.045, 0.065, 0.09],
    "quartet": [0.005, 0.01, 0.02, 0.03, 0.045, 0.065, 0.09],
    "five": [0.005, 0.01, 0.02, 0.04, 0.06, 0.09, 0.13],
    "cross": [0.002, 0.004, 0.007, 0.01, 0.015, 0.02, 0.03],
    "hmt": [0.005, 0.01, 0.02, 0.03, 0.045, 0.065, 0.09],
}


def dims():
    maps, tiers = WD.meta()
    return len(maps) + 1, len(tiers) + 1


def _hp_path(cfg):
    return os.path.join(WD.CACHE, "hp_sP{sP}_r{r}_sE{sE}.json".format(**cfg))


def _hp_one(cfg):
    if os.path.exists(_hp_path(cfg)):
        return json.load(open(_hp_path(cfg)))
    D, Dv = WD.load("train"), WD.load("val")
    nm, nt = dims()
    t0 = time.time()
    B, th = M.fit_base(D, nm, nt, max(cfg["r"], 1), cfg["sP"], cfg["sP"], cfg["sE"] if cfg["r"] else 1e-4)
    out = dict(cfg, val_nll=M.nll(M.sig(B.logit(th, Dv)), Dv["y"]), train_nll=M.nll(M.sig(B.logit(th, D)), D["y"]),
               secs=time.time() - t0)
    json.dump(out, open(_hp_path(cfg), "w"))
    print(json.dumps(out), flush=True)
    return out


def stage_hp(procs):
    from multiprocessing import get_context
    with get_context("spawn").Pool(procs) as pool:
        res = pool.map(_hp_one, HP_GRID)
    best = min(res, key=lambda d: d["val_nll"])
    json.dump({"grid": res, "best": best}, open(os.path.join(WD.RESULTS, "fit_hp.json"), "w"), indent=1)
    print("best", best)


def stage_base(workers):
    if os.path.exists(os.path.join(WD.RESULTS, "fit_base.json")) and os.path.exists(M.CHOL_NPY):
        return
    # The plan requires a low-rank triple term: the best val config among r = 8 (rule fixed before the
    # grid finished; it is also the best config overall, sP 0.05, sE 0.4).
    grid = json.load(open(os.path.join(WD.RESULTS, "fit_hp.json")))["grid"]
    hp = min([g for g in grid if g["r"] == 8], key=lambda d: d["val_nll"])
    D = WD.load("train")
    nm, nt = dims()
    r = max(hp["r"], 1)
    sE = hp["sE"] if hp["r"] else 1e-4
    B, th = M.fit_base(D, nm, nt, r, hp["sP"], hp["sP"], sE, iters=1000)
    t0 = time.time()
    Lam = M.ggn_precision(B, th, len(D["y"]), workers=workers)
    Lam[np.diag_indices_from(Lam)] += B.prior_prec(hp["sP"], hp["sP"], sE)
    print(f"GGN precision {B.P}x{B.P}: {time.time() - t0:.0f}s", flush=True)
    import scipy.linalg as sl
    L = sl.cholesky(Lam, lower=True)
    np.save(M.CHOL_NPY, L)
    cfg = dict(nm=nm, nt=nt, r=r, sS=hp["sP"], sK=hp["sP"], sE=sE, P=B.P)
    np.savez(M.BASE_NPZ, theta=th, cfg=json.dumps(cfg))
    # posterior SD per block (diag of the inverse from the Cholesky factor)
    Li = sl.solve_triangular(L, np.eye(B.P), lower=True)
    sd = np.sqrt((Li ** 2).sum(0))
    out = {"cfg": cfg, "train_games": int(len(D["y"])), "post_sd_median": {}, "map_abs_median": {}}
    for k, (a, b) in B.off.items():
        out["post_sd_median"][k] = float(np.median(sd[a:b]))
        out["map_abs_median"][k] = float(np.median(np.abs(th[a:b])))
    for s in ("val", "test", "NODRIFT"):
        Ds = WD.load(s)
        out[f"{s}_nll"] = M.nll(M.sig(B.logit(th, Ds)), Ds["y"])
    json.dump(out, open(os.path.join(WD.RESULTS, "fit_base.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def stage_layers(names):
    B, th, cfg = M.load_base()
    D = WD.load("train")
    off = B.logit(th, D)
    sp = os.path.join(WD.RESULTS, "fit_layers.json")
    summ = json.load(open(sp)) if os.path.exists(sp) else {}
    for layer in names:
        if layer in summ and os.path.exists(M.layer_npz(layer)):
            continue
        t0 = time.time()
        L = M.fit_layer(layer, D, off, cfg["nm"], cfg["nt"], LAYER_GRID[layer])
        info = {k: L[k] for k in ("layer", "s", "s_hi95", "s_lo95", "cells_seen", "grid")}
        np.savez(M.layer_npz(layer), keys=L["keys"], n=L["n"], mu=L["mu"], sd=L["sd"], info=json.dumps(info))
        # held-out check: does the layer (posterior mean, unseen cells 0) help on val / NODRIFT?
        for s in ("val", "NODRIFT"):
            Ds = WD.load(s)
            kp, kn = M.keys(layer, Ds, cfg["nm"], cfg["nt"])
            look = lambda k: np.where(L["keys"][np.clip(np.searchsorted(L["keys"], k), 0, len(L["keys"]) - 1)] == k,
                                      L["mu"][np.clip(np.searchsorted(L["keys"], k), 0, len(L["keys"]) - 1)], 0.0)
            z0 = B.logit(th, Ds)
            z1 = z0 + look(kp).sum(1) - look(kn).sum(1)
            info[f"{s}_nll_base"] = M.nll(M.sig(z0), Ds["y"])
            info[f"{s}_nll_with_layer"] = M.nll(M.sig(z1), Ds["y"])
        info["secs"] = time.time() - t0
        summ[layer] = info
        json.dump(summ, open(sp, "w"), indent=1)
        print(layer, json.dumps({k: v for k, v in info.items() if k != "grid"}), flush=True)


def stage_time():
    D = WD.load("train")
    D = WD.subset(D, slice(0, 200000))
    nm, nt = dims()
    t0 = time.time()
    B, th = M.fit_base(D, nm, nt, 8, 0.05, 0.05, 0.25, iters=50)
    print("50 iters on 200k:", time.time() - t0)
    t0 = time.time()
    J = B.jacobian(th, WD.subset(D, slice(0, 20000)))
    A = (J.T @ J).toarray()
    print("ggn 20k rows:", time.time() - t0, A.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["hp", "base", "layers", "time"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--layers", default=",".join(M.LAYERS))
    a = ap.parse_args()
    if a.stage == "hp":
        stage_hp(a.procs)
    elif a.stage == "base":
        stage_base(a.procs)
    elif a.stage == "layers":
        stage_layers(a.layers.split(","))
    else:
        stage_time()
