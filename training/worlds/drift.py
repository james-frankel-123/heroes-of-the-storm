"""
Family e (drift): how far the outcome model moves between the training period and later play,
measured on the 2.55.17 builds (balance patch 2026-07-20, T17 games; never in any training set).

With the base MAP as offset, three move layers are fit in turn on T17 (model.fit_layer: Gaussian
cells, prior SD by diag-Laplace evidence):
  dw  hero strength (cell = hero; +1 on team 0, -1 on team 1)
  dS  pair synergy (cell = pair within a team)
  dK  pair counter (directed cell (a on team 0, b on team 1) and its reverse with the opposite sign,
      so the antisymmetric move has SD sqrt(2) s)
The evidence-chosen SDs (and their 95% profile upper bounds) are the drift family's move sizes:
sd_w = s(dw), sd_S = s(dS), sd_K = sqrt(2) s(dK), each net of the same fit on NODRIFT (snapshot
builds, later uploads: no balance change), which measures what the base model misses without any
drift: drift SD = sqrt(max(s_T17^2 - s_NODRIFT^2, 0)) (rule fixed before the fit). -> site/results/drift.json
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as WD  # noqa: E402
import model as M  # noqa: E402

H = M.H
GRID = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25]


def kpn(kind, D):
    T0, T1 = D["T0"], D["T1"]
    if kind == "dw":
        return T0, T1
    if kind == "dS":
        return T0[:, M.IA] * H + T0[:, M.IB], T1[:, M.IA] * H + T1[:, M.IB]
    a, b = T0[:, :, None], T1[:, None, :]
    return (a * H + b).reshape(len(T0), 25), (b * H + a).reshape(len(T0), 25)


def measure(name):
    B, th, cfg = M.load_base()
    D = WD.load(name)
    off = B.logit(th, D)
    out = {"games": int(len(D["y"])), "nll_base": M.nll(M.sig(off), D["y"])}
    for kind in ("dw", "dS", "dK"):
        L = M.fit_layer(kind, D, off, cfg["nm"], cfg["nt"], GRID, kpn=kpn(kind, D))
        kp, kn = kpn(kind, D)
        lut = dict(zip(L["keys"].tolist(), L["mu"].tolist()))
        f = np.vectorize(lambda k: lut.get(int(k), 0.0))
        off = off + f(kp).sum(1) - f(kn).sum(1)
        scale = np.sqrt(2) if kind == "dK" else 1.0
        out[kind] = {"s": L["s"] * scale, "s_hi95": L["s_hi95"] * scale, "s_lo95": L["s_lo95"] * scale,
                     "grid": L["grid"], "nll_after": M.nll(M.sig(off), D["y"])}
        print(name, kind, json.dumps({k: v for k, v in out[kind].items() if k != "grid"}), flush=True)
    return out


def main():
    res = {"T17": measure("T17"), "NODRIFT_control": measure("NODRIFT")}
    T = res["T17"]
    C = res["NODRIFT_control"]
    net = lambda k, f: float(np.sqrt(max(T[k][f] ** 2 - C[k]["s"] ** 2, 0.0)))
    res["sizes"] = {"sd_w": net("dw", "s"), "sd_S": net("dS", "s"), "sd_K": net("dK", "s"),
                    "sd_w_hi95": net("dw", "s_hi95"), "sd_S_hi95": net("dS", "s_hi95"), "sd_K_hi95": net("dK", "s_hi95")}
    json.dump(res, open(os.path.join(WD.RESULTS, "drift.json"), "w"), indent=1)
    print(json.dumps(res["sizes"], indent=1))


if __name__ == "__main__":
    main()
