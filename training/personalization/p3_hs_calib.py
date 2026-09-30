"""
P3 hero strength, stage 5: calibrated intervals.

The raw GP posterior variance s^2 is too small for the future (p3_hs_eval
coverage: realized latent error variance ~1.6x claimed; cold cells worse).
Reasons: skill drifts over months, E residuals are in-sample for the WP
model, and players who take up a new hero differ more than the kernel says.
We fit a display variance

    s2_display = c * s2 + delta2 + kappa0^2 * [n = 0] + kappa1^2 * [1 <= n <= 5]

by least squares of (rbar - m)^2 - noise on the regressors, using cells
played in V1 with the posterior as of the cutoff (fit), and test it on
cells played in V2 with the posterior as of the V1/V2 split (test). Both
horizons are about 7 weeks. Cells need >= 5 future games.

Usage (from training/): python3 personalization/p3_hs_calib.py
Output: results/p3_hs_calib.json
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C
from p3_hs_fit import KERNELS, KOUT, main_kernels, prepare
from p3_hs_eval import decayed_state, nbin, N_LABELS

Z80 = 1.2815516


def cells(d, K, S, P, N, fut, r_adj, H):
    key = d["pid"][fut] * H + d["hero"][fut]
    u, inv, nf = np.unique(key, return_inverse=True, return_counts=True)
    rbar = np.bincount(inv, weights=r_adj[fut]) / nf
    noise = np.bincount(inv, weights=d["v"][fut]) / nf ** 2
    pid, h = u // H, u % H
    m, s2 = C.gp_query(np.ascontiguousarray(S[pid]), np.ascontiguousarray(P[pid]), h, K)
    return dict(rbar=rbar, noise=noise, m=m, s2=s2, nh=N[pid, h], nf=nf)


def report(c, s2d):
    out = {}
    z = np.abs(c["rbar"] - c["m"]) / np.sqrt(s2d + c["noise"])
    err = (c["rbar"] - c["m"]) ** 2 - c["noise"]
    b = nbin(c["nh"])
    for fl, fm in (("nf>=10", c["nf"] >= 10), ("nf>=30", c["nf"] >= 30)):
        for lab, sel in [("all", np.ones(len(b), bool))] + \
                [(N_LABELS[i], b == i) for i in range(len(N_LABELS))]:
            s = sel & fm
            if s.sum() < 50:
                continue
            out[f"{lab}|{fl}"] = {"cells": int(s.sum()),
                                  "cover80": float((z[s] < Z80).mean()),
                                  "var_ratio": float(err[s].mean() / s2d[s].mean()),
                                  "display_sd_pp": float(100 * np.sqrt(s2d[s].mean()))}
    return out


def main():
    d = C.load_slots()
    H = len(d["hero_names"])
    _, best = main_kernels()
    kz = np.load(KOUT)
    bases = {k[6:]: kz[k] for k in kz.files if k.startswith("basis_")}
    e_mask, _, _, _, r_adj = prepare(d)
    days = d["day"]
    post = ~d["in_sample"]
    _, first_row = np.unique(d["g"][post], return_index=True)
    med = np.median(days[post][first_row])
    v1, v2 = post & (days < med), post & (days >= med)
    cut = int(days[post].min())
    out = {}
    for kname in ("player+hero", best):
        K = C.kernel_from([bases[b] for b in KERNELS[kname]], kz["theta_" + kname])
        for hl in (None, 365):
            tag = f"{kname} hl={'inf' if hl is None else hl}"
            S, P, N, _, _ = decayed_state(d, e_mask, cut, hl, H, r_adj)
            cf = cells(d, K, S, P, N, v1, r_adj, H)
            S, P, N, _, _ = decayed_state(d, ~v2, int(med), hl, H, r_adj)
            ct = cells(d, K, S, P, N, v2, r_adj, H)
            del S, P, N
            fm = cf["nf"] >= 5
            reg = lambda c: np.column_stack([c["s2"], np.ones(len(c["s2"])), c["nh"] == 0,
                                             (c["nh"] >= 1) & (c["nh"] <= 5)])
            X = reg(cf)[fm]
            yv = ((cf["rbar"] - cf["m"]) ** 2 - cf["noise"])[fm]
            coef = np.linalg.lstsq(X, yv, rcond=None)[0]
            coef = np.maximum(coef, [0.5, 0.0, 0.0, 0.0])
            disp = lambda c: reg(c) @ coef
            out[tag] = {"coef": {"c": float(coef[0]), "delta_sd_pp": float(100 * np.sqrt(coef[1])),
                                 "kappa0_sd_pp": float(100 * np.sqrt(coef[2])),
                                 "kappa1_sd_pp": float(100 * np.sqrt(coef[3]))},
                        "fit_cells": int(fm.sum()), "test_cells": int((ct["nf"] >= 5).sum()),
                        "test_raw": report(ct, ct["s2"]), "test_calibrated": report(ct, disp(ct)),
                        "fit_calibrated": report(cf, disp(cf))}
            a, b_ = out[tag]["test_raw"]["all|nf>=10"], out[tag]["test_calibrated"]["all|nf>=10"]
            print(f"{tag}: c={coef[0]:.2f} delta={100*np.sqrt(coef[1]):.2f}pp "
                  f"kappa0={100*np.sqrt(coef[2]):.2f}pp kappa1={100*np.sqrt(coef[3]):.2f}pp | V2 nf>=10 cover80 raw {a['cover80']:.3f} "
                  f"(var_ratio {a['var_ratio']:.2f}) -> calibrated {b_['cover80']:.3f} "
                  f"(var_ratio {b_['var_ratio']:.2f})", flush=True)
            for fl in ("nf>=10", "nf>=30"):
                print("   " + fl + " " + " ".join(
                    f"{l}:{out[tag]['test_calibrated'][l + '|' + fl]['cover80']:.2f}"
                    f"/{out[tag]['test_calibrated'][l + '|' + fl]['var_ratio']:.2f}"
                    for l in ["all"] + N_LABELS if l + "|" + fl in out[tag]["test_calibrated"]))
    with open(os.path.join(C.RESULTS, "p3_hs_calib.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
