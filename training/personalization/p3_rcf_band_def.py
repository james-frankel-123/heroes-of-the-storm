"""
P3 definitive run: the display band of the chosen shrinkage model
(P3_SHRINKAGE.md section 6), D = s2 (1 + exp(-games on the hero / 10)),
checked on V1, V2 and OOT cells with 10+ games (moment ratio realized /
claimed with player-bootstrap CI, 80% coverage, by games on the hero), with
the cell's game noise from its own spread (p3_rcf_band.cells).

Usage: python3 personalization/p3_rcf_band_def.py --arm chosen
Output: $P3_RESULTS/rcf/band_def_<arm>.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("BAND_NOISE", "empirical")
import numpy as np

import p3_rcf as R
import p3_rcf_band as B

BINS = ((0, 0), (1, 9), (10, 29), (30, 99), (100, 299), (300, 10 ** 9))


def claimed(A, c):
    rows = c["rows"]
    D = A["s_var"][rows].astype(float) * (1 + np.exp(-A["s_nh"][rows].astype(float) / 10.0))
    n = np.bincount(c["inv"]).astype(float)
    return (np.bincount(c["inv"], weights=D) / n)[c["keep"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    z = np.load(os.path.join(R.RCF_RES, "p3c", f"{a.arm}__merged.npz"))
    A = {k: z[k] for k in z.files if k.startswith("s_")}
    out = {"arm": a.arm, "formula": "D = s2 * (1 + exp(-games_on_hero / 10)); band = estimate +- 1.28 sqrt(D)"}
    rng = np.random.default_rng(0)
    for wk, wn in ((1, "V1"), (2, "V2"), (3, "OOT")):
        c = B.cells(A, wk, 10)
        cl = claimed(A, c)
        ex = c["e"] ** 2 - c["noise"]
        zz = np.abs(c["e"]) / np.sqrt(cl + c["noise"])
        up, inv = np.unique(c["pid"], return_inverse=True)
        bs = []
        for _ in range(300):
            w = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up))[inv].astype(float)
            bs.append(np.sum(w * ex) / np.sum(w * cl))
        row = {"cells": int(len(ex)), "ratio": float(ex.sum() / cl.sum()),
               "ratio_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
               "cover80": float((zz < 1.2816).mean()), "by_games": {}}
        for lo, hi in BINS:
            m = (c["nh"] >= lo) & (c["nh"] <= hi)
            row["by_games"][f"{lo}-{hi}"] = {"cells": int(m.sum()), "cover80": float((zz[m] < 1.2816).mean()),
                                             "ratio": float(ex[m].sum() / cl[m].sum()),
                                             "sd_pp": float(100 * np.median(np.sqrt(cl[m])))}
        out[wn] = row
        print(wn, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "by_games"}),
              {k: round(v["cover80"], 3) for k, v in row["by_games"].items()}, flush=True)
    s = np.isin(A["s_win"], [2, 3])
    nh = A["s_nh"][s].astype(float)
    sd = 100 * np.sqrt(A["s_var"][s] * (1 + np.exp(-nh / 10)))
    out["display_sd_pp_by_games"] = {f"{lo}-{hi}": float(np.median(sd[(nh >= lo) & (nh <= hi)])) for lo, hi in BINS}
    print("display sd by games", {k: round(v, 2) for k, v in out["display_sd_pp_by_games"].items()})
    with open(os.path.join(R.RCF_RES, f"band_def_{a.arm}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
