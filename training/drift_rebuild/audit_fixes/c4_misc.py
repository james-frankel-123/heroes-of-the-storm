"""Consolidated audit C4: committed artifacts for (a) retrain-every-K regrets
from the stored W2c matrix (K = 1, 2, 3, 4, 6, 9, 12) and (b) the chance
precision of the blind changepoint detector (11 held-out boundaries, +-3-day
tolerance, held-out span 2025-01-01 to 2026-05-22).
Output: drift_rebuild/results/c4_misc.json"""
import datetime as dt
import json
import os
import sys

import numpy as np

T = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, T)
from drift2026 import common  # noqa: E402

R = common.RESULTS_DIR
builds = [b for b in common.load_patch_index()["builds"] if b.startswith("2.55")]
cut, last, C0 = builds.index(common.TRAIN_CUTOFF_BUILD), len(builds) - 1, 8
acc, games = {}, {}
for p in range(C0, last):
    src = ("d2", "d2c_cumprev_s42.json") if p == cut else ("w2c", f"w2c_cut{p:02d}_s42.json")
    r = json.load(open(os.path.join(R, *src)))
    acc[p] = {builds.index(b): d["acc"] for b, d in r["test_acc_per_build"].items()}
    for b, d in r["test_acc_per_build"].items():
        games[builds.index(b)] = d["n_rows"] // 2
Ns = sorted(n for n in games if n > C0)
w = np.array([games[n] for n in Ns], float)
wavg = lambda c: float(np.sum(w * np.array([c[n] for n in Ns])) / w.sum())
o = wavg({n: acc[n - 1][n] for n in Ns})
out = {"retrain_every_K_refresh": {}}
for K in (1, 2, 3, 4, 6, 9, 12):
    pts = list(range(C0, last, K))
    c = {n: acc[max(q for q in pts if q < n)][n] for n in Ns}
    out["retrain_every_K_refresh"][K] = {"retrains": len([p for p in pts if C0 < p < Ns[-1]]),
                                         "regret_pp": round(o - wavg(c), 3)}
span = (dt.date(2026, 5, 22) - dt.date(2025, 1, 1)).days + 1
out["blind_changepoint_chance"] = {"boundaries": 11, "tolerance_days": 3, "span_days": span,
                                   "covered_days": 11 * 7, "chance_precision": round(77 / span, 3)}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(common.RESULTS_DIR, "c4_misc.json"), "w"), indent=1)
