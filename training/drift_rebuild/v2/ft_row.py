"""Warm-start finetune row of the v2 deployment replay (q8_finetune_row
protocol): retrain points every 3rd build from C0 using the w2c_ft models,
statistics refreshed every build; regret vs retrain-every-build.
Output: <results>/w2c_ft_row.json"""
import json
import os

import v2env  # noqa: F401
import numpy as np
from drift2026 import common

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
ft = {}
for p in range(C0 + 3, last, 3):
    r = json.load(open(os.path.join(R, "w2c_ft", f"w2c_ft_cut{p:02d}_s42.json")))
    ft[p] = {builds.index(b): d["acc"] for b, d in r["test_acc_per_build"].items()}
ft[C0] = acc[C0]
Ns = sorted(n for n in games if n > C0)
w = np.array([games[n] for n in Ns], float)
wavg = lambda c: float(np.sum(w * np.array([c[n] for n in Ns])) / w.sum())
o = wavg({n: acc[n - 1][n] for n in Ns})
pts = list(range(C0, last, 3))
c = {n: ft[max(q for q in pts if q < n)][n] for n in Ns}
out = {"finetune_every3_refresh": {"retrains": len([p for p in pts if C0 < p < Ns[-1]]),
                                    "regret_pp": round(o - wavg(c), 3)}}
print(out)
json.dump(out, open(os.path.join(R, "w2c_ft_row.json"), "w"))
