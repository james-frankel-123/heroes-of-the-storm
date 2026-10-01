"""Consolidated audit A2: Welch t, df and two-sided p for every greedy
degeneracy contrast quoted in Secs. 2-4 (3 seeds x 500 drafts per arm).
Run from training/. Output: drift_rebuild/results/a2_welch.json"""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, "drift_rebuild")
from r8_score import t_sf  # noqa: E402


def welch(a, b):
    a, b = np.array(a, float), np.array(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return {"t": round(float(t), 2), "df": round(float(df), 2), "p": round(2 * t_sf(abs(t), df), 4)}


def seeds(fmt):
    return [json.load(open(fmt.format(s)))["degen_rate"] for s in (42, 123, 777)]


W, RB = "drift2026/results/w2b/", "drift_rebuild/results/r4_w2b/"
M, Mg = seeds(W + "d2c_cumprev_s{}.json"), seeds(W + "d2c_cumprev_s{}_gdcutoff.json")
A, Ag = seeds(RB + "r2_allhist_oof_s{}.json"), seeds(RB + "r2_allhist_oof_s{}_gdcutoff.json")
out = {"allhist_vs_maint": welch(A, M), "allhist_vs_maint_gdcutoff": welch(Ag, Mg),
       "embed_vs_maint": welch(seeds(RB + "r2_embed_oof_s{}.json"), M),
       "leaky_vs_oof_allhist": welch(seeds(W + "d2b_allhist_s{}.json"), A),
       "d90_vs_maint": welch(seeds(W + "q7_decayed90_s{}.json"), M),
       "d90k100_vs_maint": welch(seeds(W + "q7_decayed90k100_s{}.json"), M)}
print(json.dumps(out, indent=1))
json.dump(out, open("drift_rebuild/results/a2_welch.json", "w"), indent=1)
