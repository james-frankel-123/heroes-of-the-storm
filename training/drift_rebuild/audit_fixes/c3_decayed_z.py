"""Consolidated audit C3/C4: committed artifact for the decayed-aggregate
significance numbers quoted in Sec. 4 (previously only in audit scratch).

Future window: per-row correctness averaged over the 3 seeds of each arm
(q7_decayed90 / q7_decayed90k100 / q7_decayed365 vs d2c_cumprev), paired on
identical rows, SE clustered by replay (both team-swap rows of a game).
Nested window: the W8c re-selection (models trained with cutoff 2.55.13.95301,
scored on builds after it up to the 2026 cutoff), same paired, clustered SE.
Output: drift_rebuild/results/c3_decayed_z.json"""
import json
import os
import sys

import numpy as np
import torch

T = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, T)
from drift2026 import common  # noqa: E402
from drift2026.train_drift_wp import enriched_cols  # noqa: E402
from sweep_enriched_wp import WinProbEnrichedModel  # noqa: E402

torch.set_num_threads(4)
cols = enriched_cols()
builds = common.load_patch_index()["builds"]


def rows(feat, lo, hi):
    z = np.load(os.path.join(common.CACHE_DIR, f"features_{feat}.npz"))
    b = z["build_idx"].astype(np.int64)
    present = np.unique(b)
    pos_of = {int(x): i for i, x in enumerate(present)}
    pos = np.vectorize(pos_of.get, otypes=[np.int64])(b)
    lo_p, hi_p = pos_of[builds.index(lo)], (pos_of[builds.index(hi)] if hi else 10 ** 9)
    m = (pos > lo_p) & (pos <= hi_p)
    X = np.concatenate([z["bases"][m], z["enricheds"][m][:, cols]], 1).astype(np.float32)
    return torch.from_numpy(X), z["labels"][m], z["replay_ids"][m]


def correct(stem, X, y):
    acc = []
    for s in (42, 123, 777):
        ck = torch.load(os.path.join(common.MODELS_DIR, f"{stem}_s{s}.pt"), map_location="cpu", weights_only=False)
        m = WinProbEnrichedModel(X.shape[1], [256, 128], dropout=0.3)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        with torch.no_grad():
            p = torch.cat([m(X[i:i + 65536]) for i in range(0, len(X), 65536)]).numpy().ravel()
        acc.append(((p > 0.5) == (y > 0.5)).astype(np.float64))
    return np.mean(acc, 0)


def paired(a, b, rid):
    u, inv = np.unique(rid, return_inverse=True)
    g = np.bincount(inv, weights=a - b) / np.bincount(inv)
    se = g.std(ddof=1) / np.sqrt(len(u)) * 100
    d = (a - b).mean() * 100
    return {"diff_pp": round(d, 4), "se_pp": round(se, 4), "z": round(d / se, 2), "n_games": int(len(u))}


out = {}
for window, lo, hi, cells in (
        ("future", "2.55.14.95918", None,
         {"cumulative": ("cumulative_prev", "d2c_cumprev"), "decayed90": ("decayed90_prev", "q7_decayed90"),
          "decayed90k100": ("decayed90k100_prev", "q7_decayed90k100"), "decayed365": ("decayed365_prev", "q7_decayed365")}),
        ("nested", "2.55.13.95301", "2.55.14.95918",
         {"cumulative": ("cumulative_prev", "w8c_cumulative"), "decayed90": ("decayed90_prev", "w8c_decayed90"),
          "decayed90k100": ("decayed90k100_prev", "w8c_decayed90k100"), "decayed365": ("decayed365_prev", "w8c_decayed365")})):
    acc, rid0 = {}, None
    for c, (feat, stem) in cells.items():
        X, y, rid = rows(feat, lo, hi)
        if rid0 is None:
            rid0 = rid
        assert (rid == rid0).all()
        acc[c] = correct(stem, X, y)
    out[window] = {c: paired(acc[c], acc["cumulative"], rid0) for c in acc if c != "cumulative"}
    out[window]["acc_mean"] = {c: round(float(v.mean() * 100), 3) for c, v in acc.items()}
    print(window, json.dumps(out[window]), flush=True)
json.dump(out, open(os.path.join(T, "drift_rebuild/results/c3_decayed_z.json"), "w"), indent=1)
