"""Consolidated audit C8: the vintage matrix rescored on the leak-free
head-to-head (maintained vs unmaintained, drift_rebuild/results/h2h/M_vs_U.json,
15 x 6 seeds, 7,200 drafts) with the same ten naive judges, plus the
original first-version drafts for comparison. SE clustered by seed pairing.
Run from training/. Output: drift_rebuild/results/c8_vintage_leakfree.json"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from qm2026.train_qm_wp import MLP, featurize  # noqa: E402

torch.set_num_threads(4)
J = {n: f"drift2026/models/vintage_judges/wp_{n}.pt"
     for n in ["2022Q1-ranked", "2022-07", "2023-07", "2024-07", "2025-07", "2026-build"]}
J.update({n: "qm2026/results/" + f for n, f in [("QM-2021", "qm_wp_v0.pt"), ("QM-2022", "qm_wp_2022.pt"),
                                                ("QM-2024", "qm_wp_2024.pt"), ("QM-2026", "qm_wp_2026.pt")]})
files = {"leakfree_M_vs_U": "drift_rebuild/results/h2h/M_vs_U.json",
         "first_version_M_vs_Uleaky": "drift2026/results/w6_head2head.json"}
out = {}
for fk, fp in files.items():
    R = json.load(open(fp))["records"]
    if "sa" not in R[0]:
        for k, r in enumerate(R):          # W6 loop order: 25 pairings x 80
            r["sa"], r["sb"] = divmod(k // 80, 5)
    pid = np.array([r["sa"] * 100 + r["sb"] for r in R])
    Xf = torch.tensor(np.stack([featurize(r["maintained"], r["frozen"], r["game_map"], r["tier"]) for r in R]))
    Xr = torch.tensor(np.stack([featurize(r["frozen"], r["maintained"], r["game_map"], r["tier"]) for r in R]))
    out[fk] = {}
    for n, p in J.items():
        m = MLP()
        sd = torch.load(p, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        m.load_state_dict(sd)
        m.eval()
        with torch.no_grad():
            w = (0.5 * (m(Xf) + (1 - m(Xr)))).numpy().ravel()
        pm = np.array([w[pid == u].mean() for u in np.unique(pid)])
        out[fk][n] = {"mean": round(float(w.mean()), 4),
                      "se_pairing": round(float(pm.std(ddof=1) / np.sqrt(len(pm))), 4)}
        print(fk, n, out[fk][n], flush=True)
json.dump(out, open("drift_rebuild/results/c8_vintage_leakfree.json", "w"), indent=1)
