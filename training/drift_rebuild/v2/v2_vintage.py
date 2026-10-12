"""
v2 vintage matrix (Sec. 8).

  seeds  retrain the 2026-build ranked judge with two extra seeds (7, 991; the
         v1 seed-check seeds) on the v2 relabeled snapshot, same pool and cap
         as drift_v2/models/vintage_judges/wp_2026-build.pt. CPU.
  score  score the v2 maintained vs unmaintained head-to-head drafts
         (drift_v2/results/h2h/M_vs_U.json) with the ten v2 judges and the
         extra seeds: mean maintained-side score and its SE clustered by seed
         pairing; judge-seed SD over the 2026-build seeds.

Usage (from training/):
  python3 drift_rebuild/v2/run.py drift_rebuild/v2/v2_vintage.py seeds
  python3 drift_rebuild/v2/run.py drift_rebuild/v2/v2_vintage.py score
Output: drift_v2/models/vintage_judges/wp_2026-build_s<seed>.pt,
        drift_v2/results/v2_vintage.json
"""
import json
import os
import sys

import numpy as np
import torch

import v2env  # noqa: F401
from drift2026 import common
import drift2026.q2_vintage_judges as q2
from qm2026.train_qm_wp import MLP, featurize

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
JDIR = os.path.join(common.MODELS_DIR, "vintage_judges")
NAMES = ["2022Q1-ranked", "2022-07", "2023-07", "2024-07", "2025-07", "2026-build",
         "QM-2021", "QM-2022", "QM-2024", "QM-2026"]
EXTRA_SEEDS = [7, 991]


def seeds():
    common.setup()
    meta0 = json.load(open(os.path.join(JDIR, "wp_2026-build.meta.json")))
    rows, builds = common.load_data_with_patches()
    cut = common.cutoff_idx(builds)
    pool = [r for r in rows if r["build_idx"] <= cut]
    del rows
    for s in EXTRA_SEEDS:
        p = os.path.join(JDIR, f"wp_2026-build_s{s}.pt")
        if os.path.exists(p):
            continue
        q2.SEED = s
        model, meta = q2.train_judge(f"2026-build_s{s}", pool, torch.device("cpu"), meta0["n_games"])
        meta["seed"] = s
        torch.save(model.state_dict(), p)
        json.dump(meta, open(p.replace(".pt", ".meta.json"), "w"), indent=1)
        print(f"seed {s}: n {meta['n_games']} window {meta['window_min_date']}..{meta['window_max_date']} "
              f"acc {meta['holdout_acc']:.4f}", flush=True)


def score():
    R = json.load(open(os.path.join(common.RESULTS_DIR, "h2h", "M_vs_U.json")))["records"]
    pid = np.array([r["sa"] * 100 + r["sb"] for r in R])
    Xf = torch.tensor(np.stack([featurize(r["maintained"], r["frozen"], r["game_map"], r["tier"]) for r in R]))
    Xr = torch.tensor(np.stack([featurize(r["frozen"], r["maintained"], r["game_map"], r["tier"]) for r in R]))
    judges = {n: os.path.join(JDIR, f"wp_{n}.pt") for n in NAMES}
    judges.update({f"2026-build_s{s}": os.path.join(JDIR, f"wp_2026-build_s{s}.pt") for s in EXTRA_SEEDS})
    out = {"n_drafts": len(R), "n_pairings": int(len(np.unique(pid))), "judges": {}}
    for n, p in judges.items():
        m = MLP()
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval()
        with torch.no_grad():
            w = (0.5 * (m(Xf) + (1 - m(Xr)))).numpy().ravel()
        pm = np.array([w[pid == u].mean() for u in np.unique(pid)])
        out["judges"][n] = {"mean": round(float(w.mean()), 4),
                            "se_pairing": round(float(pm.std(ddof=1) / np.sqrt(len(pm))), 4)}
        print(n, out["judges"][n], flush=True)
    s26 = [out["judges"][k]["mean"] for k in ["2026-build"] + [f"2026-build_s{s}" for s in EXTRA_SEEDS]]
    out["judge_seed_2026_build"] = {"means": s26, "sd": round(float(np.std(s26, ddof=1)), 4),
                                    "range": round(float(max(s26) - min(s26)), 4)}
    print("2026-build seeds", out["judge_seed_2026_build"], flush=True)
    common.write_json(os.path.join(common.RESULTS_DIR, "v2_vintage.json"), out)


if __name__ == "__main__":
    {"seeds": seeds, "score": score}[sys.argv[1]]()
