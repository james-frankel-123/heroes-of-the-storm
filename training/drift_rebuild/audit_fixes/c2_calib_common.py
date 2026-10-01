"""Consolidated audit C2: calibration aging on one common window.
Each leak-free value function (all-history at the 2026 cutoff, 1-yr-stale,
2-yr-stale; 3 seeds) and the maintained model are scored on the same rows:
the seven builds after the 2026 cutoff (the future window), each with its own
deployment statistics (cumulative at its own cutoff; maintained: cumulative
through the previous build).
Output: drift_rebuild/results/c2_calib_common.json"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r3_eval_vf as R  # noqa: E402
import numpy as np  # noqa: E402

R.torch.set_num_threads(4)
builds = R.common.load_patch_index()["builds"]
cut = builds.index(R.CUT)
arms = {"oof_allhist_2026": ("r2_allhist_oof_s{}", f"oof_{R.CUT}", R.rb.MODELS_DIR),
        "oof_stale1yr": ("r2_stale1yr_oof_s{}", "oof_2.55.9.93613", R.rb.MODELS_DIR),
        "oof_stale2yr": ("r2_stale2yr_oof_s{}", "oof_2.55.4.91418", R.rb.MODELS_DIR),
        "maintained": ("d2c_cumprev_s{}", "cumulative_prev", R.D2_MODELS)}
out = {}
for arm, (stem, feat, mdir) in arms.items():
    X, y, bidx, _ = R.load_feats(feat)
    m = bidx > cut
    res = []
    for s in (42, 123, 777):
        model = R.load_model(os.path.join(mdir, stem.format(s) + ".pt"))
        res.append(R.metrics(R.predict(model, X[m]), y[m]))
    out[arm] = {"n_rows": res[0]["n_rows"], "per_seed": res,
                "acc_mean": float(np.mean([r["acc"] for r in res])),
                "slope_mean": float(np.mean([r["calib_slope"] for r in res])),
                "slope_range": [min(r["calib_slope"] for r in res), max(r["calib_slope"] for r in res)]}
    print(arm, out[arm]["n_rows"], round(out[arm]["acc_mean"], 2), round(out[arm]["slope_mean"], 3), out[arm]["slope_range"], flush=True)
json.dump(out, open(os.path.join(R.rb.RESULTS_DIR, "c2_calib_common.json"), "w"), indent=1)
