"""
R3 — accuracy, log loss and calibration of drift value functions, on the
training-period validation slice and on the deployment (future) window.

For each (model, feature pass) pair: rows are split exactly as
drift2026/train_drift_wp.py splits them (fixed seed-42 replay-level 2% val
slice of builds <= the model's cutoff; future = builds after it). Calibration
slope/intercept = logistic regression of the label on logit(p) (slope 1 is
calibrated; slope < 1 is overconfident).

Usage:
  python3 drift_rebuild/r3_eval_vf.py            # the standard table
Output: drift_rebuild/results/r3_eval_vf.json
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_common as rb  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from drift2026 import common  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(16)
from sweep_enriched_wp import WinProbEnrichedModel  # noqa: E402
from drift2026.train_drift_wp import enriched_cols, WPWithPatchEmbed  # noqa: E402

D2_MODELS = common.MODELS_DIR
CUT = common.TRAIN_CUTOFF_BUILD


def feat_path(p):
    if p.startswith("oof_"):
        return os.path.join(rb.CACHE_DIR, f"features_{p}.npz")
    return os.path.join(common.CACHE_DIR, f"features_{p}.npz")


_CACHE = {}


def load_feats(p):
    if p in _CACHE:
        return _CACHE[p]
    z = np.load(feat_path(p))
    cols = enriched_cols()
    X = np.concatenate([z["bases"], z["enricheds"][:, cols]], axis=1)
    out = (X, z["labels"], z["build_idx"].astype(np.int64), z["replay_ids"])
    _CACHE.clear()
    _CACHE[p] = out
    return out


def masks(bidx, rids, cutoff_build):
    builds = common.load_patch_index()["builds"]
    present = np.unique(bidx)
    pos_of = {int(b): i for i, b in enumerate(present)}
    cpos = pos_of[builds.index(cutoff_build)]
    pos = np.vectorize(pos_of.get, otypes=[np.int64])(bidx)
    train = pos <= cpos
    tr_rids = np.unique(rids[train])
    rng = np.random.RandomState(common.SEED)
    val_ids = set(tr_rids[rng.permutation(len(tr_rids))
                          [:max(1, int(len(tr_rids) * 0.02))]].tolist())
    is_val = np.fromiter((int(r) in val_ids for r in rids), bool, len(rids))
    return train & is_val, pos > cpos, pos, cpos


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("embed"):
        m = WPWithPatchEmbed(ck["input_dim"], ck["n_patches"], ck["embed_dim"],
                             arch=tuple(ck["arch"]), dropout=ck["dropout"])
        m.load_state_dict(ck["state_dict"])
        m.eval()
        p = ck["n_patches"] - 1
        return lambda x: m(x, torch.full((len(x),), p, dtype=torch.long))
    m = WinProbEnrichedModel(ck["input_dim"], list(ck["arch"]), ck["dropout"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def predict(model, X):
    out = []
    for i in range(0, len(X), 262144):
        out.append(model(torch.from_numpy(X[i:i + 262144])).numpy().ravel())
    return np.concatenate(out)


def calib(p, y):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    X = np.stack([np.ones_like(z), z], 1)
    w = np.zeros(2)
    for _ in range(50):
        eta = X @ w
        mu = 1 / (1 + np.exp(-eta))
        g = X.T @ (y - mu)
        H = (X * (mu * (1 - mu))[:, None]).T @ X
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[1]), float(w[0])


def metrics(p, y):
    p64 = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    slope, icpt = calib(p, y)
    return {"n_rows": int(len(y)),
            "acc": round(float(((p > 0.5) == (y > 0.5)).mean() * 100), 3),
            "logloss": round(float(-np.mean(y * np.log(p64)
                                            + (1 - y) * np.log(1 - p64))), 5),
            "calib_slope": round(slope, 3), "calib_intercept": round(icpt, 4)}


def evaluate(model_path, feat_pass, cutoff_build=CUT):
    X, y, bidx, rids = load_feats(feat_pass)
    va, fu, _, _ = masks(bidx, rids, cutoff_build)
    m = load_model(model_path)
    return {"val": metrics(predict(m, X[va]), y[va]),
            "future": metrics(predict(m, X[fu]), y[fu])}


def standard_jobs():
    jobs = []
    oof = f"oof_{CUT}"
    for s in common.SEEDS:
        for reg in ("allhist", "win3", "win6", "win12", "decay90", "decay365",
                    "embed"):
            jobs.append((f"leaky/{reg}/s{s}",
                         os.path.join(D2_MODELS, f"d2b_{reg}_s{s}.pt"),
                         "cutoff", CUT))
            jobs.append((f"oof/{reg}/s{s}",
                         os.path.join(rb.MODELS_DIR, f"r2_{reg}_oof_s{s}.pt"),
                         oof, CUT))
        jobs.append((f"maintained/cumprev/s{s}",
                     os.path.join(D2_MODELS, f"d2c_cumprev_s{s}.pt"),
                     "cumulative_prev", CUT))
        # leak reliance: leaky allhist weights on OOF features
        jobs.append((f"leaky_on_oof/allhist/s{s}",
                     os.path.join(D2_MODELS, f"d2b_allhist_s{s}.pt"), oof, CUT))
        for tag, b in (("stale1yr", "2.55.9.93613"), ("stale2yr", "2.55.4.91418")):
            jobs.append((f"leaky/{tag}/s{s}",
                         os.path.join(D2_MODELS, f"w7_stale_{b}_s{s}.pt"),
                         f"cutoff_{b}", b))
            jobs.append((f"oof/{tag}/s{s}",
                         os.path.join(rb.MODELS_DIR, f"r2_{tag}_oof_s{s}.pt"),
                         f"oof_{b}", b))
    jobs.append(("leaky/c0frozen/s42",
                 os.path.join(D2_MODELS, "w2c_frozenstats_cut08_s42.pt"),
                 "cutoff_2.55.3.89754", "2.55.3.89754"))
    jobs.append(("oof/c0frozen/s42",
                 os.path.join(rb.MODELS_DIR, "r2_c0frozen_oof_s42.pt"),
                 "oof_2.55.3.89754", "2.55.3.89754"))
    return jobs


def main():
    out_path = os.path.join(rb.RESULTS_DIR, "r3_eval_vf.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    jobs = standard_jobs()
    jobs.sort(key=lambda j: j[2])   # group by feature pass (cache reuse)
    for key, mp, fp, cb in jobs:
        if key in res or not os.path.exists(mp) or not os.path.exists(feat_path(fp)):
            continue
        res[key] = {"model": mp, "features": fp, **evaluate(mp, fp, cb)}
        r = res[key]
        print(f"{key:32s} val {r['val']['acc']:.2f} (slope {r['val']['calib_slope']:.2f}) "
              f"future {r['future']['acc']:.2f} (slope {r['future']['calib_slope']:.2f})",
              flush=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
