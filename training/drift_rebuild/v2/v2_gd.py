"""
v2 opponent models (GenericDraftModel, next-pick behavioral cloning) on
site-tiered data, trained with r5b_fast_gd.train_gd (train_single_model's
algorithm on GPU-resident data) and the W5 sample construction
(w5_gd_drift._gd_rows_chunk, 98/2 replay split seed 42).

Pools (--pool):
  cutoff    all games at or before the 2026 cutoff        (deployable; W2b pool, MCTS opponent)
  win6      last 6 builds through the cutoff
  win30d    30 calendar days before the cutoff, trained once
  full      the whole snapshot (saw the test period; reference row)
  future    the 7 post-cutoff builds only (test-period fit; reference row)
  era:<b>   all games at or before build <b>              (stale agents' opponent)
  roll30    for each post-cutoff build, the 30 days before its first game
            (one model per build and variant)
Every model is scored on the post-cutoff next-pick samples (top-1/top-5,
pick/ban split, per build). Output: drift_v2/models/gd_<pool>/generic_draft_<v>.pt
and drift_v2/results/gd_eval_<pool>.json.

Usage: CUDA_VISIBLE_DEVICES=<g> nice -n 19 taskset -c 48-63 \
  python3 drift_rebuild/v2/run.py drift_rebuild/v2/v2_gd.py --pool cutoff --variants 0,1
"""
import argparse
import gc
import json
import multiprocessing as mp
import os
import time

import v2env  # noqa: F401
import numpy as np
import torch

from drift2026 import common
from shared import split_data
from drift2026.w5_gd_drift import _gd_rows_chunk
from r5b_fast_gd import train_gd, _batch


def samples(rows, dev, with_builds=False):
    # Workers get their chunks by pickle; gc.freeze keeps their collector from
    # touching (and copying) the parent's row dicts. Parts are quantized to
    # uint8 (values are k/15) as they arrive and written into one buffer, so
    # no float32 copy and no concatenate copy is ever held. Peak RAM is the
    # loaded rows plus the uint8 samples.
    gc.collect()
    gc.freeze()
    cap = 20 * len(rows)
    X = np.empty((cap, 289), np.uint8)
    y = np.empty(cap, np.int64)
    b = np.empty(cap, np.int64)
    n = 0
    with mp.get_context("fork").Pool(4, maxtasksperchild=50) as pool:
        for px, py, pb in pool.imap(_gd_rows_chunk, [rows[i:i + 4000] for i in range(0, len(rows), 4000)]):
            k = len(py)
            X[n:n + k] = np.rint(px * 15).astype(np.uint8)
            y[n:n + k] = py
            b[n:n + k] = pb
            n += k
    gc.unfreeze()
    out = (torch.from_numpy(X[:n]).to(dev), torch.from_numpy(y[:n]).to(dev))
    if with_builds:
        out = out + (b[:n].copy(),)
    del X, y, b
    return out


@torch.no_grad()
def evaluate(path, Xf, yf, bf, dev):
    from train_generic_draft import GenericDraftModel
    m = GenericDraftModel()
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    m.to(dev).eval()
    t1 = np.zeros(len(yf), bool)
    t5 = np.zeros(len(yf), bool)
    pick = np.zeros(len(yf), bool)
    for i in range(0, len(yf), 16384):
        idx = torch.arange(i, min(i + 16384, len(yf)), device=dev)
        x, mk = _batch(Xf, idx)
        lg = m(x, mk)
        t1[i:i + len(idx)] = (lg.argmax(1) == yf[idx]).cpu().numpy()
        t5[i:i + len(idx)] = (lg.topk(5, 1).indices == yf[idx, None]).any(1).cpu().numpy()
        pick[i:i + len(idx)] = (x[:, 288] > 0.5).cpu().numpy()
    per = {int(b): [round(float(t1[bf == b].mean() * 100), 2), int((bf == b).sum())] for b in np.unique(bf)}
    return {"top1": round(float(t1.mean() * 100), 3), "top5": round(float(t5.mean() * 100), 3),
            "top1_pick": round(float(t1[pick].mean() * 100), 3),
            "top1_ban": round(float(t1[~pick].mean() * 100), 3), "per_build": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--variants", default="0,1")
    a = ap.parse_args()
    variants = [int(v) for v in a.variants.split(",")]
    dev = torch.device("cuda")
    t0 = time.time()
    common.setup()
    rows, builds = common.load_data_with_patches()
    cut = builds.index(common.TRAIN_CUTOFF_BUILD)
    past = [r for r in rows if r["build_idx"] <= cut]
    future = [r for r in rows if r["build_idx"] > cut]
    Xf, yf, bf = samples(future, dev, with_builds=True)
    print(f"future samples {len(yf):,} ({time.time()-t0:.0f}s)", flush=True)
    tag = a.pool.replace(":", "_")
    out_dir = os.path.join(common.MODELS_DIR, f"gd_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(common.RESULTS_DIR, f"gd_eval_{tag}.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}

    def run(pool_rows, name):
        tr, va = split_data(pool_rows, test_frac=0.02, seed=42)
        Xtr, ytr = samples(tr, dev)
        Xva, yva = samples(va, dev)
        print(f"{name}: {len(pool_rows):,} replays, {len(ytr):,} train samples ({time.time()-t0:.0f}s)", flush=True)
        for v in variants:
            path = os.path.join(out_dir, f"generic_draft_{name}_{v}.pt" if a.pool == "roll30"
                                else f"generic_draft_{v}.pt")
            if not os.path.exists(path):
                train_gd(Xtr, ytr, Xva, yva, v, path, verbose=False)
            res[f"{name}_v{v}"] = {"n_replays": len(pool_rows), "n_train_samples": int(len(ytr)),
                                   **evaluate(path, Xf, yf, bf, dev)}
            print(f"  {name} v{v}: top1 {res[f'{name}_v{v}']['top1']} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open(res_path, "w"), indent=1)
        del Xtr, ytr, Xva, yva
        torch.cuda.empty_cache()

    if a.pool == "cutoff":
        run(past, "cutoff")
    elif a.pool == "win6":
        run([r for r in past if r["build_idx"] > cut - 6], "win6")
    elif a.pool == "win30d":
        last = max(r["date_days"] for r in past)
        run([r for r in past if r["date_days"] > last - 30], "win30d")
    elif a.pool == "full":
        run(rows, "full")
    elif a.pool == "future":
        run(future, "future")
    elif a.pool.startswith("era:"):
        e = builds.index(a.pool[4:])
        run([r for r in rows if r["build_idx"] <= e], "era")
    elif a.pool == "roll30":
        tb = sorted({r["build_idx"] for r in future})
        for b in tb:
            d0 = min(r["date_days"] for r in future if r["build_idx"] == b)
            run([r for r in rows if d0 - 30 <= r["date_days"] < d0], builds[b])
    print(f"done ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
