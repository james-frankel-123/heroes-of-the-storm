"""
R13 — a 30-day opponent-model window that actually rolls (audit A8).

drift2026/w5_gd_drift.py trained its "30-day" GenericDraftModel once, on the
30 days before the training cutoff, and scored it on all seven later builds.
Here, for each post-cutoff build b, variants 0 and 1 are trained on the games
played in the 30 calendar days before b's first game (any build; all such
games are available to a deployed system at that time), with the W5 sample
construction (_gd_rows_chunk) and a 98/2 replay split, then scored on b's
next-pick samples in the stored W5 future cache. Training uses the
train_single_model algorithm on GPU-resident data (r5b_fast_gd.train_gd).

Usage: CUDA_VISIBLE_DEVICES=1 python3 drift_rebuild/r13_rolling_gd.py
Output: drift_rebuild/results/r13_rolling_gd.json
"""
import os
import sys
import json
import time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_common as rb  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from drift2026 import common  # noqa: E402

WIN_DAYS = 30


def to_u8(chunks, dev):
    X = np.concatenate([c[0] for c in chunks])
    y = np.concatenate([c[1] for c in chunks])
    return (torch.from_numpy(np.rint(X * 15).astype(np.uint8)).to(dev),
            torch.from_numpy(y.astype(np.int64)).to(dev))


def main():
    from shared import split_data
    from drift2026.w5_gd_drift import _gd_rows_chunk
    from r5b_fast_gd import train_gd, _batch
    from rerun2026 import common as rc
    from train_generic_draft import GenericDraftModel
    common.setup()
    dev = torch.device("cuda")
    t0 = time.time()
    rows, builds = common.load_data_with_patches()
    cut = builds.index(common.TRAIN_CUTOFF_BUILD)
    test_b = sorted({r["build_idx"] for r in rows if r["build_idx"] > cut})
    start = {b: min(r["date_days"] for r in rows if r["build_idx"] == b) for b in test_b}
    fut = rc.memmap_open(os.path.join(common.CACHE_DIR, "gd_w5_future_test"))
    fb = np.asarray(fut["builds"])
    out = {"per_build": {}}
    ctx = mp.get_context("fork")
    for b in test_b:
        d0 = start[b]
        win = [r for r in rows if d0 - WIN_DAYS <= r["date_days"] < d0]
        tr, va = split_data(win, test_frac=0.02, seed=42)
        with ctx.Pool(32) as pool:
            ctr = pool.map(_gd_rows_chunk, [tr[i:i + 2000] for i in range(0, len(tr), 2000)])
            cva = pool.map(_gd_rows_chunk, [va[i:i + 2000] for i in range(0, len(va), 2000)])
        Xtr, ytr = to_u8(ctr, dev)
        Xva, yva = to_u8(cva, dev)
        m = fb == b
        Xf = torch.from_numpy(np.rint(np.asarray(fut["states"][np.flatnonzero(m)]) * 15)
                              .astype(np.uint8)).to(dev)
        yf = torch.from_numpy(np.asarray(fut["actions"][np.flatnonzero(m)], np.int64)).to(dev)
        accs = []
        for v in (0, 1):
            path = os.path.join(rb.MODELS_DIR, f"gd_roll30_{builds[b]}_{v}.pt")
            if not os.path.exists(path):
                train_gd(Xtr, ytr, Xva, yva, v, path, verbose=False)
            model = GenericDraftModel()
            model.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
            model.to(dev).eval()
            c = 0
            with torch.no_grad():
                for i in range(0, len(yf), 8192):
                    idx = torch.arange(i, min(i + 8192, len(yf)), device=dev)
                    x, mk = _batch(Xf, idx)
                    c += (model(x, mk).argmax(1) == yf[idx]).sum().item()
            accs.append(100 * c / len(yf))
        out["per_build"][builds[b]] = {
            "window_replays": len(win), "train_samples": int(len(ytr)),
            "n_test_samples": int(m.sum()), "top1_by_variant": accs,
            "top1": float(np.mean(accs))}
        print(f"{builds[b]}: window {len(win):,} replays, top1 {np.mean(accs):.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    n = np.array([v["n_test_samples"] for v in out["per_build"].values()])
    a = np.array([v["top1"] for v in out["per_build"].values()])
    out["overall_top1"] = float((n * a).sum() / n.sum())
    common.write_json(os.path.join(rb.RESULTS_DIR, "r13_rolling_gd.json"), out)
    print(f"rolling 30-day overall top-1: {out['overall_top1']:.2f}")


if __name__ == "__main__":
    main()
