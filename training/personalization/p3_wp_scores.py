"""
P3 stage 1 — population win probability for every snapshot game.

Scores every row of the paper-2 cumulative-refresh feature cache
(drift2026/feature_cache/features_cumulative_prev.npz: stats through the
previous build, so every feature is causal) with the three paper-2
maintained models (drift2026/models/d2c_cumprev_s{42,123,777}.pt, trained on
builds <= 2.55.14.95918). Rows come in pairs (original, team swap), so the
per-game probability is symmetrized: wp0 = (p_orig + 1 - p_swap) / 2,
averaged over seeds.

Games after the training cutoff are out of sample for the WP model; earlier
games are in sample, which slightly shrinks their residuals (reported, not
corrected).

Usage: python3 personalization/p3_wp_scores.py
Output: personalization/cache/wp_drift.npz
    replay_ids, date_days, build_idx, y (team 0 won), wp0 (ensemble),
    wp0_seeds (3, N), in_sample (build <= cutoff)
"""
import os
import sys

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING)
from drift2026 import common

common.setup()

import numpy as np
import torch

from sweep_enriched_wp import WinProbEnrichedModel
from drift2026.train_drift_wp import enriched_cols

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "wp_drift.npz")
SEEDS = [42, 123, 777]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    z = np.load(os.path.join(common.CACHE_DIR, "features_cumulative_prev.npz"))
    X = np.concatenate([z["bases"], z["enricheds"][:, enriched_cols()]], axis=1)
    rids = z["replay_ids"]
    assert np.all(rids[0::2] == rids[1::2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(X, dtype=torch.float32)
    del X
    probs = []
    for s in SEEDS:
        ck = torch.load(os.path.join(common.MODELS_DIR, f"d2c_cumprev_s{s}.pt"),
                        map_location="cpu", weights_only=False)
        m = WinProbEnrichedModel(ck["input_dim"], ck["arch"], dropout=ck["dropout"])
        m.load_state_dict(ck["state_dict"])
        m.to(device).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xt), 262144):
                # WinProbEnrichedModel ends in a sigmoid: outputs are
                # probabilities already.
                out.append(m(Xt[i:i + 262144].to(device)).cpu().numpy())
        p = np.concatenate(out)
        probs.append((p[0::2] + 1.0 - p[1::2]) / 2.0)
        print(f"seed {s}: done", flush=True)
    seeds = np.stack(probs)
    builds = common.load_patch_index()["builds"]
    cut = builds.index(common.TRAIN_CUTOFF_BUILD)
    bidx = z["build_idx"][0::2].astype(np.int64)
    y = z["labels"][0::2]
    wp0 = seeds.mean(axis=0)
    np.savez(OUT, replay_ids=rids[0::2], date_days=z["date_days"][0::2],
             build_idx=bidx, y=y, wp0=wp0, wp0_seeds=seeds,
             in_sample=bidx <= cut)
    oos = bidx > cut
    acc = lambda m: float(((wp0[m] > 0.5) == (y[m] == 1)).mean())
    print(f"{len(y):,} games; in-sample acc {acc(~oos):.4f}, "
          f"out-of-sample acc {acc(oos):.4f} ({oos.sum():,} games)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
