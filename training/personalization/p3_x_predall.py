"""
P3 extensions: causal per-slot predictions for EVERY snapshot slot
(2024-04-01 .. 2026-05-22), used by tasks 2, 3, 4 and 7.

For each slot: GP posterior mean at the slot's hero under five phase-1
kernels (state = the player's games from days <= game day - 1, no decay),
posterior variance under "+CF rank 2", the lagged counts (games on this
hero, games overall), and the experience offset mu from those counts.
The post-snapshot games are deliberately not used here: they stay a clean
holdout for task 1.

Usage (from training/): python3 personalization/p3_x_predall.py
Output: cache/x_predall.npz (row-aligned with cache/hs_slots.npz)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C
import p3_x_common as X

OUT = os.path.join(X.CACHE, "x_predall.npz")
KN = ["+CF rank 2", "player", "player+hero", "player+role+melee+hero", "+co-play"]
TAG = {"+CF rank 2": "cf2", "player": "player", "player+hero": "ph",
       "player+role+melee+hero": "role", "+co-play": "coplay"}


def main():
    t0 = time.time()
    d = C.load_slots()
    Ks, table = X.kernels()
    n_p, n_ph = C.experience_counts(d)
    r_adj = d["r"] - table[C.exp_bins(n_p, n_ph)]
    allm = np.ones(len(d["pid"]), bool)
    qi, m, v, Nh, Np, ex = X.online_predict(d, allm, allm, Ks[KN[0]], r_adj,
                                            extra=[Ks[k] for k in KN[1:]])
    assert np.array_equal(qi, np.arange(len(d["pid"])))
    mu = table[C.exp_bins(Np.astype(np.int64), Nh.astype(np.int64))]
    out = {"mu": mu.astype(np.float32), "var_cf2": v.astype(np.float32),
           "nh_lag": Nh, "np_lag": Np, "n_p": n_p.astype(np.int32),
           "n_ph": n_ph.astype(np.int32), "r_adj": r_adj.astype(np.float32)}
    out["m_cf2"] = m.astype(np.float32)
    for k, e in zip(KN[1:], ex):
        out["m_" + TAG[k]] = e.astype(np.float32)
    np.savez(OUT, **out)
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
