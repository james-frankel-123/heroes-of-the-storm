"""
P3 hero strength: no-pooling baseline.

Each player x hero cell is shrunk toward zero on its own (kernel = t I,
fit by marginal likelihood on the same fit-half players as p3_hs_fit), so
nothing is borrowed from the player's other heroes. Scored like p3_hs_eval
(online, causal through the previous day, V2 slots by games on the hero).

Usage (from training/): python3 personalization/p3_hs_nopool.py
Output: results/p3_hs_nopool.json
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C
from p3_hs_fit import KOUT, prepare
from p3_hs_eval import by_n


def main():
    d = C.load_slots()
    H = len(d["hero_names"])
    kz = np.load(KOUT)
    e_mask, _, _, table, r_adj = prepare(d)
    S, P, _, _, _ = C.static_state(d, e_mask, d["n_players"], H, r_adj)
    fp = kz["fit_players"] & (P.sum(1) > 0)
    th, _ = C.fit_kernel(np.ascontiguousarray(S[fp]), np.ascontiguousarray(P[fp]), [np.eye(H)])
    K = np.exp(th[0]) * np.eye(H)
    del S, P
    days = d["day"]
    post = ~d["in_sample"]
    _, first_row = np.unique(d["g"][post], return_index=True)
    med = np.median(days[post][first_row])
    qidx, S, P, Nh, Np, _, _ = C.online_state(d, np.ones_like(post), post, H, r_adj)
    mu = table[C.exp_bins(Np.astype(np.int64), Nh.astype(np.int64))]
    m, s2 = C.gp_query(S, P, d["hero"][qidx], K)
    v2 = days[qidx] >= med
    out = {"hero_sd_pp": float(100 * np.sqrt(K[0, 0])),
           "V2": by_n(d["r"][qidx][v2], d["wp"][qidx][v2], (mu + m)[v2], Nh[v2].astype(np.int64),
                      np.sqrt(s2[v2]))}
    for k, v in out["V2"].items():
        print(k, {a: round(b, 3) for a, b in v.items()})
    with open(os.path.join(C.RESULTS, "p3_hs_nopool.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
