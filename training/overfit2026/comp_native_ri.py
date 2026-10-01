"""
Option 1 checked natively: the realized-outcome index with explicit
structure terms inside its own cross-fitted logistic
(gold.StructRealizedIndex), against the plain index, on held-out real games.

  N half-split  index built on one hash half of N (salt 5, as in the RN_cf
                cross-fit) scores the other half; both directions
  SNAP 25%      index built on all of N scores a 25% hash subsample of the
                snapshot (no index trained on any snapshot game)
This is the form used by the production seed-selection judge
(production_refresh/refresh.py, phase select).

Usage (from training/): nice -n 19 taskset -c 48-53 python3 overfit2026/comp_native_ri.py
Output: overfit2026/results/comp_native_ri.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

from overfit2026 import data, gold
from overfit2026.comp_common import gap_table
from overfit2026.structure import struct_matrix


def main():
    from drift2026 import common as dcommon
    dcommon._bind_statscache_methods()
    games = gold.GOLD_SETS["NODRIFT"](data)
    h = np.array([data.half(g[0], salt=5) for g in games])
    p = {"plain": np.zeros(len(games)), "struct": np.zeros(len(games))}
    for k in (0, 1):
        tr = [g for g, hh in zip(games, h) if hh == k]
        for name, cls in (("plain", gold.RealizedIndex), ("struct", gold.StructRealizedIndex)):
            idx = cls(tr, name=f"N_h{k}_{name}")
            for i in np.where(h == 1 - k)[0]:
                g = games[i]
                p[name][i] = idx.score(g[3], g[4], g[1])
    y = np.array([1.0 if g[6] == 0 else 0.0 for g in games])
    s0, s1 = struct_matrix([g[3] for g in games]), struct_matrix([g[4] for g in games])
    out = {"N_halfsplit": {n: gap_table(v, y, s0, s1) for n, v in p.items()}}
    full = gold.StructRealizedIndex(games, name="NODRIFT_struct")
    out["NODRIFT_struct_describe"] = full.describe()
    import pickle
    with open(data.art(HERE, "cache", "realized_NODRIFT_struct_s99.pkl"), "wb") as f:
        pickle.dump(full, f, protocol=pickle.HIGHEST_PROTOCOL)
    snap = [g for g in data.load_snapshot() if data.frac_bucket(g[0], 8) < 2]
    plain = gold.get_index("NODRIFT")
    ys = np.array([1.0 if g[6] == 0 else 0.0 for g in snap])
    t0, t1 = struct_matrix([g[3] for g in snap]), struct_matrix([g[4] for g in snap])
    ps = {"plain": np.array([plain.score(g[3], g[4], g[1]) for g in snap]),
          "struct": np.array([full.score(g[3], g[4], g[1]) for g in snap])}
    out["SNAP25"] = {n: gap_table(v, ys, t0, t1) for n, v in ps.items()}
    json.dump(out, open(data.art(HERE, "results", "comp_native_ri.json"), "w"), indent=1)
    for sname in ("N_halfsplit", "SNAP25"):
        for n, t in out[sname].items():
            print(f"{sname:12s} {n:6s} ll={t['logloss']:.5f} acc={t['acc']:.4f} slope={t['slope']:.2f} "
                  + " ".join(f"{k}:{v['gap_pp']:+.1f}±{v['se_pp']:.1f}" for k, v in t["by_kind"].items()))
    print("struct coef fold0:", np.round(full.fit[0]["coef"], 3).tolist())


if __name__ == "__main__":
    main()
