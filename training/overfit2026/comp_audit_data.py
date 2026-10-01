"""
Composition audit, step 1: score every independent judge on real games it
never trained on, and store per-game arrays for the calibration analysis.

Evaluation sets (overfit2026.data slim games):
  SNAP  the pinned paper-1 snapshot, 1,949,087 games. No judge trained on any
        of them (gN / gN_naive / RN on post-snapshot no-drift games, R17 on
        2.55.17, QM2026 on Quick Match), so every judge is held out here.
  N     the 291,837 post-snapshot no-drift games (BF + T97). Held out for R17
        and QM2026 only; RN gets a two-way cross-fit (index built on one hash
        half scores the other half, "RN_cf").
  T17   the 158,608 drifted 2.55.17 games. Held out for gN / gN_naive / RN /
        QM2026; R17 gets the same cross-fit ("R17_cf").

Judges, P(team0 wins), team-order symmetrized: gN (mean of the 3 OOF
enriched judges, N8 deploy statistics), gN_naive, RN (NODRIFT index), R17,
QM2026. Structure per team: no_healer, no_frontline, stack (3+ of a
stacking-bad role), non-exclusive, with shared.is_degenerate conventions.

Usage (from training/):
  nice -n 19 taskset -c 48-53 python3 overfit2026/comp_audit_data.py [SNAP N T17]
Output: overfit2026/cache/comp_audit_<set>.npz
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np
import torch

from overfit2026 import data, feats, score, gold
from overfit2026.structure import struct_matrix

NPROC = 6
GN = ["gN8_oof_s0", "gN8_oof_s1", "gN8_oof_s2"]


def games_of(name):
    if name == "SNAP":
        return data.load_snapshot()
    return gold.GOLD_SETS["NODRIFT" if name == "N" else "T17"](data)


def cross_fit(games, name):
    """Realized index built on one hash half scores the other half."""
    out = np.zeros(len(games))
    h = np.array([data.half(g[0], salt=5) for g in games])
    for k in (0, 1):
        idx = gold.RealizedIndex([g for g, hh in zip(games, h) if hh == k], name=f"{name}_h{k}")
        for i in np.where(h == 1 - k)[0]:
            g = games[i]
            out[i] = idx.score(g[3], g[4], g[1])
    return out


def main():
    torch.set_num_threads(4)
    from drift2026 import common as dcommon
    from overfit2026.stage_b_saved import qm_scores
    dcommon._bind_statscache_methods()
    sets = sys.argv[1:] or ["T17", "N", "SNAP"]
    for name in sets:
        path = os.path.join(HERE, "cache", f"comp_audit_{name}.npz")
        if os.path.exists(path):
            print(f"{name}: exists", flush=True)
            continue
        t0 = time.time()
        games = games_of(name)
        rows = [(g[3], g[4], g[2], g[1]) for g in games]
        out = {"rid": np.array([g[0] for g in games], np.int64),
               "tier": np.array([g[1] for g in games]),
               "y": np.array([1.0 if g[6] == 0 else 0.0 for g in games]),
               "s0": struct_matrix([g[3] for g in games]),
               "s1": struct_matrix([g[4] for g in games])}
        print(f"{name}: {len(games):,} games", flush=True)
        Xf, Xs = feats.featurize(rows, score._stats("N8"), nproc=NPROC)
        print(f"  featurized {time.time() - t0:.0f}s", flush=True)
        for n in GN + ["gN8_naive"]:
            m, meta = score.model(n)
            d = meta["input_dim"]
            out[n] = feats.predict_sym(m, Xf[:, :d], Xs[:, :d], "cpu")
        del Xf, Xs
        out["gN"] = np.mean([out[n] for n in GN], 0)
        out["gN_naive"] = out.pop("gN8_naive")
        print(f"  gN {time.time() - t0:.0f}s", flush=True)
        for key, idx_name in (("RN", "NODRIFT"), ("R17", "T17")):
            ri = score.realized(idx_name)
            out[key] = np.array([ri.score(o, p, t) for o, p, m, t in rows])
        print(f"  realized {time.time() - t0:.0f}s", flush=True)
        q = {}
        for i in range(0, len(rows), 200000):
            r = qm_scores({"QM2026": score.qm()["QM2026"]}, rows[i:i + 200000])
            q.setdefault("QM2026", []).append(r["QM2026"])
        out["QM2026"] = np.concatenate(q["QM2026"])
        print(f"  QM {time.time() - t0:.0f}s", flush=True)
        if name == "N":
            out["RN_cf"] = cross_fit(games, "N")
        if name == "T17":
            out["R17_cf"] = cross_fit(games, "T17")
        np.savez(path, **out)
        print(f"{name}: wrote {path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
