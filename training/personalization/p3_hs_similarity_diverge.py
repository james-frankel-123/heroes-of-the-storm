"""
P3: where skill similarity and preference similarity diverge, tested with
pooled held-out skill correlations.

Skill-structure similarity: the +CF rank 2 kernel (fit half, E window),
general-skill term removed (the "specific" version) and total.
Groups of hero pairs:
  skill-not-pref   top 20% by embedding similarity, bottom 50% by preference
  pref-not-skill   top 20% by preference, bottom 50% by embedding similarity
  both high        top 20% on both
  neither          bottom 50% on both
Each group's pooled skill correlation is estimated on the HELD-OUT players
only (p3_hs_similarity_pool estimator, 200 player-bootstrap replicates).
Example pairs of each group are listed (largest gap in rank).

Run (from training/):
  NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    nice -n 19 taskset -c 48-63 python3 personalization/p3_hs_similarity_diverge.py
Output: results/p3_hs_similarity_diverge.json
"""
import os
import sys
import json

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C
from p3_hs_fit import prepare, KERNELS
from p3_hs_similarity import half_tables, features, pref_matrix, GT
from p3_hs_similarity_pool import pool_parts, group_rho


def main():
    d = C.load_slots()
    names = np.array([str(x) for x in d["hero_names"]])
    meta = C.hero_meta(d["hero_names"])
    H = 90
    _, _, _, _, r_adj = prepare(d)
    z = np.load(GT)
    o = np.argsort(z["replay_ids"])
    ts = z["ts"][o][np.searchsorted(z["replay_ids"][o], d["replay_id"])]
    nA, nB, sA, sB = half_tables(d, r_adj, ts)
    kz = np.load(os.path.join(C.CACHE, "hs_kernels.npz"))
    fitp = kz["fit_players"]
    R_pref, _, _ = pref_matrix(d)
    bases = {k[6:]: kz[k] for k in kz.files if k.startswith("basis_")}
    th = kz["theta_+CF rank 2"]
    Kf = C.kernel_from([bases[b] for b in KERNELS["+CF rank 2"]], th)
    Ks = Kf - np.exp(th[0]) * bases["player"]
    dk = np.sqrt(np.diag(Ks))
    Rm = Ks / np.outer(dk, dk)
    iu = np.triu_indices(H, 1)
    rp = np.argsort(np.argsort(R_pref[iu])) / len(iu[0])
    rm = np.argsort(np.argsort(Rm[iu])) / len(iu[0])

    def mat(sel):
        M = np.zeros((H, H), bool)
        M[iu[0][sel], iu[1][sel]] = True
        return M | M.T
    sels = {"skill-not-pref": (rm >= 0.8) & (rp < 0.5), "pref-not-skill": (rp >= 0.8) & (rm < 0.5),
            "both high": (rm >= 0.8) & (rp >= 0.8), "neither": (rm < 0.5) & (rp < 0.5)}
    rng = np.random.RandomState(2)
    out = {"n_pairs": {k: int(v.sum()) for k, v in sels.items()}}
    for ver in ("total", "specific"):
        xA, xB, A, pidx = features(nA, nB, sA, sB, 4, ver == "specific")
        hold = ~fitp[pidx]
        xA, xB, A = xA[hold], xB[hold], A[hold]
        est = {k: group_rho(*pool_parts(xA, xB, A), mat(v)) for k, v in sels.items()}
        bs = {k: [] for k in sels}
        diffs = {"skill-not-pref minus pref-not-skill": []}
        for _ in range(200):
            w = np.bincount(rng.randint(0, len(A), len(A)), minlength=len(A)).astype(float)
            parts = pool_parts(xA, xB, A, w)
            vals = {k: group_rho(*parts, mat(v)) for k, v in sels.items()}
            for k in sels:
                bs[k].append(vals[k])
            diffs["skill-not-pref minus pref-not-skill"].append(vals["skill-not-pref"] - vals["pref-not-skill"])
        out[ver] = {k: [est[k], float(np.nanpercentile(bs[k], 2.5)), float(np.nanpercentile(bs[k], 97.5))]
                    for k in sels}
        dd = diffs["skill-not-pref minus pref-not-skill"]
        out[ver]["skill-not-pref minus pref-not-skill"] = [
            est["skill-not-pref"] - est["pref-not-skill"], float(np.nanpercentile(dd, 2.5)),
            float(np.nanpercentile(dd, 97.5))]
        print(ver, json.dumps({k: [round(x, 3) for x in v] for k, v in out[ver].items()}), flush=True)
    for k in ("skill-not-pref", "pref-not-skill"):
        idx = np.flatnonzero(sels[k])
        gap = (rm - rp)[idx] if k == "skill-not-pref" else (rp - rm)[idx]
        top = idx[np.argsort(-gap)][:15]
        out[f"examples {k}"] = [{"pair": f"{names[iu[0][t]]} ({C.BLIZZ_ROLES[meta['blizz'][iu[0][t]]]}) + "
                                         f"{names[iu[1][t]]} ({C.BLIZZ_ROLES[meta['blizz'][iu[1][t]]]})",
                                 "pref_r": float(R_pref[iu][t]), "embedding_corr": float(Rm[iu][t])}
                                for t in top]
        print(k, [e["pair"] for e in out[f"examples {k}"]], flush=True)
    with open(os.path.join(C.RESULTS, "p3_hs_similarity_diverge.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
