"""
P3: pooled (group-level) skill similarity. Pair-level skill correlations
are noise (p3_hs_similarity: bootstrap sd ~0.6 per pair, split-player
stability of the matrix ~0), so skill structure is estimated for GROUPS of
hero pairs by pooling the split-half cross products of all pairs in the
group:

  rho_G = sum_{(h,k) in G} cross_hk Nw_hk / sum_{(h,k) in G} sqrt(v_h v_k) Nw_hk

(cross, v, Nw as in p3_hs_similarity.pair_matrix; players with >= 4 games
on each hero, weights a_h a_k, a = min(games, 200)). CIs from 200
player-bootstrap replicates. Versions: total skill and skill beyond the
player's general level ("specific").

Groups:
  - same Blizzard role / cross role / same fine role
  - deciles of preference similarity (p3_pref_similarity definition)
  - deciles of the skill embedding's implied similarity (+CF rank 2 kernel,
    fit on the fit half of players), scored on the HELD-OUT half only
  - preference clusters (8, average linkage on the rank-12 preference
    embedding): within-cluster pairs of each cluster, and all between-
    cluster pairs
  - per hero: its 5 nearest preference neighbors vs its other heroes
Embedding stability: the pure rank-2 factorization (player + hero + V V')
refit on the held-out half's E games and compared with the fit-half one.

Run (from training/):
  NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    nice -n 19 taskset -c 48-63 python3 personalization/p3_hs_similarity_pool.py
Outputs: results/p3_hs_similarity_pool.json, results/fig_hero_skill_vs_pref_pooled.png
"""
import os
import sys
import json

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from p3_heroes import NUM_HEROES
import p3_hs_core as C
from p3_hs_fit import prepare, KERNELS
from p3_hs_similarity import half_tables, features, pref_matrix, GT

NBOOT = 200


def pool_parts(xA, xB, A, w=None):
    Aw = A if w is None else A * w[:, None]
    Nw = np.maximum(A.T @ Aw, 1e-12)
    mA = ((xA * A).T @ Aw) / Nw
    mB = ((xB * A).T @ Aw) / Nw
    AB = ((xA * A).T @ (xB * Aw)) / Nw
    cross = 0.5 * (AB + AB.T) - 0.5 * (mA * mB.T + mB * mA.T)
    a2 = np.maximum((A * Aw).sum(0), 1e-12)
    vh = (xA * xB * A * Aw).sum(0) / a2 - ((xA * A * Aw).sum(0) / a2) * ((xB * A * Aw).sum(0) / a2)
    return cross, Nw, vh


def group_rho(cross, Nw, vh, mask):
    sv = np.sqrt(np.clip(np.outer(vh, vh), 0, None))
    m = mask & (np.outer(vh > 0, vh > 0))
    np.fill_diagonal(m, False)
    den = (sv * Nw)[m].sum()
    return float((cross * Nw)[m].sum() / den) if den > 0 else float("nan")


def main():
    d = C.load_slots()
    names = np.array([str(x) for x in d["hero_names"]])
    meta = C.hero_meta(d["hero_names"])
    H = NUM_HEROES
    e_mask, _, _, _, r_adj = prepare(d)
    z = np.load(GT)
    o = np.argsort(z["replay_ids"])
    ts = z["ts"][o][np.searchsorted(z["replay_ids"][o], d["replay_id"])]
    nA, nB, sA, sB = half_tables(d, r_adj, ts)
    kz = np.load(os.path.join(C.CACHE, "hs_kernels.npz"))
    fitp = kz["fit_players"]
    R_pref, cos_pref, npref = pref_matrix(d)
    iu = np.triu_indices(H, 1)
    # groups
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    D = np.clip(1 - cos_pref, 0, None)
    np.fill_diagonal(D, 0)
    pcl = fcluster(linkage(squareform(D, checks=False), "average"), 8, "maxclust")
    groups = {"all pairs": np.ones((H, H), bool),
              "same Blizzard role": meta["blizz"][:, None] == meta["blizz"][None, :],
              "cross role": meta["blizz"][:, None] != meta["blizz"][None, :],
              "same fine role": meta["fine"][:, None] == meta["fine"][None, :],
              "same preference cluster": pcl[:, None] == pcl[None, :],
              "different preference cluster": pcl[:, None] != pcl[None, :]}
    pv = R_pref[iu]
    pedges = np.percentile(pv, np.arange(10, 100, 10))
    pdec = np.searchsorted(pedges, R_pref)
    for q in range(10):
        groups[f"preference decile {q + 1}"] = pdec == q
    for c in np.unique(pcl):
        groups[f"within preference cluster {c}"] = (pcl[:, None] == c) & (pcl[None, :] == c)
    # model-implied similarity (fit half), specific version drops the general term
    bases = {k[6:]: kz[k] for k in kz.files if k.startswith("basis_")}
    th = kz["theta_+CF rank 2"]
    Kf = C.kernel_from([bases[b] for b in KERNELS["+CF rank 2"]], th)
    Ks = Kf - np.exp(th[0]) * bases["player"]
    model_R = {}
    for nm, Km in (("total", Kf), ("specific", Ks)):
        dk = np.sqrt(np.diag(Km))
        model_R[nm] = Km / np.outer(dk, dk)
    out = {"preference_players": npref,
           "preference_clusters": {int(c): list(names[pcl == c]) for c in np.unique(pcl)}}
    rng = np.random.RandomState(1)
    res = {}
    for ver in ("total", "specific"):
        xA, xB, A, pidx = features(nA, nB, sA, sB, 4, ver == "specific")
        gm = dict(groups)
        mv = model_R[ver][iu]
        medges = np.percentile(mv, np.arange(10, 100, 10))
        mdec = np.searchsorted(medges, model_R[ver])
        for q in range(10):
            gm[f"model decile {q + 1}"] = mdec == q
        # per-hero nearest preference neighbors
        for h in range(H):
            nn = np.argsort(-np.where(np.arange(H) == h, -9, R_pref[h]))[:5]
            m1 = np.zeros((H, H), bool)
            m1[h, nn] = m1[nn, h] = True
            m2 = np.zeros((H, H), bool)
            others = np.setdiff1d(np.arange(H), np.r_[nn, h])
            m2[h, others] = m2[others, h] = True
            gm[f"hero {names[h]} | pref top5"] = m1
            gm[f"hero {names[h]} | other heroes"] = m2
        # held-out players for the model validation
        hold = ~fitp[pidx]
        est, boots = {}, {k: [] for k in gm}
        parts_all = pool_parts(xA, xB, A)
        parts_hold = pool_parts(xA[hold], xB[hold], A[hold])
        for k, m in gm.items():
            src = parts_hold if k.startswith("model decile") else parts_all
            est[k] = group_rho(*src, m)
        for b in range(NBOOT):
            w = np.bincount(rng.randint(0, len(A), len(A)), minlength=len(A)).astype(float)
            pa = pool_parts(xA, xB, A, w)
            ph = pool_parts(xA[hold], xB[hold], A[hold], w[hold])
            for k, m in gm.items():
                boots[k].append(group_rho(*(ph if k.startswith("model decile") else pa), m))
        res[ver] = {k: [est[k], float(np.nanpercentile(boots[k], 2.5)),
                        float(np.nanpercentile(boots[k], 97.5))] for k in gm}
        # decile trend summaries
        for fam, key in (("preference", "preference decile"), ("model (held-out)", "model decile")):
            xs = np.array([np.nanmean((pv if fam == "preference" else mv)[
                ((pdec if fam == "preference" else mdec)[iu]) == q]) for q in range(10)])
            ys = np.array([res[ver][f"{key} {q + 1}"][0] for q in range(10)])
            bs = np.array([boots[f"{key} {q + 1}"] for q in range(10)])  # 10 x NBOOT
            slope = np.polyfit(xs, ys, 1)[0]
            bsl = [np.polyfit(xs, bs[:, j], 1)[0] for j in range(bs.shape[1]) if np.all(np.isfinite(bs[:, j]))]
            top_minus_bottom = ys[-1] - ys[0]
            tmb = bs[-1] - bs[0]
            res[ver][f"{fam} trend"] = {
                "decile_x_mean": xs.tolist(), "slope": float(slope),
                "slope_ci": [float(np.percentile(bsl, 2.5)), float(np.percentile(bsl, 97.5))],
                "top_minus_bottom_decile": [float(top_minus_bottom), float(np.nanpercentile(tmb, 2.5)),
                                            float(np.nanpercentile(tmb, 97.5))]}
        print(ver, json.dumps({k: [round(x, 3) for x in v] for k, v in res[ver].items()
                               if not k.startswith("hero ") and not k.endswith("trend")}), flush=True)
        print(ver, "trends", json.dumps({k: v for k, v in res[ver].items() if k.endswith("trend")}), flush=True)
        # per-hero: transfer to preference neighbors minus to others
        per = []
        for h in range(H):
            a = res[ver][f"hero {names[h]} | pref top5"]
            b_ = res[ver][f"hero {names[h]} | other heroes"]
            diff = np.array(boots[f"hero {names[h]} | pref top5"]) - np.array(boots[f"hero {names[h]} | other heroes"])
            per.append({"hero": names[h], "rho_pref_top5": a[0], "rho_others": b_[0],
                        "diff": a[0] - b_[0], "diff_ci": [float(np.nanpercentile(diff, 2.5)),
                                                          float(np.nanpercentile(diff, 97.5))]})
        res[ver]["per_hero_transfer"] = per
    out["pooled"] = res

    # ---------- embedding stability: refit pure rank 2 on the held-out half
    SE, PE, _, _, _ = C.static_state(d, e_mask, d["n_players"], H, r_adj)
    act = PE.sum(1) > 0
    B2 = [np.ones((H, H)), np.eye(H)]
    Sh, Ph = np.ascontiguousarray(SE[~fitp & act]), np.ascontiguousarray(PE[~fitp & act])
    th0, _ = C.fit_kernel(Sh, Ph, B2, verbose=False)
    thh, Vh, _ = C.fit_kernel_lowrank(Sh, Ph, B2, 2, th0, seed=2)
    Vf = np.load(os.path.join(C.CACHE, "sd_similarity.npz"))["V"]
    Gf, Gh = Vf @ Vf.T, Vh @ Vh.T
    out["embedding_stability"] = {
        "corr_VVt_offdiag_fit_vs_heldout": float(np.corrcoef(Gf[iu], Gh[iu])[0, 1]),
        "corr_VVt_fit_vs_pref": float(np.corrcoef(Gf[iu], R_pref[iu])[0, 1]),
        "corr_VVt_heldout_vs_pref": float(np.corrcoef(Gh[iu], R_pref[iu])[0, 1]),
        "corr_CFrank2_kernel_specific_vs_pref": float(np.corrcoef(model_R["specific"][iu], R_pref[iu])[0, 1]),
        "heldout_factor_sd_pp": [float(100 * np.sqrt((Vh[:, j] ** 2).mean())) for j in range(2)]}
    # loadings agreement per axis after Procrustes rotation
    U, _, Wt = np.linalg.svd(Vh.T @ Vf)
    Vh_rot = Vh @ (U @ Wt)
    out["embedding_stability"]["per_axis_loading_corr"] = [
        float(np.corrcoef(Vf[:, j], Vh_rot[:, j])[0, 1]) for j in range(2)]
    print("embedding stability", json.dumps(out["embedding_stability"]), flush=True)

    # ---------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, fam, key in ((axs[0], "preference", "preference decile"),
                         (axs[1], "model (held-out)", "model decile")):
        for ver, col in (("total", "#4e79a7"), ("specific", "#e15759")):
            tr = res[ver][f"{fam} trend"]
            xs = np.array(tr["decile_x_mean"])
            ys = np.array([res[ver][f"{key} {q + 1}"] for q in range(10)])
            ax.errorbar(xs, ys[:, 0], yerr=[ys[:, 0] - ys[:, 1], ys[:, 2] - ys[:, 0]], fmt="o-", color=col,
                        capsize=3, label=f"{ver} skill")
        ax.axhline(0, color="#999", lw=0.5)
        ax.set_xlabel("mean " + ("preference r" if fam == "preference" else
                                 "embedding-implied correlation (fit half)") + " in decile")
        ax.set_ylabel("pooled skill correlation (95% CI)")
        ax.set_title(("Skill vs preference similarity, deciles of hero pairs" if fam == "preference" else
                      "Embedding similarity vs held-out pooled skill correlation"), fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.RESULTS, "fig_hero_skill_vs_pref_pooled.png"), dpi=130)
    with open(os.path.join(C.RESULTS, "p3_hs_similarity_pool.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("done")


if __name__ == "__main__":
    main()
