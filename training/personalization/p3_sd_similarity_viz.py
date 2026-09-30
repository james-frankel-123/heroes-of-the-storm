"""
P3 phase 2, item 4 (second part): clusters, validation and figures from the
embedding chosen in p3_sd_similarity.py (pure factorization, rank by
held-out likelihood).

- Clusters: Ward clustering of the hero embedding coordinates (the part of
  skill beyond general skill that the factors explain), k = 4..8, ARI
  against Blizzard and fine roles. Heroes are ordered by angle in the
  embedding plane.
- Validation on held-out players only (never used to fit the embedding):
  empirical cross-hero covariance of cell means (cells with >= 5 games),
  pooled over deciles of the model's predicted covariance beyond general
  skill. Pair-level empirical correlations are too noisy to rank (median
  per-cell split-half reliability 0.10), so validation is by decile.
- Variance shares of a player x hero cell from the phase-1 kernel.
- Figures: results/fig_hero_skill_corr.png (heatmap, model beyond general
  skill | held-out empirical, both ordered by embedding angle) and
  results/fig_hero_embedding.png (2-D hero map colored by role).

Usage (from training/): python3 personalization/p3_sd_similarity_viz.py
Output: results/p3_sd_similarity_viz.json and the two PNGs.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C
from p3_hs_fit import prepare
from p3_sd_similarity import ari, empirical_cov

ROLE_COL = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#9c755f"]


def main():
    d = C.load_slots()
    H = len(d["hero_names"])
    names = np.array([str(x) for x in d["hero_names"]])
    meta = C.hero_meta(d["hero_names"])
    z = np.load(os.path.join(C.CACHE, "sd_similarity.npz"))
    V, F = z["V"], z["F"]
    with open(os.path.join(C.RESULTS, "p3_sd_similarity.json")) as f:
        prev = json.load(f)
    r = prev["best_rank_pure"]
    kz = np.load(os.path.join(C.CACHE, "hs_kernels.npz"))
    fitp = kz["fit_players"]
    out = {"rank": r}

    # ---------- variance shares (phase-1 +CF rank 2 kernel, mean diagonal)
    th = kz["theta_+CF rank 2"]
    bl = ["player", "role", "fine", "melee", "hero", "cf2"]
    bases = {k[6:]: kz[k] for k in kz.files if k.startswith("basis_")}
    parts = {b: float(np.exp(t) * np.mean(np.diag(bases[b]))) for b, t in zip(bl, th)}
    tot = sum(parts.values())
    out["variance_shares_phase1_kernel"] = {b: v / tot for b, v in parts.items()}
    out["cell_sd_pp_phase1_kernel"] = float(100 * np.sqrt(tot))
    print("variance shares", {k: round(v, 3) for k, v in out["variance_shares_phase1_kernel"].items()})

    # ---------- clusters on embedding coordinates
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(F, "ward")
    ang = np.arctan2(F[:, 1], F[:, 0]) if F.shape[1] >= 2 else F[:, 0]
    order = np.argsort(ang)
    clus = {}
    for k in (4, 5, 6, 8):
        lab = fcluster(Z, k, "maxclust")
        clus[k] = {"ari_blizz": ari(lab, meta["blizz"]), "ari_fine": ari(lab, meta["fine"]),
                   "labels": lab.tolist()}
        print(f"k={k}: ARI Blizzard {clus[k]['ari_blizz']:.3f} fine {clus[k]['ari_fine']:.3f}")
    # permutation reference for ARI
    rng = np.random.RandomState(0)
    lab6 = np.array(clus[6]["labels"])
    perm = [ari(rng.permutation(lab6), meta["blizz"]) for _ in range(500)]
    out["ari_perm_95pct_k6"] = float(np.percentile(perm, 95))
    k_show = 6
    lab = np.array(clus[k_show]["labels"])
    cang = {c: np.arctan2(F[lab == c, 1].mean(), F[lab == c, 0].mean()) for c in np.unique(lab)}
    order = np.lexsort((ang, np.array([cang[c] for c in lab])))
    members = {}
    for c in np.unique(lab):
        hs = np.flatnonzero(lab == c)
        hs = hs[np.argsort(-np.linalg.norm(F[hs], axis=1))]
        roles = [C.BLIZZ_ROLES[meta["blizz"][h]] for h in hs]
        cen = F[hs].mean(0)
        members[int(c)] = {"n": int(len(hs)), "centroid": cen.tolist(),
                           "heroes": [f"{names[h]} ({C.BLIZZ_ROLES[meta['blizz'][h]][:1] if meta['blizz'][h] != 3 else 'RA'})"
                                      for h in hs],
                           "role_counts": {rr: roles.count(rr) for rr in sorted(set(roles))}}
        print(f"cluster {c} n={len(hs)} centroid {np.round(100 * cen, 2)}: "
              f"{', '.join(members[int(c)]['heroes'])}")
    out["clusters"] = {"k": k_show, "members": members,
                       "ari": {k: {"blizz": v["ari_blizz"], "fine": v["ari_fine"]}
                               for k, v in clus.items()}}

    # ---------- held-out validation by decile of model covariance
    _, _, _, _, r_adj = prepare(d)
    hold_rows = ~fitp[d["pid"]]
    S, P, N, _, _ = C.static_state(d, hold_rows, d["n_players"], H, r_adj)
    Pm = np.where(N >= 5, P, 0.0)
    Sm = np.where(N >= 5, S, 0.0)
    M, npair = empirical_cov(Sm, Pm, 5)
    Kf = V @ V.T
    iu = np.triu_indices(H, 1)
    pred = Kf[iu]
    emp = M[iu]
    wts = npair[iu]
    dec = np.searchsorted(np.percentile(pred, np.arange(10, 100, 10)), pred)
    gen = float(np.average(emp, weights=wts))
    val = []
    for q in range(10):
        s = dec == q
        val.append({"decile": q + 1, "pairs": int(s.sum()),
                    "model_cov_beyond_general_pp2": float(1e4 * pred[s].mean()),
                    "empirical_cov_pp2": float(1e4 * np.average(emp[s], weights=wts[s])),
                    "empirical_minus_mean_pp2": float(1e4 * (np.average(emp[s], weights=wts[s]) - gen))})
        print(val[-1])
    slope = np.polyfit(pred, emp, 1, w=np.sqrt(wts))[0]
    out["heldout_validation"] = {"deciles": val, "weighted_slope_emp_on_model": float(slope),
                                 "weighted_corr": float(np.corrcoef(pred * np.sqrt(wts),
                                                                    emp * np.sqrt(wts))[0, 1])}
    # same-role vs cross-role empirical covariance (held-out)
    same = meta["blizz"][iu[0]] == meta["blizz"][iu[1]]
    out["heldout_emp_cov_pp2"] = {
        "same_role": float(1e4 * np.average(emp[same], weights=wts[same])),
        "cross_role": float(1e4 * np.average(emp[~same], weights=wts[~same])),
        "model_same_role": float(1e4 * pred[same].mean()),
        "model_cross_role": float(1e4 * pred[~same].mean())}
    print("same vs cross role", out["heldout_emp_cov_pp2"])
    # top cross-role pairs by model beyond-general covariance (as correlation of
    # the factor part), with held-out empirical covariance for reference
    dk = np.sqrt(np.diag(Kf))
    Rf = Kf / np.outer(dk, dk)
    # only heroes with a sizeable loading (top 60% by norm) to avoid noisy directions
    big = np.linalg.norm(F, axis=1) >= np.percentile(np.linalg.norm(F, axis=1), 40)
    pairs = []
    for i, j in zip(*iu):
        if meta["blizz"][i] != meta["blizz"][j] and big[i] and big[j]:
            pairs.append((Kf[i, j], Rf[i, j], M[i, j], npair[i, j], i, j))
    pairs.sort(key=lambda t: -t[0])
    out["top_cross_role_pairs"] = [
        {"pair": f"{names[i]} ({C.BLIZZ_ROLES[meta['blizz'][i]]}) + {names[j]} "
                 f"({C.BLIZZ_ROLES[meta['blizz'][j]]})",
         "shared_factor_cov_pp2": float(1e4 * k), "factor_corr": float(rr),
         "heldout_emp_cov_pp2": float(1e4 * e), "heldout_players_both": int(n)}
        for k, rr, e, n, i, j in pairs[:15]]
    for p in out["top_cross_role_pairs"]:
        print(p)
    # hero loadings table
    out["hero_loadings_pp"] = {names[h]: [float(100 * x) for x in F[h]] for h in order}

    # ---------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # heatmap: model total skill correlation (general + factors + hero), and
    # the factor-only correlation, ordered by angle
    sp2 = prev["embedding_selection"][f"rank {r}"]["sd_pp"]["player"] ** 2 / 1e4
    sh2 = prev["embedding_selection"][f"rank {r}"]["sd_pp"]["hero"] ** 2 / 1e4
    Kt = sp2 + Kf + sh2 * np.eye(H)
    dt = np.sqrt(np.diag(Kt))
    Rt = Kt / np.outer(dt, dt)
    Mh = M.copy()
    dg = np.clip(np.diag(Mh), 1e-6, None)
    Re = Mh / np.sqrt(np.outer(dg, dg))
    np.fill_diagonal(Re, 1)
    fig, axs = plt.subplots(1, 2, figsize=(22, 11))
    for ax, mat, title in (
            (axs[0], Rt, f"Model: correlation of player skill across heroes (rank-{r} embedding)"),
            (axs[1], np.clip(Re, -1, 1), "Held-out players: empirical disattenuated correlation (noisy)")):
        im = ax.imshow(mat[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(H))
        ax.set_yticks(range(H))
        ax.set_xticklabels(names[order], rotation=90, fontsize=6)
        ax.set_yticklabels(names[order], fontsize=6)
        for t, h in zip(ax.get_yticklabels(), order):
            t.set_color(ROLE_COL[meta["blizz"][h]])
        for t, h in zip(ax.get_xticklabels(), order):
            t.set_color(ROLE_COL[meta["blizz"][h]])
        lo = lab[order]
        for b in np.flatnonzero(lo[1:] != lo[:-1]):
            ax.axhline(b + 0.5, color="k", lw=0.5)
            ax.axvline(b + 0.5, color="k", lw=0.5)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    handles = [plt.Line2D([0], [0], color=c, lw=6) for c in ROLE_COL]
    fig.legend(handles, C.BLIZZ_ROLES, loc="lower center", ncol=6, fontsize=10,
               title="label color = Blizzard role; heroes ordered by cluster, then by angle in the embedding; "
                     "black lines = Ward clusters (k=6)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(C.RESULTS, "fig_hero_skill_corr.png"), dpi=130)
    plt.close(fig)
    if F.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(13, 11))
        for h in range(H):
            ax.scatter(100 * F[h, 0], 100 * F[h, 1], color=ROLE_COL[meta["blizz"][h]], s=40)
            ax.annotate(names[h], (100 * F[h, 0], 100 * F[h, 1]), fontsize=7,
                        xytext=(3, 2), textcoords="offset points")
        ax.axhline(0, color="#999", lw=0.5)
        ax.axvline(0, color="#999", lw=0.5)
        ax.set_xlabel("axis 1 (pp): low-ladder, straightforward heroes  <->  high-ladder, "
                      "high-execution heroes")
        ax.set_ylabel("axis 2 (pp): popular mainstream heroes  <->  niche, unusual-kit heroes")
        ax.set_title(f"Hero embedding from player skill (rank {r}); two heroes close together "
                     "are ones the same players are good at", fontsize=11)
        ax.legend(handles, C.BLIZZ_ROLES, loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(C.RESULTS, "fig_hero_embedding.png"), dpi=130)
        plt.close(fig)
    with open(os.path.join(C.RESULTS, "p3_sd_similarity_viz.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
