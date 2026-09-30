"""
P3 phase 2, item 4: hero similarity through common strengths (descriptive).

1. Empirical hero x hero correlation of player skill. Per player and hero,
   the mean detrended residual over all snapshot games (2024-04 .. 2026-05).
   Cross-hero covariances are precision-weighted cross products; game noise
   is independent across cells, so they are unbiased. Diagonal variances
   subtract each cell's noise variance (disattenuation). A split-half check
   (odd vs even games of the same cell) gives each hero's reliability and a
   second, fully noise-free estimate of the diagonal.
2. Learned embeddings: K = s_p^2 J + s_h^2 I + V V' (pure factorization,
   no side features) for rank r = 1..8, plus the phase-1 side-feature
   version, by marginal likelihood on the fit half of players and scored on
   the held-out half. The chosen V is rotated to principal axes; each axis
   is labeled by its extreme heroes and its correlation with hero side
   features and usage statistics.
3. Hero clusters: average-linkage clustering of the correlation beyond
   general skill (the model-implied correlation of V V' + s_h^2 I, and the
   empirical excess correlation). Adjusted Rand index against Blizzard and
   fine roles; cross-role members listed.
4. Heatmap PNG ordered by cluster; top cross-role pairs.

Usage (from training/): python3 personalization/p3_sd_similarity.py
Outputs: results/p3_sd_similarity.json, results/fig_hero_skill_corr.png
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C
from p3_hs_fit import prepare

GLOBAL = {"Dehaka", "Falstad", "Brightwing", "Abathur", "Zagara", "Medivh", "Nova",
          "Stitches", "Garrosh", "Anub'arak", "Tyrande", "Malthael", "Zeratul", "Samuro",
          "Valeera", "Illidan", "Genji", "Tracer"}
# Heroes with a notably high mechanical ceiling (hand-labeled, from common
# community difficulty tiers; used only to label axes).
HARD = {"Abathur", "Alarak", "Cho", "Gall", "Chen", "Deathwing", "Genji", "Illidan",
        "Kerrigan", "Maiev", "Medivh", "Qhira", "Samuro", "The Lost Vikings", "Tracer",
        "Valeera", "Zeratul", "Hanzo", "Lunara", "Zul'jin", "Chromie", "Kel'Thuzad",
        "Li-Ming", "Tychus", "D.Va", "Lúcio", "Ana"}


def ari(a, b):
    from math import comb
    a = np.asarray(a)
    b = np.asarray(b)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    M = np.zeros((len(ua), len(ub)), int)
    np.add.at(M, (ia, ib), 1)
    s = sum(comb(int(x), 2) for x in M.ravel())
    sa = sum(comb(int(x), 2) for x in M.sum(1))
    sb = sum(comb(int(x), 2) for x in M.sum(0))
    n = comb(len(a), 2)
    exp = sa * sb / n
    return float((s - exp) / (0.5 * (sa + sb) - exp))


def empirical_cov(S, P, min_games_p):
    Y = np.where(P > 0, S / np.maximum(P, 1e-12), 0.0)
    k0 = 0.003
    W = np.where(P > 0, P / (1 + P * k0), 0.0)
    A = W * Y
    num = A.T @ A
    den = W.T @ W
    M = num / np.maximum(den, 1e-12)
    dg = (W ** 2 * (Y ** 2 - np.where(P > 0, 1 / np.maximum(P, 1e-12), 0))).sum(0) \
        / np.maximum((W ** 2).sum(0), 1e-12)
    M[np.diag_indices_from(M)] = dg
    # effective number of players supporting each pair
    npair = (W > 0).astype(float).T @ (W > 0).astype(float)
    return M, npair


def main():
    d = C.load_slots()
    H = len(d["hero_names"])
    names = np.array([str(x) for x in d["hero_names"]])
    meta = C.hero_meta(d["hero_names"])
    kz = np.load(os.path.join(C.CACHE, "hs_kernels.npz"))
    _, _, _, _, r_adj = prepare(d)
    allrows = np.ones(len(d["pid"]), bool)
    S, P, N, _, _ = C.static_state(d, allrows, d["n_players"], H, r_adj)
    out = {}

    # ---------- 1. empirical correlation (cells with >= 5 games)
    Pm = np.where(N >= 5, P, 0.0)
    Sm = np.where(N >= 5, S, 0.0)
    M, npair = empirical_cov(Sm, Pm, 5)
    # split-half diagonal: odd vs even games of each cell
    o = np.lexsort((d["replay_id"], d["day"], d["pid"] * 128 + d["hero"]))
    key = (d["pid"] * 128 + d["hero"])[o]
    brk = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    rank = np.arange(len(o)) - np.repeat(brk, np.diff(np.r_[brk, len(o)]))
    half = np.empty(len(o), bool)
    half[o] = rank % 2 == 0
    SA, PA, NA, _, _ = C.static_state(d, half, d["n_players"], H, r_adj)
    SB, PB, NB, _, _ = C.static_state(d, ~half, d["n_players"], H, r_adj)
    ok = (NA >= 5) & (NB >= 5)
    YA = np.where(ok, SA / np.maximum(PA, 1e-12), 0)
    YB = np.where(ok, SB / np.maximum(PB, 1e-12), 0)
    wts = np.where(ok, 1.0 / (1 / np.maximum(PA, 1e-12) + 1 / np.maximum(PB, 1e-12) + 0.003), 0)
    sh_var = (wts * YA * YB).sum(0) / np.maximum(wts.sum(0), 1e-12)
    tot_var = (wts * (YA ** 2 + YB ** 2) / 2).sum(0) / np.maximum(wts.sum(0), 1e-12)
    rel = sh_var / np.maximum(tot_var, 1e-12)
    out["split_half"] = {"median_reliability_per_cell": float(np.median(rel)),
                         "diag_sd_pp_splithalf_median": float(100 * np.sqrt(np.median(np.clip(sh_var, 0, None)))),
                         "diag_sd_pp_moments_median": float(100 * np.sqrt(np.median(np.clip(np.diag(M), 0, None))))}
    dg = np.clip(np.diag(M), 1e-6, None)
    R = M / np.sqrt(np.outer(dg, dg))
    np.fill_diagonal(R, 1.0)
    off = R[~np.eye(H, dtype=bool)]
    out["empirical_corr"] = {"median_offdiag": float(np.median(off)),
                             "iqr": [float(np.percentile(off, 25)), float(np.percentile(off, 75))],
                             "min_pair_players": int(npair[~np.eye(H, dtype=bool)].min())}
    print("empirical corr", out["empirical_corr"], out["split_half"], flush=True)

    # ---------- 2. embeddings by held-out likelihood
    fitp = kz["fit_players"]
    act = P.sum(1) > 0
    e_mask = d["in_sample"] & (d["day"] >= C.day_of(C.E_START))
    SE, PE, _, _, _ = C.static_state(d, e_mask, d["n_players"], H, r_adj)
    actE = PE.sum(1) > 0
    Sf, Pf = np.ascontiguousarray(SE[fitp & actE]), np.ascontiguousarray(PE[fitp & actE])
    Sh, Ph = np.ascontiguousarray(SE[~fitp & actE]), np.ascontiguousarray(PE[~fitp & actE])
    B2 = [np.ones((H, H)), np.eye(H)]
    th0, _ = C.fit_kernel(Sf, Pf, B2)
    llh0 = float(C._ml_terms(Sh, Ph, np.stack(B2), th0)[0].sum())
    emb = {"player+hero": {"heldout_ll": llh0}}
    Vs = {}
    for r in range(1, 9):
        th, V, llf = C.fit_kernel_lowrank(Sf, Pf, B2, r, th0, seed=r)
        Kb = [B2[0], B2[1], V @ V.T]
        llh = float(C._ml_terms(Sh, Ph, np.stack(Kb), np.r_[th, 0.0])[0].sum())
        emb[f"rank {r}"] = {"fit_ll": llf, "heldout_ll": llh, "gain_vs_player+hero": llh - llh0,
                            "sd_pp": {"player": float(100 * np.sqrt(np.exp(th[0]))),
                                      "hero": float(100 * np.sqrt(np.exp(th[1])))}}
        Vs[r] = (th, V)
        print(f"rank {r}: held-out gain {llh - llh0:+.1f}", flush=True)
    best_r = max(range(1, 9), key=lambda r: emb[f"rank {r}"]["heldout_ll"])
    out["embedding_selection"] = emb
    out["best_rank_pure"] = best_r
    th, V = Vs[best_r]
    # principal axes
    U, s, _ = np.linalg.svd(V, full_matrices=False)
    F = U * s
    for j in range(F.shape[1]):
        if F[np.argmax(np.abs(F[:, j])), j] < 0:
            F[:, j] *= -1
    # side features and usage stats
    tot_games = N.sum(0)
    players_h = (N > 0).sum(0)
    mmr = d["player_mmr"].astype(float)
    okm = ~np.isnan(mmr)
    mean_mmr = np.bincount(d["hero"][okm], weights=mmr[okm], minlength=H) / \
        np.maximum(np.bincount(d["hero"][okm], minlength=H), 1)
    feats = {"melee": meta["melee"].astype(float),
             "ranged AA": (meta["fine"] == meta["fine_names"].index("ranged_aa")).astype(float),
             "mage": (meta["fine"] == meta["fine_names"].index("ranged_mage")).astype(float),
             "healer": (meta["blizz"] == 2).astype(float),
             "tank": (meta["blizz"] == 0).astype(float),
             "bruiser": (meta["blizz"] == 1).astype(float),
             "melee assassin": (meta["blizz"] == 4).astype(float),
             "global": np.array([n in GLOBAL for n in names], float),
             "high mechanics (hand label)": np.array([n in HARD for n in names], float),
             "log popularity": np.log(tot_games),
             "games per player on hero": tot_games / players_h,
             "mean player MMR": mean_mmr}
    axes = []
    for j in range(F.shape[1]):
        o_ = np.argsort(F[:, j])
        cors = {k: float(np.corrcoef(F[:, j], v)[0, 1]) for k, v in feats.items()}
        axes.append({"sd_pp": float(100 * np.sqrt((F[:, j] ** 2).mean())),
                     "low": list(names[o_[:10]]), "high": list(names[o_[::-1][:10]]),
                     "corr_with_features": dict(sorted(cors.items(), key=lambda kv: -abs(kv[1])))})
        top3 = list(axes[-1]["corr_with_features"].items())[:3]
        print(f"axis {j + 1} sd {axes[-1]['sd_pp']:.2f}pp high {axes[-1]['high'][:6]} low "
              f"{axes[-1]['low'][:6]} | {top3}", flush=True)
    out["axes"] = axes

    # ---------- 3. clusters (correlation beyond general skill)
    Kh = V @ V.T + np.exp(th[1]) * np.eye(H)
    dk = np.sqrt(np.diag(Kh))
    Rm = Kh / np.outer(dk, dk)
    # empirical excess: subtract the general-skill share estimated by the model
    sp2 = np.exp(th[0])
    Ex = (M - sp2) / np.sqrt(np.outer(np.clip(dg - sp2, 1e-6, None), np.clip(dg - sp2, 1e-6, None)))
    np.fill_diagonal(Ex, 1.0)
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
    from scipy.spatial.distance import squareform
    Dm = np.clip(1 - Rm, 0, None)
    np.fill_diagonal(Dm, 0)
    Z = linkage(squareform(Dm, checks=False), "average")
    order = leaves_list(Z)
    clus = {}
    for k in (4, 6, 8, 10):
        lab = fcluster(Z, k, "maxclust")
        clus[k] = {"labels": lab.tolist(), "ari_blizz": ari(lab, meta["blizz"]),
                   "ari_fine": ari(lab, meta["fine"])}
        print(f"k={k}: ARI vs Blizzard role {clus[k]['ari_blizz']:.3f}, vs fine role "
              f"{clus[k]['ari_fine']:.3f}", flush=True)
    # also cluster the embedding directly with roles as reference: agreement of
    # model and empirical excess correlation
    iu = np.triu_indices(H, 1)
    out["model_vs_empirical_excess_corr"] = float(np.corrcoef(Rm[iu], Ex[iu])[0, 1])
    k_show = 8
    lab = np.array(clus[k_show]["labels"])
    members = {}
    for c in np.unique(lab):
        hs = np.flatnonzero(lab == c)
        roles = [C.BLIZZ_ROLES[meta["blizz"][h]] for h in hs]
        maj = max(set(roles), key=roles.count)
        members[int(c)] = {"heroes": [f"{names[h]} ({C.BLIZZ_ROLES[meta['blizz'][h]]})" for h in hs],
                           "majority_role": maj, "share_majority": roles.count(maj) / len(roles)}
    out["clusters"] = {"k": k_show, "members": members,
                       "ari": {k: {"blizz": v["ari_blizz"], "fine": v["ari_fine"]} for k, v in clus.items()}}
    for c, v in members.items():
        print(f"cluster {c} ({v['majority_role']} {v['share_majority']:.0%}): {', '.join(v['heroes'])}",
              flush=True)
    # top cross-role pairs by model beyond-general correlation, with empirical support
    pairs = []
    for i, j in zip(*iu):
        if meta["blizz"][i] != meta["blizz"][j]:
            pairs.append((Rm[i, j], Ex[i, j], R[i, j], npair[i, j], names[i], names[j],
                          C.BLIZZ_ROLES[meta["blizz"][i]], C.BLIZZ_ROLES[meta["blizz"][j]]))
    pairs.sort(key=lambda t: -t[0])
    out["top_cross_role_pairs"] = [
        {"pair": f"{a} ({ra}) + {b} ({rb})", "model_corr_beyond_general": float(m),
         "empirical_excess_corr": float(e), "empirical_total_corr": float(t), "players_both": int(n)}
        for m, e, t, n, a, b, ra, rb in pairs[:20]]
    same = Rm[iu][meta["blizz"][iu[0]] == meta["blizz"][iu[1]]]
    cross = Rm[iu][meta["blizz"][iu[0]] != meta["blizz"][iu[1]]]
    out["mean_beyond_general_corr"] = {"same_role": float(same.mean()), "cross_role": float(cross.mean())}
    samef = Rm[iu][meta["fine"][iu[0]] == meta["fine"][iu[1]]]
    out["mean_beyond_general_corr"]["same_fine_role"] = float(samef.mean())
    for p in out["top_cross_role_pairs"][:12]:
        print(p, flush=True)

    # ---------- 4. heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(22, 11))
    role_col = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#9c755f"]
    for ax, mat, title in ((axs[0], R, "Empirical skill correlation (disattenuated), all skill"),
                           (axs[1], Rm, f"Correlation beyond general skill (rank-{best_r} embedding)")):
        im = ax.imshow(mat[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(H))
        ax.set_yticks(range(H))
        ax.set_xticklabels(names[order], rotation=90, fontsize=6)
        ax.set_yticklabels(names[order], fontsize=6)
        for t, h in zip(ax.get_yticklabels(), order):
            t.set_color(role_col[meta["blizz"][h]])
        for t, h in zip(ax.get_xticklabels(), order):
            t.set_color(role_col[meta["blizz"][h]])
        ax.set_title(title, fontsize=11)
        # cluster boundaries
        lo = lab[order]
        for b in np.flatnonzero(lo[1:] != lo[:-1]):
            ax.axhline(b + 0.5, color="k", lw=0.6)
            ax.axvline(b + 0.5, color="k", lw=0.6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    handles = [plt.Line2D([0], [0], color=c, lw=6) for c in role_col]
    fig.legend(handles, C.BLIZZ_ROLES, loc="lower center", ncol=6, fontsize=10,
               title="label color = Blizzard role; black lines = 8 clusters")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    png = os.path.join(C.RESULTS, "fig_hero_skill_corr.png")
    fig.savefig(png, dpi=130)
    out["figure"] = png
    np.savez(os.path.join(C.CACHE, "sd_similarity.npz"), R=R, Rm=Rm, Ex=Ex, V=V, F=F,
             order=order, labels=lab)
    with open(os.path.join(C.RESULTS, "p3_sd_similarity.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", png)


if __name__ == "__main__":
    main()
