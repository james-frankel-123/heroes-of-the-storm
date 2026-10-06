"""
P3: hero x hero SKILL similarity with real power, validated embeddings, and
the comparison with PLAY-PREFERENCE similarity (p3_pref_similarity.py).

Skill per player x hero cell: mean detrended residual r_adj = y - WP_drift -
experience offset over all snapshot games (2024-04 .. 2026-05-22). Each
cell's games are split into odd and even games in play order (game_date,
replay_id tiebreak): halves A and B.

Pair estimator (pair h, k; players with >= MIN games on BOTH heroes):
  cross   = mean over players of (x_hA x_kB + x_hB x_kA) / 2 - mean products
  var_h   = mean(x_hA x_hB) - mean_hA mean_hB   (same players)
  rho_hk  = cross / sqrt(var_h var_k)
Halves never share games, so game noise drops out of every product: rho is
the correlation of TRUE skill, disattenuated by split-half reliability
without a noise model. Two versions:
  total     x = half mean of the cell
  specific  x = half mean minus the player's same-half mean over all other
            heroes (excluding h). Cross products pair an A half with a B
            half, so the centering adds no shared noise. This is skill on h
            relative to the player's general level.
CIs: 200 player-bootstrap replicates. Coverage: pairs with >= NPAIR
qualifying players. Stability: the whole matrix on two random halves of
the players, correlated over covered pairs (the preference matrix gets the
same treatment on its own players).

Embedding validation: the p3_hs_fit kernels (+CF rank r, r = 1, 2, 3, 4, 8,
fit on the fit half of players, E window only) and the pure rank-2
factorization (p3_sd_similarity) imply a hero x hero correlation. It is
compared with rho estimated on the held-out half of players only.

Preference: volume-centered log(1 + games) correlation over players with
>= 100 games (the p3_pref_similarity.py definition, recomputed here with
the region-aware player key) and its rank-12 SVD cosine for ordering.

Run (from training/):
  NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    nice -n 19 taskset -c 48-63 python3 personalization/p3_hs_similarity.py
Outputs: results/p3_hs_similarity.json, results/fig_hero_skill_vs_pref.png,
cache/hs_similarity.npz
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

GT = os.path.join(C.CACHE, "gametime_2024q2.npz")
NPAIR = 100
NBOOT = 200
CAP = 200


def half_tables(d, r_adj, ts):
    """Per (player, hero): n, sums of halves A/B, and player totals."""
    H = NUM_HEROES
    npl = int(d["n_players"])
    key = d["pid"] * H + d["hero"]
    o = np.lexsort((d["replay_id"], ts, key))
    ks = key[o]
    brk = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    rk = np.arange(len(o)) - np.repeat(brk, np.diff(np.r_[brk, len(o)]))
    isA = np.empty(len(o), bool)
    isA[o] = rk % 2 == 0
    size = npl * H
    nA = np.bincount(key, weights=isA, minlength=size).reshape(npl, H)
    nB = np.bincount(key, weights=~isA, minlength=size).reshape(npl, H)
    sA = np.bincount(key, weights=r_adj * isA, minlength=size).reshape(npl, H)
    sB = np.bincount(key, weights=r_adj * ~isA, minlength=size).reshape(npl, H)
    return nA, nB, sA, sB


def features(nA, nB, sA, sB, mingames, specific):
    n = nA + nB
    Q = n >= mingames
    xA = np.where(Q, sA / np.maximum(nA, 1), 0.0)
    xB = np.where(Q, sB / np.maximum(nB, 1), 0.0)
    if specific:
        tA, tB = sA.sum(1, keepdims=True), sB.sum(1, keepdims=True)
        cA_n, cB_n = nA.sum(1, keepdims=True) - nA, nB.sum(1, keepdims=True) - nB
        ok = (cA_n >= 10) & (cB_n >= 10)
        Q = Q & ok
        cA = (tA - sA) / np.maximum(cA_n, 1)
        cB = (tB - sB) / np.maximum(cB_n, 1)
        xA = np.where(Q, xA - cA, 0.0)
        xB = np.where(Q, xB - cB, 0.0)
    rows = Q.any(1)
    a = np.where(Q, np.minimum(n, CAP), 0.0)
    return xA[rows], xB[rows], a[rows], np.flatnonzero(rows)


def pair_matrix(xA, xB, A, w=None):
    """A: per-cell weights (0 = not qualified). Pair cross products are
    weighted by a_h a_k (inverse variance of a noise-dominated product);
    per-hero split-half variance by a_h^2 over all of the hero's players."""
    Aw = A if w is None else A * w[:, None]
    Q = (A > 0).astype(np.float64)
    Qw = Q if w is None else Q * w[:, None]
    N = Q.T @ Qw
    Nw = np.maximum(A.T @ Aw, 1e-12)
    mA = ((xA * A).T @ Aw) / Nw
    mB = ((xB * A).T @ Aw) / Nw
    AB = ((xA * A).T @ (xB * Aw)) / Nw
    cross = 0.5 * (AB + AB.T) - 0.5 * (mA * mB.T + mB * mA.T)
    a2 = (A * Aw).sum(0)
    vh = ((xA * xB * A * Aw).sum(0) / np.maximum(a2, 1e-12)
          - ((xA * A * Aw).sum(0) / np.maximum(a2, 1e-12)) * ((xB * A * Aw).sum(0) / np.maximum(a2, 1e-12)))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = cross / np.sqrt(np.outer(vh, vh))
    rho[:, vh <= 0] = np.nan
    rho[vh <= 0, :] = np.nan
    np.fill_diagonal(rho, 1.0)
    vv = np.diag(vh)
    return rho, N, cross, vv


def pref_matrix(d, min_games=100, k=12, players_mask=None):
    H = NUM_HEROES
    npl = int(d["n_players"])
    cnt = np.bincount(d["pid"] * H + d["hero"], minlength=npl * H).reshape(npl, H).astype(float)
    keep = cnt.sum(1) >= min_games
    if players_mask is not None:
        keep &= players_mask
    X = np.log1p(cnt[keep])
    X -= X.mean(1, keepdims=True)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    R = Z.T @ Z / len(Z)
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    emb = Vt[:k].T * S[:k]
    embn = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return R, embn @ embn.T, int(keep.sum())


def offdiag(M, mask=None):
    iu = np.triu_indices(M.shape[0], 1)
    v = M[iu]
    return v if mask is None else v[mask[iu]]


def wcorr(a, b, w):
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(w) & (w > 0)
    a, b, w = a[ok], b[ok], w[ok]
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    return float(np.sum(w * (a - ma) * (b - mb)) /
                 np.sqrt(np.sum(w * (a - ma) ** 2) * np.sum(w * (b - mb) ** 2)))


def main():
    d = C.load_slots()
    names = np.array([str(x) for x in d["hero_names"]])
    meta = C.hero_meta(d["hero_names"])
    H = NUM_HEROES
    _, _, _, _, r_adj = prepare(d)
    z = np.load(GT)
    o = np.argsort(z["replay_ids"])
    ts = z["ts"][o][np.searchsorted(z["replay_ids"][o], d["replay_id"])]
    nA, nB, sA, sB = half_tables(d, r_adj, ts)
    kz = np.load(os.path.join(C.CACHE, "hs_kernels.npz"))
    fitp = kz["fit_players"]
    rng = np.random.RandomState(0)
    out = {"NPAIR_min_players": NPAIR, "weight_cap_games": CAP}
    iu = np.triu_indices(H, 1)
    res = {}
    for MIN, ver in ((30, "total 30+"), (30, "specific 30+"), (4, "total"), (4, "specific")):
        xA, xB, Q, pidx = features(nA, nB, sA, sB, MIN, ver.startswith("specific"))
        rho, N, cross, vv = pair_matrix(xA, xB, Q)
        boots = []
        for b in range(NBOOT):
            w = np.bincount(rng.randint(0, len(Q), len(Q)), minlength=len(Q)).astype(float)
            boots.append(pair_matrix(xA, xB, Q, w)[0])
        B = np.array(boots)
        lo = np.nanpercentile(B, 2.5, axis=0)
        hi = np.nanpercentile(B, 97.5, axis=0)
        se = np.nanstd(B, axis=0)
        # clip absurd values for summaries (tiny true-variance pairs)
        covered = (N >= NPAIR) & np.isfinite(rho)
        # split-half stability over players
        half = rng.rand(len(Q)) < 0.5
        r1 = pair_matrix(xA[half], xB[half], Q[half])[0]
        r2 = pair_matrix(xA[~half], xB[~half], Q[~half])[0]
        N1 = (Q[half] > 0).astype(float).T @ (Q[half] > 0).astype(float)
        N2 = (Q[~half] > 0).astype(float).T @ (Q[~half] > 0).astype(float)
        both = covered & (N1 >= NPAIR / 2) & (N2 >= NPAIR / 2) & np.isfinite(r1) & np.isfinite(r2)
        both &= np.isfinite(se) & (se > 0)
        a, b_ = offdiag(np.clip(r1, -2, 2), both), offdiag(np.clip(r2, -2, 2), both)
        wt = 1 / offdiag(se, both) ** 2
        stab = {"pairs": int(both[iu].sum()), "r": float(np.corrcoef(a, b_)[0, 1]),
                "r_weighted": wcorr(a, b_, wt)}
        # also restricted to precise pairs (bootstrap sd < 0.15)
        prec = both & (se < 0.15)
        if prec[iu].sum() > 20:
            stab["precise_pairs"] = int(prec[iu].sum())
            stab["r_precise"] = float(np.corrcoef(offdiag(r1, prec), offdiag(r2, prec))[0, 1])
        res[ver] = dict(rho=rho, N=N, lo=lo, hi=hi, se=se, covered=covered, Q=Q, xA=xA, xB=xB,
                        pidx=pidx)
        cv = offdiag(covered.astype(float)).astype(bool)
        rv = offdiag(rho)[cv]
        out[ver] = {
            "min_games_each_hero": MIN,
            "players_with_any_qualifying_cell": int(len(Q)),
            "qualifying_cells": int((Q > 0).sum()),
            "pairs_total": int(len(iu[0])),
            "pairs_covered": int(cv.sum()),
            "coverage_share": float(cv.mean()),
            "pairs_by_min_players": {str(t): int((offdiag(N) >= t).sum()) for t in (30, 50, 100, 200, 500, 1000)},
            "median_players_per_covered_pair": float(np.median(offdiag(N)[cv])),
            "median_boot_sd_covered": float(np.nanmedian(offdiag(se)[cv])),
            "rho_covered_median": float(np.nanmedian(rv)),
            "rho_covered_iqr": [float(np.nanpercentile(rv, 25)), float(np.nanpercentile(rv, 75))],
            "share_ci_excludes_zero": float(np.mean((offdiag(lo)[cv] > 0) | (offdiag(hi)[cv] < 0))),
            "median_splithalf_var_pp2": float(1e4 * np.nanmedian(np.diag(vv))),
            "stability_split_players": stab}
        print(ver, json.dumps(out[ver]), flush=True)

    # ---------- held-out validation of embeddings (specific and total)
    out["embedding_validation"] = {}
    bases = {k[6:]: kz[k] for k in kz.files if k.startswith("basis_")}
    sim = np.load(os.path.join(C.CACHE, "sd_similarity.npz"))
    for ver in ("total", "specific"):
        r_ = res[ver]
        hold = ~fitp[r_["pidx"]]
        rho_h, N_h = pair_matrix(r_["xA"][hold], r_["xB"][hold], r_["Q"][hold])[:2]
        rho_f, N_f = pair_matrix(r_["xA"][~hold], r_["xB"][~hold], r_["Q"][~hold])[:2]
        ok = (N_h >= NPAIR / 2) & (N_f >= NPAIR / 2) & np.isfinite(rho_h) & np.isfinite(rho_f)
        ok &= np.isfinite(r_["se"]) & (r_["se"] > 0)
        wt = 1 / offdiag(r_["se"], ok) ** 2
        tgt = np.clip(offdiag(rho_h, ok), -2, 2)
        ceiling = wcorr(tgt, np.clip(offdiag(rho_f, ok), -2, 2), wt)
        v = {"pairs": int(ok[iu].sum()),
             "fit_half_empirical_rho (ceiling)": ceiling}
        cands = {}
        for nm in ("player+hero", "player+role+melee+hero", "+CF rank 1", "+CF rank 2",
                   "+CF rank 3", "+CF rank 4", "+CF rank 8"):
            bl = KERNELS[nm]
            th = kz["theta_" + nm]
            Kf = C.kernel_from([bases[b] for b in bl], th)
            if ver == "specific":
                Kf = Kf - np.exp(th[0]) * bases["player"]  # drop the general-skill term
            cands[nm] = Kf
        Vp = sim["V"]
        cands["pure factorization rank 2"] = Vp @ Vp.T + (0.0 if ver == "specific" else 0.02 ** 2)
        R_pref, cos_pref, _ = pref_matrix(d)
        cands["preference correlation"] = R_pref
        for nm, Kf in cands.items():
            dk = np.sqrt(np.abs(np.diag(Kf))) + 1e-12
            Rm = Kf / np.outer(dk, dk)
            pred = offdiag(Rm, ok)
            okf = np.isfinite(pred) & np.isfinite(tgt) & np.isfinite(wt)
            X_ = np.column_stack([np.ones(okf.sum()), pred[okf]]) * np.sqrt(wt[okf])[:, None]
            slope = np.linalg.lstsq(X_, tgt[okf] * np.sqrt(wt[okf]), rcond=None)[0][1]
            v[nm] = {"weighted_corr": wcorr(pred, tgt, wt), "slope": float(slope)}
        out["embedding_validation"][ver] = v
        print("validation", ver, json.dumps(v), flush=True)

    # ---------- skill vs preference
    R_pref, cos_pref, npref = pref_matrix(d)
    ph = np.random.RandomState(5).rand(int(d["n_players"])) < 0.5
    Ra, _, _ = pref_matrix(d, players_mask=ph)
    Rb, _, _ = pref_matrix(d, players_mask=~ph)
    out["preference"] = {"players": npref,
                         "split_half_r": float(np.corrcoef(offdiag(Ra), offdiag(Rb))[0, 1])}
    comp = {}
    for ver in ("total", "specific"):
        r_ = res[ver]
        cv = r_["covered"]
        s = np.clip(offdiag(r_["rho"], cv), -2, 2)
        p = offdiag(R_pref, cv)
        se = offdiag(r_["se"], cv)
        wt = 1 / se ** 2
        rel_s = out[ver]["stability_split_players"]["r_weighted"]
        rel_p = out["preference"]["split_half_r"]
        r_obs = wcorr(s, p, wt)
        # full-sample reliability from the half-sample one (Spearman-Brown)
        rel_s_full = 2 * rel_s / (1 + rel_s) if rel_s > 0 else float("nan")
        rel_p_full = 2 * rel_p / (1 + rel_p)
        comp[ver] = {"pairs": int(cv[iu].sum()), "corr_weighted": r_obs,
                     "corr_unweighted": float(np.corrcoef(s, p)[0, 1]),
                     "corr_disattenuated": float(r_obs / np.sqrt(rel_s_full * rel_p_full)),
                     "skill_matrix_reliability_full": rel_s_full}
        # same-role vs cross-role means
        same = offdiag((meta["blizz"][:, None] == meta["blizz"][None, :]), cv)
        comp[ver]["mean_skill_rho_same_role"] = float(np.average(s[same], weights=wt[same]))
        comp[ver]["mean_skill_rho_cross_role"] = float(np.average(s[~same], weights=wt[~same]))
        comp[ver]["mean_pref_r_same_role"] = float(p[same].mean())
        comp[ver]["mean_pref_r_cross_role"] = float(p[~same].mean())
        # divergences: residual of skill on preference (weighted fit), in SE units
        A = np.column_stack([np.ones(len(p)), p])
        beta = np.linalg.lstsq(A * np.sqrt(wt)[:, None], s * np.sqrt(wt), rcond=None)[0]
        resid = s - A @ beta
        zres = resid / se
        ii, jj = iu[0][cv[iu]], iu[1][cv[iu]]
        lo_, hi_ = offdiag(r_["lo"], cv), offdiag(r_["hi"], cv)
        Np = offdiag(r_["N"], cv)

        def row(t):
            return {"pair": f"{names[ii[t]]} + {names[jj[t]]}",
                    "roles": f"{C.BLIZZ_ROLES[meta['blizz'][ii[t]]]} / {C.BLIZZ_ROLES[meta['blizz'][jj[t]]]}",
                    "pref_r": float(p[t]), "skill_rho": float(s[t]),
                    "skill_ci": [float(lo_[t]), float(hi_[t])], "players": int(Np[t]),
                    "z_vs_pref_line": float(zres[t])}
        comp[ver]["fit_skill_on_pref"] = beta.tolist()
        # co-played but skill does not transfer: high pref, skill well below line
        hp = np.flatnonzero(p >= np.percentile(p, 90))
        comp[ver]["coplayed_no_transfer"] = [row(t) for t in hp[np.argsort(zres[hp])][:12]]
        # transferable but rarely co-played: low pref, skill well above line
        lp = np.flatnonzero(p <= np.percentile(p, 25))
        comp[ver]["transfer_not_coplayed"] = [row(t) for t in lp[np.argsort(-zres[lp])][:12]]
        comp[ver]["top_skill_pairs"] = [row(t) for t in np.argsort(-(s - 1.0 * se))[:15]]
        comp[ver]["top_pref_pairs_skill"] = [row(t) for t in np.argsort(-p)[:15]]
        print("compare", ver, {k: v for k, v in comp[ver].items() if not isinstance(v, list)}, flush=True)
    out["skill_vs_pref"] = comp

    # ---------- figure: same hero order (preference clustering)
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    D = np.clip(1 - cos_pref, 0, None)
    np.fill_diagonal(D, 0)
    order = leaves_list(linkage(squareform(D, checks=False), "average"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    role_col = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#9c755f"]
    fig = plt.figure(figsize=(30, 11))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8])
    S_show = res["specific"]["rho"].copy()
    S_show[~res["specific"]["covered"]] = np.nan
    np.fill_diagonal(S_show, np.nan)
    Pm = R_pref.copy()
    np.fill_diagonal(Pm, np.nan)
    for gi_, (mat, title, vmax) in enumerate((
            (Pm, f"Play preference: volume-centered log(1+games) correlation ({npref:,} players)", 0.4),
            (S_show, f"Skill: disattenuated correlation beyond general skill (players with {MIN}+ games "
                     f"on both; grey = <{NPAIR} players)", 1.0))):
        ax = fig.add_subplot(gs[gi_])
        cm = plt.get_cmap("RdBu_r").copy()
        cm.set_bad("#dddddd")
        im = ax.imshow(np.clip(mat[np.ix_(order, order)], -vmax, vmax), cmap=cm, vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(H))
        ax.set_yticks(range(H))
        ax.set_xticklabels(names[order], rotation=90, fontsize=5.5)
        ax.set_yticklabels(names[order], fontsize=5.5)
        for t, h in zip(ax.get_yticklabels(), order):
            t.set_color(role_col[meta["blizz"][h]])
        for t, h in zip(ax.get_xticklabels(), order):
            t.set_color(role_col[meta["blizz"][h]])
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax = fig.add_subplot(gs[2])
    cv = res["specific"]["covered"]
    p = offdiag(R_pref, cv)
    s = offdiag(res["specific"]["rho"], cv)
    se = offdiag(res["specific"]["se"], cv)
    same = offdiag((meta["blizz"][:, None] == meta["blizz"][None, :]), cv)
    ax.errorbar(p[~same], np.clip(s[~same], -1.5, 1.5), yerr=np.minimum(se[~same], 1), fmt="o", ms=2.5,
                alpha=0.35, color="#e15759", elinewidth=0.4, label="cross-role pair")
    ax.errorbar(p[same], np.clip(s[same], -1.5, 1.5), yerr=np.minimum(se[same], 1), fmt="o", ms=2.5,
                alpha=0.35, color="#4e79a7", elinewidth=0.4, label="same-role pair")
    beta = comp["specific"]["fit_skill_on_pref"]
    xs = np.linspace(p.min(), p.max(), 10)
    ax.plot(xs, beta[0] + beta[1] * xs, "k-", lw=1)
    ax.axhline(0, color="#999", lw=0.5)
    ax.set_xlabel("preference similarity (r)")
    ax.set_ylabel("skill similarity beyond general skill (rho, bootstrap sd bars)")
    ax.set_title(f"Per hero pair ({len(p)} covered pairs); weighted r = "
                 f"{comp['specific']['corr_weighted']:.2f}", fontsize=10)
    ax.legend(fontsize=8)
    handles = [plt.Line2D([0], [0], color=c, lw=6) for c in role_col]
    fig.legend(handles, C.BLIZZ_ROLES, loc="lower center", ncol=6, fontsize=9,
               title="label color = Blizzard role; both heatmaps in the preference-cluster order")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(C.RESULTS, "fig_hero_skill_vs_pref.png"), dpi=120)
    np.savez(os.path.join(C.CACHE, "hs_similarity.npz"),
             **{f"{v}_{k}": res[v][k] for v in res for k in ("rho", "N", "lo", "hi", "se", "covered")},
             pref=R_pref, pref_cos=cos_pref, order=order)
    with open(os.path.join(C.RESULTS, "p3_hs_similarity.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("done")


if __name__ == "__main__":
    main()
