"""
P3 hero strength, stage 1: fit the hero kernels on the estimation window.

Uses only E games (2024-04-01 .. WP training cutoff, build <= 2.55.14.95918).
Players are split in half at random: kernel hyperparameters and the
data-driven similarity matrices come from the fit half; kernels are compared
by marginal log likelihood on the held-out half.

Similarity matrices:
  co-strength (CF)  method-of-moments hero x hero covariance of player x hero
                    residual means (cross-products of different heroes are
                    unbiased because game noise is independent across
                    cells), minus the fitted role structure, top-10
                    eigencomponents. This is probabilistic PCA of the
                    player x hero residual matrix with missing cells.
  co-play           cosine similarity of 16-d hero vectors from an SVD of
                    players' log play-share matrix (who plays what).

Usage (from training/):
  python3 personalization/p3_hs_core.py is a library; first run
  python3 personalization/p3_hs_fit.py slots   # build cache/hs_slots.npz
  python3 personalization/p3_hs_fit.py fit
Outputs: cache/hs_kernels.npz, results/p3_hs_fit.json
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C

KOUT = os.path.join(C.CACHE, "hs_kernels.npz")

KERNELS = {
    "player": ["player"],
    "player+hero": ["player", "hero"],
    "player+role+hero": ["player", "role", "fine", "hero"],
    "player+role+melee+hero": ["player", "role", "fine", "melee", "hero"],
    "+co-play": ["player", "role", "fine", "melee", "hero", "coplay"],
    "+co-strength moments": ["player", "role", "fine", "melee", "hero", "costr"],
    "+CF rank 1": ["player", "role", "fine", "melee", "hero", "cf1"],
    "+CF rank 2": ["player", "role", "fine", "melee", "hero", "cf2"],
    "+CF rank 3": ["player", "role", "fine", "melee", "hero", "cf3"],
    "+CF rank 4": ["player", "role", "fine", "melee", "hero", "cf4"],
    "+CF rank 8": ["player", "role", "fine", "melee", "hero", "cf8"],
    "+CF rank 2 +co-play": ["player", "role", "fine", "melee", "hero", "coplay", "cf2cp"],
}
STRUCT = ["player", "role", "fine", "melee", "hero"]


def main_kernels():
    """Kernels carried through evaluation: the nested baselines, co-play,
    and the best kernel by held-out marginal likelihood."""
    with open(os.path.join(C.RESULTS, "p3_hs_fit.json")) as f:
        k = json.load(f)["kernels"]
    top = max(v["heldout_ll"] for v in k.values())
    # simplest kernel within 2 log-likelihood units of the best
    best = min((n for n in k if k[n]["heldout_ll"] >= top - 2.0),
               key=lambda n: (len(k[n]["bases"]), -k[n]["heldout_ll"]))
    main = ["player", "player+hero", "player+role+melee+hero", "+co-play"]
    if best not in main:
        main.append(best)
    return main, best


def prepare(d):
    """Experience offsets (fit on E) and detrended residuals for all rows."""
    days = d["day"]
    e_mask = d["in_sample"] & (days >= C.day_of(C.E_START))
    n_p, n_ph = C.experience_counts(d)
    table = C.fit_experience(d["r"], n_p, n_ph, e_mask)
    r_adj = d["r"] - table[C.exp_bins(n_p, n_ph)]
    return e_mask, n_p, n_ph, table, r_adj


def mom_covariance(S, P):
    """Moment estimate of the hero x hero covariance of cell means."""
    Y = np.where(P > 0, S / np.maximum(P, 1e-12), 0.0)
    k0 = 0.003
    Wt = np.where(P > 0, P / (1 + P * k0), 0.0)
    A = Wt * Y
    num = A.T @ A
    den = Wt.T @ Wt
    M = num / np.maximum(den, 1e-12)
    dg = (Wt ** 2 * (Y ** 2 - np.where(P > 0, 1 / np.maximum(P, 1e-12), 0))).sum(0) \
        / np.maximum((Wt ** 2).sum(0), 1e-12)
    M[np.diag_indices_from(M)] = dg
    return M, den


def coplay_similarity(N, k=16):
    tot = N.sum(1)
    X = N[tot >= 20]
    X = np.log1p(X) / np.log1p(X).sum(1, keepdims=True)
    X = X - X.mean(0)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    V = (Vt[:k].T * s[:k])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return V @ V.T


def fit():
    d = C.load_slots()
    H = len(d["hero_names"])
    meta = C.hero_meta(d["hero_names"])
    e_mask, n_p, n_ph, table, r_adj = prepare(d)
    S, P, N, _, _ = C.static_state(d, e_mask, d["n_players"], H, r_adj)
    rng = np.random.RandomState(7)
    half = rng.rand(int(d["n_players"])) < 0.5
    active = P.sum(1) > 0
    fit_p, hold_p = half & active, ~half & active
    print(f"E slots {e_mask.sum():,}; players fit {fit_p.sum():,}, held-out {hold_p.sum():,}")
    bases = C.base_kernels(meta)

    # structured fit first (needed to define the co-strength residual)
    Sf, Pf = np.ascontiguousarray(S[fit_p]), np.ascontiguousarray(P[fit_p])
    Sh, Ph = np.ascontiguousarray(S[hold_p]), np.ascontiguousarray(P[hold_p])
    names3 = KERNELS["player+role+melee+hero"]
    th3, _ = C.fit_kernel(Sf, Pf, [bases[b] for b in names3])
    K3 = C.kernel_from([bases[b] for b in names3], th3)
    M, den = mom_covariance(Sf, Pf)
    Rm = M - K3
    Rm = (Rm + Rm.T) / 2
    ev, U = np.linalg.eigh(Rm)
    top = np.argsort(ev)[::-1][:10]
    lam = np.clip(ev[top], 0, None)
    Scs = (U[:, top] * lam) @ U[:, top].T
    Scs = Scs / np.mean(np.diag(Scs))
    bases["costr"] = Scs
    bases["coplay"] = coplay_similarity(N[fit_p])
    print(f"co-strength residual eigenvalues (top 10, pp^2): {np.round(1e4 * ev[top], 2)}")

    out = {"experience_table": table.tolist(), "kernels": {}}
    save = {"experience_table": table, "fit_players": fit_p,
            "mom_cov": M, "mom_den": den}
    for b, v in bases.items():
        save["basis_" + b] = v
    # learned low-rank CF factors (weight fixed at 1: theta entry 0)
    thetas = {}
    for r in (1, 2, 3, 4, 8):
        th, V, _ = C.fit_kernel_lowrank(Sf, Pf, [bases[b] for b in STRUCT], r, th3)
        bases[f"cf{r}"] = V @ V.T
        thetas[f"+CF rank {r}"] = np.r_[th, 0.0]
        save[f"V_cf{r}"] = V
    th, V, _ = C.fit_kernel_lowrank(Sf, Pf, [bases[b] for b in STRUCT + ["coplay"]], 2,
                                    np.r_[th3, np.log(1e-4)])
    bases["cf2cp"] = V @ V.T
    thetas["+CF rank 2 +co-play"] = np.r_[th, 0.0]
    save["V_cf2cp"] = V
    for b, v in bases.items():
        save["basis_" + b] = v
    base_ll = None
    for name, bl in KERNELS.items():
        print(name, flush=True)
        if name in thetas:
            th = thetas[name]
            llf = float(C._ml_terms(Sf, Pf, np.stack([bases[b] for b in bl]), th)[0].sum())
        else:
            th, llf = C.fit_kernel(Sf, Pf, [bases[b] for b in bl])
        llh, _ = C._ml_terms(Sh, Ph, np.stack([bases[b] for b in bl]), th)
        llh = float(llh.sum())
        if base_ll is None:
            base_ll = llh
        out["kernels"][name] = {
            "bases": bl, "log_var": th.tolist(),
            "sd_pp": {b: float(100 * np.sqrt(np.exp(t) * np.mean(np.diag(bases[b]))))
                      for b, t in zip(bl, th)},
            "fit_ll": llf, "heldout_ll": llh, "heldout_ll_vs_player": llh - base_ll}
        save["theta_" + name] = th
        print(f"  fit ll {llf:.1f}; held-out ll {llh:.1f} ({llh - base_ll:+.1f} vs player-only)",
              flush=True)
    # hero-level summaries of the co-strength and co-play similarity
    names = list(d["hero_names"])
    nn = {}
    for key in ("costr", "coplay", "cf2"):
        Sm = bases[key] + (1e-9 * np.eye(H))
        dg = np.sqrt(np.diag(Sm))
        Cr = Sm / np.outer(dg, dg)
        nn[key] = {names[i]: [names[j] for j in np.argsort(-Cr[i]) if j != i][:4]
                   for i in range(H)}
    out["nearest_heroes"] = nn
    np.savez(KOUT, **save)
    with open(os.path.join(C.RESULTS, "p3_hs_fit.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {KOUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["slots", "fit"])
    a = ap.parse_args()
    C.build_slots() if a.stage == "slots" else fit()


if __name__ == "__main__":
    main()
