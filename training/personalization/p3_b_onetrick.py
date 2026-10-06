"""
P3 review item 7 (REVIEW section 2.3): one-trick pooling.

Problem. The GP pools a one-trick's off-main heroes toward a player level
that the main dominates. The skill model predicts a 2.8pp drop when a
one-trick's main is banned; the natural experiment (p3_x_ban_nat.py) shows
8.1pp (7.4 to 8.8).

Candidate fixes, each fit by marginal likelihood on the E fit half (the
fit_players of cache/hs_kernels.npz, E games, lag-1 experience table), with
the six "+CF rank 2" log-variances refit jointly with the new parameters
(CF factors V fixed):

  A  cap: a hero's weight in the shared components is capped at CAP games.
     Implemented as tempering: for a query on hero h, every OTHER hero j of
     the player enters with S_j, P_j scaled by c_j = min(1, CAP / n_j), i.e.
     its cell-mean noise variance inflated by 1 / c_j. The queried hero's own
     cell keeps full weight, so the main's own estimate is unchanged while
     its pull on the player level (and through it on every off-main hero)
     is limited to what CAP games could carry. Variances are fit by ML on
     the tempered likelihood (all cells tempered), the natural objective
     when each cell enters with weight c_j. Chosen over a two-stage
     estimator because it stays one GP posterior (one solve per query, same
     code path, exact posterior variance under the tempered model).
  B  concentration kernel: per player K_p = K + g_p (a1 u u' + a2 diag(u)),
     u_j = [j is not the main], g_p = n_main / (n_total + 10) (shrunk main
     share). a1 is a shared off-main level, a2 extra off-main hero-specific
     variance. A one-trick's off-main heroes then pool with each other more
     and with the main less.
  C  concentration mean: prior mean of theta_pj = beta g_p u_j. B and A only
     change variances, so on their own they can move an off-main estimate
     toward 0 or toward the player's own off-main data, never below it. The
     raw evidence (one-tricks 6 to 7pp worse off-main beyond the offset) is
     a mean shift, so C fits it directly as a GP mean function (beta by ML,
     i.e. GLS, jointly with the variances).
  BC both.

Main and share are computed from the same lag-1 counts as the state (games
on days <= t - 1), so they are causal; in the E fit they come from the E
state itself, which matches how they are used online.

Evaluation (all variants in one causal online pass, state = rows with
day <= t - 1, frozen hyperparameters):
  1. headline game protocol: logistic combiner on logit(WP) + team
     difference of (offset + GP mean), fit on V1 games, scored on V2 and OOT
     (p3_x_oot windows; OOT excludes the sealed build 2.55.17.98025 unless
     --final); arms: skill only; + forced off-main (main share x
     main unavailable at the player's pick step, from cache/pgd_games.npz
     draft order, lag-1 main, 30+ earlier games); + concentration terms
     (main share x off-main); + both. Gains vs the WP-only combiner and,
     paired, vs the phase-1 kernel. Replay bootstrap (widen ~1.3x for
     recurring players).
  2. natural experiment (snapshot rows, flags from cache/x_ban_flags.npz as
     written by p3_x_ban_nat.py): within-player OLS with the same controls
     of (a) the model's s at the hero played (predicted drop), (b) s plus
     the forced term fit on V1 slots (r_self on s and the forced feature),
     (c) realized r_self net of the other nine players' s from the same
     variant. Strata one-trick / specialist / flexible / all. Second design
     (second review): the pre-ban information set only. Main and share from
     the lag-1 counts, treated = opponents banned the main at a draft step
     before the player's pick, control = all other eligible slots (no
     conditioning on later picks, later bans or the hero played). Predicted
     drops are also given for s plus the V1-fit concentration term (main
     share x off-main, the substitution cost learned from data) and its
     combiner coefficient is reported in the game table.
  3. player-variance term after the skill model (p3_x_map decomposition of
     e = r - s), base vs variants (--no-mapvar to skip).

Usage (from training/):
  python personalization/p3_b_onetrick.py [--sample 0.1] [--fit-sample 1.0]
      [--caps 30] [--nboot 200] [--no-mapvar] [--final]
Outputs: C.RESULTS/p3_b_onetrick[_s<frac>].json,
         C.CACHE/b7_onetrick_params[_s<frac>].npz, b7_onetrick_preds[_s<frac>].npz
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from numba import njit, prange
from p3_heroes import NUM_HEROES, HKEY
import p3_hs_core as C
import p3_x_common as X

K0_SHARE = 10.0
SEALED_BUILD = "2.55.17.98025"  # scored only with --final (second review)
LOGV_BOUNDS = (np.log(1e-9), np.log(1e-1))
CF_BOUNDS = (np.log(1e-3), np.log(1e2))
BASE_NAMES = ["player", "role", "fine", "melee", "hero", "cf2"]


# ------------------------------------------------------------------ GP with per-player kernel

@njit(parallel=True, cache=True)
def ml_ext(S, P, B, theta, g, main, la1, la2, beta, use_b, use_c):
    """Per-player log marginal likelihood and gradient wrt
    [theta (nb), la1, la2, beta] for K_p = sum exp(theta_k) B_k
    + g_p (e1 u u' + e2 diag u), prior mean beta g_p u, u_j = [j != main_p]."""
    n, H = S.shape
    nb = B.shape[0]
    ll = np.zeros(n)
    gr = np.zeros((n, nb + 3))
    ew = np.exp(theta)
    e1 = np.exp(la1) if use_b else 0.0
    e2 = np.exp(la2) if use_b else 0.0
    bt = beta if use_c else 0.0
    for i in prange(n):
        m = 0
        for j in range(H):
            if P[i, j] > 0:
                m += 1
        if m == 0:
            continue
        obs = np.empty(m, np.int64)
        c = 0
        for j in range(H):
            if P[i, j] > 0:
                obs[c] = j
                c += 1
        u = np.empty(m)
        for a in range(m):
            u[a] = 1.0 if obs[a] != main[i] else 0.0
        gi = g[i]
        A = np.zeros((m, m))
        for k in range(nb):
            for a in range(m):
                for b in range(m):
                    A[a, b] += ew[k] * B[k, obs[a], obs[b]]
        r = np.empty(m)
        for a in range(m):
            for b in range(m):
                A[a, b] += gi * e1 * u[a] * u[b]
            A[a, a] += gi * e2 * u[a] + 1.0 / P[i, obs[a]]
            r[a] = S[i, obs[a]] / P[i, obs[a]] - bt * gi * u[a]
        L = np.linalg.cholesky(A)
        Ainv = np.linalg.inv(A)
        alpha = Ainv @ r
        logdet = 0.0
        for a in range(m):
            logdet += 2.0 * np.log(L[a, a])
        ll[i] = -0.5 * (r @ alpha + logdet + m * np.log(2 * np.pi))
        for k in range(nb):
            s1 = 0.0
            s2 = 0.0
            for a in range(m):
                for b in range(m):
                    Bab = B[k, obs[a], obs[b]]
                    s1 += alpha[a] * Bab * alpha[b]
                    s2 += Ainv[b, a] * Bab
            gr[i, k] = 0.5 * ew[k] * (s1 - s2)
        ua = 0.0
        uAu = 0.0
        ua2 = 0.0
        uAd = 0.0
        for a in range(m):
            ua += u[a] * alpha[a]
            ua2 += u[a] * alpha[a] * alpha[a]
            uAd += u[a] * Ainv[a, a]
            for b in range(m):
                uAu += u[a] * u[b] * Ainv[a, b]
        if use_b:
            gr[i, nb] = 0.5 * e1 * gi * (ua * ua - uAu)
            gr[i, nb + 1] = 0.5 * e2 * gi * (ua2 - uAd)
        if use_c:
            gr[i, nb + 2] = gi * ua
    return ll, gr


@njit(parallel=True, cache=True)
def gp_query_ext(S, P, h, K, main, g, e1, e2, beta):
    """Posterior mean and variance at hero h[i] under the per-player kernel
    and mean of ml_ext (e1 = e2 = beta = 0 gives C.gp_query)."""
    n, H = S.shape
    mean = np.empty(n)
    var = np.empty(n)
    for i in prange(n):
        hh = h[i]
        gi = g[i]
        uh = 1.0 if hh != main[i] else 0.0
        m = 0
        for j in range(H):
            if P[i, j] > 0:
                m += 1
        if m == 0:
            mean[i] = beta * gi * uh
            var[i] = K[hh, hh] + gi * (e1 + e2) * uh
            continue
        obs = np.empty(m, np.int64)
        c = 0
        for j in range(H):
            if P[i, j] > 0:
                obs[c] = j
                c += 1
        u = np.empty(m)
        for a in range(m):
            u[a] = 1.0 if obs[a] != main[i] else 0.0
        A = np.empty((m, m))
        Bm = np.empty((m, 2))
        for a in range(m):
            for b in range(m):
                A[a, b] = K[obs[a], obs[b]] + gi * e1 * u[a] * u[b]
            A[a, a] += gi * e2 * u[a] + 1.0 / P[i, obs[a]]
            Bm[a, 0] = S[i, obs[a]] / P[i, obs[a]] - beta * gi * u[a]
            kv = K[hh, obs[a]] + gi * e1 * uh * u[a]
            if obs[a] == hh:
                kv += gi * e2 * uh
            Bm[a, 1] = kv
        Xs = np.linalg.solve(A, Bm)
        mu = beta * gi * uh
        q = 0.0
        for a in range(m):
            mu += Bm[a, 1] * Xs[a, 0]
            q += Bm[a, 1] * Xs[a, 1]
        mean[i] = mu
        var[i] = K[hh, hh] + gi * (e1 + e2) * uh - q
    return mean, var


def main_share(N):
    """Main hero (most games, lowest index on ties), raw share, shrunk share g."""
    tot = N.sum(1)
    main = np.where(tot > 0, np.argmax(N, axis=1), -1).astype(np.int64)
    top = N.max(1) if N.shape[0] else np.zeros(0)
    share = np.where(tot > 0, top / np.maximum(tot, 1), 0.0)
    g = top / (tot + K0_SHARE)
    return main, share, g, tot


def fit_variant(S, P, Bst, g, main, theta0, use_b, use_c, label):
    from scipy.optimize import minimize
    nb = Bst.shape[0]
    x0 = np.r_[np.asarray(theta0, float), np.log(1e-4), np.log(1e-4), 0.0]
    free = list(range(nb)) + ([nb, nb + 1] if use_b else []) + ([nb + 2] if use_c else [])
    t0 = time.time()
    nev = [0]

    def f(z):
        x = x0.copy()
        x[free] = z
        ll, gr = ml_ext(S, P, Bst, x[:nb], g, main, x[nb], x[nb + 1], x[nb + 2], use_b, use_c)
        nev[0] += 1
        return -ll.sum(), -gr.sum(axis=0)[free]

    # the CF basis (last) is V V' with its scale inside V (phase-1 weight exp(0) = 1)
    bounds = [LOGV_BOUNDS] * (nb - 1) + [CF_BOUNDS] + ([LOGV_BOUNDS] * 2 if use_b else []) \
        + ([(-0.5, 0.5)] if use_c else [])
    res = minimize(f, x0[free], jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
    x = x0.copy()
    x[free] = res.x
    print(f"  fit {label}: -ll {res.fun:.1f}, {nev[0]} evals, {time.time() - t0:.0f}s; sd(pp) "
          f"{np.round(100 * np.sqrt(np.exp(x[:nb])), 2)}"
          + (f", off-main level {100 * np.sqrt(np.exp(x[nb])):.2f}pp, off-main hero {100 * np.sqrt(np.exp(x[nb + 1])):.2f}pp"
             " (x sqrt g)" if use_b else "")
          + (f", beta {100 * x[nb + 2]:+.2f}pp per unit g" if use_c else ""), flush=True)
    return x, -res.fun


def selftest():
    """Small synthetic checks: extras off reproduce C.gp_query and C._ml_terms;
    ML gradient matches finite differences; query mean matches a dense solve."""
    rng = np.random.RandomState(1)
    H = NUM_HEROES
    n = 40
    Bs = [np.ones((H, H)), np.eye(H)]
    Vr = rng.randn(H, 2) * 0.5
    Bs.append(Vr @ Vr.T)
    Bst = np.ascontiguousarray(np.stack(Bs))
    th = np.log(np.array([3e-4, 1e-3, 2e-4]))
    Nn = (rng.rand(n, H) < 0.1) * rng.randint(1, 60, (n, H))
    P = Nn / 0.245
    S = P * (rng.randn(n, H) * 0.05)
    main, share, g, _ = main_share(Nn.astype(float))
    K = np.tensordot(np.exp(th), Bst, axes=1)
    h = rng.randint(0, H, n)
    m0, v0 = C.gp_query(S, P, h, K)
    m1, v1 = gp_query_ext(S, P, h, K, main, g, 0.0, 0.0, 0.0)
    assert np.allclose(m0, m1) and np.allclose(v0, v1), "extras-off query mismatch"
    l0, _ = C._ml_terms(S, P, Bst, th)
    l1, _ = ml_ext(S, P, Bst, th, g, main, -50.0, -50.0, 0.0, False, False)
    assert np.allclose(l0, l1), "extras-off ML mismatch"
    x = np.r_[th, np.log(5e-4), np.log(3e-4), -0.04]
    f = lambda z: ml_ext(S, P, Bst, z[:3], g, main, z[3], z[4], z[5], True, True)[0].sum()
    _, gr = ml_ext(S, P, Bst, x[:3], g, main, x[3], x[4], x[5], True, True)
    gsum = gr.sum(0)
    for k in range(6):
        e = np.zeros(6)
        e[k] = 1e-5
        fd = (f(x + e) - f(x - e)) / 2e-5
        assert abs(fd - gsum[k]) < 1e-3 * max(1.0, abs(fd)), (k, fd, gsum[k])
    # dense check of one query
    i = int(np.flatnonzero((P > 0).sum(1) >= 3)[0])
    obs = np.flatnonzero(P[i] > 0)
    u = (np.arange(H) != main[i]).astype(float)
    Kp = K + g[i] * (np.exp(x[3]) * np.outer(u, u) + np.exp(x[4]) * np.diag(u))
    mu = x[5] * g[i] * u
    A = Kp[np.ix_(obs, obs)] + np.diag(1 / P[i, obs])
    yb = S[i, obs] / P[i, obs]
    hq = np.array([h[i]])
    mq, vq = gp_query_ext(S[i:i + 1], P[i:i + 1], hq, K, main[i:i + 1], g[i:i + 1],
                          np.exp(x[3]), np.exp(x[4]), x[5])
    md = mu[h[i]] + Kp[h[i], obs] @ np.linalg.solve(A, yb - mu[obs])
    vd = Kp[h[i], h[i]] - Kp[h[i], obs] @ np.linalg.solve(A, Kp[obs, h[i]])
    assert np.isclose(mq[0], md) and np.isclose(vq[0], vd), (mq, md, vq, vd)
    print("selftest ok", flush=True)


# ------------------------------------------------------------------ data helpers

def compact_state(d, mask, r_adj):
    """Static per-player x hero S, P, N over rows in mask, for players with rows."""
    pids, inv = np.unique(d["pid"][mask], return_inverse=True)
    H = NUM_HEROES
    idx = inv * H + d["hero"][mask]
    size = len(pids) * H
    S = np.bincount(idx, weights=r_adj[mask] / d["v"][mask], minlength=size).reshape(-1, H)
    P = np.bincount(idx, weights=1.0 / d["v"][mask], minlength=size).reshape(-1, H)
    N = np.bincount(idx, minlength=size).reshape(-1, H).astype(np.float64)
    return pids, S, P, N


PGD_TYPES = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1])  # 0 ban, 1 pick (p3_pgd_data)


def draft_flags(d, rows, main):
    """Per row, from cache/pgd_games.npz (16-step standard drafts, snapshot and
    post), joined on (replay_id, hero); rows of games not in the table get
    covered = False and all flags False:
      unavailable    main banned or picked (anyone) at a step before this
                     row's own pick, and the row's hero is not the main
      opp_ban_before opponents banned the main at a step before this pick
      own_ban_before own team banned the main at a step before this pick"""
    z = np.load(os.path.join(C.CACHE, "pgd_games.npz"))
    rid = z["rid"]
    o = np.argsort(rid)
    rs = rid[o]
    n = len(rows)
    out = {k: np.zeros(n, bool) for k in ("unavailable", "opp_ban_before", "own_ban_before", "covered")}
    ban_step = PGD_TYPES == 0
    for a in range(0, n, 1_000_000):
        rr = rows[a:a + 1_000_000]
        j = np.minimum(np.searchsorted(rs, d["replay_id"][rr]), len(rs) - 1)
        ok = rs[j] == d["replay_id"][rr]
        Hg = z["H"][o[j]]
        Tg = z["T"][o[j]]
        hero = d["hero"][rr]
        team = d["team"][rr].astype(np.int64)
        mn = main[a:a + 1_000_000]
        own_hit = Hg == hero[:, None]
        ok &= own_hit.any(1) & (mn >= 0)
        own = np.argmax(own_hit, 1)
        mh = Hg == mn[:, None]
        mstep = np.where(mh.any(1), np.argmax(mh, 1), 99)
        before = ok & (mstep < own)
        is_ban = ban_step[np.minimum(mstep, 15)]
        mteam = Tg[np.arange(len(rr)), np.minimum(mstep, 15)]
        sl = slice(a, a + len(rr))
        out["covered"][sl] = ok
        out["unavailable"][sl] = before & (hero != mn)
        out["opp_ban_before"][sl] = before & is_ban & (mteam == 1 - team)
        out["own_ban_before"][sl] = before & is_ban & (mteam == team)
    return out


def ban_controls(d_snap):
    """Main ban rate per build (x_bans) for the natural experiment, as in
    p3_x_ban_nat.py, aligned to snapshot rows."""
    F = np.load(os.path.join(C.CACHE, "x_ban_slotfeat.npz"))
    main = F["main"]
    g = d_snap["g"]
    n_games = int(g.max()) + 1
    grid = np.zeros(n_games, np.int64)
    grid[g] = d_snap["replay_id"]
    B = np.load(os.path.join(C.CACHE, "x_bans.npz"))
    o = np.argsort(B["replay_ids"])
    j = np.searchsorted(B["replay_ids"][o], grid)
    jm = np.minimum(j, len(o) - 1)
    okg = B["replay_ids"][o][jm] == grid
    bh = np.where(okg[:, None], B["hero"][o][jm], -1)
    w = np.load(C.WP)
    bidx = w["build_idx"][np.argsort(w["replay_ids"])]
    bg = bidx[np.searchsorted(np.sort(w["replay_ids"]), grid)]
    nbb = bg.max() + 1
    banned_any = np.zeros((nbb, NUM_HEROES))
    gcount = np.bincount(bg, minlength=nbb).astype(float)
    for k in range(6):
        m = bh[:, k] >= 0
        np.add.at(banned_any, (bg[m], bh[m, k]), 1.0)
    ban_rate = banned_any / np.maximum(gcount[:, None], 1)
    return ban_rate[bg[g], main]


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=float, default=1.0,
                    help="fraction of games queried outside V1/V2/OOT (natural experiment, map variance)")
    ap.add_argument("--eval-sample", type=float, default=1.0, help="fraction of V1/V2/OOT games queried (debug)")
    ap.add_argument("--fit-sample", type=float, default=1.0, help="fraction of fit-half players used in ML fits")
    ap.add_argument("--caps", default="30")
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--no-mapvar", action="store_true")
    ap.add_argument("--final", action="store_true",
                    help="also score the sealed OOT build 2.55.17.98025 (one final run only)")
    a = ap.parse_args()
    t0 = time.time()
    tag = "" if (a.sample == 1.0 and a.eval_sample == 1.0 and a.fit_sample == 1.0) else \
        f"_s{a.sample:g}_e{a.eval_sample:g}_f{a.fit_sample:g}"
    tag += "_final" if a.final else ""
    selftest()
    H = NUM_HEROES
    d = X.load_ext()
    n = len(d["pid"])
    post = d["post"]
    # snapshot rows are the prefix-in-order of hs_slots
    snap = np.load(C.SLOTS)
    sidx = np.flatnonzero(~post)
    assert np.array_equal(d["replay_id"][sidx], snap["replay_id"]) and np.array_equal(d["hero"][sidx], snap["hero"])
    del snap
    days = d["day"]
    e_mask = d["in_sample"] & (days >= C.day_of(C.E_START))
    n_p, n_ph = C.experience_counts(d)
    table = C.fit_experience(d["r"], n_p, n_ph, e_mask)
    r_adj = d["r"] - table[C.exp_bins(n_p, n_ph)]
    mu_exp = table[C.exp_bins(n_p, n_ph)]
    print(f"rows {n:,}, players {int(d['n_players']):,}, E rows {e_mask.sum():,} ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- kernels: fit on the E fit half
    kz = np.load(os.path.join(C.CACHE, "hs_kernels.npz"))
    Bst = np.ascontiguousarray(np.stack([kz["basis_" + b] for b in BASE_NAMES]).astype(np.float64))
    th0 = np.asarray(kz["theta_+CF rank 2"], float)
    fitp = np.zeros(int(d["n_players"]), bool)
    fitp[:len(kz["fit_players"])] = kz["fit_players"]
    pids, Se, Pe, Ne = compact_state(d, e_mask, r_adj)
    mE, shE, gE, _ = main_share(Ne)
    rng = np.random.RandomState(7)
    fsel = fitp[pids] & (rng.rand(len(pids)) < a.fit_sample)
    hsel = ~fitp[pids]
    if a.fit_sample < 1.0:
        hsel &= rng.rand(len(pids)) < a.fit_sample
    sub = lambda M, s: np.ascontiguousarray(M[s])
    Sf, Pf, Sh, Ph = sub(Se, fsel), sub(Pe, fsel), sub(Se, hsel), sub(Pe, hsel)
    print(f"E players: fit {fsel.sum():,}, held-out {hsel.sum():,}", flush=True)
    out = {"args": vars(a), "experience_table_rows": int(e_mask.sum()), "fits": {}}

    def ll_of(Sx, Px, sel, x, ub, uc):
        return float(ml_ext(Sx, Px, Bst, x[:6], gE[sel], mE[sel], x[6], x[7], x[8], ub, uc)[0].sum())

    variants = {}  # name -> dict(kind, x, K, cap)
    x_base = np.r_[th0, -50.0, -50.0, 0.0]
    variants["phase-1 kernel"] = {"kind": "plain", "x": x_base}
    for nm, ub, uc in (("refit (no change)", False, False), ("B concentration kernel", True, False),
                       ("C concentration mean", False, True), ("BC kernel + mean", True, True)):
        x, llf = fit_variant(Sf, Pf, Bst, gE[fsel], mE[fsel], th0, ub, uc, nm)
        if not ub:
            x[6:8] = -50.0
        variants[nm] = {"kind": "ext" if (ub or uc) else "plain", "x": x}
    caps = [float(c) for c in a.caps.split(",") if c]
    for cap in caps:
        cf = np.minimum(1.0, cap / np.maximum(Ne, 1))
        Sfc, Pfc = sub(Se * cf, fsel), sub(Pe * cf, fsel)
        x, llf = fit_variant(Sfc, Pfc, Bst, gE[fsel], mE[fsel], th0, False, False, f"A cap {cap:g}")
        x[6:8] = -50.0
        variants[f"A cap {cap:g}"] = {"kind": "cap", "x": x, "cap": cap}
    for nm, v in variants.items():
        x = v["x"]
        ub = v["kind"] == "ext" and x[6] > -40
        uc = v["kind"] == "ext"
        if v["kind"] == "cap":
            cf = np.minimum(1.0, v["cap"] / np.maximum(Ne, 1))
            Sx, Px = (Se * cf), (Pe * cf)
        else:
            Sx, Px = Se, Pe
        res = {"log_var": x[:6].tolist(),
               "sd_pp": {b: float(100 * np.sqrt(np.exp(t) * np.mean(np.diag(Bst[k]))))
                         for k, (b, t) in enumerate(zip(BASE_NAMES, x[:6]))},
               "fit_ll": ll_of(sub(Sx, fsel), sub(Px, fsel), fsel, x, ub, uc),
               "heldout_ll": ll_of(sub(Sx, hsel), sub(Px, hsel), hsel, x, ub, uc)}
        if ub:
            res["offmain_level_sd_pp_at_g1"] = float(100 * np.sqrt(np.exp(x[6])))
            res["offmain_hero_sd_pp_at_g1"] = float(100 * np.sqrt(np.exp(x[7])))
        if uc:
            res["beta_pp_per_unit_g"] = float(100 * x[8])
        if v["kind"] == "cap":
            res["note"] = "likelihood of tempered cell means; not comparable with the untempered rows"
        out["fits"][nm] = res
        v["K"] = np.tensordot(np.exp(x[:6]), Bst, axes=1)
    base_h = out["fits"]["refit (no change)"]["heldout_ll"]
    for nm in out["fits"]:
        out["fits"][nm]["heldout_ll_vs_refit"] = out["fits"][nm]["heldout_ll"] - base_h
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("fit_ll", "heldout_ll_vs_refit", "beta_pp_per_unit_g",
                                                                   "offmain_level_sd_pp_at_g1", "offmain_hero_sd_pp_at_g1")}
                      for k, v in out["fits"].items()}, indent=1), flush=True)
    del Se, Pe, Sf, Pf, Sh, Ph
    np.savez(os.path.join(C.CACHE, f"b7_onetrick_params{tag}.npz"),
             **{"x_" + k: v["x"] for k, v in variants.items()}, base_names=np.array(BASE_NAMES))

    # ---------------- query rows
    snap_post = (~post) & (~d["in_sample"])
    _, fr = np.unique(d["g"][snap_post], return_index=True)
    med = np.median(days[snap_post][fr])
    v1, v2 = snap_post & (days < med), snap_post & (days >= med)
    oot = post & (days > X.SNAP_LAST_DAY)
    vers = [str(v) for v in d["versions"]]
    sealed = post & np.isin(d["version"], [i for i, v in enumerate(vers) if v == SEALED_BUILD])
    if not a.final:
        oot &= ~sealed
    out["oot_scope"] = "all OOT builds (final run)" if a.final else f"OOT without the sealed build {SEALED_BUILD}"
    grng = np.random.RandomState(11)
    n_games = int(d["g"].max()) + 1
    gu = grng.rand(n_games)
    evalq = (v1 | v2 | oot) & (gu[d["g"]] < a.eval_sample)
    natq = (~post) & (days >= C.day_of("2024-04-01")) & (gu[d["g"]] < a.sample)
    query = evalq | natq
    print(f"query rows {query.sum():,} (eval {evalq.sum():,}) ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- online pass (lag 1 day, all rows feed the state)
    names = list(variants)
    npl = int(d["n_players"])
    Sst = np.zeros((npl, H))
    Pst = np.zeros((npl, H))
    Nst = np.zeros((npl, H), np.float32)
    qidx = np.flatnonzero(query)
    nq = len(qidx)
    pred = {nm: np.full(nq, np.nan, np.float32) for nm in names}
    var_base = np.empty(nq, np.float32)
    q_main = np.empty(nq, np.int64)
    q_share = np.empty(nq, np.float32)
    q_tot = np.empty(nq, np.float32)
    add_day = days
    ap_ = 0
    ud, qs = np.unique(days[qidx], return_index=True)
    qe = np.r_[qs[1:], nq]
    tl = time.time()
    for it, (t, s0, s1) in enumerate(zip(ud, qs, qe)):
        e = np.searchsorted(add_day, t - 1, side="right")
        if e > ap_:
            rr = np.arange(ap_, e)
            pi, hi = d["pid"][rr], d["hero"][rr]
            np.add.at(Sst, (pi, hi), r_adj[rr] / d["v"][rr])
            np.add.at(Pst, (pi, hi), 1.0 / d["v"][rr])
            np.add.at(Nst, (pi, hi), 1.0)
            ap_ = e
        q = qidx[s0:s1]
        pq, hq = d["pid"][q], d["hero"][q]
        Sq = np.ascontiguousarray(Sst[pq])
        Pq = np.ascontiguousarray(Pst[pq])
        Nq = Nst[pq].astype(np.float64)
        mq, shq, gq, totq = main_share(Nq)
        q_main[s0:s1], q_share[s0:s1], q_tot[s0:s1] = mq, shq, totq
        for nm in names:
            v = variants[nm]
            x = v["x"]
            if v["kind"] == "plain":
                m_, vv = C.gp_query(Sq, Pq, hq, v["K"])
                if nm == "phase-1 kernel":
                    var_base[s0:s1] = vv
            elif v["kind"] == "cap":
                cf = np.minimum(1.0, v["cap"] / np.maximum(Nq, 1))
                cf[np.arange(len(q)), hq] = 1.0
                m_, _ = C.gp_query(np.ascontiguousarray(Sq * cf), np.ascontiguousarray(Pq * cf), hq, v["K"])
            else:
                e1 = np.exp(x[6]) if x[6] > -40 else 0.0
                e2 = np.exp(x[7]) if x[7] > -40 else 0.0
                m_, _ = gp_query_ext(Sq, Pq, hq, v["K"], mq, gq, e1, e2, x[8])
            pred[nm][s0:s1] = m_
        if it % 100 == 0:
            print(f"  day {it}/{len(ud)} ({time.time() - tl:.0f}s)", flush=True)
    del Sst, Pst
    print(f"online pass done ({time.time() - t0:.0f}s)", flush=True)
    # per-row features
    dfl = draft_flags(d, qidx, q_main)
    unav, cov = dfl["unavailable"], dfl["covered"]
    hero_q = d["hero"][qidx]
    offmain = (hero_q != q_main) & (q_main >= 0)
    forced = np.where(q_tot >= 30, q_share, 0.0) * unav
    conc = q_share * offmain
    s_rows = {nm: (mu_exp[qidx] + pred[nm]).astype(np.float64) for nm in names}
    np.savez(os.path.join(C.CACHE, f"b7_onetrick_preds{tag}.npz"), qidx=qidx, var_base=var_base, main=q_main,
             share=q_share, tot=q_tot, mu=mu_exp[qidx].astype(np.float32), **dfl,
             **{"m_" + str(i): pred[nm] for i, nm in enumerate(names)}, names=np.array(names))
    out["forced_feature"] = {"draft_coverage_eval_rows": float(cov[evalq[qidx]].mean()),
                             "share_rows_main_unavailable_offmain": float(unav[evalq[qidx]].mean())}

    # ---------------- 1. headline game protocol
    n_games, y, wp0, cnt, gday = X.game_arrays(d)
    lo = np.log(wp0 / (1 - wp0))
    qcount = np.bincount(d["g"][qidx], minlength=n_games)
    gv = {}
    for nmw, mw in (("V1", v1), ("V2", v2), ("OOT", oot)):
        gm = np.zeros(n_games, bool)
        gm[d["g"][mw]] = True
        gv[nmw] = gm & (cnt == 10) & (qcount == 10)
    out["windows_games"] = {k: int(v.sum()) for k, v in gv.items()}
    g_q, t_q = d["g"][qidx], d["team"][qidx]
    Dfor = X.team_diff(forced, g_q, t_q, n_games)
    Dcon = X.team_diff(conc, g_q, t_q, n_games)
    w0 = C.fit_logistic(lo[gv["V1"]][:, None], y[gv["V1"]])
    ll0 = X.logloss(C.predict(w0, lo[:, None]), y)
    arms = {"skill": [], "skill + forced": [Dfor], "skill + concentration": [Dcon],
            "skill + concentration + forced": [Dcon, Dfor]}
    game = {}
    lls = {}
    for nm in names:
        Ds = X.team_diff(s_rows[nm], g_q, t_q, n_games)
        for an, extra in arms.items():
            Xm = np.column_stack([lo, Ds] + extra)
            w = C.fit_logistic(Xm[gv["V1"]], y[gv["V1"]])
            p = C.predict(w, Xm)
            l1 = X.logloss(p, y)
            lls[(nm, an)] = l1
            r = {"coef": w.tolist()}
            if "concentration" in an:
                # main share x off-main learned by the combiner (logit per unit; x25 = pp near 0.5)
                r["concentration_coef_logit"] = float(w[3])
                r["concentration_approx_pp_per_unit_share"] = float(25 * w[3])
            for wn in ("V2", "OOT"):
                idx = np.flatnonzero(gv[wn])
                gain, ci = X.boot_gain(ll0, l1, idx, n=a.nboot)
                mt = C.game_metrics(p[gv[wn]], y[gv[wn]])
                r[wn] = {"gain": gain, "ci": ci, "acc": mt["acc"], "cal_slope": mt["cal_slope"]}
                for refn, lab in (("phase-1 kernel", "vs_phase1_same_arm"), ("refit (no change)", "vs_refit_same_arm")):
                    if nm not in ("phase-1 kernel", refn):
                        dg, dci = X.boot_gain(lls[(refn, an)], l1, idx, n=a.nboot)
                        r[wn][lab] = {"gain": dg, "ci": dci}
                if an != "skill":
                    dg, dci = X.boot_gain(lls[(nm, "skill")], l1, idx, n=a.nboot)
                    r[wn]["vs_skill_only_same_kernel"] = {"gain": dg, "ci": dci}
            game[f"{nm} | {an}"] = r
            print(f"  {nm:24s} {an:32s} V2 {r['V2']['gain']:+.5f} [{r['V2']['ci'][0]:+.5f},{r['V2']['ci'][1]:+.5f}]"
                  f"  OOT {r['OOT']['gain']:+.5f} [{r['OOT']['ci'][0]:+.5f},{r['OOT']['ci'][1]:+.5f}]", flush=True)
    out["game"] = game
    print(f"game level done ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- 2. natural experiment (snapshot rows)
    import p3_x_ban_nat as BN
    fl = np.load(os.path.join(C.CACHE, "x_ban_flags.npz"))
    sq = ~post[qidx]
    rows_s = qidx[sq]                    # ext row index of the snapshot query rows
    snap_pos = np.cumsum(~post) - 1      # ext row -> hs_slots row (snapshot rows keep their order)
    rows_h = snap_pos[rows_s]
    assert len(fl["treat"]) == len(sidx)
    ds = {k: d[k][rows_s] for k in ("g", "team", "pid", "r", "day", "in_sample", "replay_id")}
    # whole games only (opp_S needs all ten slots)
    gc = np.bincount(ds["g"], minlength=int(ds["g"].max()) + 1)
    full = gc[ds["g"]] == 10
    br_all = ban_controls({"g": d["g"][sidx], "replay_id": d["replay_id"][sidx]})
    br = br_all[rows_h]
    treat, ctrl, share = fl["treat"][rows_h] & full, fl["ctrl"][rows_h] & full, fl["share"][rows_h]
    strata = {"one-trick (main share >= 50%)": share >= 0.5, "specialist (25-50%)": (share >= 0.25) & (share < 0.5),
              "flexible (< 25%)": share < 0.25, "all": np.ones(len(rows_s), bool)}
    gs, ts = ds["g"], ds["team"].astype(int)
    v1s = (~ds["in_sample"]) & (ds["day"] < med)
    forced_s, conc_s = forced[sq], conc[sq]
    # second definition: pre-ban information set only. Main and share from
    # earlier-day counts (the state's own lag-1 counts), treatment = the
    # opponents banned the main at a draft step before this player's pick,
    # control = every other eligible slot (no conditioning on later picks or
    # bans, or on what the player picked); own-team bans of the main before
    # the pick are dropped from both.
    sh2 = q_share[sq]
    elig2 = dfl["covered"][sq] & (q_tot[sq] >= 30) & (ds["day"] >= C.day_of("2024-07-01")) & full
    treat2 = elig2 & dfl["opp_ban_before"][sq] & ~dfl["own_ban_before"][sq]
    ctrl2 = elig2 & ~dfl["opp_ban_before"][sq] & ~dfl["own_ban_before"][sq]
    strata2 = {"one-trick (main share >= 50%)": sh2 >= 0.5, "specialist (25-50%)": (sh2 >= 0.25) & (sh2 < 0.5),
               "flexible (< 25%)": sh2 < 0.25}
    designs = {"p3_x_ban_nat definition": (treat, ctrl, strata),
               "pre-ban information set": (treat2, ctrl2, strata2)}
    nat = {"treated": int(treat.sum()), "control": int(ctrl.sum()),
           "treated_pre_ban_definition": int(treat2.sum()), "control_pre_ban_definition": int(ctrl2.sum())}
    nrng = np.random.RandomState(0)
    for nm in names:
        s = s_rows[nm][sq]
        Sg = np.zeros((int(gs.max()) + 1, 2))
        np.add.at(Sg, (gs, ts), s)
        r_self = ds["r"] - (Sg[gs, ts] - s) + Sg[gs, 1 - ts]
        opp_S = Sg[gs, 1 - ts]

        def v1fit(cols):
            A_ = np.column_stack([np.ones(v1s.sum()), s[v1s]] + [c[v1s] for c in cols])
            return np.linalg.lstsq(A_, r_self[v1s], rcond=None)[0]
        bf, bc, bcf = v1fit([forced_s]), v1fit([conc_s]), v1fit([conc_s, forced_s])
        outs = {"predicted: s on hero played": s,
                "predicted: s + forced term": s + bf[2] * forced_s,
                "predicted: s + concentration term": s + bc[2] * conc_s,
                "predicted: s + concentration + forced": s + bcf[2] * conc_s + bcf[3] * forced_s,
                "realized: r_self": r_self}
        res = {"V1_slot_fits_r_self (pp per unit share)": {
            "s + forced": {"slope_s": float(bf[1]), "forced": float(100 * bf[2])},
            "s + concentration": {"slope_s": float(bc[1]), "concentration": float(100 * bc[2])},
            "s + concentration + forced": {"slope_s": float(bcf[1]), "concentration": float(100 * bcf[2]),
                                           "forced": float(100 * bcf[3])}}}
        for dn, (tr_, ct_, st_) in designs.items():
            res[dn] = {}
            for sn, sm in st_.items():
                tm, cm = tr_ & sm, ct_ & sm
                sel = tm | cm
                if tm.sum() < 100:
                    continue
                rr = {"treated": int(tm.sum()), "control": int(cm.sum())}
                for on, ov in outs.items():
                    if sn == "all" and on not in ("predicted: s on hero played", "realized: r_self"):
                        continue
                    est, ci = BN.fe_ols(ov[sel], tm[sel], [br[sel], opp_S[sel]], ds["pid"][sel], nrng,
                                        nboot=200 if sn.startswith("one-trick") else 100)
                    rr[on] = {"drop_pp": -100 * est, "ci_pp": [-100 * ci[1], -100 * ci[0]]}
                rl = rr["realized: r_self"]["drop_pp"]
                for on in outs:
                    if on.startswith("predicted") and on in rr:
                        rr[on]["share_of_realized"] = rr[on]["drop_pp"] / rl if rl else float("nan")
                res[dn][sn] = rr
        nat[nm] = res
        r1 = res["p3_x_ban_nat definition"]
        print(f"  nat {nm:24s} " + "  ".join(
            f"{sn.split(' ')[0]}: s {r1[sn]['predicted: s on hero played']['drop_pp']:.2f} "
            f"+conc {r1[sn]['predicted: s + concentration term']['drop_pp']:.2f} "
            f"+forced {r1[sn]['predicted: s + forced term']['drop_pp']:.2f} "
            f"real {r1[sn]['realized: r_self']['drop_pp']:.2f}"
            for sn in list(strata)[:3] if sn in r1), flush=True)
    out["natural_experiment"] = nat
    print(f"natural experiment done ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- 3. player-variance term after the skill model
    if not a.no_mapvar:
        import p3_x_map as M
        import p3_x_side as XS
        sg = np.load(os.path.join(C.CACHE, "x_side_games.npz"))
        dd = {k: d[k][rows_s] for k in ("pid", "hero", "replay_id", "day")}
        dd["n_players"] = d["n_players"]
        gi = XS.game_lookup(dd, sg["replay_ids"])
        ok = gi >= 0
        mapi = np.where(ok, sg["map"][np.maximum(gi, 0)], 63).astype(np.int64)
        dk = {k: v[ok] for k, v in dd.items() if k != "n_players"}
        dk["n_players"] = d["n_players"]
        mv = {}
        for nm in names:
            e_ = (ds["r"] - s_rows[nm][sq])[ok]
            r0 = M.decompose(dk, e_, mapi[ok], np.random.RandomState(0), f"e after {nm}")
            mv[nm] = {k: r0[k] for k in ("s2_p", "s2_ph", "s2_pm", "s2_phm")}
        out["variance_after_skill_model_pp2"] = mv
    out["elapsed_s"] = time.time() - t0
    fn = os.path.join(C.RESULTS, f"p3_b_onetrick{tag}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"wrote {fn} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
