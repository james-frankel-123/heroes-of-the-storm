"""
P3 hero strength: shared core (data, hero kernels, Gaussian-process posterior,
marginal-likelihood fitting, causal online sufficient statistics).

Model. For player p and hero h, strength theta_ph is the expected residual
r = y_team - WP_drift(team) when p plays h (units: win-probability points).
Each game gives r_i with noise variance v_i = wp_i (1 - wp_i). An experience
offset mu_exp(games seen for p, games seen for p on h) is subtracted first
(new accounts and first games on a hero lose more than the WP expects).
Across the 90 heroes of one player, theta_p. ~ N(0, K) with

    K = t_player J + t_role B_role + t_fine B_fine + t_melee B_melee
        + t_sim S + t_hero I

J all-ones (overall skill), B_* same-group indicators, S a hero similarity
matrix (co-strength or co-play), I independent hero-specific skill. Special
cases give the compared methods: player-only (J), player+hero (J, I),
player+role+hero (J, B_*, I), player+similar-hero (all). The posterior for
every hero of a player, including heroes never played, is the GP posterior
given the per-hero sufficient statistics S_h = sum r_i / v_i and
P_h = sum 1 / v_i. Hyperparameters t_* are fit by maximizing the marginal
likelihood.

All statistics are causal: the online state for a game on day t only
contains games from days <= t - lag.
"""
import os
import sys
import datetime

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from p3_heroes import HKEY, check_heroes
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results")
WP = os.path.join(CACHE, "wp_drift.npz")
PLAYERS = os.path.join(CACHE, "players_2024q2.npz")
HERO_MMR = os.path.join(CACHE, "hero_mmr_2024q2.npz")
SLOTS = os.path.join(CACHE, "hs_slots.npz")
E_START = "2024-04-01"
EPOCH = datetime.date(1970, 1, 1)

BLIZZ_ROLES = ["Tank", "Bruiser", "Healer", "Ranged Assassin", "Melee Assassin", "Support"]
# Heroes whose basic attack is melee range (hand-labeled side feature).
MELEE = {
    "Anub'arak", "Arthas", "Cho", "Diablo", "E.T.C.", "Garrosh", "Johanna",
    "Mal'Ganis", "Mei", "Muradin", "Stitches", "Tyrael", "Artanis", "Chen",
    "Deathwing", "Dehaka", "Gazlowe", "Hogger", "Imperius", "Leoric",
    "Malthael", "Ragnaros", "Sonya", "Thrall", "Xul", "Yrel", "Varian",
    "Kharazim", "Rehgar", "Uther", "Alarak", "Illidan", "Kerrigan", "Maiev",
    "Qhira", "Samuro", "The Butcher", "Valeera", "Zeratul", "Murky",
    "Blaze",
}

# experience bins (games seen in window before this game)
NP_EDGES = np.array([0, 1, 5, 10, 20, 50, 100, 300], np.int64)
NPH_EDGES = np.array([0, 1, 2, 3, 5, 10, 20, 50], np.int64)


def day_of(s):
    return (datetime.date.fromisoformat(s) - EPOCH).days


# ------------------------------------------------------------------ heroes

def hero_meta(hero_names):
    from shared import HERO_ROLE_FINE, FINE_ROLE_NAMES, FINE_TO_BLIZZ_ROLE
    fine = np.array([FINE_ROLE_NAMES.index(HERO_ROLE_FINE[h]) for h in hero_names])
    blizz = np.array([BLIZZ_ROLES.index(FINE_TO_BLIZZ_ROLE[HERO_ROLE_FINE[h]])
                      for h in hero_names])
    # Varian: Blizzard lists him as Bruiser; the fine map keeps him alone.
    melee = np.array([h in MELEE for h in hero_names])
    return {"fine": fine, "blizz": blizz, "melee": melee,
            "fine_names": FINE_ROLE_NAMES, "names": np.asarray(hero_names)}


def base_kernels(meta):
    H = len(meta["fine"])
    same = lambda a: (a[:, None] == a[None, :]).astype(np.float64)
    return {"player": np.ones((H, H)), "role": same(meta["blizz"]),
            "fine": same(meta["fine"]), "melee": same(meta["melee"].astype(int)),
            "hero": np.eye(H)}


# ------------------------------------------------------------------ data

def build_slots():
    """One row per player-game with residuals, ids, day, and MMR columns.
    Rows are sorted by (day, replay_id). Player key = (region, blizz_id)."""
    w = np.load(WP)
    p = np.load(PLAYERS)
    m = np.load(HERO_MMR)
    g_rid = w["replay_ids"]
    order = np.argsort(g_rid)
    g_rid = g_rid[order]
    gi = np.searchsorted(g_rid, p["replay_ids"])
    ok = (gi < len(g_rid)) & (g_rid[np.minimum(gi, len(g_rid) - 1)] == p["replay_ids"])
    print(f"player rows without a scored game (dropped): {(~ok).sum()}")
    # align hero_mmr rows to player rows on (replay_id, blizz_id)
    ka = np.lexsort((p["blizz_ids"], p["replay_ids"]))
    kb = np.lexsort((m["blizz_ids"], m["replay_ids"]))
    assert np.array_equal(p["replay_ids"][ka], m["replay_ids"][kb])
    assert np.array_equal(p["blizz_ids"][ka], m["blizz_ids"][kb])
    inv_a = np.empty_like(ka)
    inv_a[ka] = np.arange(len(ka))
    take = kb[inv_a]  # m row for each p row
    p = {k: p[k] for k in p.files}
    take, gi = take[ok], gi[ok]
    for k in ("replay_ids", "blizz_ids", "hero", "team", "party", "mmr"):
        p[k] = p[k][ok]
    y = w["y"][order][gi]
    wp = w["wp0"][order][gi]
    team = p["team"].astype(np.int8)
    y_team = np.where(team == 0, y, 1 - y).astype(np.float32)
    wp_team = np.where(team == 0, wp, 1 - wp).astype(np.float64)
    day = w["date_days"][order][gi].astype(np.int32)
    rid = p["replay_ids"]
    region = m["region"][take].astype(np.int64)
    key = region * (1 << 40) + p["blizz_ids"]
    pid_u, pid = np.unique(key, return_inverse=True)
    srt = np.lexsort((rid, day))
    out = dict(
        replay_id=rid, g=gi, pid=pid.astype(np.int64), hero=p["hero"].astype(np.int64),
        team=team, party=p["party"], y=y_team, wp=wp_team, day=day,
        in_sample=w["in_sample"][order][gi], player_mmr=p["mmr"],
        hero_mmr=m["hero_mmr"][take], role_mmr=m["role_mmr"][take],
        hero_level=m["hero_level"][take])
    out = {k: v[srt] for k, v in out.items()}
    out["n_players"] = np.int64(len(pid_u))
    out["hero_names"] = p["hero_names"]
    out["player_keys"] = pid_u
    # game index for the sorted order
    np.savez(SLOTS, **out)
    print(f"wrote {SLOTS}: {len(srt):,} slots, {len(pid_u):,} players "
          f"(blizz_ids {len(np.unique(p['blizz_ids'])):,})")


def load_slots():
    z = np.load(SLOTS, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    check_heroes(d["hero_names"], d["hero"])
    d["r"] = d["y"] - d["wp"]
    d["v"] = d["wp"] * (1 - d["wp"])
    return d


def experience_counts(d):
    """Games seen in the window before each row (exact order), overall and
    on this hero."""
    n = len(d["pid"])
    o = np.lexsort((d["replay_id"], d["day"], d["pid"]))
    pid = d["pid"][o]
    first = np.r_[True, pid[1:] != pid[:-1]]
    starts = np.flatnonzero(first)
    lens = np.diff(np.r_[starts, n])
    k = np.arange(n) - np.repeat(starts, lens)
    n_p = np.empty(n, np.int64)
    n_p[o] = k
    key = d["pid"] * HKEY + d["hero"]
    o2 = np.lexsort((d["replay_id"], d["day"], key))
    kk = key[o2]
    first = np.r_[True, kk[1:] != kk[:-1]]
    starts = np.flatnonzero(first)
    lens = np.diff(np.r_[starts, n])
    k = np.arange(n) - np.repeat(starts, lens)
    n_ph = np.empty(n, np.int64)
    n_ph[o2] = k
    return n_p, n_ph


def exp_bins(n_p, n_ph):
    a = np.searchsorted(NP_EDGES, n_p, side="right") - 1
    b = np.searchsorted(NPH_EDGES, n_ph, side="right") - 1
    return a * len(NPH_EDGES) + b


def fit_experience(r, n_p, n_ph, mask, prior=200.0):
    """Mean residual by (games seen, games seen on hero) bin, shrunk to 0."""
    b = exp_bins(n_p, n_ph)
    nb = len(NP_EDGES) * len(NPH_EDGES)
    s = np.bincount(b[mask], weights=r[mask], minlength=nb)
    c = np.bincount(b[mask], minlength=nb)
    return s / (c + prior)


# ------------------------------------------------------------------ GP

@njit(cache=True)
def _post_one(Si, Pi, K, h):
    H = Si.shape[0]
    m = 0
    for j in range(H):
        if Pi[j] > 0:
            m += 1
    if m == 0:
        return 0.0, K[h, h]
    obs = np.empty(m, np.int64)
    c = 0
    for j in range(H):
        if Pi[j] > 0:
            obs[c] = j
            c += 1
    A = np.empty((m, m))
    B = np.empty((m, 2))
    for a in range(m):
        for b in range(m):
            A[a, b] = K[obs[a], obs[b]]
        A[a, a] += 1.0 / Pi[obs[a]]
        B[a, 0] = Si[obs[a]] / Pi[obs[a]]
        B[a, 1] = K[h, obs[a]]
    X = np.linalg.solve(A, B)
    mean = 0.0
    q = 0.0
    for a in range(m):
        mean += K[h, obs[a]] * X[a, 0]
        q += K[h, obs[a]] * X[a, 1]
    return mean, K[h, h] - q


@njit(parallel=True, cache=True)
def gp_query(S, P, h, K):
    """Posterior mean and variance of theta at hero h[i] given the per-hero
    sufficient statistics S[i], P[i] (one row per query)."""
    n = S.shape[0]
    mean = np.empty(n)
    var = np.empty(n)
    for i in prange(n):
        mean[i], var[i] = _post_one(S[i], P[i], K, h[i])
    return mean, var


@njit(parallel=True, cache=True)
def gp_all(S, P, K):
    """Posterior mean and variance for all heroes, one row per player."""
    n, H = S.shape
    mean = np.empty((n, H))
    var = np.empty((n, H))
    for i in prange(n):
        m = 0
        for j in range(H):
            if P[i, j] > 0:
                m += 1
        if m == 0:
            for j in range(H):
                mean[i, j] = 0.0
                var[i, j] = K[j, j]
            continue
        obs = np.empty(m, np.int64)
        c = 0
        for j in range(H):
            if P[i, j] > 0:
                obs[c] = j
                c += 1
        A = np.empty((m, m))
        yb = np.empty(m)
        for a in range(m):
            for b in range(m):
                A[a, b] = K[obs[a], obs[b]]
            A[a, a] += 1.0 / P[i, obs[a]]
            yb[a] = S[i, obs[a]] / P[i, obs[a]]
        Kob = np.empty((m, H))
        for a in range(m):
            for j in range(H):
                Kob[a, j] = K[obs[a], j]
        alpha = np.linalg.solve(A, yb)
        W = np.linalg.solve(A, Kob)
        for j in range(H):
            s = 0.0
            q = 0.0
            for a in range(m):
                s += Kob[a, j] * alpha[a]
                q += Kob[a, j] * W[a, j]
            mean[i, j] = s
            var[i, j] = K[j, j] - q
    return mean, var


@njit(parallel=True, cache=True)
def _ml_terms(S, P, bases, theta):
    """Per-player log marginal likelihood and its gradient wrt log-variances."""
    n, H = S.shape
    nb = bases.shape[0]
    ll = np.zeros(n)
    gr = np.zeros((n, nb))
    ew = np.exp(theta)
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
        A = np.zeros((m, m))
        for k in range(nb):
            for a in range(m):
                for b in range(m):
                    A[a, b] += ew[k] * bases[k, obs[a], obs[b]]
        yb = np.empty(m)
        for a in range(m):
            A[a, a] += 1.0 / P[i, obs[a]]
            yb[a] = S[i, obs[a]] / P[i, obs[a]]
        L = np.linalg.cholesky(A)
        Ainv = np.linalg.inv(A)
        alpha = Ainv @ yb
        logdet = 0.0
        for a in range(m):
            logdet += 2.0 * np.log(L[a, a])
        ll[i] = -0.5 * (yb @ alpha + logdet + m * np.log(2 * np.pi))
        for k in range(nb):
            s1 = 0.0
            s2 = 0.0
            for a in range(m):
                for b in range(m):
                    Bab = bases[k, obs[a], obs[b]]
                    s1 += alpha[a] * Bab * alpha[b]
                    s2 += Ainv[b, a] * Bab
            gr[i, k] = 0.5 * ew[k] * (s1 - s2)
    return ll, gr


def fit_kernel(S, P, bases, theta0=None, verbose=True):
    """Maximize the marginal likelihood over log-variances of the bases."""
    from scipy.optimize import minimize
    B = np.ascontiguousarray(np.stack(bases).astype(np.float64))
    nb = B.shape[0]
    th0 = np.full(nb, np.log(1e-4)) if theta0 is None else np.asarray(theta0, float)

    def f(th):
        ll, gr = _ml_terms(S, P, B, th)
        return -ll.sum(), -gr.sum(axis=0)

    res = minimize(f, th0, jac=True, method="L-BFGS-B",
                   bounds=[(np.log(1e-9), np.log(1e-1))] * nb)
    if verbose:
        print(f"  ml fit: -ll {res.fun:.1f}, sd(pp) "
              f"{np.round(100 * np.sqrt(np.exp(res.x)), 2)}", flush=True)
    return res.x, -res.fun


def kernel_from(bases, theta):
    return np.tensordot(np.exp(theta), np.stack(bases), axes=1)


# ------------------------------------------------------------------ state

def static_state(d, mask, n_players, H, r_adj):
    """Per-player per-hero sufficient statistics from rows in mask."""
    idx = d["pid"][mask] * H + d["hero"][mask]
    size = int(n_players) * H
    S = np.bincount(idx, weights=r_adj[mask] / d["v"][mask], minlength=size)
    P = np.bincount(idx, weights=1.0 / d["v"][mask], minlength=size)
    N = np.bincount(idx, minlength=size).astype(np.float64)
    W = np.bincount(idx, weights=d["y"][mask], minlength=size)
    R = np.bincount(idx, weights=d["r"][mask], minlength=size)
    sh = (int(n_players), H)
    return S.reshape(sh), P.reshape(sh), N.reshape(sh), W.reshape(sh), R.reshape(sh)


def online_state(d, add_mask, query_mask, H, r_adj, half_life=None, lag=1):
    """For each query row (in d's day order), per-hero stats of that player
    from add_mask rows with day <= query day - lag, exponentially decayed
    with the given half-life in days (None = no decay).
    Returns (qidx, S, P, N_hero_raw, n_player_raw, W_hero, R_hero) where S, P
    are (nq, H) and the rest refer to the query's own hero."""
    day = d["day"]
    assert np.all(np.diff(day) >= 0)
    npl = int(d["n_players"])
    lam = 0.0 if half_life is None else np.log(2) / half_life
    d0 = int(day.min())
    Sst = np.zeros((npl, H))
    Pst = np.zeros((npl, H))
    Nst = np.zeros((npl, H), np.float32)
    Wst = np.zeros((npl, H), np.float32)
    Rst = np.zeros((npl, H), np.float32)
    qidx = np.flatnonzero(query_mask)
    nq = len(qidx)
    S = np.empty((nq, H))
    P = np.empty((nq, H))
    Nh = np.empty(nq, np.float32)
    Np = np.empty(nq, np.float32)
    Wh = np.empty(nq, np.float32)
    Rh = np.empty(nq, np.float32)
    add_idx = np.flatnonzero(add_mask)
    ap = 0
    qdays = day[qidx]
    ud, qstart = np.unique(qdays, return_index=True)
    qend = np.r_[qstart[1:], nq]
    for t, a, b in zip(ud, qstart, qend):
        # add all rows with day <= t - lag
        e = np.searchsorted(day[add_idx], t - lag, side="right")
        if e > ap:
            rr = add_idx[ap:e]
            sc = np.exp(lam * (day[rr] - d0))
            pi, hi = d["pid"][rr], d["hero"][rr]
            np.add.at(Sst, (pi, hi), sc * r_adj[rr] / d["v"][rr])
            np.add.at(Pst, (pi, hi), sc / d["v"][rr])
            np.add.at(Nst, (pi, hi), 1.0)
            np.add.at(Wst, (pi, hi), d["y"][rr])
            np.add.at(Rst, (pi, hi), d["r"][rr])
            ap = e
        q = qidx[a:b]
        pq, hq = d["pid"][q], d["hero"][q]
        dec = np.exp(-lam * (t - d0))
        S[a:b] = Sst[pq] * dec
        P[a:b] = Pst[pq] * dec
        Nh[a:b] = Nst[pq, hq]
        Np[a:b] = Nst[pq].sum(axis=1)
        Wh[a:b] = Wst[pq, hq]
        Rh[a:b] = Rst[pq, hq]
    return qidx, S, P, Nh, Np, Wh, Rh


def lagged_value(d, values, group_key, lag):
    """For each row, the latest non-NaN value of `values` within the same
    group from a row with day <= day - lag (NaN if none)."""
    n = len(values)
    o = np.lexsort((d["replay_id"], d["day"], group_key))
    gk = group_key[o]
    ds = d["day"][o].astype(np.int64)
    vs = values[o]
    out = np.full(n, np.nan, np.float32)
    brk = np.flatnonzero(np.r_[True, gk[1:] != gk[:-1]])
    ends = np.r_[brk[1:], n]
    # forward-fill last non-NaN index within group
    valid = ~np.isnan(vs)
    idx = np.where(valid, np.arange(n), -1)
    # group-wise running max of idx
    grp = np.repeat(np.arange(len(brk)), ends - brk)
    runmax = np.maximum.accumulate(idx + 0)  # crosses groups; fixed below
    gstart = brk[grp]
    runmax = np.where(runmax >= gstart, runmax, -1)
    # position of the last row with day <= day - lag in the same group
    comp = grp.astype(np.int64) * 100000 + ds
    pos = np.searchsorted(comp, comp - lag, side="right") - 1
    okp = (pos >= 0) & (pos >= gstart)
    j = np.where(okp, runmax[np.maximum(pos, 0)], -1)
    good = okp & (j >= gstart)
    tmp = np.full(n, np.nan, np.float32)
    tmp[good] = vs[j[good]]
    out[o] = tmp
    return out


# ------------------------------------------------------------------ metrics

def fit_logistic(X, y, iters=100, l2=1e-6):
    X1 = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ w))
        g = X1.T @ (y - p) - l2 * w
        Hm = (X1 * (p * (1 - p))[:, None]).T @ X1 + l2 * np.eye(len(w))
        step = np.linalg.solve(Hm, g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w


def predict(w, X):
    return 1 / (1 + np.exp(-(np.column_stack([np.ones(len(X)), X]) @ w)))


def game_metrics(p, y):
    eps = 1e-7
    ll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    acc = np.mean((p > 0.5) == (y == 1))
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    lo = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    slope = fit_logistic(lo[:, None], y)[1]
    return {"logloss": float(ll), "acc": float(acc), "ece": float(ece),
            "cal_slope": float(slope)}


@njit(parallel=True, cache=True)
def _ml_G(S, P, K, nchunk):
    """Total log marginal likelihood and dLL/dK (H x H, summed over players)."""
    n, H = S.shape
    Gc = np.zeros((nchunk, H, H))
    llc = np.zeros(nchunk)
    step = (n + nchunk - 1) // nchunk
    for c in prange(nchunk):
        for i in range(c * step, min(n, (c + 1) * step)):
            m = 0
            for j in range(H):
                if P[i, j] > 0:
                    m += 1
            if m == 0:
                continue
            obs = np.empty(m, np.int64)
            q = 0
            for j in range(H):
                if P[i, j] > 0:
                    obs[q] = j
                    q += 1
            A = np.empty((m, m))
            yb = np.empty(m)
            for a in range(m):
                for b in range(m):
                    A[a, b] = K[obs[a], obs[b]]
                A[a, a] += 1.0 / P[i, obs[a]]
                yb[a] = S[i, obs[a]] / P[i, obs[a]]
            L = np.linalg.cholesky(A)
            Ainv = np.linalg.inv(A)
            alpha = Ainv @ yb
            logdet = 0.0
            for a in range(m):
                logdet += 2.0 * np.log(L[a, a])
            llc[c] += -0.5 * (yb @ alpha + logdet + m * np.log(2 * np.pi))
            for a in range(m):
                for b in range(m):
                    Gc[c, obs[a], obs[b]] += 0.5 * (alpha[a] * alpha[b] - Ainv[a, b])
    return llc.sum(), Gc.sum(axis=0)


def fit_kernel_lowrank(S, P, bases, rank, theta0, V0=None, seed=0, verbose=True, maxiter=300):
    """K = sum exp(theta_j) B_j + V V^T, V (H x rank); maximize the marginal
    likelihood over theta and V (probabilistic matrix factorization of the
    player x hero residual matrix, player factors integrated out)."""
    from scipy.optimize import minimize
    B = np.stack(bases).astype(np.float64)
    nb, H = B.shape[0], B.shape[1]
    rng = np.random.RandomState(seed)
    V = 0.002 * rng.randn(H, rank) if V0 is None else V0
    x0 = np.r_[np.asarray(theta0, float), V.ravel()]

    def unpack(x):
        return x[:nb], x[nb:].reshape(H, rank)

    def f(x):
        th, V = unpack(x)
        K = np.tensordot(np.exp(th), B, axes=1) + V @ V.T
        ll, G = _ml_G(S, P, K, 128)
        gth = np.exp(th) * np.einsum("ij,kij->k", G, B)
        gV = 2 * G @ V
        return -ll, -np.r_[gth, gV.ravel()]

    bounds = [(np.log(1e-9), np.log(1e-1))] * nb + [(None, None)] * (H * rank)
    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter})
    th, V = unpack(res.x)
    if verbose:
        print(f"  lowrank r={rank}: -ll {res.fun:.1f} ({res.nit} it), sd(pp) "
              f"{np.round(100 * np.sqrt(np.exp(th)), 2)}, factor sd(pp) "
              f"{np.round(100 * np.sqrt((V ** 2).sum(0)/H), 2)}", flush=True)
    return th, V, -res.fun
