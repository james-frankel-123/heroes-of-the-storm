"""
P3 rich CF (results/P3_RICH_CF.md): richer player x hero skill models than
the rank-2 CF kernel, all in one factor form so that every variant runs
through the same as-of state, fit and tests.

Model. For player p and hero h, skill theta_ph (detrended residual units,
as P3_HERO_STRENGTH):

    theta_ph = U_h . a_p + e_ph,   a_p ~ N(W z_p, I_K),  e_ph ~ N(0, s2_h)

U (H x K) holds the hero loadings: an overall-skill column (all ones times a
scale), optional hand-made role / fine-role / melee indicator columns (each
group with one scale), and the learned CF factors V. This is the same model
as the phase-1 GP kernel K = U U^T + diag(s2): today's "+CF rank 2" kernel is
the special case with role, fine and melee columns, V of rank 2 and one
shared s2.

Win channel: per hero, the as-of sufficient statistics S = sum r_adj / v and
P = sum 1 / v (P3_HERO_STRENGTH), i.e. a cell mean S / P with noise 1 / P.
Implicit feedback (variant 3, SVD++ style): z_p is the player's as-of play
preference (log1p game counts per hero as shares of the player's total,
standardized per hero, plus log volume); W (K x Dz) maps it to the prior
mean of a_p. Counts never contain outcomes.
Multi-target (variant 4): for metric m, the cell mean of the standardized
per-game performance (p3_rcf_data.py) is G_m,h . a_p + f + noise with
cell-specific variance rho2_m and per-game noise eps2_m, sharing a_p with the
win channel.

Everything reduces to K x K algebra per player (Woodbury), so the posterior
of a_p given win stats, metrics and z is exact:
    Lam = I + sum_h w_h U_h U_h^T + sum_m,h wm_h G_mh G_mh^T
    eta = W z + sum_h w_h ybar_h U_h + sum_m,h wm_h (xbar_mh - c_mh) G_mh
with w_h = P_h / (1 + s2_h P_h). At hero h:
    E[theta_h] = (1 - s2_h w_h) U_h a_hat + s2_h w_h ybar_h
    Var        = (1 - s2_h w_h)^2 U_h Lam^-1 U_h^T + s2_h (1 - s2_h w_h)

Fit objective (as-of by construction): split E at TZ. For each player the
conditioning set is every history row before TZ (window rows, plus Heroes
Profile rows for "merged"), its metrics and its play counts; the target is
the player's E slot rows from TZ on (win channel only). The objective is the
predictive log likelihood of the target cell means given the conditioning
set, summed over the fit half of players (same random half as phase 1). ARD
(hierarchical shrinkage per factor, Bayesian PCA with the column scale
profiled out) lets the data switch factors off. The same predictive log
likelihood on the held-out half is test 1.

Stages (from training/):
  prep   --hist window|merged                       cache/rcf/fit_<hist>.npz
  fit    --hist ... --tag ... [--kmax --ard --role --implicit --metrics]
                                                     results/rcf/params/<tag>.npz
  online --tag ...                                   results/rcf/p3c/<tag>.npz
  cassia --tag ...  / cover --tag ...                results/rcf/{cassia,cover}_<tag>.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C

RCF_CACHE = os.path.join(C.CACHE, "rcf")
RCF_RES = os.path.join(C.RESULTS, "rcf")
TZ = C.day_of("2025-03-01")
ROWS = os.environ.get("P3_RCF_ROWS") or os.path.join(RCF_CACHE, "merged_rows_oct05.npz")
KERN = os.environ.get("P3_RCF_KERNELS") or os.path.join(RCF_CACHE, "hs_kernels.npz")
H = 90
HERE_DIR = os.path.dirname(os.path.abspath(__file__))


def fit_half(n_players):
    """The phase-1 random half (p3_hs_fit: RandomState(7)); the snapshot
    players keep their phase-1 assignment."""
    return np.random.RandomState(7).rand(int(n_players)) < 0.5


# ------------------------------------------------------------------ rows

def load_aug(hist):
    """Slot rows plus (merged) the Heroes Profile rows as one day-sorted row
    set, with r_adj (experience offset from slot-row counts, p3_c_merged
    exp_counts="slots") and the slot index of each row (-1 for HP rows)."""
    import p3_x_common as X
    import p3_c_merged as M
    d = X.load_ext()
    for k in ("player_mmr", "hero_mmr", "role_mmr", "hero_level", "party"):
        d.pop(k, None)
    d["wp_pop"] = d["wp"].copy()
    mmr = os.environ.get("P3_RCF_MMR") == "1"
    if mmr:
        mmr_adjust(d)
    elif os.environ.get("P3_RCF_MMR") == "combiner":
        # control: the MMR term in the game-level combiner only, residual histories unchanged
        mmr_adjust(d, apply=False)
    cpath = os.path.join(RCF_CACHE, f"aug_{hist}{'_mmr' if mmr else ''}.npz")
    if os.path.exists(cpath):
        z = np.load(cpath)
        aug = {k: z[k] for k in z.files}
        aug["n_players"] = d["n_players"]
        assert int(aug["n_slots"]) == len(d["pid"]) and str(aug["fingerprint"]) == aug_fingerprint(d)
        print(f"  loaded aug cache {cpath}", flush=True)
        return d, aug
    ex = None
    if hist == "merged":
        ex, _ = M.load_rows(ROWS, d)
    n = len(d["pid"])
    q = np.zeros(n, bool)
    A = M.merged_predict(d, ex, None, q, exp_counts="slots", aug_only=True)
    o = A["order"]
    aug = A["aug"]
    aug["r_adj"] = A["r_adj"]
    aug["slot"] = np.where(o < n, o, -1)
    aug["table"] = A["table"]
    from p3_keys import key_blizz
    rid_all = d["replay_id"].astype(np.int64) if ex is None else np.r_[d["replay_id"].astype(np.int64),
                                                                      ex["replay_id"].astype(np.int64)]
    aug["rid"] = rid_all[o]
    # compact dtypes (identical values); bl is derived from pid when needed
    for k, dt in (("day", np.int32), ("pid", np.int32), ("hero", np.int16), ("slot", np.int32), ("y", np.float32)):
        assert np.array_equal(aug[k].astype(dt), aug[k])
        aug[k] = aug[k].astype(dt)
    if os.environ.get("P3_RCF_WRITE_AUG") == "1":
        os.makedirs(RCF_CACHE, exist_ok=True)
        np.savez(cpath, **{k: v for k, v in aug.items() if k != "n_players"}, n_slots=np.int64(len(d["pid"])),
                 fingerprint=np.array(aug_fingerprint(d)))
        print(f"  wrote aug cache {cpath}", flush=True)
    return d, aug


def aug_fingerprint(d):
    import hashlib
    h = hashlib.sha1()
    for k in ("replay_id", "pid", "hero", "y"):
        h.update(np.ascontiguousarray(d[k]).tobytes())
    h.update(np.ascontiguousarray(d["wp_pop"]).tobytes())
    if "mmr_coef" in d:
        h.update(np.round(np.asarray(d["mmr_coef"]), 12).tobytes())
    return h.hexdigest()[:16]


def aug_bl(aug, d):
    from p3_keys import key_blizz
    return key_blizz(d["player_keys"])[aug["pid"]].astype(np.int64)


def mmr_adjust(d, apply=True):
    """Variant (e): the MMR-adjusted expectation. Each slot's baseline
    becomes wp' = sigmoid(logit wp + b . x), x = at-game MMR covariates of
    p3_mmr_feats.design (team difference, own minus team mean, hero minus
    player MMR; zero where unknown), b fit by logistic regression with
    logit wp as offset on E slots. r and v are recomputed from wp'; every
    later stage (offset table, state, targets) then works on MMR-adjusted
    residuals. d["wp_pop"] keeps the population WP; d["mmr_adj"] = wp' - wp."""
    import p3_mmr_feats as MF
    f = MF.slot_features(d["replay_id"], d["player_keys"][d["pid"]], d["team"], MF.load_stamps(C.CACHE))
    Xm = MF.design(f)
    e = d["in_sample"] & (d["day"] >= C.day_of(C.E_START))
    off = np.log(d["wp"] / (1 - d["wp"]))
    yv = d["y"].astype(np.float64)
    cf = os.path.join(RCF_CACHE, "mmr_coef.npy")
    w = np.load(cf) if os.path.exists(cf) else np.zeros(Xm.shape[1])
    for _ in range(0 if os.path.exists(cf) else 50):
        p = 1 / (1 + np.exp(-(off[e] + Xm[e] @ w)))
        g = Xm[e].T @ (yv[e] - p)
        Hm = (Xm[e] * (p * (1 - p))[:, None]).T @ Xm[e] + 1e-9 * np.eye(len(w))
        st = np.linalg.solve(Hm, g)
        w += st
        if np.abs(st).max() < 1e-10:
            break
    if not os.path.exists(cf):
        os.makedirs(RCF_CACHE, exist_ok=True)
        np.save(cf, w)
    wp2 = 1 / (1 + np.exp(-(off + Xm @ w)))
    d["mmr_adj"] = wp2 - d["wp"]
    if not apply:
        return
    d["mmr_coef"] = w
    d["wp"] = wp2
    d["r"] = d["y"] - d["wp"]
    d["v"] = d["wp"] * (1 - d["wp"])
    print(f"  MMR-adjusted expectation: coef (dteam, own, hrel, hrel known) {np.round(w, 4).tolist()}; "
          f"slots with full MMR {f['full'].mean():.3f}; mean |adj| {np.abs(d['mmr_adj']).mean() * 100:.2f}pp", flush=True)


def slot_metrics():
    z = np.load(os.path.join(RCF_CACHE, "sb_z.npz"))
    return z["Z"], [str(m) for m in z["metrics"]]


def aug_metrics(aug, card, d=None):
    """Metric rows (float16, n_aug x M) and evidence weights for every aug
    row. card "legacy": the first export (sb_z.npz, window slot rows only;
    met2_hs2 as committed). Other cards: p3_rcf_cards.py, joined on
    (replay_id, blizz_id), so HP history rows that are also corpus games get
    their scoreboard too."""
    n = len(aug["day"])
    if card in (None, "legacy"):
        Z, _ = slot_metrics()
        out = np.full((n, Z.shape[1]), np.nan, np.float16)
        sl = aug["slot"] >= 0
        out[sl] = Z[aug["slot"][sl]]
        return out, np.ones(n, np.float32)
    zc = os.path.join(RCF_CACHE, f"zaug_{card}_{n}.npz")
    if os.path.exists(zc):
        z = np.load(zc)
        print(f"  loaded metric cache {zc}", flush=True)
        return z["Z"], z["W"]
    z = np.load(os.path.join(RCF_CACHE, f"sb_{card}.npz"))
    comp = (z["replay_id"].astype(np.int64) << 40) | z["blizz_id"].astype(np.int64)
    o = np.argsort(comp)
    cs = comp[o]
    bl = aug["bl"] if "bl" in aug else aug_bl(aug, d)
    q = (aug["rid"].astype(np.int64) << 40) | bl
    j = np.minimum(np.searchsorted(cs, q), len(cs) - 1)
    hit = cs[j] == q
    Zs = z["Z"]
    out = np.full((n, Zs.shape[1]), np.nan, np.float16)
    out[hit] = Zs[o[j[hit]]]
    W = np.ones(n, np.float32)
    W[hit] = z["W"][o[j[hit]]].astype(np.float32)
    sl = aug["slot"] >= 0
    print(f"  metrics {card}: slot rows with a value {(~np.isnan(out[sl])).any(1).mean():.3f}, "
          f"other rows {(~np.isnan(out[~sl])).any(1).sum():,}", flush=True)
    if os.environ.get("P3_RCF_WRITE_AUG") == "1":
        np.savez(zc, Z=out, W=W)
    return out, W


def metric_sums(aug, Zaug, Waug, mask, pos, n):
    """(n, H, M) weighted sums of metric z and weighted counts over aug rows
    in mask, players mapped by pos (-1 = skip)."""
    M = Zaug.shape[1]
    rows = np.flatnonzero(mask & (pos[aug["pid"]] >= 0))
    cell = pos[aug["pid"][rows]] * H + aug["hero"][rows]
    X = np.zeros((n * H, M), np.float32)
    Cc = np.zeros((n * H, M), np.float32)
    w = Waug[rows]
    for k in range(M):
        x = Zaug[rows, k].astype(np.float32)
        g = ~np.isnan(x)
        X[:, k] = np.bincount(cell[g], weights=(w * x)[g], minlength=n * H)
        Cc[:, k] = np.bincount(cell[g], weights=w[g], minlength=n * H)
    return X.reshape(n, H, M), Cc.reshape(n, H, M)


def z_features(N, mean=None, sd=None):
    """Play preference from counts (n, H): log1p shares standardized per
    hero, plus standardized log volume. Players without history get 0."""
    L = np.log1p(N)
    tot = L.sum(1, keepdims=True)
    s = np.where(tot > 0, L / np.maximum(tot, 1e-9), 0.0)
    vol = np.log1p(N.sum(1, keepdims=True))
    F = np.concatenate([s, vol], 1)
    if mean is None:
        has = tot[:, 0] > 0
        mean, sd = F[has].mean(0), F[has].std(0) + 1e-6
    Z = (F - mean) / sd
    Z[tot[:, 0] <= 0] = 0.0
    return Z.astype(np.float32), mean, sd


# ------------------------------------------------------------------ prep

def prep(hist, card="legacy"):
    """Static fit arrays for players with E target rows (day >= TZ):
    S1, P1, N1 and metric sums X1, C1 (rows before TZ); S2, P2 (target E slot
    rows from TZ on); SE, PE (all E slot rows: the phase-1 marginal test)."""
    t0 = time.time()
    d, aug = load_aug(hist)
    if card == "legacy":
        mets = slot_metrics()[1]
    else:
        mets = [str(m) for m in np.load(os.path.join(RCF_CACHE, f"sb_{card}.npz"))["metrics"]]
    M = len(mets)
    npl = int(d["n_players"])
    e_start = C.day_of(C.E_START)
    e_last = int(d["day"][d["in_sample"]].max())
    slot = aug["slot"]
    is_slot = slot >= 0
    ins = np.zeros(len(slot), bool)
    ins[is_slot] = d["in_sample"][slot[is_slot]]
    day, pid, hero = aug["day"], aug["pid"], aug["hero"]
    r_adj, v = aug["r_adj"], aug["v"]
    e_slot = is_slot & ins & (day >= e_start) & (day <= e_last)
    tgt = e_slot & (day >= TZ)
    cond = (day < TZ)
    players = np.unique(pid[tgt])
    pos = np.full(npl, -1, np.int64)
    pos[players] = np.arange(len(players))
    n = len(players)
    print(f"[prep {hist}] rows {len(day):,}; target rows {tgt.sum():,}; cond rows {cond.sum():,}; "
          f"players {n:,}", flush=True)

    def acc(mask, w):
        idx = pos[pid[mask]] * H + hero[mask]
        ok = pos[pid[mask]] >= 0
        return np.bincount(idx[ok], weights=w[ok] if w is not None else None,
                           minlength=n * H).reshape(n, H)
    out = {"players": players, "fit": fit_half(npl)[players]}
    out["S1"] = acc(cond, r_adj[cond] / v[cond]).astype(np.float32)
    out["P1"] = acc(cond, 1.0 / v[cond]).astype(np.float32)
    out["N1"] = acc(cond, None).astype(np.float32)
    out["S2"] = acc(tgt, r_adj[tgt] / v[tgt]).astype(np.float32)
    out["P2"] = acc(tgt, 1.0 / v[tgt]).astype(np.float32)
    out["SE"] = acc(e_slot, r_adj[e_slot] / v[e_slot]).astype(np.float32)
    out["PE"] = acc(e_slot, 1.0 / v[e_slot]).astype(np.float32)
    # metrics of conditioning rows (every row with a scoreboard)
    Zaug, Waug = aug_metrics(aug, card, d)
    X1, C1 = metric_sums(aug, Zaug, Waug, cond, pos, n)
    if card != "legacy":
        p = os.path.join(RCF_CACHE, f"fitm_{hist}_{card}.npz")
        np.savez_compressed(p, X1=X1, C1=C1, players=players)
        print(f"[prep {hist} {card}] wrote {p} ({time.time() - t0:.0f}s)", flush=True)
        if os.path.exists(os.path.join(RCF_CACHE, f"fit_{hist}.npz")):
            return
    out["X1"] = X1
    out["C1"] = C1
    out["metrics"] = np.array(mets)
    out["hero_names"] = d["hero_names"]
    out["tz"] = np.int64(TZ)
    out["hist"] = np.array(hist)
    os.makedirs(RCF_CACHE, exist_ok=True)
    p = os.path.join(RCF_CACHE, f"fit_{hist}.npz")
    np.savez_compressed(p, **out)
    f = out["fit"]
    print(f"[prep {hist}] wrote {p}: players {n:,} (fit {f.sum():,}, held-out {(~f).sum():,}); "
          f"median cond games {np.median(out['N1'].sum(1)):.0f}; {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ model (numpy posterior)

class Params:
    """Fitted variant: U (H, K), s2 (H,), optional W (K, Dz) with z mean/sd,
    optional metric loadings G (M, H, K), intercepts c (M, H), rho2, eps2 (M)."""

    def __init__(self, U, s2, W=None, zmean=None, zsd=None, G=None, c=None, rho2=None, eps2=None, meta=None):
        self.U = np.asarray(U, np.float64)
        self.s2 = np.broadcast_to(np.asarray(s2, np.float64), (self.U.shape[0],)).copy()
        self.W, self.zmean, self.zsd = W, zmean, zsd
        self.G, self.c, self.rho2, self.eps2 = G, c, rho2, eps2
        self.meta = meta or {}

    @property
    def K(self):
        return self.U.shape[1]

    def kernel(self):
        return self.U @ self.U.T + np.diag(self.s2)

    def save(self, path):
        d = {"U": self.U, "s2": self.s2, "meta": np.array(json.dumps(self.meta))}
        for k in ("W", "zmean", "zsd", "G", "c", "rho2", "eps2"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        np.savez(path, **d)

    @staticmethod
    def load(path):
        z = np.load(path)
        g = lambda k: z[k] if k in z.files else None
        return Params(z["U"], z["s2"], g("W"), g("zmean"), g("zsd"), g("G"), g("c"), g("rho2"), g("eps2"),
                      json.loads(str(z["meta"])))

    @staticmethod
    def from_kernel_file(path, name="+CF rank 2", drop=()):
        """Today's phase-1 kernel in factor form (exact: every structural
        basis is low rank plus the identity)."""
        from p3_hs_fit import KERNELS
        kz = np.load(path)
        bases = {k[6:]: kz[k] for k in kz.files if k.startswith("basis_")}
        th = kz["theta_" + name]
        cols, s2 = [], 0.0
        for b, t in zip(KERNELS[name], th):
            if b in drop:
                continue
            var = np.exp(t)
            if b == "hero":
                s2 = var
            elif b.startswith("cf"):
                V = kz["V_" + b] if ("V_" + b) in kz.files else None
                cols.append(np.sqrt(var) * V)
            else:
                B = bases[b]
                # B is a same-group indicator matrix: one column per group
                _, grp = np.unique(B, axis=0, return_inverse=True)
                ind = np.eye(grp.max() + 1)[grp.ravel()]
                assert np.allclose(ind @ ind.T, B)
                cols.append(np.sqrt(var) * ind)
        return Params(np.concatenate(cols, 1), s2, meta={"source": path, "name": name, "drop": list(drop)})


def posterior(par, S, P, h, N=None, X=None, Cm=None, want_all=False, chunk=8000):
    """Chunked wrapper of _posterior (bounds the float64 temporaries)."""
    n = S.shape[0]
    if n <= chunk:
        return _posterior(par, S, P, h, N, X, Cm, want_all)
    outs = []
    for i0 in range(0, n, chunk):
        sl = slice(i0, i0 + chunk)
        outs.append(_posterior(par, S[sl], P[sl], h[sl], None if N is None else N[sl],
                               None if X is None else X[sl], None if Cm is None else Cm[sl], want_all))
    return np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])


def _posterior(par, S, P, h, N=None, X=None, Cm=None, want_all=False):
    """Posterior mean and variance of theta at hero h[i] for each row i
    given win stats S, P (n, H), counts N (n, H) for z, metric sums X and
    counts Cm (n, H, M). Batched K x K algebra (float64)."""
    U, s2 = par.U, par.s2
    n, K = S.shape[0], par.K
    S = S.astype(np.float64)
    P = P.astype(np.float64)
    den = 1.0 + s2[None, :] * P
    w = P / den
    wy = S / den
    UU = (U[:, :, None] * U[:, None, :]).reshape(H, K * K)
    Lam = (w @ UU).reshape(n, K, K) + np.eye(K)[None]
    eta = wy @ U
    if par.W is not None and N is not None:
        zf, _, _ = z_features(N, par.zmean, par.zsd)
        eta += zf.astype(np.float64) @ par.W.T
    if par.G is not None and X is not None:
        Mn = par.G.shape[0]
        Cm = Cm.astype(np.float64)
        wm = Cm / (par.rho2[None, None, :] * Cm + par.eps2[None, None, :])     # (n, H, M)
        xb = np.where(Cm > 0, X / np.maximum(Cm, 1e-9), 0.0) - par.c.T[None]   # (n, H, M)
        Gt = np.transpose(par.G, (1, 0, 2)).reshape(H * Mn, K)                   # (H*M, K)
        GG = (Gt[:, :, None] * Gt[:, None, :]).reshape(H * Mn, K * K)
        Lam += (wm.reshape(n, H * Mn) @ GG).reshape(n, K, K)
        eta += (wm * xb).reshape(n, H * Mn) @ Gt
    L = np.linalg.cholesky(Lam)
    ahat = np.linalg.solve(Lam, eta[:, :, None])[:, :, 0]
    if want_all:
        a = 1.0 - s2[None, :] * w
        mean = a * (ahat @ U.T) + s2[None, :] * wy
        Linv = np.linalg.inv(L)                                    # (n, K, K)
        Q = np.einsum("nkl,hl->nhk", Linv, U)
        var = a ** 2 * (Q ** 2).sum(2) + s2[None, :] * a
        return mean, var
    Uh = U[h]                                                      # (n, K)
    ah = 1.0 - s2[h] * w[np.arange(n), h]
    mean = ah * (Uh * ahat).sum(1) + s2[h] * wy[np.arange(n), h]
    q = np.linalg.solve(L, Uh[:, :, None])[:, :, 0]
    var = ah ** 2 * (q ** 2).sum(1) + s2[h] * ah
    return mean, var


# ------------------------------------------------------------------ fit (torch)

def fit(a):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    torch.manual_seed(a.seed)
    z = dict(np.load(os.path.join(RCF_CACHE, f"fit_{a.hist}.npz")))
    if a.sb != "legacy" and a.metrics:
        zm = np.load(os.path.join(RCF_CACHE, f"fitm_{a.hist}_{a.sb}.npz"))
        assert np.array_equal(zm["players"], z["players"])
        z["X1"], z["C1"] = zm["X1"], zm["C1"]
    fitm = z["fit"]
    if getattr(a, "flip_half", False):
        # judges for the P3 tournament: the same fit on the OTHER player half
        fitm = ~fitm
    use_m = a.metrics
    mets = [str(m) for m in z["metrics"]]
    Mn = len(mets)
    N1 = z["N1"]
    zf, zmean, zsd = z_features(N1[fitm])
    Dz = zf.shape[1]
    meta_h = C.hero_meta(z["hero_names"]) if a.role else None

    def tens(sel):
        g = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)
        T = {k: g(z[k][sel]) for k in ("S1", "P1", "S2", "P2")}
        T["Z"] = g(z_features(N1[sel], zmean, zsd)[0])
        if use_m:
            T["X1"] = g(z["X1"][sel])
            T["C1"] = g(z["C1"][sel])
        return T
    if a.objective == "marginal":
        # phase-1 style: marginal likelihood of all E slot rows, no conditioning
        def tens(sel):  # noqa: F811
            n = int(sel.sum())
            T = {"S1": torch.zeros(n, H, device=dev), "P1": torch.zeros(n, H, device=dev),
                 "S2": torch.from_numpy(z["SE"][sel]).to(dev), "P2": torch.from_numpy(z["PE"][sel]).to(dev),
                 "Z": torch.zeros(n, Dz, device=dev)}
            if use_m:
                T["X1"] = torch.zeros(n, H, Mn, device=dev)
                T["C1"] = torch.zeros(n, H, Mn, device=dev)
            return T
    if a.val:
        # inner validation: 20% of the fit half plays the held-out role; the
        # real held-out half is not touched (used to choose lam and kmax)
        rv = np.random.RandomState(11).rand(len(fitm)) < 0.2
        hold_sel, fitm = fitm & rv, fitm & ~rv
    else:
        hold_sel = ~fitm
    Tf, Th = tens(fitm), tens(hold_sel)
    nf, nh = int(fitm.sum()), int(hold_sel.sum())
    print(f"[fit {a.tag}] {a.hist}, fit players {nf:,}, held-out {nh:,}, device {dev}", flush=True)
    dt = torch.float64 if a.f64 else torch.float32
    K = a.kmax
    # parameters
    p = {}
    p["log_sp"] = torch.tensor(np.log(0.017), dtype=dt)
    p["V"] = torch.tensor(0.005 * np.random.RandomState(a.seed).randn(H, K), dtype=dt)
    p["log_s2"] = torch.full((H,) if a.hero_s2 else (1,), np.log(0.022 ** 2), dtype=dt)
    groups = []
    if a.role:
        for nm, lab in (("role", meta_h["blizz"]), ("fine", meta_h["fine"]), ("melee", meta_h["melee"].astype(int))):
            ind = np.eye(lab.max() + 1)[lab]
            groups.append((nm, torch.tensor(ind, dtype=dt)))
            p["log_s_" + nm] = torch.tensor(np.log(0.01), dtype=dt)
    if use_m:
        p["c"] = torch.zeros(Mn, H, dtype=dt)
        p["log_rho2"] = torch.full((Mn,), np.log(0.1), dtype=dt)
        p["log_eps2"] = torch.full((Mn,), np.log(0.8), dtype=dt)
    for k in p:
        p[k] = p[k].to(dev).requires_grad_(True)

    def build_U():
        cols = [torch.exp(p["log_sp"]) * torch.ones(H, 1, dtype=dt, device=dev), p["V"]]
        for nm, ind in groups:
            cols.append(torch.exp(p["log_s_" + nm]) * ind.to(dev))
        return torch.cat(cols, 1)
    nU = 1 + K + sum(g[1].shape[1] for g in groups)
    if a.kit:
        import json as _j
        kj = _j.load(open(os.path.join(os.path.dirname(HERE_DIR), "scilearn", "kit", "kit_2.57.json")))
        names = [str(x) for x in z["hero_names"]]
        miss = [n for n in names if n not in kj["heroes"]]
        assert not miss, f"kit file misses {miss}"
        A = np.array([[kj["heroes"][n]["values"][ax] for ax in kj["axes"]] for n in names], float)
        A = (A - A.mean(0)) / (A.std(0) + 1e-9)
        for j, ax in enumerate(kj["axes"]):
            groups.append(("kit_" + ax, torch.tensor(A[:, j:j + 1], dtype=dt)))
            p["log_s_kit_" + ax] = torch.tensor(np.log(0.003), dtype=dt, device=dev, requires_grad=True)
        nU = 1 + K + sum(g[1].shape[1] for g in groups)
    if a.implicit:
        p["W"] = torch.zeros(nU, Dz, dtype=dt, device=dev, requires_grad=True)
    if use_m:
        # G = shared loadings per metric (B, broadcast over heroes) + hero deviations D (L2, sd gsd)
        if a.grole:
            role_idx = torch.tensor(C.hero_meta(z["hero_names"])["blizz"], device=dev)
        shape = (Mn, 6, nU) if a.grole else (Mn, 1, nU) if a.gshare else (Mn, H, nU)
        p["G"] = torch.tensor(0.01 * np.random.RandomState(a.seed + 1).randn(*shape), dtype=dt, device=dev,
                              requires_grad=True)
        if (a.gshare or a.grole) and a.gsd > 0:
            p["D"] = torch.zeros(Mn, H, nU, dtype=dt, device=dev, requires_grad=True)
    # metric availability per (hero, metric): never-informative pairs are fixed at 0
    if use_m:
        avail = torch.tensor((z["C1"][fitm].sum(0) > 0).T.astype(np.float64), dtype=dt, device=dev)  # (M, H)

    def Geff():
        G = p["G"][:, role_idx] if a.grole else p["G"].expand(Mn, H, nU)
        return G + p["D"] if "D" in p else G

    def batch_ll(T, i0, i1):
        U = build_U()
        s2 = torch.exp(p["log_s2"]).expand(H)
        S1 = T["S1"][i0:i1].to(dev, dt)
        P1 = T["P1"][i0:i1].to(dev, dt)
        S2 = T["S2"][i0:i1].to(dev, dt)
        P2 = T["P2"][i0:i1].to(dev, dt)
        B = S1.shape[0]
        den = 1 + s2 * P1
        w1 = P1 / den
        wy1 = S1 / den
        UU = (U[:, :, None] * U[:, None, :]).reshape(H, nU * nU)
        Lam = (w1 @ UU).reshape(B, nU, nU) + torch.eye(nU, dtype=dt, device=dev)
        eta = wy1 @ U
        if a.implicit:
            eta = eta + T["Z"][i0:i1].to(dev, dt) @ p["W"].T
        if use_m:
            X1 = T["X1"][i0:i1].to(dev, dt)
            C1 = T["C1"][i0:i1].to(dev, dt)
            G = Geff() * avail[:, :, None]
            rho2 = torch.exp(p["log_rho2"])
            eps2 = torch.exp(p["log_eps2"])
            wm = C1 / (rho2 * C1 + eps2)                                  # (B, H, M)
            xb = torch.where(C1 > 0, X1 / C1.clamp_min(1e-9), torch.zeros_like(X1)) - p["c"].T
            Gt = G.permute(1, 0, 2).reshape(H * Mn, nU)
            GG = (Gt[:, :, None] * Gt[:, None, :]).reshape(H * Mn, nU * nU)
            Lam = Lam + (wm.reshape(B, H * Mn) @ GG).reshape(B, nU, nU)
            eta = eta + (wm * xb).reshape(B, H * Mn) @ Gt
        L = torch.linalg.cholesky(Lam)
        ahat = torch.cholesky_solve(eta[:, :, None], L)[:, :, 0]
        m2 = P2 > 0
        aa = 1 - s2 * w1
        cc = s2 * wy1
        Up = aa[:, :, None] * U[None]                                    # (B, H, nU)
        t = torch.where(m2, s2 * aa + 1 / P2.clamp_min(1e-9), torch.ones_like(P2))
        invt = torch.where(m2, 1 / t, torch.zeros_like(t))
        y2 = torch.where(m2, S2 / P2.clamp_min(1e-9), torch.zeros_like(S2))
        r = torch.where(m2, y2 - cc - (Up @ ahat[:, :, None])[:, :, 0], torch.zeros_like(y2))
        Lam2 = Lam + torch.einsum("bhk,bh,bhl->bkl", Up, invt, Up)
        beta = torch.einsum("bhk,bh->bk", Up, invt * r)
        L2 = torch.linalg.cholesky(Lam2)
        sol = torch.cholesky_solve(beta[:, :, None], L2)[:, :, 0]
        quad = (invt * r * r).sum(1) - (beta * sol).sum(1)
        ld = (torch.where(m2, torch.log(t), torch.zeros_like(t))).sum(1) \
            + 2 * torch.log(torch.diagonal(L2, dim1=1, dim2=2)).sum(1) \
            - 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
        mcount = m2.sum(1).to(dt)
        out = -0.5 * (quad + ld + mcount * np.log(2 * np.pi))
        if a.joint and joint_on[0]:
            # + log p(conditioning set | z): every conditioning win cell and metric
            # cell, so the metric loadings are a generative factor model of
            # the performance data (Woodbury with the same Lam, eta)
            o1 = P1 > 0
            q1 = torch.where(o1, S1 * S1 / (P1.clamp_min(1e-9) * den), torch.zeros_like(S1)).sum(1)
            lw = torch.where(o1, torch.log(w1.clamp_min(1e-30)), torch.zeros_like(w1)).sum(1)
            n1 = o1.sum(1).to(dt)
            mu0 = T["Z"][i0:i1].to(dev, dt) @ p["W"].T if a.implicit else torch.zeros_like(eta)
            if use_m:
                om = C1 > 0
                q1 = q1 + torch.where(om, wm * xb * xb, torch.zeros_like(wm)).sum((1, 2))
                lw = lw + torch.where(om, torch.log(wm.clamp_min(1e-30)), torch.zeros_like(wm)).sum((1, 2))
                n1 = n1 + om.sum((1, 2)).to(dt)
            ldL = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
            ll1 = -0.5 * (q1 + (mu0 * mu0).sum(1) - (eta * ahat).sum(1) + ldL - lw + n1 * np.log(2 * np.pi))
            out = out + ll1
        return out

    def glasso(x, dims, size):
        # group lasso: lam * sqrt(group size) * ||group||, one group per factor
        # (a Laplace scale-mixture prior per factor: whole factors switch off)
        return a.lam * np.sqrt(size) * torch.sqrt((x ** 2).sum(dims) + 1e-12).sum()

    def penalty():
        pen = torch.zeros((), dtype=dt, device=dev)
        if a.lam > 0:
            pen = pen + glasso(p["V"], 0, H)
        else:
            pen = pen + (p["V"] ** 2).sum() / (2 * 0.05 ** 2)
        if a.implicit:
            pen = pen + (p["W"] ** 2).sum() / (2 * a.wsd ** 2)
        if use_m:
            pen = pen + (p["G"] ** 2).sum() / (2 * 0.3 ** 2)
            if "D" in p:
                pen = pen + (p["D"] ** 2).sum() / (2 * a.gsd ** 2)
            pen = pen + (p["c"] ** 2).sum() / (2 * 0.3 ** 2)
        if a.hero_s2:
            ls = p["log_s2"]
            pen = pen + ((ls - ls.mean()) ** 2).sum() / (2 * 0.5 ** 2)
        return pen

    CH = a.chunk
    joint_on = [True]

    def total_ll(T, n, grad=False):
        tot = 0.0
        for i0 in range(0, n, CH):
            ll = batch_ll(T, i0, min(n, i0 + CH)).sum()
            if grad:
                (-ll).backward()
            tot += float(ll.detach())
        return tot

    params = list(p.values())
    opt = torch.optim.LBFGS(params, lr=1, max_iter=a.iters, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10)
    it = [0]
    t0 = time.time()

    def closure():
        opt.zero_grad()
        pen = penalty()
        pen.backward()
        ll = total_ll(Tf, nf, grad=True)
        it[0] += 1
        if it[0] % 25 == 1:
            print(f"  eval {it[0]}: fit ll {ll:.1f} pen {float(pen):.1f} ({time.time() - t0:.0f}s)", flush=True)
        return torch.tensor(-ll + float(pen))
    for rnd in range(a.rounds):
        opt.step(closure)
    joint_on[0] = False
    with torch.no_grad():
        llf = total_ll(Tf, nf)
        # per-player held-out ll
        hl = []
        for i0 in range(0, nh, CH):
            hl.append(batch_ll(Th, i0, min(nh, i0 + CH)).double().cpu().numpy())
        hl = np.concatenate(hl)
        U = build_U().double().cpu().numpy()
        s2 = torch.exp(p["log_s2"]).expand(H).double().cpu().numpy()
    colsd = 100 * np.sqrt((U ** 2).mean(0))
    vsd = colsd[1:1 + K]
    eff = int((vsd > 0.3).sum())
    print(f"[fit {a.tag}] fit ll {llf:.1f}; held-out ll {hl.sum():.1f}; V column sd (pp) "
          f"{np.round(np.sort(vsd)[::-1], 2).tolist()}; effective rank {eff}; "
          f"overall sd {colsd[0]:.2f}pp; hero sd {100 * np.sqrt(s2.mean()):.2f}pp; {time.time() - t0:.0f}s", flush=True)
    meta = {"tag": a.tag, "hist": a.hist, "kmax": K, "ard": a.ard, "role": a.role, "implicit": a.implicit,
            "metrics": use_m, "wsd": a.wsd, "gshare": a.gshare, "grole": a.grole, "kit": a.kit, "joint": a.joint, "sb": a.sb, "gsd": a.gsd, "objective": a.objective, "hero_s2": a.hero_s2, "fit_ll": llf,
            "heldout_ll": float(hl.sum()), "val": bool(a.val), "lam": a.lam, "v_sd_pp": vsd.tolist(), "effective_rank": eff,
            "overall_sd_pp": float(colsd[0]), "hero_sd_pp": float(100 * np.sqrt(s2.mean())),
            "seconds": round(time.time() - t0), "n_fit": nf, "n_heldout": nh}
    kw = {}
    if a.implicit:
        kw.update(W=p["W"].detach().double().cpu().numpy(), zmean=zmean, zsd=zsd)
    if use_m:
        G = (Geff() * avail[:, :, None]).detach().double().cpu().numpy()
        kw.update(G=G, c=p["c"].detach().double().cpu().numpy(),
                  rho2=torch.exp(p["log_rho2"]).detach().double().cpu().numpy(),
                  eps2=torch.exp(p["log_eps2"]).detach().double().cpu().numpy())
        meta["metric_loading_norm"] = dict(zip(mets, np.sqrt((G ** 2).sum((1, 2))).round(4).tolist()))
        meta["rho2"] = dict(zip(mets, kw["rho2"].round(4).tolist()))
        meta["eps2"] = dict(zip(mets, kw["eps2"].round(4).tolist()))
    if a.implicit:
        meta["W_row_norm"] = np.sqrt((kw["W"] ** 2).sum(1)).round(4).tolist()
    par = Params(U, s2, meta=meta, **kw)
    os.makedirs(os.path.join(RCF_RES, "params"), exist_ok=True)
    par.save(os.path.join(RCF_RES, "params", f"{a.tag}.npz"))
    np.savez(os.path.join(RCF_RES, "params", f"{a.tag}_heldout.npz"), ll=hl,
             players=z["players"][hold_sel])
    with open(os.path.join(RCF_RES, "params", f"{a.tag}.json"), "w") as f:
        json.dump(meta, f, indent=1)


def heldout_fixed(a):
    """Held-out predictive ll of a fixed parameter file (e.g. today's kernel
    in factor form), numpy, same arrays as fit()."""
    z = np.load(os.path.join(RCF_CACHE, f"fit_{a.hist}.npz"))
    par = Params.from_kernel_file(KERN, drop=tuple(a.drop.split(",")) if a.drop else ()) \
        if a.tag.startswith("today") else Params.load(os.path.join(RCF_RES, "params", f"{a.tag}.npz"))
    sel = ~z["fit"]
    if a.objective == "marginal":
        S1 = np.zeros((sel.sum(), H)); P1 = S1; S2, P2 = z["SE"][sel], z["PE"][sel]
    else:
        S1, P1, S2, P2 = z["S1"][sel], z["P1"][sel], z["S2"][sel], z["P2"][sel]
    ll = pred_ll_np(par, S1, P1, S2, P2)
    out = a.tag + ("_" + a.hist) + ("_marg" if a.objective == "marginal" else "")
    os.makedirs(os.path.join(RCF_RES, "params"), exist_ok=True)
    np.savez(os.path.join(RCF_RES, "params", f"{out}_heldout.npz"), ll=ll, players=z["players"][sel])
    print(f"[heldout {out}] {ll.sum():.1f} over {sel.sum():,} players")


def pred_ll_np(par, S1, P1, S2, P2, chunk=20000):
    """Numpy predictive ll (win channel only, no z or metrics)."""
    U, s2 = par.U, par.s2
    K = U.shape[1]
    out = []
    for i0 in range(0, len(S1), chunk):
        s1, p1 = S1[i0:i0 + chunk].astype(np.float64), P1[i0:i0 + chunk].astype(np.float64)
        sv, pv = S2[i0:i0 + chunk].astype(np.float64), P2[i0:i0 + chunk].astype(np.float64)
        B = len(s1)
        den = 1 + s2 * p1
        w1, wy1 = p1 / den, s1 / den
        UU = (U[:, :, None] * U[:, None, :]).reshape(H, K * K)
        Lam = (w1 @ UU).reshape(B, K, K) + np.eye(K)
        ah = np.linalg.solve(Lam, (wy1 @ U)[:, :, None])[:, :, 0]
        m2 = pv > 0
        aa = 1 - s2 * w1
        Up = aa[:, :, None] * U[None]
        t = np.where(m2, s2 * aa + 1 / np.maximum(pv, 1e-9), 1.0)
        invt = np.where(m2, 1 / t, 0.0)
        y2 = np.where(m2, sv / np.maximum(pv, 1e-9), 0.0)
        r = np.where(m2, y2 - s2 * wy1 - np.einsum("bhk,bk->bh", Up, ah), 0.0)
        Lam2 = Lam + np.einsum("bhk,bh,bhl->bkl", Up, invt, Up)
        beta = np.einsum("bhk,bh->bk", Up, invt * r)
        sol = np.linalg.solve(Lam2, beta[:, :, None])[:, :, 0]
        quad = (invt * r * r).sum(1) - (beta * sol).sum(1)
        ld = np.where(m2, np.log(t), 0).sum(1) + np.linalg.slogdet(Lam2)[1] - np.linalg.slogdet(Lam)[1]
        out.append(-0.5 * (quad + ld + m2.sum(1) * np.log(2 * np.pi)))
    return np.concatenate(out)


# ------------------------------------------------------------------ online

def online_walk(aug, query_slots, par, lag=1, metrics=None, use_z=True, mweights=None, hl_win=None, hl_met=None,
                parts=False, want_all=False, q_win=None, q_met=None, metric_feats=False, feat_mask=None):
    """As-of posterior for each query slot row (indices into the slot table,
    any order). State holds aug rows with day <= query day - lag; every
    query sees only rows on earlier days, so its own row (same day) and its
    own metrics never enter. Returns dict over query slots in the given
    order: mean, var, nh, np (counts in the state, all rows).
    hl_win, hl_met: optional recency half-lives (days) on the win and the
    metric evidence (variant 6a, forgetting = a random walk on skill);
    counts for the play preference are never decayed.
    q_win, q_met: state-space drift (variance per day; win in residual units,
    metrics in z units). Each player x hero cell's own evidence is an
    information filter for a cell level that drifts as a random walk: between
    the cell's updates, and up to the query day, its precision P becomes
    P / (1 + q P dt) with the mean S / P unchanged (exact for the local-level
    model with a flat prior on the cell's own evidence). Metric cells use
    precision C / eps2_m."""
    day, hero = aug["day"], aug["hero"]
    d0 = int(day.min())
    # state only for players that have a query (rows of other players never
    # reach a query); pid is the compact index from here on
    slot = aug["slot"]
    inv0 = np.full(int(slot.max()) + 1, -1, np.int64)
    inv0[slot[slot >= 0]] = np.flatnonzero(slot >= 0)
    qplayers = np.unique(aug["pid"][inv0[query_slots]])
    cmap = np.full(int(aug["n_players"]), -1, np.int64)
    cmap[qplayers] = np.arange(len(qplayers))
    pid = cmap[aug["pid"]]
    act = pid >= 0
    lw = 0.0 if hl_win is None else np.log(2) / hl_win
    lm = 0.0 if hl_met is None else np.log(2) / hl_met
    npl = len(qplayers)
    use_m = par.G is not None and metrics is not None
    Mn = par.G.shape[0] if use_m else 0
    Sst = np.zeros((npl, H))
    Pst = np.zeros((npl, H))
    Nst = np.zeros((npl, H), np.float32)
    kal = (q_win or 0) > 0 or ((q_met or 0) > 0 and use_m)
    if kal:
        Lst = np.full((npl, H), -1, np.int32)
        qw = float(q_win or 0.0)
        qm = float(q_met or 0.0)
        eps2 = np.asarray(par.eps2, np.float32) if use_m and par.eps2 is not None else None
    if use_m:
        mdt = np.float32
        Xst = np.zeros((npl, H, Mn), mdt)
        Cst = np.zeros((npl, H, Mn), mdt)
    # query rows in the aug order
    inv = np.full(int(slot.max()) + 1, -1, np.int64)
    inv[slot[slot >= 0]] = np.flatnonzero(slot >= 0)
    qa = inv[query_slots]
    assert (qa >= 0).all()
    oq = np.argsort(qa, kind="stable")
    qa_s = qa[oq]
    nq = len(qa)
    mean = np.empty(nq)
    var = np.empty(nq)
    Nh = np.empty(nq, np.float32)
    Np = np.empty(nq, np.float32)
    if want_all:
        mall = np.empty((nq, H), np.float32)
        vall = np.empty((nq, H), np.float32)
        nall = np.empty((nq, H), np.float32)
    if metric_feats and use_m:
        # direct metric features of the query: the player's mean z on the
        # query hero and over all heroes (shrunk with 5 pseudo-games), as-of;
        # stored only for queries in feat_mask (default all), in query order
        fm_q = np.ones(nq, bool) if feat_mask is None else np.asarray(feat_mask, bool)
        fpos = np.full(nq, -1, np.int64)
        fpos[np.flatnonzero(fm_q)] = np.arange(int(fm_q.sum()))
        fpos_s = fpos[oq]                                   # in the walk's sorted order
        mfeat = np.empty((int(fm_q.sum()), 2 * Mn), np.float32)
    if parts:
        # the prior for theta at the query hero from everything except that
        # hero's own win evidence (m0, v0), and the own evidence (Sh, Ph)
        m0 = np.empty(nq)
        v0 = np.empty(nq)
        Sh = np.empty(nq)
        Ph = np.empty(nq)
    qd = day[qa_s]
    ud, qs = np.unique(qd, return_index=True)
    qe = np.r_[qs[1:], nq]
    ap = 0
    t0 = time.time()
    for t, a_, b_ in zip(ud, qs, qe):
        e = np.searchsorted(day, t - lag, side="right")
        if e > ap and kal:
            # state-space: inflate each touched cell to the day of its new
            # rows, then add them, one day at a time
            rr_all = np.arange(ap, e)
            rr_all = rr_all[act[rr_all]]
            dd_all = day[rr_all]
            ub, bs = np.unique(dd_all, return_index=True)
            be = np.r_[bs[1:], len(rr_all)]
            for dd, s_, e_ in zip(ub, bs, be):
                rr = rr_all[s_:e_]
                pi, hi = pid[rr], hero[rr]
                cell = np.unique(pi.astype(np.int64) * H + hi)
                cp_, ch_ = cell // H, cell % H
                dt = np.where(Lst[cp_, ch_] >= 0, dd - Lst[cp_, ch_], 0).astype(np.float64)
                if qw > 0:
                    f = 1.0 / (1.0 + qw * Pst[cp_, ch_] * dt)
                    Sst[cp_, ch_] *= f
                    Pst[cp_, ch_] *= f
                if use_m and qm > 0:
                    fm_ = (1.0 / (1.0 + qm * (Cst[cp_, ch_] / eps2[None, :]) * dt[:, None])).astype(np.float32)
                    Xst[cp_, ch_] *= fm_
                    Cst[cp_, ch_] *= fm_
                Lst[cp_, ch_] = dd
                np.add.at(Sst, (pi, hi), aug["r_adj"][rr] / aug["v"][rr])
                np.add.at(Pst, (pi, hi), 1.0 / aug["v"][rr])
                np.add.at(Nst, (pi, hi), 1.0)
                if use_m:
                    Xm = metrics[rr].astype(np.float32)
                    g = ~np.isnan(Xm)
                    wr = np.ones(len(rr), np.float32) if mweights is None else mweights[rr].astype(np.float32)
                    np.add.at(Xst, (pi, hi), np.where(g, Xm, 0) * wr[:, None])
                    np.add.at(Cst, (pi, hi), g.astype(np.float32) * wr[:, None])
            ap = e
        elif e > ap:
            rr_all = np.arange(ap, e)
            rr_all = rr_all[act[rr_all]]
            for c0 in range(0, len(rr_all), 1_000_000):     # chunks bound the temporaries
                rr = rr_all[c0:c0 + 1_000_000]
                pi, hi = pid[rr], hero[rr]
                scw = np.exp(lw * (day[rr] - d0))
                np.add.at(Sst, (pi, hi), scw * aug["r_adj"][rr] / aug["v"][rr])
                np.add.at(Pst, (pi, hi), scw / aug["v"][rr])
                np.add.at(Nst, (pi, hi), 1.0)
                if use_m:
                    # metrics are indexed by aug row (every row with a scoreboard)
                    Xm = metrics[rr].astype(np.float32)
                    g = ~np.isnan(Xm)
                    wr = np.ones(len(rr), np.float32) if mweights is None else mweights[rr].astype(np.float32)
                    wr = (wr * np.exp(lm * (day[rr] - d0))).astype(mdt)
                    np.add.at(Xst, (pi, hi), np.where(g, Xm, 0) * wr[:, None])
                    np.add.at(Cst, (pi, hi), g.astype(mdt) * wr[:, None])
            ap = e
        q = qa_s[a_:b_]
        pq, hq = pid[q], hero[q]
        kw = {}
        if use_z and par.W is not None:
            kw["N"] = Nst[pq].astype(np.float64)
        if kal:
            dtq = np.where(Lst[pq] >= 0, t - Lst[pq], 0).astype(np.float64)
            fw_ = 1.0 / (1.0 + qw * Pst[pq] * dtq)
            Sq, Pq = Sst[pq] * fw_, Pst[pq] * fw_
            if use_m:
                Cq = Cst[pq]
                fm_ = 1.0 / (np.float32(1.0) + np.float32(qm) * (Cq / eps2[None, None, :])
                             * dtq.astype(np.float32)[:, :, None])
                kw["X"], kw["Cm"] = Xst[pq] * fm_, Cq * fm_
        else:
            if use_m:
                fm = np.float32(np.exp(-lm * (t - d0)))
                kw["X"], kw["Cm"] = Xst[pq] * fm, Cst[pq] * fm
            fw = np.exp(-lw * (t - d0))
            Sq, Pq = Sst[pq] * fw, Pst[pq] * fw
        mean[a_:b_], var[a_:b_] = posterior(par, Sq, Pq, hq, **kw)
        if want_all:
            ma, va = posterior(par, Sq, Pq, hq, want_all=True, **kw)
            mall[a_:b_], vall[a_:b_], nall[a_:b_] = ma, va, Nst[pq]
        if parts:
            ar = np.arange(len(q))
            Sh[a_:b_], Ph[a_:b_] = Sq[ar, hq], Pq[ar, hq]
            Sq[ar, hq] = 0.0
            Pq[ar, hq] = 0.0
            m0[a_:b_], v0[a_:b_] = posterior(par, Sq, Pq, hq, **kw)
        if metric_feats and use_m:
            fp = fpos_s[a_:b_]
            sel = fp >= 0
            if sel.any():
                ar_ = np.flatnonzero(sel)
                Xq, Cq_ = kw["X"], kw["Cm"]
                mfeat[fp[sel], :Mn] = Xq[ar_, hq[ar_]] / (Cq_[ar_, hq[ar_]] + 5.0)
                mfeat[fp[sel], Mn:] = Xq[ar_].sum(1) / (Cq_[ar_].sum(1) + 5.0)
        Nh[a_:b_] = Nst[pq, hq]
        Np[a_:b_] = Nst[pq].sum(1)
    back = np.empty(nq, np.int64)
    back[oq] = np.arange(nq)
    print(f"  online walk: {nq:,} queries, {time.time() - t0:.0f}s", flush=True)
    out = {"mean": mean[back], "var": var[back], "nh": Nh[back], "np": Np[back]}
    if parts:
        out.update(m0=m0[back], v0=v0[back], Sh=Sh[back], Ph=Ph[back])
    if want_all:
        out.update(mean_all=mall[back], var_all=vall[back], n_all=nall[back])
    if metric_feats and use_m:
        out["mfeat"] = mfeat                                 # rows = queries in feat_mask, in query order
    return out


def get_params(tag):
    if tag.startswith("today"):
        drop = ("role", "fine", "melee") if tag == "today_norole" else ()
        return Params.from_kernel_file(KERN, drop=drop)
    return Params.load(os.path.join(RCF_RES, "params", f"{tag}.npz"))


def gap_days(d, aug, rows):
    """Days since the player's last game on the hero on an EARLIER day (any
    history row), for slot rows; -1 if none."""
    from p3_heroes import HKEY
    sub = np.flatnonzero(np.isin(aug["pid"], np.unique(d["pid"][rows])))
    kk = aug["pid"][sub].astype(np.int64) * HKEY + aug["hero"][sub]
    oo = np.lexsort((aug["day"][sub], kk))
    ks, ds = kk[oo], aug["day"][sub][oo].astype(np.int64)
    ar_ = np.arange(len(oo))
    newg = np.r_[True, ks[1:] != ks[:-1]]
    gst = np.maximum.accumulate(np.where(newg, ar_, 0))
    bst = np.maximum.accumulate(np.where(newg | np.r_[True, ds[1:] != ds[:-1]], ar_, 0))
    prev = bst - 1
    last = np.where(prev >= gst, ds[np.maximum(prev, 0)], -1)
    last_aug = np.empty(len(oo), np.int64)
    last_aug[oo] = last
    inv_slot = np.full(len(d["pid"]), -1, np.int64)
    ss = aug["slot"][sub]
    inv_slot[ss[ss >= 0]] = np.flatnonzero(ss >= 0)
    lq = last_aug[inv_slot[rows]]
    return np.where(lq >= 0, d["day"][rows] - lq, -1)


def online(a):
    """One arm on V1, V2 and OOT slots (as p3_c_tests.predict): combiner fit on V1."""
    import p3_x_common as X
    import p3_c_tests as T
    import p3_c_merged as M
    t0 = time.time()
    par = get_params(a.tag)
    d, aug = load_aug(a.hist)
    w, med = T.windows(d)
    qi = np.flatnonzero(w > 0)
    layer = getattr(a, "layer", False)
    qwalk = np.arange(len(d["pid"])) if layer else qi
    tail = getattr(a, "tail", None)
    ppath = getattr(a, "parts", None)
    if ppath and os.path.exists(ppath):
        z = np.load(ppath)
        assert np.array_equal(z["qwalk"], qwalk)
        res = {k: z[k] for k in z.files if k != "qwalk"}
        print(f"  loaded walk parts {ppath}", flush=True)
    else:
        mets, mw = aug_metrics(aug, par.meta.get("sb", "legacy"), d) if par.G is not None else (None, None)
        fmask = (w[qwalk] > 0) if getattr(a, "statlin", False) else None
        kw_walk = dict(metrics=mets, mweights=mw, hl_win=getattr(a, "hl_win", None),
                       hl_met=getattr(a, "hl_met", None), parts=bool(ppath or tail),
                       q_win=getattr(a, "q_win", None), q_met=getattr(a, "q_met", None),
                       metric_feats=bool(getattr(a, "statlin", False)))
        nsh = int(os.environ.get("P3_RCF_SHARDS", "1"))
        if nsh <= 1:
            res = online_walk(aug, qwalk, par, feat_mask=fmask, **kw_walk)
        else:
            # players are independent in the walk: shard them to bound the state's memory (exact)
            res = {}
            fidx = None if fmask is None else np.cumsum(fmask) - 1
            for s_ in range(nsh):
                sel = np.flatnonzero(d["pid"][qwalk] % nsh == s_)
                r_ = online_walk(aug, qwalk[sel], par, feat_mask=None if fmask is None else fmask[sel], **kw_walk)
                for k, v in r_.items():
                    if k == "mfeat":
                        if k not in res:
                            res[k] = np.empty((int(fmask.sum()), v.shape[1]), v.dtype)
                        res[k][fidx[sel][fmask[sel]]] = v
                    else:
                        if k not in res:
                            res[k] = np.empty((len(qwalk),) + v.shape[1:], v.dtype)
                        res[k][sel] = v
                del r_
                import gc
                gc.collect()
        if ppath:
            np.savez(ppath, qwalk=qwalk, **res)
        del mets, mw
        import gc
        gc.collect()
    table = aug["table"]
    # experience offset from slot-row lag-1 counts (A2wS)
    from p3_heroes import HKEY
    np_s = M.lag1_key_sums(d["day"], d["pid"].astype(np.int64)).astype(np.int64)
    nph_s = M.lag1_key_sums(d["day"], d["pid"].astype(np.int64) * HKEY + d["hero"]).astype(np.int64)
    mu_all = table[C.exp_bins(np_s[qwalk], nph_s[qwalk])]
    tail_meta = None
    if tail and tail != "gauss_ref":
        import p3_rcf_tails as TL
        res["gap"] = gap_days(d, aug, qwalk)
        res["mu"] = mu_all
        res["mean"], res["var"], tail_meta = TL.apply(tail, res, d["r"][qwalk] - mu_all, d["v"][qwalk] if "v" in d else
                                                      d["wp"][qwalk] * (1 - d["wp"][qwalk]),
                                                      w[qwalk] == 1, getattr(a, "tail_args", None))
    pred_all = mu_all + res["mean"]
    n_games, y, wp0, cnt, gday = X.game_arrays(dict(d, wp=d["wp_pop"]))
    lo = np.log(wp0 / (1 - wp0))
    gw = np.zeros(n_games, np.int8)
    gw[d["g"][qi]] = w[qi]
    full = cnt == 10
    fitg = (gw == 1) & full
    lay_meta = None
    if layer:
        if ppath:
            os.environ.setdefault("P3_RCF_LAYER_SAVE", ppath.replace(".npz", "_layer.npz"))
        pred, lay_meta, extra_cols = temporal_layer(d, pred_all, res["nh"], qi, gw * full, fitg, lo, y, n_games)
        res = {k: (v if k == "mfeat" else v[qi]) for k, v in res.items()}
    else:
        pred, extra_cols = pred_all, []
    mu = mu_all if not layer else mu_all[qi]
    stat_meta = None
    if getattr(a, "statlin", False):
        # direct linear metric channel: ridge of the remaining residual on the
        # 30 metric features, fit on V1 slots (weights 1 / v), applied to all
        F = res["mfeat"]                                     # already restricted to qi (w > 0)
        F = np.nan_to_num(F.astype(np.float64))
        wv = (d["wp_pop"] * (1 - d["wp_pop"]))[qi]
        tgt = d["r"][qi] - pred
        v1 = w[qi] == 1
        mF, sF = F[v1].mean(0), F[v1].std(0) + 1e-9
        Fz = (F - mF) / sF
        A = (Fz[v1].T / wv[v1]) @ Fz[v1]
        best = None
        for lam in (1e2, 1e3, 1e4, 1e5, 1e6, 1e7):
            # choose the ridge on V1 by 2-fold split of V1 players
            fold = (d["pid"][qi][v1] % 2) == 0
            err = 0.0
            for f_ in (fold, ~fold):
                Ff, tf, wf = Fz[v1][f_], tgt[v1][f_], wv[v1][f_]
                b_ = np.linalg.solve((Ff.T / wf) @ Ff + lam * np.eye(F.shape[1]), (Ff.T / wf) @ tf)
                Fo, to, wo = Fz[v1][~f_], tgt[v1][~f_], wv[v1][~f_]
                err += float(np.sum((to - Fo @ b_) ** 2 / wo))
            if best is None or err < best[0]:
                best = (err, lam)
        lam = best[1]
        beta = np.linalg.solve(A + lam * np.eye(F.shape[1]), (Fz[v1].T / wv[v1]) @ tgt[v1])
        pred = pred + Fz @ beta
        stat_meta = {"ridge": lam, "beta": beta.tolist(), "mean": mF.tolist(), "sd": sF.tolist(),
                     "features": "per metric: X_hero / (C_hero + 5), then X_all / (C_all + 5), as-of walk state"}
        del F, Fz, A
        res.pop("mfeat", None)
        import gc
        gc.collect()
        print(f"  direct metric channel: ridge {lam:g}, |beta| max {np.abs(beta).max():.4f}", flush=True)
    gap = gap_days(d, aug, qi)
    post_meta = None
    if getattr(a, "postcal", False):
        # smooth calibration of the whole slot prediction: r ~ sum_k c_k pred B_k(z),
        # B = (1, l, l^2, sh, sh^2, g, g^2, l g, l sh) of games on the hero,
        # hero share and days since the last game on it; WLS on V1 slots
        import p3_rcf_tails as TL
        B = TL.shrink_basis({"nh": res["nh"], "np": res["np"], "gap": gap},
                            rich=2 if os.environ.get("P3_RCF_POSTCAL") == "2" else True)
        Xp = B * pred[:, None]
        v1 = w[qi] == 1
        wv = 1.0 / (d["wp_pop"] * (1 - d["wp_pop"]))[qi][v1]
        cpar = np.linalg.solve((Xp[v1].T * wv) @ Xp[v1] + 1e-9 * np.eye(B.shape[1]), (Xp[v1].T * wv) @ d["r"][qi][v1])
        mult = B @ cpar
        pred = pred * mult
        post_meta = {"coef": cpar.tolist(), "multiplier_quantiles": np.percentile(mult, [5, 25, 50, 75, 95]).tolist(),
                     "basis": "p3_rcf_tails.shrink_basis(rich=%s)" % (2 if os.environ.get("P3_RCF_POSTCAL") == "2" else True)}
        print(f"  post-calibration multiplier, 5/50/95%: {np.round(np.percentile(mult, [5, 50, 95]), 3).tolist()}", flush=True)
    pr = np.zeros(len(d["pid"]))
    pr[qi] = pred
    D = X.team_diff(pr, d["g"], d["team"], n_games)
    cols = [lo, D]
    if "mmr_adj" in d:
        # the MMR part of the expectation is a game-level term with its own
        # coefficient (summing a team-level shift over five slots would count it five times)
        adj = np.zeros(len(d["pid"]))
        adj[qi] = d["mmr_adj"][qi]
        cols.append(X.team_diff(adj, d["g"], d["team"], n_games))
    Xc = np.column_stack(cols)
    wc = C.fit_logistic(Xc[fitg], y[fitg])
    w0 = C.fit_logistic(lo[fitg][:, None], y[fitg])
    p = C.predict(wc, Xc)
    p0 = C.predict(w0, lo[:, None])
    gsel = (gw > 0) & full
    grid = np.zeros(n_games, np.int64)
    grid[d["g"]] = d["replay_id"]
    pulled_pid = np.load(ROWS)["pulled_pid"]
    pulled = np.isin(d["pid"], pulled_pid)
    n_pull = np.bincount(d["g"], weights=pulled, minlength=n_games)
    rec_slot = M.lag1_key_sums(d["day"], d["pid"].astype(np.int64) * HKEY + d["hero"], d["r"])[qi]
    # record to date over every history row (slot rows: y - WP; HP rows: y - proxy), lag 1 day
    # (rows of players with a query only, to bound memory)
    if os.environ.get("P3_RCF_NO_REC") == "1":
        rec_full = np.full(len(qi), np.nan)
    else:
        sub = np.flatnonzero(np.isin(aug["pid"], np.unique(d["pid"][qi])))
        rec_sub = M.lag1_key_sums(aug["day"][sub], aug["pid"][sub].astype(np.int64) * HKEY + aug["hero"][sub],
                                  aug["r"][sub])
        inv_slot = np.full(len(d["pid"]), -1, np.int64)
        ss = aug["slot"][sub]
        inv_slot[ss[ss >= 0]] = np.flatnonzero(ss >= 0)
        rec_full = rec_sub[inv_slot[qi]]
        del rec_sub, sub
    out = {"s_replay_id": d["replay_id"][qi], "s_key": d["player_keys"][d["pid"][qi]], "s_pid": d["pid"][qi],
           "s_team": d["team"][qi], "s_hero": d["hero"][qi].astype(np.int16), "s_day": d["day"][qi],
           "s_win": w[qi], "s_r": d["r"][qi].astype(np.float32), "s_wp": d["wp_pop"][qi].astype(np.float32),
           "s_pred": pred.astype(np.float32), "s_mean": res["mean"].astype(np.float32),
           "s_var": res["var"].astype(np.float32), "s_mu": mu.astype(np.float32),
           "s_nh": res["nh"], "s_np": res["np"], "s_nh_slot": nph_s[qi].astype(np.float32),
           "s_np_slot": np_s[qi].astype(np.float32), "s_rec_slot": rec_slot.astype(np.float32),
           "s_pulled": pulled[qi], "s_rec": rec_full.astype(np.float32), "s_gap": gap.astype(np.int32),
           "s_kdiag": np.diag(par.kernel())[d["hero"][qi]].astype(np.float32),
           "kernel_hero_sd_pp": 100 * np.sqrt(np.diag(par.kernel())),
           "g_replay_id": grid[gsel], "g_win": gw[gsel], "g_y": y[gsel], "g_lo": lo[gsel], "g_D": D[gsel],
           "g_p": p[gsel], "g_p0": p0[gsel], "g_day": gday[gsel], "g_all10_pulled": (n_pull[gsel] == 10),
           "coef": wc, "coef0": w0}
    ll = X.logloss(p, y)
    ll0 = X.logloss(p0, y)
    summ = {nm: {"games": int(((gw == k) & full).sum()),
                 "gain": float((ll0 - ll)[(gw == k) & full].mean())} for k, nm in T.WIN.items()}
    name = getattr(a, "out", None) or a.tag
    print(f"[online {name} {a.hist}] {json.dumps(summ)} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.join(RCF_RES, "p3c"), exist_ok=True)
    np.savez(os.path.join(RCF_RES, "p3c", f"{name}__{a.hist}.npz"), **out,
             meta=np.array(json.dumps({"tag": a.tag, "hist": a.hist, "summary": summ, "split_day": med,
                                       "hl_win": getattr(a, "hl_win", None), "hl_met": getattr(a, "hl_met", None),
                                       "q_win": getattr(a, "q_win", None), "q_met": getattr(a, "q_met", None),
                                       "layer": lay_meta, "tail": tail_meta, "statlin": stat_meta,
                                       "postcal": post_meta})))


def layer_stats(d, u, early, hl_d):
    """As-of per-player residual statistics for every slot row (games on
    EARLIER days only): exponentially decayed sum and weight of u (half-life
    hl_d days), and the plain sum and count of u over the player's early
    games on a hero (early = fewer than 5 earlier games on it)."""
    import p3_c_merged as M
    day = d["day"].astype(np.int64)
    pid = d["pid"].astype(np.int64)
    lam = np.log(2) / hl_d
    d0 = int(day.min())
    sc = np.exp(lam * (day - d0))
    dec = np.exp(-lam * (day - d0))
    Sd = M.lag1_key_sums(day, pid, sc * u) * dec
    Nd = M.lag1_key_sums(day, pid, sc) * dec
    Se = M.lag1_key_sums(day, pid, np.where(early, u, 0.0))
    Ne = M.lag1_key_sums(day, pid, early.astype(np.float64))
    return Sd, Nd, Se, Ne


def temporal_layer(d, pred_all, nh_all, qi, gw, fitg, lo, y, n_games):
    """Variant 6b/6c: per-player drift (decayed mean of the player's recent
    residuals against the model, u = r - pred, shrunk) and a per-player
    new-hero learning term (mean u over the player's early games on heroes,
    shrunk, applied when the query hero has fewer than 5 earlier games).
    Each u comes from the model's own as-of prediction; the statistics read
    games on earlier days only. Half-life and shrinkage are chosen on V1
    games by the log loss of the combiner [logit WP, D, D_drift, D_learn];
    the per-slot prediction adds the two terms scaled by their coefficient
    relative to D."""
    import p3_x_common as X
    u = d["r"] - pred_all
    early = nh_all < 5
    lp = os.environ.get("P3_RCF_LAYER_SAVE")
    if lp:
        # the layer's inputs for every slot (state writers rebuild the per-player terms from them)
        np.savez(lp, u=u.astype(np.float32), early=early)
    best = None
    qm = np.zeros(len(u), bool)
    qm[qi] = True
    td = lambda v: X.team_diff(np.where(qm, v, 0.0), d["g"], d["team"], n_games)
    D0 = td(pred_all)
    for hl_d in (45, 180, 365, 730):
        Sd, Nd, Se, Ne = layer_stats(d, u, early, hl_d)
        for kd in (10, 40, 160):
            rho = Sd / (Nd + kd)
            for ke in (1, 2, 5, 20):
                nu = np.where(early, Se / (Ne + ke), 0.0)
                Xg = np.column_stack([lo, D0, td(rho), td(nu)])
                wc = C.fit_logistic(Xg[fitg], y[fitg])
                ll = float(X.logloss(C.predict(wc, Xg[fitg]), y[fitg]).mean())
                if best is None or ll < best[0]:
                    best = (ll, hl_d, kd, ke, wc, rho, nu)
                print(f"    layer hl {hl_d} kd {kd} ke {ke}: V1 ll {ll:.6f} coef {np.round(wc, 3).tolist()}", flush=True)
    ll, hl_d, kd, ke, wc, rho, nu = best
    # ablations at the chosen hyperparameters (combiner refit on V1 each)
    abl = {}
    gwin = gw
    for nm, cols in (("drift only", [D0, td(rho)]), ("learning only", [D0, td(nu)]), ("both", [D0, td(rho), td(nu)])):
        Xg = np.column_stack([lo] + cols)
        wa = C.fit_logistic(Xg[fitg], y[fitg])
        pa = C.predict(wa, Xg)
        w0 = C.fit_logistic(np.column_stack([lo, D0])[fitg], y[fitg])
        p0 = C.predict(w0, np.column_stack([lo, D0]))
        abl[nm] = {}
        for k, wn in ((1, "V1"), (2, "V2"), (3, "OOT")):
            mm = (gwin == k) & np.isfinite(lo)
            dd = X.logloss(p0[mm], y[mm]) - X.logloss(pa[mm], y[mm])
            abl[nm][wn] = [1000 * float(dd.mean()), 1000 * float(dd.std(ddof=1) / np.sqrt(mm.sum()))]
    print(f"  layer ablations (gain over the same model without the layer, x1000, SE): {abl}", flush=True)
    gd, gr, gn = wc[2], wc[3], wc[4]
    pred = pred_all[qi] + (gr / gd) * rho[qi] + (gn / gd) * nu[qi]
    meta = {"hl_d": hl_d, "kd": kd, "ke": ke, "coef": wc.tolist(), "v1_ll": ll, "ablations": abl,
            "drift_scale": float(gr / gd), "learn_scale": float(gn / gd)}
    print(f"  layer chosen on V1: {meta}", flush=True)
    return pred, meta, [rho[qi], nu[qi]]


# ------------------------------------------------------------------ cassia

def cassia(a):
    """Masked cold start (P3_HERO_STRENGTH section 4) on the static E state
    (plus, for merged, every HP row up to the end of E): held-out players with
    300+ E slot games, target cells with 10-60 E slot games on the hero and at
    most 5% of their E games. All of the target cell's rows (slot and HP),
    its counts and its metrics are hidden."""
    from p3_hs_cassia import score
    par = get_params(a.tag)
    d, aug = load_aug(a.hist)
    npl = int(d["n_players"])
    e_last = int(d["day"][d["in_sample"]].max())
    slot = aug["slot"]
    is_slot = slot >= 0
    ins = np.zeros(len(slot), bool)
    ins[is_slot] = d["in_sample"][slot[is_slot]]
    day, pid, hero = aug["day"], aug["pid"], aug["hero"]
    e_slot = is_slot & ins & (day >= C.day_of(C.E_START))
    state = (day <= e_last)
    if a.hist == "window":
        state &= e_slot
    cnt = np.bincount(pid[e_slot] * H + hero[e_slot], minlength=npl * H).reshape(npl, H)
    tot = cnt.sum(1)
    hold = ~fit_half(npl)
    deep = hold & (tot >= 300)
    pi, hi = np.nonzero((cnt >= 10) & (cnt <= 60) & (cnt <= 0.05 * tot[:, None]) & deep[:, None])
    up = np.unique(pi)
    pos = np.full(npl, -1, np.int64)
    pos[up] = np.arange(len(up))
    sel = state & (pos[pid] >= 0)
    idx = pos[pid[sel]] * H + hero[sel]
    S = np.bincount(idx, weights=aug["r_adj"][sel] / aug["v"][sel], minlength=len(up) * H).reshape(-1, H)
    P = np.bincount(idx, weights=1 / aug["v"][sel], minlength=len(up) * H).reshape(-1, H)
    N = np.bincount(idx, minlength=len(up) * H).reshape(-1, H).astype(np.float64)
    kw = {}
    tg = np.arange(len(pi))
    rows_p = pos[pi]
    Sm, Pm, Nm = S[rows_p].copy(), P[rows_p].copy(), N[rows_p].copy()
    Sm[tg, hi] = 0
    Pm[tg, hi] = 0
    Nm[tg, hi] = 0
    if par.W is not None:
        kw["N"] = Nm
    if par.G is not None:
        Zaug, Waug = aug_metrics(aug, par.meta.get("sb", "legacy"), d)
        Xs, Cs = metric_sums(aug, Zaug, Waug, sel, pos, len(up))
        Xs, Cs = Xs[rows_p], Cs[rows_p]
        Xs[tg, hi] = 0
        Cs[tg, hi] = 0
        kw["X"], kw["Cm"] = Xs, Cs
    m, s2 = [], []
    for i0 in range(0, len(pi), 20000):
        sl = slice(i0, i0 + 20000)
        mm, vv = posterior(par, Sm[sl], Pm[sl], hi[sl], **{k: v[sl] for k, v in kw.items()})
        m.append(mm)
        s2.append(vv)
    m, s2 = np.concatenate(m), np.concatenate(s2)
    # targets: cell mean of r_adj over E slot rows
    tk = pi * H + hi
    key = pid * H + hero
    sel_t = e_slot & np.isin(key, tk)
    o = np.argsort(tk)
    ci = o[np.searchsorted(tk[o], key[sel_t])]
    ns = np.bincount(ci, minlength=len(tk))
    rbar = np.bincount(ci, weights=aug["r_adj"][sel_t], minlength=len(tk)) / ns
    noise = np.bincount(ci, weights=aug["v"][sel_t], minlength=len(tk)) / ns ** 2
    meta = C.hero_meta(d["hero_names"])
    roles = meta["blizz"][hi]
    out = {"tag": a.tag, "hist": a.hist, "cells": int(len(pi)), "players": int(len(up)),
           "all": score(rbar, noise, m, s2),
           "by_role": {C.BLIZZ_ROLES[r]: score(rbar[roles == r], noise[roles == r], m[roles == r], s2[roles == r])
                       for r in range(6) if (roles == r).sum() > 200}}
    print(f"[cassia {a.tag} {a.hist}] R2 {out['all']['latent_R2']:+.3f}; " + ", ".join(
        f"{k} {v['latent_R2']:+.2f}" for k, v in out["by_role"].items()), flush=True)
    os.makedirs(RCF_RES, exist_ok=True)
    np.savez(os.path.join(RCF_RES, f"cassia_{a.tag}__{a.hist}.npz"), pi=pi, hi=hi, m=m, s2=s2, rbar=rbar,
             noise=noise, roles=roles)
    with open(os.path.join(RCF_RES, f"cassia_{a.tag}__{a.hist}.json"), "w") as f:
        json.dump(out, f, indent=1)


# ------------------------------------------------------------------ coverage

def layer_shift_at(d, lay, as_of, cells):
    """The temporal layer's shift for each cell (pid * H + hero) as of a day:
    drift_scale * decayed mean of u + learn_scale * early-game mean of u (if
    the cell had fewer than 5 games), from slot games on days <= as_of.
    lay: dict with u (per slot, NaN where unknown), early (per slot), nh_cell
    (games on the hero at as_of per cell) and the arm's layer meta."""
    m = lay["meta"]
    day = d["day"].astype(np.int64)
    ok = (day <= as_of) & np.isfinite(lay["u"])
    pid = d["pid"].astype(np.int64)
    npl = int(d["n_players"])
    lam = np.log(2) / m["hl_d"]
    w = np.exp(-lam * (as_of + 1 - day[ok]))
    u = lay["u"][ok]
    Sd = np.bincount(pid[ok], weights=w * u, minlength=npl)
    Nd = np.bincount(pid[ok], weights=w, minlength=npl)
    e = lay["early"][ok]
    Se = np.bincount(pid[ok], weights=np.where(e, u, 0.0), minlength=npl)
    Ne = np.bincount(pid[ok], weights=e.astype(float), minlength=npl)
    cp = cells // H
    rho = Sd[cp] / (Nd[cp] + m["kd"])
    nu = Se[cp] / (Ne[cp] + m["ke"])
    return m["drift_scale"] * rho + m["learn_scale"] * nu * (lay["nh_cell"] < 5)


def cells_at(par, d, aug, as_of, target, min_games, Zaug=None, Waug=None):
    """Posterior as of day `as_of` (rows with day <= as_of) for every cell
    with min_games+ slot games in target (bool over aug rows), and the
    realized mean r_adj of the cell over those games with its noise."""
    day, pid, hero = aug["day"], aug["pid"], aug["hero"]
    npl = int(d["n_players"])
    key = pid * H + hero
    fk, fc = np.unique(key[target], return_counts=True)
    cells = fk[fc >= min_games]
    cp = cells // H
    up = np.unique(cp)
    pos = np.full(npl, -1, np.int64)
    pos[up] = np.arange(len(up))
    st = (day <= as_of) & (pos[pid] >= 0)
    idx = pos[pid[st]] * H + hero[st]
    S = np.bincount(idx, weights=aug["r_adj"][st] / aug["v"][st], minlength=len(up) * H).reshape(-1, H)
    P = np.bincount(idx, weights=1 / aug["v"][st], minlength=len(up) * H).reshape(-1, H)
    N = np.bincount(idx, minlength=len(up) * H).reshape(-1, H).astype(np.float64)
    kw = {}
    rows_p = pos[cp]
    hq = cells % H
    if par.W is not None:
        kw["N"] = N[rows_p]
    if par.G is not None:
        Xs, Cs = metric_sums(aug, Zaug, Waug, st, pos, len(up))
        kw["X"], kw["Cm"] = Xs[rows_p], Cs[rows_p]
    m, s2 = [], []
    for i0 in range(0, len(cells), 20000):
        sl = slice(i0, i0 + 20000)
        mm, vv = posterior(par, S[rows_p][sl], P[rows_p][sl], hq[sl], **{k: v[sl] for k, v in kw.items()})
        m.append(mm)
        s2.append(vv)
    m, s2 = np.concatenate(m), np.concatenate(s2)
    ci = np.searchsorted(cells, key[target])
    okc = (ci < len(cells)) & (cells[np.minimum(ci, len(cells) - 1)] == key[target])
    ci = ci[okc]
    nf = np.bincount(ci, minlength=len(cells))
    rbar = np.bincount(ci, weights=aug["r_adj"][target][okc], minlength=len(cells)) / nf
    noise = np.bincount(ci, weights=aug["v"][target][okc], minlength=len(cells)) / nf ** 2
    return m, s2, rbar, noise, N[rows_p, hq], cells


def fit_scale(m, s2, rbar, noise):
    """Display variance c s2 + k0 by maximum likelihood of rbar - m ~
    N(0, c s2 + k0 + noise) (P3_HERO_STRENGTH section 5)."""
    from scipy.optimize import minimize
    e2 = (rbar - m) ** 2

    def nll(x):
        v = np.exp(x[0]) * s2 + np.exp(x[1]) + noise
        return 0.5 * np.sum(np.log(v) + e2 / v)
    r = minimize(nll, [0.0, np.log(1e-6)], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    return float(np.exp(r.x[0])), float(np.exp(r.x[1]))


def cover(a):
    """Interval coverage (P3_HERO_STRENGTH section 5). Display scale fit on
    V1: the posterior as of the end of E vs cells with 5+ V1 games, display
    variance c s2 + k0. Test on V2: the posterior as of the V1/V2 split vs
    cells with 10+ V2 games; 80% interval m +- 1.28 sqrt(display + noise / n),
    raw (c = 1, k0 = 0) and scaled."""
    import p3_c_tests as T
    par = get_params(a.tag)
    d, aug = load_aug(a.hist)
    w, med = T.windows(d)
    slot = aug["slot"]
    sw = np.zeros(len(slot), np.int8)
    sw[slot >= 0] = w[slot[slot >= 0]]
    Zaug, Waug = aug_metrics(aug, par.meta.get("sb", "legacy"), d) if par.G is not None else (None, None)
    e_last = int(d["day"][d["in_sample"]].max())
    m1, v1, r1, n1, nat1, cells1 = cells_at(par, d, aug, e_last, sw == 1, 5, Zaug, Waug)
    m, s2, rbar, noise, n_at, cells2 = cells_at(par, d, aug, med - 1, sw == 2, 10, Zaug, Waug)
    lay_arm = getattr(a, "layer_arm", None)
    if lay_arm:
        # the temporal layer of an arm (online --layer --parts): its shift
        # as of each date is added to the cell mean before fitting the scale
        za = np.load(os.path.join(RCF_RES, "p3c", f"{lay_arm}__{a.hist}.npz"))
        meta = json.loads(str(za["meta"]))["layer"]
        zp = np.load(a.parts)
        assert len(zp["qwalk"]) == len(d["pid"])
        from p3_heroes import HKEY
        import p3_c_merged as M
        np_s = M.lag1_key_sums(d["day"], d["pid"].astype(np.int64)).astype(np.int64)
        nph_s = M.lag1_key_sums(d["day"], d["pid"].astype(np.int64) * HKEY + d["hero"]).astype(np.int64)
        mu_all = aug["table"][C.exp_bins(np_s, nph_s)]
        u = d["r"] - (mu_all + zp["mean"])
        lay = {"u": u, "early": zp["nh"] < 5, "meta": meta}
        m1 = m1 + layer_shift_at(d, dict(lay, nh_cell=nat1), e_last, cells1)
        m = m + layer_shift_at(d, dict(lay, nh_cell=n_at), med - 1, cells2)
    c, k0 = fit_scale(m1, v1, r1, n1)
    z = np.abs(rbar - m) / np.sqrt(s2 + noise)
    zc = np.abs(rbar - m) / np.sqrt(c * s2 + k0 + noise)
    bins = [(0, 0), (1, 5), (6, 20), (21, 100), (101, 10 ** 9)]
    out = {"tag": a.tag, "hist": a.hist, "layer_arm": lay_arm, "cells": int(len(m)), "cover80": float((z < 1.2816).mean()),
           "v1_cells": int(len(m1)), "scale_c": c, "scale_k0_sd_pp": 100 * float(np.sqrt(k0)),
           "cover80_scaled": float((zc < 1.2816).mean()),
           "var_ratio": float((((rbar - m) ** 2).mean() - noise.mean()) / s2.mean()),
           "var_ratio_scaled": float((((rbar - m) ** 2).mean() - noise.mean()) / (c * s2 + k0).mean()),
           "display_sd_pp_median": float(100 * np.median(np.sqrt(c * s2 + k0))), "by_n": {}}
    for lo_, hi_ in bins:
        s = (n_at >= lo_) & (n_at <= hi_)
        out["by_n"][f"{lo_}-{hi_ if hi_ < 10 ** 9 else ''}"] = {
            "cells": int(s.sum()), "cover80": float((z[s] < 1.2816).mean()),
            "cover80_scaled": float((zc[s] < 1.2816).mean()),
            "display_sd_pp": float(100 * np.median(np.sqrt(c * s2[s] + k0))),
            "var_ratio": float((((rbar[s] - m[s]) ** 2).mean() - noise[s].mean()) / s2[s].mean())}
    print(f"[cover {a.tag} {a.hist}] V1 scale c {c:.2f} k0 sd {out['scale_k0_sd_pp']:.2f}pp; V2 cells {len(m):,} "
          f"cover80 raw {out['cover80']:.3f} scaled {out['cover80_scaled']:.3f}; var_ratio {out['var_ratio']:.2f} -> "
          f"{out['var_ratio_scaled']:.2f}; " + ", ".join(f"{k}: {v['cover80_scaled']:.2f}" for k, v in out["by_n"].items()),
          flush=True)
    with open(os.path.join(RCF_RES, f"cover_{lay_arm or a.tag}__{a.hist}.json"), "w") as f:
        json.dump(out, f, indent=1)


# ------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "fit", "heldout", "online", "cassia", "cover"])
    ap.add_argument("--hist", default="merged", choices=["window", "merged"])
    ap.add_argument("--tag", default="v")
    ap.add_argument("--kmax", type=int, default=2)
    ap.add_argument("--ard", action="store_true", help="(kept for old specs) same as --lam 1")
    ap.add_argument("--lam", type=float, default=0.0, help="group-lasso weight per factor (0: Gaussian prior)")
    ap.add_argument("--hl-win", type=float, help="online: recency half-life on the win evidence (days)")
    ap.add_argument("--hl-met", type=float, help="online: recency half-life on the metric evidence (days)")
    ap.add_argument("--q-win", type=float, help="online: state-space drift variance per day, win evidence")
    ap.add_argument("--q-met", type=float, help="online: state-space drift variance per day, metric evidence (z units)")
    ap.add_argument("--layer", action="store_true", help="online: per-player drift and new-hero learning layer")
    ap.add_argument("--out", help="online: arm name (default the tag)")
    ap.add_argument("--parts", help="online: cache file for the walk (with leave-own-hero-out parts)")
    ap.add_argument("--tail", help="online: tail card (p3_rcf_tails.py), fit on V1 slots")
    ap.add_argument("--postcal", action="store_true", help="online: smooth calibration of the slot prediction (V1)")
    ap.add_argument("--statlin", action="store_true", help="online: direct linear metric channel (ridge on V1)")
    ap.add_argument("--layer-arm", help="cover: add this online arm's temporal layer (needs --parts)")
    ap.add_argument("--tail-args", help="online: tail card options (json)")
    ap.add_argument("--sb", default="legacy", help="scoreboard card (p3_rcf_cards.py); legacy = sb_z.npz")
    ap.add_argument("--kit", action="store_true", help="kit-axis columns (scilearn kit_2.57.json), one scale each")
    ap.add_argument("--joint", action="store_true", help="add log p(conditioning set): generative metric factors")
    ap.add_argument("--wsd", type=float, default=0.3, help="prior sd of the implicit map W")
    ap.add_argument("--gshare", action="store_true", help="metric loadings shared across heroes (+ deviations)")
    ap.add_argument("--grole", action="store_true", help="metric loadings shared within Blizzard role")
    ap.add_argument("--gsd", type=float, default=0.0, help="prior sd of hero deviations of metric loadings")
    ap.add_argument("--val", action="store_true", help="inner validation split of the fit half")
    ap.add_argument("--role", action="store_true", help="hand-made role, fine-role and melee columns")
    ap.add_argument("--implicit", action="store_true")
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--hero-s2", action="store_true", help="per-hero hero-specific variance")
    ap.add_argument("--objective", default="predictive", choices=["predictive", "marginal"])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--f64", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--drop", default="")
    ap.add_argument("--flip-half", action="store_true",
                    help="fit on the phase-1 held-out half of players (the tournament judge's fit)")
    a = ap.parse_args()
    if a.ard and a.lam == 0:
        a.lam = 1.0
    {"prep": lambda: prep(a.hist, a.sb), "fit": lambda: fit(a), "heldout": lambda: heldout_fixed(a),
     "online": lambda: online(a), "cassia": lambda: cassia(a), "cover": lambda: cover(a)}[a.stage]()


if __name__ == "__main__":
    main()
