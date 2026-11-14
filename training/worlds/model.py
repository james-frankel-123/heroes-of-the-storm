"""
World model: an explicit Bayesian outcome model of ranked games, fit on the paper-1 training split.

logit P(team 0 wins) = s(T0) - s(T1) + sum_{a in T0, b in T1} C[a, b]  (+ higher-order layers)
  s(T) = sum_h (w_h + V[map, h] + U[tier, h])           hero, hero x map, hero x tier
       + sum_{a<b in T} S[a, b]                          pair synergy within a team
       + sum_k e3_k(E[T])                                low-rank (CP rank r) triple term:
                                                         sum over the team's 10 triples of sum_k E_ak E_bk E_ck
  C antisymmetric (C[a, b] = K_ab for a < b, -K_ba otherwise): pair counter across teams.
No intercept: the model is antisymmetric in the two teams, like the symmetrized WP models.

Priors (Gaussian, independent): w ~ N(0, 1), V, U ~ N(0, 0.25^2) (as overrating.py's baseline),
S ~ N(0, sS^2), K ~ N(0, sK^2), E ~ N(0, sE^2); (sS, sK, r, sE) chosen on the val split.
Posterior: Laplace at the MAP with the generalized Gauss-Newton precision
  Lambda = J' diag(p(1-p)) J + diag(prior precisions)
(J the Jacobian of the logit; exact for every linear block, Gauss-Newton for the CP block, whose
sign/permutation symmetry is irrelevant to a local Gaussian). Draws: theta = theta_hat + L^-T z.

Higher-order layers (each its own error family; world.py): independent cell effects tau_c added to a
team (or a matchup) whenever the cell occurs,
  triple    the 10 three-hero subsets of a team (residual beyond the CP term)
  quartet   the 5 four-hero subsets
  five      the team itself
  cross     (pair in one team, hero in the other): 50 cells per team
  hmt       hero x map x tier (beyond the additive V and U)
tau_c ~ N(0, s^2), fit conditional on the base MAP (offset) by MAP over the seen cells; s by the
diagonal-Laplace evidence (type-II ML) on the training games,
  log p(y | s) ~ loglik(tau_hat) - sum tau_hat^2 / 2s^2 - 1/2 sum log(1 + h_c s^2),
h_c = sum of p(1-p) over the rows touching c; posterior tau_c ~ N(tau_hat_c, 1 / (h_c + 1/s^2))
(mean-field: overlaps between a team's cells are ignored in the variance). Unseen cells: N(0, s^2).
"""
import os
import sys
import json
import time
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import data as WD  # noqa: E402

H = WD.H
IA, IB = np.triu_indices(5, 1)
NPAIR = H * (H - 1) // 2
PID = -np.ones((H, H), np.int64)
PID[np.triu_indices(H, 1)] = np.arange(NPAIR)
PID = np.maximum(PID, PID.T)
SD_W, SD_MT = 1.0, 0.25
THREADS = int(os.environ.get("WORLDS_THREADS", "8"))


def sig(z):
    return 1 / (1 + np.exp(-z))


def nll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ── base model ───────────────────────────────────────────────────────────
class Base:
    """Parameter vector layout and the logit, in numpy (any number of games, any parameter draw)."""

    def __init__(self, nm, nt, r):
        self.nm, self.nt, self.r = nm, nt, r       # map / tier rows include the 'unknown' row
        sizes = [("w", H), ("V", nm * H), ("U", nt * H), ("S", NPAIR), ("K", NPAIR), ("E", H * r)]
        self.off, o = {}, 0
        for k, n in sizes:
            self.off[k] = (o, o + n)
            o += n
        self.P = o

    def unpack(self, th):
        g = lambda k: th[self.off[k][0]:self.off[k][1]]
        return dict(w=g("w"), V=g("V").reshape(self.nm, H), U=g("U").reshape(self.nt, H), S=g("S"), K=g("K"),
                    E=g("E").reshape(H, self.r))

    def prior_prec(self, sS, sK, sE):
        d = np.empty(self.P)
        for k, s in (("w", SD_W), ("V", SD_MT), ("U", SD_MT), ("S", sS), ("K", sK), ("E", sE)):
            d[self.off[k][0]:self.off[k][1]] = 1 / s ** 2
        return d

    @staticmethod
    def e3(X):
        """X (..., 5, r) -> sum_k e3 of the 5 values: the CP triple term of a team."""
        p1, p2, p3 = X.sum(-2), (X ** 2).sum(-2), (X ** 3).sum(-2)
        return ((p1 ** 3 - 3 * p1 * p2 + 2 * p3) / 6).sum(-1)

    def parts(self, th, D):
        """-> dict of per-game pieces: main0/main1 (hero + map + tier), int0/int1 (synergy + CP triple),
        cross (counters), all as logits of the team's strength (cross: team 0's view)."""
        q = self.unpack(th)
        T0, T1, m, t = D["T0"], D["T1"], D["m"], D["t"]
        out = {}
        for s, T in (("0", T0), ("1", T1)):
            out["main" + s] = (q["w"][T] + q["V"][m[:, None], T] + q["U"][t[:, None], T]).sum(1)
            out["int" + s] = q["S"][PID[T[:, IA], T[:, IB]]].sum(1) + self.e3(q["E"][T])
        a, b = T0[:, :, None], T1[:, None, :]
        k = q["K"][PID[np.minimum(a, b), np.maximum(a, b)]]
        out["cross"] = np.where(a < b, k, -k).sum((1, 2))
        return out

    def logit(self, th, D):
        p = self.parts(th, D)
        return p["main0"] - p["main1"] + p["int0"] - p["int1"] + p["cross"]

    # Jacobian of the logit (sparse, one row per game), for the Gauss-Newton precision
    def jacobian(self, th, D):
        import scipy.sparse as sp
        q = self.unpack(th)
        T0, T1, m, t = D["T0"], D["T1"], D["m"], D["t"]
        n, r = len(T0), self.r
        cols, vals = [], []
        for T, sg in ((T0, 1.0), (T1, -1.0)):
            cols += [self.off["w"][0] + T, self.off["V"][0] + m[:, None] * H + T, self.off["U"][0] + t[:, None] * H + T,
                     self.off["S"][0] + PID[T[:, IA], T[:, IB]]]
            vals += [np.full((n, 5), sg)] * 3 + [np.full((n, 10), sg)]
            X = q["E"][T]                                         # (n, 5, r)
            p1, p2 = X.sum(1, keepdims=True), (X ** 2).sum(1, keepdims=True)
            d = ((p1 - X) ** 2 - (p2 - X ** 2)) / 2               # de3/dx = e2 of the other four
            cols.append((self.off["E"][0] + T[:, :, None] * r + np.arange(r)).reshape(n, -1))
            vals.append(sg * d.reshape(n, -1))
        a, b = T0[:, :, None], T1[:, None, :]
        cols.append((self.off["K"][0] + PID[np.minimum(a, b), np.maximum(a, b)]).reshape(n, 25))
        vals.append(np.where(a < b, 1.0, -1.0).reshape(n, 25))
        C = np.concatenate(cols, 1)
        V = np.concatenate(vals, 1)
        k = C.shape[1]
        return sp.csr_matrix((V.ravel(), C.ravel(), np.arange(n + 1) * k), shape=(n, self.P))


def fit_base(D, nm, nt, r, sS, sK, sE, init=None, iters=400, seed=0, verbose=True):
    """MAP of the base model (torch LBFGS, float64, CPU)."""
    import torch
    torch.set_num_threads(THREADS)
    B = Base(nm, nt, r)
    T0, T1 = torch.tensor(D["T0"]), torch.tensor(D["T1"])
    m, t = torch.tensor(D["m"])[:, None], torch.tensor(D["t"])[:, None]
    y = torch.tensor(D["y"])
    pid = torch.tensor(PID)
    ia, ib = torch.tensor(IA), torch.tensor(IB)
    rng = np.random.default_rng(seed)
    th0 = np.zeros(B.P)
    th0[B.off["E"][0]:B.off["E"][1]] = rng.normal(0, 0.5 * sE, H * r)   # the CP term has a saddle at 0
    if init is not None:
        th0[:len(init)] = init[:B.P] if len(init) >= B.P else np.pad(init, (0, B.P - len(init)))
    th = torch.tensor(th0, requires_grad=True)
    prec = torch.tensor(B.prior_prec(sS, sK, sE))
    a, b = T0[:, :, None], T1[:, None, :]
    kid = pid[torch.minimum(a, b), torch.maximum(a, b)]
    ksg = torch.where(a < b, 1.0, -1.0).double()
    o = B.off

    def logit(th):
        w = th[o["w"][0]:o["w"][1]]
        V = th[o["V"][0]:o["V"][1]].view(nm, H)
        U = th[o["U"][0]:o["U"][1]].view(nt, H)
        S, K = th[o["S"][0]:o["S"][1]], th[o["K"][0]:o["K"][1]]
        E = th[o["E"][0]:o["E"][1]].view(H, r)

        def s(T):
            X = E[T]
            p1, p2, p3 = X.sum(1), (X ** 2).sum(1), (X ** 3).sum(1)
            return ((w[T] + V[m, T] + U[t, T]).sum(1) + S[pid[T[:, ia], T[:, ib]]].sum(1)
                    + ((p1 ** 3 - 3 * p1 * p2 + 2 * p3) / 6).sum(1))
        return s(T0) - s(T1) + (K[kid] * ksg).sum((1, 2))

    opt = torch.optim.LBFGS([th], lr=1, max_iter=iters, tolerance_grad=1e-7, tolerance_change=1e-12,
                            history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit(th), y, reduction="sum") + 0.5 * (prec * th ** 2).sum()
        loss.backward()
        return loss
    t0 = time.time()
    opt.step(closure)
    out = th.detach().numpy().copy()
    if verbose:
        print(f"  base r={r} sS={sS} sK={sK} sE={sE}: {time.time() - t0:.0f}s, "
              f"train nll {nll(sig(B.logit(out, D)), D['y']):.5f}", flush=True)
    return B, out


def _ggn_chunk(args):
    path, nm, nt, r, th, lo, hi, step = args
    B = Base(nm, nt, r)
    D = WD.load("train")
    acc = np.zeros((B.P, B.P))
    for i in range(lo, hi, step):
        sl = slice(i, min(i + step, hi))
        Dc = {k: v[sl] for k, v in D.items()}
        p = sig(B.logit(th, Dc))
        J = B.jacobian(th, Dc)
        acc += (J.T @ J.multiply((p * (1 - p))[:, None]).tocsr()).toarray()
    np.save(path, acc)
    return path


def ggn_precision(B, th, n, workers=8, step=20000):
    """J' diag(p(1-p)) J over the training games, in parallel row blocks (dense P x P)."""
    from multiprocessing import get_context
    edges = np.linspace(0, n, workers + 1).astype(int)
    jobs = [(os.path.join(WD.CACHE, f"_ggn_{i}.npy"), B.nm, B.nt, B.r, th, edges[i], edges[i + 1], step)
            for i in range(workers)]
    with get_context("fork").Pool(workers) as pool:
        paths = pool.map(_ggn_chunk, jobs)
    Lam = np.zeros((B.P, B.P))
    for p in paths:
        Lam += np.load(p)
        os.remove(p)
    return Lam


# ── higher-order layers ──────────────────────────────────────────────────
LAYERS = ("triple", "quartet", "five", "cross", "hmt")
C3, C4 = list(combinations(range(5), 3)), list(combinations(range(5), 4))


def team_keys(layer, T, m=None, t=None, nm=None, nt=None):
    """(n, k) int64 keys of the team's cells (own-team layers)."""
    if layer == "triple":
        return np.stack([(T[:, a] * H + T[:, b]) * H + T[:, c] for a, b, c in C3], 1)
    if layer == "quartet":
        return np.stack([((T[:, a] * H + T[:, b]) * H + T[:, c]) * H + T[:, d] for a, b, c, d in C4], 1)
    if layer == "five":
        k = T[:, 0].copy()
        for j in range(1, 5):
            k = k * H + T[:, j]
        return k[:, None]
    if layer == "hmt":
        return (T * nm + m[:, None]) * nt + t[:, None]
    raise ValueError(layer)


def cross_keys(T, O):
    """(n, 50) keys (pair of T, hero of O): the cells where T's pairs meet O's heroes."""
    pk = (T[:, IA] * H + T[:, IB])                    # (n, 10)
    return (pk[:, :, None] * H + O[:, None, :]).reshape(len(T), 50)


def keys(layer, D, nm, nt):
    """-> (pos keys of team 0, neg keys of team 1): each cell adds tau to team 0's logit (pos) or subtracts (neg)."""
    if layer == "cross":
        return cross_keys(D["T0"], D["T1"]), cross_keys(D["T1"], D["T0"])
    return (team_keys(layer, D["T0"], D["m"], D["t"], nm, nt), team_keys(layer, D["T1"], D["m"], D["t"], nm, nt))


def fit_layer(layer, D, off, nm, nt, s_grid, iters=200, kpn=None):
    """MAP of the layer's cell effects for each prior SD in s_grid (warm-started, increasing s);
    diag-Laplace evidence per s; -> dict with the chosen s and the posterior of the seen cells.
    kpn: (pos keys, neg keys) in place of the named layer's (any cell definition)."""
    import torch
    torch.set_num_threads(THREADS)
    kp, kn = kpn if kpn is not None else keys(layer, D, nm, nt)
    uk, inv = np.unique(np.concatenate([kp.ravel(), kn.ravel()]), return_inverse=True)
    ip = torch.tensor(inv[:kp.size].reshape(kp.shape))
    ineg = torch.tensor(inv[kp.size:].reshape(kn.shape))
    nc = np.bincount(inv, minlength=len(uk))
    y = torch.tensor(D["y"])
    o = torch.tensor(off)
    tau = torch.zeros(len(uk), dtype=torch.float64, requires_grad=True)
    res = []
    t0 = time.time()
    for s in sorted(s_grid):
        opt = torch.optim.LBFGS([tau], lr=1, max_iter=iters, tolerance_grad=1e-7, tolerance_change=1e-12,
                                history_size=30, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = o + tau[ip].sum(1) - tau[ineg].sum(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="sum") + 0.5 * (tau ** 2).sum() / s ** 2
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            z = (o + tau[ip].sum(1) - tau[ineg].sum(1)).numpy()
        th = tau.detach().numpy().copy()
        p = sig(z)
        w = p * (1 - p)
        h = np.bincount(inv, weights=np.concatenate([np.repeat(w, kp.shape[1]), np.repeat(w, kn.shape[1])]), minlength=len(uk))
        ll = float(np.sum(D["y"] * np.log(np.clip(p, 1e-12, 1)) + (1 - D["y"]) * np.log(np.clip(1 - p, 1e-12, 1))))
        ev = ll - 0.5 * float((th ** 2).sum()) / s ** 2 - 0.5 * float(np.log1p(h * s ** 2).sum())
        res.append(dict(s=float(s), loglik=ll, evidence=ev, tau=th, h=h, tau_sd=float(th.std())))
        print(f"  layer {layer} s={s:.4f}: evidence {ev:.1f} loglik {ll:.1f} sd(tau_hat) {th.std():.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    best = max(res, key=lambda d: d["evidence"])
    # profile bound: the largest s whose evidence is within 1.92 of the best (95%, one parameter)
    ok = [d["s"] for d in res if d["evidence"] >= best["evidence"] - 1.92]
    return dict(layer=layer, keys=uk, n=nc, mu=best["tau"], sd=1 / np.sqrt(best["h"] + 1 / best["s"] ** 2),
                s=best["s"], s_hi95=max(ok), s_lo95=min(ok), cells_seen=int(len(uk)),
                grid=[{k: d[k] for k in ("s", "loglik", "evidence", "tau_sd")} for d in res])


# ── persistence ──────────────────────────────────────────────────────────
BASE_NPZ = os.path.join(WD.CACHE, "base.npz")
CHOL_NPY = os.path.join(WD.CACHE, "base_chol.npy")


def layer_npz(layer):
    return os.path.join(WD.CACHE, f"layer_{layer}.npz")


def load_base():
    z = np.load(BASE_NPZ)
    cfg = json.loads(str(z["cfg"]))
    return Base(cfg["nm"], cfg["nt"], cfg["r"]), z["theta"], cfg


def load_layer(layer):
    z = np.load(layer_npz(layer))
    d = {k: z[k] for k in ("keys", "n", "mu", "sd")}
    d.update(json.loads(str(z["info"])))
    return d
