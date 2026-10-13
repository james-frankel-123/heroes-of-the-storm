"""Fitting backends (FRAMING 6.2).

Penalized MLE on the antisymmetric design (L2 per group, random-walk GMRF for state terms),
solved with L-BFGS-B (bounds carry the shape and sign constraints). Shrinkage precisions
come from empirical Bayes with a diagonal Laplace approximation (MacKay updates). The
design is antisymmetric with no intercept, so swap augmentation adds nothing: the
likelihood of (d, y) equals that of (-d, 1 - y). The diagonal Laplace posterior variances
are also the Bayesian backend's output used by the leaf's lower confidence bound.

Every fit records the row IDs it read (fit-score disjointness, FRAMING 6.3 rule 2).
"""
import hashlib
import json
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit, log1p

from training.scilearn import terms as TM


def _penalty_parts(blocks, offs, lam):
    """Sparse precision matrix Q from blocks and per-block precisions lam."""
    p = offs[-1]
    diag = np.zeros(p)
    rows, cols, vals = [], [], []
    for bi, b in enumerate(blocks):
        o = offs[bi]
        q = b.X.shape[1]
        if b.kind == "rw":
            diag[o:o + q] += b.state.get("level_precision", 1.0)
            if b.rw_pairs:
                pr = np.asarray(b.rw_pairs) + o
                i, j = pr[:, 0], pr[:, 1]
                l = lam[bi]
                diag[i] += l; diag[j] += l
                rows += [i, j]; cols += [j, i]; vals += [np.full(len(i), -l)] * 2
        else:
            diag[o:o + q] += lam[bi]
    Q = sp.diags(diag)
    if rows:
        Q = Q + sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(p, p))
    return Q.tocsr()


def nll(eta, y):
    # sum of log(1 + exp(-s*eta)) with s = 2y - 1, stable
    z = np.where(y > 0, eta, -eta)
    return np.logaddexp(0, -z)


def fit_design(X, y, blocks, offs, offset=None, weights=None, eb_iters=4, maxiter=400, lam_init=None,
               verbose=False):
    n, p = X.shape
    y = y.astype(np.float64)
    w = np.ones(n) if weights is None else weights
    off = np.zeros(n) if offset is None else offset
    lam = np.array([b.lam0 for b in blocks], float) if lam_init is None else np.asarray(lam_init, float)
    lo = np.concatenate([b.lo for b in blocks]); hi = np.concatenate([b.hi for b in blocks])
    bounds = [(None if np.isinf(a) else a, None if np.isinf(b_) else b_) for a, b_ in zip(lo, hi)]
    beta = np.clip(np.zeros(p), np.where(np.isinf(lo), -1e9, lo), np.where(np.isinf(hi), 1e9, hi))
    Xt = X.T.tocsr()
    hist = []
    for it in range(eb_iters + 1):
        Q = _penalty_parts(blocks, offs, lam)

        def f(b):
            eta = off + X @ b
            pr = expit(eta)
            val = (w * nll(eta, y)).sum() + 0.5 * b @ (Q @ b)
            g = Xt @ (w * (pr - y)) + Q @ b
            return val, g

        res = minimize(f, beta, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "maxcor": 20, "ftol": 1e-10, "gtol": 1e-6})
        beta = res.x
        eta = off + X @ beta
        pr = expit(eta)
        Hd = np.asarray((X.multiply(X)).T @ (w * pr * (1 - pr))).ravel() + Q.diagonal()
        var = 1.0 / np.maximum(Hd, 1e-12)
        hist.append({"iter": it, "nll": float((w * nll(eta, y)).sum()), "lam": lam.tolist(),
                     "converged": bool(res.success), "nit": int(res.nit)})
        if verbose:
            print(hist[-1], flush=True)
        if it == eb_iters:
            break
        new = lam.copy()
        for bi, b in enumerate(blocks):
            if b.kind == "free" or b.X.shape[1] == 0:
                continue
            o, q = offs[bi], b.X.shape[1]
            bb, vv = beta[o:o + q], var[o:o + q]
            if b.kind == "rw":
                if b.rw_pairs:
                    pr_ = np.asarray(b.rw_pairs) + o
                    d = beta[pr_[:, 1]] - beta[pr_[:, 0]]
                    dv = var[pr_[:, 1]] + var[pr_[:, 0]]
                    # only differences between builds that both have data carry evidence
                    used = (Hd[pr_[:, 0]] - Q.diagonal()[pr_[:, 0]] > 1e-3) & (Hd[pr_[:, 1]] - Q.diagonal()[pr_[:, 1]] > 1e-3)
                    if used.sum() > 5:
                        new[bi] = used.sum() / max(float((d[used] ** 2 + dv[used]).sum()), 1e-9)
            elif q >= 3 and np.all(np.isinf(b.lo)) and np.all(np.isinf(b.hi)):
                # MacKay: gamma = sum(1 - lam*var); lam = gamma / sum(beta^2)
                gamma = float(np.sum(1 - lam[bi] * vv))
                new[bi] = max(gamma, 1.0) / max(float((bb ** 2).sum()), 1e-9)
        lam = np.clip(new, 1e-3, 1e7)
    return beta, var, lam, hist


class Model:
    """A fitted additive model: specs + per-term compile states + coefficients."""

    def __init__(self, specs, states, beta, var, lam, offs, names, meta, fit_info):
        self.specs, self.states = specs, states
        self.beta, self.var, self.lam = beta, var, lam
        self.offs, self.names = offs, names
        self.meta, self.fit_info = meta, fit_info

    def design(self, ctx):
        ctx.max_fit_build = self.fit_info["max_fit_build"]
        X, _, _ = TM.compile_model(self.specs, ctx, self.states)
        return X

    def logit(self, ctx, offset=None):
        eta = self.design(ctx) @ self.beta
        return eta if offset is None else eta + offset

    def term_contrib(self, ctx):
        X = self.design(ctx)
        out = {}
        for i, s in enumerate(self.specs):
            o0, o1 = self.offs[i], self.offs[i + 1]
            out[s.get("name", f"t{i}")] = X[:, o0:o1] @ self.beta[o0:o1]
        return out

    def save(self, path):
        np.savez(path.with_suffix(".npz"), beta=self.beta, var=self.var, lam=self.lam, offs=self.offs)
        path.with_suffix(".json").write_text(json.dumps(
            {"specs": self.specs, "states": self.states, "names": self.names, "fit_info": self.fit_info},
            default=float))

    @classmethod
    def load(cls, path, meta):
        a = np.load(path.with_suffix(".npz"))
        j = json.loads(path.with_suffix(".json").read_text())
        return cls(j["specs"], j["states"], a["beta"], a["var"], a["lam"], a["offs"], j["names"], meta, j["fit_info"])

    def model_hash(self):
        h = hashlib.sha256(json.dumps(self.specs, sort_keys=True).encode())
        h.update(self.beta.astype(np.float64).tobytes())
        return h.hexdigest()


def fit(specs, ctx, y, offset=None, verbose=False, **kw):
    t0 = time.time()
    ctx.max_fit_build = int(ctx.build.max())
    X, blocks, offs = TM.compile_model(specs, ctx)
    beta, var, lam, hist = fit_design(X, y, blocks, offs, offset=offset, verbose=verbose, **kw)
    names = [n for b in blocks for n in b.names]
    info = {"n": int(ctx.n), "p": int(X.shape[1]), "seconds": round(time.time() - t0, 1),
            "max_fit_build": int(ctx.max_fit_build), "eb_history": hist,
            "row_ids_sha256": hashlib.sha256(np.sort(ctx.replay_id).tobytes()).hexdigest()}
    return Model(specs, [b.state for b in blocks], beta, var, lam, offs, names, ctx.meta, info)
