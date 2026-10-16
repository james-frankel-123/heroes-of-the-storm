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
               verbose=False, **conv):
    """Legacy path (the wave-0 to cycle-2 incumbents): eb_iters MacKay updates, each inner
    L-BFGS-B solve capped at maxiter. Passing `converge=...` options switches to
    fit_design_converged (warm start, preconditioning, a checked convergence criterion)."""
    if conv:
        return fit_design_converged(X, y, blocks, offs, offset=offset, weights=weights, lam_init=lam_init,
                                    verbose=verbose, **conv)
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


# --------------------------------------------------------------------------- converged fits
#
# Convergence criterion (cycle 3 on; CYCLE3_PREP.md section 1). Two levels:
#   inner: every coefficient's projected gradient, in units of its own diagonal posterior SD,
#          is at most PGTOL: max_i |pg_i| / sqrt(Hd_i) <= PGTOL. This is scale-free (a sum over
#          2.28M games and over 330K games is judged the same way) and says each coefficient is
#          within about PGTOL posterior SDs of its stationary point.
#   outer: the empirical-Bayes precisions are at a fixed point: no adaptive block's lam moves
#          by more than EB_TOL in log10 in an update (blocks shrunk to nothing, lam >= 1e6, and
#          blocks at the lower clip count as fixed).
# The legacy fits (gtol 1e-6 on the raw sum, 400 iterations, 4 EB updates) met neither.
PGTOL = 1e-3
EB_TOL = 0.01
LAM_CLIP = (1e-3, 1e7)


def _hdiag(X, w, pr, Qd):
    return np.asarray((X.multiply(X)).T @ (w * pr * (1 - pr))).ravel() + Qd


def _eb_update(blocks, offs, lam, beta, var, Hd, Q):
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
                used = (Hd[pr_[:, 0]] - Q.diagonal()[pr_[:, 0]] > 1e-3) & (Hd[pr_[:, 1]] - Q.diagonal()[pr_[:, 1]] > 1e-3)
                if used.sum() > 5:
                    new[bi] = used.sum() / max(float((d[used] ** 2 + dv[used]).sum()), 1e-9)
        elif q >= 3 and np.all(np.isinf(b.lo)) and np.all(np.isinf(b.hi)):
            gamma = float(np.sum(1 - lam[bi] * vv))
            new[bi] = max(gamma, 1.0) / max(float((bb ** 2).sum()), 1e-9)
    return np.clip(new, *LAM_CLIP)


LAM_SETTLED = 1e6     # prior SD at most 0.001 logit: the block is shrunk to nothing


def _eb_moved(lam, new):
    """Largest log10 change over blocks that are not settled. A block counts as settled when it
    sits at the lower clip, or when its precision is at least LAM_SETTLED before and after the
    update: MacKay drives a null block's precision up only geometrically, and from there on its
    coefficients are below 0.001 logit whatever the precision does."""
    a, b = np.log10(lam), np.log10(new)
    pinned = ((lam >= LAM_SETTLED) & (new >= LAM_SETTLED)) | (
        np.isclose(new, LAM_CLIP[0]) & np.isclose(lam, LAM_CLIP[0]))
    d = np.abs(b - a)
    d[pinned] = 0.0
    return float(d.max()) if len(d) else 0.0


def fit_design_converged(X, y, blocks, offs, offset=None, weights=None, lam_init=None, verbose=False,
                         beta_init=None, pgtol=PGTOL, eb_tol=EB_TOL, max_eb_rounds=60, max_iter_round=20000,
                         max_iter_total=None, precondition=True, log_every=100, mode="cold"):
    """Penalized MLE with EB precisions iterated to a fixed point, each inner solve run until the
    criterion above holds. Same objective and same MacKay updates as fit_design; only the
    starting point, the variable scaling and the stopping rules differ.

    beta_init / lam_init: warm start (a previous incumbent's coefficients and precisions mapped
    onto this design; new terms at 0 or at their Q2 offset-fit values).
    precondition: L-BFGS-B on u = b * sqrt(Hd) (diagonal Hessian scaling, recomputed each EB
    round). Bounds scale with it. The optimum is unchanged.
    max_iter_total: an overall iteration budget (mid-cycle warm refits); the fit stops there and
    reports converged=False."""
    import time as _t
    n, p = X.shape
    t0 = _t.time()
    y = y.astype(np.float64)
    w = np.ones(n) if weights is None else weights
    off = np.zeros(n) if offset is None else offset
    lam = np.array([b.lam0 for b in blocks], float) if lam_init is None else np.asarray(lam_init, float).copy()
    lo = np.concatenate([b.lo for b in blocks]); hi = np.concatenate([b.hi for b in blocks])
    lo_c, hi_c = np.where(np.isinf(lo), -1e9, lo), np.where(np.isinf(hi), 1e9, hi)
    beta = np.clip(np.zeros(p) if beta_init is None else np.asarray(beta_init, float), lo_c, hi_c)
    Xt = X.T.tocsr()
    hist, total_it, inner_ok, eb_ok = [], 0, False, False

    def crit(b, g, Hd):
        # projected gradient: zero where a bound is active and the gradient pushes outward
        pg = g.copy()
        pg[(b <= lo_c) & (g > 0)] = 0.0
        pg[(b >= hi_c) & (g < 0)] = 0.0
        return float(np.max(np.abs(pg) / np.sqrt(Hd))) if p else 0.0

    for rnd in range(max_eb_rounds):
        Q = _penalty_parts(blocks, offs, lam)
        Qd = Q.diagonal()

        def fg(b):
            eta = off + X @ b
            pr = expit(eta)
            return (w * nll(eta, y)).sum() + 0.5 * b @ (Q @ b), Xt @ (w * (pr - y)) + Q @ b

        pr0 = expit(off + X @ beta)
        Hd = np.maximum(_hdiag(X, w, pr0, Qd), 1e-12)
        s = 1.0 / np.sqrt(Hd) if precondition else np.ones(p)
        bnds = [(None if np.isinf(a) else a / si, None if np.isinf(c) else c / si) for a, c, si in zip(lo, hi, s)]
        f0, g0 = fg(beta)
        c_start = crit(beta, g0, Hd)
        nit_round, restarts, msg = 0, 0, ""
        state = {"k": 0, "t": _t.time()}

        def cb(u):
            state["k"] += 1
            if verbose and log_every and state["k"] % log_every == 0:
                print(f"  round {rnd} it {state['k']} ({_t.time() - state['t']:.0f}s)", flush=True)

        c_end = c_start
        while c_end > pgtol:
            budget = max_iter_round - nit_round
            if max_iter_total is not None:
                budget = min(budget, max_iter_total - total_it)
            if budget <= 0:
                break
            res = minimize(lambda u: (lambda v, g: (v, g * s))(*fg(u * s)), beta / s, jac=True,
                           method="L-BFGS-B", bounds=bnds,
                           options={"maxiter": budget, "maxcor": 20, "ftol": 1e-15, "gtol": pgtol},
                           callback=cb)
            beta = np.clip(res.x * s, lo_c, hi_c)
            nit_round += int(res.nit); total_it += int(res.nit)
            msg = str(res.message)
            _, g = fg(beta)
            pr_ = expit(off + X @ beta)
            Hd_now = np.maximum(_hdiag(X, w, pr_, Qd), 1e-12)
            c_end = crit(beta, g, Hd_now)
            if c_end > pgtol and int(res.nit) < budget:
                # stopped short of the criterion (line search at precision, or the ftol floor):
                # rescale at the current point and restart the L-BFGS memory, at most 5 times
                restarts += 1
                Hd = Hd_now
                s = 1.0 / np.sqrt(Hd) if precondition else np.ones(p)
                bnds = [(None if np.isinf(a) else a / si, None if np.isinf(c) else c / si) for a, c, si in zip(lo, hi, s)]
                if restarts > 5:
                    break
        eta = off + X @ beta
        pr = expit(eta)
        Hd = _hdiag(X, w, pr, Qd)
        var = 1.0 / np.maximum(Hd, 1e-12)
        inner_ok = c_end <= pgtol
        new = _eb_update(blocks, offs, lam, beta, var, Hd, Q)
        moved = _eb_moved(lam, new)
        hist.append({"iter": rnd, "nll": float((w * nll(eta, y)).sum()), "lam": lam.tolist(),
                     "converged": bool(inner_ok), "nit": nit_round, "restarts": restarts,
                     "crit_start": c_start, "crit_end": c_end, "eb_max_log10_move": moved,
                     "seconds": round(_t.time() - t0, 1), "message": msg[:120]})
        if verbose:
            print({k: v for k, v in hist[-1].items() if k != "lam"}, flush=True)
        if moved <= eb_tol and inner_ok:
            eb_ok = True        # lam is a fixed point and beta is converged at it: stop, keep lam
            break
        if max_iter_total is not None and total_it >= max_iter_total:
            break
        lam = new
    status = {"mode": mode, "converged": bool(inner_ok and eb_ok), "inner_converged": bool(inner_ok),
              "eb_converged": bool(eb_ok), "eb_rounds": len(hist), "iterations": total_it,
              "final_crit": hist[-1]["crit_end"] if hist else None, "pgtol": pgtol, "eb_tol": eb_tol,
              "precondition": bool(precondition), "seconds": round(_t.time() - t0, 1)}
    return beta, var, lam, hist, status


def warm_start(names_new, blocks_new, specs_new, sources):
    """Map coefficients and precisions onto a new design.

    sources: list of (Model) in priority order (the previous incumbent first, then Q2 offset
    fits of new terms). A coefficient is copied by its name when the term spec that produced it
    is identical in the source (so a term whose vocabulary or knots differ is not mixed in);
    a block's lam is copied the same way. Everything else starts at 0 and lam0.
    Returns beta0, lam0, report."""
    p = len(names_new)
    beta0 = np.zeros(p)
    lam0 = np.array([b.lam0 for b in blocks_new], float)
    have = np.zeros(p, bool)
    spec_key = lambda s: json.dumps(s, sort_keys=True)
    # block index of each new coefficient, and the spec for each block
    offs, o = [0], 0
    for b in blocks_new:
        o += b.X.shape[1]
        offs.append(o)
    report = {"copied": 0, "from": []}
    lam_have = np.zeros(len(blocks_new), bool)
    for src in sources:
        sk = {spec_key(s): i for i, s in enumerate(src.specs)}
        idx = {nm: j for j, nm in enumerate(src.names)}
        n_src = 0
        for bi, s in enumerate(specs_new):
            si = sk.get(spec_key(s))
            if si is None:
                continue
            for j in range(offs[bi], offs[bi + 1]):
                if have[j]:
                    continue
                k = idx.get(names_new[j])
                if k is not None and int(src.offs[si]) <= k < int(src.offs[si + 1]):
                    beta0[j] = src.beta[k]
                    have[j] = True
                    n_src += 1
            if not lam_have[bi] and si < len(src.lam):
                lam0[bi] = float(src.lam[si])
                lam_have[bi] = True
        report["from"].append({"model": getattr(src, "tag", "?"), "coefficients": n_src})
        report["copied"] += n_src
    report["zero"] = int(p - have.sum())
    return beta0, lam0, report


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


def fit(specs, ctx, y, offset=None, verbose=False, converge=None, warm_from=(), **kw):
    """converge: None for the legacy fit, or a dict of fit_design_converged options (e.g.
    {"mode": "cold"} or {"mode": "warm", "max_iter_total": 4000}). warm_from: Models whose
    matching terms seed the start point (warm mode only)."""
    t0 = time.time()
    ctx.max_fit_build = int(ctx.build.max())
    X, blocks, offs = TM.compile_model(specs, ctx)
    names = [n for b in blocks for n in b.names]
    warm = None
    if converge is None:
        beta, var, lam, hist = fit_design(X, y, blocks, offs, offset=offset, verbose=verbose, **kw)
        status = {"mode": "legacy", "converged": all(h["converged"] for h in hist[-1:]),
                  "iterations": int(sum(h["nit"] for h in hist)), "eb_rounds": len(hist)}
    else:
        opts = dict(converge)
        if opts.get("mode") == "warm" and warm_from:
            b0, l0, warm = warm_start(names, blocks, specs, list(warm_from))
            opts.setdefault("beta_init", b0)
            opts.setdefault("lam_init", l0)
        beta, var, lam, hist, status = fit_design_converged(X, y, blocks, offs, offset=offset, verbose=verbose,
                                                            **opts)
    info = {"n": int(ctx.n), "p": int(X.shape[1]), "seconds": round(time.time() - t0, 1),
            "max_fit_build": int(ctx.max_fit_build), "eb_history": hist, "fit_status": status,
            "row_ids_sha256": hashlib.sha256(np.sort(ctx.replay_id).tobytes()).hexdigest()}
    if warm is not None:
        info["warm_start"] = warm
    return Model(specs, [b.state for b in blocks], beta, var, lam, offs, names, ctx.meta, info)
