"""
P3 tails (results/P3_TAILS.md): non-Gaussian priors for a player's effect on
the query hero, on top of the rich-CF model.

For a query slot the factor model gives, exactly, the prior for theta at the
query hero from everything except that hero's own win evidence, N(m0, v0)
(p3_rcf.online_walk parts=True), and the own evidence ybar = Sh / Ph with
noise 1 / Ph. The Gaussian posterior is the product of the two. A card
replaces the Gaussian prior shape by f, on the standardized deviation
x = (theta - m0) / sqrt(v0):

  scale   N(mu, s^2)                                  (control: a plain refit)
  asym    split normal: mode mu, sd s_neg below and s_pos above
  mix     (1 - pi) N(0, s1^2) + pi N(mu2, s2^2)
  t       Student-t: location mu, scale s, df nu
  vol     N(mu_b, s_b^2) by play-volume cell b: games on the hero before the
          game (0, 1-9, 10-29, 30-99, 100+) x specialist (hero share of the
          player's games >= 20%)
  vol_t   vol with a Student-t shape (df per cell)

The posterior of theta is computed by quadrature on a fixed x grid. The
parameters are fit by maximum likelihood of the scored game's own residual
r_adj = r - offset ~ N(theta, wp (1 - wp)) under the posterior predictive,
on V1 slots only; V2 and OOT are the test. Everything a slot uses is as-of
(the parts come from the lag-1 state).
"""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

XG = np.linspace(-10, 10, 401)
NB_EDGES = np.array([0, 1, 10, 30, 100])


def vol_bin(nh, npl):
    b = np.searchsorted(NB_EDGES, nh, side="right") - 1
    spec = (nh / np.maximum(npl, 1)) >= 0.2
    return b * 2 + spec.astype(int)


def logf(card, p, ctx):
    """log prior density on XG (unnormalized), shape (n, len(XG)) or (len(XG),)."""
    x = XG
    if card == "scale":
        mu, s = p[0], np.exp(p[1])
        return -0.5 * ((x - mu) / s) ** 2 - np.log(s)
    if card == "asym":
        mu, sn, sp = p[0], np.exp(p[1]), np.exp(p[2])
        s = np.where(x < mu, sn, sp)
        return -0.5 * ((x - mu) / s) ** 2 - np.log((sn + sp) / 2)
    if card == "mix":
        pi = 1 / (1 + np.exp(-p[0]))
        s1, mu2, s2 = np.exp(p[1]), p[2], np.exp(p[3])
        a = np.log(1 - pi) - 0.5 * (x / s1) ** 2 - np.log(s1)
        b = np.log(pi) - 0.5 * ((x - mu2) / s2) ** 2 - np.log(s2)
        return np.logaddexp(a, b)
    if card == "t":
        mu, s, nu = p[0], np.exp(p[1]), 2 + np.exp(np.clip(p[2], -5, 8))
        z = (x - mu) / s
        return gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu) - np.log(s) - (nu + 1) / 2 * np.log1p(z * z / nu)
    if card in ("vol", "vol_t"):
        b = ctx["bin"]
        nb = 10
        mu = np.asarray(p[:nb])[b][:, None]
        s = np.exp(np.asarray(p[nb:2 * nb]))[b][:, None]
        z = (x[None, :] - mu) / s
        if card == "vol":
            return -0.5 * z ** 2 - np.log(s)
        nu = (2 + np.exp(np.clip(np.asarray(p[2 * nb:3 * nb]), -5, 8)))[b][:, None]
        return gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu) - (nu + 1) / 2 * np.log1p(z * z / nu) - np.log(s)
    raise ValueError(card)


def p0(card):
    return {"scale": [0.0, 0.0], "asym": [0.0, 0.0, 0.0], "mix": [-2.0, 0.0, -1.0, 0.7], "t": [0.0, 0.0, 2.0],
            "vol": [0.0] * 20, "vol_t": [0.0] * 20 + [2.0] * 10}[card]


def post(card, p, m0, v0, Sh, Ph, ctx, chunk=200000):
    """Posterior mean and variance of theta, and the log weights."""
    n = len(m0)
    mean = np.empty(n)
    var = np.empty(n)
    for i0 in range(0, n, chunk):
        sl = slice(i0, min(n, i0 + chunk))
        s0 = np.sqrt(v0[sl])[:, None]
        lf = logf(card, p, {k: v[sl] for k, v in ctx.items()})
        lf = np.broadcast_to(lf, (sl.stop - sl.start, len(XG)))
        P = Ph[sl][:, None]
        yb = np.where(P > 0, Sh[sl][:, None] / np.maximum(P, 1e-12), 0.0)
        th = m0[sl][:, None] + s0 * XG[None, :]
        ll = np.where(P > 0, -0.5 * P * (yb - th) ** 2, 0.0)
        lw = lf + ll
        lw = lw - lw.max(1, keepdims=True)
        w = np.exp(lw)
        w /= w.sum(1, keepdims=True)
        mean[sl] = (w * th).sum(1)
        var[sl] = (w * th * th).sum(1) - mean[sl] ** 2
    return mean, var


def nll(card, p, m0, v0, Sh, Ph, r, v, ctx):
    """- sum log posterior predictive of the scored residual r (Gaussian game
    noise v) under the card's posterior."""
    s0 = np.sqrt(v0)[:, None]
    lf = np.broadcast_to(logf(card, p, ctx), (len(m0), len(XG)))
    P = Ph[:, None]
    yb = np.where(P > 0, Sh[:, None] / np.maximum(P, 1e-12), 0.0)
    th = m0[:, None] + s0 * XG[None, :]
    lw = lf + np.where(P > 0, -0.5 * P * (yb - th) ** 2, 0.0)
    lw = lw - np.logaddexp.reduce(lw, axis=1, keepdims=True)
    lr = -0.5 * (r[:, None] - th) ** 2 / v[:, None] - 0.5 * np.log(2 * np.pi * v[:, None])
    return -float(np.logaddexp.reduce(lw + lr, axis=1).sum())


def apply(card, res, r_target, v, v1_mask, args=None, max_fit=250000, seed=0):
    """Fit the card on V1 slots (subsample), then the posterior for every slot.
    Returns (mean, var, meta)."""
    m0, v0, Sh, Ph = res["m0"], np.maximum(res["v0"], 1e-10), res["Sh"], res["Ph"]
    ctx = {}
    if card.startswith("vol"):
        ctx["bin"] = vol_bin(res["nh"], res["np"])
    idx = np.flatnonzero(v1_mask)
    rng = np.random.default_rng(seed)
    if len(idx) > max_fit:
        idx = np.sort(rng.choice(idx, max_fit, replace=False))
    sub = {k: x[idx] for k, x in ctx.items()}
    f = lambda p: nll(card, p, m0[idx], v0[idx], Sh[idx], Ph[idx], r_target[idx], v[idx], sub)
    pg = p0(card)
    base = nll("scale", [0.0, 0.0], m0[idx], v0[idx], Sh[idx], Ph[idx], r_target[idx], v[idx], sub)
    if card.startswith("vol"):
        # the likelihood separates by volume cell: fit each cell's (mu, s) on
        # its own (vol_t: plus its own df), then assemble
        nb = 10
        x = np.array(pg, float)
        if card == "vol_t":
            x = np.r_[np.zeros(nb), np.zeros(nb), np.full(nb, 2.0)]
        fun = 0.0
        for b in range(nb):
            sb = idx[sub["bin"] == b]
            if len(sb) < 500:
                continue
            one = {"bin": np.zeros(len(sb), int)}
            if card == "vol":
                fb = lambda q: nll("vol", np.r_[q[0], np.zeros(nb - 1), q[1], np.zeros(nb - 1)],
                                   m0[sb], v0[sb], Sh[sb], Ph[sb], r_target[sb], v[sb], one)
                rb = minimize(fb, [0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-4})
                x[b], x[nb + b] = rb.x
            else:
                fb = lambda q: nll("vol_t", np.r_[q[0], np.zeros(nb - 1), q[1], np.zeros(nb - 1), q[2]],
                                   m0[sb], v0[sb], Sh[sb], Ph[sb], r_target[sb], v[sb], one)
                rb = minimize(fb, [0.0, 0.0, 2.0], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-4})
                x[b], x[nb + b], x[2 * nb + b] = rb.x
            fun += rb.fun
        r = type("R", (), {"x": x, "fun": f(x)})()
    else:
        r = minimize(f, pg, method="Powell", options={"maxiter": 20000, "xtol": 1e-3, "ftol": 1e-7})
    mean, var = post(card, r.x, m0, v0, Sh, Ph, ctx)
    meta = {"card": card, "params": np.asarray(r.x).tolist(), "v1_fit_slots": int(len(idx)),
            "v1_nll": float(r.fun), "v1_nll_gaussian": base, "v1_gain_vs_gaussian": float(base - r.fun)}
    print(f"  tail {card}: V1 log-lik gain over Gaussian {base - r.fun:+.1f} on {len(idx):,} slots; "
          f"params {np.round(r.x, 3).tolist()}", flush=True)
    return mean, var, meta
