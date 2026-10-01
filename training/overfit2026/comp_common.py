"""
Shared helpers for the composition audit: team-level calibration gaps by
structure type, offset logistic fits, log loss.
"""
import os
import sys

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING_DIR)

import numpy as np

from overfit2026.structure import KINDS, kind_of

CONSENSUS = ["gN", "gN_naive", "RN", "QM2026"]
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_offset(X, y, offset, iters=50, l2=1e-6):
    """Logistic regression y ~ sigmoid(offset + c0 + X w). Returns (w_full, se)."""
    X1 = np.column_stack([np.ones(len(y)), X]) if X is not None and np.size(X) else np.ones((len(y), 1))
    w = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = sig(offset + X1 @ w)
        g = X1.T @ (y - p) - l2 * w
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    p = sig(offset + X1 @ w)
    H = (X1 * (p * (1 - p))[:, None]).T @ X1
    return w, np.sqrt(np.diag(np.linalg.inv(H)))


def teams(p0, y0, s0, s1):
    """Stack both teams of every game: (p_team, y_team, struct_team, kind_team)."""
    p = np.r_[p0, 1 - p0]
    y = np.r_[y0, 1 - y0]
    S = np.r_[s0, s1]
    return p, y, S, kind_of(S)


def recalibrate(p, y):
    """Global (slope, intercept) recalibration on all teams; symmetric data."""
    z = logit(p)
    w, _ = fit_offset(z[:, None], y, np.zeros(len(y)))
    return sig(w[0] + w[1] * z), float(w[1])


def gap_table(p0, y0, s0, s1, recal=True):
    p, y, S, k = teams(p0, y0, s0, s1)
    out = {"n_games": int(len(p0)), "logloss": logloss(p0, y0),
           "acc": float(np.mean((p0 > .5) == (y0 > .5)))}
    pr, slope = recalibrate(p, y) if recal else (None, None)
    out["slope"] = slope
    rows = {}
    for name in KINDS + ["any_degenerate"]:
        m = (k > 0) if name == "any_degenerate" else (k == KINDS.index(name))
        n = int(m.sum())
        if n == 0:
            continue
        g = y[m] - p[m]
        r = {"teams": n, "share_pct": 100 * n / len(y), "realized_wr": float(y[m].mean()),
             "pred": float(p[m].mean()), "gap_pp": float(100 * g.mean()),
             "se_pp": float(100 * g.std(ddof=1) / np.sqrt(n))}
        if recal:
            r["gap_recal_pp"] = float(100 * (y[m] - pr[m]).mean())
        rows[name] = r
    out["by_kind"] = rows
    return out
