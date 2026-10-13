"""Evaluator and acceptance gate (FRAMING 5.4, 7.1, 8, 11.9). Harness side only.

Not mounted into proposer sandboxes. Scores per-game paired log loss with a block
bootstrap (week x player cluster), applies the dev gate z >= 3.0 + 0.59 sqrt(I_g), records
what the alternative rules would decide, checks invariance over recent S builds, and
refuses to score any row a fit has read (fit-score disjointness).
"""
import hashlib
import math

import numpy as np
from scipy import stats
from scipy.special import expit

from training.scilearn import config as C
from training.scilearn.fit import nll


class DisjointnessError(RuntimeError):
    pass


def check_disjoint(score_ids, fit_read_ids):
    overlap = np.intersect1d(np.asarray(score_ids), np.asarray(fit_read_ids))
    if len(overlap):
        raise DisjointnessError(f"refusing to score {len(overlap)} rows that a fit read")


def clusters(game_ts, player_key, n_player_buckets=64):
    """Block id = (week, bucket of the game's smallest known player key)."""
    week = (game_ts // (7 * 86400)).astype(np.int64)
    pk = np.where(player_key >= 0, player_key, np.iinfo(np.int64).max).min(1)
    b = np.where(pk == np.iinfo(np.int64).max, -1, pk % n_player_buckets)
    _, blk = np.unique(week * 1000 + b, return_inverse=True)
    return blk


def block_se(x, blocks, n_boot=400, seed=0):
    """SE of mean(x) by a multiplier bootstrap over blocks (Poisson-like weights)."""
    rng = np.random.default_rng(seed)
    nb = blocks.max() + 1
    sums = np.bincount(blocks, weights=x, minlength=nb)
    cnts = np.bincount(blocks, minlength=nb).astype(float)
    W = rng.exponential(1.0, size=(n_boot, nb))
    means = (W @ sums) / (W @ cnts)
    se_boot = float(means.std(ddof=1))
    se_iid = float(x.std(ddof=1) / math.sqrt(len(x)))
    return max(se_boot, se_iid), se_boot / max(se_iid, 1e-12)


def metrics(eta, y):
    p = expit(eta)
    ll = nll(eta, y)
    out = {"n": int(len(y)), "logloss": float(ll.mean()), "acc": float(((eta > 0) == (y > 0)).mean()),
           "brier": float(((p - y) ** 2).mean())}
    # calibration: logistic regression of y on eta (slope, intercept)
    import scipy.optimize as so

    def f(ab):
        e = ab[0] + ab[1] * eta
        pr = expit(e)
        return nll(e, y).sum(), np.array([(pr - y).sum(), ((pr - y) * eta).sum()])
    r = so.minimize(f, np.array([0.0, 1.0]), jac=True, method="L-BFGS-B")
    out["cal_intercept"], out["cal_slope"] = float(r.x[0]), float(r.x[1])
    return out


def paired_gain(eta_inc, eta_new, y, blocks):
    """Gain = mean(ll_inc - ll_new) > 0 means the new model is better."""
    d = nll(eta_inc, y) - nll(eta_new, y)
    se, deff = block_se(d, blocks)
    return {"gain": float(d.mean()), "se": se, "z": float(d.mean() / se) if se > 0 else 0.0,
            "design_effect": float(deff ** 2), "n": int(len(d)), "sum2": float(2 * d.sum())}


def threshold(group_bits_before, margin=C.GATE_MARGIN):
    return C.GATE_Z0 + margin * math.sqrt(max(group_bits_before, 0.0))


def gate_decisions(z, group_bits_before, k_params=1, sum2=None):
    """Primary rule plus the alternatives the ledger records (FRAMING 3, row 2 and 3)."""
    thr = threshold(group_bits_before)
    p_one = float(stats.norm.sf(z))
    alt = {
        "half_margin": bool(z >= thr),
        "full_margin": bool(z >= C.GATE_Z0 + 2 * C.GATE_MARGIN * math.sqrt(max(group_bits_before, 0))),
        "fwer_z3.3": bool(z >= 3.3),
        "p_one_sided": p_one,     # FDR (BH at 10%) is decided over the cycle's p-values at close
    }
    if k_params > 1 and sum2 is not None:
        # in-sample family rule, recorded for comparison (FRAMING 5.4 last paragraph)
        alt["family_lrt"] = bool(sum2 >= stats.chi2.isf(stats.norm.sf(thr), k_params))
    # Terms are fit on the fit tranches and scored on held-out dev games, so the paired z
    # already pays for every parameter; the k-df LRT (an in-sample rule) is recorded only.
    decision = alt["half_margin"]
    near = (not decision) and z >= C.GATE_NEAR_MISS
    return decision, thr, alt, near


def invariance(score, offset, y, build, recent_builds, min_n=2000):
    """Fit y ~ offset + alpha_b * score per build; sign must hold in >= 4 of 6 and
    Cochran's Q must not reject at 0.05."""
    from scipy.optimize import minimize_scalar
    alphas, ses = [], []
    for b in recent_builds:
        m = build == b
        if m.sum() < min_n or np.abs(score[m]).sum() == 0:
            continue
        s, o, yy = score[m], offset[m], y[m]
        r = minimize_scalar(lambda a: nll(o + a * s, yy).sum(), bounds=(-20, 20), method="bounded")
        p = expit(o + r.x * s)
        info = float((s * s * p * (1 - p)).sum())
        alphas.append(r.x); ses.append(1 / math.sqrt(max(info, 1e-12)))
    alphas, ses = np.array(alphas), np.array(ses)
    if len(alphas) < 3:
        return {"ok": False, "reason": "fewer than 3 builds with support", "n_builds": len(alphas)}
    w = 1 / ses ** 2
    pooled = float((w * alphas).sum() / w.sum())
    Q = float((w * (alphas - pooled) ** 2).sum())
    p_q = float(stats.chi2.sf(Q, len(alphas) - 1))
    n_pos = int((alphas > 0).sum())
    need = max(4, math.ceil(len(alphas) * 4 / 6)) if len(alphas) >= 6 else len(alphas) - 1
    return {"ok": bool(n_pos >= need and p_q >= 0.05), "n_pos": n_pos, "n_builds": len(alphas),
            "alphas": alphas.round(4).tolist(), "cochran_q": Q, "p_q": p_q}


def shuffle_labels(y, game_ts, map_, tier, build, key, rid):
    """Shuffled-outcome control: whole game records permuted within (map, tier, build)
    blocks (the outcome moves with its trajectory; here only the outcome is used)."""
    seed = int(hashlib.sha256(f"{key}:shuffle".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    blk = (map_.astype(np.int64) * 10 + tier) * 1000 + build
    out = y.copy()
    order = np.argsort(blk, kind="stable")
    sb = blk[order]
    starts = np.flatnonzero(np.r_[True, sb[1:] != sb[:-1]])
    ends = np.r_[starts[1:], len(sb)]
    for s, e in zip(starts, ends):
        idx = order[s:e]
        out[idx] = y[rng.permutation(idx)]
    return out


def post_stratify_weights(cells_sample, cells_population):
    """Weights so the sample's cell mix matches the population's; returns w, n_eff."""
    u, inv = np.unique(cells_sample, return_inverse=True)
    pop = np.array([(cells_population == c).sum() for c in u], float)
    smp = np.bincount(inv).astype(float)
    w = (pop / pop.sum()) / (smp / smp.sum())
    w = w[inv]
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    return w, n_eff
