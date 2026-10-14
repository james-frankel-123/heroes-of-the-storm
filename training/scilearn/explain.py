"""Compression / explanation cards (cycle 2). Harness side only.

An accepted lookup table (hero x map, hero x tier, ally pairs, hero-vs-hero counters) is the
accurate layer. A theory card proposes a small kit- or role-based function of the table's
keys. It does not add predictive terms; it earns a place by explaining and transferring:

  (a) explain   cross-fitted, precision-weighted share of the table's signal it explains, on
                a CONFIRM table fit on Dpre+G (disjoint from S, where sessions explore and
                see their table). 10 keyed folds over cells; the theory never sees the cells
                it is scored on.
  (b) transfer  leave-heroes-out: theory fit without any cell of the O3 heroes (20 heroes
                with >= 20K S games, keyed hash, 5 folds), scored on their cells (FRAMING O3).
  (c) onset     the incumbent's table shrunk toward the theory instead of toward 0, scored on
                the O2 slices (first 10K games of each Dpost build). A guard: must not hurt.
  (d) permutation  kit features are recomputed from P within-role permutations of the kit
                table; the real theory must beat 95% of them on (a). A theory whose kit axes
                carry only role-level information fails here.

Cell model. For each table cell j the incumbent fit gives a posterior mean beta_j, variance
v_j and the block's prior precision lam. With the fitter's diagonal Laplace approximation the
data-only estimate is b_j = beta_j (1/v_j) / (1/v_j - lam) with variance s_j = 1/(1/v_j - lam).
Weights are w_j = 1/s_j. Main effects the table shares with other terms (hero strength,
context means) are projected out of both target and features by weighted least squares
(nuisance design N), so a theory is credited only for interaction structure.

For a held-out cell, d_j = w_j [r_j^2 - (r_j - f_j)^2], with r the projected target and f the
theory's prediction. Sum(d) is the drop in the weighted residual sum of squares against
"explains nothing"; it is <= 0 in expectation for a theory with no real content. Signal
sum of squares S = sum(w r^2) - (n - rank N), since noise contributes 1 per cell after
weighting. Share = sum(d) / S. With Gaussian cell noise, sum(d) / (2 ln 2) is the number of
bits the theory saves in coding the held-out cells, and it must exceed the theory's own
description cost 0.5 k log2(n) (the MDL charge).
"""
import hashlib
import math

import numpy as np
from scipy.optimize import lsq_linear

TABLE_KINDS = ("hero_ctx", "pair", "cross")
MAX_THEORY_PARAMS = 30
N_FOLDS = 10
N_PERM = 200
O3_N_HEROES, O3_MIN_GAMES, O3_FOLDS = 20, 20000, 5

# Acceptance gate for explanation cards (CYCLE2_PREP.md section 3).
GATE = {
    "z_explain_base": 3.0,       # plus 0.59 sqrt(I_g), as the Q2 dev gate
    "z_transfer": 2.0,           # leave-heroes-out, one-sided
    "z_onset_floor": -2.0,       # onset must not get worse
    "p_perm": 0.05,              # beat 95% of within-role kit permutations
}


class ExplainError(ValueError):
    pass


def _seed(*parts):
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:12], 16)


# --------------------------------------------------------------------------- table cells

def table_kind(spec):
    t = spec.get("type")
    if t == "lookup" and len(spec.get("keys", [])) == 2 and spec["keys"][0] == "hero":
        return "hero_ctx"
    if t == "pair_family":
        return "pair"
    if t == "cross_family":
        return "cross"
    raise ExplainError(f"term type {t!r} with keys {spec.get('keys')} is not an explainable table "
                       "(hero x map / map_class / tier lookups, pair_family, cross_family)")


def table_cells(model, table_name, meta):
    """Cells of one table term in a fitted Model (training.scilearn.fit.Model)."""
    names = [s.get("name") for s in model.specs]
    if table_name not in names:
        raise ExplainError(f"table {table_name!r} is not a term of this model")
    i = names.index(table_name)
    spec, state = model.specs[i], model.states[i] or {}
    kind = table_kind(spec)
    o0, o1 = int(model.offs[i]), int(model.offs[i + 1])
    beta, var = np.asarray(model.beta[o0:o1], float), np.asarray(model.var[o0:o1], float)
    lam = float(model.lam[i])
    H = len(meta["heroes"])
    if kind == "hero_ctx":
        ctx = spec["keys"][1]
        labels = {"map": meta["maps"], "map_class": ["lanes2", "lanes3"], "tier": meta["tiers"]}[ctx]
        nc = len(labels)
        j = np.arange(o1 - o0)
        a, b = j // nc, j % nc
    else:
        ctx, labels, nc = None, None, None
        codes = np.asarray(state["vocab"], np.int64)
        a, b = codes // H, codes % H
    return cells_from_posterior(kind, a, b, beta, var, lam, ctx=ctx, ctx_labels=labels, n_ctx=nc,
                                n_heroes=H, table_name=table_name)


def cells_from_posterior(kind, a, b, beta, var, lam, **info):
    prec = 1.0 / np.maximum(var, 1e-300)
    pdata = prec - lam
    keep = pdata > 1e-3 * max(lam, 1e-6)          # cells with data behind them
    s = 1.0 / pdata[keep]
    y = beta[keep] * prec[keep] / pdata[keep]
    out = {"kind": kind, "a": np.asarray(a)[keep].astype(np.int64), "b": np.asarray(b)[keep].astype(np.int64),
           "y": y, "s": s, "w": 1.0 / s, "beta": beta[keep], "lam": lam, "keep": keep}
    out.update(info)
    return out


# --------------------------------------------------------------------------- theory features

def validate_theory(theory, kind, axes, roles, ctx=None):
    """Check a card's theory spec. Returns the number of parameters it can use."""
    feats = theory.get("features")
    if not isinstance(feats, list) or not feats:
        raise ExplainError("explain term needs a non-empty `features` list")
    k = 0
    for f in feats:
        if not isinstance(f, dict):
            raise ExplainError("each feature is a mapping")
        if kind == "hero_ctx":
            if "kit" in f:
                if f["kit"] not in axes:
                    raise ExplainError(f"unknown kit axis {f['kit']!r}")
            elif "role" in f:
                if f["role"] not in roles:
                    raise ExplainError(f"unknown role {f['role']!r}")
            else:
                raise ExplainError("hero x context tables take features {kit: <axis>, by: ...} or {role: <Role>, by: ...}")
            by = f.get("by", ctx)
            if by not in ("map", "map_class", "tier", "lanes2"):
                raise ExplainError("`by` must be map, map_class, lanes2 or tier")
            if ctx == "tier" and by != "tier":
                raise ExplainError("a hero x tier table takes `by: tier`")
            if ctx in ("map", "map_class") and by == "tier":
                raise ExplainError("a hero x map table takes `by: map`, `map_class` or `lanes2`")
            if ctx == "map_class" and by == "map":
                raise ExplainError("a hero x map_class table takes `by: map_class` or `lanes2`")
            k += {"map": 15, "map_class": 2, "lanes2": 1, "tier": 4}[by]
        else:
            if "kit_pair" in f:
                p = f["kit_pair"]
                if not (isinstance(p, list) and len(p) == 2 and all(x in axes for x in p)):
                    raise ExplainError("kit_pair takes [axis, axis]")
                k += 1
            elif "kit_sum" in f and kind == "pair":
                if f["kit_sum"] not in axes:
                    raise ExplainError(f"unknown kit axis {f['kit_sum']!r}")
                k += 1
            elif "same_role" in f:
                k += 1
            elif "role_pair" in f:
                k += 21 if kind == "pair" else 15
            else:
                raise ExplainError("pair tables take {kit_pair: [a, b]}, {kit_sum: axis}, {same_role: true} "
                                   "or {role_pair: true}; counter tables take kit_pair or role_pair")
        if f.get("sign") not in (None, "+", "-"):
            raise ExplainError("sign must be + or -")
    if k > MAX_THEORY_PARAMS:
        raise ExplainError(f"theory has up to {k} parameters; the cap is {MAX_THEORY_PARAMS}")
    return k


def has_kit(theory):
    return any(("kit" in f) or ("kit_pair" in f) or ("kit_sum" in f) for f in theory["features"])


def theory_features(theory, cells, kit, axes, hero_roles, roles, maps=None, lanes2=None):
    """Design F (n_cells, k) and per-column bounds. `kit` is (H, K) aligned to axes."""
    a, b, kind = cells["a"], cells["b"], cells["kind"]
    cols, lo, hi, names = [], [], [], []

    def bounds(f, m):
        s = f.get("sign")
        lo.extend([0.0 if s == "+" else -np.inf] * m)
        hi.extend([0.0 if s == "-" else np.inf] * m)

    if kind == "hero_ctx":
        ctx = cells["ctx"]
        labels = cells["ctx_labels"]
        for f in theory["features"]:
            v = kit[a, axes.index(f["kit"])] if "kit" in f else (hero_roles[a] == roles.index(f["role"])).astype(float)
            by = f.get("by", ctx)
            if by == ctx or (by == "map" and ctx == "map"):
                lev = b
                levels = list(range(len(labels)))
            elif by in ("map_class", "lanes2") and ctx == "map":
                cls = np.array([0 if m in lanes2 else 1 for m in labels])
                lev = cls[b]
                levels = [0, 1] if by == "map_class" else [0]
            elif by == "lanes2" and ctx == "map_class":
                lev, levels = b, [0]
            else:
                raise ExplainError(f"by {by!r} does not fit a hero x {ctx} table")
            for L in levels:
                cols.append(v * (lev == L))
                names.append(f"{f.get('kit', f.get('role'))}x{by}[{L}]")
            bounds(f, len(levels))
    else:
        R = len(roles)
        for f in theory["features"]:
            if "kit_pair" in f:
                k, l = axes.index(f["kit_pair"][0]), axes.index(f["kit_pair"][1])
                if kind == "cross":     # effect of lo-hero's team vs hi-hero's team, antisymmetric
                    cols.append(kit[a, k] * kit[b, l] - kit[b, k] * kit[a, l])
                else:                   # within-team pair, symmetric
                    cols.append(kit[a, k] * kit[b, l] + kit[b, k] * kit[a, l])
                names.append(f"{f['kit_pair'][0]}x{f['kit_pair'][1]}")
                bounds(f, 1)
            elif "kit_sum" in f:
                k = axes.index(f["kit_sum"])
                cols.append(kit[a, k] * kit[b, k])     # pair-level stacking of one axis
                names.append(f"stack[{f['kit_sum']}]")
                bounds(f, 1)
            elif "same_role" in f:
                cols.append((hero_roles[a] == hero_roles[b]).astype(float))
                names.append("same_role")
                bounds(f, 1)
            elif "role_pair" in f:
                ra, rb = hero_roles[a], hero_roles[b]
                m = 0
                for i in range(R):
                    for j in range(i, R):
                        if kind == "cross":
                            if i == j:
                                continue
                            cols.append(((ra == i) & (rb == j)).astype(float) - ((ra == j) & (rb == i)).astype(float))
                        else:
                            cols.append((((ra == i) & (rb == j)) | ((ra == j) & (rb == i))).astype(float))
                        names.append(f"roles[{roles[i]},{roles[j]}]")
                        m += 1
                bounds(f, m)
    F = np.stack(cols, 1) if cols else np.zeros((len(a), 0))
    return F, np.array(lo), np.array(hi), names


# --------------------------------------------------------------------------- projection and fit

def nuisance_design(cells, n_heroes):
    a, b, kind = cells["a"], cells["b"], cells["kind"]
    n = len(a)
    if kind == "hero_ctx":
        N = np.zeros((n, n_heroes + cells["n_ctx"]))
        N[np.arange(n), a] = 1
        N[np.arange(n), n_heroes + b] = 1
    elif kind == "pair":
        N = np.zeros((n, n_heroes))
        N[np.arange(n), a] += 1
        N[np.arange(n), b] += 1
    else:
        N = np.zeros((n, n_heroes))
        N[np.arange(n), a] += 1
        N[np.arange(n), b] -= 1
    used = np.abs(N).sum(0) > 0
    return N[:, used]


class Projector:
    """Weighted residualization on the nuisance design: x -> x - N (N'WN)^+ N'W x."""

    def __init__(self, N, w):
        sw = np.sqrt(w)[:, None]
        U, sv, _ = np.linalg.svd(N * sw, full_matrices=False)
        r = int((sv > sv.max() * 1e-9).sum()) if len(sv) else 0
        self.Q, self.sw, self.rank = U[:, :r], np.sqrt(w), r

    def __call__(self, X):
        X = np.asarray(X, float)
        one = X.ndim == 1
        X2 = X[:, None] if one else X
        Z = X2 * self.sw[:, None]
        Z = Z - self.Q @ (self.Q.T @ Z)
        out = Z / self.sw[:, None]
        return out[:, 0] if one else out


def wls_bounded(F, r, w, lo, hi, ridge=1e-6):
    """argmin sum w (r - F g)^2 + ridge |g|^2 subject to lo <= g <= hi."""
    k = F.shape[1]
    if k == 0:
        return np.zeros(0)
    sw = np.sqrt(w)
    A = np.vstack([F * sw[:, None], math.sqrt(ridge * max(w.sum(), 1.0)) * np.eye(k)])
    t = np.concatenate([r * sw, np.zeros(k)])
    lo_ = np.where(np.isfinite(lo), lo, -np.inf)
    hi_ = np.where(np.isfinite(hi), hi, np.inf)
    if np.all(np.isinf(lo_)) and np.all(np.isinf(hi_)):
        return np.linalg.lstsq(A, t, rcond=None)[0]
    return lsq_linear(A, t, bounds=(lo_, hi_), method="bvls").x


def _block_se(d, clusters, n_boot=400, seed=0):
    rng = np.random.default_rng(seed)
    _, cl = np.unique(clusters, return_inverse=True)
    nb = cl.max() + 1
    sums = np.bincount(cl, weights=d, minlength=nb)
    W = rng.exponential(1.0, size=(n_boot, nb))
    tot = W @ sums                       # bootstrap of the SUM over cells (clusters reweighted)
    se_sum = float(tot.std(ddof=1)) if n_boot > 1 else 0.0
    se_iid = float(d.std(ddof=1) * math.sqrt(len(d))) if len(d) > 1 else 0.0
    return max(se_sum, se_iid)


def crossfit_score(F, r, w, lo, hi, folds, clusters, seed=0):
    """Out-of-fold predictions; returns (sum d, se of sum, z, d, pred)."""
    pred = np.zeros(len(r))
    for f in np.unique(folds):
        te = folds == f
        tr = ~te
        if tr.sum() <= F.shape[1] or te.sum() == 0:
            continue
        g = wls_bounded(F[tr], r[tr], w[tr], lo, hi)
        pred[te] = F[te] @ g
    d = w * (r ** 2 - (r - pred) ** 2)
    sd = float(d.sum())
    se = _block_se(d, clusters, seed=seed)
    return sd, se, (sd / se if se > 0 else 0.0), d, pred


def signal_ss(r, w, rank):
    return float((w * r ** 2).sum() - (len(r) - rank))


# --------------------------------------------------------------------------- folds

def cell_folds(cells, key, k=N_FOLDS):
    h = np.array([_seed(key, int(x), int(y)) % k for x, y in zip(cells["a"], cells["b"])])
    return h


def o3_heroes(hero_names, hero_game_counts, key, n=O3_N_HEROES, min_games=O3_MIN_GAMES):
    """FRAMING O3: 20 heroes with at least 20K games, chosen by keyed hash. Returns indices."""
    elig = [i for i, c in enumerate(hero_game_counts) if c >= min_games]
    elig.sort(key=lambda i: _seed(key, hero_names[i]))
    return elig[:n]


def cell_clusters(cells):
    return cells["a"] if cells["kind"] == "hero_ctx" else np.minimum(cells["a"], cells["b"])


# --------------------------------------------------------------------------- the three scores

def score_explain(cells, theory, kit, axes, hero_roles, roles, n_heroes, *, fold_key, perm_key,
                  o3=None, lanes2=None, n_perm=N_PERM):
    """(a) cross-fitted explained share, (b) leave-heroes-out transfer, (d) permutation p."""
    F, lo, hi, names = theory_features(theory, cells, kit, axes, hero_roles, roles, lanes2=lanes2)
    w = cells["w"]
    N = nuisance_design(cells, n_heroes)
    P = Projector(N, w)
    r = P(cells["y"])
    Ft = P(F)
    S = signal_ss(r, w, P.rank)
    n = len(r)
    k_eff = int(np.linalg.matrix_rank(Ft * np.sqrt(w)[:, None])) if F.shape[1] else 0
    folds = cell_folds(cells, fold_key)
    cl = cell_clusters(cells)
    sd, se, z, d, pred = crossfit_score(Ft, r, w, lo, hi, folds, cl)
    g_all = wls_bounded(Ft, r, w, lo, hi)
    bits_saved = sd / (2 * math.log(2))
    param_bits = 0.5 * k_eff * math.log2(max(n, 2))
    out = {"n_cells": n, "k": int(F.shape[1]), "k_eff": k_eff, "feature_names": names,
           "coefs": [float(x) for x in g_all],
           "signal_ss": S, "noise_ss": float(n - P.rank),
           "explain": {"sum_d": sd, "se": se, "z": z, "share": sd / S if S > 0 else float("nan"),
                       "share_se": se / S if S > 0 else float("nan"),
                       "bits_saved": bits_saved, "param_bits": param_bits}}
    # (b) leave-heroes-out on the O3 heroes
    if o3:
        o3 = list(o3)
        fold_of = {h: i % O3_FOLDS for i, h in enumerate(o3)}
        dd, cc, rr = [], [], []
        for f in range(O3_FOLDS):
            hs = [h for h in o3 if fold_of[h] == f]
            touch = np.isin(cells["a"], hs) | np.isin(cells["b"], hs)
            if touch.sum() == 0 or (~touch).sum() <= F.shape[1]:
                continue
            g = wls_bounded(Ft[~touch], r[~touch], w[~touch], lo, hi)
            p = Ft[touch] @ g
            dd.append(w[touch] * (r[touch] ** 2 - (r[touch] - p) ** 2))
            cc.append(np.where(np.isin(cells["a"][touch], hs), cells["a"][touch], cells["b"][touch]))
            rr.append(r[touch] * np.sqrt(w[touch]))
        if dd:
            dd, cc = np.concatenate(dd), np.concatenate(cc)
            sb = float(dd.sum())
            seb = _block_se(dd, cc, seed=1)
            n_t = len(dd)
            S_t = float((np.concatenate(rr) ** 2).sum()) - n_t * (n - P.rank) / max(n, 1)
            out["transfer"] = {"n_cells": n_t, "n_heroes": len(o3), "sum_d": sb, "se": seb,
                               "z": sb / seb if seb > 0 else 0.0,
                               "share": sb / S_t if S_t > 0 else float("nan")}
    # (d) within-role permutations of the kit table
    if has_kit(theory) and n_perm:
        from training.scilearn.terms import permuted_kit
        null = []
        for i in range(n_perm):
            kp = permuted_kit(kit, hero_roles, f"{perm_key}:{i}")
            Fp, _, _, _ = theory_features(theory, cells, kp, axes, hero_roles, roles, lanes2=lanes2)
            sdp, _, _, _, _ = crossfit_score(P(Fp), r, w, lo, hi, folds, cl)
            null.append(sdp)
        null = np.array(null)
        out["permutation"] = {"n": n_perm, "p": float((1 + (null >= sd).sum()) / (n_perm + 1)),
                              "null_q95": float(np.quantile(null, 0.95)), "null_mean": float(null.mean())}
    else:
        out["permutation"] = {"n": 0, "p": None, "note": "no kit features (role-only theory)"}
    out["_pred_all"] = Ft @ g_all
    return out


def shrink_to_theory(y_raw, s_raw, lam, prior_mean):
    """Posterior mean of each cell under N(prior_mean, 1/lam) instead of N(0, 1/lam)."""
    return (y_raw / s_raw + lam * prior_mean) / (1.0 / s_raw + lam)


def theory_prior_for(cells_inc, theory, coefs, kit, axes, hero_roles, roles, n_heroes, lanes2=None):
    """Theory predictions for the incumbent table's cells, projected on its own nuisance."""
    F, _, _, _ = theory_features(theory, cells_inc, kit, axes, hero_roles, roles, lanes2=lanes2)
    P = Projector(nuisance_design(cells_inc, n_heroes), cells_inc["w"])
    return P(F) @ np.asarray(coefs, float)


def gate(res, group_bits_before, margin=0.59):
    """Decision for an explanation card. Returns (accept, reasons, threshold)."""
    thr = GATE["z_explain_base"] + margin * math.sqrt(max(group_bits_before, 0.0))
    reasons = []
    ex = res["explain"]
    if not res["signal_ss"] > 0:
        reasons.append("the table has no signal beyond noise to explain")
    if ex["z"] < thr:
        reasons.append(f"explain z {ex['z']:.2f} below {thr:.2f}")
    if ex["bits_saved"] <= ex["param_bits"]:
        reasons.append(f"does not compress: saves {ex['bits_saved']:.1f} bits, costs {ex['param_bits']:.1f}")
    tr = res.get("transfer")
    if tr is None:
        reasons.append("no leave-heroes-out cells")
    elif tr["z"] < GATE["z_transfer"] or not tr["sum_d"] > 0:
        reasons.append(f"transfer z {tr['z']:.2f} below {GATE['z_transfer']}")
    on = res.get("onset")
    if on is not None and on["z"] < GATE["z_onset_floor"]:
        reasons.append(f"onset worse (z {on['z']:.2f})")
    pm = res.get("permutation", {})
    if pm.get("p") is not None and pm["p"] > GATE["p_perm"]:
        reasons.append(f"kit axes do no better than within-role permutations (p {pm['p']:.3f})")
    return not reasons, reasons, thr


def onset_gain(X_tab, beta_tab, cells_inc, prior, eta0, y, blocks):
    """(c) Paired log-loss gain on onset games when the incumbent table is shrunk toward the
    theory. X_tab: the table's design columns on those games; eta0: incumbent logit."""
    from training.scilearn import evaluator as E
    beta_new = np.array(beta_tab, float).copy()
    beta_new[cells_inc["keep"]] = shrink_to_theory(cells_inc["y"], cells_inc["s"], cells_inc["lam"], prior)
    eta1 = eta0 + X_tab @ (beta_new - beta_tab)
    out = E.paired_gain(eta0, eta1, y, blocks)
    out["mean_abs_shift"] = float(np.abs(eta1 - eta0).mean())
    return out
