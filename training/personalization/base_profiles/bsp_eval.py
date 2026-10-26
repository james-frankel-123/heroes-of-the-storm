"""
BSP: do as-of scoreboard profiles add to paper 1's base draft WP?

Stages (5090, CPU, nice 19, 4 threads, < 8 GB):
  prep   per evaluation game: base WP, labels, map, tier, build, week, and the
         ten heroes' as-of profiles (rolling 26-week window; for the post set
         also frozen at the snapshot end, and 8-week and all-history windows)
         plus the as-of hero x map x tier win rate (control).
  fit    team features, offset logistic models, test once, CIs, ablations,
         tier/map splits, drift, coefficients. -> $BSP/results/bsp_results.json

Evaluation designs
  POST (time-based, primary): games dated 2026-05-23 to 2026-09-27, after the
       snapshot paper 1's WP was trained on. Offset-train 05-23..07-12,
       validation 07-13..08-09, test 08-10..09-27.
  P1 (paper 1's own split): offset fit on paper 1's validation games, model
       choices by 5-fold week-block CV inside them, test once on paper 1's
       test games. Paper 1's split is a replay-hash split inside the
       snapshot, not a time split.

Every model: logit P(team 0 wins) = b + a logit(p_base) + w . x, with every x
a team-0-minus-team-1 difference (antisymmetric), L2 on w only. The
reference is the recalibrated base (b, a only) fit on the same rows, so a
lift is never just recalibration of a stale base.
"""
import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import bsp_profiles as BP

BSP = BP.BSP
RES = os.path.join(BSP, "results")
os.makedirs(RES, exist_ok=True)
EPOCH = datetime.date(1970, 1, 1)


def dnum(s):
    return (datetime.date.fromisoformat(s) - EPOCH).days


POST0, TR1, VA1, POST1 = dnum("2026-05-23"), dnum("2026-07-12"), dnum("2026-08-09"), dnum("2026-09-27")
L_MAIN = 26
D = {n: i for i, n in enumerate(BP.DIM_NAMES)}


# ───────────────────────────────────────────────────────────── prep

def prep():
    t0 = time.time()
    G = BP.load_games()
    C = BP.build_cells(G)
    print(f"cells built ({time.time() - t0:.0f}s): {json.dumps(C['info'])[:600]}", flush=True)
    PS, PN = BP.cumulate(C)
    w0 = C["w0"]
    del C
    PG, PW, w0r = BP.wr_cells(G)
    W = np.load(os.path.join(BP.DATA, "wp_base.npz"))
    out = {}
    snap_week = int(BP.week_of(POST0))
    for name in ("p1val", "p1test", "post"):
        rid = W[f"{name}_rid"]
        j = np.searchsorted(G["replay_id"], rid)
        j = np.minimum(j, len(G["replay_id"]) - 1)
        hit = G["replay_id"][j] == rid
        ok = hit & (G["map"][j] >= 0) & (G["tier"][j] >= 0) & (G["t0"][j] >= 0).all(1) & (G["t1"][j] >= 0).all(1)
        y_db = (G["winner"][j] == 0).astype(np.float32)
        agree = float((y_db[ok] == W[f"{name}_y"][ok]).mean())
        print(f"{name}: {len(rid):,} games, {ok.sum():,} joined; label agreement {agree:.5f}", flush=True)
        sel = np.where(ok)[0]
        jj = j[sel]
        heroes = np.stack([G["t0"][jj], G["t1"][jj]], 1).astype(np.int64)      # (n,2,5)
        mp, tr = G["map"][jj].astype(np.int64), G["tier"][jj].astype(np.int64)
        day = G["day"][jj]
        wk = BP.week_of(day)
        rec = {"rid": rid[sel], "y": W[f"{name}_y"][sel], "p": W[f"{name}_p"][:, sel], "day": day,
               "map": mp, "tier": tr, "heroes": heroes, "version": G["version"][jj]}
        windows = {"roll": L_MAIN}
        if name == "post":
            windows.update({"roll8": 8, "rollall": 0, "frozen": L_MAIN})
        for tag, L in windows.items():
            P = np.zeros(heroes.shape + (len(BP.DIMS),), np.float32)
            n_hmt = np.zeros(heroes.shape, np.float32)
            WR = np.zeros(heroes.shape, np.float32)
            weeks = np.full_like(wk, snap_week) if tag == "frozen" else wk
            for w in np.unique(weeks):
                m = weeks == w
                prof, ncell, _ = BP.profile_table(PS, PN, w0, int(w), L)
                h = heroes[m]
                P[m] = prof[h, mp[m][:, None, None], tr[m][:, None, None]]
                n_hmt[m] = ncell[h, mp[m][:, None, None], tr[m][:, None, None]]
                wr = BP.wr_table(PG, PW, w0r, int(w), L)
                WR[m] = wr[h, mp[m][:, None, None], tr[m][:, None, None]]
            rec[f"P_{tag}"], rec[f"n_{tag}"], rec[f"wr_{tag}"] = P, n_hmt, WR
            print(f"  {name} {tag}: median cell games {np.median(n_hmt):.0f} ({time.time() - t0:.0f}s)", flush=True)
        np.savez(os.path.join(BSP, "data", f"prep_{name}.npz"), **rec)
        out[name] = {"n": int(len(sel)), "label_agreement": agree}
    json.dump(out, open(os.path.join(RES, "prep_meta.json"), "w"), indent=1)


def load_prep(name):
    z = np.load(os.path.join(BSP, "data", f"prep_{name}.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


# ───────────────────────────────────────────────────────────── features

SUM_DIMS = ["dmg", "siege", "heal", "self_heal", "shield", "taken", "xp", "cc", "merc", "deaths",
            "escapes", "tf_dmg"]
MAX_DIMS = ["dmg", "heal", "cc", "taken", "self_heal"]
RICH_DIMS = SUM_DIMS + ["stun", "root", "silence", "struct", "globes", "dead_share"]


class Featurizer:
    """Thresholds and scales are fit once on the offset-train rows and frozen."""

    def fit(self, P):
        A = P.reshape(-1, 5, P.shape[-1])
        tmax = A.max(1)
        self.thr = {k: 0.5 * float(np.median(tmax[:, D[k]])) for k in ("heal", "cc", "self_heal", "taken")}
        S = A.sum(1)
        self.mu = S.mean(0)
        self.sd = S.std(0) + 1e-9
        self.hmu = A.reshape(-1, A.shape[-1]).mean(0)
        self.hsd = A.reshape(-1, A.shape[-1]).std(0) + 1e-9
        return self

    def team(self, A):
        """A (n,5,Dm) -> dict of per-team features (n,)."""
        f = {}
        S = A.sum(1)
        zS = (S - self.mu) / self.sd
        srt = np.sort(A, 1)
        for k in SUM_DIMS:
            f[f"SUM:{k}"] = S[:, D[k]]
        for k in MAX_DIMS:
            f[f"MAX:{k}"] = srt[:, -1, D[k]]
        heal = A[:, :, D["heal"]]
        nh = (heal >= self.thr["heal"]).sum(1)
        ncc = (A[:, :, D["cc"]] >= self.thr["cc"]).sum(1)
        nsus = (A[:, :, D["self_heal"]] >= self.thr["self_heal"]).sum(1)
        nfront = (A[:, :, D["taken"]] >= self.thr["taken"]).sum(1)
        cc = A[:, :, D["cc"]]
        sh = cc / np.maximum(cc.sum(1, keepdims=True), 1e-9)
        f["COV:n_healers"] = nh.astype(np.float64)
        f["COV:no_healer"] = (nh == 0).astype(np.float64)
        f["COV:two_plus_healers"] = (nh >= 2).astype(np.float64)
        f["COV:n_heavy_cc"] = ncc.astype(np.float64)
        f["COV:second_cc"] = srt[:, -2, D["cc"]]
        f["COV:cc_concentration"] = (sh ** 2).sum(1)
        f["COV:second_self_heal"] = srt[:, -2, D["self_heal"]]
        f["COV:two_plus_sustain"] = (nsus >= 2).astype(np.float64)
        f["COV:no_frontline"] = (nfront == 0).astype(np.float64)
        f["INT:cc_x_dmg"] = zS[:, D["cc"]] * zS[:, D["dmg"]]
        f["INT:heal_x_taken"] = zS[:, D["heal"]] * zS[:, D["taken"]]
        f["INT:sustain_x_heal"] = zS[:, D["self_heal"]] * zS[:, D["heal"]]
        return f, zS

    def small(self, P):
        """P (n,2,5,Dm) -> (X (n,k) diffs, names)."""
        fa, za = self.team(P[:, 0])
        fb, zb = self.team(P[:, 1])
        names = list(fa)
        X = [fa[k] - fb[k] for k in names]
        # cross-team terms (own x opponent minus the reverse)
        for a, b in (("cc", "escapes"), ("dmg", "heal")):
            names.append(f"INT:{a}_vs_opp_{b}")
            X.append(za[:, D[a]] * zb[:, D[b]] - zb[:, D[a]] * za[:, D[b]])
        return np.stack(X, 1), names

    def rich(self, P, mp, tr, nmaps):
        """richer regularized set: sum/max/min/second of RICH_DIMS, within-team
        products and cross-team products of z-sums, map x sum and tier x sum."""
        X, names = [], []
        za_all, zb_all = [], []
        for t in (0, 1):
            A = P[:, t]
            Z = (A - self.hmu) / self.hsd
            za_all.append(Z)
        Za, Zb = za_all
        idx = [D[k] for k in RICH_DIMS]
        for nm, fn in (("sum", lambda Z: Z.sum(1)), ("max", lambda Z: Z.max(1)), ("min", lambda Z: Z.min(1)),
                       ("second", lambda Z: np.sort(Z, 1)[:, -2])):
            a, b = fn(Za[:, :, idx]), fn(Zb[:, :, idx])
            for i, k in enumerate(RICH_DIMS):
                names.append(f"R{nm}:{k}")
                X.append(a[:, i] - b[:, i])
        sa, sb = Za[:, :, idx].sum(1) / np.sqrt(5), Zb[:, :, idx].sum(1) / np.sqrt(5)
        ns = len(SUM_DIMS)          # SUM_DIMS lead RICH_DIMS
        for i in range(ns):
            for j in range(i, ns):
                names.append(f"Rwith:{RICH_DIMS[i]}*{RICH_DIMS[j]}")
                X.append(sa[:, i] * sa[:, j] - sb[:, i] * sb[:, j])
        for i in range(ns):
            for j in range(i + 1, ns):
                names.append(f"Rcross:{RICH_DIMS[i]}_vs_opp_{RICH_DIMS[j]}")
                X.append(sa[:, i] * sb[:, j] - sb[:, i] * sa[:, j])
        for m in range(nmaps):
            on = (mp == m).astype(np.float64)
            for i, k in enumerate(SUM_DIMS):
                names.append(f"Rmap{m}:{k}")
                X.append(on * (sa[:, idx.index(D[k])] - sb[:, idx.index(D[k])]))
        for t in range(3):
            on = (tr == t).astype(np.float64)
            for k in SUM_DIMS:
                names.append(f"Rtier{t}:{k}")
                X.append(on * (sa[:, idx.index(D[k])] - sb[:, idx.index(D[k])]))
        return np.stack([x.astype(np.float32) for x in X], 1), names


def wr_feature(WR):
    lw = np.log(np.clip(WR, 1e-3, 1 - 1e-3) / (1 - np.clip(WR, 1e-3, 1 - 1e-3)))
    return (lw[:, 0].mean(1) - lw[:, 1].mean(1))[:, None], ["W:asof_hero_map_tier_wr"]


# ───────────────────────────────────────────────────────────── models

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _Z(lb, X, sc, a, b):
    n = b - a
    cols = [np.ones((n, 1)), lb[a:b, None]]
    if X is not None and X.shape[1]:
        cols.append(X[a:b].astype(np.float64) / sc)
    return np.concatenate(cols, 1)


def fit_offset(lb, X, y, lam=0.0, sc=None, iters=30, chunk=40000):
    """logit P = b + a lb + (X / sc) w; L2 penalty lam * n / 2 on w only.
    Chunked so a wide X never exists in float64 at full length. -> coef, cov"""
    n = len(y)
    k = 2 + (X.shape[1] if X is not None else 0)
    pen = np.full(k, lam * n)
    pen[:2] = 0.0
    w = np.zeros(k)
    w[1] = 1.0
    for _ in range(iters):
        g = pen * w
        Hs = np.diag(pen)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            Z = _Z(lb, X, sc, a, b)
            p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
            g = g + Z.T @ (p - y[a:b])
            Hs = Hs + (Z * (p * (1 - p))[:, None]).T @ Z
        step = np.linalg.solve(Hs, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w, np.linalg.inv(Hs)


def predict(w, lb, X, sc=None, chunk=40000):
    out = np.empty(len(lb))
    for a in range(0, len(lb), chunk):
        b = min(len(lb), a + chunk)
        out[a:b] = _Z(lb, X, sc, a, b) @ w
    return np.clip(1 / (1 + np.exp(-out)), 1e-7, 1 - 1e-7)


def ll_vec(p, y):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def block_ci(delta, blocks, B=4000, seed=0):
    """mean of delta with a week-block bootstrap 95% CI."""
    u, inv = np.unique(blocks, return_inverse=True)
    s = np.bincount(inv, weights=delta)
    c = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    r = rng.integers(0, len(u), (B, len(u)))
    boots = s[r].sum(1) / c[r].sum(1)
    return float(delta.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), int(len(u))


def compare(pa, pb, y, blocks):
    """pb vs pa: log-loss lift x1000 (positive = pb better) and accuracy pp."""
    dl = (ll_vec(pa, y) - ll_vec(pb, y)) * 1000
    da = (((pb > .5) == (y > .5)).astype(float) - ((pa > .5) == (y > .5)).astype(float)) * 100
    return {"ll_lift_x1000": block_ci(dl, blocks), "acc_lift_pp": block_ci(da, blocks)}


def summary(p, y):
    return {"ll_x1000": float(ll_vec(p, y).mean() * 1000), "acc": float(((p > .5) == (y > .5)).mean() * 100),
            "n": int(len(y))}


# ───────────────────────────────────────────────────────────── fit

FAMILIES = ("SUM", "MAX", "COV", "INT")
LAMS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1)


def scale_fit(X):
    s = np.zeros(X.shape[1])
    for a in range(0, len(X), 50000):
        s += (X[a:a + 50000].astype(np.float64) ** 2).sum(0)
    s = np.sqrt(s / len(X))
    return np.where(s > 1e-6, s, np.inf)      # a constant-zero column contributes nothing


def design_post():
    R = load_prep("post")
    tr = R["day"] <= TR1
    va = (R["day"] > TR1) & (R["day"] <= VA1)
    te = R["day"] > VA1
    return R, tr, va, te


def run_design(name, R, tr, va, te, cv_folds=None, extra_windows=()):
    """One design. cv_folds: if given, validation = week-block CV inside tr (P1 design)."""
    out = {"n": {"train": int(tr.sum()), "val": int(va.sum()) if va is not None else 0, "test": int(te.sum())}}
    y = R["y"].astype(np.float64)
    lb = logit(R["p"][0].astype(np.float64))
    wk = BP.week_of(R["day"])
    fz = Featurizer().fit(R["P_roll"][tr])
    Xs, names = fz.small(R["P_roll"].astype(np.float64))
    Xw, wn = wr_feature(R["wr_roll"])
    nmaps = int(R["map"].max()) + 1

    # validation protocol
    if cv_folds is None:
        splits = [(tr, va)]
    else:
        wtr = wk[tr]
        uw = np.unique(wtr)
        fold_of_week = {w: i % cv_folds for i, w in enumerate(np.random.default_rng(1).permutation(uw))}
        f = np.array([fold_of_week.get(w, -1) for w in wk])
        splits = [(tr & (f != k), tr & (f == k)) for k in range(cv_folds)]

    def val_ll(Xall, lam):
        tot, cnt = 0.0, 0
        for a, b in splits:
            Xa = Xall[a] if Xall is not None else None
            sc = scale_fit(Xa) if Xa is not None else None
            w, _ = fit_offset(lb[a], Xa, y[a], lam, sc)
            p = predict(w, lb[b], Xall[b] if Xall is not None else None, sc)
            tot += ll_vec(p, y[b]).sum()
            cnt += b.sum()
        return tot / cnt * 1000

    def fit_test(Xall, lam):
        Xa = Xall[tr] if Xall is not None else None
        sc = scale_fit(Xa) if Xa is not None else None
        w, cov = fit_offset(lb[tr], Xa, y[tr], lam, sc)
        return predict(w, lb, Xall, sc), w, cov, sc

    fam_cols = {fam: [i for i, n in enumerate(names) if n.startswith(fam + ":")] for fam in FAMILIES}
    cand = {"base_recal": None, "W": Xw, "small": Xs, "small+W": np.concatenate([Xs, Xw], 1)}
    for fam in FAMILIES:
        cand[f"only_{fam}"] = Xs[:, fam_cols[fam]]
        cand[f"drop_{fam}"] = np.delete(Xs, fam_cols[fam], 1)
    t0 = time.time()
    Xr, rnames = fz.rich(R["P_roll"].astype(np.float64), R["map"], R["tier"], nmaps)
    print(f"  {name}: rich features {Xr.shape} ({time.time() - t0:.0f}s)", flush=True)

    vals = {}
    for k, X in cand.items():
        lams = (0.0,) if X is None else (0.0, 1e-4, 1e-3, 1e-2)
        vals[k] = {lam: val_ll(X, lam) for lam in lams}
        print(f"  val {k}: " + " ".join(f"{l:g}:{v:.3f}" for l, v in vals[k].items()), flush=True)
    vals["rich"] = {lam: val_ll(Xr, lam) for lam in (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)}
    print("  val rich: " + " ".join(f"{l:g}:{v:.3f}" for l, v in vals["rich"].items()), flush=True)
    XrW = np.concatenate([Xr, Xw], 1)
    vals["rich+W"] = {lam: val_ll(XrW, lam) for lam in (1e-2, 3e-2, 1e-1, 3e-1, 1.0)}
    print("  val rich+W: " + " ".join(f"{l:g}:{v:.3f}" for l, v in vals["rich+W"].items()), flush=True)
    best_lam = {k: min(v, key=v.get) for k, v in vals.items()}
    out["val_ll_x1000"] = {k: {str(l): x for l, x in v.items()} for k, v in vals.items()}
    out["best_lam"] = {k: float(v) for k, v in best_lam.items()}
    # selection among the profile models (W excluded: it is the control)
    prof_models = ["small", "rich"]
    selected = min(prof_models, key=lambda k: vals[k][best_lam[k]])
    out["selected"] = selected
    allX = dict(cand)
    allX["rich"] = Xr
    allX["rich+W"] = XrW

    # test once
    preds, coefs = {}, {}
    for k, X in allX.items():
        p, w, cov, sc = fit_test(X, best_lam[k])
        preds[k] = p
        coefs[k] = (w, cov, sc)
    yt, bt = y[te], wk[te]
    raw = np.clip(R["p"][0][te].astype(np.float64), 1e-7, 1 - 1e-7)
    out["test"] = {"base_raw": summary(raw, yt), "base_seedmean_raw": summary(
        np.clip(R["p"].mean(0)[te].astype(np.float64), 1e-7, 1 - 1e-7), yt)}
    for k, p in preds.items():
        out["test"][k] = summary(p[te], yt)
    ref = preds["base_recal"][te]
    out["lift_vs_recal"] = {k: compare(ref, p[te], yt, bt) for k, p in preds.items() if k != "base_recal"}
    out["lift_vs_raw"] = {"base_recal": compare(raw, ref, yt, bt)}
    out["lift_over_W"] = {"small+W": compare(preds["W"][te], preds["small+W"][te], yt, bt),
                          "rich+W": compare(preds["W"][te], preds["rich+W"][te], yt, bt)}
    sel = preds[selected][te]
    out["by_tier"] = {}
    for t, tn in enumerate(["low", "mid", "high"]):
        m = R["tier"][te] == t
        if m.sum() > 1000:
            out["by_tier"][tn] = {"n": int(m.sum()), "small": compare(ref[m], preds["small"][te][m], yt[m], bt[m]),
                                  "selected": compare(ref[m], sel[m], yt[m], bt[m])}
    out["by_map"] = {}
    maps = list(BP.load_games_maps())
    for mi in range(nmaps):
        m = R["map"][te] == mi
        if m.sum() > 1000:
            out["by_map"][maps[mi]] = {"n": int(m.sum()), "small": compare(ref[m], preds["small"][te][m], yt[m], bt[m]),
                                       "selected": compare(ref[m], sel[m], yt[m], bt[m])}
    # coefficients of the small model (per SD of each feature, logit units) with SE
    w, cov, sc = coefs["small"]
    se = np.sqrt(np.diag(cov))
    out["coef_small"] = {n: {"per_sd": float(w[2 + i]), "se": float(se[2 + i]), "sd": float(sc[i])}
                         for i, n in enumerate(names)}
    out["coef_small"]["_a_base"] = {"per_sd": float(w[1]), "se": float(se[1])}
    w, cov, sc = coefs["rich"]
    order = np.argsort(-np.abs(w[2:]))[:30]
    out["coef_rich_top"] = [{"name": rnames[i], "per_sd": float(w[2 + i])} for i in order]
    out["rich_n_features"] = len(rnames)
    out["featurizer"] = {"thr": fz.thr}
    # test-period mean feature values for the interpretation section
    out["small_feature_sd"] = {n: float(s) for n, s in zip(names, coefs["small"][2])}
    out["_preds"] = {"ref": ref, "small": preds["small"][te], "sel": sel, "te": te}

    # drift (post only): frozen vs rolling, window length, build age, changed heroes
    if extra_windows:
        out["drift"] = drift(R, tr, te, fz, lb, y, wk, best_lam["small"], preds, names)
    return out


def drift(R, tr, te, fz, lb, y, wk, lam, preds, names):
    res = {}
    ref = preds["base_recal"]
    for tag in ("frozen", "roll8", "rollall"):
        X, _ = fz.small(R[f"P_{tag}"].astype(np.float64))
        sc = scale_fit(X[tr])
        w, _ = fit_offset(lb[tr], X[tr], y[tr], lam, sc)
        p = predict(w, lb, X, sc)
        res[f"small_{tag}"] = {"test": summary(p[te], y[te]),
                               "lift_vs_recal": compare(ref[te], p[te], y[te], wk[te]),
                               "vs_roll26": compare(p[te], preds["small"][te], y[te], wk[te])}
        if tag == "frozen":
            p_frozen = p
    # build age: days since the build's first day with >= 200 games in the post set
    ver = R["version"]
    age = np.zeros(len(ver))
    for v in np.unique(ver):
        m = ver == v
        days, cnt = np.unique(R["day"][m], return_counts=True)
        first = days[np.argmax(cnt >= 200)] if (cnt >= 200).any() else days.min()
        age[m] = R["day"][m] - first
    res["build_first_days"] = {str(v): int(R["day"][ver == v].min()) for v in np.unique(ver)}
    for lab, m in (("build_age_0_13", age < 14), ("build_age_14plus", age >= 14)):
        mm = te & m
        if mm.sum() > 1000:
            res[lab] = {"n": int(mm.sum()),
                        "rolling_vs_recal": compare(ref[mm], preds["small"][mm], y[mm], wk[mm]),
                        "rolling_vs_frozen": compare(p_frozen[mm], preds["small"][mm], y[mm], wk[mm])}
    # games whose heroes' profiles moved most since the snapshot (standardized distance)
    dz = (R["P_roll"] - R["P_frozen"]) / fz.hsd
    chg = np.sqrt((dz ** 2).mean(-1)).mean((1, 2))
    q = np.quantile(chg[te], 0.75)
    res["change_q75"] = float(q)
    for lab, m in (("changed_top25", chg >= q), ("changed_rest", chg < q)):
        mm = te & m
        res[lab] = {"n": int(mm.sum()),
                    "rolling_vs_recal": compare(ref[mm], preds["small"][mm], y[mm], wk[mm]),
                    "rolling_vs_frozen": compare(p_frozen[mm], preds["small"][mm], y[mm], wk[mm])}
    # heroes whose profile moved most (test period mean over slots)
    H = R["heroes"][te].reshape(-1)
    per = np.sqrt((dz[te] ** 2).mean(-1)).reshape(-1)
    nh = int(H.max()) + 1
    s = np.bincount(H, weights=per, minlength=nh) / np.maximum(np.bincount(H, minlength=nh), 1)
    heroes = BP.load_games_heroes()
    top = np.argsort(-s)[:10]
    res["most_changed_heroes"] = [(str(heroes[i]), float(s[i])) for i in top]
    return res


def fit_all():
    BP_res = {}
    t0 = time.time()
    R, tr, va, te = design_post()
    print(f"POST: train {tr.sum():,} val {va.sum():,} test {te.sum():,}", flush=True)
    BP_res["POST"] = run_design("POST", R, tr, va, te, extra_windows=True)
    print(f"POST done ({time.time() - t0:.0f}s)", flush=True)
    del R
    A, B = load_prep("p1val"), load_prep("p1test")
    R = {k: np.concatenate([A[k], B[k]], -1 if k == "p" else 0) for k in A if k in B}
    n1 = len(A["y"])
    tr = np.arange(len(R["y"])) < n1
    te = ~tr
    BP_res["P1"] = run_design("P1", R, tr, None, te, cv_folds=5)
    print(f"P1 done ({time.time() - t0:.0f}s)", flush=True)
    for k in BP_res:
        BP_res[k].pop("_preds", None)
    json.dump(BP_res, open(os.path.join(RES, "bsp_results.json"), "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "fit"])
    a = ap.parse_args()
    {"prep": prep, "fit": fit_all}[a.cmd]()


if __name__ == "__main__":
    main()
