"""
P3 H1: do patches reshuffle individual skill on the patched hero, in a jump
at the boundary?

Boundaries: the 27 sizable build boundaries of paper 2's detector study
(drift2026/results/w3_changepoints.json). Changed heroes per boundary:
  notes     patch-note truth (bug-fix-only and ARAM-only mentions excluded)
  detector  heroes the win-rate detector flagged (detected.wr_bh)
  unchanged every hero in neither set
A boundary sits at the first build of its new side. Segments run between
consecutive boundaries, capped at 90 days on each side of a boundary.

Population part removed first: u = r_adj - mean r_adj of (build, hero), so a
hero's patch-wide shift (and the WP's one-build lag in catching it) is gone
and only the player-specific part is tested.

A. Difference-in-differences on cells (boundary, player, hero) with >= 10
   games on each side. Split-half design, so no game-noise model is needed
   (wp(1 - wp) overstates the noise of in-sample residuals by about 2%,
   which is as large as the signal here): each side's games are split into
   odd and even games in play order; means are centered within (boundary,
   hero, side, half) and scaled by g / (g - 1) for the centering, then
     true pre/post variance  = mean(pre_odd * pre_even), same for post
     cross covariance        = mean(pre * post)
     stability               = cross / sqrt(var_pre * var_post)
     excess change variance  = var_pre + var_post - 2 cross
   compared across changed and unchanged heroes, with player-cluster
   bootstrap CIs. Raw Spearman rank correlations (groups with >= 30 players
   per boundary x hero) are reported alongside. With --ab-only only A and B
   run (output results/p3_ph_patch_ab.json).
B. Magnitude: excess change variance by the hero's aggregate win-rate shift
   |WR_post - WR_pre| at that boundary.
C. State-space model with an extra jump variance J added to the hero-specific
   state of changed heroes when a player's history crosses a boundary. J is
   fit by likelihood (other parameters fixed at the p3_sd_kalman drift fit)
   on the random-40k and heavy-6k E samples; placebo: the same jump applied
   to unchanged heroes instead.
D. Out of sample after the 2026-04-20 patch (2.55.16, inside V2): lag-1-day
   predictions with and without J; per-slot log-loss gain on changed-hero
   slots in the following 30 days; cell coverage of 80% intervals frozen at
   the day before the patch.

Run (from training/, 4 threads):
  NUMBA_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_ph_patch.py
Output: results/p3_ph_patch.json
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING)
import numpy as np
from numba import njit, prange
import p3_hs_core as C
import p3_sd_kalman as K
from p3_hs_fit import prepare
from p3_hs_eval import slot_stats

W3 = os.path.join(TRAINING, "drift2026", "results", "w3_changepoints.json")
CAP_DAYS = 90
MIN_SIDE = 10
Z80 = 1.2815516


def slot_builds(d):
    w = np.load(C.WP)
    order = np.argsort(w["replay_ids"])
    return w["build_idx"][order][d["g"]].astype(np.int64)


def boundaries(names):
    from drift2026 import common
    builds = common.load_patch_index()["builds"]
    w = json.load(open(W3))
    idx = {n: i for i, n in enumerate(names)}
    out = []
    for b in w["boundaries"]:
        ib = b["intermediate_builds"]
        if isinstance(ib, str):
            ib = json.loads(ib.replace("'", '"'))
        pos = builds.index(ib[0])
        notes = set(b.get("truth", {}).get("heroes", []) or [])
        det = set(b["detected"].get("wr_bh", []))
        out.append({"pos": pos, "new_build": b["new_build"], "prev_build": b["prev_build"],
                    "notes": np.array([n in notes for n in names]),
                    "detector": np.array([n in det for n in names])})
    out.sort(key=lambda x: x["pos"])
    return out, builds


# ------------------------------------------------------------------ A, B

def cell_table(d, u, build, bnds):
    H = 90
    rows = []
    days = d["day"]
    for k, b in enumerate(bnds):
        lo = bnds[k - 1]["pos"] if k > 0 else 0
        hi = bnds[k + 1]["pos"] if k + 1 < len(bnds) else 10 ** 6
        pre = (build >= lo) & (build < b["pos"])
        post = (build >= b["pos"]) & (build < hi)
        if not pre.any() or not post.any():
            continue
        t_b = days[post].min()
        pre &= days >= t_b - CAP_DAYS
        post &= days < t_b + CAP_DAYS
        for side, m in ((0, pre), (1, post)):
            key = d["pid"][m] * H + d["hero"][m]
            uk, inv, n = np.unique(key, return_inverse=True, return_counts=True)
            su = np.bincount(inv, weights=u[m])
            sv = np.bincount(inv, weights=d["v"][m])
            sy = np.bincount(inv, weights=d["y"][m])
            # odd/even games in play order within the cell
            oo = np.lexsort((d["replay_id"][m], d["day"][m], inv))
            ks = inv[oo]
            brk = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
            rk = np.arange(len(oo)) - np.repeat(brk, np.diff(np.r_[brk, len(oo)]))
            odd = np.empty(len(oo), bool)
            odd[oo] = rk % 2 == 0
            sa = np.bincount(inv, weights=u[m] * odd, minlength=len(uk))
            na = np.bincount(inv, weights=odd, minlength=len(uk))
            rows.append((k, side, uk, n, su, sv, sy, sa, na))
    # join pre and post per boundary
    cells = []
    for k in sorted(set(r[0] for r in rows)):
        pr = [r for r in rows if r[0] == k and r[1] == 0][0]
        po = [r for r in rows if r[0] == k and r[1] == 1][0]
        common_keys, ia, ib = np.intersect1d(pr[2], po[2], return_indices=True)
        ok = (pr[3][ia] >= MIN_SIDE) & (po[3][ib] >= MIN_SIDE)
        ck = common_keys[ok]
        a, b_ = ia[ok], ib[ok]
        cells.append(dict(k=np.full(len(ck), k), pid=ck // 90, hero=ck % 90,
                          n0=pr[3][a], n1=po[3][b_],
                          u0=pr[4][a] / pr[3][a], u1=po[4][b_] / po[3][b_],
                          z0=pr[5][a] / pr[3][a] ** 2, z1=po[5][b_] / po[3][b_] ** 2,
                          u0a=pr[7][a] / pr[8][a], u0b=(pr[4][a] - pr[7][a]) / (pr[3][a] - pr[8][a]),
                          u1a=po[7][b_] / po[8][b_], u1b=(po[4][b_] - po[7][b_]) / (po[3][b_] - po[8][b_])))
    cat = {key: np.concatenate([c[key] for c in cells]) for key in cells[0]}
    # hero aggregate win-rate shift per boundary (all players, capped segments)
    agg = {}
    for k in set(r[0] for r in rows):
        pr = [r for r in rows if r[0] == k and r[1] == 0][0]
        po = [r for r in rows if r[0] == k and r[1] == 1][0]
        h0 = np.bincount(pr[2] % 90, weights=pr[6], minlength=90) / np.maximum(
            np.bincount(pr[2] % 90, weights=pr[3], minlength=90), 1)
        h1 = np.bincount(po[2] % 90, weights=po[6], minlength=90) / np.maximum(
            np.bincount(po[2] % 90, weights=po[3], minlength=90), 1)
        agg[k] = h1 - h0
    # center within (boundary, hero, side)
    g = cat["k"] * 90 + cat["hero"]
    ug, gi = np.unique(g, return_inverse=True)
    for s in ("u0", "u1", "u0a", "u0b", "u1a", "u1b"):
        mean = np.bincount(gi, weights=cat[s]) / np.bincount(gi)
        cat[s] = cat[s] - mean[gi]
    cat["gsize"] = np.bincount(gi)[gi]
    cat["gcorr"] = cat["gsize"] / np.maximum(cat["gsize"] - 1, 1)
    cat["dwr"] = np.array([agg[k][h] for k, h in zip(cat["k"], cat["hero"])])
    return cat


def group_stats(c, sel):
    sel = sel & (c["gsize"] >= 3)
    if sel.sum() < 30:
        return None
    g = c["gcorr"][sel]
    v0 = np.mean(g * c["u0a"][sel] * c["u0b"][sel])
    v1 = np.mean(g * c["u1a"][sel] * c["u1b"][sel])
    cov = np.mean(g * c["u0"][sel] * c["u1"][sel])
    exc = v0 + v1 - 2 * cov
    stab = cov / np.sqrt(v0 * v1) if v0 > 0 and v1 > 0 else float("nan")
    return {"cells": int(sel.sum()), "var_pre_pp2": 1e4 * v0, "var_post_pp2": 1e4 * v1,
            "cov_pp2": 1e4 * cov, "stability": float(stab),
            "excess_change_var_pp2": 1e4 * exc}


def spearman_by_group(c, sel):
    out = []
    g = c["k"] * 90 + c["hero"]
    for gg in np.unique(g[sel]):
        m = sel & (g == gg)
        if m.sum() < 30:
            continue
        r0 = np.argsort(np.argsort(c["u0"][m]))
        r1 = np.argsort(np.argsort(c["u1"][m]))
        out.append((np.corrcoef(r0, r1)[0, 1], m.sum()))
    if not out:
        return None
    r, n = np.array(out).T
    return {"groups": int(len(r)), "mean_rank_corr": float(np.average(r, weights=n))}


def did(c, masks, rng, nboot=200):
    res = {}
    for nm, m in masks.items():
        res[nm] = group_stats(c, m)
        if res[nm] is not None:
            res[nm]["spearman"] = spearman_by_group(c, m)
    # player-cluster bootstrap of changed - unchanged
    up, pinv = np.unique(c["pid"], return_inverse=True)
    diffs = {k: {"stability": [], "excess_change_var_pp2": []} for k in masks if k != "unchanged"}
    for _ in range(nboot):
        w = np.bincount(rng.randint(0, len(up), len(up)), minlength=len(up))[pinv]
        idx = np.repeat(np.arange(len(w)), w)
        cb = {k: v[idx] for k, v in c.items()}
        mb = {k: m[idx] for k, m in masks.items()}
        base = group_stats(cb, mb["unchanged"])
        for k in diffs:
            s = group_stats(cb, mb[k])
            if s is None:
                continue
            for q in diffs[k]:
                diffs[k][q].append(s[q] - base[q])
    for k in diffs:
        res[f"{k} - unchanged"] = {
            q: [res[k][q] - res["unchanged"][q], float(np.nanpercentile(v, 2.5)),
                float(np.nanpercentile(v, 97.5))] for q, v in diffs[k].items()}
    return res


# ------------------------------------------------------------------ C, D

@njit(parallel=True, cache=True)
def kalman_jump(starts, ends, tval, day, build, hero, y, v, upd, lidx, lw, s2, lam, hs_off,
                jtab, out_m, out_s2, ll_rows):
    d = s2.shape[0]
    L = lidx.shape[1]
    H = jtab.shape[1]
    for p in prange(starts.shape[0]):
        a, b = starts[p], ends[p]
        x = np.zeros(d)
        P = np.zeros((d, d))
        for i in range(d):
            P[i, i] = s2[i]
        f = np.empty(d)
        pz = np.empty(d)
        t_prev = tval[a]
        b_prev = build[a]
        i = a
        while i < b:
            j = i
            while j < b and day[j] == day[i]:
                j += 1
            dt = tval[i] - t_prev
            if dt > 0:
                for k in range(d):
                    f[k] = np.exp(-lam[k] * dt)
                for k in range(d):
                    x[k] *= f[k]
                    for l in range(d):
                        P[k, l] *= f[k] * f[l]
                    P[k, k] += s2[k] * (1.0 - f[k] * f[k])
                t_prev = tval[i]
            if build[i] > b_prev:
                for bb in range(b_prev + 1, build[i] + 1):
                    for h in range(H):
                        if jtab[bb, h] > 0:
                            P[hs_off + h, hs_off + h] += jtab[bb, h]
                b_prev = build[i]
            for r in range(i, j):
                h = hero[r]
                m = 0.0
                s = 0.0
                for q1 in range(L):
                    m += lw[h, q1] * x[lidx[h, q1]]
                    for q2 in range(L):
                        s += lw[h, q1] * lw[h, q2] * P[lidx[h, q1], lidx[h, q2]]
                out_m[r] = m
                out_s2[r] = s
            for r in range(i, j):
                if not upd[r]:
                    continue
                h = hero[r]
                for k in range(d):
                    acc = 0.0
                    for q1 in range(L):
                        acc += P[k, lidx[h, q1]] * lw[h, q1]
                    pz[k] = acc
                m = 0.0
                S = v[r]
                for q1 in range(L):
                    m += lw[h, q1] * x[lidx[h, q1]]
                    S += lw[h, q1] * pz[lidx[h, q1]]
                e = y[r] - m
                ll_rows[r] = -0.5 * (np.log(2 * np.pi * S) + e * e / S)
                for k in range(d):
                    x[k] += pz[k] * e / S
                for k in range(d):
                    c = pz[k] / S
                    for l in range(d):
                        P[k, l] -= c * pz[l]
            i = j


class JumpFilter(K.Filter):
    def __init__(self, d, meta, V, r_adj, build, rows_mask=None):
        super().__init__(d, meta, V, r_adj, rows_mask)
        self.build = build[self.rows].astype(np.int64)

    def run_jump(self, p, jtab, upd=None):
        sd, lam, br = K.split(p)
        s2 = np.array([sd.get(K.COMPS[c], 0.0) ** 2 for c in K.state_comp()])
        n = len(self.rows)
        m, s, ll = np.zeros(n), np.zeros(n), np.zeros(n)
        u = np.ones(n, bool) if upd is None else upd
        kalman_jump(self.starts, self.ends, self.tval, self.day, self.build, self.hero, self.y,
                    self.v, u, self.lidx, self.lw, s2, self.lam_state(lam, br), K.OFF["hero_s"],
                    jtab, m, s, ll)
        return m, s, ll


def jtab_for(bnds, nb, which, J):
    t = np.zeros((nb + 1, 90))
    for b in bnds:
        if which == "notes":
            m = b["notes"]
        elif which == "detector":
            m = b["detector"]
        elif which == "notes or detector":
            m = b["notes"] | b["detector"]
        else:  # placebo: unchanged heroes
            m = ~(b["notes"] | b["detector"])
        t[b["pos"], m] = J
    return t


def fit_jump(jf, p, bnds, nb, which):
    from scipy.optimize import minimize_scalar
    ll0 = jf.run_jump(p, jtab_for(bnds, nb, which, 0.0))[2].sum()

    def f(z):
        return -jf.run_jump(p, jtab_for(bnds, nb, which, float(np.exp(z))))[2].sum()
    r = minimize_scalar(f, bounds=(np.log(1e-7), np.log(0.01)), method="bounded",
                        options={"xatol": 0.1, "maxiter": 20})
    return float(np.exp(r.x)), float(-r.fun - ll0)


def main():
    t0 = time.time()
    d = C.load_slots()
    meta = C.hero_meta(d["hero_names"])
    names = [str(x) for x in d["hero_names"]]
    kz = np.load(K.KOUT)
    e_mask, n_p, n_ph, table, r_adj = prepare(d)
    build = slot_builds(d)
    bnds, builds = boundaries(names)
    nb = len(builds)
    out = {"boundaries": [{"new_build": b["new_build"], "pos": b["pos"],
                           "n_notes": int(b["notes"].sum()), "n_detector": int(b["detector"].sum())}
                          for b in bnds]}
    # population part: per (build, hero) mean of r_adj
    bh = build * 90 + d["hero"]
    ubh, inv = np.unique(bh, return_inverse=True)
    mean_bh = np.bincount(inv, weights=r_adj) / np.bincount(inv)
    u = r_adj - mean_bh[inv]

    # ---------- A, B
    c = cell_table(d, u, build, bnds)
    notes = np.array([bnds[k]["notes"][h] for k, h in zip(c["k"], c["hero"])])
    det = np.array([bnds[k]["detector"][h] for k, h in zip(c["k"], c["hero"])])
    masks = {"unchanged": ~notes & ~det, "notes": notes, "detector": det, "notes and detector": notes & det}
    rng = np.random.RandomState(0)
    out["did"] = did(c, masks, rng)
    big = (c["n0"] >= 20) & (c["n1"] >= 20)
    out["did_min20"] = did({k: v[big] for k, v in c.items()},
                           {k: m[big] for k, m in masks.items()}, rng)
    for k, v in out["did_min20"].items():
        print("min20", k, json.dumps(v)[:300], flush=True)
    out["n_boundaries_used"] = int(len(np.unique(c["k"])))
    for k, v in out["did"].items():
        print(k, json.dumps(v)[:400], flush=True)
    mag = {}
    adw = np.abs(c["dwr"])
    for lo, hi in ((0, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 1)):
        for nm, m in (("all heroes", np.ones(len(adw), bool)), ("notes-changed", notes)):
            s = m & (adw >= lo) & (adw < hi)
            gs = group_stats(c, s)
            if gs:
                mag[f"{nm} |dWR| {100 * lo:.0f}-{100 * hi:.0f}pp"] = gs
                print(f"magnitude {nm} |dWR| {100 * lo:.0f}-{100 * hi:.0f}pp: cells {gs['cells']} "
                      f"stability {gs['stability']:.3f} excess {gs['excess_change_var_pp2']:.2f}", flush=True)
    out["magnitude"] = mag
    # weighted regression of cell excess change on |dWR| (all heroes)
    ex = c["gcorr"] * (c["u0a"] * c["u0b"] + c["u1a"] * c["u1b"] - 2 * c["u0"] * c["u1"])
    X = np.column_stack([np.ones(len(adw)), 100 * adw])
    beta = np.linalg.lstsq(X, 1e4 * ex, rcond=None)[0]
    bs = []
    up, pinv = np.unique(c["pid"], return_inverse=True)
    for _ in range(200):
        w = np.bincount(rng.randint(0, len(up), len(up)), minlength=len(up))[pinv].astype(float)
        sw = np.sqrt(w)
        bs.append(np.linalg.lstsq(X * sw[:, None], 1e4 * ex * sw, rcond=None)[0][1])
    out["magnitude_slope_pp2_per_pp_dWR"] = [float(beta[1]), float(np.percentile(bs, 2.5)),
                                             float(np.percentile(bs, 97.5))]
    print("magnitude slope", out["magnitude_slope_pp2_per_pp_dWR"], flush=True)
    if "--ab-only" in sys.argv:
        # likelihood profile of the jump size (random 40k E sample)
        with open(os.path.join(C.RESULTS, "p3_sd_fit.json")) as f:
            fits = json.load(f)
        rng2 = np.random.RandomState(11)
        fitp = kz["fit_players"]
        nE = np.bincount(d["pid"][e_mask], minlength=int(d["n_players"]))
        samp = np.zeros(int(d["n_players"]), bool)
        samp[rng2.choice(np.flatnonzero(fitp & (nE > 0)), 40000, replace=False)] = True
        p = fits["random 40k"]["fits"]["drift days"]["params"]
        jf = JumpFilter(d, meta, kz["V_cf2"], r_adj, build, e_mask & samp[d["pid"]])
        base = jf.run_jump(p, jtab_for(bnds, nb, "notes", 0.0))[2].sum()
        prof = {}
        for which in ("notes", "detector", "placebo (unchanged)"):
            for sdpp in (0.5, 1.0, 2.0, 3.0):
                ll = jf.run_jump(p, jtab_for(bnds, nb, which, (sdpp / 100) ** 2))[2].sum()
                prof[f"{which} | J sd {sdpp}pp"] = float(ll - base)
                print(f"profile {which} J sd {sdpp}pp: dLL {ll - base:+.1f}", flush=True)
        out["jump_profile_random40k"] = prof
        with open(os.path.join(C.RESULTS, "p3_ph_patch_ab.json"), "w") as f:
            json.dump(out, f, indent=1, default=float)
        return

    # ---------- C
    with open(os.path.join(C.RESULTS, "p3_sd_fit.json")) as f:
        fits = json.load(f)
    rng2 = np.random.RandomState(11)
    fitp = kz["fit_players"]
    nE = np.bincount(d["pid"][e_mask], minlength=int(d["n_players"]))
    samp = np.zeros(int(d["n_players"]), bool)
    samp[rng2.choice(np.flatnonzero(fitp & (nE > 0)), 40000, replace=False)] = True
    heavy = np.zeros(int(d["n_players"]), bool)
    cand = np.flatnonzero(fitp & (nE >= 300))
    heavy[rng2.choice(cand, min(6000, len(cand)), replace=False)] = True
    out["jump_fit"] = {}
    for sname, sm in (("random 40k", samp), ("heavy 6k (>=300 games)", heavy)):
        p = fits[sname]["fits"]["drift days"]["params"]
        jf = JumpFilter(d, meta, kz["V_cf2"], r_adj, build, e_mask & sm[d["pid"]])
        for which in ("notes", "detector", "notes or detector", "placebo (unchanged)"):
            J, gain = fit_jump(jf, p, bnds, nb, which)
            out["jump_fit"][f"{sname} | {which}"] = {"J_sd_pp": 100 * np.sqrt(J), "ll_gain": gain,
                                                     "hero_s_sd_pp": 100 * p["sd_hero_s"]}
            print(f"jump {sname} | {which}: J sd {100 * np.sqrt(J):.2f}pp, ll gain {gain:+.1f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---------- D: out of sample after 2.55.16
    bpos = [b for b in bnds if b["new_build"].startswith("2.55.16")]
    out["oos"] = {}
    if bpos:
        b16 = bpos[0]
        p = fits["random 40k"]["fits"]["drift days"]["params"]
        key = "random 40k | notes"
        J = (out["jump_fit"][key]["J_sd_pp"] / 100) ** 2
        jf = JumpFilter(d, meta, kz["V_cf2"], r_adj, build)
        t_b = int(d["day"][build >= b16["pos"]].min())
        mu = table[C.exp_bins(n_p, n_ph)]
        rows = jf.rows
        changed_slot = b16["notes"][d["hero"]]
        win = (d["day"] >= t_b) & (d["day"] < t_b + 30) & (build >= b16["pos"])
        res = {}
        for lab, jt in (("no jump", jtab_for(bnds, nb, "notes", 0.0)),
                        ("jump (notes)", jtab_for(bnds, nb, "notes", J))):
            m, s, _ = jf.run_jump(p, jt)
            pm = np.empty(len(d["pid"]))
            ps = np.empty(len(d["pid"]))
            pm[rows], ps[rows] = m, s
            r_ = {}
            for nm, sel in (("changed heroes", win & changed_slot), ("unchanged heroes", win & ~changed_slot)):
                r_[nm] = slot_stats(d["r"][sel], d["wp"][sel], (mu + pm)[sel], np.sqrt(ps[sel]))
            # frozen coverage: state at t_b - 1, cells with >= 10 games in window
            upd = d["day"][rows] < t_b
            mf, sf, _ = jf.run_jump(p, jt, upd)
            pmf = np.empty(len(d["pid"]))
            psf = np.empty(len(d["pid"]))
            pmf[rows], psf[rows] = mf, sf
            for nm, sel in (("changed heroes", win & changed_slot), ("unchanged heroes", win & ~changed_slot)):
                key_ = d["pid"][sel] * 90 + d["hero"][sel]
                uk, inv, nf = np.unique(key_, return_inverse=True, return_counts=True)
                rbar = np.bincount(inv, weights=r_adj[sel]) / nf
                noise = np.bincount(inv, weights=d["v"][sel]) / nf ** 2
                mbar = np.bincount(inv, weights=pmf[sel]) / nf
                s2b = np.bincount(inv, weights=psf[sel]) / nf
                ok = nf >= 10
                z = np.abs(rbar - mbar) / np.sqrt(s2b + noise)
                err = (rbar - mbar) ** 2 - noise
                r_[nm]["coverage_cells"] = int(ok.sum())
                r_[nm]["cover80"] = float((z[ok] < Z80).mean())
                r_[nm]["var_ratio"] = float(err[ok].mean() / s2b[ok].mean())
            res[lab] = r_
            print(lab, {k: (round(v["ll_gain_x1000"], 3), v.get("cover80"), round(v.get("var_ratio", 0), 2),
                            v.get("coverage_cells")) for k, v in r_.items()}, flush=True)
        out["oos"] = {"boundary": b16["new_build"], "window_days": 30, "results": res}
    with open(os.path.join(C.RESULTS, "p3_ph_patch.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
