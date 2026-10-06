"""
P3 phase 2: per player x hero skill as a state-space model (Kalman filter).

State per player (d = 202): overall skill in slow, medium and fast parts (3),
Blizzard-role effects (6), fine-role effects (9), melee/ranged effects (2),
CF factors (2, loadings V from p3_hs_fit rank 2), hero-specific effects in
a slow and a fast part (90 + 90). Every component is an Ornstein-Uhlenbeck
process with stationary variance s_c^2 and mean-reversion rate lam_c per
clock unit:

    x(t + D) = f x(t) + e,  f = exp(-lam D),  Var e = s^2 (1 - f^2)

so over short gaps a component is a random walk with increment variance
q = 2 lam s^2 per unit time. With every lam = 0 and the fast parts at zero
variance the model is exactly the static kernel of p3_hs_fit (checked to
machine precision). A game by the player on hero h gives
r_adj = z_h' x + noise, noise variance wp (1 - wp).

Filtering runs per player in time order. Predictions for a game use only
the state after the previous day's games (days < t). The log likelihood
uses the exact sequential predictive. Parameters are fit by coordinate-wise
bounded 1-D maximization on 40,000 fit-half players' E games. Variants: the
fast hero rate split by Blizzard role; the clock is calendar days or the
player's own game count.

Library + CLI:
  python3 personalization/p3_sd_kalman.py fit   # fit on E, fit-half players
Output: cache/sd_params.json, results/p3_sd_fit.json
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from numba import njit, prange
from p3_heroes import NUM_HEROES
import p3_hs_core as C

KOUT = os.path.join(C.CACHE, "hs_kernels.npz")
PARAMS = os.path.join(C.CACHE, "sd_params.json")
# components: (name, number of states)
LAYOUT = [("player_s", 1), ("player_m", 1), ("player_f", 1), ("role", 6), ("fine", 9), ("melee", 2),
          ("cf", 2), ("hero_s", NUM_HEROES), ("hero_f", NUM_HEROES)]
COMPS = [c for c, _ in LAYOUT]
OFF = {}
_o = 0
for _c, _n in LAYOUT:
    OFF[_c] = _o
    _o += _n
D_STATE = _o


def loadings(meta, V):
    H = len(meta["fine"])
    idx = np.zeros((H, 10), np.int64)
    w = np.ones((H, 10))
    for h in range(H):
        idx[h] = [OFF["player_s"], OFF["player_m"], OFF["player_f"], OFF["role"] + meta["blizz"][h],
                  OFF["fine"] + meta["fine"][h], OFF["melee"] + int(meta["melee"][h]),
                  OFF["cf"], OFF["cf"] + 1, OFF["hero_s"] + h, OFF["hero_f"] + h]
        w[h, 6], w[h, 7] = V[h, 0], V[h, 1]
    return idx, w


def state_comp():
    c = np.empty(D_STATE, np.int64)
    for i, (name, n) in enumerate(LAYOUT):
        c[OFF[name]:OFF[name] + n] = i
    return c


@njit(parallel=True, cache=True)
def kalman_run(starts, ends, tval, day, hero, y, v, upd, lidx, lw, s2, lam_row_state,
               out_m, out_s2, ll_rows):
    """lam_row_state: (H, d) per-hero-row rate override is not needed; lam
    is per state but hero states may carry role-specific rates (encoded in
    the state vector already). Returns nothing; fills outputs."""
    npl = starts.shape[0]
    d = s2.shape[0]
    L = lidx.shape[1]
    for p in prange(npl):
        a, b = starts[p], ends[p]
        x = np.zeros(d)
        P = np.zeros((d, d))
        for i in range(d):
            P[i, i] = s2[i]
        f = np.empty(d)
        pz = np.empty(d)
        t_prev = tval[a]
        i = a
        while i < b:
            # block of rows with the same day
            j = i
            while j < b and day[j] == day[i]:
                j += 1
            dt = tval[i] - t_prev
            if dt > 0:
                for k in range(d):
                    f[k] = np.exp(-lam_row_state[k] * dt)
                for k in range(d):
                    x[k] *= f[k]
                    for l in range(d):
                        P[k, l] *= f[k] * f[l]
                    P[k, k] += s2[k] * (1.0 - f[k] * f[k])
                t_prev = tval[i]
            # day-start predictions
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
            # sequential updates
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


class Filter:
    def __init__(self, d, meta, V, r_adj, rows_mask=None, clock="days", n_p=None):
        """Rows sorted by (pid, day, replay_id) restricted to rows_mask."""
        sel = np.ones(len(d["pid"]), bool) if rows_mask is None else rows_mask
        idx = np.flatnonzero(sel)
        o = np.lexsort((d["replay_id"][idx], d["day"][idx], d["pid"][idx]))
        self.rows = idx[o]
        pid = d["pid"][self.rows]
        brk = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
        self.starts = brk.astype(np.int64)
        self.ends = np.r_[brk[1:], len(pid)].astype(np.int64)
        self.day = d["day"][self.rows].astype(np.int64)
        if clock == "days":
            self.tval = self.day.astype(np.float64)
        else:  # player's own game count
            self.tval = n_p[self.rows].astype(np.float64)
        self.hero = d["hero"][self.rows].astype(np.int64)
        self.y = r_adj[self.rows].astype(np.float64)
        self.v = d["v"][self.rows].astype(np.float64)
        self.lidx, self.lw = loadings(meta, V)
        self.blizz = meta["blizz"]

    def lam_state(self, lam, by_role=None):
        """lam: dict comp -> rate per clock unit (missing = 0). by_role:
        dict hero component -> array of 6 rates by Blizzard role."""
        comp = state_comp()
        out = np.array([lam.get(COMPS[c], 0.0) for c in comp], float)
        for c, arr in (by_role or {}).items():
            for h in range(NUM_HEROES):
                out[OFF[c] + h] = arr[self.blizz[h]]
        return out

    def run(self, sd, lam, upd=None):
        n = len(self.rows)
        s2 = np.array([sd.get(COMPS[c], 0.0) ** 2 for c in state_comp()])
        out_m = np.zeros(n)
        out_s2 = np.zeros(n)
        ll = np.zeros(n)
        u = np.ones(n, bool) if upd is None else upd
        kalman_run(self.starts, self.ends, self.tval, self.day, self.hero, self.y, self.v, u,
                   self.lidx, self.lw, s2, lam, out_m, out_s2, ll)
        return out_m, out_s2, ll


def coord_fit(obj, params, free, bounds, sweeps=2, label=""):
    """Coordinate-wise bounded 1-D maximization (Brent) of obj(params) over
    the log of each free parameter; robust to the flat, slightly noisy
    likelihood surface where finite-difference L-BFGS failed."""
    from scipy.optimize import minimize_scalar
    p = dict(params)
    best = obj(p)
    t0 = time.time()
    for sw in range(sweeps):
        for k in free:
            lo, hi = bounds[k]

            def f(z, k=k):
                q = dict(p)
                q[k] = float(np.exp(z))
                return -obj(q)
            r = minimize_scalar(f, bounds=(np.log(lo), np.log(hi)), method="bounded",
                                options={"xatol": 0.05, "maxiter": 25})
            if -r.fun > best:
                best = -r.fun
                p[k] = float(np.exp(r.x))
        print(f"  {label} sweep {sw + 1}: ll {best:.1f} ({time.time() - t0:.0f}s) " +
              ", ".join(f"{k}={v:.4g}" for k, v in p.items() if k in free), flush=True)
    return p, best


def split(p):
    sd = {c: p.get("sd_" + c, 0.0) for c in COMPS}
    lam = {c: p.get("lam_" + c, 0.0) for c in COMPS}
    by_role = {}
    for c in ("hero_s", "hero_f"):
        if f"lam_{c}_Tank" in p:
            by_role[c] = np.array([p[f"lam_{c}_" + r] for r in C.BLIZZ_ROLES])
    return sd, lam, by_role


def halflives(p, unit="days"):
    out = {}
    for c in COMPS:
        lam = p.get("lam_" + c, 0.0)
        sd = p.get("sd_" + c, 0.0)
        out[c] = {"sd_pp": 100 * sd, "lam_per_" + unit: lam,
                  "corr_halflife_" + unit: (float(np.log(2) / lam) if lam > 0 else None),
                  "info_halflife_" + unit: (float(np.log(2) / (2 * lam)) if lam > 0 else None),
                  "rw_sd_pp_per_30" + unit[0]: (float(100 * np.sqrt(2 * lam * sd ** 2 * 30))
                                                  if lam > 0 else 0.0)}
    return out


def fit_sample(d, meta, V, r_adj, n_p, rows, label, out, p_static):
    filt = Filter(d, meta, V, r_adj, rows)

    def obj_for(fl):
        def obj(p):
            sd, lam, br = split(p)
            return float(fl.run(sd, fl.lam_state(lam, br))[2].sum())
        return obj

    obj = obj_for(filt)
    B = {k: (1e-4, 0.15) for k in ["sd_" + c for c in COMPS]}
    B.update({"lam_player_s": (1e-5, 0.01), "lam_player_m": (1e-3, 0.1),
              "lam_player_f": (0.02, 2.0), "lam_cf": (1e-5, 0.05),
              "lam_hero_s": (1e-5, 0.02), "lam_hero_f": (2e-4, 0.2)})
    for c in ("hero_s", "hero_f"):
        for r in C.BLIZZ_ROLES:
            B[f"lam_{c}_" + r] = B["lam_" + c]
    res = {}
    llA = obj(p_static)
    res["static (phase-1 kernel)"] = {"params": p_static, "ll": llA}
    print(f"[{label}] static ll {llA:.1f}", flush=True)
    p0 = dict(p_static)
    p0.update({"sd_player_s": 0.012, "sd_player_m": 0.015, "lam_player_m": 0.02,
               "sd_player_f": 0.015, "lam_player_f": 0.2, "sd_hero_f": 0.01, "lam_hero_f": 0.005,
               "lam_player_s": 1e-4, "lam_cf": 1e-4, "lam_hero_s": 1e-4})
    free = ["sd_player_f", "lam_player_f", "sd_player_m", "lam_player_m", "sd_player_s",
            "lam_player_s", "sd_hero_f", "lam_hero_f", "sd_hero_s", "lam_hero_s", "lam_cf"]
    pB, llB = coord_fit(obj, p0, free, B, sweeps=2, label=f"{label} drift days")
    res["drift days"] = {"params": pB, "ll": llB}
    pC0 = dict(pB)
    freeC = []
    for c in ("hero_s", "hero_f"):
        for r in C.BLIZZ_ROLES:
            pC0[f"lam_{c}_" + r] = pB["lam_" + c]
            freeC.append(f"lam_{c}_" + r)
    pC, llC = coord_fit(obj, pC0, freeC, B, sweeps=1, label=f"{label} hero rates by role")
    res["drift days, hero rates by role"] = {"params": pC, "ll": llC}
    filt_g = Filter(d, meta, V, r_adj, rows, clock="games", n_p=n_p)
    pD, llD = coord_fit(obj_for(filt_g), pB, free, B, sweeps=2, label=f"{label} drift games")
    res["drift games"] = {"params": pD, "ll": llD}
    for k, v in res.items():
        v["ll_gain_vs_static"] = v["ll"] - llA
        v["components"] = halflives(v["params"], "games" if "games" in k else "days")
        print(f"[{label}] {k}: dLL {v['ll_gain_vs_static']:+.1f}", flush=True)
    out[label] = {"n_rows": int(rows.sum()), "fits": res}
    return res


def main():
    d = C.load_slots()
    meta = C.hero_meta(d["hero_names"])
    kz = np.load(KOUT)
    from p3_hs_fit import prepare
    e_mask, n_p, n_ph, table, r_adj = prepare(d)
    V = kz["V_cf2"]
    th = kz["theta_+CF rank 2"]  # player, role, fine, melee, hero, cf2
    ex = lambda i: float(np.sqrt(np.exp(th[i])))
    p_static = {"sd_player_s": ex(0), "sd_role": ex(1), "sd_fine": ex(2), "sd_melee": ex(3),
                "sd_hero_s": ex(4), "sd_cf": ex(5)}
    rng = np.random.RandomState(11)
    fitp = kz["fit_players"]
    nE = np.bincount(d["pid"][e_mask], minlength=int(d["n_players"]))
    out = {}
    samp = np.zeros(int(d["n_players"]), bool)
    samp[rng.choice(np.flatnonzero(fitp & (nE > 0)), 40000, replace=False)] = True
    res_r = fit_sample(d, meta, V, r_adj, n_p, e_mask & samp[d["pid"]], "random 40k", out, p_static)
    with open(PARAMS, "w") as f:
        json.dump({k: v["params"] for k, v in res_r.items()}, f, indent=1)
    heavy = np.zeros(int(d["n_players"]), bool)
    cand = np.flatnonzero(fitp & (nE >= 300))
    heavy[rng.choice(cand, min(6000, len(cand)), replace=False)] = True
    fit_sample(d, meta, V, r_adj, n_p, e_mask & heavy[d["pid"]], "heavy 6k (>=300 games)", out,
               p_static)
    with open(os.path.join(C.RESULTS, "p3_sd_fit.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
