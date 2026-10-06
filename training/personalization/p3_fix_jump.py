"""
Audit fix P3-14: do patches reset individual hero skill? Redone with the
demeaned residual, a fitted noise scale, out-of-sample tests and matched
placebos.

Old test (p3_ph_patch): the jump filter was fed r_adj, not the demeaned u the
write-up described; the likelihood was in-sample with the noise fixed at
wp(1 - wp) (AUDIT P3-14, P3-10).

Here:
  residual   u = r_adj - mean of r_adj over (build, hero), r_adj from the
             lag-1 count contract (p3_fix_counts). The population's patch
             shift is removed; only the player-specific part is tested.
  model      the drift state-space model (p3_sd_kalman "drift days", random
             40k fit), observation noise c * wp (1 - wp) with c profiled
  jump       extra variance J on the hero-specific state of (boundary, hero)
             pairs when a player's history crosses the boundary
  sets       patch notes; win-rate detector; aggregate |dWR| >= 2pp
  placebos   (a) matched: at the same boundary, for each changed hero an
             unchanged hero of the same Blizzard role with the closest pick
             volume in the pre segment; (b) time: the changed heroes, jump at
             a build half-way through the pre segment (no patch there)
  samples    fit: 40k fit-half players, E games (2024-04 .. 2026-02-09)
             held-out players: 40k other players, E games
             out of time: held-out players' full history, log likelihood
             scored only on post-cutoff games (2026) within 60 days after
             the 2026 boundaries (2.55.15, 2.55.16), jumps at every boundary
Reported: log-likelihood gain over J = 0 on a J grid for real and placebo
sets, the J maximizing the fit-sample likelihood with a profile 95% bound,
and real minus placebo at each J. Plus the split-half DiD of p3_ph_patch on
u restricted to the 2026 boundaries (out of sample).

Run (from training/): OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_fix_jump.py
Output: results/fix/p3_fix_jump.json
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from p3_heroes import NUM_HEROES
import p3_hs_core as C
import p3_sd_kalman as K
import p3_ph_patch as PP
import p3_fix_counts as F

J_GRID_PP = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
NOISE_GRID = [0.94, 0.96, 0.98, 1.0, 1.02]


def main():
    t0 = time.time()
    rng = np.random.RandomState(0)
    d = C.load_slots()
    meta = C.hero_meta(d["hero_names"])
    names = [str(x) for x in d["hero_names"]]
    kz = np.load(K.KOUT)
    e_mask, n_p, n_ph, table, r_adj = F.prepare_l1(d)
    build = PP.slot_builds(d)
    bnds, builds = PP.boundaries(names)
    nb = len(builds)
    bh = build * NUM_HEROES + d["hero"]
    _, inv = np.unique(bh, return_inverse=True)
    u = r_adj - (np.bincount(inv, weights=r_adj) / np.bincount(inv))[inv]
    days = d["day"]
    post = ~d["in_sample"]
    # ---- aggregate shift per (boundary, hero) and pre-segment pick volume
    dwr = np.zeros((len(bnds), NUM_HEROES))
    vol = np.zeros((len(bnds), NUM_HEROES))
    mid_pos = np.full(len(bnds), -1)
    for k, b in enumerate(bnds):
        lo = bnds[k - 1]["pos"] if k > 0 else 0
        hi = bnds[k + 1]["pos"] if k + 1 < len(bnds) else 10 ** 6
        pre = (build >= lo) & (build < b["pos"])
        po = (build >= b["pos"]) & (build < hi)
        if not pre.any() or not po.any():
            continue
        t_b = days[po].min()
        pre &= days >= t_b - 90
        po &= days < t_b + 90
        c0 = np.bincount(d["hero"][pre], minlength=NUM_HEROES)
        c1 = np.bincount(d["hero"][po], minlength=NUM_HEROES)
        dwr[k] = np.bincount(d["hero"][po], weights=d["y"][po], minlength=NUM_HEROES) / np.maximum(c1, 1) - \
            np.bincount(d["hero"][pre], weights=d["y"][pre], minlength=NUM_HEROES) / np.maximum(c0, 1)
        vol[k] = c0
        pb = np.unique(build[pre])
        if len(pb) >= 2:
            mid_pos[k] = int(pb[len(pb) // 2]) if pb[len(pb) // 2] > pb[0] else -1
    sets = {"notes": [(k, h) for k, b in enumerate(bnds) for h in np.flatnonzero(b["notes"])],
            "detector": [(k, h) for k, b in enumerate(bnds) for h in np.flatnonzero(b["detector"])],
            "|dWR| >= 2pp": [(k, h) for k in range(len(bnds)) for h in np.flatnonzero(np.abs(dwr[k]) >= 0.02)]}
    sets = {s: [(k, h) for k, h in v if vol[k, h] > 0] for s, v in sets.items()}

    def matched(pairs):
        out, used = [], set()
        for k, h in pairs:
            ch = bnds[k]["notes"] | bnds[k]["detector"] | (np.abs(dwr[k]) >= 0.02)
            cand = [h2 for h2 in range(NUM_HEROES) if not ch[h2] and meta["blizz"][h2] == meta["blizz"][h]
                    and (k, h2) not in used and vol[k, h2] > 0]
            if not cand:
                continue
            h2 = min(cand, key=lambda x: abs(np.log1p(vol[k, x]) - np.log1p(vol[k, h])))
            used.add((k, h2))
            out.append((k, h2))
        return out

    def table_for(pairs, J, time_placebo=False):
        t = np.zeros((nb + 1, NUM_HEROES))
        for k, h in pairs:
            pos = mid_pos[k] if time_placebo else bnds[k]["pos"]
            if pos >= 0:
                t[pos, h] = J
        return t

    variants = {}
    for s, pairs in sets.items():
        variants[(s, "real")] = (pairs, False)
        variants[(s, "matched placebo")] = (matched(pairs), False)
        variants[(s, "time placebo")] = ([(k, h) for k, h in pairs if mid_pos[k] >= 0], True)
    out = {"n_pairs": {f"{s} | {v}": len(p) for (s, v), (p, _) in variants.items()}}
    print(json.dumps(out["n_pairs"]), flush=True)
    # ---- samples
    fits = __import__("json").load(open(os.path.join(C.RESULTS, "p3_sd_fit.json")))
    p = fits["random 40k"]["fits"]["drift days"]["params"]
    fitp = kz["fit_players"]
    nE = np.bincount(d["pid"][e_mask], minlength=int(d["n_players"]))
    samp_fit = np.zeros(int(d["n_players"]), bool)
    samp_fit[rng.choice(np.flatnonzero(fitp & (nE > 0)), 40000, replace=False)] = True
    samp_hold = np.zeros(int(d["n_players"]), bool)
    samp_hold[rng.choice(np.flatnonzero(~fitp & (nE > 0)), 40000, replace=False)] = True
    jf_fit = PP.JumpFilter(d, meta, kz["V_cf2"], u, build, e_mask & samp_fit[d["pid"]])
    jf_hold = PP.JumpFilter(d, meta, kz["V_cf2"], u, build, e_mask & samp_hold[d["pid"]])
    jf_oot = PP.JumpFilter(d, meta, kz["V_cf2"], u, build, samp_hold[d["pid"]])
    v_fit, v_hold, v_oot = jf_fit.v.copy(), jf_hold.v.copy(), jf_oot.v.copy()
    zero = np.zeros((nb + 1, NUM_HEROES))
    # noise scale profile (fit sample, no jump)
    prof = {}
    for c in NOISE_GRID:
        jf_fit.v = v_fit * c
        prof[c] = float(jf_fit.run_jump(p, zero)[2].sum())
    c_hat = max(prof, key=prof.get)
    out["noise_scale_profile"] = {str(k): v - prof[1.0] for k, v in prof.items()}
    out["noise_scale"] = c_hat
    jf_fit.v, jf_hold.v, jf_oot.v = v_fit * c_hat, v_hold * c_hat, v_oot * c_hat
    print(f"noise scale {c_hat} ({time.time() - t0:.0f}s)", flush=True)
    # out-of-time scoring mask: post-cutoff rows within 60 days after a 2026 boundary
    b26 = [k for k, b in enumerate(bnds) if b["new_build"].startswith("2.55.15") or b["new_build"].startswith("2.55.16")]
    rows_o = jf_oot.rows
    t_b = {k: int(days[build >= bnds[k]["pos"]].min()) for k in b26}
    win = np.zeros(len(rows_o), bool)
    for k in b26:
        dd = days[rows_o]
        win |= (dd >= t_b[k]) & (dd < t_b[k] + 60) & (build[rows_o] >= bnds[k]["pos"])
    win &= post[rows_o]
    hero_o = d["hero"][rows_o]
    base = {"fit": jf_fit.run_jump(p, zero)[2].sum(), "hold": jf_hold.run_jump(p, zero)[2].sum()}
    ll0_oot = jf_oot.run_jump(p, zero)[2]
    res = {}
    for (s, vname), (pairs, tp) in variants.items():
        heroes26 = set(h for k, h in pairs if k in b26)
        sel_o = win & np.isin(hero_o, list(heroes26)) if heroes26 else win & False
        r = {"pairs": len(pairs), "oot_scored_rows": int(sel_o.sum())}
        for J in J_GRID_PP[1:]:
            tj = table_for(pairs, (J / 100) ** 2, tp)
            r[f"J {J}pp"] = {"fit": float(jf_fit.run_jump(p, tj)[2].sum() - base["fit"]),
                             "held-out players": float(jf_hold.run_jump(p, tj)[2].sum() - base["hold"]),
                             "out of time (2026 boundaries)": float((jf_oot.run_jump(p, tj)[2] - ll0_oot)[sel_o].sum())}
        res[f"{s} | {vname}"] = r
        print(s, vname, json.dumps({k: v for k, v in r.items()}), f"({time.time() - t0:.0f}s)", flush=True)
    out["ll_gain_vs_no_jump"] = res
    # ML J and profile bound (fit sample) for the real sets
    from scipy.optimize import minimize_scalar
    mle = {}
    for s in sets:
        pairs, tp = variants[(s, "real")]
        f = lambda z: -(jf_fit.run_jump(p, table_for(pairs, np.exp(z), tp))[2].sum() - base["fit"])
        r = minimize_scalar(f, bounds=(np.log(1e-8), np.log(4e-4)), method="bounded", options={"xatol": 0.05})
        j_hat = float(np.exp(r.x))
        g_hat = -r.fun
        # profile upper bound: largest J with gain >= g_hat - 1.92
        hi = None
        for J in np.arange(0.25, 4.01, 0.25):
            g = jf_fit.run_jump(p, table_for(pairs, (J / 100) ** 2, tp))[2].sum() - base["fit"]
            if g < max(g_hat, 0.0) - 1.92:
                hi = float(J)
                break
        mle[s] = {"J_hat_sd_pp": 100 * np.sqrt(j_hat), "ll_gain_at_J_hat": float(g_hat),
                  "profile_95_upper_bound_sd_pp": hi}
        print("mle", s, mle[s], flush=True)
    out["mle_fit_sample"] = mle
    # out-of-sample DiD on the 2026 boundaries (split-half design of p3_ph_patch)
    c = PP.cell_table(d, u, build, bnds)
    oos = np.isin(c["k"], b26)
    cc = {k: v[oos] for k, v in c.items()}
    notes = np.array([bnds[k]["notes"][h] for k, h in zip(cc["k"], cc["hero"])])
    det = np.array([bnds[k]["detector"][h] for k, h in zip(cc["k"], cc["hero"])])
    big = np.abs(np.array([dwr[k, h] for k, h in zip(cc["k"], cc["hero"])])) >= 0.02
    masks = {"unchanged": ~notes & ~det & ~big, "notes": notes, "detector": det, "|dWR| >= 2pp": big}
    out["did_2026_boundaries"] = PP.did(cc, masks, rng)
    print("did 2026", json.dumps(out["did_2026_boundaries"])[:1500], flush=True)
    os.makedirs(F.FIX_RESULTS, exist_ok=True)
    with open(os.path.join(F.FIX_RESULTS, "p3_fix_jump.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
