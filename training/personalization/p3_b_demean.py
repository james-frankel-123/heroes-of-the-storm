"""
P3 review item 3 (REVIEW section 3.1): causal (build, hero, tier) demeaning
of residuals before the personal state.

Mechanism. Residuals are taken against a WP whose hero statistics run one
build behind. When a patch moves a hero, every player of that hero gets a
residual of the same sign until the WP catches up, and the GP reads it as
hero-specific skill.

Fix. For each slot, subtract from its residual (r_adj = y - WP - experience
offset) the shrunken mean residual of its cell (build, hero, WP skill tier)
over EARLIER days of the SAME build only (lag 1 day; a new build starts every
cell at 0):

    nowcast_i = S_c(< day_i) / (n_c(< day_i) + k),   k = noise / tau2

an empirical-Bayes posterior mean with prior mean 0, noise = mean wp(1-wp),
and tau2 the variance of true cell means, estimated by moments on E
(cells with 50+ slots: mean of m_c^2 - noise_c / n_c, weighted by n_c).
The V1-estimated tau2 is reported as a check (and run as an arm). The
demeaned residual r_adj - nowcast is what enters the GP state (phase-1
"+CF rank 2" kernel and experience table, unchanged), and optionally the
state-space filter of p3_sd_kalman.py ("drift days" parameters, --kalman).

`causal_cell_demean(d, r, cell_key, tau2, lag=1)` is the reusable function
(returns r_demeaned, nowcast, n_before); `row_builds`, `row_tiers` give the
build and tier of every row of the extended table.

Leakage controls. The nowcast for a slot uses only rows of the same build on
days <= day - 1 (the slot's own game and same-day games never count); a row
enters the personal state at day + 1 with the nowcast it had on its own day.
tau2 comes from E (or V1, the combiner-fit window). OOT excludes the sealed
build 2.55.17.98025 unless --final. "Earlier days" means every eventually
uploaded game with an earlier date (no ingestion-time filter; audit P3-04).

Evaluation (headline protocol: logistic combiner on logit WP + team
difference of the slot estimate, fit on V1, scored on V2 and OOT; replay
bootstrap, widen about 1.3x for recurring players):
  arms   none | demeaned state | demeaned state + nowcast as a separate
         combiner feature; cell variants (build, hero, tier) with tau2 from
         E and from V1, and (build, hero) with tau2 from E
  where  all of V2 and OOT, by OOT build, and the first 30 days after every
         build that starts inside V1, V2 or OOT (V2 holds 2.55.16 on
         2026-04-20, OOT the 2.55.17 builds); slot-level log-loss gain and
         slope of r on the slot estimate there, split by patched heroes
         (|hero win-rate shift| >= 2pp with |z| >= 2 over 30 days either
         side, plus the patch-note and detector heroes of drift2026
         w3_changepoints.json where the boundary is listed; read from the repo
         or, on a worker without it, from cache/b3_w3_changepoints.json) and
         unpatched
  table  cache/b3_nowcast.npz: per (build, hero, tier) cell and day with
         games, the running sum, count and shrunken mean through that day
         (usable from the next day)

Usage (from training/):
  python personalization/p3_b_demean.py [--eval-sample 0.1] [--kalman] [--final]
Outputs: C.RESULTS/p3_b_demean[...].json, C.CACHE/b3_nowcast[...].npz
"""
import os
import sys
import json
import gzip
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from p3_heroes import NUM_HEROES
import p3_hs_core as C
import p3_x_common as X

SEALED_BUILD = "2.55.17.98025"
W3 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "drift2026", "results",
                  "w3_changepoints.json")
NTIER = 4  # low, mid, high, unknown


# ------------------------------------------------------------------ reusable pieces

def row_builds(d):
    """Build (full game version) of every row of the extended table (or of
    hs_slots): snapshot rows from wp_drift build_idx, post rows from the
    version column. Returns (code per row, list of build names), codes
    ordered by first day seen."""
    from drift2026 import common
    names_all = common.load_patch_index()["builds"]
    w = np.load(C.WP)
    o = np.argsort(w["replay_ids"])
    bidx = w["build_idx"][o]
    post = d["post"] if "post" in d else np.zeros(len(d["pid"]), bool)
    names = np.empty(len(d["pid"]), object)
    snap = ~post
    names[snap] = np.asarray(names_all, object)[bidx[d["g"][snap]]]
    if post.any():
        vers = np.asarray([str(v) for v in d["versions"]], object)
        names[post] = vers[d["version"][post]]
    u, code = np.unique(names.astype(str), return_inverse=True)
    first = np.full(len(u), 10 ** 9)
    np.minimum.at(first, code, d["day"])
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(u), np.int64)
    rank[order] = np.arange(len(u))
    return rank[code], [str(u[i]) for i in order]


def row_tiers(d):
    """WP skill tier per row (shared.SKILL_TIERS index; 3 = unknown)."""
    from shared import SKILL_TIERS
    import p3_x_side as XS
    out = np.full(len(d["pid"]), 3, np.int64)
    sg = np.load(os.path.join(C.CACHE, "x_side_games.npz"), allow_pickle=True)
    tmap = np.array([SKILL_TIERS.index(str(n)) for n in sg["tier_names"]])
    gi = XS.game_lookup(d, sg["replay_ids"])
    ok = gi >= 0
    out[ok] = tmap[sg["tier"][gi[ok]]]
    post = d["post"] if "post" in d else np.zeros(len(d["pid"]), bool)
    if post.any():
        pg = json.load(gzip.open(os.path.join(C.CACHE, "x_post_games.json.gz"), "rt"))
        rid = np.array([g["replay_id"] for g in pg], np.int64)
        tr = np.array([SKILL_TIERS.index(g["skill_tier"]) if g.get("skill_tier") in SKILL_TIERS else 3
                       for g in pg], np.int64)
        o = np.argsort(rid)
        j = np.minimum(np.searchsorted(rid[o], d["replay_id"]), len(rid) - 1)
        okp = post & (rid[o][j] == d["replay_id"])
        out[okp] = tr[o][j][okp]
    return out


def estimate_tau2(r, v, cell_key, mask, min_n=50):
    """Moment estimate of the variance of true cell means of r (prior mean 0)."""
    u, inv = np.unique(cell_key[mask], return_inverse=True)
    n = np.bincount(inv).astype(float)
    m = np.bincount(inv, weights=r[mask]) / n
    nv = np.bincount(inv, weights=v[mask]) / n
    ok = n >= min_n
    return float(np.sum(n[ok] * (m[ok] ** 2 - nv[ok] / n[ok])) / np.sum(n[ok])), int(ok.sum())


def causal_cell_demean(d, r, cell_key, tau2, lag=1, noise=None):
    """Shrunken mean of r over rows of the same cell with day <= day_i - lag.
    Put the build in cell_key so a new build starts from 0. Returns
    (r - nowcast, nowcast, n_before). noise defaults to mean wp(1 - wp)."""
    noise = float(np.mean(d["v"])) if noise is None else noise
    k = noise / max(tau2, 1e-9)
    day = d["day"].astype(np.int64)
    o = np.lexsort((d["replay_id"], day, cell_key))
    ck, dy = cell_key[o], day[o]
    cs = np.r_[0.0, np.cumsum(r[o])]
    gstart = np.searchsorted(ck, ck, side="left")
    # rows of the same cell with day <= day_i - lag: composite search on (cell, day)
    span = int(day.max() - day.min() + lag + 2)
    comp = (ck - ck.min()) * span + (dy - day.min())
    pos = np.searchsorted(comp, comp - lag, side="right")
    pos = np.maximum(pos, gstart)
    s = cs[pos] - cs[gstart]
    nb = (pos - gstart).astype(np.float64)
    now = np.empty(len(r))
    nbo = np.empty(len(r))
    now[o] = s / (nb + k)
    nbo[o] = nb
    return r - now, now, nbo


def nowcast_table(d, r, cell_key, tau2, noise=None):
    """Per (cell, day with games): running sum and count through that day and
    the shrunken mean (the nowcast valid from the next day on)."""
    noise = float(np.mean(d["v"])) if noise is None else noise
    k = noise / max(tau2, 1e-9)
    key = cell_key.astype(np.int64) * 100000 + d["day"].astype(np.int64)
    u, inv = np.unique(key, return_inverse=True)
    s = np.bincount(inv, weights=r)
    n = np.bincount(inv).astype(float)
    cell = u // 100000
    brk = np.r_[True, cell[1:] != cell[:-1]]
    gid = np.cumsum(brk) - 1
    cs, cn = np.cumsum(s), np.cumsum(n)
    first = np.flatnonzero(brk)
    base_s = np.r_[0.0, cs][first][gid]
    base_n = np.r_[0.0, cn][first][gid]
    S, N = cs - base_s, cn - base_n
    return {"cell": cell, "day": (u % 100000).astype(np.int32), "sum": S, "n": N, "nowcast": S / (N + k),
            "k": k}


# ------------------------------------------------------------------ evaluation helpers

def slot_stats(r, wp, pred):
    p1 = np.clip(wp + pred, 1e-3, 1 - 1e-3)
    y = r + wp
    ll0 = -(y * np.log(wp) + (1 - y) * np.log(1 - wp))
    ll1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    o = {"n": int(len(r)), "ll_gain_x1000": float(1000 * (ll0 - ll1).mean())}
    if len(r) > 50 and pred.std() > 1e-9:
        o["slope"] = float(np.cov(r, pred)[0, 1] / pred.var())
    return o


def patched_sets(d, build, names, starts, hero_names, use_rows=None):
    """Per boundary build b: heroes whose win rate moved >= 2pp (|z| >= 2)
    between the 30 days before and after the build's first day, plus the w3
    note and detector heroes if the boundary is in w3_changepoints.json."""
    src = W3 if os.path.exists(W3) else os.path.join(C.CACHE, "b3_w3_changepoints.json")
    w3 = json.load(open(src)) if os.path.exists(src) else {"boundaries": []}
    if not w3["boundaries"]:
        print("w3_changepoints.json not found: patched sets from win-rate shifts only", flush=True)
    hn = [str(h) for h in hero_names]
    w3sets = {}
    for b in w3["boundaries"]:
        ib = b["intermediate_builds"]
        if isinstance(ib, str):
            ib = json.loads(ib.replace("'", '"'))
        notes = set(b.get("truth", {}).get("heroes", []) or [])
        det = set(b["detected"].get("wr_bh", []))
        for x in list(ib) + [b["new_build"]]:
            w3sets[x] = np.array([h in notes or h in det for h in hn])
    out = {}
    day = d["day"]
    use = np.ones(len(day), bool) if use_rows is None else use_rows
    for bi, t in starts.items():
        pre = use & (day >= t - 30) & (day < t)
        po = use & (day >= t) & (day < t + 30)
        c0 = np.bincount(d["hero"][pre], minlength=NUM_HEROES).astype(float)
        c1 = np.bincount(d["hero"][po], minlength=NUM_HEROES).astype(float)
        w0 = np.bincount(d["hero"][pre], weights=d["y"][pre], minlength=NUM_HEROES) / np.maximum(c0, 1)
        w1 = np.bincount(d["hero"][po], weights=d["y"][po], minlength=NUM_HEROES) / np.maximum(c1, 1)
        se = np.sqrt(0.25 / np.maximum(c0, 1) + 0.25 / np.maximum(c1, 1))
        dwr = w1 - w0
        mv = (np.abs(dwr) >= 0.02) & (np.abs(dwr) / se >= 2) & (c0 >= 200) & (c1 >= 200)
        listed = w3sets.get(names[bi])
        out[bi] = {"patched": mv | (listed if listed is not None else False), "dwr": dwr,
                   "w3_listed": listed is not None}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-sample", type=float, default=1.0, help="fraction of V1/V2/OOT games queried (debug)")
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--kalman", action="store_true", help="also run the state-space filter on both residuals")
    ap.add_argument("--final", action="store_true", help="also score the sealed OOT build 2.55.17.98025")
    a = ap.parse_args()
    t0 = time.time()
    tag = ("" if a.eval_sample == 1.0 else f"_e{a.eval_sample:g}") + ("_final" if a.final else "")
    H = NUM_HEROES
    d = X.load_ext()
    n = len(d["pid"])
    days = d["day"]
    post = d["post"]
    e_mask = d["in_sample"] & (days >= C.day_of(C.E_START))
    n_p, n_ph = C.experience_counts(d)
    table = C.fit_experience(d["r"], n_p, n_ph, e_mask)
    mu_exp = table[C.exp_bins(n_p, n_ph)]
    r_adj = d["r"] - mu_exp
    build, bnames = row_builds(d)
    tier = row_tiers(d)
    snap_post = (~post) & (~d["in_sample"])
    _, fr = np.unique(d["g"][snap_post], return_index=True)
    med = np.median(days[snap_post][fr])
    v1, v2 = snap_post & (days < med), snap_post & (days >= med)
    sealed = post & (build == (bnames.index(SEALED_BUILD) if SEALED_BUILD in bnames else -1))
    oot = post & (days > X.SNAP_LAST_DAY) & (a.final | ~sealed)
    out = {"args": vars(a), "rows": n, "tier_unknown_share": float((tier == 3).mean()),
           "oot_scope": "all OOT builds (final run)" if a.final else f"OOT without the sealed build {SEALED_BUILD}",
           "builds_in_eval": {}}
    print(f"rows {n:,}; builds {len(bnames)}; tier unknown {out['tier_unknown_share']:.4f} ({time.time() - t0:.0f}s)",
          flush=True)

    # ---------------- cells, tau2, demeaned residuals
    key_bht = (build * H + d["hero"]) * NTIER + tier
    key_bh = build * H + d["hero"]
    tau = {}
    tau["bht_E"], nE = estimate_tau2(r_adj, d["v"], key_bht, e_mask)
    tau["bht_V1"], nV = estimate_tau2(r_adj, d["v"], key_bht, v1)
    tau["bh_E"], nE2 = estimate_tau2(r_adj, d["v"], key_bh, e_mask)
    noise = float(np.mean(d["v"]))
    out["tau2"] = {k: {"pp2": 1e4 * v, "sd_pp": 100 * np.sqrt(max(v, 0)), "k_games": noise / max(v, 1e-9)}
                   for k, v in tau.items()}
    out["tau2_cells"] = {"bht_E": nE, "bht_V1": nV, "bh_E": nE2}
    print("tau2", json.dumps(out["tau2"]), flush=True)
    arms = {"none": (r_adj, np.zeros(n))}
    for nm, key, t2 in (("build x hero x tier, tau2 E", key_bht, tau["bht_E"]),
                        ("build x hero x tier, tau2 V1", key_bht, tau["bht_V1"]),
                        ("build x hero, tau2 E", key_bh, tau["bh_E"])):
        rd, now, nb = causal_cell_demean(d, r_adj, key, t2, lag=1, noise=noise)
        arms[nm] = (rd, now)
        print(f"  {nm}: nowcast sd {100 * now.std():.3f}pp, rows with 0 earlier-day cell games "
              f"{(nb == 0).mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
    # nowcast table (primary arm)
    nt = nowcast_table(d, r_adj, key_bht, tau["bht_E"], noise)
    np.savez(os.path.join(C.CACHE, f"b3_nowcast{tag}.npz"), build=(nt["cell"] // NTIER) // H,
             hero=(nt["cell"] // NTIER) % H, tier=nt["cell"] % NTIER, day=nt["day"], sum=nt["sum"], n=nt["n"],
             nowcast=nt["nowcast"], k=nt["k"], build_names=np.array(bnames), hero_names=d["hero_names"],
             note=np.array("stats through `day` inclusive; valid as a nowcast from day + 1; tier 3 = unknown"))

    # ---------------- online GP pass (shared P, one S per arm)
    Ks, _ = X.kernels()
    K = Ks["+CF rank 2"]
    gu = np.random.RandomState(11).rand(int(d["g"].max()) + 1)
    query = (v1 | v2 | oot) & (gu[d["g"]] < a.eval_sample)
    qidx = np.flatnonzero(query)
    nq = len(qidx)
    names = list(arms)
    npl = int(d["n_players"])
    Sst = {nm: np.zeros((npl, H)) for nm in names}
    Pst = np.zeros((npl, H))
    pred = {nm: np.empty(nq) for nm in names}
    ap_ = 0
    ud, qs = np.unique(days[qidx], return_index=True)
    qe = np.r_[qs[1:], nq]
    for it, (t, s0, s1) in enumerate(zip(ud, qs, qe)):
        e = np.searchsorted(days, t - 1, side="right")
        if e > ap_:
            rr = np.arange(ap_, e)
            pi, hi = d["pid"][rr], d["hero"][rr]
            np.add.at(Pst, (pi, hi), 1.0 / d["v"][rr])
            for nm in names:
                np.add.at(Sst[nm], (pi, hi), arms[nm][0][rr] / d["v"][rr])
            ap_ = e
        q = qidx[s0:s1]
        pq, hq = d["pid"][q], d["hero"][q]
        Pq = np.ascontiguousarray(Pst[pq])
        for nm in names:
            pred[nm][s0:s1] = C.gp_query(np.ascontiguousarray(Sst[nm][pq]), Pq, hq, K)[0]
        if it % 50 == 0:
            print(f"  day {it}/{len(ud)} ({time.time() - t0:.0f}s)", flush=True)
    del Sst, Pst
    s_rows = {nm: mu_exp[qidx] + pred[nm] for nm in names}
    now_rows = {nm: arms[nm][1][qidx] for nm in names}
    print(f"GP pass done ({time.time() - t0:.0f}s)", flush=True)

    if a.kalman:
        import p3_sd_kalman as KF
        meta = C.hero_meta(d["hero_names"])
        kz = np.load(KF.KOUT)
        prm = json.load(open(KF.PARAMS))["drift days"]
        for nm in ("none", "build x hero x tier, tau2 E"):
            filt = KF.Filter(d, meta, kz["V_cf2"], arms[nm][0])
            sd, lam, br = KF.split(prm)
            m_, _, _ = filt.run(sd, filt.lam_state(lam, br))
            pm = np.empty(n)
            pm[filt.rows] = m_
            key = f"state-space | {nm}"
            s_rows[key] = mu_exp[qidx] + pm[qidx]
            now_rows[key] = arms[nm][1][qidx]
            del filt
            print(f"  kalman {nm} ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- game level
    n_games, y, wp0, cnt, gday = X.game_arrays(d)
    lo = np.log(wp0 / (1 - wp0))
    qcount = np.bincount(d["g"][qidx], minlength=n_games)
    gbuild = np.zeros(n_games, np.int64)
    gbuild[d["g"]] = build
    gv = {}
    for wn, mw in (("V1", v1), ("V2", v2), ("OOT", oot)):
        gm = np.zeros(n_games, bool)
        gm[d["g"][mw & query]] = True
        gv[wn] = gm & (cnt == 10) & (qcount == 10)
    periods = {"V2": gv["V2"], "OOT": gv["OOT"]}
    for b in np.unique(gbuild[gv["OOT"]]):
        m_ = gv["OOT"] & (gbuild == b)
        if m_.sum() > 1000:
            periods[f"OOT build {bnames[b]}"] = m_
    # boundaries: builds whose first day falls inside an evaluation window
    bstart = np.full(len(bnames), 10 ** 9)
    np.minimum.at(bstart, build, days)
    evdays = {wn: (days[mw].min(), days[mw].max()) for wn, mw in (("V1", v1), ("V2", v2), ("OOT", oot)) if mw.any()}
    starts = {}
    for b, t in enumerate(bstart):
        if bnames[b] == SEALED_BUILD and not a.final:
            continue
        for wn, (lo_d, hi_d) in evdays.items():
            if lo_d < t <= hi_d:
                starts[b] = int(t)
                periods[f"{wn} first 30 days of {bnames[b]}"] = gv[wn] & (gday >= t) & (gday < t + 30)
                out["builds_in_eval"][bnames[b]] = {"window": wn, "first_day": str(np.datetime64(int(t), "D"))}
    print("boundaries:", json.dumps(out["builds_in_eval"]), flush=True)
    g_q, t_q = d["g"][qidx], d["team"][qidx]
    fitm = gv["V1"]
    w0 = C.fit_logistic(lo[fitm][:, None], y[fitm])
    ll0 = X.logloss(C.predict(w0, lo[:, None]), y)
    specs = {}
    for nm in s_rows:
        Ds = X.team_diff(s_rows[nm], g_q, t_q, n_games)
        specs[nm] = [Ds]
        if nm != "none" and not nm.endswith("| none"):
            specs[nm + " + nowcast feature"] = [Ds, X.team_diff(now_rows[nm], g_q, t_q, n_games)]
        if nm == "none":
            # the nowcast's own value without demeaning the state
            for nm2 in ("build x hero x tier, tau2 E", "build x hero, tau2 E"):
                specs[f"none + nowcast feature ({nm2})"] = [Ds, X.team_diff(now_rows[nm2], g_q, t_q, n_games)]
    lls = {}
    game = {}
    for sn, cols in specs.items():
        Xm = np.column_stack([lo] + cols)
        w = C.fit_logistic(Xm[fitm], y[fitm])
        p = C.predict(w, Xm)
        l1 = X.logloss(p, y)
        lls[sn] = l1
        ref = "state-space | none" if sn.startswith("state-space") else "none"
        res = {"coef": w.tolist()}
        for pn, pm in periods.items():
            idx = np.flatnonzero(pm)
            if len(idx) < 200:
                continue
            nb_ = a.nboot if pn in ("V2", "OOT") else max(100, a.nboot // 2)
            gain, ci = X.boot_gain(ll0, l1, idx, n=nb_)
            r_ = {"gain": gain, "ci": ci, "games": int(len(idx))}
            if sn != ref and ref in lls:
                dg, dci = X.boot_gain(lls[ref], l1, idx, n=nb_)
                r_["vs_" + ref] = {"gain": dg, "ci": dci}
            res[pn] = r_
        game[sn] = res
        print(f"  {sn:52s} V2 {res['V2']['gain']:+.5f}  OOT {res['OOT']['gain']:+.5f}"
              + "".join(f"  d{k[:3]} {res[k]['vs_' + ref]['gain']:+.5f} [{res[k]['vs_' + ref]['ci'][0]:+.5f},"
                        f"{res[k]['vs_' + ref]['ci'][1]:+.5f}]" for k in ("V2", "OOT") if 'vs_' + ref in res[k]),
              flush=True)
    out["game"] = game

    # ---------------- slot level after each boundary, patched vs unpatched heroes
    ps = patched_sets(d, build, bnames, starts, d["hero_names"], use_rows=~sealed | a.final)
    hero_q, day_q = d["hero"][qidx], days[qidx]
    r_q, wp_q = d["r"][qidx], d["wp"][qidx]
    slot = {}
    for b, t in starts.items():
        win = (day_q >= t) & (day_q < t + 30)
        pat = ps[b]["patched"][hero_q]
        res = {"patched_heroes": [str(d["hero_names"][h]) for h in np.flatnonzero(ps[b]["patched"])],
               "w3_listed": bool(ps[b]["w3_listed"])}
        for nm in s_rows:
            for sub, m_ in (("all", win), ("patched", win & pat), ("unpatched", win & ~pat)):
                if m_.sum() < 200:
                    continue
                res.setdefault(nm, {})[sub] = slot_stats(r_q[m_], wp_q[m_], s_rows[nm][m_])
                if nm != "none" and not nm.startswith("state-space"):
                    res[nm][sub + " (+ nowcast)"] = slot_stats(r_q[m_], wp_q[m_], s_rows[nm][m_] + now_rows[nm][m_])
        slot[bnames[b]] = res
        nn = res.get("none", {})
        dd_ = res.get("build x hero x tier, tau2 E", {})
        print(f"  slot {bnames[b]}: patched {len(res['patched_heroes'])}; "
              + "; ".join(f"{sub} slope {nn[sub].get('slope', float('nan')):.3f} -> {dd_[sub].get('slope', float('nan')):.3f}, "
                          f"ll {nn[sub]['ll_gain_x1000']:+.3f} -> {dd_[sub]['ll_gain_x1000']:+.3f}"
                          for sub in ("patched", "unpatched") if sub in nn and sub in dd_), flush=True)
    out["slot_after_boundaries"] = slot
    out["elapsed_s"] = time.time() - t0
    fn = os.path.join(C.RESULTS, f"p3_b_demean{tag}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"wrote {fn} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
