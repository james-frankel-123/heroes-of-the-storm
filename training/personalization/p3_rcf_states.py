"""
P3 player states of the chosen shrinkage model for the tournament
(P3T_WALK_RECORD / P3T_WALK_REPLAY files of p3t_data.walk).

A query file holds one row per query: player_key (region << 40 | blizz_id),
as_of_day, ev_lo, ev_parity, exclude_lo, exclude_hi, half, hero. The answer
file echoes those fields and adds, over all 90 heroes in shared.HEROES order:
  s   the chosen model's estimate at each hero, in fractions of win
      probability on the residual scale y - wp: experience offset (slot-row
      counts) + state-space posterior (win drift q_win, metric drift q_met)
      reshaped by the V1 shrinkage function + direct scoreboard channel +
      temporal layer (drift and new-hero learning) times the V1 smooth
      calibration multiplier. The lobby-level MMR term is left out.
  sd  the base sd: sqrt(s2), s2 the state-space Gaussian posterior variance
      (same meaning as the old interface's sqrt(var_all)). It is the
      posterior of the Gaussian walk, not a validated sd of the full estimator
      (shrinkage function, direct channel and calibration change the mean only).
  sd_display  the site's display band: sqrt(s2 (1 + exp(-games on the hero / 10)))
      (P3_SHRINKAGE.md section 6); not for the tournament
  n   merged play counts (Heroes Profile and corpus rows) with day <= as_of_day
Evidence for a query: history rows with day <= as_of_day, day >= ev_lo (if
>= 0), the parity filter (1 odd days, 2 even days) and not in
[exclude_lo, exclude_hi]. A filtered query (any of the three) gets no
temporal layer: the layer's residuals come from unfiltered predictions, so
they would carry excluded outcomes (as the old judge contract, no layer).
half 0 uses the drafter fit (full_ms, the arm "chosen"), half 1 the judge fit
on the other player half (full_ms_flip, the arm "chosen_flip"); every V1
coefficient comes from that arm's metadata.

Usage (definitive env, remote/def_env.sh):
  python3 personalization/p3_rcf_states.py answer <query dir> <answer dir>
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C
import p3_rcf as R
import p3_rcf_tails as TL

H = 90
# half -> (arm, params tag, walk-parts stem); P3_RCF_STATE_ARMS (json) overrides
ARMS = {0: ("chosen", "full_ms", "parts_chosen"), 1: ("chosen_flip", "full_ms_flip", "parts_chosen_flip")}
if os.environ.get("P3_RCF_STATE_ARMS"):
    ARMS = {int(k): tuple(v) for k, v in json.loads(os.environ["P3_RCF_STATE_ARMS"]).items()}
QFIELDS = ("player_key", "as_of_day", "ev_lo", "ev_parity", "exclude_lo", "exclude_hi", "half", "hero")


def sh_perm(hero_names):
    """column permutation: shared.HEROES index i -> slot-table hero index"""
    from shared import HEROES
    s = {h: i for i, h in enumerate(HEROES)}
    to_sh = np.array([s[str(h)] for h in hero_names], np.int64)
    inv = np.empty(H, np.int64)
    inv[to_sh] = np.arange(H)
    return inv


def arm_meta(name):
    z = np.load(os.path.join(R.RCF_RES, "p3c", f"{name}__merged.npz"))
    return json.loads(str(z["meta"]))


def evidence_mask(day, ev_lo, parity, ex_lo, ex_hi):
    m = np.ones(len(day), bool)
    if ev_lo >= 0:
        m &= day >= ev_lo
    if parity == 1:
        m &= (day % 2) == 1
    elif parity == 2:
        m &= (day % 2) == 0
    if ex_lo >= 0:
        m &= ~((day >= ex_lo) & (day <= ex_hi))
    return m


def state_walk(aug, par, qpid, qday, row_mask, metrics, mweights, q_win, q_met, statlin, xmetrics=None,
               hero_names=None):
    """All-hero state of each query (player qpid, rows with day <= qday that
    pass row_mask). Same information filter as p3_rcf.online_walk (Kalman
    drift on win and metric cells). Returns per query (nq, H): mean, var
    (Gaussian posterior), S, P (filtered own evidence), n (merged counts),
    nsl (slot-row counts), last (last evidence day, -1 none), stat (direct
    metric channel contribution)."""
    day, pid0, hero = aug["day"], aug["pid"], aug["hero"]
    slot = aug["slot"]
    up = np.unique(qpid)
    cmap = np.full(int(aug["n_players"]), -1, np.int64)
    cmap[up] = np.arange(len(up))
    pid = cmap[pid0]
    act = pid >= 0                                           # rows of queried players (counts use all of them)
    npl = len(up)
    Mn = par.G.shape[0]
    Sst = np.zeros((npl, H))
    Pst = np.zeros((npl, H))
    Nst = np.zeros((npl, H), np.float32)                     # evidence rows (inside the filter)
    Nfull = np.zeros((npl, H), np.float32)                   # every history row (full-history counts)
    Nslf = np.zeros((npl, H), np.float32)                    # every slot row (experience offset)
    Xst = np.zeros((npl, H, Mn), np.float32)
    Cst = np.zeros((npl, H, Mn), np.float32)
    Mx = 0 if xmetrics is None else xmetrics.shape[1]
    if Mx:
        XXst = np.zeros((npl, H, Mx), np.float32)
        CXst = np.zeros((npl, H, Mx), np.float32)
    Lst = np.full((npl, H), -1, np.int32)
    eps2 = np.asarray(par.eps2, np.float32)
    beta, mF, sF = (np.asarray(statlin["beta"]), np.asarray(statlin["mean"]), np.asarray(statlin["sd"]))
    if statlin.get("group"):
        # metric x group weights: column h uses the shared weights plus its group's deviations
        gidx, _ = R.statlin_groups(statlin["group"], np.arange(H), hero_names)
        Bh = beta[None, :] + np.asarray(statlin["group_dev"])[gidx]   # (H, K)
    else:
        Bh = None
    nq = len(qpid)
    out = {k: np.empty((nq, H), np.float64) for k in ("mean", "var", "S", "P")}
    out.update({k: np.empty((nq, H), np.float32) for k in ("n", "nfull", "nslf", "stat")})
    out["last"] = np.empty((nq, H), np.int32)
    o = np.argsort(qday, kind="stable")
    ud, qs = np.unique(qday[o], return_index=True)
    qe = np.r_[qs[1:], nq]
    ap = 0
    for t, a_, b_ in zip(ud, qs, qe):
        e = np.searchsorted(day, t, side="right")          # rows with day <= as_of_day
        if e > ap:
            rr_all = np.arange(ap, e)
            rr_all = rr_all[act[rr_all]]
            np.add.at(Nfull, (pid[rr_all], hero[rr_all]), 1.0)
            slr = rr_all[slot[rr_all] >= 0]
            np.add.at(Nslf, (pid[slr], hero[slr]), 1.0)
            rr_all = rr_all[row_mask[rr_all]]
            dd_all = day[rr_all]
            ub, bs = np.unique(dd_all, return_index=True)
            be = np.r_[bs[1:], len(rr_all)]
            for dd, s_, e_ in zip(ub, bs, be):
                rr = rr_all[s_:e_]
                pi, hi = pid[rr], hero[rr]
                cell = np.unique(pi.astype(np.int64) * H + hi)
                cp_, ch_ = cell // H, cell % H
                dt = np.where(Lst[cp_, ch_] >= 0, dd - Lst[cp_, ch_], 0).astype(np.float64)
                f = 1.0 / (1.0 + q_win * Pst[cp_, ch_] * dt)
                Sst[cp_, ch_] *= f
                Pst[cp_, ch_] *= f
                fm_ = (1.0 / (1.0 + q_met * (Cst[cp_, ch_] / eps2[None, :]) * dt[:, None])).astype(np.float32)
                Xst[cp_, ch_] *= fm_
                Cst[cp_, ch_] *= fm_
                if Mx:
                    fx_ = (1.0 / (1.0 + q_met * CXst[cp_, ch_] * dt[:, None])).astype(np.float32)
                    XXst[cp_, ch_] *= fx_
                    CXst[cp_, ch_] *= fx_
                Lst[cp_, ch_] = dd
                np.add.at(Sst, (pi, hi), aug["r_adj"][rr] / aug["v"][rr])
                np.add.at(Pst, (pi, hi), 1.0 / aug["v"][rr])
                np.add.at(Nst, (pi, hi), 1.0)
                Xm = metrics[rr].astype(np.float32)
                g = ~np.isnan(Xm)
                wr = mweights[rr].astype(np.float32)
                np.add.at(Xst, (pi, hi), np.where(g, Xm, 0) * wr[:, None])
                np.add.at(Cst, (pi, hi), g.astype(np.float32) * wr[:, None])
                if Mx:
                    Xx = xmetrics[rr].astype(np.float32)
                    gx = ~np.isnan(Xx)
                    np.add.at(XXst, (pi, hi), np.where(gx, Xx, 0))
                    np.add.at(CXst, (pi, hi), gx.astype(np.float32))
            ap = e
        q = o[a_:b_]
        pq = cmap[qpid[q]]
        tq = t + 1                                           # the query's game day
        dtq = np.where(Lst[pq] >= 0, tq - Lst[pq], 0).astype(np.float64)
        fw = 1.0 / (1.0 + q_win * Pst[pq] * dtq)
        Sq, Pq = Sst[pq] * fw, Pst[pq] * fw
        Cq = Cst[pq]
        fm_ = 1.0 / (np.float32(1.0) + np.float32(q_met) * (Cq / eps2[None, None, :]) * dtq.astype(np.float32)[:, :, None])
        Xq, Cq = Xst[pq] * fm_, Cq * fm_
        hq = np.zeros(len(q), np.int64)
        m, v = R.posterior(par, Sq, Pq, hq, X=Xq, Cm=Cq, want_all=True)
        out["mean"][q], out["var"][q] = m, v
        out["S"][q], out["P"][q] = Sq, Pq
        out["n"][q], out["nfull"][q], out["nslf"][q] = Nst[pq], Nfull[pq], Nslf[pq]
        out["last"][q] = Lst[pq]
        Fh = Xq / (Cq + 5.0)                                 # (b, H, M)
        Fa = Xq.sum(1) / (Cq.sum(1) + 5.0)                   # (b, M)
        parts_ = [Fh, np.repeat(Fa[:, None, :], H, 1)]
        if Mx:
            CXq = CXst[pq]
            fx_ = 1.0 / (np.float32(1.0) + np.float32(q_met) * CXq * dtq.astype(np.float32)[:, :, None])
            XXq, CXq = XXst[pq] * fx_, CXq * fx_
            parts_ += [XXq / (CXq + 5.0), np.repeat((XXq.sum(1) / (CXq.sum(1) + 5.0))[:, None, :], H, 1)]
        F = np.concatenate(parts_, 2)
        del parts_
        st_ = np.empty(F.shape[:2], np.float32)
        for c0 in range(0, len(F), 4096):                    # bounded float64 temporaries
            Fz = (F[c0:c0 + 4096] - mF) / sF
            st_[c0:c0 + 4096] = Fz @ beta if Bh is None else np.einsum("bhk,hk->bh", Fz, Bh)
        out["stat"][q] = st_
        del F, Fz
    return out


def layer_terms(d, lay, meta, qpid, qday, filt):
    """Per-player drift and early-game learning statistics as of each query
    (slot games with day <= as_of_day that pass the evidence filter), by
    per-player cumulative sums (vectorized)."""
    day = d["day"].astype(np.int64)
    pid = d["pid"].astype(np.int64)
    u = lay["u"].astype(np.float64)
    early = lay["early"]
    keep = filt(day) & np.isfinite(u) & np.isin(pid, np.unique(qpid))
    r = np.flatnonzero(keep)
    o = np.lexsort((day[r], pid[r]))
    r = r[o]
    p_, d_ = pid[r], day[r]
    lam = np.log(2) / meta["hl_d"]
    d0 = int(d_.min()) if len(d_) else 0
    ew = np.exp(lam * (d_ - d0))
    cs = lambda x: np.r_[0.0, np.cumsum(x)]
    c_wu, c_w = cs(ew * u[r]), cs(ew)
    c_eu, c_e = cs(np.where(early[r], u[r], 0.0)), cs(early[r].astype(float))
    BIG = 1 << 20
    key = p_ * BIG + d_
    qk = qpid.astype(np.int64) * BIG + qday.astype(np.int64)
    hi = np.searchsorted(key, qk, side="right")             # rows of this player with day <= as_of
    lo = np.searchsorted(key, qpid.astype(np.int64) * BIG, side="left")
    sum_wu = c_wu[hi] - c_wu[lo]
    sum_w = c_w[hi] - c_w[lo]
    dec = np.exp(-lam * (qday.astype(np.int64) + 1 - d0))
    rho = sum_wu * dec / (sum_w * dec + meta["kd"])
    nu = (c_eu[hi] - c_eu[lo]) / ((c_e[hi] - c_e[lo]) + meta["ke"])
    return rho, nu


def filtered(ev_lo, parity, ex_lo):
    """True when the query's evidence filter drops any history (window start,
    parity or exclusion). Such states carry no temporal layer: the layer's
    residuals u = r - pred are built on unfiltered histories, so excluded
    outcomes would reach the state through pred (old judge contract: no layer)."""
    return ev_lo >= 0 or parity != 0 or ex_lo >= 0


def answer(qdir, adir, only=None, ctx=None):
    """ctx (optional dict) keeps the loaded data, metric caches, layer inputs and
    arm caches across calls (filtercheck mutates them in place)."""
    t0 = time.time()
    os.makedirs(adir, exist_ok=True)
    ctx = {} if ctx is None else ctx
    if "d" not in ctx:
        ctx["d"], ctx["aug"] = R.load_aug("merged")
        ctx["cache"], ctx["mcache"] = {}, {}
    d, aug = ctx["d"], ctx["aug"]
    keys = d["player_keys"]
    ko = np.argsort(keys)
    files = sorted(glob.glob(os.path.join(qdir, "*.npz")))
    if only:
        files = [os.path.join(qdir, f) for f in only]
    cache, mcache = ctx["cache"], ctx["mcache"]
    for fq in files:
        Qz = np.load(fq)
        Q = {k: Qz[k] for k in Qz.files}                       # read once (npz access re-reads the zip)
        qk = Q["player_key"].astype(np.int64)
        j = np.minimum(np.searchsorted(keys[ko], qk), len(keys) - 1)
        known = keys[ko][j] == qk
        qpid = np.where(known, ko[j], -1)
        nq = len(qk)
        S_out = np.zeros((nq, H), np.float32)
        SD_out = np.zeros((nq, H), np.float32)
        SDD_out = np.zeros((nq, H), np.float32)
        N_out = np.zeros((nq, H), np.float32)
        groups = {}
        G = np.column_stack([Q["half"].astype(np.int64), Q["ev_lo"].astype(np.int64), Q["ev_parity"].astype(np.int64),
                             Q["exclude_lo"].astype(np.int64), Q["exclude_hi"].astype(np.int64)])
        ok = np.flatnonzero(qpid >= 0)
        ug, ginv = np.unique(G[ok], axis=0, return_inverse=True)
        for k_, g in enumerate(ug):
            groups[tuple(int(x) for x in g)] = ok[ginv.ravel() == k_].tolist()
        items = []
        for g, idx in groups.items():
            idx = np.array(idx)
            for c0 in range(0, len(idx), 100_000):           # bound memory of the all-hero arrays
                items.append((g, idx[c0:c0 + 100_000]))
        for g, idx in items:
            half, ev_lo, par_, ex_lo, ex_hi = g
            armn, ptag, stem = ARMS[half]
            if armn not in cache:
                meta = arm_meta(armn)
                par = R.get_params(ptag)
                card = par.meta.get("sb", "legacy")
                if card not in mcache:                       # one copy per card, shared by both halves
                    mcache[card] = R.aug_metrics(aug, card, d)
                mets, mw = mcache[card]
                ext = (meta.get("statlin") or {}).get("extra")
                if ext and ext not in mcache:
                    mcache[ext] = R.aug_metrics(aug, ext, d)
                xm = mcache[ext][0] if ext else None
                lz = np.load(os.path.join(R.RCF_CACHE, f"{stem}_layer.npz"))
                lay = {k: lz[k] for k in lz.files}             # materialized (npz access re-reads)
                cache[armn] = (meta, par, mets, mw, lay, xm)
            meta, par, mets, mw, lay, xm = cache[armn]
            filt = lambda dd: evidence_mask(dd, ev_lo, par_, ex_lo, ex_hi)
            rm = filt(aug["day"].astype(np.int64))
            st = state_walk(aug, par, qpid[idx], Q["as_of_day"][idx].astype(np.int64), rm, mets, mw,
                            meta["q_win"], meta["q_met"], meta["statlin"], xmetrics=xm,
                            hero_names=d["hero_names"])
            tq = Q["as_of_day"][idx].astype(np.int64) + 1
            # n_all / np_all: evidence-window counts (the record the shrinkage and calibration describe);
            # n_full: every history row (output n, the new-hero flag); slot counts for the offset
            n_all, np_all = st["n"].astype(np.float64), st["n"].sum(1, keepdims=True).astype(np.float64)
            n_full = st["nfull"].astype(np.float64)
            # experience offset from slot-row counts
            table = aug["table"]
            nb = len(C.NPH_EDGES)
            bp = np.searchsorted(C.NP_EDGES, st["nslf"].sum(1), side="right") - 1
            bh = np.searchsorted(C.NPH_EDGES, st["nslf"], side="right") - 1
            mu = table[bp[:, None] * nb + bh]
            # shrinkage function on the own-hero record (closed-form leave-own-out)
            var = st["var"].astype(np.float64)
            mean = st["mean"].astype(np.float64)
            P = st["P"].astype(np.float64)
            Sv = st["S"].astype(np.float64)
            v0 = 1.0 / np.maximum(1.0 / np.maximum(var, 1e-12) - P, 1e-12)
            m0 = v0 * (mean / np.maximum(var, 1e-12) - Sv)
            gap = np.where(st["last"] >= 0, tq[:, None] - st["last"], -1)
            fl = lambda x: x.reshape(-1)
            res = {"nh": fl(n_all), "np": fl(np.repeat(np_all, H, 1)), "gap": fl(gap.astype(np.float64))}
            tm = meta["tail"]
            B = TL.shrink_basis(res, rich=True)
            w = fl(P) / (fl(P) + 1.0 / fl(v0))
            yb = np.where(fl(P) > 0, fl(Sv) / np.maximum(fl(P), 1e-12), fl(m0))
            own = w * (yb - fl(m0))
            # the arm's prediction is offset + shrinkage-function mean (which has its own offset term)
            est = fl(mu) + (B @ np.asarray(tm["beta_prior"])) * fl(m0) + (B @ np.asarray(tm["beta_own"])) * own \
                + (B @ np.asarray(tm["beta_offset"])) * fl(mu)
            # temporal layer
            lm = meta["layer"]
            est = est.reshape(len(idx), H)
            if not filtered(ev_lo, par_, ex_lo):
                rho, nu = layer_terms(d, lay, lm, qpid[idx], Q["as_of_day"][idx].astype(np.int64), filt)
                est = est + lm["drift_scale"] * rho[:, None] + lm["learn_scale"] * nu[:, None] * (n_full < 5)
            # direct scoreboard channel, then the smooth calibration
            est = est + st["stat"]
            B2 = TL.shrink_basis(res, rich=2 if "rich=2" in meta["postcal"]["basis"] else True)
            est = est * (B2 @ np.asarray(meta["postcal"]["coef"])).reshape(len(idx), H)
            S_out[idx] = est
            SD_out[idx] = np.sqrt(var)                         # base Gaussian posterior sd (old sqrt(var_all))
            SDD_out[idx] = np.sqrt(var * (1 + np.exp(-n_full / 10.0)))   # display band (site only)
            N_out[idx] = n_full
        # columns in shared.HEROES order (column i = shared.HEROES[i])
        perm = sh_perm(d["hero_names"])
        outp = {k: Q[k] for k in QFIELDS}
        outp.update(s=S_out[:, perm], sd=SD_out[:, perm], sd_display=SDD_out[:, perm], n=N_out[:, perm],
                    known=qpid >= 0)
        np.savez(os.path.join(adir, os.path.basename(fq)), **outp)
        print(f"answered {os.path.basename(fq)}: {nq:,} queries, {int((qpid >= 0).sum()):,} known players "
              f"({time.time() - t0:.0f}s)", flush=True)


SELFCHECK_TOL = 1e-6      # float32 answer files; observed differences are ~1.5e-8


def selfcheck(half=0, n=3000, seed=0, ctx=None):
    """The unfiltered state at the played hero must equal the arm's own slot
    prediction (minus the MMR term, which the arm adds only in the game
    combiner) for random V2 slots: query = (player, day - 1, no filter, half).
    Raises unless every difference is finite and below SELFCHECK_TOL."""
    import tempfile
    arm = ARMS[half][0]
    z = np.load(os.path.join(R.RCF_RES, "p3c", f"{arm}__merged.npz"))
    rng = np.random.default_rng(seed)
    i = rng.choice(np.flatnonzero(z["s_win"] == 2), n, replace=False)
    with tempfile.TemporaryDirectory() as td:
        qd, ad = os.path.join(td, "q"), os.path.join(td, "a")
        os.makedirs(qd)
        np.savez(os.path.join(qd, "check.npz"), player_key=z["s_key"][i].astype(np.int64),
                 as_of_day=(z["s_day"][i] - 1).astype(np.int32), ev_lo=np.full(n, -1, np.int32),
                 ev_parity=np.zeros(n, np.int8), exclude_lo=np.full(n, -1, np.int32),
                 exclude_hi=np.full(n, -1, np.int32), half=np.full(n, half, np.int8), hero=np.full(n, -1, np.int16))
        answer(qd, ad, ctx=ctx)
        a = np.load(os.path.join(ad, "check.npz"))
        from shared import HEROES
        names = [str(h) for h in np.load(os.path.join(R.C.CACHE, "x_slots_ext.npz"), mmap_mode="r")["hero_names"]]
        to_sh = np.array([HEROES.index(h) for h in names])
        s = a["s"][np.arange(n), to_sh[z["s_hero"][i]]]
        diff = np.abs(s.astype(np.float64) - z["s_pred"][i])
        print(f"selfcheck {arm} (half {half}): max |state - arm prediction| {np.nanmax(diff):.2e}, "
              f"median {np.nanmedian(diff):.2e}", flush=True)
        assert bool(a["known"].all()), "selfcheck: unknown players"
        assert np.isfinite(a["s"]).all() and np.isfinite(a["sd"]).all(), f"selfcheck {arm}: non-finite state"
        assert np.isfinite(diff).all() and diff.max() < SELFCHECK_TOL, \
            f"selfcheck {arm}: max difference {np.nanmax(diff):.2e} >= {SELFCHECK_TOL:g}"
        return diff


FILTERS = {  # name -> (half, ev_lo, parity, exclude_lo, exclude_hi); days as in p3t_data
    "judge": (1, 20635, 0, -1, -1),                  # judge window from 2026-07-01 (J0)
    "split_even": (1, -1, 2, 20596, 20634),          # even days, outside the tournament window
    "split_odd": (0, -1, 1, -1, -1),                 # odd days
    "exclude": (0, -1, 0, 20596, 20634),             # exclusion only
}


def filtercheck(n_players=50, seed=0, ctx=None):
    """Two checks of every filter form (FILTERS), on the real data:
    (a) poison: perturb, for n_players players, every outcome and scoreboard
        value outside the query's filter and every layer residual; filtered
        answers must not change at all, while unfiltered answers of the same
        players must change (the perturbation reaches them) and perturbing
        evidence inside the filter must change the filtered answers;
    (b) reference: the filtered walk state equals the state of a walk over a
        copy of the history with the out-of-filter rows physically removed
        (posterior mean, variance, own evidence, counts, direct channel)."""
    import tempfile
    ctx = {} if ctx is None else ctx
    if "d" not in ctx:
        ctx["d"], ctx["aug"] = R.load_aug("merged")
        ctx["cache"], ctx["mcache"] = {}, {}
    d, aug = ctx["d"], ctx["aug"]
    rng = np.random.default_rng(seed)
    day = aug["day"].astype(np.int64)
    qday = 20707                                      # as_of_day 2026-09-11, the judge window end (J1)
    # players with history before, inside the tournament window and inside the judge window
    before = np.unique(aug["pid"][day < 20596])
    tw = np.unique(aug["pid"][(day >= 20596) & (day <= 20634)])
    inside = np.unique(aug["pid"][(day >= 20635) & (day <= qday)])
    cand = np.intersect1d(np.intersect1d(before, tw), inside)
    keys = np.asarray(d["player_keys"])
    pick = rng.choice(cand[cand < len(keys)], n_players, replace=False)
    rows_q = []
    for nm, (half, lo, par_, xlo, xhi) in FILTERS.items():
        rows_q.append((half, lo, par_, xlo, xhi))
    rows_q.append((0, -1, 0, -1, -1))                 # unfiltered control, half 0
    rows_q.append((1, -1, 0, -1, -1))                 # unfiltered control, half 1
    nq = len(rows_q) * n_players
    Q = {"player_key": np.tile(keys[pick].astype(np.int64), len(rows_q)),
         "as_of_day": np.full(nq, qday, np.int32),
         "ev_lo": np.repeat([r[1] for r in rows_q], n_players).astype(np.int32),
         "ev_parity": np.repeat([r[2] for r in rows_q], n_players).astype(np.int8),
         "exclude_lo": np.repeat([r[3] for r in rows_q], n_players).astype(np.int32),
         "exclude_hi": np.repeat([r[4] for r in rows_q], n_players).astype(np.int32),
         "half": np.repeat([r[0] for r in rows_q], n_players).astype(np.int8),
         "hero": np.full(nq, -1, np.int16)}
    with tempfile.TemporaryDirectory() as td:
        qd = os.path.join(td, "q")
        os.makedirs(qd)
        np.savez(os.path.join(qd, "fc.npz"), **Q)

        def run(tag):
            ad = os.path.join(td, tag)
            answer(qd, ad, ctx=ctx)
            z = np.load(os.path.join(ad, "fc.npz"))
            return {k: z[k].copy() for k in ("s", "sd", "n")}

        A = run("a")
        prow = np.flatnonzero(np.isin(aug["pid"], pick))
        lays = [v[4] for v in ctx["cache"].values()]
        mets = [v for v in ctx["mcache"].values()]

        def poison(sel):
            r = prow[sel]
            aug["r_adj"][r] = rng.normal(0, 1, len(r)).astype(aug["r_adj"].dtype)
            for Z, W in mets:
                Z[r] = rng.normal(0, 3, (len(r), Z.shape[1])).astype(Z.dtype)
            return r

        # (a) per filter: poison the rows outside it; that filter's answers must not move
        nf = len(FILTERS) * n_players
        sp = np.flatnonzero(np.isin(d["pid"], pick))
        for lay in lays:                              # layer residuals reach unfiltered states only
            lay["u"][sp] = rng.normal(0, 1, len(sp)).astype(lay["u"].dtype)
        prev = A
        ctl = 0.0
        for j, (nm, (half, lo, par_, xlo, xhi)) in enumerate(FILTERS.items()):
            ex = (~evidence_mask(day[prow], lo, par_, xlo, xhi)) & (day[prow] <= qday)
            assert ex.sum() > 0, f"filtercheck: no excluded rows for {nm}"
            poison(ex)
            B = run(f"b{j}")
            sl = slice(j * n_players, (j + 1) * n_players)
            for k in ("s", "sd"):
                dmax = np.abs(B[k][sl].astype(np.float64) - prev[k][sl]).max()
                assert dmax == 0.0, f"filtercheck {nm}: {k} changed by {dmax:.3e} after poisoning excluded evidence"
            assert np.array_equal(B["n"], A["n"]), "filtercheck: counts changed"
            ctl = max(ctl, np.abs(B["s"][nf:].astype(np.float64) - prev["s"][nf:]).max())
            print(f"  poison {nm}: {int(ex.sum()):,} excluded rows perturbed, filtered answers unchanged", flush=True)
            prev = B
        print(f"  unfiltered control max change {ctl:.2e}", flush=True)
        assert ctl > 1e-4, "filtercheck: perturbation did not reach the unfiltered control (vacuous)"
        # sensitivity: perturbing evidence inside the judge window moves every filtered group
        poison((day[prow] >= 20635) & (day[prow] <= qday))
        C_ = run("c")
        for j, nm in enumerate(FILTERS):
            sl = slice(j * n_players, (j + 1) * n_players)
            dch = np.abs(C_["s"][sl].astype(np.float64) - prev["s"][sl]).max()
            print(f"  sensitivity {nm}: max change {dch:.2e}", flush=True)
            assert dch > 1e-4, f"filtercheck: {nm} answers ignore evidence inside the filter"
    # (b) reference walk on a physically filtered copy (only the picked players' rows)
    for nm, (half, lo, par_, xlo, xhi) in FILTERS.items():
        meta, par, mets_, mw, lay, xm = ctx["cache"][ARMS[half][0]]
        qpid = pick.astype(np.int64)
        qd_ = np.full(len(qpid), qday, np.int64)
        keep = evidence_mask(day, lo, par_, xlo, xhi)
        st_f = state_walk(aug, par, qpid, qd_, keep, mets_, mw, meta["q_win"], meta["q_met"], meta["statlin"],
                          xmetrics=xm, hero_names=d["hero_names"])
        sub = np.flatnonzero(np.isin(aug["pid"], pick) & keep)
        aug_s = {k: (v[sub] if (hasattr(v, "shape") and v.shape[:1] == aug["day"].shape) else v) for k, v in aug.items()}
        st_r = state_walk(aug_s, par, qpid, qd_, np.ones(len(sub), bool), mets_[sub], mw[sub], meta["q_win"],
                          meta["q_met"], meta["statlin"], xmetrics=None if xm is None else xm[sub],
                          hero_names=d["hero_names"])
        for k in ("mean", "var", "S", "P", "n", "stat", "last"):
            a_, b_ = np.asarray(st_f[k], np.float64), np.asarray(st_r[k], np.float64)
            dmax = np.abs(a_ - b_).max()
            assert np.isfinite(a_).all() and dmax < SELFCHECK_TOL, f"filtercheck reference {nm}: {k} differs by {dmax:.3e}"
        print(f"  reference {nm}: filtered walk = walk on the physically filtered history", flush=True)
    print(f"filtercheck passed ({n_players} players, filters {list(FILTERS)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("answer")
    p.add_argument("qdir")
    p.add_argument("adir")
    p.add_argument("--only", nargs="*", help="file names in this order (default: every file)")
    p = sub.add_parser("selfcheck", help="both halves (unfiltered) and the filter checks; raises on failure")
    p.add_argument("--halves", default="0,1")
    p.add_argument("--no-filtercheck", action="store_true")
    a = ap.parse_args()
    if a.cmd == "answer":
        answer(a.qdir, a.adir, a.only)
    else:
        ctx = {}
        for h in (int(x) for x in a.halves.split(",")):
            selfcheck(h, ctx=ctx)
        if not a.no_filtercheck:
            filtercheck(ctx=ctx)       # mutates ctx's data: last


if __name__ == "__main__":
    main()
