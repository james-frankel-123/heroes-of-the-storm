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
  sd  the display sd: sqrt(s2 (1 + exp(-games on the hero / 10))), s2 the
      state-space posterior variance (P3_SHRINKAGE.md section 6)
  n   merged play counts (Heroes Profile and corpus rows) with day <= as_of_day
Evidence for a query: history rows with day <= as_of_day, day >= ev_lo (if
>= 0), the parity filter (1 odd days, 2 even days) and not in
[exclude_lo, exclude_hi]. The same filter applies to the layer's statistics.
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


def state_walk(aug, par, qpid, qday, row_mask, metrics, mweights, q_win, q_met, statlin):
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
    Lst = np.full((npl, H), -1, np.int32)
    eps2 = np.asarray(par.eps2, np.float32)
    beta, mF, sF = (np.asarray(statlin["beta"]), np.asarray(statlin["mean"]), np.asarray(statlin["sd"]))
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
                Lst[cp_, ch_] = dd
                np.add.at(Sst, (pi, hi), aug["r_adj"][rr] / aug["v"][rr])
                np.add.at(Pst, (pi, hi), 1.0 / aug["v"][rr])
                np.add.at(Nst, (pi, hi), 1.0)
                Xm = metrics[rr].astype(np.float32)
                g = ~np.isnan(Xm)
                wr = mweights[rr].astype(np.float32)
                np.add.at(Xst, (pi, hi), np.where(g, Xm, 0) * wr[:, None])
                np.add.at(Cst, (pi, hi), g.astype(np.float32) * wr[:, None])
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
        F = np.concatenate([Fh, np.repeat(Fa[:, None, :], H, 1)], 2)
        out["stat"][q] = (((F - mF) / sF) @ beta).astype(np.float32)
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


def answer(qdir, adir, only=None):
    t0 = time.time()
    os.makedirs(adir, exist_ok=True)
    d, aug = R.load_aug("merged")
    keys = d["player_keys"]
    ko = np.argsort(keys)
    files = sorted(glob.glob(os.path.join(qdir, "*.npz")))
    if only:
        files = [os.path.join(qdir, f) for f in only]
    cache = {}
    for fq in files:
        Q = np.load(fq)
        qk = Q["player_key"].astype(np.int64)
        j = np.minimum(np.searchsorted(keys[ko], qk), len(keys) - 1)
        known = keys[ko][j] == qk
        qpid = np.where(known, ko[j], -1)
        nq = len(qk)
        S_out = np.zeros((nq, H), np.float32)
        SD_out = np.zeros((nq, H), np.float32)
        N_out = np.zeros((nq, H), np.float32)
        groups = {}
        for i in range(nq):
            if qpid[i] < 0:
                continue
            g = (int(Q["half"][i]), int(Q["ev_lo"][i]), int(Q["ev_parity"][i]), int(Q["exclude_lo"][i]),
                 int(Q["exclude_hi"][i]))
            groups.setdefault(g, []).append(i)
        for g, idx in groups.items():
            idx = np.array(idx)
            half, ev_lo, par_, ex_lo, ex_hi = g
            armn, ptag, stem = ARMS[half]
            if armn not in cache:
                meta = arm_meta(armn)
                par = R.get_params(ptag)
                mets, mw = R.aug_metrics(aug, par.meta.get("sb", "legacy"), d)
                lay = np.load(os.path.join(R.RCF_CACHE, f"{stem}_layer.npz"))
                cache[armn] = (meta, par, mets, mw, lay)
            meta, par, mets, mw, lay = cache[armn]
            filt = lambda dd: evidence_mask(dd, ev_lo, par_, ex_lo, ex_hi)
            rm = filt(aug["day"].astype(np.int64))
            st = state_walk(aug, par, qpid[idx], Q["as_of_day"][idx].astype(np.int64), rm, mets, mw,
                            meta["q_win"], meta["q_met"], meta["statlin"])
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
            rho, nu = layer_terms(d, lay, lm, qpid[idx], Q["as_of_day"][idx].astype(np.int64), filt)
            est = est.reshape(len(idx), H) + lm["drift_scale"] * rho[:, None] \
                + lm["learn_scale"] * nu[:, None] * (n_full < 5)
            # direct scoreboard channel, then the smooth calibration
            est = est + st["stat"]
            B2 = TL.shrink_basis(res, rich=2 if "rich=2" in meta["postcal"]["basis"] else True)
            est = est * (B2 @ np.asarray(meta["postcal"]["coef"])).reshape(len(idx), H)
            S_out[idx] = est
            SD_out[idx] = np.sqrt(var * (1 + np.exp(-n_full / 10.0)))
            N_out[idx] = n_full
        # columns in shared.HEROES order (column i = shared.HEROES[i])
        perm = sh_perm(d["hero_names"])
        outp = {k: Q[k] for k in QFIELDS}
        outp.update(s=S_out[:, perm], sd=SD_out[:, perm], n=N_out[:, perm], known=qpid >= 0)
        np.savez(os.path.join(adir, os.path.basename(fq)), **outp)
        print(f"answered {os.path.basename(fq)}: {nq:,} queries, {int((qpid >= 0).sum()):,} known players "
              f"({time.time() - t0:.0f}s)", flush=True)


def selfcheck(arm=None, n=3000, seed=0):
    """The state at the played hero must equal the arm's own slot prediction
    (minus the MMR term, which the arm adds only in the game combiner) for
    random V2 slots: query = (player, day - 1, no filter, half 0)."""
    import tempfile
    arm = arm or ARMS[0][0]
    z = np.load(os.path.join(R.RCF_RES, "p3c", f"{arm}__merged.npz"))
    rng = np.random.default_rng(seed)
    i = rng.choice(np.flatnonzero(z["s_win"] == 2), n, replace=False)
    with tempfile.TemporaryDirectory() as td:
        qd, ad = os.path.join(td, "q"), os.path.join(td, "a")
        os.makedirs(qd)
        np.savez(os.path.join(qd, "check.npz"), player_key=z["s_key"][i].astype(np.int64),
                 as_of_day=(z["s_day"][i] - 1).astype(np.int32), ev_lo=np.full(n, -1, np.int32),
                 ev_parity=np.zeros(n, np.int8), exclude_lo=np.full(n, -1, np.int32),
                 exclude_hi=np.full(n, -1, np.int32), half=np.zeros(n, np.int8), hero=np.full(n, -1, np.int16))
        answer(qd, ad)
        a = np.load(os.path.join(ad, "check.npz"))
        from shared import HEROES
        names = [str(h) for h in np.load(os.path.join(R.C.CACHE, "x_slots_ext.npz"), mmap_mode="r")["hero_names"]]
        to_sh = np.array([HEROES.index(h) for h in names])
        s = a["s"][np.arange(n), to_sh[z["s_hero"][i]]]
        diff = np.abs(s - z["s_pred"][i])
        print(f"selfcheck {arm}: max |state - arm prediction| {diff.max():.2e}, median {np.median(diff):.2e}")
        return diff


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("answer")
    p.add_argument("qdir")
    p.add_argument("adir")
    p.add_argument("--only", nargs="*", help="file names in this order (default: every file)")
    sub.add_parser("selfcheck")
    a = ap.parse_args()
    if a.cmd == "answer":
        answer(a.qdir, a.adir, a.only)
    else:
        selfcheck()


if __name__ == "__main__":
    main()
