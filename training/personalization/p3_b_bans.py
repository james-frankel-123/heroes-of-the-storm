"""
P3 review item 8 (REVIEW section 4.2 and 6.9; second review sections 4 and 5):
the personalized ban model, checked against the main-ban natural experiment.

Five parts, all on snapshot games (the ban table cache/x_bans.npz covers the
snapshot only, so no post-snapshot game and never the sealed build
2.55.17.98025 is read; --final is accepted for the common interface and only
changes the output name).

Features. Two causal per-slot walks (p3_x_ban_feat._walk):
  "time" games that ended before this game started (the fixed walker of
         p3_x_ban_feat), used to replicate the published natural experiment;
  "lag1" games of earlier days only (the lag-1 contract), used for every
         model fit, every lobby query and the pre-ban-information version.
The lobby ban model uses lag-1 features throughout (strength arm fit on V1,
per-hero query features, personal tables, imitation recency).

1. Calibration. At the first ban decision of each held-out V2 lobby
   (p3_x_ban_model protocol), for each opponent with 30+ earlier-day games and
   his main available: the primary arm's predicted personal drop if his main
   is banned, with his main-pick probability from (a) the imitation model and
   (b) the personal GD (cache/pgd_personal.pt pick head on the meta GD at the
   same state), against the realized drop from the natural experiment, by
   type (one-trick >= 50% main share, specialist 25-50%, flexible < 25%).
   Recovered share = predicted / realized, CI from the opponent bootstrap of
   the prediction and the player-cluster bootstrap of the realized effect.
   Also the real frequency with which those opponents picked the main when it
   was still available at their pick.
2. Winner's curse. The +3.4pp is value(own argmax) - value(population ban)
   under one model. Two cross-fits:
   (a) coefficients: the strength arm refit on two random halves of V1
       players (A, B); choose with A, value with B and the reverse;
   (b) history: per player, earlier-day games split by alternating game
       index into two halves; the per-hero query features (counts, MAWP,
       EWMA residual, main, share) rebuilt from each half; choose with one,
       value with the other. The GP skill s and the imitation pick
       probabilities keep the full history in both halves, so only the noise
       in the count, MAWP and EWMA features is cross-fit. Each half has half the data, so its selection
       bias is larger than the full model's: the half-sample shrink is a
       conservative bound; the point correction scales the half-sample bias
       by 1/sqrt(2) (bias of a max grows with the noise sd).
   CIs: lobby bootstrap. Plus the realized check on real bans (team residual
   on own minus opponents' personal value of the real bans), split by
   whether the real ban was the model's argmax.
3. Targeting by tier: P(opponents ban the main) against the per-team ban rate
   of the hero in the same build (as published) and in the same build and
   real league tier. Player-cluster bootstrap CIs on the ratios.
4. Confounding: (a) opponents' largest premade size (dummies 2..5) and the
   player's own party size as controls, for the main effects and the placebo;
   (b) one-tricks split by MAWP of the main (hot >= 0.55, mid, cold < 0.50):
   one within-player regression with treated x state and the state main
   effects, and the published form (separate regressions per stratum); the
   bound on targeting is the overall effect minus the cold effect (if all of
   the hot excess were targeting);
   (c) pre-ban-information version: main from earlier-day history, treated =
   opponents banned it at a step before the player's pick, control = nobody
   banned it before his pick; no conditioning on later picks or on bans
   after his pick. Outcome y (intent to treat), with r_self for reference.
5. The V2 window label: the split day is computed from the data and printed.

Usage (from training/): python personalization/p3_b_bans.py [--sample 0.1] [--nboot 200] [--final]
Output: C.RESULTS/p3_b_bans[...].json; cache b8_slotfeat_{time,lag1}.npz
"""
import argparse
import datetime
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from numba import njit

from p3_heroes import NUM_HEROES, HKEY
import p3_hs_core as C
import p3_x_common as X
import p3_x_ban_feat as F
import p3_x_ban_model as BM

BANS = os.path.join(C.CACHE, "x_bans.npz")
PICKO = os.path.join(C.CACHE, "pickorder_2024q2.npz")
FEAT_VER = "v1"
PRIMARY_COLS = list(range(10))  # "combo + forced off main" in p3_x_ban_model.ARMS
NAT_START = "2024-07-01"
TYPES = {"one-trick": (0.5, 1.01), "specialist": (0.25, 0.5), "flexible": (0.0, 0.25)}

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def date_of(day):
    return str(C.EPOCH + datetime.timedelta(days=int(day)))


# ------------------------------------------------------------------ features

def fake_times(d):
    """Day-level times: a game ends at the last second of its day label and
    the query time is the start of the day, so only earlier days enter."""
    day = d["day"].astype(np.float64)
    return (day + 1) * 86400.0 - 1.0, day * 86400.0


def slot_feats(d, mode):
    path = os.path.join(C.CACHE, f"b8_slotfeat_{mode}_{FEAT_VER}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    t_end, t_start = F.play_times(d) if mode == "time" else fake_times(d)
    n = len(d["pid"])
    srt = np.lexsort((d["replay_id"], t_end, d["pid"]))
    pid = d["pid"][srt]
    brk = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    ends = np.r_[brk[1:], n]
    outs = [np.zeros(n, np.int64)] + [np.zeros(n) for _ in range(8)]
    qa = [np.zeros((1, NUM_HEROES), np.float32) for _ in range(6)]
    F._walk(brk.astype(np.int64), ends.astype(np.int64), d["hero"][srt].astype(np.int64),
            d["y"][srt].astype(np.float64), (d["y"] - d["wp"])[srt].astype(np.float64),
            t_end[srt], t_start[srt], np.full(n, -1, np.int64), *outs, *qa)
    names = ["main", "tot", "top_cnt", "dshare_main", "mawp_main", "ewr_main", "mawp_own", "ewr_own", "n_own"]
    res = {}
    for nm, a in zip(names, outs):
        b = np.empty_like(a)
        b[srt] = a
        res[nm] = b
    np.savez(path, **res)
    return res


@njit(cache=True)
def _walk_q(starts, ends, hero, y, r, t_end, t_start, inhist, qslot, q_cnt, q_mawp, q_ewr10):
    """Per-hero query features (counts, MAWP, EWMA residual 10) for query
    rows; rows with inhist False never enter the history."""
    a10 = 1 - np.exp(-np.log(2) / 10.0)
    bt = np.zeros((NUM_HEROES, 400))
    by = np.zeros((NUM_HEROES, 400))
    for p in range(starts.shape[0]):
        cnt = np.zeros(NUM_HEROES)
        ewr = np.zeros(NUM_HEROES)
        w10 = np.zeros(NUM_HEROES)
        head = np.zeros(NUM_HEROES, np.int64)
        ap = starts[p]
        for i in range(starts[p], ends[p]):
            now = t_start[i]
            while ap < i and t_end[ap] <= now:
                if inhist[ap]:
                    h0 = hero[ap]
                    cnt[h0] += 1
                    ewr[h0] = (1 - a10) * ewr[h0] + a10 * r[ap]
                    w10[h0] = (1 - a10) * w10[h0] + a10
                    bt[h0, head[h0] % 400] = t_end[ap]
                    by[h0, head[h0] % 400] = y[ap]
                    head[h0] += 1
                ap += 1
            q = qslot[i]
            if q >= 0:
                for h in range(NUM_HEROES):
                    q_cnt[q, h] = cnt[h]
                    q_mawp[q, h] = F._mawp(bt[h], by[h], int(cnt[h]), head[h], now)
                    q_ewr10[q, h] = ewr[h] / w10[h] if w10[h] > 0 else 0.0


def query_feats(d, rows, inhist=None):
    t_end, t_start = fake_times(d)
    n = len(d["pid"])
    srt = np.lexsort((d["replay_id"], t_end, d["pid"]))
    pid = d["pid"][srt]
    brk = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    ends = np.r_[brk[1:], n]
    qpos = np.full(n, -1, np.int64)
    qpos[rows] = np.arange(len(rows))
    ih = np.ones(n, bool) if inhist is None else inhist
    out = [np.zeros((len(rows), NUM_HEROES), np.float32) for _ in range(3)]
    _walk_q(brk.astype(np.int64), ends.astype(np.int64), d["hero"][srt].astype(np.int64),
            d["y"][srt].astype(np.float64), (d["y"] - d["wp"])[srt].astype(np.float64),
            t_end[srt], t_start[srt], ih[srt], qpos[srt], *out)
    return dict(zip(["q_cnt", "q_mawp", "q_ewr10"], out))


def history_halves(d):
    """Alternating split of each player's games (play order by day, replay)."""
    o = np.lexsort((d["replay_id"], d["day"], d["pid"]))
    pid = d["pid"][o]
    brk = np.r_[True, pid[1:] != pid[:-1]]
    first = np.maximum.accumulate(np.where(brk, np.arange(len(o)), 0))
    idx = np.arange(len(o)) - first
    half = np.empty(len(o), bool)
    half[o] = (idx % 2) == 0
    return half


def pgd_walk(d, rows):
    """p3_pgd_feats.walk on the lag-1 contract (earlier days only)."""
    import p3_pgd_feats as PF
    t_end, t_start = fake_times(d)
    n = len(d["pid"])
    srt = np.lexsort((d["replay_id"], t_end, d["pid"]))
    pid = d["pid"][srt]
    brk = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    ends = np.r_[brk[1:], n]
    qpos = np.full(n, -1, np.int64)
    qpos[rows] = np.arange(len(rows))
    arrs = [np.zeros((len(rows), NUM_HEROES), np.float32) for _ in range(5)]
    PF._walk(brk.astype(np.int64), ends.astype(np.int64), d["hero"][srt].astype(np.int64),
             d["y"][srt].astype(np.float64), t_end[srt], t_start[srt], qpos[srt], *arrs)
    return dict(zip(["n", "e20", "e100", "days", "mawp"], arrs))


# ------------------------------------------------------------------ regression

class Boot:
    """Poisson bootstrap weights per player id, shared by every regression so
    that draws pair across strata and specifications."""

    def __init__(self, n_players, nboot, seed=7):
        rng = np.random.RandomState(seed)
        self.W = rng.poisson(1.0, (nboot, int(n_players))).astype(np.uint8)


def fe_fit(yv, Xk, ctrl, grp, boot):
    """Within-player OLS of yv on [Xk, ctrl]; returns (beta of Xk, draws)."""
    Xk = np.atleast_2d(np.asarray(Xk, float).T).T if np.ndim(Xk) == 1 else np.asarray(Xk, float)
    m = Xk.shape[1]
    u, inv = np.unique(grp, return_inverse=True)
    Xm = np.column_stack([Xk] + [np.asarray(c, float) for c in ctrl])
    cnt = np.bincount(inv).astype(float)

    def demean(a):
        return a - (np.bincount(inv, weights=a) / cnt)[inv]
    Xd = np.column_stack([demean(Xm[:, j]) for j in range(Xm.shape[1])])
    yd = demean(np.asarray(yv, float))
    k = Xd.shape[1]
    XX = np.zeros((len(u), k, k))
    Xy = np.zeros((len(u), k))
    for a in range(k):
        Xy[:, a] = np.bincount(inv, weights=Xd[:, a] * yd, minlength=len(u))
        for b in range(a, k):
            XX[:, a, b] = XX[:, b, a] = np.bincount(inv, weights=Xd[:, a] * Xd[:, b], minlength=len(u))
    ridge = 1e-9 * np.eye(k)
    beta = np.linalg.solve(XX.sum(0) + ridge, Xy.sum(0))[:m]
    draws = np.zeros((boot.W.shape[0], m))
    for bi in range(boot.W.shape[0]):
        w = boot.W[bi, u].astype(float)
        draws[bi] = np.linalg.solve(np.tensordot(w, XX, 1) + ridge, w @ Xy)[:m]
    return beta, draws


def pp(beta, draws, j=0, sign=1.0):
    e = sign * 100 * float(beta[j])
    q = np.percentile(sign * 100 * draws[:, j], [2.5, 97.5])
    return {"pp": e, "ci": [float(min(q)), float(max(q))]}


# ------------------------------------------------------------------ natural experiment

def nat_setup(d):
    n = len(d["pid"])
    g, team = d["g"], d["team"].astype(int)
    n_games = int(g.max()) + 1
    B = np.load(BANS)
    grid = np.zeros(n_games, np.int64)
    grid[g] = d["replay_id"]
    o = np.argsort(B["replay_ids"])
    j = np.minimum(np.searchsorted(B["replay_ids"][o], grid), len(o) - 1)
    okg = B["replay_ids"][o][j] == grid
    bh = np.where(okg[:, None], B["hero"][o][j], -1)
    bt = np.where(okg[:, None], B["team"][o][j], -1)
    bk = np.where(okg[:, None], B["pick_number"][o][j], -1)
    po = np.load(PICKO)
    names = [str(x) for x in d["hero_names"]]
    hid = {h: i for i, h in enumerate(names)}
    pk = po["replay_ids"].astype(np.int64) * HKEY + np.array([hid.get(h, HKEY - 1) for h in po["hero"]])
    so = np.argsort(pk)
    key = d["replay_id"].astype(np.int64) * HKEY + d["hero"]
    jj = np.minimum(np.searchsorted(pk[so], key), len(so) - 1)
    okp = pk[so][jj] == key
    rank = np.where(okp, po["pick_rank"][so][jj], -1)
    log(f"bans matched {okg.mean():.4f}, pick ranks matched {okp.mean():.4f}")
    # builds and real tiers per game
    w = np.load(C.WP)
    ow = np.argsort(w["replay_ids"])
    bg = w["build_idx"][ow][np.searchsorted(w["replay_ids"][ow], grid)]
    tier_g = F.real_tier(grid)
    # party sizes
    party = d["party"]
    gteam = g.astype(np.int64) * 2 + team
    nz = party != 0
    pkey = np.where(nz, (gteam << 32) ^ (party.astype(np.int64) % 2147483647), -1)
    sz = np.ones(n, np.int64)
    _, pinv, pc = np.unique(pkey[nz], return_inverse=True, return_counts=True)
    sz[nz] = pc[pinv]
    big = np.ones(2 * n_games, np.int64)
    np.maximum.at(big, gteam, sz)
    opp_party = big[g.astype(np.int64) * 2 + 1 - team]
    # skill sums
    P = np.load(os.path.join(C.CACHE, "x_predall.npz"))
    s_self = (P["mu"] + P["m_cf2"]).astype(np.float64)
    S = np.zeros((n_games, 2))
    np.add.at(S, (g, team), s_self)
    r_self = d["r"] - (S[g, team] - s_self) + S[g, 1 - team]
    return dict(n_games=n_games, grid=grid, bh=bh, bt=bt, bk=bk, rank=rank, bg=bg, tier_g=tier_g,
                own_party=sz, opp_party=opp_party, s_self=s_self, r_self=r_self, opp_S=S[g, 1 - team])


def ban_flags(d, N, main):
    g, team = d["g"], d["team"].astype(int)
    n = len(g)
    rank = N["rank"]
    bh, bt, bk = N["bh"], N["bt"], N["bk"]
    f = {k: np.zeros(n, bool) for k in ("opp_ban", "opp_ban_late", "own_ban", "phase2", "first_ph",
                                          "any_opp", "own_pre", "opp_pre")}
    for k in range(6):
        hit = bh[g, k] == main
        is_opp = bt[g, k] == 1 - team
        is_own = bt[g, k] == team
        second = bk[g, k] >= 5
        before = ~second | (rank >= 5)
        valid = hit & is_opp & before
        f["opp_ban"] |= valid
        f["phase2"] |= valid & second
        f["opp_ban_late"] |= hit & is_opp & second & (rank < 5) & (rank >= 0)
        f["own_ban"] |= hit & is_own
        f["first_ph"] |= hit & is_opp & ~second
        f["any_opp"] |= hit & is_opp
        f["own_pre"] |= hit & is_own & before
        f["opp_pre"] |= valid
    gk = g.astype(np.int64) * HKEY + d["hero"]
    sgk = np.sort(gk)
    mk = g.astype(np.int64) * HKEY + main
    j2 = np.minimum(np.searchsorted(sgk, mk), len(sgk) - 1)
    f["taken"] = (sgk[j2] == mk) & (d["hero"] != main)
    return f


def ban_rate_tables(d, N):
    """Per-(build) and per-(build, tier) share of games in which hero h was
    banned by either team (per-team rate = half)."""
    bg, bh = N["bg"], N["bh"]
    tg = N["tier_g"]
    ti = np.where(tg < 0, 0, tg)  # 0 unknown, 1..6 Bronze..Master
    nb = int(bg.max()) + 1
    any_b = np.zeros((nb, NUM_HEROES))
    any_bt = np.zeros((nb, 7, NUM_HEROES))
    gc = np.bincount(bg, minlength=nb).astype(float)
    gct = np.zeros((nb, 7))
    np.add.at(gct, (bg, ti), 1.0)
    for k in range(6):
        m = bh[:, k] >= 0
        np.add.at(any_b, (bg[m], bh[m, k]), 1.0)
        np.add.at(any_bt, (bg[m], ti[m], bh[m, k]), 1.0)
    return any_b / np.maximum(gc[:, None], 1), any_bt / np.maximum(gct[:, :, None], 1), ti


def natural_experiment(d, N, Fz, Fl, boot, sel_players, out):
    n = len(d["pid"])
    g = d["g"]
    rank = N["rank"]
    main = Fz["main"]
    fl = ban_flags(d, N, main)
    share = Fz["top_cnt"] / np.maximum(Fz["tot"], 1)
    elig = (Fz["tot"] >= 30) & (d["day"] >= C.day_of(NAT_START)) & (rank >= 0) & sel_players
    treat = elig & fl["opp_ban"] & ~fl["own_ban"]
    ctrl = elig & ~fl["opp_ban"] & ~fl["own_ban"] & ~fl["opp_ban_late"] & ~fl["taken"]
    br, brt, ti = ban_rate_tables(d, N)
    br_main = br[N["bg"][g], main]
    post = ~d["in_sample"]
    _, fr = np.unique(g[post], return_index=True)
    med = float(np.median(d["day"][post][fr]))
    v2_label = f"V2 window ({date_of(np.ceil(med))} on)"
    out["windows"] = {"split_day": date_of(np.ceil(med)), "median_day_value": med,
                      "V1": f"{date_of(d['day'][post].min())} .. {date_of(np.ceil(med) - 1)}",
                      "V2": f"{date_of(np.ceil(med))} .. {date_of(d['day'][post].max())}",
                      "label_fix": f"p3_x_ban_nat.py:187 labels the V2 stratum '2026-04-01 on'; "
                                   f"the split day is {date_of(np.ceil(med))}"}
    log("windows", out["windows"])
    opp_dum = [(N["opp_party"] == k).astype(float) for k in (2, 3, 4, 5)]
    own_sz = N["own_party"].astype(float)
    base_ctrl = lambda m: [br_main[m], N["opp_S"][m]]
    prem_ctrl = lambda m: [br_main[m], N["opp_S"][m]] + [c[m] for c in opp_dum] + [own_sz[m]]
    r_self, y = N["r_self"], d["y"].astype(float)
    strata = {t: (share >= lo) & (share < hi) for t, (lo, hi) in TYPES.items()}
    strata["all"] = np.ones(n, bool)
    strata["one-trick, " + v2_label] = strata["one-trick"] & (d["day"] >= med)

    # (1) replication by type, with and without the premade control
    eff, draws_type = {}, {}
    for sn, sm in strata.items():
        tm, cm = treat & sm, ctrl & sm
        if tm.sum() < 50:
            continue
        sel = tm | cm
        res = {"treated": int(tm.sum()), "control": int(cm.sum()),
               "played_main_when_available": float((d["hero"][cm] == main[cm]).mean())}
        for on, ov in (("r_self", r_self), ("y", y)):
            b, dr = fe_fit(ov[sel], tm[sel], base_ctrl(sel), d["pid"][sel], boot)
            res[on] = pp(b, dr)
            if on == "r_self":
                draws_type[sn] = (b[0], dr[:, 0])
            b, dr = fe_fit(ov[sel], tm[sel], prem_ctrl(sel), d["pid"][sel], boot)
            res[on + ", + premade control"] = pp(b, dr)
        eff[sn] = res
        log("effect", sn, {k: (round(v["pp"], 2) if isinstance(v, dict) else v) for k, v in res.items()})
    out["effects_by_type"] = eff
    prem_share = {sn: float((N["opp_party"][treat & sm] >= 2).mean()) for sn, sm in strata.items()
                  if (treat & sm).sum() >= 50}
    prem_share_c = {sn: float((N["opp_party"][ctrl & sm] >= 2).mean()) for sn, sm in strata.items()
                    if (treat & sm).sum() >= 50}
    out["opponents_premade_share"] = {"treated": prem_share, "control": prem_share_c}

    # (4b) hot / mid / cold one-tricks in one regression; bound on targeting
    ot = strata["one-trick"]
    mw = Fz["mawp_main"]
    sel = (treat | ctrl) & ot
    tm = treat[sel]
    hot, cold = (mw >= 0.55)[sel], (mw < 0.50)[sel]
    mid = ~hot & ~cold
    Xk = np.column_stack([tm & hot, tm & mid, tm & cold]).astype(float)
    hc = {}
    hotf, midf = hot.astype(float), mid.astype(float)
    with_state = lambda ctl: (lambda m: ctl(m) + [hotf, midf])
    for spec, ctl in (("joint, base controls + MAWP state", with_state(base_ctrl)),
                      ("joint, + premade control + MAWP state", with_state(prem_ctrl))):
        bA, dA = fe_fit(r_self[sel], tm, ctl(sel), d["pid"][sel], boot)
        bH, dH = fe_fit(r_self[sel], Xk, ctl(sel), d["pid"][sel], boot)
        bound = bA[0] - bH[2]
        bdraw = dA[:, 0] - dH[:, 2]
        hc[spec] = {"one-trick, all": pp(bA, dA, 0, -1), "hot (main MAWP >= 0.55)": pp(bH, dH, 0, -1),
                    "mid": pp(bH, dH, 1, -1), "cold (main MAWP < 0.50)": pp(bH, dH, 2, -1),
                    "hot minus cold": {"pp": float(-100 * (bH[0] - bH[2])),
                                       "ci": sorted(np.percentile(-100 * (dH[:, 0] - dH[:, 2]), [2.5, 97.5]).tolist())},
                    "targeting bound: all minus cold": {
                        "pp": float(-100 * bound), "ci": sorted(np.percentile(-100 * bdraw, [2.5, 97.5]).tolist()),
                        "share_of_all": float(bound / bA[0]),
                        "share_ci": sorted(np.percentile(bdraw / dA[:, 0], [2.5, 97.5]).tolist())},
                    "treated": {"hot": int((tm & hot).sum()), "mid": int((tm & mid).sum()),
                                "cold": int((tm & cold).sum())}}
    # published form: separate within-player regressions per stratum (paired draws)
    sep = {}
    for lab, sm in (("all", np.ones(sel.sum(), bool)), ("hot", hot), ("cold", cold)):
        ss = np.flatnonzero(sel)[sm]
        sep[lab] = fe_fit(r_self[ss], treat[ss], base_ctrl(ss), d["pid"][ss], boot)
    bnd = sep["all"][0][0] - sep["cold"][0][0]
    bdr = sep["all"][1][:, 0] - sep["cold"][1][:, 0]
    hc["separate strata (published form)"] = {
        "one-trick, all": pp(*sep["all"], 0, -1), "hot": pp(*sep["hot"], 0, -1), "cold": pp(*sep["cold"], 0, -1),
        "targeting bound: all minus cold": {"pp": float(-100 * bnd),
                                            "ci": sorted(np.percentile(-100 * bdr, [2.5, 97.5]).tolist()),
                                            "share_of_all": float(bnd / sep["all"][0][0]),
                                            "share_ci": sorted(np.percentile(bdr / sep["all"][1][:, 0], [2.5, 97.5]).tolist())}}
    out["hot_cold_bound (drops, pp)"] = hc
    log("hot/cold", json.dumps(hc))

    # (4a) placebo with the premade control
    base_pl = elig & (rank < 5) & (d["hero"] != main) & ~fl["own_ban"] & ~fl["taken"] & ~fl["first_ph"]
    plc = {}
    for sn in ("all", "one-trick", "specialist"):
        sm = strata[sn] & base_pl
        tmk = sm & fl["opp_ban_late"]
        if tmk.sum() < 50:
            continue
        plc[sn] = {"placebo_treated": int(tmk.sum())}
        for on, ov in (("y", y), ("r_self", r_self)):
            b, dr = fe_fit(ov[sm], tmk[sm], base_ctrl(sm), d["pid"][sm], boot)
            plc[sn][on] = pp(b, dr)
            b, dr = fe_fit(ov[sm], tmk[sm], prem_ctrl(sm), d["pid"][sm], boot)
            plc[sn][on + ", + premade control"] = pp(b, dr)
    out["placebo_ban_after_player_picked"] = plc
    log("placebo", json.dumps(plc))

    # (4c) pre-ban information version: lag-1 main, ITT
    main_l = Fl["main"]
    fl2 = ban_flags(d, N, main_l)
    share_l = Fl["top_cnt"] / np.maximum(Fl["tot"], 1)
    elig_l = (Fl["tot"] >= 30) & (d["day"] >= C.day_of(NAT_START)) & (rank >= 0) & sel_players
    tr_l = elig_l & fl2["opp_pre"] & ~fl2["own_pre"]
    ct_l = elig_l & ~fl2["opp_pre"] & ~fl2["own_pre"]
    br_l = br[N["bg"][g], main_l]
    pre = {}
    for t, (lo, hi) in list(TYPES.items()) + [("all", (0.0, 1.01))]:
        sm = (share_l >= lo) & (share_l < hi)
        tm_, cm_ = tr_l & sm, ct_l & sm
        if tm_.sum() < 50:
            continue
        sel = tm_ | cm_
        res = {"treated": int(tm_.sum()), "control": int(cm_.sum()),
               "control: main picked by someone else later": float(fl2["taken"][cm_].mean()),
               "control: main banned after his pick": float(fl2["opp_ban_late"][cm_].mean())}
        c1 = lambda m: [br_l[m]] + [c_[m] for c_ in opp_dum] + [own_sz[m]]
        for on, ov in (("y (ITT)", y), ("r_self", r_self)):
            b, dr = fe_fit(ov[sel], tm_[sel], c1(sel), d["pid"][sel], boot)
            res[on] = pp(b, dr)
        pre[t] = res
        log("pre-ban ITT", t, {k: (round(v["pp"], 2) if isinstance(v, dict) else v) for k, v in res.items()})
    out["pre_ban_information_ITT"] = pre

    # (3) targeting with build-level and build x tier base rates
    tg = N["tier_g"][g]
    ti_s = ti[g]
    base_b = br[N["bg"][g], main] / 2.0
    base_t = brt[N["bg"][g], ti_s, main] / 2.0
    any_opp = fl["any_opp"]
    tgt = {}
    groups = [("all", np.ones(n, bool)), ("one-trick", strata["one-trick"]),
              ("specialist", strata["specialist"]), ("flexible", strata["flexible"])]
    groups += [(f"one-trick, {F.TIER_NAMES[t]}", strata["one-trick"] & (tg == t)) for t in range(1, 7)]
    groups += [(f"one-trick or specialist, {F.TIER_NAMES[t]}", (share >= 0.25) & (tg == t)) for t in range(1, 7)]
    elig_t = (Fz["tot"] >= 30) & (d["day"] >= C.day_of(NAT_START)) & (rank >= 0) & sel_players & ~fl["own_ban"]
    for gn, gm in groups:
        sm = gm & elig_t
        if sm.sum() < 200:
            continue
        u, inv = np.unique(d["pid"][sm], return_inverse=True)
        a = np.bincount(inv, weights=any_opp[sm].astype(float), minlength=len(u))
        eb = np.bincount(inv, weights=base_b[sm], minlength=len(u))
        et = np.bincount(inv, weights=base_t[sm], minlength=len(u))
        Wb = boot.W[:, u].astype(float)
        rb = (Wb @ a) / np.maximum(Wb @ eb, 1e-12)
        rt = (Wb @ a) / np.maximum(Wb @ et, 1e-12)
        tgt[gn] = {"slots": int(sm.sum()), "opponents_ban_main": float(a.sum() / sm.sum()),
                   "expected, build rate": float(eb.sum() / sm.sum()),
                   "expected, build x tier rate": float(et.sum() / sm.sum()),
                   "ratio, build rate": float(a.sum() / eb.sum()),
                   "ratio, build rate ci": np.percentile(rb, [2.5, 97.5]).tolist(),
                   "ratio, build x tier rate": float(a.sum() / et.sum()),
                   "ratio, build x tier rate ci": np.percentile(rt, [2.5, 97.5]).tolist()}
    out["targeting"] = tgt
    log("targeting", json.dumps({k: (round(v["ratio, build rate"], 2), round(v["ratio, build x tier rate"], 2))
                                 for k, v in tgt.items()}))
    flags_lag = {"unavailable": elig_l & (fl2["opp_ban"] | fl2["own_ban"] | fl2["taken"]) & (d["hero"] != main_l)}
    return med, draws_type, flags_lag


# ------------------------------------------------------------------ ban model

def fit_primary(d, Fl, unavailable, s, r_self, mask):
    Xs = BM.slot_design(s, Fl["mawp_own"], Fl["ewr_own"], Fl["n_own"], Fl["tot"], Fl["top_cnt"],
                        d["hero"] == Fl["main"], unavailable)
    A = np.column_stack([np.ones(mask.sum()), Xs[mask][:, PRIMARY_COLS]])
    return np.linalg.lstsq(A, r_self[mask], rcond=None)[0]


def ban_model(d, N, Fl, flags_lag, med, a, draws_type, out):
    import torch
    torch.set_num_threads(2)
    import p3_dr_core as D
    import p3_dr_imitation as I
    import p3_pgd_common as PG
    import p3_pgd_feats as PF
    import p3_pgd_model as PM
    from drift2026 import common
    rng = np.random.RandomState(0)
    meta = C.hero_meta(d["hero_names"])
    names = [str(h) for h in d["hero_names"]]
    g = d["g"]
    post = ~d["in_sample"]
    v1 = post & (d["day"] < med)
    # strength arm fits: full V1, and two halves of V1 players
    half_p = np.random.RandomState(5).rand(int(d["n_players"])) < 0.5
    hp = half_p[d["pid"]]
    s, r_self = N["s_self"], N["r_self"]
    un = flags_lag["unavailable"]
    coefs = {"full": fit_primary(d, Fl, un, s, r_self, v1),
             "A": fit_primary(d, Fl, un, s, r_self, v1 & hp),
             "B": fit_primary(d, Fl, un, s, r_self, v1 & ~hp)}
    out["strength_fit"] = {k: (100 * v).tolist() for k, v in coefs.items()}
    log("forced off-main coef (pp x main share):", {k: round(100 * v[-1], 2) for k, v in coefs.items()})

    L = D.load_lobbies()
    T = D.load_personal()
    picks = L["steps"][:, :, 3]
    prow = picks[picks >= 0].reshape(len(L["replay_id"]), 10)
    npos = np.vectorize(lambda r_: T["pos"][int(r_)])(prow)
    est = (T["n_p"][npos] >= 50).all(1)
    lob = np.flatnonzero((L["day"] >= med) & est)
    if a.sample < 1:
        lob = np.sort(rng.choice(lob, max(50, int(a.sample * len(lob))), replace=False))
    rows = np.unique(prow[lob].ravel())
    qpos = {int(r_): i for i, r_ in enumerate(rows)}
    Q = {"full": query_feats(d, rows)}
    hh = history_halves(d)
    Q["hA"] = query_feats(d, rows, hh)
    Q["hB"] = query_feats(d, rows, ~hh)
    rec = I.recency_features(d, rows)
    iw = np.load(I.OUT)["w"]
    gd = D.GDPolicy(d["hero_names"])
    log(f"lobbies {len(lob):,}, slots {len(rows):,}")
    # personal GD (pick head) at the first ban state
    ck = torch.load(PM.OUT, weights_only=False)
    assert list(ck["pick_feats"]) == list(PF.FEATS)
    head = PM.Head(len(PF.FEATS))
    head.load_state_dict(ck["pick"])
    head.eval()
    hist = pgd_walk(d, rows)
    Fpg, _ = PF.context(hist, meta["fine"], use_cluster=bool(ck.get("use_cluster", True)))
    mg = PG.load_metagd()
    G = PG.load_games()
    gpos = {int(r_): i for i, r_ in enumerate(G["rid"])}
    out["personal_gd"] = {"use_cluster": bool(ck.get("use_cluster", True)), "alpha_pick": float(head.alpha)}

    w = np.load(C.WP)
    o = np.argsort(w["replay_ids"])
    bidx_all = w["build_idx"][o]
    present = np.unique(w["build_idx"])
    builds = common.load_patch_index()["builds"]
    stats_cache = {}

    def pop_strength(gi):
        bi = int(bidx_all[L["g"][gi]])
        key = builds[present[int(np.searchsorted(present, bi)) - 1]]
        if key not in stats_cache:
            stats_cache[key] = common.load_patch_stats("cumulative", key)
        hw = stats_cache[key].hero_wr.get(str(L["tier"][gi]), {})
        return np.array([(hw.get(h, 50.0) - 50.0) / 100.0 for h in names])

    VARS = {"full": ("full", "full"), "A": ("A", "full"), "B": ("B", "full"),
            "hA": ("full", "hA"), "hB": ("full", "hB")}

    def strength(row, cname, qname):
        q = qpos[int(row)]
        p = T["pos"][int(row)]
        n_h = Q[qname]["q_cnt"][q].astype(float)
        tot, top = n_h.sum(), n_h.max()
        ism = np.arange(NUM_HEROES) == np.argmax(n_h)
        Xh = BM.slot_design(T["s"][p].astype(float), Q[qname]["q_mawp"][q].astype(float),
                            Q[qname]["q_ewr10"][q].astype(float), n_h, np.full(NUM_HEROES, tot),
                            np.full(NUM_HEROES, top), ism)
        b = coefs[cname]
        return b[0] + Xh[:, PRIMARY_COLS[:9]] @ b[1:10]

    recs, calib = [], []
    pgd_miss = 0
    for li, gi in enumerate(lob):
        st = L["steps"][gi]
        mo, to = D.one_hots(str(L["map"][gi]), str(L["tier"][gi]))
        sp = pop_strength(gi)
        pick_step = {int(row): k for k, (h, ty, tm, row) in enumerate(st) if ty == 1}
        team_of = {int(row): tm for (h, ty, tm, row) in st if ty == 1}
        hero_of = {int(row): h for (h, ty, tm, row) in st if ty == 1}
        sig = {v: {r_: strength(r_, *VARS[v]) for r_ in pick_step} for v in VARS}
        t0_ = np.zeros(NUM_HEROES, np.float32)
        t1_ = np.zeros(NUM_HEROES, np.float32)
        bans = np.zeros(NUM_HEROES, np.float32)
        for k, (h, ty, tm, row) in enumerate(st):
            if ty == 1:
                (t0_ if tm == 0 else t1_)[h] = 1
                continue
            avail = (t0_ + t1_ + bans) == 0
            todo = [r_ for r_, kk in pick_step.items() if kk > k]
            Bn = len(todo)
            lp = gd.logprobs(np.repeat(t0_[None], Bn, 0), np.repeat(t1_[None], Bn, 0), np.repeat(bans[None], Bn, 0),
                             np.repeat(mo[None], Bn, 0), np.repeat(to[None], Bn, 0),
                             np.array([pick_step[r_] for r_ in todo], np.float32), np.ones(Bn, np.float32),
                             np.repeat(avail[None], Bn, 0))
            fake = {"row": np.array(todo), "recpos": np.array([qpos[r_] for r_ in todo]), "lp": lp.astype(np.float32)}
            Xf = I.feature_tensor(fake, T, rec, meta)
            u = np.where(avail[None], (Xf * iw).sum(-1), -1e9)
            Pi = np.exp(u - u.max(1, keepdims=True))
            Pi /= Pi.sum(1, keepdims=True)
            Pg = np.where(avail[None], np.exp(lp), 0)
            Pg /= Pg.sum(1, keepdims=True)
            sign = np.array([1.0 if team_of[r_] != tm else -1.0 for r_ in todo])

            def value(Pm, sr, qn, fc):
                cnts = np.stack([Q[qn]["q_cnt"][qpos[r_]] for r_ in todo])
                mains = cnts.argmax(1)
                msh = cnts.max(1) / np.maximum(cnts.sum(1), 1)
                E = (Pm * sr).sum(1)
                Ec = (E[:, None] - Pm * sr) / np.maximum(1 - Pm, 1e-9)
                Ec[np.arange(Bn), mains] += fc * msh
                return (sign[:, None] * (E[:, None] - Ec)).sum(0)
            vals, pers = {}, {}
            for v, (cn, qn) in VARS.items():
                fc = float(coefs[cn][-1])
                sr = np.stack([sp + sig[v][r_] for r_ in todo])
                vals[v] = np.where(avail, value(Pi, sr, qn, fc), -1e9)
                if v == "full":
                    pers[v] = np.where(avail, value(Pi, np.stack([sig[v][r_] for r_ in todo]), qn, fc), -1e9)
            vpop = np.where(avail, value(Pg, np.repeat(sp[None], Bn, 0), "full", 0.0), -1e9)
            c = {v: int(np.argmax(vals[v])) for v in VARS}
            cp = int(np.argmax(vpop))
            gain = lambda chooser, scorer: float(vals[scorer][c[chooser]] - vals[scorer][cp])
            recs.append({"gi": int(gi), "team": int(tm), "phase": 1 if k < 4 else 2,
                         "own_full": gain("full", "full"),
                         "own_A": gain("A", "A"), "own_B": gain("B", "B"),
                         "cf_AB": gain("A", "B"), "cf_BA": gain("B", "A"),
                         "own_hA": gain("hA", "hA"), "own_hB": gain("hB", "hB"),
                         "cf_hAB": gain("hA", "hB"), "cf_hBA": gain("hB", "hA"),
                         "hA_scored_full": gain("hA", "full"), "hB_scored_full": gain("hB", "full"),
                         "real_personal": float(pers["full"][h]) if avail[h] else 0.0,
                         "matched": bool(c["full"] == h)})
            # calibration at the lobby's first ban decision
            if k == 0:
                gp = gpos.get(int(L["replay_id"][gi]))
                if gp is None:
                    pgd_miss += 1
                Ppg = None
                if gp is not None:
                    Xg = np.zeros((Bn, PG.DIM), np.float32)
                    Xg[:, 3 * NUM_HEROES + G["map"][gp]] = 1
                    Xg[:, 284 + G["tier"][gp]] = 1
                    Xg[:, 287] = np.array([pick_step[r_] for r_ in todo]) / 15.0
                    Xg[:, 288] = 1
                    di = G["day"][gp] - int(G["day0"])
                    Xg[:, 289:379] = G["meta_pick"][di, G["tier"][gp]]
                    Xg[:, 379:469] = G["meta_ban"][di, G["tier"][gp]]
                    M = np.repeat(avail[None], Bn, 0)
                    lpm = PG.logp_metagd(mg, Xg, M)
                    with torch.no_grad():
                        Fq = torch.from_numpy(Fpg[[qpos[r_] for r_ in todo]])
                        ub = head(torch.from_numpy(lpm), Fq, torch.from_numpy(M))
                        Ppg = torch.softmax(ub, -1).numpy().astype(np.float64)
                fc = float(coefs["full"][-1])
                for jj, r_ in enumerate(todo):
                    if sign[jj] < 0:
                        continue
                    n_h = Q["full"]["q_cnt"][qpos[r_]]
                    m_ = int(np.argmax(n_h))
                    if n_h.sum() < 30 or not avail[m_]:
                        continue
                    ms = float(n_h[m_] / n_h.sum())
                    sj = sig["full"][r_]

                    def drop(pv):
                        Ej = (pv * sj).sum()
                        Ec = (Ej - pv[m_] * sj[m_]) / max(1 - pv[m_], 1e-9) + fc * ms
                        return float(Ej - Ec)
                    # realized: main still available at his pick, did he pick it
                    ps = pick_step[r_]
                    gone = any(st[kk][0] == m_ for kk in range(ps))
                    calib.append({"share": ms, "p_imit": float(Pi[jj, m_]), "drop_imit": drop(Pi[jj]),
                                  "p_pgd": float(Ppg[jj, m_]) if Ppg is not None else np.nan,
                                  "drop_pgd": drop(Ppg[jj]) if Ppg is not None else np.nan,
                                  "avail_at_pick": not gone, "picked_main": int(hero_of[r_] == m_)})
            bans[h] = 1
        if (li + 1) % 1000 == 0:
            log(f"lobbies {li + 1}/{len(lob)}")
    out["lobbies"] = int(len(lob))
    out["ban_decisions"] = len(recs)
    out["pgd_lobbies_unmatched"] = pgd_miss

    # ---- item 2: winner's curse
    gis = np.array([r_["gi"] for r_ in recs])
    ug, ginv = np.unique(gis, return_inverse=True)
    nb = a.nboot

    def col(k):
        return np.array([r_[k] for r_ in recs], float)
    cols = {k: col(k) for k in ("own_full", "own_A", "own_B", "cf_AB", "cf_BA", "own_hA", "own_hB",
                                "cf_hAB", "cf_hBA", "hA_scored_full", "hB_scored_full")}
    sums = {k: np.bincount(ginv, weights=v, minlength=len(ug)) for k, v in cols.items()}
    cnt = np.bincount(ginv, minlength=len(ug)).astype(float)
    brng = np.random.RandomState(3)
    Wl = brng.poisson(1.0, (nb, len(ug))).astype(float)

    def stat(fn, scale=100.0, key="pp"):
        m = {k: v.sum() / cnt.sum() for k, v in sums.items()}
        val = fn(m)
        bs = []
        for b in range(nb):
            den = Wl[b] @ cnt
            mb = {k: (Wl[b] @ v) / den for k, v in sums.items()}
            bs.append(fn(mb))
        return {key: scale * float(val), "ci": (scale * np.percentile(bs, [2.5, 97.5])).tolist()}
    own_h = lambda m: 0.5 * (m["own_hA"] + m["own_hB"])
    cf_h = lambda m: 0.5 * (m["cf_hAB"] + m["cf_hBA"])
    wc = {"in-sample gain (own argmax, full model, lag-1 features)": stat(lambda m: m["own_full"]),
          "coefficient halves: own argmax, same half": stat(lambda m: 0.5 * (m["own_A"] + m["own_B"])),
          "coefficient halves: cross-fit (choose A, value B and reverse)": stat(lambda m: 0.5 * (m["cf_AB"] + m["cf_BA"])),
          "history halves: own argmax, same half": stat(own_h),
          "history halves: cross-fit": stat(cf_h),
          "history halves: half-sample selection bias": stat(lambda m: own_h(m) - cf_h(m)),
          "history halves: shrink ratio (cross / own)": stat(lambda m: cf_h(m) / own_h(m), 1.0, "ratio"),
          "full gain x shrink ratio (conservative)": stat(lambda m: m["own_full"] * cf_h(m) / own_h(m)),
          "full gain - half-sample bias / sqrt 2 (point correction)": stat(
              lambda m: m["own_full"] - (own_h(m) - cf_h(m)) / np.sqrt(2)),
          "half-history choice scored by the full model": stat(
              lambda m: 0.5 * (m["hA_scored_full"] + m["hB_scored_full"]))}
    out["winners_curse"] = wc
    log("winner's curse", json.dumps(wc, indent=1))
    # realized check on real bans, split by whether the real ban was the argmax
    n_games, yg, wpg, _, _ = X.game_arrays(d)
    acc = {}
    for r_ in recs:
        a_ = acc.setdefault(r_["gi"], np.zeros((2, 2)))
        a_[int(r_["matched"]), r_["team"]] += r_["real_personal"]
    gl = np.array(list(acc))
    gg = L["g"][gl]
    res0 = yg[gg] - wpg[gg]
    xm = np.array([acc[x][1, 0] - acc[x][1, 1] for x in gl])
    xn = np.array([acc[x][0, 0] - acc[x][0, 1] for x in gl])
    A = np.column_stack([np.ones(len(gl)), xm + xn])
    A2 = np.column_stack([np.ones(len(gl)), xm, xn])
    b1 = np.linalg.lstsq(A, res0, rcond=None)[0]
    b2 = np.linalg.lstsq(A2, res0, rcond=None)[0]
    bs1, bs2 = [], []
    for _ in range(nb):
        ii = brng.randint(0, len(gl), len(gl))
        bs1.append(np.linalg.lstsq(A[ii], res0[ii], rcond=None)[0][1])
        bs2.append(np.linalg.lstsq(A2[ii], res0[ii], rcond=None)[0][1:])
    bs2 = np.array(bs2)
    out["realized_check_real_bans"] = {
        "share of real bans that were the model's argmax": float(np.mean([r_["matched"] for r_ in recs])),
        "slope, all real bans": [float(b1[1])] + np.percentile(bs1, [2.5, 97.5]).tolist(),
        "slope, real bans that were the argmax": [float(b2[1])] + np.percentile(bs2[:, 0], [2.5, 97.5]).tolist(),
        "slope, other real bans": [float(b2[2])] + np.percentile(bs2[:, 1], [2.5, 97.5]).tolist(),
        "sd of the matched-ban difference (pp)": float(100 * xm.std()), "lobbies": int(len(gl))}
    log("realized", json.dumps(out["realized_check_real_bans"]))

    # ---- item 1: calibration by type
    cal = {}
    sh = np.array([x["share"] for x in calib])
    for t, (lo, hi) in TYPES.items():
        m = (sh >= lo) & (sh < hi)
        xs = [calib[i] for i in np.flatnonzero(m)]
        if not xs or t not in draws_type:
            continue
        di = np.array([x["drop_imit"] for x in xs])
        dp = np.array([x["drop_pgd"] for x in xs])
        okp = ~np.isnan(dp)
        av = np.array([x["avail_at_pick"] for x in xs])
        pk = np.array([x["picked_main"] for x in xs])
        real_b, real_d = draws_type[t]
        realized = -100 * real_b
        rd = -100 * real_d
        ib = brng.randint(0, len(xs), (nb, len(xs)))
        pi_b = 100 * di[ib].mean(1)
        pg_b = 100 * np.nanmean(np.where(okp, dp, np.nan)[ib], 1)
        cal[t] = {"opponents": len(xs), "with personal GD": int(okp.sum()),
                  "p(main), imitation": float(np.mean([x["p_imit"] for x in xs])),
                  "p(main), personal GD": float(np.nanmean([x["p_pgd"] for x in xs])),
                  "p(main), real (main still free at his pick)": float(pk[av].mean()) if av.any() else None,
                  "predicted drop, imitation (pp)": float(100 * di.mean()),
                  "predicted drop, personal GD (pp)": float(100 * dp[okp].mean()),
                  "realized drop (pp)": {"pp": realized, "ci": np.percentile(rd, [2.5, 97.5]).tolist()},
                  "recovered share, imitation": {"share": float(100 * di.mean() / realized),
                                                 "ci": np.percentile(pi_b / rd, [2.5, 97.5]).tolist()},
                  "recovered share, personal GD": {"share": float(100 * dp[okp].mean() / realized),
                                                   "ci": np.percentile(pg_b / rd, [2.5, 97.5]).tolist()}}
    out["calibration_main_ban"] = cal
    log("calibration", json.dumps(cal, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of players (natural experiment) and lobbies")
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--final", action="store_true",
                    help="output name only: this script reads snapshot games, never the sealed build")
    ap.add_argument("--skip-model", action="store_true")
    a = ap.parse_args()
    tag = ("" if a.sample >= 1 else f"_s{a.sample:g}") + ("_final" if a.final else "")
    d = C.load_slots()
    out = {"sample": a.sample, "nboot": a.nboot,
           "scope": "snapshot games only (ban table ends at the snapshot); sealed build 2.55.17.98025 never read"}
    Fz = slot_feats(d, "time")
    Fl = slot_feats(d, "lag1")
    log("slot features")
    N = nat_setup(d)
    sel_players = np.ones(len(d["pid"]), bool)
    if a.sample < 1:
        sel_players = (np.random.RandomState(9).rand(int(d["n_players"])) < a.sample)[d["pid"]]
    boot = Boot(d["n_players"], a.nboot)
    med, draws_type, flags_lag = natural_experiment(d, N, Fz, Fl, boot, sel_players, out)
    del boot
    if not a.skip_model:
        ban_model(d, N, Fl, flags_lag, med, a, draws_type, out)
    os.makedirs(C.RESULTS, exist_ok=True)
    path = os.path.join(C.RESULTS, f"p3_b_bans{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    log("wrote", path)


if __name__ == "__main__":
    main()
