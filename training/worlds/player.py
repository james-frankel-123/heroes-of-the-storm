"""
Family f: player-hero skill. Outcomes depend on who plays each hero, fit JOINTLY with the
hero-generic population model so that, averaged over the assignments humans actually chose, the
player term is zero for every hero and the hero-generic fit is reproduced.

Data: paper 2's slot table (Storm League games with the ten players; sealed build and sealed ids
excluded exactly as p3_team_support.residual). Fit window: 2026-01-01..06-30 (ends at the drafter
state's as-of date; skill is static within it, so it is kept recent: a first fit on every game since
2021 gave player terms nearly uncorrelated with the drafter's, slope ~0.28 against the real ~0.8);
familiarity counts use the full history. Check window: the judge window 2026-07-01..09-11.

World structure (deliberately NOT the drafter's: no hero similarity, no record pooling):
  logit P(team 0) = logit WP_pop + sum_{team 0 slots} u'(p, h) - sum_{team 1 slots} u'(p, h)
  u(p, h)  = gamma[bin(n_ph)] + alpha_p + eps_ph
  u'(p, h) = u(p, h) - ubar_h,   ubar_h = mean of u over the fit window's observed (p, h) slots
n_ph = the player's earlier games on the hero (familiarity, 8 bins on 0, 1-2, 3-5, 6-10, 11-25,
26-50, 51-100, 101+); alpha_p ~ N(0, s_a^2) general skill beyond MMR matchmaking; eps_ph ~ N(0, s_e^2)
player-hero cell; gamma ~ N(0, 1). WP_pop = the slot table's population WP (hero-generic). MAP by
LBFGS, (s_a, s_e) by diag-Laplace evidence, posterior alpha, eps ~ N(mu, 1 / (h + 1/s^2)); unseen
players and cells from the prior. Dials: k scales the eps SD of thin cells (n < FLOOR_N), delta is
an extra shift (points) on never-played cells (n = 0), the forced-unfamiliar case.

Stages:
  extract  (main box, light) slot-table arrays -> site/cache/player_src.npz (integer ids only)
  fit      (3090) -> site/cache/player_fit.npz, site/results/player_fit.json
  slope    (3090) the paper-2 check: world outcomes on the judge-window games with their actual
           assignments; the slope of r = y - WP_pop on the drafter's personal term P per min3 bin,
           against the real 0.74-0.84 (P3_TEAM_SUPPORT.md section 2) -> check_player_slope.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, TRAINING_DIR)
import data as WD  # noqa: E402
import model as M  # noqa: E402

H = M.H
SRC = os.path.join(WD.CACHE, "player_src.npz")
FIT = os.path.join(WD.CACHE, "player_fit.npz")
NBINS = [0, 1, 3, 6, 11, 26, 51, 101]
FLOOR_N = 5
SALT_A, SALT_E = 31, 37
P2 = os.path.join(TRAINING_DIR, "personalization")


def nbin(n):
    return np.searchsorted(np.asarray(NBINS), np.asarray(n), side="right") - 1


# ── extract (main box) ───────────────────────────────────────────────────
def extract(work):
    sys.path.insert(0, P2)
    import p3_hs_core as HC
    import p3_sealed
    import p3_ood_audit as OA
    import p3t_support as PS
    from shared import HEROES
    t0 = time.time()
    z = np.load(os.path.join(P2, "cache_def_def1027", "x_slots_ext.npz"), mmap_mode="r")
    d = {k: np.asarray(z[k]) for k in ("replay_id", "pid", "hero", "team", "y", "wp", "day", "version", "versions",
                                       "hero_names", "player_keys")}
    vs = [str(x) for x in d["versions"]]
    sb = vs.index("2.55.17.98025") if "2.55.17.98025" in vs else -999
    J0, J1 = HC.day_of("2026-07-01"), HC.day_of("2026-09-11")
    m = (d["version"] != sb) & (d["day"] <= J1)
    w = np.flatnonzero(m)
    w = w[~np.isin(d["replay_id"][w].astype(np.int64), p3_sealed.sealed_ids())]
    rid = d["replay_id"][w].astype(np.int64)
    o = np.lexsort((d["team"][w], rid))
    w, rid = w[o], rid[o]
    _, st, cnt = np.unique(rid, return_index=True, return_counts=True)
    ok = cnt == 10
    rows = w[st[ok][:, None] + np.arange(10)[None]]
    team = d["team"][rows]
    good = (team[:, :5] == team[:, :1]).all(1) & (team[:, 5:] == team[:, 5:6]).all(1) & (team[:, 0] != team[:, 5])
    rows = rows[good]
    sh = np.array([HEROES.index(str(x)) for x in d["hero_names"]], np.int64)
    hero = sh[d["hero"][rows]]
    pid = d["pid"][rows].astype(np.int64)
    day = d["day"][rows[:, 0]].astype(np.int64)
    y0 = d["y"][rows[:, 0]].astype(float)
    wp0 = d["wp"][rows[:, 0]].astype(float)
    grid = d["replay_id"][rows[:, 0]].astype(np.int64)
    # familiarity: earlier games of this player on this hero (chronological; ties by replay id)
    G = len(rows)
    fp, fh = pid.ravel(), hero.ravel()
    fd = np.repeat(day, 10)
    fr = np.repeat(grid, 10)
    oo = np.lexsort((fr, fd, fh, fp))
    key = fp[oo] * H + fh[oo]
    first = np.r_[True, key[1:] != key[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(key)), 0))
    nprior = np.empty(len(key), np.int64)
    nprior[oo] = np.arange(len(key)) - start
    nprior = nprior.reshape(G, 10)
    fit = day < J0
    jw = (day >= J0) & (day <= J1)
    # the drafter's personal term on the judge-window games (as p3_team_support.residual)
    c = np.load(os.path.join(work, "cells.npz"))
    jq = np.load(os.path.join(P2, "cache_def_def1027", "judge_q", "p3t_data_states__3.npz"))
    bkeys = jq["player_key"].astype(np.int64)
    keys = np.asarray(d["player_keys"]).astype(np.int64)[pid[jw]]
    ob = np.argsort(bkeys)
    j = np.minimum(np.searchsorted(bkeys[ob], keys), len(bkeys) - 1)
    known = bkeys[ob][j] == keys
    brow = ob[j]
    hj = hero[jw]
    s = np.where(known, c["b_s"][brow, hj], 0.0)
    off = np.where(known, OA.off_role_shared(c["b_n"])[brow, hj], False).astype(float)
    b0, b1, b2, b3 = PS.HEADLINE_COMBINER
    sg = lambda x: 1 / (1 + np.exp(-x))
    lw = np.log(wp0[jw] / (1 - wp0[jw]))
    dS = s[:, :5].sum(1) - s[:, 5:].sum(1)
    dO = off[:, :5].sum(1) - off[:, 5:].sum(1)
    P0 = sg(b0 + b1 * lw + b2 * dS + b3 * dO) - sg(b0 + b1 * lw)              # team 0's personal term
    P1 = sg(b0 - b1 * lw - b2 * dS - b3 * dO) - sg(b0 - b1 * lw)              # team 1's (its own WP = 1 - wp0)
    c3 = np.load(os.path.join(TRAINING_DIR, "paper1_revision", "site", "results", "support_mitigation", "triple_counts.npy"))
    T = np.sort(hero[jw].reshape(-1, 2, 5), 2)
    def min3(A):
        return np.stack([c3[(A[:, a] * H + A[:, b]) * H + A[:, cc]] for a, b, cc in M.C3], 1).min(1)
    np.savez(SRC, pid=pid, hero=hero, y0=y0, wp0=wp0, day=day, nprior=nprior, fit=fit, jw=jw,
             j_P=np.stack([P0, P1], 1), j_known=known.all(1), j_min3=np.stack([min3(T[:, 0]), min3(T[:, 1])], 1),
             n_players=len(d["player_keys"]), J0=J0, J1=J1)
    print(f"player_src: {G:,} games ({fit.sum():,} fit, {jw.sum():,} judge window), {time.time() - t0:.0f}s", flush=True)


P2_LOBBIES = os.path.join(WD.CACHE, "p2_lobbies.npz")


def extract_p2():
    """Paper 2's headline and tune lobbies: the five players of each draft team (draft team t = real team
    t ^ first_team), their earlier games per hero as of the lobby day, map, tier. -> p2_lobbies.npz"""
    hl = os.path.join(P2, "cache_def_def1027_hlsup", "p3t")
    L = np.load(os.path.join(hl, "lobbies.npz"))
    pidtab = np.asarray(np.load(os.path.join(P2, "cache_def_def1027", "x_slots_ext.npz"), mmap_mode="r")["pid"])
    src = np.load(SRC)
    sp, sh, sd = src["pid"].ravel(), src["hero"].ravel(), np.repeat(src["day"], 10)
    out = {}
    for name in ("tourney", "tourney_tune"):
        gi = np.load(os.path.join(hl, name, "lobbies.npz"))["gi"]
        n = len(gi)
        pid = np.zeros((n, 2, 5), np.int64)
        for i, g in enumerate(gi):
            first = int(L["first_team"][g])
            c = [0, 0]
            for h, ty, tm, r in L["steps"][g]:
                if ty == 1:
                    kt = int(tm) ^ first
                    pid[i, kt, c[kt]] = pidtab[int(r)]
                    c[kt] += 1
        day = L["day"][gi].astype(np.int64)
        keep = np.isin(sp, pid.ravel())
        kp, kh, kd = sp[keep], sh[keep], sd[keep]
        cnt = np.zeros((n, 2, 5, H), np.int32)
        for i in range(n):
            for t in range(2):
                for j in range(5):
                    m = (kp == pid[i, t, j]) & (kd < day[i])
                    cnt[i, t, j] = np.bincount(kh[m], minlength=H)
        tag = name.replace("tourney", "headline") if name == "tourney" else "tune"
        out.update({f"{tag}_gi": gi, f"{tag}_pid": pid, f"{tag}_cnt": cnt, f"{tag}_day": day,
                    f"{tag}_map": L["map"][gi], f"{tag}_tier": L["tier"][gi]})
        print(name, n, "lobbies", flush=True)
    np.savez(P2_LOBBIES, **out)


COV = os.path.join(WD.CACHE, "player_cov.npz")


def extract_cov(work, n_pool=3000):
    """f-cov inputs (main box): the drafter state's skill s(p, h) and off-role flag (as of 2026-06-30,
    cells.npz b rows; unknown players 0) for (1) the judge-window slots, (2) the paper-2 lobby players,
    (3) a pool of real judge-window lobbies attached to paper-1 drafts (players, earlier games per hero
    as of the game day, state). Integer ids only. -> player_cov.npz"""
    sys.path.insert(0, P2)
    import p3_ood_audit as OA
    src = dict(np.load(SRC))
    c = np.load(os.path.join(work, "cells.npz"))
    jq = np.load(os.path.join(P2, "cache_def_def1027", "judge_q", "p3t_data_states__3.npz"))
    bkeys = jq["player_key"].astype(np.int64)
    pkeys = np.asarray(np.load(os.path.join(P2, "cache_def_def1027", "x_slots_ext.npz"), mmap_mode="r")["player_keys"]).astype(np.int64)
    ob = np.argsort(bkeys)
    OFF = OA.off_role_shared(c["b_n"])
    bs = c["b_s"]

    def brow(pid):
        k = pkeys[pid]
        j = np.minimum(np.searchsorted(bkeys[ob], k), len(bkeys) - 1)
        return ob[j], bkeys[ob][j] == k

    out = {}
    jw = src["jw"]
    pid, hero = src["pid"][jw], src["hero"][jw]
    r, kn = brow(pid)
    out["j_s"] = np.where(kn, bs[r, hero], 0.0).astype(np.float32)
    out["j_off"] = np.where(kn, OFF[r, hero], False)
    out["j_known"] = kn
    L = np.load(P2_LOBBIES)
    for tag in ("headline", "tune"):
        r, kn = brow(L[f"{tag}_pid"])
        out[f"{tag}_s"] = np.where(kn[..., None], bs[r], 0.0).astype(np.float32)          # (n, 2, 5, 90)
        out[f"{tag}_off"] = np.where(kn[..., None], OFF[r], False)
    # pool of judge-window lobbies for paper-1 drafts
    rng = np.random.default_rng(17)
    gi = rng.choice(np.flatnonzero(jw), n_pool, replace=False)
    ppid = src["pid"][gi].reshape(n_pool, 2, 5)
    pday = src["day"][gi]
    sp, sh, sd = src["pid"].ravel(), src["hero"].ravel(), np.repeat(src["day"], 10)
    keep = np.isin(sp, ppid.ravel())
    kp, kh, kd = sp[keep], sh[keep], sd[keep]
    o = np.lexsort((kd, kp))
    kp, kh, kd = kp[o], kh[o], kd[o]
    cnt = np.zeros((n_pool, 2, 5, H), np.int32)
    for i in range(n_pool):
        for t in range(2):
            for j in range(5):
                a, b = np.searchsorted(kp, ppid[i, t, j]), np.searchsorted(kp, ppid[i, t, j], side="right")
                m = kd[a:b] < pday[i]
                cnt[i, t, j] = np.bincount(kh[a:b][m], minlength=H)
    r, kn = brow(ppid)
    out.update(pool_pid=ppid, pool_cnt=cnt, pool_s=np.where(kn[..., None], bs[r], 0.0).astype(np.float32),
               pool_off=np.where(kn[..., None], OFF[r], False), pool_known=kn)
    np.savez(COV, **out)
    print("player_cov:", {k: v.shape for k, v in out.items()}, flush=True)


# ── fit (3090) ───────────────────────────────────────────────────────────
def _map(S, sa, se, init=None, iters=300):
    import torch
    torch.set_num_threads(M.THREADS)
    ip, ib, ic = S["ip"], S["ib"], S["ic"]
    nP, nC = S["nP"], S["nC"]
    y = torch.tensor(S["y"])
    off = torch.tensor(S["off"])
    sgn = torch.tensor(np.r_[np.ones(5), -np.ones(5)])
    tp, tb, tc = torch.tensor(ip), torch.tensor(ib), torch.tensor(ic)
    th = torch.zeros(len(NBINS) + nP + nC, dtype=torch.float64)
    if init is not None:
        th[:] = torch.tensor(init)
    th.requires_grad_(True)
    nb = len(NBINS)
    prec = torch.cat([torch.ones(nb), torch.full((nP,), 1 / sa ** 2), torch.full((nC,), 1 / se ** 2)]).double()

    def logit():
        g, a, e = th[:nb], th[nb:nb + nP], th[nb + nP:]
        return off + ((g[tb] + a[tp] + e[tc]) * sgn).sum(1)
    opt = torch.optim.LBFGS([th], lr=1, max_iter=iters, tolerance_grad=1e-7, tolerance_change=1e-12, history_size=30,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit(), y, reduction="sum") + 0.5 * (prec * th ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        p = torch.sigmoid(logit()).numpy()
    t = th.detach().numpy().copy()
    w = np.repeat(p * (1 - p), 10)
    hP = np.bincount(ip.ravel(), weights=w, minlength=nP)
    hC = np.bincount(ic.ravel(), weights=w, minlength=nC)
    yy = S["y"]
    ll = float(np.sum(yy * np.log(np.clip(p, 1e-12, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-12, 1))))
    a, e = t[nb:nb + nP], t[nb + nP:]
    ev = ll - 0.5 * (a ** 2).sum() / sa ** 2 - 0.5 * (e ** 2).sum() / se ** 2 - 0.5 * np.log1p(hP * sa ** 2).sum() \
        - 0.5 * np.log1p(hC * se ** 2).sum()
    return t, dict(sa=sa, se=se, loglik=ll, evidence=float(ev)), hP, hC


def setup(src, sel):
    pid, hero = src["pid"][sel], src["hero"][sel]
    up, ip = np.unique(pid, return_inverse=True)
    ck = pid * H + hero
    uc, ic = np.unique(ck, return_inverse=True)
    wp = np.clip(src["wp0"][sel], 1e-6, 1 - 1e-6)
    return dict(ip=ip.reshape(pid.shape), ic=ic.reshape(pid.shape), ib=nbin(src["nprior"][sel]), nP=len(up), nC=len(uc),
                up=up, uc=uc, y=src["y0"][sel], off=np.log(wp / (1 - wp)), hero=hero)


FIT_FROM = os.environ.get("WORLDS_PLAYER_FROM", "2026-01-01")


def fit():
    import datetime
    src = dict(np.load(SRC))
    d0 = (datetime.date.fromisoformat(FIT_FROM) - datetime.date(1970, 1, 1)).days
    S = setup(src, src["fit"] & (src["day"] >= d0))
    print(f"fit window: {len(S['y']):,} games, {S['nP']:,} players, {S['nC']:,} cells", flush=True)
    grid = []
    t, best = None, None
    for sa in (0.02, 0.05, 0.1, 0.2):          # s_a at s_e = 0.05, then s_e at the best s_a
        t, info, _, _ = _map(S, sa, 0.05, init=t)
        grid.append(info)
        print(json.dumps(info), flush=True)
    sa = max(grid, key=lambda d: d["evidence"])["sa"]
    for se in (0.02, 0.1, 0.2, 0.3):
        t, info, _, _ = _map(S, sa, se, init=t)
        grid.append(info)
        print(json.dumps(info), flush=True)
    cand = [g for g in grid if g["sa"] == sa]
    se = max(cand, key=lambda d: d["evidence"])["se"]
    t, info, hP, hC = _map(S, sa, se, init=t)
    nb = len(NBINS)
    g, a, e = t[:nb], t[nb:nb + S["nP"]], t[nb + S["nP"]:]
    u = g[S["ib"]] + a[S["ip"]] + e[S["ic"]]
    ubar = np.bincount(S["hero"].ravel(), weights=u.ravel(), minlength=H) / np.maximum(np.bincount(S["hero"].ravel(), minlength=H), 1)
    ncell = np.bincount(S["ic"].ravel(), minlength=S["nC"])
    np.savez(FIT, gamma=g, players=S["up"], a_mu=a, a_sd=1 / np.sqrt(hP + 1 / sa ** 2), cells=S["uc"], e_mu=e,
             e_sd=1 / np.sqrt(hC + 1 / se ** 2), e_n=ncell, ubar=ubar, sa=sa, se=se)
    # marginal check: the centred term averages to zero per game over the observed assignments
    uc_ = u - ubar[S["hero"]]
    net = (uc_[:, :5].sum(1) - uc_[:, 5:].sum(1))
    out = {"fit_from": FIT_FROM, "games": int(len(S["y"])), "players": int(S["nP"]), "cells": int(S["nC"]), "s_a": sa, "s_e": se,
           "gamma_logit": g.tolist(), "gamma_pts_rel_101plus": (25 * (g - g[-1])).tolist(), "nbins": NBINS,
           "grid": grid, "final": info,
           "marginal_check": {"mean_net_term_logit": float(net.mean()), "max_abs_ubar_after_centering": 0.0,
                              "sd_net_term_logit": float(net.std())},
           "nll_pop": M.nll(1 / (1 + np.exp(-S["off"])), S["y"]),
           "nll_with_player": M.nll(1 / (1 + np.exp(-(S["off"] + (g[S['ib']] + a[S['ip']] + e[S['ic']]) @ np.r_[np.ones(5), -np.ones(5)]))), S["y"])}
    json.dump(out, open(os.path.join(WD.RESULTS, "player_fit.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "grid"}, indent=1))


# ── v2: time-varying skill and role-level pooling, full history ─────────
GROUPS = ("A", "AP", "R", "E")      # player, player x half-year, player x role, player-hero cell
EPOCH_DAY = 18628                   # 2021-01-01; periods are half-years from here


def period(day):
    return (np.asarray(day) - EPOCH_DAY) // 182


def role_idx():
    sys.path.insert(0, os.path.join(TRAINING_DIR, "paper1_revision"))
    from paper1_revision import team_support as ts
    return np.asarray(ts.ROLE_IDX, np.int64)


def group_keys(pid, hero, day):
    per = period(day)[:, None] if np.ndim(day) == 1 else period(day)
    ro = role_idx()[hero]
    return {"A": pid, "AP": pid * 64 + per, "R": pid * 8 + ro, "E": pid * H + hero}


def setup2(src, sel):
    pid, hero = src["pid"][sel], src["hero"][sel]
    K = group_keys(pid, hero, src["day"][sel])
    S = {"ib": nbin(src["nprior"][sel]), "y": src["y0"][sel], "hero": hero}
    wp = np.clip(src["wp0"][sel], 1e-6, 1 - 1e-6)
    S["off"] = np.log(wp / (1 - wp))
    for g in GROUPS:
        u, inv = np.unique(K[g], return_inverse=True)
        S["u" + g], S["i" + g] = u, inv.reshape(pid.shape)
    return S


def _map2(S, sd, init=None, iters=300):
    import torch
    torch.set_num_threads(M.THREADS)
    nb = len(NBINS)
    sizes = [nb] + [len(S["u" + g]) for g in GROUPS]
    offs = np.cumsum([0] + sizes)
    y, off = torch.tensor(S["y"]), torch.tensor(S["off"])
    sgn = torch.tensor(np.r_[np.ones(5), -np.ones(5)])
    idx = [torch.tensor(S["ib"])] + [torch.tensor(S["i" + g]) for g in GROUPS]
    th = torch.zeros(offs[-1], dtype=torch.float64)
    if init is not None:
        th[:] = torch.tensor(init)
    th.requires_grad_(True)
    prec = torch.cat([torch.ones(nb)] + [torch.full((n,), 1 / sd[g] ** 2) for g, n in zip(GROUPS, sizes[1:])]).double()

    def logit():
        u = sum(th[offs[k]:offs[k + 1]][idx[k]] for k in range(len(idx)))
        return off + (u * sgn).sum(1)
    opt = torch.optim.LBFGS([th], lr=1, max_iter=iters, tolerance_grad=1e-7, tolerance_change=1e-12, history_size=30,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit(), y, reduction="sum") + 0.5 * (prec * th ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        p = torch.sigmoid(logit()).numpy()
    t = th.detach().numpy().copy()
    w = np.repeat(p * (1 - p), 10)
    yy = S["y"]
    ll = float(np.sum(yy * np.log(np.clip(p, 1e-12, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-12, 1))))
    ev, post = ll, {}
    for k, g in enumerate(GROUPS):
        v = t[offs[k + 1]:offs[k + 2]]
        h = np.bincount(S["i" + g].ravel(), weights=w, minlength=len(v))
        ev += -0.5 * (v ** 2).sum() / sd[g] ** 2 - 0.5 * np.log1p(h * sd[g] ** 2).sum()
        post[g] = (v, 1 / np.sqrt(h + 1 / sd[g] ** 2))
    return t, dict(sd=dict(sd), loglik=ll, evidence=float(ev)), post, t[:nb]


SD2_GRID = [dict(A=0.05, AP=ap, R=r, E=0.05) for ap in (0.05, 0.1, 0.2) for r in (0.03, 0.08, 0.15)]


def _fit2_one(sd):
    p = os.path.join(WD.CACHE, "player2_ev_" + "_".join(f"{g}{sd[g]}" for g in GROUPS) + ".json")
    if os.path.exists(p):
        return json.load(open(p))
    src = dict(np.load(SRC))
    S = setup2(src, src["fit"])
    _, info, _, _ = _map2(S, sd)
    json.dump(info, open(p, "w"))
    print(json.dumps(info), flush=True)
    return info


def fit2(procs):
    from multiprocessing import get_context
    with get_context("spawn").Pool(procs) as pool:
        grid = pool.map(_fit2_one, SD2_GRID)
    best = max(grid, key=lambda d: d["evidence"])["sd"]
    src = dict(np.load(SRC))
    S = setup2(src, src["fit"])
    t, info, post, g = _map2(S, best)
    u = g[S["ib"]] + sum(post[k][0][S["i" + k]] for k in GROUPS)
    ubar = np.bincount(S["hero"].ravel(), weights=u.ravel(), minlength=H) / np.maximum(np.bincount(S["hero"].ravel(), minlength=H), 1)
    arrs = {"gamma": g, "ubar": ubar, "sd": json.dumps(best)}
    for k in GROUPS:
        arrs["keys_" + k], arrs["mu_" + k], arrs["psd_" + k] = S["u" + k], post[k][0], post[k][1]
    np.savez(FIT2, **arrs)
    zf = S["off"] + (u * np.r_[np.ones(5), -np.ones(5)]).sum(1)
    out = {"version": 2, "groups": GROUPS, "games": int(len(S["y"])), "sd": best, "grid": grid, "final": info,
           "gamma_pts_rel_101plus": (25 * (g - g[-1])).tolist(), "nbins": NBINS,
           "cells": {k: int(len(S["u" + k])) for k in GROUPS},
           "nll_pop": M.nll(1 / (1 + np.exp(-S["off"])), S["y"]), "nll_with_player": M.nll(1 / (1 + np.exp(-zf)), S["y"])}
    json.dump(out, open(os.path.join(WD.RESULTS, "player_fit2.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "grid"}, indent=1))


FIT2 = os.path.join(WD.CACHE, "player_fit2.npz")


class PlayerWorlds2:
    """v2 player term of world wid. The judge window / play period uses the latest fitted half-year's
    player x period effect where it exists (as-of 06-30), plus a fresh period innovation drawn from the
    prior (the play period is a new half-year)."""
    SALT = {"A": 41, "AP": 43, "R": 47, "E": 53, "NEW": 59}

    def __init__(self):
        z = np.load(FIT2)
        self.z = {k: z[k] for k in z.files}
        self.sd = json.loads(str(self.z["sd"]))
        self.last = int(period(np.array([20634]))[0])        # 2026-06-30's half-year

    def _draw(self, g, K, wid, k_mult=1.0, mean_only=False):
        import world as W
        keys = self.z["keys_" + g]
        i = np.clip(np.searchsorted(keys, K), 0, len(keys) - 1)
        hit = keys[i] == K
        mu = np.where(hit, self.z["mu_" + g][i], 0.0)
        sd = np.where(hit, self.z["psd_" + g][i], self.sd[g]) * k_mult
        return mu if mean_only else mu + sd * W.hnorm(K, wid, self.SALT[g])

    def u(self, pid, hero, nprior, wid, k=1.0, delta=0.0, mean_only=False, s=None, off=None):
        import world as W
        ro = role_idx()[hero]
        mo = mean_only
        A = self._draw("A", pid, wid, mean_only=mo)
        AP = self._draw("AP", pid * 64 + self.last, wid, mean_only=mo) + (0.0 if mo else self.sd["AP"] * W.hnorm(pid, wid, self.SALT["NEW"]))
        R = self._draw("R", pid * 8 + ro, wid, mean_only=mo)
        E = self._draw("E", pid * H + hero, wid, k_mult=np.where(nprior < FLOOR_N, k, 1.0), mean_only=mo)
        return self.z["gamma"][nbin(nprior)] + A + AP + R + E - self.z["ubar"][hero] + np.where(nprior == 0, W.PTS * delta, 0.0)


# ── f-cov: the drafter state's skill as a jointly fit covariate (partly circular) ──
FITCOV = os.path.join(WD.CACHE, "player_fitcov.json")


def fitcov():
    """On half A of the judge window (even game index), with the v2 outcome-only term (posterior mean) and
    logit WP_pop as offset: logistic fit of the outcome on the team differences of the drafter state's
    s(p, h) and off-role flag (beta_s, beta_off). The covariate enters every world as beta . x, centred per
    hero over half A's observed slots. Labelled partly circular: the drafter's own skill estimates are
    part of the world's truth."""
    src = dict(np.load(SRC))
    cov = dict(np.load(COV))
    jw = src["jw"]
    G = int(jw.sum())
    A = np.arange(G) % 2 == 0
    PW = PlayerWorlds2()
    pid, hero, npr = src["pid"][jw], src["hero"][jw], src["nprior"][jw]
    U = PW.u(pid, hero, npr, 0, mean_only=True)
    sg = np.r_[np.ones(5), -np.ones(5)]
    wp = np.clip(src["wp0"][jw], 1e-6, 1 - 1e-6)
    off = np.log(wp / (1 - wp)) + U @ sg
    X = np.column_stack([cov["j_s"] @ sg, cov["j_off"].astype(float) @ sg])
    y = src["y0"][jw]
    b = np.zeros(2)
    for _ in range(30):
        p = 1 / (1 + np.exp(-(off[A] + X[A] @ b)))
        g = X[A].T @ (y[A] - p)
        Hm = (X[A] * (p * (1 - p))[:, None]).T @ X[A]
        b += np.linalg.solve(Hm, g)
    se = np.sqrt(np.diag(np.linalg.inv(Hm)))
    hA = hero[A].ravel()
    v = (b[0] * cov["j_s"][A] + b[1] * cov["j_off"][A]).ravel()
    ch = np.bincount(hA, weights=v, minlength=H) / np.maximum(np.bincount(hA, minlength=H), 1)
    out = {"beta_s": float(b[0]), "beta_off": float(b[1]), "se": se.tolist(), "fit_games": int(A.sum()),
           "centre": ch.tolist(), "what": "f-cov: v2 outcome-only term + beta . (drafter state s, off-role), fit on "
                                       "judge-window half A; partly circular"}
    json.dump(out, open(FITCOV, "w"), indent=1)
    json.dump({k: v for k, v in out.items() if k != "centre"}, open(os.path.join(WD.RESULTS, "player_fitcov.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "centre"}), flush=True)


class PlayerWorldsCov:
    def __init__(self):
        self.v2 = PlayerWorlds2()
        c = json.load(open(FITCOV))
        self.b = (c["beta_s"], c["beta_off"])
        self.ch = np.asarray(c["centre"])

    def u(self, pid, hero, nprior, wid, k=1.0, delta=0.0, s=None, off=None):
        return self.v2.u(pid, hero, nprior, wid, k, delta) + self.b[0] * s + self.b[1] * off - self.ch[hero]


# ── f-cal: outcome-only player term from the drafter's raw information, not its model ──
# Same raw information the drafter state reads (slot rows plus the merged Heroes Profile rows, each game's
# scoreboard card a_shrink), before 2026-07-01 only; a different model class: a logistic regression of the
# outcome (offset logit WP_pop, team-difference sums of the ten slots) on shrunk aggregates of each player's
# record and scoreboard (player x hero, player x role, player; all-time and a 90-day half-life), log play
# counts and the familiarity bins. No drafter state, kernel, walk or calibration enters. Fit on Jan-Jun 2026
# games with features as of the first of each game's month; L2 chosen on June (fit Jan-May); the term is
# then applied with features as of 2026-06-30. World: u(p, h) = kcal * (x . w + gamma[bin n] - ubar_h)
# + residual player, player-period, player-role and player-hero effects drawn from N(0, SD) with f-out's
# evidence SDs (independent of the features). kcal = 1 is the plain fit; kcal > 1 the slope-matched scale.
CAL = os.path.join(WD.CACHE, "player_cal.npz")
CAL_JSON = os.path.join(WD.RESULTS, "player_fitcal.json")
CAL_HL = 90.0
CAL_MONTHS = ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01"]
CAL_LAMS = (1.0, 10.0, 100.0, 1000.0)     # L2 on standardized coefficients, summed log loss
CAL_RESID = {"A": 0.05, "AP": 0.05, "R": 0.08, "E": 0.05}


def _day(s):
    import datetime
    return (datetime.date.fromisoformat(s) - datetime.date(1970, 1, 1)).days


class _CalRows:
    """Slot and merged HP rows before J0 (not the sealed ids), day-sorted, with the scoreboard card; running
    sums per player x hero, player, player x role (all-time) and per player x hero, player (90-day half-life)."""
    GROUPS = ("ph", "p", "pr")

    def __init__(self, J0):
        sys.path.insert(0, P2)
        import p3_sealed
        from shared import HEROES
        C = os.path.join(P2, "cache_def_def1027")
        x = np.load(os.path.join(C, "x_slots_ext.npz"), mmap_mode="r")
        sh = np.array([HEROES.index(str(v)) for v in x["hero_names"]], np.int64)
        a = np.load(os.path.join(C, "rcf", "aug_merged.npz"), mmap_mode="r")
        zz = np.load(os.path.join(C, "rcf", "zaug_a_shrink_58686654.npz"), mmap_mode="r")
        day = np.asarray(a["day"])
        keep = np.flatnonzero(day < J0)
        keep = keep[~np.isin(np.asarray(a["rid"])[keep], p3_sealed.sealed_ids())]
        keep = keep[np.argsort(day[keep], kind="stable")]
        self.pid = np.asarray(a["pid"])[keep].astype(np.int64)
        self.hero = sh[np.asarray(a["hero"])[keep]]
        self.r = np.asarray(a["r"])[keep]
        self.day = day[keep].astype(np.int64)
        Z = np.asarray(zz["Z"])[keep].astype(np.float32)
        w = np.asarray(zz["W"])[keep].astype(np.float32)
        ok = ~np.isnan(Z)
        self.Z = np.nan_to_num(Z) * w[:, None]
        self.Zn = ok * w[:, None]
        self.role = role_idx()
        self.keys = {}
        for g, key in (("ph", self.pid * H + self.hero), ("p", self.pid), ("pr", self.pid * 8 + self.role[self.hero])):
            u, inv = np.unique(key, return_inverse=True)
            self.keys[g] = (u, inv)
        self.state = {}
        self.T = None

    def advance(self, T):
        """running sums to rows with day < T (all-time and decayed to T)."""
        lo = -10 ** 9 if self.T is None else self.T
        if self.T is not None:
            f = 0.5 ** ((T - self.T) / CAL_HL)
            for k, v in self.state.items():
                if k.endswith(":r"):
                    for arr in v:
                        arr *= f
        i0, i1 = np.searchsorted(self.day, lo), np.searchsorted(self.day, T)
        w1 = np.ones(i1 - i0)
        wr = 0.5 ** ((T - self.day[i0:i1]) / CAL_HL)
        for g, (u, inv) in self.keys.items():
            ii = inv[i0:i1]
            L = len(u)
            for tag, ww in (("", w1), ("r", wr)):
                if tag == "r" and g == "pr":
                    continue
                st = self.state.setdefault(g + ":" + tag, [np.zeros(L), np.zeros(L), np.zeros((L, 15)), np.zeros((L, 15))])
                st[0] += np.bincount(ii, weights=ww, minlength=L)
                st[1] += np.bincount(ii, weights=ww * self.r[i0:i1], minlength=L)
                for k in range(15):
                    st[2][:, k] += np.bincount(ii, weights=ww * self.Z[i0:i1, k], minlength=L)
                    st[3][:, k] += np.bincount(ii, weights=ww * self.Zn[i0:i1, k], minlength=L)
        self.T = T

    def _look(self, name, key):
        u = self.keys[name.split(":")[0]][0]
        i = np.clip(np.searchsorted(u, key), 0, len(u) - 1)
        hit = u[i] == key
        return [np.where(hit.reshape(hit.shape + (1,) * (v.ndim - 1)), v[i], 0) for v in self.state[name]]

    def feats(self, pid, hero):
        """(..., 149) features of (player, hero) cells as of the current T."""
        F = []
        ro = self.role[hero]

        def add(name, key, kr, km, withn):
            n, R, Ms, Mn = self._look(name, key)
            F.extend([R / (n + kr)] + ([np.log1p(n)] if withn else []) + list(np.moveaxis(Ms / (Mn + km), -1, 0)))
        add("ph:", pid * H + hero, 20, 10, True)
        add("ph:r", pid * H + hero, 10, 5, True)
        add("p:", pid, 50, 20, True)
        add("p:r", pid, 20, 10, True)
        add("pr:", pid * 8 + ro, 20, 10, True)
        add("pr:", pid * 8 + ro, 5, 3, False)
        add("pr:", pid * 8 + ro, 80, 40, False)
        add("ph:", pid * H + hero, 5, 3, False)
        add("ph:", pid * H + hero, 80, 40, False)
        return np.stack(F, -1).astype(np.float32)


def _cal_newton(X, off, y, lam, iters=30):
    """L2 logistic regression with offset (summed log loss + lam/2 |w|^2), Newton steps."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(off + X @ w)))
        g = X.T @ (y - p) - lam * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X + lam * np.eye(len(w))
        step = np.linalg.solve(Hm, g)
        w += step
        if np.abs(step).max() < 1e-9:
            break
    return w, Hm


def extract_cal():
    """(main box, light) features, fit and the f-cal tables -> site/cache/player_cal.npz, results/player_fitcal.json"""
    t0 = time.time()
    src = dict(np.load(SRC))
    J0 = int(src["J0"])
    R = _CalRows(J0)
    print(f"f-cal rows {len(R.day):,} ({time.time() - t0:.0f}s)", flush=True)
    cuts = [_day(m) for m in CAL_MONTHS] + [J0]
    sgn = np.r_[np.ones(5), -np.ones(5)]
    nb = len(NBINS)
    blocks = []
    for j, T in enumerate(cuts[:-1]):
        R.advance(T)
        sel = (src["day"] >= T) & (src["day"] < cuts[j + 1])
        Fx = R.feats(src["pid"][sel], src["hero"][sel])
        Fg = np.eye(nb, dtype=np.float32)[nbin(src["nprior"][sel])][..., 1:]
        blocks.append((sel, np.concatenate([Fx, Fg], -1)))
        print(f"  month {CAL_MONTHS[j]}: {sel.sum():,} games ({time.time() - t0:.0f}s)", flush=True)
    R.advance(J0)
    allF = np.concatenate([F for _, F in blocks]).reshape(-1, blocks[0][1].shape[-1])
    mu, sd = allF.mean(0), allF.std(0) + 1e-6
    del allF

    def design(js):
        X = np.concatenate([((blocks[j][1] - mu) / sd).transpose(0, 2, 1) @ sgn for j in js])
        sel = np.concatenate([np.flatnonzero(blocks[j][0]) for j in js])
        wp = np.clip(src["wp0"][sel], 1e-6, 1 - 1e-6)
        return X, np.log(wp / (1 - wp)), src["y0"][sel]

    def nll(z, y):
        p = np.clip(1 / (1 + np.exp(-z)), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    Xa, oa, ya = design(range(5))
    Xv, ov, yv = design([5])
    val = {}
    for lam in CAL_LAMS:
        w, _ = _cal_newton(Xa, oa, ya, lam)
        val[lam] = nll(ov + Xv @ w, yv)
        print(f"  lam {lam}: June nll {val[lam]:.5f}", flush=True)
    lam = min(val, key=val.get)
    X, o, y = design(range(6))
    w, Hm = _cal_newton(X, o, y, lam)
    cov = np.linalg.inv(Hm)
    wr = w / sd                                      # raw-feature coefficients (the mean shift cancels in team differences)
    K = len(w)
    # centring: mean of x . w over the fit window's observed slots per hero
    hero_fit = np.concatenate([src["hero"][blocks[j][0]] for j in range(6)])
    u_fit = np.concatenate([blocks[j][1] @ wr for j in range(6)])
    ubar = np.bincount(hero_fit.ravel(), weights=u_fit.ravel(), minlength=H) / np.maximum(np.bincount(hero_fit.ravel(), minlength=H), 1)
    gamma = np.r_[0.0, wr[K - (nb - 1):]]
    wx = wr[:K - (nb - 1)]
    # tables: judge-window slots and the paper-2 lobby players x every hero, features as of J0
    jw = src["jw"]
    jp, jh = src["pid"][jw], src["hero"][jw]
    L = np.load(P2_LOBBIES)
    cells = [(jp * H + jh).ravel()]
    for tag in ("headline", "tune"):
        P_ = np.repeat(L[f"{tag}_pid"][..., None], H, -1)
        cells.append((P_ * H + np.arange(H)).ravel())
    keys = np.unique(np.concatenate(cells))
    xb = R.feats(keys // H, keys % H) @ wx
    out = dict(keys=keys, xb=xb, gamma=gamma, ubar=ubar, w=w, sd=sd, mu=mu, cov=cov)
    np.savez(CAL, **out)
    # held-out judge window: log loss of each model's mean, and the expected slope of the world's outcome
    # residual on the drafter's personal term (no outcome noise; the 200-world check is stage slopecal)
    PWc = PlayerWorldsCal(kcal=1.0)
    m_j = PWc.mean(jp, jh, src["nprior"][jw]) @ sgn
    wp = np.clip(src["wp0"][jw], 1e-6, 1 - 1e-6)
    lw = np.log(wp / (1 - wp))
    yj = src["y0"][jw]
    Pj = src["j_P"][:, 0]
    fout = PlayerWorlds2().u(jp, jh, src["nprior"][jw], 0, mean_only=True) @ sgn
    sl = lambda a_, b_: float(np.linalg.lstsq(np.column_stack([np.ones(len(a_)), a_]), b_, rcond=None)[0][1])
    sg = lambda z: 1 / (1 + np.exp(-z))
    scan = {}
    for kc in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0):
        q = sg(lw + kc * m_j)
        scan[str(kc)] = {"judge_nll": nll(lw + kc * m_j, yj), "expected_slope_on_P": sl(Pj, q - wp)}
    res = {"what": "f-cal: outcome-only player term (records + scoreboard aggregates, logistic, no drafter state or model)",
           "fit_games": int(len(y)), "features": int(K), "lambda_grid_june_nll": {str(k): v for k, v in val.items()},
           "lambda": lam, "june_nll_pop": nll(ov, yv), "residual_sd": CAL_RESID, "half_life_days": CAL_HL,
           "familiarity_gamma_pts_rel_0": (25 * gamma).tolist(), "cells": int(len(keys)),
           "judge_window": {"games": int(jw.sum()), "nll_pop": nll(lw, yj),
                            "nll_drafter_P": float(-np.mean(yj * np.log(np.clip(wp + Pj, 1e-9, 1)) + (1 - yj) * np.log(np.clip(1 - wp - Pj, 1e-9, 1)))),
                            "nll_fout_mean": nll(lw + fout, yj),
                            "fout_expected_slope_on_P": sl(Pj, sg(lw + fout) - wp),
                            "slope_of_r_on_P": sl(Pj, yj - wp),
                            "slope_of_r_on_fcal": sl(sg(lw + m_j) - wp, yj - wp),
                            "corr_fcal_P": float(np.corrcoef(sg(lw + m_j) - wp, Pj)[0, 1]),
                            "kcal_scan": scan}}
    json.dump(res, open(CAL_JSON, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


class PlayerWorldsCal:
    """f-cal player term of world wid. kcal scales the fitted mean (default: player_fitcal.json 'kcal', else 1);
    env WORLDS_FCAL_K overrides."""
    SALT = {"A": 71, "AP": 73, "R": 79, "E": 83}

    def __init__(self, kcal=None):
        z = np.load(CAL)
        self.keys, self.xb, self.gamma, self.ubar = z["keys"], z["xb"], z["gamma"], z["ubar"]
        if kcal is None:
            kcal = os.environ.get("WORLDS_FCAL_K")
            if kcal is None and os.path.exists(CAL_JSON):
                kcal = json.load(open(CAL_JSON)).get("kcal")
        self.kcal = float(kcal if kcal is not None else 1.0)

    def mean(self, pid, hero, nprior):
        key = pid * H + hero
        i = np.clip(np.searchsorted(self.keys, key), 0, len(self.keys) - 1)
        x = np.where(self.keys[i] == key, self.xb[i], 0.0)
        return x + self.gamma[nbin(nprior)] - self.ubar[hero]

    def u(self, pid, hero, nprior, wid, k=1.0, delta=0.0, mean_only=False, s=None, off=None):
        import world as W
        m = self.kcal * self.mean(pid, hero, nprior)
        if mean_only:
            return m
        sd = CAL_RESID
        ro = role_idx()[hero]
        e = sd["A"] * W.hnorm(pid, wid, self.SALT["A"]) + sd["AP"] * W.hnorm(pid * 64 + 7, wid, self.SALT["AP"]) \
            + sd["R"] * W.hnorm(pid * 8 + ro, wid, self.SALT["R"]) \
            + sd["E"] * np.where(nprior < FLOOR_N, k, 1.0) * W.hnorm(pid * H + hero, wid, self.SALT["E"])
        return m + e + np.where(nprior == 0, W.PTS * delta, 0.0)


# ── the player term in a world ───────────────────────────────────────────
class PlayerWorlds:
    def __init__(self):
        z = np.load(FIT)
        self.z = {k: z[k] for k in z.files}

    def u(self, pid, hero, nprior, wid, k=1.0, delta=0.0):
        """u'(p, h) of world wid (any shape); unseen players / cells from the prior."""
        import world as W
        z = self.z
        P, C = z["players"], z["cells"]
        i = np.clip(np.searchsorted(P, pid), 0, len(P) - 1)
        hitP = P[i] == pid
        a = np.where(hitP, z["a_mu"][i], 0.0) + np.where(hitP, z["a_sd"][i], float(z["sa"])) * W.hnorm(pid, wid, SALT_A)
        ck = pid * H + hero
        j = np.clip(np.searchsorted(C, ck), 0, len(C) - 1)
        hitC = C[j] == ck
        n = nprior
        esd = np.where(hitC, z["e_sd"][j], float(z["se"])) * np.where(n < FLOOR_N, k, 1.0)
        e = np.where(hitC, z["e_mu"][j], 0.0) + esd * W.hnorm(ck, wid, SALT_E)
        return z["gamma"][nbin(n)] + a + e - z["ubar"][hero] + np.where(n == 0, W.PTS * delta, 0.0)


def slope(n_worlds, version=1, kcal=None, tag=""):
    src = dict(np.load(SRC))
    jw = src["jw"]
    if version == "cal":
        PW = PlayerWorldsCal(kcal)
    else:
        PW = PlayerWorlds() if version == 1 else (PlayerWorldsCov() if version == "cov" else PlayerWorlds2())
    if version == "cov":
        cov = dict(np.load(COV))
    pid, hero, npr = src["pid"][jw], src["hero"][jw], src["nprior"][jw]
    wp0 = np.clip(src["wp0"][jw], 1e-6, 1 - 1e-6)
    P = src["j_P"]
    mn = src["j_min3"]
    known = src["j_known"]
    G = len(wp0)
    sgn = np.r_[np.ones(5), -np.ones(5)]
    bins = [(0, 4, "0-4"), (5, 24, "5-24"), (25, 99, "25-99"), (100, 499, "100-499"), (500, 10 ** 12, "500+")]
    rows_P = np.r_[P[:, 0], P[:, 1]]
    rows_m = np.r_[mn[:, 0], mn[:, 1]]
    keep = np.r_[known, known]
    half = np.ones(2 * G, bool)
    if version == "cov":                       # f-cov's beta is fit on half A: check on half B only
        hb = np.arange(G) % 2 == 1
        half = np.r_[hb, hb]

    def fit_slopes(y0):
        r = np.r_[y0 - wp0, (1 - y0) - (1 - wp0)]
        out = {}
        for sample, km in (("all_games", half), ("all_known", keep & half)):
            o = {}
            for lo, hi, lab in bins:
                s = km & (rows_m >= lo) & (rows_m <= hi)
                X = np.column_stack([np.ones(s.sum()), rows_P[s]])
                o[lab] = float(np.linalg.lstsq(X, r[s], rcond=None)[0][1])
            out[sample] = o
        return out
    res = []
    for w in range(n_worlds):
        U = PW.u(pid, hero, npr, w, s=cov["j_s"], off=cov["j_off"].astype(float)) if version == "cov" else PW.u(pid, hero, npr, w)
        z = np.log(wp0 / (1 - wp0)) + U @ sgn
        y0 = (np.random.default_rng([w, 13]).random(G) < 1 / (1 + np.exp(-z))).astype(float)
        res.append(fit_slopes(y0))
    real = fit_slopes(src["y0"][jw])
    out = {"worlds": n_worlds, "games": int(G), "real": real, "worlds_summary": {},
           "reference": "personalization/results/P3_TEAM_SUPPORT.md section 2 (0.74-0.84 per bin, all_games)"}
    for sample in ("all_games", "all_known"):
        out["worlds_summary"][sample] = {}
        for _, _, lab in bins:
            v = np.array([r[sample][lab] for r in res])
            out["worlds_summary"][sample][lab] = {"mean": float(v.mean()), "sd": float(v.std()),
                                                  "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
                                                  "real": real[sample][lab], "real_pct": float(100 * (v < real[sample][lab]).mean())}
    out["model_version"] = version
    if version == "cal":
        out["kcal"] = PW.kcal
    suf = {1: "", 2: "2", "cov": "_cov", "cal": "_cal" + tag}[version]
    json.dump(out, open(os.path.join(WD.RESULTS, "check_player_slope" + suf + ".json"), "w"), indent=1)
    print(json.dumps(out["worlds_summary"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "extract_p2", "extract_cov", "fit", "slope", "fit2", "slope2", "fitcov",
                                      "slopecov", "extract_cal", "slopecal"])
    ap.add_argument("--kcal", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--work", default="")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    if a.stage == "extract":
        extract(a.work)
    elif a.stage == "extract_p2":
        extract_p2()
    elif a.stage == "extract_cov":
        extract_cov(a.work)
    elif a.stage == "extract_cal":
        extract_cal()
    elif a.stage == "slopecal":
        slope(a.n, version="cal", kcal=a.kcal, tag=a.tag)
    elif a.stage == "fitcov":
        fitcov()
    elif a.stage == "slopecov":
        slope(a.n, version="cov")
    elif a.stage == "fit":
        fit()
    elif a.stage == "fit2":
        fit2(a.procs)
    elif a.stage == "slope2":
        slope(a.n, version=2)
    else:
        slope(a.n)
