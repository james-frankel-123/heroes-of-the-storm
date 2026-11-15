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

    def _draw(self, g, K, wid, k_mult=1.0):
        import world as W
        keys = self.z["keys_" + g]
        i = np.clip(np.searchsorted(keys, K), 0, len(keys) - 1)
        hit = keys[i] == K
        mu = np.where(hit, self.z["mu_" + g][i], 0.0)
        sd = np.where(hit, self.z["psd_" + g][i], self.sd[g]) * k_mult
        return mu + sd * W.hnorm(K, wid, self.SALT[g])

    def u(self, pid, hero, nprior, wid, k=1.0, delta=0.0):
        import world as W
        ro = role_idx()[hero]
        A = self._draw("A", pid, wid)
        AP = self._draw("AP", pid * 64 + self.last, wid) + self.sd["AP"] * W.hnorm(pid, wid, self.SALT["NEW"])
        R = self._draw("R", pid * 8 + ro, wid)
        E = self._draw("E", pid * H + hero, wid, k_mult=np.where(nprior < FLOOR_N, k, 1.0))
        return self.z["gamma"][nbin(nprior)] + A + AP + R + E - self.z["ubar"][hero] + np.where(nprior == 0, W.PTS * delta, 0.0)


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


def slope(n_worlds, version=1):
    src = dict(np.load(SRC))
    jw = src["jw"]
    PW = PlayerWorlds() if version == 1 else PlayerWorlds2()
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

    def fit_slopes(y0):
        r = np.r_[y0 - wp0, (1 - y0) - (1 - wp0)]
        out = {}
        for sample, km in (("all_games", np.ones(2 * G, bool)), ("all_known", keep)):
            o = {}
            for lo, hi, lab in bins:
                s = km & (rows_m >= lo) & (rows_m <= hi)
                X = np.column_stack([np.ones(s.sum()), rows_P[s]])
                o[lab] = float(np.linalg.lstsq(X, r[s], rcond=None)[0][1])
            out[sample] = o
        return out
    res = []
    for w in range(n_worlds):
        U = PW.u(pid, hero, npr, w)
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
    json.dump(out, open(os.path.join(WD.RESULTS, "check_player_slope" + ("" if version == 1 else "2") + ".json"), "w"), indent=1)
    print(json.dumps(out["worlds_summary"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "extract_p2", "fit", "slope", "fit2", "slope2"])
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--work", default="")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    if a.stage == "extract":
        extract(a.work)
    elif a.stage == "extract_p2":
        extract_p2()
    elif a.stage == "fit":
        fit()
    elif a.stage == "fit2":
        fit2(a.procs)
    elif a.stage == "slope2":
        slope(a.n, version=2)
    else:
        slope(a.n)
