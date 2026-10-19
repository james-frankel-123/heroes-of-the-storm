"""
P3 tournament (paper 2, player-aware drafters): lobbies and personal states.

Windows (all games before CAP_TS 2026-09-28; the sealed personalization
build 2.55.17.98025 is never read):
  population data of the current drafters  pinned snapshot, through 2026-05-22
  OOT lobbies                              2026-05-23 to 2026-09-11 (backtests)
  tournament lobbies                       2026-05-23 to 2026-06-30
  judge window                             2026-07-01 to 2026-09-11 (after every lobby)

Personal state of a player at a query day (the drafter's s(p, h) for all 90
heroes, in shared.HEROES order), from the main skill model (full_ms: met2_hs2
structure, full scoreboard export, a_shrink, merged full_1017 history):
  s = experience offset (lifetime slot-row counts before the day)
      + posterior mean (all evidence rows on days before the day)
      + temporal layer (drift and new-hero learning, slot games before the day)
Variants (VINTAGES):
  cur  state through the day before the lobby (the maintained drafter)
  s4m  state frozen through 2026-02-10 (build 2.55.14.95918, the 4-month vintage)
  s1y  state frozen through 2025-06-02 (build 2.55.10.94189, the 1-year vintage)
  judge  evidence from the judge window only, queried at its end; no layer.
         Shares no outcome with any drafter state (drafter evidence ends
         before the lobby, at the latest 2026-06-29).
Pools: heroes with at least one earlier game (merged counts, lag 1 day).
Off-role: fine role under 10% of the player's earlier games (50+ games).

Stages (3090, env of remote/run_rcf_full.sh; cores 20-35):
  python personalization/p3t_data.py lobbies
  python personalization/p3t_data.py states [--vintages cur,s4m,s1y,judge]
  python personalization/p3t_data.py judgefit
Outputs: $P3_CACHE/p3t/{lobbies,state_<v>,judgefit}.npz (player keys stay in caches).
"""
import argparse
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import p3_hs_core as C

OUT = os.path.join(C.CACHE, "p3t")
os.makedirs(OUT, exist_ok=True)
H = 90
SEALED = "2.55.17.98025"
OOT_START, OOT_END = C.day_of("2026-05-23"), C.day_of("2026-09-11")
LOBBY_END = C.day_of("2026-06-30")
J0, J1 = C.day_of("2026-07-01"), C.day_of("2026-09-11")
VINTAGES = {"s4m": C.day_of("2026-02-10"), "s1y": C.day_of("2025-06-02")}
PICK_STEPS = [4, 5, 6, 7, 8, 11, 12, 13, 14, 15]


def site_tier(league_tier, avg_mmr):
    from overfit2026.data import site_tier as st
    return st(league_tier, avg_mmr)


def to_shared(names):
    """index map: d hero index -> shared.HEROES index"""
    from shared import HEROES
    s = {h: i for i, h in enumerate(HEROES)}
    return np.array([s[str(h)] for h in names], np.int64)


# ------------------------------------------------------------------ lobbies

def build_lobbies(d):
    """Every OOT game (2026-05-23 to 09-11, not the sealed build) with a
    16-step draft whose 10 picks all join to slot rows. steps (G, 16, 4):
    shared hero index, type (0 ban, 1 pick), real team, slot row (-1 bans)."""
    names = [str(x) for x in d["hero_names"]]
    hidx = {n: i for i, n in enumerate(names)}
    sh = to_shared(names)
    post = np.flatnonzero(d["post"] & (d["day"] >= OOT_START) & (d["day"] <= OOT_END))
    key = {(int(d["replay_id"][i]), int(d["hero"][i])): int(i) for i in post}
    games = json.load(gzip.open(os.path.join(C.CACHE, "x_post_games.json.gz"), "rt"))
    keep, steps, skip = [], [], {}
    for g in games:
        if g["game_version"] == SEALED:
            skip["sealed"] = skip.get("sealed", 0) + 1
            continue
        o = g["draft_order"]
        if len(o) != 16:
            skip["not16"] = skip.get("not16", 0) + 1
            continue
        tier = site_tier(g.get("league_tier"), g.get("avg_mmr"))
        if tier == "unknown":
            skip["tier"] = skip.get("tier", 0) + 1
            continue
        b0 = set(g["team0_bans"])
        b1 = set(g["team1_bans"])
        st, ok = [], True
        for h, ty, pn, sl in o:
            if h not in hidx:
                ok = False
                break
            if ty == 1:
                row = key.get((int(g["replay_id"]), hidx[h]))
                if row is None:
                    ok = False
                    break
                st.append((int(sh[hidx[h]]), 1, int(d["team"][row]), row))
            else:
                tm = 0 if h in b0 else 1 if h in b1 else (0 if sl == 1 else 1)
                st.append((int(sh[hidx[h]]), 0, tm, -1))
        if not ok:
            skip["join"] = skip.get("join", 0) + 1
            continue
        if sum(s[1] for s in st) != 10 or [s[1] for s in st] != [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]:
            skip["order"] = skip.get("order", 0) + 1
            continue
        # Storm League alternation: the first-pick team bans at steps 0, 2, 10
        ft = st[4][2]
        exp = [ft, 1 - ft, ft, 1 - ft, ft, 1 - ft, 1 - ft, ft, ft, 1 - ft, ft, 1 - ft, 1 - ft, ft, ft, 1 - ft]
        if [s[2] for s in st] != exp:
            skip["teams"] = skip.get("teams", 0) + 1
            continue
        keep.append(g)
        steps.append(st)
    steps = np.array(steps, np.int64)
    rows = steps[:, :, 3]
    prow = rows[rows >= 0].reshape(len(keep), 10)
    out = dict(
        replay_id=np.array([g["replay_id"] for g in keep], np.int64),
        map=np.array([g["game_map"] for g in keep]),
        tier=np.array([site_tier(g.get("league_tier"), g.get("avg_mmr")) for g in keep]),
        version=np.array([g["game_version"] for g in keep]),
        winner=np.array([g["winner"] for g in keep], np.int8),
        avg_mmr=np.array([g["avg_mmr"] if g["avg_mmr"] is not None else np.nan for g in keep], np.float32),
        steps=steps, slot_rows=prow, day=d["day"][prow[:, 0]],
        first_team=steps[:, 4, 2].astype(np.int8))
    # sanity: the real winner agrees with the slot outcome
    y0 = np.where(d["team"][prow[:, 0]] == 0, d["y"][prow[:, 0]], 1 - d["y"][prow[:, 0]])
    agree = float(np.mean((y0 == 1) == (out["winner"] == 0)))
    np.savez(os.path.join(OUT, "lobbies.npz"), **out)
    print(f"lobbies: {len(keep):,} OOT games kept; skipped {skip}; winner/slot agreement {agree:.4f}; "
          f"tournament window {int((out['day'] <= LOBBY_END).sum()):,}", flush=True)


def load_lobbies():
    z = np.load(os.path.join(OUT, "lobbies.npz"))
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ walk

def walk(aug, par, mets, mw, q_pid, q_day, ev_lo=None, q_hero=None, want_all=True):
    """As-of posterior for (player, day) queries: evidence rows with
    ev_lo <= day <= q_day - 1 (ev_lo None: from the start); merged counts
    (all rows) and slot-row counts with day <= q_day - 1. State is kept only
    for the queried players. Returns dict in query order:
      mean_all/var_all (nq, H) [want_all] or mean/var (nq,) at q_hero,
      n_all (nq, H) merged counts, n_slot (nq, H) slot-row counts.
    Same algebra as p3_rcf.online_walk (no recency weights; full_ms has none)."""
    import p3_rcf as R
    day = aug["day"]
    npl = int(aug["n_players"])
    upid = np.unique(q_pid)
    pos = np.full(npl, -1, np.int64)
    pos[upid] = np.arange(len(upid))
    sel = np.flatnonzero(pos[aug["pid"]] >= 0)          # day-sorted
    sday = day[sel]
    sp = pos[aug["pid"][sel]]
    shero = aug["hero"][sel]
    use_m = par.G is not None and mets is not None
    n = len(upid)
    Sst = np.zeros((n, H))
    Pst = np.zeros((n, H))
    Nall = np.zeros((n, H), np.float32)
    Nsl = np.zeros((n, H), np.float32)
    if use_m:
        Mn = par.G.shape[0]
        Xst = np.zeros((n, H, Mn), np.float32)
        Cst = np.zeros((n, H, Mn), np.float32)
    nq = len(q_pid)
    o = np.argsort(q_day, kind="stable")
    qd = q_day[o]
    ud, qs = np.unique(qd, return_index=True)
    qe = np.r_[qs[1:], nq]
    res = {}
    if want_all:
        res["mean_all"] = np.empty((nq, H), np.float32)
        res["var_all"] = np.empty((nq, H), np.float32)
    else:
        res["mean"] = np.empty(nq)
        res["var"] = np.empty(nq)
    res["n_all"] = np.empty((nq, H), np.float32)
    res["n_slot"] = np.empty((nq, H), np.float32)
    ap = 0
    t0 = time.time()
    for t, a_, b_ in zip(ud, qs, qe):
        e = np.searchsorted(sday, t - 1, side="right")
        if e > ap:
            rr = np.arange(ap, e)
            pi, hi = sp[rr], shero[rr]
            np.add.at(Nall, (pi, hi), 1.0)
            isl = aug["slot"][sel[rr]] >= 0
            np.add.at(Nsl, (pi[isl], hi[isl]), 1.0)
            ev = rr if ev_lo is None else rr[sday[rr] >= ev_lo]
            if len(ev):
                ai = sel[ev]
                pe, he = sp[ev], shero[ev]
                np.add.at(Sst, (pe, he), aug["r_adj"][ai] / aug["v"][ai])
                np.add.at(Pst, (pe, he), 1.0 / aug["v"][ai])
                if use_m:
                    Xm = mets[ai].astype(np.float32)
                    g = ~np.isnan(Xm)
                    wr = mw[ai].astype(np.float32)
                    np.add.at(Xst, (pe, he), np.where(g, Xm, 0) * wr[:, None])
                    np.add.at(Cst, (pe, he), g.astype(np.float32) * wr[:, None])
            ap = e
        q = o[a_:b_]
        pq = pos[q_pid[q]]
        kw = {}
        if par.W is not None:
            kw["N"] = Nall[pq].astype(np.float64)
        if use_m:
            kw["X"], kw["Cm"] = Xst[pq], Cst[pq]
        hq = np.zeros(len(q), np.int64) if q_hero is None else q_hero[q]
        for c0 in range(0, len(q), 20000):
            cq = slice(c0, c0 + 20000)
            kc = {k: v[cq] for k, v in kw.items()}
            if want_all:
                m_, v_ = R.posterior(par, Sst[pq[cq]], Pst[pq[cq]], hq[cq], want_all=True, **kc)
                res["mean_all"][q[cq]] = m_
                res["var_all"][q[cq]] = v_
            else:
                m_, v_ = R.posterior(par, Sst[pq[cq]], Pst[pq[cq]], hq[cq], **kc)
                res["mean"][q[cq]] = m_
                res["var"][q[cq]] = v_
        res["n_all"][q] = Nall[pq]
        res["n_slot"][q] = Nsl[pq]
    print(f"  walk: {nq:,} queries, {n:,} players, {time.time() - t0:.0f}s", flush=True)
    return res


def offset(table, n_slot):
    """Experience offset for all heroes from lifetime slot-row counts (nq, H)."""
    n_p = n_slot.sum(1)
    nb = len(C.NPH_EDGES)
    bp = np.searchsorted(C.NP_EDGES, n_p, side="right") - 1
    bh = np.searchsorted(C.NPH_EDGES, n_slot, side="right") - 1
    return table[bp[:, None] * nb + bh]


class Layer:
    """The full_ms temporal layer at arbitrary (player, day) queries:
    shift(h) = drift_scale * rho_p + learn_scale * nu_p * [n_all(h) < 5],
    rho = decayed mean of u over the player's slot games before the day
    (half-life hl_d, shrink kd), nu = mean u over his early games (fewer than
    5 earlier games on the hero; shrink ke). u = r - (offset + posterior mean)
    of each slot game, from the cached as-of walk parts (parts_full_ms.npz)."""

    def __init__(self, d, aug, meta):
        import p3_c_merged as M
        from p3_heroes import HKEY
        z = np.load(os.path.join(C.CACHE, "rcf", "parts_full_ms.npz"))
        assert len(z["qwalk"]) == len(d["pid"]) and (z["qwalk"] == np.arange(len(d["pid"]))).all()
        np_s = M.lag1_key_sums(d["day"], d["pid"].astype(np.int64)).astype(np.int64)
        nph_s = M.lag1_key_sums(d["day"], d["pid"].astype(np.int64) * HKEY + d["hero"]).astype(np.int64)
        mu = aug["table"][C.exp_bins(np_s, nph_s)]
        u = d["r"] - (mu + z["mean"])
        ok = np.isfinite(u)
        early = (z["nh"] < 5) & ok
        u = np.where(ok, u, 0.0)
        self.m = meta
        lam = np.log(2) / meta["hl_d"]
        self.lam = lam
        o = np.lexsort((d["day"], d["pid"]))
        self.pid = d["pid"][o].astype(np.int64)
        self.day = d["day"][o].astype(np.int64)
        self.d0 = int(self.day.min())
        w = np.exp(lam * (self.day - self.d0)) * ok[o]
        self.cw = np.r_[0.0, np.cumsum(w)]
        self.cwu = np.r_[0.0, np.cumsum(w * u[o])]
        self.ceu = np.r_[0.0, np.cumsum(np.where(early[o], u[o], 0.0))]
        self.ce = np.r_[0.0, np.cumsum(early[o].astype(float))]
        self.key = self.pid * 100000 + self.day

    def at(self, q_pid, q_day, n_all):
        lo = np.searchsorted(self.key, q_pid.astype(np.int64) * 100000)
        hi = np.searchsorted(self.key, q_pid.astype(np.int64) * 100000 + (q_day.astype(np.int64) - 1), side="right")
        sc = np.exp(-self.lam * (q_day.astype(np.int64) - self.d0))
        Sd = (self.cwu[hi] - self.cwu[lo]) * sc
        Nd = (self.cw[hi] - self.cw[lo]) * sc
        rho = Sd / (Nd + self.m["kd"])
        nu = (self.ceu[hi] - self.ceu[lo]) / (self.ce[hi] - self.ce[lo] + self.m["ke"])
        return (self.m["drift_scale"] * rho)[:, None] + (self.m["learn_scale"] * nu)[:, None] * (n_all < 5)


def layer_meta():
    j = json.load(open(os.path.join(C.RESULTS, "p3_c_tests_main.json")))
    m = j["arms"]["full_ms_layer"]["layer"]
    return {k: m[k] for k in ("hl_d", "kd", "ke", "drift_scale", "learn_scale", "coef")}


def off_role(d, n_all):
    meta = C.hero_meta(d["hero_names"])
    fine = meta["fine"]
    n_p = n_all.sum(1)
    cnt = np.zeros((len(n_all), fine.max() + 1), np.float32)
    for f in range(fine.max() + 1):
        cnt[:, f] = n_all[:, fine == f].sum(1)
    share = cnt[:, fine] / np.maximum(n_p[:, None], 1)
    return (n_p[:, None] >= 50) & (share < 0.10)


def load_model(tag="full_ms"):
    import p3_rcf as R
    par = R.get_params(tag)
    d, aug = R.load_aug("merged")
    mets, mw = R.aug_metrics(aug, par.meta.get("sb", "legacy")) if par.G is not None else (None, None)
    return par, d, aug, mets, mw


def states(vintages, judge_tag=None):
    """judge_tag: parameters of the judge's skill model (the fit on the other
    player half, full_ms_flip); the drafter states use full_ms."""
    t0 = time.time()
    par, d, aug, mets, mw = load_model()
    jpar = _jpar(judge_tag) if judge_tag else par
    sh = to_shared(d["hero_names"])
    L = load_lobbies()
    meta = layer_meta()
    lay = Layer(d, aug, meta) if any(v != "judge" for v in vintages) else None
    print(f"model loaded ({time.time() - t0:.0f}s)", flush=True)
    for v in vintages:
        if v == "cur":
            gsel = np.arange(len(L["replay_id"]))
            qday = np.repeat(L["day"][gsel], 10)
        else:
            gsel = np.flatnonzero(L["day"] <= LOBBY_END)
            fix = (J1 + 1) if v == "judge" else (VINTAGES[v] + 1)
            qday = np.full(len(gsel) * 10, fix, np.int64)
        rows = L["slot_rows"][gsel].ravel()
        qpid = d["pid"][rows].astype(np.int64)
        # one walk per distinct (player, day)
        ukey, inv = np.unique(qpid * 100000 + qday, return_inverse=True)
        upid, uday = ukey // 100000, ukey % 100000
        res = walk(aug, jpar if v == "judge" else par, mets, mw, upid, uday, ev_lo=J0 if v == "judge" else None)
        s = offset(aug["table"], res["n_slot"]) + res["mean_all"]
        if v != "judge":
            s = s + lay.at(upid, uday, res["n_all"])
        off = off_role(d, res["n_all"])
        out = dict(gsel=gsel, s=s[inv][:, sh_inv(sh)].astype(np.float32 if v != "cur" else np.float16),
                   sd=np.sqrt(res["var_all"])[inv][:, sh_inv(sh)].astype(np.float16),
                   n=res["n_all"][inv][:, sh_inv(sh)].astype(np.uint16),
                   off=off[inv][:, sh_inv(sh)], qday=qday.astype(np.int32))
        if v == "judge":
            # the raw judge-window record per hero (judge-free personal score):
            # sum and count of r = y - WP_pop over the player's window slot games
            win = np.flatnonzero((d["day"] >= J0) & (d["day"] <= J1) & d["post"] & (d["version"] != _vidx(d, SEALED)))
            pos = {int(p): i for i, p in enumerate(upid)}
            m = np.isin(d["pid"][win], upid)
            win = win[m]
            pi = np.array([pos[int(p)] for p in d["pid"][win]])
            cell = pi * H + sh[d["hero"][win]]
            rs = np.bincount(cell, weights=d["r"][win], minlength=len(upid) * H).reshape(-1, H)
            rn = np.bincount(cell, minlength=len(upid) * H).reshape(-1, H)
            out.update(rec_sum=rs[inv].astype(np.float32), rec_n=rn[inv].astype(np.uint16))
        np.savez(os.path.join(OUT, f"state_{v}.npz"), **out)
        print(f"state {v}: {len(rows):,} slots ({len(ukey):,} player-days) ({time.time() - t0:.0f}s)", flush=True)


def sh_inv(sh):
    """column permutation: shared index j -> d index"""
    inv = np.empty(H, np.int64)
    inv[sh] = np.arange(H)
    return inv


def _vidx(d, ver):
    vs = [str(x) for x in d["versions"]]
    return vs.index(ver) if ver in vs else -999


def _jpar(tag):
    import p3_rcf as R
    p = R.get_params(tag)
    print(f"judge parameters: {tag} (n_fit {p.meta.get('n_fit')})", flush=True)
    return p


def judgefit(tag="full_ms_flip"):
    """Judge-window as-of predictions at the played hero for every slot of
    every judge-window game (evidence from J0 to the day before), the input
    of the judge combiner. Also the d2c WP and outcome."""
    t0 = time.time()
    par, d, aug, mets, mw = load_model(tag)
    sv = _vidx(d, SEALED)
    q = np.flatnonzero(d["post"] & (d["day"] >= J0) & (d["day"] <= J1) & (d["version"] != sv))
    res = walk(aug, par, mets, mw, d["pid"][q].astype(np.int64), d["day"][q].astype(np.int64),
               ev_lo=J0, q_hero=d["hero"][q].astype(np.int64), want_all=False)
    hq = d["hero"][q]
    ns = res["n_slot"]
    n_p = ns.sum(1)
    nb = len(C.NPH_EDGES)
    bp = np.searchsorted(C.NP_EDGES, n_p, side="right") - 1
    bh = np.searchsorted(C.NPH_EDGES, ns[np.arange(len(q)), hq], side="right") - 1
    mu = aug["table"][bp * nb + bh]
    off = off_role(d, res["n_all"])[np.arange(len(q)), hq]
    np.savez(os.path.join(OUT, "judgefit.npz"), replay_id=d["replay_id"][q], g=d["g"][q], team=d["team"][q],
             hero=to_shared(d["hero_names"])[hq], pred=(mu + res["mean"]).astype(np.float32),
             var=res["var"].astype(np.float32), off=off, y=d["y"][q], wp=d["wp"][q].astype(np.float32),
             day=d["day"][q], mmr=d["player_mmr"][q])
    print(f"judgefit: {len(q):,} slots ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["lobbies", "states", "judgefit"])
    ap.add_argument("--vintages", default="cur,s4m,s1y,judge")
    ap.add_argument("--judge-params", default="full_ms_flip")
    a = ap.parse_args()
    if a.stage == "lobbies":
        import p3_x_common as X
        build_lobbies(X.load_ext())
    elif a.stage == "states":
        states(a.vintages.split(","), a.judge_params)
    else:
        judgefit(a.judge_params)


if __name__ == "__main__":
    main()
