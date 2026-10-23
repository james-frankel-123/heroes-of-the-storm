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

os.environ.setdefault("P1_TIERS", "site")   # paper-1 site tiers for every slim-table read

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
    from shared import MAPS
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
        if g["game_map"] not in MAPS:
            # Haunted Mines returned in 2.55.17: no model or judge has seen it
            skip["map"] = skip.get("map", 0) + 1
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


def ranks():
    """Who holds the bans (Max, 2026-10-23: the highest-ranked player of each team bans; pick
    order is free, whoever locks first takes the turn). Per lobby slot in pick order (as
    slot_rows): rank = the player's as-of MMR (Heroes Profile's causal rating, the latest stamp
    parsed before the game started: cache/mmr_at_game.npz, player_mmr_causal). Per-player league
    tiers are not stored (only the game's), so a player without a causal MMR is ranked at the
    lobby's average MMR, else at the median causal MMR of the lobby's site tier. party = the
    Heroes Profile party id (0 = solo; groups are per game and team). lobby_ranks.npz:
    rank (G, 10) float32, rank_src (G, 10) int8 (0 causal, 1 lobby average, 2 tier median),
    party (G, 10) int64."""
    import p3_x_common as X
    import p3_mmr_feats as MF
    t0 = time.time()
    d = X.load_ext()
    L = load_lobbies()
    rows = L["slot_rows"]
    G = len(rows)
    rid = np.repeat(L["replay_id"], 10)
    key = d["player_keys"][d["pid"][rows.ravel()]]
    assert (d["replay_id"][rows.ravel()] == rid).all(), "slot rows out of step with the lobbies"
    pm = MF.slot_features(rid, key, d["team"][rows.ravel()], MF.load_stamps(C.CACHE))["pm"].reshape(G, 10)
    rank = pm.astype(np.float32)
    src = np.where(np.isfinite(pm), 0, 1).astype(np.int8)
    tiers = np.unique(L["tier"])
    med = {t_: float(np.nanmedian(np.where(np.isfinite(pm[L["tier"] == t_]), pm[L["tier"] == t_], np.nan)))
           for t_ in tiers}
    avg = L["avg_mmr"].astype(np.float32)
    fb = np.where(np.isfinite(avg), avg, np.array([med[t_] for t_ in L["tier"]], np.float32))
    miss = ~np.isfinite(pm)
    rank[miss] = np.broadcast_to(fb[:, None], (G, 10))[miss]
    src[miss & ~np.isfinite(avg)[:, None]] = 2
    party = d["party"][rows.ravel()].astype(np.int64).reshape(G, 10)
    np.savez(os.path.join(OUT, "lobby_ranks.npz"), rank=rank, rank_src=src, party=party)
    win = L["day"] <= LOBBY_END
    print(f"lobby ranks: {G:,} lobbies; causal MMR on {(src[win] == 0).mean():.4f} of window slots, lobby "
          f"average {(src[win] == 1).mean():.4f}, tier median {(src[win] == 2).mean():.4f}; solo slots "
          f"{(party[win] == 0).mean():.3f} ({time.time() - t0:.0f}s)", flush=True)


def load_lobbies():
    z = np.load(os.path.join(OUT, "lobbies.npz"))
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ walk

# ------------------------------------------------------------------ external player states
# The definitive rerun takes the player states from the personalization lane's model
# (coordinator, 2026-10-22). Two env switches, both keyed by the walk calls in order:
#   P3T_WALK_RECORD=<dir>  compute as usual and also write every walk's query list to
#                          <dir>/<stage>__<k>.npz (player_key, as_of_day, ev_lo, ev_parity,
#                          exclude_lo, exclude_hi, half, hero)
#   P3T_WALK_REPLAY=<dir>  read the state for those queries from <dir>/<stage>__<k>.npz
#                          (fields s, sd, n; same order, the queries echoed and checked) in
#                          place of the walk. The external s already holds the experience
#                          offset and the temporal layer, so offset(), the offset-table
#                          lookups and Layer.at contribute zero in this mode.
WALK_RECORD = os.environ.get("P3T_WALK_RECORD")
WALK_REPLAY = os.environ.get("P3T_WALK_REPLAY")
_HALF = {}            # id(params) -> 0 drafter fit, 1 judge fit (other player half)
_KEYS = {}            # "keys": d["player_keys"] of the loaded slot table
_CALLS = [0]


def _walk_name():
    pos = [a_ for a_ in sys.argv[1:] if not a_.startswith("-")][:1]
    stage = os.path.basename(sys.argv[0]).replace(".py", "") + ("_" + pos[0] if pos else "")
    name = f"{stage}__{_CALLS[0]}"
    _CALLS[0] += 1
    return name


def _walk_queries(par, q_pid, q_day, ev_lo, q_hero, ev_ok):
    n = len(q_pid)
    if ev_ok is None:
        par_, xlo, xhi = 0, -1, -1
    elif ev_ok is SPLIT_D:
        par_, xlo, xhi = 1, -1, -1
    elif ev_ok is SPLIT_J:
        par_, xlo, xhi = 2, OOT_START, LOBBY_END
    else:
        raise SystemExit("external states: unknown evidence filter")
    if id(par) not in _HALF:
        raise SystemExit("external states: walk called with parameters not loaded through load_model/_jpar")
    return {"player_key": _KEYS["keys"][q_pid].astype(np.int64), "as_of_day": (q_day - 1).astype(np.int32),
            "ev_lo": np.full(n, -1 if ev_lo is None else ev_lo, np.int32), "ev_parity": np.full(n, par_, np.int8),
            "exclude_lo": np.full(n, xlo, np.int32), "exclude_hi": np.full(n, xhi, np.int32),
            "half": np.full(n, _HALF[id(par)], np.int8),
            "hero": (np.full(n, -1, np.int16) if q_hero is None else _KEYS["sh"][q_hero].astype(np.int16))}


def _merged_counts(aug, q_pid, q_day):
    """(nq, H) merged play counts (every row with day <= q_day - 1, no evidence filter), in d hero
    order: exactly the walk's n_all."""
    day = aug["day"]
    upid = np.unique(q_pid)
    pos = np.full(int(aug["n_players"]), -1, np.int64)
    pos[upid] = np.arange(len(upid))
    sel = np.flatnonzero(pos[aug["pid"]] >= 0)
    sday, sp, sh_ = day[sel], pos[aug["pid"][sel]], aug["hero"][sel]
    out = np.empty((len(q_pid), H), np.float32)
    N = np.zeros((len(upid), H), np.float32)
    o = np.argsort(q_day, kind="stable")
    ap = 0
    for t_ in np.unique(q_day):
        e = np.searchsorted(sday, t_ - 1, side="right")
        if e > ap:
            np.add.at(N, (sp[ap:e], sh_[ap:e]), 1.0)
            ap = e
        q = o[q_day[o] == t_]
        out[q] = N[pos[q_pid[q]]]
    return out


def _walk_external(qs, want_all, q_hero, aug=None, q_pid=None, q_day=None):
    name = _walk_name()
    z = np.load(os.path.join(WALK_REPLAY, name + ".npz"))
    for k_, v_ in qs.items():
        if k_ in z.files and not np.array_equal(z[k_], v_):
            raise SystemExit(f"external states {name}: query field {k_} differs from this run's queries")
    s, sd, n_ = z["s"].astype(np.float32), z["sd"].astype(np.float32), z["n"].astype(np.float32)
    # the external files are in shared.HEROES order; this module works in d order: d column i = shared column sh[i]
    s, sd, n_ = s[:, _KEYS["sh"]], sd[:, _KEYS["sh"]], n_[:, _KEYS["sh"]]
    if aug is not None:
        # guard: the external n must be the merged count before the as-of day, in shared order;
        # a wrong hero order or a windowed count fails here
        want = _merged_counts(aug, q_pid, q_day)
        if not np.array_equal(n_, want):
            frac = float((n_ == want).mean())
            raise SystemExit(f"external states {name}: n differs from the merged counts ({frac:.3f} of cells agree); "
                             "check the hero order (shared.HEROES) and that n ignores the evidence filters")
    res = {"n_all": n_, "n_slot": np.zeros_like(n_)}
    if want_all:
        res["mean_all"], res["var_all"] = s, sd ** 2
    else:
        hq = np.zeros(len(s), np.int64) if q_hero is None else q_hero
        res["mean"], res["var"] = s[np.arange(len(s)), hq], sd[np.arange(len(s)), hq] ** 2
    print(f"  external states {name}: {len(s):,} queries", flush=True)
    return res


def walk(aug, par, mets, mw, q_pid, q_day, ev_lo=None, q_hero=None, want_all=True, ev_ok=None):
    if WALK_RECORD or WALK_REPLAY:
        qs = _walk_queries(par, q_pid, q_day, ev_lo, q_hero, ev_ok)
        if WALK_REPLAY:
            return _walk_external(qs, want_all, q_hero, aug, q_pid, q_day)
        os.makedirs(WALK_RECORD, exist_ok=True)
        np.savez(os.path.join(WALK_RECORD, _walk_name() + ".npz"), **qs)
    return _walk(aug, par, mets, mw, q_pid, q_day, ev_lo, q_hero, want_all, ev_ok)


def _walk(aug, par, mets, mw, q_pid, q_day, ev_lo=None, q_hero=None, want_all=True, ev_ok=None):
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
            if ev_ok is not None:
                ev = ev[ev_ok(sday[ev])]
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
        if WALK_REPLAY:
            return np.zeros(np.shape(n_all), np.float64)     # the external s already holds the layer
        lo = np.searchsorted(self.key, q_pid.astype(np.int64) * 100000)
        hi = np.searchsorted(self.key, q_pid.astype(np.int64) * 100000 + (q_day.astype(np.int64) - 1), side="right")
        sc = np.exp(-self.lam * (q_day.astype(np.int64) - self.d0))
        Sd = (self.cwu[hi] - self.cwu[lo]) * sc
        Nd = (self.cw[hi] - self.cw[lo]) * sc
        rho = Sd / (Nd + self.m["kd"])
        nu = (self.ceu[hi] - self.ceu[lo]) / (self.ce[hi] - self.ce[lo] + self.m["ke"])
        return (self.m["drift_scale"] * rho)[:, None] + (self.m["learn_scale"] * nu)[:, None] * (n_all < 5)


def eval_lobbies():
    """Lobbies reserved for evaluation (tournament, tune, party study): the lobbies that fit a
    drafter component (probe, player-feature prior targets, opponent models) are drawn
    from the other rich tournament-window lobbies."""
    out = set()
    for sub in ("tourney", "tourney_tune", "tourney_party"):
        p = os.path.join(OUT, sub, "lobbies.npz")
        if os.path.exists(p):
            out |= set(np.load(p)["gi"].tolist())
    return out


def layer_meta():
    # rcf_full layout: p3_c_tests_main.json; definitive builds (results/def_<D>): p3_c_tests_def.json
    for name in ("p3_c_tests_main.json", "p3_c_tests_def.json"):
        path = os.path.join(C.RESULTS, name)
        if os.path.exists(path):
            break
    else:
        raise SystemExit(f"layer meta: neither p3_c_tests_main.json nor p3_c_tests_def.json in {C.RESULTS}")
    j = json.load(open(path))
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
    _HALF[id(par)] = 1 if tag.endswith("_flip") else 0
    _KEYS["keys"] = np.asarray(d["player_keys"])
    _KEYS["sh"] = np.asarray(to_shared(d["hero_names"]))    # d hero index -> shared.HEROES index
    if WALK_REPLAY:
        aug["table"] = np.zeros_like(aug["table"])         # the external s already holds the experience offset
        return par, d, aug, None, None
    mets, mw = R.aug_metrics(aug, par.meta.get("sb", "legacy"), d) if par.G is not None else (None, None)
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
    _HALF[id(p)] = 1 if tag.endswith("_flip") else 0
    print(f"judge parameters: {tag} (n_fit {p.meta.get('n_fit')})", flush=True)
    return p


def _slot_mmr(d):
    """The slot table's player_mmr (post-game, kept for reference only; the judge does not read
    it). p3_rcf.load_aug drops it since b169d46, so NaN when absent."""
    v = d.get("player_mmr")
    return np.full(len(d["pid"]), np.nan, np.float32) if v is None else v


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
             day=d["day"][q], mmr=_slot_mmr(d)[q])
    print(f"judgefit: {len(q):,} slots ({time.time() - t0:.0f}s)", flush=True)


# split-history robustness: the drafter and the judge see disjoint halves of
# every player's history (odd and even days), so both have years of evidence
SPLIT_D = lambda day: (day % 2) == 1                                    # drafter: odd days
SPLIT_J = lambda day: ((day % 2) == 0) & ((day < OOT_START) | (day > LOBBY_END))   # judge: even days, never the tournament window


def split_states(judge_tag="full_ms_flip"):
    """state_splitd (drafter, odd-day evidence before the lobby day, full_ms,
    no layer) and state_splitj (judge, even-day evidence outside the
    tournament window through J1, the other-half fit), tournament-window lobbies."""
    t0 = time.time()
    par, d, aug, mets, mw = load_model()
    jpar = _jpar(judge_tag)
    sh = to_shared(d["hero_names"])
    L = load_lobbies()
    gsel = np.flatnonzero(L["day"] <= LOBBY_END)
    rows = L["slot_rows"][gsel].ravel()
    qpid = d["pid"][rows].astype(np.int64)
    for v, pr, ok, qday in (("splitd", par, SPLIT_D, np.repeat(L["day"][gsel], 10).astype(np.int64)),
                            ("splitj", jpar, SPLIT_J, np.full(len(rows), J1 + 1, np.int64))):
        ukey, inv = np.unique(qpid * 100000 + qday, return_inverse=True)
        upid, uday = ukey // 100000, ukey % 100000
        res = walk(aug, pr, mets, mw, upid, uday, ev_ok=ok)
        s = offset(aug["table"], res["n_slot"]) + res["mean_all"]
        np.savez(os.path.join(OUT, f"state_{v}.npz"), gsel=gsel, s=s[inv][:, sh_inv(sh)].astype(np.float32),
                 sd=np.sqrt(res["var_all"])[inv][:, sh_inv(sh)].astype(np.float16),
                 n=res["n_all"][inv][:, sh_inv(sh)].astype(np.uint16),
                 off=off_role(d, res["n_all"])[inv][:, sh_inv(sh)], qday=qday.astype(np.int32))
        print(f"state {v}: {len(rows):,} slots ({time.time() - t0:.0f}s)", flush=True)


def split_judgefit(tag="full_ms_flip"):
    """judgefit with the judge's split evidence (even days outside the tournament window)."""
    t0 = time.time()
    par, d, aug, mets, mw = load_model(tag)
    sv = _vidx(d, SEALED)
    q = np.flatnonzero(d["post"] & (d["day"] >= J0) & (d["day"] <= J1) & (d["version"] != sv))
    res = walk(aug, par, mets, mw, d["pid"][q].astype(np.int64), d["day"][q].astype(np.int64),
               q_hero=d["hero"][q].astype(np.int64), want_all=False, ev_ok=SPLIT_J)
    hq = d["hero"][q]
    ns = res["n_slot"]
    nb = len(C.NPH_EDGES)
    bp = np.searchsorted(C.NP_EDGES, ns.sum(1), side="right") - 1
    bh = np.searchsorted(C.NPH_EDGES, ns[np.arange(len(q)), hq], side="right") - 1
    mu = aug["table"][bp * nb + bh]
    off = off_role(d, res["n_all"])[np.arange(len(q)), hq]
    np.savez(os.path.join(OUT, "judgefit_split.npz"), replay_id=d["replay_id"][q], g=d["g"][q], team=d["team"][q],
             hero=to_shared(d["hero_names"])[hq], pred=(mu + res["mean"]).astype(np.float32),
             var=res["var"].astype(np.float32), off=off, y=d["y"][q], wp=d["wp"][q].astype(np.float32),
             day=d["day"][q], mmr=_slot_mmr(d)[q])
    print(f"judgefit split: {len(q):,} slots ({time.time() - t0:.0f}s)", flush=True)


def prelobbies():
    """Snapshot-era lobbies with full draft orders (cache/drafts_post.json.gz,
    2026-02-11 to 05-22, before every tournament lobby) and their as-of
    player states, for training draft priors: pre_lobbies.npz (same fields as
    lobbies.npz, plus wp = the slot table's causal WP for team 0) and
    pre_state.npz (s, sd, n, off per slot in pick order, shared hero order)."""
    from shared import MAPS
    from overfit2026 import data as odata
    t0 = time.time()
    par, d, aug, mets, mw = load_model()
    sh = to_shared(d["hero_names"])
    names = [str(x) for x in d["hero_names"]]
    hidx = {n_: i for i, n_ in enumerate(names)}
    site = odata.site_tiers()
    pre = np.flatnonzero(~d["in_sample"] & (d["day"] <= C.day_of("2026-05-22")))
    key = {(int(d["replay_id"][i]), int(d["hero"][i])): int(i) for i in pre}
    dr = json.load(gzip.open(os.path.join(C.CACHE, "drafts_post.json.gz"), "rt"))
    keep, steps = [], []
    for rid, g in dr.items():
        o = g["order"]
        tier = site.get(int(rid), "unknown")
        if len(o) != 16 or tier == "unknown" or g["map"] not in MAPS:
            continue
        st, ok = [], True
        for h, ty, pn, sl in o:
            if h not in hidx:
                ok = False
                break
            if ty == 1:
                row = key.get((int(rid), hidx[h]))
                if row is None:
                    ok = False
                    break
                st.append((int(sh[hidx[h]]), 1, int(d["team"][row]), row))
            else:
                st.append((int(sh[hidx[h]]), 0, 0 if sl == 1 else 1, -1))
        if not ok or [x[1] for x in st] != [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]:
            continue
        ft = st[4][2]
        exp = [ft, 1 - ft, ft, 1 - ft, ft, 1 - ft, 1 - ft, ft, ft, 1 - ft, ft, 1 - ft, 1 - ft, ft, ft, 1 - ft]
        if [x[2] for x in st] != exp:
            continue
        keep.append((int(rid), g["map"], tier))
        steps.append(st)
    steps = np.array(steps, np.int64)
    prow = steps[:, :, 3][steps[:, :, 1] == 1].reshape(len(keep), 10)
    r0 = prow[:, 0]
    y0 = np.where(d["team"][r0] == 0, d["y"][r0], 1 - d["y"][r0])
    wp0 = np.where(d["team"][r0] == 0, d["wp"][r0], 1 - d["wp"][r0])
    np.savez(os.path.join(OUT, "pre_lobbies.npz"), replay_id=np.array([k_[0] for k_ in keep], np.int64),
             map=np.array([k_[1] for k_ in keep]), tier=np.array([k_[2] for k_ in keep]), steps=steps, slot_rows=prow,
             day=d["day"][r0], first_team=steps[:, 4, 2].astype(np.int8), winner=(1 - y0).astype(np.int8),
             wp=wp0.astype(np.float32))
    rows = prow.ravel()
    qpid, qday = d["pid"][rows].astype(np.int64), d["day"][rows].astype(np.int64)
    ukey, inv = np.unique(qpid * 100000 + qday, return_inverse=True)
    upid, uday = ukey // 100000, ukey % 100000
    res = walk(aug, par, mets, mw, upid, uday)
    lay = Layer(d, aug, layer_meta())
    s_ = offset(aug["table"], res["n_slot"]) + res["mean_all"] + lay.at(upid, uday, res["n_all"])
    si = sh_inv(sh)
    np.savez(os.path.join(OUT, "pre_state.npz"), s=s_[inv][:, si].astype(np.float16),
             sd=np.sqrt(res["var_all"])[inv][:, si].astype(np.float16), n=res["n_all"][inv][:, si].astype(np.uint16),
             off=off_role(d, res["n_all"])[inv][:, si])
    print(f"pre-lobbies: {len(keep):,} games ({time.time() - t0:.0f}s)", flush=True)


ROSTER_WIN = ("2026-03-29", "2026-05-22")


def rosters(n_max=20000):
    """Training rosters for self-play (no draft order, no outcome used):
    snapshot-era games 2026-03-29 to 05-22 (before every lobby) with all ten
    players known and 50+ earlier games each; player state as of the day
    before the game (offset + posterior + layer), pools = heroes played before.
    Map and tier come from the slim snapshot or backfill tables (site tiers)."""
    from shared import MAPS, SKILL_TIERS
    import p3t_value as PV
    t0 = time.time()
    par, d, aug, mets, mw = load_model()
    sh = to_shared(d["hero_names"])
    meta = PV.game_meta()
    lo, hi = (C.day_of(x) for x in ROSTER_WIN)
    q = np.flatnonzero((d["day"] >= lo) & (d["day"] <= hi))
    cnt = np.bincount(d["g"][q], minlength=int(d["g"].max()) + 1)
    q = q[cnt[d["g"][q]] == 10]
    q = q[np.isin(d["replay_id"][q], np.fromiter(meta.keys(), np.int64))]
    o = np.lexsort((d["team"][q], d["g"][q]))
    q = q[o].reshape(-1, 10)                      # 10 slots per game, team 0 first
    assert (d["team"][q[:, :5]] == 0).all() and (d["team"][q[:, 5:]] == 1).all()
    rng = np.random.RandomState(12)
    if len(q) > 3 * n_max:
        q = q[np.sort(rng.choice(len(q), 3 * n_max, replace=False))]
    flat = q.ravel()
    res = walk(aug, par, mets, mw, d["pid"][flat].astype(np.int64), d["day"][flat].astype(np.int64))
    rich = (res["n_all"].sum(1) >= 50).reshape(-1, 10).all(1)
    lay = Layer(d, aug, layer_meta())
    s = offset(aug["table"], res["n_slot"]) + res["mean_all"] + lay.at(d["pid"][flat].astype(np.int64),
                                                                        d["day"][flat].astype(np.int64), res["n_all"])
    inv = sh_inv(sh)
    keep = np.flatnonzero(rich)[:n_max]
    sel = (keep[:, None] * 10 + np.arange(10)[None]).ravel()
    tm = [meta[int(r)] for r in d["replay_id"][q[keep, 0]]]
    ok = np.array([t[1] in MAPS and t[0] in SKILL_TIERS for t in tm])
    keep, sel = keep[ok], (keep[ok][:, None] * 10 + np.arange(10)[None]).ravel()
    tm = [t for t, k in zip(tm, ok) if k]
    np.savez(os.path.join(OUT, "rosters.npz"),
             s=s[sel][:, inv].astype(np.float32).reshape(-1, 10, H),
             off=off_role(d, res["n_all"][sel])[:, inv].reshape(-1, 10, H),
             n=res["n_all"][sel][:, inv].astype(np.uint16).reshape(-1, 10, H),
             map=np.array([MAPS.index(t[1]) for t in tm], np.int32),
             tier=np.array([SKILL_TIERS.index(t[0]) for t in tm], np.int32),
             replay_id=d["replay_id"][q[keep, 0]])
    print(f"rosters: {len(keep):,} games ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["lobbies", "ranks", "states", "judgefit", "rosters", "split_states", "split_judgefit", "prelobbies"])
    ap.add_argument("--vintages", default="cur,s4m,s1y,judge")
    ap.add_argument("--judge-params", default="full_ms_flip")
    a = ap.parse_args()
    if a.stage == "lobbies":
        import p3_x_common as X
        build_lobbies(X.load_ext())
    elif a.stage == "ranks":
        ranks()
    elif a.stage == "states":
        states(a.vintages.split(","), a.judge_params)
    elif a.stage == "rosters":
        rosters()
    elif a.stage == "prelobbies":
        prelobbies()
    elif a.stage == "split_states":
        split_states(a.judge_params)
    elif a.stage == "split_judgefit":
        split_judgefit(a.judge_params)
    else:
        judgefit(a.judge_params)


if __name__ == "__main__":
    main()
