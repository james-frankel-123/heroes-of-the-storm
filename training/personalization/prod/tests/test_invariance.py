"""
Fixture tests for the P3 RESEARCH feature code (second review, section 5,
"fast regression checks"). No database, no cache files: every input is a
tiny synthetic replay fixture built here, and the few functions that read a
cache file are pointed at a temporary directory.

  1  future-deletion invariance: deleting or changing every game after day t
     leaves every feature of games on or before t unchanged
     (p3_fix_counts.lag1_counts, recency_features_l1, p3_hs_core.online_state,
     p3_hero_level_causal latest_before / running_max_before / account_status,
     and both history walkers)
  2  late arrival: a delayed parse time changes hero_level_causal only from
     the new parse time on
  3  walkers (p3_pgd_feats._walk, p3_x_ban_feat._walk): a game that ended after
     the queried game's start never counts, checked against a brute-force
     reference, including overlapping timestamps of one player's games
  4  draft harness (p3_dr_drafter team mode): permuting the realized pick order
     within a team, or which teammate played which hero, leaves _assign_terms,
     the team pools and a whole stubbed draft unchanged; p3_assign with all-zero
     personal inputs gives zero personal terms and V equals the population WP
  5  hero set: a 91-hero list fails p3_heroes.check_heroes

Most research modules import numba (and the drafter imports torch). Missing
packages skip the tests that need them and the skips are printed.
Run: python3 training/personalization/prod/tests/test_invariance.py
     (/usr/bin/python3 on the main box has numba and torch)
"""
import gzip
import importlib.util
import json
import os
import sys
import tempfile
import traceback

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")

P3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, P3)
import numpy as np

HAVE = {m: importlib.util.find_spec(m) is not None for m in ("numba", "torch")}
DAY = 86400
DAY0 = 19800
HEROES_USED = np.array([0, 3, 7, 12, 25, 40, 61, 89])
TMP = tempfile.mkdtemp(prefix="p3_invariance_")


# ------------------------------------------------------------------ fixture

def fixture(seed=0, n_games=160, n_players=8, days=30):
    """Slot table in the hs_slots layout, sorted by (day, replay_id), plus
    per-slot times. Replay ids are NOT in day order (upload order differs from
    play order); several games share a day; some lengths are missing (0); some
    parse times are missing (-1) or late by days. Player 0 gets three
    overlapping games on day DAY0 + 15 (bad timestamps)."""
    rng = np.random.RandomState(seed)
    rows = []
    rid_pool = rng.permutation(np.arange(1000, 1000 + n_games + 3))

    def add_game(g, day, end, gl, players):
        y = float(rng.rand() < 0.5)
        wp = rng.uniform(0.3, 0.7)
        for k, p in enumerate(players):
            team = k // 2
            parse = end + int(rng.exponential(3600)) + (rng.randint(1, 6) * DAY if rng.rand() < 0.15 else 0)
            if rng.rand() < 0.05:
                parse = -1
            lvl = rng.choice([np.nan, 2.0, 5.0, 9.0, 30.0])
            rows.append(dict(replay_id=int(rid_pool[g]), pid=int(p), hero=int(rng.choice(HEROES_USED)),
                             team=team, day=day, end=end, gl=gl, y=y if team == 0 else 1 - y,
                             wp=wp if team == 0 else 1 - wp, parse=parse, lvl=lvl))

    for g in range(n_games):
        day = DAY0 + rng.randint(days)
        end = day * DAY + rng.randint(3600, DAY)
        add_game(g, day, end, int(rng.choice([0, 600, 1200, 1800])),
                 rng.choice(np.arange(1, n_players), 3, replace=False).tolist() + [0] * (rng.rand() < 0.4))
    base = (DAY0 + 15) * DAY + 36000
    # ends 8500 < 10000 < 10500; the third game starts (8000) before the first ends
    for j, (e, gl) in enumerate([(8500, 1000), (10000, 1000), (10500, 2500)]):
        add_game(n_games + j, DAY0 + 15, base + e, gl, [0, 1 + j])
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    o = np.lexsort((a["pid"], a["replay_id"], a["day"]))
    a = {k: v[o] for k, v in a.items()}
    d = {"replay_id": a["replay_id"].astype(np.int64), "pid": a["pid"].astype(np.int64),
         "hero": a["hero"].astype(np.int64), "team": a["team"].astype(np.int8),
         "day": a["day"].astype(np.int32), "y": a["y"].astype(np.float32), "wp": a["wp"].astype(np.float64),
         "n_players": np.int64(n_players), "in_sample": np.ones(len(o), bool)}
    d["r"] = d["y"] - d["wp"]
    d["v"] = d["wp"] * (1 - d["wp"])
    gl = np.where(a["gl"] > 0, a["gl"], 1200).astype(np.float64)
    t = {"end": a["end"].astype(np.float64), "gl": a["gl"].astype(np.int64), "start": a["end"] - gl,
         "parse": a["parse"].astype(np.int64), "lvl": a["lvl"].astype(np.float64)}
    return d, t


def cut(d, t, keep):
    """Rows where keep, n_players unchanged."""
    dd = {k: (v[keep] if np.ndim(v) else v) for k, v in d.items()}
    tt = {k: v[keep] for k, v in t.items()}
    return dd, tt


def changed_future(d, t, day_t, seed=5):
    """Same rows; every game after day_t gets new heroes, outcomes, WP, levels."""
    rng = np.random.RandomState(seed)
    f = d["day"] > day_t
    d2 = {k: (v.copy() if np.ndim(v) else v) for k, v in d.items()}
    t2 = {k: v.copy() for k, v in t.items()}
    d2["hero"][f] = rng.choice(HEROES_USED, f.sum())
    d2["y"][f] = 1 - d2["y"][f]
    d2["wp"][f] = rng.uniform(0.2, 0.8, f.sum())
    d2["r"] = d2["y"] - d2["wp"]
    d2["v"] = d2["wp"] * (1 - d2["wp"])
    t2["lvl"][f] = rng.choice([1.0, 80.0], f.sum())
    return d2, t2


def write_cache(cache, d, t):
    """gametime_2024q2.npz and an empty x_post_games.json.gz for the walkers."""
    os.makedirs(cache, exist_ok=True)
    u, i = np.unique(d["replay_id"], return_index=True)
    np.savez(os.path.join(cache, "gametime_2024q2.npz"), replay_ids=u, ts=t["end"][i].astype(np.int64),
             game_length=t["gl"][i])
    with gzip.open(os.path.join(cache, "x_post_games.json.gz"), "wt") as f:
        json.dump([], f)


def eq(a, b, what):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), \
        f"{what}: {int((~((a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == 'f' else (a == b))).sum())} entries differ"


DAY_T = DAY0 + 17


def variants():
    d, t = fixture()
    past = d["day"] <= DAY_T
    assert past.sum() < len(past) and past.any()
    return d, t, past, [("deleted", *cut(d, t, past)), ("changed", *changed_future(d, t, DAY_T))]


# ------------------------------------------------------------------ 1. future invariance

def test_lag1_counts():
    import p3_fix_counts as F
    d, t, past, alts = variants()
    a0, b0 = F.lag1_counts(d)
    # brute force on the fixture
    for i in range(len(a0)):
        m = (d["pid"] == d["pid"][i]) & (d["day"] < d["day"][i])
        assert a0[i] == m.sum() and b0[i] == (m & (d["hero"] == d["hero"][i])).sum()
    for name, d2, _ in alts:
        a, b = F.lag1_counts(d2)
        n = past.sum()
        eq(a[:n], a0[:n], f"lag1 n_p ({name})")
        eq(b[:n], b0[:n], f"lag1 n_ph ({name})")


def test_recency_l1():
    import p3_fix_counts as F
    import p3_hs_core as C
    d, t, past, alts = variants()
    old = C.CACHE
    try:
        C.CACHE = os.path.join(TMP, "rec")
        write_cache(C.CACHE, d, t)
        q = np.flatnonzero(past)
        ref = F.recency_features_l1(d, q)
        for name, d2, _ in alts:
            q2 = np.flatnonzero(d2["day"] <= DAY_T)
            got = F.recency_features_l1(d2, q2)
            for k, a, b in zip(("e20", "e100", "last"), ref, got):
                eq(b, a, f"recency {k} ({name})")
    finally:
        C.CACHE = old


def test_online_state():
    import p3_hs_core as C
    from p3_heroes import NUM_HEROES
    d, t, past, alts = variants()
    for hl in (None, 30.0):
        ref = C.online_state(d, np.ones(len(d["pid"]), bool), past, NUM_HEROES, d["r"], half_life=hl, lag=1)
        for name, d2, _ in alts:
            q2 = d2["day"] <= DAY_T
            got = C.online_state(d2, np.ones(len(d2["pid"]), bool), q2, NUM_HEROES, d2["r"], half_life=hl, lag=1)
            for k, a, b in zip(("qidx", "S", "P", "Nh", "Np", "Wh", "Rh"), ref, got):
                eq(b, a, f"online_state {k} hl={hl} ({name})")


def _hl_inputs(d, t):
    from p3_heroes import HKEY
    grp = d["pid"] * HKEY + d["hero"]
    return grp, t["parse"], t["lvl"], grp, t["start"].astype(np.int64)


def brute_latest(g, ts, v, qg, qt):
    out = []
    for a, b in zip(qg, qt):
        m = (g == a) & (ts >= 0) & ~np.isnan(v) & (ts < b)
        if not m.any():
            out.append({np.nan})
        else:
            out.append(set(v[m & (ts == ts[m].max())].tolist()))
    return out


def brute_runmax(g, ts, v, qg, qt):
    out = []
    for a, b in zip(qg, qt):
        m = (g == a) & (ts >= 0) & ~np.isnan(v) & (ts < b)
        out.append(v[m].max() if m.any() else np.nan)
    return np.array(out)


def test_hero_level_causal_future():
    import p3_hero_level_causal as L
    d, t, past, alts = variants()
    g, ts, v, qg, qt = _hl_inputs(d, t)
    lb = L.latest_before(g, ts, v, qg, qt)
    rm = L.running_max_before(d["pid"], ts, v, d["pid"], qt)
    # brute force on the full fixture (ties: any tied stamp is acceptable)
    for x, s in zip(lb, brute_latest(g, ts, v, qg, qt)):
        assert (np.isnan(x) and any(np.isnan(list(s)))) or x in s, (x, s)
    eq(rm, brute_runmax(d["pid"], ts, v, d["pid"], qt), "running_max_before vs brute force")
    st = L.account_status(d, {"acct_lo": rm, "acct_hi": rm}, late_day=DAY0 + 10)
    n = past.sum()
    for name, d2, t2 in alts:
        g2, ts2, v2, qg2, qt2 = _hl_inputs(d2, t2)
        eq(L.latest_before(g2, ts2, v2, qg2[:n], qt2[:n]), lb[:n], f"latest_before ({name})")
        rm2 = L.running_max_before(d2["pid"], ts2, v2, d2["pid"], qt2)
        eq(rm2[:n], rm[:n], f"running_max_before ({name})")
        st2 = L.account_status(d2, {"acct_lo": rm2, "acct_hi": rm2}, late_day=DAY0 + 10)
        eq(st2[:n], st[:n], f"account_status ({name})")


# ------------------------------------------------------------------ 2. late arrival

def test_hero_level_late_arrival():
    import p3_hero_level_causal as L
    d, t = fixture(seed=3)
    g, ts, v, qg, qt = _hl_inputs(d, t)
    ok = (ts >= 0) & ~np.isnan(v)
    # a source row with later queries in its group
    cand = [i for i in np.flatnonzero(ok) if ((qg == g[i]) & (qt > ts[i])).sum() >= 2]
    assert cand, "fixture has no stamp with later queries"
    i = cand[0]
    p0, p1 = ts[i], ts[i] + 3 * DAY
    ts_late = ts.copy()
    ts_late[i] = p1
    ts_gone = ts.copy()
    ts_gone[i] = -1  # the stamp never arrives
    for fn, grp, qgrp in ((L.latest_before, g, qg), (L.running_max_before, d["pid"], d["pid"])):
        base = fn(grp, ts, v, qgrp, qt)
        late = fn(grp, ts_late, v, qgrp, qt)
        gone = fn(grp, ts_gone, v, qgrp, qt)
        before = qt <= p0
        between = (qt > p0) & (qt <= p1)
        eq(late[before], base[before], f"{fn.__name__}: queries before the old parse time")
        eq(late[between], gone[between], f"{fn.__name__}: stamp must be invisible until its new parse time")
        if fn is L.running_max_before:
            after = qt > p1
            eq(late[after], base[after], "running_max_before: after the new parse time the stamp counts again")
    # and the change is real for this group (between the times, the stamp was in use before)
    m = (qg == g[i]) & (qt > p0) & (qt <= p1)
    if m.any():
        b0 = L.latest_before(g, ts, v, qg[m], qt[m])
        assert np.any(b0 == v[i]), "fixture: the delayed stamp was not the latest one in the window"


# ------------------------------------------------------------------ 3. walkers

def _walker_brute(d, t_end, t_start, rows):
    """Per query row: games of the same player (other rows) that ended at or
    before the query's start, counted per hero; days since last such game."""
    from p3_heroes import NUM_HEROES
    n = np.zeros((len(rows), NUM_HEROES))
    days = np.full((len(rows), NUM_HEROES), -1.0)
    for k, i in enumerate(rows):
        m = (d["pid"] == d["pid"][i]) & (t_end <= t_start[i])
        m[i] = False
        for h in np.unique(d["hero"][m]):
            mh = m & (d["hero"] == h)
            n[k, h] = mh.sum()
            days[k, h] = (t_start[i] - t_end[mh].max()) / DAY
    return n, days


def _run_walkers(d, t):
    import p3_hs_core as C
    import p3_pgd_feats as PF
    import p3_x_ban_feat as B
    oldc, oldg = C.CACHE, B.GAMETIME
    try:
        C.CACHE = os.path.join(TMP, "walk%d" % len(d["pid"]))
        write_cache(C.CACHE, d, t)
        B.GAMETIME = os.path.join(C.CACHE, "gametime_2024q2.npz")
        q = np.arange(len(d["pid"]))
        pg = PF.walk(d, q)
        bf = B.compute(d, q)
        t_end, t_start = PF.play_end_times(d)
        bt_end, bt_start = B.play_times(d)
    finally:
        C.CACHE, B.GAMETIME = oldc, oldg
    eq(t_end, bt_end, "walker end times agree")
    eq(t_start, bt_start, "walker start times agree")
    return pg, bf, t_end, t_start


def test_walkers_visibility():
    d, t = fixture(seed=1)
    pg, bf, t_end, t_start = _run_walkers(d, t)
    rows = np.arange(len(d["pid"]))
    n_ref, days_ref = _walker_brute(d, t_end, t_start, rows)
    bad_pg = np.flatnonzero((pg["n"] != n_ref).any(1))
    bad_b = np.flatnonzero((bf["q_cnt"] != n_ref).any(1))
    msg = []
    for name, bad in (("p3_pgd_feats._walk", bad_pg), ("p3_x_ban_feat._walk", bad_b)):
        if len(bad):
            i = bad[0]
            msg.append(f"{name}: {len(bad)} query rows count games that ended after the query started "
                       f"(e.g. replay {d['replay_id'][i]}, player {d['pid'][i]}, start {t_start[i]:.0f}; "
                       f"walker n={int((pg['n'] if 'pgd' in name else bf['q_cnt'])[i].sum())}, "
                       f"reference n={int(n_ref[i].sum())})")
    assert not msg, "; ".join(msg)
    assert np.allclose(np.where(n_ref > 0, pg["days"], -1), days_ref, atol=1e-3)


def test_walkers_future():
    d, t, past, alts = variants()
    n = past.sum()
    pg0, bf0, _, _ = _run_walkers(d, t)
    for name, d2, t2 in alts:
        pg, bf, _, _ = _run_walkers(d2, t2)
        for k in pg0:
            eq(pg[k][:n], pg0[k][:n], f"pgd walker {k} ({name})")
        for k in bf0:
            eq(bf[k][:n], bf0[k][:n], f"ban walker {k} ({name})")


# ------------------------------------------------------------------ 4. draft harness

def _lob(seed=0, s_scale=0.05):
    from p3_assign import DRAFT_TEAM, IS_PICK
    from p3_heroes import NUM_HEROES
    rng = np.random.RandomState(seed)
    heroes = rng.choice(NUM_HEROES, 16, replace=False)
    rows = {0: [10, 11, 12, 13, 14], 1: [20, 21, 22, 23, 24]}
    nxt = {0: 0, 1: 0}
    steps = []
    for k in range(16):
        tm = DRAFT_TEAM[k]
        if IS_PICK[k]:
            steps.append((int(heroes[k]), 1, tm, rows[tm][nxt[tm]]))
            nxt[tm] += 1
        else:
            steps.append((int(heroes[k]), 0, tm, -1))
    allrows = rows[0] + rows[1]
    T = {"pos": {r: i for i, r in enumerate(allrows)},
         "s": rng.normal(0, s_scale, (10, NUM_HEROES)), "off": rng.normal(0, s_scale, (10, NUM_HEROES)),
         "pool": rng.rand(10, NUM_HEROES) < 0.3}
    ipers = {r: rng.normal(0, 1, NUM_HEROES).astype(np.float32) for r in allrows}
    import shared
    L = {"steps": [np.array(steps)], "map": np.array([shared.MAPS[0]]), "tier": np.array(["mid"]),
         "g": np.array([0])}
    return L, T, ipers


def _permute_team(L, mode, seed=1):
    """mode 'order': permute which player picks at each of a team's pick steps
    (heroes stay at their steps, so who played what changes too);
    'heroes': permute the heroes among a team's pick steps (players stay)."""
    rng = np.random.RandomState(seed)
    st = L["steps"][0].copy()
    for tm in (0, 1):
        idx = np.flatnonzero((st[:, 1] == 1) & (st[:, 2] == tm))
        col = 3 if mode == "order" else 0
        st[idx, col] = st[idx[rng.permutation(len(idx))], col]
    return dict(L, steps=[st])


def test_drafter_team_invariance():
    import p3_dr_drafter as DR
    L, T, ipers = _lob()
    lob0 = DR.lobby_payload(L, T, 0, ipers, 0, np.array([0]))
    rng = np.random.RandomState(9)
    teams = [[list(rng.choice(90, 5, replace=False)) for _ in range(64)] for _ in (0, 1)]
    b = np.array([0.1, 1.0, 3.7, 1.2])
    for mode in ("order", "heroes"):
        lob1 = DR.lobby_payload(_permute_team(L, mode), T, 0, ipers, 0, np.array([0]))
        for tm in (0, 1):
            assert sorted(lob1["team_rows"][tm]) == sorted(lob0["team_rows"][tm])
            eq(lob1["team_pool"][tm], lob0["team_pool"][tm], f"team_pool t{tm} ({mode})")
            assert np.allclose(lob1["ipers_team"][tm], lob0["ipers_team"][tm], atol=1e-5), mode
            S0, O0 = DR._assign_terms(teams[tm], lob0, tm, b)
            S1, O1 = DR._assign_terms(teams[tm], lob1, tm, b)
            assert np.allclose(S0, S1, atol=1e-12) and np.allclose(O0, O1, atol=1e-12), (mode, tm)
            # any hero order within a team gives the same terms
            S2, O2 = DR._assign_terms([list(np.random.RandomState(i).permutation(h)) for i, h in enumerate(teams[tm])],
                                      lob0, tm, b)
            assert np.allclose(S0, S2, atol=1e-12) and np.allclose(O0, O2, atol=1e-12)


class _StubGD:
    """Deterministic GD stand-in: fixed hero logits plus a small dependence on
    the team-0 multiset; log-softmax over available heroes."""

    def __init__(self, H):
        r = np.random.RandomState(4)
        self.a, self.c = r.normal(0, 1, H), r.normal(0, 0.3, H)

    def logprobs(self, t0, t1, bans, mo, to, step, is_pick, avail):
        z = self.a[None, :] + (t0.sum(1, keepdims=True) - t1.sum(1, keepdims=True)) * self.c[None, :]
        z = np.where(avail, z, -1e9)
        z = z - z.max(1, keepdims=True)
        return z - np.log(np.exp(z).sum(1, keepdims=True))


class _StubWP:
    def __init__(self, H):
        self.w = np.random.RandomState(6).normal(0, 0.1, H)

    def wp(self, drafts, bidx, game_map, tier):
        return np.array([1 / (1 + np.exp(-(self.w[list(a)].sum() - self.w[list(c)].sum()))) for a, c in drafts])


def test_drafter_whole_draft_invariance():
    """A full stubbed team-mode draft (personalized team 0 vs GD) gives the same
    picks and values when the realized pick order within each team is permuted."""
    import p3_dr_drafter as DR
    from p3_heroes import NUM_HEROES
    L, T, ipers = _lob(seed=2)
    DR._W.update(gd=_StubGD(NUM_HEROES), wp=_StubWP(NUM_HEROES), b=np.array([0.0, 1.0, 3.7, 1.2]), iw0=1.0)
    old_r = DR.R
    DR.R = 2
    try:
        out = {}
        for mode in (None, "order", "heroes"):
            lob = DR.lobby_payload(L if mode is None else _permute_team(L, mode), T, 0, ipers, 0, np.array([0]))
            r = DR._run_draft(lob, 123, {0: "pers", 1: "gd"}, "gd")
            out[mode] = (sorted(r["t0"].values()), sorted(r["t1"].values()), r["V"],
                         [(int(x["pers"]), int(x["pop"])) for x in r["decisions"]])
        for mode in ("order", "heroes"):
            a, b = out[None], out[mode]
            assert a[0] == b[0] and a[1] == b[1] and a[3] == b[3], (mode, a, b)
            assert abs(a[2] - b[2]) < 1e-9
    finally:
        DR.R = old_r


def test_assign_zero_personal():
    """numpy-only: p3_assign with zero personal inputs gives zero terms."""
    import p3_assign as A
    from p3_heroes import NUM_HEROES
    z = np.zeros((10, NUM_HEROES))
    rng = np.random.RandomState(0)
    acts = rng.choice(NUM_HEROES, 16, replace=False)
    S, O, slots = A.assignment_terms(z, z, acts, 3.7, 1.2)
    assert S == [0.0, 0.0] and O == [0.0, 0.0]
    for n in range(6):
        s, o, perm = A.best_assignment(z[:5], z[:5], acts[:n], 3.7, 1.2)
        assert s == 0.0 and o == 0.0 and perm == (0, 1, 2, 3, 4)


def test_assign_parity_and_zero_drafter():
    """p3_assign.best_assignment and p3_dr_drafter._assign_terms agree on full
    teams; with zero personal inputs the drafter's V is the population WP."""
    import p3_assign as A
    import p3_dr_drafter as DR
    from p3_heroes import NUM_HEROES
    L, T, ipers = _lob(seed=3)
    lob = DR.lobby_payload(L, T, 0, ipers, 0, np.array([0]))
    b = np.array([0.0, 1.0, 3.7, 1.2])
    rng = np.random.RandomState(1)
    for tm in (0, 1):
        rows = lob["team_rows"][tm]
        s5 = np.stack([lob["s"][r] for r in rows])
        o5 = np.stack([lob["off"][r] for r in rows])
        for _ in range(20):
            h = list(rng.choice(NUM_HEROES, 5, replace=False))
            S, O = DR._assign_terms([h], lob, tm, b)
            s, o, _ = A.best_assignment(s5, o5, h, b[2], b[3])
            assert np.isclose(S[0], s, atol=1e-12) and np.isclose(O[0], o, atol=1e-12)
    zl = dict(lob, s={r: np.zeros(NUM_HEROES) for r in lob["s"]}, off={r: np.zeros(NUM_HEROES) for r in lob["off"]})
    DR._W.update(wp=_StubWP(NUM_HEROES))
    finals = [({10 + i: int(x) for i, x in enumerate(rng.choice(45, 5, replace=False))},
               {20 + i: int(x) for i, x in enumerate(45 + rng.choice(45, 5, replace=False))}) for _ in range(16)]
    wp, V = DR._value(finals, zl, b)
    assert np.allclose(V, wp, atol=1e-6), np.abs(V - wp).max()


# ------------------------------------------------------------------ 5. hero set

def test_hero_set():
    import p3_heroes as H
    import shared
    H.check_heroes(shared.HEROES)
    for bad in (list(shared.HEROES) + ["Xal'atath"], list(shared.HEROES[:-1]) + ["Xal'atath"]):
        try:
            H.check_heroes(bad)
            raise RuntimeError(f"check_heroes accepted a {len(bad)}-hero list")
        except AssertionError:
            pass
    try:
        H.check_heroes(shared.HEROES, np.array([0, 90]))
        raise RuntimeError("check_heroes accepted hero index 90")
    except AssertionError:
        pass


# ------------------------------------------------------------------ runner

TESTS = [
    ("1 lag1_counts future invariance", test_lag1_counts, ("numba",)),
    ("1 recency_features_l1 future invariance", test_recency_l1, ("numba",)),
    ("1 online_state future invariance", test_online_state, ("numba",)),
    ("1 hero_level_causal future invariance", test_hero_level_causal_future, ("numba",)),
    ("1 walkers future invariance", test_walkers_future, ("numba",)),
    ("2 hero_level_causal late arrival", test_hero_level_late_arrival, ("numba",)),
    ("3 walkers: no game that ended after the query started", test_walkers_visibility, ("numba",)),
    ("4 drafter team mode: pools and _assign_terms", test_drafter_team_invariance, ("numba", "torch")),
    ("4 drafter team mode: whole stubbed draft", test_drafter_whole_draft_invariance, ("numba", "torch")),
    ("4 p3_assign zero personal inputs", test_assign_zero_personal, ()),
    ("4 assign parity, zero inputs give population WP", test_assign_parity_and_zero_drafter, ("numba", "torch")),
    ("5 hero set", test_hero_set, ()),
]


def main():
    passed, failed, skipped = [], [], []
    for name, fn, needs in TESTS:
        miss = [m for m in needs if not HAVE[m]]
        if miss:
            skipped.append((name, miss))
            continue
        try:
            fn()
            passed.append(name)
            print(f"PASS  {name}", flush=True)
        except Exception as e:  # report every failure, keep going
            failed.append(name)
            print(f"FAIL  {name}: {e}", flush=True)
            if os.environ.get("P3_TEST_TRACE"):
                traceback.print_exc()
    for name, miss in skipped:
        print(f"SKIP  {name} (needs {', '.join(miss)})")
    print(f"test_invariance: {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
