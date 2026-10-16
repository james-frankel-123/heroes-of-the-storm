"""Arm A2 (p3_c_merged): end-to-end leak fixtures through online_predict on a
50-player fixture (a slot table plus a merged store).
  M1 planted future HP rows (on the scored game's day, before its start, and
     a day later) change no prediction of that game or of any earlier game;
     a leaky reader (lag 0) is caught
  M2 own label: the scored game's own HP copy, timed 10 minutes before the
     game, never enters (excluded by replay_id); flipping its outcome changes
     nothing; a leaky reader (no slot exclusion, lag 0) is caught
  M3 rule "seen": a match played 30 days before but uploaded after the game
     equals deleting it; rule "game" counts it
  M4 counts in the state equal a brute-force count of slot plus extra rows on
     earlier days; extra rows never repeat a slot row
  M5 without extras the A2 code path equals the A1 code path (p3_x_oot);
     with exp_counts="slots" the offset equals it too
  M6 clock guard delays rows from 2026-07-09 on by the Eastern offset only
  M7 wp_proxy reads only in-sample slot rows
Run: python3 personalization/tests/test_merged.py (or pytest)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import p3_full_history as FH
import p3_c_merged as M
import p3_hs_core as C
import p3_x_common as X

DAY = 86400
D0 = C.day_of("2024-03-01")
N_PL = 50


def fixture(seed=0):
    """Slot table (sorted by day, replay_id) and the store rows: slot rows
    (src 3), HP-only rows (src 2) and corpus-only non-slot rows (src 1)."""
    rng = np.random.default_rng(seed)
    hero_names = FH.hero_index()[1]
    H = len(hero_names)
    keys = np.sort(rng.choice(np.arange(1, 10 ** 6), N_PL, replace=False) + (1 << 40))
    ns = 3000
    pid = rng.integers(0, N_PL, ns)
    fav = rng.integers(0, H, N_PL)
    hero = np.where(rng.random(ns) < 0.5, fav[pid], rng.integers(0, H, ns))
    day = D0 + rng.integers(0, 300, ns)
    rid = 5_000_000 + rng.permutation(ns)
    s = np.lexsort((rid, day))
    pid, hero, day, rid = pid[s], hero[s], day[s], rid[s]
    wp = np.clip(0.5 + 0.1 * rng.standard_normal(ns), 0.2, 0.8)
    y = (rng.random(ns) < wp).astype(np.float32)
    d = {"pid": pid.astype(np.int64), "hero": hero.astype(np.int64), "day": day.astype(np.int32),
         "replay_id": rid.astype(np.int64), "y": y, "wp": wp, "r": y - wp, "v": wp * (1 - wp),
         "in_sample": day < D0 + 200, "n_players": np.int64(N_PL), "player_keys": keys,
         "hero_names": np.array(hero_names)}
    # store: slot rows, HP-only rows (older and interleaved), corpus-only rows
    nh = 4000
    hp_pid = rng.integers(0, N_PL, nh)
    hp_hero = np.where(rng.random(nh) < 0.5, fav[hp_pid], rng.integers(0, H, nh))
    hp_end = (D0 - 400 + rng.integers(0, 700, nh)) * DAY + rng.integers(0, DAY, nh)
    hp_rid = 1_000_000 + np.arange(nh)
    nc = 300
    c_pid = rng.integers(0, N_PL, nc)
    st_rows = {
        "key": np.r_[keys[pid], keys[hp_pid], keys[c_pid]],
        "replay_id": np.r_[rid, hp_rid, 9_000_000 + np.arange(nc)],
        "end_ts": np.r_[day.astype(np.int64) * DAY + rng.integers(0, DAY, ns), hp_end,
                        (D0 + rng.integers(0, 300, nc)) * DAY],
        "hero": np.r_[hero, hp_hero, rng.integers(0, H, nc)],
        "win": np.r_[y.astype(np.int8), rng.integers(0, 2, nh), rng.integers(0, 2, nc)],
        "src": np.r_[np.full(ns, 3), np.full(nh, 2), np.full(nc, 1)],
    }
    st_rows["seen_ts"] = st_rows["end_ts"] + np.where(rng.random(len(st_rows["end_ts"])) < 0.1,
                                                      rng.integers(1, 60, len(st_rows["end_ts"])) * DAY, 0)
    rng2 = np.random.default_rng(seed + 100)
    A = 0.02 * rng2.standard_normal((H, 4))
    K = 0.0016 * np.ones((H, H)) + 0.0009 * np.eye(H) + A @ A.T
    return d, st_rows, K


def store_of(rows):
    return FH.Store.from_rows(**rows)


def rows_of(st, d, exclude=True):
    if exclude:
        return M.extra_rows(st, d["player_keys"], d["pid"], d["replay_id"], d["hero_names"])
    return M.extra_rows(st, d["player_keys"], np.zeros(0, np.int64), np.zeros(0, np.int64), d["hero_names"])


def run(d, ex, K, q, rule="game", lag=1, table=None):
    return M.merged_predict(d, ex, K, q, rule=rule, lag=lag, table=table)


FEATS = ("pred", "mean", "var", "nh", "np", "rec")


def same_on(a, b, rows):
    return all(np.allclose(a[k][rows], b[k][rows], rtol=0, atol=1e-12) for k in FEATS)


def add_rows(rows, **kw):
    return {k: np.r_[rows[k], np.asarray(kw[k])] for k in rows}


N_PROBES = 25


def probes(d, seed):
    rng = np.random.default_rng(seed)
    late = np.flatnonzero(d["day"] > D0 + 150)
    return rng.choice(late, N_PROBES, replace=False)


def test_m1_future_rows():
    d, rows, K = fixture(1)
    q = np.ones(len(d["pid"]), bool)
    base_st = store_of(rows)
    ex0 = rows_of(base_st, d)
    tab = run(d, ex0, K, q)["table"]           # one table so only the state can move
    base = run(d, ex0, K, q, table=tab)
    lbase = run(d, ex0, K, q, lag=0, table=tab)
    caught = 0
    for i in probes(d, 2):
        k = d["player_keys"][d["pid"][i]]
        t = int(d["day"][i]) * DAY
        r2 = add_rows(rows, key=[k, k], replay_id=[7_000_001, 7_000_002], end_ts=[t + 60, t + DAY + 60],
                      hero=[d["hero"][i]] * 2, win=[1, 1], src=[2, 2], seen_ts=[t + 60, t + DAY + 60])
        ex2 = rows_of(store_of(r2), d)
        got = run(d, ex2, K, q, table=tab)
        before = np.flatnonzero(d["day"] <= d["day"][i])
        assert same_on(got, base, before), i
        lk = run(d, ex2, K, q, lag=0, table=tab)
        caught += not same_on(lk, lbase, [i])
    assert caught == N_PROBES, f"leaky reader caught on {caught} of {N_PROBES}"


def test_m2_own_label():
    d, rows, K = fixture(3)
    q = np.ones(len(d["pid"]), bool)
    tab = run(d, rows_of(store_of(rows), d), K, q)["table"]
    caught = 0
    nslot = len(d["pid"])
    for i in probes(d, 4):
        a = {k: v.copy() for k, v in rows.items()}
        # the own row is store row i (slot rows come first): time it 10 min before the game
        t = int(d["day"][i]) * DAY + 3600
        a["end_ts"][i] = t - 600
        a["seen_ts"][i] = t - 600
        b = {k: v.copy() for k, v in a.items()}
        b["win"][i] = 1 - b["win"][i]
        sa, sb = store_of(a), store_of(b)
        ea, eb = rows_of(sa, d), rows_of(sb, d)
        assert not np.isin(d["replay_id"][i], ea["replay_id"])
        pa, pb = run(d, ea, K, q, table=tab), run(d, eb, K, q, table=tab)
        assert same_on(pa, pb, np.flatnonzero(d["day"] <= d["day"][i])), i
        # leaky: no slot exclusion and lag 0 (the own HP copy counts on its day)
        la = run(d, rows_of(sa, d, exclude=False), K, q, lag=0, table=tab)
        lb = run(d, rows_of(sb, d, exclude=False), K, q, lag=0, table=tab)
        caught += not same_on(la, lb, [i])
    assert caught == N_PROBES and nslot == len(d["pid"])


def test_m3_seen_rule():
    d, rows, K = fixture(5)
    q = np.ones(len(d["pid"]), bool)
    rows = dict(rows)
    rows["seen_ts"] = rows["end_ts"].copy()       # every base row on time
    tab = run(d, rows_of(store_of(rows), d), K, q)["table"]
    base_seen = run(d, rows_of(store_of(rows), d), K, q, rule="seen", table=tab)
    base_game = run(d, rows_of(store_of(rows), d), K, q, rule="game", table=tab)
    for i in probes(d, 6)[:10]:
        k = d["player_keys"][d["pid"][i]]
        t = int(d["day"][i]) * DAY
        up = t + DAY + 100
        r2 = add_rows(rows, key=[k], replay_id=[7_100_001], end_ts=[t - 30 * DAY], hero=[d["hero"][i]], win=[1],
                      src=[2], seen_ts=[up])
        e2 = rows_of(store_of(r2), d)
        s2 = run(d, e2, K, q, rule="seen", table=tab)
        before = np.flatnonzero(d["day"] <= up // DAY)
        assert same_on(s2, base_seen, before), i
        g2 = run(d, e2, K, q, rule="game", table=tab)
        assert g2["nh"][i] == base_game["nh"][i] + 1


def test_m4_brute_counts():
    d, rows, K = fixture(7)
    st = store_of(rows)
    ex = rows_of(st, d)
    slot_pairs = set(zip(d["pid"].tolist(), d["replay_id"].tolist()))
    assert not slot_pairs & set(zip(ex["pid"].tolist(), ex["replay_id"].tolist()))
    assert (ex["src"] & 2).all() and len(ex["pid"]) == 4000
    q = np.ones(len(d["pid"]), bool)
    for rule in ("game", "seen"):
        res = run(d, ex, K, q, rule=rule)
        av = ex["end_day"] if rule == "game" else np.maximum(ex["end_day"], ex["seen_day"])
        for i in np.random.default_rng(8).choice(len(d["pid"]), 300, replace=False):
            p, h, t = d["pid"][i], d["hero"][i], d["day"][i]
            ns = ((d["pid"] == p) & (d["day"] < t)).sum() + ((ex["pid"] == p) & (av < t)).sum()
            nh = ((d["pid"] == p) & (d["hero"] == h) & (d["day"] < t)).sum() + \
                ((ex["pid"] == p) & (ex["hero"] == h) & (av < t)).sum()
            assert res["np"][i] == ns and res["nh"][i] == nh, (rule, i)
            assert res["nh_slot"][i] == ((d["pid"] == p) & (d["hero"] == h) & (d["day"] < t)).sum()


def test_m5_no_extras_equals_a1_path():
    d, _, K = fixture(9)
    q = d["day"] > D0 + 200
    res = run(d, None, K, q)
    n_p, n_ph = C.experience_counts(d)
    e_mask = d["in_sample"] & (d["day"] >= C.day_of(C.E_START))
    table = C.fit_experience(d["r"], n_p, n_ph, e_mask)
    r_adj = d["r"] - table[C.exp_bins(n_p, n_ph)]
    qi, m, v, Nh, Np, _ = X.online_predict(d, np.ones(len(d["pid"]), bool), q, K, r_adj)
    mu = table[C.exp_bins(Np.astype(np.int64), Nh.astype(np.int64))]
    assert np.array_equal(res["qidx"], qi)
    assert np.allclose(res["pred"], mu + m, atol=1e-12) and np.allclose(res["var"], v, atol=1e-12)


def test_m5b_slot_counts_offset():
    """exp_counts="slots": the offset and its table equal the arm without
    extras; only the GP state moves."""
    d, rows, K = fixture(10)
    q = d["day"] > D0 + 200
    ex = rows_of(store_of(rows), d)
    a = M.merged_predict(d, ex, K, q, exp_counts="slots")
    b = M.merged_predict(d, None, K, q)
    assert np.array_equal(a["table"], b["table"]) and np.array_equal(a["mu"], b["mu"])
    assert not np.allclose(a["mean"], b["mean"])
    assert np.array_equal(a["nh_slot"], b["nh"])


def test_m5c_half_life_matches_online_state():
    """The recency weight in online_predict equals p3_hs_core.online_state's."""
    d, _, K = fixture(12)
    q = d["day"] > D0 + 200
    n_p, n_ph = C.experience_counts(d)
    r_adj = d["r"] - 0.001 * np.minimum(n_ph, 5)
    allm = np.ones(len(d["pid"]), bool)
    qi, m, v, _, _, _ = X.online_predict(d, allm, q, K, r_adj, half_life=90.0)
    qi2, S, P, _, _, _, _ = C.online_state(d, allm, q, len(d["hero_names"]), r_adj, half_life=90.0)
    m2, v2 = C.gp_query(S, P, d["hero"][qi2], K)
    assert np.array_equal(qi, qi2) and np.allclose(m, m2, atol=1e-10) and np.allclose(v, v2, atol=1e-10)
    # future rows still never enter with a half-life (planted far-future rows)
    m3 = X.online_predict(d, allm, q, K, r_adj, half_life=90.0, lag=1)[1]
    assert np.allclose(m, m3)


def test_m6_clock_guard():
    import datetime
    t_jun = int(datetime.datetime(2026, 6, 30, 22, 0, tzinfo=datetime.timezone.utc).timestamp())
    t_jul = int(datetime.datetime(2026, 7, 20, 22, 0, tzinfo=datetime.timezone.utc).timestamp())
    t_jul_am = int(datetime.datetime(2026, 7, 20, 10, 0, tzinfo=datetime.timezone.utc).timestamp())
    g = M.guard_day([t_jun, t_jul, t_jul_am])
    assert g.tolist() == [t_jun // DAY, t_jul // DAY + 1, t_jul_am // DAY]


def test_m7_proxy_in_sample_only():
    d, rows, K = fixture(11)
    ex = rows_of(store_of(rows), d)
    a = M.wp_proxy(d, ex)
    d2 = dict(d)
    d2["y"] = np.where(d["in_sample"], d["y"], 1 - d["y"]).astype(np.float32)
    assert np.array_equal(a, M.wp_proxy(d2, ex))
    assert ((a > 0.2) & (a < 0.8)).all()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
