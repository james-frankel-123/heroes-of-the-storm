"""Merged as-of history store (p3_full_history): leak fixtures.

Ideas from training/scilearn/fixtures.py: each check runs a legitimate
reader and must also catch a planted leak.
  F1 planted future rows (after the scored game, and exactly at its time)
     leave every feature unchanged, i.e. equal to deleting them
  F2 own-label exclusion: flipping the scored game's own outcome changes
     none of its features, even when its HP row carries an earlier
     (offset) time than the query
  F3 late upload under rule "seen" equals deleting the row for queries
     before the upload; under rule "game" it still counts (documented gap)
  F4 vectorized counts equal a brute-force loop on a random store
  F5 merge dedups a match in both sources once, corpus values win, and the
     builder's cap drops rows from 2026-09-28 on
  F6 p3_rows hands P3 only HP-only rows of P3 players
  planted leak: a reader with <= instead of < (or no own-row exclusion)
     must fail F1/F2
Run: python3 personalization/tests/test_full_history.py (or pytest)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import p3_full_history as FH

DAY = 86400
T0 = 1_700_000_000


def random_rows(n_players=30, n_rows=3000, seed=0):
    rng = np.random.default_rng(seed)
    key = (rng.integers(1, 3, n_rows) << 40) | rng.integers(1, n_players, n_rows)
    rid = rng.permutation(np.arange(1, n_rows + 1)) + 10_000
    end = T0 + rng.integers(0, 400, n_rows) * DAY + rng.integers(0, DAY, n_rows)
    end[:50] = end[50:100]                       # exact time ties across rows
    hero = rng.integers(0, 90, n_rows)
    win = rng.integers(0, 2, n_rows)
    src = rng.choice([1, 2, 3], n_rows)
    seen = end + np.where(rng.random(n_rows) < 0.1, rng.integers(1, 200, n_rows) * DAY, 0)
    return dict(key=key, replay_id=rid, end_ts=end, hero=hero, win=win, src=src, seen_ts=seen)


def brute(r, qkey, qts, qrid, qhero, rule="game"):
    col = r["end_ts"] if rule == "game" else np.maximum(r["seen_ts"], r["end_ts"])
    out = {k: [] for k in ("n", "w", "n_corpus", "n_hero", "w_hero", "n_corpus_hero")}
    for k, t, rid, h in zip(qkey, qts, qrid, qhero):
        m = (r["key"] == k) & (col < t) & (r["replay_id"] != rid)
        mh = m & (r["hero"] == h)
        out["n"].append(m.sum())
        out["w"].append(r["win"][m].sum())
        out["n_corpus"].append(((r["src"][m] & 1) > 0).sum())
        out["n_hero"].append(mh.sum())
        out["w_hero"].append(r["win"][mh].sum())
        out["n_corpus_hero"].append(((r["src"][mh] & 1) > 0).sum())
    return {k: np.array(v) for k, v in out.items()}


def queries(r, n=400, seed=1):
    """Scored games = some of the store's own rows (their own row is in the
    store, so own-label exclusion is exercised), plus unknown players."""
    rng = np.random.default_rng(seed)
    i = rng.choice(len(r["key"]), n, replace=False)
    qkey = r["key"][i].copy()
    qkey[:20] = (3 << 40) | 999                  # players not in the store
    return qkey, r["end_ts"][i].copy(), r["replay_id"][i].copy(), r["hero"][i].copy(), i


def same(a, b):
    return all(np.array_equal(a[k], b[k]) for k in a)


class LeakyStore(FH.Store):
    """Planted leak: <= instead of <, and no own-row exclusion."""

    def asof_counts(self, qkey, qts, qrid, qhero=None, rule="game"):
        return super().asof_counts(qkey, np.asarray(qts) + 1, np.full(len(qts), -1), qhero, rule)


def test_f4_brute_force():
    r = random_rows()
    st = FH.Store.from_rows(**r)
    qk, qt, qr, qh, _ = queries(r)
    for rule in ("game", "seen"):
        got = st.asof_counts(qk, qt, qr, qh, rule=rule)
        want = brute(r, qk, qt, qr, qh, rule)
        assert same(got, want), rule


def _add(r, **add):
    return {k: np.r_[r[k], np.asarray(add[k])] for k in r}


def _stores(r, cls=FH.Store):
    return cls.from_rows(**r)


N_PROBES = 60


def test_f1_future_rows_excluded():
    """Per probe query q: plant a win on q's player and hero exactly at q's
    time and one a day later. Every query that starts at or before the
    planted time (q itself included) must be unchanged; the leaky reader
    must change q."""
    r = random_rows(seed=2)
    qk, qt, qr, qh, _ = queries(r, seed=3)
    base = _stores(r).asof_counts(qk, qt, qr, qh)
    lbase = _stores(r, LeakyStore).asof_counts(qk, qt, qr, qh)
    caught = 0
    for q in range(20, 20 + N_PROBES):
        r2 = _add(r, key=[qk[q]] * 2, replay_id=[900_001, 900_002], end_ts=[qt[q], qt[q] + DAY], hero=[qh[q]] * 2,
                  win=[1, 1], src=[3, 3], seen_ts=[qt[q], qt[q] + DAY])
        got = _stores(r2).asof_counts(qk, qt, qr, qh)
        before = qt <= qt[q]
        assert all(np.array_equal(got[k][before], base[k][before]) for k in got), q
        lk = _stores(r2, LeakyStore).asof_counts(qk, qt, qr, qh)
        caught += any(lk[k][q] != lbase[k][q] for k in lk)
    assert caught == N_PROBES, "planted future rows not caught in the leaky reader"


def test_f2_own_label_excluded():
    """Per probe query q (a store row): its HP copy carries a time 10 minutes
    BEFORE the query time (API game_date offset), so only the replay_id rule
    removes it. Flipping q's outcome must leave q's features (and every
    earlier query's) unchanged; the leaky reader must change q."""
    r = random_rows(seed=4)
    qk, qt, qr, qh, i = queries(r, seed=5)
    r_off = {k: v.copy() for k, v in r.items()}
    caught = 0
    probes = [q for q in range(20, len(qk)) if qk[q] == r["key"][i[q]]][:N_PROBES]
    for q in probes:
        a_rows = {k: v.copy() for k, v in r_off.items()}
        a_rows["end_ts"][i[q]] = qt[q] - 600
        a_rows["seen_ts"][i[q]] = qt[q] - 600
        b_rows = {k: v.copy() for k, v in a_rows.items()}
        b_rows["win"][i[q]] = 1 - b_rows["win"][i[q]]
        a = _stores(a_rows).asof_counts(qk, qt, qr, qh)
        b = _stores(b_rows).asof_counts(qk, qt, qr, qh)
        before = qt <= qt[q]
        assert all(np.array_equal(a[k][before], b[k][before]) for k in a), q
        la = _stores(a_rows, LeakyStore).asof_counts(qk, qt, qr, qh)
        lb = _stores(b_rows, LeakyStore).asof_counts(qk, qt, qr, qh)
        caught += la["w"][q] != lb["w"][q]
        # the own row is in the store before qt: a reader that keeps it counts one more
        keep_own = _stores(a_rows).asof_counts(qk[q:q + 1], qt[q:q + 1], [-1], qh[q:q + 1])
        assert keep_own["n"][0] == a["n"][q] + 1
    assert len(probes) == N_PROBES and caught == N_PROBES, "own label not caught in the leaky reader"


def test_f3_late_upload():
    """A match played 30 days before probe q but uploaded a day after it:
    under rule "seen" equal to deleting it for every query before the
    upload; under rule "game" q counts it (the documented gap)."""
    r = random_rows(seed=6)
    qk, qt, qr, qh, i = queries(r, seed=7)
    base_seen = _stores(r).asof_counts(qk, qt, qr, qh, rule="seen")
    base_game = _stores(r).asof_counts(qk, qt, qr, qh)
    for q in range(20, 20 + N_PROBES):
        up = qt[q] + DAY
        r2 = _add(r, key=[qk[q]], replay_id=[800_001], end_ts=[qt[q] - 30 * DAY], hero=[qh[q]], win=[1], src=[2],
                  seen_ts=[up])
        s2 = _stores(r2).asof_counts(qk, qt, qr, qh, rule="seen")
        before = qt <= up
        assert all(np.array_equal(s2[k][before], base_seen[k][before]) for k in s2), q
        g2 = _stores(r2).asof_counts(qk, qt, qr, qh)
        assert g2["n"][q] == base_game["n"][q] + 1


def test_f5_merge_and_cap(tmp_dir=None):
    corpus = dict(rid=np.array([1, 2, 3]), region=np.array([1, 1, 2]), blizz=np.array([7, 7, 7]),
                  hero=np.array([5, 6, 7], np.int8), end=np.array([T0, T0 + 10, T0 + 20]),
                  gl=np.array([600, 700, 800], np.int32), win=np.array([1, 0, 1], np.int8),
                  mmr=np.array([2000, 2010, 2020], np.float32))
    hp = dict(rid=np.array([2, 4, 3]), region=np.array([1, 1, 1]), blizz=np.array([7, 7, 7]),
              hero=np.array([6, 9, 7], np.int8), end=np.array([T0 + 40, T0 - 100, T0 + 20]),
              win=np.array([0, 1, 1], np.int8), map=np.array([3, 4, 5], np.int16),
              mmr=np.array([1, 2, 3], np.float32))
    m, agree = FH.merge(corpus, hp)
    k1, k2 = (1 << 40) | 7, (2 << 40) | 7
    rows = {(int(k), int(r)): (int(s), int(e), float(mm)) for k, r, s, e, mm in
            zip(m["key"], m["replay_id"], m["src"], m["end_ts"], m["post_mmr"])}
    assert len(m["key"]) == 5                    # (k1,1) (k1,2) (k1,4) (k2,3) (k1,3)
    assert rows[(k1, 2)] == (3, T0 + 40, 2010.0)  # in both: corpus values, HP clock, src 3
    assert rows[(k1, 4)][0] == 2 and rows[(k2, 3)][0] == 1 and rows[(k1, 3)][0] == 2
    assert agree["pairs"] == 1 and agree["winner_disagree"] == 0
    # cap: the shard parser drops rows from 2026-09-28 on
    import gzip
    import json
    import tempfile
    d = tempfile.mkdtemp()
    p = os.path.join(d, "s.jsonl.gz")
    cols = ["replayID", "game_date", "winner", "game_map", "hero", "mmr"]
    rec = {"region": 1, "blizz_id": 7, "cols": cols, "rows": [
        [10, "2026-09-27 23:59:59", "True", 1, "Ana", 2500],
        [11, "2026-09-28 00:00:00", "False", 1, "Ana", 2490],
        [12, "2026-10-02 10:00:00", "True", 1, "Ana", 2500]]}
    with gzip.open(p, "wt") as f:
        f.write(json.dumps(rec) + "\n")
    cat, acc, n_cap, _, _ = FH._parse_shard(p)
    assert cat["rid"].tolist() == [10] and n_cap == 2 and acc == [(1, 7, "")]


def test_f7_clock_fix():
    """Corpus clock bug: from CLOCK_FIX_RID on, a corpus-only row is shifted
    back by the Eastern offset (4 h in July, 5 h in January); a pair takes
    HP's time; rows below the id are untouched."""
    jul = 1_783_600_000                           # 2026-07-09
    jan = 1_767_600_000                           # 2026-01-05
    F = FH.CLOCK_FIX_RID
    corpus = dict(rid=np.array([F - 5, F + 1, F + 2, F + 3]), region=np.ones(4, int), blizz=np.full(4, 9),
                  hero=np.zeros(4, np.int8), end=np.array([jul, jul + 14400, jan + 18000, jul + 14400]),
                  gl=np.full(4, 600, np.int32), win=np.zeros(4, np.int8), mmr=np.zeros(4, np.float32))
    hp = dict(rid=np.array([F + 3]), region=np.ones(1, int), blizz=np.full(1, 9), hero=np.zeros(1, np.int8),
              end=np.array([jul]), win=np.zeros(1, np.int8), map=np.zeros(1, np.int16), mmr=np.zeros(1, np.float32))
    m, agree = FH.merge(corpus, hp)
    got = dict(zip(m["replay_id"].tolist(), m["end_ts"].tolist()))
    assert got == {F - 5: jul, F + 1: jul, F + 2: jan, F + 3: jul}, got
    assert agree["pairs_from_fix_id_with_eastern_offset"] == 1.0 and agree["corpus_only_rows_clock_fixed"] == 2


def test_f6_p3_rows():
    r = random_rows(seed=8)
    st = FH.Store.from_rows(**r)
    pk = np.unique(r["key"])[::2]
    names = list(st.a["hero_names"])
    rev = names[::-1]                            # a slot table with another hero order
    got = st.p3_rows(pk, rev)
    assert (np.diff(got["end_ts"]) >= 0).all()
    order = np.lexsort((got["replay_id"], pk[got["pid"]]))
    sel_rows = np.flatnonzero((r["src"] == 2) & np.isin(r["key"], pk))
    want = {(int(k), int(rr)): names[h] for k, rr, h in zip(r["key"][sel_rows], r["replay_id"][sel_rows], r["hero"][sel_rows])}
    for i in order:
        assert rev[got["hero"][i]] == want[(int(pk[got["pid"][i]]), int(got["replay_id"][i]))]
    assert np.array_equal(st.to_store_heroes(np.arange(90), rev), np.arange(90)[::-1])
    keys = pk[got["pid"]]
    sel = (r["src"] == 2) & np.isin(r["key"], pk)
    assert len(got["pid"]) == sel.sum()
    assert set(zip(keys.tolist(), got["replay_id"].tolist())) == set(zip(r["key"][sel].tolist(), r["replay_id"][sel].tolist()))


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
