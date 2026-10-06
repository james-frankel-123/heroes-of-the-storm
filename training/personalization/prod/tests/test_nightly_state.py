"""Synthetic tests for prod/nightly_state.py. Run: python3 training/personalization/prod/tests/test_nightly_state.py"""
import csv
import gzip
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import numpy as np

from personalization.prod import nightly_state as N
from personalization.prod import wp_chain as W

DAY = 86400
D0 = N.LATE_DAY - 20  # synthetic games start 20 days before the status cut-off


def synth(rng, n_players=150, n_games=1500, days=60):
    """Ten-player games; players drawn with replacement from a small pool so
    players repeat heroes. Fetch delays: mostly hours, some weeks, a few never."""
    pool_region = rng.choice([1, 2, 0], n_players, p=[0.6, 0.35, 0.05])
    pool_blizz = rng.randint(1, 10_000_000, n_players)
    pool_blizz[1] = pool_blizz[0]  # same blizz_id in two regions: distinct players
    pool_region[0], pool_region[1] = 1, 2
    rows = []
    games = {"replay_id": [], "end_ts": [], "y": [], "build": [], "game_length": [], "g_fetched": []}
    for g in range(n_games):
        rid = 10_000 + g
        end = (D0 + rng.randint(0, days)) * DAY + rng.randint(0, DAY)
        delay = int(rng.exponential(6 * 3600)) + (rng.randint(3, 30) * DAY if rng.rand() < 0.1 else 0)
        g_f = end + delay
        games["replay_id"].append(rid)
        games["end_ts"].append(end)
        games["y"].append(float(rng.rand() < 0.5))
        games["build"].append("2.55.12.3000")
        games["game_length"].append(1200)
        games["g_fetched"].append(g_f)
        players = rng.choice(n_players, 10, replace=False)
        for t, pl in enumerate(players):
            f = g_f + rng.randint(0, 3600)
            if rng.rand() < 0.01:
                f = -1  # never fetched / missing timestamp
            lv = rng.choice([np.nan, 3.0, 8.0, 40.0], p=[0.2, 0.3, 0.2, 0.3])
            rows.append((rid, N.make_keys([pool_region[pl]], [pool_blizz[pl]])[0], pl % 7 * 3 + rng.randint(0, 2),
                         t // 5, end, 1200, f, lv, lv))
    a = np.array(rows, dtype=object)
    slots = {"rid": a[:, 0].astype(np.int64), "key": a[:, 1].astype(np.int64), "hero": a[:, 2].astype(np.int64),
             "team": a[:, 3].astype(np.int64), "end_ts": a[:, 4].astype(np.int64),
             "game_length": a[:, 5].astype(np.int64), "fetched": a[:, 6].astype(np.int64),
             "lo": a[:, 7].astype(np.float64), "hi": a[:, 8].astype(np.float64)}
    games = {k: np.array(v) for k, v in games.items()}
    return slots, games


def residuals_at(games, as_of, rng_seed, rescore_after=None):
    """Residual rows available at as_of: games ended 2+ days before as_of are
    scored by v1; once as_of passes rescore_after, games after day 30 carry
    v2 scores (a monthly rescore)."""
    rng = np.random.RandomState(rng_seed)
    wp1 = rng.uniform(0.3, 0.7, len(games["y"]))
    wp2 = np.clip(wp1 + rng.normal(0, 0.03, len(wp1)), 0.05, 0.95)
    ok = games["end_ts"] < as_of - 2 * DAY
    v2 = (rescore_after is not None) & (as_of >= rescore_after) & (games["end_ts"] >= (D0 + 30) * DAY)
    return {"replay_id": games["replay_id"][ok], "y": games["y"][ok],
            "wp": np.where(v2, wp2, wp1)[ok],
            "vintage_id": np.where(v2, "v2", "v1")[ok].astype("U64")}


def assert_same(a, b, ctx):
    for k in a:
        if k.startswith(("row_", "ledger_")) or k in ("vintage_names", "as_of", "late_day"):
            assert a[k].dtype.kind == b[k].dtype.kind and np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"), (ctx, k)
    pa = set(zip(a["pending_rid"].tolist(), a["pending_key"].tolist()))
    pb = set(zip(b["pending_rid"].tolist(), b["pending_key"].tolist()))
    assert pa == pb, (ctx, "pending")


def test_incremental_equals_full(rng):
    slots, games = synth(rng)
    order = ["v1", "v2"]
    rescore_at = (D0 + 45) * DAY
    as_of = (D0 + 3) * DAY
    state = N.full_rebuild(slots, residuals_at(games, as_of, 7, rescore_at), as_of, order)
    rescored_seen = 0
    for night in range(4, 75):
        as_of = (D0 + night) * DAY + 3 * 3600
        res = residuals_at(games, as_of, 7, rescore_at)
        cand = N.candidates(slots, state)
        sub = {k: v[cand] for k, v in slots.items()}
        state = N.incremental(state, sub, res, as_of, order)
        rescored_seen += state["stats"]["rescored"]
        full = N.full_rebuild(slots, res, as_of, order)
        assert_same(state, full, f"night {night}")
        # visibility: nothing fetched at or after as_of, no game unfinished
        assert (state["row_max_fetched_ts"] < as_of).all()
        assert (state["ledger_fetched"] >= 0).all() and (state["ledger_fetched"] < as_of).all()
        assert (state["ledger_day"] * DAY < as_of).all()
    assert rescored_seen > 0, "the rescore path was not exercised"
    assert 1 in set(np.unique(state["row_vin"]).tolist()) and (state["ledger_vin"] == 0).any()
    return slots, games, state


def test_values(slots, games, state):
    # one row by hand
    i = int(np.argmax(state["row_n"]))
    k, h = state["row_key"][i], state["row_hero"][i]
    as_of = int(state["as_of"])
    res = residuals_at(games, as_of, 7, (D0 + 45) * DAY)
    m = (slots["key"] == k) & (slots["hero"] == h) & N.visible(slots, as_of) & np.isin(slots["rid"], res["replay_id"])
    j = np.searchsorted(res["replay_id"], slots["rid"][m])
    yt = np.where(slots["team"][m] == 0, res["y"][j], 1 - res["y"][j])
    wt = np.where(slots["team"][m] == 0, res["wp"][j], 1 - res["wp"][j])
    v = wt * (1 - wt)
    assert state["row_n"][i] == m.sum()
    assert np.isclose(state["row_S"][i], np.sum((yt - wt) / v)) and np.isclose(state["row_P"][i], np.sum(1 / v))
    assert state["row_last_day"][i] == (slots["end_ts"][m] // DAY).max()
    # keys: the two regions of one blizz_id stay separate players
    reg, bl = N.split_keys(state["row_key"])
    both = np.intersect1d(bl[reg == 1], bl[reg == 2])
    assert len(both) >= 1
    # status: first seen before LATE_DAY -> 0; later players never 0
    late = state["row_first_seen_day"] >= N.LATE_DAY
    assert (state["row_status"][~late] == 0).all() and (state["row_status"][late] != 0).all()
    assert set(np.unique(state["row_status"][late])) <= {1, 2, 3}


def test_refetch_not_double_counted(slots, state):
    """replay_players upserts overwrite fetched_at; a refetched row shows up
    again as a candidate and must not be added a second time."""
    as_of = int(state["as_of"]) + DAY
    i = np.flatnonzero(np.isin(slots["rid"], state["ledger_rid"]))[:5]
    sub = {k: v[i].copy() for k, v in slots.items()}
    sub["fetched"] = np.full(len(i), as_of - 60)
    res = {"replay_id": np.zeros(0, np.int64), "y": np.zeros(0), "wp": np.zeros(0), "vintage_id": np.zeros(0, "U64")}
    st2 = N.incremental(state, sub, res, as_of)
    assert st2["stats"]["added"] == 0 and np.array_equal(st2["row_n"], state["row_n"])


def test_save_load(state, tmp):
    p = os.path.join(tmp, "state.npz")
    N.save_state(state, p)
    st = N.load_state(p)
    assert_same(state, st, "save/load")
    z = np.load(p)
    ddl = open(N.DDL).read()
    cols = set(re.findall(r"^\s+([a-z_]+)\s+(?:smallint|bigint|integer|double|varchar|timestamptz)", ddl, re.M))
    tab = {k[len("table__"):] for k in z.files if k.startswith("table__")}
    assert tab == cols, (tab ^ cols)
    assert "battletag" not in ddl.lower().split("no battletags")[0]


def test_lookup(state):
    reg, bl = N.split_keys(state["row_key"])
    a = (int(reg[0]), int(bl[0]))
    out = N.lookup(state, [(a[0], a[1], 3.0), (9, 123, 1.0)],
                   posterior_fn=lambda S, P: (S / (P + 400.0), 1 / (P + 400.0)))
    assert out["found"] == [True, False] and np.allclose(out["weights"], [0.75, 0.25])
    sel = state["row_key"] == state["row_key"][0]
    S0 = np.zeros(N.NUM_HEROES)
    S0[state["row_hero"][sel]] = state["row_S"][sel]
    assert np.allclose(out["S_mean"], 0.75 * S0)
    m0 = S0 / (out["P"][0] + 400.0)
    assert np.allclose(out["mean"], 0.75 * m0)
    v0, v1 = 1 / (out["P"][0] + 400.0), np.full(N.NUM_HEROES, 1 / 400.0)
    assert np.allclose(out["var"], 0.75 * (v0 + m0 ** 2) + 0.25 * v1 - out["mean"] ** 2)
    assert (out["var"] >= v0.min() * 0.999).all()  # collision uncertainty adds variance


def write_export(tmp):
    """A tiny export in the p3_export.py column layout."""
    stamps_cols = ["replay_id", "blizz_id", "region", "region_was_null", "hero", "hero_level", "hero_level_band",
                   "parse_ts", "end_ts", "game_length", "game_version", "team", "party", "winner", "player_mmr",
                   "hero_mmr", "role_mmr", "fetched_ts"]
    games_cols = ["replay_id", "game_version", "end_ts", "game_length", "game_map", "skill_tier", "league_tier",
                  "region", "team0_heroes", "team1_heroes", "team0_bans", "team1_bans", "winner", "avg_mmr",
                  "draft_order", "fetched_ts"]
    t0 = (N.LATE_DAY + 100) * DAY
    st, gm = [], []
    heroes = ["Abathur", "Alarak", "Xal'atath", "Valla"]
    for g in range(4):
        rid = 500 + g
        end = t0 + g * DAY
        gm.append({"replay_id": rid, "game_version": "2.55.12.3000", "end_ts": end, "game_length": 1200,
                   "winner": g % 2, "fetched_ts": end + 600, "region": 1})
        for p in range(2):
            st.append({"replay_id": rid, "blizz_id": 77 + p, "region": "" if (g == 0 and p == 1) else 1,
                       "region_was_null": 0, "hero": heroes[(g + p) % 4], "team": p,
                       "hero_level": "" if p else 12, "hero_level_band": "25-50" if p else "",
                       "parse_ts": end + 100, "end_ts": end, "game_length": 1200, "game_version": "2.55.12.3000",
                       "winner": int(g % 2 == p), "fetched_ts": end + 700})
    for name, cols, recs in (("stamps", stamps_cols, st), ("games", games_cols, gm)):
        with gzip.open(os.path.join(tmp, f"{name}.csv.gz"), "wt", newline="") as f:
            w = csv.DictWriter(f, cols)
            w.writeheader()
            for r in recs:
                w.writerow({c: r.get(c, "") for c in cols})
    return t0


def test_frozen_path(tmp):
    t0 = write_export(tmp)
    slots, games = N.load_frozen(tmp)
    assert slots["dropped_unknown_hero"] == 2  # Xal'atath is outside the v1 set
    assert (N.split_keys(slots["key"])[0] == 0).sum() == 1  # NULL region -> explicit unknown region
    assert np.isnan(slots["lo"]).sum() == 0 and set(slots["hi"].tolist()) >= {12.0, 50.0}
    reg = {"schema": 1, "vintages": [{"vintage_id": "v1", "weights": [], "stats_path": "",
                                      "stats_cutoff_date": "2024-01-01", "trained_through_date": "2024-01-01",
                                      "build_cutoff": "2.55.11.0", "created_at": ""}]}
    rows, rep = W.rescore(games, reg, lambda v, g: np.full(len(g["y"]), 0.5), min_games=10 ** 9)
    assert rep["scored"] == 4 and (rows["y"] == np.array([1, 0, 1, 0])).all()
    st = N.full_rebuild(slots, rows, t0 + 10 * DAY)
    assert st["stats"]["counted"] == len(slots["rid"]) == 6
    assert np.isclose(st["row_P"].sum(), 6 * 4.0)  # v = 0.25 for every game
    # player 78 had a 25-50 band: lowest level 25 > 5, first seen late -> status 3
    t = N.table_rows(st)
    assert set(t["account_status"].tolist()) == {3}
    # an as_of before the fetches sees nothing
    assert N.full_rebuild(slots, rows, t0 + 650)["stats"]["counted"] == 0


def test_neon_refused():
    old = os.environ.get("DATABASE_URL_RESEARCH")
    os.environ["DATABASE_URL_RESEARCH"] = "postgres://u:p@ep-x.neon.tech/db"
    try:
        N.load_db(0, 1)
        raise AssertionError("neon URL accepted")
    except SystemExit as e:
        assert "Neon" in str(e)
    finally:
        if old is None:
            del os.environ["DATABASE_URL_RESEARCH"]
        else:
            os.environ["DATABASE_URL_RESEARCH"] = old


def main():
    rng = np.random.RandomState(1)
    slots, games, state = test_incremental_equals_full(rng)
    test_values(slots, games, state)
    test_refetch_not_double_counted(slots, state)
    with tempfile.TemporaryDirectory() as td:
        test_save_load(state, td)
        test_frozen_path(td)
    test_lookup(state)
    test_neon_refused()
    print("test_nightly_state: all passed")


if __name__ == "__main__":
    main()
