"""
Nightly player_skill_state builder for the personal layer (review 5.5).

One row per (region, blizz_id, hero) the player has a counted game on:
  S               sum over counted games of r / v, r = y_team - wp_team,
                  v = wp_team (1 - wp_team) (the GP sufficient statistic)
  P               sum of 1 / v
  n               counted games
  last_day        UTC day (days since 1970-01-01) of the latest counted game
  first_seen_day  the player's earliest counted game, any hero
  status          account status 0-3 (p3_hero_level_causal.account_status):
                  0 first seen before LATE_DAY; 1 later, no usable level;
                  2 later, highest level <= 5; 3 later, some level > 5
  wp_vintage_id   newest WP vintage (registry order) among the row's games
  max_fetched_ts  latest effective fetched_at among the row's games
  as_of           the snapshot time (same for every row)
No battletags are stored. Battletags are resolved at serve time by lookup().

Visibility contract (review 5.1). A player-game counts in the state as of T
only if its effective fetched_at < T and its game ended before T. Effective
fetched_at is the later of replay_players.fetched_at and
replay_draft_data.fetched_at (both rows must exist for the residual). A state
as of T may serve a game only if T <= the game's start, so no state row was
fetched after the predicted game started.

Residuals come only from prod/wp_chain.py rows (one per replay, team-0 y and
wp, vintage_id). A visible player-game whose replay has no residual yet is
"pending": it is not counted and is retried on the next night.

Two build paths that must give identical state (bit for bit):
  full_rebuild(slots, residuals, as_of)            from all source rows
  incremental(state, slots, residuals, as_of)      from the previous state plus
      candidate rows (fetched_at >= the previous as_of, or pending)
Both keep a ledger of counted player-games with their per-game contribution.
The ledger (a) stops a refetched row from being counted twice, (b) lets a
monthly rescore under a new WP vintage replace a game's contribution, and
(c) makes every row a sum in one canonical order (player, hero, replay_id),
so the two paths add the same floats in the same order.

Sources: frozen export files (p3_export.py stamps.csv.gz + games.csv.gz),
fully implemented; or the local research DB through DATABASE_URL_RESEARCH
(read-only, Neon URLs refused), a thin query layer with the same columns.
"""
import csv
import datetime
import gzip
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P3_DIR = os.path.dirname(HERE)
if P3_DIR not in sys.path:
    sys.path.insert(0, P3_DIR)

from p3_heroes import NUM_HEROES  # noqa: E402  (hero-set contract: 90 v1 heroes)
from p3_keys import REGION_SHIFT, UNKNOWN_REGION  # noqa: E402
import shared  # noqa: E402  (on sys.path through p3_heroes)

DDL = os.path.join(HERE, "player_skill_state.sql")
LATE_DAY = (datetime.date(2024, 7, 1) - datetime.date(1970, 1, 1)).days
NEW_ACCOUNT_MAX_LEVEL = 5
LEDGER = ("rid", "key", "hero", "team", "day", "fetched", "s", "p", "vin", "lo", "hi")
ROW = ("key", "hero", "S", "P", "n", "last_day", "first_seen_day", "status", "vin", "max_fetched_ts")


# ------------------------------------------------------------------ keys

def make_keys(region, blizz_id):
    """(region << 40) | blizz_id; a missing region is the explicit unknown
    region, never a negative code (review 6.7)."""
    r = np.asarray(region, np.int64)
    b = np.asarray(blizz_id, np.int64)
    assert (b >= 0).all() and (b < (1 << REGION_SHIFT)).all(), "blizz_id outside [0, 2^40)"
    r = np.where(r > 0, r, UNKNOWN_REGION)
    return (r << REGION_SHIFT) | b


def split_keys(keys):
    keys = np.asarray(keys, np.int64)
    return keys >> REGION_SHIFT, keys & ((1 << REGION_SHIFT) - 1)


def _pair_member(a_r, a_k, b_r, b_k):
    """Bool per a: is (a_r, a_k) among the (b_r, b_k) pairs."""
    if not len(a_r) or not len(b_r):
        return np.zeros(len(a_r), bool)
    r = np.r_[b_r, a_r]
    k = np.r_[b_k, a_k]
    tag = np.r_[np.zeros(len(b_r), np.int8), np.ones(len(a_r), np.int8)]
    o = np.lexsort((tag, k, r))
    rs, ks, ts = r[o], k[o], tag[o]
    grp = np.cumsum(np.r_[True, (rs[1:] != rs[:-1]) | (ks[1:] != ks[:-1])]) - 1
    has_b = np.bincount(grp, weights=(ts == 0)) > 0  # the pair occurs among the b rows
    out = np.zeros(len(a_r), bool)
    is_a = ts == 1
    out[o[is_a] - len(b_r)] = has_b[grp[is_a]]
    return out


# ------------------------------------------------------------------ sources

def hero_index(names):
    """v1 hero index per name; -1 for a hero outside the 90-hero set."""
    idx = {h: i for i, h in enumerate(shared.HEROES)}
    assert len(idx) == NUM_HEROES
    return np.array([idx.get(str(h), -1) for h in names], np.int64)


def level_interval(level, band):
    """(lo, hi) from a numeric hero_level or a v1-API band string such as
    "25-50" or "100+"; (nan, nan) if neither parses. Same rule as
    p3_hero_level_causal.band (copied: that module needs numba)."""
    if level not in (None, ""):
        x = float(level)
        return x, x
    s = (band or "").strip()
    if not s:
        return np.nan, np.nan
    try:
        if s.endswith("+"):
            return float(s[:-1]), np.inf
        if "-" in s:
            a, b = s.split("-", 1)
            return float(a), float(b)
        return float(s), float(s)
    except ValueError:
        return np.nan, np.nan


def _int(x, default=-1):
    return default if x in (None, "") else int(float(x))


def slots_from_records(stamps, games):
    """Slot arrays from player-game records (dicts with the p3_export stamps
    columns) and game records (games columns). Effective fetched = the later
    of the two fetched timestamps; -1 (never visible) if either is missing."""
    gf = {int(g["replay_id"]): _int(g.get("fetched_ts")) for g in games}
    cols = {k: [] for k in ("rid", "region", "blizz", "hero_name", "team", "end_ts", "game_length",
                            "fetched", "lo", "hi")}
    for r in stamps:
        if r.get("blizz_id") in (None, ""):
            continue
        rid = int(r["replay_id"])
        fp, fg = _int(r.get("fetched_ts")), gf.get(rid, -1)
        lo, hi = level_interval(r.get("hero_level"), r.get("hero_level_band"))
        cols["rid"].append(rid)
        cols["region"].append(_int(r.get("region"), 0))
        cols["blizz"].append(int(r["blizz_id"]))
        cols["hero_name"].append(r["hero"])
        cols["team"].append(int(r["team"]))
        cols["end_ts"].append(int(r["end_ts"]))
        cols["game_length"].append(_int(r.get("game_length")))
        cols["fetched"].append(max(fp, fg) if fp >= 0 and fg >= 0 else -1)
        cols["lo"].append(lo)
        cols["hi"].append(hi)
    hero = hero_index(cols.pop("hero_name"))
    out = {"rid": np.array(cols["rid"], np.int64),
           "key": make_keys(np.array(cols["region"], np.int64), np.array(cols["blizz"], np.int64)),
           "hero": hero, "team": np.array(cols["team"], np.int64),
           "end_ts": np.array(cols["end_ts"], np.int64), "game_length": np.array(cols["game_length"], np.int64),
           "fetched": np.array(cols["fetched"], np.int64),
           "lo": np.array(cols["lo"], np.float64), "hi": np.array(cols["hi"], np.float64)}
    keep = out["hero"] >= 0
    out = {k: v[keep] for k, v in out.items()}
    out["dropped_unknown_hero"] = int((~keep).sum())
    return out


def games_from_records(games):
    """Game arrays for wp_chain.rescore: replay_id, build, end_ts, game_length,
    y (1 = team 0 won; replay_draft_data.winner is 0 or 1)."""
    g = [x for x in games if str(x.get("winner")) in ("0", "1")]
    return {"replay_id": np.array([int(x["replay_id"]) for x in g], np.int64),
            "build": np.array([str(x["game_version"]) for x in g]),
            "end_ts": np.array([int(x["end_ts"]) for x in g], np.int64),
            "game_length": np.array([_int(x.get("game_length")) for x in g], np.int64),
            "y": np.array([1.0 if str(x["winner"]) == "0" else 0.0 for x in g])}


def _read_csv_gz(path):
    with gzip.open(path, "rt", newline="") as f:
        yield from csv.DictReader(f)


def load_frozen(export_dir):
    """(slots, games) from p3_export.py files <export_dir>/stamps.csv.gz and
    games.csv.gz. Streams with the csv module (no pandas on the workers)."""
    games = list(_read_csv_gz(os.path.join(export_dir, "games.csv.gz")))
    slots = slots_from_records(_read_csv_gz(os.path.join(export_dir, "stamps.csv.gz")), games)
    return slots, games_from_records(games)


DB_STAMPS = """
SELECT rp.replay_id, rp.blizz_id, COALESCE(rp.region, d.region) AS region, rp.hero, rp.hero_level,
       rp.raw_extras->>'hero_level_band' AS hero_level_band,
       EXTRACT(EPOCH FROM d.game_date)::bigint AS end_ts, d.game_length, rp.team,
       EXTRACT(EPOCH FROM rp.fetched_at)::bigint AS fetched_ts
FROM replay_players rp JOIN replay_draft_data d USING (replay_id)
WHERE (rp.fetched_at >= to_timestamp(%(since)s) OR d.fetched_at >= to_timestamp(%(since)s)
       OR d.game_date >= to_timestamp(%(since)s) OR rp.replay_id = ANY(%(pending)s))
  AND rp.fetched_at < to_timestamp(%(as_of)s) AND d.fetched_at < to_timestamp(%(as_of)s)"""
DB_GAMES = """
SELECT d.replay_id, d.game_version, EXTRACT(EPOCH FROM d.game_date)::bigint AS end_ts, d.game_length,
       d.winner, EXTRACT(EPOCH FROM d.fetched_at)::bigint AS fetched_ts
FROM replay_draft_data d WHERE d.replay_id = ANY(%(rids)s)"""


def research_url():
    url = os.environ.get("DATABASE_URL_RESEARCH")
    if not url:
        raise SystemExit("DATABASE_URL_RESEARCH is not set (local read-only role)")
    if "neon" in url.lower():
        raise SystemExit("refusing a Neon URL: the personal state reads the local store only")
    return url


def load_db(since_ts, as_of_ts, pending_rids=(), url=None):
    """(slots, games) for rows fetched in [since_ts, as_of_ts) plus pending
    replays, from the local research store (read-only session). since_ts = 0
    reads everything (full rebuild)."""
    url = url or research_url()
    if "neon" in url.lower():
        raise SystemExit("refusing a Neon URL")
    import psycopg2
    conn = psycopg2.connect(url)
    conn.set_session(readonly=True)
    try:
        with conn.cursor() as c:
            c.execute(DB_STAMPS, {"since": int(since_ts), "as_of": int(as_of_ts),
                                  "pending": [int(x) for x in pending_rids]})
            names = [d[0] for d in c.description]
            stamps = [dict(zip(names, ["" if v is None else v for v in row])) for row in c.fetchall()]
            rids = sorted({int(r["replay_id"]) for r in stamps})
            c.execute(DB_GAMES, {"rids": rids})
            names = [d[0] for d in c.description]
            games = [dict(zip(names, ["" if v is None else v for v in row])) for row in c.fetchall()]
    finally:
        conn.close()
    return slots_from_records(stamps, games), games_from_records(games)


# ------------------------------------------------------------------ core

def visible(slots, as_of):
    """The visibility contract: fetched before as_of and the game over."""
    f = slots["fetched"]
    return (f >= 0) & (f < as_of) & (slots["end_ts"] < as_of)


def _vintage_names(old, residuals, order):
    names = list(old)
    if order is not None:
        if list(order[:len(names)]) != names:
            raise ValueError("vintage order must extend the state's vintage list (append-only registry)")
        names = list(order)
    for v in np.unique(residuals["vintage_id"]) if len(residuals["vintage_id"]) else []:
        if str(v) not in names:
            if order is not None:
                raise ValueError(f"residual vintage {v!r} not in the given vintage order")
            names.append(str(v))
    return names


def _join_residuals(rid, team, residuals, names):
    """Per slot: has a residual, s = r/v, p = 1/v, vintage index (team view).
    Slots without a residual get s = p = 0 and vintage -1 (never used)."""
    rr = np.asarray(residuals["replay_id"], np.int64)
    n = len(rid)
    if len(np.unique(rr)) != len(rr):
        raise ValueError("residual rows must be one per replay")
    if not len(rr):
        return np.zeros(n, bool), np.zeros(n), np.zeros(n), np.full(n, -1, np.int64)
    o = np.argsort(rr)
    pos = np.minimum(np.searchsorted(rr[o], rid), len(rr) - 1)
    j = o[pos]
    has = rr[j] == rid
    y = np.asarray(residuals["y"], np.float64)[j]
    wp = np.asarray(residuals["wp"], np.float64)[j]
    yt = np.where(team == 0, y, 1 - y)
    wt = np.where(team == 0, wp, 1 - wp)
    v = wt * (1 - wt)
    idx = {nm: i for i, nm in enumerate(names)}
    vin_r = np.array([idx[str(x)] for x in np.asarray(residuals["vintage_id"])], np.int64)
    s = np.where(has, (yt - wt) / v, 0.0)
    p = np.where(has, 1.0 / v, 0.0)
    return has, s, p, np.where(has, vin_r[j], -1)


def _canon(led):
    o = np.lexsort((led["rid"], led["hero"], led["key"]))
    return {k: v[o] for k, v in led.items()}


def _segmax(x, starts, fill):
    """Max per segment, NaN treated as missing (fill) and returned as NaN."""
    z = np.where(np.isnan(x), fill, x)
    m = np.maximum.reduceat(z, starts)
    return np.where(m == fill, np.nan, m)


def _aggregate(led, late_day):
    """State rows from a canonical-order ledger (all games of each player)."""
    n = len(led["rid"])
    if n == 0:
        return {k: np.zeros(0, np.float64 if k in ("S", "P") else np.int64) for k in ROW}
    k, h = led["key"], led["hero"]
    brk = np.flatnonzero(np.r_[True, (k[1:] != k[:-1]) | (h[1:] != h[:-1])])
    row = np.cumsum(np.r_[True, (k[1:] != k[:-1]) | (h[1:] != h[:-1])]) - 1
    nr = len(brk)
    S = np.bincount(row, weights=led["s"], minlength=nr)
    P = np.bincount(row, weights=led["p"], minlength=nr)
    cnt = np.bincount(row, minlength=nr).astype(np.int64)
    last = np.maximum.reduceat(led["day"], brk)
    vin = np.maximum.reduceat(led["vin"], brk)
    mf = np.maximum.reduceat(led["fetched"], brk)
    pbrk = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    prow = np.cumsum(np.r_[True, k[1:] != k[:-1]]) - 1
    first = np.minimum.reduceat(led["day"], pbrk)
    alo = _segmax(led["lo"], pbrk, -1.0)
    ahi = _segmax(led["hi"], pbrk, -1.0)
    late = first >= late_day
    st = np.where(~late, 0, np.where(ahi <= NEW_ACCOUNT_MAX_LEVEL, 2,
                                     np.where(alo > NEW_ACCOUNT_MAX_LEVEL, 3, 1)))
    rp = prow[brk]
    return {"key": k[brk], "hero": h[brk], "S": S, "P": P, "n": cnt, "last_day": last,
            "first_seen_day": first[rp], "status": st[rp].astype(np.int64), "vin": vin,
            "max_fetched_ts": mf}


def _ledger_from(slots, sel, s, p, vin):
    return {"rid": slots["rid"][sel], "key": slots["key"][sel], "hero": slots["hero"][sel],
            "team": slots["team"][sel], "day": slots["end_ts"][sel] // 86400,
            "fetched": slots["fetched"][sel], "s": s[sel], "p": p[sel], "vin": vin[sel],
            "lo": slots["lo"][sel], "hi": slots["hi"][sel]}


def _empty_ledger():
    z = {k: np.zeros(0, np.int64) for k in LEDGER}
    for k in ("s", "p", "lo", "hi"):
        z[k] = np.zeros(0, np.float64)
    return z


def _pack(rows, led, pend_rid, pend_key, names, as_of, late_day, stats):
    st = {f"row_{k}": v for k, v in rows.items()}
    st.update({f"ledger_{k}": v for k, v in led.items()})
    st.update(pending_rid=pend_rid, pending_key=pend_key,
              vintage_names=np.array(names, dtype="U64"), as_of=np.int64(as_of),
              late_day=np.int64(late_day))
    st["stats"] = stats
    return st


def _check_unique_slots(slots):
    o = np.lexsort((slots["key"], slots["rid"]))
    r, k = slots["rid"][o], slots["key"][o]
    dup = (r[1:] == r[:-1]) & (k[1:] == k[:-1])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate (replay_id, player) slots in the input")


def full_rebuild(slots, residuals, as_of, vintage_order=None, late_day=LATE_DAY):
    """State as of `as_of` from every source row (see the module docstring)."""
    _check_unique_slots(slots)
    names = _vintage_names([], residuals, vintage_order)
    vis = visible(slots, as_of)
    has, s, p, vin = _join_residuals(slots["rid"], slots["team"], residuals, names)
    led = _canon(_ledger_from(slots, vis & has, s, p, vin))
    pend = vis & ~has
    rows = _aggregate(led, late_day)
    stats = {"counted": int(len(led["rid"])), "pending": int(pend.sum()), "not_visible": int((~vis).sum())}
    return _pack(rows, led, slots["rid"][pend], slots["key"][pend], names, as_of, late_day, stats)


def candidates(slots, state):
    """Rows the nightly job must look at: fetched (or ended) at or after the
    previous as_of, or still pending. (The DB path selects the same set in SQL.)"""
    pend = _pair_member(slots["rid"], slots["key"], state["pending_rid"], state["pending_key"])
    prev = int(state["as_of"])
    return (slots["fetched"] >= prev) | (slots["end_ts"] >= prev) | pend


def incremental(state, slots, residuals, as_of, vintage_order=None):
    """Advance `state` to `as_of` from candidate slots (see candidates()) and
    residual rows (new games, plus any games rescored under a newer vintage).
    Already-counted player-games are never added twice; a rescored game's
    contribution is replaced. Only players with a changed game are recomputed."""
    if as_of < int(state["as_of"]):
        raise ValueError("as_of moves backwards")
    _check_unique_slots(slots)
    late_day = int(state["late_day"])
    names = _vintage_names([str(x) for x in state["vintage_names"]], residuals, vintage_order)
    led = {k: state[f"ledger_{k}"] for k in LEDGER}
    rows = {k: state[f"row_{k}"] for k in ROW}

    # 1. rescored games already in the ledger: replace their contribution
    has_l, s_l, p_l, vin_l = _join_residuals(led["rid"], led["team"], residuals, names)
    changed = has_l & ((s_l != led["s"]) | (p_l != led["p"]) | (vin_l != led["vin"]))
    led["s"] = np.where(changed, s_l, led["s"])
    led["p"] = np.where(changed, p_l, led["p"])
    led["vin"] = np.where(changed, vin_l, led["vin"])

    # 2. new visible player-games not yet counted
    vis = visible(slots, as_of)
    counted = _pair_member(slots["rid"], slots["key"], led["rid"], led["key"])
    has, s, p, vin = _join_residuals(slots["rid"], slots["team"], residuals, names)
    add = vis & ~counted & has
    pend = vis & ~counted & ~has
    new = _ledger_from(slots, add, s, p, vin)
    touched = np.unique(np.r_[led["key"][changed], new["key"]])
    led = _canon({k: np.r_[led[k], new[k]] for k in LEDGER})

    # 3. recompute every row of a touched player from the ledger
    in_t = np.isin(led["key"], touched)
    sub = {k: v[in_t] for k, v in led.items()}  # still canonical order
    fresh = _aggregate(sub, late_day)
    keep = ~np.isin(rows["key"], touched)
    merged = {k: np.r_[rows[k][keep], fresh[k]] for k in ROW}
    o = np.lexsort((merged["hero"], merged["key"]))
    rows = {k: v[o] for k, v in merged.items()}
    stats = {"added": int(add.sum()), "rescored": int(changed.sum()), "pending": int(pend.sum()),
             "touched_players": int(len(touched)), "counted": int(len(led["rid"]))}
    return _pack(rows, led, slots["rid"][pend], slots["key"][pend], names, as_of, late_day, stats)


# ------------------------------------------------------------------ output

def table_rows(state):
    """The player_skill_state columns, named as in player_skill_state.sql.
    Timestamps (max_fetched_at, as_of) are epoch seconds here; the loader
    converts them with to_timestamp."""
    region, blizz = split_keys(state["row_key"])
    names = state["vintage_names"]
    return {"region": region, "blizz_id": blizz, "hero": state["row_hero"], "s": state["row_S"],
            "p": state["row_P"], "n_games": state["row_n"], "last_played_day": state["row_last_day"],
            "first_seen_day": state["row_first_seen_day"], "account_status": state["row_status"],
            "wp_vintage_id": names[state["row_vin"]] if len(names) else np.zeros(0, "U64"),
            "max_fetched_at": state["row_max_fetched_ts"],
            "as_of": np.full(len(region), int(state["as_of"]), np.int64)}


def save_state(state, path):
    """npz with the ledger (needed for the next incremental run) and the
    table columns (table__*) ready for loading into player_skill_state."""
    z = {k: v for k, v in state.items() if k != "stats"}
    z.update({f"table__{k}": v for k, v in table_rows(state).items()})
    tmp = path + ".tmp.npz"
    np.savez(tmp, **z)
    os.replace(tmp, path)


def load_state(path):
    z = np.load(path, allow_pickle=False)
    st = {k: z[k] for k in z.files if not k.startswith("table__")}
    st["stats"] = {}
    return st


# ------------------------------------------------------------------ serve

def lookup(state, candidates_, posterior_fn=None):
    """Probability-weighted state for one battletag (review 4.2 item 3).

    candidates_: list of (region, blizz_id, weight), the accounts the battletag
      may refer to in its region, with weights >= 0 (normalized here). A
      candidate missing from the table is a never-seen account: zero stats,
      weight kept.
    Returns weights, per-candidate (S, P, n) arrays over the 90 heroes, their
    weighted averages (a convenience, not a posterior), and, if posterior_fn
    (S_vec, P_vec) -> (mean, var) is given, the mixture posterior per hero:
    mean = sum w_c m_c, var = sum w_c (v_c + m_c^2) - mean^2."""
    if not candidates_:
        raise ValueError("lookup needs at least one candidate account")
    w = np.array([float(c[2]) for c in candidates_])
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("candidate weights must be >= 0 with a positive sum")
    w = w / w.sum()
    keys = make_keys([c[0] for c in candidates_], [c[1] for c in candidates_])
    rk, rh = state["row_key"], state["row_hero"]
    H = NUM_HEROES
    S = np.zeros((len(keys), H))
    P = np.zeros((len(keys), H))
    N = np.zeros((len(keys), H), np.int64)
    found = []
    for i, k in enumerate(keys):
        a, b = np.searchsorted(rk, k, "left"), np.searchsorted(rk, k, "right")
        S[i, rh[a:b]] = state["row_S"][a:b]
        P[i, rh[a:b]] = state["row_P"][a:b]
        N[i, rh[a:b]] = state["row_n"][a:b]
        found.append(bool(b > a))
    out = {"weights": w, "found": found, "S": S, "P": P, "n": N,
           "S_mean": w @ S, "P_mean": w @ P, "n_mean": w @ N}
    if posterior_fn is not None:
        mv = [posterior_fn(S[i], P[i]) for i in range(len(keys))]
        m = np.array([x[0] for x in mv])
        v = np.array([x[1] for x in mv])
        mean = w @ m
        out["mean"] = mean
        out["var"] = w @ (v + m ** 2) - mean ** 2
    return out
