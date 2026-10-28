"""
Offline player cards and name index for the site (docs/personalization-site-design.md).

The website never calls Heroes Profile. Everything it shows about a player is
computed here, on the main box, from our research store, then published.

Model: the paper's definitive chosen shrinkage model (P3_SHRINKAGE.md; build
def1027, results/def_def1027/P3_DEFINITIVE_def1027.md). The cards hold the
answers of its state writer (p3_rcf_states.py, half 0 = the drafter fit
"chosen") as of the last day of the store, unfiltered: s for all 90 heroes
and the display band sd_display = sqrt(s2 (1 + exp(-games / 10))).

Stages (run_site_publish.sh runs them in this order):
  build     (numpy + psycopg2) read the merged history store
            (cache/full_history/<store>/history.npz) plus the site's post-cap
            corpus rows (below), pick every account with a game in the last
            WINDOW days, read its latest visible name, discriminator and MMR
            from the local research DB, drop private accounts, and write
            cache/site/pre.npz and the state writer's query file.
  states    (numpy + numba, the def env: personalization/remote/def_env.sh)
            p3_rcf_states.answer on that query file.
  encode    cards from pre.npz and the answers: cache/site/accounts.npz and a
            size report.
  check     (def env) fresh state-writer answers for random published
            accounts must equal the decoded cards (s within the card's 1e-4
            quantization, the band within 1e-3, the band being sd_display);
            writes results/site_publish_check.json (errors only, no ids).
  load      replace the local table personal_accounts with the build (one
            transaction). sync/publish-site.ts then pushes it to Neon with
            the other site tables (it is in SITE_TABLES).
  artifact  (def env) src/lib/personal/model-artifact.json: the draft
            combiner refit with the off-role term (p3_site_offrole.py on
            the chosen arm) and the validation numbers the page quotes.
  simulate  name-matching accuracy on held-out lobbies (store data only).
  fixture   synthetic cards for src/lib/personal/__tests__.

Data rules. The research store holds no game from its cap on (2026-09-28 for
def1027). The site adds, here only, the corpus games from that date on (the
2.57 builds), read from the local DB with the 5,000 sealed replays
(training/replays/cohorts/sealed_ids.txt) excluded at the query; every other
row set is filtered against the same list. Post-cap rows only feed the game
counts, last played, the recent-games filter and the name index: the model's
estimates are as of the store's last day, because the state writer reads the
definitive build's caches. Battletags are read from the local store at build
time and go only to the local table and Neon, never to the repo.

Card format (bytea, little endian), version 2:
  u8 version, u8 model (3 = chosen shrinkage model), u32 games,
  90 x i16 s for every hero in model-artifact heroNames order (0.01 pp),
  u8 k, then k played heroes (most games first), each:
      u8 hero, u16 games, u16 last played (days since 2014-01-01), u8 band sd (0.1 pp)
  u8 j, then j never-played suggestions (highest s first): u8 hero, u8 band sd (0.1 pp)
The displayed estimate of a hero is s; its 80% band is s +- 1.2816 sd.

Usage (from training/, source ../.env):
  nice -n 19 taskset -c 44-47 python3 personalization/p3_site_publish.py build
  (. personalization/remote/def_env.sh $PWD def1027; python3 personalization/p3_site_publish.py states)
"""
import argparse
import datetime
import json
import os
import shutil
import struct
import sys
import time
import unicodedata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ARTIFACT = os.path.join(ROOT, "src", "lib", "personal", "model-artifact.json")
OUT = os.path.join(HERE, "cache", "site")
QDIR = os.path.join(OUT, "states_q")
ADIR = os.path.join(OUT, "states_a")
REGION_SHIFT = 40
DAY = 86400
EPOCH14 = (datetime.date(2014, 1, 1) - datetime.date(1970, 1, 1)).days
CARD_VERSION = 2
MODEL_CHOSEN = 3
DEF_TAG = "def1027"
H = 90
S_SCALE = 1e4          # s in 0.01 pp
SD_SCALE = 1e3         # band sd in 0.1 pp


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def hero_names():
    return list(json.load(open(ARTIFACT))["heroNames"])


def load_store(tag):
    z = np.load(os.path.join(HERE, "cache", "full_history", tag, "history.npz"))
    meta = json.load(open(os.path.join(HERE, "cache", "full_history", tag, "meta.json")))
    return z, meta


def def_env(tag):
    """The state writer reads the definitive build's caches through these variables."""
    got = os.environ.get("P3_RESULTS", "")
    assert os.environ.get("DEF_TAG") == tag and os.path.basename(os.path.normpath(got)) == f"def_{tag}" \
        and os.path.isdir(got), f"source personalization/remote/def_env.sh <training dir> {tag} first (P3_RESULTS={got!r})"


def states_module():
    sys.path.insert(0, os.path.dirname(HERE))
    import p3_rcf_states as RS
    from shared import HEROES
    assert len(HEROES) == H, "shared.HEROES must be the v1 set (unset HOTS_HERO_SET)"
    assert RS.ARMS[0][0] == "chosen", f"half 0 must be the chosen arm, got {RS.ARMS[0]}"
    return RS, list(HEROES)


def to_names_order(arr, sh_heroes, names):
    """state-writer columns (shared.HEROES order) -> model-artifact heroNames order"""
    perm = np.array([sh_heroes.index(n) for n in names])
    return arr[:, perm]


# ------------------------------------------------------------------ store rows

SEALED = os.path.join(os.path.dirname(HERE), "replays", "cohorts", "sealed_ids.txt")


def sealed_ids():
    """The 5,000 sealed replays (training/replays/cohorts/sealed_rule.json): never read."""
    ids = np.array(sorted(int(x) for x in open(SEALED).read().split()), np.int64)
    assert len(ids) == 5000, f"sealed list has {len(ids)} ids"
    return ids


def post_cap_rows(cap_ts, sealed, names):
    """Site only: corpus Storm League rows of games from the store cap on (the 2.57 builds),
    excluding the sealed replays at the query. The research store and its cap are untouched.
    No scoreboard for these rows (the a_shrink card covers the capped export only), so they
    feed the win channel and counts. replay_draft_data.game_date is UTC (timezone fix of
    2026-10-16, sync/docs/timezone-fix-2026-10.md)."""
    import psycopg2
    con = psycopg2.connect(os.environ["DATABASE_URL_RESEARCH"])
    con.set_session(readonly=True, autocommit=True)
    cur = con.cursor()
    hi = datetime.datetime.fromtimestamp(cap_ts, datetime.timezone.utc).replace(tzinfo=None)
    t0 = time.time()
    cur.execute("""
        SELECT rp.replay_id, rp.blizz_id, COALESCE(rp.region, d.region), rp.hero, rp.winner::int,
               EXTRACT(EPOCH FROM d.game_date)::bigint, rp.player_mmr
        FROM replay_draft_data d JOIN replay_players rp USING (replay_id)
        WHERE d.game_date >= %s AND rp.blizz_id > 0 AND rp.hero IS NOT NULL AND rp.winner IS NOT NULL
          AND NOT (d.replay_id = ANY(%s))""", (hi, [int(x) for x in sealed]))
    rows = cur.fetchall()
    con.close()
    hidx = {n: i for i, n in enumerate(names)}
    rid = np.array([r[0] for r in rows], np.int64)
    key = np.array([(int(r[2] or 0) << REGION_SHIFT) | int(r[1]) for r in rows], np.int64)
    hero = np.array([hidx.get(r[3], -1) for r in rows], np.int64)
    ok = (hero >= 0) & np.array([r[2] is not None for r in rows], bool)
    out = {"key": key[ok], "rid": rid[ok], "hero": hero[ok],
           "win": np.array([r[4] for r in rows], np.float64)[ok],
           "end": np.array([r[5] for r in rows], np.int64)[ok],
           "mmr": np.array([np.nan if r[6] is None else r[6] for r in rows], np.float64)[ok],
           "src": np.ones(int(ok.sum()), np.int8)}
    assert not np.isin(out["rid"], sealed).any()
    log(f"post-cap corpus rows: {len(out['rid']):,} ({time.time() - t0:.0f}s), last game "
        f"{datetime.datetime.fromtimestamp(int(out['end'].max()), datetime.timezone.utc):%Y-%m-%d}")
    return out


def site_rows(z, cap_ts, window_days, names, post=None, sealed=None):
    """Store rows (before the research cap) plus the site's post-cap rows; accounts with a game
    in the last window_days of the combined data; known heroes only."""
    ptr = z["ptr"]
    keys = z["keys"]
    kidx = np.repeat(np.arange(len(keys)), np.diff(ptr))
    hmap = np.array([names.index(str(h)) if str(h) in names else -1 for h in z["hero_names"]])
    h0 = z["hero"]
    hero = np.where(h0 >= 0, hmap[np.maximum(h0, 0)], -1)
    end = z["end_ts"]
    rid = z["replay_id"]
    keep = (hero >= 0) & (end < cap_ts)
    if sealed is not None:
        keep &= ~np.isin(rid, sealed)
    A = {"key": keys[kidx[keep]], "rid": rid[keep].astype(np.int64), "hero": hero[keep],
         "win": z["win"][keep].astype(np.float64), "end": end[keep].astype(np.int64),
         "mmr": z["post_mmr"][keep].astype(np.float64), "src": z["src"][keep]}
    if post is not None and len(post["rid"]):
        A = {k: np.r_[A[k], post[k]] for k in A}
    end_max = int(A["end"].max())
    lo = end_max - window_days * DAY
    uk, inv = np.unique(A["key"], return_inverse=True)
    recent = np.bincount(inv, weights=A["end"] >= lo, minlength=len(uk))
    act = np.flatnonzero(recent > 0)
    pos = np.full(len(uk), -1, np.int64)
    pos[act] = np.arange(len(act))
    pa = pos[inv]
    m = pa >= 0
    o = np.lexsort((A["end"][m], pa[m]))
    sel = np.flatnonzero(m)[o]
    return {"keys": uk[act], "recent": recent[act].astype(np.int64), "acct": pa[sel],
            "hero": A["hero"][sel], "win": A["win"][sel], "day": A["end"][sel] // DAY, "rid": A["rid"][sel],
            "src": A["src"][sel], "mmr": A["mmr"][sel], "end_max": end_max}


# ------------------------------------------------------------------ build

def query_arrays(keys, as_of):
    """Unfiltered queries of the drafter half (p3_rcf_states.QFIELDS)."""
    q = len(keys)
    return {"player_key": np.asarray(keys, np.int64), "as_of_day": np.full(q, as_of, np.int32),
            "ev_lo": np.full(q, -1, np.int32), "ev_parity": np.zeros(q, np.int8),
            "exclude_lo": np.full(q, -1, np.int32), "exclude_hi": np.full(q, -1, np.int32),
            "half": np.zeros(q, np.int8), "hero": np.full(q, -1, np.int16)}


def build(a):
    t0 = time.time()
    names = hero_names()
    z, meta = load_store(a.store)
    cap = int(meta["cap_ts"])
    sealed = sealed_ids()
    post = None if a.research_cap else post_cap_rows(cap, sealed, names)
    R = site_rows(z, cap, a.window, names, post, sealed)
    end = z["end_ts"]
    as_of = int(end[end < cap].max() // DAY)                   # the store's last day
    through = datetime.datetime.fromtimestamp(R["end_max"], datetime.timezone.utc).date()
    log(f"data through {through}; model estimates as of {datetime.date(1970, 1, 1) + datetime.timedelta(days=as_of)}")
    n = len(R["keys"])
    cell = R["acct"] * H + R["hero"]
    N = np.bincount(cell, minlength=n * H).reshape(n, H)
    last = np.full(n * H, -1, np.int64)
    np.maximum.at(last, cell, R["day"])
    last = last.reshape(n, H)
    tags, n_private = read_names(R["keys"], R["end_max"] + 1, a.window, sealed)
    lastrow = np.full(n, -1, np.int64)                         # latest MMR: HP post-game rating of the last row
    np.maximum.at(lastrow, R["acct"], np.arange(len(R["acct"])))
    mmr = np.where(lastrow >= 0, R["mmr"][np.maximum(lastrow, 0)], np.nan)
    keep = []
    nm, disc = [], []
    for i, t in enumerate(tags):
        if t is None:
            continue
        name, d = t.rsplit("#", 1)
        if not d.isdigit() or not name:
            continue
        keep.append(i)
        nm.append(name[:24])
        disc.append(int(d))
    keep = np.array(keep, np.int64)
    log(f"accounts with a visible name: {len(keep):,} of {n:,} (private-listed tags: {n_private:,})")
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "pre.npz"), keys=R["keys"][keep], N=N[keep].astype(np.int32),
             last=last[keep].astype(np.int32), recent=R["recent"][keep], mmr=mmr[keep],
             name=np.array(nm, dtype=object), disc=np.array(disc, np.int64),
             info=json.dumps({"store": a.store, "research_cap": datetime.datetime.fromtimestamp(cap, datetime.timezone.utc).date().isoformat(),
                              "as_of_day": as_of, "through": through.isoformat(), "window_days": a.window,
                              "post_cap_rows": 0 if post is None else int(len(post["rid"])),
                              "build_seconds": round(time.time() - t0)}))
    shutil.rmtree(QDIR, ignore_errors=True)
    shutil.rmtree(ADIR, ignore_errors=True)
    os.makedirs(QDIR)
    np.savez(os.path.join(QDIR, "site.npz"), **query_arrays(R["keys"][keep], as_of))
    log(f"pre.npz and {len(keep):,} state queries written ({time.time() - t0:.0f}s)")


def states(a):
    """The chosen model's states of every built account (p3_rcf_states.answer)."""
    def_env(a.def_tag)
    RS, sh = states_module()
    shutil.rmtree(ADIR, ignore_errors=True)
    RS.answer(QDIR, ADIR)
    with open(os.path.join(ADIR, "heroes.json"), "w") as f:
        json.dump({"def_tag": a.def_tag, "arm": RS.ARMS[0][0], "heroes": sh}, f)


def encode_card(s, sdd, N, last, games, n_sugg=5):
    """s, sdd: (H,) in heroNames order; N, last: (H,) site counts and last day (-1 never)."""
    b = bytearray(struct.pack("<BBI", CARD_VERSION, MODEL_CHOSEN, int(games)))
    b += np.clip(np.round(s * S_SCALE), -32767, 32767).astype("<i2").tobytes()
    played = np.flatnonzero(N > 0)
    played = played[np.argsort(-N[played], kind="stable")][:255]
    b += struct.pack("<B", len(played))
    for h in played:
        lastd = int(last[h]) - EPOCH14
        b += struct.pack("<BHHB", int(h), int(min(N[h], 65535)), max(0, min(lastd, 65535)),
                         int(np.clip(round(sdd[h] * SD_SCALE), 0, 255)))
    never = np.flatnonzero(N == 0)
    top = never[np.argsort(-s[never], kind="stable")][:n_sugg]
    b += struct.pack("<B", len(top))
    for h in top:
        b += struct.pack("<BB", int(h), int(np.clip(round(sdd[h] * SD_SCALE), 0, 255)))
    return bytes(b)


def decode_card(buf):
    """Python mirror of src/lib/personal/card.ts decodeCard."""
    v, model, games = struct.unpack_from("<BBI", buf, 0)
    assert v == CARD_VERSION and model == MODEL_CHOSEN
    o = 6
    s = np.frombuffer(buf, "<i2", H, o).astype(np.float64) / S_SCALE
    o += 2 * H
    k = buf[o]
    o += 1
    played = []
    for _ in range(k):
        h, n, lastd, sd = struct.unpack_from("<BHHB", buf, o)
        played.append((h, n, lastd + EPOCH14, sd / SD_SCALE))
        o += 6
    j = buf[o]
    o += 1
    sugg = []
    for _ in range(j):
        h, sd = struct.unpack_from("<BB", buf, o)
        sugg.append((h, sd / SD_SCALE))
        o += 2
    assert o == len(buf)
    return {"games": games, "s": s, "played": played, "suggestions": sugg}


def read_answers(names):
    hz = json.load(open(os.path.join(ADIR, "heroes.json")))
    A = np.load(os.path.join(ADIR, "site.npz"))
    out = {k: A[k] for k in A.files}
    for k in ("s", "sd", "sd_display", "n"):
        out[k] = to_names_order(out[k], hz["heroes"], names)
    return out, hz


def encode(a):
    t0 = time.time()
    names = hero_names()
    P = np.load(os.path.join(OUT, "pre.npz"), allow_pickle=True)
    P = {k: P[k] for k in P.files}
    info = json.loads(str(P["info"]))
    A, hz = read_answers(names)
    assert np.array_equal(A["player_key"], P["keys"]), "answers do not match pre.npz (rerun states)"
    assert (A["as_of_day"] == info["as_of_day"]).all()
    known = A["known"]
    assert np.isfinite(A["s"][known]).all() and np.isfinite(A["sd_display"][known]).all()
    log(f"accounts {len(known):,}, known to the model {int(known.sum()):,} (others have only post-cap games: dropped)")
    rows, sizes = [], []
    games = P["N"].sum(1)
    for i in np.flatnonzero(known):
        N = P["N"][i]
        card = encode_card(A["s"][i], A["sd_display"][i], N, P["last"][i], games[i])
        top = [int(h) for h in np.argsort(-N, kind="stable")[:3] if N[h] > 0]
        lastday = int(P["last"][i].max())
        mmr = P["mmr"][i]
        rows.append((int(P["keys"][i] >> REGION_SHIFT), int(P["keys"][i] & ((1 << REGION_SHIFT) - 1)),
                     str(P["name"][i]), int(P["disc"][i]), None if not np.isfinite(mmr) else int(round(float(mmr))),
                     (datetime.date(1970, 1, 1) + datetime.timedelta(days=lastday)).isoformat(),
                     int(games[i]), int(P["recent"][i]), bytes(top), card))
        sizes.append(len(card))
    cols = ["region", "blizz_id", "name", "discriminator", "mmr", "last_played", "games", "recent_games",
            "top_heroes", "card"]
    np.savez(os.path.join(OUT, "accounts.npz"), cols=np.array(cols),
             **{c: np.array([r[j] for r in rows], dtype=object) for j, c in enumerate(cols)})
    sizes = np.array(sizes)
    names_len = np.array([len(r[2].encode()) for r in rows])
    # Postgres estimate: 24 B tuple header + 4 B item pointer, fixed columns
    # 2+4+4+2+4+4+4 = 24 B, varlena headers, plus two btree indexes
    heap = 28 + 24 + (names_len + 1) + (3 + 1) + (sizes + 4)
    idx = 24 + 16 + names_len
    recent = np.array([r[7] for r in rows])
    rep = dict(info, model=f"chosen shrinkage model ({hz['def_tag']}, arm {hz['arm']}, p3_rcf_states.py)",
               estimates_as_of=(datetime.date(1970, 1, 1) + datetime.timedelta(days=info["as_of_day"])).isoformat(),
               accounts=len(rows), dropped_unknown=int((~known).sum()),
               card_bytes_mean=float(sizes.mean()), card_bytes_p95=float(np.quantile(sizes, 0.95)),
               by_recent_games={str(t): int((recent >= t).sum()) for t in (1, 3, 5, 10, 20)},
               built_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               encode_seconds=round(time.time() - t0))
    for t in (1, 3, 5, 10):
        m = recent >= t
        rep[f"neon_estimate_mb_min{t}"] = float((heap[m].sum() * 1.12 + idx[m].sum() * 1.3) / 1e6)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    log(json.dumps(rep))


def read_names(keys, cap_ts, window_days, sealed):
    """Latest battletag, region and player MMR per account from the local
    research DB, over corpus rows of games in the window ending at cap_ts (sealed
    replays excluded)."""
    import psycopg2
    con = psycopg2.connect(os.environ["DATABASE_URL_RESEARCH"])
    con.set_session(readonly=True, autocommit=True)
    cur = con.cursor()
    lo = datetime.datetime.fromtimestamp(cap_ts - window_days * DAY, datetime.timezone.utc).replace(tzinfo=None)
    hi = datetime.datetime.fromtimestamp(cap_ts, datetime.timezone.utc).replace(tzinfo=None)
    cur.execute("SELECT min(replay_id), max(replay_id) FROM replay_draft_data WHERE game_date >= %s AND game_date < %s",
                (lo, hi))
    rlo, rhi = cur.fetchone()
    t0 = time.time()
    cur.execute("""
        SELECT DISTINCT ON (r, rp.blizz_id) COALESCE(rp.region, d.region) AS r, rp.blizz_id, rp.battletag
        FROM replay_players rp JOIN replay_draft_data d USING (replay_id)
        WHERE rp.replay_id BETWEEN %s AND %s AND rp.blizz_id > 0 AND rp.battletag LIKE '%%#%%'
          AND d.game_date < %s AND NOT (rp.replay_id = ANY(%s))
        ORDER BY r, rp.blizz_id, rp.replay_id DESC""", (rlo, rhi, hi, [int(x) for x in sealed]))
    names = {}
    for reg, bl, tag in cur:
        if reg is None:
            continue
        names[(int(reg) << REGION_SHIFT) | int(bl)] = tag
    log(f"names from the window: {len(names):,} accounts ({time.time() - t0:.0f}s)")
    cur.execute("SELECT battletag, region FROM player_privacy WHERE state = 'private'")
    private = {(t, int(r)) for t, r in cur.fetchall()}
    con.close()
    out = []
    for k in keys:
        tag = names.get(int(k))
        reg = int(k) >> REGION_SHIFT
        out.append(None if tag is None or (tag, reg) in private else tag)
    return out, len(private)


# ------------------------------------------------------------------ load (local table)

DDL = """
CREATE TABLE IF NOT EXISTS personal_accounts (
  region smallint NOT NULL,
  blizz_id integer NOT NULL,
  name varchar(24) NOT NULL,
  discriminator integer NOT NULL,
  mmr smallint,
  last_played date NOT NULL,
  games integer NOT NULL,
  recent_games integer NOT NULL,
  top_heroes bytea NOT NULL,
  card bytea NOT NULL,
  CONSTRAINT personal_accounts_region_blizz_id_pk PRIMARY KEY (region, blizz_id)
);
CREATE INDEX IF NOT EXISTS personal_accounts_name_idx ON personal_accounts (lower(name));
"""


def load(a):
    import io
    import psycopg2
    zf = np.load(os.path.join(OUT, "accounts.npz"), allow_pickle=True)
    z = {k: zf[k] for k in zf.files}                  # NpzFile re-reads an array on every access
    keep = np.array(z["recent_games"], dtype=np.int64) >= a.min_recent
    n = int(keep.sum())
    url = os.environ["DATABASE_URL"]
    assert "neon.tech" not in url, "load writes the LOCAL store; publish-site.ts copies it to Neon"
    con = psycopg2.connect(url)
    cur = con.cursor()
    cur.execute(DDL)
    buf = io.StringIO()
    for i in np.flatnonzero(keep):
        mmr = z["mmr"][i]
        # NFC: the site normalizes lookups the same way (src/lib/personal/accounts.ts nfc)
        name = unicodedata.normalize("NFC", str(z["name"][i])).replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
        buf.write("\t".join([str(z["region"][i]), str(z["blizz_id"][i]), name, str(z["discriminator"][i]),
                             "\\N" if mmr is None else str(int(mmr)), str(z["last_played"][i]), str(z["games"][i]),
                             str(z["recent_games"][i]), "\\\\x" + bytes(z["top_heroes"][i]).hex(),
                             "\\\\x" + bytes(z["card"][i]).hex()]) + "\n")
    buf.seek(0)
    cur.execute("DELETE FROM personal_accounts")
    cur.copy_expert("COPY personal_accounts FROM STDIN", buf)
    con.commit()
    con.autocommit = True
    cur.execute("VACUUM personal_accounts")       # dead rows of the previous build become reusable
    cur.execute("SELECT pg_total_relation_size('personal_accounts')")
    log(f"personal_accounts: {n:,} rows (recent games >= {a.min_recent}); local size {cur.fetchone()[0] / 1e6:.1f} MB")
    con.close()


# ------------------------------------------------------------------ name-matching simulation

def match_score(cand, ctx, p):
    """Higher is better. cand: dict arrays for the candidates of one name."""
    s = np.where(cand["region"] == ctx["region"], p["region"], 0.0)
    known = np.isfinite(cand["mmr"]) & np.isfinite(ctx["mmr"])
    s = s - np.where(known, np.abs(cand["mmr"] - ctx["mmr"]) / p["mmr_scale"], p["mmr_unknown"])
    s = s + p["recency"] * np.exp(-np.maximum(ctx["day"] - cand["last"], 0) / p["recency_days"])
    return s


PARAMS_MATCH = {"region": 10.0, "mmr_scale": 300.0, "mmr_unknown": 1.0, "recency": 1.0, "recency_days": 30.0,
                "margin": 1.5, "min_score": 9.5}


def simulate(a):
    """Hide the #numbers of the other nine players in held-out lobbies and
    match each by visible name against an index built only from games
    before the test period, with the party's region and MMR as context.

    Index as of CUT (accounts with a game in the WINDOW days before CUT;
    last played, games and MMR as of CUT). Test lobbies: corpus games from
    CUT to the store cap. The party is one random player of the lobby; its
    region and MMR as of CUT are the context (the lobby MMR context uses the
    mean MMR of the party only, as the page knows nothing else)."""
    z, meta = load_store(a.store)
    cap = int(meta["cap_ts"])
    cut = cap - a.test_days * DAY
    keys_all = z["keys"]
    ptr = z["ptr"]
    kidx = np.repeat(np.arange(len(keys_all)), np.diff(ptr))
    end = z["end_ts"]
    before = end < cut
    # as-of index stats per account
    inwin = before & (end >= cut - a.window * DAY)
    nk = len(keys_all)
    idx_acct = np.bincount(kidx, weights=inwin, minlength=nk) >= a.min_recent
    lastday = np.full(nk, -1, np.int64)
    np.maximum.at(lastday, kidx[before], end[before] // DAY)
    lastrow = np.full(nk, -1, np.int64)
    br = np.flatnonzero(before)
    np.maximum.at(lastrow, kidx[br], br)
    mmr = np.where(lastrow >= 0, z["post_mmr"][np.maximum(lastrow, 0)], np.nan).astype(np.float64)
    # names: the production build's name map (the latest name in the window)
    accf = np.load(os.path.join(OUT, "accounts.npz"), allow_pickle=True)
    acc = {k: accf[k] for k in ("region", "blizz_id", "name")}
    akey = (acc["region"].astype(np.int64) << REGION_SHIFT) | acc["blizz_id"].astype(np.int64)
    kpos = np.searchsorted(keys_all, akey)
    ok = (kpos < nk) & (keys_all[np.minimum(kpos, nk - 1)] == akey)
    name_of = np.full(nk, None, dtype=object)
    name_of[kpos[ok]] = [str(x).lower() for x in acc["name"][ok]]
    # index: name -> account indices
    from collections import defaultdict
    by_name = defaultdict(list)
    for k in np.flatnonzero(idx_acct & (name_of != None)):  # noqa: E711
        by_name[name_of[k]].append(k)
    # test lobbies: corpus rows of games in [cut, cap)
    test = (end >= cut) & (end < cap) & ((z["src"] & 1) > 0)
    tr = np.flatnonzero(test)
    rid = z["replay_id"][tr]
    o = np.argsort(rid, kind="stable")
    tr, rid = tr[o], rid[o]
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    stops = np.r_[starts[1:], len(rid)]
    rng = np.random.default_rng(a.seed)
    games = np.flatnonzero((stops - starts) >= 6)
    games = rng.choice(games, min(a.games, len(games)), replace=False)
    region = keys_all >> REGION_SHIFT
    rec = []
    p = dict(PARAMS_MATCH, **json.loads(a.params or "{}"))
    for gi, g in enumerate(games):
        members = kidx[tr[starts[g]:stops[g]]]
        gday = int(end[tr[starts[g]]] // DAY)
        party = members[rng.integers(len(members))]
        ctx = {"region": region[party], "mmr": mmr[party], "day": gday}
        for k in members:
            if k == party:
                continue
            nm = name_of[k]
            if nm is None:
                rec.append((0, 0, 0, 0, 0, 0, gi))     # no visible name known
                continue
            cands = by_name.get(nm, [])
            in_idx = k in cands
            if not cands:
                rec.append((1, 0, 0, 0, 0, 0, gi))
                continue
            cand = {"region": region[cands], "mmr": mmr[cands], "last": lastday[cands]}
            sc = match_score(cand, ctx, p)
            order = np.argsort(-sc, kind="stable")
            ranked = [cands[i] for i in order]
            top1 = ranked[0] == k
            top3 = k in ranked[:3]
            clear = (sc[order[0]] >= p["min_score"]) and (
                len(cands) == 1 or (sc[order[0]] - sc[order[1]] >= p["margin"]))
            rec.append((1, int(in_idx), int(top1), int(top3), int(clear), len(cands), gi))
    R = np.array(rec)
    half = R[:, 6] >= len(games) // 2          # report on the second half; tune on the first
    out = {"cut": datetime.date.fromtimestamp(cut).isoformat(), "test_days": a.test_days,
           "lobbies": int(len(games)), "params": p, "min_recent": a.min_recent}
    for nm_, part in (("tune", R[~half]), ("report", R[half])):
        named, inidx = part[:, 0] == 1, part[:, 1] == 1
        hasc = part[:, 5] > 0
        clear = hasc & (part[:, 4] == 1)
        shown = hasc & ~clear
        out[nm_] = {
            "players": int(len(part)),
            "share_in_index": float(inidx.mean()),
            "indexed_top1": float(part[inidx, 2].mean()), "indexed_top3": float(part[inidx, 3].mean()),
            "indexed_unique_name": float((part[inidx, 5] == 1).mean()),
            "auto_selected": float(clear.mean()),
            "auto_correct": float((clear & (part[:, 2] == 1)).mean()),
            "auto_wrong": float((clear & (part[:, 2] == 0)).mean()),
            "auto_precision": float(part[clear, 2].mean()),
            "choices_shown": float(shown.mean()),
            "choices_contain_true_top3": float(part[shown, 3].mean()) if shown.any() else None,
            "no_candidate": float((named & ~hasc).mean()),
        }
    print(json.dumps(out, indent=1))
    with open(os.path.join(HERE, "results", "site_match_sim.json"), "w") as f:
        json.dump(out, f, indent=1)


# ------------------------------------------------------------------ checks, artifact and fixtures

def check(a):
    """Fresh state-writer answers for random published accounts (a new query file, a new
    answer run) against the decoded cards. Needs the def env (numba)."""
    import tempfile
    def_env(a.def_tag)
    RS, sh = states_module()
    names = hero_names()
    acc = np.load(os.path.join(OUT, "accounts.npz"), allow_pickle=True)
    acc = {k: acc[k] for k in ("region", "blizz_id", "card")}
    keys = (acc["region"].astype(np.int64) << REGION_SHIFT) | acc["blizz_id"].astype(np.int64)
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(keys), min(a.n_check, len(keys)), replace=False)
    info = json.loads(str(np.load(os.path.join(OUT, "pre.npz"), allow_pickle=True)["info"]))
    with tempfile.TemporaryDirectory() as td:
        qd, ad = os.path.join(td, "q"), os.path.join(td, "a")
        os.makedirs(qd)
        np.savez(os.path.join(qd, "check.npz"), **query_arrays(keys[pick], info["as_of_day"]))
        RS.answer(qd, ad)
        F = np.load(os.path.join(ad, "check.npz"))
        assert F["known"].all()
        fs, fsd, fsdd = (to_names_order(F[k], sh, names).astype(np.float64) for k in ("s", "sd", "sd_display"))
    # the build's own answer file, full precision, for the same accounts
    B, _ = read_answers(names)
    bpos = np.searchsorted(B["player_key"], keys[pick]) if (np.diff(B["player_key"]) > 0).all() else None
    if bpos is None:
        o = np.argsort(B["player_key"])
        bpos = o[np.searchsorted(B["player_key"][o], keys[pick])]
    assert np.array_equal(B["player_key"][bpos], keys[pick])
    # the band must follow its formula from the base sd and the games on the hero
    fn = to_names_order(F["n"], sh, names).astype(np.float64)
    fband = np.sqrt(RS.display_var(fsd ** 2, fn))
    err0 = float(np.abs(fband - fsdd).max())
    assert err0 < 1e-5, f"sd_display does not follow band {RS.BAND}: {err0:.2e}"
    err = {"s_build_vs_fresh": float(np.abs(B["s"][bpos] - fs).max()),
           "sd_display_build_vs_fresh": float(np.abs(B["sd_display"][bpos] - fsdd).max()),
           "s_card_vs_fresh": 0.0, "sd_card_vs_sd_display": 0.0, "suggestion_sd_vs_sd_display": 0.0}
    sd_vs_base = []
    for j, i in enumerate(pick):
        c = decode_card(bytes(acc["card"][i]))
        err["s_card_vs_fresh"] = max(err["s_card_vs_fresh"], float(np.abs(c["s"] - fs[j]).max()))
        for h, _, _, sd in c["played"]:
            err["sd_card_vs_sd_display"] = max(err["sd_card_vs_sd_display"], abs(sd - fsdd[j, h]),
                                               abs(sd - fband[j, h]))
            sd_vs_base.append(abs(sd - fsd[j, h]))
        for h, sd in c["suggestions"]:
            err["suggestion_sd_vs_sd_display"] = max(err["suggestion_sd_vs_sd_display"], abs(sd - fsdd[j, h]),
                                                     abs(sd - fband[j, h]))
    tol = {"s_build_vs_fresh": 1e-6, "sd_display_build_vs_fresh": 1e-6, "s_card_vs_fresh": 0.5 / S_SCALE + 1e-6,
           "sd_card_vs_sd_display": 0.5 / SD_SCALE + 1e-6, "suggestion_sd_vs_sd_display": 0.5 / SD_SCALE + 1e-6}
    out = {"accounts": int(len(pick)), "model": f"chosen ({a.def_tag}), p3_rcf_states.py half 0, unfiltered",
           "band": RS.BAND, "band_formula": RS.BANDS[RS.BAND]["formula"], "band_vs_formula_max": err0,
           "as_of_day": info["as_of_day"], "max_abs_error": err, "tolerance": tol,
           "band_sd_minus_base_sd_max": float(max(sd_vs_base)) if sd_vs_base else None,
           "note": "win-probability units; card values are quantized to 1e-4 (s) and 1e-3 (band sd)"}
    print(json.dumps(out, indent=1))
    with open(os.path.join(HERE, "results", "site_publish_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    bad = {k: v for k, v in err.items() if not v <= tol[k]}
    assert not bad, f"check failed: {bad}"
    # the band must be the display band, not the base posterior sd
    assert out["band_sd_minus_base_sd_max"] is None or out["band_sd_minus_base_sd_max"] > 2 / SD_SCALE


def artifact(a):
    """src/lib/personal/model-artifact.json for the chosen model (hero order and roles kept)."""
    def_env(a.def_tag)
    res = os.environ["P3_RESULTS"]
    old = json.load(open(ARTIFACT))
    off = json.load(open(os.path.join(res, "site", "offrole_refit.json")))["chosen"]
    RS, _ = states_module()
    bv = RS.BAND
    if bv == "def1027":
        band = json.load(open(os.path.join(res, "rcf", "band_def_chosen.json")))
        cover = {"v2": round(band["V2"]["cover80"], 4), "oot": round(band["OOT"]["cover80"], 4)}
        band_entry = {"version": bv, "formula": band["formula"], "z80": 1.2815515655446004}
        anchored = None
    else:
        an = json.load(open(os.path.join(res, "rcf", "band_anchor_chosen_k10_binomial.json")))
        f = an["forms"]["smooth_plus_floor"]
        assert np.allclose(f["b"], RS.BANDS[bv]["b"], rtol=0, atol=0), "band coefficients differ from the fit"
        ho = f["held_out"]
        cover = {"v2": round(ho["V2"]["summary"]["cover80"], 4), "oot": round(ho["OOT"]["summary"]["cover80"], 4)}
        by = [v["cover80"] for w in ("V2", "OOT") for v in ho[w]["summary"]["by_games"].values()]
        band_entry = {"version": bv, "formula": RS.BANDS[bv]["formula"], "b": RS.BANDS[bv]["b"],
                      "target": RS.BANDS[bv]["target"], "z80": 1.2815515655446004,
                      "fit": "maximum likelihood on V1 cells of the def1027 chosen arm (P3_BAND_ANCHOR.md)"}
        anchored = {"realizedOverClaimed": {"v2": round(ho["V2"]["summary"]["ratio"], 3),
                                            "oot": round(ho["OOT"]["summary"]["ratio"], 3)},
                    "cover80ByGamesRange": [round(min(by), 3), round(max(by), 3)], "horizonGames": an["k"]}
    z = np.load(os.path.join(res, "rcf", "p3c", "chosen__merged.npz"))
    acc = {}
    for k, nm in ((2, "v2"), (3, "oot")):
        m = z["g_win"] == k
        acc[nm] = round(float(((z["g_p"][m] > 0.5) == (z["g_y"][m] > 0.5)).mean()), 4)
    _, meta = load_store(a.store)
    as_of = datetime.datetime.fromtimestamp(int(meta["cap_ts"]), datetime.timezone.utc).date() - datetime.timedelta(days=1)
    b = off["coef_with"]
    A = {"version": f"{a.def_tag}-chosen",
         "source": ("training/personalization/p3_site_publish.py artifact; chosen shrinkage model of the "
                    f"definitive build {a.def_tag} (P3_SHRINKAGE.md, P3_DEFINITIVE_{a.def_tag}.md); draft combiner "
                    "refit with the off-role term by p3_site_offrole.py on V1 games"),
         "model": "chosen",
         "estimatesAsOf": as_of.isoformat(),
         "heroNames": old["heroNames"], "fineRoles": old["fineRoles"], "heroFineRole": old["heroFineRole"],
         "draft": {"b0": b[0], "b1": b[1], "b2": b[2], "b3": b[3],
                   "offRoleShare": 0.1, "offRoleMinGames": 50,
                   "offRoleGainX1000": {k: off[k]["gain_x1000"] for k in ("V2", "OOT")}},
         "band": band_entry,
         "validation": dict({"accuracy": acc, "cover80": cover}, **({"band": anchored} if anchored else {}))}
    with open(ARTIFACT, "w") as f:
        json.dump(A, f, indent=1)
        f.write("\n")
    print(json.dumps({k: A[k] for k in ("version", "estimatesAsOf", "draft", "validation")}, indent=1))


def fixture(a):
    """Synthetic cards (no real accounts) for src/lib/personal/__tests__: the encoded bytes,
    what they decode to, and match scores."""
    rng = np.random.default_rng(a.seed)
    names = hero_names()
    cases = []
    for c in range(4):
        k = [0, 3, 25, 40][c]
        heroes = rng.choice(H, k, replace=False)
        N = np.zeros(H, np.int64)
        N[heroes] = rng.integers(1, 300, k)
        last = np.where(N > 0, 20500 + rng.integers(0, 300, H), -1)
        s = rng.normal(-0.02, 0.04, H)
        sdd = rng.uniform(0.02, 0.12, H)
        card = encode_card(s, sdd, N, last, int(N.sum()))
        q = lambda x, sc: np.round(np.asarray(x) * sc) / sc
        never = np.flatnonzero(N == 0)
        top = never[np.argsort(-s[never], kind="stable")][:5]
        cases.append({"card_hex": card.hex(), "games": int(N.sum()), "s": q(s, S_SCALE).tolist(),
                      "played": {names[h]: {"games": int(N[h]), "sd": float(q(sdd[h], SD_SCALE)),
                                            "mean": float(q(s[h], S_SCALE))} for h in heroes},
                      "suggestions": [names[h] for h in top]})
    ctx = {"region": 2, "mmr": 2600.0, "day": 20700}
    cand = {"region": np.array([2, 2, 1, 2]), "mmr": np.array([2650.0, np.nan, 2600.0, 3400.0]),
            "last": np.array([20690, 20500, 20700, 20699])}
    sc = match_score(cand, ctx, PARAMS_MATCH)
    iso = lambda d: (datetime.date(1970, 1, 1) + datetime.timedelta(days=int(d))).isoformat()
    match = {"ctx": {"region": 2, "mmr": 2600, "asOf": iso(20700)},
             "candidates": [{"region": int(cand["region"][i]), "mmr": None if np.isnan(cand["mmr"][i]) else
                             float(cand["mmr"][i]), "lastPlayed": iso(cand["last"][i])} for i in range(4)],
             "scores": sc.tolist(), "params": PARAMS_MATCH}
    p = os.path.join(ROOT, "src", "lib", "personal", "__tests__", "fixtures", "site_cards.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump({"note": "Synthetic cards (no real accounts), from p3_site_publish.py fixture",
                   "cards": cases, "match": match}, f)
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "states", "encode", "check", "load", "artifact", "simulate", "fixture"])
    ap.add_argument("--n-check", type=int, default=200)
    ap.add_argument("--research-cap", action="store_true",
                    help="build: stop at the research store cap (no 2.57 rows)")
    ap.add_argument("--store", default=DEF_TAG)
    ap.add_argument("--def-tag", default=DEF_TAG, help="definitive build whose chosen model makes the cards")
    ap.add_argument("--window", type=int, default=180, help="days before the data's end that make an account active")
    ap.add_argument("--min-recent", type=int, default=3, help="load and simulate: minimum games in the window")
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--params", help="simulate: JSON overrides of PARAMS_MATCH")
    a = ap.parse_args()
    {"build": build, "states": states, "encode": encode, "check": check, "load": load, "artifact": artifact,
     "simulate": simulate, "fixture": fixture}[a.stage](a)


if __name__ == "__main__":
    main()
