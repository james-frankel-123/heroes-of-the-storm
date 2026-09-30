"""
P3: MMR as it stood when each game started, from Heroes Profile's own
rating chain, plus validation.

Facts this relies on (checked on live replay_players rows 2026-09-30 and on
the v1 /players/mmr/history fixture):
  * HP stamps each player-game with its ratings AFTER that game, computed in
    HP's processing (parse) order: post_k = post_{k-1} + change_k.
  * The stamp time mmr_date_parsed is US Eastern local time; game_date is
    the UTC game END; start = end - game_length.
So a stamp is legitimate information for a game only if it was parsed
before the game started. Sources: cache/mmr_stamps_db.npz (every stored SL
player row, p3_fetch_mmr_stamps.py) and, when present, the HP MMR-history
pull in cache/mmr_history/shards/*.jsonl.gz (sync/fetch-mmr-history.ts),
which adds the player's games we never stored.

Per player-game (and per hero / role for hero_mmr / role_mmr):
  causal  post-game rating of the player's latest stamp parsed strictly
          before this game's start. Clean by construction; can be stale
          (batch uploaders).
  fresh   HP's own pre-game rating = post-game rating of this game's
          parse-order predecessor, used only when every game HP processed up
          to that predecessor ended before this game started (so nothing
          played later is in it); else falls back to causal.
Baselines: prev_stamp = stamped MMR of the player's previous game in play
order (what lagged analyses used; leaky when that game was parsed after
this one started), own_stamp = this game's own post-game stamp (certain
leak; positive control).

Validation on the P3 snapshot slots (hs_slots.npz, with WP residuals):
coverage, time since the last point, and a leak check: slope and log-loss
gain of r = y - wp on the lobby-centered value, per source, on the same
games; at-game MMR must not beat the clean previous-game MMR.

Usage (from training/):
  nice -n 19 taskset -c 48-63 python3 personalization/p3_mmr_at_game.py
Output: cache/mmr_at_game.npz, results/p3_mmr_at_game.json/.txt
"""
import glob
import gzip
import json
import os
import sys
import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import p3_hs_core as C

STAMPS = os.path.join(C.CACHE, "mmr_stamps_db.npz")
SHARDS = os.path.join(C.CACHE, "mmr_history", "shards")
OUT = os.path.join(C.CACHE, "mmr_at_game.npz")
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ------------------------------------------------------------ time helpers

def _naive_epoch(strs):
    """'YYYY-MM-DD HH:MM:SS' -> epoch seconds treating the text as UTC."""
    a = np.array(strs, dtype="datetime64[s]")
    return a.astype(np.int64)


def eastern_to_utc(naive_epoch):
    """US Eastern wall-clock (as naive epoch) -> UTC epoch, via DST lookup."""
    ny = ZoneInfo("America/New_York")
    out = np.empty_like(naive_epoch)
    days = naive_epoch // 86400
    for d in np.unique(days):
        # offset at local noon of that day (DST switches at 02:00; the
        # 2-hour transition window is negligible here)
        dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(days=int(d), hours=12)
        off = ny.utcoffset(dt).total_seconds()
        m = days == d
        out[m] = naive_epoch[m] - int(off)
    return out


def match(key_a, rid_a, key_b, rid_b):
    """Row of B with the same (key, replay_id) for each row of A, or -1."""
    _, inv = np.unique(np.concatenate([key_a, key_b]), return_inverse=True)
    ca = inv[:len(key_a)].astype(np.int64) << 32 | rid_a
    cb = inv[len(key_a):].astype(np.int64) << 32 | rid_b
    o = np.argsort(cb, kind="stable")
    pos = np.minimum(np.searchsorted(cb[o], ca), len(cb) - 1)
    return np.where(cb[o][pos] == ca, o[pos], -1)


# ------------------------------------------------------------ sources

def load_db():
    z = np.load(STAMPS)
    s = {k: z[k] for k in z.files}
    s["start_ts"] = s["end_ts"] - np.where(s["game_length"] > 0, s["game_length"], 3600)
    s["src"] = np.zeros(len(s["replay_ids"]), np.int8)
    return s


def load_api(hero_names, db):
    """HP history pull rows, converted to the DB stamp layout. Rows for games
    we already store keep our end time; others use the API game_date (UTC
    check reported)."""
    files = sorted(glob.glob(os.path.join(SHARDS, "*.jsonl.gz")))
    if not files:
        return None
    hidx = {h: i for i, h in enumerate(hero_names)}
    cols = {k: [] for k in ("rid", "reg", "bid", "hero", "pm", "hm", "rm", "pts", "gd", "win")}
    for f in files:
        try:
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("data_source") == "fixture":
                        continue
                    ci = {c: i for i, c in enumerate(r["cols"])}
                    for row in r["rows"]:
                        if row[ci["mmr_date_parsed"]] is None or row[ci["player_conservative_rating"]] is None:
                            continue
                        cols["rid"].append(row[ci["replayID"]])
                        cols["reg"].append(r["region"])
                        cols["bid"].append(r["blizz_id"])
                        cols["hero"].append(hidx.get(row[ci["hero"]], -1))
                        cols["pm"].append(1800 + 40 * row[ci["player_conservative_rating"]])
                        hc, rc = row[ci["hero_conservative_rating"]], row[ci["role_conservative_rating"]]
                        cols["hm"].append(np.nan if hc is None else 1800 + 40 * hc)
                        cols["rm"].append(np.nan if rc is None else 1800 + 40 * rc)
                        cols["pts"].append(row[ci["mmr_date_parsed"]])
                        cols["gd"].append(row[ci["game_date"]] or "1970-01-01 00:00:00")
                        cols["win"].append(str(row[ci["winner"]]).lower() in ("true", "1"))
        except EOFError:
            pass  # shard of a killed run: records before the last flush are intact
    if not cols["rid"]:
        return None
    a = {"replay_ids": np.array(cols["rid"], np.int64), "region": np.array(cols["reg"], np.int16),
         "blizz_ids": np.array(cols["bid"], np.int64), "hero": np.array(cols["hero"], np.int16),
         "player_mmr": np.array(cols["pm"], np.float32), "hero_mmr": np.array(cols["hm"], np.float32),
         "role_mmr": np.array(cols["rm"], np.float32), "winner": np.array(cols["win"])}
    a["parse_ts"] = eastern_to_utc(_naive_epoch(cols["pts"]))
    gd = _naive_epoch(cols["gd"])
    # align to DB end times where both exist; report the API game_date offset
    kd = db["region"].astype(np.int64) << 40 | db["blizz_ids"]
    ka = a["region"].astype(np.int64) << 40 | a["blizz_ids"]
    j = match(ka, a["replay_ids"], kd, db["replay_ids"])
    hit = j >= 0
    hit_idx = np.maximum(j, 0)
    if hit.any():
        off = gd[hit] - db["end_ts"][hit_idx[hit]]
        say(f"API rows matching DB rows: {hit.sum():,}/{len(hit):,}; API game_date - DB end: "
            f"median {np.median(off):.0f}s, p1 {np.percentile(off, 1):.0f}s, p99 {np.percentile(off, 99):.0f}s")
        med = int(np.median(off))
    else:
        med = 0
    a["end_ts"] = np.where(hit, db["end_ts"][hit_idx], gd - med)
    gl = np.where(hit, db["game_length"][hit_idx], -1)
    a["game_length"] = gl.astype(np.int32)
    a["start_ts"] = a["end_ts"] - np.where(gl > 0, gl, 3600)
    a["src"] = np.ones(len(gl), np.int8)
    return a


def merge(db, api):
    """Union keyed on (region, blizz_id, replay_id); API values win (they
    are HP's current chain), DB rows fill the rest."""
    if api is None:
        return db
    keys = ("replay_ids", "region", "blizz_ids", "hero", "player_mmr", "hero_mmr", "role_mmr",
            "parse_ts", "end_ts", "start_ts", "game_length", "winner", "src")
    cat = {k: np.concatenate([api[k], db[k]]) for k in keys}
    k = cat["region"].astype(np.int64) << 40 | cat["blizz_ids"]
    o = np.lexsort((-cat["src"], cat["replay_ids"], k))
    first = np.r_[True, (k[o][1:] != k[o][:-1]) | (cat["replay_ids"][o][1:] != cat["replay_ids"][o][:-1])]
    keep = o[first]
    say(f"merged stamps: {len(keep):,} (API {int((cat['src'][keep] == 1).sum()):,}, "
        f"DB-only {int((cat['src'][keep] == 0).sum()):,})")
    return {kk: cat[kk][keep] for kk in keys}


# ------------------------------------------------------------ core join

def at_game(group, parse_ts, start_ts, end_ts, value):
    """For each row i (a player-game), within its group:
      causal[i]: value of the latest row j with parse_ts[j] < start_ts[i]
      fresh[i]:  value of i's parse-order predecessor p if every row parsed
                 up to p ended by start_ts[i], else causal[i]
    Rows with parse_ts < 0 or NaN value never serve as sources. Returns
    causal, causal source index, fresh, fresh-used-predecessor flag."""
    n = len(group)
    valid = (parse_ts >= 0) & ~np.isnan(value)
    # dense group ids
    gu, gid = np.unique(group, return_inverse=True)
    gid = gid.astype(np.int64)
    BIG = np.int64(1) << 32
    src_rows = np.flatnonzero(valid)
    comp = gid[src_rows] * BIG + parse_ts[src_rows]
    o = np.argsort(comp, kind="stable")
    comp_s, rows_s = comp[o], src_rows[o]
    q = gid * BIG + start_ts
    pos = np.searchsorted(comp_s, q, side="left") - 1
    ok = pos >= 0
    ok[ok] &= (comp_s[pos[ok]] // BIG) == gid[ok]
    cidx = np.where(ok, rows_s[np.maximum(pos, 0)], -1)
    causal = np.where(cidx >= 0, value[np.maximum(cidx, 0)], np.nan)

    # fresh: predecessor in parse order among valid rows of the group
    # running max of end_ts along parse order within group
    end_s = end_ts[rows_s]
    grp_s = comp_s // BIG
    starts = np.r_[True, grp_s[1:] != grp_s[:-1]]
    # segmented cumulative max (end_ts < 2^31, so group offsets never overlap)
    seg = (np.cumsum(starts) - 1).astype(np.int64)
    offs = seg << 31
    runmax = np.maximum.accumulate(end_s.astype(np.int64) + offs) - offs
    # position of each valid row in the sorted order
    rank = np.empty(n, np.int64)
    rank[:] = -1
    rank[rows_s] = np.arange(len(rows_s))
    r = rank
    has_self = r >= 1  # r == 0 has no predecessor anywhere
    prev = np.where(has_self, r - 1, 0)
    same = has_self & (grp_s[prev] == grp_s[np.maximum(r, 0)])
    clean = same & (runmax[prev] <= start_ts)
    fresh = np.where(clean, value[rows_s[prev]], causal)
    return causal, cidx, fresh, clean


def prev_in_play_order(key, start_ts, replay_id):
    """Index of the same player's previous game by start time (or -1)."""
    o = np.lexsort((replay_id, start_ts, key))
    ks = key[o]
    prev = np.full(len(key), -1, np.int64)
    same = np.r_[False, ks[1:] == ks[:-1]]
    prev[o[same]] = o[np.flatnonzero(same) - 1]
    return prev


def q(x, ps=(10, 25, 50, 75, 90, 99)):
    x = x[np.isfinite(x)]
    return {f"p{p}": float(np.percentile(x, p)) for p in ps} if len(x) else {}


def main():
    t0 = datetime.datetime.now()
    db = load_db()
    names = [str(h) for h in db["hero_names"]]
    say(f"DB stamps: {len(db['replay_ids']):,} SL player rows; with parse stamp "
        f"{(db['parse_ts'] >= 0).mean():.1%}; with player_mmr {np.isfinite(db['player_mmr']).mean():.1%}")
    api = load_api(names, db)
    s = merge(db, api)
    n = len(s["replay_ids"])
    # A rating that includes game j cannot exist before j ended; ~0.5% of
    # stamps sit slightly before the stored end (clock skew, up to ~3h), and
    # a few before the game's own start. Clamp so a game can never source
    # its own post-game stamp.
    raw_parse = s["parse_ts"].copy()
    s["parse_ts"] = np.where(raw_parse >= 0, np.maximum(raw_parse, s["end_ts"]), -1)
    say(f"stamps clamped to game end: {((raw_parse >= 0) & (raw_parse < s['end_ts'])).mean():.4%}")
    meta = C.hero_meta(names)
    hero = s["hero"].astype(np.int64)
    role = np.where(hero >= 0, meta["blizz"][np.maximum(hero, 0)], -1)
    pkey = s["region"].astype(np.int64) << 40 | s["blizz_ids"]

    # --- sanity: parse time vs game end for a game's own stamp (tz check)
    has = raw_parse >= 0
    lag = (raw_parse - s["end_ts"]).astype(np.float64)
    lag[~has] = np.nan
    month = (s["end_ts"] // 86400 * 86400).astype("datetime64[s]").astype("datetime64[M]")
    tz_rows = {}
    for mo in np.unique(month[has])[::3]:
        m = has & (month == mo)
        tz_rows[str(mo)] = {"n": int(m.sum()), "share_parse_before_end": float((lag[m] < 0).mean()),
                            "p1_lag_s": float(np.percentile(lag[m], 1)),
                            "median_lag_s": float(np.median(lag[m]))}
    say("own stamp parse - game end, by month (tz check; negative share must be ~0):")
    for k, v in tz_rows.items():
        say(f"  {k}: n {v['n']:,}  parse<end {v['share_parse_before_end']:.4f}  "
            f"p1 {v['p1_lag_s']:.0f}s  median {v['median_lag_s']:.0f}s")

    res = {"tz_check": tz_rows}
    out = {"replay_ids": s["replay_ids"], "blizz_ids": s["blizz_ids"], "region": s["region"],
           "start_ts": s["start_ts"], "hero": s["hero"]}
    for nm, grp, val in (("player", pkey, s["player_mmr"]),
                         ("hero", pkey * 128 + np.maximum(hero, 0) + np.where(hero < 0, 127, 0), s["hero_mmr"]),
                         ("role", pkey * 8 + np.maximum(role, 0) + np.where(role < 0, 7, 0), s["role_mmr"])):
        causal, cidx, fresh, used = at_game(grp, s["parse_ts"], s["start_ts"], s["end_ts"],
                                            val.astype(np.float64))
        age = np.where(cidx >= 0, s["start_ts"] - s["parse_ts"][np.maximum(cidx, 0)], -1)
        out[f"{nm}_mmr_causal"] = causal.astype(np.float32)
        out[f"{nm}_mmr_causal_age_s"] = age.astype(np.int64)
        out[f"{nm}_mmr_fresh"] = fresh.astype(np.float32)
        out[f"{nm}_fresh_is_pre"] = used
        say(f"{nm}: causal coverage {np.isfinite(causal).mean():.1%}, fresh-pre used {used.mean():.1%}")

    prev = prev_in_play_order(pkey, s["start_ts"], s["replay_ids"])
    pv = np.where(prev >= 0, s["player_mmr"][np.maximum(prev, 0)], np.nan)
    pclean = (prev >= 0) & (s["parse_ts"][np.maximum(prev, 0)] >= 0) & \
        (s["parse_ts"][np.maximum(prev, 0)] < s["start_ts"])
    out["prev_stamp_mmr"] = pv.astype(np.float32)
    out["prev_stamp_parsed_before_start"] = pclean
    out["own_stamp_mmr"] = s["player_mmr"]
    out["own_parse_lag_s"] = np.where(has, raw_parse - s["end_ts"], np.iinfo(np.int64).min)
    out["source_api"] = s["src"] == 1
    out["hero_names"] = np.array(names)
    np.savez(OUT, **out)
    say(f"wrote {OUT}: {n:,} player-games")

    # --- validation on the P3 snapshot slots
    d = C.load_slots()
    pk = d["player_keys"][d["pid"]]
    idx = match(pk, d["replay_id"], pkey, s["replay_ids"])
    ok = idx >= 0
    say(f"snapshot slots: {len(pk):,}; matched to stamp rows {ok.mean():.2%}")
    gi = np.maximum(idx, 0)
    V = {k: np.where(ok, out[k][gi].astype(np.float64), np.nan) for k in
         ("player_mmr_causal", "player_mmr_fresh", "hero_mmr_causal", "hero_mmr_fresh",
          "role_mmr_causal", "role_mmr_fresh", "prev_stamp_mmr", "own_stamp_mmr")}
    V["slot_player_mmr_stamp"] = d["player_mmr"].astype(np.float64)
    age_d = np.where(ok, out["player_mmr_causal_age_s"][gi], -1) / 86400.0
    pre = ok & out["player_fresh_is_pre"][gi]
    pclean_s = ok & out["prev_stamp_parsed_before_start"][gi]

    # panel subset (>= 100 games in window) and overall
    cnt = np.bincount(d["pid"])
    heavy = cnt[d["pid"]] >= 100
    cov = {}
    for nm, sel in (("all slots", np.ones(len(pk), bool)), ("panel (>=100 games)", heavy)):
        cov[nm] = {"n": int(sel.sum()),
                   "player_causal": float(np.isfinite(V["player_mmr_causal"][sel]).mean()),
                   "player_fresh_pre_used": float(pre[sel].mean()),
                   "hero_causal": float(np.isfinite(V["hero_mmr_causal"][sel]).mean()),
                   "role_causal": float(np.isfinite(V["role_mmr_causal"][sel]).mean()),
                   "prev_stamp_exists": float(np.isfinite(V["prev_stamp_mmr"][sel]).mean()),
                   "prev_stamp_parsed_before_start": float(pclean_s[sel].mean()),
                   "age_days_causal": q(np.where(age_d[sel] >= 0, age_d[sel], np.nan))}
    res["coverage"] = cov
    for k, v in cov.items():
        say(f"coverage {k}: {json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if a != 'age_days_causal'})}")
        say(f"  days since causal point: {json.dumps({a: round(b, 2) for a, b in v['age_days_causal'].items()})}")
    fr = V["player_mmr_fresh"] - V["player_mmr_causal"]
    res["fresh_minus_causal"] = q(np.abs(fr[pre]))
    say(f"|fresh - causal| where fresh differs (pre used): {json.dumps({a: round(b, 1) for a, b in res['fresh_minus_causal'].items()})}")

    # --- leak check
    n_games = int(d["g"].max()) + 1
    r = d["r"]

    def centered(x, need_full=True):
        okx = np.isfinite(x)
        s_ = np.bincount(d["g"][okx], weights=x[okx], minlength=n_games)
        n_ = np.bincount(d["g"][okx], minlength=n_games)
        xc = np.where(okx, x - s_[d["g"]] / np.maximum(n_[d["g"]], 1), np.nan)
        return xc, n_[d["g"]] == 10

    cols = ["player_mmr_causal", "player_mmr_fresh", "prev_stamp_mmr", "own_stamp_mmr",
            "slot_player_mmr_stamp"]
    XC, FULL = {}, np.ones(len(r), bool)
    for c in cols:
        XC[c], f = centered(V[c])
        FULL &= f
    # Center every source on the lobby mean of the CAUSAL values (clean for
    # all 10), so only the tested player's own value changes between rows of
    # the table. Centering on the source's own lobby mean would let the other
    # nine players' leaky values into slot i (teammates share the outcome).
    cmean = V["player_mmr_causal"] - XC["player_mmr_causal"]
    for c in cols:
        XC[c] = V[c] - cmean
    prev_late = FULL & ~pclean_s
    strata = {"all": FULL, "prev game parsed BEFORE this start": FULL & pclean_s,
              "prev game parsed AFTER this start (leak-prone)": prev_late}
    say(f"leak check: {FULL.sum():,} slots in {FULL.sum() // 10:,} full lobbies where causal, fresh, "
        f"prev and own stamps exist for all 10 players; slope and ll gain of r on (own value - lobby "
        f"mean of causal MMR)")
    lk = {}
    for sn, sm in strata.items():
        say(f" stratum {sn}: {sm.sum():,} slots")
        lk[sn] = {}
        for c in cols:
            st = stratum(r[sm], d["wp"][sm], XC[c][sm])
            lk[sn][c] = st
            say(f"  {c:24s} slope/100 {st['slope_per100']:+.4f} (se {st['slope_se']:.4f})  "
                f"ll gain x1000 {st['ll_gain_x1000']:+.3f}")
    # Lobbies where all 10 previous-game stamps were parsed before the start:
    # there the previous stamp is clean too and should tie with at-game MMR.
    allclean = np.bincount(d["g"], weights=pclean_s.astype(float), minlength=n_games)[d["g"]] == 10
    sm = FULL & allclean
    lk["lobbies with all 10 prev stamps parsed before start (own-lobby centering)"] = {}
    say(f" lobbies with all 10 prev stamps clean: {sm.sum() // 10:,} lobbies")
    for c in ("player_mmr_causal", "player_mmr_fresh", "prev_stamp_mmr", "own_stamp_mmr"):
        xc, _ = centered(V[c])
        st = stratum(r[sm], d["wp"][sm], xc[sm])
        lk["lobbies with all 10 prev stamps parsed before start (own-lobby centering)"][c] = st
        say(f"  {c:24s} slope/100 {st['slope_per100']:+.4f} (se {st['slope_se']:.4f})  "
            f"ll gain x1000 {st['ll_gain_x1000']:+.3f}")
    # joint regressions: does at-game MMR add anything beyond the previous
    # game's stamp (it should not), and does the previous stamp add beyond
    # at-game MMR in the leak-prone stratum (it will, if it leaks)?
    for sn, sm in strata.items():
        for a_, b_ in (("player_mmr_causal", "prev_stamp_mmr"), ("player_mmr_fresh", "prev_stamp_mmr")):
            X = np.stack([XC[a_][sm], XC[b_][sm], np.ones(sm.sum())], 1)
            beta, *_ = np.linalg.lstsq(X, r[sm], rcond=None)
            resid = r[sm] - X @ beta
            cov_b = np.linalg.inv(X.T @ X) * resid.var()
            lk[sn][f"joint {a_} + {b_}"] = {"b_a": float(100 * beta[0]), "se_a": float(100 * np.sqrt(cov_b[0, 0])),
                                            "b_b": float(100 * beta[1]), "se_b": float(100 * np.sqrt(cov_b[1, 1]))}
            say(f"  [{sn}] joint: {a_} {100 * beta[0]:+.4f} ({100 * np.sqrt(cov_b[0, 0]):.4f})  "
                f"{b_} {100 * beta[1]:+.4f} ({100 * np.sqrt(cov_b[1, 1]):.4f})")
    # hero / role: full lobbies for that source only
    for c in ("hero_mmr_causal", "role_mmr_causal"):
        xc, f = centered(V[c])
        st = stratum(r[f], d["wp"][f], xc[f])
        lk[c] = st
        say(f"  {c} ({f.sum():,} slots, own full lobbies) slope/100 {st['slope_per100']:+.4f} "
            f"(se {st['slope_se']:.4f})  ll gain x1000 {st['ll_gain_x1000']:+.3f}")
    # where fresh actually differs from the clean-prev value: late-uploaded
    # neighbours are the risky cases
    res["leak_check"] = lk
    res["elapsed_s"] = (datetime.datetime.now() - t0).total_seconds()
    with open(os.path.join(C.RESULTS, "p3_mmr_at_game.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(C.RESULTS, "p3_mmr_at_game.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


def stratum(r, wp, x):
    x = x - x.mean()
    b = np.cov(r, x)[0, 1] / x.var()
    a = r.mean()
    pred = a + b * x
    y = r + wp
    p1 = np.clip(wp + pred, 1e-3, 1 - 1e-3)
    ll0 = -(y * np.log(wp) + (1 - y) * np.log(1 - wp))
    ll1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    se = np.sqrt(np.mean((r - pred) ** 2) / (len(x) * x.var()))
    return {"n": int(len(r)), "slope_per100": float(100 * b), "slope_se": float(100 * se),
            "corr": float(np.corrcoef(r, x)[0, 1]),
            "ll_gain_x1000": float(1000 * (ll0 - ll1).mean())}


if __name__ == "__main__":
    main()
