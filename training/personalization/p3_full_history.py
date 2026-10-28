"""
P3 merged as-of history store: our corpus player rows UNION the Heroes
Profile per-match Storm League histories, one row per (player, match).

Sources
  corpus  cache/export/<tag>/stamps.csv.gz (p3_export.py stamps: every
          replay_players row of a stored SL draft, local research store)
  hp      cache/mmr_history/shards/*.jsonl.gz (sync/fetch-mmr-history.ts:
          the account's full SL history from HP, 2019 on)
Player key = (region, blizz_id) as p3_keys.player_keys (region << 40 | blizz).
Rows are deduped on (key, replay_id); a match in both sources keeps the
corpus values (hero, win, game length) with HP's end time, and gets src = 3.
Corpus-only rows from replay id CLOCK_FIX_RID on get their end time shifted
back by the US Eastern offset (corpus clock bug, see CLOCK_FIX_RID).

Cap: every row with game end on or after 2026-09-28 is dropped while
parsing, before anything is stored. That is the P3 snapshot rule (no game
from a build released after 2026-09-27) and it keeps every 2.57 game, so the
scilearn C, F, R and pool tranches and the sealed replays, out of the store.
The raw HP shards do hold such rows (pulled Sep 30 on); nothing else here
reads them.

As-of rule (asof_counts, p3_rows): history for a scored game at time t uses
only rows with end_ts < t (strictly), and never the scored game's own row
(matched by replay_id, so an HP row whose game_date is offset from our end
time cannot leak its own label). The P3 GP state adds rows on whole days
(day <= query day - lag), which is stricter.

Late uploads: HP replay ids are assigned at upload, so the id says roughly
when a match became visible. seen_ts = max(end_ts, upload time estimated
from the id: the 95th percentile game end among ~2,000 neighbouring ids,
made monotone in the id). rule="seen" adds seen_ts < t. HP histories were
pulled long after most games, so the default rule ("game") counts a match
played before t even if it was uploaded after t; `late` reports how often.

Usage (from training/, nice 19, <= 4 cores on the main box):
  nice -n 19 taskset -c 44-47 python3 personalization/p3_full_history.py build --tag oct05
  nice -n 19 taskset -c 44-47 python3 personalization/p3_full_history.py stats --tag oct05
Output: cache/full_history/<tag>/history.npz, meta.json (no battletags);
results/p3_full_history_<tag>.json/.txt (stats)
"""
import argparse
import csv
import datetime
import glob
import gzip
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from p3_keys import REGION_SHIFT, player_keys

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("P3_CACHE") or os.path.join(HERE, "cache")
RESULTS = os.environ.get("P3_RESULTS") or os.path.join(HERE, "results")
SHARDS = os.path.join(HERE, "cache", "mmr_history", "shards")
DONE = os.path.join(HERE, "cache", "mmr_history", "done.jsonl")
# P3_CAP_DATE (YYYY-MM-DD) moves the cap for the post-freeze confirmation build only
_CAP = os.environ.get("P3_CAP_DATE", "2026-09-28")
CAP_TS = int(datetime.datetime.strptime(_CAP, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc).timestamp())
SRC_CORPUS, SRC_HP = 1, 2
# Corpus clock bug (found 2026-10-16): for replay ids >= CLOCK_FIX_RID (games
# from about 2026-07-09 on), replay_draft_data.game_date is the true UTC end
# PLUS the US Eastern offset (4 h in summer, 5 h in winter). Evidence on
# 11.1M (player, match) pairs in both sources: below the id the HP and corpus
# times agree on every pair; from it on they differ by exactly the Eastern
# offset on every pair, and our own fetch time precedes the corpus end on
# 20.6% of those rows (0.03% against the HP time), so HP's game_date is the
# right clock. The store uses HP's time for pairs and shifts corpus-only rows
# from this id on.
# DB corrected 2026-10-16 (sync/docs/timezone-fix-2026-10.md): the bug follows
# the fetch time (daemon restart 2026-07-09 20:22:44 UTC), so the first shifted
# id is 64264412 (64264402..64264408 were stored by the old process and are
# right). Exports made after the correction are already UTC; merge() checks the
# pairs and shifts only when the export still carries the offset.
CLOCK_FIX_RID = 64264412
SEEN_BIN = 2000          # replay ids per upload-time bin
SEEN_Q = 95              # percentile of game end within a bin = upload time
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def hero_index():
    import p3_heroes  # noqa: F401  (puts training/ on the path, asserts the v1 set)
    import shared
    return {h: i for i, h in enumerate(shared.HEROES)}, list(shared.HEROES)


def eastern_offset(utc_epoch):
    """Seconds US Eastern is behind UTC (14400 or 18000) at each time."""
    from zoneinfo import ZoneInfo
    ny = ZoneInfo("America/New_York")
    t = np.asarray(utc_epoch, np.int64)
    days = t // 86400
    out = np.empty(len(t), np.int64)
    ud, inv = np.unique(days, return_inverse=True)
    offs = np.array([-ny.utcoffset(datetime.datetime(1970, 1, 1) + datetime.timedelta(days=int(d), hours=12))
                     .total_seconds() for d in ud], np.int64)
    out[:] = offs[inv]
    return out


def _naive_epoch(strs):
    return np.array(strs, dtype="datetime64[s]").astype(np.int64)


# ------------------------------------------------------------------ parse

def _parse_shard(path):
    """One HP shard -> column arrays (rows with game end before CAP only)."""
    hidx, _ = hero_index()
    out = {k: [] for k in ("region", "blizz", "rid", "end", "hero", "win", "map", "mmr")}
    accounts, n_cap, n_bad, unmapped = [], 0, 0, {}
    try:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("data_source") == "fixture" or not r.get("rows"):
                    if r.get("data_source") != "fixture":
                        accounts.append((r["region"], r["blizz_id"], r.get("fetched_at", "")))
                    continue
                accounts.append((r["region"], r["blizz_id"], r.get("fetched_at", "")))
                ci = {c: i for i, c in enumerate(r["cols"])}
                rows = r["rows"]
                gd = [row[ci["game_date"]] for row in rows]
                ok = [g is not None and row[ci["replayID"]] is not None for g, row in zip(gd, rows)]
                n_bad += len(rows) - sum(ok)
                rows = [row for row, o in zip(rows, ok) if o]
                if not rows:
                    continue
                end = _naive_epoch([row[ci["game_date"]] for row in rows])
                keep = end < CAP_TS
                n_cap += int((~keep).sum())
                idx = np.flatnonzero(keep)
                if not len(idx):
                    continue
                n = len(idx)
                out["region"].append(np.full(n, r["region"], np.int64))
                out["blizz"].append(np.full(n, r["blizz_id"], np.int64))
                out["rid"].append(np.array([rows[i][ci["replayID"]] for i in idx], np.int64))
                out["end"].append(end[idx])
                hs = []
                for i in idx:
                    h = rows[i][ci["hero"]]
                    j = hidx.get(h, -1)
                    if j < 0:
                        unmapped[h] = unmapped.get(h, 0) + 1
                    hs.append(j)
                out["hero"].append(np.array(hs, np.int8))
                out["win"].append(np.array([str(rows[i][ci["winner"]]).lower() in ("true", "1") for i in idx], np.int8))
                out["map"].append(np.array([rows[i][ci["game_map"]] if rows[i][ci["game_map"]] is not None else -1
                                            for i in idx], np.int16))
                mm = [rows[i][ci["mmr"]] for i in idx]
                out["mmr"].append(np.array([np.nan if m is None else m for m in mm], np.float32))
    except EOFError:
        pass  # shard of a killed run: records before the last flush are intact
    cat = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}
    return cat, accounts, n_cap, n_bad, unmapped


def load_hp(shard_glob=None, procs=3):
    files = sorted(glob.glob(shard_glob or os.path.join(SHARDS, "*.jsonl.gz")))
    if not files:
        raise SystemExit(f"no HP shards under {SHARDS}")
    if procs > 1 and len(files) > 1:
        from multiprocessing import Pool
        with Pool(min(procs, len(files))) as pool:
            parts = pool.map(_parse_shard, files)
    else:
        parts = [_parse_shard(f) for f in files]
    cols = {k: np.concatenate([p[0][k] for p in parts if len(p[0]["rid"])]) for k in parts[0][0]}
    accounts = [a for p in parts for a in p[1]]
    n_cap = sum(p[2] for p in parts)
    n_bad = sum(p[3] for p in parts)
    unmapped = {}
    for p in parts:
        for h, n in p[4].items():
            unmapped[h] = unmapped.get(h, 0) + n
    say(f"HP shards {len(files)}: {len(cols['rid']):,} rows before the cap, {n_cap:,} rows on/after "
        f"2026-09-28 dropped unread, {n_bad} rows without date or id; accounts {len(accounts):,}; "
        f"unmapped heroes {unmapped}")
    return cols, accounts


def load_corpus(tag):
    """Export stamps -> column arrays (already capped by the export query)."""
    path = os.path.join(CACHE, "export", tag, "stamps.csv.gz")
    hidx, _ = hero_index()
    rid, bid, reg, hero, end, gl, win, mmr = [], [], [], [], [], [], [], []
    unmapped = {}
    with gzip.open(path, "rt", newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        ix = {c: hdr.index(c) for c in ("replay_id", "blizz_id", "region", "hero", "end_ts",
                                        "game_length", "winner", "player_mmr")}
        for r in rd:
            if not r[ix["blizz_id"]]:
                continue
            rid.append(int(r[ix["replay_id"]]))
            bid.append(int(r[ix["blizz_id"]]))
            reg.append(int(r[ix["region"]]) if r[ix["region"]] else 0)
            h = r[ix["hero"]]
            j = hidx.get(h, -1)
            if j < 0:
                unmapped[h] = unmapped.get(h, 0) + 1
            hero.append(j)
            end.append(int(r[ix["end_ts"]]))
            gl.append(int(r[ix["game_length"]]) if r[ix["game_length"]] else -1)
            win.append(1 if r[ix["winner"]] == "1" else 0)
            mmr.append(float(r[ix["player_mmr"]]) if r[ix["player_mmr"]] else np.nan)
    c = {"rid": np.array(rid, np.int64), "blizz": np.array(bid, np.int64), "region": np.array(reg, np.int64),
         "hero": np.array(hero, np.int8), "end": np.array(end, np.int64), "gl": np.array(gl, np.int32),
         "win": np.array(win, np.int8), "mmr": np.array(mmr, np.float32)}
    keep = c["end"] < CAP_TS
    c = {k: v[keep] for k, v in c.items()}
    say(f"corpus export {tag}: {len(c['rid']):,} player rows (unmapped heroes {unmapped})")
    return c


# ------------------------------------------------------------------ merge

def merge(corpus, hp):
    """Union deduped on (key, replay_id). Corpus values win except the end
    time (HP's clock, see CLOCK_FIX_RID); src bits OR.
    Both inputs are dicts of arrays with rid, region, blizz, hero, end, win,
    mmr (corpus also gl; hp also map). Returns rows sorted by (key, end, rid)."""
    nc, nh = len(corpus["rid"]), len(hp["rid"])
    kc = player_keys(corpus["region"], corpus["blizz"], label="corpus")
    kh = player_keys(hp["region"], hp["blizz"], label="hp")
    key = np.concatenate([kc, kh])
    rid = np.concatenate([corpus["rid"], hp["rid"]])
    src = np.r_[np.full(nc, SRC_CORPUS, np.int8), np.full(nh, SRC_HP, np.int8)]
    o = np.lexsort((src, rid, key))          # corpus first within a (key, rid) pair
    ks, rs = key[o], rid[o]
    first = np.r_[True, (ks[1:] != ks[:-1]) | (rs[1:] != rs[:-1])]
    grp = np.cumsum(first) - 1
    keep = o[first]
    # src bits per group, and the HP row of each pair (for agreement checks)
    bits = np.zeros(len(keep), np.int8)
    np.bitwise_or.at(bits, grp, src[o])
    hp_of = np.full(len(keep), -1, np.int64)
    is_hp = src[o] == SRC_HP
    hp_of[grp[is_hp]] = o[is_hp] - nc
    both = bits == 3
    ci = keep[both]
    hi = hp_of[both]
    agree = {}
    if both.any():
        agree = {"pairs": int(both.sum()),
                 "winner_disagree": int((corpus["win"][ci] != hp["win"][hi]).sum()),
                 "hero_disagree": int(((corpus["hero"][ci] != hp["hero"][hi]) & (hp["hero"][hi] >= 0)).sum())}
        off = hp["end"][hi] - corpus["end"][ci]
        agree.update({"hp_minus_corpus_end_s_median": float(np.median(off)),
                      "p1": float(np.percentile(off, 1)), "p99": float(np.percentile(off, 99))})
    say(f"merge: corpus {nc:,} + hp {nh:,} -> {len(keep):,} rows "
        f"(both {int(both.sum()):,}, corpus only {int((bits == 1).sum()):,}, hp only {int((bits == 2).sum()):,}); "
        f"agreement {agree}")

    def col(name, dtype, fill):
        a = np.concatenate([corpus.get(name, np.full(nc, fill)), hp.get(name, np.full(nh, fill))]).astype(dtype)
        return a[keep]
    m = {"key": key[keep], "replay_id": rid[keep], "end_ts": col("end", np.int64, 0),
         "game_length": col("gl", np.int32, -1), "hero": col("hero", np.int8, -1),
         "win": col("win", np.int8, 0), "hp_map": col("map", np.int16, -1),
         "post_mmr": col("mmr", np.float32, np.nan), "src": bits}
    # clock: a pair takes HP's end time; a corpus-only row from CLOCK_FIX_RID
    # on is shifted back by the Eastern offset (see CLOCK_FIX_RID)
    m["end_ts"][both] = hp["end"][hi]
    m["hp_map"][both] = hp["map"][hi] if "map" in hp else -1
    fix = (bits == SRC_CORPUS) & (m["replay_id"] >= CLOCK_FIX_RID)
    hp_new_pairs = hp["rid"][hi] >= CLOCK_FIX_RID
    export_shifted = bool(hp_new_pairs.any() and
                          (hp["end"][hi][hp_new_pairs] != corpus["end"][ci][hp_new_pairs]).mean() > 0.5)
    agree["export_clock_shifted"] = export_shifted
    if not export_shifted:
        fix[:] = False
    if fix.any():
        e = m["end_ts"][fix]
        m["end_ts"][fix] = e - eastern_offset(e - 14400)
    if both.any():
        hi_new = hp["rid"][hi] >= CLOCK_FIX_RID
        off = hp["end"][hi] - corpus["end"][ci]
        agree["pairs_from_fix_id_with_eastern_offset"] = float(
            (off[hi_new] == -eastern_offset(hp["end"][hi][hi_new])).mean()) if hi_new.any() else None
        agree["pairs_below_fix_id_exact"] = float((off[~hi_new] == 0).mean()) if (~hi_new).any() else None
    agree["corpus_only_rows_clock_fixed"] = int(fix.sum())
    say(f"clock fix: {int(fix.sum()):,} corpus-only rows shifted; pair checks {agree}")
    s = np.lexsort((m["replay_id"], m["end_ts"], m["key"]))
    return {k: v[s] for k, v in m.items()}, agree


def seen_times(rid, end_ts, bin_rows=SEEN_BIN, q=SEEN_Q):
    """Upload-time estimate per row from its replay id. Unique replays are
    cut into bins of bin_rows consecutive ids; a bin's upload time is the
    q-th percentile of game end in it, then a running max over bins (so the
    top id range, uploaded after the cap and holding only late uploads of
    older games, stays at the cap-era level). seen = max(end, upload)."""
    ur, ui = np.unique(rid, return_index=True)
    ue = end_ts[ui]
    nb = len(ur) // bin_rows
    if nb == 0:
        return end_ts.copy(), {}
    edges = ur[::bin_rows][:nb]
    up = np.percentile(ue[:nb * bin_rows].reshape(nb, bin_rows), q, axis=1)
    up = np.maximum.accumulate(up).astype(np.int64)
    b = np.searchsorted(edges, rid, side="right") - 1
    u = np.where(b >= 0, up[np.maximum(b, 0)], end_ts)
    span_h = np.diff(up) / 3600.0
    return np.maximum(end_ts, u), {"bins": int(nb), "bin_rows": bin_rows, "q": q,
                                   "bin_span_hours_median": float(np.median(span_h)) if len(span_h) else None}


def build(tag, out=None, shard_glob=None, procs=3):
    t0 = time.time()
    hp, accounts = load_hp(shard_glob, procs)
    corpus = load_corpus(tag)
    st, agree = merge(corpus, hp)
    del hp, corpus
    st["seen_ts"], seen_meta = seen_times(st["replay_id"], st["end_ts"])
    keys, ptr = np.unique(st["key"], return_index=True)
    ptr = np.r_[ptr, len(st["key"])].astype(np.int64)
    # accounts whose full HP history is in the store (status ok in done.jsonl)
    ok = set()
    if os.path.exists(DONE):
        for line in open(DONE):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("status") == "ok":
                ok.add((int(d["region"]), int(d["blizz_id"])))
    pulled = np.unique(np.array([(int(r) << REGION_SHIFT) | int(b) for r, b, _ in accounts
                                 if (int(r), int(b)) in ok], np.int64))
    out = out or os.path.join(CACHE, "full_history", tag)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "history.npz")
    np.savez(path + ".tmp.npz", keys=keys, ptr=ptr, pulled_keys=pulled, hero_names=np.array(hero_index()[1]),
             **{k: v for k, v in st.items() if k != "key"})
    os.replace(path + ".tmp.npz", path)
    meta = {"tag": tag, "rows": int(len(st["replay_id"])), "players": int(len(keys)),
            "pulled_players": int(len(pulled)), "cap_ts": CAP_TS,
            "src_counts": {"corpus_only": int((st["src"] == 1).sum()), "hp_only": int((st["src"] == 2).sum()),
                           "both": int((st["src"] == 3).sum())},
            "agreement": agree, "seen": seen_meta, "built_at": datetime.datetime.now(
                datetime.timezone.utc).isoformat(timespec="seconds"), "seconds": round(time.time() - t0),
            "columns": "keys, ptr (CSR over rows sorted by key, end_ts, replay_id), pulled_keys; per row: "
                       "replay_id, end_ts, game_length, hero (shared.HEROES v1 index, -1 unknown), win, hp_map, "
                       "post_mmr (HP POST-game player MMR of that row: never use a row's own value for that "
                       "game), src (1 corpus, 2 hp, 3 both), seen_ts"}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    say(f"wrote {path}: {meta['rows']:,} rows, {meta['players']:,} players, {meta['pulled_players']:,} pulled "
        f"({time.time() - t0:.0f}s)")
    return path


# ------------------------------------------------------------------ read

class Store:
    """Merged history, with lazily built sort orders for as-of prefix sums."""

    def __init__(self, arrays):
        self.a = arrays
        self.keys = arrays["keys"]
        self.ptr = arrays["ptr"]
        self.n = len(arrays["replay_id"])
        self.kidx = np.repeat(np.arange(len(self.keys), dtype=np.int64), np.diff(self.ptr))
        self._ord = {}
        self._own = None

    @classmethod
    def load(cls, tag_or_path):
        p = tag_or_path if tag_or_path.endswith(".npz") else os.path.join(CACHE, "full_history", tag_or_path, "history.npz")
        z = np.load(p)
        a = {k: z[k] for k in z.files}
        a.setdefault("hero_names", np.array(hero_index()[1]))
        return cls(a)

    @classmethod
    def from_rows(cls, key, replay_id, end_ts, hero, win, src, seen_ts=None, **extra):
        """A store from already-merged rows (tests and small builds)."""
        s = np.lexsort((replay_id, end_ts, key))
        a = {"replay_id": np.asarray(replay_id, np.int64)[s], "end_ts": np.asarray(end_ts, np.int64)[s],
             "hero": np.asarray(hero, np.int8)[s], "win": np.asarray(win, np.int8)[s],
             "src": np.asarray(src, np.int8)[s]}
        a["seen_ts"] = a["end_ts"].copy() if seen_ts is None else np.maximum(np.asarray(seen_ts, np.int64)[s], a["end_ts"])
        for k, v in extra.items():
            a[k] = np.asarray(v)[s]
        k = np.asarray(key, np.int64)[s]
        keys, ptr = np.unique(k, return_index=True)
        a["keys"], a["ptr"] = keys, np.r_[ptr, len(k)].astype(np.int64)
        a.setdefault("pulled_keys", keys)
        a.setdefault("hero_names", np.array(hero_index()[1]))
        return cls(a)

    def _order(self, by_hero, ts_col):
        """Rows sorted by (key, [hero,] ts) with the composite sort key and
        prefix sums of games, wins and corpus rows."""
        name = (by_hero, ts_col)
        if name not in self._ord:
            ts = self.a[ts_col]
            grp = self.kidx * 128 + (self.a["hero"].astype(np.int64) + 1) if by_hero else self.kidx
            o = np.lexsort((ts, grp))
            comp = (grp[o] << 34) | ts[o]
            pre = lambda x: np.r_[0, np.cumsum(x[o], dtype=np.int64)]
            self._ord[name] = (comp, pre(np.ones(self.n, np.int64)), pre(self.a["win"].astype(np.int64)),
                               pre((self.a["src"] & SRC_CORPUS).astype(np.int64)))
        return self._ord[name]

    def hero_map(self, hero_names):
        """Index array: store hero index -> position in hero_names (the
        caller's order, e.g. hs_slots hero_names); -1 if absent."""
        pos = {str(h): i for i, h in enumerate(hero_names)}
        return np.array([pos.get(str(h), -1) for h in self.a["hero_names"]], np.int64)

    def to_store_heroes(self, hero, hero_names):
        """Caller-order hero indices -> store (shared.HEROES) indices."""
        pos = {str(h): i for i, h in enumerate(self.a["hero_names"])}
        m = np.array([pos[str(h)] for h in hero_names], np.int64)
        return m[np.asarray(hero, np.int64)]

    def key_index(self, qkey):
        qkey = np.asarray(qkey, np.int64)
        j = np.minimum(np.searchsorted(self.keys, qkey), len(self.keys) - 1)
        return np.where(self.keys[j] == qkey, j, -1)

    def own_row(self, qkey, qrid):
        """Row index of (key, replay_id) or -1."""
        if self._own is None:
            comp = (self.kidx << 32) | self.a["replay_id"]
            o = np.argsort(comp, kind="stable")
            self._own = (comp[o], o)
        comp, o = self._own
        k = self.key_index(qkey)
        q = (np.maximum(k, 0) << 32) | np.asarray(qrid, np.int64)
        j = np.minimum(np.searchsorted(comp, q), len(comp) - 1)
        return np.where((k >= 0) & (comp[j] == q), o[j], -1)

    def asof_counts(self, qkey, qts, qrid, qhero=None, rule="game"):
        """Per query: games, wins, corpus games (n_corpus = what the thin
        corpus history would show), and the same on qhero, from rows with
        end_ts < qts (rule "game") or also seen_ts < qts (rule "seen"),
        always excluding the query's own replay_id. qhero is in the store's
        hero order (shared.HEROES); convert with to_store_heroes."""
        # "game": end_ts < t; "seen": seen_ts < t (seen >= end, so it implies
        # end < t); any other name: that column of the store, which must be
        # >= end_ts row by row (e.g. a tolerance variant of seen_ts)
        ts_col = {"game": "end_ts", "seen": "seen_ts"}.get(rule, rule)
        assert ts_col in self.a and (ts_col == "end_ts" or (self.a[ts_col] >= self.a["end_ts"]).all())
        qkey = np.asarray(qkey, np.int64)
        qts = np.asarray(qts, np.int64)
        k = self.key_index(qkey)
        has = k >= 0
        kk = np.maximum(k, 0)
        out = {}
        groups = [("", False, kk)]
        if qhero is not None:
            qh = np.asarray(qhero, np.int64)
            groups.append(("_hero", True, kk * 128 + qh + 1))
        own = self.own_row(qkey, qrid)
        own_in = (own >= 0) & (self.a[ts_col][np.maximum(own, 0)] < qts)
        own_win = np.where(own_in, self.a["win"][np.maximum(own, 0)], 0)
        own_cor = np.where(own_in, self.a["src"][np.maximum(own, 0)] & SRC_CORPUS, 0)
        for sfx, by_hero, grp in groups:
            comp, cn, cw, cc = self._order(by_hero, ts_col)
            lo = np.searchsorted(comp, grp << 34, side="left")
            hi = np.searchsorted(comp, (grp << 34) | qts, side="left")
            sub = own_in if not by_hero else own_in & (self.a["hero"][np.maximum(own, 0)] == qh)
            n = np.where(has, cn[hi] - cn[lo], 0) - sub
            w = np.where(has, cw[hi] - cw[lo], 0) - np.where(sub, own_win, 0)
            c = np.where(has, cc[hi] - cc[lo], 0) - np.where(sub, own_cor, 0)
            out["n" + sfx], out["w" + sfx], out["n_corpus" + sfx] = n, w, c
        return out

    def p3_rows(self, player_keys, hero_names, src_mask=SRC_HP, exclude_src=SRC_CORPUS):
        """History rows for P3's online state, mapped to P3 pid indices
        (player_keys = hs_slots player_keys, sorted). Default: HP-only rows
        (the corpus rows are already P3 slot rows). hero comes back in the
        order of hero_names (the slot table's). Returns pid, hero, end_ts,
        day, y, seen_ts, sorted by end_ts."""
        pk = np.asarray(player_keys, np.int64)
        sel = ((self.a["src"] & src_mask) > 0) & ((self.a["src"] & exclude_src) == 0) & (self.a["hero"] >= 0)
        key = self.keys[self.kidx]
        j = np.minimum(np.searchsorted(pk, key), len(pk) - 1)
        sel &= pk[j] == key
        idx = np.flatnonzero(sel)
        idx = idx[np.argsort(self.a["end_ts"][idx], kind="stable")]
        hm = self.hero_map(hero_names)
        assert (hm >= 0).all(), "slot hero list misses a store hero"
        return {"pid": j[idx].astype(np.int64), "hero": hm[self.a["hero"][idx].astype(np.int64)],
                "end_ts": self.a["end_ts"][idx], "day": (self.a["end_ts"][idx] // 86400).astype(np.int32),
                "y": self.a["win"][idx].astype(np.float32), "seen_ts": self.a["seen_ts"][idx],
                "replay_id": self.a["replay_id"][idx]}


# ------------------------------------------------------------------ stats

def stats(tag):
    st = Store.load(tag)
    a = st.a
    out = {"tag": tag}
    key_of_row = st.keys[st.kidx]
    pulled = np.isin(key_of_row, a["pulled_keys"])
    # coverage per pulled player: all games before the cap, full vs corpus
    kp = st.kidx[pulled]
    n_full = np.bincount(kp, minlength=len(st.keys))
    n_cor = np.bincount(kp, weights=(a["src"][pulled] & 1) > 0, minlength=len(st.keys))
    pre25 = a["end_ts"][pulled] < int(datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc).timestamp())
    n_full_pre = np.bincount(kp[pre25], minlength=len(st.keys))
    n_cor_pre = np.bincount(kp[pre25], weights=(a["src"][pulled][pre25] & 1) > 0, minlength=len(st.keys))
    m = n_full > 0
    ratio = n_full[m] / np.maximum(n_cor[m], 1)
    out["pulled_players"] = int(m.sum())
    out["career_games_full"] = {"median": float(np.median(n_full[m])), "p10": float(np.percentile(n_full[m], 10)),
                                "p90": float(np.percentile(n_full[m], 90)), "total": int(n_full[m].sum())}
    out["career_games_corpus"] = {"median": float(np.median(n_cor[m])), "total": int(n_cor[m].sum())}
    out["ratio_full_to_corpus"] = {"median": float(np.median(ratio)), "p25": float(np.percentile(ratio, 25)),
                                   "p75": float(np.percentile(ratio, 75)),
                                   "pooled": float(n_full[m].sum() / max(n_cor[m].sum(), 1))}
    out["corpus_share_of_career"] = {"median": float(np.median(n_cor[m] / n_full[m])),
                                     "before_2025_pooled": float(n_cor_pre.sum() / max(n_full_pre.sum(), 1)),
                                     "since_2025_pooled": float((n_cor[m].sum() - n_cor_pre.sum()) /
                                                                max(n_full[m].sum() - n_full_pre.sum(), 1))}
    yrs = (a["end_ts"][pulled] // 86400).astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970
    out["pulled_rows_by_year"] = {int(y): {"full": int((yrs == y).sum()),
                                           "corpus": int(((yrs == y) & ((a["src"][pulled] & 1) > 0)).sum())}
                                  for y in np.unique(yrs)}
    # late uploads: one row per replay
    ur, ui = np.unique(a["replay_id"], return_index=True)
    lag = (a["seen_ts"][ui] - a["end_ts"][ui]) / 86400.0
    yr = (a["end_ts"][ui] // 86400).astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970
    out["upload_lag_days"] = {"replays": int(len(ur)), "median": float(np.median(lag)),
                              **{f"share_over_{d}d": float((lag > d).mean()) for d in (1, 7, 30, 365)},
                              "by_year_share_over_7d": {int(y): float((lag[yr == y] > 7).mean()) for y in np.unique(yr)}}
    srcu = a["src"][ui]
    for nm, msk in (("corpus_replays", (srcu & 1) > 0), ("hp_only_replays", srcu == 2)):
        out["upload_lag_days"][nm] = {"replays": int(msk.sum()), "median": float(np.median(lag[msk])),
                                      **{f"share_over_{d}d": float((lag[msk] > d).mean()) for d in (1, 7, 30, 365)}}
    say(json.dumps(out, indent=1))
    return st, out


def scored_stats(st, out, slots_path=None):
    """On P3's scored slots (hs_slots.npz): as-of history of pulled players
    full vs corpus, the late-upload share of that history, and lobbies with
    all 10 players pulled."""
    slots_path = slots_path or os.path.join(CACHE, "hs_slots.npz")
    if not os.path.exists(slots_path):
        say(f"no {slots_path}; skipping scored-slot stats")
        return out
    z = np.load(slots_path)
    pk = z["player_keys"][z["pid"]]
    gts = os.path.join(CACHE, "gametime_2024q2.npz")
    g = np.load(gts)
    o = np.argsort(g["replay_ids"])
    j = np.minimum(np.searchsorted(g["replay_ids"][o], z["replay_id"]), len(o) - 1)
    okt = g["replay_ids"][o][j] == z["replay_id"]
    end = np.where(okt, g["ts"][o][j], z["day"].astype(np.int64) * 86400 + 43200)
    # gametime_2024q2.npz was built 2026-09-30, before the DB correction, so it still carries the offset
    late_clock = okt & (z["replay_id"] >= CLOCK_FIX_RID)       # corpus clock bug
    end[late_clock] -= eastern_offset(end[late_clock] - 14400)
    gl = np.where(okt, g["game_length"][o][j], -1)
    start = end - np.where(gl > 0, gl, 0)
    pulled = np.isin(pk, st.a["pulled_keys"])
    out["scored_slots"] = int(len(pk))
    out["scored_slots_pulled_share"] = float(pulled.mean())
    sel = np.flatnonzero(pulled)
    rng = np.random.default_rng(0)
    if len(sel) > 2_000_000:
        sel = np.sort(rng.choice(sel, 2_000_000, replace=False))
    hero = st.to_store_heroes(z["hero"], z["hero_names"])
    c = st.asof_counts(pk[sel], start[sel], z["replay_id"][sel], hero[sel])
    cs = st.asof_counts(pk[sel], start[sel], z["replay_id"][sel], hero[sel], rule="seen")
    # tolerance variant: an upload within a day of the game counts as on time
    # (the id bins resolve upload time only to a few hours)
    sa = st.a
    sa["seen1d"] = np.where(sa["seen_ts"] - sa["end_ts"] > 86400, sa["seen_ts"], sa["end_ts"])
    c1 = st.asof_counts(pk[sel], start[sel], z["replay_id"][sel], hero[sel], rule="seen1d")
    r = (c["n"] + 1) / (c["n_corpus"] + 1)
    rh = (c["n_hero"] + 1) / (c["n_corpus_hero"] + 1)
    out["asof_pulled_slots"] = {
        "sampled": int(len(sel)),
        "games_before_median_full": float(np.median(c["n"])), "games_before_median_corpus": float(np.median(c["n_corpus"])),
        "ratio_median_(n+1)/(corpus+1)": float(np.median(r)),
        "hero_games_before_median_full": float(np.median(c["n_hero"])),
        "hero_games_before_median_corpus": float(np.median(c["n_corpus_hero"])),
        "hero_ratio_median": float(np.median(rh)),
        "share_with_50plus_hero_games_full": float((c["n_hero"] >= 50).mean()),
        "share_with_50plus_hero_games_corpus": float((c["n_corpus_hero"] >= 50).mean()),
        "late_share_of_history_rows_(game rule rows not yet uploaded at start)":
            float((c["n"] - cs["n"]).sum() / max(c["n"].sum(), 1)),
        "share_of_slots_with_any_late_row": float(((c["n"] - cs["n"]) > 0).mean()),
        "late_share_over_1d_of_history_rows": float((c["n"] - c1["n"]).sum() / max(c["n"].sum(), 1)),
        "late_share_over_1d_of_hero_rows": float((c["n_hero"] - c1["n_hero"]).sum() / max(c["n_hero"].sum(), 1)),
        "share_of_slots_with_any_late_row_over_1d": float(((c["n"] - c1["n"]) > 0).mean()),
        "corpus_late_share_over_1d": float((c["n_corpus"] - c1["n_corpus"]).sum() / max(c["n_corpus"].sum(), 1)),
    }
    # lobbies with all 10 pulled
    gid = z["replay_id"]
    ug, inv = np.unique(gid, return_inverse=True)
    n_slot = np.bincount(inv)
    n_pull = np.bincount(inv, weights=pulled)
    full10 = (n_slot == 10) & (n_pull == 10)
    day_g = np.zeros(len(ug), np.int64)
    day_g[inv] = z["day"]
    yr = day_g.astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970
    # projection: the pulled set plus the tranche-2 panel once it is pulled
    p2 = os.path.join(HERE, "cache", "mmr_history", "panel2.tsv")
    if os.path.exists(p2):
        rb = np.loadtxt(p2, delimiter="\t", skiprows=1, usecols=(1, 2), dtype=np.int64, ndmin=2)
        more = np.isin(pk, (rb[:, 0] << REGION_SHIFT) | rb[:, 1])
        n_pull2 = np.bincount(inv, weights=pulled | more)
        full10_2 = (n_slot == 10) & (n_pull2 == 10)
        out["lobbies_after_panel2"] = {"all10": int(full10_2.sum()),
                                       "all10_by_year": {int(y): int((full10_2 & (yr == y)).sum()) for y in np.unique(yr)},
                                       "slots_pulled_share": float((pulled | more).mean())}
    out["lobbies"] = {"scored": int(len(ug)), "all10_pulled": int(full10.sum()),
                      "all10_pulled_by_year": {int(y): int((full10 & (yr == y)).sum()) for y in np.unique(yr)},
                      "8plus_pulled": int(((n_slot == 10) & (n_pull >= 8)).sum())}
    say(json.dumps({k: out[k] for k in ("scored_slots", "scored_slots_pulled_share", "asof_pulled_slots", "lobbies",
                                        "lobbies_after_panel2") if k in out}, indent=1))
    return out


def corpus_lag_check(tag):
    """Cross-check of the id-based upload estimate on corpus rows: HP's own
    parse stamp (mmr_date_parsed at our fetch) minus game end."""
    path = os.path.join(CACHE, "export", tag, "stamps.csv.gz")
    rid, end, pts = [], [], []
    seen = set()
    with gzip.open(path, "rt", newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        a, b, c = hdr.index("replay_id"), hdr.index("end_ts"), hdr.index("parse_ts")
        for r in rd:
            if r[a] in seen or not r[c]:
                continue
            seen.add(r[a])
            rid.append(int(r[a]))
            end.append(int(r[b]))
            pts.append(int(r[c]))
    return np.array(rid, np.int64), np.array(end, np.int64), np.array(pts, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["build", "stats"])
    ap.add_argument("--tag", required=True, help="export tag (cache/export/<tag>/stamps.csv.gz)")
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    if a.what == "build":
        build(a.tag, procs=a.procs)
        return
    st, out = stats(a.tag)
    out = scored_stats(st, out)
    # id-based lag vs HP parse stamps on corpus replays
    rid, end, pts = corpus_lag_check(a.tag)
    ur, ui = np.unique(st.a["replay_id"], return_index=True)
    j = np.minimum(np.searchsorted(ur, rid), len(ur) - 1)
    ok = (ur[j] == rid) & (pts > 0)
    est = (st.a["seen_ts"][ui][j[ok]] - end[ok]) / 86400.0
    hp = (pts[ok] - end[ok]) / 86400.0
    out["lag_check_corpus"] = {"replays": int(ok.sum()),
                               "id_estimate_share_over_7d": float((est > 7).mean()),
                               "parse_stamp_share_over_7d": float((hp > 7).mean()),
                               "agree_on_over_7d": float(((est > 7) == (hp > 7)).mean()),
                               "spearman_like_corr": float(np.corrcoef(np.argsort(np.argsort(est)),
                                                                       np.argsort(np.argsort(hp)))[0, 1])}
    say(json.dumps(out["lag_check_corpus"], indent=1))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"p3_full_history_{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(RESULTS, f"p3_full_history_{a.tag}.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
