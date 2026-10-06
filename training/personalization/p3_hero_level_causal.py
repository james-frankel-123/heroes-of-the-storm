"""
P3: hero level as it stood when each game started (parse-order safe).

Heroes Profile stamps hero_level when it parses a replay, after the game, so
a row's own level and a lagged level from a late-parsed game can both carry
experience gained after the game being predicted. As for MMR
(p3_mmr_at_game.py), the only clean value for game g is the stamp of the
player's latest game parsed strictly before g started.

Two quantities per player-game:
  hero   level on this hero: latest (player, hero) stamp parsed before start
  acct   the player's highest level on any hero over stamps parsed before
         start (used for the new-account status)
Levels are kept as an interval [lo, hi]: an integer level gives lo = hi; a
v1-API band string such as "25-50" gives lo = 25, hi = 50; "100+" gives
lo = 100, hi = inf. No stamp gives NaN (unknown), never a sentinel level.

Stages (from training/):
  parse   cache/export/<tag>/stamps.csv.gz -> stamps.npz (numeric)
  build   per-slot arrays for hs_slots.npz (snapshot) and x_slots_ext.npz
          (snapshot + post) -> cache/hero_level_causal_{snap,ext}.npz
Usage: python3 personalization/p3_hero_level_causal.py parse --tag oct05
       python3 personalization/p3_hero_level_causal.py build --tag oct05
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C
from p3_keys import player_keys

OUT = {"snap": os.path.join(C.CACHE, "hero_level_causal_snap.npz"),
       "ext": os.path.join(C.CACHE, "hero_level_causal_ext.npz")}


def band(s):
    """(lo, hi) from a level band string; (nan, nan) if unparseable."""
    if not isinstance(s, str) or not s.strip():
        return np.nan, np.nan
    s = s.strip()
    try:
        if s.endswith("+"):
            return float(s[:-1]), np.inf
        if "-" in s:
            a, b = s.split("-", 1)
            return float(a), float(b)
        return float(s), float(s)
    except ValueError:
        return np.nan, np.nan


def parse(tag):
    """Stream the CSV with the csv module (the worker venv has no pandas)."""
    import csv
    import gzip
    d = os.path.join(C.CACHE, "export", tag)
    names = [str(h) for h in np.load(C.PLAYERS)["hero_names"]]
    hidx = {h: i for i, h in enumerate(names)}
    keys = ("replay_ids", "blizz_ids", "region", "region_was_null", "hero", "lo", "hi",
            "parse_ts", "end_ts", "game_length", "fetched_ts", "team", "party", "winner",
            "player_mmr", "hero_mmr", "role_mmr")
    cols = {k: [] for k in keys}
    bands, unknown_heroes = {}, {}
    num = lambda x, dflt: dflt if x == "" else x
    with gzip.open(os.path.join(d, "stamps.csv.gz"), "rt", newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        c = {k: hdr.index(k) for k in ("replay_id", "blizz_id", "region", "region_was_null", "hero",
                                         "hero_level", "hero_level_band", "parse_ts", "end_ts",
                                         "game_length", "fetched_ts", "team", "party", "winner",
                                         "player_mmr", "hero_mmr", "role_mmr")}
        n = 0
        for r in rd:
            if r[c["blizz_id"]] == "":
                continue
            lv = r[c["hero_level"]]
            if lv != "":
                lo = hi = float(lv)
            else:
                b = r[c["hero_level_band"]]
                if b not in bands:
                    bands[b] = band(b)
                lo, hi = bands[b]
            h = r[c["hero"]]
            hi_ = hidx.get(h, -1)
            if hi_ < 0:
                unknown_heroes[h] = unknown_heroes.get(h, 0) + 1
            cols["replay_ids"].append(int(r[c["replay_id"]]))
            cols["blizz_ids"].append(int(r[c["blizz_id"]]))
            cols["region"].append(int(num(r[c["region"]], 0)))
            cols["region_was_null"].append(int(num(r[c["region_was_null"]], 0)))
            cols["hero"].append(hi_)
            cols["lo"].append(lo)
            cols["hi"].append(hi)
            cols["parse_ts"].append(int(num(r[c["parse_ts"]], -1)))
            cols["end_ts"].append(int(r[c["end_ts"]]))
            cols["game_length"].append(int(num(r[c["game_length"]], -1)))
            cols["fetched_ts"].append(int(num(r[c["fetched_ts"]], -1)))
            cols["team"].append(int(num(r[c["team"]], -1)))
            cols["party"].append(int(num(r[c["party"]], 0)))
            cols["winner"].append(int(num(r[c["winner"]], -1)))
            for k in ("player_mmr", "hero_mmr", "role_mmr"):
                cols[k].append(float(num(r[c[k]], "nan")))
            n += 1
            if n % 2_000_000 == 0:
                print(f"  {n:,} rows", flush=True)
    dt = {"replay_ids": np.int64, "blizz_ids": np.int64, "region": np.int16, "region_was_null": np.int8,
          "hero": np.int16, "lo": np.float32, "hi": np.float32, "parse_ts": np.int64, "end_ts": np.int64,
          "game_length": np.int32, "fetched_ts": np.int64, "team": np.int8, "party": np.int64,
          "winner": np.int8, "player_mmr": np.float32, "hero_mmr": np.float32, "role_mmr": np.float32}
    z = {k: np.array(v, dt[k]) for k, v in cols.items()}
    del cols
    z["hero_names"] = np.array(names)
    np.savez(os.path.join(d, "stamps.tmp.npz"), **z)
    os.replace(os.path.join(d, "stamps.tmp.npz"), os.path.join(d, "stamps.npz"))
    info = {"rows": int(len(z["replay_ids"])), "bands": {k: list(v) for k, v in bands.items()},
            "unknown_heroes": unknown_heroes, "no_level": int(np.isnan(z["lo"]).sum()),
            "band_rows": int((~np.isnan(z["lo"]) & (z["lo"] != z["hi"])).sum()),
            "no_parse_ts": int((z["parse_ts"] < 0).sum()), "region_was_null": int(z["region_was_null"].sum())}
    with open(os.path.join(d, "stamps_parse.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "bands"}), f"bands {len(bands)}")


def latest_before(group, ts, value, q_group, q_ts):
    """For each query: value of the latest source row of the same group with
    ts < q_ts (strict); NaN if none. Sources with ts < 0 or NaN value are
    skipped."""
    ok = (ts >= 0) & ~np.isnan(value)
    g, t, v = group[ok], ts[ok], value[ok]
    o = np.lexsort((t, g))
    g, t, v = g[o], t[o], v[o]
    out = np.full(len(q_group), np.nan, np.float64)
    gu, gi = np.unique(g, return_inverse=True)
    BIG = np.int64(1) << 34
    comp = gi.astype(np.int64) * BIG + t
    qi = np.searchsorted(gu, q_group)
    qi_ok = (qi < len(gu)) & (gu[np.minimum(qi, len(gu) - 1)] == q_group)
    qc = np.where(qi_ok, qi.astype(np.int64) * BIG + q_ts, 0)
    pos = np.searchsorted(comp, qc, side="left") - 1
    good = qi_ok & (pos >= 0)
    good[good] &= gi[pos[good]] == qi[good]
    out[good] = v[pos[good]]
    return out


def running_max_before(group, ts, value, q_group, q_ts):
    """Max of value over source rows of the same group with ts < q_ts."""
    ok = (ts >= 0) & ~np.isnan(value)
    g, t, v = group[ok], ts[ok], value[ok].astype(np.float64)
    o = np.lexsort((t, g))
    g, t, v = g[o], t[o], v[o]
    gu, gi = np.unique(g, return_inverse=True)
    first = np.r_[True, gi[1:] != gi[:-1]]
    seg = np.cumsum(first) - 1
    # segmented cumulative max: offset each segment so maxima never cross
    off = seg.astype(np.float64) * 1e7
    cm = np.maximum.accumulate(np.where(np.isinf(v), 1e6, v) + off) - off
    cm = np.where(cm >= 1e6 - 1, np.inf, cm)
    BIG = np.int64(1) << 34
    comp = gi.astype(np.int64) * BIG + t
    qi = np.searchsorted(gu, q_group)
    qi_ok = (qi < len(gu)) & (gu[np.minimum(qi, len(gu) - 1)] == q_group)
    qc = np.where(qi_ok, qi.astype(np.int64) * BIG + q_ts, 0)
    pos = np.searchsorted(comp, qc, side="left") - 1
    good = qi_ok & (pos >= 0)
    good[good] &= gi[pos[good]] == qi[good]
    out = np.full(len(q_group), np.nan)
    out[good] = cm[pos[good]]
    return out


def build(tag):
    z = np.load(os.path.join(C.CACHE, "export", tag, "stamps.npz"))
    s = {k: z[k] for k in z.files}
    skey = player_keys(s["region"], s["blizz_ids"], label="stamps")
    start = s["end_ts"] - np.where(s["game_length"] > 0, s["game_length"], 3600)
    from p3_heroes import HKEY
    from p3_mmr_at_game import match
    stats = {}
    for which, path in (("snap", C.SLOTS), ("ext", os.path.join(C.CACHE, "x_slots_ext.npz"))):
        if not os.path.exists(path):
            print(f"skip {which}: {path} missing")
            continue
        d = np.load(path)
        pkey = d["player_keys"][d["pid"]]
        rid = d["replay_id"]
        j = match(pkey, rid, skey, s["replay_ids"])  # the slot's own stamp row
        hit = j >= 0
        q_start = np.where(hit, start[np.maximum(j, 0)], -(1 << 40))
        stats[which] = {"slots": int(len(rid)), "matched_to_a_stamp_row": float(hit.mean())}
        hero_names = [str(h) for h in d["hero_names"]]
        assert hero_names == [str(h) for h in s["hero_names"]]
        # per (player, hero)
        sgrp = skey * HKEY + s["hero"].astype(np.int64)
        qgrp = pkey * HKEY + d["hero"].astype(np.int64)
        valid_src = s["hero"] >= 0
        lo_h = latest_before(sgrp[valid_src], s["parse_ts"][valid_src], s["lo"][valid_src].astype(np.float64),
                             qgrp, q_start)
        hi_h = latest_before(sgrp[valid_src], s["parse_ts"][valid_src], s["hi"][valid_src].astype(np.float64),
                             qgrp, q_start)
        lo_a = running_max_before(skey, s["parse_ts"], s["lo"].astype(np.float64), pkey,
                                  q_start)
        hi_a = running_max_before(skey, s["parse_ts"], s["hi"].astype(np.float64), pkey,
                                  q_start)
        # the old unlagged stamp, for the comparison table
        own = d["hero_level"].astype(np.float64)
        own[own < 0] = np.nan
        both = ~np.isnan(lo_h) & ~np.isnan(own) & (lo_h == hi_h)
        stats[which].update({
            "hero_level_known": float((~np.isnan(lo_h)).mean()),
            "acct_level_known": float((~np.isnan(lo_a)).mean()),
            "own_stamp_minus_causal_mean": float(np.mean(own[both] - lo_h[both])),
            "own_stamp_above_causal_share": float(np.mean(own[both] > lo_h[both])),
        })
        out = {"replay_id": rid, "pid": d["pid"], "start_ts": q_start,
               "hero_lo": lo_h.astype(np.float32), "hero_hi": hi_h.astype(np.float32),
               "acct_lo": lo_a.astype(np.float32), "acct_hi": hi_a.astype(np.float32)}
        np.savez(OUT[which] + ".tmp.npz", **out)
        os.replace(OUT[which] + ".tmp.npz", OUT[which])
        print(which, json.dumps(stats[which]), flush=True)
    with open(os.path.join(C.RESULTS, "p3_hero_level_causal.json"), "w") as f:
        json.dump(stats, f, indent=1)


def load(d, which=None):
    """Causal hero-level arrays aligned to slot table d (snapshot slots, or
    the extended table when d has a 'post' column)."""
    which = which or ("ext" if "post" in d else "snap")
    z = np.load(OUT[which])
    assert np.array_equal(z["replay_id"], d["replay_id"]) and np.array_equal(z["pid"], d["pid"]), \
        f"{OUT[which]} is not aligned to this slot table; rebuild it"
    return {k: z[k].astype(np.float64) for k in ("hero_lo", "hero_hi", "acct_lo", "acct_hi")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["parse", "build"])
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    parse(a.tag) if a.stage == "parse" else build(a.tag)


if __name__ == "__main__":
    main()


def account_status(d, hl, late_day=None):
    """New-account status at each game, causal (P3_EXTENSIONS section 4):
      0  first seen in the window before 2024-07-01
      1  first seen later, and no level stamp parsed before the game started
         (or a band that straddles level 5)
      2  first seen later, highest earlier-parsed level <= 5 (new account)
      3  first seen later, some earlier-parsed level > 5 (old account new to
         the corpus)"""
    late_day = C.day_of("2024-07-01") if late_day is None else late_day
    first_day = np.full(int(d["n_players"]), 10 ** 9)
    np.minimum.at(first_day, d["pid"], d["day"])
    late = first_day[d["pid"]] >= late_day
    lo, hi = hl["acct_lo"], hl["acct_hi"]
    st = np.where(~late, 0, np.where(hi <= 5, 2, np.where(lo > 5, 3, 1)))
    return st.astype(np.int64)
