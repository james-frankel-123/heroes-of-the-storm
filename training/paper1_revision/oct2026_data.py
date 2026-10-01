"""
oct2026 expert-study refresh, data side: snapshot from the relabeled live DB,
leak-free (out-of-fold) decayed statistics, and the phase0 caches that the
unchanged rerun2026 scripts read, written into namespace rerun2026/ns/oct2026.

  snapshot   read-only DB pull: every replay_draft_data row with
             game_date < 2026-09-01 and skill_tier <> 'unknown' (site-scheme
             tiers after the 2026-09-30 relabel: low = Bronze+Silver, mid =
             Gold+Platinum, high = Diamond+Master). Rows carry game_date and
             game_version. Verifies every row's tier against the site rule
             from league_tier/avg_mmr. -> training/snapshots/
             replay_snapshot_2026-09-01_sitetiers_<N>.json
  stats      split = rerun2026.common.load_split() (patch-2.55 filter, 2%
             replay-level test, seed 42) on the new snapshot. Statistics over
             TRAIN rows only, per-game exponentially decayed (half-life 90 d,
             reference 2026-09-01, the September run's recipe), in the
             frozen-stats schema: deploy (all train rows) and oof0..4 (train
             minus hash fold k), plus own-corpus role-composition tables
             (decayed win rate; admitted at >= 50 raw games per tier).
  features   FULL_TRAIN_NPZ: train rows with out-of-fold features (row of fold
             k uses oof{k}), both team orders adjacent (2p, 2p+1) exactly like
             sweep_enriched_wp._extract_chunk, plus rids. FULL_TEST_NPZ: test
             rows with deploy statistics. CQL_ENR_TRAIN/TEST: enriched CQL
             transitions, train out-of-fold, test deploy. (cql-naive and gd
             caches come from phase0_features.py unchanged; they read no
             statistics.)

Env for every stage: RERUN_NS=oct2026, REPLAY_SNAPSHOT_PATH=<new snapshot>.
"""
import os
import sys
import json
import time
import argparse
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np

CUTOFF = "2026-09-01"
HALF_LIFE = 90.0
N_FOLDS = 5
COMP_MIN_RAW = 50
NPROC = int(os.environ.get("P1R_NPROC", "4"))
SNAP_DIR = os.path.join(TRAINING_DIR, "snapshots")


def site_tier(lt, mmr):
    if lt is None:
        return "unknown" if mmr is None else "high"
    if lt <= 3:
        return "low"
    if lt <= 5:
        return "mid"
    return "high"


def fold_of(rid):
    from overfit2026.data import splitmix64
    return splitmix64(int(rid) * 1000003 + 31) % N_FOLDS


def snapshot_path():
    import glob
    p = sorted(glob.glob(os.path.join(SNAP_DIR, f"replay_snapshot_{CUTOFF}_sitetiers_*.json")))
    return p[-1] if p else None


def cmd_snapshot():
    import psycopg2
    url = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(url)
    conn.set_session(readonly=True)
    cur = conn.cursor(name="snap")
    cur.itersize = 50_000
    cur.execute("""
        SELECT replay_id, game_map, skill_tier, draft_order, team0_heroes, team1_heroes,
               team0_bans, team1_bans, winner, avg_mmr, league_tier, game_date, game_version
        FROM replay_draft_data
        WHERE game_date < %s AND skill_tier <> 'unknown'
        ORDER BY replay_id""", (CUTOFF,))
    rows, mism, vers, tiers = [], 0, {}, {}
    t0 = time.time()
    for (rid, gm, st, do, t0h, t1h, t0b, t1b, w, mmr, lt, gd, gv) in cur:
        if site_tier(lt, mmr) != st:
            mism += 1
        d = {"replay_id": rid, "game_map": gm, "skill_tier": st, "draft_order": do,
             "team0_heroes": t0h, "team1_heroes": t1h, "team0_bans": t0b, "team1_bans": t1b,
             "winner": w, "avg_mmr": mmr, "league_tier": lt,
             "game_date": gd.isoformat() if gd else None, "game_version": gv}
        for f in ("draft_order", "team0_heroes", "team1_heroes", "team0_bans", "team1_bans"):
            if isinstance(d[f], str):
                d[f] = json.loads(d[f])
        rows.append(d)
        key = ".".join((gv or "?").split(".")[:3])
        vers[key] = vers.get(key, 0) + 1
        tiers[st] = tiers.get(st, 0) + 1
    cur.close()
    conn.close()
    path = os.path.join(SNAP_DIR, f"replay_snapshot_{CUTOFF}_sitetiers_{len(rows)}.json")
    json.dump(rows, open(path, "w"))
    meta = {"path": path, "n": len(rows), "tier_mismatch_vs_site_rule": mism, "tiers": tiers,
            "versions": vers, "max_game_date": max(r["game_date"] for r in rows if r["game_date"]),
            "max_replay_id": max(r["replay_id"] for r in rows), "secs": time.time() - t0}
    json.dump(meta, open(path[:-5] + ".meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


def _ns():
    from rerun2026 import common
    common.setup()
    return common


def stats_dir():
    from rerun2026 import common
    d = os.path.join(common.CACHE_DIR, "stats")
    os.makedirs(d, exist_ok=True)
    return d


def _accumulate(rows):
    """Decayed and raw counts per tier, drift2026/make_snapshot conventions."""
    from drift2026.build_patch_stats import comp_key
    ref = datetime.date.fromisoformat(CUTOFF).toordinal()
    cells = {}
    for d in rows:
        t0h, t1h = d["team0_heroes"] or [], d["team1_heroes"] or []
        if len(t0h) != 5 or len(t1h) != 5 or d["winner"] not in (0, 1) or not d.get("game_date"):
            continue
        gd = datetime.datetime.fromisoformat(d["game_date"])
        age = max(0.0, ref - (gd.toordinal() + gd.hour / 24.0))
        w = 0.5 ** (age / HALF_LIFE)
        c = cells.setdefault(d["skill_tier"], {"games": 0.0, "bans": {}, "hero": {}, "hmap": {},
                                               "with": {}, "against": {}, "comp": {}})
        c["games"] += w
        for h in set((d["team0_bans"] or []) + (d["team1_bans"] or [])):
            c["bans"][h] = c["bans"].get(h, 0.0) + w
        teams = (t0h, t1h)
        for ti, team in enumerate(teams):
            ww = w if d["winner"] == ti else 0.0
            for h in team:
                e = c["hero"].setdefault(h, [0.0, 0.0]); e[0] += w; e[1] += ww
                e = c["hmap"].setdefault((d["game_map"], h), [0.0, 0.0]); e[0] += w; e[1] += ww
            hs = sorted(team)
            for i in range(5):
                for j in range(i + 1, 5):
                    e = c["with"].setdefault((hs[i], hs[j]), [0.0, 0.0]); e[0] += w; e[1] += ww
            e = c["comp"].setdefault(comp_key(team), [0.0, 0.0, 0]); e[0] += w; e[1] += ww; e[2] += 1
        for a in t0h:
            for b in t1h:
                key = (a, b) if a < b else (b, a)
                wa = w if d["winner"] == 0 else 0.0
                e = c["against"].setdefault(key, [0.0, 0.0])
                e[0] += w
                e[1] += wa if a < b else w - wa
    return cells


def _write_stats(cells, name, n_rows):
    hero_stats, hmap, pw = [], [], []
    for tier, c in cells.items():
        tot = c["games"]
        for h, (g, wn) in c["hero"].items():
            if g >= 20:
                hero_stats.append({"hero": h, "tier": tier, "games": round(g, 1),
                                   "win_rate": round(100 * wn / g, 3), "pick_rate": round(100 * g / tot, 3),
                                   "ban_rate": round(100 * c["bans"].get(h, 0.0) / tot, 3)})
        for (m, h), (g, wn) in c["hmap"].items():
            if g >= 5:
                hmap.append({"hero": h, "map": m, "tier": tier, "games": round(g, 1),
                             "win_rate": round(100 * wn / g, 3)})
        for (a, b), (g, wn) in c["with"].items():
            if g >= 10:
                for x, y in ((a, b), (b, a)):
                    pw.append({"hero_a": x, "hero_b": y, "tier": tier, "relationship": "with",
                               "win_rate": round(100 * wn / g, 3), "games": round(g, 1)})
        for (a, b), (g, wa) in c["against"].items():
            if g >= 10:
                pw.append({"hero_a": a, "hero_b": b, "tier": tier, "relationship": "against",
                           "win_rate": round(100 * wa / g, 3), "games": round(g, 1)})
                pw.append({"hero_a": b, "hero_b": a, "tier": tier, "relationship": "against",
                           "win_rate": round(100 - 100 * wa / g, 3), "games": round(g, 1)})
    comps = {tier: [{"roles": k.split(","), "winRate": round(100 * wn / g, 3), "games": round(g, 1),
                     "raw_games": raw}
                    for k, (g, wn, raw) in c["comp"].items() if raw >= COMP_MIN_RAW]
             for tier, c in cells.items()}
    sd = stats_dir()
    json.dump({"_meta": {"snapshot_date": CUTOFF, "patch": "2.55", "kind": "decayed90_oof",
                         "subset": name, "rows": n_rows, "half_life_days": HALF_LIFE},
               "hero_stats": hero_stats, "hero_map_stats": hmap, "pairwise_stats": pw},
              open(os.path.join(sd, f"{name}.json"), "w"))
    json.dump(comps, open(os.path.join(sd, f"{name}_compositions.json"), "w"))
    print(f"stats {name}: {n_rows:,} rows, comps per tier "
          f"{ {t: len(v) for t, v in comps.items()} }", flush=True)


def cmd_stats():
    common = _ns()
    train, test = common.load_split()
    folds = np.array([fold_of(r["replay_id"]) for r in train])
    jobs = [("deploy", train)] + [(f"oof{k}", [r for r, f in zip(train, folds) if f != k])
                                  for k in range(N_FOLDS)]
    for name, rows in jobs:
        if os.path.exists(os.path.join(stats_dir(), f"{name}.json")):
            continue
        _write_stats(_accumulate(rows), name, len(rows))
    json.dump({"n_train": len(train), "n_test": len(test),
               "folds": {int(k): int((folds == k).sum()) for k in range(N_FOLDS)}},
              open(os.path.join(stats_dir(), "split.json"), "w"), indent=1)


def load_stats(name):
    from overfit2026 import feats
    from paper1_revision.core import comps_from_json
    sd = stats_dir()
    st = feats.stats_from_json(os.path.join(sd, f"{name}.json"), compositions=False)
    st.comp_data = comps_from_json(os.path.join(sd, f"{name}_compositions.json"))
    return st


def _full_chunk(args):
    rows, sd = args
    from sweep_enriched_wp import StatsCache, extract_features, FEATURE_GROUPS, _swap_features
    st = object.__new__(StatsCache)
    for k, v in sd.items():
        setattr(st, k, v)
    mask = [True] * len(FEATURE_GROUPS)
    B, E, L, R = [], [], [], []
    for d in rows:
        try:
            b, e = extract_features(d, st, mask)
        except Exception:
            continue
        y = float(d["winner"] == 0)
        bs, es = _swap_features(b, e)
        B += [b, bs]
        E += [e, es]
        L += [y, 1.0 - y]
        R += [d["replay_id"], d["replay_id"]]
    return (np.array(B, np.float32), np.array(E, np.float32), np.array(L, np.float32),
            np.array(R, np.int64))


def _extract(rows, st):
    import multiprocessing as mp
    from paper1_revision.core import stats_dict
    sd = stats_dict(st)
    parts = [(rows[i:i + 4000], sd) for i in range(0, len(rows), 4000)]
    with mp.get_context("fork").Pool(NPROC) as pool:
        res = pool.map(_full_chunk, parts)
    return [np.concatenate([r[i] for r in res]) for i in range(4)]


def cmd_features():
    common = _ns()
    train, test = common.load_split()
    if not os.path.exists(common.FULL_TEST_NPZ):
        b, e, l, r = _extract(test, load_stats("deploy"))
        np.savez(common.FULL_TEST_NPZ, bases=b, enricheds=e, labels=l, rids=r)
        print(f"full test: {len(l):,} rows", flush=True)
    if not os.path.exists(common.FULL_TRAIN_NPZ):
        folds = np.array([fold_of(r["replay_id"]) for r in train])
        parts = []
        for k in range(N_FOLDS):
            rows = [r for r, f in zip(train, folds) if f == k]
            parts.append(_extract(rows, load_stats(f"oof{k}")))
            print(f"  fold {k}: {len(rows):,} replays", flush=True)
        b, e, l, r = [np.concatenate([p[i] for p in parts]) for i in range(4)]
        np.savez(common.FULL_TRAIN_NPZ, bases=b, enricheds=e, labels=l, rids=r)
        print(f"full train (out-of-fold): {len(l):,} rows", flush=True)
    # enriched CQL transitions
    from rerun2026.phase0_features import _cql_enriched_chunk
    from sweep_enriched_wp import compute_group_indices
    from experiment_cql_enriched import get_enriched_cols
    from paper1_revision.core import stats_dict
    import multiprocessing as mp
    dim = 289 + len(get_enriched_cols(compute_group_indices()))
    fields = [("actions", "int64"), ("outcomes", "float32")]

    def stream(data, stats_of):
        by = {}
        for r in data:
            by.setdefault(stats_of(r), []).append(r)
        with mp.get_context("fork").Pool(NPROC) as pool:
            for sname, rs in sorted(by.items()):
                sd = stats_dict(load_stats(sname))
                for out in pool.imap(_cql_enriched_chunk,
                                     [(rs[i:i + 2000], sd) for i in range(0, len(rs), 2000)]):
                    yield out
                print(f"  cql-enriched {sname}: {len(rs):,} replays", flush=True)

    for d, data, fn in ((common.CQL_ENR_TEST, test, lambda r: "deploy"),
                        (common.CQL_ENR_TRAIN, train, lambda r: f"oof{fold_of(r['replay_id'])}")):
        if not os.path.exists(os.path.join(d, "meta.json")):
            common.memmap_write(d, stream(data, fn), dim, fields)


def cmd_phase0(which):
    """Statistic-free phase0 caches (gd, cql-naive) with phase0's own chunk
    functions, but streamed in small chunks through a fork pool of NPROC
    workers. Replaces phase0_features.py --only gd/cql, whose 4-worker pool of
    giant chunks hung for 5 h on 2026-10-01 (a worker died; imap waited)."""
    import multiprocessing as mp
    from rerun2026.phase0_features import _cql_naive_chunk, _gd_chunk
    common = _ns()
    train, test = common.load_split()
    if which == "gd":
        plans = [(common.GD_TRAIN, train), (common.GD_TEST, test)]
        fn, fields = _gd_chunk, [("actions", "int64")]
    else:
        plans = [(common.CQL_NAIVE_TRAIN, train), (common.CQL_NAIVE_TEST, test)]
        fn, fields = _cql_naive_chunk, [("actions", "int64"), ("outcomes", "float32")]
    for d, data in plans:
        if os.path.exists(os.path.join(d, "meta.json")):
            continue

        def stream():
            t0 = time.time()
            with mp.get_context("fork").Pool(NPROC, maxtasksperchild=50) as pool:
                for i, out in enumerate(pool.imap(fn, [data[k:k + 2000] for k in range(0, len(data), 2000)])):
                    if (i + 1) % 100 == 0:
                        print(f"  {os.path.basename(d)}: chunk {i + 1} ({time.time() - t0:.0f}s)", flush=True)
                    yield out
        common.memmap_write(d, stream(), 289, fields)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["snapshot", "stats", "features", "phase0_gd", "phase0_cql"])
    a = ap.parse_args()
    {"snapshot": cmd_snapshot, "stats": cmd_stats, "features": cmd_features,
     "phase0_gd": lambda: cmd_phase0("gd"), "phase0_cql": lambda: cmd_phase0("cql")}[a.cmd]()
