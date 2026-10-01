"""
paper1_revision core: leak-free data design for the paper-1 revision.

Corpus: the pinned paper-1 snapshot (1,949,087 replays, patch 2.55), read from
overfit2026's slim cache (identical rows to rerun2026.common.load_data()).

Splits (replay level)
  test   the paper's own test set: shared.split_data(test_frac=0.02, seed=42)
         over rerun2026.common.load_data() -> 38,981 replays. Kept so every
         revised accuracy is on the same games as the submitted Table I.
  val    2% of the remaining replays by a salted hash (early stopping and
         seed selection; the submission had no validation split, audit C1).
  train  everything else.
  fold   5-fold salted hash over train rows (out-of-fold statistics).

Statistics: built only from our own corpus in the frozen-stats schema, plus an
own-corpus role-composition table (replaces the external Heroes Profile
compositions.json). No Heroes Profile aggregate enters any feature.
  deploy   all train games. Used for val/test/post-snapshot features, greedy
           search, the MCTS kernel lookup tables, and every judge's scoring
           of generated drafts. Contains no val or test game.
  oof{k}   train games outside fold k. Features of a training row in fold k
           use oof{k}, so no training row's features contain its own outcome.
  leak     = deploy used for training rows too (the submission's design, with
           our own statistics in place of the external file). Control only.

Feature layout per row: base (197) + full enriched vector (all 15
FEATURE_GROUPS, 160 dims), for both team orders (the swap orientation is
re-extracted, not permuted).
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)
os.environ.setdefault("REPLAY_SNAPSHOT", "1")

import numpy as np

from overfit2026 import data as odata

CACHE = odata.art(HERE, "cache")
STATS_DIR = os.path.join(CACHE, "stats")
FEAT_DIR = os.path.join(CACHE, "feats")
MODEL_DIR = odata.art(HERE, "models")
RESULTS = odata.art(HERE, "results")
LOGS = odata.art(HERE, "logs")
MCTS_RUNS = odata.art(HERE, "mcts_runs")
TEST_IDS = os.path.join(CACHE, "paper_test_ids.json")

N_FOLDS = 5
COMP_MIN_GAMES = 50       # own role-composition cell admitted at >= 50 games
                          # (the external table's smallest cell is 50)
for _d in (CACHE, STATS_DIR, FEAT_DIR, MODEL_DIR, RESULTS, LOGS):
    os.makedirs(_d, exist_ok=True)


# ── splits ──────────────────────────────────────────────────────────────

def fold_of(rid):
    return odata.splitmix64(int(rid) * 1000003 + 31) % N_FOLDS


def is_val(rid):
    return odata.splitmix64(int(rid) * 998244353 + 23) % 50 == 0


_TEST = None


def test_ids():
    global _TEST
    if _TEST is None:
        if not os.path.exists(TEST_IDS):
            from rerun2026 import common
            common.setup()
            _, test = common.load_split()
            json.dump(sorted(int(r["replay_id"]) for r in test), open(TEST_IDS, "w"))
        _TEST = set(json.load(open(TEST_IDS)))
    return _TEST


def split_games():
    """-> dict(train=[...], val=[...], test=[...]) of slim snapshot games."""
    games = odata.load_snapshot()
    T = test_ids()
    out = {"train": [], "val": [], "test": []}
    for g in games:
        if g[0] in T:
            out["test"].append(g)
        elif is_val(g[0]):
            out["val"].append(g)
        else:
            out["train"].append(g)
    return out


def gold_games(name):
    """Post-snapshot games (never in any training set): NODRIFT, T17, T97, BF."""
    from overfit2026 import gold
    return gold.GOLD_SETS[name](odata)


# ── statistics ──────────────────────────────────────────────────────────

def counts_for(games):
    from overfit2026.split import counts_for as _c
    return _c(games)


def stats_json(merged, meta):
    """frozen-stats schema (overfit2026.split.frozen_json thresholds) plus a
    role-composition table in the compositions.json schema."""
    from overfit2026.split import frozen_json
    js = frozen_json(merged, meta)
    comps = {}
    for tier, cell in merged.items():
        rows = []
        for key, (g, w) in cell["comp"].items():
            if g >= COMP_MIN_GAMES:
                rows.append({"roles": key.split(","), "winRate": round(100.0 * w / g, 3),
                             "games": int(g)})
        comps[tier] = rows
    js["compositions"] = comps
    return js


def stats_path(name):
    return os.path.join(STATS_DIR, f"{name}.json")


def comp_path(name):
    return os.path.join(STATS_DIR, f"{name}_compositions.json")


def build_stats():
    sp = split_games()
    train = sp["train"]
    folds = np.array([fold_of(g[0]) for g in train])
    jobs = {"deploy": train}
    for k in range(N_FOLDS):
        jobs[f"oof{k}"] = [g for g, f in zip(train, folds) if f != k]
    # half-corpus stats for the in-corpus realized index are not needed; the
    # realized indices come from post-snapshot games (overfit2026.gold).
    for name, games in jobs.items():
        if os.path.exists(stats_path(name)):
            continue
        t0 = time.time()
        js = stats_json(counts_for(games), {"subset": name, "games_used": len(games)})
        comps = js.pop("compositions")
        json.dump(js, open(stats_path(name), "w"))
        json.dump(comps, open(comp_path(name), "w"))
        print(f"stats {name}: {len(games):,} games ({time.time() - t0:.0f}s)", flush=True)
    meta = {k: len(v) for k, v in sp.items()}
    meta["folds"] = {int(k): int((folds == k).sum()) for k in range(N_FOLDS)}
    json.dump(meta, open(os.path.join(CACHE, "split_meta.json"), "w"), indent=1)
    print(meta)


_STATS = {}


def load_stats(name):
    """StatsCache over own-corpus stats `name` with its own composition table.
    name 'hp' = the submission's external Heroes Profile file + compositions."""
    if name in _STATS:
        return _STATS[name]
    from overfit2026 import feats
    if name == "hp":
        st = feats.paper_stats()
    else:
        st = feats.stats_from_json(stats_path(name), compositions=False)
        st.comp_data = comps_from_json(comp_path(name))
    _STATS[name] = st
    return st


def comps_from_json(path):
    raw = json.load(open(path))
    out = {}
    for tier, comps in raw.items():
        out[tier] = {",".join(sorted(c["roles"])): (c["winRate"], c["games"]) for c in comps}
    return out


def stats_dict(st):
    return {"hero_wr": st.hero_wr, "hero_meta": st.hero_meta,
            "hero_map_wr": st.hero_map_wr, "pairwise": st.pairwise,
            "comp_data": st.comp_data}


# ── featurization ───────────────────────────────────────────────────────

def _chunk(args):
    rows, sd = args
    from sweep_enriched_wp import StatsCache, extract_features, FEATURE_GROUPS
    st = object.__new__(StatsCache)
    for k, v in sd.items():
        setattr(st, k, v)
    mask = [True] * len(FEATURE_GROUPS)
    F, S = [], []
    for t0, t1, gm, tier in rows:
        d = {"team0_heroes": list(t0), "team1_heroes": list(t1), "game_map": gm,
             "skill_tier": tier, "winner": 0}
        b, e = extract_features(d, st, mask)
        F.append(np.concatenate([b, e]))
        d = {"team0_heroes": list(t1), "team1_heroes": list(t0), "game_map": gm,
             "skill_tier": tier, "winner": 0}
        b, e = extract_features(d, st, mask)
        S.append(np.concatenate([b, e]))
    return np.array(F, np.float32), np.array(S, np.float32)


def featurize(rows, st, nproc=int(os.environ.get("P1R_NPROC", "6")), chunk=2000):
    """rows: (team0, team1, map, tier). -> (X_forward, X_swapped), 357 cols."""
    import multiprocessing as mp
    rows = list(rows)
    sd = stats_dict(st)
    if len(rows) <= chunk:
        return _chunk((rows, sd))
    parts = [(rows[i:i + chunk], sd) for i in range(0, len(rows), chunk)]
    with mp.get_context("fork").Pool(nproc) as pool:
        res = pool.map(_chunk, parts)
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


def rows_of(games):
    return [(g[3], g[4], g[2], g[1]) for g in games]


def labels(games):
    return np.array([1.0 if g[6] == 0 else 0.0 for g in games], np.float32)


def feat_path(name):
    return os.path.join(FEAT_DIR, f"{name}.npz")


def build_features(which=("train_oof", "train_leak", "val", "test", "test_hp",
                          "NODRIFT", "T17", "NODRIFT_hp")):
    sp = split_games()
    for name in which:
        path = feat_path(name)
        if os.path.exists(path):
            continue
        t0 = time.time()
        if name == "train_oof":
            games = sp["train"]
            folds = np.array([fold_of(g[0]) for g in games])
            Xf = np.zeros((len(games), 357), np.float32)
            Xs = np.zeros_like(Xf)
            for k in range(N_FOLDS):
                idx = np.where(folds == k)[0]
                a, b = featurize(rows_of([games[i] for i in idx]), load_stats(f"oof{k}"))
                Xf[idx], Xs[idx] = a, b
                print(f"  train_oof fold {k}: {len(idx):,} rows", flush=True)
        elif name == "train_leak":
            games = sp["train"]
            Xf, Xs = featurize(rows_of(games), load_stats("deploy"))
        elif name in ("val", "test"):
            games = sp[name]
            Xf, Xs = featurize(rows_of(games), load_stats("deploy"))
        elif name == "test_hp":        # the submission's features (external stats)
            games = sp["test"]
            Xf, Xs = featurize(rows_of(games), load_stats("hp"))
        elif name in ("NODRIFT", "T17"):
            games = gold_games(name)
            Xf, Xs = featurize(rows_of(games), load_stats("deploy"))
        elif name == "NODRIFT_hp":
            games = gold_games("NODRIFT")
            Xf, Xs = featurize(rows_of(games), load_stats("hp"))
        else:
            raise ValueError(name)
        np.savez(path, Xf=Xf, Xs=Xs, y=labels(games),
                 rid=np.array([g[0] for g in games]),
                 tier=np.array([g[1] for g in games]))
        print(f"features {name}: {len(games):,} rows ({time.time() - t0:.0f}s)", flush=True)


def load_features(name):
    z = np.load(feat_path(name))
    return z["Xf"], z["Xs"], z["y"], z["rid"], z["tier"]


# ── column selection ────────────────────────────────────────────────────

def group_cols(groups):
    """Column indices into the 357-wide row for base + the given groups."""
    from sweep_enriched_wp import compute_group_indices
    gi = compute_group_indices()
    cols = list(range(197))
    for g in groups:
        s, e = gi[g]
        cols.extend(range(197 + s, 197 + e))
    return np.array(cols)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ids":
        print(len(test_ids()))
    elif cmd == "stats":
        build_stats()
    elif cmd == "features":
        build_features(tuple(sys.argv[2:]) or None) if len(sys.argv) > 2 else build_features()
