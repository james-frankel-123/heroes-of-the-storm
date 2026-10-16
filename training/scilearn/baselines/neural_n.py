"""
N contrast: the frozen neural win-probability models, scoreable by the
scientific-learning harness (FRAMING sections 1 and 10.2 "N", WAVE0_BRIEF item 7).

Two frozen models, both swap-symmetrized: p = (p(t0, t1) + 1 - p(t1, t0)) / 2.

  n_drift           The drift-aware WP (paper 2's d2c_cumprev, the population WP
                    under P3). drift2026/models/d2c_cumprev_s{42,123,777}.pt, mean
                    of the three seeds. Weights trained on builds up to
                    2.55.14.95918 (through 2026-02-10, replay_id <= 63,653,039);
                    every later build is out of time for the weights.
                    Features: cumulative statistics through the PREVIOUS build
                    (legacy research tiers):
                      build <= 2.55.16.96881  drift2026/patch_stats/cumulative/<prev>.json.gz
                                              (snapshot games only), exactly as in
                                              drift2026/build_drift_features.py
                      later builds            snapshot counts through 96881 plus
                                              every history game of the builds
                                              strictly between 96881 and the
                                              scored build (personalization/
                                              p3_x_wp_post.py convention, which
                                              produced the 56.94% headline)
                    No statistic contains a game of the scored build.
  p1_site_enriched  The paper-1 revision's leak-free enriched WP on site tiers
                    (paper1_revision/site/models/enriched_s42.pt, the selected
                    seed). Weights trained on the pinned paper-1 snapshot
                    (replay_id <= 63,653,039, builds up to 2.55.16.97039, minus
                    the paper test set and the 2% hash validation set) with
                    out-of-fold statistics. Scoring uses the frozen "deploy"
                    statistics (all snapshot training games;
                    paper1_revision/site/cache/stats/deploy*.json). Dpre and the
                    snapshot part of G are therefore IN SAMPLE for this model.

Input rows (dicts): replay_id, game_version, game_map, league_tier, avg_mmr,
team0_heroes, team1_heroes, team0_bans, team1_bans; winner only for history
rows. Both tier schemes are derived from league_tier / avg_mmr (the DB's
skill_tier column was relabelled to the site scheme on 2026-09-30, and the
drift model needs the legacy labels). Other keys (bans, draft_order,
game_date, players) are accepted and ignored: neither model reads player
information. Unknown heroes (e.g. a hero released after training) get a zero
one-hot and default statistics.

CLI (DB access only in `extract`, local, research role; 2.57 rows refused
unless SCILEARN_ALLOW_C=1, sealed replay ids always dropped):
  python3 training/scilearn/baselines/neural_n.py extract Dpre G Dpost
  python3 training/scilearn/baselines/neural_n.py score --rows <dir>/rows_Dpost.pkl.gz \
      --history <dir>/rows_G.pkl.gz --tag Dpost --out-dir <dir> --nproc 12
  python3 training/scilearn/baselines/neural_n.py report --dir <dir> --tags Dpre,G,Dpost
Default <dir>: /home/max/scilearn-private/fits/neural_n (private, out of git).
"""
import os
import sys
import gzip
import json
import pickle
import time
import argparse
import importlib.util
from pathlib import Path

import numpy as np

TRAINING = Path(os.environ.get("NEURAL_N_TRAINING") or Path(__file__).resolve().parents[2])
if os.environ.get("P1_TIERS", "site") != "site":
    raise RuntimeError("neural_n needs P1_TIERS=site (the paper-1 site namespace)")
os.environ["P1_TIERS"] = "site"            # before any overfit2026 import
os.environ.setdefault("REPLAY_SNAPSHOT", "1")
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

OUT_DIR = Path(os.environ.get("SCILEARN_PRIVATE", "/home/max/scilearn-private")) / "fits" / "neural_n"
SEALED_IDS = Path("/home/max/hots-replays/sealed/sealed_ids.txt")
SNAPSHOT_BOUND = 63653039
DRIFT_BASE_BUILD = "2.55.16.96881"
DRIFT_SEEDS = (42, 123, 777)
P1_SEED = 42
MODELS = ("n_drift", "p1_site_enriched")

# Tranches by build (FRAMING 4.1). S is everything before 2.55.15.
DPRE_EXTRA = {"2.55.16.96846", "2.55.16.96870", "2.55.16.96881"}
G_BUILDS = {"2.55.16.97039"}


def bkey(v):
    return tuple(int(p) for p in v.split("."))


def tranche_of(v):
    k = bkey(v)
    if k[:2] == (2, 57) or k[:2] > (2, 55):
        return "C"
    if k < (2, 55, 15):
        return "S"
    if k[:3] == (2, 55, 15) or v in DPRE_EXTRA:
        return "Dpre"
    if v in G_BUILDS:
        return "G"
    if k[:3] == (2, 55, 17):
        return "Dpost"
    return "other"


# ── tiers ───────────────────────────────────────────────────────────────

def legacy_tier(league_tier, avg_mmr):
    """Pre-2026-09-30 research labels (the drift model's training scheme). Matches
    replay_draft_skill_tier_backup_20260930 on every 2.55.1x row: league_tier
    2 -> low, 3-4 -> mid, 5-6 -> high, missing -> mid."""
    if league_tier is None:
        return "mid"
    lt = int(league_tier)
    return "low" if lt <= 2 else ("mid" if lt <= 4 else "high")


def site_tier(league_tier, avg_mmr):
    """Site scheme (overfit2026.data.site_tier); 'unknown' = no rank and no MMR."""
    if league_tier is None:
        return "unknown" if avg_mmr is None else "high"
    lt = int(league_tier)
    return "low" if lt <= 3 else ("mid" if lt <= 5 else "high")


# ── data guards ─────────────────────────────────────────────────────────

def sealed_ids():
    if not SEALED_IDS.exists():
        return set()
    return {int(x) for x in SEALED_IDS.read_text().split() if x.strip().isdigit()}


def guard(rows, what="rows"):
    """Drop sealed ids; refuse 2.57+ rows unless SCILEARN_ALLOW_C=1 (the evaluator at unlock)."""
    allow_c = os.environ.get("SCILEARN_ALLOW_C") == "1"
    bad = [r["game_version"] for r in rows if tranche_of(r["game_version"]) == "C"]
    if bad and not allow_c:
        raise RuntimeError(f"{what}: {len(bad)} rows of 2.57+ builds; C is locked")
    S = sealed_ids()
    out = [r for r in rows if int(r["replay_id"]) not in S]
    if len(out) != len(rows):
        print(f"{what}: dropped {len(rows) - len(out)} sealed replay ids", flush=True)
    return out


# ── paper-1 site enriched WP ───────────────────────────────────────────

_P1 = None


def _p1():
    global _P1
    if _P1 is None:
        from paper1_revision import core, train_wp
        assert os.path.basename(os.path.dirname(core.CACHE)) == "site", core.CACHE
        m, cols = train_wp.load("enriched", seed=P1_SEED)
        _P1 = (core, m, cols, core.load_stats("deploy"))
    return _P1


def predict_p1(rows, nproc=4, chunk=100000):
    import torch
    core, m, cols, st = _p1()
    out = np.empty(len(rows), np.float64)
    for i in range(0, len(rows), chunk):
        part = rows[i:i + chunk]
        rr = []
        for g in part:
            t = site_tier(g.get("league_tier"), g.get("avg_mmr"))
            rr.append((tuple(g["team0_heroes"]), tuple(g["team1_heroes"]), g["game_map"],
                       "mid" if t == "unknown" else t))
        Xf, Xs = core.featurize(rr, st, nproc=nproc)
        with torch.no_grad():
            pf = m(torch.from_numpy(np.ascontiguousarray(Xf[:, cols]))).view(-1).double().numpy()
            ps = m(torch.from_numpy(np.ascontiguousarray(Xs[:, cols]))).view(-1).double().numpy()
        out[i:i + len(part)] = 0.5 * (pf + 1.0 - ps)
    return out


# ── drift-aware WP (d2c_cumprev) ───────────────────────────────────────

_DR = {}


def _wp_post():
    """personalization/p3_x_wp_post.py (the code behind the 56.94% headline): its
    rounded-count stats builder and its feature worker."""
    if "mod" not in _DR:
        path = TRAINING / "personalization" / "p3_x_wp_post.py"
        spec = importlib.util.spec_from_file_location("p3_x_wp_post", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["p3_x_wp_post"] = mod       # pool workers resolve _feat by module name
        spec.loader.exec_module(mod)
        _DR["mod"] = mod
    return _DR["mod"]


def _drift_models():
    if "models" not in _DR:
        import torch
        from sweep_enriched_wp import WinProbEnrichedModel
        from drift2026 import common
        from drift2026.train_drift_wp import enriched_cols
        ms = []
        # NEURAL_N_DRIFT_CKPT: a checkpoint pattern with {seed}, for the matched-cutoff
        # retrain (compare/n_matched.py); default is the frozen N
        pat = os.environ.get("NEURAL_N_DRIFT_CKPT") or os.path.join(common.MODELS_DIR, "d2c_cumprev_s{seed}.pt")
        for s in DRIFT_SEEDS:
            ck = torch.load(pat.format(seed=s),
                            map_location="cpu", weights_only=False)
            m = WinProbEnrichedModel(ck["input_dim"], ck["arch"], dropout=ck["dropout"])
            m.load_state_dict(ck["state_dict"])
            ms.append(m.eval())
        _DR["models"] = ms
        _DR["ecols"] = enriched_cols()
    return _DR["models"], _DR["ecols"]


def _base_counts():
    """Cumulative counts through DRIFT_BASE_BUILD from the snapshot (p3_x_wp_post)."""
    if "base" not in _DR:
        from drift2026 import common
        from drift2026.build_patch_stats import merge_cell, _new_cell
        mod = _wp_post()
        with gzip.open(common.COUNTS_PKL, "rb") as f:
            pc = pickle.load(f)
        base_i = pc["builds"].index(DRIFT_BASE_BUILD)
        cum = {}
        for bi, by_tier in pc["per_build"].items():
            if bi > base_i:
                continue
            for tier, cell in mod.to_cells(by_tier).items():
                merge_cell(cum.setdefault(tier, _new_cell()), cell)
        ref = common.load_patch_stats("cumulative", DRIFT_BASE_BUILD)
        mine = mod.stats_from_counts_rounded(cum)
        assert mine.hero_wr == ref.hero_wr and mine.comp_data == ref.comp_data, \
            "rounded-count stats do not reproduce the 96881 cumulative file"
        _DR["base"] = cum
    return _DR["base"]


def drift_stats(build, history):
    """Stats object for scoring `build`: everything strictly before it."""
    import copy
    from drift2026 import common
    from drift2026.build_patch_stats import count_chunk, merge_cell, _new_cell
    if bkey(build) <= bkey(DRIFT_BASE_BUILD):
        builds = common.load_patch_index()["builds"]
        i = builds.index(build)
        if i == 0:
            return common.empty_stats()
        return common.load_patch_stats("cumulative", builds[i - 1])
    cum = copy.deepcopy(_base_counts())
    hist = [g for g in history
            if bkey(DRIFT_BASE_BUILD) < bkey(g["game_version"]) < bkey(build)
            and g.get("winner") in (0, 1)
            and len(g["team0_heroes"]) == 5 and len(g["team1_heroes"]) == 5]
    slim = [(0, legacy_tier(g.get("league_tier"), g.get("avg_mmr")), g["game_map"],
             tuple(g["team0_heroes"]), tuple(g["team1_heroes"]),
             tuple(g.get("team0_bans") or []) + tuple(g.get("team1_bans") or []), g["winner"])
            for g in hist]
    for i in range(0, len(slim), 20000):
        for (_, tier), cell in count_chunk(slim[i:i + 20000]).items():
            merge_cell(cum.setdefault(tier, _new_cell()), cell)
    hb = sorted({g["game_version"] for g in hist}, key=bkey)
    print(f"  stats for {build}: base through {DRIFT_BASE_BUILD} + {len(hist):,} history "
          f"games of {hb}", flush=True)
    return _wp_post().stats_from_counts_rounded(cum)


def predict_drift(rows, history=None, nproc=4, return_seeds=False):
    """p(team0 wins) under n_drift. history: rows (with winner) usable as earlier-build
    statistics; the scored rows themselves are added (only builds strictly before the
    scored build ever enter its statistics)."""
    import torch
    import multiprocessing as mp
    mod = _wp_post()
    models, ecols = _drift_models()
    seen, hist = set(), []
    for g in list(history or []) + list(rows):
        if g["replay_id"] not in seen:
            seen.add(g["replay_id"])
            hist.append(g)
    out = np.full((len(DRIFT_SEEDS), len(rows)), np.nan)
    idx_of = {}
    for i, g in enumerate(rows):
        idx_of.setdefault(g["game_version"], []).append(i)
    for b in sorted(idx_of, key=bkey):
        ii = idx_of[b]
        mod._STATS = drift_stats(b, hist)
        feats = [{"team0_heroes": rows[i]["team0_heroes"], "team1_heroes": rows[i]["team1_heroes"],
                  "game_map": rows[i]["game_map"],
                  "skill_tier": legacy_tier(rows[i].get("league_tier"), rows[i].get("avg_mmr")),
                  "winner": 0, "avg_mmr": rows[i].get("avg_mmr"), "replay_id": i} for i in ii]
        chunks = [feats[k:k + 2000] for k in range(0, len(feats), 2000)]
        if nproc > 1 and len(chunks) > 1:
            with mp.get_context("fork").Pool(nproc) as pool:
                parts = pool.map(mod._feat, chunks)
        else:
            parts = [mod._feat(c) for c in chunks]
        parts = [p for p in parts if len(p[0])]
        B = np.concatenate([p[0] for p in parts])
        E = np.concatenate([p[1] for p in parts])
        pos = np.concatenate([p[2] for p in parts])[0::2]
        X = torch.from_numpy(np.concatenate([B, E[:, ecols]], axis=1))
        with torch.no_grad():
            for s, m in enumerate(models):
                p = np.concatenate([m(X[k:k + 131072]).view(-1).double().numpy()
                                    for k in range(0, len(X), 131072)])
                out[s, pos] = (p[0::2] + 1.0 - p[1::2]) / 2.0
        print(f"  n_drift {b}: {len(ii):,} games", flush=True)
    miss = np.isnan(out[0]).sum()
    if miss:
        print(f"  n_drift: {miss} rows failed feature extraction; p = 0.5", flush=True)
        out[:, np.isnan(out[0])] = 0.5
    return out if return_seeds else out.mean(axis=0)


def predict(rows, model="n_drift", history=None, nproc=4):
    """p(team0 wins) for each row, under the frozen model `model` (see MODELS)."""
    rows = list(rows)
    if model == "n_drift":
        return predict_drift(rows, history=history, nproc=nproc)
    if model == "p1_site_enriched":
        return predict_p1(rows, nproc=nproc)
    raise ValueError(model)


# ── extraction (local, DB) ──────────────────────────────────────────────

def _db():
    import psycopg2
    url = None
    for line in open(TRAINING.parent / ".env"):
        if line.startswith("DATABASE_URL_RESEARCH="):
            url = line.split("=", 1)[1].strip().strip('"').strip("'")
    conn = psycopg2.connect(url)
    conn.set_session(readonly=True)
    return conn


def tranche_versions(cur, tranche):
    cur.execute("SELECT DISTINCT game_version FROM replay_draft_data "
                "WHERE game_version NOT LIKE '2.57%%'")
    vs = sorted((v for (v,) in cur.fetchall() if v), key=bkey)
    return [v for v in vs if tranche_of(v) == tranche]


def extract(tranches, out_dir):
    conn = _db()
    cur = conn.cursor()
    S = sealed_ids()
    for t in tranches:
        if t not in ("Dpre", "G", "Dpost"):
            raise SystemExit(f"extract supports Dpre, G, Dpost (got {t})")
        vs = tranche_versions(cur, t)
        assert vs and all(tranche_of(v) == t for v in vs) and not any(v.startswith("2.57") for v in vs)
        c2 = conn.cursor(name=f"nn_{t}")
        c2.itersize = 50000
        c2.execute("""SELECT replay_id, game_version, EXTRACT(EPOCH FROM game_date)::bigint,
                             game_map, league_tier, avg_mmr, team0_heroes, team1_heroes,
                             team0_bans, team1_bans, winner
                      FROM replay_draft_data WHERE game_version = ANY(%s)""", (vs,))
        rows, dropped = [], 0
        for rid, v, ts, gm, lt, am, t0, t1, b0, b1, w in c2:
            lst = lambda x: (json.loads(x) if isinstance(x, str) else x) or []
            t0, t1 = lst(t0), lst(t1)
            if rid in S or len(t0) != 5 or len(t1) != 5 or w not in (0, 1):
                dropped += 1
                continue
            rows.append({"replay_id": int(rid), "game_version": v, "ts": int(ts), "game_map": gm,
                         "league_tier": lt, "avg_mmr": None if am is None else float(am),
                         "team0_heroes": t0, "team1_heroes": t1,
                         "team0_bans": lst(b0), "team1_bans": lst(b1), "winner": int(w)})
        c2.close()
        rows.sort(key=lambda g: (g["ts"], g["replay_id"]))
        path = Path(out_dir) / f"rows_{t}.pkl.gz"
        with gzip.open(path, "wb") as f:
            pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{t}: {len(rows):,} games ({dropped} dropped: sealed, not 5v5 or no winner) "
              f"builds {vs} -> {path}", flush=True)


def load_rows(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


# ── scoring ─────────────────────────────────────────────────────────────

def score(rows_path, history_paths, tag, out_dir, models, nproc):
    import torch
    torch.set_num_threads(max(1, nproc))
    rows = guard(load_rows(rows_path), "rows")
    hist = []
    for h in history_paths:
        hist += guard(load_rows(h), "history")
    rid = np.array([g["replay_id"] for g in rows], np.int64)
    for name in models:
        path = Path(out_dir) / f"{name}_{tag}.npz"
        if path.exists():
            print(f"skip {path} (exists)")
            continue
        t0 = time.time()
        extra = {}
        if name == "n_drift":
            ps = predict_drift(rows, history=hist, nproc=nproc, return_seeds=True)
            p = ps.mean(axis=0)
            extra["p_seeds"] = ps.astype(np.float32)
        else:
            p = predict_p1(rows, nproc=nproc)
        np.savez(path, replay_id=rid, p=p.astype(np.float64), model=name, tag=tag, **extra)
        print(f"{name} {tag}: {len(rows):,} games in {time.time() - t0:.0f}s -> {path}", flush=True)


# ── metrics (local; reads labels from the rows file) ──────────────────

def fit_logistic(x, y, iters=50):
    """y ~ sigmoid(a + b x) by Newton. -> (a, b)."""
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def metrics(p, y, n_boot=2000, seed=0):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    y = np.asarray(y, np.float64)
    ll_i = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    lg = np.log(p / (1 - p))
    a, b = fit_logistic(lg, y)
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.array([ll_i[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {"n": int(n), "log_loss": float(ll_i.mean()),
            "log_loss_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "accuracy": float(np.mean((p > 0.5) == (y > 0.5))),
            "brier": float(np.mean((p - y) ** 2)),
            "cal_slope": b, "cal_intercept": a, "base_rate_team0": float(y.mean())}


def paired_diff(pa, pb, y, n_boot=2000, seed=1):
    """log loss of a minus b, with a game-level bootstrap CI."""
    f = lambda p: -(y * np.log(np.clip(p, 1e-6, 1 - 1e-6)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    d = f(pa) - f(pb)
    rng = np.random.default_rng(seed)
    n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)),
                                              float(np.percentile(bs, 97.5))]}


def hero_roles():
    """Hero -> role from src/lib/data/hero-roles.ts."""
    import re
    src = (TRAINING.parent / "src" / "lib" / "data" / "hero-roles.ts").read_text()
    body = src.split("HERO_ROLES", 1)[1].split("{", 1)[1].split("}", 1)[0]
    out = {}
    for m in re.finditer(r"""(['"])((?:\\.|(?!\1).)*)\1\s*:\s*'([^']+)'""", body):
        out[m.group(2).replace("\\'", "'")] = m.group(3)
    return out


def role_key(team, roles):
    return tuple(sorted(roles.get(h, "Unknown") for h in team))


def s_role_shares(cache_path):
    """Share of S teams per role multiset (heroes only, no outcomes)."""
    if Path(cache_path).exists():
        return json.load(open(cache_path))
    roles = hero_roles()
    conn = _db()
    cur = conn.cursor()
    vs = tranche_versions(cur, "S")
    cur.execute("""SELECT h, count(*) FROM (
                     SELECT team0_heroes::text h FROM replay_draft_data WHERE game_version = ANY(%s)
                     UNION ALL
                     SELECT team1_heroes::text h FROM replay_draft_data WHERE game_version = ANY(%s)) x
                   GROUP BY h""", (vs, vs))
    cnt, tot, unknown = {}, 0, set()
    for h, c in cur.fetchall():
        team = json.loads(h) if h else []
        if len(team) != 5:
            continue
        unknown |= {x for x in team if x not in roles}
        k = "|".join(role_key(team, roles))
        cnt[k] = cnt.get(k, 0) + c
        tot += c
    res = {"n_teams": tot, "n_builds": len(vs), "unknown_heroes": sorted(unknown),
           "share": {k: v / tot for k, v in cnt.items()}}
    json.dump(res, open(cache_path, "w"))
    return res


def report(out_dir, tags, out_json, n_boot):
    out_dir = Path(out_dir)
    roles = hero_roles()
    shares = s_role_shares(out_dir / "s_role_shares.json")
    res = {"models": {}, "o5": {"threshold_share": 0.002, "n_s_teams": shares["n_teams"],
                                "unknown_heroes_in_S": shares["unknown_heroes"]},
           "tranches": {}}
    test_ids = set(json.load(open(TRAINING / "paper1_revision" / "site" / "cache" /
                                  "paper_test_ids.json")))
    from overfit2026.data import splitmix64
    is_val = lambda r: splitmix64(int(r) * 998244353 + 23) % 50 == 0
    for tag in tags:
        rows = load_rows(out_dir / f"rows_{tag}.pkl.gz")
        rid = np.array([g["replay_id"] for g in rows], np.int64)
        y = np.array([1.0 if g["winner"] == 0 else 0.0 for g in rows])
        ver = np.array([g["game_version"] for g in rows])
        ts = np.array([g["ts"] for g in rows], np.int64)
        in_snap = rid <= SNAPSHOT_BOUND
        p1_train = in_snap & ~np.isin(rid, list(test_ids)) & ~np.array([is_val(r) for r in rid])
        unk = np.array([site_tier(g.get("league_tier"), g.get("avg_mmr")) == "unknown" for g in rows])
        P = {}
        for name in MODELS:
            z = np.load(out_dir / f"{name}_{tag}.npz")
            assert np.array_equal(z["replay_id"], rid)
            P[name] = z["p"]
        T = res["tranches"].setdefault(tag, {"n": int(len(rows)), "builds": {}, "subsets": {}})
        T["n_in_paper1_snapshot"] = int(in_snap.sum())
        T["n_in_paper1_train"] = int(p1_train.sum())
        T["n_site_tier_unknown"] = int(unk.sum())
        for name in MODELS:
            T.setdefault("all", {})[name] = metrics(P[name], y, n_boot)
        T["all"]["ll_diff_n_drift_minus_p1"] = paired_diff(P["n_drift"], P["p1_site_enriched"], y, n_boot)
        subsets = {}
        if in_snap.any():
            subsets["post_snapshot_upload"] = ~in_snap
        subsets["site_tier_known"] = ~unk
        if tag == "Dpost":
            rare = np.array([shares["share"].get("|".join(role_key(g["team0_heroes"], roles)), 0.0) <= 0.002
                             or shares["share"].get("|".join(role_key(g["team1_heroes"], roles)), 0.0) <= 0.002
                             for g in rows])
            subsets["O5_rare_role_multiset"] = rare
            for v in sorted(set(ver), key=bkey):
                order = np.where(ver == v)[0]
                order = order[np.lexsort((rid[order], ts[order]))]
                for k in (1000, 10000):
                    m = np.zeros(len(rows), bool)
                    m[order[:k]] = True
                    subsets[f"O2_first{k // 1000}k_{v}"] = m
        for sname, m in subsets.items():
            if m.sum() == 0:
                continue
            S_ = T["subsets"].setdefault(sname, {"n": int(m.sum())})
            for name in MODELS:
                S_[name] = metrics(P[name][m], y[m], n_boot)
            S_["ll_diff_n_drift_minus_p1"] = paired_diff(P["n_drift"][m], P["p1_site_enriched"][m],
                                                         y[m], n_boot)
        for v in sorted(set(ver), key=bkey):
            m = ver == v
            B = T["builds"].setdefault(v, {"n": int(m.sum())})
            for name in MODELS:
                B[name] = metrics(P[name][m], y[m], 200)
        print(f"{tag}: done", flush=True)
    res["models"] = {
        "n_drift": {"checkpoints": [f"training/drift2026/models/d2c_cumprev_s{s}.pt" for s in DRIFT_SEEDS],
                    "combine": "mean of 3 seeds, each swap-symmetrized",
                    "weights_train_cutoff": "builds <= 2.55.14.95918 (game_date <= 2026-02-10)",
                    "stats": "cumulative through the previous build (legacy tiers)",
                    "in_sample_tranches": []},
        "p1_site_enriched": {"checkpoint": f"training/paper1_revision/site/models/enriched_s{P1_SEED}.pt",
                             "combine": "selected seed, swap-symmetrized (features re-extracted per order)",
                             "weights_train_cutoff": "paper-1 snapshot, replay_id <= 63,653,039 "
                                                     "(builds through 2.55.16.97039, played to 2026-05-22)",
                             "stats": "frozen deploy stats = all snapshot training games (site tiers)",
                             "in_sample_tranches": ["Dpre", "G (snapshot part)"]},
    }
    json.dump(res, open(out_json, "w"), indent=1)
    print(f"wrote {out_json}")


def main():
    ap = argparse.ArgumentParser(description="N contrast scoring")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("tranches", nargs="+")
    e.add_argument("--out-dir", default=str(OUT_DIR))
    s = sub.add_parser("score")
    s.add_argument("--rows", required=True)
    s.add_argument("--history", nargs="*", default=[])
    s.add_argument("--tag", required=True)
    s.add_argument("--out-dir", default=str(OUT_DIR))
    s.add_argument("--models", default=",".join(MODELS))
    s.add_argument("--nproc", type=int, default=4)
    r = sub.add_parser("report")
    r.add_argument("--dir", default=str(OUT_DIR))
    r.add_argument("--tags", default="Dpre,G,Dpost")
    r.add_argument("--out", required=True)
    r.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    if a.cmd == "extract":
        os.makedirs(a.out_dir, exist_ok=True)
        extract(a.tranches, a.out_dir)
    elif a.cmd == "score":
        os.makedirs(a.out_dir, exist_ok=True)
        score(a.rows, a.history, a.tag, a.out_dir, a.models.split(","), a.nproc)
    else:
        report(a.dir, a.tags.split(","), a.out, a.n_boot)


if __name__ == "__main__":
    main()
