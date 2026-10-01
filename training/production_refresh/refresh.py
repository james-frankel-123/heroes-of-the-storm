"""
Production model refresh — the drift paper's recommendation, in production.

Retrains the three site models on the FULL live corpus through today with
DECAYED AGGREGATE statistics (exponential decay, half-life 90 days — the
paper's q7_decayed90 flavor), and emits a matching stats artifact for the
site so training and serving see the same statistics. Runs on a standing
cadence (see cadence.sh); each run writes to a dated directory.

Phases (subcommands; `all` chains them):
  stats   decayed90 stats from live replay_draft_data -> training stats JSON
          (frozen_stats schema for StatsCache) + site artifact
          (src/lib/data/draft-stats-decayed.json shape) + keep/exclude id sets
  data    fresh corpus load (REPLAY_SNAPSHOT=0, 2.55 only), 98/2 split,
          WP feature cache (decayed-stats features)
  wp      WinProbEnrichedModel 283-d [256,128], seeds {42,123,777}, keep best
  partial partial-draft WP (drafter projections; recent-500K cap, step embed)
  gd      generic_draft_0 (behavior cloning; benefits from fresh meta data)
  mcts    2 seeds at 400 sims (300K episodes; F_400sim operating point).
          Completion = process exit 0 + draft_policy.pt (NEVER file existence
          alone: the worker checkpoints DURING training)
  select  pick the deployed seed with an INDEPENDENT judge, not the worker's
          own eval WP (a proxy: the policy's score under the value function
          it searched against). Each seed's policy argmax drafts a fixed
          benchmark against the fresh GD model; drafts are scored by a
          realized-outcome index (overfit2026/gold.RealizedIndex) fitted on
          the last JUDGE_DAYS of real games. Rule: highest judge score among
          seeds whose degenerate-comp rate is within DEGEN_TOL of the best.
  gates   check every deploy gate, record them in refresh_meta.json, exit
          nonzero if any fails (export runs this first)
  export  ONNX -> public/models/ via export_site_models.py (env-overridden
          paths) + copy site stats artifact into src/lib/data/
  all     everything in order; refuses to export if gates fail

Why 400 sims and judge-based selection (training/overfit2026/REPORT.md;
paper/paper 1/REVISION_NOTES.md section 3.8): 400 -> 800 training sims raises
the proxy but not independent judges, even with a leak-free value function,
and picking the seed with the best proxy rewards exploiting the value
function's errors.

Deploy gates (checked before export):
  - WP best test acc >= GATE_WP_MIN (drift-era models land ~57-58%)
  - WP calibration slope within GATE_CAL_SLOPE
  - partial-WP overall test acc >= GATE_PARTIAL_MIN
  - selected MCTS seed: judge score > population-greedy baseline (GD argmax
    vs the same GD opponent) under the same judge on the same benchmark
  - selected MCTS seed: proxy eval WP >= GATE_MCTS_PROXY_FLOOR (sanity only)
  - export parity asserts (in export_site_models.py) must pass

Usage:
    source .env first (DATABASE_URL required), then e.g.
    python training/production_refresh/refresh.py all
    python training/production_refresh/refresh.py stats
"""
import argparse
import datetime
import gzip
import json
import math
import os
import subprocess
import sys
import time

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(TRAINING_DIR)
BASE = os.path.dirname(os.path.abspath(__file__))

RUN_DATE = os.environ.get("REFRESH_DATE") or datetime.date.today().isoformat()
RUN_DIR = os.path.join(BASE, RUN_DATE)
STATS_JSON = os.path.join(RUN_DIR, "stats_decayed90.json")
# Out-of-fold training stats: fold k = replay_id % OOF_FOLDS. Rows in fold k
# get features from every OTHER fold's games, so no training (or test) row's
# features contain its own outcome. STATS_JSON (all games) is only for
# serving, search and the site artifact. Leaky training stats made the WP
# overconfident on unseen games (calibration slope 0.61 vs 1.0-1.1 out of
# fold; training/overfit2026/REPORT.md).
# skill_tier uses the site's scheme (low = Bronze+Silver, mid = Gold+Platinum,
# high = Diamond+Master; relabelled 2026-09-30). Rows with no tier or MMR are
# 'unknown' and carry no tier information, so they are left out of training.
TRAIN_TIERS = ("low", "mid", "high")
OOF_FOLDS = 5
REFRESH_WORKERS = 16  # CPU pool size (cadence.sh also pins to 16 cores)
OOF_STATS_JSON = os.path.join(RUN_DIR, "stats_decayed90_oof{k}.json")
SITE_STATS_JSON = os.path.join(RUN_DIR, "draft-stats-decayed.json")
EXCLUDE_IDS_JSON = os.path.join(RUN_DIR, "pre255_exclude_ids.json")
FEATURE_CACHE_TRAIN = os.path.join(RUN_DIR, "wp_features_train.npz")
FEATURE_CACHE_TEST = os.path.join(RUN_DIR, "wp_features_test.npz")
WP_PT = os.path.join(RUN_DIR, "wp_enriched_256.pt")
PARTIAL_PT = os.path.join(RUN_DIR, "partial_wp_prod.pt")
GD_PT = os.path.join(RUN_DIR, "generic_draft_0.pt")
META_JSON = os.path.join(RUN_DIR, "refresh_meta.json")

HALF_LIFE_DAYS = 90.0
WP_SEEDS = [42, 123, 777]
MCTS_SEEDS = [0, 1]   # 2 seeds: HotS work is capped to ~25% of this shared box
MCTS_SIMS = 400       # operating point; 800 buys proxy WP, not judged quality
MCTS_EPISODES = int(os.environ.get("REFRESH_MCTS_EPISODES", "300000"))
GATE_WP_MIN = 56.0    # % test accuracy floor
GATE_PARTIAL_MIN = 0.525  # partial-WP overall test acc floor (all-step mix)
# Proxy sanity floor only (catches broken training). Leak-free 400-sim runs
# land ~0.73-0.74 on the paper stats; decayed-stats 800-sim runs 0.75-0.80.
GATE_MCTS_PROXY_FLOOR = 0.70
# Seed selection (phase select)
JUDGE_DAYS = 90           # realized-index window, ending at the stats ref date
JUDGE_SALT = 99           # gold.RealizedIndex fold-hash salt
BENCH_DRAFTS = 1260       # 14 maps x 3 tiers x 2 sides x 15
BENCH_SEED = 20261101     # GD opponent sampling; same for every policy
DEGEN_TOL = 0.02          # eligible seeds: degen rate <= best seed's + 2pp
SELECT_JSON = os.path.join(RUN_DIR, "seed_selection.json")
GATE_CAL_SLOPE = (0.8, 1.25)  # WP calibration slope on out-of-fold test rows
MCTS_MIN_FREE_MIB = 16000  # only launch MCTS seeds on GPUs with this much free

# refresh.py lives under training/; make training modules importable and pin
# the stats override BEFORE any sweep_enriched_wp import.
sys.path.insert(0, TRAINING_DIR)
os.environ["WP_STATS_PATH"] = STATS_JSON
os.environ["REPLAY_SNAPSHOT"] = "0"


def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def meta_update(**kv):
    m = {}
    if os.path.exists(META_JSON):
        m = json.load(open(META_JSON))
    m.update(kv)
    json.dump(m, open(META_JSON, "w"), indent=2)


# ── Phase: stats ─────────────────────────────────────────────────────

def _db_conn():
    import psycopg2
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL required (source .env)")
    return psycopg2.connect(url)


def _merge_cells(folds, skip=None):
    """Sum per-fold decayed count cells (optionally leaving one fold out)."""
    out = {}
    for k, cells in enumerate(folds):
        if k == skip:
            continue
        for tier, c in cells.items():
            d = out.setdefault(tier, {"games": 0.0, "bans": {}, "hero": {},
                                      "hmap": {}, "with": {}, "against": {}})
            d["games"] += c["games"]
            for h, v in c["bans"].items():
                d["bans"][h] = d["bans"].get(h, 0.0) + v
            for key in ("hero", "hmap", "with", "against"):
                dd = d[key]
                for kk, (g, wn) in c[key].items():
                    e = dd.get(kk)
                    if e is None:
                        dd[kk] = [g, wn]
                    else:
                        e[0] += g
                        e[1] += wn
    return out


def _write_training_stats(cells, path, n_used, note):
    """Training stats JSON (frozen_stats schema). Storage thresholds mirror the
    drift decayed arm: hero>=20, map>=5, pair>=10 DECAYED effective games;
    consumers apply their own 30/50 reliability gates on top."""
    hero_stats, hero_map_stats, pairwise_stats = [], [], []
    for tier, cell in cells.items():
        total = cell["games"]
        if total <= 0:
            continue
        for h, (g, wn) in cell["hero"].items():
            if g < 20:
                continue
            hero_stats.append({
                "hero": h, "tier": tier, "games": round(g, 1),
                "win_rate": round(100.0 * wn / g, 3),
                "pick_rate": round(100.0 * g / total, 3),
                "ban_rate": round(100.0 * cell["bans"].get(h, 0.0) / total, 3)})
        for (m, h), (g, wn) in cell["hmap"].items():
            if g < 5:
                continue
            hero_map_stats.append({
                "hero": h, "map": m, "tier": tier, "games": round(g, 1),
                "win_rate": round(100.0 * wn / g, 3)})
        for (a, b), (g, wn) in cell["with"].items():
            if g < 10:
                continue
            wr = round(100.0 * wn / g, 3)
            for x, y in ((a, b), (b, a)):
                pairwise_stats.append({
                    "hero_a": x, "hero_b": y, "tier": tier,
                    "relationship": "with", "win_rate": wr,
                    "games": round(g, 1)})
        for (a, b), (g, wa) in cell["against"].items():
            if g < 10:
                continue
            wr_a = round(100.0 * wa / g, 3)
            pairwise_stats.append({
                "hero_a": a, "hero_b": b, "tier": tier,
                "relationship": "against", "win_rate": wr_a,
                "games": round(g, 1)})
            pairwise_stats.append({
                "hero_a": b, "hero_b": a, "tier": tier,
                "relationship": "against", "win_rate": round(100.0 - wr_a, 3),
                "games": round(g, 1)})

    json.dump({
        "_meta": {"snapshot_date": RUN_DATE, "patch": "2.55",
                  "kind": "decayed90", "half_life_days": HALF_LIFE_DAYS,
                  "source": "live replay_draft_data (own corpus)",
                  "games_used": n_used, "note": note},
        "hero_stats": hero_stats,
        "hero_map_stats": hero_map_stats,
        "pairwise_stats": pairwise_stats,
    }, open(path, "w"))
    log(f"wrote {path} ({len(hero_stats)} hero, {len(hero_map_stats)} "
        f"hero-map, {len(pairwise_stats)} pairwise rows)")
    return hero_stats, hero_map_stats, pairwise_stats


def phase_stats():
    """Per-game exponentially decayed counts over the live 2.55 corpus.
    Equivalent to the paper's per-build incremental decay up to within-build
    granularity (weights compose multiplicatively either way)."""
    from drift2026.build_patch_stats import HP_ROLE_MAP  # exact role mapping
    from shared import HERO_ROLE_FINE

    os.makedirs(RUN_DIR, exist_ok=True)
    conn = _db_conn()
    cur = conn.cursor()
    cur.execute("SELECT COALESCE(max(game_date), now()) FROM replay_draft_data")
    ref_date = cur.fetchone()[0]
    ref_ord = ref_date.toordinal() + (ref_date.hour / 24.0)
    log(f"decay reference date: {ref_date}")

    cur.execute("""
        SELECT replay_id, game_map, skill_tier, team0_heroes, team1_heroes,
               team0_bans, team1_bans, winner, game_date, game_version
        FROM replay_draft_data ORDER BY replay_id""")

    def new_cell():
        return {"games": 0.0, "bans": {}, "hero": {}, "hmap": {},
                "with": {}, "against": {}}

    def bump(d, key, w, win_w):
        e = d.get(key)
        if e is None:
            d[key] = [w, win_w]
        else:
            e[0] += w
            e[1] += win_w

    folds = [{} for _ in range(OOF_FOLDS)]   # fold -> tier -> cell
    exclude_ids = []
    n_used = 0
    t0 = time.time()
    while True:
        batch = cur.fetchmany(50_000)
        if not batch:
            break
        for (rid, gmap, tier, t0h, t1h, t0b, t1b, winner, gdate, gver) in batch:
            if not (gver or "").startswith("2.55"):
                exclude_ids.append(rid)
                continue
            teams = []
            for raw in (t0h, t1h):
                teams.append(json.loads(raw) if isinstance(raw, str) else (raw or []))
            bans = []
            for raw in (t0b, t1b):
                b = json.loads(raw) if isinstance(raw, str) else (raw or [])
                bans.extend(b)
            if gdate is None or len(teams[0]) != 5 or len(teams[1]) != 5:
                continue
            if tier not in TRAIN_TIERS:   # 'unknown': no league tier or MMR
                continue
            age = max(0.0, ref_ord - (gdate.toordinal() + gdate.hour / 24.0))
            w = 0.5 ** (age / HALF_LIFE_DAYS)
            cells = folds[rid % OOF_FOLDS]
            cell = cells.get(tier)
            if cell is None:
                cell = cells[tier] = new_cell()
            cell["games"] += w
            for h in set(bans):
                cell["bans"][h] = cell["bans"].get(h, 0.0) + w
            for ti, heroes in enumerate(teams):
                team_won = (winner == ti)
                win_w = w if team_won else 0.0
                for h in heroes:
                    bump(cell["hero"], h, w, win_w)
                    bump(cell["hmap"], (gmap, h), w, win_w)
                hs = sorted(heroes)
                for i in range(5):
                    for j in range(i + 1, 5):
                        bump(cell["with"], (hs[i], hs[j]), w, win_w)
            for a in teams[0]:
                for b in teams[1]:
                    key = (a, b) if a < b else (b, a)
                    wins_of_a = w if winner == 0 else 0.0
                    if a < b:
                        bump(cell["against"], key, w, wins_of_a)
                    else:
                        bump(cell["against"], key, w, w - wins_of_a)
            n_used += 1
    cur.close()
    conn.close()
    cells = _merge_cells(folds)
    log(f"counted {n_used:,} games ({len(exclude_ids):,} pre-2.55 excluded) "
        f"in {time.time() - t0:.0f}s; effective decayed games/tier: "
        + ", ".join(f"{t}={c['games']:.0f}" for t, c in sorted(cells.items())))

    hero_stats, hero_map_stats, pairwise_stats = _write_training_stats(
        cells, STATS_JSON, n_used, "all games (serving / search / site)")
    for k in range(OOF_FOLDS):
        _write_training_stats(_merge_cells(folds, skip=k),
                              OOF_STATS_JSON.format(k=k), n_used,
                              f"out-of-fold: excludes fold {k} (replay_id % "
                              f"{OOF_FOLDS} == {k})")

    # Site artifact: same numbers reshaped for getDraftData. Sub-threshold
    # rows are dropped at generation (browser thresholds: pairwise>=30 games,
    # hero-map>=50 — omitted rows behave identically to below-threshold).
    site = {"_meta": {"generated": RUN_DATE, "halfLifeDays": HALF_LIFE_DAYS,
                      "source": "own corpus, decayed aggregates (drift paper)"},
            "tiers": {}}
    by_tier = site["tiers"]
    for r in hero_stats:
        t = by_tier.setdefault(r["tier"], {"heroStats": {}, "heroMapWinRates": {},
                                           "synergies": {}, "counters": {}})
        t["heroStats"][r["hero"]] = {
            "winRate": r["win_rate"], "pickRate": r["pick_rate"],
            "banRate": r["ban_rate"], "games": r["games"]}
    for r in hero_map_stats:
        if r["games"] < 50:
            continue
        t = by_tier.setdefault(r["tier"], {"heroStats": {}, "heroMapWinRates": {},
                                           "synergies": {}, "counters": {}})
        t["heroMapWinRates"].setdefault(r["map"], {})[r["hero"]] = {
            "winRate": r["win_rate"], "games": r["games"]}
    for r in pairwise_stats:
        if r["games"] < 30:
            continue
        t = by_tier.setdefault(r["tier"], {"heroStats": {}, "heroMapWinRates": {},
                                           "synergies": {}, "counters": {}})
        sec = "synergies" if r["relationship"] == "with" else "counters"
        t[sec].setdefault(r["hero_a"], {})[r["hero_b"]] = {
            "winRate": r["win_rate"], "games": round(r["games"])}
    json.dump(site, open(SITE_STATS_JSON, "w"))
    log(f"wrote {SITE_STATS_JSON} ({os.path.getsize(SITE_STATS_JSON) // 1024} KB)")

    json.dump(exclude_ids, open(EXCLUDE_IDS_JSON, "w"))
    meta_update(stats={"games_used": n_used, "ref_date": str(ref_date),
                       "excluded_pre255": len(exclude_ids)})


# ── Phase: data ──────────────────────────────────────────────────────

def _load_fresh_corpus():
    import shared
    # Always read the DB: a local cache can hold stale labels (skill_tier was
    # relabelled in place on 2026-09-30).
    force = True
    rows = shared.load_replay_data(force_refresh=force)
    exclude = set(json.load(open(EXCLUDE_IDS_JSON)))
    rows = [r for r in rows if r["replay_id"] not in exclude
            and r.get("skill_tier") in TRAIN_TIERS]
    log(f"corpus: {len(rows):,} 2.55 replays")
    return rows


def oof_stats_cache(k):
    """StatsCache over fold k's out-of-fold stats (compositions unchanged)."""
    from sweep_enriched_wp import StatsCache
    st = StatsCache.__new__(StatsCache)
    st._load_frozen(OOF_STATS_JSON.format(k=k))
    st._load_compositions()
    return st


def oof_features(rows, cache_path):
    """Feature cache where every row's aggregate features come from the
    stats that exclude its own fold (replay_id % OOF_FOLDS)."""
    import numpy as np
    from sweep_enriched_wp import precompute_all_features
    parts = []
    for k in range(OOF_FOLDS):
        fold_rows = [r for r in rows if r["replay_id"] % OOF_FOLDS == k]
        if fold_rows:
            b, e, y = precompute_all_features(fold_rows, oof_stats_cache(k),
                                              num_workers=REFRESH_WORKERS)
            parts.append((b.numpy(), e.numpy(), y.numpy()))
    np.savez(cache_path,
             bases=np.concatenate([p[0] for p in parts]),
             enricheds=np.concatenate([p[1] for p in parts]),
             labels=np.concatenate([p[2] for p in parts]))


def phase_data():
    from shared import split_data

    rows = _load_fresh_corpus()
    train_rows, test_rows = split_data(rows, test_frac=0.02, seed=42)
    log(f"building out-of-fold WP feature caches ({len(train_rows):,} train / "
        f"{len(test_rows):,} test, {OOF_FOLDS} folds)")
    oof_features(train_rows, FEATURE_CACHE_TRAIN)
    oof_features(test_rows, FEATURE_CACHE_TEST)
    meta_update(data={"train": len(train_rows), "test": len(test_rows),
                      "oof_folds": OOF_FOLDS})


# ── Phase: wp ────────────────────────────────────────────────────────

def phase_wp():
    import numpy as np
    import torch
    from sweep_enriched_wp import WinProbEnrichedModel, compute_group_indices
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    from retrain_frozen_stats import train_wp_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # The cache holds ALL enriched groups; the site model uses the 9-group
    # ENRICHED_GROUPS preset (86 of the enriched columns) — same column
    # selection as rerun2026's train_jobs._cols_for.
    gi = compute_group_indices()
    cols = []
    for g in ENRICHED_GROUPS:
        s, e = gi[g]
        cols.extend(range(s, e))

    def tensors(path):
        z = np.load(path)
        X = np.concatenate([z["bases"], z["enricheds"][:, cols]], axis=1)
        return (torch.tensor(X, dtype=torch.float32).to(device),
                torch.tensor(z["labels"], dtype=torch.float32).to(device))

    train_X, train_y = tensors(FEATURE_CACHE_TRAIN)
    test_X, test_y = tensors(FEATURE_CACHE_TEST)
    dim = train_X.shape[1]
    assert dim == 283, f"expected 283-d features, got {dim}"

    best = {"acc": -1.0}
    accs = []
    for seed in WP_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = WinProbEnrichedModel(dim, [256, 128], dropout=0.3)
        model, acc = train_wp_model(model, train_X, test_X, train_y, test_y,
                                    f"prod_wp-s{seed}", device)
        accs.append(acc)
        if acc > best["acc"]:
            best = {"acc": acc, "seed": seed,
                    "state": {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}}
    torch.save(best["state"], WP_PT)
    model = WinProbEnrichedModel(dim, [256, 128], dropout=0.3)
    model.load_state_dict(best["state"])
    model.to(device).eval()
    with torch.no_grad():
        p = model(test_X).clamp(1e-6, 1 - 1e-6).cpu().numpy()
    slope = calibration_slope(p, test_y.cpu().numpy())
    log(f"WP best acc {best['acc']:.2f}% (seed {best['seed']}; all {accs}), "
        f"calibration slope {slope:.3f} on out-of-fold test rows -> {WP_PT}")
    meta_update(wp={"best_acc": best["acc"], "best_seed": best["seed"],
                    "all_accs": accs, "cal_slope": slope})


def calibration_slope(p, y):
    """Slope of a logistic regression of y on logit(p). 1.0 = calibrated;
    well below 1 = overconfident (the signature of label leakage)."""
    import numpy as np
    x = np.log(p / (1 - p))
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(50):
        q = 1 / (1 + np.exp(-X @ w))
        H = (X * (q * (1 - q))[:, None]).T @ X
        step = np.linalg.solve(H, X.T @ (y - q))
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[1])


# ── Phase: partial ───────────────────────────────────────────────────

def phase_partial():
    """Partial-draft WP model (drafter projections; 4th site model since
    2026-08-07). Trained on the same fresh corpus + decayed stats as the WP
    model, capped to the most recent 500K replays for tractable per-step
    feature extraction."""
    import torch
    env = dict(os.environ)
    env.update({"PARTIAL_WP_MAX_REPLAYS": "500000", "PARTIAL_WP_OUT": PARTIAL_PT,
                "WP_STATS_PATH": STATS_JSON, "REPLAY_SNAPSHOT": "0",
                "PARTIAL_WP_OOF_STATS": OOF_STATS_JSON,
                "PARTIAL_WP_OOF_FOLDS": str(OOF_FOLDS)})
    subprocess.run([sys.executable, "-u",
                    os.path.join(TRAINING_DIR, "train_partial_wp.py")],
                   env=env, check=True, cwd=TRAINING_DIR)
    ckpt = torch.load(PARTIAL_PT, weights_only=True, map_location="cpu")
    acc = float(ckpt.get("best_test_acc", 0.0))
    meta_update(partial={"best_acc": acc})
    log(f"partial-WP done (overall test acc {acc:.4f}) -> {PARTIAL_PT}")


# ── Phase: gd ────────────────────────────────────────────────────────

def phase_gd():
    import torch
    import train_generic_draft as tgd
    from shared import split_data

    # all of tgd's saves are relative to its __file__; redirect into RUN_DIR
    tgd.__file__ = os.path.join(RUN_DIR, "train_generic_draft.py")
    rows = _load_fresh_corpus()
    train_rows, test_rows = split_data(rows, test_frac=0.02, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_ds = tgd.DraftDataset(train_rows)
    test_ds = tgd.DraftDataset(test_rows)
    log(f"GD: {len(train_ds):,} train / {len(test_ds):,} test samples")
    loss = tgd.train_single_model(0, tgd.MODEL_VARIANTS[0], train_ds, test_ds, device)
    meta_update(gd={"best_test_loss": loss})
    log(f"GD variant 0 done (loss {loss:.4f}) -> {GD_PT}")


# ── Phase: mcts ──────────────────────────────────────────────────────

def free_gpus(min_free_mib=MCTS_MIN_FREE_MIB):
    """GPU indices with at least min_free_mib free (other jobs share this box)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    gpus = [int(i) for i, free in (l.split(",") for l in out.strip().splitlines())
            if int(free) >= min_free_mib]
    if not gpus:
        sys.exit(f"no GPU with >= {min_free_mib} MiB free — not training MCTS")
    return gpus


def phase_mcts():
    procs = []
    # cadence.sh pins the whole refresh to one GPU (REFRESH_GPU); the box is
    # shared and HotS work is capped at ~25% of it.
    gpus = ([int(os.environ["REFRESH_GPU"])] if os.environ.get("REFRESH_GPU")
            else free_gpus())
    log(f"MCTS on GPUs {gpus} (>= {MCTS_MIN_FREE_MIB} MiB free each)")
    for seed in MCTS_SEEDS:
        run_dir = os.path.join(RUN_DIR, f"mcts_s{seed}")
        os.makedirs(run_dir, exist_ok=True)
        env = dict(os.environ)
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpus[seed % len(gpus)]),
            "MCTS_SAVE_DIR": run_dir,
            "MCTS_NUM_EPISODES": str(MCTS_EPISODES),
            "MCTS_NUM_SIMS": str(MCTS_SIMS),
            "MCTS_NET_SIZE": "base",
            "MCTS_POLICY_HEAD": "linear",
            "MCTS_WP_MODEL": "enriched_full",
            "MCTS_WP_PATH": WP_PT,
            "MCTS_GD_PATH": GD_PT,
            "MCTS_EXCLUDE_IDS": EXCLUDE_IDS_JSON,
            "MCTS_FRESH": "1",
            "MCTS_BATCH_EPISODES": "128",
            "WANDB_RUN_NAME": f"prod_refresh_{RUN_DATE}_s{seed}",
            "WP_STATS_PATH": STATS_JSON,
            "REPLAY_SNAPSHOT": "0",
        })
        logf = open(os.path.join(run_dir, "train.log"), "w")
        p = subprocess.Popen(
            [sys.executable, "-u", os.path.join(TRAINING_DIR, "train_mcts_worker.py")],
            env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=TRAINING_DIR)
        procs.append((seed, p, run_dir))
        log(f"launched MCTS seed {seed} (pid {p.pid}, "
            f"gpu {gpus[seed % len(gpus)]})")

    # Completion = exit code 0 AND draft_policy.pt present. The worker saves
    # draft_policy.pt DURING training on every new-best eval, so the file
    # alone is NEVER a completion signal.
    results = {}
    for seed, p, run_dir in procs:
        rc = p.wait()
        ckpt = os.path.join(run_dir, "draft_policy.pt")
        ok = rc == 0 and os.path.exists(ckpt)
        # Worker eval lines: "  EVAL @ {ep}: avg_wp=0.7712 win_rate=..."
        # best checkpoint = max avg_wp (worker saves draft_policy.pt on new best)
        best_wp = None
        logp = os.path.join(run_dir, "train.log")
        for line in open(logp, errors="replace"):
            if "avg_wp=" in line:
                try:
                    v = float(line.rsplit("avg_wp=", 1)[1].split()[0].rstrip(","))
                    best_wp = v if best_wp is None else max(best_wp, v)
                except ValueError:
                    pass
        results[seed] = {"rc": rc, "ok": ok, "best_wp": best_wp}
        log(f"MCTS seed {seed}: rc={rc} ok={ok} best_wp={best_wp}")

    ok_seeds = {s: r for s, r in results.items() if r["ok"] and r["best_wp"]}
    # best_wp here is the PROXY (worker's own eval under the value function it
    # searched against). It is logged, not used to choose: see phase select.
    meta_update(mcts={"results": results, "sims": MCTS_SIMS,
                      "episodes": MCTS_EPISODES,
                      "ok_seeds": sorted(ok_seeds)})
    if not ok_seeds:
        sys.exit("no MCTS seed completed successfully")
    log(f"MCTS done: ok seeds {sorted(ok_seeds)}; proxy best_wp "
        + ", ".join(f"s{s}={r['best_wp']:.4f}" for s, r in sorted(ok_seeds.items())))


# ── Phase: select ────────────────────────────────────────────────────

def _judge_games():
    """Slim games (gold.py format) from the last JUDGE_DAYS of the corpus the
    other phases train on: replay_draft_data, 2.55 only (pre-2.55 exclude
    set), known tiers only, 5v5 with a recorded winner. The window ends at the
    stats phase's decay reference date so a rerun sees the same games."""
    exclude = set(json.load(open(EXCLUDE_IDS_JSON)))
    ref = json.load(open(META_JSON)).get("stats", {}).get("ref_date")
    conn = _db_conn()
    conn.set_session(readonly=True)
    cur = conn.cursor()
    if ref is None:
        cur.execute("SELECT max(game_date) FROM replay_draft_data")
        ref = str(cur.fetchone()[0])
    cur.execute("""
        SELECT replay_id, skill_tier, game_map, team0_heroes, team1_heroes,
               team0_bans, team1_bans, winner
        FROM replay_draft_data
        WHERE game_date > %s::timestamptz - make_interval(days => %s)
          AND game_date <= %s::timestamptz
          AND game_version LIKE '2.55%%'""", (ref, JUDGE_DAYS, ref))

    def lst(x):
        return json.loads(x) if isinstance(x, str) else (x or [])
    games = []
    for rid, tier, gmap, t0, t1, b0, b1, w in cur:
        t0, t1 = tuple(lst(t0)), tuple(lst(t1))
        if (rid in exclude or tier not in TRAIN_TIERS or w not in (0, 1)
                or len(t0) != 5 or len(t1) != 5):
            continue
        games.append((int(rid), tier, gmap, t0, t1,
                      tuple(lst(b0)) + tuple(lst(b1)), int(w)))
    cur.close()
    conn.close()
    return games, ref


def _bench_configs():
    """Fixed benchmark: every (map, tier) cell, both sides, BENCH_DRAFTS total."""
    from shared import MAPS
    n_m, n_t = len(MAPS), len(TRAIN_TIERS)
    return [(i, MAPS[i % n_m], TRAIN_TIERS[(i // n_m) % n_t],
             (i // (n_m * n_t)) % 2) for i in range(BENCH_DRAFTS)]


def _run_bench(choose, gd):
    """Play the benchmark. Our side acts with choose(state, team, step_type);
    the opponent samples the fresh GD model at T=1 (same RNG stream for every
    policy, so drafts are paired by config). Returns [(our, opp, map, tier)]."""
    import random
    import torch
    from shared import HEROES, NUM_HEROES
    from train_draft_policy import DraftState, DRAFT_ORDER
    random.seed(BENCH_SEED)
    torch.manual_seed(BENCH_SEED)
    cpu = torch.device("cpu")
    out = []
    for _, gmap, tier, our in _bench_configs():
        s = DraftState(gmap, tier, our_team=our)
        while not s.is_terminal():
            team, typ = DRAFT_ORDER[s.step]
            if team == our:
                h = choose(s, team, typ)
            else:
                with torch.no_grad():
                    p = torch.softmax(gd(s.to_tensor_gd(cpu), s.valid_mask(cpu)), 1)
                h = torch.multinomial(p, 1).item()
            s.apply_action(h, team, typ)
        ov, pv = ((s.team0_picks, s.team1_picks) if our == 0
                  else (s.team1_picks, s.team0_picks))
        out.append((tuple(HEROES[i] for i in range(NUM_HEROES) if ov[i] > 0),
                    tuple(HEROES[i] for i in range(NUM_HEROES) if pv[i] > 0),
                    gmap, tier))
    return out


def _policy_chooser(path):
    """Site inference mode for the exported policy: masked policy-head argmax
    (same network config export_site_models.py exports)."""
    import torch
    from train_draft_policy import AlphaZeroDraftNet
    net = AlphaZeroDraftNet(size="base", policy_head_type="linear")
    net.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    net.eval()
    cpu = torch.device("cpu")

    def choose(s, team, typ):
        x = s.to_tensor(cpu)
        x[0, -1] = float(team)
        with torch.no_grad():
            logits, _ = net(x, s.valid_mask(cpu))
        return int(logits.argmax(1).item())
    return choose


def _gd_argmax_chooser(gd):
    """Population-greedy baseline: the GD behavior-cloning model's most likely
    pick/ban (what the population would most often do here)."""
    import torch
    cpu = torch.device("cpu")

    def choose(s, team, typ):
        with torch.no_grad():
            return int(gd(s.to_tensor_gd(cpu), s.valid_mask(cpu)).argmax(1).item())
    return choose


def _hero_wr_chooser(stats):
    """Context baseline (not gated): pick/ban the available hero with the
    highest tier win rate in the serving stats."""
    from shared import HEROES

    def choose(s, team, typ):
        wr = stats.hero_wr.get(s.skill_tier, {})
        mask = s.valid_mask_np()
        return max((i for i in range(len(HEROES)) if mask[i] > 0),
                   key=lambda i: wr.get(HEROES[i], 0.0))
    return choose


def _summ(drafts, judge, proxy_fn):
    import numpy as np
    from shared import is_degenerate, HERO_ROLE_FINE
    healers = {h for h, r in HERO_ROLE_FINE.items() if r == "healer"}
    j = np.array([judge.score(o, p, t) for o, p, m, t in drafts])
    px = np.array([0.5 * (proxy_fn(list(o), list(p), m, t)
                          + 1 - proxy_fn(list(p), list(o), m, t))
                   for o, p, m, t in drafts])
    deg = np.array([float(is_degenerate(list(o))) for o, p, m, t in drafts])
    return j, {
        "judge": float(j.mean()), "judge_se": float(j.std(ddof=1) / np.sqrt(len(j))),
        "judge_by_tier": {t: float(np.mean([x for x, d in zip(j, drafts) if d[3] == t]))
                          for t in TRAIN_TIERS},
        "proxy_bench": float(px.mean()),
        "degen": float(deg.mean()),
        "healer": float(np.mean([any(h in healers for h in o) for o, p, m, t in drafts])),
        "distinct_heroes": len({h for o, p, m, t in drafts for h in o}),
    }


def phase_select():
    import pickle
    import numpy as np
    import torch
    from overfit2026 import gold
    from drift2026 import common as dcommon
    from sweep_enriched_wp import (StatsCache, WinProbEnrichedModel,
                                   compute_group_indices)
    from experiment_synthetic_augmentation import ENRICHED_GROUPS, make_eval_fn
    from train_generic_draft import GenericDraftModel

    m = json.load(open(META_JSON))
    results = m.get("mcts", {}).get("results", {})
    ok = sorted(int(s) for s, r in results.items() if r.get("ok") and r.get("best_wp"))
    if not ok:
        sys.exit("select: no completed MCTS seed in refresh_meta.json")

    # Judge: realized outcomes of recent real games, cross-fitted on two hash
    # folds (statistics from one, logistic weights from the other). It never
    # sees the WP's features, weights or predictions.
    dcommon._bind_statscache_methods()
    judge_pkl = os.path.join(RUN_DIR, f"judge_realized_{JUDGE_DAYS}d_s{JUDGE_SALT}.pkl")
    if os.path.exists(judge_pkl):
        with open(judge_pkl, "rb") as f:
            judge, ref = pickle.load(f)
    else:
        t0 = time.time()
        games, ref = _judge_games()
        log(f"judge: {len(games):,} games in the {JUDGE_DAYS} days to {ref}")
        judge = gold.RealizedIndex(games, salt=JUDGE_SALT, name=f"recent{JUDGE_DAYS}d")
        with open(judge_pkl, "wb") as f:
            pickle.dump((judge, ref), f, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"judge built in {time.time() - t0:.0f}s; held-out fold acc "
            + ", ".join(f"{x['acc']:.4f}" for x in judge.fit))

    # Proxy on the same drafts: the fresh WP with the serving stats (what the
    # MCTS searched against), team-order symmetrized.
    st = StatsCache.__new__(StatsCache)
    st._load_frozen(STATS_JSON)
    st._load_compositions()
    gi = compute_group_indices()
    cols = [c for g in ENRICHED_GROUPS for c in range(*gi[g])]
    wp = WinProbEnrichedModel(283, [256, 128], dropout=0.3)
    wp.load_state_dict(torch.load(WP_PT, weights_only=True, map_location="cpu"))
    wp.eval()
    proxy_fn = make_eval_fn(wp, cols, st, torch.device("cpu"))

    gd = GenericDraftModel()
    gd.load_state_dict(torch.load(GD_PT, weights_only=True, map_location="cpu"))
    gd.eval()

    per_draft = {}
    rows = {}
    for name, chooser in ([("pop_greedy", _gd_argmax_chooser(gd)),
                           ("hero_wr_greedy", _hero_wr_chooser(st))]
                          + [(f"s{s}", _policy_chooser(os.path.join(
                              RUN_DIR, f"mcts_s{s}", "draft_policy.pt"))) for s in ok]):
        t0 = time.time()
        drafts = _run_bench(chooser, gd)
        per_draft[name], rows[name] = _summ(drafts, judge, proxy_fn)
        r = rows[name]
        log(f"bench {name:15s} judge {r['judge']:.4f}±{r['judge_se']:.4f} "
            f"proxy_bench {r['proxy_bench']:.4f} degen {r['degen']:.3f} "
            f"distinct {r['distinct_heroes']} ({time.time() - t0:.0f}s)")

    base = per_draft["pop_greedy"]
    seeds = {}
    for s in ok:
        r = rows[f"s{s}"]
        d = per_draft[f"s{s}"] - base
        r.update({"proxy_best_wp": results[str(s)]["best_wp"],
                  "vs_pop_greedy": float(d.mean()),
                  "vs_pop_greedy_se": float(d.std(ddof=1) / np.sqrt(len(d)))})
        seeds[s] = r
    min_degen = min(r["degen"] for r in seeds.values())
    eligible = [s for s in ok if seeds[s]["degen"] <= min_degen + DEGEN_TOL + 1e-12]
    chosen = max(eligible, key=lambda s: seeds[s]["judge"])
    proxy_pick = max(ok, key=lambda s: seeds[s]["proxy_best_wp"])
    sel = {
        "seed": chosen,
        "rule": (f"max judge among seeds with degen <= min degen + {DEGEN_TOL}"),
        "eligible": eligible,
        "proxy_pick_would_have_been": proxy_pick,
        "judge": {"kind": "overfit2026.gold.RealizedIndex", "days": JUDGE_DAYS,
                  "ref_date": str(ref), "salt": JUDGE_SALT, **judge.describe()},
        "bench": {"drafts": BENCH_DRAFTS, "seed": BENCH_SEED,
                  "policy_mode": "policy-head argmax",
                  "opponent": "fresh GD (generic_draft_0.pt), sampled T=1"},
        "seeds": {str(s): r for s, r in seeds.items()},
        "baselines": {"pop_greedy": rows["pop_greedy"],
                      "hero_wr_greedy": rows["hero_wr_greedy"]},
    }
    json.dump({**sel, "per_draft": {k: np.round(v, 5).tolist()
                                    for k, v in per_draft.items()}},
              open(SELECT_JSON, "w"), indent=1)
    meta_update(select=sel)
    log(f"selected MCTS seed {chosen} (judge {seeds[chosen]['judge']:.4f}, "
        f"proxy {seeds[chosen]['proxy_best_wp']:.4f}); proxy rule would have "
        f"picked {proxy_pick}")


# ── Gates ────────────────────────────────────────────────────────────

def check_gates():
    """Evaluate every deploy gate, record all of them, return True iff all pass."""
    m = json.load(open(META_JSON))
    wp = m.get("wp", {})
    sel = m.get("select", {})
    seed = sel.get("seed")
    srow = sel.get("seeds", {}).get(str(seed), {})
    base = sel.get("baselines", {}).get("pop_greedy", {})
    slope = wp.get("cal_slope")
    g = {
        "wp_acc": {"value": wp.get("best_acc"), "min": GATE_WP_MIN},
        "wp_cal_slope": {"value": slope, "range": list(GATE_CAL_SLOPE)},
        "partial_acc": {"value": m.get("partial", {}).get("best_acc"),
                        "min": GATE_PARTIAL_MIN},
        "mcts_judge_vs_pop_greedy": {"value": srow.get("judge"),
                                     "must_exceed": base.get("judge"),
                                     "seed": seed},
        "mcts_proxy_floor": {"value": srow.get("proxy_best_wp"),
                             "min": GATE_MCTS_PROXY_FLOOR, "seed": seed},
    }
    for k in ("wp_acc", "partial_acc", "mcts_proxy_floor"):
        g[k]["pass"] = g[k]["value"] is not None and g[k]["value"] >= g[k]["min"]
    g["wp_cal_slope"]["pass"] = (slope is not None
                                 and GATE_CAL_SLOPE[0] <= slope <= GATE_CAL_SLOPE[1])
    j = g["mcts_judge_vs_pop_greedy"]
    j["pass"] = (j["value"] is not None and j["must_exceed"] is not None
                 and j["value"] > j["must_exceed"])
    ok = all(v["pass"] for v in g.values())
    meta_update(gates={"pass": ok, "checked": datetime.datetime.now().isoformat(),
                       **g})
    for k, v in g.items():
        log(f"gate {k:26s} {'PASS' if v['pass'] else 'FAIL'}  "
            + json.dumps({kk: vv for kk, vv in v.items() if kk != 'pass'}))
    return ok


def phase_gates():
    if not check_gates():
        sys.exit("GATE FAIL — not exporting (see refresh_meta.json 'gates')")
    log("all gates pass")


# ── Phase: export ────────────────────────────────────────────────────

def phase_export():
    phase_gates()
    best_seed = json.load(open(META_JSON))["select"]["seed"]

    env = dict(os.environ)
    env.update({
        "SITE_POLICY_PT": os.path.join(RUN_DIR, f"mcts_s{best_seed}", "draft_policy.pt"),
        "SITE_GD_PT": GD_PT,
        "SITE_WP_PT": WP_PT,
    })
    subprocess.run([sys.executable,
                    os.path.join(TRAINING_DIR, "export_site_models.py")],
                   env=env, check=True, cwd=TRAINING_DIR)
    penv = dict(os.environ)
    penv["PARTIAL_WP_CKPT"] = PARTIAL_PT
    subprocess.run([sys.executable,
                    os.path.join(TRAINING_DIR, "production_refresh", "export_partial_wp.py")],
                   env=penv, check=True, cwd=TRAINING_DIR)

    import shutil
    dst = os.path.join(REPO_DIR, "src", "lib", "data", "draft-stats-decayed.json")
    shutil.copy(SITE_STATS_JSON, dst)
    log(f"site stats artifact -> {dst}")
    meta_update(exported=True, export_date=datetime.datetime.now().isoformat())
    log("export complete — commit public/models/ + src/lib/data/draft-stats-decayed.json to deploy")


PHASES = {"stats": phase_stats, "data": phase_data, "wp": phase_wp,
          "partial": phase_partial, "gd": phase_gd, "mcts": phase_mcts,
          "select": phase_select, "gates": phase_gates,
          "export": phase_export}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=list(PHASES) + ["all"])
    args = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)
    if args.phase == "all":
        for name in ("stats", "data", "wp", "partial", "gd", "mcts", "select",
                     "export"):
            log(f"=== phase {name} ===")
            PHASES[name]()
    else:
        PHASES[args.phase]()


if __name__ == "__main__":
    main()
