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
  gd      generic_draft_0 (behavior cloning; benefits from fresh meta data)
  mcts    4x J_800sim seeds (800 sims, 300K episodes), one per GPU, pick the
          best final eval WP. Completion = process exit 0 + draft_policy.pt
          (NEVER file existence alone: the worker checkpoints DURING training)
  export  ONNX -> public/models/ via export_site_models.py (env-overridden
          paths) + copy site stats artifact into src/lib/data/
  all     everything in order; refuses to export if gates fail

Deploy gates (checked before export):
  - WP best test acc >= GATE_WP_MIN (drift-era models land ~57-58%)
  - MCTS best eval WP >= GATE_MCTS_MIN
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
SITE_STATS_JSON = os.path.join(RUN_DIR, "draft-stats-decayed.json")
EXCLUDE_IDS_JSON = os.path.join(RUN_DIR, "pre255_exclude_ids.json")
FEATURE_CACHE_TRAIN = os.path.join(RUN_DIR, "wp_features_train.npz")
FEATURE_CACHE_TEST = os.path.join(RUN_DIR, "wp_features_test.npz")
WP_PT = os.path.join(RUN_DIR, "wp_enriched_256.pt")
GD_PT = os.path.join(RUN_DIR, "generic_draft_0.pt")
META_JSON = os.path.join(RUN_DIR, "refresh_meta.json")

HALF_LIFE_DAYS = 90.0
WP_SEEDS = [42, 123, 777]
MCTS_SEEDS = [0, 1, 2, 3]
MCTS_SIMS = 800
MCTS_EPISODES = 300_000
GATE_WP_MIN = 56.0    # % test accuracy floor
GATE_MCTS_MIN = 0.70  # eval WP floor (J_800sim best seeds land ~0.77)

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

    cells = {}
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
            age = max(0.0, ref_ord - (gdate.toordinal() + gdate.hour / 24.0))
            w = 0.5 ** (age / HALF_LIFE_DAYS)
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
    log(f"counted {n_used:,} games ({len(exclude_ids):,} pre-2.55 excluded) "
        f"in {time.time() - t0:.0f}s; effective decayed games/tier: "
        + ", ".join(f"{t}={c['games']:.0f}" for t, c in sorted(cells.items())))

    # Training stats JSON (frozen_stats schema; storage thresholds mirror the
    # drift decayed arm: hero>=20, map>=5, pair>=10 DECAYED effective games —
    # consumers apply their own 30/50 reliability gates on top).
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
                  "games_used": n_used},
        "hero_stats": hero_stats,
        "hero_map_stats": hero_map_stats,
        "pairwise_stats": pairwise_stats,
    }, open(STATS_JSON, "w"))
    log(f"wrote {STATS_JSON} ({len(hero_stats)} hero, {len(hero_map_stats)} "
        f"hero-map, {len(pairwise_stats)} pairwise rows)")

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
    # force a fresh DB read (the 24h cache may hold yesterday's corpus)
    if os.path.exists(shared._REPLAY_CACHE_PATH):
        age_h = (time.time() - os.path.getmtime(shared._REPLAY_CACHE_PATH)) / 3600
        force = age_h > 6
    else:
        force = True
    rows = shared.load_replay_data(force_refresh=force)
    exclude = set(json.load(open(EXCLUDE_IDS_JSON)))
    rows = [r for r in rows if r["replay_id"] not in exclude]
    log(f"corpus: {len(rows):,} 2.55 replays")
    return rows


def phase_data():
    from shared import split_data
    from sweep_enriched_wp import StatsCache, precompute_all_features

    rows = _load_fresh_corpus()
    train_rows, test_rows = split_data(rows, test_frac=0.02, seed=42)
    stats = StatsCache()  # WP_STATS_PATH -> decayed stats
    log(f"building WP feature caches ({len(train_rows):,} train / "
        f"{len(test_rows):,} test)")
    precompute_all_features(train_rows, stats, cache_path=FEATURE_CACHE_TRAIN)
    precompute_all_features(test_rows, stats, cache_path=FEATURE_CACHE_TEST)
    meta_update(data={"train": len(train_rows), "test": len(test_rows)})


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
    log(f"WP best acc {best['acc']:.2f}% (seed {best['seed']}; all {accs}) -> {WP_PT}")
    meta_update(wp={"best_acc": best["acc"], "best_seed": best["seed"],
                    "all_accs": accs})


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

def phase_mcts():
    procs = []
    for seed in MCTS_SEEDS:
        run_dir = os.path.join(RUN_DIR, f"mcts_s{seed}")
        os.makedirs(run_dir, exist_ok=True)
        env = dict(os.environ)
        env.update({
            "CUDA_VISIBLE_DEVICES": str(seed % 4),
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
        log(f"launched MCTS seed {seed} (pid {p.pid}, gpu {seed % 4})")

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
    if not ok_seeds:
        meta_update(mcts={"results": results, "best_seed": None})
        sys.exit("no MCTS seed completed successfully")
    best_seed = max(ok_seeds, key=lambda s: ok_seeds[s]["best_wp"])
    meta_update(mcts={"results": results, "best_seed": best_seed,
                      "best_wp": ok_seeds[best_seed]["best_wp"]})
    log(f"best MCTS seed: {best_seed} (eval WP {ok_seeds[best_seed]['best_wp']:.4f})")


# ── Phase: export ────────────────────────────────────────────────────

def phase_export():
    m = json.load(open(META_JSON))
    wp_acc = m.get("wp", {}).get("best_acc", -1)
    mcts_wp = m.get("mcts", {}).get("best_wp", -1)
    best_seed = m.get("mcts", {}).get("best_seed")
    if wp_acc < GATE_WP_MIN:
        sys.exit(f"GATE FAIL: WP acc {wp_acc:.2f}% < {GATE_WP_MIN}% — not exporting")
    if best_seed is None or mcts_wp < GATE_MCTS_MIN:
        sys.exit(f"GATE FAIL: MCTS best_wp {mcts_wp} < {GATE_MCTS_MIN} — not exporting")

    env = dict(os.environ)
    env.update({
        "SITE_POLICY_PT": os.path.join(RUN_DIR, f"mcts_s{best_seed}", "draft_policy.pt"),
        "SITE_GD_PT": GD_PT,
        "SITE_WP_PT": WP_PT,
    })
    subprocess.run([sys.executable,
                    os.path.join(TRAINING_DIR, "export_site_models.py")],
                   env=env, check=True, cwd=TRAINING_DIR)

    import shutil
    dst = os.path.join(REPO_DIR, "src", "lib", "data", "draft-stats-decayed.json")
    shutil.copy(SITE_STATS_JSON, dst)
    log(f"site stats artifact -> {dst}")
    meta_update(exported=True, export_date=datetime.datetime.now().isoformat())
    log("export complete — commit public/models/ + src/lib/data/draft-stats-decayed.json to deploy")


PHASES = {"stats": phase_stats, "data": phase_data, "wp": phase_wp,
          "gd": phase_gd, "mcts": phase_mcts, "export": phase_export}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=list(PHASES) + ["all"])
    args = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)
    if args.phase == "all":
        for name in ("stats", "data", "wp", "gd", "mcts", "export"):
            log(f"=== phase {name} ===")
            PHASES[name]()
    else:
        PHASES[args.phase]()


if __name__ == "__main__":
    main()
