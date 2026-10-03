"""
Production model refresh — the drift paper's recommendation, in production.

Retrains the three site models on the FULL live corpus through today with
DECAYED AGGREGATE statistics (exponential decay, half-life 90 days — the
paper's q7_decayed90 flavor), and emits a matching stats artifact for the
site so training and serving see the same statistics. Runs on a standing
cadence (see cadence.sh); each run writes to a dated directory.

Encoding: HOTS_HERO_SET=v2 (set below): 91 heroes (Xal'atath, patch 2.57,
appended) and 15 maps (Haunted Mines appended); the MCTS kernel is the
separate cuda_mcts/h91 build. Games from patch 2.55 on are used; decayed win
rates are shrunk toward role / hero / additive priors (HERO_PRIOR_GAMES etc.),
so a new hero with few games gets sane statistics.

Phases (subcommands; several may be given; `all` chains stats..export):
  dump    read-only DB dump of the corpus to corpus.pkl.gz, for runs on
          machines without DB access (set HOTS_CORPUS_PATH to it there)
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
  kparity v2 kernel vs Python WP on finished self-play drafts (all 15 maps);
          gated (gate kernel_parity)
  select  pick the deployed seed with an INDEPENDENT judge, not the worker's
          own eval WP (a proxy: the policy's score under the value function
          it searched against). Each seed's policy argmax drafts a fixed
          benchmark against the fresh GD model; drafts are scored by a
          structure-aware realized-outcome index
          (overfit2026/gold.StructRealizedIndex: the realized index plus
          no-healer / no-frontline / stacked-role terms, all cross-fitted on
          real outcomes) fitted on the last JUDGE_DAYS of real games. The
          plain index under-penalized degenerate teams by ~5pp and ranked a
          55%-degenerate hero-win-rate drafter above every policy. Rule: highest judge score among
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

Remote (pause-robust) runs: HOTS_CORPUS_PATH=<dump> refresh.py --resume stats
data wp partial gd; each MCTS seed as its own job (`refresh.py mcts_cmd
--seed N` prints the command); then REFRESH_MCTS_EXTERNAL=1
REFRESH_EXPORT_DIR=<dir> refresh.py mcts select export.

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
# compositions.json snapshot: sync rewrites src/lib/data/compositions.json
# daily, so every phase (and the export) reads this copy instead.
COMPOSITIONS_JSON = os.path.join(RUN_DIR, "compositions.json")
SRC_COMPOSITIONS = os.path.join(REPO_DIR, "src", "lib", "data", "compositions.json")
# Corpus dump for machines without DB access (phase dump; HOTS_CORPUS_PATH).
CORPUS_DUMP = os.path.join(RUN_DIR, "corpus.pkl.gz")
LOW_DATA_JSON = os.path.join(RUN_DIR, "low_data_test_drafts.json")
# test drafts of each low-data hero's most-played role peers (swing reference)
PEER_DRAFTS_JSON = os.path.join(RUN_DIR, "peer_test_drafts.json")

# Games from this major version on (2.57 shipped Xal'atath on 2026-09-28).
MIN_VERSION = (2, 55)
# Shrinkage of decayed win rates, in pseudo-games (decayed effective games):
# hero -> games-weighted mean of its fine role in the tier; hero-map -> the
# hero's (shrunk) tier rate; pair -> the additive expectation from the two
# (shrunk) hero rates. Matters for new or rarely played heroes; for heroes
# with thousands of decayed games per tier it moves rates by < 0.1pp.
HERO_PRIOR_GAMES = 200.0
MAP_PRIOR_GAMES = 50.0
PAIR_PRIOR_GAMES = 30.0
# Low-data heroes (fewer than this many picks in the corpus) get a fixed hash
# share of their games in the WP test split, so their calibration can be
# measured, and a swap check against same-role heroes.
LOW_DATA_PICKS = 5000
LOW_DATA_TEST_SHARE = 0.2
# Gates for low-data heroes. Level: on held-out games containing the hero,
# the WP's mean P(hero's team wins) must match the realized rate within
# max(LEVEL_TOL, 2 SE); a genuinely strong new hero is then allowed a large
# edge over its role peers, but not one the outcomes do not support. Swing:
# replacing the hero by any same-role peer in a real draft must move the WP
# by at most SWING_REF_FACTOR times the worst swap measured the same way for
# its most-played role peers (catches broken or exploding inputs; on
# 2026-10-02 established ranged mages reached 0.17-0.26).
LEVEL_TOL = 0.03
SWING_REF_PEERS = 4          # most-played same-role heroes measured as reference
SWING_REF_FACTOR = 1.5       # hero's worst swap <= 1.5 x the peers' worst

HALF_LIFE_DAYS = 90.0
WP_SEEDS = [42, 123, 777]
MCTS_SEEDS = [0, 1]   # 2 seeds: HotS work is capped to ~25% of this shared box
MCTS_SIMS = 400       # operating point; 800 buys proxy WP, not judged quality
# Kernel search (X2 fix, 2026-10-01): opponent chance nodes, so the tree plans
# through the opponent's replies to our later picks and bans. Runs before
# 2026-11-01 used the pre-fix search ("legacy"), whose tree stopped at the
# current own-pick block.
MCTS_SEARCH_MODE = os.environ.get("REFRESH_MCTS_SEARCH", "chance")
MCTS_EPISODES = int(os.environ.get("REFRESH_MCTS_EPISODES", "300000"))
GATE_WP_MIN = 56.0    # % test accuracy floor
GATE_PARTIAL_MIN = 0.525  # partial-WP overall test acc floor (all-step mix)
# Proxy sanity floor only (catches broken training). Leak-free 400-sim runs
# land ~0.73-0.74 on the paper stats; decayed-stats 800-sim runs 0.75-0.80.
GATE_MCTS_PROXY_FLOOR = 0.70
# Seed selection (phase select)
JUDGE_DAYS = 90           # realized-index window, ending at the stats ref date
JUDGE_SALT = 99           # gold.RealizedIndex fold-hash salt
JUDGE_CLASS = "StructRealizedIndex"  # gold.<class>; v1 was RealizedIndex
BENCH_DRAFTS = 1260       # v2: 15 maps x 3 tiers x 2 sides x 14
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
# Production encoding: 91 heroes (Xal'atath appended) and 15 maps (Haunted
# Mines appended); research stays on shared.py's v1 default. Subprocesses
# (partial-WP trainer, MCTS workers, exports) inherit all of these.
os.environ["HOTS_HERO_SET"] = "v2"
os.environ["WP_COMPOSITIONS_PATH"] = COMPOSITIONS_JSON


def version_ok(gver):
    try:
        major = tuple(int(x) for x in (gver or "").split(".")[:2])
    except ValueError:
        return False
    return len(major) == 2 and major >= MIN_VERSION


def require_compositions():
    if not os.path.exists(COMPOSITIONS_JSON):
        sys.exit(f"{COMPOSITIONS_JSON} missing: run the stats phase first "
                 "(it snapshots src/lib/data/compositions.json)")


def wp_dim():
    from sweep_enriched_wp import INPUT_DIM_BASE, FEATURE_GROUP_DIMS
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    return INPUT_DIM_BASE + sum(FEATURE_GROUP_DIMS[g] for g in ENRICHED_GROUPS)


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


def shrink(wins, games, prior, k):
    """Posterior mean of a win rate (fraction) under k pseudo-games at prior."""
    return (wins + k * prior) / (games + k)


def shrunk_hero_rates(cell):
    """hero -> win rate (fraction) shrunk toward its fine role's games-weighted
    mean in this tier cell. A hero with no fine role shrinks toward 0.5."""
    from shared import HERO_ROLE_FINE
    role_g, role_w = {}, {}
    for h, (g, wn) in cell["hero"].items():
        r = HERO_ROLE_FINE.get(h)
        role_g[r] = role_g.get(r, 0.0) + g
        role_w[r] = role_w.get(r, 0.0) + wn
    out = {}
    for h, (g, wn) in cell["hero"].items():
        r = HERO_ROLE_FINE.get(h)
        prior = role_w[r] / role_g[r] if r is not None and role_g[r] > 0 else 0.5
        out[h] = shrink(wn, g, prior, HERO_PRIOR_GAMES)
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
        hwr = shrunk_hero_rates(cell)
        for h, (g, wn) in cell["hero"].items():
            if g < 20:
                continue
            hero_stats.append({
                "hero": h, "tier": tier, "games": round(g, 1),
                "win_rate": round(100.0 * hwr[h], 3),
                "pick_rate": round(100.0 * g / total, 3),
                "ban_rate": round(100.0 * cell["bans"].get(h, 0.0) / total, 3)})
        for (m, h), (g, wn) in cell["hmap"].items():
            if g < 5:
                continue
            hero_map_stats.append({
                "hero": h, "map": m, "tier": tier, "games": round(g, 1),
                "win_rate": round(100.0 * shrink(wn, g, hwr[h], MAP_PRIOR_GAMES), 3)})
        for (a, b), (g, wn) in cell["with"].items():
            if g < 10:
                continue
            prior = min(max(hwr[a] + hwr[b] - 0.5, 0.0), 1.0)
            wr = round(100.0 * shrink(wn, g, prior, PAIR_PRIOR_GAMES), 3)
            for x, y in ((a, b), (b, a)):
                pairwise_stats.append({
                    "hero_a": x, "hero_b": y, "tier": tier,
                    "relationship": "with", "win_rate": wr,
                    "games": round(g, 1)})
        for (a, b), (g, wa) in cell["against"].items():
            if g < 10:
                continue
            prior = min(max(0.5 + hwr[a] - hwr[b], 0.0), 1.0)
            wr_a = round(100.0 * shrink(wa, g, prior, PAIR_PRIOR_GAMES), 3)
            pairwise_stats.append({
                "hero_a": a, "hero_b": b, "tier": tier,
                "relationship": "against", "win_rate": wr_a,
                "games": round(g, 1)})
            pairwise_stats.append({
                "hero_a": b, "hero_b": a, "tier": tier,
                "relationship": "against", "win_rate": round(100.0 - wr_a, 3),
                "games": round(g, 1)})

    json.dump({
        "_meta": {"snapshot_date": RUN_DATE,
                  "patch": ">=" + ".".join(map(str, MIN_VERSION)),
                  "shrinkage": {"hero_to_role": HERO_PRIOR_GAMES,
                                "map_to_hero": MAP_PRIOR_GAMES,
                                "pair_to_additive": PAIR_PRIOR_GAMES},
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


DUMP_COLUMNS = ("replay_id", "game_map", "skill_tier", "draft_order",
                "team0_heroes", "team1_heroes", "team0_bans", "team1_bans",
                "winner", "avg_mmr", "league_tier", "game_date", "game_version")


def phase_dump():
    """Read-only DB dump of the whole corpus (the columns shared.load_replay_data
    returns, plus game_date/game_version) for runs on machines without DB
    access. Point HOTS_CORPUS_PATH at the file there."""
    import pickle
    os.makedirs(RUN_DIR, exist_ok=True)
    conn = _db_conn()
    conn.set_session(readonly=True)
    cur = conn.cursor(name="refresh_dump")
    cur.itersize = 50_000
    cur.execute(f"SELECT {', '.join(DUMP_COLUMNS)} FROM replay_draft_data ORDER BY replay_id")
    rows = []
    for rec in cur:
        d = dict(zip(DUMP_COLUMNS, rec))
        for f in ("draft_order", "team0_heroes", "team1_heroes", "team0_bans", "team1_bans"):
            if isinstance(d[f], str):
                d[f] = json.loads(d[f])
        rows.append(d)
    cur.close()
    conn.close()
    tmp = CORPUS_DUMP + ".tmp"
    with gzip.open(tmp, "wb", compresslevel=3) as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CORPUS_DUMP)
    ref = max(r["game_date"] for r in rows if r["game_date"] is not None)
    log(f"dumped {len(rows):,} replays (latest game {ref}) -> {CORPUS_DUMP} "
        f"({os.path.getsize(CORPUS_DUMP) >> 20} MB)")
    meta_update(dump={"rows": len(rows), "latest_game": str(ref),
                      "dumped_at": datetime.datetime.now().isoformat(timespec="seconds")})


def _corpus_rows():
    """The dump's rows when HOTS_CORPUS_PATH is set, else None (read the DB)."""
    if not os.environ.get("HOTS_CORPUS_PATH"):
        return None
    import shared
    return shared.load_replay_data()


def _stats_records():
    """(ref_date, iterator of (rid, map, tier, t0, t1, b0, b1, winner, date, version))."""
    rows = _corpus_rows()
    if rows is not None:
        ref = max(r["game_date"] for r in rows if r.get("game_date") is not None)
        return ref, ((r["replay_id"], r["game_map"], r["skill_tier"], r["team0_heroes"],
                      r["team1_heroes"], r["team0_bans"], r["team1_bans"], r["winner"],
                      r["game_date"], r["game_version"]) for r in rows)
    conn = _db_conn()
    cur = conn.cursor()
    cur.execute("SELECT COALESCE(max(game_date), now()) FROM replay_draft_data")
    ref = cur.fetchone()[0]
    cur.execute("""
        SELECT replay_id, game_map, skill_tier, team0_heroes, team1_heroes,
               team0_bans, team1_bans, winner, game_date, game_version
        FROM replay_draft_data ORDER BY replay_id""")

    def it():
        while True:
            batch = cur.fetchmany(50_000)
            if not batch:
                break
            yield from batch
        cur.close()
        conn.close()
    return ref, it()


def phase_stats():
    """Per-game exponentially decayed counts over the live corpus (patch 2.55
    on). Equivalent to the paper's per-build incremental decay up to
    within-build granularity (weights compose multiplicatively either way)."""
    import shutil

    os.makedirs(RUN_DIR, exist_ok=True)
    shutil.copy(SRC_COMPOSITIONS, COMPOSITIONS_JSON)
    ref_date, records = _stats_records()
    ref_ord = ref_date.toordinal() + (ref_date.hour / 24.0)
    log(f"decay reference date: {ref_date}")

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
    for (rid, gmap, tier, t0h, t1h, t0b, t1b, winner, gdate, gver) in records:
        if not version_ok(gver):
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
    log(f"corpus: {len(rows):,} replays (patch >= 2.55, known tiers)")
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


def low_data_heroes(rows):
    """{hero: picks} for encoded heroes with fewer than LOW_DATA_PICKS picks."""
    from collections import Counter
    from shared import HEROES
    c = Counter(h for r in rows for h in (r["team0_heroes"] or []) + (r["team1_heroes"] or []))
    return {h: c.get(h, 0) for h in HEROES if c.get(h, 0) < LOW_DATA_PICKS}


def wp_split(rows):
    """98/2 random split (seed 42, as before), except that games with a
    low-data hero go to test by a fixed replay-id hash share
    (LOW_DATA_TEST_SHARE), so the hero's calibration is measurable."""
    from shared import split_data
    from overfit2026.data import splitmix64
    low = low_data_heroes(rows)
    has_low = [any(h in low for h in (r["team0_heroes"] or []) + (r["team1_heroes"] or []))
               for r in rows]
    rest = [r for r, f in zip(rows, has_low) if not f]
    train_rows, test_rows = split_data(rest, test_frac=0.02, seed=42)
    cut = int(LOW_DATA_TEST_SHARE * 1000)
    for r, f in zip(rows, has_low):
        if f:
            (test_rows if splitmix64(int(r["replay_id"]) * 7907 + 11) % 1000 < cut
             else train_rows).append(r)
    return train_rows, test_rows, low


def peer_test_drafts(rows, test_rows, low):
    """{low-data hero: {peer: [<=200 test drafts containing that peer]}} for its
    SWING_REF_PEERS most-played fine-role peers."""
    import random
    from collections import Counter
    from shared import HERO_ROLE_FINE
    picks = Counter(h for r in rows for h in (r["team0_heroes"] or []) + (r["team1_heroes"] or []))
    keep = ("replay_id", "game_map", "skill_tier", "team0_heroes", "team1_heroes", "winner")
    rng = random.Random(0)
    out = {}
    for h in low:
        role = HERO_ROLE_FINE.get(h)
        peers = sorted((p for p, r in HERO_ROLE_FINE.items() if r == role and p not in low),
                       key=lambda p: -picks.get(p, 0))[:SWING_REF_PEERS]
        out[h] = {}
        for p in peers:
            ds = [{k: r[k] for k in keep} for r in test_rows
                  if p in (r["team0_heroes"] or []) + (r["team1_heroes"] or [])
                  and len(r["team0_heroes"] or []) == 5 and len(r["team1_heroes"] or []) == 5]
            out[h][p] = rng.sample(ds, min(200, len(ds)))
    return out


def phase_data():
    rows = _load_fresh_corpus()
    train_rows, test_rows, low = wp_split(rows)
    log(f"low-data heroes (< {LOW_DATA_PICKS} picks): {low or 'none'}")
    # raw drafts of the low-data test games, for the WP swap check
    json.dump([{k: r[k] for k in ("replay_id", "game_map", "skill_tier", "team0_heroes",
                                  "team1_heroes", "winner")}
               for r in test_rows
               if any(h in low for h in (r["team0_heroes"] or []) + (r["team1_heroes"] or []))],
              open(LOW_DATA_JSON, "w"))
    json.dump(peer_test_drafts(rows, test_rows, low), open(PEER_DRAFTS_JSON, "w"))
    log(f"building out-of-fold WP feature caches ({len(train_rows):,} train / "
        f"{len(test_rows):,} test, {OOF_FOLDS} folds)")
    oof_features(train_rows, FEATURE_CACHE_TRAIN)
    oof_features(test_rows, FEATURE_CACHE_TEST)
    meta_update(data={"train": len(train_rows), "test": len(test_rows),
                      "oof_folds": OOF_FOLDS, "low_data_heroes": low})


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
    assert dim == wp_dim(), f"expected {wp_dim()}-d features, got {dim}"

    from shared import tie_hero_columns
    tied = tied_heroes()
    log(f"WP identity columns tied to role mean: {tied or 'none'}")
    best = {"acc": -1.0}
    accs = []
    for seed in WP_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = WinProbEnrichedModel(dim, [256, 128], dropout=0.3)
        model, acc = train_wp_model(model, train_X, test_X, train_y, test_y,
                                    f"prod_wp-s{seed}", device,
                                    after_step=(lambda m: tie_hero_columns(m.net[0].weight, tied))
                                    if tied else None)
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
    y = test_y.cpu().numpy()
    slope = calibration_slope(p, y)
    log(f"WP best acc {best['acc']:.2f}% (seed {best['seed']}; all {accs}), "
        f"calibration slope {slope:.3f} on out-of-fold test rows -> {WP_PT}")
    meta_update(wp={"best_acc": best["acc"], "best_seed": best["seed"],
                    "all_accs": accs, "cal_slope": slope, "identity_tied": tied,
                    "finite": bool(np.isfinite(p).all())})


def phase_lowdata():
    """Calibration and swap checks for low-data heroes on the trained WP
    (held-out games; feature cache + raw drafts from the data phase)."""
    import numpy as np
    import torch
    from sweep_enriched_wp import WinProbEnrichedModel, compute_group_indices
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    require_compositions()
    gi = compute_group_indices()
    cols = [c for g in ENRICHED_GROUPS for c in range(*gi[g])]
    z = np.load(FEATURE_CACHE_TEST)
    X = np.concatenate([z["bases"], z["enricheds"][:, cols]], axis=1).astype(np.float32)
    y = z["labels"].astype(np.float32)
    model = WinProbEnrichedModel(wp_dim(), [256, 128], dropout=0.3)
    model.load_state_dict(torch.load(WP_PT, weights_only=True, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        p = torch.cat([model(torch.from_numpy(X[i:i + 65536]))
                       for i in range(0, len(X), 65536)]).clamp(1e-6, 1 - 1e-6).numpy()
    low = json.load(open(META_JSON)).get("data", {}).get("low_data_heroes", {})
    peers = json.load(open(PEER_DRAFTS_JSON)) if os.path.exists(PEER_DRAFTS_JSON) else {}
    res = {}
    for h in low:
        hero_drafts = [d for d in json.load(open(LOW_DATA_JSON))
                       if h in d["team0_heroes"] + d["team1_heroes"]][:200]
        r = {**hero_calibration(h, X, p, y), **swap_check(h, hero_drafts, model)}
        ref = {pr: swap_check(pr, ds, model, quiet=True) for pr, ds in peers.get(h, {}).items()}
        r["swap_ref"] = {pr: {k: v.get(k) for k in ("swap_mean_delta", "swap_worst_abs_delta")}
                         for pr, v in ref.items()}
        worst = [v["swap_worst_abs_delta"] for v in ref.values() if "swap_worst_abs_delta" in v]
        r["swap_ref_worst"] = max(worst) if worst else None
        log(f"low-data {h}: swap worst {r.get('swap_worst_abs_delta')} vs role peers' "
            f"worst {r['swap_ref_worst']} ({r['swap_ref']})")
        res[h] = r
    meta_update(low_data=res)


def tied_heroes():
    """Low-data heroes (data phase): their one-hot identity columns in the WP
    and partial-WP first layers are tied to their role peers' mean during
    training (shared.tie_hero_columns). With ~1K games a free identity column
    overfits (2026-10-02, Xal'atath: +0.12 of a +0.18 swap edge came from it,
    and the WP predicted .66 for her team vs .61 realized); tied, her value
    comes from her shrunk statistics. She gets her own column once she has
    LOW_DATA_PICKS picks."""
    return sorted(json.load(open(META_JSON)).get("data", {}).get("low_data_heroes", {}))


def hero_calibration(hero, X, p, y):
    """Calibration of P(hero's team wins) on the test games that contain the
    hero (oriented to the hero's team)."""
    import numpy as np
    from shared import HERO_TO_IDX, NUM_HEROES
    i = HERO_TO_IDX[hero]
    on0, on1 = X[:, i] > 0.5, X[:, NUM_HEROES + i] > 0.5
    ph = np.concatenate([p[on0], 1 - p[on1]])
    yh = np.concatenate([y[on0], 1 - y[on1]])
    # the feature cache holds every game in both team orders: 2 rows per game
    n_games = len(ph) // 2
    out = {"test_rows": int(len(ph)), "test_games": int(n_games)}
    if n_games < 30:
        return out
    pc = np.clip(ph, 1e-6, 1 - 1e-6)
    out.update({"mean_pred": float(ph.mean()), "realized": float(yh.mean()),
                "realized_se": float(np.sqrt(yh.mean() * (1 - yh.mean()) / n_games)),
                "brier": float(np.mean((ph - yh) ** 2)),
                "brier_const": float(np.mean((yh.mean() - yh) ** 2)),
                "logloss": float(-np.mean(yh * np.log(pc) + (1 - yh) * np.log(1 - pc))),
                "acc": float(np.mean((ph > 0.5) == (yh > 0.5))),
                "pred_sd": float(ph.std())})
    try:
        out["cal_slope"] = calibration_slope(pc, yh)
    except np.linalg.LinAlgError:
        out["cal_slope"] = None
    log(f"WP calibration on {hero} test games: {out}")
    return out


def swap_check(hero, drafts, model, quiet=False):
    """Mean and worst WP change when `hero` is replaced, in real test drafts
    containing it, by each available hero of the same fine role (features from
    the serving stats, team-order symmetrized)."""
    import numpy as np
    import torch
    from shared import HERO_ROLE_FINE
    from sweep_enriched_wp import StatsCache, compute_group_indices
    from experiment_synthetic_augmentation import ENRICHED_GROUPS, make_eval_fn
    st = StatsCache.__new__(StatsCache)
    st._load_frozen(STATS_JSON)
    st._load_compositions()
    gi = compute_group_indices()
    cols = [c for g in ENRICHED_GROUPS for c in range(*gi[g])]
    model.eval()
    f = make_eval_fn(model, cols, st, torch.device("cpu"))

    def sym(a, b, m, t):
        return 0.5 * (f(a, b, m, t) + 1 - f(b, a, m, t))
    role = HERO_ROLE_FINE[hero]
    peers = [h for h, r in HERO_ROLE_FINE.items() if r == role and h != hero]
    deltas, worst = [], 0.0
    for d in drafts:
        own, opp = ((d["team0_heroes"], d["team1_heroes"]) if hero in d["team0_heroes"]
                    else (d["team1_heroes"], d["team0_heroes"]))
        base = sym(own, opp, d["game_map"], d["skill_tier"])
        alts = [sym([x if x != hero else h for x in own], opp, d["game_map"], d["skill_tier"])
                for h in peers if h not in own and h not in opp]
        dl = [base - a for a in alts]
        deltas.append(float(np.mean(dl)))
        worst = max(worst, float(np.max(np.abs(dl))))
    out = {"swap_drafts": len(drafts), "swap_peers": len(peers)}
    if deltas:
        out.update({"swap_mean_delta": float(np.mean(deltas)),
                    "swap_worst_abs_delta": worst,
                    "swap_finite": bool(np.isfinite(deltas).all())})
    if not quiet:
        log(f"WP swap check for {hero} vs {len(peers)} {role} peers: {out}")
    return out


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
                "PARTIAL_WP_OOF_FOLDS": str(OOF_FOLDS),
                "PARTIAL_WP_TIE_HEROES": json.dumps(tied_heroes())})
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
    # epoch-level resume: a paused or killed run continues at its last epoch
    os.environ.setdefault("GD_RESUME_PATH", os.path.join(RUN_DIR, "gd_resume.pt"))
    # Samples are streamed to a compact on-disk cache and memory-mapped
    # (tgd.CompactDraftDataset): as float32 arrays in RAM the full corpus's
    # ~30M samples need ~46 GB (OOM-killed on a 47 GB worker, 2026-10-02).
    # The corpus is freed before training.
    import gc
    cache = os.path.join(RUN_DIR, "gd_cache")
    if not all(os.path.exists(os.path.join(cache, s, "meta.json")) for s in ("train", "test")):
        rows = _load_fresh_corpus()
        train_rows, test_rows = split_data(rows, test_frac=0.02, seed=42)
        del rows
        tgd.CompactDraftDataset(train_rows, os.path.join(cache, "train"))
        tgd.CompactDraftDataset(test_rows, os.path.join(cache, "test"))
        del train_rows, test_rows
        gc.collect()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_ds = tgd.CompactDraftDataset(None, os.path.join(cache, "train"))
    test_ds = tgd.CompactDraftDataset(None, os.path.join(cache, "test"))
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


def ensure_kernel():
    """Build (or confirm) the v2 kernel in cuda_mcts/h91 for these sources."""
    subprocess.run([sys.executable, os.path.join(TRAINING_DIR, "cuda_mcts", "build_h91.py"),
                    "--if-stale"], check=True, cwd=TRAINING_DIR)
    info = json.load(open(os.path.join(TRAINING_DIR, "cuda_mcts", "h91", "BUILD_INFO.json")))
    meta_update(kernel=info)
    return info


def mcts_env(seed):
    run_dir = os.path.join(RUN_DIR, f"mcts_s{seed}")
    env = {
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
        "MCTS_SEARCH_MODE": MCTS_SEARCH_MODE,
        "WANDB_RUN_NAME": f"prod_refresh_{RUN_DATE}_s{seed}",
        "WP_STATS_PATH": STATS_JSON,
        "REPLAY_SNAPSHOT": "0",
        "HOTS_HERO_SET": os.environ["HOTS_HERO_SET"],
        "WP_COMPOSITIONS_PATH": COMPOSITIONS_JSON,
    }
    if os.environ.get("HOTS_CORPUS_PATH"):
        env["HOTS_CORPUS_PATH"] = os.environ["HOTS_CORPUS_PATH"]
    return run_dir, env


def mcts_cmd(seed):
    """Shell command for one seed as a separate, pause-robust job (remote
    workers: training/remote_workers/run_remote.sh recognizes the bare worker
    and resumes it with MCTS_FRESH=0). Completion marker: DONE in its dir."""
    import shlex
    run_dir, env = mcts_env(seed)
    os.makedirs(run_dir, exist_ok=True)
    assigns = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    log_path = shlex.quote(os.path.join(run_dir, "train.log"))
    return (f"{assigns} python -u train_mcts_worker.py >> {log_path} 2>&1 "
            f"&& touch {shlex.quote(os.path.join(run_dir, 'DONE'))}")


def phase_kparity():
    """The v2 kernel's in-kernel WP must match the Python WP on finished
    self-play drafts across all 15 maps (gated)."""
    require_compositions()
    if os.environ.get("REFRESH_MCTS_EXTERNAL") != "1":
        ensure_kernel()
    sys.path.insert(0, BASE)
    import kernel_parity
    res = kernel_parity.run(WP_PT, GD_PT, wp_dim())
    log(f"kernel parity: {res}")
    meta_update(kernel_parity=res)


def phase_mcts():
    require_compositions()
    procs = []
    # REFRESH_MCTS_EXTERNAL=1: the seeds ran as separate jobs (mcts_cmd); only
    # collect them here. Completion = DONE marker + draft_policy.pt.
    external = os.environ.get("REFRESH_MCTS_EXTERNAL") == "1"
    if not external:
        ensure_kernel()
        # cadence.sh pins the whole refresh to one GPU (REFRESH_GPU); the box
        # is shared and HotS work is capped at ~25% of it.
        gpus = ([int(os.environ["REFRESH_GPU"])] if os.environ.get("REFRESH_GPU")
                else free_gpus())
        log(f"MCTS on GPUs {gpus} (>= {MCTS_MIN_FREE_MIB} MiB free each)")
    for seed in MCTS_SEEDS:
        run_dir, extra = mcts_env(seed)
        os.makedirs(run_dir, exist_ok=True)
        if external:
            procs.append((seed, None, run_dir))
            continue
        env = dict(os.environ)
        env.update(extra)
        env["CUDA_VISIBLE_DEVICES"] = str(gpus[seed % len(gpus)])
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
        if p is None:
            rc = 0 if os.path.exists(os.path.join(run_dir, "DONE")) else None
        else:
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
        ki = os.path.join(run_dir, "kernel_info.json")
        results[seed] = {"rc": rc, "ok": ok, "best_wp": best_wp,
                         "kernel_info": json.load(open(ki)) if os.path.exists(ki) else None}
        log(f"MCTS seed {seed}: rc={rc} ok={ok} best_wp={best_wp}")

    ok_seeds = {s: r for s, r in results.items() if r["ok"] and r["best_wp"]}
    # best_wp here is the PROXY (worker's own eval under the value function it
    # searched against). It is logged, not used to choose: see phase select.
    meta_update(mcts={"results": results, "sims": MCTS_SIMS,
                      "episodes": MCTS_EPISODES, "search_mode": MCTS_SEARCH_MODE,
                      "ok_seeds": sorted(ok_seeds)})
    if not ok_seeds:
        sys.exit("no MCTS seed completed successfully")
    log(f"MCTS done: ok seeds {sorted(ok_seeds)}; proxy best_wp "
        + ", ".join(f"s{s}={r['best_wp']:.4f}" for s, r in sorted(ok_seeds.items())))


# ── Phase: select ────────────────────────────────────────────────────

def _judge_games():
    """Slim games (gold.py format) from the last JUDGE_DAYS of the corpus the
    other phases train on: replay_draft_data, patch 2.55 on (pre-2.55 exclude
    set), known tiers only, 5v5 with a recorded winner. The window ends at the
    stats phase's decay reference date so a rerun sees the same games."""
    exclude = set(json.load(open(EXCLUDE_IDS_JSON)))
    ref = json.load(open(META_JSON)).get("stats", {}).get("ref_date")
    rows = _corpus_rows()
    if rows is not None:
        hi = max(r["game_date"] for r in rows if r.get("game_date") is not None)
        if ref is not None:
            hi = min(hi, datetime.datetime.fromisoformat(ref).replace(tzinfo=hi.tzinfo))
        lo = hi - datetime.timedelta(days=JUDGE_DAYS)
        games = []
        for r in rows:
            gd, t0, t1 = r.get("game_date"), tuple(r["team0_heroes"] or []), tuple(r["team1_heroes"] or [])
            if (gd is None or not (lo < gd <= hi) or not version_ok(r.get("game_version"))
                    or r["replay_id"] in exclude or r["skill_tier"] not in TRAIN_TIERS
                    or r["winner"] not in (0, 1) or len(t0) != 5 or len(t1) != 5):
                continue
            games.append((int(r["replay_id"]), r["skill_tier"], r["game_map"], t0, t1,
                          tuple(r["team0_bans"] or []) + tuple(r["team1_bans"] or []),
                          int(r["winner"])))
        return games, str(hi)
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
          AND CASE WHEN game_version ~ '^[0-9]+[.][0-9]+'
                   THEN split_part(game_version, '.', 1)::int * 1000
                        + split_part(game_version, '.', 2)::int
                   ELSE 0 END >= %s""",
                (ref, JUDGE_DAYS, ref, MIN_VERSION[0] * 1000 + MIN_VERSION[1]))

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
    # v2 (2026-10-01): the index carries explicit structure terms. The plain
    # v1 index is still built from the same games and reported per bench row
    # ("judge_v1") for continuity; it does not enter the rule.
    games = None
    judges = {}
    for cls, tag in ((JUDGE_CLASS, "struct_"), ("RealizedIndex", "")):
        judge_pkl = os.path.join(RUN_DIR, f"judge_realized_{tag}{JUDGE_DAYS}d_s{JUDGE_SALT}.pkl")
        if os.path.exists(judge_pkl):
            with open(judge_pkl, "rb") as f:
                judges[cls], ref = pickle.load(f)
            continue
        if games is None:
            games, ref = _judge_games()
            log(f"judge: {len(games):,} games in the {JUDGE_DAYS} days to {ref}")
        t0 = time.time()
        j = getattr(gold, cls)(games, salt=JUDGE_SALT, name=f"recent{JUDGE_DAYS}d_{cls}")
        with open(judge_pkl, "wb") as f:
            pickle.dump((j, ref), f, protocol=pickle.HIGHEST_PROTOCOL)
        judges[cls] = j
        log(f"judge {cls} built in {time.time() - t0:.0f}s; held-out fold acc "
            + ", ".join(f"{x['acc']:.4f}" for x in j.fit))
    judge, judge_v1 = judges[JUDGE_CLASS], judges["RealizedIndex"]

    # Proxy on the same drafts: the fresh WP with the serving stats (what the
    # MCTS searched against), team-order symmetrized.
    st = StatsCache.__new__(StatsCache)
    st._load_frozen(STATS_JSON)
    st._load_compositions()
    gi = compute_group_indices()
    cols = [c for g in ENRICHED_GROUPS for c in range(*gi[g])]
    wp = WinProbEnrichedModel(wp_dim(), [256, 128], dropout=0.3)
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
        r["judge_v1"] = float(np.mean([judge_v1.score(o, p, t) for o, p, m, t in drafts]))
        log(f"bench {name:15s} judge {r['judge']:.4f}±{r['judge_se']:.4f} (v1 {r['judge_v1']:.4f}) "
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
        "judge": {"kind": f"overfit2026.gold.{JUDGE_CLASS}", "days": JUDGE_DAYS,
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
        "wp_finite": {"value": wp.get("finite")},
        "kernel_parity": {"value": m.get("kernel_parity", {}).get("max_abs_diff"),
                          "max": m.get("kernel_parity", {}).get("tol"),
                          "pass": m.get("kernel_parity", {}).get("pass") is True},
    }
    # low-data heroes (e.g. a hero added by a new patch): finite WP and no
    # large swing against the same-role heroes it replaces
    missing = sorted(set(m.get("data", {}).get("low_data_heroes", {})) - set(m.get("low_data", {})))
    if missing:
        g["low_data_checked"] = {"value": missing, "pass": False}   # lowdata phase not run
    for h, r in m.get("low_data", {}).items():
        if "mean_pred" in r:
            tol = max(LEVEL_TOL, 2 * r["realized_se"])
            gap = r["mean_pred"] - r["realized"]
            g[f"low_data_level:{h}"] = {"value": gap, "max_abs": tol,
                                        "test_games": r["test_games"],
                                        "pass": abs(gap) <= tol}
        else:   # too few held-out games to measure: report, do not block
            g[f"low_data_level:{h}"] = {"value": None, "test_games": r.get("test_games"),
                                        "pass": True}
        v, ref = r.get("swap_worst_abs_delta"), r.get("swap_ref_worst")
        lim = SWING_REF_FACTOR * ref if ref is not None else None
        g[f"low_data_swing:{h}"] = {"value": v, "max": lim, "peers_worst": ref,
                                    "mean_delta": r.get("swap_mean_delta"),
                                    "pass": (v is not None and lim is not None
                                             and r.get("swap_finite", False) and v <= lim)}
    g["wp_finite"]["pass"] = g["wp_finite"]["value"] is True
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

def site_encoding_matches():
    """The deployed site must encode heroes/maps exactly as these models do:
    compare training/shared.py with src/lib/draft/encoding.ts on origin/main
    (what cadence.sh deploys onto), else the working tree's copy."""
    import re
    from shared import HEROES, MAPS
    src = None
    try:
        subprocess.run(["git", "-C", REPO_DIR, "fetch", "-q", "origin", "main"],
                       check=True, timeout=120)
        src = subprocess.run(["git", "-C", REPO_DIR, "show",
                              "origin/main:src/lib/draft/encoding.ts"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        p = os.path.join(REPO_DIR, "src", "lib", "draft", "encoding.ts")
        src = open(p).read() if os.path.exists(p) else ""

    def lst(name):
        m = re.search(name + r"[^=]*=\s*\[(.*?)\]", src, re.S)
        if not m:
            return None
        body = re.sub(r"//[^\n]*", "", m.group(1))
        return re.findall(r'"((?:[^"\\]|\\.)*)"', body)
    ok = lst("export const HEROES") == list(HEROES) and lst("export const MAPS") == list(MAPS)
    return ok


def phase_export():
    phase_gates()
    out_root = os.environ.get("REFRESH_EXPORT_DIR") or REPO_DIR
    if out_root == REPO_DIR and not site_encoding_matches():
        sys.exit("EXPORT REFUSED: the site's src/lib/draft/encoding.ts (origin/main) does "
                 "not encode the same heroes/maps as training/shared.py "
                 f"(HOTS_HERO_SET={os.environ['HOTS_HERO_SET']}); deploy the site code first")
    best_seed = json.load(open(META_JSON))["select"]["seed"]

    # REFRESH_EXPORT_DIR: export into <dir>/public/models and <dir>/src/lib/data
    # (remote runs; the files are copied back for the deploy). Default: this
    # repo, as cadence.sh expects.
    models_dir = os.path.join(out_root, "public", "models")
    data_dir = os.path.join(out_root, "src", "lib", "data")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "SITE_POLICY_PT": os.path.join(RUN_DIR, f"mcts_s{best_seed}", "draft_policy.pt"),
        "SITE_GD_PT": GD_PT,
        "SITE_WP_PT": WP_PT,
        "SITE_MODELS_DIR": models_dir,
    })
    subprocess.run([sys.executable,
                    os.path.join(TRAINING_DIR, "export_site_models.py")],
                   env=env, check=True, cwd=TRAINING_DIR)
    penv = dict(os.environ)
    penv["PARTIAL_WP_CKPT"] = PARTIAL_PT
    penv["SITE_MODELS_DIR"] = models_dir
    subprocess.run([sys.executable,
                    os.path.join(TRAINING_DIR, "production_refresh", "export_partial_wp.py")],
                   env=penv, check=True, cwd=TRAINING_DIR)

    import shutil
    dst = os.path.join(data_dir, "draft-stats-decayed.json")
    shutil.copy(SITE_STATS_JSON, dst)
    log(f"site stats artifact -> {dst}")
    # the comp_wr features were computed from this snapshot; cadence.sh deploys
    # it with the models so the site serves the same table
    shutil.copy(COMPOSITIONS_JSON, os.path.join(data_dir, "compositions.json"))
    meta_update(exported=True, export_date=datetime.datetime.now().isoformat())
    log("export complete — commit public/models/ + src/lib/data/draft-stats-decayed.json "
        "+ src/lib/data/compositions.json to deploy")


PHASES = {"dump": phase_dump, "stats": phase_stats, "data": phase_data, "wp": phase_wp, "lowdata": phase_lowdata,
          "partial": phase_partial, "gd": phase_gd,
          "kparity": phase_kparity, "mcts": phase_mcts,
          "select": phase_select, "gates": phase_gates,
          "export": phase_export}


ALL = ("stats", "data", "wp", "lowdata", "partial", "gd", "kparity", "mcts", "select",
       "export")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phases", nargs="+", choices=list(PHASES) + ["all", "mcts_cmd"])
    ap.add_argument("--resume", action="store_true",
                    help="skip phases refresh_meta.json records as done (pause-robust "
                         "remote runs: a re-run continues at the first unfinished phase)")
    ap.add_argument("--seed", type=int, help="mcts_cmd: the MCTS seed")
    args = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)
    if args.phases == ["mcts_cmd"]:
        print(mcts_cmd(args.seed))
        return
    names = list(ALL) if args.phases == ["all"] else args.phases
    for name in names:
        done = json.load(open(META_JSON)).get("phases_done", []) if os.path.exists(META_JSON) else []
        if args.resume and name in done:
            log(f"=== phase {name}: done, skipping ===")
            continue
        log(f"=== phase {name} ===")
        PHASES[name]()
        if name not in ("gates",):
            m = json.load(open(META_JSON)) if os.path.exists(META_JSON) else {}
            meta_update(phases_done=sorted(set(m.get("phases_done", [])) | {name}))


if __name__ == "__main__":
    main()
