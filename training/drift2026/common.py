"""
Shared utilities for the drift2026 suite (Paper 2 — patch drift).

Builds on rerun2026 conventions:
  - Dataset of record: the SAME pinned snapshot + 2.55 filter as rerun2026
    (rerun2026.common.load_data()). drift2026 NEVER refetches full replays.
  - Patch identity comes from a compact sidecar (drift2026/patch_sidecar.npz:
    replay_id -> game_version/build index + game_date), fetched once from the
    DB by fetch_patch_data.py for all replay_id <= SNAPSHOT_BOUND.
  - GPU pool / Job class are reused from rerun2026.common (logs redirected to
    drift2026/logs/).

PATCH UNIT: the *build* (full game_version, e.g. "2.55.14.95918"), NOT the
minor patch. The 2.55 corpus has only 17 minor patches but 45 builds (28 with
>= 20K games); HotS late-life balance patches ship as builds — e.g. minor
patch 2.55.3 spans 17 months across 8 builds. Builds are ordered by
(minor, build_number), which matches chronological order of min(game_date).

TEMPORAL SPLIT (D2): strictly-future evaluation.
  - TRAIN_CUTOFF_BUILD = "2.55.14.95918" (through 2026-02-10). Train pool =
    all builds <= cutoff (~1.80M replays, build indices 0..CUTOFF_IDX).
  - TEST = every build after the cutoff (all of 2.55.15 + 2.55.16,
    ~144K replays), reported per build; the 4 sizable test builds
    (>= 10K games: 2.55.15.96370, 2.55.15.96477, 2.55.16.96881,
    2.55.16.97039) are the headline held-out patches.

LEAKAGE RULES:
  - Per-build stats ("local") use only that build's games.
  - Cumulative stats through build N use builds 0..N only.
  - D2 regime-comparison features use cumulative-through-CUTOFF stats for
    BOTH train and test rows (nothing after the cutoff touches any feature).
  - The paper-1 frozen stats (2026-05-19, end-of-time) appear only as the
    explicitly-leaky control arm.
"""
import os
import sys
import json
import gzip

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIFT_DIR = os.path.join(TRAINING_DIR, "drift2026")
MODELS_DIR = os.path.join(DRIFT_DIR, "models")
LOGS_DIR = os.path.join(DRIFT_DIR, "logs")
RESULTS_DIR = os.path.join(DRIFT_DIR, "results")
CACHE_DIR = os.path.join(DRIFT_DIR, "feature_cache")
STATS_DIR = os.path.join(DRIFT_DIR, "patch_stats")

SIDECAR_NPZ = os.path.join(DRIFT_DIR, "patch_sidecar.npz")
PATCH_INDEX_JSON = os.path.join(DRIFT_DIR, "patch_index.json")
COUNTS_PKL = os.path.join(STATS_DIR, "patch_counts.pkl.gz")

# Snapshot bound: max replay_id covered by the pinned 2026-05-22 snapshot
# (same constant rerun2026 uses for the pre-2.55 exclusion query).
SNAPSHOT_BOUND = 63653039

SEED = 42
SEEDS = [42, 123, 777]

# Temporal split (see header). CUTOFF build index is derived from the ordered
# build list at load time; these constants are asserted against it.
TRAIN_CUTOFF_BUILD = "2.55.14.95918"
SIZABLE_TEST_BUILDS = ["2.55.15.96370", "2.55.15.96477",
                       "2.55.16.96881", "2.55.16.97039"]
SIZABLE_GAMES = 20000   # a build is "sizable" at >= 20K games (28 of 45)

sys.path.insert(0, TRAINING_DIR)
from rerun2026 import common as rerun_common  # noqa: E402


def setup():
    """Create dirs, pin env/paths via rerun2026.common.setup(), redirect the
    shared GPU pool's log directory to drift2026/logs/."""
    for d in (MODELS_DIR, LOGS_DIR, RESULTS_DIR, CACHE_DIR, STATS_DIR):
        os.makedirs(d, exist_ok=True)
    rerun_common.setup()
    rerun_common.LOGS_DIR = LOGS_DIR  # pool logs land here


def build_sort_key(version):
    """Chronological ordering key for a full game_version string."""
    parts = version.split(".")
    return tuple(int(p) for p in parts)


def minor_key(version):
    """'2.55.14.95918' -> '2.55.14'."""
    return ".".join(version.split(".")[:3])


def load_patch_index():
    """Ordered build list + metadata written by fetch_patch_data.py.
    Returns dict with keys: builds (ordered list of full versions),
    build_meta {version: {n_db, min_date, max_date}}."""
    with open(PATCH_INDEX_JSON) as f:
        return json.load(f)


def load_sidecar():
    """replay_id -> (build_idx, date_days) arrays plus the ordered build list.
    date_days = days since 1970-01-01 (UTC) of game_date."""
    import numpy as np
    z = np.load(SIDECAR_NPZ, allow_pickle=False)
    idx = load_patch_index()
    return z["replay_ids"], z["build_idx"], z["date_days"], idx["builds"]


def load_data_with_patches():
    """rerun2026.common.load_data() (pinned snapshot, 2.55-filtered) joined
    with the patch sidecar. Attaches to every row:
        build      full game_version
        build_idx  index into the ordered build list
        date_days  int days since epoch
    Rows absent from the sidecar are dropped (count printed; expected 0).
    Returns (rows, builds)."""
    rows = rerun_common.load_data()
    ids, bidx, days, builds = load_sidecar()
    lookup = {}
    for i in range(len(ids)):
        lookup[int(ids[i])] = i
    kept, missing = [], 0
    for r in rows:
        j = lookup.get(r["replay_id"])
        if j is None:
            missing += 1
            continue
        r["build_idx"] = int(bidx[j])
        r["build"] = builds[r["build_idx"]]
        r["date_days"] = int(days[j])
        kept.append(r)
    print(f"Patch join: {len(kept)} rows with build info, {missing} missing "
          f"from sidecar (dropped)")
    return kept, builds


def cutoff_idx(builds):
    i = builds.index(TRAIN_CUTOFF_BUILD)
    # sanity: every SIZABLE_TEST_BUILD must be strictly after the cutoff
    for b in SIZABLE_TEST_BUILDS:
        assert builds.index(b) > i, b
    return i


# ── Patch-stats objects (StatsCache drop-in) ──

class PatchStats:
    """StatsCache-shaped object built from a drift2026 per-patch/cumulative
    stats file (or raw dicts). Duck-types sweep_enriched_wp.StatsCache: has
    hero_wr / hero_meta / hero_map_wr / pairwise / comp_data attributes and
    borrows the getter methods, so extract_features() consumes it as-is."""

    def __init__(self, hero_wr, hero_meta, hero_map_wr, pairwise, comp_data):
        self.hero_wr = hero_wr
        self.hero_meta = hero_meta
        self.hero_map_wr = hero_map_wr
        self.pairwise = pairwise
        self.comp_data = comp_data


def _bind_statscache_methods():
    from sweep_enriched_wp import StatsCache
    for m in ("get_comp_wr", "get_hero_wr", "get_hero_map_wr",
              "get_counter", "get_synergy"):
        setattr(PatchStats, m, getattr(StatsCache, m))


def stats_path(kind, build):
    """kind: 'per_patch' | 'cumulative'. One file per build."""
    return os.path.join(STATS_DIR, kind, f"{build}.json.gz")


def load_patch_stats(kind, build):
    """Load a drift2026 stats file into a PatchStats object."""
    _bind_statscache_methods()
    with gzip.open(stats_path(kind, build), "rt") as f:
        raw = json.load(f)
    hero_wr, hero_meta = {}, {}
    for r in raw["hero_stats"]:
        hero_wr.setdefault(r["tier"], {})[r["hero"]] = r["win_rate"]
        hero_meta.setdefault(r["tier"], {})[r["hero"]] = (r["pick_rate"], r["ban_rate"])
    hero_map_wr = {}
    for r in raw["hero_map_stats"]:
        hero_map_wr.setdefault(r["tier"], {}).setdefault(r["map"], {})[r["hero"]] = \
            (r["win_rate"], r["games"])
    pairwise = {}
    for r in raw["pairwise_stats"]:
        pairwise.setdefault(r["tier"], {}).setdefault(r["relationship"], {}) \
            .setdefault(r["hero_a"], {})[r["hero_b"]] = (r["win_rate"], r["games"])
    comp_data = {}
    for r in raw["comp_stats"]:
        comp_data.setdefault(r["tier"], {})[r["roles"]] = (r["win_rate"], r["games"])
    return PatchStats(hero_wr, hero_meta, hero_map_wr, pairwise, comp_data)


def stats_from_counts(counts_by_tier, pair_min=10):
    """Derive a PatchStats directly from raw count cells ({tier: cell} in the
    patch_counts.pkl.gz format), mirroring build_patch_stats.derive_stats_file
    thresholds (hero >= 20 games, map/comp >= 5, pairwise >= pair_min).
    Used e.g. to build 'future-period ground truth' stats by merging the
    post-cutoff builds' counts."""
    _bind_statscache_methods()
    hero_wr, hero_meta, hero_map_wr, pairwise, comp_data = {}, {}, {}, {}, {}
    for tier, cell in counts_by_tier.items():
        total = cell["games"]
        if not total:
            continue
        hw = hero_wr.setdefault(tier, {})
        hm = hero_meta.setdefault(tier, {})
        for h, (g, w) in cell["hero"].items():
            if g < 20:
                continue
            hw[h] = 100.0 * w / g
            hm[h] = (100.0 * g / total, 100.0 * cell["bans"].get(h, 0) / total)
        hmap = hero_map_wr.setdefault(tier, {})
        for (m, h), (g, w) in cell["hmap"].items():
            if g >= 5:
                hmap.setdefault(m, {})[h] = (100.0 * w / g, g)
        pw = pairwise.setdefault(tier, {"with": {}, "against": {}})
        for (a, b), (g, w) in cell["with"].items():
            if g < pair_min:
                continue
            wr = 100.0 * w / g
            pw["with"].setdefault(a, {})[b] = (wr, g)
            pw["with"].setdefault(b, {})[a] = (wr, g)
        for (a, b), (g, wa) in cell["against"].items():
            if g < pair_min:
                continue
            wr_a = 100.0 * wa / g
            pw["against"].setdefault(a, {})[b] = (wr_a, g)
            pw["against"].setdefault(b, {})[a] = (100.0 - wr_a, g)
        cd = comp_data.setdefault(tier, {})
        for ck, (g, w) in cell["comp"].items():
            if g >= 5:
                cd[ck] = (100.0 * w / g, g)
    return PatchStats(hero_wr, hero_meta, hero_map_wr, pairwise, comp_data)


def empty_stats():
    """All-defaults stats: what a causal system has before any 2.55 data
    exists (used for build 0 in the cumulative_prev feature pass)."""
    _bind_statscache_methods()
    return PatchStats({}, {}, {}, {}, {})


def stats_as_dict(stats):
    """Same serialization shape rerun2026.common.stats_as_dict uses."""
    return {
        "hero_wr": stats.hero_wr,
        "hero_meta": stats.hero_meta,
        "hero_map_wr": stats.hero_map_wr,
        "pairwise": stats.pairwise,
        "comp_data": stats.comp_data,
    }


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Wrote {path}")


# Re-exports for phase scripts
Job = rerun_common.Job
run_pool = rerun_common.run_pool
NUM_GPUS = rerun_common.NUM_GPUS
