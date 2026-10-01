"""
Drift paper v2 environment (2026-10-01): every drift analysis rebuilt on
correctly tiered data.

Tier rule = the site scheme (sync/relabel-skill-tier.ts, sync-replays.ts
leagueTierToSkillTier), applied to the pinned snapshot's stored league_tier
and avg_mmr:
    league_tier NULL -> 'high' (Master) if avg_mmr is set, else 'unknown'
    league_tier <= 3 -> 'low'   (Bronze + Silver)
    league_tier <= 5 -> 'mid'   (Gold + Platinum)
    else             -> 'high'  (Diamond)
Rows labeled 'unknown' are dropped.

Importing this module (before any drift2026 / drift_rebuild module):
  * patches rerun2026.common.load_data and shared.load_replay_data so every
    loader returns relabeled rows with 'unknown' dropped;
  * redirects drift2026.common's and drift_rebuild.rb_common's output
    directories to training/drift_v2/ (the patch sidecar and patch index,
    which carry no tiers, stay shared);
  * caps multiprocessing at V2_WORKERS (default 8) worker processes.
Nothing in drift2026/ or the pre-registration's frozen files is modified.
"""
import multiprocessing as _mp
import os
import sys

TRAINING = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2 = os.path.join(TRAINING, "drift_v2")
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.join(TRAINING, "drift_rebuild"))

_W = int(os.environ.get("V2_WORKERS", "8"))
# fork, so pool workers inherit the patched loaders and redirected paths
# (Python 3.14 defaults to forkserver, whose workers re-import modules fresh).
_mp.set_start_method("fork", force=True)
_mp.cpu_count = lambda: _W
os.cpu_count = lambda: _W


def site_tier(league_tier, avg_mmr):
    if league_tier is None:
        return "unknown" if avg_mmr is None else "high"
    lt = int(league_tier)
    return "low" if lt <= 3 else ("mid" if lt <= 5 else "high")


def relabel(rows):
    out = []
    for r in rows:
        t = site_tier(r.get("league_tier"), r.get("avg_mmr"))
        if t == "unknown":
            continue
        r["skill_tier"] = t
        out.append(r)
    return out


import shared as _shared  # noqa: E402
from rerun2026 import common as _rc  # noqa: E402

_orig_rc_load = _rc.load_data
_orig_shared_load = _shared.load_replay_data


def _rc_load_data(*a, **k):
    rows = relabel(_orig_rc_load(*a, **k))
    print(f"[v2] relabeled to site tiers; {len(rows):,} rows after dropping unknown", flush=True)
    return rows


def _shared_load(*a, **k):
    return relabel(_orig_shared_load(*a, **k))


_rc.load_data = _rc_load_data
_shared.load_replay_data = _shared_load

from drift2026 import common as _dc  # noqa: E402

_dc.DRIFT_DIR = V2
_dc.MODELS_DIR = os.path.join(V2, "models")
_dc.LOGS_DIR = os.path.join(V2, "logs")
_dc.RESULTS_DIR = os.path.join(V2, "results")
_dc.CACHE_DIR = os.path.join(V2, "feature_cache")
_dc.STATS_DIR = os.path.join(V2, "patch_stats")
_dc.COUNTS_PKL = os.path.join(_dc.STATS_DIR, "patch_counts.pkl.gz")

import rb_common as _rb  # noqa: E402

_rb.CACHE_DIR = _dc.CACHE_DIR
_rb.MODELS_DIR = _dc.MODELS_DIR
_rb.RESULTS_DIR = _dc.RESULTS_DIR
_rb.LOGS_DIR = _dc.LOGS_DIR
_rb.RUNS_DIR = os.path.join(V2, "mcts_runs")
for _d in (_dc.MODELS_DIR, _dc.LOGS_DIR, _dc.RESULTS_DIR, _dc.CACHE_DIR, _dc.STATS_DIR, _rb.RUNS_DIR):
    os.makedirs(_d, exist_ok=True)
