"""
Stale vintages of the paper-1 drafter (paper 2 drift question, player-aware
drafter tournament). A vintage rebuilds the whole paper-1 pipeline (stats,
features, enriched WP, MCTS config F) from ONLY the paper-1 train/val games
whose build is at or before a cutoff build; the test split is unchanged.

Enabled by P1_VINTAGE=<name> (requires P1_TIERS=site). paper1_revision/core.py
then puts every artifact under paper1_revision/site/vint/<name>/ and filters
split_games()' train and val; train_mcts.py takes the vintage's GD opponent
and a pretrain exclude list of every snapshot row outside the vintage train
split. With P1_VINTAGE unset nothing changes.

Builds are ordered as in drift2026 (patch_index.json, sidecar build index):
"at or before the cutoff" = build_idx <= index(cutoff), the rule the drift
GD pools (drift_rebuild/v2/v2_gd.py --pool era:<b> / cutoff) use.
"""
import os
import json
import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)

VINTAGES = {
    # 4 months stale: the drift cutoff (last game 2026-02-10)
    "s4m": {"cutoff": "2.55.14.95918", "last_game": "2026-02-10",
            "gd": "drift_v2/models/gd_cutoff/generic_draft_0.pt"},
    # 1 year stale (last game 2025-06-02)
    "s1y": {"cutoff": "2.55.10.94189", "last_game": "2025-06-02",
            "gd": "drift_v2/models/gd_era_2.55.10.94189/generic_draft_0.pt"},
}

NAME = os.environ.get("P1_VINTAGE", "")
if NAME and NAME not in VINTAGES:
    raise SystemExit(f"P1_VINTAGE={NAME}: unknown vintage (have {sorted(VINTAGES)})")

SIDECAR = os.path.join(TRAINING_DIR, "drift2026", "patch_sidecar.npz")
PATCH_INDEX = os.path.join(TRAINING_DIR, "drift2026", "patch_index.json")
_SC = None


def spec(name=None):
    return VINTAGES[name or NAME]


def gd_path(name=None):
    return os.path.join(TRAINING_DIR, spec(name)["gd"])


def _sidecar():
    global _SC
    if _SC is None:
        z = np.load(SIDECAR, allow_pickle=False)
        ids = z["replay_ids"].astype(np.int64)
        o = np.argsort(ids)
        builds = json.load(open(PATCH_INDEX))["builds"]
        _SC = (ids[o], z["build_idx"][o].astype(np.int64), z["date_days"][o].astype(np.int64), builds)
    return _SC


def build_info(rids):
    """-> (build_idx, date_days) arrays for replay ids; KeyError if any id is
    missing from the sidecar (a game of unknown build is never admitted)."""
    ids, bidx, days, _ = _sidecar()
    r = np.asarray(rids, np.int64)
    j = np.searchsorted(ids, r)
    j = np.minimum(j, len(ids) - 1)
    bad = ids[j] != r
    if bad.any():
        raise KeyError(f"{int(bad.sum())} replay ids missing from the patch sidecar, e.g. {r[bad][:5].tolist()}")
    return bidx[j], days[j]


def cutoff_idx(name=None):
    return _sidecar()[3].index(spec(name)["cutoff"])


def keep(games, name=None):
    """Games (slim tuples, replay id first) on builds <= the vintage cutoff."""
    if not games:
        return []
    b, _ = build_info([g[0] for g in games])
    cut = cutoff_idx(name)
    return [g for g, x in zip(games, b) if x <= cut]


def _day(s):
    return (datetime.date.fromisoformat(s) - datetime.date(1970, 1, 1)).days


def check(rids, what, name=None):
    """Assert no replay after the cutoff: max build index and max game date."""
    if len(rids) == 0:
        return
    b, d = build_info(rids)
    cut = cutoff_idx(name)
    last = _day(spec(name)["last_game"])
    assert int(b.max()) <= cut, f"{what}: build {_sidecar()[3][int(b.max())]} after cutoff"
    assert int(d.max()) <= last, f"{what}: game on day {int(d.max())} after {spec(name)['last_game']}"
    mx = datetime.date(1970, 1, 1) + datetime.timedelta(days=int(d.max()))
    print(f"[vintage {name or NAME}] {what}: {len(rids):,} games, max build "
          f"{_sidecar()[3][int(b.max())]}, last game {mx} (cutoff {spec(name)['cutoff']})", flush=True)
