"""
The 5,000 sealed 2.57 replays (training/replays/cohorts/SEALED_README.md):
never read. Every P3 path that reads games on or after 2026-09-28 excludes
them through this module (no-op for builds capped at 2026-09-28).
"""
import os

import numpy as np

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "training", "replays", "cohorts", "sealed_ids.txt")


def sealed_ids():
    ids = np.array(sorted(int(x) for x in open(_PATH).read().split()), np.int64)
    assert len(ids) == 5000, f"sealed list has {len(ids)} ids"
    return ids


def sql_exclude(col="d.replay_id"):
    """SQL predicate excluding the sealed replays (ids inlined; no parameters)."""
    return f"NOT ({col} = ANY('{{{','.join(str(int(x)) for x in sealed_ids())}}}'::bigint[]))"
