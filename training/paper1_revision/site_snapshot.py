"""
The pinned paper-1 snapshot with site-scheme tiers, for the rerun2026
pipeline run in namespace p1site (GD, CQL, BC-CQL, MCQ, IQL, the
discriminator, the Gourdeau estimator, rich evaluation, tournament).

Same rows in the same order as snapshots/replay_snapshot_2026-05-22_1956753.json
(so rerun2026.common's permutation split assigns every replay exactly as
before); only skill_tier changes, to the site scheme from the stored
league_tier / avg_mmr (overfit2026.data.site_tiers). Unranked rows get
'unknown' and are dropped after the split (RERUN_SPLIT=p1val). Rows outside
the tier backup (pre-2.55 replays, excluded anyway) are 'unknown'.

Usage (from training/): python3 paper1_revision/site_snapshot.py
Output: snapshots/replay_snapshot_2026-05-22_1956753_p1site.json
"""
import os
import sys
import json
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

SRC = os.path.join(TRAINING_DIR, "snapshots", "replay_snapshot_2026-05-22_1956753.json")
OUT = os.path.join(TRAINING_DIR, "snapshots", "replay_snapshot_2026-05-22_1956753_p1site.json")


def main():
    from overfit2026.data import site_tiers
    if os.path.exists(OUT):
        print("exists", OUT)
        return
    S = site_tiers()
    rows = json.load(open(SRC))
    c = Counter()
    for r in rows:
        r["skill_tier"] = S.get(int(r["replay_id"]), "unknown")
        c[r["skill_tier"]] += 1
    json.dump(rows, open(OUT + ".tmp", "w"))
    os.replace(OUT + ".tmp", OUT)
    print(len(rows), dict(c))


if __name__ == "__main__":
    main()
