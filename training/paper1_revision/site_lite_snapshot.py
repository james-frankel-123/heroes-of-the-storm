"""
A lean copy of the site-tier snapshot for MCTS value pretraining on
memory-limited workers (the 3080: ~15 GB per job). The worker's value-head
pretraining (train_draft_policy.pretrain_value_head) reads only replay_id,
game_map, skill_tier, team0/team1 heroes and bans, and winner; the full rows
also carry the draft order and listing metadata, which make the parsed
snapshot several times larger. Same rows in the same order, so
shared.split_data and the worker's exclude filter select exactly the same
games as with the full file.

Usage (on a host with enough RAM, from training/): python paper1_revision/site_lite_snapshot.py
Output: snapshots/replay_snapshot_2026-05-22_1956753_p1site_lite.json
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
SRC = os.path.join(TRAINING_DIR, "snapshots", "replay_snapshot_2026-05-22_1956753_p1site.json")
OUT = SRC[:-5] + "_lite.json"
KEEP = ("replay_id", "game_map", "skill_tier", "team0_heroes", "team1_heroes",
        "team0_bans", "team1_bans", "winner")


def main():
    rows = json.load(open(SRC))
    lite = [{k: r.get(k) for k in KEEP} for r in rows]
    del rows
    json.dump(lite, open(OUT + ".tmp", "w"))
    os.replace(OUT + ".tmp", OUT)
    print(len(lite), os.path.getsize(OUT))


if __name__ == "__main__":
    main()
