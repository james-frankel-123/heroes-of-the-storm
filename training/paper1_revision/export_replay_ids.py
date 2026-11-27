"""
Replay-id list for the code release: every replay of paper 1's corpus with its split and fold, plus the
post-snapshot games. Raw replays are not redistributed (Heroes Profile terms); these ids, with the code,
rebuild every split and statistic from Heroes Profile's public replay endpoint.

Columns (CSV, gzip):
  replay_ids_snapshot.csv.gz   replay_id, split (train | val | test | unknown_tier), fold (0..4, out-of-fold
                               statistics; paper1_revision.oct2026_data.fold_of)
  replay_ids_post.csv.gz       replay_id, tier (the 260,539 post-snapshot games, site tiers)

Usage (from training/):
  P1_TIERS=site RERUN_NS=p1site RERUN_SPLIT=p1val REPLAY_SNAPSHOT=1 \
  REPLAY_SNAPSHOT_PATH=snapshots/replay_snapshot_2026-05-22_1956753_p1site.json \
  nice -n 19 taskset -c 44-47 python3 paper1_revision/export_replay_ids.py <out_dir>
"""
import gzip
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.dirname(HERE)
sys.path.insert(0, T)
sys.path.insert(0, HERE)

import numpy as np


def main(out):
    os.makedirs(out, exist_ok=True)
    from rerun2026 import common
    from shared import split_data
    from oct2026_data import fold_of
    from overfit2026.data import splitmix64
    rows = common._load_all()
    train, test = split_data(rows, test_frac=common.FULL_TEST_FRAC, seed=common.SEED)
    is_val = lambda rid: splitmix64(int(rid) * 998244353 + 23) % 50 == 0
    known = lambda r: r.get("skill_tier") in ("low", "mid", "high")
    out_rows = []
    for r in train:
        rid = int(r["replay_id"])
        s = "unknown_tier" if not known(r) else ("val" if is_val(rid) else "train")
        out_rows.append((rid, s))
    for r in test:
        out_rows.append((int(r["replay_id"]), "test" if known(r) else "unknown_tier"))
    out_rows.sort()
    with gzip.open(os.path.join(out, "replay_ids_snapshot.csv.gz"), "wt") as f:
        f.write("replay_id,split,fold\n")
        for rid, s in out_rows:
            f.write(f"{rid},{s},{fold_of(rid)}\n")
    cnt = {}
    for _, s in out_rows:
        cnt[s] = cnt.get(s, 0) + 1
    print("snapshot", len(out_rows), cnt, flush=True)
    z = np.load(os.path.join(T, "overfit2026", "site", "cache", "comp_audit_N.npz"))
    o = np.argsort(z["rid"])
    with gzip.open(os.path.join(out, "replay_ids_post.csv.gz"), "wt") as f:
        f.write("replay_id,tier\n")
        for rid, t in zip(z["rid"][o], z["tier"][o]):
            f.write(f"{int(rid)},{t}\n")
    print("post", len(o), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
