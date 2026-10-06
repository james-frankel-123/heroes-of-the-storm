"""
P3 full-history inputs (review item #2, after the player backfill).

The window caches hold player rows of Storm League games from 2024-04-01 on
(players_2024q2.npz, hero_mmr_2024q2.npz). Once the backfill has stored
players for the older replays, this script rebuilds both files from a
frozen export (p3_export.py stamps, parsed by p3_hero_level_causal.py parse)
over the whole scored range (games with population WP scores in
wp_drift.npz, back to 2021-12, replay_id within the snapshot bound), in a
separate cache directory. Every other input is linked from the window cache;
derived tables are rebuilt there by the pipeline (P3_CACHE=<dir>).

Rows keep the window file layout and hero order, so the same code runs on
both. The E window (tables and kernels) stays 2024-04-01 on; older games
only add to counts and to the skill state, and the scored games (V1, V2,
OOT) are the same, so the comparison separates coverage from estimation.

Usage (from training/):
  python3 personalization/p3_c_history.py --tag oct15 --out personalization/cache_full
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C

DERIVED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote", "derived_caches.txt")
SNAPSHOT_BOUND = 63653039


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = C.CACHE
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    z = np.load(os.path.join(src, "export", a.tag, "stamps.npz"))
    w = np.load(os.path.join(src, "wp_drift.npz"))
    scored = np.sort(w["replay_ids"])
    rid = z["replay_ids"]
    j = np.minimum(np.searchsorted(scored, rid), len(scored) - 1)
    keep = (scored[j] == rid) & (rid <= SNAPSHOT_BOUND) & (z["hero"] >= 0) & (z["team"] >= 0)
    old = np.load(os.path.join(src, "players_2024q2.npz"))
    names = [str(h) for h in old["hero_names"]]
    assert names == [str(h) for h in z["hero_names"]], "hero order differs from the window cache"
    k = np.flatnonzero(keep)
    print(f"export rows {len(rid):,}; in scored snapshot games {len(k):,} "
          f"(window file had {len(old['replay_ids']):,})", flush=True)
    np.savez(os.path.join(out, "players_2024q2.npz"), replay_ids=rid[k], blizz_ids=z["blizz_ids"][k],
             hero=z["hero"][k].astype(np.int16), team=z["team"][k], party=z["party"][k],
             mmr=z["player_mmr"][k], hero_names=old["hero_names"])
    hl = np.where(np.isnan(z["lo"][k]), -1, np.where(z["lo"][k] == z["hi"][k], z["lo"][k], -1)).astype(np.int16)
    np.savez(os.path.join(out, "hero_mmr_2024q2.npz"), replay_ids=rid[k], blizz_ids=z["blizz_ids"][k],
             region=z["region"][k], hero_mmr=z["hero_mmr"][k], role_mmr=z["role_mmr"][k], hero_level=hl)
    # HP rating stamps with parse times for p3_mmr_at_game (all exported rows)
    ok = z["hero"] >= 0
    np.savez(os.path.join(out, "mmr_stamps_db.npz"), replay_ids=rid[ok], blizz_ids=z["blizz_ids"][ok],
             region=z["region"][ok], hero=z["hero"][ok], player_mmr=z["player_mmr"][ok],
             hero_mmr=z["hero_mmr"][ok], role_mmr=z["role_mmr"][ok], parse_ts=z["parse_ts"][ok],
             end_ts=z["end_ts"][ok], game_length=z["game_length"][ok], winner=z["winner"][ok] == 1,
             hero_names=old["hero_names"])
    # game times for every exported game (the window file starts 2024-04-01)
    import csv
    import gzip
    gr, gt, gl = [], [], []
    with gzip.open(os.path.join(src, "export", a.tag, "games.csv.gz"), "rt", newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        ir, ie, il = hdr.index("replay_id"), hdr.index("end_ts"), hdr.index("game_length")
        for r in rd:
            gr.append(int(r[ir]))
            gt.append(int(r[ie]))
            gl.append(int(r[il]) if r[il] else -1)
    np.savez(os.path.join(out, "gametime_2024q2.npz"), replay_ids=np.array(gr, np.int64),
             ts=np.array(gt, np.int64), game_length=np.array(gl, np.int32))
    # every other input from the window cache, never a derived table nor a
    # table aligned to the window slot order (mmr_at_game is rebuilt)
    import re
    pats = [re.compile(l.strip()) for l in open(DERIVED) if l.strip()]
    linked = 0
    for f in os.listdir(src):
        if f in ("players_2024q2.npz", "hero_mmr_2024q2.npz", "gametime_2024q2.npz", "mmr_at_game.npz", "mmr_stamps_db.npz",
                 "_legacy_pre_oct26", "mcts_v2", "fix"):
            continue
        if any(p.search("/" + f) for p in pats):
            continue
        dst = os.path.join(out, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(src, f), dst)
            linked += 1
    print(f"linked {linked} inputs into {out}")


if __name__ == "__main__":
    main()
