"""
P3 personalized GD, stage 1: one draft table for every game.

Games: every game of the extended slot table (cache/x_slots_ext.npz): the
snapshot (2024-04-01 .. 2026-05-22) and the post-snapshot Storm League games
(build 2.55.16.97039 after the snapshot and 2.55.17.*, dated before
2026-09-28). Kept: standard 16-step drafts whose ten picks join to the ten
player rows.

Sources: snapshot picks from cache/pickorder_2024q2.npz (pick rank), bans
from cache/x_bans.npz (pick_number, team, first-pick team), map and tier
from cache/x_side_games.npz; post-snapshot drafts from
cache/x_post_games.json.gz (draft_order).

Standard order with f = the first-picking team and o the other:
  steps  0-3   bans  f o f o
  steps  4-8   picks f o o f f
  steps  9-10  bans  o f
  steps 11-15  picks o o f f o
Every kept game is checked against this pattern.

Also: per-day, per-tier hero pick and ban counts (for the causal 28-day meta
rates used as model inputs).

Run (from training/): OMP_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_pgd_data.py
Output: cache/pgd_games.npz
"""
import os
import sys
import gzip
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C
import p3_x_common as X

OUT = os.path.join(C.CACHE, "pgd_games.npz")
TYPES = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1])
FIRST = np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0])  # 1 = first-picking team
PICK_STEPS = np.flatnonzero(TYPES == 1)
BAN_STEPS = np.flatnonzero(TYPES == 0)


def main():
    t0 = time.time()
    from shared import MAPS, SKILL_TIERS
    d = X.load_ext()
    names = [str(h) for h in d["hero_names"]]
    hidx = {n: i for i, n in enumerate(names)}
    n_rows = len(d["pid"])
    rkey = d["replay_id"] * 128 + d["hero"]
    ro = np.argsort(rkey)
    rkey_s = rkey[ro]

    def rows_of(rid, hero):
        k = rid * 128 + hero
        j = np.searchsorted(rkey_s, k)
        ok = (j < n_rows) & (rkey_s[np.minimum(j, n_rows - 1)] == k)
        return np.where(ok, ro[np.minimum(j, n_rows - 1)], -1)

    # ---------- snapshot games
    po = np.load(os.path.join(C.CACHE, "pickorder_2024q2.npz"))
    uh, inv = np.unique(po["hero"], return_inverse=True)
    hmap = np.array([hidx.get(str(h), -1) for h in uh])
    ph = hmap[inv]
    prid = po["replay_ids"]
    prow = rows_of(prid, np.maximum(ph, 0))
    prow[ph < 0] = -1
    o = np.lexsort((po["pick_rank"], prid))
    prid, ph, prank, prow = prid[o], ph[o], po["pick_rank"][o], prow[o]
    ug, gs, gc = np.unique(prid, return_index=True, return_counts=True)
    good = gc == 10
    snap = {}
    for gi, s0 in zip(ug[good], gs[good]):
        pass
    sel = np.concatenate([np.arange(s, s + 10) for s in gs[good]]) if good.any() else np.array([], int)
    P_h = ph[sel].reshape(-1, 10)
    P_row = prow[sel].reshape(-1, 10)
    P_rank = prank[sel].reshape(-1, 10)
    rid_s = ug[good]
    okp = (P_h >= 0).all(1) & (P_row >= 0).all(1) & (P_rank == np.arange(10)).all(1)
    bz = np.load(os.path.join(C.CACHE, "x_bans.npz"))
    bo = np.argsort(bz["replay_ids"])
    bj = np.searchsorted(bz["replay_ids"][bo], rid_s)
    bj = np.minimum(bj, len(bo) - 1)
    okb = bz["replay_ids"][bo][bj] == rid_s
    bi = bo[bj]
    B_h = bz["hero"][bi]
    B_t = bz["team"][bi]
    B_pn = bz["pick_number"][bi]
    first = bz["first_pick_team"][bi]
    okb &= (bz["n_bans"][bi] == 6) & (B_h >= 0).all(1) & np.isin(first, [0, 1])
    # sort bans by pick_number
    bs = np.argsort(np.where(B_pn < 0, 99, B_pn), axis=1)
    B_h = np.take_along_axis(B_h, bs, 1)
    B_t = np.take_along_axis(B_t, bs, 1)
    sm = np.load(os.path.join(C.CACHE, "x_side_games.npz"), allow_pickle=True)
    so = np.argsort(sm["replay_ids"])
    sj = np.minimum(np.searchsorted(sm["replay_ids"][so], rid_s), len(so) - 1)
    oks = sm["replay_ids"][so][sj] == rid_s
    mp = np.array([MAPS.index(str(n)) for n in sm["map_names"]])[sm["map"][so][sj]]
    tr = np.array([SKILL_TIERS.index(str(n)) for n in sm["tier_names"]])[sm["tier"][so][sj]]
    G = len(rid_s)
    H = np.full((G, 16), -1, np.int64)
    T = np.full((G, 16), -1, np.int64)
    R = np.full((G, 16), -1, np.int64)
    H[:, PICK_STEPS] = P_h
    R[:, PICK_STEPS] = P_row
    T[:, PICK_STEPS] = d["team"][np.maximum(P_row, 0)]
    H[:, BAN_STEPS] = B_h
    T[:, BAN_STEPS] = B_t
    exp_team = np.where(FIRST[None, :] == 1, first[:, None], 1 - first[:, None])
    okt = (T == exp_team).all(1)
    keep = okp & okb & oks & okt
    print(f"snapshot: {G:,} games with 10 ranked picks; kept {keep.sum():,} "
          f"(picks {okp.mean():.3f}, bans {okb.mean():.3f}, meta {oks.mean():.3f}, order {okt[okp & okb].mean():.4f})",
          flush=True)
    games = dict(rid=rid_s[keep], H=H[keep], T=T[keep], R=R[keep], map=mp[keep], tier=tr[keep],
                 first=first[keep], post=np.zeros(keep.sum(), bool))
    # ---------- post-snapshot games
    pg = json.load(gzip.open(os.path.join(C.CACHE, "x_post_games.json.gz"), "rt"))
    in_ext = set(d["replay_id"][d["post"]].tolist())
    rows_p = []
    n_skip_map = 0
    for g in pg:
        if g["replay_id"] not in in_ext or not g.get("draft_order") or len(g["draft_order"]) != 16:
            continue
        hs, ts_, ty = [], [], []
        ok = True
        for h, t, pn, sl in g["draft_order"]:
            if h not in hidx:
                ok = False
                break
            hs.append(hidx[h])
            ty.append(int(t))
            ts_.append((0 if sl == 1 else 1) if int(t) == 0 else (0 if sl < 5 else 1))
        if not ok or ty != TYPES.tolist():
            continue
        f = ts_[4]
        if ts_ != np.where(FIRST == 1, f, 1 - f).tolist():
            continue
        if g["game_map"] not in MAPS or g["skill_tier"] not in SKILL_TIERS:
            n_skip_map += 1
            continue
        rows_p.append((g["replay_id"], hs, ts_, MAPS.index(g["game_map"]), SKILL_TIERS.index(g["skill_tier"]), f))
    rp = np.array([r[0] for r in rows_p], np.int64)
    Hp = np.array([r[1] for r in rows_p], np.int64)
    Tp = np.array([r[2] for r in rows_p], np.int64)
    Rp = np.full(Hp.shape, -1, np.int64)
    Rp[:, PICK_STEPS] = rows_of(np.repeat(rp, 10), Hp[:, PICK_STEPS].ravel()).reshape(-1, 10)
    okr = (Rp[:, PICK_STEPS] >= 0).all(1)
    okr &= (d["team"][np.maximum(Rp[:, PICK_STEPS], 0)] == Tp[:, PICK_STEPS]).all(1)
    print(f"post: {len(pg):,} games, {len(rp):,} standard drafts on the 14 GD maps "
          f"({n_skip_map:,} skipped: map outside the GD map set), kept {okr.sum():,}", flush=True)
    for k, v in dict(rid=rp, H=Hp, T=Tp, R=Rp, map=np.array([r[3] for r in rows_p]),
                     tier=np.array([r[4] for r in rows_p]), first=np.array([r[5] for r in rows_p]),
                     post=np.ones(len(rp), bool)).items():
        games[k] = np.concatenate([games[k], v[okr]])
    games["day"] = d["day"][games["R"][:, 4]]
    games["g"] = d["g"][games["R"][:, 4]]
    o = np.lexsort((games["rid"], games["day"]))
    games = {k: v[o] for k, v in games.items()}
    # ---------- per-day, per-tier pick and ban counts
    days = games["day"]
    d0 = int(days.min())
    nd = int(days.max()) - d0 + 1
    pc = np.zeros((nd, 3, 90), np.float32)
    bc = np.zeros((nd, 3, 90), np.float32)
    gcnt = np.zeros((nd, 3), np.float32)
    di = days - d0
    np.add.at(gcnt, (di, games["tier"]), 1)
    for s in PICK_STEPS:
        np.add.at(pc, (di, games["tier"], games["H"][:, s]), 1)
    for s in BAN_STEPS:
        np.add.at(bc, (di, games["tier"], games["H"][:, s]), 1)
    np.savez(OUT, **games, day0=d0, pick_counts=pc, ban_counts=bc, game_counts=gcnt)
    print(f"wrote {OUT}: {len(games['rid']):,} games ({games['post'].sum():,} post) ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
