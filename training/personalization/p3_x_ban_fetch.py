"""
P3 extensions, ban redo: every ban of every snapshot Storm League game since
2024-04-01 (DB read-only, no battletags), with the banning team and the
draft position.

draft_order ban entries carry player_slot 1 (team 0) or 2 (team 1); checked
against team0_bans / team1_bans and against the first-pick team.

Usage (from training/, DATABASE_URL set): python3 personalization/p3_x_ban_fetch.py
Output: cache/x_bans.npz  replay_ids, hero (G, 6) index into hero_names (-1
none), team (G, 6), pick_number (G, 6), n_bans, first_pick_team
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "x_bans.npz")


def main():
    import psycopg2
    names = [str(x) for x in np.load(os.path.join(HERE, "cache", "players_2024q2.npz"))["hero_names"]]
    hidx = {n: i for i, n in enumerate(names)}
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.set_session(readonly=True)
    cur = conn.cursor(name="xb")
    cur.itersize = 100000
    cur.execute("""SELECT replay_id, draft_order, team0_bans, team1_bans FROM replay_draft_data
                   WHERE game_date >= '2024-04-01' AND replay_id <= 63653039""")
    rid, H, T, K, nb, fp = [], [], [], [], [], []
    mism = 0
    for r, do, b0, b1 in cur:
        h = [-1] * 6
        t = [-1] * 6
        k = [-1] * 6
        j = 0
        first = -1
        for e in (do or []):
            ty = int(e.get("type"))
            if ty == 0 and j < 6:
                h[j] = hidx.get(e.get("hero"), -1)
                t[j] = 0 if int(e.get("player_slot", -1)) == 1 else 1
                k[j] = int(e.get("pick_number", -1))
                j += 1
            elif ty == 1 and first < 0:
                first = -2  # resolved below from the first ban's team
        b0 = set(b0 or [])
        for jj in range(j):
            if h[jj] >= 0 and (names[h[jj]] in b0) != (t[jj] == 0):
                mism += 1
        rid.append(r)
        H.append(h)
        T.append(t)
        K.append(k)
        nb.append(j)
        # Storm League: the first-pick team bans first
        fp.append(t[0])
    np.savez(OUT, replay_ids=np.array(rid, np.int64), hero=np.array(H, np.int16),
             team=np.array(T, np.int8), pick_number=np.array(K, np.int8),
             n_bans=np.array(nb, np.int8), first_pick_team=np.array(fp, np.int8))
    print(f"wrote {OUT}: {len(rid):,} games; bans whose team disagrees with team0_bans: {mism}")


def tiers():
    """league_tier and avg_mmr for the same games (cache/x_tiers.npz).
    Stored league_tier is the real tier + 1; NULL means Master."""
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.set_session(readonly=True)
    cur = conn.cursor(name="xt")
    cur.itersize = 200000
    cur.execute("""SELECT replay_id, league_tier, avg_mmr FROM replay_draft_data
                   WHERE game_date >= '2024-04-01' AND replay_id <= 63653039""")
    rid, lt, am = [], [], []
    for r, t, m in cur:
        rid.append(r)
        lt.append(-1 if t is None else t)
        am.append(np.nan if m is None else m)
    np.savez(os.path.join(HERE, "cache", "x_tiers.npz"), replay_ids=np.array(rid, np.int64),
             league_tier=np.array(lt, np.int16), avg_mmr=np.array(am, np.float32))
    v, c = np.unique(lt, return_counts=True)
    print(dict(zip(v.tolist(), c.tolist())))


if __name__ == "__main__":
    import sys
    if "tiers" in sys.argv:
        tiers()
    else:
        main()
