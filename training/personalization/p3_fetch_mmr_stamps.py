"""
P3 at-game MMR, source 1: Heroes Profile's own rating chain as already stored
in replay_players (no API calls).

What the stored numbers are (checked 2026-09-30 on live rows): for every
player-game, player_mmr / hero_mmr / role_mmr are HP's ratings AFTER that
game, in HP's processing (parse) order, and raw_extras.mmr_date_parsed is
the time HP processed it, in US Eastern local time (game_date is UTC, the
game END). Consecutive rows in parse order chain: post_k = post_{k-1} +
change_k. So a row's MMR includes its own result, and a row is causal
information only from its parse time on.

This script pulls every Storm League player row with its parse stamp, so
p3_mmr_at_game.py can take, for each game, the latest stamp parsed strictly
before the game started.

Usage (from training/, needs psycopg2 and DATABASE_URL):
  nice -n 19 taskset -c 48-63 python3 personalization/p3_fetch_mmr_stamps.py
Output: personalization/cache/mmr_stamps_db.npz
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache", "mmr_stamps_db.npz")
HERO_NAMES = os.path.join(HERE, "cache", "players_2024q2.npz")
# Upper bound on game date: rows after the snapshot end can only matter for
# the chain-order check of snapshot games parsed late, so keep a margin.
END = "2026-07-15"


def main():
    import psycopg2
    names = [str(h) for h in np.load(HERO_NAMES)["hero_names"]]
    hidx = {h: i for i, h in enumerate(names)}
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    with conn.cursor() as c:
        c.execute("SET statement_timeout='60min'")
    cur = conn.cursor(name="p3mmrstamps")
    cur.itersize = 500000
    # rp.region can be NULL on early rows; the draft row always has it.
    cur.execute(f"""
        SELECT rp.replay_id, rp.blizz_id, d.region, rp.hero,
               rp.player_mmr, rp.hero_mmr, rp.role_mmr,
               EXTRACT(EPOCH FROM ((rp.raw_extras->>'mmr_date_parsed')::timestamp
                                   AT TIME ZONE 'America/New_York'))::bigint,
               EXTRACT(EPOCH FROM d.game_date)::bigint, d.game_length, rp.winner
        FROM replay_players rp JOIN replay_draft_data d USING (replay_id)
        WHERE d.game_date < '{END}'""")
    cols = {k: [] for k in ("rid", "bid", "reg", "hero", "pm", "hm", "rm", "pts", "ets", "gl", "win")}
    unk = set()
    n = 0
    for r in cur:
        cols["rid"].append(r[0])
        cols["bid"].append(r[1])
        cols["reg"].append(r[2])
        h = hidx.get(r[3], -1)
        if h < 0:
            unk.add(r[3])
        cols["hero"].append(h)
        cols["pm"].append(np.nan if r[4] is None else r[4])
        cols["hm"].append(np.nan if r[5] is None else r[5])
        cols["rm"].append(np.nan if r[6] is None else r[6])
        cols["pts"].append(-1 if r[7] is None else r[7])
        cols["ets"].append(r[8])
        cols["gl"].append(-1 if r[9] is None else r[9])
        cols["win"].append(bool(r[10]))
        n += 1
        if n % 2_000_000 == 0:
            print(f"  {n:,} rows", flush=True)
    np.savez(OUT,
             replay_ids=np.array(cols["rid"], np.int64),
             blizz_ids=np.array(cols["bid"], np.int64),
             region=np.array(cols["reg"], np.int16),
             hero=np.array(cols["hero"], np.int16),
             player_mmr=np.array(cols["pm"], np.float32),
             hero_mmr=np.array(cols["hm"], np.float32),
             role_mmr=np.array(cols["rm"], np.float32),
             parse_ts=np.array(cols["pts"], np.int64),
             end_ts=np.array(cols["ets"], np.int64),
             game_length=np.array(cols["gl"], np.int32),
             winner=np.array(cols["win"], np.bool_),
             hero_names=np.array(names))
    print(f"wrote {OUT}: {n:,} SL player rows; unknown heroes: {sorted(unk)[:10]}")


if __name__ == "__main__":
    main()
