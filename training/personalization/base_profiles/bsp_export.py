"""
Base scoreboard profiles (BSP), data step: one read-only export from the
LOCAL research store (DATABASE_URL_RESEARCH; never Neon), written as compact
numpy arrays. No battletag or blizz_id is exported: profiles are hero-level
aggregates, so player identity is never needed.

Scope: Storm League drafts dated before 2026-09-28, no 2.57+ build (the P3
data rule), and never a replay in training/replays/cohorts/sealed_ids.txt
(excluded in SQL, so those rows are never read).

Steps (main box, nice 19, 2 cores):
  copy     COPY games and scoreboard rows to csv.gz (streamed, no parsing)
  arrays   csv.gz -> npz (games.npz; sb.npz with one float32 column per metric)

Usage (from repo root):
  set -a; . ./.env; set +a
  nice -n 19 taskset -c 48-49 /home/linuxbrew/.linuxbrew/bin/python3 \
      training/personalization/base_profiles/bsp_export.py copy --out DIR
  nice -n 19 taskset -c 48-49 /usr/bin/python3 \
      training/personalization/base_profiles/bsp_export.py arrays --out DIR
"""
import argparse
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SEALED = os.path.join(REPO, "training", "replays", "cohorts", "sealed_ids.txt")

WHERE = ("d.game_date < '2026-09-28' AND d.game_version NOT LIKE '2.57%%' "
         "AND d.game_version NOT LIKE '2.58%%' AND d.game_version NOT LIKE '3.%%' "
         "AND NOT (d.replay_id = ANY(%(sealed)s))")

METRICS = ["hero_damage", "siege_damage", "structure_damage", "minion_damage", "creep_damage",
           "summon_damage", "healing", "self_healing", "protection_allies", "damage_taken",
           "deaths", "time_spent_dead", "kills", "assists", "takedowns",
           "experience_contribution", "merc_camp_captures", "watch_tower_captures",
           "time_cc_enemy_heroes", "stunning_enemies", "rooting_enemies", "silencing_enemies",
           "regen_globes", "escapes", "teamfight_hero_damage", "teamfight_damage_taken",
           "teamfight_healing", "outnumbered_deaths", "clutch_heals", "highest_kill_streak"]

QUERIES = {
    "games": f"""
        SELECT d.replay_id, d.game_version, EXTRACT(EPOCH FROM d.game_date)::bigint AS end_ts,
               d.game_length, d.game_map, d.skill_tier, d.league_tier, d.avg_mmr,
               d.team0_heroes, d.team1_heroes, d.winner
        FROM replay_draft_data d WHERE {WHERE}""",
    "scoreboard": f"""
        SELECT rp.replay_id, rp.hero, rp.team, rp.winner::int AS winner,
               {", ".join(f"(rp.scoreboard->>'{k}')::real AS {k}" for k in METRICS)}
        FROM replay_players rp JOIN replay_draft_data d USING (replay_id)
        WHERE {WHERE}""",
}


def sealed_ids():
    return [int(x) for x in open(SEALED).read().split()]


def cmd_copy(out):
    import psycopg2
    url = os.environ.get("DATABASE_URL_RESEARCH")
    if not url:
        raise SystemExit("DATABASE_URL_RESEARCH is not set")
    if "neon" in url.lower():
        raise SystemExit("refusing a Neon URL")
    os.makedirs(out, exist_ok=True)
    conn = psycopg2.connect(url)
    conn.set_session(readonly=True)
    sealed = sealed_ids()
    meta = {}
    for what, q in QUERIES.items():
        path = os.path.join(out, f"{what}.csv.gz")
        t0 = time.time()
        with conn.cursor() as c:
            c.execute("SET statement_timeout = 0")
            sql = c.mogrify(q, {"sealed": sealed}).decode()
            with gzip.open(path + ".tmp", "wb", compresslevel=3) as f:
                c.copy_expert(f"COPY ({sql}) TO STDOUT WITH (FORMAT csv, HEADER true)", f)
        os.replace(path + ".tmp", path)
        meta[what] = {"seconds": round(time.time() - t0), "query": " ".join(q.split()),
                      "n_sealed_excluded_ids": len(sealed)}
        print(what, meta[what]["seconds"], "s", flush=True)
    json.dump(meta, open(os.path.join(out, "export_meta.json"), "w"), indent=1)


def cmd_arrays(out):
    import numpy as np
    import pandas as pd
    sys.path.insert(0, os.path.join(REPO, "training"))
    from shared import HEROES, MAPS
    hidx = {h: i for i, h in enumerate(HEROES)}
    midx = {m: i for i, m in enumerate(MAPS)}
    tiers = {"low": 0, "mid": 1, "high": 2}
    sealed = set(sealed_ids())

    g = pd.read_csv(os.path.join(out, "games.csv.gz"))
    assert not g["replay_id"].isin(sealed).any()
    def team(s):
        h = json.loads(s) if isinstance(s, str) and s.startswith("[") else str(s).strip("{}").split(",")
        h = [x.strip().strip('"') for x in h]
        return [hidx.get(x, -1) for x in h] if len(h) == 5 else [-1] * 5
    t0 = np.array([team(s) for s in g["team0_heroes"]], np.int16)
    t1 = np.array([team(s) for s in g["team1_heroes"]], np.int16)
    gm = g["game_map"].map(midx).fillna(-1).to_numpy(np.int16)
    tr = g["skill_tier"].map(tiers).fillna(-1).to_numpy(np.int8)
    np.savez(os.path.join(out, "games.npz"),
             replay_id=g["replay_id"].to_numpy(np.int64), end_ts=g["end_ts"].to_numpy(np.int64),
             game_length=g["game_length"].fillna(0).to_numpy(np.int32), map=gm, tier=tr,
             version=g["game_version"].astype(str).to_numpy(), t0=t0, t1=t1,
             winner=g["winner"].fillna(-1).to_numpy(np.int8),
             heroes=np.array(HEROES), maps=np.array(MAPS))
    print(f"games {len(g):,}; valid teams {((t0 >= 0).all(1) & (t1 >= 0).all(1)).mean():.4f}", flush=True)
    del g

    cols = {"replay_id": [], "hero": [], "team": [], "winner": []}
    cols.update({m: [] for m in METRICS})
    for ch in pd.read_csv(os.path.join(out, "scoreboard.csv.gz"), chunksize=2_000_000,
                          dtype={m: np.float32 for m in METRICS}):
        cols["replay_id"].append(ch["replay_id"].to_numpy(np.int64))
        cols["hero"].append(ch["hero"].map(hidx).fillna(-1).to_numpy(np.int16))
        cols["team"].append(ch["team"].fillna(-1).to_numpy(np.int8))
        cols["winner"].append(ch["winner"].fillna(-1).to_numpy(np.int8))
        for m in METRICS:
            cols[m].append(ch[m].to_numpy(np.float32))
        print(f"  sb chunk {sum(len(x) for x in cols['replay_id']):,}", flush=True)
    sb = {k: np.concatenate(v) for k, v in cols.items()}
    assert not np.isin(sb["replay_id"], np.fromiter(sealed, np.int64)).any()
    np.savez(os.path.join(out, "sb.npz"), metrics=np.array(METRICS), **sb)
    print(f"scoreboard rows {len(sb['replay_id']):,}; nan share per metric:", flush=True)
    for m in METRICS:
        print(f"  {m:26s} {np.isnan(sb[m]).mean():.3f}  nonzero {np.nanmean(sb[m] > 0):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["copy", "arrays"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"copy": cmd_copy, "arrays": cmd_arrays}[a.cmd](a.out)


if __name__ == "__main__":
    main()
