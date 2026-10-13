"""Read-only snapshot of the research DB into the broker's private store (FRAMING 4, 11.1).

  python -m training.scilearn.snapshot build [--as-of 2026-10-13]
  python -m training.scilearn.snapshot verify <snap_id>

Writes PRIVATE/snapshots/<snap_id>/:
  games.npz     draft-time covariates only (no winner, length or parse-time MMR)
  labels.npz    y0 (team 0 won), game_length, avg_mmr: broker and evaluator only
  players.npz   per-slot player key (-1 for private accounts), party, hero
  quarantine.json  counts only: sealed rows are never written anywhere
  pool.npz      the interesting pool (covariates only), apart from every view
  manifest.json hashes, vocabularies, tranche counts, rules

Sealed replay IDs (cohort A) are dropped at read time. Private accounts (player_privacy
state = 'private') have their player key removed, per the Heroes Profile terms.
Uses DATABASE_URL_RESEARCH only.
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import sys

import numpy as np
import psycopg2

from training.scilearn import config as C
from training.scilearn import tranches as T

ROLES = ["Tank", "Bruiser", "Healer", "Support", "Ranged Assassin", "Melee Assassin"]
TIERS = ["low", "mid", "high", "unknown"]


def research_dsn():
    for line in open(C.REPO / ".env"):
        if line.startswith("DATABASE_URL_RESEARCH="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("DATABASE_URL_RESEARCH missing from .env")


def hero_roles():
    import re
    txt = (C.REPO / "src/lib/data/hero-roles.ts").read_text()
    out = {}
    for m in re.finditer(r"""^\s*['"](.+?)['"]\s*:\s*'([A-Za-z ]+)'""", txt, re.M):
        out[m.group(1).replace("\\'", "'")] = m.group(2)
    return out


def build_key(v):
    return tuple(int(x) for x in v.split("."))


def copy_rows(cur, sql):
    buf = io.StringIO()
    cur.copy_expert(f"COPY ({sql}) TO STDOUT WITH CSV", buf)
    buf.seek(0)
    return csv.reader(buf)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(as_of):
    sealed = {int(x) for x in open(C.SEALED_IDS_FILE).read().split()}
    cohort_b = T.cohort_b_ids()
    keys = T.load_keys()
    conn = psycopg2.connect(research_dsn())
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cutoff = f"timestamp '{as_of}'"

    rows = list(copy_rows(cur, f"""
        SELECT replay_id, region, game_map, extract(epoch from game_date)::bigint, game_version,
               coalesce(league_tier, -1), skill_tier, team0_heroes::text, team1_heroes::text,
               team0_bans::text, team1_bans::text, draft_order::text, winner, game_length, avg_mmr
        FROM replay_draft_data WHERE game_date < {cutoff} ORDER BY replay_id"""))
    n_db = len(rows)
    heroes = sorted({h for r in rows for h in json.loads(r[7]) + json.loads(r[8])})
    maps = sorted({r[2] for r in rows})
    builds = sorted({r[4] for r in rows}, key=build_key)
    hidx = {h: i for i, h in enumerate(heroes)}
    midx = {m: i for i, m in enumerate(maps)}
    bidx = {b: i for i, b in enumerate(builds)}
    roles = hero_roles()
    missing = [h for h in heroes if h not in roles]
    if missing:
        raise SystemExit(f"heroes without a role: {missing}")

    keep, pool, n_sealed = [], [], 0
    for r in rows:
        rid = int(r[0])
        if rid in sealed:
            n_sealed += 1
            continue
        tr = T.assign(rid, r[4], int(r[3]), sealed=sealed, cohort_b=cohort_b, keys=keys)
        (pool if tr == "pool" else keep).append((r, tr))
    del rows

    def arrays(items):
        n = len(items)
        a = {
            "replay_id": np.zeros(n, np.int64), "game_ts": np.zeros(n, np.int64),
            "first_seen_ts": np.zeros(n, np.int64), "build": np.zeros(n, np.int16),
            "map": np.zeros(n, np.int8), "tier": np.zeros(n, np.int8),
            "league_tier": np.zeros(n, np.int8), "region": np.zeros(n, np.int8),
            "heroes": np.full((n, 10), -1, np.int16), "bans": np.full((n, 6), -1, np.int16),
            "pick_seq": np.full((n, 10), -1, np.int16), "pick_team": np.full((n, 10), -1, np.int8),
            "tranche": np.zeros(n, np.int8),
        }
        lab = {"replay_id": a["replay_id"], "y0": np.zeros(n, np.int8),
               "game_length": np.zeros(n, np.int32), "avg_mmr": np.zeros(n, np.float32)}
        for i, (r, tr) in enumerate(items):
            a["replay_id"][i] = int(r[0])
            a["game_ts"][i] = int(r[3])
            # Retrospective first-seen convention (P3): game date plus one day.
            a["first_seen_ts"][i] = int(r[3]) + 86400
            a["build"][i] = bidx[r[4]]
            a["map"][i] = midx[r[2]]
            a["tier"][i] = TIERS.index(r[6]) if r[6] in TIERS else 3
            a["league_tier"][i] = int(r[5])
            a["region"][i] = int(r[1] or -1)
            t0, t1 = json.loads(r[7]), json.loads(r[8])
            a["heroes"][i, :5] = [hidx[h] for h in t0]
            a["heroes"][i, 5:] = [hidx[h] for h in t1]
            bans = [b for b in (json.loads(r[9] or "[]") + json.loads(r[10] or "[]")) if b in hidx][:6]
            a["bans"][i, :len(bans)] = [hidx[b] for b in bans]
            picks = [d for d in json.loads(r[11] or "[]") if str(d.get("type")) == "1" and d.get("hero") in hidx]
            picks.sort(key=lambda d: d.get("pick_number", 0))
            for j, d in enumerate(picks[:10]):
                a["pick_seq"][i, j] = hidx[d["hero"]]
                a["pick_team"][i, j] = 0 if d["hero"] in t0 else 1
            a["tranche"][i] = T.CODES[tr]
            lab["y0"][i] = 1 if int(r[12]) == 0 else 0
            lab["game_length"][i] = int(r[13] or 0)
            lab["avg_mmr"][i] = float(r[14]) if r[14] else np.nan
        return a, lab

    games, labels = arrays(keep)
    pool_games, _ = arrays(pool)

    # Players: slot order follows the team hero lists; private accounts lose their key.
    private = set()
    for bt, reg in copy_rows(cur, "SELECT battletag, region FROM player_privacy WHERE state = 'private'"):
        private.add((bt, int(reg)))
    rid_pos = {int(r): i for i, r in enumerate(games["replay_id"])}
    n = len(games["replay_id"])
    pkey = np.full((n, 10), -1, np.int64)
    party = np.zeros((n, 10), np.int64)
    n_priv = 0
    for rid, blz, bt, reg, hero, team, prt in copy_rows(cur, f"""
            SELECT p.replay_id, p.blizz_id, p.battletag, coalesce(p.region, -1), p.hero, p.team, coalesce(p.party, 0)
            FROM replay_players p JOIN replay_draft_data d USING (replay_id)
            WHERE d.game_date < {cutoff}"""):
        i = rid_pos.get(int(rid))
        if i is None or hero not in hidx:
            continue
        h = hidx[hero]
        row = games["heroes"][i]
        base = 0 if int(team) == 0 else 5
        hit = np.nonzero(row[base:base + 5] == h)[0]
        if len(hit) == 0:
            continue
        s = base + int(hit[0])
        if (bt, int(reg)) in private:
            n_priv += 1
            pkey[i, s] = -1
        else:
            pkey[i, s] = int(blz) if blz else -1
        party[i, s] = int(prt)
    conn.close()

    snap_id = f"snap_{as_of.replace('-', '')}"
    out = C.SNAPSHOTS / snap_id
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "games.npz", **games)
    np.savez(out / "labels.npz", **labels)
    np.savez(out / "players.npz", replay_id=games["replay_id"], player_key=pkey, party=party)
    np.savez(out / "pool.npz", **pool_games)
    counts = {tr: int((games["tranche"] == T.CODES[tr]).sum()) for tr in T.CODES}
    counts["pool"] = len(pool)
    counts["R_sealed_rows_dropped"] = n_sealed
    (out / "quarantine.json").write_text(json.dumps(
        {"sealed_rows_dropped": n_sealed, "pool_rows_set_apart": len(pool),
         "rule": "sealed IDs never written; pool rows only in pool.npz, excluded from every view"}, indent=1))
    manifest = {
        "snap_id": snap_id, "as_of": as_of, "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "DATABASE_URL_RESEARCH (local research role, read-only session)",
        "db_rows_read": n_db, "heroes": heroes, "maps": maps, "builds": builds, "tiers": TIERS,
        "roles": ROLES, "hero_roles": [ROLES.index(roles[h]) for h in heroes],
        "tranche_codes": T.CODES, "tranche_counts": counts,
        "private_player_slots_unkeyed": n_priv,
        "first_seen": "retrospective: game_date + 1 day for every row in this snapshot",
        "cohort_b_ids": len(cohort_b),
        "files": {f: sha_file(out / f) for f in ["games.npz", "labels.npz", "players.npz", "pool.npz"]},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for f in out.iterdir():
        os.chmod(f, 0o400)
    print(json.dumps({"snap_id": snap_id, "tranche_counts": counts, "private_slots": n_priv}, indent=1))
    return snap_id


def verify(snap_id):
    d = C.SNAPSHOTS / snap_id
    m = json.loads((d / "manifest.json").read_text())
    ok = all(sha_file(d / f) == h for f, h in m["files"].items())
    sealed = {int(x) for x in open(C.SEALED_IDS_FILE).read().split()}
    g = np.load(d / "games.npz")
    leak = int(np.isin(g["replay_id"], list(sealed)).sum())
    p = np.load(d / "pool.npz")
    leak += int(np.isin(p["replay_id"], list(sealed)).sum())
    print(json.dumps({"hashes_ok": ok, "sealed_rows_present": leak}))
    return ok and leak == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--as-of", default=C.SNAPSHOT_AS_OF)
    v = sp.add_parser("verify")
    v.add_argument("snap_id")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.as_of)
    else:
        sys.exit(0 if verify(a.snap_id) else 1)
