"""
Check parsed timelines against our DB rows for the same replays (local research role only):
winner, heroes per team, per-player hero and talents (matched on toon_id = blizz_id),
game length, and map. Prints a summary and writes <root>/timelines/validation.json.

  python3 training/replays/validate_timelines.py [--root /home/max/hots-replays]
"""
import argparse
import gzip
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

import psycopg2

LEVELS = [1, 4, 7, 10, 13, 16, 20]


def norm(x):
    return re.sub(r"[^a-z0-9]", "", (x or "").lower())


def talent_match(tl_t, db_name):
    """'ref' if the reference name matches, 'fuzzy' if the DB name is inside the internal id,
    'miss' otherwise."""
    if tl_t.get("name") and norm(tl_t["name"]) == norm(db_name):
        return "ref"
    nd, ni = norm(db_name), norm(tl_t.get("id"))
    if nd and nd in ni:
        return "fuzzy"
    return "miss"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/max/hots-replays")
    a = ap.parse_args()
    url = os.environ.get("DATABASE_URL_RESEARCH", "")
    if not url or "neon" in url:
        sys.exit("DATABASE_URL_RESEARCH must be the local store")

    files = {}
    for dp, _, fs in os.walk(os.path.join(a.root, "timelines")):
        for f in fs:
            if f.endswith(".json.gz"):
                files[int(f.split(".")[0])] = os.path.join(dp, f)
    ids = sorted(files)
    conn = psycopg2.connect(url)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    games, players = {}, defaultdict(list)
    for i in range(0, len(ids), 5000):
        chunk = ids[i:i + 5000]
        cur.execute("""SELECT replay_id, winner, team0_heroes, team1_heroes, game_length, game_map
                       FROM replay_draft_data WHERE replay_id = ANY(%s)""", (chunk,))
        for rid, w, t0, t1, gl, mp in cur.fetchall():
            games[rid] = {"winner": w, "t0": t0, "t1": t1, "len": gl, "map": mp}
        cur.execute("""SELECT replay_id, blizz_id, hero, team, winner, talents
                       FROM replay_players WHERE replay_id = ANY(%s)""", (chunk,))
        for rid, bid, hero, team, win, tal in cur.fetchall():
            players[rid].append({"blizz_id": bid, "hero": hero, "team": team, "win": win, "talents": tal})
    conn.close()

    c = Counter()
    len_diff, len_diff_gates = [], []
    talent_kinds = Counter()
    talent_miss_examples = Counter()
    hero_miss = Counter()
    map_pairs = defaultdict(Counter)
    for rid in ids:
        with gzip.open(files[rid], "rt") as f:
            tl = json.load(f)
        c["timelines"] += 1
        g = games.get(rid)
        if not g:
            c["not_in_db"] += 1
            continue
        c["in_db"] += 1
        c["winner_agree"] += tl["winner_team"] == g["winner"]
        teams = {0: set(), 1: set()}
        for p in tl["players"]:
            teams[p["team"]].add(norm(p["hero"]))
        db_teams = {0: {norm(h) for h in (g["t0"] or [])}, 1: {norm(h) for h in (g["t1"] or [])}}
        ok = teams == db_teams
        c["heroes_agree"] += ok
        if not ok:
            for t in (0, 1):
                for h in teams[t] ^ db_teams[t]:
                    hero_miss[h] += 1
        if g["len"]:
            len_diff.append(tl["length_s"] - g["len"])
            if tl.get("gates_open_s") is not None:
                len_diff_gates.append(tl["length_s"] - tl["gates_open_s"] - g["len"])
        map_pairs[g["map"]][tl.get("map_id")] += 1
        dbp = {p["blizz_id"]: p for p in players.get(rid, [])}
        if dbp:
            c["games_with_db_players"] += 1
        for p in tl["players"]:
            d = dbp.get(p["toon_id"])
            if d is None:
                c["player_unmatched"] += 1 if dbp else 0
                continue
            c["player_matched"] += 1
            c["player_hero_agree"] += norm(d["hero"]) == norm(p["hero"])
            c["player_team_agree"] += d["team"] == p["team"]
            c["player_win_agree"] += bool(d["win"]) == p["win"]
            dt = d["talents"] or {}
            c["talents_db_count"] += len(dt)
            c["talents_tl_count"] += len(p["talents"])
            c["talent_count_agree"] += len(dt) == len(p["talents"])
            for t in p["talents"]:
                lvl = str(LEVELS[t["tier"] - 1]) if t["tier"] <= 7 else None
                if lvl in dt:
                    k = talent_match(t, dt[lvl])
                    talent_kinds[k] += 1
                    if k == "miss":
                        talent_miss_examples[f"{t['id']} | {t.get('name')} vs {dt[lvl]}"] += 1

    def pct(n, d):
        return round(100.0 * n / d, 2) if d else None

    abs_len = [abs(x) for x in len_diff]
    abs_gate = [abs(x) for x in len_diff_gates]
    tk = sum(talent_kinds.values())
    out = {
        "timelines": c["timelines"], "in_db": c["in_db"], "not_in_db": c["not_in_db"],
        "winner_agree_pct": pct(c["winner_agree"], c["in_db"]),
        "team_heroes_agree_pct": pct(c["heroes_agree"], c["in_db"]),
        "games_with_db_players": c["games_with_db_players"],
        "players_matched_on_toon_id": c["player_matched"],
        "players_unmatched": c["player_unmatched"],
        "player_hero_agree_pct": pct(c["player_hero_agree"], c["player_matched"]),
        "player_team_agree_pct": pct(c["player_team_agree"], c["player_matched"]),
        "player_win_agree_pct": pct(c["player_win_agree"], c["player_matched"]),
        "talent_count_agree_pct": pct(c["talent_count_agree"], c["player_matched"]),
        "talents_compared": tk,
        "talent_agree_pct": pct(talent_kinds["ref"] + talent_kinds["fuzzy"], tk),
        "talent_match_kinds": dict(talent_kinds),
        "talent_miss_top": talent_miss_examples.most_common(15),
        "hero_mismatch_top": hero_miss.most_common(10),
        "length_vs_db_total": {
            "median_abs_s": statistics.median(abs_len) if abs_len else None,
            "within_2s_pct": pct(sum(x <= 2 for x in abs_len), len(abs_len)),
            "median_signed_s": statistics.median(len_diff) if len_diff else None},
        "length_vs_db_from_gates": {
            "median_abs_s": statistics.median(abs_gate) if abs_gate else None,
            "within_2s_pct": pct(sum(x <= 2 for x in abs_gate), len(abs_gate))},
        "map_id_consistency": {m: dict(v) for m, v in map_pairs.items()},
    }
    with open(os.path.join(a.root, "timelines", "validation.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
