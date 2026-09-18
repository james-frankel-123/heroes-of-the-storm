"""
Pin a replay snapshot + matching stats file for a rerun2026 namespace.

Writes, for a game_date cutoff C (exclusive):
  training/snapshots/replay_snapshot_<C>_<N>.json
      every replay_draft_data row with game_date < C, in the exact column
      set shared.load_replay_data() returns (replay_id, game_map, skill_tier,
      draft_order, team0/1_heroes, team0/1_bans, winner, avg_mmr,
      league_tier), ordered by replay_id. The pre-2.55 exclusion list
      (pre255_exclude_ids.json) still applies at load time via
      common.load_data(), exactly as for the paper snapshot.
  training/snapshots/stats_decayed90_<C>.json
      per-game exponentially decayed hero / hero-map / pairwise stats
      (half-life 90 days, reference = C) over the same rows, in the
      frozen_stats schema StatsCache loads. Same computation as
      production_refresh.refresh.phase_stats (the deployed data-side refresh),
      restricted to games before the cutoff so nothing after C leaks into
      the features of post-cutoff study anchors.

Usage:
    set -a && source .env && set +a
    python rerun2026/make_snapshot.py --cutoff 2026-09-01
"""
import os
import sys
import json
import time
import argparse
import datetime

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP_DIR = os.path.join(TRAINING_DIR, "snapshots")
HALF_LIFE_DAYS = 90.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True, help="game_date cutoff, exclusive (YYYY-MM-DD)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cutoff = datetime.date.fromisoformat(args.cutoff)
    stats_path = os.path.join(SNAP_DIR, f"stats_decayed90_{args.cutoff}.json")

    import psycopg2
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL required (source .env)")
    conn = psycopg2.connect(url)
    cur = conn.cursor(name="snap")  # server-side cursor: stream 2M+ rows
    cur.itersize = 50_000
    cur.execute("""
        SELECT replay_id, game_map, skill_tier, draft_order,
               team0_heroes, team1_heroes, team0_bans, team1_bans, winner,
               avg_mmr, league_tier, game_date, game_version
        FROM replay_draft_data
        WHERE game_date < %s
        ORDER BY replay_id""", (cutoff,))

    ref_ord = float(cutoff.toordinal())

    def new_cell():
        return {"games": 0.0, "bans": {}, "hero": {}, "hmap": {}, "with": {}, "against": {}}

    def bump(d, key, w, win_w):
        e = d.get(key)
        if e is None:
            d[key] = [w, win_w]
        else:
            e[0] += w
            e[1] += win_w

    cells = {}
    rows = []
    n_stats = n_pre255 = 0
    max_id = 0
    t0 = time.time()
    for (rid, gmap, tier, draft_order, t0h, t1h, t0b, t1b, winner, avg_mmr,
         league_tier, gdate, gver) in cur:
        d = {"replay_id": rid, "game_map": gmap, "skill_tier": tier,
             "draft_order": draft_order, "team0_heroes": t0h, "team1_heroes": t1h,
             "team0_bans": t0b, "team1_bans": t1b, "winner": winner,
             "avg_mmr": avg_mmr, "league_tier": league_tier}
        for f in ("draft_order", "team0_heroes", "team1_heroes", "team0_bans", "team1_bans"):
            if isinstance(d[f], str):
                d[f] = json.loads(d[f])
        rows.append(d)
        max_id = max(max_id, rid)
        if not (gver or "").startswith("2.55"):
            n_pre255 += 1
            continue
        teams = [d["team0_heroes"] or [], d["team1_heroes"] or []]
        if gdate is None or len(teams[0]) != 5 or len(teams[1]) != 5:
            continue
        age = max(0.0, ref_ord - (gdate.toordinal() + gdate.hour / 24.0))
        w = 0.5 ** (age / HALF_LIFE_DAYS)
        cell = cells.get(tier)
        if cell is None:
            cell = cells[tier] = new_cell()
        cell["games"] += w
        bans = list(d["team0_bans"] or []) + list(d["team1_bans"] or [])
        for h in set(bans):
            cell["bans"][h] = cell["bans"].get(h, 0.0) + w
        for ti, heroes in enumerate(teams):
            win_w = w if winner == ti else 0.0
            for h in heroes:
                bump(cell["hero"], h, w, win_w)
                bump(cell["hmap"], (gmap, h), w, win_w)
            hs = sorted(heroes)
            for i in range(5):
                for j in range(i + 1, 5):
                    bump(cell["with"], (hs[i], hs[j]), w, win_w)
        for a in teams[0]:
            for b in teams[1]:
                key = (a, b) if a < b else (b, a)
                wins_of_a = w if winner == 0 else 0.0
                if a < b:
                    bump(cell["against"], key, w, wins_of_a)
                else:
                    bump(cell["against"], key, w, w - wins_of_a)
        n_stats += 1
        if len(rows) % 250_000 == 0:
            print(f"  {len(rows):,} rows ({time.time() - t0:.0f}s)", flush=True)
    cur.close()
    conn.close()

    snap_path = os.path.join(SNAP_DIR, f"replay_snapshot_{args.cutoff}_{len(rows)}.json")
    if os.path.exists(snap_path) and not args.force:
        sys.exit(f"exists: {snap_path} (use --force)")
    os.makedirs(SNAP_DIR, exist_ok=True)
    with open(snap_path, "w") as f:
        json.dump(rows, f)
    print(f"wrote {snap_path}: {len(rows):,} replays (max replay_id {max_id}, "
          f"{n_pre255:,} pre-2.55 rows kept for the exclusion filter) in {time.time() - t0:.0f}s")

    hero_stats, hero_map_stats, pairwise_stats = [], [], []
    for tier, cell in cells.items():
        total = cell["games"]
        if total <= 0:
            continue
        for h, (g, wn) in cell["hero"].items():
            if g < 20:
                continue
            hero_stats.append({
                "hero": h, "tier": tier, "games": round(g, 1),
                "win_rate": round(100.0 * wn / g, 3),
                "pick_rate": round(100.0 * g / total, 3),
                "ban_rate": round(100.0 * cell["bans"].get(h, 0.0) / total, 3)})
        for (m, h), (g, wn) in cell["hmap"].items():
            if g < 5:
                continue
            hero_map_stats.append({
                "hero": h, "map": m, "tier": tier, "games": round(g, 1),
                "win_rate": round(100.0 * wn / g, 3)})
        for (a, b), (g, wn) in cell["with"].items():
            if g < 10:
                continue
            wr = round(100.0 * wn / g, 3)
            for x, y in ((a, b), (b, a)):
                pairwise_stats.append({
                    "hero_a": x, "hero_b": y, "tier": tier,
                    "relationship": "with", "win_rate": wr, "games": round(g, 1)})
        for (a, b), (g, wa) in cell["against"].items():
            if g < 10:
                continue
            wr_a = round(100.0 * wa / g, 3)
            pairwise_stats.append({
                "hero_a": a, "hero_b": b, "tier": tier,
                "relationship": "against", "win_rate": wr_a, "games": round(g, 1)})
            pairwise_stats.append({
                "hero_a": b, "hero_b": a, "tier": tier,
                "relationship": "against", "win_rate": round(100.0 - wr_a, 3),
                "games": round(g, 1)})
    json.dump({
        "_meta": {"snapshot_date": args.cutoff, "patch": "2.55",
                  "kind": "decayed90", "half_life_days": HALF_LIFE_DAYS,
                  "source": f"own corpus, game_date < {args.cutoff}",
                  "games_used": n_stats, "snapshot_file": os.path.basename(snap_path),
                  "max_replay_id": max_id},
        "hero_stats": hero_stats,
        "hero_map_stats": hero_map_stats,
        "pairwise_stats": pairwise_stats,
    }, open(stats_path, "w"))
    print(f"wrote {stats_path}: {len(hero_stats)} hero, {len(hero_map_stats)} hero-map, "
          f"{len(pairwise_stats)} pairwise rows; effective decayed games/tier: "
          + ", ".join(f"{t}={c['games']:.0f}" for t, c in sorted(cells.items())))


if __name__ == "__main__":
    main()
