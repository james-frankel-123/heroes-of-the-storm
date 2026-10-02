"""
Tier-banner audit for the expert-study v6 pool (oct2026), read-only on the DB.

Shown ranges (src/app/rate/rate-client.tsx): Low = Bronze-Silver,
Mid = Gold-Platinum, High = Diamond-Master.

  * Ladder items (anchors, calibration, screener/catch real drafts): the
    game's actual rank from the live replay_draft_data.league_tier (stored one
    above the rank; NULL with an MMR = Master), compared with the item's tier
    banner. Also checks that the item tier equals the DB's site-scheme
    skill_tier.
  * Machine items: their tier label conditioned agents and judges trained on
    the site-tier snapshot (replay_snapshot_2026-09-01_sitetiers_*). Reported:
    the rank population behind each site label in the training window
    (2.55 builds, game_date < 2026-09-01), share inside the shown range.

  python3 paper1_revision/expert_tier_audit_v6.py [pool.json]
Output: results/expert_v6/tier_audit.json
"""
import os
import sys
import json
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SHOWN = {"low": {"Bronze", "Silver"}, "mid": {"Gold", "Platinum"}, "high": {"Diamond", "Master"}}
NAMES = {1: "Wood", 2: "Bronze", 3: "Silver", 4: "Gold", 5: "Platinum", 6: "Diamond"}
OUT = os.environ.get("TIER_AUDIT_OUT", os.path.join(HERE, "results", "expert_v6", "tier_audit.json"))


def rank(lt, mmr):
    if lt is None:
        return "Master" if mmr is not None else "unknown"
    return NAMES.get(lt, f"id{lt}")


def main():
    import psycopg2
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "expert_v6",
                                                             "rating-items-v6.json")
    pool = json.load(open(path))
    url = os.environ.get("DATABASE_URL") or [l.split("=", 1)[1].strip().strip('"')
                                              for l in open(os.path.join(REPO, ".env"))
                                              if l.startswith("DATABASE_URL=")][0]
    conn = psycopg2.connect(url)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    lad = [it for it in pool["items"] if it["provenance"].get("replayId")]
    rids = [int(it["provenance"]["replayId"]) for it in lad]
    cur.execute("SELECT replay_id, league_tier, avg_mmr, skill_tier FROM replay_draft_data "
                "WHERE replay_id = ANY(%s)", (rids,))
    db = {r[0]: r[1:] for r in cur.fetchall()}
    by = Counter()
    outside, label_mismatch, missing = Counter(), Counter(), 0
    for it in lad:
        r = db.get(int(it["provenance"]["replayId"]))
        if r is None:
            missing += 1
            continue
        rk = rank(r[0], r[1])
        by[(it["block"], it["tier"], rk)] += 1
        outside[it["block"]] += rk not in SHOWN[it["tier"]]
        label_mismatch[it["block"]] += r[2] != it["tier"]
    nb = Counter(it["block"] for it in lad)
    out = {"pool": os.path.relpath(path, REPO), "pool_seed": pool.get("seed"),
           "shown_ranges": {k: sorted(v) for k, v in SHOWN.items()},
           "ladder_items": len(lad), "ladder_missing_in_db": missing,
           "ladder_outside_shown_range": {b: {"items": nb[b], "outside": outside[b]} for b in nb},
           "ladder_outside_total": sum(outside.values()),
           "ladder_item_tier_ne_db_skill_tier": sum(label_mismatch.values()),
           "ladder_rank_counts": {}}
    for (b, t, rk), n in sorted(by.items()):
        out["ladder_rank_counts"].setdefault(b, {}).setdefault(t, {})[rk] = n
    cur.execute("""SELECT skill_tier, league_tier, (avg_mmr IS NOT NULL), count(*)
                   FROM replay_draft_data
                   WHERE game_date < '2026-09-01' AND game_version LIKE '2.55.%%'
                   GROUP BY 1, 2, 3""")
    pop = {}
    for st, lt, has_mmr, n in cur.fetchall():
        pop.setdefault(st, Counter())[rank(lt, 1 if has_mmr else None)] += n
    out["training_window_label_population"] = {st: dict(c) for st, c in pop.items()}
    out["machine_label_share_inside_shown_range"] = {
        st: sum(n for rk, n in c.items() if rk in SHOWN[st]) / sum(c.values())
        for st, c in pop.items() if st in SHOWN}
    out["machine_items_by_block_tier"] = dict(Counter(
        f"{it['block']}/{it['tier']}" for it in pool["items"] if not it["provenance"].get("replayId")))
    conn.close()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
