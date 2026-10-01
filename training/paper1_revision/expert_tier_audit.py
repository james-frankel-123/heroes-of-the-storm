"""
Expert-study v2 item pool (data/rating-items.json, pool v5, generated
2026-09-18) vs the tier descriptions shown to raters (src/app/rate/
rate-client.tsx: Low = Bronze-Silver, Mid = Gold-Platinum, High =
Diamond-Master).

Ladder-sourced items (anchors, calibration, screener/catch real drafts) carry a
replayId: their actual rank comes from the pre-relabel backup
(backups/replay_draft_skill_tier_20260930.csv.gz; league_tier is stored one
above the rank, NULL with an MMR = Master). Machine items carry a tier label
that conditioned the agents and judges, trained on the 2026-09-01 snapshot's
old labels; their meaning is the population behind that old label, measured
here on the same snapshot.
Output: results/expert_tier_audit.json (read-only on data; no DB writes).
"""
import os
import sys
import json
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(TRAINING_DIR)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core
from paper1_revision.tier_sensitivity import real_tiers, real_tier

SHOWN = {"low": {"Bronze", "Silver"}, "mid": {"Gold", "Platinum"}, "high": {"Diamond", "Master"}}


def main():
    pool = json.load(open(os.path.join(REPO, "data", "rating-items.json")))
    R = real_tiers()
    out = {"pool_seed": pool.get("seed"), "generated_at": pool.get("generatedAt"),
           "shown_ranges": {k: sorted(v) for k, v in SHOWN.items()}, "ladder": {}, "machine": {}}
    lad = Counter()
    lad_items = Counter()
    lad_match = Counter()
    mach = Counter()
    for it in pool["items"]:
        prov = it.get("provenance", {})
        rids = []
        for k in ("replayId", "realReplayId"):
            if prov.get(k):
                rids.append(prov[k])
        if prov.get("real") and isinstance(prov["real"], dict) and prov["real"].get("replayId"):
            rids.append(prov["real"]["replayId"])
        if rids:
            for rid in rids:
                r = R.get(int(rid))
                rt = real_tier(r[1], r[2]) if r else "missing"
                lad[(it["block"], it["tier"], rt)] += 1
                lad_items[it["block"]] += 1
                lad_match[it["block"]] += rt in SHOWN.get(it["tier"], set())
        else:
            mach[(it["block"], it["tier"])] += 1
    for (b, t, rt), n in sorted(lad.items()):
        out["ladder"].setdefault(b, {}).setdefault(t, {})[rt] = n
    out["ladder_mismatch"] = {b: {"items": lad_items[b], "outside_shown_range": lad_items[b] - lad_match[b],
                                  "share_outside": (lad_items[b] - lad_match[b]) / lad_items[b]}
                              for b in lad_items}
    tot = sum(lad_items.values())
    out["ladder_mismatch"]["all_ladder"] = {"items": tot, "outside_shown_range": tot - sum(lad_match.values()),
                                            "share_outside": (tot - sum(lad_match.values())) / tot}
    for (b, t), n in sorted(mach.items()):
        out["machine"].setdefault(b, {})[t] = n
    # population behind each OLD label in the September snapshot used by the
    # tournament agents (by the same backup); fraction inside the shown range
    snap = os.path.join(TRAINING_DIR, "snapshots", "replay_snapshot_2026-09-01_2303238.json")
    pop = Counter()
    import re
    pat = re.compile(r'"replay_id": (\d+), "game_map": "[^"]*", "skill_tier": "(\w+)"')
    with open(snap) as f:
        while True:
            chunk = f.read(1 << 26)
            if not chunk:
                break
            for rid, st in pat.findall(chunk):
                r = R.get(int(rid))
                pop[(st, real_tier(r[1], r[2]) if r else "missing")] += 1
    lab = {}
    for (st, rt), n in pop.items():
        lab.setdefault(st, Counter())[rt] += n
    out["sept_snapshot_label_population"] = {st: dict(c) for st, c in lab.items()}
    out["machine_label_share_inside_shown_range"] = {
        st: sum(n for rt, n in c.items() if rt in SHOWN.get(st, set())) / sum(c.values())
        for st, c in lab.items() if st in SHOWN}
    json.dump(out, open(os.path.join(core.RESULTS, "expert_tier_audit.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("ladder_mismatch", "machine",
                                          "machine_label_share_inside_shown_range")}, indent=1))
    print(json.dumps(out["ladder"], indent=1))
    print(json.dumps(out["sept_snapshot_label_population"], indent=1))


if __name__ == "__main__":
    main()
