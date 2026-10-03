"""
Patch-2.57 sensitivity counts for an expert-study pool (prereg v3 §4).

Source: Blizzard, "Heroes of the Storm live patch notes, September 28, 2026",
https://news.blizzard.com/en-us/article/24303007/heroes-of-the-storm-live-patch-notes-september-28-2026
(new hero Xal'atath, who appears in no pool item).

  (a) primary: drop items with any of the 7 heroes whose balance changed in 2.57
  (b) broad:   (a) plus the 4 heroes with gameplay-relevant bug fixes, plus
               items on the 2 maps that changed

  python3 paper1_revision/patch257_counts.py <pool.json> [out.json]
"""
import sys
import json

CHANGED = ["Abathur", "Alexstrasza", "Garrosh", "Mal'Ganis", "Qhira", "Whitemane", "Yrel"]
BUGFIX = ["Ragnaros", "Malfurion", "Anduin", "Xul"]
MAPS = ["Garden of Terror", "Volskaya Foundry"]
BLOCKS = ["screener", "calibration", "pairs", "anchors", "catch"]
STRATA = ["constrained_mcts_vs_mcts", "mcts_vs_enriched", "vs_anchored", "other_pairs"]
TIERS = ["low", "mid", "high"]


def heroes(it):
    return set(it["teams"]["team0"]) | set(it["teams"]["team1"])


def main():
    pool = json.load(open(sys.argv[1]))
    items = pool["items"]
    allh = set().union(*(heroes(i) for i in items))
    missing = [h for h in CHANGED + BUGFIX if h not in allh]
    if missing:
        raise SystemExit(f"hero names not found in pool: {missing}")
    variants = {
        "a_changed_heroes": lambda i: bool(heroes(i) & set(CHANGED)),
        "b_broad": lambda i: bool(heroes(i) & set(CHANGED + BUGFIX)) or i["map"] in MAPS,
    }
    pairs = [i for i in items if i["block"] == "pairs"]
    cons = lambda i: i["provenance"]["wpTeam0Sym"]["consensus"]
    out = {"pool_seed": pool.get("seed"), "changed_heroes": CHANGED, "bugfix_heroes": BUGFIX,
           "changed_maps": MAPS, "xalatath_items": sum("Xal'atath" in heroes(i) for i in items),
           "hero_appearances": {h: sum(h in heroes(i) for i in items) for h in CHANGED + BUGFIX},
           "variants": {}}
    for name, hit in variants.items():
        v = {"blocks": {}, "strata": {}, "effective": {}}
        for b in BLOCKS:
            xs = [i for i in items if i["block"] == b]
            v["blocks"][b] = {"items": len(xs), "affected": sum(map(hit, xs)),
                              "affected_by_tier": {t: sum(hit(i) for i in xs if i["tier"] == t) for t in TIERS}}
        for st in STRATA:
            xs = [i for i in pairs if i["provenance"]["stratum"] == st]
            v["strata"][st] = {"items": len(xs), "affected": sum(map(hit, xs))}
        for thr in (0.01, 0.02, 0.05):
            keep = [i for i in pairs if abs(cons(i) - 0.5) > thr]
            v["effective"][f"H1_{thr}"] = {"before": len(keep), "after": sum(not hit(i) for i in keep)}
        keep = [i for i in pairs if abs(cons(i) - 0.5) > 0.02 and i["provenance"]["stratum"] == "vs_anchored"]
        v["effective"]["S5_0.02"] = {"before": len(keep), "after": sum(not hit(i) for i in keep)}
        an = [i for i in items if i["block"] == "anchors"]
        v["effective"]["S1_anchors"] = {t: {"before": sum(i["tier"] == t for i in an),
                                            "after": sum(i["tier"] == t and not hit(i) for i in an)}
                                        for t in TIERS}
        out["variants"][name] = v
    text = json.dumps(out, indent=1)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
