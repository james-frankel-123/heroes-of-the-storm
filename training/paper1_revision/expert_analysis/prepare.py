"""
Expert study v6.2 (prereg v3, osf.io/fspuc): build the analysis tables from a
frozen ratings export. No DB access; no model inference.

  python3 prepare.py <ratings.csv> <sequences.json> <out_dir> [--partials strict|lenient]

Inputs: the read-only export (export_ratings.sh), the app's assignment for each
rater code (dump_sequences.ts), data/rating-items.json and config.json.

Rules implemented (prereg v3 section in brackets):
  * drop is_test rows and rows created before the registration [§3.1, §3.2]
  * every rating must be an item of the rater's assigned slot (checked; abort
    otherwise); the final stored row per (rater, item) is the one analyzed [§2]
  * side correction: choice and slider in team0 terms [§2, §6 S2]
  * correct = forced choice matches the recorded winner / real side [§2]
  * gates: screener >= 7/8, catch >= 2/3, median ms >= 10,000 [§3.3-3.6]
    - strict (primary): a gate whose items the rater never reached cannot be
      passed, so a partial rater who stopped before the catch items is
      excluded; the median is over the items rated
    - lenient (sensitivity, --partials lenient): a partial rater is held only
      to the gate items they reached (screener must be complete; catch >= the
      prereg's 2/3 rate on the catch items rated, vacuous if none)
  * calibration accuracy <= 50% is a flag only [§3.5]
Outputs: pairs.csv, anchors.csv, raters.json in out_dir.
"""
import os
import sys
import csv
import json
import argparse
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ratings")
    ap.add_argument("sequences")
    ap.add_argument("out")
    ap.add_argument("--partials", choices=["strict", "lenient"], default="strict")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    pool = json.load(open(os.path.join(REPO, cfg["pool"]["file"])))
    items = {it["id"]: it for it in pool["items"]}
    seqs = json.load(open(a.sequences))
    reg = cfg["prereg"]["registered_utc"].replace(" ", "T")
    G = cfg["gates"]

    rows = list(csv.DictReader(open(a.ratings)))
    by_rater = defaultdict(dict)
    dropped = {"is_test": 0, "before_registration": 0}
    for r in rows:
        if r["is_test"] in ("t", "true", "True", "1"):
            dropped["is_test"] += 1
            continue
        if r["created_at"] < reg:
            dropped["before_registration"] += 1
            continue
        code = r["rater"]
        if code not in cfg["raters"]:
            raise SystemExit(f"rating from a code not in config.json: {code!r}")
        by_rater[code][int(r["item_id"])] = r   # unique (rater, item) in the DB

    raters, pair_rows, anchor_rows = {}, [], []
    for code in sorted(by_rater):
        seq = seqs[code]["sequence"]
        pos = {iid: i + 1 for i, iid in enumerate(seq)}
        rr = by_rater[code]
        bad = [iid for iid in rr if iid not in pos]
        if bad:
            raise SystemExit(f"{code}: {len(bad)} rated items not in slot {seqs[code]['slot']}'s assignment")
        recs = []
        for iid, r in rr.items():
            it = items[iid]
            if r["block"] != it["block"]:
                raise SystemExit(f"{code} item {iid}: stored block {r['block']} != pool block {it['block']}")
            a_is_0 = r["team_a_is_team0"] in ("t", "true", "True", "1")
            pick0 = (r["better_team"] == "A") == a_is_0
            pa = int(r["p_team_a"]) / 100
            slider0 = pa if a_is_0 else 1 - pa
            w = it["provenance"].get("winner")
            recs.append({"item": iid, "block": it["block"], "tier": it["tier"], "map": it["map"],
                         "pos": pos[iid], "pick0": pick0, "slider0": slider0,
                         "ms": int(r["ms_taken"]) if r["ms_taken"] else None,
                         "correct": (pick0 == (w == 0)) if w is not None and it["block"] != "pairs" else None})
        n = len(recs)
        last = max(x["pos"] for x in recs)
        blk = lambda b: [x for x in recs if x["block"] == b]
        scr, cat, cal = blk("screener"), blk("catch"), blk("calibration")
        scr_ok = sum(x["correct"] for x in scr)
        cat_ok = sum(x["correct"] for x in cat)
        ms = [x["ms"] for x in recs if x["ms"] is not None]
        med = statistics.median(ms) if ms else None
        complete = n == 240
        g_scr = len(scr) == 8 and scr_ok >= G["screener_min"]
        g_cat_strict = cat_ok >= G["catch_min"]
        g_cat_len = (cat_ok >= G["catch_min"]) if len(cat) == 3 else (len(cat) == 0 or cat_ok / len(cat) >= G["catch_min"] / 3)
        g_spd = med is not None and med >= G["speed_median_ms_min"]
        inc_strict = g_scr and g_cat_strict and g_spd
        inc_len = g_scr and g_cat_len and g_spd
        included = inc_strict if a.partials == "strict" else inc_len
        cal_acc = (sum(x["correct"] for x in cal) / len(cal)) if cal else None
        raters[code] = {"slot": cfg["raters"][code]["slot"], "group": cfg["raters"][code]["group"],
                        "n_rated": n, "complete": complete, "last_position_reached": last,
                        "positions_contiguous": sorted(x["pos"] for x in recs) == list(range(1, n + 1)),
                        "screener": [scr_ok, len(scr)], "catch": [cat_ok, len(cat)],
                        "median_ms": med, "calibration_acc": cal_acc,
                        "calibration_flag_le_50pct": cal_acc is not None and cal_acc <= G["calibration_flag_max"],
                        "gate_screener": g_scr, "gate_catch_strict": g_cat_strict, "gate_catch_lenient": g_cat_len,
                        "gate_speed": g_spd, "included_strict": inc_strict, "included_lenient": inc_len,
                        "included": included}
        if not included:
            continue
        for x in recs:
            it = items[x["item"]]
            pv = it["provenance"]
            heroes = set(it["teams"]["team0"]) | set(it["teams"]["team1"])
            p257a = bool(heroes & set(cfg["patch257"]["changed_heroes"]))
            p257b = p257a or bool(heroes & set(cfg["patch257"]["bugfix_heroes"])) or it["map"] in cfg["patch257"]["changed_maps"]
            if x["block"] == "pairs":
                c, u = pv["wpTeam0Sym"], pv["wpTeam0Sym_uncorrected"]
                s0, s1 = pv["team0Strategy"], pv["team1Strategy"]
                row = {"rater": code, "item": x["item"], "tier": x["tier"], "map": x["map"],
                       "stratum": pv["stratum"], "s0": s0, "s1": s1,
                       "chosen": s0 if x["pick0"] else s1, "other": s1 if x["pick0"] else s0,
                       "pick0": int(x["pick0"]), "slider0": x["slider0"],
                       "ood_var_max": pv["ood_var_max"], "ood_var_mean": pv["ood_var_mean"],
                       "ood_var_matchup": pv["ood_var_matchup"], "p257a": int(p257a), "p257b": int(p257b)}
                for name, val in (("cons", c["consensus"]), ("cons_unc", u["consensus"]),
                                  ("naive", c["naive"]), ("herostrength", c["herostrength"]),
                                  ("enriched", c["enriched"])):
                    row[f"p_{name}"] = val
                    row[f"agree_{name}"] = int(x["pick0"] == (val > 0.5))
                pair_rows.append(row)
            elif x["block"] == "anchors":
                anchor_rows.append({"rater": code, "item": x["item"], "tier": x["tier"], "map": x["map"],
                                    "group": cfg["raters"][code]["group"], "correct": int(x["correct"]),
                                    "slider0": x["slider0"], "winner0": int(pv["winner"] == 0),
                                    "p257a": int(p257a), "p257b": int(p257b)})
    os.makedirs(a.out, exist_ok=True)
    for name, rows_ in (("pairs.csv", pair_rows), ("anchors.csv", anchor_rows)):
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0].keys()))
            w.writeheader()
            w.writerows(rows_)
    json.dump({"partials_rule": a.partials, "dropped_rows": dropped, "raters": raters,
               "included": sorted(c for c, r in raters.items() if r["included"])},
              open(os.path.join(a.out, "raters.json"), "w"), indent=1)
    print(f"{len(raters)} raters, {sum(r['included'] for r in raters.values())} included ({a.partials}); "
          f"{len(pair_rows)} pair and {len(anchor_rows)} anchor judgments")


if __name__ == "__main__":
    main()
