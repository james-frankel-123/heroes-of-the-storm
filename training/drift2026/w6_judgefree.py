"""
Q1 — judge-free rescoring of the W6 head-to-head drafts.

Replaces learned-judge WP (vintage-confounded: the consensus judges are
era-aligned with the maintained agent) with metrics that carry no judge
vintage at all:
  - degenerate / healer rates per side (rule-based),
  - counter and synergy deltas per side scored against FUTURE-PERIOD
    ground-truth statistics (the actual post-cutoff meta's outcomes; the
    W4 benchmark protocol applied to the W6 terminal drafts).

Usage: python3 drift2026/w6_judgefree.py
Output: drift2026/results/W6_JUDGEFREE.{json,md}
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

common.setup()

import numpy as np

from shared import is_degenerate, HERO_ROLE_FINE
from drift2026.phase_w4_mcts import future_truth_stats

HERE = os.path.dirname(os.path.abspath(__file__))
IN_JSON = os.path.join(HERE, "results", "w6_head2head.json")
OUT_JSON = os.path.join(HERE, "results", "w6_judgefree.json")
OUT_MD = os.path.join(HERE, "results", "W6_JUDGEFREE.md")


def main():
    global IN_JSON, OUT_JSON, OUT_MD
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="head2head file suffix, e.g. _stale1yr")
    args = ap.parse_args()
    if args.suffix:
        IN_JSON = os.path.join(HERE, "results", f"w6_head2head{args.suffix}.json")
        OUT_JSON = os.path.join(HERE, "results", f"w6_judgefree{args.suffix}.json")
        OUT_MD = os.path.join(HERE, "results", f"W6_JUDGEFREE{args.suffix.upper()}.md")
    truth = future_truth_stats()
    healers = set(h for h, r in HERO_ROLE_FINE.items() if r == "healer")
    res = json.load(open(IN_JSON))
    records = res["records"]
    print(f"{len(records)} W6 drafts")

    def ctr_d(ha, hb, tier):
        r = truth.get_counter(ha, hb, tier)
        if r is None:
            return None
        return r - (truth.get_hero_wr(ha, tier)
                    + (100 - truth.get_hero_wr(hb, tier)) - 50)

    def syn_d(ha, hb, tier):
        r = truth.get_synergy(ha, hb, tier)
        if r is None:
            return None
        return r - (50 + (truth.get_hero_wr(ha, tier) - 50)
                    + (truth.get_hero_wr(hb, tier) - 50))

    def hero_wr_mean(team, tier):
        vals = [truth.get_hero_wr(h, tier) for h in team]
        vals = [v for v in vals if v is not None]
        return np.mean(vals) if vals else None

    sides = {"maintained": {"ctr": [], "syn": [], "deg": [], "heal": [], "hwr": []},
             "frozen": {"ctr": [], "syn": [], "deg": [], "heal": [], "hwr": []}}
    for r in records:
        tier = r["tier"]
        for side, own, opp in (("maintained", r["maintained"], r["frozen"]),
                               ("frozen", r["frozen"], r["maintained"])):
            d = sides[side]
            cd = [x for o in opp for h in own
                  for x in [ctr_d(h, o, tier)] if x is not None]
            d["ctr"].append(np.mean(cd) if cd else 0.0)
            sy = [x for j, h1 in enumerate(own) for h2 in own[j + 1:]
                  for x in [syn_d(h1, h2, tier)] if x is not None]
            d["syn"].append(np.mean(sy) if sy else 0.0)
            d["deg"].append(is_degenerate(own))
            d["heal"].append(any(h in healers for h in own))
            hw = hero_wr_mean(own, tier)
            if hw is not None:
                d["hwr"].append(hw)

    out = {}
    for side, d in sides.items():
        out[side] = {
            "counter_future": float(np.mean(d["ctr"])),
            "counter_se": float(np.std(d["ctr"], ddof=1) / np.sqrt(len(d["ctr"]))),
            "synergy_future": float(np.mean(d["syn"])),
            "synergy_se": float(np.std(d["syn"], ddof=1) / np.sqrt(len(d["syn"]))),
            "degen_pct": float(np.mean(d["deg"]) * 100),
            "healer_pct": float(np.mean(d["heal"]) * 100),
            "future_hero_wr_mean": float(np.mean(d["hwr"])),
        }
    # Paired per-draft deltas (maintained - frozen), tighter than side SEs.
    dctr = np.array(sides["maintained"]["ctr"]) - np.array(sides["frozen"]["ctr"])
    dsyn = np.array(sides["maintained"]["syn"]) - np.array(sides["frozen"]["syn"])
    dhwr = np.array(sides["maintained"]["hwr"]) - np.array(sides["frozen"]["hwr"])
    out["paired_deltas"] = {
        "counter": {"mean": float(dctr.mean()),
                    "se": float(dctr.std(ddof=1) / np.sqrt(len(dctr)))},
        "synergy": {"mean": float(dsyn.mean()),
                    "se": float(dsyn.std(ddof=1) / np.sqrt(len(dsyn)))},
        "future_hero_wr": {"mean": float(dhwr.mean()),
                           "se": float(dhwr.std(ddof=1) / np.sqrt(len(dhwr)))},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    m, fz, pd = out["maintained"], out["frozen"], out["paired_deltas"]
    lines = [
        "# W6 judge-free rescoring (future-truth stats; no learned judge)",
        "",
        "| metric | maintained | frozen | paired delta (m − f) |",
        "|---|---|---|---|",
        f"| degenerate % | {m['degen_pct']:.1f} | {fz['degen_pct']:.1f} | "
        f"{m['degen_pct']-fz['degen_pct']:+.1f} pp |",
        f"| healer % | {m['healer_pct']:.1f} | {fz['healer_pct']:.1f} | "
        f"{m['healer_pct']-fz['healer_pct']:+.1f} pp |",
        f"| counter vs future truth | {m['counter_future']:+.3f} | "
        f"{fz['counter_future']:+.3f} | {pd['counter']['mean']:+.3f} "
        f"± {pd['counter']['se']:.3f} |",
        f"| synergy vs future truth | {m['synergy_future']:+.3f} | "
        f"{fz['synergy_future']:+.3f} | {pd['synergy']['mean']:+.3f} "
        f"± {pd['synergy']['se']:.3f} |",
        f"| mean future hero WR of picks | {m['future_hero_wr_mean']:.2f} | "
        f"{fz['future_hero_wr_mean']:.2f} | {pd['future_hero_wr']['mean']:+.2f} "
        f"± {pd['future_hero_wr']['se']:.2f} |",
        "",
        "All scoring against merged post-cutoff per-build ground-truth counts "
        "(the actual future meta) — no learned judge, no vintage confound.",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
