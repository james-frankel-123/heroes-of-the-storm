"""
Do the gold references see broken compositions?

On real post-snapshot no-drift games (BF + T97, the NODRIFT set), for every
team: the rule-based degenerate flag (shared.is_degenerate) and its type.
Compares each team's realized win rate with what the realized-outcome index
predicts for it, so the gap measures how much the judge under-penalizes
degenerate teams in real play. Also reports how many games the index's
role-composition cells hold for degenerate role multisets.

Usage (from training/): nice -n 19 taskset -c 48-51 python3 overfit2026/degen_judge_check.py
Output: overfit2026/results/degen_judge_check.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np

from shared import is_degenerate, HERO_ROLE_FINE, FINE_TO_BLIZZ_ROLE, DEGEN_STACK_ROLES
from overfit2026 import data
from overfit2026.gold import RealizedIndex, TeamFeatures, GOLD_SETS, stats_for
from drift2026 import common as dcommon

OUT = os.path.join(HERE, "results", "degen_judge_check.json")


def kind(team):
    roles = [FINE_TO_BLIZZ_ROLE.get(HERO_ROLE_FINE.get(h, ""), "Ranged Assassin") for h in team]
    has_heal = "Healer" in roles
    has_front = any(r in ("Tank", "Bruiser") for r in roles)
    if not has_front and "Uther" in team and roles.count("Healer") >= 2:
        has_front = True
    stack = any(roles.count(r) >= 3 for r in DEGEN_STACK_ROLES)
    if not has_heal:
        return "no_healer"
    if not has_front:
        return "no_frontline"
    if stack:
        return "stack"
    return "ok"


def main():
    dcommon._bind_statscache_methods()
    games = GOLD_SETS["NODRIFT"](data)
    print(f"{len(games):,} NODRIFT games")
    # judge built on one half, evaluated on the other: no game scores itself
    a = [g for g in games if data.half(g[0], salt=5) == 0]
    b = [g for g in games if data.half(g[0], salt=5) == 1]
    idx = RealizedIndex(a, name="half_a")
    rows = []
    for rid, tier, gmap, t0, t1, bans, w in b:
        p0 = idx.score(t0, t1, tier)
        for team, opp, won, p in ((t0, t1, w == 0, p0), (t1, t0, w == 1, 1 - p0)):
            rows.append((kind(team), float(won), p))
    out = {"n_games_eval": len(b), "by_kind": {}}
    kinds = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    p = np.array([r[2] for r in rows])
    for k in ("ok", "no_healer", "no_frontline", "stack", "any_degenerate"):
        m = (kinds != "ok") if k == "any_degenerate" else (kinds == k)
        n = int(m.sum())
        if n == 0:
            continue
        gap = y[m] - p[m]
        out["by_kind"][k] = {
            "teams": n, "share_pct": 100 * n / len(y),
            "realized_wr": float(y[m].mean()), "judge_pred": float(p[m].mean()),
            "realized_minus_pred_pp": float(100 * gap.mean()),
            "se_pp": float(100 * gap.std(ddof=1) / np.sqrt(n))}
    # how much data the comp term has for degenerate role multisets
    tf = TeamFeatures(stats_for(a))
    comp_games = {}
    for rid, tier, gmap, t0, t1, bans, w in a:
        for team in (t0, t1):
            k = kind(team)
            key = (tier, tf.comp_key(team))
            comp_games.setdefault(k, {})[key] = tf._comp.get(tier, {}).get(tf.comp_key(team), (0, 0))[0]
    out["comp_cell_games_median"] = {k: float(np.median(list(v.values()))) for k, v in comp_games.items()}
    out["index_coef"] = idx.describe()["folds"][0]["coef"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
