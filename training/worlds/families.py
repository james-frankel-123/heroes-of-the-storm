"""
Freeze the dials (world.py) before any agent is scored: python3 worlds/families.py freeze -> worlds/dials.json.

Floors per family: the 5th percentile of held-out human teams' support statistic (NODRIFT teams),
the rule that set paper 1's FLOOR = 25 for min3; family a uses 25 itself.

Defaults come from the step-1 real-data measurements (paper1_revision/site/results/overrating):
  rare compositions are not overrated beyond in-support ones in ranked play (level shift 0 +- 1 pt,
  slope below/in support 0.92 / 0.87, difference +0.05 +- 0.05)       -> default k = 1, delta = 0, a = 1
Named alternatives, all fixed here:
  qm          delta = -1     Quick Match's below-floor level bias (-1.0 +- 0.3 pts)
  certify14   delta = -2.7   certify's -1.8 +- 1.2 pts at min3 1-4, at full grade (g(2) = 0.66)
  atten       a = 0.84       the low end of the below-floor slope bound (0.84-1.1)
  sigma3      sigma = 3   extra off-support team noise of 3 pts SD (unseen team, full grade)
  stress      k = 2, delta = -3, a = 1, sigma = 3, drift = 2   outside every step-1 bound
  stress_hard k = 3, delta = -6, a = 0.5, sigma = 6
Grid for the break-even surface (pre-registered): k in {1, 1.5, 2, 3}, delta in {+1, 0, -1, -2, -3,
-4, -6, -8} points, a in {1.2, 1, 0.85, 0.7, 0.5}, sigma in {0, 3, 6} points, crossed only as the planes
delta x sigma, delta x k and delta x a (world.grid_cells), the other dials at default. Which cells step 1 allows is computed, not
chosen (checks.py allowed).
Worlds: 1,000 per family x dial cell for rescoring (ids 0-999); plausibility checks on ids 0-199.
"""
import os
import sys
import json
import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as WD  # noqa: E402
import world as W  # noqa: E402


def freeze():
    if os.path.exists(W.DIALS_JSON):
        raise SystemExit(f"{W.DIALS_JSON} exists: dials are frozen (delete by hand only before any scoring)")
    D = WD.load("NODRIFT")
    with open(W.DIALS_JSON, "w") as f:   # placeholder so Worlds() loads; overwritten below
        json.dump({"floors": {}, "drift": None}, f)
    Ws = W.Worlds()
    floors = {"triple": 25}
    stats = {}
    for fam, name in (("a", "triple"), ("b", "cross"), ("c", "body45"), ("d", "hmt")):
        m = np.concatenate(Ws.team_support(fam, D))
        p5 = float(np.percentile(m, 5))
        stats[name] = {"p5": p5, "p1": float(np.percentile(m, 1)), "median": float(np.median(m))}
        if name != "triple":
            floors[name] = max(1, int(round(p5)))
    drift = json.load(open(os.path.join(WD.RESULTS, "drift.json")))["sizes"]
    out = {
        "frozen": datetime.date.today().isoformat(),
        "floors": floors, "human_support_nodrift": stats, "drift": drift,
        "named": {
            "default": dict(k=1.0, delta=0.0, a=1.0, sigma=0.0, drift=1.0),
            "qm": dict(k=1.0, delta=-1.0, a=1.0, sigma=0.0, drift=1.0),
            "certify14": dict(k=1.0, delta=-2.7, a=1.0, sigma=0.0, drift=1.0),
            "atten": dict(k=1.0, delta=0.0, a=0.84, sigma=0.0, drift=1.0),
            "sigma3": dict(k=1.0, delta=0.0, a=1.0, sigma=3.0, drift=1.0),
            "stress": dict(k=2.0, delta=-3.0, a=1.0, sigma=3.0, drift=2.0),
            "stress_hard": dict(k=3.0, delta=-6.0, a=0.5, sigma=6.0, drift=2.0),
        },
        "grid": {"k": [1.0, 1.5, 2.0, 3.0], "delta": [1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0],
                 "a": [1.2, 1.0, 0.85, 0.7, 0.5], "sigma": [0.0, 3.0, 6.0]},
        "worlds": {"rescore": 1000, "checks": 200, "ensemble_g": 20},
        "grade": "g(m) = clip(1 - log(1+m)/log(1+F), 0, 1); delta in win-prob points at p = 0.5 (logit 0.04 per point)",
    }
    json.dump(out, open(W.DIALS_JSON, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["freeze"]:
        freeze()
