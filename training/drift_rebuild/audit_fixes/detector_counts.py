"""Boundary- and hero-level detector counts, Fisher/binomial tests and
latencies from w3_changepoints.json (active results dir), for both truth
variants (any mention; balance-only = bug-fix-only mentions removed).
Output: <results>/detector_counts.json"""
import json
import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from drift2026 import common  # noqa: E402


def fisher_greater(a, b, c, d):
    r1, c1, n = a + b, a + c, a + b + c + d
    p = lambda x: comb(c1, x) * comb(n - c1, r1 - x) / comb(n, r1)
    return sum(p(x) for x in range(a, min(r1, c1) + 1))


def binom_sf(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


j = json.load(open(os.path.join(common.RESULTS_DIR, "w3_changepoints.json")))
gt = json.load(open(os.path.join(common.DRIFT_DIR, "patch_notes_ground_truth.json")))
bal = {}
for p in gt["patches"]:
    b = p.get("mapped_build")
    if b:
        bal.setdefault(b, set()).update(set(p.get("heroes_changed") or []) - set(p.get("heroes_bugfix_only") or []))
vb = [b for b in j["boundaries"] if b["truth"]["validatable"]]
out = {"n_boundaries": len(vb)}
fire = np.array([len(b["detected"]["wr_bh"]) > 0 for b in vb])
for name, truth in (("any", lambda b: set(b["truth"]["heroes"])),
                    ("balance", lambda b: set().union(*[bal.get(x, set()) for x in b["intermediate_builds"] + [b["new_build"]]]))):
    T = [truth(b) for b in vb]
    flags = sum(len(b["detected"]["wr_bh"]) for b in vb)
    tp = sum(len(set(b["detected"]["wr_bh"]) & t) for b, t in zip(vb, T))
    chg = np.array([len(t) > 0 for t in T])
    a, b_, c, d = int((fire & chg).sum()), int((~fire & chg).sum()), int((fire & ~chg).sum()), int((~fire & ~chg).sum())
    out[name] = {"flags": flags, "tp_flags": tp, "hero_precision": round(tp / flags, 3),
                 "fire_changed": [a, a + b_], "fire_unchanged": [c, c + d],
                 "fisher_p": round(fisher_greater(a, b_, c, d), 4),
                 "boundary_precision": [a, a + c], "base_rate": round((a + b_) / len(vb), 3),
                 "binom_p": round(binom_sf(a, a + c, (a + b_) / len(vb)), 4),
                 "chance_hero_rate": round(float(np.mean([len(t) / 90 for t in T])), 3)}


def g(L, k):
    v = L.get(k)
    return v["build_games"] if isinstance(v, dict) else None


first, tz, tb = [], [], []
for b in vb:
    bh, T = set(b["detected"]["wr_bh"]), set(b["truth"]["heroes"])
    zs = [g(b["latency_wr"].get(h, {}), "z_naive") for h in bh]
    zs = [x for x in zs if x is not None]
    if zs:
        first.append(min(zs))
    for h in bh & T:
        tz.append(g(b["latency_wr"].get(h, {}), "z_naive"))
        tb.append(g(b["latency_wr"].get(h, {}), "z_bonf"))
out["latency"] = {"first_flag_median": float(np.median(first)), "n_firing": len(first),
                  "tp_n": len(tz), "naive_median": float(np.median([x for x in tz if x])),
                  "bonf_median_crossing": float(np.median([x for x in tb if x])),
                  "bonf_never": sum(x is None for x in tb)}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(common.RESULTS_DIR, "detector_counts.json"), "w"), indent=1)
