"""
Descriptive, post hoc (not in the registered plan): accuracy of the study's
draft-only models on the same real-game items the experts rated. Uses only
data/rating-items.json and the oct2026 value functions; no ratings.

Models: the study's consensus label (mean of the naive, hero-strength and
enriched value functions, composition-corrected; the H1 label) and each value
function alone (corrected). A model "picks" the side it gives > 0.5.
Run in the oct2026 env:
  bash paper1_revision/oct2026_env.sh python3 paper1_revision/expert_analysis/exploratory_anchor_models.py <out.json>
"""
import sys, json, math
import numpy as np
sys.path.insert(0, "/home/max/heroes-of-the-storm/training")
from rerun2026 import common
common.setup()
from paper1_revision import oct2026_struct_correction as sc


def wilson(k, n, z=1.959963984540054):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"k": int(k), "n": int(n), "rate": p, "lo": c - h, "hi": c + h}


pool = json.load(open("/home/max/heroes-of-the-storm/data/rating-items.json"))["items"]
an = [it for it in pool if it["block"] == "anchors"]
rows = [(it["teams"]["team0"], it["teams"]["team1"], it["map"], it["tier"]) for it in an]
y = np.array([it["provenance"]["winner"] == 0 for it in an])
models = {"consensus": sc.consensus_sc(rows), "naive": sc.naive_sc(rows),
          "herostrength": sc.herostrength_sc(rows), "enriched": sc.enriched_sc(rows)}
out = {"note": __doc__.strip().splitlines()[1], "models": {}}
for name, p in models.items():
    p = np.asarray(p, float)
    out["models"][name] = {t: wilson(int(((p[m] > .5) == y[m]).sum()), int(m.sum()))
                           for t in ("low", "mid", "high", "all")
                           for m in [np.array([t == "all" or it["tier"] == t for it in an])]}
json.dump(out, open(sys.argv[1], "w"), indent=1)
for name, v in out["models"].items():
    print(name, {t: f"{x['rate']:.3f} [{x['lo']:.3f},{x['hi']:.3f}]" for t, x in v.items()})
