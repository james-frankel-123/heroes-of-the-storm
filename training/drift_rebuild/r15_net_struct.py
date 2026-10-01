"""
R15 — Net team win-probability gap with structure terms (consolidated audit
N3, B9).

The W13 / r10 "Net" index uses hero WR, synergy and counter only. Here every
head-to-head file is also scored with overfit2026.gold.StructRealizedIndex:
the same cross-fitted realized index plus a role-composition term and three
structure indicators (no healer, no frontline, 3+ stacked role), whose
weights are fitted on real outcomes of the held-out fold.

Games: the r10 cache (drift_rebuild/feature_cache/r10_clean_games.json,
fetched 2026-09-29 before the 2026-09-30 tier relabel, so its tiers are the
snapshot scheme the agents drafted under): builds 2.55.16.97039 and the four
2.55.17 builds, game_date < 2026-09-28; 308,375 games.

Net(draft) = 100 * (0.5 * (score(m, f) + 1 - score(f, m)) - 0.5), crossed
seed random effects with the Satterthwaite t of r8_score.

Usage: OMP_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 drift_rebuild/r15_net_struct.py
Output: drift_rebuild/results/r15_net_struct.json
"""
import os
import sys
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_common as rb  # noqa: E402

from drift2026 import common  # noqa: E402

common.setup()
import numpy as np  # noqa: E402
from overfit2026.gold import RealizedIndex, StructRealizedIndex  # noqa: E402
from drift2026.w8_inference import attach_labels  # noqa: E402
from r8_score import infer, PAPER_FILES  # noqa: E402


def main():
    games = [tuple(tuple(x) if isinstance(x, list) else x for x in g)
             for g in json.load(open(os.path.join(rb.CACHE_DIR, "r10_clean_games.json")))]
    print(f"{len(games):,} games", flush=True)
    idx = {"plain": RealizedIndex(games, name="plain"),
           "struct": StructRealizedIndex(games, name="struct")}
    files = {os.path.splitext(os.path.basename(p))[0]: p
             for p in sorted(glob.glob(os.path.join(rb.RESULTS_DIR, "h2h", "*.json")))}
    for k, f in PAPER_FILES.items():
        files[k] = os.path.join(common.RESULTS_DIR, f)
    files["paper_Md90_vs_M"] = os.path.join(common.RESULTS_DIR,
                                            "w6_head2head_champ_vs_cumprev.json")
    out = {"n_games": len(games), "fits": {k: v.describe() for k, v in idx.items()},
           "files": {}}
    for key, path in files.items():
        res = json.load(open(path))
        recs = (res["records"] if res["records"] and "sa" in res["records"][0]
                else attach_labels(res))
        sa = np.array([r["sa"] for r in recs])
        sb = np.array([r["sb"] for r in recs])
        e = {}
        for name, ix in idx.items():
            v = np.array([100 * (0.5 * (ix.score(tuple(r["maintained"]), tuple(r["frozen"]), r["tier"])
                                        + 1 - ix.score(tuple(r["frozen"]), tuple(r["maintained"]), r["tier"])) - 0.5)
                          for r in recs])
            x = infer(v, sa, sb)
            e[name] = {k2: x[k2] for k2 in ("est", "se", "z", "df", "p_t")}
        out["files"][key] = e
        print(f"{key:24s} plain {e['plain']['est']:+.2f} ± {e['plain']['se']:.2f}  "
              f"struct {e['struct']['est']:+.2f} ± {e['struct']['se']:.2f} "
              f"(t {e['struct']['z']:.1f}, df {e['struct']['df']:.0f})", flush=True)
    common.write_json(os.path.join(rb.RESULTS_DIR, "r15_net_struct.json"), out)


if __name__ == "__main__":
    main()
