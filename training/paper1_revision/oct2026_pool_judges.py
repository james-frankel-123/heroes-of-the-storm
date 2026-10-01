"""
Pluggable judge step for the expert-study v6 pool (oct2026). NOT RUN until the
judge set is decided (coordinator, 2026-10-01: the independent judges are
being audited for under-penalizing degenerate compositions).

Where judges enter the pool
  * Item SAMPLING does not use any judge: generate-rating-items.ts draws
    machine pairs by matchup stratum (cycling strategy pairs, then least-used
    tier/map), real games by tier and map, and screener/catch teams by role
    rules.
  * The only judge-derived item data is provenance.wpTeam0Sym on the 280
    machine pairs: P(team0 wins), team-order symmetrized, per evaluator. The
    tournament scripts write the four namespace evaluators (naive,
    herostrength, enriched, augmented) into every record; the generator
    copies them. The preregistration uses these frozen numbers for H1
    (consensus side), the near-tie exclusion (|consensus - 0.5| <= 0.02, also
    0.01/0.05), the per-evaluator sensitivities, S2 (slider calibration), and
    S5. Nothing filters items on them at generation time.
  * OOD covariates (rating_items_ood.py) come from the 20-member WP ensemble,
    a separate model family.

This script (re)writes provenance.wpTeam0Sym for every machine pair from a
configurable judge set and records which judges were used and the consensus
definition in pool['judges']. Judges:
  ns:<name>         a namespace WP evaluator (naive, herostrength, enriched,
                    augmented), scored with the namespace deploy statistics
  py:<file>:<func>  any Python callable func(rows) -> np.ndarray of
                    P(team0 wins), rows = [(team0, team1, map, tier)]; this is
                    the hook for the corrected, composition-aware judges
The previous wpTeam0Sym is kept as provenance.wpTeam0Sym_tournament.

Usage (env RERUN_NS=oct2026 etc., see oct2026_refresh.base_env):
  python3 paper1_revision/oct2026_pool_judges.py --judges ns:naive,ns:herostrength,ns:enriched,ns:augmented \
      [--consensus naive,herostrength,enriched,augmented] [--pool data/rating-items.json] [--dry-run]
"""
import os
import sys
import json
import argparse
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(TRAINING_DIR)
sys.path.insert(0, TRAINING_DIR)

import numpy as np

NS_EVAL = {"naive": ("wp_naive.pt", [256, 128], []),
           "herostrength": ("wp_herostrength.pt", [256, 128], ["hero_wr", "team_avg_wr"]),
           "enriched": ("wp_enriched_256.pt", [256, 128], "E"),
           "augmented": ("wp_aug_v2_512.pt", [512, 256, 128], "E")}


def ns_judge(name):
    import torch
    from rerun2026 import common
    from sweep_enriched_wp import WinProbEnrichedModel
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    from paper1_revision.oct2026_wp import cols_for
    from paper1_revision import core
    fn, arch, groups = NS_EVAL[name]
    groups = list(ENRICHED_GROUPS) if groups == "E" else groups
    cols = cols_for(groups)
    m = WinProbEnrichedModel(197 + len(cols), arch, dropout=0.3)
    m.load_state_dict(torch.load(os.path.join(common.MODELS_DIR, fn), map_location="cpu",
                                 weights_only=True))
    m.eval()
    st = common.stats_cache()

    def score(rows):
        Xf, Xs = core.featurize(rows, st, nproc=4)
        sel = list(range(197)) + [197 + c for c in cols]
        with torch.no_grad():
            a = m(torch.tensor(Xf[:, sel])).view(-1).numpy()
            b = m(torch.tensor(Xs[:, sel])).view(-1).numpy()
        return 0.5 * (a + 1 - b)
    return score


def py_judge(spec):
    path, func = spec.rsplit(":", 1)
    sp = importlib.util.spec_from_file_location("judge_mod", path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return getattr(mod, func)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", required=True)
    ap.add_argument("--consensus", default=None, help="judge names averaged for the consensus")
    ap.add_argument("--pool", default=os.path.join(REPO, "data", "rating-items.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from rerun2026 import common
    common.setup()
    pool = json.load(open(a.pool))
    machine = [it for it in pool["items"] if it["provenance"].get("source") == "tournament"]
    rows = [(it["teams"]["team0"], it["teams"]["team1"], it["map"], it["tier"]) for it in machine]
    scores = {}
    for spec in a.judges.split(","):
        name = spec.split(":", 1)[1] if spec.startswith("ns:") else spec.rsplit(":", 1)[1]
        fn = ns_judge(name) if spec.startswith("ns:") else py_judge(spec[3:])
        scores[name] = np.asarray(fn(rows), float)
        print(f"{name}: mean P(team0) {scores[name].mean():.4f}", flush=True)
    cons_names = a.consensus.split(",") if a.consensus else list(scores)
    cons = np.mean([scores[n] for n in cons_names], 0)
    summary = {thr: int(np.sum(np.abs(cons - 0.5) <= thr)) for thr in (0.01, 0.02, 0.05)}
    print("near-ties (excluded) at 0.01/0.02/0.05:", summary)
    if a.dry_run:
        return
    for i, it in enumerate(machine):
        prov = it["provenance"]
        if "wpTeam0Sym_tournament" not in prov:
            prov["wpTeam0Sym_tournament"] = prov.get("wpTeam0Sym", {})
        prov["wpTeam0Sym"] = {n: float(scores[n][i]) for n in scores}
    pool["judges"] = {"specs": a.judges.split(","), "consensus": cons_names,
                      "near_ties_excluded": {str(k): v for k, v in summary.items()}}
    json.dump(pool, open(a.pool, "w"), indent=1)
    print(f"wrote {a.pool}")


if __name__ == "__main__":
    main()
