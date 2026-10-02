"""
Judge (label) step for the expert-study v6/v6.1 pool (oct2026).

Where judges enter the pool
  * Item SAMPLING does not use any judge: generate-rating-items.ts draws
    machine pairs by matchup stratum (cycling strategy pairs, then least-used
    tier/map, with a per-team appearance cap), real games by tier and map,
    and screener/catch teams by role rules.
  * The only judge-derived item data is provenance.wpTeam0Sym* on the 280
    machine pairs: P(team0 wins), team-order symmetrized, per evaluator. The
    tournament scripts write the three namespace evaluators (naive,
    herostrength, enriched) into every record; the generator copies them,
    and this script keeps that copy as provenance.wpTeam0Sym_tournament.
  * OOD covariates (rating_items_ood.py) come from the 20-member WP ensemble,
    a separate model family.

v6/v6.1 runs this twice:
  1. --judges ns:naive,ns:herostrength,ns:enriched --consensus naive,herostrength,enriched
     --field wpTeam0Sym_uncorrected
  2. --judges py:<oct2026_struct_correction.py>:{naive,herostrength,enriched,consensus}_sc
     --consensus-judge consensus_sc --rename ..._sc:... --field wpTeam0Sym
     (the structure-corrected labels the preregistration reads)
Each run records its judge specs, consensus definition and near-tie counts in
pool['judges'][field]. Judges:
  ns:<name>         a namespace WP evaluator (naive, herostrength, enriched),
                    scored with the namespace deploy statistics
  py:<file>:<func>  any Python callable func(rows) -> np.ndarray of
                    P(team0 wins), rows = [(team0, team1, map, tier)]
Env: oct2026_refresh.base_env (RERUN_NS=oct2026, deploy stats, oct2026_site
hook); rerun2026.common.setup() refuses to run without it.
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
           }  # wp_aug_v2_512 (synthetic augmentation) dropped in v6


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
    ap.add_argument("--consensus-judge", default=None,
                    help="use this judge's output directly as the consensus (e.g. a separately "
                         "corrected consensus) instead of averaging")
    ap.add_argument("--field", default="wpTeam0Sym", help="provenance field to write")
    ap.add_argument("--rename", default="", help="name map for stored keys, e.g. naive_sc:naive,...")
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
    if a.consensus_judge:
        cons_names = [a.consensus_judge]
        cons = scores[a.consensus_judge]
    else:
        cons_names = a.consensus.split(",") if a.consensus else list(scores)
        cons = np.mean([scores[n] for n in cons_names], 0)
    summary = {thr: int(np.sum(np.abs(cons - 0.5) <= thr)) for thr in (0.01, 0.02, 0.05)}
    print("near-ties (excluded) at 0.01/0.02/0.05:", summary)
    if a.dry_run:
        return
    ren = dict(x.split(":") for x in a.rename.split(",") if x)
    for i, it in enumerate(machine):
        prov = it["provenance"]
        if "wpTeam0Sym_tournament" not in prov:
            prov["wpTeam0Sym_tournament"] = prov.get("wpTeam0Sym", {})
        d = {ren.get(n, n): float(scores[n][i]) for n in scores}
        d["consensus"] = float(cons[i])
        prov[a.field] = d
    pool.setdefault("judges", {})[a.field] = {
        "specs": a.judges.split(","), "consensus": cons_names,
        "near_ties_excluded": {str(k): v for k, v in summary.items()}}
    json.dump(pool, open(a.pool, "w"), indent=1)
    print(f"wrote {a.pool}")


if __name__ == "__main__":
    main()
