"""
Table II/III redone with the revision's leak-free models: greedy drafters
(naive / hero-strength / enriched, 256x128) each draft the same 480
configurations (rerun2026 phase3 protocol: random.seed(42) configs, GD
opponents sampled at T=1 with per-step seeds, archive
experiment_value_function_quality.run_single_draft unchanged), with own
deploy statistics. Terminal teams are scored by every revision model (the
self-evaluation matrix of the submission) and by the independent references.
Scores are team-order symmetrized P(our team wins).
Output: results/crosseval.json
"""
import os
import sys
import json
import random

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core

import numpy as np
import torch

DRAFTERS = ["naive", "herostrength", "enriched"]


def _worker(args):
    configs, = args
    torch.set_num_threads(1)
    from archive.experiment_value_function_quality import run_single_draft
    from rerun2026.phase3_benchmarks import load_gd_models
    from sweep_enriched_wp import compute_group_indices
    from paper1_revision import train_wp
    S = train_wp.specs()
    dev = torch.device("cpu")
    gd = load_gd_models(dev)
    st = core.load_stats("deploy")
    gi = compute_group_indices()
    models = {n: train_wp.load(n)[0] for n in DRAFTERS}
    groups = {n: S[n]["groups"] for n in DRAFTERS}
    out = []
    for (ci, gm, tier, ou) in configs:
        for n in DRAFTERS:
            r = run_single_draft(ci, gm, tier, ou, n, models[n], groups[n], models, groups,
                                 gd, st, gi, dev)
            rec = {k: r[k] for k in ("wp_model", "game_map", "skill_tier", "our_team",
                                     "our_picks", "opp_picks", "comp_has_healer",
                                     "comp_has_ranged_damage", "comp_is_absurd",
                                     "comp_has_frontline", "comp_roles")}
            rec["steps"] = [{"step": st["step"], "chosen_hero": st["chosen_hero"]}
                            for st in r["steps"]]
            out.append(rec)
    return out


def main():
    import multiprocessing as mp
    from shared import MAPS, SKILL_TIERS
    path = os.path.join(core.RESULTS, "crosseval_records.json")
    if not os.path.exists(path):
        random.seed(42)
        configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2) for i in range(480)]
        parts = [(configs[i::12],) for i in range(12)]
        with mp.get_context("fork").Pool(12) as pool:
            recs = [r for p in pool.map(_worker, parts) for r in p]
        json.dump(recs, open(path, "w"))
    recs = json.load(open(path))
    from paper1_revision.bench_mcts import sym_predict
    from paper1_revision.score_tournament import score_rows
    from paper1_revision import train_wp
    rows = [(r["our_picks"], r["opp_picks"], r["game_map"], r["skill_tier"]) for r in recs]
    sc = score_rows(rows)
    Xf, Xs = core.featurize(rows, core.load_stats("deploy"), nproc=int(os.environ.get("P1R_NPROC", "6")))
    for n in DRAFTERS:
        m, cols = train_wp.load(n)
        sc[f"by_{n}"] = sym_predict(m, cols, Xf, Xs)
    res = {"n_per_drafter": 480, "matrix": {}, "composition": {}}
    for d in DRAFTERS:
        mask = np.array([r["wp_model"] == d for r in recs])
        res["matrix"][d] = {k: [float(v[mask].mean()), float(v[mask].std(ddof=1) / np.sqrt(mask.sum()))]
                            for k, v in sc.items()}
        sub = [r for r in recs if r["wp_model"] == d]
        res["composition"][d] = {
            "healer": 100 * np.mean([r["comp_has_healer"] for r in sub]),
            "ranged": 100 * np.mean([r["comp_has_ranged_damage"] for r in sub]),
            "frontline": 100 * np.mean([r["comp_has_frontline"] for r in sub]),
            "degen": 100 * np.mean([r["comp_is_absurd"] for r in sub])}
        print(d, {k: round(v[0], 4) for k, v in res["matrix"][d].items()},
              res["composition"][d], flush=True)
    # pick agreement at the same step of the same configuration (submission's
    # "divergence" definition), and the 50 naive drafts the naive model rates
    # most above the enriched model
    import math
    byc = {}
    for r in recs:
        byc.setdefault((r["game_map"], r["skill_tier"], r["our_team"]), {})[r["wp_model"]] = r
    phases = {"bans": {0, 1, 2, 3, 9, 10}, "early_picks": {4, 5, 6, 7, 8},
              "late_picks": {11, 12, 13, 14, 15}}
    res["agreement"] = {}
    for ph, steps in phases.items():
        agree = {"naive_vs_enriched": [0, 0], "herostr_vs_enriched": [0, 0]}
        for d in byc.values():
            if not all(k in d for k in DRAFTERS):
                continue
            e = {s["step"]: s["chosen_hero"] for s in d["enriched"]["steps"]}
            for k, other in (("naive_vs_enriched", "naive"), ("herostr_vs_enriched", "herostrength")):
                o = {s["step"]: s["chosen_hero"] for s in d[other]["steps"]}
                for st in steps:
                    if st in e and st in o:
                        agree[k][0] += e[st] == o[st]
                        agree[k][1] += 1
        res["agreement"][ph] = {k: v[0] / v[1] for k, v in agree.items() if v[1]}
    nidx = [i for i, r in enumerate(recs) if r["wp_model"] == "naive"]
    gap = sorted(nidx, key=lambda i: -(sc["by_naive"][i] - sc["by_enriched"][i]))[:50]
    res["top50_gap"] = {"no_healer": int(sum(not recs[i]["comp_has_healer"] for i in gap)),
                        "degenerate": int(sum(recs[i]["comp_is_absurd"] for i in gap)),
                        "role_stacking": int(sum(any(v >= 3 for v in recs[i]["comp_roles"].values())
                                                 for i in gap))}
    hn = sum(r["comp_has_healer"] for r in recs if r["wp_model"] == "naive")
    he = sum(r["comp_has_healer"] for r in recs if r["wp_model"] == "enriched")
    n = 480
    tab = [hn, n - hn, he, n - he]
    tot = sum(tab)
    col1, col2 = hn + he, 2 * n - hn - he
    exp = [n * col1 / tot, n * col2 / tot, n * col1 / tot, n * col2 / tot]
    chi = sum((o - e) ** 2 / e for o, e in zip(tab, exp))
    res["healer_naive_vs_enriched"] = {"diff_pp": 100 * (he - hn) / n, "chi2": chi,
                                       "p": math.erfc(math.sqrt(chi / 2))}
    print(res["agreement"], res["top50_gap"], res["healer_naive_vs_enriched"])
    json.dump(res, open(os.path.join(core.RESULTS, "crosseval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
