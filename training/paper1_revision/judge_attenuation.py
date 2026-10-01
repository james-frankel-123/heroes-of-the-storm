"""
Judge attenuation benchmark (audit P1-A2).

An independent judge is a noisier model than the value function an agent
optimized, so even with no over-optimization a gain measured by the value
function shows up smaller under the judge (regression dilution). The
benchmark: on real games neither side trained on (the paper's 38,981 test
replays), regress each judge's prediction on the value function's prediction,
P(team0 wins), both team-order symmetrized:

    slope = cov(judge, proxy) / var(proxy)

A draft-quality gain of d under the proxy is expected to appear as about
slope * d under the judge if the proxy's errors on the agents' drafts are like
its errors on human drafts. A judge gain well below slope * d is
over-optimization; a gain near it is fully transmitted.

Proxies: the leak-free enriched WP with own deploy statistics (what B/F/J_oof
optimized) and the submission's enriched WP with external statistics
(proxy_sub; what the submitted configurations optimized; its statistics
contain the test games, so its slope is reported for completeness).
Judges: v1 and composition-corrected v2 (overfit2026/judges_v2.py).

Usage (from training/): nice -n 19 taskset -c 48-53 python3 paper1_revision/judge_attenuation.py
Output: results/judge_attenuation.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np

from paper1_revision import core


def slope(j, p):
    j, p = np.asarray(j, float), np.asarray(p, float)
    c = np.cov(j, p)
    return float(c[0, 1] / c[1, 1])


def boot_se(j, p, n=200, seed=0):
    rng = np.random.RandomState(seed)
    v = [slope(j[i], p[i]) for i in (rng.randint(0, len(j), len(j)) for _ in range(n))]
    return float(np.std(v, ddof=1))


def main():
    from paper1_revision.score_tournament import score_rows
    from paper1_revision.bench_mcts import sub_enriched, sym_predict
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    from overfit2026 import judges_v2
    games = core.split_games()["test"]
    rows = [(list(g[3]), list(g[4]), g[2], g[1]) for g in games]
    sc = score_rows(rows)
    Xf, Xs = core.featurize(rows, core.load_stats("hp"))
    proxies = {"proxy_rev": sc["self_enriched"],
               "proxy_sub": sym_predict(sub_enriched(), core.group_cols(ENRICHED_GROUPS), Xf, Xs)}
    v1 = {j: sc[j] for j in ("gN", "gN_naive", "RN", "QM2026")}
    v1["consensus"] = np.mean([v1[j] for j in judges_v2.CONSENSUS], 0)
    v2 = judges_v2.correct(v1, rows)
    out = {"n_games": len(rows)}
    for pn, p in proxies.items():
        out[pn] = {}
        for tag, js in (("v1", v1), ("v2", v2)):
            for j, v in js.items():
                out[pn][f"{j}|{tag}"] = {"slope": slope(v, p), "se": boot_se(np.asarray(v), np.asarray(p))}
        print(pn, {k: round(x["slope"], 3) for k, x in out[pn].items()}, flush=True)
    json.dump(out, open(os.path.join(core.RESULTS, "judge_attenuation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
