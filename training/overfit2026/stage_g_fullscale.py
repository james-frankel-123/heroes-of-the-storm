"""
Stage G: paper-scale proxies (whole snapshot) — the paper's own WP
(wp_enriched_256 + frozen HP stats), a replication with own in-sample stats
(pS8_leak), and the leakage-free version (pS8_oof, 2-fold out-of-fold stats).

1. Held-out quality on post-snapshot no-drift games (BF + T97).
2. Search with each as the leaf value (BC prior, argmax root, 256/1024/4096
   sims, same 400 configs as stage E), scored on post-snapshot gold only:
   gN (judges trained on no-drift post-snapshot games), RN, R17 (2.55.17
   realized index), QM2026.
Output: results/fullscale.json
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np

from overfit2026 import search, score, feats, data, gold
from overfit2026.stage_e import N_DRAFTS, SEED, policy_only

PAPER = os.path.join(TRAINING_DIR, "rerun2026", "models", "wp_enriched_256.pt")


def proxies():
    return {
        "paper": (feats.load_wp(PAPER), feats.paper_stats()),
        "pS8_leak": (score.model("pS8_leak")[0], score._stats("S8")),
        "pS8_oof": (score.model("pS8_oof")[0], score._stats("S8")),
    }


def heldout(P):
    fut, bf = data.load_future(), data.load_backfill()
    games = [g for v in bf.values() for g in v] + list(fut["2.55.16.97039"])
    rows = [(g[3], g[4], g[2], g[1]) for g in games]
    y = np.array([1.0 if g[6] == 0 else 0.0 for g in games])
    out = {}
    for n, (m, st) in P.items():
        Xf, Xs = feats.featurize(rows, st)
        p = np.clip(feats.predict_sym(m, Xf, Xs, "cuda"), 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p))
        c, _ = gold.fit_logistic(z[:, None], y)
        out[n] = {"acc": float(np.mean((p > .5) == (y > .5))),
                  "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                  "calib_slope": float(c[1]), "mean_abs_logit": float(np.abs(z).mean())}
        print(n, out[n], flush=True)
    return out


def gold_scores(drafts, P):
    rows = [(tuple(d["our"]), tuple(d["opp"]), d["map"], d["tier"]) for d in drafts]
    GN = ["gN8_oof_s0", "gN8_oof_s1", "gN8_oof_s2"]
    ms = score.model_scores(GN + ["gN8_naive"], rows)
    sc = {"gN": np.mean([ms[n] for n in GN], 0), "gN_naive": ms["gN8_naive"]}
    sc["RN"] = np.array([score.realized("NODRIFT").score(o, p, t) for o, p, m, t in rows])
    sc["R17"] = np.array([score.realized("T17").score(o, p, t) for o, p, m, t in rows])
    from overfit2026.stage_b_saved import qm_scores
    sc.update(qm_scores(score.qm(), rows))
    for n, (m, st) in P.items():
        Xf, Xs = feats.featurize(rows, st)
        sc[f"proxy:{n}"] = feats.predict_sym(m, Xf, Xs, "cuda")
    from shared import is_degenerate
    sc["degen"] = np.array([float(is_degenerate(list(o))) for o, p, m, t in rows])
    return sc


def main():
    P = proxies()
    res = {"heldout": heldout(P), "runs": {}}
    kernel = search.load_kernel("ofit")
    pf = search.policy_flat("bc")
    base = policy_only("bc", N_DRAFTS, SEED, argmax=True)
    res["runs"]["bc_policy_only"] = {k: [float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))]
                                     for k, v in gold_scores(base, P).items()}
    for n, (m, st) in P.items():
        wcfg = search.wp_config([m], st)
        for sims in (256, 1024, 4096):
            t0 = time.time()
            drafts = search.run(kernel, pf, wcfg, n=N_DRAFTS, sims=sims, seed=SEED,
                                root_temp=0.0, guard=1)
            sc = gold_scores(drafts, P)
            key = f"{n}__sims{sims}"
            res["runs"][key] = {k: [float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))]
                                for k, v in sc.items()}
            res["runs"][key]["per_draft"] = {k: np.round(v, 5).tolist() for k, v in sc.items()}
            r = res["runs"][key]
            print(f"{key:22s} self={r[f'proxy:{n}'][0]:.4f} gN={r['gN'][0]:.4f} "
                  f"gNnaive={r['gN_naive'][0]:.4f} RN={r['RN'][0]:.4f} R17={r['R17'][0]:.4f} "
                  f"QM={r['QM2026'][0]:.4f} deg={r['degen'][0]:.3f} ({time.time() - t0:.0f}s)",
                  flush=True)
    json.dump(res, open(os.path.join(HERE, "results", "fullscale.json"), "w"))


if __name__ == "__main__":
    main()
