"""
Step 4b, paper 1: the retrained arm. In world w (family a, default dials), the whole value-function
pipeline is rerun on outcomes drawn from the world, then the search drafts with it and is scored
under the world's truth. This measures the pipeline's winner's curse with the truth known.

Refit (the cheapest faithful one): the training, val and test games keep their real drafts; their
winners are redrawn from world w. Everything outcome-dependent is rebuilt from the redrawn winners
exactly as paper1_revision/core.py builds it: the out-of-fold statistics (oof0-4) and the deploy
statistics, the train_oof and val features, and the enriched WP (train_wp spec 'enriched', AdamW,
early stopping on val). Simplifications, documented: one seed (42) instead of three; the search
prior (F_oof MCTS policies) and the GD rollout model are not retrained (they were trained against
the real WP); the opponents keep their real models.
Drafting: search_c and search_sf (floor_search, 200 simulations) with the refit leaf against the
common opponent set, N = 42 per seat, the seeds of truthleaf.py; scored by the world's truth, next to
the same agents with the real leaf (truthleaf 'real' run) and with the truth leaf (truthleaf a/default).

Stages (3090; per world: CPU stats + features, GPU training, CPU drafting):
  world --wid W      one world end to end (resumable by artifact)
  queue --wids ...   sequentially
  score              -> site/results/retrain_p1.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, TRAINING_DIR)
import data as WD  # noqa: E402
import model as M  # noqa: E402
import world as W  # noqa: E402

AGENTS = ["search_c", "search_sf"]
RT = os.path.join(WD.CACHE, "retrain")


def wdir(wid, *p):
    d = os.path.join(RT, f"w{wid}", *p)
    os.makedirs(d if not os.path.splitext(d)[1] else os.path.dirname(d), exist_ok=True)
    return d


def patch_core(wid):
    """Point paper1_revision.core at this world's stats / features / models and redraw the winners."""
    from paper1_revision import core
    G = core.split_games()
    Ws = W.Worlds()
    out = {}
    for k, games in G.items():
        D = WD.from_rows([(g[3], g[4], g[2], g[1]) for g in games])
        p = M.sig(Ws.logit("a", D, wid))
        rng = np.random.default_rng([int(wid), 31, ["train", "val", "test"].index(k)])
        y0 = rng.random(len(p)) < p
        out[k] = [g[:6] + (0 if a else 1,) + tuple(g[7:]) for g, a in zip(games, y0)]
    core.split_games = lambda: out
    core.STATS_DIR = wdir(wid, "stats")
    core.FEAT_DIR = wdir(wid, "feats")
    core.MODEL_DIR = wdir(wid, "models")
    core.CACHE = wdir(wid)
    core._STATS.clear()
    return core


def refit(wid):
    import torch
    core = patch_core(wid)
    t0 = time.time()
    core.build_stats()
    core.build_features(("train_oof", "val"))
    print(f"w{wid}: stats + features {time.time() - t0:.0f}s", flush=True)
    from paper1_revision import train_wp
    train_wp.evaluate = lambda model, cols, fs, device: {"acc": 0.0, "slope": 0.0, "acc_sym": 0.0, "ll": 0.0}
    spec = dict(train_wp.specs()["enriched"], seeds=(42,))
    train_wp.run_spec("enriched", spec, torch.device("cuda"))
    print(f"w{wid}: WP refit {time.time() - t0:.0f}s", flush=True)
    for f in os.listdir(core.FEAT_DIR):          # 5 GB per world; the model and statistics are kept
        os.remove(os.path.join(core.FEAT_DIR, f))


def draft(wid):
    import truthleaf as TLm
    out = os.path.join(wdir(wid), "drafts.json")
    if os.path.exists(out):
        return
    from paper1_revision import core
    from paper1_revision import search_actor as sa
    from paper1_revision import floor_search as fs
    keep = {k: getattr(core, k) for k in ("STATS_DIR", "FEAT_DIR", "MODEL_DIR", "CACHE")}
    core.STATS_DIR, core.MODEL_DIR = wdir(wid, "stats"), wdir(wid, "models")
    core._STATS.clear()
    leaf = sa.LeafWP("enriched")              # the refit WP with its own deploy statistics
    for k, v in keep.items():
        setattr(core, k, v)
    core._STATS.clear()
    sa._SHARED["leaf:enriched"] = leaf
    res = {"world": int(wid), "n": TLm.N_DRAFTS, "pairs": {}, "claimed": {}}
    for a in AGENTS:
        for o in TLm.OPPS:
            for x, y in ((a, o), (o, a)):
                R = fs.run_pair(x, y, TLm.N_DRAFTS)
                recs = [{"m": r["game_map"], "t": r["tier"], "t0": r["team0"]["picks"], "t1": r["team1"]["picks"],
                         "s0": [h for h, _ in r["team0_steps"]], "s1": [h for h, _ in r["team1_steps"]]} for r in R["records"]]
                res["pairs"][f"{x}__{y}"] = recs
                from shared import HERO_TO_IDX
                keys = [(tuple(sorted(HERO_TO_IDX[h] for h in r["t0"])), tuple(sorted(HERO_TO_IDX[h] for h in r["t1"])), r["m"], r["t"])
                        for r in recs]
                res["claimed"][f"{x}__{y}"] = [v for v, _ in leaf.value_batch(keys)]
        print(f"w{wid} {a} drafted", flush=True)
    json.dump(res, open(out, "w"))


def world_job(wid):
    if not os.path.exists(os.path.join(wdir(wid, "models"), "enriched.json")):
        import subprocess
        # refit in its own process (core patched for this world only)
        rc = subprocess.call([sys.executable, "-u", os.path.abspath(__file__), "refit", "--wid", str(wid)], cwd=TRAINING_DIR)
        assert rc == 0, rc
    draft(wid)


def score():
    import rescore as RS
    import truthleaf as TLm
    real = json.load(open(os.path.join(TLm.TL, "real_real_w0.json")))
    rows = []
    for d in sorted(os.listdir(RT)):
        p = os.path.join(RT, d, "drafts.json")
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        w = R["world"]
        tlp = os.path.join(TLm.TL, TLm.tag_of("a", w, "default") + ".json")
        TL = json.load(open(tlp)) if os.path.exists(tlp) else None
        row = {"world": w}
        for a in AGENTS:
            pt = {k: TLm.truth_p("a", w, "default", TLm._arrays(v)) for k, v in R["pairs"].items()}
            row[f"refit:{a}"] = TLm.agent_value(R["pairs"], a, lambda k: pt[k])
            cl = {k: np.asarray(v) for k, v in R["claimed"].items()}
            row[f"refit_claimed:{a}"] = TLm.agent_value(R["pairs"], a, lambda k: cl[k])
            pr = {k: TLm.truth_p("a", w, "default", TLm._arrays(v)) for k, v in real["pairs"].items() if a in k}
            row[f"real:{a}"] = TLm.agent_value(real["pairs"], a, lambda k: pr[k])
            if TL:
                ptl = {k: TLm.truth_p("a", w, "default", TLm._arrays(v)) for k, v in TL["pairs"].items() if a in k}
                row[f"TL:{a}"] = TLm.agent_value(TL["pairs"], a, lambda k: ptl[k])
            row[f"conc:{a}"] = TLm.concentration(R["pairs"], a)
        rows.append(row)
    out = {"worlds": [r["world"] for r in rows], "rows": rows, "summary": {}}
    for a in AGENTS:
        S = {"refit_true_pts": float(100 * np.mean([r[f"refit:{a}"] for r in rows])),
             "refit_claimed_pts": float(100 * np.mean([r[f"refit_claimed:{a}"] for r in rows])),
             "winners_curse_claimed_minus_true": RS.contrast([r[f"refit_claimed:{a}"] - r[f"refit:{a}"] for r in rows]),
             "refit_minus_real_leaf": RS.contrast([r[f"refit:{a}"] - r[f"real:{a}"] for r in rows])}
        if all(f"TL:{a}" in r for r in rows):
            S["regret_refit_vs_truth_leaf"] = RS.contrast([r[f"TL:{a}"] - r[f"refit:{a}"] for r in rows])
            S["regret_real_vs_truth_leaf"] = RS.contrast([r[f"TL:{a}"] - r[f"real:{a}"] for r in rows])
        out["summary"][a] = S
    out["summary"]["floor_refit_sf_minus_c"] = RS.contrast([r["refit:search_sf"] - r["refit:search_c"] for r in rows])
    json.dump(out, open(os.path.join(WD.RESULTS, "retrain_p1.json"), "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["refit", "world", "queue", "score"])
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--wids", default="0,1,2,3,4,5,6,7,8,9")
    a = ap.parse_args()
    if a.stage == "refit":
        refit(a.wid)
    elif a.stage == "world":
        world_job(a.wid)
    elif a.stage == "queue":
        for w in [int(x) for x in a.wids.split(",")]:
            world_job(w)
    else:
        score()
