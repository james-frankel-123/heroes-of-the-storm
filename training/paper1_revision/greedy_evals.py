"""
Greedy-search evaluations with the revision's leak-free models (own deploy
statistics for features; external statistics for the interaction METRICS, as
in the submission's Table IV, so rows stay on one scale).

  rich <strategy> <seed>   Table IV row: 1000 drafts for one eval seed,
                           (mcts / constrained_mcts: F_oof policy argmax, eval
                           seed s uses checkpoint seed s; constrained_greedy:
                           the role mask on enriched greedy)
                           experiment_rich_evaluation.run_drafts_with_strategy
                           (GD opponents), draft configs random.seed(42) as in
                           the submission. strategies: enriched (256x128),
                           enriched_aug (aug_wr10_512).
  wr <config> <seed>       Table V row: 28-test sanity suite + 5-tank WP, and
                           1000 greedy drafts (GD-argmax rollouts/opponents,
                           rerun2026 phase3 task_wr_sweep protocol) for one
                           eval seed. configs: enriched_512 (no augmentation),
                           aug_wr{0,5,10,50}_512.
  aggregate                results/greedy_rich.json, results/wr_sweep.json
Usage: python3 paper1_revision/greedy_evals.py pool [--procs 10]
"""
import os
import sys
import json
import random
import argparse
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core

import numpy as np
import torch

OUT = os.path.join(core.RESULTS, "greedy")
RICH = {"enriched": "enriched", "enriched_aug": "aug_wr10_512", "constrained_greedy": "enriched",
        "mcts": None, "constrained_mcts": None}
WR = ["enriched_512", "aug_wr0_512", "aug_wr5_512", "aug_wr10_512", "aug_wr50_512"]
RICH_SEEDS = range(5)
WR_SEEDS = range(2)


def task_rich(strategy, seed):
    from experiment_rich_evaluation import make_wp_greedy_strategy, run_drafts_with_strategy
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    from sweep_enriched_wp import compute_group_indices
    from rerun2026.phase3_benchmarks import load_gd_models
    from rerun2026.constrained_search import (constrain, make_mcts_policy_strategy,
                                              load_mcts_policy)
    from shared import MAPS, SKILL_TIERS
    from paper1_revision import train_wp
    from paper1_revision.tournament import MCTS_CONFIG
    dev = torch.device("cpu")
    st = core.load_stats("deploy")
    if strategy in ("mcts", "constrained_mcts"):
        # eval seed s plays MCTS checkpoint seed s (policy argmax, no search)
        net = load_mcts_policy(os.path.join(core.MCTS_RUNS, f"{MCTS_CONFIG}_s{seed}",
                                            "draft_policy.pt"))
        fn = make_mcts_policy_strategy(net)
        if strategy == "constrained_mcts":
            fn = constrain(fn)
    else:
        m, cols = train_wp.load(RICH[strategy])
        fn = make_wp_greedy_strategy(m, ENRICHED_GROUPS, st, compute_group_indices(), dev)
        if strategy == "constrained_greedy":
            fn = constrain(fn)
    gd = load_gd_models(dev)
    random.seed(42)
    configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2) for i in range(1000)]
    random.seed(seed)
    torch.manual_seed(seed)
    drafts = run_drafts_with_strategy(fn, configs, gd, st, dev)
    keep = []
    for d in drafts:
        steps = [{"hero_idx": s["hero_idx"], "state": [round(float(x), 4) for x in s["state"]],
                  "mask": [int(x) for x in s["mask"]]}
                 for s in d["steps"] if s["is_ours"] and s["type"] != "ban"]
        keep.append({"our": d["our_picks"], "opp": d["opp_picks"], "map": d["game_map"],
                     "tier": d["tier"], "has_healer": d["has_healer"], "is_degen": d["is_degen"],
                     "our_pick_steps": steps})
    return keep


def task_wr(config, seed):
    from experiment_synthetic_augmentation import evaluate_config
    from experiment_synthetic_augmentation import make_eval_fn
    from rerun2026.phase3_benchmarks import load_gd_models
    from train_draft_policy import DraftState, DRAFT_ORDER
    from shared import MAPS, SKILL_TIERS, HEROES, NUM_HEROES, HERO_ROLE_FINE, is_degenerate
    from paper1_revision import train_wp
    dev = torch.device("cpu")
    m, cols = train_wp.load(config)
    st = core.load_stats("deploy")
    ecols = list(cols[197:] - 197)
    stage1 = evaluate_config(m, ecols, st, dev) if seed == 0 else None
    eval_fn = make_eval_fn(m, ecols, st, dev)
    gd_models = load_gd_models(dev)
    healers = {h for h, r in HERO_ROLE_FINE.items() if r == "healer"}
    ranged = {h for h, r in HERO_ROLE_FINE.items() if r in ("ranged_aa", "ranged_mage", "pusher")}

    def gd_pick(state):
        g = random.choice(gd_models)
        with torch.no_grad():
            return g(state.to_tensor_gd(dev), state.valid_mask(dev)).argmax(dim=1).item()

    def greedy(state, team, our):
        mask = state.valid_mask_np()
        best, bw = None, -1
        for h in [i for i in range(NUM_HEROES) if mask[i] > 0]:
            s = state.clone()
            s.apply_action(h, team, "pick")
            for rt, rty in DRAFT_ORDER[s.step:]:
                s.apply_action(gd_pick(s), rt, rty)
            t0 = [HEROES[i] for i in range(NUM_HEROES) if s.team0_picks[i] > 0]
            t1 = [HEROES[i] for i in range(NUM_HEROES) if s.team1_picks[i] > 0]
            w = eval_fn(t0, t1, state.game_map, state.skill_tier)
            w = 1 - w if our == 1 else w
            if w > bw:
                bw, best = w, h
        return best

    random.seed(seed)
    torch.manual_seed(seed)
    cfgs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2) for i in range(1000)]
    teams = []
    for _, gm, tier, our in cfgs:
        state = DraftState(gm, tier, our_team=our)
        while not state.is_terminal():
            team, typ = DRAFT_ORDER[state.step]
            h = greedy(state, team, our) if (team == our and typ == "pick") else gd_pick(state)
            state.apply_action(h, team, typ)
        v = state.team0_picks if our == 0 else state.team1_picks
        teams.append([HEROES[i] for i in range(NUM_HEROES) if v[i] > 0])
    return {"stage1": stage1, "healer": 100 * np.mean([any(h in healers for h in t) for t in teams]),
            "ranged": 100 * np.mean([any(h in ranged for h in t) for t in teams]),
            "degen": 100 * np.mean([is_degenerate(t) for t in teams]), "teams": teams}


def path(kind, name, seed):
    return os.path.join(OUT, f"{kind}_{name}_s{seed}.json")


def ready(n):
    if n in ("mcts", "constrained_mcts"):
        from paper1_revision.run_tournament import ready as tready
        return tready("mcts")
    return os.path.exists(os.path.join(core.MODEL_DIR, f"{RICH[n]}.json"))


def jobs():
    out = []
    for s in RICH_SEEDS:
        for n in RICH:
            if ready(n):
                out.append(("rich", n, s))
    for s in WR_SEEDS:
        for n in WR:
            if os.path.exists(os.path.join(core.MODEL_DIR, f"{n}.json")):
                out.append(("wr", n, s))
    return [j for j in out if not os.path.exists(path(*j))]


def run_job(j):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    log = open(os.path.join(core.LOGS, f"greedy_{j[0]}_{j[1]}_s{j[2]}.log"), "w")
    r = subprocess.run([sys.executable, "-u", __file__, j[0], j[1], str(j[2])], cwd=TRAINING_DIR,
                       env=env, stdout=log, stderr=subprocess.STDOUT)
    return j, r.returncode


def aggregate():
    from experiment_rich_evaluation import counter_responsiveness, synergy_exploitation
    from rerun2026.phase3_benchmarks import load_gd_models
    from paper1_revision.score_tournament import score_rows
    hp = core.load_stats("hp")
    gd = load_gd_models(torch.device("cpu"))
    res = {}
    for n in RICH:
        drafts = []
        for s in RICH_SEEDS:
            p = path("rich", n, s)
            if os.path.exists(p):
                drafts += json.load(open(p))
        if not drafts:
            continue
        c = Counter(h for d in drafts for h in d["our"])
        tot = sum(c.values())
        pr = np.array(list(c.values())) / tot
        agree = total = 0
        for d in drafts:
            for st in d["our_pick_steps"]:
                s_t = torch.tensor(st["state"], dtype=torch.float32).unsqueeze(0)
                m_t = torch.tensor(st["mask"], dtype=torch.float32).unsqueeze(0)
                votes = Counter()
                with torch.no_grad():
                    for g in gd:
                        votes[g(s_t, m_t).argmax(dim=1).item()] += 1
                agree += st["hero_idx"] == votes.most_common(1)[0][0]
                total += 1
        rows = [(d["our"], d["opp"], d["map"], d["tier"]) for d in drafts]
        sc = score_rows(rows)
        res[n] = {"n": len(drafts),
                  "healer": 100 * np.mean([d["has_healer"] for d in drafts]),
                  "degen": 100 * np.mean([d["is_degen"] for d in drafts]),
                  "counter": float(np.mean([counter_responsiveness(d["our"], d["opp"], hp, d["tier"])
                                            for d in drafts])),
                  "synergy": float(np.mean([synergy_exploitation(d["our"], hp, d["tier"])
                                            for d in drafts])),
                  "distinct": len(c), "entropy": float(-(pr * np.log2(pr)).sum()),
                  "top10": 100 * sum(v for _, v in c.most_common(10)) / tot,
                  "gd_similarity": 100 * agree / max(total, 1),
                  "refs": {k: float(np.mean(v)) for k, v in sc.items()}}
        print(n, {k: v for k, v in res[n].items() if k != "refs"}, flush=True)
    json.dump(res, open(os.path.join(core.RESULTS, "greedy_rich.json"), "w"), indent=1)
    wr = {}
    for n in WR:
        rs = [json.load(open(path("wr", n, s))) for s in WR_SEEDS if os.path.exists(path("wr", n, s))]
        if not rs:
            continue
        meta = json.load(open(os.path.join(core.MODEL_DIR, f"{n}.json")))
        s1 = rs[0]["stage1"]
        wr[n] = {"test_acc": meta["test_mean"]["acc"], "NODRIFT_acc": meta["NODRIFT_mean"]["acc"],
                 "NODRIFT_slope": meta["NODRIFT_mean"]["slope"],
                 "sanity": f"{s1['sanity_passed']}/{s1['sanity_total']}" if s1 else None,
                 "degen_scores": s1["degen_scores"] if s1 else None,
                 "healer": float(np.mean([r["healer"] for r in rs])),
                 "ranged": float(np.mean([r["ranged"] for r in rs])),
                 "degen": float(np.mean([r["degen"] for r in rs])),
                 "degen_per_seed": [r["degen"] for r in rs], "n_drafts": 1000 * len(rs)}
        print(n, {k: v for k, v in wr[n].items() if k != "degen_scores"}, flush=True)
    json.dump(wr, open(os.path.join(core.RESULTS, "wr_sweep.json"), "w"), indent=1)


def main():
    if sys.argv[1] == "pool":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--procs", type=int, default=10)
        a = ap.parse_args()
        os.makedirs(OUT, exist_ok=True)
        with ThreadPoolExecutor(a.procs) as ex:
            for j, rc in ex.map(run_job, jobs()):
                print(j, rc, flush=True)
    elif sys.argv[1] == "aggregate":
        aggregate()
    else:
        kind, name, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
        torch.set_num_threads(1)
        res = task_rich(name, seed) if kind == "rich" else task_wr(name, seed)
        os.makedirs(OUT, exist_ok=True)
        json.dump(res, open(path(kind, name, seed), "w"))


if __name__ == "__main__":
    main()
