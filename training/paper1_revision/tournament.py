"""
Revised round-robin tournament (12 strategies, every ordered pair, 200 drafts,
maps/tiers stratified exactly as rerun2026/phase3b_roundrobin.py).

Changes from the submission's tournament
  * Agents that carry a win-probability model use the revision's leak-free
    models and own deploy statistics: enriched greedy, enriched+aug greedy,
    constrained greedy, MCTS, constrained MCTS.
  * MCTS and constrained MCTS use the SAME checkpoints: the five leak-free
    F_oof (400 training sims, 300K episodes) seeds, pooled (draft i uses seed
    i mod 5). The configuration and the pooling rule were fixed before any
    tournament or reference score of the retrained agents existed.
  * Both MCTS actors play the policy-head argmax with no search at decision
    time (as in the submission, where the text said otherwise).
  * K_truebase (the submission's hero-identity-WP MCTS agent, 15 seeds pooled)
    is added, so features vs. no features is tested head to head.
  * Pairs in which neither side changed (the six baselines below) reuse the
    submission's draft records; only their scores are recomputed.
  * Judges are computed afterwards by score_tournament.py from the recorded
    teams; none of them is any contestant's value function.

Unchanged strategies (submission checkpoints; no aggregate statistics enter
their policies except cql_enr_a2.0, whose features keep the submission's
external statistics): gourdeau, gourdeau_disc, cql_naive_a1.0, cql_enr_a2.0,
gd, mcq_t0.5.

Interaction metrics per side (counter, synergy, resilience) use the external
Heroes Profile statistics, as in the submission, so reused and new records are
on one scale; they are evaluation instruments only.

Usage: python3 paper1_revision/tournament.py --pair A__B
       python3 paper1_revision/tournament.py --list-todo
"""
import os
import sys
import json
import time
import random
import argparse
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core

import numpy as np
import torch

STRATEGIES = ["mcts", "constrained_mcts", "enriched", "enriched_aug", "constrained_greedy",
              "k_truebase", "gourdeau", "gourdeau_disc", "cql_naive_a1.0", "cql_enr_a2.0",
              "gd", "mcq_t0.5"]
UNCHANGED = {"gourdeau", "gourdeau_disc", "cql_naive_a1.0", "cql_enr_a2.0", "gd", "mcq_t0.5"}
MCTS_CONFIG = "F_oof"
MCTS_SEEDS = [0, 1, 2, 3, 4]
K_SEEDS = list(range(15))
OUT = os.path.join(core.RESULTS, "tournament")
RR = os.path.join(TRAINING_DIR, "rerun2026")

CTX = {"draft": 0}


def pair_path(a, b):
    return os.path.join(OUT, f"{a}__{b}.json")


def reused_path(a, b):
    return os.path.join(RR, "results", "roundrobin", f"{a}__{b}.json")


def load_policy(path):
    from rerun2026.constrained_search import load_mcts_policy
    return load_mcts_policy(path)


def pooled_policy(paths):
    from rerun2026.constrained_search import make_mcts_policy_strategy
    strats = [make_mcts_policy_strategy(load_policy(p)) for p in paths]

    def strategy(state, team, step_type, game_map, tier, gd_models, device):
        return strats[CTX["draft"] % len(strats)](state, team, step_type, game_map, tier,
                                                  gd_models, device)
    return strategy


def build(name, device, gd_models):
    from rerun2026 import phase3b_roundrobin as p3b
    from rerun2026.constrained_search import constrain
    from experiment_rich_evaluation import make_wp_greedy_strategy
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    from sweep_enriched_wp import compute_group_indices
    from paper1_revision import train_wp
    gi = compute_group_indices()
    if name in UNCHANGED:
        act = p3b.build_actor(name, device, core.load_stats("hp"), gi, gd_models, None)
        return act
    if name in ("mcts", "constrained_mcts"):
        paths = [os.path.join(core.MCTS_RUNS, f"{MCTS_CONFIG}_s{s}", "draft_policy.pt")
                 for s in MCTS_SEEDS]
        inner = pooled_policy(paths)
        if name == "constrained_mcts":
            inner = constrain(inner)
    elif name == "k_truebase":
        paths = [os.path.join(RR, "mcts_runs", f"K_truebase_s{s}", "draft_policy.pt")
                 for s in K_SEEDS]
        inner = pooled_policy(paths)
    else:
        wp = {"enriched": "enriched", "constrained_greedy": "enriched",
              "enriched_aug": "aug_wr10_512"}[name]
        m, cols = train_wp.load(wp, device=device)
        inner = make_wp_greedy_strategy(m, ENRICHED_GROUPS, core.load_stats("deploy"), gi, device)
        if name == "constrained_greedy":
            inner = constrain(inner)

    def act(state, team, step_type, game_map, tier):
        return inner(state, team, step_type, game_map, tier, gd_models, device)
    return act


def run_pair(a, b, n):
    from rerun2026 import phase3b_roundrobin as p3b
    from rerun2026.phase3_benchmarks import load_gd_models
    from shared import HEROES, NUM_HEROES, MAPS, SKILL_TIERS
    from train_draft_policy import DraftState, DRAFT_ORDER
    device = torch.device("cpu")
    gd_models = load_gd_models(device)
    act_a, act_b = build(a, device, gd_models), build(b, device, gd_models)
    pair_idx = STRATEGIES.index(a) * len(STRATEGIES) + STRATEGIES.index(b)
    random.seed(1000 + pair_idx)
    torch.manual_seed(1000 + pair_idx)
    hp = core.load_stats("hp")
    records = []
    for i in range(n):
        CTX["draft"] = i
        game_map = MAPS[i % len(MAPS)]
        tier = SKILL_TIERS[(i // len(MAPS)) % len(SKILL_TIERS)]
        state = DraftState(game_map, tier, our_team=0)
        picks0, picks1 = [], []
        while not state.is_terminal():
            step_team, step_type = DRAFT_ORDER[state.step]
            step_num = state.step
            state.our_team = step_team
            actor = act_a if step_team == 0 else act_b
            h = actor(state, step_team, step_type, game_map, tier)
            if step_type == "pick":
                (picks0 if step_team == 0 else picks1).append((HEROES[h], step_num))
            state.apply_action(h, step_team, step_type)
        records.append({
            "draft": i, "game_map": game_map, "tier": tier,
            "team0": p3b.side_metrics(picks0, picks1, hp, tier),
            "team1": p3b.side_metrics(picks1, picks0, hp, tier),
            "team0_steps": picks0, "team1_steps": picks1,
            "bans": [HEROES[j] for j in range(NUM_HEROES) if state.bans[j] > 0],
        })
    return {"team0_strategy": a, "team1_strategy": b, "n_drafts": n,
            "mcts_config": MCTS_CONFIG, "mcts_seeds": MCTS_SEEDS, "records": records}


def todo():
    out = []
    for a, b in itertools.permutations(STRATEGIES, 2):
        if a in UNCHANGED and b in UNCHANGED:
            continue
        if not os.path.exists(pair_path(a, b)):
            out.append(f"{a}__{b}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair")
    ap.add_argument("--drafts", type=int, default=200)
    ap.add_argument("--list-todo", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.list_todo:
        print("\n".join(todo()))
        return
    torch.set_num_threads(1)
    x, y = a.pair.split("__")
    if os.path.exists(pair_path(x, y)):
        return
    t0 = time.time()
    res = run_pair(x, y, a.drafts)
    res["secs"] = time.time() - t0
    json.dump(res, open(pair_path(x, y), "w"))
    print(f"{a.pair}: {a.drafts} drafts in {res['secs']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
