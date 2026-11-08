"""
Matched-state opponent-swap test of interaction-awareness (reviewer request).

For each of N base states: a draft is played by the behavioral-cloning pool
(GD, sampled at temperature 1) up to one of our picks (team 0, steps 7, 8, 13,
14). A counterfactual state keeps the map, tier, every ban and every one of our
picks, and resamples only the opponent's picks from GD (avoiding our later
picks and bans). Each agent then picks in both states: h for the real
opponents O, h' for the swapped opponents O'.

Crossover score (Heroes Profile counter deltas, the paper's interaction
instrument):  0.5 * [(c(h,O) - c(h',O)) + (c(h',O') - c(h,O'))],
c(x, S) = mean counter delta of hero x against heroes S. A policy that ignores
the opponent scores exactly 0 (pairs where h is taken in O' or h' in O are
dropped, so availability alone never produces a change). Positive = the pick
made for an opponent counters that opponent better than the pick made for the
other one.

Usage (from training/, p1site env):
  python3 paper1_revision/opp_swap.py --states 1000 --agent constrained_mcts
  python3 paper1_revision/opp_swap.py --pool 4      # all agents, then summary
"""
import os
import sys
import json
import random
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core

OUT = os.path.join(core.RESULTS, "opp_swap")
AGENTS = ["constrained_mcts", "mcts", "k_oof", "constrained_k", "constrained_greedy", "enriched",
          "gourdeau", "constrained_wr", "mcq_t0.5", "cql_naive_a1.0", "cql_enr_a2.0", "gd", "gourdeau_disc"]
OUR_STEPS = [7, 8, 13, 14]


def make_states(n, gd_models, seed=7):
    """Pairs of (actions, cf_actions, map, tier, step): action lists up to our pick at `step`."""
    import torch
    from shared import MAPS, SKILL_TIERS
    from train_draft_policy import DraftState, DRAFT_ORDER
    rng = random.Random(seed)
    torch.manual_seed(seed)

    def gd_sample(state, team, forbid=()):
        state.our_team = team
        gd = rng.choice(gd_models)
        mask = state.valid_mask("cpu").clone()
        for f in forbid:
            mask[0, f] = 0
        with torch.no_grad():
            logits = gd(state.to_tensor_gd("cpu"), mask)
        logits = logits.masked_fill(mask < 0.5, -1e9)
        p = torch.softmax(logits[0], 0)
        return int(torch.multinomial(p, 1, generator=None).item())

    out = []
    while len(out) < n:
        i = len(out)
        game_map = MAPS[i % len(MAPS)]
        tier = SKILL_TIERS[(i // len(MAPS)) % len(SKILL_TIERS)]
        step = OUR_STEPS[(i // (len(MAPS) * len(SKILL_TIERS))) % len(OUR_STEPS)]
        s = DraftState(game_map, tier, our_team=0)
        acts = []
        while s.step < step:
            team, typ = DRAFT_ORDER[s.step]
            h = gd_sample(s, team)
            acts.append(h)
            s.apply_action(h, team, typ)
        opp_steps = [k for k in range(step) if DRAFT_ORDER[k] == (1, "pick")]
        ok = False
        for _ in range(20):
            s2 = DraftState(game_map, tier, our_team=0)
            cf = []
            for k in range(step):
                team, typ = DRAFT_ORDER[k]
                if k in opp_steps:
                    h = gd_sample(s2, team, forbid=[acts[j] for j in range(k + 1, step) if j not in opp_steps])
                else:
                    h = acts[k]
                if h in s2.taken:
                    break
                cf.append(h)
                s2.apply_action(h, team, typ)
            else:
                if {acts[k] for k in opp_steps} != {cf[k] for k in opp_steps}:
                    ok = True
                    break
        if ok:
            out.append((acts, cf, game_map, tier, step))
    return out


def replay(acts, game_map, tier):
    from train_draft_policy import DraftState, DRAFT_ORDER
    s = DraftState(game_map, tier, our_team=0)
    for h in acts:
        team, typ = DRAFT_ORDER[s.step]
        s.apply_action(h, team, typ)
    s.our_team = 0
    return s


def run_agent(name, n):
    import torch
    import numpy as np
    torch.set_num_threads(1)
    from rerun2026.phase3_benchmarks import load_gd_models
    from shared import HEROES
    from train_draft_policy import DRAFT_ORDER
    from experiment_rich_evaluation import counter_responsiveness
    from paper1_revision import wr_tour, mk_tour  # noqa: F401  (register constrained_wr, constrained_k)
    from paper1_revision import tournament as T
    device = torch.device("cpu")
    gd_models = load_gd_models(device)
    states = make_states(n, gd_models)
    act = T.build(name, device, gd_models)
    hp = core.load_stats("hp")
    random.seed(123)
    torch.manual_seed(123)
    rows = []
    for i, (a, cf, game_map, tier, step) in enumerate(states):
        T.CTX["draft"] = i
        opp_steps = [k for k in range(step) if DRAFT_ORDER[k] == (1, "pick")]
        O = [HEROES[a[k]] for k in opp_steps]
        O2 = [HEROES[cf[k]] for k in opp_steps]
        h = HEROES[act(replay(a, game_map, tier), 0, "pick", game_map, tier)]
        T.CTX["draft"] = i
        h2 = HEROES[act(replay(cf, game_map, tier), 0, "pick", game_map, tier)]
        valid = h not in O2 and h2 not in O
        c = lambda x, S: float(counter_responsiveness([x], S, hp, tier))
        x = 0.5 * ((c(h, O) - c(h2, O)) + (c(h2, O2) - c(h, O2))) if valid else None
        rows.append({"step": step, "tier": tier, "h": h, "h2": h2, "valid": valid, "cross": x,
                     "c_own": c(h, O)})
    v = [r for r in rows if r["valid"]]
    xs = np.array([r["cross"] for r in v])
    ch = np.array([r["h"] != r["h2"] for r in v])
    res = {"agent": name, "n_states": n, "n_valid": len(v), "changed": float(ch.mean()),
           "cross": float(xs.mean()), "cross_se": float(xs.std(ddof=1) / np.sqrt(len(xs))),
           "cross_if_changed": float(xs[ch].mean()) if ch.any() else 0.0,
           "counter_own": float(np.mean([r["c_own"] for r in rows])),
           "by_step": {str(k): float(np.mean([r["cross"] for r in v if r["step"] == k])) for k in OUR_STEPS},
           "rows": rows}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"))
    print(name, {k: res[k] for k in ("n_valid", "changed", "cross", "cross_se", "cross_if_changed")}, flush=True)


def pool(procs, n):
    logs = os.path.join(core.LOGS, "opp_swap")
    os.makedirs(logs, exist_ok=True)

    def run(a):
        if os.path.exists(os.path.join(OUT, f"{a}.json")):
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with open(os.path.join(logs, f"{a}.log"), "w") as log:
            r = subprocess.run([sys.executable, "-u", __file__, "--agent", a, "--states", str(n)],
                               cwd=TRAINING_DIR, stdout=log, stderr=subprocess.STDOUT, env=env)
        print(a, "rc", r.returncode, flush=True)
    with ThreadPoolExecutor(procs) as ex:
        list(ex.map(run, AGENTS))
    summary()


def summary():
    out = {}
    for a in AGENTS:
        p = os.path.join(OUT, f"{a}.json")
        if os.path.exists(p):
            r = json.load(open(p))
            r.pop("rows")
            out[a] = r
            print(f"{a:20s} valid {r['n_valid']:4d} changed {r['changed']:.3f} cross {r['cross']:+.3f} ± {r['cross_se']:.3f}"
                  f"  if-changed {r['cross_if_changed']:+.3f}")
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent")
    ap.add_argument("--states", type=int, default=1000)
    ap.add_argument("--pool", type=int)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary()
    elif a.agent:
        run_agent(a.agent, a.states)
    elif a.pool:
        pool(a.pool, a.states)


if __name__ == "__main__":
    main()
