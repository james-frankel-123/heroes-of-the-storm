"""
Decision-time search agents for paper 1 (reviewer request): the F_oof
checkpoints WITH the training search at decision time (search_actor.py).

  mcts_search   200 sims, leaf = the enriched leak-free WP (deploy stats)
  mcts_lcb      200 sims, leaf = 20-member ensemble mean - 1 sd
  mcts_lcb2     200 sims, leaf = ensemble mean - 2 sd
Checkpoints pooled as in tournament.py (draft i uses F_oof seed i mod 5).

Subcommands (from training/, env P1_TIERS=site RERUN_NS=p1site + P1SITE_ENV):
  bench <value> <ckpt_seed> <temp> <shard> [--nshards 10]
        kernel-benchmark mirror (bench_mcts.py / overfit2026.search.run): the
        same 1,000 configs (seed 20260929, mid tier, random map and side), GD
        pool cycled per batch of 100 (batch b plays GD b mod 5, which is also
        the search's opponent/rollout model, as in the kernel benchmark)
  pair <A__B>          one ordered tournament pair, 200 drafts, tournament.py
                       stratification and record format; seeds 7000 + index
                       in ALL (existing pair seeds untouched)
  rich <agent> <eval_seed>   Table IV protocol (greedy_evals.task_rich):
                       configs random.seed(42), eval seed s plays checkpoint s
  todo                 list missing outputs (pairs, rich, bench shards)
  score_bench / score_tournament / score_rich   (run where the judges live)
Outputs: site/results/search_agents/{bench,tournament,rich}/..., scored
summaries site/results/search_agents/*.json. Nothing existing is overwritten.
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

OUT = os.path.join(core.RESULTS, "search_agents")
NEW = {"mcts_search": "enriched", "mcts_lcb": "lcb:1", "mcts_lcb2": "lcb:2"}
PRIMARY = ["mcts_search", "mcts_lcb"]
SEED_OFFSET = 7000
BENCH_SEED = 20260929


def strategies():
    from paper1_revision.tournament import STRATEGIES
    return list(STRATEGIES)


def all_names():
    return strategies() + list(NEW)


def pairs():
    out = []
    for a, b in itertools.permutations(all_names(), 2):
        if a in NEW or b in NEW:
            out.append((a, b))
    return out


def pair_path(a, b):
    return os.path.join(OUT, "tournament", f"{a}__{b}.json")


def rich_path(agent, s):
    return os.path.join(OUT, "rich", f"rich_{agent}_s{s}.json")


def bench_path(value, ck, temp, shard):
    return os.path.join(OUT, "bench", f"F_oof_s{ck}__{value.replace(':', '')}__T{temp:g}__part{shard}.json")


# ── bench mirror ────────────────────────────────────────────────────────

def run_bench(value, ck, temp, shard, nshards):
    """Lockstep over the shard's drafts (one batched policy forward per search
    round across drafts); per-draft RNG streams, so the order does not matter."""
    import torch
    from overfit2026 import search as osearch
    from paper1_revision import search_actor as sa
    from shared import HEROES, MAPS, SKILL_TIERS
    from train_draft_policy import DraftState
    torch.set_num_threads(1)
    cfgs = osearch.draft_configs(1000, BENCH_SEED, 1)
    idx = [i for i in range(1000) if (i // 100) % nshards == shard % nshards] if nshards == 10 else \
        [i for i in range(1000) if i % nshards == shard]
    pol = sa._policy(sa.f_oof_paths([ck])[0])
    mode, k = sa.parse_value(value)
    leaf = sa._leaf(mode)
    D = []
    for i in idx:
        mi, ti, ou = (int(x) for x in cfgs[i])
        x = sa.state_vec(DraftState(MAPS[mi], SKILL_TIERS[ti], our_team=ou))
        D.append({"i": i, "mi": mi, "ti": ti, "ou": ou, "x": x,
                  "gd": sa.gd_np((i // 100) % 5),            # kernel: batch b runs GD b mod 5
                  "rng": np.random.default_rng([BENCH_SEED, i, int(temp * 10), ck])})
    stats = {"sims": 0, "leaf_steps": 0, "max_depth": 0}
    for step in range(16):
        gens, who = [], []
        for d in D:
            if sa.TEAM_OF[step] == d["ou"]:
                srng = np.random.default_rng([BENCH_SEED, d["i"], int(temp * 10), ck, step])
                s = sa.Search(pol, d["gd"], leaf, MAPS[d["mi"]], SKILL_TIERS[d["ti"]], k_lcb=k)
                gens.append(s.gen(d["x"], d["ou"], srng, root_temp=temp))
                who.append((d, s))
            else:
                a = sa.sample(d["gd"].probs(d["x"]), 1.0 - d["rng"].random())
                d["x"], _ = sa.apply(d["x"], a)
        for (d, s), (a, _) in zip(who, sa.run_batched(gens)):
            d["x"], _ = sa.apply(d["x"], a)
            stats["sims"] += s.stats["sims"]
            stats["leaf_steps"] += s.stats["leaf_steps"]
            stats["max_depth"] = max(stats["max_depth"], s.stats["max_depth"])
    out = []
    for d in D:
        x, ou = d["x"], d["ou"]
        t0 = [HEROES[j] for j in range(90) if x[j] > 0.5]
        t1 = [HEROES[j] for j in range(90) if x[90 + j] > 0.5]
        out.append({"i": d["i"], "our": t0 if ou == 0 else t1, "opp": t1 if ou == 0 else t0,
                    "map": MAPS[d["mi"]], "tier": SKILL_TIERS[d["ti"]], "side": ou})
    return out, stats


# ── tournament pair ─────────────────────────────────────────────────────

def build_any(name, device, gd_models, pair_seed):
    from paper1_revision import tournament as T
    from paper1_revision import search_actor as sa
    if name in NEW:
        return sa.build_search_actor(NEW[name], draft_ctx=T.CTX, seed_base=pair_seed)
    return T.build(name, device, gd_models)


def run_pair(a, b, n):
    import torch
    from paper1_revision import tournament as T
    from rerun2026 import phase3b_roundrobin as p3b
    from rerun2026.phase3_benchmarks import load_gd_models
    from shared import HEROES, NUM_HEROES, MAPS, SKILL_TIERS
    from train_draft_policy import DraftState, DRAFT_ORDER
    assert T.site()
    device = torch.device("cpu")
    gd_models = load_gd_models(device)
    names = all_names()
    pair_idx = names.index(a) * len(names) + names.index(b)
    seed = SEED_OFFSET + pair_idx
    act_a, act_b = build_any(a, device, gd_models, seed), build_any(b, device, gd_models, seed)
    random.seed(seed)
    torch.manual_seed(seed)
    hp = core.load_stats("hp")
    # lockstep over the n drafts, step by step: a search side searches all its
    # positions of the step in one batched round (search_actor.run_batched);
    # other sides are called draft by draft with tournament.CTX set, as in
    # tournament.run_pair. Maps/tiers stratified exactly as tournament.py.
    cfg = [(MAPS[i % len(MAPS)], SKILL_TIERS[(i // len(MAPS)) % len(SKILL_TIERS)]) for i in range(n)]
    states = [DraftState(m, t, our_team=0) for m, t in cfg]
    picks = [([], []) for _ in range(n)]
    for step in range(16):
        step_team, step_type = DRAFT_ORDER[step]
        actor = act_a if step_team == 0 else act_b
        for st in states:
            st.our_team = step_team
        if hasattr(actor, "batch"):
            hs = actor.batch([(i, states[i], step_team, cfg[i][0], cfg[i][1]) for i in range(n)])
        else:
            hs = []
            for i in range(n):
                T.CTX["draft"] = i
                hs.append(actor(states[i], step_team, step_type, cfg[i][0], cfg[i][1]))
        for i, h in enumerate(hs):
            if step_type == "pick":
                picks[i][step_team].append((HEROES[h], step))
            states[i].apply_action(h, step_team, step_type)
    records = []
    for i in range(n):
        picks0, picks1 = picks[i]
        game_map, tier = cfg[i]
        state = states[i]
        records.append({
            "draft": i, "game_map": game_map, "tier": tier,
            "team0": p3b.side_metrics(picks0, picks1, hp, tier),
            "team1": p3b.side_metrics(picks1, picks0, hp, tier),
            "team0_steps": picks0, "team1_steps": picks1,
            "bans": [HEROES[j] for j in range(NUM_HEROES) if state.bans[j] > 0],
        })
    return {"team0_strategy": a, "team1_strategy": b, "n_drafts": n,
            "mcts_config": T.MCTS_CONFIG, "mcts_seeds": T.MCTS_SEEDS,
            "k_config": T.K_CONFIG, "k_seeds": T.K_SEEDS, "seed": seed,
            "search": {k: NEW[k] for k in (a, b) if k in NEW}, "records": records}


# ── rich evaluation (Table IV protocol) ─────────────────────────────────

def run_rich(agent, seed, n=1000):
    """Table IV protocol (greedy_evals.task_rich -> experiment_rich_evaluation.
    run_drafts_with_strategy), run in lockstep over the 1,000 configs: same
    configs (random.seed(42)), same opponent (a GD model drawn with random.choice
    per opponent step, sampled at T=1 with torch.multinomial, global RNG seeded
    by the eval seed), same per-draft fields; the global RNG is consumed draft by
    draft within each step instead of draft after draft."""
    import torch
    import torch.nn.functional as F
    from rerun2026.phase3_benchmarks import load_gd_models
    from shared import MAPS, SKILL_TIERS, HEROES, NUM_HEROES, HERO_ROLE_FINE, is_degenerate, \
        map_to_one_hot, tier_to_one_hot
    from train_draft_policy import DraftState, DRAFT_ORDER
    from paper1_revision import search_actor as sa
    torch.set_num_threads(1)
    act = sa.build_search_actor(NEW[agent], seeds=(seed,), seed_base=9000 + seed)
    gd = load_gd_models(torch.device("cpu"))
    random.seed(42)
    configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2) for i in range(1000)][:n]
    random.seed(seed)
    torch.manual_seed(seed)
    healers = {h for h, r in HERO_ROLE_FINE.items() if r == "healer"}
    states = [DraftState(m, t, our_team=o) for _, m, t, o in configs]
    steps = [[] for _ in configs]
    for step in range(16):
        step_team, step_type = DRAFT_ORDER[step]
        vecs = []
        for st, (_, m, t, o) in zip(states, configs):
            vecs.append((np.concatenate([st.team0_picks.copy(), st.team1_picks.copy(), st.bans.copy(),
                                         map_to_one_hot(m), tier_to_one_hot(t),
                                         [st.step / 15.0, 0.0 if step_type == "ban" else 1.0]]),
                         st.valid_mask_np()))
        ours = [i for i, c in enumerate(configs) if c[3] == step_team]
        hs = dict(zip(ours, act.batch([(i, states[i], step_team, configs[i][1], configs[i][2])
                                        for i in ours])))
        for i, (st, (_, m, t, o)) in enumerate(zip(states, configs)):
            if i not in hs:
                g = random.choice(gd)
                with torch.no_grad():
                    probs = F.softmax(g(st.to_tensor_gd(torch.device("cpu")),
                                        st.valid_mask(torch.device("cpu"))) / 1.0, dim=1)
                    hs[i] = torch.multinomial(probs, 1).item()
            steps[i].append({"type": step_type, "hero_idx": hs[i], "state": vecs[i][0], "mask": vecs[i][1],
                             "is_ours": o == step_team})
            st.apply_action(hs[i], step_team, step_type)
    keep = []
    for st, (_, m, t, o), sts in zip(states, configs, steps):
        ov = st.team0_picks if o == 0 else st.team1_picks
        pv = st.team1_picks if o == 0 else st.team0_picks
        our = [HEROES[j] for j in range(NUM_HEROES) if ov[j] > 0]
        opp = [HEROES[j] for j in range(NUM_HEROES) if pv[j] > 0]
        keep.append({"our": our, "opp": opp, "map": m, "tier": t,
                     "has_healer": any(h in healers for h in our), "is_degen": bool(is_degenerate(our)),
                     "our_pick_steps": [{"hero_idx": s_["hero_idx"],
                                         "state": [round(float(x), 4) for x in s_["state"]],
                                         "mask": [int(x) for x in s_["mask"]]}
                                        for s_ in sts if s_["is_ours"] and s_["type"] != "ban"]})
    return keep


# ── todo ────────────────────────────────────────────────────────────────

def todo(agents=("mcts_search", "mcts_lcb", "mcts_lcb2")):
    out = []
    for a, b in pairs():
        if (a in NEW and a not in agents) or (b in NEW and b not in agents):
            continue
        if not os.path.exists(pair_path(a, b)):
            out.append(f"pair {a}__{b}")
    for ag in agents:
        for s in range(5):
            if not os.path.exists(rich_path(ag, s)):
                out.append(f"rich {ag} {s}")
    return out


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--nshards", type=int, default=10)
    ap.add_argument("--drafts", type=int, default=200)
    ap.add_argument("--agents", default="mcts_search,mcts_lcb,mcts_lcb2")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    if a.cmd == "bench":
        value, ck, temp, shard = a.args[0], int(a.args[1]), float(a.args[2]), int(a.args[3])
        p = bench_path(value, ck, temp, shard)
        if os.path.exists(p):
            return
        drafts, stats = run_bench(value, ck, temp, shard, a.nshards)
        dump(p, {"value": value, "ckpt_seed": ck, "temp": temp, "shard": shard, "nshards": a.nshards,
                 "drafts": drafts, "search_stats": stats, "secs": time.time() - t0})
    elif a.cmd == "pair":
        x, y = a.args[0].split("__")
        p = pair_path(x, y)
        if os.path.exists(p):
            return
        res = run_pair(x, y, a.drafts)
        res["secs"] = time.time() - t0
        dump(p, res)
    elif a.cmd == "rich":
        ag, s = a.args[0], int(a.args[1])
        p = rich_path(ag, s)
        if os.path.exists(p):
            return
        dump(p, run_rich(ag, s))
    elif a.cmd == "todo":
        print("\n".join(todo(tuple(a.agents.split(",")))))
        return
    elif a.cmd.startswith("score_"):
        from paper1_revision import search_score
        getattr(search_score, a.cmd)()
        return
    else:
        raise SystemExit(a.cmd)
    print(f"{a.cmd} {' '.join(a.args)}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
