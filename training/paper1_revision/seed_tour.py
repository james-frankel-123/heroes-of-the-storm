"""
Per-seed tournament drafts for the floor-trained (F_floor) and ordinary (F_oof) MCTS policies, so each
checkpoint is scored on its own drafts instead of its share of a pooled run (tournament.pooled_policy
plays draft i with checkpoint i % n_seeds, which ties the seed to the map/tier cycle).

Entrants "<base>@s<k>": one checkpoint, the play-time wrapper of the pooled strategy of the same base:
  constrained_mcts        F_oof_s<k>, role mask (tournament.build)
  constrained_mcts_sf     F_oof_s<k>, role mask + support floor (support_mitigation.floor_constrain)
  search_sf               floor_search search_sf with the F_oof_s<k> prior
  constrained_mcts_ft     F_floor_s<k>, role mask
  constrained_mcts_ft_sf  F_floor_s<k>, role mask + support floor
  search_ft_sf            floor_search search_ft_sf with the F_floor_s<k> prior
Opponents: OPP8 minus constrained_mcts (the worlds rescoring's common set), both seats, 200 drafts per
ordered pair, maps/tiers as tournament.py. The pair seed depends on (base, opponent, seat) only, so all
checkpoints of a base face the same opponent randomness (common random numbers across seeds).

Usage (from training/, p1site env, CPU):
  python3 paper1_revision/seed_tour.py pair 'constrained_mcts_ft@s3__gd' [200]
  python3 paper1_revision/seed_tour.py queue --procs 6 --kinds policy [--floor-seeds 0-7] [--oof-seeds 0-7]
  python3 paper1_revision/seed_tour.py status
A checkpoint is used only once its training log says it completed (originals: always).
Output: site/results/seed_tour/tournament/<a>__<b>.json (tournament record format).
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core

OUT = os.path.join(core.RESULTS, "seed_tour", "tournament")
OPP7 = ["mcts", "enriched", "constrained_greedy", "k_oof", "gourdeau", "gourdeau_disc", "gd"]
BASES = ["constrained_mcts", "constrained_mcts_sf", "search_sf", "constrained_mcts_ft", "constrained_mcts_ft_sf",
         "search_ft_sf"]
SEARCH = {"search_sf", "search_ft_sf"}
SEED_OFFSET = 20000
ORIGINAL = {"F_oof": range(5), "F_floor": range(3)}


def cfg_of(base):
    return "F_floor" if "_ft" in base else "F_oof"


def policy_path(cfg, k):
    return os.path.join(core.MCTS_RUNS, f"{cfg}_s{k}", "draft_policy.pt")


def trained(cfg, k):
    if k in ORIGINAL[cfg]:
        return os.path.exists(policy_path(cfg, k))
    log = os.path.join(core.LOGS, f"mcts_{cfg}_s{k}.log")
    if not (os.path.exists(log) and os.path.exists(policy_path(cfg, k))):
        return False
    with open(log, errors="replace") as f:
        return any(line.startswith("Complete. ") for line in f)


def parse(name):
    base, k = name.split("@s")
    assert base in BASES, name
    return base, int(k)


def pair_seed(a, b):
    """Seed of an ordered pair from (base, opponent, seat), independent of the checkpoint."""
    if "@s" in a:
        return SEED_OFFSET + 100 * BASES.index(parse(a)[0]) + 2 * OPP7.index(b)
    return SEED_OFFSET + 100 * BASES.index(parse(b)[0]) + 2 * OPP7.index(a) + 1


def build_seeded(name, device, gd_models, seed):
    from paper1_revision import tournament as T
    from paper1_revision import support_mitigation as sm
    from paper1_revision import floor_search as fs
    from paper1_revision import search_actor as sa
    from paper1_revision import wr_tour  # noqa: F401  registers constrained_wr in T.build
    if "@s" not in name:
        return fs.build_any(name, device, gd_models, seed)
    base, k = parse(name)
    path = policy_path(cfg_of(base), k)
    if base in SEARCH:
        # one pair per process: point the prior of floor_search.build_actor at this checkpoint only
        if base == "search_ft_sf":
            sm.FT_SEEDS = [k]
        else:
            sa.f_oof_paths = lambda seeds=None: [path]
        return fs.build_actor(base, T.CTX, seed)
    from rerun2026.constrained_search import constrain
    pol = T.pooled_policy([path])
    strat = sm.floor_constrain(pol) if base.endswith("_sf") else constrain(pol)

    def act(state, team, step_type, game_map, tier):
        return strat(state, team, step_type, game_map, tier, gd_models, device)
    return act


def run_pair(a, b, n):
    from paper1_revision import search_eval as se
    from paper1_revision import support_mitigation as sm
    from paper1_revision import floor_search as fs
    seed = pair_seed(a, b)
    se.all_names = lambda: [a, b]          # pair index 1 -> seed = SEED_OFFSET + 1
    se.SEED_OFFSET = seed - 1
    se.build_any = lambda name, device, gd_models, pair_seed_: build_seeded(name, device, gd_models, seed)
    R = se.run_pair(a, b, n)
    assert R["seed"] == seed
    for s in (a, b):
        if "@s" in s:
            base, k = parse(s)
            R.setdefault("checkpoints", {})[s] = policy_path(cfg_of(base), k)
    R["floor_fallbacks"] = dict(fs.FALLBACKS)
    R["play_floor_fallbacks"] = dict(sm.FLOOR_FALLBACKS)
    return R


def pair_path(a, b):
    return os.path.join(OUT, f"{a}__{b}.json")


def cmd_pair(spec, n):
    a, b = spec.split("__")
    p = pair_path(a, b)
    if os.path.exists(p):
        print("exists", p)
        return
    t = time.time()
    R = run_pair(a, b, n)
    R["secs"] = time.time() - t
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
    print(f"{a}__{b} {R['secs']:.0f}s", flush=True)


def seeds(spec):
    lo, hi = (int(x) for x in spec.split("-"))
    return range(lo, hi + 1)


def entrants(kinds, floor_seeds, oof_seeds):
    out = []
    for base in BASES:
        if (base in SEARCH) != (kinds == "search"):
            continue
        cfg = cfg_of(base)
        for k in (floor_seeds if cfg == "F_floor" else oof_seeds):
            if trained(cfg, k):
                out.append(f"{base}@s{k}")
    return out


def todo(kinds, floor_seeds, oof_seeds):
    out = []
    for e in entrants(kinds, floor_seeds, oof_seeds):
        for o in OPP7:
            for a, b in ((e, o), (o, e)):
                if not os.path.exists(pair_path(a, b)):
                    out.append(f"{a}__{b}")
    return out


def cmd_queue(procs, n, kinds, floor_seeds, oof_seeds):
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    work = todo(kinds, floor_seeds, oof_seeds)
    print(f"{len(work)} pairs ({kinds})", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")

    def one(s):
        rc = subprocess.call([sys.executable, "-u", os.path.abspath(__file__), "pair", s, str(n)],
                             cwd=TRAINING_DIR, env=env)
        print(s, "rc", rc, flush=True)
        return rc
    with ThreadPoolExecutor(procs) as ex:
        rcs = list(ex.map(one, work))
    print(f"QUEUE DONE ({kinds}) failed={sum(r != 0 for r in rcs)}", flush=True)


def cmd_status():
    for kinds in ("policy", "search"):
        ents = entrants(kinds, range(8), range(8))
        left = todo(kinds, range(8), range(8))
        print(f"{kinds}: {len(ents)} trained entrants, {len(left)} pairs to play")
    for cfg in ("F_oof", "F_floor"):
        print(cfg, "trained seeds:", [k for k in range(8) if trained(cfg, k)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pair", "queue", "status"])
    ap.add_argument("spec", nargs="?")
    ap.add_argument("n", nargs="?", type=int, default=200)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--kinds", choices=["policy", "search"], default="policy")
    ap.add_argument("--floor-seeds", default="0-7")
    ap.add_argument("--oof-seeds", default="0-7")
    a = ap.parse_args()
    if a.cmd == "pair":
        cmd_pair(a.spec, a.n)
    elif a.cmd == "queue":
        cmd_queue(a.procs, a.n, a.kinds, seeds(a.floor_seeds), seeds(a.oof_seeds))
    else:
        cmd_status()
