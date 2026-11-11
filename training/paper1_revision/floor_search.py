"""
Decision-time search under the support floor. The F_oof checkpoints with the
training search (search_actor.Search) at decision time, where our own picks are
restricted INSIDE the tree, not only at the root:

  decision nodes   children = role mask (constrained_search.role_constraint_mask)
                   AND support floor (support_mitigation.floor_mask: every triple
                   with our current picks has >= FLOOR training teams); priors
                   renormalized over the allowed heroes
  rollouts         our picks drawn from GD restricted to the same mask, so every
                   leaf the value is read on is a supported, role-valid team
  bans, opponent   unchanged (GD at temperature 1, progressive widening)
If the floor leaves no hero, the role mask alone is used (counted).

Agents
  search_c       role mask in-tree, no floor, 200 sims
  search_sf      role mask + floor in-tree, 200 sims
  search_sf800   role mask + floor in-tree, 800 sims
Opponents are tournament / support_mitigation strategies (the floored ones are
floored at play time).

Subcommands (from training/, p1site env):
  pair A__B [n]    one ordered pair, n drafts (default 200), tournament.py
                   stratification; seeds 9000 + index; skips existing output
  queue P          run every missing pair of PAIRS with P processes
  h2h              head to head per judge and correction form -> floor_search_h2h.json
Outputs: site/results/support_mitigation/floor_search/{tournament/*.json, floor_search_h2h.json}
"""
import os
import sys
import json
import random
import itertools
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
import numpy as np

from paper1_revision import core

OUT = os.path.join(core.RESULTS, "support_mitigation", "floor_search")
AGENTS = {"search_c": (False, 200), "search_sf": (True, 200), "search_sf800": (True, 800)}
OPPS = ["constrained_wr_sf", "constrained_wr", "constrained_greedy", "gd", "constrained_mcts_sf", "constrained_mcts"]
PAIRS = ([(a, o) for a in ("search_c", "search_sf") for o in OPPS]
         + [("search_sf800", o) for o in ("constrained_wr_sf", "constrained_greedy", "constrained_mcts_sf")]
         + [("search_sf", "search_c"), ("search_sf800", "search_sf")])
PAIRS = PAIRS + [(b, a) for a, b in PAIRS]
SEED_OFFSET = 9000
FALLBACKS = {"floor_empty": 0}


class _View:
    """The attributes role_constraint_mask / floor_mask read, from a state vector."""

    def __init__(self, x):
        from paper1_revision.search_actor import free_mask
        self.team0_picks, self.team1_picks = x[0:90], x[90:180]
        self._free = free_mask(x).astype(np.float32)

    def valid_mask_np(self):
        return self._free


def own_mask(x, our, floor):
    """bool[90]: heroes our team may pick at this state."""
    from rerun2026.constrained_search import role_constraint_mask
    from paper1_revision.support_mitigation import triple_counts, FLOOR
    v = _View(x)
    m = role_constraint_mask(v, our) > 0.5
    if not floor:
        return m
    ours = np.flatnonzero((x[0:90] if our == 0 else x[90:180]) > 0.5)
    if len(ours) >= 2:
        c3 = triple_counts()
        ok = np.ones(90, bool)
        h = np.arange(90)
        for a, b in combinations(ours.tolist(), 2):
            t = np.sort(np.stack([np.full(90, a), np.full(90, b), h]), axis=0)
            cnt = c3[(t[0] * 90 + t[1]) * 90 + t[2]]
            ok &= (cnt >= FLOOR) | (h == a) | (h == b)
        if (m & ok).any():
            m = m & ok
        else:
            FALLBACKS["floor_empty"] += 1
    return m


def make_search_cls():
    from paper1_revision import search_actor as sa

    class FloorSearch(sa.Search):
        floor = True

        def _mask(self, x):
            k = (x[:270].tobytes(), self.our_)
            m = self._mcache.get(k)
            if m is None:
                m = own_mask(x, self.our_, self.floor)
                self._mcache[k] = m
            return m

        def gen(self, x_root, our, rng, root_temp=0.0):
            self.our_ = our
            self._mcache = {}
            self.decision_mask = self._decision_mask
            return super().gen(x_root, our, rng, root_temp)

        def _decision_mask(self, x):
            fm = sa.free_mask(x)
            return fm & self._mask(x) if sa.IS_PICK[sa.step_of(x)] == 1.0 else fm

        def _restrict_prior(self, x, prior):
            if sa.IS_PICK[sa.step_of(x)] == 0.0:
                return prior
            m = self._mask(x)
            p = np.where(m, prior, 0.0)
            s = p.sum()
            return p / s if s > 0 else m / m.sum()

        def _rollout_key(self, x, rng):
            step = sa.step_of(x)
            while step < 16:
                p = self.gd.probs(x)
                if sa.TEAM_OF[step] == self.our_ and sa.IS_PICK[step] == 1.0:
                    q = np.where(self._mask(x), p, 0.0)
                    if q.sum() > 0:
                        p = q / q.sum()
                a = sa.sample(p, 1.0 - rng.random())
                if a < 0:
                    a = int(np.flatnonzero(sa.free_mask(x))[0])
                x, step = sa.apply(x, a)
            return (tuple(np.flatnonzero(x[0:90] > 0.5).tolist()),
                    tuple(np.flatnonzero(x[90:180] > 0.5).tolist()), self.map, self.tier)

    return FloorSearch


class _MaskedPolicy:
    """Policy wrapper used per search: prior(x) restricted to the search's own mask
    at our pick nodes (decision nodes are always ours). Keeps the batched path."""

    def __init__(self, pol, search):
        self.pol, self.s = pol, search

    def cached(self, x):
        p = self.pol.cached(x)
        return None if p is None else self.s._restrict_prior(x, p)

    def prior(self, x):
        return self.s._restrict_prior(x, self.pol.prior(x))

    def prior_batch(self, xs):
        return [self.s._restrict_prior(x, p) for x, p in zip(xs, self.pol.prior_batch(xs))]


def build_actor(name, draft_ctx, seed_base):
    from paper1_revision import search_actor as sa
    floor, sims = AGENTS[name]
    FS = make_search_cls()
    pols = [sa._policy(p) for p in sa.f_oof_paths()]
    leaf = sa._leaf("enriched")
    gd = sa.gd_np(0)

    def batch_act(items):
        gens, xs, ss = [], [], []
        for i, state, team, game_map, tier in items:
            x = sa.state_vec(state)
            rng = np.random.default_rng([seed_base, i, state.step, 7])
            s = FS(None, gd, leaf, game_map, tier, sims=sims)
            s.floor = floor
            s.pol = _MaskedPolicy(pols[i % len(pols)], s)
            gens.append(s.gen(x, team, rng))
            xs.append(x)
            ss.append(s)
        out = []
        for x, s, (a, _) in zip(xs, ss, sa.run_batched(gens)):
            assert x[a] + x[90 + a] + x[180 + a] < 0.5
            if sa.IS_PICK[sa.step_of(x)] == 1.0:
                assert s._mask(x)[a], "search left the mask"
            out.append(a)
        return out

    def act(state, team, step_type, game_map, tier):
        return batch_act([(draft_ctx["draft"], state, team, game_map, tier)])[0]
    act.batch = batch_act
    return act


def all_names():
    return list(AGENTS) + OPPS


def pair_path(a, b):
    return os.path.join(OUT, "tournament", f"{a}__{b}.json")


def build_any(name, device, gd_models, seed):
    from paper1_revision import tournament as T
    from paper1_revision import support_mitigation as sm
    from paper1_revision import wr_tour  # noqa: F401  registers constrained_wr in T.build
    if name in AGENTS:
        return build_actor(name, T.CTX, seed)
    if name in sm.VARIANTS:
        return sm.build_variant(name, device, gd_models)
    return T.build(name, device, gd_models)


def run_pair(a, b, n):
    import torch
    from paper1_revision import search_eval as se
    from paper1_revision import tournament as T
    names = all_names()
    seed = SEED_OFFSET + names.index(a) * len(names) + names.index(b)
    se.build_any = lambda name, device, gd_models, pair_seed: build_any(name, device, gd_models, seed)
    se.all_names = lambda: names
    se.SEED_OFFSET = SEED_OFFSET
    R = se.run_pair(a, b, n)
    R["seed"] = seed
    R["search"] = {k: {"floor": AGENTS[k][0], "sims": AGENTS[k][1]} for k in (a, b) if k in AGENTS}
    R["floor_fallbacks"] = dict(FALLBACKS)
    from paper1_revision import support_mitigation as sm
    R["play_floor_fallbacks"] = dict(sm.FLOOR_FALLBACKS)
    return R


def cmd_pair(spec, n):
    a, b = spec.split("__")
    p = pair_path(a, b)
    if os.path.exists(p):
        print("exists", p)
        return
    import time
    t = time.time()
    R = run_pair(a, b, n)
    R["secs"] = time.time() - t
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(R, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
    print(f"{a}__{b} {R['secs']:.0f}s", flush=True)


def cmd_queue(procs, n):
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    todo = [f"{a}__{b}" for a, b in PAIRS if not os.path.exists(pair_path(a, b))]
    # longest first
    todo.sort(key=lambda s: -max(AGENTS.get(x, (0, 0))[1] for x in s.split("__")))
    print(len(todo), "pairs", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")

    def one(s):
        rc = subprocess.call([sys.executable, "-u", os.path.abspath(__file__), "pair", s, str(n)],
                             cwd=TRAINING_DIR, env=env)
        print(s, "rc", rc, flush=True)
    with ThreadPoolExecutor(procs) as ex:
        list(ex.map(one, todo))
    print("QUEUE DONE", flush=True)


def cmd_h2h():
    import glob
    from paper1_revision import support_mitigation as sm
    from paper1_revision import judge_ood_calibration as joc
    from shared import HERO_TO_IDX
    recs = []
    for f in sorted(glob.glob(os.path.join(OUT, "tournament", "*.json"))):
        R = json.load(open(f))
        recs += [(R["team0_strategy"], R["team1_strategy"], r) for r in R["records"]]
    games = [(None, r["tier"], r["game_map"], r["team0"]["picks"], r["team1"]["picks"]) for _, _, r in recs]
    sc = joc.judge_scores(games)
    C = sm.sup_corpus()
    A = lambda side: np.array([sorted(HERO_TO_IDX[h] for h in r[side]["picks"]) for _, _, r in recs], np.int64)
    m0, m1 = C.metrics(A("team0")), C.metrics(A("team1"))
    jc = json.load(open(os.path.join(sm.OUT, "judge_corr.json")))
    S0 = np.array([a for a, _, _ in recs])
    S1 = np.array([b for _, b, _ in recs])
    cfg = np.array([f"{r['game_map']}|{r['tier']}" for _, _, r in recs])
    rows = sorted({(a, b) for a, b in PAIRS if a in AGENTS and (b not in AGENTS or list(AGENTS).index(a) > list(AGENTS).index(b))})
    out = {"support": {}, "h2h": {}}
    for s in list(AGENTS) + OPPS:
        sel0, sel1 = S0 == s, S1 == s
        if not (sel0.any() or sel1.any()):
            continue
        mn3 = np.concatenate([m0["min3"][sel0], m1["min3"][sel1]])
        pmi = np.concatenate([m0["pmi"][sel0], m1["pmi"][sel1]])
        out["support"][s] = {"n": int(len(mn3)), "min3_median": float(np.median(mn3)),
                             "share_min3_lt25": float((mn3 < 25).mean()), "pmi_mean": float(pmi.mean())}
    for j in ("consensus", "gN", "RN", "QM2026"):
        wl, wb = np.array(jc["judges"][j]["w_linear"]), np.array(jc["judges"][j]["w_binned"])
        forms = {"none": np.asarray(sc[j]),
                 "linear": sm.sig(sm.logit(sc[j]) + (sm.lin_feats(m0) - sm.lin_feats(m1)) @ wl),
                 "binned": sm.sig(sm.logit(sc[j]) + (sm.bin_feats(m0, jc["edges"]) - sm.bin_feats(m1, jc["edges"])) @ wb)}
        for f, v in forms.items():
            for a, b in rows:
                x = np.concatenate([v[(S0 == a) & (S1 == b)], 1 - v[(S0 == b) & (S1 == a)]])
                c = np.concatenate([cfg[(S0 == a) & (S1 == b)], cfg[(S0 == b) & (S1 == a)]])
                if not len(x):
                    continue
                mu = x.mean()
                u = np.unique(c, return_inverse=True)[1]
                se = np.sqrt((np.bincount(u, x - mu) ** 2).sum()) / len(x)
                out["h2h"].setdefault(f"{a} vs {b}", {})[f"{j}/{f}"] = [round(float(mu), 4), round(float(se), 4), int(len(x))]
    json.dump(out, open(os.path.join(OUT, "floor_search_h2h.json"), "w"), indent=1)
    for k, v in out["support"].items():
        print(f"{k:22s} {v}")
    for k, v in out["h2h"].items():
        print(f"{k:40s}", "  ".join(f"{jf} {m:.3f}±{s:.3f}" for jf, (m, s, n) in v.items()
                                     if jf.split('/')[0] in ('consensus', 'gN', 'RN') and jf.split('/')[1] in ('none', 'binned')))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "pair":
        cmd_pair(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 200)
    elif c == "queue":
        cmd_queue(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 200)
    elif c == "h2h":
        cmd_h2h()
