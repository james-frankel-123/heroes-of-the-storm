"""
Step 4a, paper 1: truth-leaf search. The same decision-time search as paper 1's search agents
(paper1_revision/floor_search.py: search_actor.Search, 200 simulations, F_oof prior or the
floor-trained prior, GD opponent and rollouts, role mask, and the support floor for _sf agents),
with only the leaf swapped: search_actor's shared 'enriched' leaf is replaced, inside the worker
process, by the world's true win probability (family / world / dial). Opponents and seeds are the
tournament's (floor_search.run_pair); every ordered pair against the common opponent set (OPP8
minus constrained_mcts) cut to three opponents (mcts, constrained_greedy, gd) and N_DRAFTS = 28 per seat
(14 maps x the first two tiers): the search costs ~9 s per draft on the 3090 CPU.

Agents:  search_c (role mask, no floor)        truth counterpart of constrained_mcts, mcts, search_c
         search_sf (role mask + floor)         ... of constrained_mcts_sf, search_sf
         search_ft_sf (floor-trained prior)    ... of constrained_mcts_ft_sf, search_ft_sf
A 'real' run (the enriched leaf, world-free) on the same seeds pairs every truth-leaf draft with
the agent's own draft.

Stages (from training/, p1site env, CPU):
  run --fam F --wid W --dial D     one world setting, all agents x pairs -> site/cache/tl/<tag>.json
  real                              the enriched-leaf runs (once)
  queue --procs P                   every world setting of the plan, one subprocess each
  score                             regret and hero concentration -> site/results/truthleaf_p1.json
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

AGENTS = ["search_c", "search_sf", "search_ft_sf"]
OPPS = ["mcts", "constrained_greedy", "gd"]     # a policy, a greedy WP drafter, the human-imitation model
N_DRAFTS = 28                                   # 14 maps x 2 tiers per seat (paper 1's search costs ~9 s per draft)
TL = os.path.join(WD.CACHE, "tl")
PLAN = [("a", "default", range(10)), ("a", "stress", range(10)), ("g", "member", range(10)), ("c", "default", range(5))]
COUNTERPART = {"constrained_mcts": "search_c", "mcts": "search_c", "search_c": "search_c",
               "constrained_mcts_sf": "search_sf", "search_sf": "search_sf",
               "constrained_mcts_ft_sf": "search_ft_sf", "search_ft_sf": "search_ft_sf"}
os.makedirs(TL, exist_ok=True)


class TruthLeaf:
    """search_actor.LeafWP interface: value_batch(keys) -> [(P(team 0 wins), 0.0)]."""

    def __init__(self, fam, wid, dial):
        from shared import HEROES, MAPS, SKILL_TIERS
        self.fam, self.wid, self.dial = fam, int(wid), dict(dial)
        self.cache = {}
        self.HEROES = HEROES
        maps, tiers = WD.meta()
        self.mi = {m: (maps.index(m) if m in maps else len(maps)) for m in MAPS}
        self.ti = {t: (tiers.index(t) if t in tiers else len(tiers)) for t in SKILL_TIERS}
        if fam == "g":
            from paper1_revision import search_actor as sa
            self.ens = sa.LeafWP("ensemble")
            self.ens.members = [self.ens.members[self.wid]]
            self.ens._np_members()
        else:
            self.W = W.Worlds()
            self.theta = self.W.base_draws([self.wid])[0]

    def value_batch(self, keys):
        if self.fam == "g":
            out = [(m, 0.0) for m, _ in self.ens.value_batch(keys)]
            self.cache.update(zip(keys, out))
            return out
        todo = [k for k in dict.fromkeys(keys) if k not in self.cache]
        if todo:
            D = {"T0": np.sort(np.array([list(a) for a, _, _, _ in todo], np.int64), 1),
                 "T1": np.sort(np.array([list(b) for _, b, _, _ in todo], np.int64), 1),
                 "m": np.array([self.mi[m] for _, _, m, _ in todo]), "t": np.array([self.ti[t] for _, _, _, t in todo])}
            c = self.W.components(self.fam, D, self.wid, theta=self.theta)
            p = M.sig(self.W.combine(c, self.fam, **self.dial))
            if len(self.cache) > 2_000_000:
                self.cache.clear()
            for k, v in zip(todo, p):
                self.cache[k] = (float(v), 0.0)      # search_actor reads leaf.cache as (mean, sd)
        return [self.cache[k] for k in keys]

    def value_t0(self, t0, t1, game_map, tier):
        return self.value_batch([(tuple(t0), tuple(t1), game_map, tier)])[0]


def tag_of(fam, wid, dial):
    return f"{fam}_{dial}_w{wid}"


def run_setting(fam, wid, dial_name):
    out = os.path.join(TL, tag_of(fam, wid, dial_name) + ".json")
    if os.path.exists(out):
        return
    from paper1_revision import search_actor as sa
    from paper1_revision import floor_search as fs
    if fam != "real":
        dial = {} if fam == "g" else {k: v for k, v in W.dials()["named"][dial_name].items() if k != "drift"}
        sa._SHARED["leaf:enriched"] = TruthLeaf(fam, wid, dial)
    res = {"family": fam, "world": int(wid), "dial": dial_name, "n": N_DRAFTS, "pairs": {}}
    pdir = os.path.join(TL, tag_of(fam, wid, dial_name))       # per-pair checkpoints (resume)
    os.makedirs(pdir, exist_ok=True)
    t0 = time.time()
    for a in AGENTS:
        for o in OPPS:
            for x, y in ((a, o), (o, a)):
                pp = os.path.join(pdir, f"{x}__{y}.json")
                if not os.path.exists(pp):
                    R = fs.run_pair(x, y, N_DRAFTS)
                    recs = [{"m": r["game_map"], "t": r["tier"], "t0": r["team0"]["picks"],
                             "t1": r["team1"]["picks"], "s0": [h for h, _ in r["team0_steps"]],
                             "s1": [h for h, _ in r["team1_steps"]]} for r in R["records"]]
                    json.dump(recs, open(pp + ".tmp", "w"))
                    os.replace(pp + ".tmp", pp)
                res["pairs"][f"{x}__{y}"] = json.load(open(pp))
        print(f"{tag_of(fam, wid, dial_name)} {a} done ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(out + ".tmp", "w"))
    os.replace(out + ".tmp", out)


def queue(procs):
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    jobs = [("real", 0, "real")] + [(f, w, d) for f, d, ws in PLAN for w in ws]
    jobs = [j for j in jobs if not os.path.exists(os.path.join(TL, tag_of(*j) + ".json"))]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    print(len(jobs), "settings", flush=True)

    def one(j):
        rc = subprocess.call([sys.executable, "-u", os.path.abspath(__file__), "run", "--fam", j[0], "--wid", str(j[1]),
                              "--dial", j[2]], cwd=TRAINING_DIR, env=env)
        print(tag_of(*j), "rc", rc, flush=True)
    W.Worlds().base_draws([0])
    with ThreadPoolExecutor(procs) as ex:
        list(ex.map(one, jobs))
    print("TL QUEUE DONE", flush=True)


# ── scoring ──────────────────────────────────────────────────────────────
def _arrays(recs):
    from shared import HERO_TO_IDX
    maps, tiers = WD.meta()
    return {"T0": np.array([sorted(HERO_TO_IDX[h] for h in r["t0"]) for r in recs], np.int64),
            "T1": np.array([sorted(HERO_TO_IDX[h] for h in r["t1"]) for r in recs], np.int64),
            "m": np.array([maps.index(r["m"]) if r["m"] in maps else len(maps) for r in recs]),
            "t": np.array([tiers.index(r["t"]) if r["t"] in tiers else len(tiers) for r in recs])}


def truth_p(fam, wid, dial_name, D, ens=None):
    if fam == "g":
        return ens
    Ws = W.Worlds()
    dial = {k: v for k, v in W.dials()["named"][dial_name].items() if k != "drift"}
    return M.sig(Ws.logit(fam, D, wid, **dial))


def agent_value(pairs, a, p_of):
    """mean over opponents of the agent's mean true win prob, both seats."""
    v = []
    for o in OPPS:
        x = np.concatenate([p_of(f"{a}__{o}"), 1 - p_of(f"{o}__{a}")])
        v.append(x.mean())
    return float(np.mean(v))


def concentration(pairs, a):
    """the agent's own picks: share of the most picked hero among first picks and among all picks, entropy
    (bits) of all picks, and the top-3 first picks."""
    from collections import Counter
    first, allp = Counter(), Counter()
    for o in OPPS:
        for k, side in ((f"{a}__{o}", "s0"), (f"{o}__{a}", "s1")):
            for r in pairs[k]:
                if r[side]:
                    first[r[side][0]] += 1
                allp.update(r[side])
    n1, na = sum(first.values()), sum(allp.values())
    q = np.array(list(allp.values()), float) / na
    return {"first_top": first.most_common(3), "first_top_share": first.most_common(1)[0][1] / n1,
            "all_top": allp.most_common(3), "all_top_share": allp.most_common(1)[0][1] / na,
            "entropy_bits": float(-(q * np.log2(q)).sum()), "distinct_heroes": len(allp)}


def score():
    import rescore as RS
    real = json.load(open(os.path.join(TL, "real_real_w0.json")))
    Dreal = {k: _arrays(v) for k, v in real["pairs"].items()}
    P1 = RS.p1_drafts()
    out = {"n_drafts_per_seat": N_DRAFTS, "opponents": OPPS, "settings": {}, "concentration": {}}
    out["concentration"]["real_leaf"] = {a: concentration(real["pairs"], a) for a in AGENTS}
    ens = {}
    for fam, dial, ws in PLAN:
        rows = []
        for w in ws:
            p = os.path.join(TL, tag_of(fam, w, dial) + ".json")
            if not os.path.exists(p):
                continue
            R = json.load(open(p))
            Dt = {k: _arrays(v) for k, v in R["pairs"].items()}
            if fam == "g":
                pt = _ens_split(Dt, f"tl_{tag_of(fam, w, dial)}", w)
                if "g_real" not in ens:
                    ens["g_real"] = _ens_split(Dreal, "tlreal", None)
                pr = {k: v[w] for k, v in ens["g_real"].items()}
            else:
                pt = {k: truth_p(fam, w, dial, d) for k, d in Dt.items()}
                pr = {k: truth_p(fam, w, dial, d) for k, d in Dreal.items()}
            row = {"world": w}
            for a in AGENTS:
                row[f"TL:{a}"] = agent_value(R["pairs"], a, lambda k: pt[k])
                row[f"real:{a}"] = agent_value(real["pairs"], a, lambda k: pr[k])
            # the tournament agents' true values in the same world, against the same three opponents
            # (their tournament drafts, all 200 per ordered pair)
            if fam == "g":
                pa = np.asarray(ens.setdefault("g_p1", _ens_all(P1))[w], float)
            else:
                pa = truth_p(fam, w, dial, P1)
            for x in COUNTERPART:
                v = _p1_agent_value(P1, pa, x)
                if v is not None:
                    row[f"agent:{x}"] = v
            row["concentration"] = {a: concentration(R["pairs"], a) for a in AGENTS}
            rows.append(row)
        if not rows:
            continue
        S = {"worlds": len(rows), "regret_pts": {}, "tl_value_pts": {}, "real_leaf_value_pts": {}}
        for a in AGENTS:
            S["tl_value_pts"][a] = float(100 * np.mean([r[f"TL:{a}"] for r in rows]))
            S["real_leaf_value_pts"][a] = float(100 * np.mean([r[f"real:{a}"] for r in rows]))
            S["regret_pts"][f"{a} (real leaf, same seeds)"] = RS.contrast([r[f"TL:{a}"] - r[f"real:{a}"] for r in rows])
        for x, c in COUNTERPART.items():
            if all(f"agent:{x}" in r for r in rows):
                S["regret_pts"][x] = RS.contrast([r[f"TL:{c}"] - r[f"agent:{x}"] for r in rows])
        S["floor_true_effect_pts"] = {
            "TL:search_sf - TL:search_c": RS.contrast([r["TL:search_sf"] - r["TL:search_c"] for r in rows]),
            "TL:search_ft_sf - TL:search_c": RS.contrast([r["TL:search_ft_sf"] - r["TL:search_c"] for r in rows])}
        S["concentration_mean"] = {a: {k: float(np.mean([r["concentration"][a][k] for r in rows]))
                                       for k in ("first_top_share", "all_top_share", "entropy_bits", "distinct_heroes")}
                                   for a in AGENTS}
        S["first_top_heroes"] = {a: _top_tally([r["concentration"][a]["first_top"][0][0] for r in rows]) for a in AGENTS}
        out["settings"][f"{fam}/{dial}"] = S
    json.dump(out, open(os.path.join(WD.RESULTS, "truthleaf_p1.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:6000])


def _p1_agent_value(P1, p, x):
    v = []
    for o in OPPS:
        f = (P1["S0"] == x) & (P1["S1"] == o)
        g = (P1["S0"] == o) & (P1["S1"] == x)
        if not (f.any() and g.any()):
            return None
        v.append(np.concatenate([p[f], 1 - p[g]]).mean())
    return float(np.mean(v))


def _ens_all(P1):
    import ensemble as EN
    return EN.values(EN.rows_of(P1), "p1drafts")


def _ens_split(Dd, tag, w):
    """one batched ensemble call over every pair's drafts -> {pair: P (member w, or all members if w is None)}."""
    import ensemble as EN
    keys = list(Dd)
    D = {c: np.concatenate([Dd[k][c] for k in keys]) for c in ("T0", "T1", "m", "t")}
    P = np.asarray(EN.values(EN.rows_of(D), tag), float)
    out, i = {}, 0
    for k in keys:
        n = len(Dd[k]["T0"])
        out[k] = P[w, i:i + n] if w is not None else P[:, i:i + n]
        i += n
    return out


def _top_tally(xs):
    from collections import Counter
    return Counter(xs).most_common(5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "real", "queue", "score"])
    ap.add_argument("--fam", default="a")
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--dial", default="default")
    ap.add_argument("--procs", type=int, default=24)
    a = ap.parse_args()
    if a.stage == "run":
        run_setting(a.fam, a.wid, a.dial)
    elif a.stage == "real":
        run_setting("real", 0, "real")
    elif a.stage == "queue":
        queue(a.procs)
    else:
        score()
