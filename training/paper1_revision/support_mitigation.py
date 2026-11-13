"""
Drafter-side mitigations of the judges' support bias (judge_ood_calibration.py), and a
binned judge correction to score them.

Variants (rules fixed before any score):
  constrained_greedy_sp   constrained greedy whose terminal win probability carries a
                          fitted support term: sigmoid(logit wp + h(team0) - h(team1)).
                          h is fit for the greedy's OWN value function (the enriched WP,
                          deploy statistics) on the post-snapshot NODRIFT games, which
                          share no game with the snapshot games the judge correction is
                          fit on, so no drafter is graded by the correction it optimized.
  constrained_greedy_sf   constrained greedy with a support floor: a pick is allowed only
                          if every 3-hero subset it forms with our earlier picks occurs in
                          at least FLOOR training-split teams (FLOOR = 25, the 5th
                          percentile of held-out human teams' rarest triple).
  constrained_wr_sf       the masked win-rate drafter with the same floor.
The floor is applied with the structural mask; if no hero passes both, the structural
mask alone decides (counted in FLOOR_FALLBACKS).

Opponents: the eight tournament strategies whose models are unchanged by the baseline
retrain (mcts, constrained_mcts, enriched, constrained_greedy, k_oof, gourdeau,
gourdeau_disc, gd) plus the win-rate drafter: 18 ordered pairs, 200 drafts each, seeds as
tournament.run_pair (the variant appended to the strategy list). Each variant is compared
with its base on the same opponents (base rows already exist).

Judge correction for scoring: per judge, antisymmetric logistic offset on binned support
features (rarest-triple and pair-affinity bins on human quantiles, one-swap and
two-plus-swap indicators), fit on the leave-one-out training-split games of
judge_ood_calibration_teams.npz; the linear form is kept for comparison; both are checked
on the held-out games with finer bins.

Usage (from training/, p1site env):
  python3 paper1_revision/support_mitigation.py fit-agent        # h for the greedy's WP
  python3 paper1_revision/support_mitigation.py fit-judges       # binned + linear corrections
  python3 paper1_revision/support_mitigation.py tour --pool 16   # variant pairs
  python3 paper1_revision/support_mitigation.py pair A__B        # one pair (worker)
  python3 paper1_revision/support_mitigation.py score            # all records, both corrections
Outputs: site/results/support_mitigation/{agent_h.json,judge_corr.json,tournament/,scores.json}
"""
import os
import sys
import json
import glob
import time
import argparse
import subprocess
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core
from paper1_revision import team_support as ts

OUT = os.path.join(core.RESULTS, "support_mitigation")
FLOOR = 25
SWEEP_RE = r"^constrained_mcts_sf(\d+)$"  # floor-threshold sweep on the MCTS-trained policy (ood_frontier.py)
# MCTS trained with the floor inside self-play (floor_kernel.py, train_mcts --floor), pooled over its seeds;
# constrained_mcts_ft = role mask only at play, constrained_mcts_ft_sf = role mask + floor at play
FT_CONFIG, FT_SEEDS = "F_floor", [0, 1, 2]
FT_NAMES = ["constrained_mcts_ft", "constrained_mcts_ft_sf"]
VARIANTS = ["constrained_greedy_sp", "constrained_greedy_sf", "constrained_wr_sf", "constrained_mcts_sf", "constrained_k_sf"]
# floor at play on the trained search policies: the MCTS-trained policy (F_oof) and masked K, argmax as in the tournament
EXTRA_PAIRS = ["constrained_k_sf__constrained_k", "constrained_k__constrained_k_sf",
               # floored vs floored: the search policies against the floored hero-strength drafter
               "constrained_mcts_sf__constrained_wr_sf", "constrained_wr_sf__constrained_mcts_sf",
               "constrained_k_sf__constrained_wr_sf", "constrained_wr_sf__constrained_k_sf"]
OPPONENTS = ["mcts", "constrained_mcts", "enriched", "constrained_greedy", "k_oof", "gourdeau",
             "gourdeau_disc", "gd", "constrained_wr"]
FLOOR_FALLBACKS = {"n": 0}
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))


# ── support features ────────────────────────────────────────────────────
def corpus():
    from paper1_revision import judge_ood_calibration as joc
    G = core.split_games()
    return ts.Corpus(np.concatenate([joc.idx_teams(G["train"], 3), joc.idx_teams(G["train"], 4)])), G


def lin_feats(m):
    d = np.asarray(m["dist"])
    return np.stack([np.log1p(np.maximum(m["min3"], 0)), np.log1p(np.maximum(m["min2"], 0)), m["pmi"],
                     (d == 1).astype(float), (d >= 2).astype(float)], 1)


QS = [0, 1, 2.5, 5, 10, 25, 50, 100]


def bin_feats(m, edges):
    """One-hot bins of rarest triple and pair affinity (top bin dropped) + swap indicators."""
    cols = []
    for k in ("min3", "pmi"):
        e = np.asarray(edges[k])
        b = np.clip(np.searchsorted(e, m[k], side="right") - 1, 0, len(e) - 2)
        cols += [(b == i).astype(float) for i in range(len(e) - 2)]
    d = np.asarray(m["dist"])
    cols += [(d == 1).astype(float), (d >= 2).astype(float)]
    return np.stack(cols, 1)


def fit_offset(off, X, y, l2=1e-4, iters=50):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sig(off + X @ w)
        g = X.T @ (p - y) / len(y) + l2 * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X / len(y) + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w


# ── agent-side h: the greedy's own WP on post-snapshot games ──────────────
def fit_agent():
    import torch
    from paper1_revision import train_wp
    from shared import HERO_TO_IDX
    C, _ = corpus()
    games = core.gold_games("NODRIFT")
    rows = [(list(g[3]), list(g[4]), g[2], g[1]) for g in games]
    y = np.array([1.0 if g[6] == 0 else 0.0 for g in games])
    Xf, Xs = core.featurize(rows, core.load_stats("deploy"), nproc=int(os.environ.get("P1R_NPROC", "6")))
    m, cols = train_wp.load("enriched")
    with torch.no_grad():
        a = m(torch.tensor(Xf[:, cols])).view(-1).numpy()
        b = m(torch.tensor(Xs[:, cols])).view(-1).numpy()
    p = 0.5 * (a + 1 - b)
    A0 = np.array([sorted(HERO_TO_IDX[h] for h in g[3]) for g in games], np.int64)
    A1 = np.array([sorted(HERO_TO_IDX[h] for h in g[4]) for g in games], np.int64)
    m0, m1 = C.metrics(A0), C.metrics(A1)
    X = lin_feats(m0) - lin_feats(m1)
    # fit on a random 70%, report the other 30%
    rng = np.random.default_rng(0)
    te = rng.random(len(y)) < 0.3
    w = fit_offset(logit(p[~te]), X[~te], y[~te])
    ll = lambda q, yy: float(-np.mean(yy * np.log(np.clip(q, 1e-6, 1)) + (1 - yy) * np.log(np.clip(1 - q, 1e-6, 1))))
    q = sig(logit(p) + X @ w)
    gap = lambda qq, s: float((y[s] - qq[s]).mean())
    low = m0["min3"] <= np.percentile(m0["min3"], 5)
    rec = {"games": int(len(y)), "w": w.tolist(), "features": ["log1p_min3", "log1p_min2", "pmi", "one_swap", "two_plus"],
           "test_ll_before": ll(p[te], y[te]), "test_ll_after": ll(q[te], y[te]),
           "test_gap_team0_rarest5pct_triple_before": gap(p, te & low), "test_gap_team0_rarest5pct_triple_after": gap(q, te & low)}
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(os.path.join(OUT, "agent_h.json"), "w"), indent=1)
    print(json.dumps(rec, indent=1))


# ── judge corrections: linear and binned ─────────────────────────────────
def fit_judges():
    from paper1_revision.support_correction import games, JUDGES
    z = dict(np.load(os.path.join(core.RESULTS, "judge_ood_calibration_teams.npz"), allow_pickle=True))
    tr, ho = games(z, "train_loo"), games(z, "heldout")
    # team metrics in game order for both sides (games() keeps only team0's)
    def m_side(s, side):
        sel = z["set"] == s
        rid, team = z["rid"][sel], z["team"][sel]
        o = np.where(team == side)[0]
        o = o[np.argsort(rid[o], kind="stable")]
        return {k: z["m_" + k][sel][o] for k in ("min3", "min2", "pmi", "dist")}
    trm0, trm1, hom0, hom1 = m_side("train_loo", 0), m_side("train_loo", 1), m_side("heldout", 0), m_side("heldout", 1)
    allm = {k: np.concatenate([trm0[k], trm1[k]]) for k in trm0}
    edges = {k: np.unique(np.percentile(allm[k], QS)).tolist() for k in ("min3", "pmi")}
    fine = {k: np.unique(np.percentile(np.concatenate([hom0[k], hom1[k]]), [0, 1, 2.5, 5, 10, 25, 50, 75, 100])).tolist()
            for k in ("min3", "pmi")}
    res = {"edges": edges, "fine_edges": fine, "judges": {}}

    def team_gaps(p, y, mA, mB, e):
        """Gap per bin over both teams of each held-out game."""
        pp = np.concatenate([p, 1 - p]); yy = np.concatenate([y, 1 - y])
        out = {}
        for k in ("min3", "pmi"):
            v = np.concatenate([mA[k], mB[k]])
            b = np.clip(np.searchsorted(e[k], v, side="right") - 1, 0, len(e[k]) - 2)
            out[k] = [[int((b == i).sum()), float((yy[b == i] - pp[b == i]).mean())] for i in range(len(e[k]) - 1)]
        return out
    for j in JUDGES:
        o_tr = logit(tr["p"][j])
        wl = fit_offset(o_tr, lin_feats(trm0) - lin_feats(trm1), tr["y"])
        wb = fit_offset(o_tr, bin_feats(trm0, edges) - bin_feats(trm1, edges), tr["y"])
        p0 = ho["p"][j]
        pl = sig(logit(p0) + (lin_feats(hom0) - lin_feats(hom1)) @ wl)
        pb = sig(logit(p0) + (bin_feats(hom0, edges) - bin_feats(hom1, edges)) @ wb)
        ll = lambda q: float(-np.mean(ho["y"] * np.log(q) + (1 - ho["y"]) * np.log(1 - q)))
        res["judges"][j] = {"w_linear": wl.tolist(), "w_binned": wb.tolist(),
                            "heldout_ll": {"none": ll(p0), "linear": ll(pl), "binned": ll(pb)},
                            "heldout_gap": {"none": team_gaps(p0, ho["y"], hom0, hom1, fine),
                                            "linear": team_gaps(pl, ho["y"], hom0, hom1, fine),
                                            "binned": team_gaps(pb, ho["y"], hom0, hom1, fine)}}
        print(j, res["judges"][j]["heldout_ll"], flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "judge_corr.json"), "w"), indent=1)


# ── strategies ───────────────────────────────────────────────────────────
_C3 = None


def triple_counts():
    global _C3
    if _C3 is None:
        p = os.path.join(OUT, "triple_counts.npy")
        if not os.path.exists(p):
            C, _ = corpus()
            os.makedirs(OUT, exist_ok=True)
            np.save(p, C.c3.astype(np.int32))
        _C3 = np.load(p)
    return _C3


def floor_mask(state, team, floor=None):
    """1 for heroes whose every triple with our current picks has >= floor (default FLOOR) training teams."""
    floor = FLOOR if floor is None else floor
    from shared import NUM_HEROES
    H = NUM_HEROES
    c3 = triple_counts()
    vec = state.team0_picks if team == 0 else state.team1_picks
    ours = [int(i) for i in np.where(np.asarray(vec) > 0)[0]]
    ok = np.ones(H, np.float32)
    if len(ours) < 2:
        return ok
    for h in range(H):
        for a, b in combinations(ours, 2):
            t = sorted((a, b, h))
            if len(set(t)) == 3 and c3[(t[0] * H + t[1]) * H + t[2]] < floor:
                ok[h] = 0.0
                break
    return ok


def floor_constrain(inner, floor=None):
    """Structural mask (constrained_search.constrain) combined with the support floor."""
    from rerun2026.constrained_search import role_constraint_mask, _constrained_view

    def strategy(state, team, step_type, game_map, tier, gd_models, device):
        if step_type != "pick":
            return inner(state, team, step_type, game_map, tier, gd_models, device)
        cmask = role_constraint_mask(state, team)
        m = cmask * floor_mask(state, team, floor)
        if m.sum() < 0.5:
            FLOOR_FALLBACKS["n"] += 1
            m = cmask
        idx = inner(_constrained_view(state, m), team, step_type, game_map, tier, gd_models, device)
        if m[idx] < 0.5:
            idx = int(np.argmax(m))
        return idx
    return strategy


def is_variant(name):
    import re
    return name in VARIANTS or name in FT_NAMES or re.match(SWEEP_RE, name) is not None


def build_variant(name, device, gd_models):
    from rerun2026.constrained_search import constrain
    import experiment_rich_evaluation as ere
    from paper1_revision import train_wp
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    from sweep_enriched_wp import compute_group_indices
    gi = compute_group_indices()
    if name == "constrained_wr_sf":
        from paper1_revision import wr_tour
        st = core.load_stats("deploy")
        cache = {}

        def inner(state, team, step_type, game_map, tier, gd_models_, device_):
            key = (game_map, tier)
            if key not in cache:
                cache[key] = np.array(wr_tour.scores(st, game_map, tier))
            s = np.where(state.valid_mask_np() > 0.5, cache[key], -np.inf)
            return int(np.argmax(s))
        strat = floor_constrain(inner)
    elif name == "constrained_greedy_sf":
        m, cols = train_wp.load("enriched", device=device)
        strat = floor_constrain(ere.make_wp_greedy_strategy(m, ENRICHED_GROUPS, core.load_stats("deploy"), gi, device))
    elif name in FT_NAMES:
        from paper1_revision import tournament as T
        pol = T.pooled_policy([os.path.join(core.MCTS_RUNS, f"{FT_CONFIG}_s{s}", "draft_policy.pt") for s in FT_SEEDS])
        strat = floor_constrain(pol) if name.endswith("_sf") else constrain(pol)
    elif is_variant(name) and name not in VARIANTS:
        import re
        from paper1_revision import tournament as T
        t = int(re.match(SWEEP_RE, name).group(1))
        strat = floor_constrain(T.pooled_policy([os.path.join(core.MCTS_RUNS, f"{T.MCTS_CONFIG}_s{s}", "draft_policy.pt")
                                                 for s in T.MCTS_SEEDS]), floor=t)
    elif name in ("constrained_mcts_sf", "constrained_k_sf"):
        from paper1_revision import tournament as T
        cfg, seeds = (T.MCTS_CONFIG, T.MCTS_SEEDS) if name == "constrained_mcts_sf" else (T.K_CONFIG, T.K_SEEDS)
        strat = floor_constrain(T.pooled_policy([os.path.join(core.MCTS_RUNS, f"{cfg}_s{s}", "draft_policy.pt") for s in seeds]))
    elif name == "constrained_greedy_sp":
        m, cols = train_wp.load("enriched", device=device)
        strat = constrain(make_sp_greedy(m, gi, device))
    else:
        raise KeyError(name)

    def act(state, team, step_type, game_map, tier):
        return strat(state, team, step_type, game_map, tier, gd_models, device)
    return act


def make_sp_greedy(wp_model, gi, device):
    """Greedy over candidates with GD completions, terminal value = WP with the support term."""
    import random
    import torch
    import experiment_rich_evaluation as ere
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    from shared import HEROES, NUM_HEROES, HERO_TO_IDX
    from sweep_enriched_wp import extract_features, FEATURE_GROUPS
    w = np.array(json.load(open(os.path.join(OUT, "agent_h.json")))["w"])
    C = sup_corpus()
    stats = core.load_stats("deploy")
    all_mask = [True] * len(FEATURE_GROUPS)
    cols = []
    for g in ENRICHED_GROUPS:
        s, e = gi[g]
        cols.extend(range(s, e))

    def h(team):
        A = np.array([sorted(HERO_TO_IDX[x] for x in team)], np.int64)
        return float(lin_feats(C.metrics(A))[0] @ w)

    def eval_terminal(t0h, t1h, game_map, tier):
        d = {"team0_heroes": t0h, "team1_heroes": t1h, "game_map": game_map, "skill_tier": tier, "winner": 0}
        base, enriched = extract_features(d, stats, all_mask)
        x = np.concatenate([base, enriched[cols]])
        with torch.no_grad():
            p = wp_model(torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)).item()
        return float(sig(logit(p) + h(t0h) - h(t1h)))

    def strategy(state, team, step_type, game_map, tier, gd_models, dev):
        mask = state.valid_mask_np()
        best, best_wp = None, -1
        for hero_idx in [i for i in range(NUM_HEROES) if mask[i] > 0]:
            s = state.clone()
            s.apply_action(hero_idx, team, step_type)
            for rs_team, rs_type in ere.DRAFT_ORDER[s.step:]:
                gd = random.choice(gd_models)
                x = s.to_tensor_gd(torch.device("cpu"))
                mk = s.valid_mask(torch.device("cpu"))
                with torch.no_grad():
                    hh = gd(x, mk).argmax(dim=1).item()
                s.apply_action(hh, rs_team, rs_type)
            t0h = [HEROES[i] for i in range(NUM_HEROES) if s.team0_picks[i] > 0]
            t1h = [HEROES[i] for i in range(NUM_HEROES) if s.team1_picks[i] > 0]
            wp = eval_terminal(t0h, t1h, game_map, tier)
            if state.our_team == 1:
                wp = 1 - wp
            if wp > best_wp:
                best_wp, best = wp, hero_idx
        return best
    return strategy


_SUPC = None


def sup_corpus():
    """Training-split support counts, cached to disk for the tournament workers."""
    global _SUPC
    if _SUPC is None:
        p = os.path.join(OUT, "corpus.npz")
        if not os.path.exists(p):
            C, _ = corpus()
            os.makedirs(OUT, exist_ok=True)
            np.savez(p, k5k=C.k5[0], k5c=C.k5[1], c4=C.c4.astype(np.int32), c3=C.c3.astype(np.int32),
                     c2=C.c2.astype(np.int64), c1=C.c1, n=C.n, rck=C.rc[0], rcc=C.rc[1])
        z = np.load(p)
        C = ts.Corpus.__new__(ts.Corpus)
        C.n, C.k5, C.c4, C.c3, C.c2, C.c1, C.rc = int(z["n"]), (z["k5k"], z["k5c"]), z["c4"], z["c3"], z["c2"], z["c1"], (z["rck"], z["rcc"])
        _SUPC = C
    return _SUPC


# ── tournament ───────────────────────────────────────────────────────────
def pairs():
    return [f"{v}__{o}" for v in VARIANTS for o in OPPONENTS] + [f"{o}__{v}" for v in VARIANTS for o in OPPONENTS] + EXTRA_PAIRS


def path(p):
    return os.path.join(OUT, "tournament", f"{p}.json")


def run_one(p, n=200):
    from paper1_revision import tournament as T
    from paper1_revision import wr_tour  # appends constrained_wr (12th) and wraps T.build
    if os.path.exists(path(p)):
        return
    a, b = p.split("__")
    if "constrained_k" in (a, b):
        from paper1_revision import mk_tour  # noqa: F401  appends constrained_k and wraps T.build
    T.STRATEGIES = list(T.STRATEGIES) + [x for x in (a, b) if is_variant(x)]
    _build = T.build

    def build(name, device, gd_models):
        if is_variant(name):
            return build_variant(name, device, gd_models)
        return _build(name, device, gd_models)
    T.build = build
    t0 = time.time()
    out = T.run_pair(a, b, n)
    out["floor_fallbacks"] = FLOOR_FALLBACKS["n"]
    os.makedirs(os.path.dirname(path(p)), exist_ok=True)
    json.dump(out, open(path(p) + ".tmp", "w"))
    os.replace(path(p) + ".tmp", path(p))
    print(f"{p}: {n} drafts in {time.time() - t0:.0f}s, floor fallbacks {FLOOR_FALLBACKS['n']}", flush=True)


def tour(procs, n=200):
    logs = os.path.join(core.LOGS, "support_mitigation")
    os.makedirs(logs, exist_ok=True)
    triple_counts(); sup_corpus()  # build the shared caches once, before the workers start
    todo = [p for p in pairs() if not os.path.exists(path(p))]

    def run(p):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with open(os.path.join(logs, f"{p}.log"), "w") as log:
            r = subprocess.run([sys.executable, "-u", __file__, "pair", p, "--drafts", str(n)],
                               cwd=TRAINING_DIR, stdout=log, stderr=subprocess.STDOUT, env=env)
        print(p, "rc", r.returncode, flush=True)
    with ThreadPoolExecutor(procs) as ex:
        list(ex.map(run, todo))


# ── scoring ──────────────────────────────────────────────────────────────
BASE_OF = {"constrained_greedy_sp": "constrained_greedy", "constrained_greedy_sf": "constrained_greedy",
           "constrained_wr_sf": "constrained_wr", "constrained_mcts_sf": "constrained_mcts", "constrained_k_sf": "constrained_k"}


def score():
    from paper1_revision import judge_ood_calibration as joc
    from shared import HERO_TO_IDX
    recs = []
    for d in ("tournament", "wr_baseline/tournament", "masked_k/tournament", "support_mitigation/tournament"):
        for f in sorted(glob.glob(os.path.join(core.RESULTS, d, "*.json"))):
            R = json.load(open(f))
            a, b = R["team0_strategy"], R["team1_strategy"]
            keep = set(BASE_OF.values()) | set(VARIANTS)
            if not ({a, b} & keep):
                continue
            for r in R["records"]:
                recs.append((a, b, r))
    games = [(None, r["tier"], r["game_map"], r["team0"]["picks"], r["team1"]["picks"]) for _, _, r in recs]
    sc = joc.judge_scores(games)
    C = sup_corpus()
    A = lambda side: np.array([sorted(HERO_TO_IDX[h] for h in r[side]["picks"]) for _, _, r in recs], np.int64)
    m0, m1 = C.metrics(A("team0")), C.metrics(A("team1"))
    jc = json.load(open(os.path.join(OUT, "judge_corr.json")))
    edges = jc["edges"]
    P = {}
    for j in joc.JUDGES:
        wl, wb = np.array(jc["judges"][j]["w_linear"]), np.array(jc["judges"][j]["w_binned"])
        P[(j, "none")] = np.asarray(sc[j])
        P[(j, "linear")] = sig(logit(sc[j]) + (lin_feats(m0) - lin_feats(m1)) @ wl)
        P[(j, "binned")] = sig(logit(sc[j]) + (bin_feats(m0, edges) - bin_feats(m1, edges)) @ wb)
    S0 = np.array([a for a, _, _ in recs]); S1 = np.array([b for _, b, _ in recs])

    def mean_vs(s, opps, v):
        vals = []
        for o in opps:
            x = list(v[(S0 == s) & (S1 == o)]) + list(1 - v[(S0 == o) & (S1 == s)])
            if x:
                vals.append(np.mean(x))
        return float(np.mean(vals)) if vals else None, len(vals)
    res = {"floor": FLOOR, "comparisons": {}, "support": {}}
    for v, base in BASE_OF.items():
        has = lambda s_, o: bool(((S0 == s_) & (S1 == o)).any() or ((S0 == o) & (S1 == s_)).any())
        opps = [o for o in OPPONENTS if o not in (base, v) and has(v, o) and has(base, o)]  # shared opponents only
        row = {}
        for (j, form), pv in P.items():
            mv, nv = mean_vs(v, opps, pv)
            mb, nb = mean_vs(base, opps, pv)
            h, _ = mean_vs(v, [base], pv)
            row[f"{j}/{form}"] = {"variant": mv, "base": mb, "diff": None if mv is None or mb is None else mv - mb,
                                  "pairs": [nv, nb], "h2h_vs_base": h}
        res["comparisons"][v] = row
    for s in [*VARIANTS, *sorted(set(BASE_OF.values()))]:
        sel0, sel1 = S0 == s, S1 == s
        mm = {k: np.concatenate([np.asarray(m0[k])[sel0], np.asarray(m1[k])[sel1]]) for k in ("min3", "pmi", "dist", "n5")}
        if len(mm["min3"]):
            res["support"][s] = {"teams": int(len(mm["min3"])), "min3_median": float(np.median(mm["min3"])),
                                 "min3_below_25": float((mm["min3"] < FLOOR).mean()), "pmi_mean": float(mm["pmi"].mean()),
                                 "seen_exact": float((mm["n5"] > 0).mean()), "two_plus": float((mm["dist"] >= 2).mean())}
    json.dump(res, open(os.path.join(OUT, "scores.json"), "w"), indent=1)
    for v in BASE_OF:
        print("==", v, "vs", BASE_OF[v])
        for k in ("consensus/none", "consensus/linear", "consensus/binned", "gN/binned", "RN/binned"):
            r = res["comparisons"][v][k]
            if r["variant"] is not None:
                print(f"   {k:18s} variant {r['variant']:.3f} base {r['base']:.3f} diff {r['diff']*100:+.2f} pts  h2h {r['h2h_vs_base']}")
    for s, r in res["support"].items():
        print(s, {k: round(x, 3) if isinstance(x, float) else x for k, x in r.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit-agent", "fit-judges", "tour", "pair", "score"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--drafts", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "fit-agent":
        fit_agent()
    elif a.cmd == "fit-judges":
        fit_judges()
    elif a.cmd == "tour":
        tour(a.pool, a.drafts)
    elif a.cmd == "pair":
        import torch
        torch.set_num_threads(1)
        run_one(a.arg, a.drafts)
    elif a.cmd == "score":
        score()
