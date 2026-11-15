"""
Step 3: sample-efficient exploration inside hidden worlds (3090 CPU).

Decision problem (a completion bandit, cheap enough for thousands of rounds): a context is a real NODRIFT game;
our team has three of its heroes already (a random 3 of the real five), the opponent's five, map and tier are the
real ones. The drafter recommends the last two heroes. Candidates: completions whose role composition occurs in
>= 100 training teams (no absurd compositions: an additive outcome model rates e.g. healer-less five-assassin teams
highly); of those, the learner's top 100 plus the humans' own two plus 20 random ones. The true win probability is
clipped to [0.25, 0.75], the 1st-99th percentile of the worlds' p on the humans' own teams (diagnostics: stage diag,
results/explore_diag.json): no completion is truly better than the best 1% of human matchups. A real lobby (player_cov pool) is attached to each context:
the two remaining players.

Compliance: each recommended hero is played with probability c * phi(n_ph) * 0.8^(rank - 1), c in {5, 20, 50}%,
phi = 1 (10+ earlier games on the hero), 0.6 (1-9), 0.3 (never); rank = the recommendation's rank under the
learner's mean (exploration that recommends a lower-ranked option is followed less). Both picks must be followed
for the recommended pair to be played; otherwise the humans' own completion is played.

Learner (deliberately the family-a model class, mismatched to worlds c, g and f-cov): the world model's base MAP
plus the triple layer as Gaussian cells (posterior from the fit, prior s for unseen), updated game by game with a
diagonal Laplace step. Learner 'refit' = the real-pipeline stand-in: posterior mean only, triple cells refit by MAP
every 500 games, exploit. Organic inflow: every round also feeds the learner ORGANIC human games (real NODRIFT games
with their outcomes drawn from the world).

Policies: exploit, tiebreak (among actions within 0.5 pp of the best mean, the largest posterior SD), floor_anneal
(rarest triple >= 25 / (1 + t / 2000) counting the learner's own games), lcb (mean - 1 SD), egreedy (0.1, uniform
over the top 20), top2ts (top-two Thompson, beta 0.5), refit (pipeline stand-in).
Stress: learner SD x 1.5 (the recovery check found the worlds 1.2-1.5x overconfident on search-chosen teams).

Outputs per run: cumulative true regret against the best candidate, realized true cost of each played exploration
(true value of the exploit choice minus the played one), P(cost > 2 pp), games each policy put on below-floor
compositions, and the value-of-information map (stage voi).
"""
import os
import sys
import json
import time
import argparse
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, TRAINING_DIR)
import data as WD  # noqa: E402
import model as M  # noqa: E402
import world as W  # noqa: E402

H = 90
ROUNDS = int(os.environ.get("EXP_ROUNDS", "4000"))
ORGANIC = 4
POLICIES = ["exploit", "tiebreak", "floor_anneal", "lcb", "egreedy", "top2ts", "refit"]
FAMS = ["a", "c", "g", "fc"]
COMPLIANCE = [0.05, 0.2, 0.5]
OUT = os.path.join(WD.CACHE, "explore")
os.makedirs(OUT, exist_ok=True)
C3_FILE = os.path.join(TRAINING_DIR, "paper1_revision", "site", "results", "support_mitigation", "triple_counts.npy")


def tri_index(T):
    """(n, 5) sorted teams -> (n, 10) dense triple keys."""
    return np.stack([(T[:, a] * H + T[:, b]) * H + T[:, c] for a, b, c in M.C3], 1)


class Learner:
    def __init__(self, sd_mult=1.0, mean_only=False):
        self.B, self.th, self.cfg = M.load_base()
        L = M.load_layer("triple")
        self.mu = np.zeros(H ** 3)
        self.prec = np.full(H ** 3, 1 / L["s"] ** 2)
        self.mu[L["keys"]] = L["mu"]
        self.prec[L["keys"]] = 1 / L["sd"] ** 2
        self.prior_prec = 1 / L["s"] ** 2
        self.sd_mult = sd_mult
        self.mean_only = mean_only
        self.cnt = np.load(C3_FILE).astype(np.float64)
        self.buf = []          # refit learner: games since the last refit

    def base(self, D):
        return self.B.logit(self.th, D)

    def value(self, D):
        """-> mean logit, sd (logit) of P(team 0 wins)."""
        k0, k1 = tri_index(D["T0"]), tri_index(D["T1"])
        z = self.base(D) + self.mu[k0].sum(1) - self.mu[k1].sum(1)
        v = (1 / self.prec[k0]).sum(1) + (1 / self.prec[k1]).sum(1)
        return z, self.sd_mult * np.sqrt(v)

    def sample(self, D, rng):
        k0 = tri_index(D["T0"])
        u, inv = np.unique(k0, return_inverse=True)
        draw = self.mu[u] + self.sd_mult * rng.standard_normal(len(u)) / np.sqrt(self.prec[u])
        k1 = tri_index(D["T1"])
        return self.base(D) + draw[inv.reshape(k0.shape)].sum(1) - self.mu[k1].sum(1)

    def update(self, D, y):
        k0, k1 = tri_index(D["T0"]), tri_index(D["T1"])
        np.add.at(self.cnt, k0.ravel(), 1)
        np.add.at(self.cnt, k1.ravel(), 1)
        if self.mean_only:
            self.buf.append((D, y))
            return
        z, _ = self.value(D)
        p = M.sig(z)
        w = p * (1 - p)
        for sgn, K in ((1, k0), (-1, k1)):
            for j in range(K.shape[1]):
                np.add.at(self.prec, K[:, j], w)
                np.add.at(self.mu, K[:, j], sgn * (y - p) / self.prec[K[:, j]])

    def refit(self):
        """periodic MAP refit of the triple cells touched by the buffered games (the pipeline stand-in)."""
        if not self.buf:
            return
        D = {k: np.concatenate([d[k] for d, _ in self.buf]) for k in ("T0", "T1", "m", "t")}
        y = np.concatenate([yy for _, yy in self.buf])
        for _ in range(5):
            z, _ = self.value(D)
            p = M.sig(z)
            k0, k1 = tri_index(D["T0"]), tri_index(D["T1"])
            g = np.zeros_like(self.mu)
            h = np.zeros_like(self.mu)
            for sgn, K in ((1, k0), (-1, k1)):
                for j in range(K.shape[1]):
                    np.add.at(g, K[:, j], sgn * (y - p))
                    np.add.at(h, K[:, j], p * (1 - p))
            touched = h > 0
            self.mu[touched] += (g[touched] - self.prior_prec * self.mu[touched]) / (h[touched] + self.prior_prec)


class Truth:
    def __init__(self, fam, wid):
        self.fam, self.wid = fam, wid
        if fam == "g":
            from paper1_revision import search_actor as sa
            from shared import HEROES, MAPS, SKILL_TIERS
            self.ens = sa.LeafWP("ensemble")
            self.ens.members = [self.ens.members[wid]]
            self.ens._np_members()
            maps, tiers = WD.meta()
            self.names = (HEROES, maps, tiers)
        else:
            self.W = W.Worlds()
            self.theta = self.W.base_draws([wid])[0]
        if fam == "fc":
            import player as PL
            self.PW = PL.PlayerWorldsCov()

    def p(self, D, lob=None):
        return np.clip(self._p(D, lob), *P_CAP)

    def _p(self, D, lob=None):
        if self.fam == "g":
            HEROES, maps, tiers = self.names
            keys = [(tuple(a), tuple(b), maps[m] if m < len(maps) else "?", tiers[t] if t < len(tiers) else "?")
                    for a, b, m, t in zip(D["T0"].tolist(), D["T1"].tolist(), D["m"], D["t"])]
            return np.array([v for v, _ in self.ens.value_batch(keys)])
        z = self.W.combine(self.W.components("a" if self.fam == "fc" else self.fam, D, self.wid, theta=self.theta),
                           "a" if self.fam == "fc" else self.fam)
        if self.fam == "fc":
            z = z + self.player_term(D, lob)
        return M.sig(z)

    def player_term(self, D, lob):
        """team 0's true player term (best assignment of its five lobby players) minus team 1's (humans' own team
        and lobby players, best assignment)."""
        import rescore as RS
        Dl = dict(D, H0=D["T0"], H1=D["T1"], lob=np.arange(len(D["T0"])), **lob)
        A = RS.assign_terms(self.PW, Dl, self.wid, np.random.default_rng(0))
        return A["best"]


def contexts(n, seed):
    rng = np.random.default_rng(seed)
    D = WD.load("NODRIFT")
    gi = rng.choice(len(D["y"]), n, replace=False)
    keep = np.array([rng.choice(5, 3, replace=False) for _ in range(n)])
    import player as PL
    cz = np.load(PL.COV)
    li = rng.integers(0, len(cz["pool_pid"]), n)
    return {"T0": D["T0"][gi], "T1": D["T1"][gi], "m": D["m"][gi], "t": D["t"][gi], "keep": keep,
            "pid": cz["pool_pid"][li], "cnt": cz["pool_cnt"][li], "s": cz["pool_s"][li], "off": cz["pool_off"][li]}


RC_MIN = 100      # plausibility: a candidate's role composition must occur in >= 100 training teams
P_CAP = (0.25, 0.75)   # plausibility cap on the truth: the 1st-99th percentile of the worlds' p on the humans' own teams
_RC = {}


def role_ok(teams):
    """True where the team's role composition (paper 1's role classes) has >= RC_MIN training teams."""
    if "C" not in _RC:
        from paper1_revision import support_mitigation as sm
        from paper1_revision import team_support as ts
        _RC["C"], _RC["ts"] = sm.sup_corpus(), ts
    C, ts = _RC["C"], _RC["ts"]
    rk = ts.key(range(5), np.sort(ts.ROLE_IDX[teams], 1).astype(np.int64))
    return C._lookup(C.rc, rk) >= RC_MIN


def candidates(L, ctx, i, rng):
    T0, T1 = ctx["T0"][i], ctx["T1"][i]
    base3 = T0[ctx["keep"][i]]
    human = np.setdiff1d(T0, base3)
    free = np.setdiff1d(np.arange(H), np.concatenate([T0, T1]))
    free = np.concatenate([free, human])
    pairs = np.array(list(combinations(free, 2)))
    teams = np.sort(np.concatenate([np.repeat(base3[None], len(pairs), 0), pairs], 1), 1)
    ok = role_ok(teams) | (teams == np.sort(T0)).all(1)          # the humans' own team is always a candidate
    pairs, teams = pairs[ok], teams[ok]
    D = {"T0": teams, "T1": np.repeat(T1[None], len(teams), 0), "m": np.full(len(teams), ctx["m"][i]),
         "t": np.full(len(teams), ctx["t"][i])}
    z, sd = L.value(D)
    top = np.argsort(-z)[:100]
    hidx = np.flatnonzero((teams == np.sort(T0)).all(1))
    rnd = rng.choice(len(teams), min(20, len(teams)), replace=False)
    sel = np.unique(np.concatenate([top, hidx, rnd]))
    D = {k: v[sel] for k, v in D.items()}
    return D, z[sel], sd[sel], int(np.flatnonzero((D["T0"] == np.sort(T0)).all(1))[0]), pairs[sel], base3


def choose(policy, z, sd, D, L, rng, t, step_ctx):
    order = np.argsort(-z)
    best = int(order[0])
    if policy in ("exploit", "refit"):
        return best
    if policy == "tiebreak":
        p = M.sig(z)
        ok = np.flatnonzero(p >= p[best] - 0.005)
        return int(ok[np.argmax(sd[ok])])
    if policy == "lcb":
        return int(np.argmax(z - sd))
    if policy == "egreedy":
        return int(rng.choice(order[:20])) if rng.random() < 0.1 else best
    if policy == "top2ts":
        a = int(np.argmax(L.sample(D, rng)))
        if rng.random() < 0.5:
            return a
        for _ in range(20):
            b = int(np.argmax(L.sample(D, rng)))
            if b != a:
                return b
        return a
    if policy == "floor_anneal":
        T = 25.0 / (1.0 + t / 2000.0)
        m3 = L.cnt[tri_index(D["T0"])].min(1)
        ok = np.flatnonzero(m3 >= T)
        return int(ok[np.argmax(z[ok])]) if len(ok) else best
    raise ValueError(policy)


def phi(n):
    return np.where(n >= 10, 1.0, np.where(n >= 1, 0.6, 0.3))


def run(fam, wid, policy, comp, sd_mult, seed=0):
    tag = f"{fam}_w{wid}_{policy}_c{int(100 * comp)}_k{sd_mult}"
    out = os.path.join(OUT, tag + ".json")
    if os.path.exists(out):
        return
    rng = np.random.default_rng([wid, seed, POLICIES.index(policy), int(100 * comp), int(10 * sd_mult)])
    ctx = contexts(ROUNDS, seed=1000 + wid)                  # the same contexts for every policy in a world
    org = contexts(ROUNDS * ORGANIC, seed=5000 + wid)
    L = Learner(sd_mult, mean_only=(policy == "refit"))
    TR = Truth(fam, wid)
    log = {"regret": [], "cost": [], "explored": [], "followed": [], "below_floor_played": 0, "rare_triples": {}}
    c3 = np.load(C3_FILE)
    t0 = time.time()
    for t in range(ROUNDS):
        D, z, sd, ih, pairs, base3 = candidates(L, ctx, t, rng)
        a = choose(policy, z, sd, D, L, rng, t, None)
        a_ex = int(np.argmax(z))
        rank = int((z > z[a]).sum()) + 1
        lob = {"pid": ctx["pid"][t:t + 1], "cnt": ctx["cnt"][t:t + 1], "s": ctx["s"][t:t + 1], "off": ctx["off"][t:t + 1]}
        n_ph = [ctx["cnt"][t, 0, 3, int(pairs[a][0])], ctx["cnt"][t, 0, 4, int(pairs[a][1])]]
        pf = comp * phi(np.array(n_ph)) * 0.8 ** (rank - 1)
        follow = bool((rng.random(2) < pf).all())
        played = a if follow else ih
        lobs = {k: np.repeat(v, len(D["T0"]), 0) for k, v in lob.items()}
        ptrue = TR.p(D, lobs if fam == "fc" else None)
        log["regret"].append(float(ptrue.max() - ptrue[played]))
        if follow and a != a_ex:
            log["cost"].append(float(ptrue[a_ex] - ptrue[a]))
        log["explored"].append(bool(a != a_ex))
        log["followed"].append(follow)
        Dp = {k: v[played:played + 1] for k, v in D.items()}
        if follow and c3[tri_index(Dp["T0"])].min() < 25:
            log["below_floor_played"] += 1
        y = (rng.random(1) < ptrue[played:played + 1]).astype(float)
        L.update(Dp, y)
        # organic inflow: human games from the same world
        oi = slice(t * ORGANIC, (t + 1) * ORGANIC)
        Do = {k: org[k][oi] for k in ("T0", "T1", "m", "t")}
        po = TR.p(Do, {k: org[k][oi] for k in ("pid", "cnt", "s", "off")} if fam == "fc" else None)
        L.update(Do, (rng.random(ORGANIC) < po).astype(float))
        if policy == "refit" and (t + 1) % 500 == 0:
            L.refit()
            L.buf = []
    r = np.array(log["regret"])
    cost = np.array(log["cost"]) if log["cost"] else np.zeros(0)
    res = {"family": fam, "world": wid, "policy": policy, "compliance": comp, "sd_mult": sd_mult, "rounds": ROUNDS,
           "organic_per_round": ORGANIC, "mean_regret_pts": float(100 * r.mean()),
           "regret_by_quarter_pts": [float(100 * x.mean()) for x in np.array_split(r, 4)],
           "explored_share": float(np.mean(log["explored"])), "followed_share": float(np.mean(log["followed"])),
           "played_explorations": int(len(cost)), "cost_mean_pts": float(100 * cost.mean()) if len(cost) else 0.0,
           "cost_p90_pts": float(100 * np.percentile(cost, 90)) if len(cost) else 0.0,
           "p_cost_gt_2pp": float((cost > 0.02).mean()) if len(cost) else 0.0,
           "below_floor_played": log["below_floor_played"], "secs": time.time() - t0}
    json.dump(res, open(out, "w"))
    print(tag, json.dumps({k: res[k] for k in ("mean_regret_pts", "explored_share", "followed_share", "cost_mean_pts",
                                                "p_cost_gt_2pp", "secs")}), flush=True)


def queue(procs, n_worlds):
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    jobs = []
    for w in range(n_worlds):
        for fam in FAMS:
            for comp in COMPLIANCE:
                for pol in POLICIES:
                    for k in ((1.0, 1.5) if pol in ("tiebreak", "lcb", "top2ts") else (1.0,)):
                        jobs.append((fam, w, pol, comp, k))
    jobs = [j for j in jobs if not os.path.exists(os.path.join(OUT, f"{j[0]}_w{j[1]}_{j[2]}_c{int(100 * j[3])}_k{j[4]}.json"))]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    print(len(jobs), "runs", flush=True)

    def one(j):
        subprocess.call([sys.executable, "-u", os.path.abspath(__file__), "run", "--fam", j[0], "--wid", str(j[1]),
                         "--policy", j[2], "--comp", str(j[3]), "--k", str(j[4])], cwd=TRAINING_DIR, env=env)
    W.Worlds().base_draws([0])
    with ThreadPoolExecutor(procs) as ex:
        list(ex.map(one, jobs))
    print("EXPLORE QUEUE DONE", flush=True)


def summarize():
    import glob
    import rescore as RS
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, "*.json"))]
    out = {"rounds": ROUNDS, "organic_per_round": ORGANIC, "cells": []}
    for fam in FAMS:
        for comp in COMPLIANCE:
            ex = {r["world"]: r for r in rows if r["family"] == fam and r["compliance"] == comp and r["policy"] == "exploit"}
            for pol in POLICIES:
                for k in (1.0, 1.5):
                    R = [r for r in rows if r["family"] == fam and r["compliance"] == comp and r["policy"] == pol and r["sd_mult"] == k]
                    if not R:
                        continue
                    d = [r["mean_regret_pts"] - ex[r["world"]]["mean_regret_pts"] for r in R if r["world"] in ex]
                    out["cells"].append({"family": fam, "compliance": comp, "policy": pol, "sd_mult": k, "worlds": len(R),
                                         "mean_regret_pts": float(np.mean([r["mean_regret_pts"] for r in R])),
                                         "last_quarter_regret_pts": float(np.mean([r["regret_by_quarter_pts"][-1] for r in R])),
                                         "minus_exploit": RS.contrast(np.array(d) / 100) if d else None,
                                         "followed_share": float(np.mean([r["followed_share"] for r in R])),
                                         "explored_share": float(np.mean([r["explored_share"] for r in R])),
                                         "cost_mean_pts": float(np.mean([r["cost_mean_pts"] for r in R])),
                                         "p_cost_gt_2pp": float(np.mean([r["p_cost_gt_2pp"] for r in R])),
                                         "played_explorations": float(np.mean([r["played_explorations"] for r in R])),
                                         "below_floor_played": float(np.mean([r["below_floor_played"] for r in R]))})
    json.dump(out, open(os.path.join(WD.RESULTS, "explore.json"), "w"), indent=1)
    for c in out["cells"]:
        m = c["minus_exploit"]
        print(f"{c['family']} c{c['compliance']} {c['policy']:13s} k{c['sd_mult']} regret {c['mean_regret_pts']:.2f} "
              f"(last q {c['last_quarter_regret_pts']:.2f}) vs exploit {m['mean'] if m else 0:+.3f} "
              f"followed {c['followed_share']:.3f} cost {c['cost_mean_pts']:.2f} P>2pp {c['p_cost_gt_2pp']:.2f}")


# ── diagnostics: what drives extreme true values among candidates ───────
def diag(n_ctx=300, worlds=(0, 1, 2), fams=None):
    rng = np.random.default_rng(5)
    ctx = contexts(n_ctx, seed=9000)
    L = Learner()
    out = {}
    for fam in fams or FAMS:
        rows = {"human": [], "top100": [], "random": []}
        comp = {"main": [], "pairsyn": [], "cp": [], "tri_mu": [], "tri_noise": [], "counter": [], "player": [], "total": []}
        z_by = {"human_minus_best": []}
        for w in worlds:
            TR = Truth(fam, w)
            for i in range(n_ctx):
                D, z, sd, ih, pairs, base3 = candidates(L, ctx, i, rng)
                lob = {k: np.repeat(ctx[k][i:i + 1], len(D["T0"]), 0) for k in ("pid", "cnt", "s", "off")}
                pt = TR._p(D, lob if fam == "fc" else None)        # uncapped: the diagnostics look at the raw tails
                top = set(np.argsort(-z)[:100].tolist())
                for j in range(len(pt)):
                    rows["human" if j == ih else ("top100" if j in top else "random")].append(pt[j])
                z_by["human_minus_best"].append(pt[ih] - pt.max())
                z_by.setdefault("learner_p_of_true_best", []).append(float(M.sig(z[int(np.argmax(pt))])))
                z_by.setdefault("learner_p_human", []).append(float(M.sig(z[ih])))
                if fam in ("a", "c", "fc"):
                    Ws = TR.W
                    th = TR.theta
                    q = Ws.B.unpack(th)
                    P = Ws.B.parts(th, D)
                    c = Ws.components("a" if fam == "fc" else fam, D, w, theta=th)
                    cp0, cp1 = Ws.B.e3(q["E"][D["T0"]]), Ws.B.e3(q["E"][D["T1"]])
                    j = int(np.argmax(pt))
                    comp["main"].append(float(P["main0"][j] - P["main1"][j]))
                    comp["cp"].append(float(cp0[j] - cp1[j]))
                    comp["pairsyn"].append(float(P["int0"][j] - P["int1"][j] - (cp0[j] - cp1[j])))
                    comp["counter"].append(float(P["cross"][j]))
                    comp["tri_mu"].append(float(c["lm0"][j] - c["lm1"][j]))
                    comp["tri_noise"].append(float(c["ln00"][j] - c["ln01"][j]))
                    zt = np.log(pt[j] / (1 - pt[j]))
                    comp["total"].append(float(zt))
                    if fam == "fc":
                        comp["player"].append(float(zt - Ws.combine(c, "a")[j]))
        a = {k: np.array(v) for k, v in rows.items()}
        out[fam] = {"p_quantiles": {k: np.percentile(v, [1, 10, 50, 90, 99]).round(3).tolist() for k, v in a.items()},
                    "share_gt_0.7": {k: float((v > 0.7).mean()) for k, v in a.items()},
                    "share_gt_0.8": {k: float((v > 0.8).mean()) for k, v in a.items()},
                    "human_minus_best_pts": np.percentile(100 * np.array(z_by["human_minus_best"]), [10, 50, 90]).round(2).tolist(),
                    "learner_p_of_true_best_median": float(np.median(z_by["learner_p_of_true_best"])),
                    "learner_p_human_median": float(np.median(z_by["learner_p_human"]))}
        if comp["total"]:
            out[fam]["best_candidate_logit_parts_mean_abs"] = {k: float(np.mean(np.abs(v))) for k, v in comp.items() if v}
            out[fam]["best_candidate_logit_parts_mean"] = {k: float(np.mean(v)) for k, v in comp.items() if v}
        print(fam, json.dumps(out[fam]), flush=True)
    json.dump(out, open(os.path.join(WD.RESULTS, "explore_diag.json"), "w"), indent=1)


# ── value-of-information map ─────────────────────────────────────────────
def voi(fam, wid, n_ctx=3000):
    """Missed true value by hero pair and by role archetype: for each context, every candidate the learner ranks below
    its choice but that is truly better contributes its true gain, attributed to the candidate's added pair and to its
    role composition, weighted by the context's frequency (real games) and P(both players follow at c = 20%)."""
    from paper1_revision import team_support as ts
    out = os.path.join(OUT, f"voi_{fam}_w{wid}.npz")
    if os.path.exists(out):
        return
    rng = np.random.default_rng([wid, 77])
    ctx = contexts(n_ctx, seed=9000)                        # the same contexts in every world and family
    L = Learner()
    TR = Truth(fam, wid)
    pair_v = np.zeros((H, H))
    pair_m3 = np.zeros((H, H))          # gain-weighted sum of log1p(min3) of the teams the pair completes
    pair_w = np.zeros((H, H))
    arch = {}
    for i in range(n_ctx):
        D, z, sd, ih, pairs, base3 = candidates(L, ctx, i, rng)
        lob = {k: np.repeat(ctx[k][i:i + 1], len(D["T0"]), 0) for k in ("pid", "cnt", "s", "off")}
        pt = TR.p(D, lob if fam == "fc" else None)
        a = int(np.argmax(z))
        gain = np.maximum(pt - pt[a], 0)
        n_ph = np.stack([ctx["cnt"][i, 0, 3][pairs[:, 0]], ctx["cnt"][i, 0, 4][pairs[:, 1]]], 1)
        pf = (0.2 * phi(n_ph)).prod(1)
        g = gain * pf
        np.add.at(pair_v, (pairs[:, 0], pairs[:, 1]), g)
        m3 = L.cnt[tri_index(D["T0"])].min(1)
        np.add.at(pair_m3, (pairs[:, 0], pairs[:, 1]), g * np.log1p(m3))
        np.add.at(pair_w, (pairs[:, 0], pairs[:, 1]), g)
        roles = np.sort(ts.ROLE_IDX[D["T0"]], 1)
        for r_, v in zip(map(tuple, roles), g):
            arch[r_] = arch.get(r_, 0.0) + float(v)
    pair_v = pair_v + pair_v.T
    pm = np.where(pair_w + pair_w.T > 0, (pair_m3 + pair_m3.T) / np.maximum(pair_w + pair_w.T, 1e-12), np.nan)
    keys = sorted(arch)
    np.savez(out, pair=pair_v, pair_log1p_min3=pm, n_ctx=n_ctx, arch_keys=np.array(keys),
             arch_v=np.array([arch[k] for k in keys]))


def voi_summary():
    import glob
    from scipy.stats import spearmanr
    from shared import HEROES
    from paper1_revision import team_support as ts
    P = {}
    for f in glob.glob(os.path.join(OUT, "voi_*.npz")):
        fam, w = os.path.basename(f)[4:-4].split("_w")
        P.setdefault(fam, []).append(np.load(f)["pair"])
    iu = np.triu_indices(H, 1)
    M_ = {fam: np.mean(v, 0)[iu] for fam, v in P.items()}
    out = {"families": {f: len(v) for f, v in P.items()}, "spearman_pair_voi": {}, "top_pairs": {}}
    fl = sorted(M_)
    for i, a in enumerate(fl):
        for b in fl[i + 1:]:
            out["spearman_pair_voi"][f"{a}~{b}"] = float(spearmanr(M_[a], M_[b]).correlation)
        # within-family rank stability across worlds
        if len(P[a]) > 1:
            ws = [x[iu] for x in P[a]]
            out.setdefault("spearman_across_worlds", {})[a] = float(np.mean([spearmanr(ws[j], ws[k]).correlation
                                                                             for j in range(len(ws)) for k in range(j + 1, len(ws))]))
        o = np.argsort(-M_[a])[:15]
        out["top_pairs"][a] = [(HEROES[iu[0][j]], HEROES[iu[1][j]], float(M_[a][j])) for j in o]
    out["projection"] = project(P)
    json.dump(out, open(os.path.join(WD.RESULTS, "explore_voi.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:6000])


SCENARIOS = [25, 300, 2000]       # recommendation-served drafts per week (Max, 2026-11-15: about 24 today)


def pick_follow(comp):
    """P(one player follows a top-ranked recommendation) = c * mean phi over the lobbies' players and heroes."""
    import player as PL
    cz = np.load(PL.COV)
    return float(comp * phi(cz["pool_cnt"][:, 0, 3:5].reshape(-1, H)).mean())


def partner_share():
    """per hero: the share of training teams that contain it (the chance the humans' own other pick is that hero)."""
    c1 = np.load(C3_FILE.replace("triple_counts.npy", "corpus.npz"))
    return np.asarray(c1["c1"], float) / float(c1["n"])


def pair_follow(comp):
    """P(both remaining players follow a top-ranked recommendation) at pick-level compliance c, averaged over the
    exploration contexts' lobbies (phi from their earlier games on the recommended heroes: here the typical mix)."""
    import player as PL
    cz = np.load(PL.COV)
    n = cz["pool_cnt"][:, 0, 3:5].reshape(-1, H)
    ph = phi(n).mean()                                     # average over heroes and players
    return float((comp * ph) ** 2)


def project(P):
    """Games and weeks to bring the top-VOI pairs into support (min3 >= 25) and to measure a 2 pt effect (~600 games
    per composition at 1 SE), per traffic scenario and compliance level; what a directed 2,000-draft/week push at
    c = 20% buys in 3 and 12 months (pairs filled greedily by VOI per game needed)."""
    from shared import HEROES
    import glob
    iu = np.triu_indices(H, 1)
    q = partner_share()
    qbar = float(q.mean())
    out = {"pair_follow_prob": {str(c): pair_follow(c) for c in COMPLIANCE}, "scenarios_drafts_per_week": SCENARIOS,
           "modes": {"both": "both remaining players follow independently: (c phi)^2",
                     "lobby": "coordinated lobby (premade / partner team): one decision covers both picks: c phi",
                     "single": "only the last pick is recommended; the other hero is the humans' own, so a target pair "
                               "is filled only when the humans happen to pick its partner: c phi x partner share "
                               f"(mean share of training teams containing a hero = {qbar:.3f})"},
           "table": [], "push_2000_c20": {}}
    for c in COMPLIANCE:
        for mode in ("both", "lobby", "single"):
            pf = {"both": pair_follow(c), "lobby": pick_follow(c), "single": pick_follow(c) * qbar}[mode]
            for d in SCENARIOS:
                g = d * pf
                out["table"].append({"mode": mode, "drafts_per_week": d, "compliance": c, "pair_games_per_week": g,
                                     "weeks_for_25_games": 25 / g if g else None, "weeks_for_600_games": 600 / g if g else None,
                                     "speed_vs_both": pf / pair_follow(c)})
    for fam in sorted(P):
        Z = [np.load(f) for f in glob.glob(os.path.join(OUT, f"voi_{fam}_w*.npz"))]
        if not Z or "pair_log1p_min3" not in Z[0].files:
            continue
        v = np.mean([z["pair"] for z in Z], 0)[iu] / float(Z[0]["n_ctx"])         # true win gain per draft, if resolved
        m3 = np.expm1(np.nanmean([z["pair_log1p_min3"] for z in Z], 0)[iu])
        need = np.maximum(25 - np.nan_to_num(m3, nan=0.0), 0)
        rank = np.argsort(-(v / np.maximum(need, 1)))
        res = {}
        for mode, months in [(m, mo) for m in ("both", "lobby", "single") for mo in (3, 12)]:
            pf = {"both": pair_follow(0.2), "lobby": pick_follow(0.2), "single": pick_follow(0.2) * qbar}[mode]
            budget = 2000 * pf * months * 52 / 12
            used, gain, pairs = 0.0, 0.0, []
            for j in rank:
                if v[j] <= 0:
                    break
                if used + need[j] > budget:
                    continue
                used += need[j]
                gain += v[j]
                if need[j] > 0:
                    pairs.append((HEROES[iu[0][j]], HEROES[iu[1][j]], round(float(need[j]), 1), round(float(100 * v[j]), 4)))
            res[f"{mode}_{months}_months"] = {"followed_games": budget, "pairs_into_support": len(pairs),
                                       "expected_true_gain_pts_per_draft": float(100 * gain), "top_pairs": pairs[:15]}
        out["push_2000_c20"][fam] = res
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "queue", "summary", "voi", "voi_queue", "voi_summary", "diag"])
    ap.add_argument("--fam", default="a")
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--policy", default="exploit")
    ap.add_argument("--comp", type=float, default=0.2)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--worlds", type=int, default=5)
    a = ap.parse_args()
    if a.stage == "run":
        run(a.fam, a.wid, a.policy, a.comp, a.k)
    elif a.stage == "queue":
        queue(a.procs, a.worlds)
    elif a.stage == "summary":
        summarize()
    elif a.stage == "diag":
        diag()
    elif a.stage == "voi":
        voi(a.fam, a.wid)
    elif a.stage == "voi_queue":
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.procs) as ex:
            list(ex.map(voi, [f for w in range(a.worlds) for f in FAMS], [w for w in range(a.worlds) for f in FAMS]))
    else:
        voi_summary()
