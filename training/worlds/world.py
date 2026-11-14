"""
Hidden worlds: ground-truth outcome models drawn from the world model's posterior (model.py), one
error family at a time, with explicit off-support dials (dials.json, fixed before any agent is scored).

Families (never pooled):
  a  triple     own-team interactions: synergy S, CP triple term, triple layer; support = min3
  b  cross      cross-team: counters K and the (pair, opposing hero) layer; support = the team's
                rarest (own pair, opposing hero) cell
  c  body45     own-team interactions with the quartet and five-hero layers; support = min4
  d  hmt        hero x map x tier: V, U and the hmt layer; support = the team's rarest (hero, map, tier)
  e  drift      family a at default dials, plus a play-time move of w, S, K drawn at the size measured
                from the training split to 2.55.17 (drift.json); dial = drift scale
  f  player     player-hero skill (player.py)
  g  ensemble   the drafter's own network class (ensemble.py); self-grading for ensemble-LCB agents

A world w of family F: base parameters theta_w = theta_hat + L^-T z_w (Laplace), layer cells
tau_c = mu_c + k(n_c) sd_c zeta(w, c) (unseen cells: mu = 0, sd = s, n = 0), zeta a counter-based
normal (hash of cell key, world id, layer), so a world is a pure function of any team, seen or not.

Dials (per family, graded by support through g(m) = clip(1 - log(1+m) / log(1+F), 0, 1), F the
family floor: 1 for an unseen cell or team, 0 at or above the floor):
  k      off-support scale: layer cell SD x (1 + (k-1) g(n_c))
  delta  mean shift (win-prob points at p = 0.5, i.e. logit 0.04 delta) on a team's strength, x g(m_t)
  a      interaction attenuation: the family's interaction class of team t x (1 + (a-1) g(m_t))
  sigma  extra off-support team noise (points): sigma g(m_t) zeta(w, team), one draw per five-hero team
         and family. Added after the fit showed the layers' evidence-chosen SDs are tiny (the data rule
         out large residual effects on average), so k alone cannot move the off-support spread.
Base draws (theta_w) are shared across families for the same world id, so families are paired.
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import data as WD  # noqa: E402
import model as M  # noqa: E402

FAMILIES = {"a": "triple", "b": "cross", "c": "body45", "d": "hmt", "e": "drift"}
FAM_LAYERS = {"a": ["triple"], "b": ["cross"], "c": ["quartet", "five"], "d": ["hmt"], "e": ["triple"]}
LAYER_SALT = {"triple": 11, "quartet": 13, "five": 17, "cross": 19, "hmt": 23}
DIALS_JSON = os.path.join(HERE, "dials.json")
PTS = 0.04          # logit per win-prob point at p = 0.5
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix(x):
    x = (np.asarray(x).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15))
    z = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def hnorm(keys, wid, salt):
    """Standard normal per (cell key, world, layer), reproducible and order-free."""
    from scipy.special import ndtri
    with np.errstate(over="ignore"):
        s = splitmix(np.uint64(wid) * np.uint64(1000003) + np.uint64(salt))
        h = splitmix(np.asarray(keys).astype(np.uint64) ^ s)
    u = ((h >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0 ** -53
    return ndtri(u)


def grade(m, F):
    return np.clip(1 - np.log1p(np.asarray(m, float)) / np.log1p(F), 0, 1)


def dials():
    return json.load(open(DIALS_JSON))


def grid_cells(G):
    """The pre-registered break-even planes (not the full factorial): delta x sigma, delta x k and
    delta x a, each with the other dials at default; duplicates dropped."""
    out, seen = [], set()
    planes = [dict(k=1.0, a=1.0, sigma=sg) for sg in G["sigma"]] + [dict(k=k, a=1.0, sigma=0.0) for k in G["k"]] \
        + [dict(k=1.0, a=a, sigma=0.0) for a in G["a"]]
    for p in planes:
        for d in G["delta"]:
            c = dict(p, delta=d)
            key = tuple(sorted(c.items()))
            if key not in seen:
                seen.add(key)
                out.append(c)
    return out


class Layer:
    def __init__(self, name):
        d = M.load_layer(name)
        self.name, self.keys, self.n, self.mu, self.sd, self.s = name, d["keys"], d["n"], d["mu"], d["sd"], d["s"]

    def look(self, K):
        i = np.clip(np.searchsorted(self.keys, K), 0, len(self.keys) - 1)
        hit = self.keys[i] == K
        return (np.where(hit, self.mu[i], 0.0), np.where(hit, self.sd[i], self.s), np.where(hit, self.n[i], 0))

    def count(self, K):
        i = np.clip(np.searchsorted(self.keys, K), 0, len(self.keys) - 1)
        return np.where(self.keys[i] == K, self.n[i], 0)


class Worlds:
    """All hero-generic families on one fitted world model."""

    def __init__(self):
        self.B, self.theta, self.cfg = M.load_base()
        self.nm, self.nt = self.cfg["nm"], self.cfg["nt"]
        self._L = None
        self._draws = {}
        self.layers = {}
        self.D = dials()
        self.drift = self.D.get("drift")

    # base draws
    def L(self):
        if self._L is None:
            self._L = np.load(M.CHOL_NPY, mmap_mode="r")
        return self._L

    def base_draws(self, wids):
        """theta_w for the given world ids (cached on disk in blocks of 1,000 ids)."""
        import scipy.linalg as sl
        out = []
        for w in wids:
            blk = int(w) // 1000
            if blk not in self._draws:
                p = os.path.join(WD.CACHE, f"base_draws_{blk}.npy")
                if not os.path.exists(p):
                    Z = np.stack([np.random.default_rng([7, blk * 1000 + i]).standard_normal(self.B.P) for i in range(1000)], 1)
                    X = sl.solve_triangular(np.asarray(self.L()), Z, lower=True, trans="T")
                    np.save(p, X.astype(np.float32))
                self._draws[blk] = np.load(p, mmap_mode="r")
            out.append(self.theta + np.asarray(self._draws[blk][:, int(w) % 1000], np.float64))
        return out

    def layer(self, name):
        if name not in self.layers:
            self.layers[name] = Layer(name)
        return self.layers[name]

    # support statistics
    def team_support(self, fam, D):
        """(m0, m1): the family's support statistic of each team (training-corpus counts)."""
        if fam in ("a", "e"):
            L = self.layer("triple")
            return [L.count(M.team_keys("triple", T)).min(1) for T in (D["T0"], D["T1"])]
        if fam == "c":
            L = self.layer("quartet")
            return [L.count(M.team_keys("quartet", T)).min(1) for T in (D["T0"], D["T1"])]
        if fam == "b":
            L = self.layer("cross")
            return [L.count(M.cross_keys(D["T0"], D["T1"])).min(1), L.count(M.cross_keys(D["T1"], D["T0"])).min(1)]
        if fam == "d":
            L = self.layer("hmt")
            return [L.count(M.team_keys("hmt", T, D["m"], D["t"], self.nm, self.nt)).min(1) for T in (D["T0"], D["T1"])]
        raise ValueError(fam)

    def floor(self, fam):
        return self.D["floors"][FAMILIES[fam] if fam != "e" else "triple"]

    # per-world components: everything a dial setting needs, linear pieces kept apart
    def components(self, fam, D, wid, theta=None):
        th = self.base_draws([wid])[0] if theta is None else theta
        P = self.B.parts(th, D)
        q = self.B.unpack(th)
        F = self.floor(fam)
        m0, m1 = self.team_support(fam, D)
        c = dict(main=P["main0"] - P["main1"], g0=grade(m0, F), g1=grade(m1, F), m0=m0, m1=m1)
        if fam == "d":   # the interaction class is the map / tier specific part
            for s, T in (("0", D["T0"]), ("1", D["T1"])):
                vu = (q["V"][D["m"][:, None], T] + q["U"][D["t"][:, None], T]).sum(1)
                c["main"] = c["main"] - (vu if s == "0" else -vu)
                c["own" + s] = vu
            c["own0"] = c["own0"] + 0.0
            c["rest"] = P["int0"] - P["int1"] + P["cross"]
        elif fam == "b":
            c["own0"], c["own1"] = np.zeros(len(c["main"])), np.zeros(len(c["main"]))
            c["crossK"] = P["cross"]
            c["rest"] = P["int0"] - P["int1"]
        else:
            c["own0"], c["own1"] = P["int0"], P["int1"]
            c["rest"] = P["cross"]
        # layer cells: mean and the two noise pieces (all teams' own cells; family b: cross cells)
        for s in ("0", "1"):
            c["lm" + s] = 0.0
            c["ln0" + s] = 0.0
            c["ln1" + s] = 0.0
        for name in FAM_LAYERS[fam]:
            L = self.layer(name)
            for s, (T, O) in (("0", (D["T0"], D["T1"])), ("1", (D["T1"], D["T0"]))):
                K = M.cross_keys(T, O) if name == "cross" else M.team_keys(name, T, D["m"], D["t"], self.nm, self.nt)
                mu, sd, n = L.look(K)
                z = hnorm(K, wid, LAYER_SALT[name]) * sd
                c["lm" + s] = c["lm" + s] + mu.sum(1)
                c["ln0" + s] = c["ln0" + s] + z.sum(1)
                c["ln1" + s] = c["ln1" + s] + (z * grade(n, F)).sum(1)
        for s, T in (("0", D["T0"]), ("1", D["T1"])):
            c["zt" + s] = hnorm(M.team_keys("five", T)[:, 0], wid, 61 + ord(fam))
        if fam == "e":
            c["drift"] = self.drift_logit(D, wid)
        return c

    @staticmethod
    def combine(c, fam, k=1.0, delta=0.0, a=1.0, drift=1.0, sigma=0.0):
        """logit of team 0 winning under one dial setting."""
        A0 = 1 + (a - 1) * c["g0"]
        A1 = 1 + (a - 1) * c["g1"]
        t0 = A0 * (c["own0"] + c["lm0"] + c["ln00"] + (k - 1) * c["ln10"]) + PTS * delta * c["g0"]
        t1 = A1 * (c["own1"] + c["lm1"] + c["ln01"] + (k - 1) * c["ln11"]) + PTS * delta * c["g1"]
        z = c["main"] + c["rest"] + t0 - t1
        if sigma:
            z = z + PTS * sigma * (c["g0"] * c["zt0"] - c["g1"] * c["zt1"])
        if fam == "b":
            z = z + 0.5 * (A0 + A1) * c["crossK"]
        if fam == "e":
            z = z + drift * c["drift"]
        return z

    def logit(self, fam, D, wid, **dial):
        return self.combine(self.components(fam, D, wid), fam, **dial)

    def posterior_mean_logit(self, fam, D):
        """The world model's own prediction (posterior mean of the base, layer means, default dials)."""
        c = self.components(fam, D, 0, theta=self.theta)
        for s in ("0", "1"):
            c["ln0" + s] = 0.0 * c["ln0" + s]
            c["ln1" + s] = 0.0 * c["ln1" + s]
        if fam == "e":
            c["drift"] = 0.0 * c["drift"]
        return self.combine(c, fam)

    # drift: a play-time move of w, S, K (sizes in dials.json "drift", measured by drift.py)
    def drift_logit(self, D, wid):
        d = self.drift
        rng = np.random.default_rng([99, int(wid)])
        dw = rng.normal(0, d["sd_w"], M.H)
        dS = rng.normal(0, d["sd_S"], M.NPAIR)
        dK = rng.normal(0, d["sd_K"], M.NPAIR)
        th = np.zeros(self.B.P)
        o = self.B.off
        th[o["w"][0]:o["w"][1]] = dw
        th[o["S"][0]:o["S"][1]] = dS
        th[o["K"][0]:o["K"][1]] = dK
        th[o["E"][0]:o["E"][1]] = 0.0
        P = self.B.parts(th, D)
        return P["main0"] - P["main1"] + P["int0"] - P["int1"] + P["cross"]


class Oracle:
    """One world as an environment for a sequential learner (step 3): query outcome probabilities, play
    games, and keep a log of what was played."""

    def __init__(self, worlds, fam, wid, seed=0, **dial):
        self.W, self.fam, self.wid, self.dial = worlds, fam, wid, dial
        self.theta = worlds.base_draws([wid])[0]
        self.rng = np.random.default_rng([int(wid), seed])
        self.log = []

    def prob(self, D):
        return M.sig(self.W.combine(self.W.components(self.fam, D, self.wid, theta=self.theta), self.fam, **self.dial))

    def play(self, D):
        p = self.prob(D)
        y = (self.rng.random(len(p)) < p).astype(float)
        self.log.append({k: v for k, v in D.items()} | {"y": y})
        return y
