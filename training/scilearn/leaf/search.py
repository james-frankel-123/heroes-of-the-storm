"""Draft search with a LeafModel as the value function.

Hooks into the CPU reference search training/cuda_mcts/ref_search.py without
changing it: RefSearch takes any object with the game interface
(kind, legal, prior, chance, apply, value, n_actions), and HotsDraftGame
below is that object for the ranked draft with an additive leaf.

The game follows the v2 kernel (search_v2.cuh, SEARCH_CHANCE):
  draft order     train_draft_policy.DRAFT_ORDER (16 steps, bans included)
  decision nodes  our team to move (picks and bans), PUCT c = 2.0
  chance nodes    opponent to move, outcome ~ GD; progressive widening
                  pw_k = 1.0, pw_alpha = 0.5
  leaf            partial states are rolled out to the complete draft with
                  GD sampling for both teams, then the leaf is evaluated on
                  the complete draft: sigmoid(eta - kappa * sd), our team in
                  slots 0-4. The kernel does the same with its WP network.
  root action     argmax of root visits (root_temp 0, the paper-1 operating
                  point), lowest hero index on ties
Differences from the kernel, all documented in the README:
  priors          the kernel uses a trained policy head; here the prior is GD
                  (ensemble mean) mixed with a uniform floor prior_eps over
                  legal heroes, so heroes GD never picks (Xal'atath) stay
                  reachable. prior="uniform" is also available.
  GD              the 5 v1 GD models (90 heroes, 14 maps) averaged; leafpack
                  heroes are matched to GD by name. A leafpack hero GD does
                  not know (Xal'atath) has GD probability 0, so opponents
                  never pick or ban it; a leafpack map GD does not know
                  (Haunted Mines) gets an all-zero map one-hot.
Greedy follows paper1_revision/greedy_evals.py task_wr: at each own pick,
every legal hero is tried, the draft is completed by GD argmax (a random
ensemble member per step, both teams), the leaf is evaluated, and the argmax
is played; own bans and all opponent moves are GD argmax of a random member.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TRAINING / "cuda_mcts"))
sys.path.insert(0, str(TRAINING))
from leaf import LeafModel, sigmoid  # noqa: E402
from ref_search import (KIND_CHANCE, KIND_DECISION, KIND_TERMINAL,  # noqa: E402
                        RefSearch)
import shared  # noqa: E402

# Paper-1 draft order (train_draft_policy.DRAFT_ORDER), copied to avoid a torch import.
DRAFT_ORDER = [
    (0, "ban"), (1, "ban"), (0, "ban"), (1, "ban"),
    (0, "pick"), (1, "pick"), (1, "pick"), (0, "pick"), (0, "pick"),
    (1, "ban"), (0, "ban"),
    (1, "pick"), (1, "pick"), (0, "pick"), (0, "pick"), (1, "pick"),
]
N_STEPS = len(DRAFT_ORDER)
GD_HEROES = list(shared.HEROES[:90])     # v1 encoding the GD models were trained on
GD_MAPS = list(shared.MAPS[:14])
GD_TIERS = ["low", "mid", "high"]
GD_DIM = 90 * 3 + 14 + 3 + 2
DEFAULT_GD_DIR = TRAINING / "rerun2026" / "models"
C_PUCT, PW_K, PW_ALPHA = 2.0, 1.0, 0.5


class GDOpponent:
    """Numpy forward of the GD ensemble (GenericDraftModel: 289-256-128-90 ReLU MLP)."""

    def __init__(self, leaf: LeafModel, gd_dir=DEFAULT_GD_DIR, members=(0, 1, 2, 3, 4), kind="gd"):
        self.kind, self.H = kind, leaf.H
        self.leaf2gd = np.array([GD_HEROES.index(h) if h in GD_HEROES else -1 for h in leaf.heroes])
        self.gd_known = self.leaf2gd >= 0
        self.gd2leaf = np.full(90, -1)
        self.gd2leaf[self.leaf2gd[self.gd_known]] = np.flatnonzero(self.gd_known)
        if kind == "uniform":
            return
        import torch
        Ws = [torch.load(Path(gd_dir) / f"generic_draft_{i}.pt", map_location="cpu", weights_only=True)
              for i in members]
        f = lambda k: np.stack([w[k].numpy().astype(np.float64) for w in Ws])  # noqa: E731
        self.W1T = f("net.0.weight").transpose(0, 2, 1)   # [G, 289, 256]
        self.b1 = f("net.0.bias")
        self.W2T = f("net.3.weight").transpose(0, 2, 1)   # [G, 256, 128]
        self.b2 = f("net.3.bias")
        self.W3T = f("net.6.weight").transpose(0, 2, 1)   # [G, 128, 90]
        self.b3 = f("net.6.bias")
        if self.W1T.shape[1] != GD_DIM:
            raise ValueError(f"GD input dim {self.W1T.shape[1]} != {GD_DIM}")
        self.G = len(members)

    def encode(self, state, map_name, tier):
        """GD input (DraftState.to_numpy minus our_team), as active indices and values."""
        step, t0, t1, bans = state
        idx, val = [], []
        for off, heroes in ((0, t0), (90, t1), (180, bans)):
            for h in heroes:
                g = self.leaf2gd[h]
                if g >= 0:
                    idx.append(off + g)
                    val.append(1.0)
        if map_name in GD_MAPS:
            idx.append(270 + GD_MAPS.index(map_name)); val.append(1.0)
        if tier in GD_TIERS:
            idx.append(284 + GD_TIERS.index(tier)); val.append(1.0)
        idx += [287, 288]
        val += [step / 15.0, 0.0 if DRAFT_ORDER[step][1] == "ban" else 1.0]
        return np.array(idx), np.array(val)

    def logits(self, X_idx, X_val):
        """One state -> GD logits [G, 90]."""
        h = self.b1 + np.einsum("gid,i->gd", self.W1T[:, X_idx, :], X_val)
        h = np.maximum(h, 0)
        h = np.maximum(np.einsum("gd,gde->ge", h, self.W2T) + self.b2, 0)
        return np.einsum("ge,gef->gf", h, self.W3T) + self.b3

    def logits_batch(self, X):
        """Dense states [C, 289] -> logits [G, C, 90]."""
        h = np.maximum(np.einsum("ci,gid->gcd", X, self.W1T) + self.b1[:, None], 0)
        h = np.maximum(np.einsum("gcd,gde->gce", h, self.W2T) + self.b2[:, None], 0)
        return np.einsum("gce,gef->gcf", h, self.W3T) + self.b3[:, None]

    def probs(self, state, map_name, tier, legal):
        """Ensemble-mean GD distribution over leafpack heroes (0 for unknown or taken)."""
        p = np.zeros(self.H)
        if self.kind == "uniform":
            p[legal] = 1.0
            return p / p.sum()
        gl = legal[self.gd2leaf.clip(0)] & (self.gd2leaf >= 0)
        if not gl.any():          # only GD-unknown heroes left
            p[legal] = 1.0
            return p / p.sum()
        lg = self.logits(*self.encode(state, map_name, tier))
        lg = np.where(gl[None], lg, -np.inf)
        e = np.exp(lg - lg.max(1, keepdims=True))
        q = (e / e.sum(1, keepdims=True)).mean(0)
        p[self.gd2leaf[gl]] = q[gl]
        return p / p.sum()


def new_state():
    return (0, (), (), ())


def apply_action(state, a):
    step, t0, t1, bans = state
    team, typ = DRAFT_ORDER[step]
    if typ == "ban":
        return (step + 1, t0, t1, bans + (a,))
    if team == 0:
        return (step + 1, t0 + (a,), t1, bans)
    return (step + 1, t0, t1 + (a,), bans)


def legal_mask(state, H):
    m = np.ones(H, bool)
    for part in state[1:]:
        for h in part:
            m[h] = False
    return m


class HotsDraftGame:
    """RefSearch game interface for one draft configuration (map, tier, our team)."""

    def __init__(self, leaf: LeafModel, gd: GDOpponent, map_name, tier, our_team,
                 kappa=None, prior="gd", prior_eps=0.1):
        self.leaf, self.gd = leaf, gd
        self.map, self.tier, self.our = map_name, tier, our_team
        self.kappa = leaf.kappa if kappa is None else float(kappa)
        self.prior_kind, self.prior_eps = prior, prior_eps
        self.n_actions = leaf.H
        self.n_leaf_evals = 0

    def kind(self, s):
        if s[0] >= N_STEPS:
            return KIND_TERMINAL
        return KIND_DECISION if DRAFT_ORDER[s[0]][0] == self.our else KIND_CHANCE

    def legal(self, s):
        return legal_mask(s, self.n_actions)

    def prior(self, s):
        leg = self.legal(s)
        u = leg / leg.sum()
        if self.prior_kind == "uniform":
            return u
        return (1 - self.prior_eps) * self.gd.probs(s, self.map, self.tier, leg) + self.prior_eps * u

    def chance(self, s):
        return self.gd.probs(s, self.map, self.tier, self.legal(s))

    def apply(self, s, a):
        return apply_action(s, int(a))

    def draft_row(self, s):
        ours, theirs = (s[1], s[2]) if self.our == 0 else (s[2], s[1])
        return np.array(list(ours) + list(theirs))

    def leaf_value(self, s):
        self.n_leaf_evals += 1
        return float(self.leaf.leaf_value(self.draft_row(s)[None], self.map, self.tier, self.kappa)[0])

    def value(self, s, rng):
        """GD rollout (both teams) to the complete draft, then the leaf."""
        while s[0] < N_STEPS:
            p = self.chance(s)
            s = apply_action(s, int(rng.choice(self.n_actions, p=p)))
        return self.leaf_value(s)


def _cfg_seed(*parts):
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


def mcts_draft(leaf, gd, map_name, tier, our_team, sims, search_seed, opp_seed,
               kappa=None, prior="gd", prior_eps=0.1, gd_bans=False):
    """Play one full draft: our moves by RefSearch at `sims` simulations, opponent
    moves sampled from GD with an RNG that depends only on opp_seed. gd_bans=True
    plays our bans as the GD ensemble-mean argmax instead of searching them."""
    g = HotsDraftGame(leaf, gd, map_name, tier, our_team, kappa, prior, prior_eps)
    opp_rng = np.random.default_rng(opp_seed)
    s, steps = new_state(), []
    while s[0] < N_STEPS:
        team, typ = DRAFT_ORDER[s[0]]
        if team == our_team and typ == "ban" and gd_bans:
            a = int(np.argmax(g.chance(s)))
            steps.append({"step": s[0], "type": typ, "ours": True, "hero": a})
        elif team == our_team:
            rs = RefSearch(g, c_puct=C_PUCT, pw_k=PW_K, pw_alpha=PW_ALPHA,
                           seed=_cfg_seed(search_seed, s[0], opp_seed))
            nodes = rs.search(s, sims)
            root = nodes[0]
            vis = np.array([nodes[c].visits if c >= 0 else -1 for c in root.child])
            a = int(np.argmax(vis))
            steps.append({"step": s[0], "type": typ, "ours": True, "hero": a,
                          "root_q": nodes[root.child[a]].q})
        else:
            a = int(opp_rng.choice(g.n_actions, p=g.chance(s)))
            steps.append({"step": s[0], "type": typ, "ours": False, "hero": a})
        s = apply_action(s, a)
    return _record(leaf, g, s, steps)


def greedy_draft(leaf, gd, map_name, tier, our_team, seed, kappa=None):
    """paper1_revision/greedy_evals.py task_wr protocol with the leaf as eval_fn."""
    g = HotsDraftGame(leaf, gd, map_name, tier, our_team, kappa)
    rng = np.random.default_rng(seed)
    s, steps = new_state(), []

    def gd_argmax_batch(states):
        """GD argmax of a random member per state, over legal GD-known heroes."""
        if gd.kind == "uniform":
            return [int(rng.choice(np.flatnonzero(legal_mask(x, leaf.H)))) for x in states]
        X = np.zeros((len(states), 289))
        for i, x in enumerate(states):
            ix, v = gd.encode(x, map_name, tier)
            X[i, ix] = v
        lg = gd.logits_batch(X)                                   # [G, C, 90]
        mem = rng.integers(0, gd.G, len(states))
        out = []
        for i, x in enumerate(states):
            leg = legal_mask(x, leaf.H)
            gl = leg[gd.gd2leaf.clip(0)] & (gd.gd2leaf >= 0)
            row = np.where(gl, lg[mem[i], i], -np.inf)
            out.append(int(gd.gd2leaf[int(np.argmax(row))]))
        return out

    while s[0] < N_STEPS:
        team, typ = DRAFT_ORDER[s[0]]
        if team == our_team and typ == "pick":
            cands = np.flatnonzero(legal_mask(s, leaf.H))
            sts = [apply_action(s, int(h)) for h in cands]
            while sts[0][0] < N_STEPS:
                acts = gd_argmax_batch(sts)
                sts = [apply_action(x, a) for x, a in zip(sts, acts)]
            rows = np.stack([g.draft_row(x) for x in sts])
            v = leaf.leaf_value(rows, map_name, tier, g.kappa)
            a = int(cands[int(np.argmax(v))])
            steps.append({"step": s[0], "type": typ, "ours": True, "hero": a, "root_q": float(v.max())})
        else:
            a = gd_argmax_batch([s])[0]
            steps.append({"step": s[0], "type": typ, "ours": team == our_team, "hero": a})
        s = apply_action(s, a)
    return _record(leaf, g, s, steps)


def _record(leaf, g, s, steps):
    row = g.draft_row(s)
    names = [leaf.heroes[h] for h in row]
    return {"map": g.map, "tier": g.tier, "our_team": g.our, "our": names[:5], "opp": names[5:],
            "draft": [int(h) for h in row], "steps": steps,
            "degenerate": bool(shared.is_degenerate(names[:5])),
            "has_healer": any(shared.FINE_TO_BLIZZ_ROLE.get(shared.HERO_ROLE_FINE.get(h, ""))
                              == "Healer" for h in names[:5]),
            "p_self": float(sigmoid(leaf.evaluate(row[None], g.map, g.tier)[0][0])),
            "leaf_evals": g.n_leaf_evals}
