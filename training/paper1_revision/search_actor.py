"""
Decision-time MCTS for the F_oof agents: a CPU (numpy) port of the v2 kernel
search (cuda_mcts/search_v2.cuh, SEARCH_CHANCE; the algorithm RefSearch in
cuda_mcts/ref_search.py mirrors), callable as a tournament / rich-eval actor.

Per search (one own decision), exactly the kernel's rules:
  root            expanded with the policy-head priors (masked softmax), no
                  Dirichlet noise (dir_eps = 0 in training and in the bench)
  decision nodes  PUCT, score = Q + c_puct * P * sqrt(N_parent) / (1 + N),
                  Q = N = 0 for untried actions, first maximum (lowest hero
                  index) on ties; children created lazily
  chance nodes    opponent to move: GD masked softmax (temperature 1) at the
                  node's exact state, computed once; progressive widening
                  limit(N) = max(1, ceil(pw_k * N^pw_alpha - 1e-3)); below the
                  limit the reply is drawn from the full GD distribution, at
                  the limit from GD renormalized over the existing children
  leaf            first unexpanded own decision (expanded with policy priors)
                  or a terminal draft; GD rollout to the terminal draft for
                  both teams; value = symmetrized leaf WP, our perspective
  backup          every node on the path: N += 1, W += v
  move            root-visit argmax (temp 0) or a draw proportional to the
                  root visits (temp 1)
Parameters (training: paper1_revision/train_mcts.py -> train_mcts_worker.py
run_episodes_into_buffer(c_puct 2.0, root_temp 1.0, dir_eps 0.0), search mode
chance, MCTS_PW_K 1.0, MCTS_PW_ALPHA 0.5, as recorded in the F_oof runs'
kernel_info.json; benchmark: overfit2026/search.run, the same with 200 sims):
  C_PUCT = 2.0, PW_K = 1.0, PW_ALPHA = 0.5, GD temperature 1, no root noise.
The value head is not used (the kernel does not use it).

Leaf values
  enriched    the leaf the F_oof agents trained against: paper1_revision
              site/models/enriched_s<selected>.pt (283-d) with deploy
              statistics, symmetrized 0.5 * (p(t0, t1) + 1 - p(t1, t0)).
  lcb:k       the 20-member out-of-fold ensemble of deferred.py job_ensemble
              (retrained by train_ensemble_site.py into site/models/ensemble_lcb;
              P1_ENSEMBLE_DIR overrides): each member symmetrized the same way,
              value = mean_m p_m - k * sd_m p_m (population sd over members,
              the variance job_ensemble reports), our perspective.

Search-internal opponent and rollout policy: the GD model the agent trained
against (MCTS_GD_PATH = rerun2026/ns/p1site/models/generic_draft_0.pt), unless
a GD index is given (the benchmark mirror uses the batch's GD, as the kernel
benchmark does).

All the deterministic network outputs (policy priors, GD distributions, leaf
values) are memoized by exact state, so results equal the uncached search.
"""
import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)

import numpy as np

C_PUCT = 2.0
PW_K = 1.0
PW_ALPHA = 0.5
SIMS = 200
NH = 90
CHILD_NONE, CHILD_LAZY = -2, -1
KIND_DECISION, KIND_CHANCE, KIND_TERMINAL = 0, 1, 2
CACHE_CAP = 400_000


# ── draft order and state vector (train_draft_policy.DraftState.to_numpy layout) ──

def _order():
    from train_draft_policy import DRAFT_ORDER
    return [t for t, _ in DRAFT_ORDER], [1.0 if k == "pick" else 0.0 for _, k in DRAFT_ORDER]


TEAM_OF, IS_PICK = _order()
STEP_COL, TYPE_COL, OUR_COL = 287, 288, 289       # 3*90 + 14 maps + 3 tiers


def state_vec(state):
    """DraftState -> 290-d float32 (to_numpy layout; last entry = our team)."""
    return state.to_numpy().astype(np.float32).copy()


def apply(x, a):
    """New state vector after the current step's team takes hero a."""
    step = int(round(x[STEP_COL] * 15.0))
    y = x.copy()
    if IS_PICK[step] == 0.0:
        y[180 + a] = 1.0
    else:
        y[(0 if TEAM_OF[step] == 0 else 90) + a] = 1.0
    step += 1
    y[STEP_COL] = step / 15.0
    y[TYPE_COL] = IS_PICK[step] if step < 16 else 0.0
    return y, step


def step_of(x):
    return int(round(x[STEP_COL] * 15.0))


def free_mask(x):
    return (x[0:90] + x[90:180] + x[180:270]) < 0.5


# ── numpy networks (eval mode, BatchNorm folded) ──

def _fold(lin, bn=None):
    W = lin.weight.detach().double().cpu().numpy()
    b = lin.bias.detach().double().cpu().numpy()
    if bn is not None:
        g = (bn.weight.detach().double() / (bn.running_var.detach().double() + bn.eps).sqrt()).cpu().numpy()
        W = W * g[:, None]
        b = (b - bn.running_mean.detach().double().cpu().numpy()) * g + bn.bias.detach().double().cpu().numpy()
    return np.ascontiguousarray(W.T, dtype=np.float32), b.astype(np.float32)


def _softmax_masked(logits, free):
    z = np.where(free, logits.astype(np.float64), -np.inf)
    z = np.exp(z - z.max())
    return z / z.sum()


class PolicyNP:
    """AlphaZeroDraftNet policy head (linear head, base size), numpy."""

    def __init__(self, path):
        from rerun2026.constrained_search import load_mcts_policy
        net = load_mcts_policy(path)
        assert net._policy_head_type == "linear"
        self.inp = _fold(net.input_fc, net.input_bn)
        self.blocks = [(_fold(b.fc1, b.bn1), _fold(b.fc2, b.bn2)) for b in net.res_blocks]
        self.c1 = _fold(net.compress1, net.compress1_bn)
        self.c2 = _fold(net.compress2, net.compress2_bn)
        self.head = _fold(net.policy_head)
        self.cache = {}
        self.path = path

    def logits(self, x289):
        h = np.maximum(x289 @ self.inp[0] + self.inp[1], 0)
        for (W1, b1), (W2, b2) in self.blocks:
            o = np.maximum(h @ W1 + b1, 0)
            h = np.maximum(o @ W2 + b2 + h, 0)
        h = np.maximum(h @ self.c1[0] + self.c1[1], 0)
        h = np.maximum(h @ self.c2[0] + self.c2[1], 0)
        return h @ self.head[0] + self.head[1]

    def cached(self, x):
        return self.cache.get(x[:289].tobytes())

    def prior_batch(self, xs):
        """Priors for a list of state vectors in one batched forward (cached)."""
        A = np.stack([x[:289] for x in xs])
        L = self.logits(A)
        out = []
        if len(self.cache) > CACHE_CAP:
            self.cache.clear()
        for x, lg in zip(xs, L):
            p = _softmax_masked(lg, free_mask(x))
            self.cache[x[:289].tobytes()] = p
            out.append(p)
        return out

    def prior(self, x):
        k = x[:289].tobytes()
        p = self.cache.get(k)
        if p is None:
            if len(self.cache) > CACHE_CAP:
                self.cache.clear()
            p = _softmax_masked(self.logits(x[:289]), free_mask(x))
            self.cache[k] = p
        return p


class GDNP:
    """GenericDraftModel (289 -> 256 -> 128 -> 90), numpy; masked softmax at T=1."""

    def __init__(self, model):
        lins = [m for m in model.net if hasattr(m, "weight")]
        self.L = [_fold(l) for l in lins]
        self.cache = {}

    def probs(self, x):
        k = x[:289].tobytes()
        p = self.cache.get(k)
        if p is None:
            if len(self.cache) > CACHE_CAP:
                self.cache.clear()
            h = x[:289]
            for i, (W, b) in enumerate(self.L):
                h = h @ W + b
                if i < len(self.L) - 1:
                    h = np.maximum(h, 0)
            p = _softmax_masked(h, free_mask(x))
            self.cache[k] = p
        return p


def sample(p, r):
    """v2_sample: first index whose cumulative mass exceeds r; last positive entry on overrun."""
    cs = np.cumsum(p)
    i = int(np.searchsorted(cs, r, side="right"))
    if i >= len(p):
        nz = np.flatnonzero(p > 0)
        return int(nz[-1]) if len(nz) else -1
    return i


# ── leaf values ──

class LeafWP:
    """Symmetrized leaf WP over full drafts, team-0 perspective, memoized.
    mode 'enriched': the F_oof leaf. mode 'ensemble': (mean, sd) over the 20
    members of deferred.py job_ensemble."""

    def __init__(self, mode="enriched"):
        import torch
        from paper1_revision import core, train_wp
        from experiment_synthetic_augmentation import ENRICHED_GROUPS
        from sweep_enriched_wp import FEATURE_GROUPS, compute_group_indices
        from shared import HEROES
        self.torch = torch
        self.HEROES = HEROES
        self.st = core.load_stats("deploy")
        self.mode = mode
        if mode == "enriched":
            m, cols = train_wp.load("enriched")
            self.members = [(m, np.asarray(cols))]
            groups = set(ENRICHED_GROUPS)
        else:
            from paper1_revision.deferred import _ensemble_specs
            from sweep_enriched_wp import WinProbEnrichedModel
            mdir = os.environ.get("P1_ENSEMBLE_DIR", os.path.join(core.MODEL_DIR, "ensemble_lcb"))
            self.members = []
            groups = set()
            for sp in _ensemble_specs():
                cols = core.group_cols(sp["groups"])
                m = WinProbEnrichedModel(len(cols), sp["arch"], dropout=0.3)
                m.load_state_dict(torch.load(os.path.join(mdir, sp["name"] + ".pt"),
                                             map_location="cpu", weights_only=True))
                m.eval()
                self.members.append((m, np.asarray(cols)))
                groups |= set(sp["groups"])
            assert len(self.members) == 20
        # extract only the groups the members read; scatter into the 357-wide row
        self.mask = [g in groups for g in FEATURE_GROUPS]
        gi = compute_group_indices()
        pos = list(range(197))
        for g in FEATURE_GROUPS:
            if g in groups:
                s, e = gi[g]
                pos.extend(range(197 + s, 197 + e))
        self.pos = np.array(pos)
        self._np_members()
        self.cache = {}

    def _np_members(self):
        """Members as folded numpy MLPs, grouped by architecture for one batched pass."""
        nets = []
        for m, cols in self.members:
            mods = list(m.net)
            L = []
            i = 0
            while i < len(mods):
                mod = mods[i]
                if isinstance(mod, self.torch.nn.Linear):
                    bn = mods[i + 1] if i + 1 < len(mods) and isinstance(mods[i + 1], self.torch.nn.BatchNorm1d) else None
                    L.append(_fold(mod, bn))
                i += 1
            assert isinstance(mods[-1], self.torch.nn.Sigmoid)   # sigmoid applied in _forward
            nets.append((L, cols))
        self.np_nets = nets
        # first layer over the full 357-wide row (zero rows for unused columns)
        self.W1 = []
        for L, cols in nets:
            W = np.zeros((357, L[0][0].shape[1]), np.float32)
            W[cols] = L[0][0]
            self.W1.append(W)
        self.W1cat = np.concatenate(self.W1, 1)
        self.b1cat = np.concatenate([L[0][1] for L, _ in nets])
        self.split = np.cumsum([w.shape[1] for w in self.W1])[:-1]

    def features(self, t0, t1, game_map, tier):
        from sweep_enriched_wp import extract_features
        out = np.zeros((2, 357), np.float32)
        for k, (a, b) in enumerate(((t0, t1), (t1, t0))):
            d = {"team0_heroes": a, "team1_heroes": b, "game_map": game_map,
                 "skill_tier": tier, "winner": 0}
            bb, e = extract_features(d, self.st, self.mask)
            out[k, self.pos] = np.concatenate([bb, e])
        return out

    def _forward(self, X):
        """X (2, 357) -> per-member probabilities (n_members, 2)."""
        H = np.maximum(X @ self.W1cat + self.b1cat, 0)
        parts = np.split(H, self.split, axis=1)
        P = []
        for (L, _), h in zip(self.np_nets, parts):
            for j, (W, b) in enumerate(L[1:]):
                h = h @ W + b
                if j < len(L) - 2:
                    h = np.maximum(h, 0)
            P.append(h[:, 0])
        return 1.0 / (1.0 + np.exp(-np.array(P, np.float64)))

    def value_batch(self, keys):
        """keys: (t0_idx, t1_idx, map, tier) tuples -> [(mean, sd)], one batched forward."""
        if len(self.cache) > CACHE_CAP:
            self.cache.clear()
        X = np.concatenate([self.features([self.HEROES[i] for i in a], [self.HEROES[i] for i in b], m, t)
                            for a, b, m, t in keys])
        P = self._forward(X)                      # (members, 2 * len(keys))
        out = []
        for j, key in enumerate(keys):
            sym = 0.5 * (P[:, 2 * j] + 1.0 - P[:, 2 * j + 1])
            v = (float(sym.mean()), float(sym.std()))
            self.cache[key] = v
            out.append(v)
        return out

    def value_t0(self, t0_idx, t1_idx, game_map, tier):
        """-> (mean, sd) of the members' symmetrized P(team 0 wins); sd = 0 for one member."""
        key = (t0_idx, t1_idx, game_map, tier)
        v = self.cache.get(key)
        if v is None:
            if len(self.cache) > CACHE_CAP:
                self.cache.clear()
            X = self.features([self.HEROES[i] for i in t0_idx], [self.HEROES[i] for i in t1_idx],
                              game_map, tier)
            P = self._forward(X)
            sym = 0.5 * (P[:, 0] + 1.0 - P[:, 1])
            v = (float(sym.mean()), float(sym.std()))
            self.cache[key] = v
        return v

    def torch_reference(self, t0, t1, game_map, tier):
        """Members' symmetrized P(team0) via the torch modules (for checks)."""
        X = self.features(t0, t1, game_map, tier)
        out = []
        with self.torch.no_grad():
            for m, cols in self.members:
                p = m(self.torch.tensor(X[:, cols])).view(-1).numpy()
                out.append(0.5 * (p[0] + 1 - p[1]))
        return np.array(out)


def make_leaf(mode):
    return LeafWP(mode)


# ── the search ──

class Search:
    def __init__(self, policy, gd, leaf, game_map, tier, k_lcb=None, sims=SIMS,
                 c_puct=C_PUCT, pw_k=PW_K, pw_alpha=PW_ALPHA):
        self.pol, self.gd, self.leaf = policy, gd, leaf
        self.map, self.tier = game_map, tier
        self.k = k_lcb
        self.sims, self.c, self.pk, self.pa = sims, c_puct, pw_k, pw_alpha
        self.stats = {"sims": 0, "leaf_steps": 0, "max_depth": 0}

    def _kind(self, x, step, our):
        if step >= 16:
            return KIND_TERMINAL
        return KIND_DECISION if TEAM_OF[step] == our else KIND_CHANCE

    def _rollout_key(self, x, rng):
        """GD rollout to the terminal draft -> leaf cache key."""
        step = step_of(x)
        while step < 16:
            p = self.gd.probs(x)
            a = sample(p, 1.0 - rng.random())
            if a < 0:
                a = int(np.flatnonzero(free_mask(x))[0])
            x, step = apply(x, a)
        return (tuple(np.flatnonzero(x[0:90] > 0.5).tolist()),
                tuple(np.flatnonzero(x[90:180] > 0.5).tolist()), self.map, self.tier)

    def _combine(self, mu, sd, our):
        p = mu if our == 0 else 1.0 - mu
        if self.k:
            p = p - self.k * sd
        return p

    def _value(self, x, our, rng):
        step = step_of(x)
        while step < 16:
            p = self.gd.probs(x)
            a = sample(p, 1.0 - rng.random())
            if a < 0:
                a = int(np.flatnonzero(free_mask(x))[0])
            x, step = apply(x, a)
        t0 = tuple(np.flatnonzero(x[0:90] > 0.5).tolist())
        t1 = tuple(np.flatnonzero(x[90:180] > 0.5).tolist())
        mu, sd = self.leaf.value_t0(t0, t1, self.map, self.tier)
        p = mu if our == 0 else 1.0 - mu
        if self.k:
            p = p - self.k * sd
        return p

    def run(self, x_root, our, rng, root_temp=0.0):
        """-> (hero index, root visit counts); policy priors computed inline."""
        g = self.gen(x_root, our, rng, root_temp)
        send = None
        while True:
            try:
                kind, obj, x = g.send(send)
            except StopIteration as e:
                return e.value
            send = obj.prior(x) if kind == "prior" else obj.value_batch([x])[0]

    def gen(self, x_root, our, rng, root_temp=0.0):
        """Generator form of the search: yields ("prior", policy, state) when a
        policy prior is not cached and ("leaf", leaf, key) when a leaf value is
        not cached, and expects the value sent back (run_batched batches these
        across concurrent searches); returns (hero index, root visits)."""
        x_root = x_root.copy()
        x_root[OUR_COL] = float(our)
        root_step = step_of(x_root)
        assert TEAM_OF[root_step] == our
        # node arrays (python lists; slot arrays per expanded node)
        X, KIND, VIS, VAL, NK, P, CH, CN, CW, PAR, ACT = [], [], [], [], [], [], [], [], [], [], []

        def new(x, step, parent, action):
            X.append(x)
            KIND.append(self._kind(x, step, our))
            VIS.append(0)
            VAL.append(0.0)
            NK.append(0)
            P.append(None)
            CH.append(None)
            CN.append(None)
            CW.append(None)
            PAR.append(parent)
            ACT.append(action)
            return len(X) - 1

        def expand(i, prior=None):
            x = X[i]
            if KIND[i] == KIND_DECISION:
                fm = free_mask(x)
                p = np.where(fm, prior, 0.0)
                ch = np.where(fm, CHILD_LAZY, CHILD_NONE)
            else:
                p = self.gd.probs(x)
                ch = np.where(p > 0, CHILD_LAZY, CHILD_NONE)
            P[i], CH[i] = p, ch.astype(np.int64)
            CN[i], CW[i] = np.zeros(NH), np.zeros(NH)

        new(x_root, root_step, -1, -1)
        pr = self.pol.cached(x_root)
        if pr is None:
            pr = yield ("prior", self.pol, x_root)
        expand(0, pr)
        for _ in range(self.sims):
            path, i = [0], 0
            while True:
                kd = KIND[i]
                if kd == KIND_TERMINAL:
                    break
                if kd == KIND_DECISION:
                    if P[i] is None:
                        pr = self.pol.cached(X[i])
                        if pr is None:
                            pr = yield ("prior", self.pol, X[i])
                        expand(i, pr)        # leaf: expanded with priors, then rolled out
                        break
                    n = CN[i]
                    q = np.divide(CW[i], n, out=np.zeros(NH), where=n > 0)
                    sc = q + self.c * P[i] * math.sqrt(VIS[i]) / (1.0 + n)
                    sc[CH[i] == CHILD_NONE] = -np.inf
                    a = int(np.argmax(sc))
                else:
                    if P[i] is None:
                        expand(i)
                    r = 1.0 - rng.random()
                    lim = (1 << 30) if self.pk <= 0 else max(1, math.ceil(self.pk * VIS[i] ** self.pa - 1e-3))
                    if NK[i] < lim:
                        a = sample(P[i], r)
                    else:
                        a = sample(np.where(CH[i] >= 0, P[i], 0.0) / P[i][CH[i] >= 0].sum(), r)
                c = int(CH[i][a])
                if c < 0:
                    y, s2 = apply(X[i], a)
                    c = new(y, s2, i, a)
                    CH[i][a] = c
                    NK[i] += 1
                i = c
                path.append(i)
            key = self._rollout_key(X[i], rng)
            ms = self.leaf.cache.get(key)
            if ms is None:
                ms = yield ("leaf", self.leaf, key)
            v = self._combine(ms[0], ms[1], our)
            depth = step_of(X[i]) - root_step
            self.stats["sims"] += 1
            self.stats["leaf_steps"] += depth
            self.stats["max_depth"] = max(self.stats["max_depth"], depth)
            for j in path:
                VIS[j] += 1
                VAL[j] += v
                pj = PAR[j]
                if pj >= 0:
                    CN[pj][ACT[j]] += 1
                    CW[pj][ACT[j]] += v
        visits = CN[0].copy()
        if root_temp < 0.05:
            a = int(np.argmax(visits))
        else:
            w = visits ** (1.0 / root_temp)
            a = sample(w / w.sum(), 1.0 - rng.random())
        return a, visits


def run_batched(gens):
    """Drive search generators together; policy priors and leaf values requested
    in the same round are computed in one batched forward per network (exact same
    values as the inline path; amortizes the weight reads). -> list of results."""
    n = len(gens)
    res = [None] * n
    send = [None] * n
    active = list(range(n))
    while active:
        reqs = {}
        nxt = []
        for j in active:
            try:
                kind, obj, x = gens[j].send(send[j])
            except StopIteration as e:
                res[j] = e.value
                continue
            reqs.setdefault((kind, id(obj)), (kind, obj, []))[2].append((j, x))
            nxt.append(j)
        for kind, obj, items in reqs.values():
            xs = [x for _, x in items]
            if kind == "prior":
                ps = obj.prior_batch(xs)
            else:
                uniq = list(dict.fromkeys(xs))
                vals = dict(zip(uniq, obj.value_batch(uniq)))
                ps = [vals[x] for x in xs]
            for (j, _), p in zip(items, ps):
                send[j] = p
        active = nxt
    return res


# ── actors ──

_SHARED = {}


def _policy(path):
    if path not in _SHARED:
        _SHARED[path] = PolicyNP(path)
    return _SHARED[path]


def _leaf(mode):
    key = "leaf:" + mode
    if key not in _SHARED:
        _SHARED[key] = make_leaf(mode)
    return _SHARED[key]


def gd_np(i):
    """GDNP for p1site generic_draft_<i> (the search-internal opponent/rollout model)."""
    key = f"gd:{i}"
    if key not in _SHARED:
        import torch
        from train_generic_draft import GenericDraftModel
        from overfit2026 import search as osearch
        g = GenericDraftModel()
        g.load_state_dict(torch.load(os.path.join(osearch.gd_dir(), f"generic_draft_{i}.pt"),
                                     weights_only=True, map_location="cpu"))
        g.eval()
        _SHARED[key] = GDNP(g)
    return _SHARED[key]


def f_oof_paths(seeds=(0, 1, 2, 3, 4)):
    from paper1_revision import core
    return [os.path.join(core.MCTS_RUNS, f"F_oof_s{s}", "draft_policy.pt") for s in seeds]


def parse_value(spec):
    """'enriched' -> ('enriched', None); 'lcb:1' -> ('ensemble', 1.0)."""
    if spec == "enriched":
        return "enriched", None
    if spec.startswith("lcb:"):
        return "ensemble", float(spec[4:])
    raise ValueError(spec)


def build_search_actor(value="enriched", seeds=(0, 1, 2, 3, 4), draft_ctx=None, seed_base=0,
                       gd_index=0, sims=SIMS, root_temp=0.0):
    """Tournament-style actor act(state, team, step_type, game_map, tier) -> hero_idx.
    Draft i (draft_ctx['draft']) uses checkpoint seeds[i % len(seeds)], as the
    tournament pools F_oof. The search RNG is seeded by (seed_base, draft, step).
    act.batch(items) searches several drafts' positions at once (lockstep)."""
    pols = [_policy(p) for p in f_oof_paths(seeds)]
    mode, k = parse_value(value)
    leaf = _leaf(mode)
    gd = gd_np(gd_index)
    ctx = draft_ctx if draft_ctx is not None else {"draft": 0}

    def batch_act(items):
        """items: list of (draft index, state, team, game_map, tier) -> hero indices."""
        gens, xs = [], []
        for i, state, team, game_map, tier in items:
            x = state_vec(state)
            rng = np.random.default_rng([seed_base, i, state.step, 7])
            s = Search(pols[i % len(pols)], gd, leaf, game_map, tier, k_lcb=k, sims=sims)
            gens.append(s.gen(x, team, rng, root_temp=root_temp))
            xs.append(x)
        out = []
        for x, (a, _) in zip(xs, run_batched(gens)):
            assert x[a] + x[90 + a] + x[180 + a] < 0.5
            out.append(a)
        return out

    def act(state, team, step_type, game_map, tier):
        return batch_act([(ctx["draft"], state, team, game_map, tier)])[0]
    act.batch = batch_act
    return act


def as_rich_strategy(act):
    """Rich-eval signature (state, team, step_type, game_map, tier, gd_models, device)."""
    def strategy(state, team, step_type, game_map, tier, gd_models, device):
        return act(state, team, step_type, game_map, tier)
    return strategy
