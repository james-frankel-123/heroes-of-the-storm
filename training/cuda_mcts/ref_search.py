"""
CPU reference of the v2 search (search_v2.cuh, SEARCH_CHANCE) and the tree
invariant checker shared by the GPU trace test.

RefSearch runs the same algorithm as the kernel on any small game:
  decision nodes  PUCT, Q = 0 / N = 0 for untried actions (the kernel's FPU)
  chance nodes    outcome sampled from the chance distribution, children
                  created on demand, progressive widening
                  limit(N) = max(1, ceil(pw_k * N**pw_alpha - 1e-3)); at the limit the
                  outcome is drawn from the distribution renormalized over
                  the existing children (same law as draw-full-then-resample)
  backup          every node on the path: N += 1, W += v (one perspective)
It is used to unit-test the chance-node statistics and the backup (exact
expectimax on toy games), and check_tree() verifies a kernel tree dump.
"""
import math
import numpy as np

KIND_DECISION, KIND_CHANCE, KIND_TERMINAL = 0, 1, 2
CHILD_NONE, CHILD_LAZY = -2, -1


def sample(p, r):
    """Mirror of v2_sample: first index whose cumulative mass exceeds r,
    zero entries skipped, last positive entry on rounding overrun."""
    cum, last = 0.0, -1
    for i, x in enumerate(p):
        if x <= 0.0:
            continue
        last = i
        cum += x
        if cum > r:
            return i
    return last


def sample_existing(p, child, r):
    """Mirror of v2_sample_existing: p restricted to child >= 0, renormalized."""
    tot = sum(x for x, c in zip(p, child) if c >= 0)
    x0, cum, last = r * tot, 0.0, -1
    for i, (x, c) in enumerate(zip(p, child)):
        if c < 0:
            continue
        last = i
        cum += x
        if cum > x0:
            return i
    return last


def pw_limit(n, k, alpha):
    if k <= 0:
        return 1 << 30
    return max(1, math.ceil(k * n ** alpha - 1e-3))   # same epsilon as the kernel


class Node:
    __slots__ = ("parent", "action", "visits", "value_sum", "p", "child", "n_kids", "kind", "state")

    def __init__(self, parent, action, kind, state):
        self.parent, self.action, self.kind, self.state = parent, action, kind, state
        self.visits, self.value_sum, self.n_kids = 0, 0.0, 0
        self.p, self.child = None, None

    @property
    def q(self):
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


class RefSearch:
    """game interface:
        kind(s) -> KIND_*; n_actions; legal(s) -> bool array;
        prior(s) -> probs (decision); chance(s) -> probs (chance);
        apply(s, a) -> s'; value(s, rng) -> leaf value in [0, 1]
    """

    def __init__(self, game, c_puct=2.0, pw_k=1.0, pw_alpha=0.5, seed=0):
        self.g, self.c, self.k, self.alpha = game, c_puct, pw_k, pw_alpha
        self.rng = np.random.default_rng(seed)
        self.nodes = []

    def _new(self, parent, action, state):
        n = Node(parent, action, self.g.kind(state), state)
        self.nodes.append(n)
        return len(self.nodes) - 1

    def _expand(self, i):
        n = self.nodes[i]
        legal = self.g.legal(n.state)
        if n.kind == KIND_DECISION:
            pr = self.g.prior(n.state)
            n.p = [float(pr[a]) if legal[a] else 0.0 for a in range(self.g.n_actions)]
            n.child = [CHILD_LAZY if legal[a] else CHILD_NONE for a in range(self.g.n_actions)]
        else:
            pr = self.g.chance(n.state)
            n.p = [float(x) for x in pr]
            n.child = [CHILD_LAZY if x > 0 else CHILD_NONE for x in pr]

    def search(self, root_state, sims):
        self.nodes = []
        self._new(-1, -1, root_state)
        self._expand(0)
        for _ in range(sims):
            path, i = [0], 0
            while True:
                n = self.nodes[i]
                if n.kind == KIND_TERMINAL:
                    break
                if n.kind == KIND_DECISION:
                    if n.p is None:
                        self._expand(i)
                        break
                    sq = math.sqrt(n.visits)
                    best, ba = -1e30, -1
                    for a, c in enumerate(n.child):
                        if c == CHILD_NONE:
                            continue
                        nn, q = (0, 0.0) if c < 0 else (self.nodes[c].visits, self.nodes[c].q)
                        sc = q + self.c * n.p[a] * sq / (1.0 + nn)
                        if sc > best:
                            best, ba = sc, a
                    a = ba
                else:
                    if n.p is None:
                        self._expand(i)
                    r = 1.0 - self.rng.random()   # (0, 1], like curand_uniform
                    lim = pw_limit(n.visits, self.k, self.alpha)
                    a = sample(n.p, r) if n.n_kids < lim else sample_existing(n.p, n.child, r)
                c = n.child[a]
                if c < 0:
                    c = self._new(i, a, self.g.apply(n.state, a))
                    n.child[a] = c
                    n.n_kids += 1
                i = c
                path.append(i)
            v = self.g.value(self.nodes[i].state, self.rng)
            for j in path:
                self.nodes[j].visits += 1
                self.nodes[j].value_sum += v
        return self.nodes


# ── exact expectimax for toy games ──

def expectimax(game, s, memo=None):
    memo = {} if memo is None else memo
    key = game.key(s)
    if key in memo:
        return memo[key]
    k = game.kind(s)
    if k == KIND_TERMINAL:
        v = game.terminal_value(s)
    elif k == KIND_DECISION:
        legal = game.legal(s)
        v = max(expectimax(game, game.apply(s, a), memo) for a in range(game.n_actions) if legal[a])
    else:
        p = game.chance(s)
        v = sum(p[a] * expectimax(game, game.apply(s, a), memo) for a in range(game.n_actions) if p[a] > 0)
    memo[key] = v
    return v


class ToyDraft:
    """n heroes, alternating turn pattern (0 = us, 1 = opponent), each turn
    takes one free hero. Terminal value = sigmoid(sum of our weights + pair
    synergies - sum of theirs) (deterministic). Opponent: fixed softmax of
    hero weights over free heroes. Leaf value = exact terminal value after a
    uniform-random rollout (the tests use exact expectimax as the target,
    so rollouts are replaced by exact values when exact_leaf=True)."""

    def __init__(self, n=6, turns=(0, 1, 1, 0, 0, 1), seed=0, exact_leaf=True):
        rng = np.random.default_rng(seed)
        self.n_actions, self.turns = n, turns
        self.w = rng.normal(0, 1, n)
        self.syn = rng.normal(0, 0.7, (n, n))
        self.prior_logits = rng.normal(0, 1, n)
        self.exact_leaf = exact_leaf
        self._memo = {}

    def key(self, s):
        return s

    def init(self):
        return (tuple(), tuple())          # (our picks, their picks), in order

    def _step(self, s):
        return len(s[0]) + len(s[1])

    def kind(self, s):
        st = self._step(s)
        if st >= len(self.turns):
            return KIND_TERMINAL
        return KIND_DECISION if self.turns[st] == 0 else KIND_CHANCE

    def legal(self, s):
        m = np.ones(self.n_actions, bool)
        for a in s[0] + s[1]:
            m[a] = False
        return m

    def _softmax(self, x, legal):
        x = np.where(legal, x, -np.inf)
        e = np.exp(x - x[legal].max())
        e[~legal] = 0
        return e / e.sum()

    def prior(self, s):
        return self._softmax(self.prior_logits, self.legal(s))

    def chance(self, s):
        return self._softmax(1.5 * self.w, self.legal(s))

    def apply(self, s, a):
        if self.turns[self._step(s)] == 0:
            return (s[0] + (a,), s[1])
        return (s[0], s[1] + (a,))

    def terminal_value(self, s):
        o, t = s
        x = sum(self.w[a] for a in o) - sum(self.w[a] for a in t)
        x += sum(self.syn[a, b] for i, a in enumerate(o) for b in o[i + 1:])
        return 1.0 / (1.0 + math.exp(-x))

    def value(self, s, rng):
        if self.exact_leaf:
            return expectimax(self, s, self._memo)
        while self.kind(s) != KIND_TERMINAL:
            leg = np.flatnonzero(self.legal(s))
            s = self.apply(s, int(rng.choice(leg)))
        return self.terminal_value(s)


# ── tree invariants (shared with the GPU trace test) ──

def check_tree(t, sims, pw_k, pw_alpha, kind_of, legal_of, apply_of, root_state,
               value_tol=2e-3, chance_probs_of=None, prior_of=None, prob_tol=2e-4,
               root_noise=False):
    """t: dict with arrays parent, action, visits, value_sum, slot, n_kids,
    step, kind, slot_p, slot_child (kernel debug_tree layout).
    kind_of(state) / legal_of(state) / apply_of(state, parent_step, a)
    reconstruct states independently of the kernel. Returns (states, report)."""
    n = len(t["parent"])
    rep = {"nodes": n, "errors": []}
    err = rep["errors"].append
    kids = [[] for _ in range(n)]
    for i in range(1, n):
        kids[t["parent"][i]].append(i)
    states = [None] * n
    states[0] = root_state
    order = [0]
    for i in order:
        order.extend(kids[i])
    n_prob_checked = 0
    for i in order:
        s = states[i]
        if i > 0:
            p = t["parent"][i]
            a = t["action"][i]
            if not legal_of(states[p])[a]:
                err(f"node {i}: action {a} illegal at parent state")
            if t["step"][i] != t["step"][p] + 1:
                err(f"node {i}: step {t['step'][i]} != parent step + 1")
            ps = t["slot"][p]
            if ps < 0 or t["slot_child"][ps][a] != i:
                err(f"node {i}: parent slot does not point back")
            s = states[i] = apply_of(states[p], t["step"][p], a)
        k = kind_of(s)
        if k != t["kind"][i]:
            err(f"node {i}: kind {t['kind'][i]} != reconstructed {k}")
        cv = sum(t["visits"][c] for c in kids[i])
        cw = sum(t["value_sum"][c] for c in kids[i])
        v, w = t["visits"][i], t["value_sum"][i]
        if i == 0:
            if v != sims or cv != sims:
                err(f"root visits {v} / children {cv} != sims {sims}")
        elif k == KIND_DECISION:
            if t["slot"][i] < 0:
                err(f"node {i}: decision node never expanded")
            if v != cv + 1:
                err(f"node {i}: decision visits {v} != 1 + children {cv}")
            if not (-value_tol <= w - cw <= 1 + value_tol):
                err(f"node {i}: decision leaf value {w - cw} outside [0,1]")
        elif k == KIND_CHANCE:
            if v != cv:
                err(f"node {i}: chance visits {v} != children {cv}")
            if abs(w - cw) > value_tol * max(1.0, v ** 0.5):
                err(f"node {i}: chance value_sum {w} != children {cw}")
            if t["n_kids"][i] > pw_limit(max(v - 1, 0), pw_k, pw_alpha):
                err(f"node {i}: {t['n_kids'][i]} children over widening limit at N={v}")
        else:
            if kids[i]:
                err(f"node {i}: terminal node has children")
        if t["n_kids"][i] != len(kids[i]):
            err(f"node {i}: n_kids {t['n_kids'][i]} != {len(kids[i])}")
        sl = t["slot"][i]
        if sl >= 0:
            sp, sc = np.asarray(t["slot_p"][sl]), np.asarray(t["slot_child"][sl])
            leg = legal_of(s)
            if np.any(sp[~leg] != 0):
                err(f"node {i}: positive mass on an illegal action")
            if k == KIND_DECISION and np.any((sc == CHILD_NONE) != ~leg):
                err(f"node {i}: decision child mask != legal mask")
            if not (i == 0 and root_noise) and abs(sp.sum() - 1) > 1e-4:
                err(f"node {i}: slot mass {sp.sum()}")
            ref = None
            if k == KIND_CHANCE and chance_probs_of is not None:
                ref = chance_probs_of(s)
            if k == KIND_DECISION and prior_of is not None and not (i == 0 and root_noise):
                ref = prior_of(s)
            if ref is not None:
                n_prob_checked += 1
                d = float(np.abs(np.asarray(ref) - sp).max())
                rep["max_prob_diff"] = max(rep.get("max_prob_diff", 0.0), d)
                if d > prob_tol:
                    err(f"node {i}: kernel probs differ from reference by {d:.2e}")
    rep["prob_checked"] = n_prob_checked
    return states, rep
