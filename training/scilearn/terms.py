"""Term library and DSL compiler (FRAMING 6.1).

A model is a list of term specs (dicts, the `term:` block of a card). Each term compiles to
a sparse block of game-level columns that are antisymmetric by construction:

    d(A, B) = phi(A | ctx) - phi(B | ctx)            team terms
    d(A, B) = psi(A, B) - psi(B, A)                  cross terms

and eta = d @ beta with no intercept, so swapping the teams negates eta exactly and no
context effect can favour a side by itself. Only the infrastructure owner adds term types.

Allowed inputs: heroes, bans, pick order, map, tier, build, date, party, as-of player
histories and the dated kit table. Anything else is refused at compile time.
"""
import hashlib
import json

import numpy as np
import scipy.sparse as sp

from training.scilearn import config as C

TERM_TYPES = ("lookup", "state", "multiset", "indicator", "attr_sum", "bilinear", "product",
              "pair_family", "cross_family", "decay", "rule", "hash_null")
HARNESS_ONLY = {"hash_null"}   # canary terms; the broker refuses them in session cards
ALLOWED_FIELDS = {"hero", "map", "map_class", "tier", "build", "role_count", "kit_sum",
                  "hero_in_team", "party_max", "pick_position", "date"}
FORBIDDEN_FIELDS = {"game_length", "avg_mmr", "player_mmr", "hero_mmr", "role_mmr", "scoreboard",
                    "talents", "winner", "y0", "outcome", "duration", "raw_extras", "hero_level",
                    "replay", "timeline", "label"}
LANES2 = {"Battlefield of Eternity", "Braxis Holdout", "Hanamura Temple", "Haunted Mines"}
MAP_CLASSES = ["lanes2", "lanes3"]
INTERACTION_TYPES = {"bilinear", "product", "rule", "cross_family", "pair_family"}


class TermError(ValueError):
    pass


def term_hash(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- context

class Ctx:
    """Draft-time inputs for a set of games. Holds no labels."""

    def __init__(self, games, meta, kit=None, players=None, max_fit_build=None):
        self.heroes = games["heroes"].astype(np.int64)
        self.map = games["map"].astype(np.int64)
        self.tier = games["tier"].astype(np.int64)
        self.build = games["build"].astype(np.int64)
        self.game_ts = games["game_ts"].astype(np.int64)
        self.first_seen_ts = games["first_seen_ts"].astype(np.int64)
        self.replay_id = games["replay_id"].astype(np.int64)
        self.n = len(self.replay_id)
        self.meta = meta
        self.H = len(meta["heroes"])
        self.M = len(meta["maps"])
        self.B = len(meta["builds"])
        self.hero_roles = np.asarray(meta["hero_roles"], np.int64)
        self.map_class = np.array([1 - int(m in LANES2) for m in meta["maps"]], np.int64)
        self.kit = kit                      # (H, K) float or None
        self.axes = list(meta.get("kit_axes", [])) if kit is not None else []
        self.players = players              # dict with player_key (n,10), party (n,10) or None
        self.max_fit_build = max_fit_build if max_fit_build is not None else self.B - 1
        self._cache = {}
        self._root, self._idx = None, None   # history features are computed on the root

    def team(self, side):
        return self.heroes[:, :5] if side == 0 else self.heroes[:, 5:]

    def role_counts(self, side):
        k = ("rc", side)
        if k not in self._cache:
            r = self.hero_roles[self.team(side)]
            self._cache[k] = np.stack([(r == j).sum(1) for j in range(6)], 1)
        return self._cache[k]

    def kit_sums(self, side, perm_key=None):
        if self.kit is None:
            raise TermError("term needs the kit table, which is not loaded")
        k = ("ks", side, perm_key)
        if k not in self._cache:
            kit = self.kit if perm_key is None else permuted_kit(self.kit, self.hero_roles, perm_key)
            self._cache[k] = kit[self.team(side)].sum(1)
        return self._cache[k]

    def axis(self, name):
        if name not in self.axes:
            raise TermError(f"unknown kit axis {name!r}; axes: {self.axes}")
        return self.axes.index(name)

    def subset(self, idx):
        g = {"heroes": self.heroes[idx], "map": self.map[idx], "tier": self.tier[idx],
             "build": self.build[idx], "game_ts": self.game_ts[idx],
             "first_seen_ts": self.first_seen_ts[idx], "replay_id": self.replay_id[idx]}
        pl = None
        if self.players is not None:
            pl = {k: v[idx] for k, v in self.players.items()}
        sub = Ctx(g, self.meta, self.kit, pl, self.max_fit_build)
        idx = np.arange(self.n)[idx]
        sub._root = self._root or self
        sub._idx = idx if self._idx is None else self._idx[idx]
        return sub

    def history_feature(self, key, fn):
        """Compute fn(root_ctx) once on the root context (full history), then slice."""
        root = self._root or self
        if key not in root._cache:
            root._cache[key] = fn(root)
        v = root._cache[key]
        return v if self._idx is None else v[self._idx]


def permuted_kit(kit, hero_roles, key):
    """Kit table with every axis permuted among heroes of the same role (null canary)."""
    import hashlib as _h
    rng = np.random.default_rng(int(_h.sha256(str(key).encode()).hexdigest()[:12], 16))
    out = kit.copy()
    for r in np.unique(hero_roles):
        idx = np.flatnonzero(hero_roles == r)
        for k in range(kit.shape[1]):
            out[idx, k] = kit[rng.permutation(idx), k]
    return out


# --------------------------------------------------------------------------- predicates

def _check_fields(obj, path="term"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_FIELDS or (isinstance(v, str) and v in FORBIDDEN_FIELDS):
                raise TermError(f"{path}: field {k if k in FORBIDDEN_FIELDS else v!r} is not visible at draft time")
            _check_fields(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str) and v in FORBIDDEN_FIELDS:
                raise TermError(f"{path}[{i}]: field {v!r} is not visible at draft time")
            _check_fields(v, f"{path}[{i}]")


OPS = {">=": np.greater_equal, "<=": np.less_equal, "==": np.equal, ">": np.greater, "<": np.less}


def eval_predicate(ctx, side, p):
    f = p["field"]
    if f not in ALLOWED_FIELDS:
        raise TermError(f"predicate field {f!r} not allowed")
    op = OPS[p.get("op", "==")]
    if f == "role_count":
        v = ctx.role_counts(side)[:, ctx.meta["roles"].index(p["key"])]
    elif f == "kit_sum":
        v = ctx.kit_sums(side)[:, ctx.axis(p["key"])]
    elif f == "map":
        return ctx.map == ctx.meta["maps"].index(p["value"])
    elif f == "map_class":
        return ctx.map_class[ctx.map] == MAP_CLASSES.index(p["value"])
    elif f == "tier":
        return ctx.tier == ctx.meta["tiers"].index(p["value"])
    elif f == "hero_in_team":
        h = ctx.meta["heroes"].index(p["value"])
        return (ctx.team(side) == h).any(1)
    else:
        raise TermError(f"predicate field {f!r} not supported in predicates")
    return op(v, p["value"])


# --------------------------------------------------------------------------- blocks

class Block:
    """Columns of one term plus their penalty structure and bounds."""

    def __init__(self, X, names, kind="ridge", lo=None, hi=None, groups=None, rw_pairs=None,
                 lam0=1.0, interaction=False, state=None):
        self.X = sp.csr_matrix(X, dtype=np.float64)
        self.names = names
        self.kind = kind          # ridge | rw | free
        p = self.X.shape[1]
        self.lo = np.full(p, -np.inf) if lo is None else np.asarray(lo, float)
        self.hi = np.full(p, np.inf) if hi is None else np.asarray(hi, float)
        self.groups = groups      # optional per-column group label for EB
        self.rw_pairs = rw_pairs  # list of (i, j) column pairs for random-walk differences
        self.lam0 = lam0
        self.interaction = interaction
        self.state = state or {}  # compile-time state needed to rebuild on new games


def _onehot_diff(idxA, idxB, ncol, signA=None):
    """Rows = games; idxA/idxB are (n, k) column indices (-1 = skip)."""
    n = idxA.shape[0]
    rows, cols, vals = [], [], []
    for idx, s in ((idxA, 1.0), (idxB, -1.0)):
        r = np.repeat(np.arange(n), idx.shape[1])
        c = idx.ravel()
        m = c >= 0
        rows.append(r[m]); cols.append(c[m]); vals.append(np.full(m.sum(), s))
    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, ncol))
    X.sum_duplicates()
    return X


def _sign_bounds(spec, p):
    s = spec.get("sign")
    lo, hi = np.full(p, -np.inf), np.full(p, np.inf)
    if s in ("+", "positive"):
        lo[:] = 0
    elif s in ("-", "negative"):
        hi[:] = 0
    cap = spec.get("cap")
    if cap is not None:
        lo = np.maximum(lo, -cap); hi = np.minimum(hi, cap)
    return lo, hi


def compile_term(spec, ctx, state=None):
    """Return a Block for this term on ctx. `state` (from a fit) freezes data-dependent
    column vocabularies so prediction on new games uses the fitted columns."""
    _check_fields(spec)
    t = spec.get("type")
    if t not in TERM_TYPES:
        raise TermError(f"unknown term type {t!r}")
    state = dict(state or {})
    name = spec.get("name") or f"{t}_{term_hash(spec)}"
    lam = float(spec.get("prior_precision", 1.0))
    A, Bt = ctx.team(0), ctx.team(1)
    H, M = ctx.H, ctx.M

    if t == "lookup":
        keys = spec["keys"]
        if keys[0] != "hero":
            raise TermError("lookup keys must start with hero")
        if keys == ["hero"]:
            X = _onehot_diff(A, Bt, H)
            names = [f"{name}[{h}]" for h in ctx.meta["heroes"]]
        else:
            k = keys[1]
            if k == "map":
                c, nc, labels = ctx.map, M, ctx.meta["maps"]
            elif k == "map_class":
                c, nc, labels = ctx.map_class[ctx.map], 2, MAP_CLASSES
            elif k == "tier":
                c, nc, labels = ctx.tier, 4, ctx.meta["tiers"]
            else:
                raise TermError(f"lookup key {k!r} not supported")
            X = _onehot_diff(A * nc + c[:, None], Bt * nc + c[:, None], H * nc)
            names = [f"{name}[{h}|{l}]" for h in ctx.meta["heroes"] for l in labels]
        return Block(X, names, "ridge", lam0=lam)

    if t == "state":
        # theta(hero, build) as a random walk across builds (GMRF on first differences).
        Bn = ctx.B
        b = np.minimum(ctx.build, ctx.max_fit_build)
        X = _onehot_diff(A * Bn + b[:, None], Bt * Bn + b[:, None], H * Bn)
        pairs = [(h * Bn + j - 1, h * Bn + j) for h in range(H) for j in range(1, Bn)]
        names = [f"{name}[{h}@{bb}]" for h in ctx.meta["heroes"] for bb in ctx.meta["builds"]]
        jump = float(spec.get("jump_precision", 400.0))
        return Block(X, names, "rw", rw_pairs=pairs, lam0=jump,
                     state={"level_precision": float(spec.get("level_precision", 1.0))})

    if t == "multiset":
        rcA, rcB = ctx.role_counts(0), ctx.role_counts(1)
        codeA = (rcA * (6 ** np.arange(6))).sum(1)
        codeB = (rcB * (6 ** np.arange(6))).sum(1)
        vocab = state.get("vocab") or spec.get("_vocab")   # _vocab: harness-frozen cells
        if vocab is None:
            u, cnt = np.unique(np.concatenate([codeA, codeB]), return_counts=True)
            vocab = [int(x) for x, c in zip(u, cnt) if c >= int(spec.get("min_teams", 200))]
        pos = {c: i for i, c in enumerate(vocab)}
        ia = np.array([pos.get(int(c), -1) for c in codeA])[:, None]
        ib = np.array([pos.get(int(c), -1) for c in codeB])[:, None]
        X = _onehot_diff(ia, ib, len(vocab))
        return Block(X, [f"{name}[{c}]" for c in vocab], "ridge", lam0=lam, state={"vocab": vocab})

    if t in ("indicator", "rule"):
        preds = spec["predicates"]
        if t == "rule" and len(preds) > 3:
            raise TermError("rule terms take at most 3 predicates")
        fa = np.ones(ctx.n, bool); fb = np.ones(ctx.n, bool)
        for p in preds:
            fa &= eval_predicate(ctx, 0, p)
            fb &= eval_predicate(ctx, 1, p)
        X = sp.csr_matrix((fa.astype(float) - fb.astype(float))[:, None])
        lo, hi = _sign_bounds(spec, 1)
        if t == "rule":
            lo = np.maximum(lo, -C.RULE_MAX_BETA); hi = np.minimum(hi, C.RULE_MAX_BETA)
        return Block(X, [name], "ridge", lo, hi, lam0=lam, interaction=(t == "rule"))

    if t == "attr_sum":
        k = ctx.axis(spec["axis"])
        pk = spec.get("_perm_key")
        sA, sB = ctx.kit_sums(0, pk)[:, k], ctx.kit_sums(1, pk)[:, k]
        knots = state.get("knots")
        if knots is None:
            knots = list(np.quantile(np.concatenate([sA, sB]), np.linspace(0, 1, int(spec.get("n_knots", 5)))))
        knots = np.asarray(knots, float)
        shape = spec.get("shape", "monotone_inc")

        def seg(x):
            # segment-slope basis: f(x) = sum_j beta_j * clip(x - k_j, 0, k_{j+1} - k_j)
            w = np.diff(knots)
            return np.clip(x[:, None] - knots[:-1][None, :], 0, w[None, :])
        G = seg(sA) - seg(sB)
        p = G.shape[1]
        lo, hi = np.full(p, -np.inf), np.full(p, np.inf)
        if shape == "concave_inc":
            # beta_j = sum_{i>=j} gamma_i, gamma >= 0 : nonnegative, nonincreasing slopes
            G = np.cumsum(G, axis=1)
            lo[:] = 0
        elif shape == "monotone_inc":
            lo[:] = 0
        elif shape == "monotone_dec":
            hi[:] = 0
        elif shape != "free":
            raise TermError(f"unknown shape {shape!r}")
        return Block(sp.csr_matrix(G), [f"{name}[{j}]" for j in range(p)], "ridge", lo, hi, lam0=lam,
                     state={"knots": knots.tolist(), "shape": shape})

    if t == "bilinear":
        ents = spec["entries"]
        if len(ents) > C.MAX_INTERACTIONS:
            raise TermError("bilinear W takes at most 30 nonzero entries")
        SA, SB = ctx.kit_sums(0), ctx.kit_sums(1)
        cols, lo, hi = [], [], []
        for e in ents:
            k, l = ctx.axis(e["a"]), ctx.axis(e["b"])
            cols.append(SA[:, k] * SB[:, l] - SB[:, k] * SA[:, l])
            l0, h0 = _sign_bounds(e, 1)
            lo.append(l0[0]); hi.append(h0[0])
        return Block(sp.csr_matrix(np.stack(cols, 1)), [f"{name}[{e['a']}x{e['b']}]" for e in ents],
                     "ridge", lo, hi, lam0=lam, interaction=True)

    if t == "product":
        k = ctx.axis(spec["axis"])
        d = ctx.kit_sums(0)[:, k] - ctx.kit_sums(1)[:, k]
        by = spec["by"]
        if by == "map_class":
            c, labels = ctx.map_class[ctx.map], MAP_CLASSES
        elif by == "map":
            c, labels = ctx.map, ctx.meta["maps"]
        elif by == "tier":
            c, labels = ctx.tier, ctx.meta["tiers"]
        else:
            raise TermError(f"product by {by!r} not supported")
        X = sp.csr_matrix((d, (np.arange(ctx.n), c)), shape=(ctx.n, len(labels)))
        return Block(X, [f"{name}[{l}]" for l in labels], "ridge", lam0=lam, interaction=True)

    if t in ("pair_family", "cross_family"):
        within = t == "pair_family"
        vocab = state.get("vocab") or spec.get("_vocab")   # _vocab: harness-frozen cells
        if within:
            ia, ib = np.triu_indices(5, 1)
            pa = np.sort(np.stack([A[:, ia], A[:, ib]], -1), -1)
            pb = np.sort(np.stack([Bt[:, ia], Bt[:, ib]], -1), -1)
            codeA = pa[..., 0] * H + pa[..., 1]
            codeB = pb[..., 0] * H + pb[..., 1]
        else:
            # cross pair (a in A, b in B): canonical column (min, max), sign +1 if a < b.
            aa = np.repeat(A, 5, 1); bb = np.tile(Bt, (1, 5))
            lo_, hi_ = np.minimum(aa, bb), np.maximum(aa, bb)
            codeA = lo_ * H + hi_
            sgn = np.where(aa < bb, 1.0, -1.0)
            sgn[aa == bb] = 0
        if vocab is None:
            allc = np.concatenate([codeA.ravel(), codeB.ravel()]) if within else codeA.ravel()
            u, cnt = np.unique(allc, return_counts=True)
            vocab = [int(x) for x, c in zip(u, cnt) if c >= int(spec.get("min_games", 500))]
        pos = np.full(H * H, -1, np.int64)
        pos[np.asarray(vocab, np.int64)] = np.arange(len(vocab))
        n = ctx.n
        if within:
            X = _onehot_diff(pos[codeA], pos[codeB], len(vocab))
        else:
            ci = pos[codeA]
            m = (ci >= 0) & (sgn != 0)
            r = np.repeat(np.arange(n), 25).reshape(n, 25)
            X = sp.csr_matrix((sgn[m], (r[m], ci[m])), shape=(n, len(vocab)))
            X.sum_duplicates()
        names = [f"{name}[{ctx.meta['heroes'][c // H]}+{ctx.meta['heroes'][c % H]}]" for c in vocab]
        return Block(X, names, "ridge", lam0=float(spec.get("prior_precision", 50.0)),
                     interaction=True, state={"vocab": vocab})

    if t == "hash_null":
        # Null canary: a keyed-hash coin per (game, team), antisymmetric, outcome-blind.
        import hashlib as _h
        key = spec["_key"]
        def coin(rid, team):
            return int.from_bytes(_h.sha256(f"{key}:{rid}:{sorted(team.tolist())}".encode()).digest()[:4], "big")
        a = np.array([coin(r, t) for r, t in zip(ctx.replay_id, A)])
        b = np.array([coin(r, t) for r, t in zip(ctx.replay_id, Bt)])
        X = sp.csr_matrix(np.sign(a.astype(np.int64) - b.astype(np.int64)).astype(float)[:, None])
        return Block(X, [name], "ridge", lam0=lam)

    if t == "decay":
        from training.scilearn import features
        hl = float(spec["half_life_days"])
        v = ctx.history_feature(("phx", hl), lambda r: features.player_hero_experience(r, hl))
        X = sp.csr_matrix((v[:, :5].sum(1) - v[:, 5:].sum(1))[:, None])
        lo, hi = _sign_bounds(spec, 1)
        return Block(X, [name], "ridge", lo, hi, lam0=lam)

    raise TermError(f"term type {t!r} not implemented")


def compile_model(specs, ctx, states=None):
    """Compile a list of term specs. Returns (X csr, blocks, offsets)."""
    n_inter = sum(1 for s in specs if s.get("type") in INTERACTION_TYPES)
    if n_inter > C.MAX_INTERACTIONS:
        raise TermError(f"{n_inter} interaction/rule terms exceed the cap of {C.MAX_INTERACTIONS}")
    states = states or [None] * len(specs)
    blocks = [compile_term(s, ctx, st) for s, st in zip(specs, states)]
    p = sum(b.X.shape[1] for b in blocks)
    if p > C.MAX_PARAMS:
        raise TermError(f"{p} parameters exceed the cap of {C.MAX_PARAMS}")
    X = sp.hstack([b.X for b in blocks], format="csr")
    offs = np.cumsum([0] + [b.X.shape[1] for b in blocks])
    return X, blocks, offs
