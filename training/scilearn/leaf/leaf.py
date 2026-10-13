"""Additive glass-box WP as an MCTS leaf (FRAMING 6.1 and 11 item 11).

Reads a leafpack (leafpack.npz + leafpack.json, written by the harness model
exporter) and evaluates complete drafts, vectorized over N drafts:

  T(team | m, t) = sum_h hero_ctx[h, m, t]
                 + sum_{i<j} pair_within[h_i, h_j]
                 + role_table[sum_r count_r * 6**r]
                 + sum_k shape(k, S_k) + sum_k map_axis[m, k] * S_k      (S_k = sum_h kit[h, k])
                 + sum of rule betas whose predicates all hold
  X(A, B)        = sum_{a, b} cross[a, b] + S_A^T W S_B
  eta(A, B)      = T(A) - T(B) + X(A, B) - X(B, A),   P(A wins) = sigmoid(eta)

The leaf value seen by the search is sigmoid(eta - kappa * sd), from the
perspective of the team in slots 0-4. Partial drafts are never evaluated: the
search (search.py) rolls partial states out to a complete draft first, like
the CUDA kernel (search_v2.cuh).

Usage:
  m = LeafModel.load("path/to/leafpack_dir")
  logit, sd, contrib = m.evaluate(drafts, "Cursed Hollow", "mid")
  m.export_check()
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N_ROLES = 6
ROLE_CODES = N_ROLES ** N_ROLES
ROLE_POW = N_ROLES ** np.arange(N_ROLES)
DEFAULT_ROLES = ["Tank", "Bruiser", "Healer", "Support", "Ranged Assassin", "Melee Assassin"]
DEFAULT_TIERS = ["low", "mid", "high", "unknown"]
OPS = {">=": np.greater_equal, "<=": np.less_equal, "==": np.equal, "!=": np.not_equal}

# Pairs of slots inside one team (10 pairs) and across teams (25 pairs).
_IW, _JW = np.triu_indices(5, 1)
_AX, _BX = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
_AX, _BX = _AX.ravel(), _BX.ravel()

REQUIRED = ("hero_ctx", "pair_within", "role_table", "kit", "shape_knots", "shape_vals",
            "map_axis", "cross", "W")
VARIANCES = ("hero_ctx_var", "pair_within_var", "role_table_var")
OPTIONAL_VARIANCES = ("cross_var",)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


def role_code(counts) -> np.ndarray:
    """counts[..., 6] role counts in json `roles` order -> role_table index."""
    return np.asarray(counts, dtype=np.int64) @ ROLE_POW


def interp_shape(knots: np.ndarray, vals: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Piecewise-linear through (knots, vals), flat outside the knot range."""
    return np.interp(x, knots, vals)


class LeafModel:
    def __init__(self, arrays: dict, meta: dict, source: str = "<memory>"):
        self.meta, self.source = meta, source
        self.heroes = list(meta["heroes"])
        self.maps = list(meta["maps"])
        self.tiers = list(meta.get("tiers", DEFAULT_TIERS))
        self.roles = list(meta.get("roles", DEFAULT_ROLES))
        self.axes = list(meta.get("axes", [f"k{i}" for i in range(int(meta.get("K", 12)))]))
        self.kappa = float(meta.get("kappa", 0.0))
        self.version = meta.get("version")
        self.model_id = meta.get("model_id")
        self.term_ids = meta.get("term_ids")
        self.card_hashes = meta.get("card_hashes")
        self.build = meta.get("build")
        self.rules = list(meta.get("rules", []))
        self.hero_idx = {h: i for i, h in enumerate(self.heroes)}
        self.map_idx = {m: i for i, m in enumerate(self.maps)}
        self.tier_idx = {t: i for i, t in enumerate(self.tiers)}
        self.role_idx = {r: i for i, r in enumerate(self.roles)}
        self.axis_idx = {a: i for i, a in enumerate(self.axes)}

        a = {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items() if k != "hero_role"}
        for k in REQUIRED:
            if k not in a:
                raise ValueError(f"leafpack missing array {k!r}")
        self.H, self.M, self.T = len(self.heroes), len(self.maps), len(self.tiers)
        self.K = len(self.axes)
        H, M, T, K = self.H, self.M, self.T, self.K
        want = {"hero_ctx": (H, M, T), "pair_within": (H, H), "role_table": (ROLE_CODES,),
                "kit": (H, K), "map_axis": (M, K), "cross": (H, H), "W": (K, K)}
        for k, shp in want.items():
            if a[k].shape != shp:
                raise ValueError(f"{k} has shape {a[k].shape}, expected {shp}")
        if a["shape_knots"].shape != a["shape_vals"].shape or a["shape_knots"].shape[0] != K:
            raise ValueError("shape_knots / shape_vals must both be [K, n_knots]")
        if np.any(np.diff(a["shape_knots"], axis=1) <= 0):
            raise ValueError("shape_knots must be strictly increasing per axis")
        if not np.allclose(a["pair_within"], a["pair_within"].T, atol=1e-12):
            raise ValueError("pair_within must be symmetric")
        for k in VARIANCES + OPTIONAL_VARIANCES:
            if k in a:
                base = k[:-4]
                if a[k].shape != a[base].shape:
                    raise ValueError(f"{k} shape {a[k].shape} != {base} shape {a[base].shape}")
                if np.any(a[k] < 0):
                    raise ValueError(f"{k} has negative entries")
            elif k in VARIANCES:
                a[k] = np.zeros_like(a[k[:-4]])
        self.a = a

        # Hero -> role index (0..5 in `roles` order). See README "Leafpack additions".
        if "hero_role" in arrays:
            hr = np.asarray(arrays["hero_role"], dtype=np.int64)
        elif "hero_roles" in meta:
            hr = np.array([self.role_idx[r] for r in meta["hero_roles"]], dtype=np.int64)
        else:
            raise ValueError("leafpack needs a hero -> role map: npz `hero_role` [H] int or "
                             "json `hero_roles` [H] role names")
        if hr.shape != (H,) or hr.min() < 0 or hr.max() >= N_ROLES:
            raise ValueError("hero_role must be [H] ints in 0..5")
        self.hero_role = hr
        self.role_onehot = np.eye(N_ROLES, dtype=np.int64)[hr]          # [H, 6]
        self._compile_rules()

    # ── loading ──

    @classmethod
    def load(cls, d) -> "LeafModel":
        d = Path(d)
        meta = json.loads((d / "leafpack.json").read_text())
        with np.load(d / "leafpack.npz") as z:
            arrays = {k: z[k] for k in z.files}
        return cls(arrays, meta, str(d))

    def save(self, d) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        arrs = dict(self.a)
        arrs["hero_role"] = self.hero_role
        np.savez_compressed(d / "leafpack.npz", **arrs)
        (d / "leafpack.json").write_text(json.dumps(self.meta, indent=1, ensure_ascii=False))

    # ── rules ──

    def _compile_rules(self):
        self._rules = []
        for r in self.rules:
            preds = []
            for p in r["predicates"]:
                f, op, v = p["field"], p["op"], p["value"]
                if op not in OPS:
                    raise ValueError(f"rule {r['id']}: unknown op {op!r}")
                if f == "role_count":
                    k = p["key"]
                    preds.append(("role", self.role_idx[k] if isinstance(k, str) else int(k), op, float(v)))
                elif f == "kit_sum":
                    k = p["key"]
                    preds.append(("kit", self.axis_idx[k] if isinstance(k, str) else int(k), op, float(v)))
                elif f == "map":
                    preds.append(("map", None, op, self.map_idx[v] if isinstance(v, str) else int(v)))
                elif f == "tier":   # extension, not in the format spec
                    preds.append(("tier", None, op, self.tier_idx[v] if isinstance(v, str) else int(v)))
                else:
                    raise ValueError(f"rule {r['id']}: unknown field {f!r}")
            self._rules.append((str(r["id"]), float(r["beta"]), preds))

    def _rule_hits(self, counts, S, m, t):
        """counts [N, 6], S [N, K], m/t [N] -> {rule_id: bool [N]}"""
        out = {}
        for rid, _, preds in self._rules:
            ok = np.ones(len(counts), bool)
            for kind, key, op, v in preds:
                x = counts[:, key] if kind == "role" else S[:, key] if kind == "kit" else \
                    m if kind == "map" else t
                ok &= OPS[op](x, v)
            out[rid] = ok
        return out

    # ── index helpers ──

    def _ctx(self, x, table, n, what):
        if isinstance(x, (str, int, np.integer)):
            i = table[x] if isinstance(x, str) else int(x)
            return np.full(n, i, dtype=np.int64)
        arr = np.asarray([table[v] if isinstance(v, str) else int(v) for v in x], dtype=np.int64)
        if arr.shape != (n,):
            raise ValueError(f"{what} must be a scalar or length-{n}")
        return arr

    def hero_ids(self, names) -> list[int]:
        return [self.hero_idx[h] for h in names]

    # ── team value ──

    def _team(self, P, m, t):
        """P [N, 5] hero indices -> per-term team values (dict of [N]), counts, S, codes"""
        a = self.a
        counts = self.role_onehot[P].sum(1)                               # [N, 6]
        codes = counts @ ROLE_POW
        S = a["kit"][P].sum(1)                                            # [N, K]
        terms = {
            "hero_ctx": a["hero_ctx"][P, m[:, None], t[:, None]].sum(1),
            "pair_within": a["pair_within"][P[:, _IW], P[:, _JW]].sum(1),
            "role_table": a["role_table"][codes],
            "shape": sum(interp_shape(a["shape_knots"][k], a["shape_vals"][k], S[:, k])
                         for k in range(self.K)) if self.K else np.zeros(len(P)),
            "map_axis": (a["map_axis"][m] * S).sum(1),
        }
        hits = self._rule_hits(counts, S, m, t)
        for rid, beta, _ in self._rules:
            terms[f"rule:{rid}"] = beta * hits[rid]
        return terms, counts, S, codes

    # ── public API ──

    def evaluate(self, drafts, map, tier, build=None, kappa=None, per_side=False):
        """drafts [N, 10] hero indices (team A slots 0-4, team B 5-9); map/tier are a
        name or index, or a length-N sequence. Returns (eta[N], sd[N], contributions),
        contributions = {term: [N]} summing to eta. Terms: hero_ctx, pair_within,
        role_table, shape, map_axis, rule:<id>, cross_hero, cross_kit.
        `build` is checked against json `build` when both are given."""
        if build is not None and self.build is not None and build != self.build:
            raise ValueError(f"leafpack is for build {self.build!r}, asked for {build!r}")
        D = np.asarray(drafts, dtype=np.int64)
        if D.ndim == 1:
            D = D[None]
        if D.shape[1] != 10:
            raise ValueError("drafts must be [N, 10]")
        if D.min() < 0 or D.max() >= self.H:
            raise ValueError("hero index out of range")
        N = len(D)
        m = self._ctx(map, self.map_idx, N, "map")
        t = self._ctx(tier, self.tier_idx, N, "tier")
        A, B = D[:, :5], D[:, 5:]
        tA, cA, SA, codeA = self._team(A, m, t)
        tB, cB, SB, codeB = self._team(B, m, t)
        a = self.a
        contrib = {k: tA[k] - tB[k] for k in tA}
        cr = a["cross"]
        contrib["cross_hero"] = (cr[A[:, _AX], B[:, _BX]] - cr[B[:, _AX], A[:, _BX]]).sum(1)
        Wm = a["W"]
        contrib["cross_kit"] = np.einsum("nk,kl,nl->n", SA, Wm, SB) - np.einsum("nk,kl,nl->n", SB, Wm, SA)
        eta = np.sum(list(contrib.values()), axis=0) if contrib else np.zeros(N)

        # sd of eta under independent table entries: each touched entry enters with
        # coefficient (uses by A) - (uses by B); hero and pair entries are +-1.
        v = a["hero_ctx_var"][A, m[:, None], t[:, None]].sum(1) + \
            a["hero_ctx_var"][B, m[:, None], t[:, None]].sum(1)
        v = v + a["pair_within_var"][A[:, _IW], A[:, _JW]].sum(1) + \
            a["pair_within_var"][B[:, _IW], B[:, _JW]].sum(1)
        same = codeA == codeB
        v = v + np.where(same, 0.0, a["role_table_var"][codeA] + a["role_table_var"][codeB])
        if "cross_var" in a:
            cv = a["cross_var"]
            v = v + (cv[A[:, _AX], B[:, _BX]] + cv[B[:, _AX], A[:, _BX]]).sum(1)
        sd = np.sqrt(v)
        if per_side:
            return eta, sd, contrib, (tA, tB)
        return eta, sd, contrib

    def leaf_value(self, drafts, map, tier, kappa=None):
        """sigmoid(eta - kappa * sd) for team A (slots 0-4)."""
        k = self.kappa if kappa is None else float(kappa)
        eta, sd, _ = self.evaluate(drafts, map, tier)
        return sigmoid(eta - k * sd)

    def p_win(self, drafts, map, tier):
        return sigmoid(self.evaluate(drafts, map, tier)[0])

    def explain(self, draft, map, tier, top=10):
        """Plain-language breakdown of one draft: list of (term, value) by |value|."""
        eta, sd, c = self.evaluate(np.asarray(draft)[None], map, tier)
        items = sorted(((k, float(v[0])) for k, v in c.items()), key=lambda x: -abs(x[1]))
        return {"eta": float(eta[0]), "sd": float(sd[0]), "p": float(sigmoid(eta[0])),
                "terms": items[:top]}

    def reference_eta(self, draft, m: int, t: int) -> float:
        """Scalar loop implementation straight from the format spec (for checks)."""
        a = self.a

        def team_value(P):
            v = sum(a["hero_ctx"][h, m, t] for h in P)
            v += sum(a["pair_within"][P[i], P[j]] for i in range(5) for j in range(i + 1, 5))
            cnt = [0] * N_ROLES
            for h in P:
                cnt[self.hero_role[h]] += 1
            v += a["role_table"][sum(c * N_ROLES ** r for r, c in enumerate(cnt))]
            S = [sum(a["kit"][h, k] for h in P) for k in range(self.K)]
            for k in range(self.K):
                v += float(np.interp(S[k], a["shape_knots"][k], a["shape_vals"][k]))
                v += a["map_axis"][m, k] * S[k]
            env = {"role": cnt, "kit": S}
            for _, beta, preds in self._rules:
                ok = True
                for kind, key, op, val in preds:
                    x = m if kind == "map" else t if kind == "tier" else env[kind][key]
                    ok = ok and bool(OPS[op](x, val))
                v += beta if ok else 0.0
            return v, S

        def cross(P, Q, SP, SQ):
            return sum(a["cross"][p, q] for p in P for q in Q) + \
                sum(SP[k] * a["W"][k, l] * SQ[l] for k in range(self.K) for l in range(self.K))

        A, B = list(draft[:5]), list(draft[5:])
        tA, SA = team_value(A)
        tB, SB = team_value(B)
        return float(tA - tB + cross(A, B, SA, SB) - cross(B, A, SB, SA))

    def export_check(self, n=2000, seed=0, tol=1e-8, n_ref=50) -> dict:
        """Swap antisymmetry, contribution sum, slot-order invariance, and agreement
        with the scalar reference on random complete drafts. Raises on failure."""
        rng = np.random.default_rng(seed)
        D = np.stack([rng.choice(self.H, 10, replace=False) for _ in range(n)])
        m = rng.integers(0, self.M, n)
        t = rng.integers(0, self.T, n)
        eta, sd, c = self.evaluate(D, m, t)
        eta_s, sd_s, _ = self.evaluate(np.concatenate([D[:, 5:], D[:, :5]], 1), m, t)
        # slot order inside a team must not matter either
        perm = np.concatenate([rng.permutation(5), 5 + rng.permutation(5)])
        eta_p, _, _ = self.evaluate(D[:, perm], m, t)
        ref = np.array([self.reference_eta(D[i], int(m[i]), int(t[i])) for i in range(min(n, n_ref))])
        rep = {
            "n": n,
            "max_swap_err": float(np.abs(eta + eta_s).max()),
            "max_sd_swap_err": float(np.abs(sd - sd_s).max()),
            "max_contrib_err": float(np.abs(np.sum(list(c.values()), 0) - eta).max()),
            "max_perm_err": float(np.abs(eta - eta_p).max()),
            "max_ref_err": float(np.abs(ref - eta[:len(ref)]).max()),
            "terms": sorted(c),
        }
        rep["ok"] = all(rep[k] < tol for k in ("max_swap_err", "max_sd_swap_err",
                                               "max_contrib_err", "max_perm_err", "max_ref_err"))
        if not rep["ok"]:
            raise AssertionError(f"leafpack export check failed: {rep}")
        return rep
