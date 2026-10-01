"""
Judge-free gold scorer built from realized outcomes of a set of real games.

RealizedIndex(games): the W13 "net realized win probability" index, extended
with a role-composition term, cross-fitted so no game's own outcome is used
both to build a statistic and to fit the weights:

  fold k in {0,1}: games split by a salted replay-level hash.
    statistics  <- fold k games       (hero WR, pair synergy/counter, role-comp WR)
    weights     <- logistic regression of the result of fold 1-k games on the
                   team differences (d_hwr, d_syn, d_ctr, d_comp) computed with
                   fold-k statistics
  score(own, opp, tier) = mean over folds of sigmoid(w . d(own, opp)),
  intercept dropped (side is balanced in every draft set we score).

Feature definitions match drift2026/w13_counter_dig.Scorer (additive
residuals, the paper's synergy/counter convention). d_comp is the difference
of the two teams' role-composition realized WR (drift2026 comp_key: HP role
multiset), shrunk toward 50% with a 200-game pseudo-count.
"""
import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np

from overfit2026.data import splitmix64

COMP_K = 200.0


def stats_for(games):
    """StatsCache-shaped truth object from slim games (drift2026 thresholds)."""
    from drift2026 import common as dcommon
    from drift2026.build_patch_stats import count_chunk, merge_cell, _new_cell
    merged = {}
    for i in range(0, len(games), 20000):
        chunk = [(0, t, m, a, b, bans, w) for _, t, m, a, b, bans, w
                 in games[i:i + 20000]]
        for (_, tier), cell in count_chunk(chunk).items():
            dst = merged.get(tier)
            if dst is None:
                dst = merged[tier] = _new_cell()
            merge_cell(dst, cell)
    st = dcommon.stats_from_counts(merged, pair_min=10)
    st._counts = merged
    return st


def fit_logistic(X, y, iters=60, l2=1e-6):
    X1 = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ w))
        g = X1.T @ (y - p) - l2 * w
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X1 @ w))
    H = (X1 * (p * (1 - p))[:, None]).T @ X1
    return w, np.sqrt(np.diag(np.linalg.inv(H)))


class TeamFeatures:
    """Team-level realized features against one truth object."""

    def __init__(self, truth):
        from drift2026.build_patch_stats import comp_key
        self.t = truth
        self.comp_key = comp_key
        self._comp = {}
        for tier, cell in truth._counts.items():
            self._comp[tier] = {k: (g, w) for k, (g, w) in cell["comp"].items()}
        if hasattr(truth, "_counts"):
            del truth._counts   # defaultdict cells are not picklable

    def hwr(self, team, tier):
        v = [self.t.hero_wr.get(tier, {}).get(h) for h in team]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else 50.0

    def syn(self, team, tier):
        t = self.t
        out = []
        for j, a in enumerate(team):
            for b in team[j + 1:]:
                r = t.get_synergy(a, b, tier)
                if r is not None:
                    out.append(r - (50 + (t.get_hero_wr(a, tier) - 50)
                                    + (t.get_hero_wr(b, tier) - 50)))
        return float(np.mean(out)) if out else 0.0

    def ctr(self, own, opp, tier):
        t = self.t
        add = []
        for a in own:
            for b in opp:
                r = t.get_counter(a, b, tier)
                if r is None:
                    continue
                add.append(r - (t.get_hero_wr(a, tier)
                                + (100 - t.get_hero_wr(b, tier)) - 50))
        return float(np.mean(add)) if add else 0.0

    def comp(self, team, tier):
        g, w = self._comp.get(tier, {}).get(self.comp_key(team), (0, 0))
        return 100.0 * (w + 0.5 * COMP_K) / (g + COMP_K)

    def diff(self, a, b, tier):
        return np.array([self.hwr(a, tier) - self.hwr(b, tier),
                         self.syn(a, tier) - self.syn(b, tier),
                         self.ctr(a, b, tier) - self.ctr(b, a, tier),
                         self.comp(a, tier) - self.comp(b, tier)])


NAMES = ["hwr", "syn", "ctr", "comp"]


class RealizedIndex:
    def __init__(self, games, salt=99, name=""):
        self.name = name
        self.n_games = len(games)
        folds = [[], []]
        for g in games:
            folds[splitmix64(g[0] * 7919 + salt) & 1].append(g)
        self.folds = []
        self.fit = []
        for k in (0, 1):
            tf = TeamFeatures(stats_for(folds[k]))
            X, y = [], []
            for rid, tier, gmap, t0, t1, bans, w in folds[1 - k]:
                X.append(self._x(tf, t0, t1, tier))
                y.append(1.0 if w == 0 else 0.0)
            X, y = np.array(X), np.array(y)
            coef, se = fit_logistic(X, y)
            # held-out log-loss / acc of the index on the outcome fold
            z = X @ coef[1:] + coef[0]
            p = 1 / (1 + np.exp(-z))
            ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
            acc = float(np.mean((p > 0.5) == (y > 0.5)))
            self.folds.append(tf)
            self.fit.append({"coef": coef.tolist(), "se": se.tolist(),
                             "n": len(y), "logloss": ll, "acc": acc})
        # full-set truth for raw component reporting
        self.full = TeamFeatures(stats_for(games))

    def _x(self, tf, a, b, tier):
        return tf.diff(a, b, tier)

    def score(self, own, opp, tier):
        ps = []
        for tf, f in zip(self.folds, self.fit):
            c = np.array(f["coef"])
            ps.append(1 / (1 + math.exp(-(self._x(tf, own, opp, tier) @ c[1:]))))
        return float(np.mean(ps))

    def components(self, own, opp, tier):
        return dict(zip(NAMES, self.full.diff(own, opp, tier).tolist()))

    def describe(self):
        return {"name": self.name, "n_games": self.n_games, "folds": self.fit,
                "features": NAMES}


class StructRealizedIndex(RealizedIndex):
    """RealizedIndex plus explicit team-structure terms (2026-10-01).

    The role-composition cell term shrinks rare cells (median 2.5-10 games for
    degenerate role multisets) to 50% with a 200-game prior, so the plain
    index barely penalizes structurally broken teams (-5pp held-out
    calibration gap on real degenerate teams; comp_judges.py). This variant
    adds the differences of three indicators (no_healer, no_frontline,
    3+ stacked role; overfit2026.structure) to the cross-fitted logistic, so
    their weights are fitted on real outcomes of the outcome fold, exactly
    like the other four terms. Indicators need no statistics, so nothing else
    changes. What it cannot learn: the penalty of a structure no real team
    shows (e.g. five tanks) beyond the additive indicator model.
    """

    def _x(self, tf, a, b, tier):
        from overfit2026.structure import struct_vec
        return np.concatenate([tf.diff(a, b, tier),
                               np.subtract(struct_vec(a), struct_vec(b))])

    def describe(self):
        d = super().describe()
        d["features"] = NAMES + ["no_healer", "no_frontline", "stack"]
        return d


# ── Named gold sets (cached) ──

GOLD_SETS = {
    # post-snapshot uploads of snapshot-era games: no balance drift
    "BF": lambda d: [g for v in d.load_backfill().values() for g in v],
    # same build as the snapshot's last build, played after the snapshot
    "T97": lambda d: d.load_future()["2.55.16.97039"],
    # 2.55.17 builds (balance patch 2026-07-20; drifted meta)
    "T17": lambda d: [g for b, v in d.load_future().items()
                      if b.startswith("2.55.17.") for g in v],
    "NODRIFT": lambda d: GOLD_SETS["BF"](d) + GOLD_SETS["T97"](d),
    "FUT": lambda d: GOLD_SETS["T97"](d) + GOLD_SETS["T17"](d),
}


def get_index(name, games=None, salt=99):
    """Build (or load cached) RealizedIndex for a named gold set."""
    import pickle
    from overfit2026 import data
    from drift2026 import common as dcommon
    dcommon._bind_statscache_methods()
    path = data.art(HERE, "cache", f"realized_{name}_s{salt}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    if games is None:
        games = GOLD_SETS[name](data)
    ri = RealizedIndex(games, salt=salt, name=name)
    with open(path, "wb") as f:
        pickle.dump(ri, f, protocol=pickle.HIGHEST_PROTOCOL)
    return ri
