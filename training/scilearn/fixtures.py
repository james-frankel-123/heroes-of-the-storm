"""Self-inclusion and leak fixtures (FRAMING 6.3 rule 3), run on every library change.

  python -m training.scilearn.fixtures            run all fixtures, print PASS/FAIL, exit 1 on any miss

Each fixture is checked twice: a legitimate builder must pass, and a planted leak must be
caught. Planted leaks:
  P1 self-inclusive hero win rate (target encoding over all games, the game itself included)
  P2 win rate keyed on game time instead of first-seen time (late uploads leak)
  P3 future-inclusive win rate (every other game, past or future)
  P4 a card term reading the game's own length (future scoreboard)
  P5 a card term reading parse-time MMR
  P6 a card term reading talents / scoreboard
  P7 a side-biased term (team 0 bonus) that breaks swap antisymmetry
  P8 scoring rows that a fit read (fit-score disjointness)
"""
import sys

import numpy as np
import scipy.sparse as sp

from training.scilearn import evaluator as E
from training.scilearn import features as FE
from training.scilearn import terms as TM

TOL = 1e-8


# --------------------------------------------------------------------------- planted builders

def leak_self_inclusive_wr(ctx, y0):
    hero = ctx.heroes
    win = np.concatenate([np.repeat(y0[:, None], 5, 1), 1 - np.repeat(y0[:, None], 5, 1)], 1)
    w = np.bincount(hero.ravel(), weights=win.ravel().astype(float), minlength=ctx.H)
    c = np.bincount(hero.ravel(), minlength=ctx.H)
    return ((w + 100) / (c + 200))[hero]


def leak_game_time_wr(ctx, y0):
    """As-of on game time (not first-seen): a late upload changes earlier features."""
    class C2:
        pass
    c2 = C2()
    c2.__dict__.update(ctx.__dict__)
    c2.first_seen_ts = ctx.game_ts + 1          # pretends every game was seen right away
    return FE.asof_hero_winrate(c2, y0)


def leak_future_wr(ctx, y0):
    """Leave-one-out win rate over every other game, past and future."""
    hero = ctx.heroes
    win = np.concatenate([np.repeat(y0[:, None], 5, 1), 1 - np.repeat(y0[:, None], 5, 1)], 1).astype(float)
    w = np.bincount(hero.ravel(), weights=win.ravel(), minlength=ctx.H)
    c = np.bincount(hero.ravel(), minlength=ctx.H)
    return (w[hero] - win + 100) / (c[hero] - 1 + 200)


LEGIT = {"asof_hero_winrate": FE.asof_hero_winrate,
         "player_hero_experience_90d": lambda ctx, y0: FE.player_hero_experience(ctx, 90.0)}
PLANTED = {"P1_self_inclusive_wr": leak_self_inclusive_wr,
           "P2_game_time_wr": leak_game_time_wr,
           "P3_future_wr": leak_future_wr}


# --------------------------------------------------------------------------- invariance checks

def _ctx_without(ctx, drop):
    keep = np.setdiff1d(np.arange(ctx.n), [drop])
    return ctx.subset(keep), keep


def _fresh(ctx):
    """A root context (no parent), so history features are recomputed on these rows only."""
    g = {"heroes": ctx.heroes, "map": ctx.map, "tier": ctx.tier, "build": ctx.build,
         "game_ts": ctx.game_ts, "first_seen_ts": ctx.first_seen_ts, "replay_id": ctx.replay_id}
    return TM.Ctx(g, ctx.meta, ctx.kit, ctx.players, ctx.max_fit_build)


def delete_game_invariance(fn, ctx, y0, probes):
    """Deleting game g must not change features of games that start before g is first seen."""
    base = fn(ctx, y0)
    for g in probes:
        sub, keep = _ctx_without(ctx, g)
        f2 = fn(_fresh(sub), y0[keep])
        before = ctx.game_ts[keep] <= ctx.first_seen_ts[g]
        if not np.allclose(base[keep][before], f2[before], atol=TOL, rtol=0):
            return False
    return True


def late_upload_invariance(fn, ctx, y0, probes):
    """Uploading g late (first seen far in the future) must equal deleting g for every game
    that starts before the new first-seen time."""
    for g in probes:
        late = _fresh(ctx)
        late.first_seen_ts = ctx.first_seen_ts.copy()
        late.first_seen_ts[g] = ctx.game_ts.max() + 10 * 86400
        f_late = fn(late, y0)
        sub, keep = _ctx_without(ctx, g)
        f_del = fn(_fresh(sub), y0[keep])
        if not np.allclose(f_late[keep], f_del, atol=TOL, rtol=0):
            return False
    return True


def own_label_invariance(fn, ctx, y0, probes):
    """Flipping g's outcome must not change g's own features (or any earlier game's)."""
    base = fn(ctx, y0)
    for g in probes:
        y2 = y0.copy()
        y2[g] = 1 - y2[g]
        f2 = fn(ctx, y2)
        before = ctx.game_ts <= ctx.first_seen_ts[g]
        if not np.allclose(base[before], f2[before], atol=TOL, rtol=0):
            return False
    return True


# --------------------------------------------------------------------------- term checks

def synthetic_ctx(meta, kit=None, n=400, seed=0):
    rng = np.random.default_rng(seed)
    H = len(meta["heroes"])
    heroes = np.stack([rng.choice(H, 10, replace=False) for _ in range(n)])
    g = {"heroes": heroes, "map": rng.integers(0, len(meta["maps"]), n), "tier": rng.integers(0, 4, n),
         "build": rng.integers(0, len(meta["builds"]), n), "game_ts": np.sort(rng.integers(1.6e9, 1.7e9, n)),
         "replay_id": np.arange(n) + 1}
    g["first_seen_ts"] = g["game_ts"] + 86400
    players = {"player_key": rng.integers(0, 60, (n, 10)), "party": np.zeros((n, 10), np.int64)}
    return TM.Ctx(g, meta, kit=kit, players=players)


def swapped(ctx):
    g = {"heroes": np.concatenate([ctx.heroes[:, 5:], ctx.heroes[:, :5]], 1), "map": ctx.map,
         "tier": ctx.tier, "build": ctx.build, "game_ts": ctx.game_ts, "first_seen_ts": ctx.first_seen_ts,
         "replay_id": ctx.replay_id}
    pl = None
    if ctx.players is not None:
        pl = {k: np.concatenate([v[:, 5:], v[:, :5]], 1) for k, v in ctx.players.items()}
    return TM.Ctx(g, ctx.meta, ctx.kit, pl, ctx.max_fit_build)


def antisymmetry_error(X, Xs):
    d = (X + Xs)
    return float(abs(d).max()) if d.nnz else 0.0


def check_term(term, meta, kit=None, axes=None):
    """Compile a card's term on synthetic drafts: allowed inputs, caps, swap antisymmetry."""
    m = dict(meta)
    if kit is not None:
        m["kit_axes"] = axes
    ctx = synthetic_ctx(m, kit)
    try:
        X, blocks, _ = TM.compile_model([term], ctx)
        state = [b.state for b in blocks]
        Xs, _, _ = TM.compile_model([term], swapped(ctx), state)
    except TM.TermError as e:
        return {"ok": False, "reason": str(e)}
    except (KeyError, ValueError, TypeError, IndexError) as e:
        return {"ok": False, "reason": f"term does not compile: {type(e).__name__}: {e}"}
    err = antisymmetry_error(sp.csr_matrix(X), sp.csr_matrix(Xs))
    if err > TOL:
        return {"ok": False, "reason": f"swap antisymmetry error {err:.2e}"}
    return {"ok": True, "p": int(X.shape[1]), "swap_error": err}


def side_biased_design(ctx):
    """P7: a term that rewards team 0 for having a healer (not antisymmetric)."""
    rc = ctx.role_counts(0)
    return sp.csr_matrix((rc[:, 2] > 0).astype(float)[:, None])


# --------------------------------------------------------------------------- runner

def run_all(sample_n=6000, seed=1):
    from training.scilearn import data as D
    results = []
    g, meta, y, pl = D.load(tranches=["Dpre"], labels=True, players=True)
    rng = np.random.default_rng(seed)
    # a contiguous slice keeps player histories dense enough to matter
    start = rng.integers(0, len(y) - sample_n)
    idx = np.arange(start, start + sample_n)
    sub = {k: v[idx] for k, v in g.items()}
    ctx = TM.Ctx(sub, meta, players={k: v[idx] for k, v in pl.items()})
    y0 = y[idx].astype(np.int64)
    order = np.argsort(ctx.game_ts)
    probes = order[[len(order) // 5, len(order) // 3, len(order) // 2]]

    checks = {"delete_game": delete_game_invariance, "late_upload": late_upload_invariance,
              "own_label": own_label_invariance}
    for name, fn in LEGIT.items():
        for cname, chk in checks.items():
            ok = chk(fn, ctx, y0, probes)
            results.append((f"legit {name} / {cname}", ok, "passes" if ok else "FAILS (builder breaks the contract)"))
    for name, fn in PLANTED.items():
        caught = [c for c, chk in checks.items() if not chk(fn, ctx, y0, probes)]
        results.append((f"planted {name}", bool(caught), f"caught by {caught}" if caught else "NOT CAUGHT"))

    kit, axes = D.load_kit(meta)
    for pname, term in [
        ("P4_future_scoreboard_length", {"type": "indicator", "name": "long_game", "predicates": [{"field": "game_length", "op": ">", "value": 1200}]}),
        ("P5_parse_time_mmr", {"type": "product", "axis": "avg_mmr", "by": "tier", "name": "mmr"}),
        ("P5b_mmr_in_predicate", {"type": "rule", "name": "high_mmr", "predicates": [{"field": "kit_sum", "key": "burst", "op": ">=", "value": 5}, {"field": "avg_mmr", "op": ">", "value": 2500}]}),
        ("P6_talents_scoreboard", {"type": "lookup", "keys": ["hero", "talents"], "name": "tal"}),
        ("P6b_scoreboard_field", {"type": "indicator", "name": "sb", "predicates": [{"field": "scoreboard", "key": "deaths", "op": "<", "value": 3}]}),
    ]:
        r = check_term(term, meta, kit, axes)
        results.append((f"planted {pname}", not r["ok"], f"rejected: {r.get('reason')}" if not r["ok"] else "NOT CAUGHT"))

    # swap antisymmetry for every library term type, on synthetic drafts
    lib = [{"type": "lookup", "keys": ["hero"]}, {"type": "lookup", "keys": ["hero", "map"]},
           {"type": "state"}, {"type": "multiset", "min_teams": 1},
           {"type": "indicator", "predicates": [{"field": "role_count", "key": "Healer", "op": "==", "value": 0}]},
           {"type": "pair_family", "min_games": 1}, {"type": "cross_family", "min_games": 1},
           {"type": "decay", "half_life_days": 30},
           {"type": "rule", "predicates": [{"field": "role_count", "key": "Tank", "op": ">=", "value": 2}, {"field": "map_class", "value": "lanes3"}]}]
    if kit is not None:
        lib += [{"type": "attr_sum", "axis": axes[0], "shape": "concave_inc"},
                {"type": "bilinear", "entries": [{"a": axes[0], "b": axes[1]}, {"a": axes[2], "b": axes[2]}]},
                {"type": "product", "axis": axes[1], "by": "map_class"}]
    worst = 0.0
    bad = []
    for t in lib:
        r = check_term(t, meta, kit, axes)
        if not r["ok"]:
            bad.append((t["type"], r["reason"]))
        else:
            worst = max(worst, r["swap_error"])
    results.append(("legit library swap antisymmetry", not bad and worst <= TOL,
                    f"{len(lib)} term types, max error {worst:.1e}" + (f"; failures {bad}" if bad else "")))
    sctx = synthetic_ctx(meta, kit)
    err = antisymmetry_error(side_biased_design(sctx), side_biased_design(swapped(sctx)))
    results.append(("planted P7_side_biased_term", err > TOL, f"swap error {err:.2f} > 1e-8, rejected"))

    try:
        E.check_disjoint(np.array([1, 2, 3]), np.array([3, 4]))
        results.append(("planted P8_fit_score_overlap", False, "NOT CAUGHT"))
    except E.DisjointnessError as e:
        results.append(("planted P8_fit_score_overlap", True, f"refused: {e}"))
    return results


if __name__ == "__main__":
    res = run_all()
    for name, ok, msg in res:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}")
    n_ok = sum(ok for _, ok, _ in res)
    print(f"{n_ok}/{len(res)} passed")
    sys.exit(0 if n_ok == len(res) else 1)
