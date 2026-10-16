"""Paper 2's personal layer (P3, "+CF rank 2") on the harness's as-of contract.

P3 (training/personalization, P3_EXTENSIONS section 1): for player p on hero h,
    s = mu_exp(n_p, n_ph) + GP posterior mean of theta_ph,
where n_p and n_ph count the player's earlier games (overall, on h), mu_exp is the lag-1
experience table, and the GP over the 90 heroes uses the "+CF rank 2" kernel. The GP's
sufficient statistics are the player's per-hero sums of r / v and 1 / v, with
r = y_team - wp_team - mu_exp at that earlier game and v = wp_team (1 - wp_team), wp being
the population WP (N). A game is scored as
    eta = b0 + b1 logit(wp) + b2 (sum_team0 s - sum_team1 s).
Kernels and the experience table are paper 2's, fit on window E (builds up to
2.55.14.95918; cache/fix_hs_kernels_l1.npz, the lag-1 refit). Nothing here refits them.

What changes from paper 2 is the as-of rule. Paper 2 used every game dated on an earlier
UTC day, including games uploaded after the scored game (audit P3-04). Here a game h
enters the state and the counts of game g only if first_seen_ts[h] < game_ts[g], the
contract in features.py, and `fixtures` runs fixtures.py's three invariance checks
(delete a game, upload it late, flip its own label) on this builder.

Population WP inputs (per game, team-0 win probability), shipped as npz (replay_id, wp):
  frozen N  wp_drift.npz for snapshot games (paper 2's own WP), frozen N on the training
            matrix for later uploads of S, Dpre and G, fits/neural_n for Dpost
  N SDG     the matched-cutoff retrain on the same games (in sample on S, Dpre, G)
Combiner: b fit on Dpre+G games with frozen N (a fit tranche, out of time for frozen N);
P3 on N SDG reuses those weights (fitting them on its own in-sample WP would favour it).

  python -m training.scilearn.compare.p3_asof fixtures --data <extract.npz> --meta <extract.json> --wp <wp.npz> --kernels <npz>
  python -m training.scilearn.compare.p3_asof run --data ... --meta ... --wp <wp.npz> --kernels <npz> --tag <tag> --out <dir> [--combiner <json>]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from numba import njit
except ImportError:          # the fixtures also run, slowly, without numba
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

NP_EDGES = np.array([0, 1, 5, 10, 20, 50, 100, 300], np.int64)
NPH_EDGES = np.array([0, 1, 2, 3, 5, 10, 20, 50], np.int64)
BEST = "+CF rank 2"
BASES = ["player", "role", "fine", "melee", "hero", "cf2"]
CODES = {"S": 0, "Dpre": 1, "G": 2, "Dpost": 3}


def load_kernel(path, p3_heroes, our_heroes):
    """K (H_ours, H_ours) in our hero order; heroes P3 never saw get no row (mapped -1)."""
    kz = np.load(path)
    K = np.tensordot(np.exp(kz["theta_" + BEST]), np.stack([kz["basis_" + b] for b in BASES]), axes=1)
    pos = {h: i for i, h in enumerate(p3_heroes)}
    hmap = np.array([pos.get(h, -1) for h in our_heroes], np.int64)
    return K, kz["experience_table"].astype(np.float64), hmap


@njit(cache=True)
def _post_one(Si, Pi, K, h):
    H = Si.shape[0]
    m = 0
    for j in range(H):
        if Pi[j] > 0:
            m += 1
    if m == 0:
        return 0.0
    obs = np.empty(m, np.int64)
    c = 0
    for j in range(H):
        if Pi[j] > 0:
            obs[c] = j
            c += 1
    A = np.empty((m, m))
    B = np.empty(m)
    for a in range(m):
        for b in range(m):
            A[a, b] = K[obs[a], obs[b]]
        A[a, a] += 1.0 / Pi[obs[a]]
        B[a] = Si[obs[a]] / Pi[obs[a]]
    x = np.linalg.solve(A, B)
    mean = 0.0
    for a in range(m):
        mean += K[h, obs[a]] * x[a]
    return mean


@njit(cache=True)
def _bin(n_p, n_ph, e1, e2):
    a = np.searchsorted(e1, n_p, side="right") - 1
    b = np.searchsorted(e2, n_ph, side="right") - 1
    return a * len(e2) + b


@njit(cache=True)
def _sweep(pid, hero, q_t, e_t, y, wp, query, n_players, H, K, table, e1, e2):
    """Slots in any order. A slot's query (at its game's start) sees exactly the slots whose
    event time (first seen) is strictly earlier. Returns per slot: mu, gp mean (0 where not
    queried), n_p, n_ph."""
    n = len(pid)
    qo = np.argsort(q_t, kind="mergesort")
    eo = np.argsort(e_t, kind="mergesort")
    S = np.zeros((n_players, H))
    P = np.zeros((n_players, H))
    N = np.zeros((n_players, H), np.int64)
    Nt = np.zeros(n_players, np.int64)
    mu = np.zeros(n)
    gpm = np.zeros(n)
    npl = np.zeros(n, np.int64)
    nph = np.zeros(n, np.int64)
    radj = np.full(n, np.nan)
    ep = 0
    for k in range(n):
        i = qo[k]
        t = q_t[i]
        while ep < n and e_t[eo[ep]] < t:
            j = eo[ep]
            if np.isnan(radj[j]):
                # an event before its own query cannot happen when first_seen > game_ts
                raise ValueError("event before its own query")
            p, h = pid[j], hero[j]
            if not np.isnan(wp[j]):
                v = wp[j] * (1.0 - wp[j])
                S[p, h] += radj[j] / v
                P[p, h] += 1.0 / v
            N[p, h] += 1
            Nt[p] += 1
            ep += 1
        p, h = pid[i], hero[i]
        npl[i], nph[i] = Nt[p], N[p, h]
        mu[i] = table[_bin(Nt[p], N[p, h], e1, e2)]
        radj[i] = (y[i] - wp[i] - mu[i]) if not np.isnan(wp[i]) else 0.0
        if query[i]:
            gpm[i] = _post_one(S[p], P[p], K, h)
    return mu, gpm, npl, nph


def slot_values(heroes, player_key, game_ts, first_seen_ts, y0, wp0, K, table, hmap, query_games=None):
    """Per-slot P3 values s (n, 10) (0 for private players and unmapped heroes), as-of."""
    n = len(y0)
    if np.any(first_seen_ts <= game_ts):
        raise ValueError("first_seen_ts must be after game_ts for every game")
    ph = hmap[heroes]
    ok = (player_key >= 0) & (ph >= 0)
    gi, sl = np.nonzero(ok)
    _, pid = np.unique(player_key[gi, sl], return_inverse=True)
    team0 = sl < 5
    y = np.where(team0, y0[gi], 1 - y0[gi]).astype(np.float64)
    w = np.where(team0, wp0[gi], 1 - wp0[gi]).astype(np.float64)
    q = np.ones(len(gi), np.bool_) if query_games is None else query_games[gi]
    mu, gpm, npl, nph = _sweep(pid.astype(np.int64), ph[gi, sl].astype(np.int64),
                               game_ts[gi].astype(np.int64), first_seen_ts[gi].astype(np.int64),
                               y, w, q, int(pid.max()) + 1 if len(pid) else 1, K.shape[0],
                               np.ascontiguousarray(K), table, NP_EDGES, NPH_EDGES)
    s = np.zeros((n, 10))
    s[gi, sl] = mu + gpm
    cnt = np.zeros((n, 10), np.int64)
    cnt[gi, sl] = nph
    return s, cnt


def team_diff(s):
    return s[:, :5].sum(1) - s[:, 5:].sum(1)


def fit_combiner(lo, D, y):
    from scipy.optimize import minimize
    X = np.column_stack([np.ones_like(lo), lo, D])

    def f(b):
        e = X @ b
        p = 1 / (1 + np.exp(-e))
        return np.logaddexp(0, -np.where(y > 0, e, -e)).sum(), X.T @ (p - y)
    r = minimize(f, np.array([0.0, 1.0, 1.0]), jac=True, method="L-BFGS-B")
    return r.x


def load_inputs(a):
    d = np.load(a.data)
    meta = json.loads(Path(a.meta).read_text())
    games = {k: d[k] for k in ("replay_id", "game_ts", "first_seen_ts", "heroes", "tranche", "build")}
    y0 = d["y0"].astype(np.float64)
    pk = d["player_key"]
    w = np.load(a.wp)
    w_rid, w_wp = w["replay_id"], w["wp"].astype(np.float64)
    o = np.argsort(w_rid)
    w_rid, w_wp = w_rid[o], w_wp[o]
    j = np.clip(np.searchsorted(w_rid, games["replay_id"]), 0, len(w_rid) - 1)
    wp0 = np.where(w_rid[j] == games["replay_id"], w_wp[j], np.nan)
    p3_heroes = [str(h) for h in np.load(a.p3_heroes)["hero_names"]] if a.p3_heroes.endswith(".npz") \
        else json.loads(Path(a.p3_heroes).read_text())
    K, table, hmap = load_kernel(a.kernels, p3_heroes, meta["heroes"])
    return games, meta, y0, pk, wp0, K, table, hmap


def cmd_fixtures(a):
    """fixtures.py invariance checks on this builder, on 6,000 contiguous Dpre games (as in
    fixtures.run_all), plus a planted leak (state built on game time) that must be caught."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from training.scilearn import fixtures as FX
    from training.scilearn import terms as TM
    games, meta, y0, pk, wp0, K, table, hmap = load_inputs(a)
    m = np.flatnonzero(games["tranche"] == CODES["Dpre"])
    rng = np.random.default_rng(1)
    start = rng.integers(0, len(m) - 6000)
    idx = m[start:start + 6000]
    sub = {k: games[k][idx] for k in ("replay_id", "game_ts", "first_seen_ts", "heroes", "build")}
    sub.update(map=np.zeros(6000, np.int64), tier=np.zeros(6000, np.int64))
    wp_of = dict(zip(games["replay_id"][idx].tolist(), wp0[idx].tolist()))
    ctx = TM.Ctx(sub, meta, players={"player_key": pk[idx], "party": np.zeros((6000, 10), np.int64)})
    yy = y0[idx].astype(np.int64)

    def legit(c, y):
        w = np.array([wp_of[int(r)] for r in c.replay_id])
        return slot_values(c.heroes, c.players["player_key"], c.game_ts, c.first_seen_ts,
                           y.astype(float), w, K, table, hmap)[0]

    def leak_game_time(c, y):
        w = np.array([wp_of[int(r)] for r in c.replay_id])
        return slot_values(c.heroes, c.players["player_key"], c.game_ts, c.game_ts + 1,
                           y.astype(float), w, K, table, hmap)[0]

    order = np.argsort(ctx.game_ts)
    probes = order[[len(order) // 5, len(order) // 3, len(order) // 2]]
    checks = {"delete_game": FX.delete_game_invariance, "late_upload": FX.late_upload_invariance,
              "own_label": FX.own_label_invariance}
    res = []
    for cn, chk in checks.items():
        ok = chk(legit, ctx, yy, probes)
        res.append((f"legit p3_slot_values / {cn}", ok))
    caught = [cn for cn, chk in checks.items() if not chk(leak_game_time, ctx, yy, probes)]
    res.append((f"planted p3 state on game time, caught by {caught}", bool(caught)))
    s = legit(ctx, yy)
    res.append((f"builder is not trivially zero (mean |s| {np.abs(s).mean():.4f})", np.abs(s).mean() > 0))
    for name, ok in res:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    n_ok = sum(ok for _, ok in res)
    print(f"{n_ok}/{len(res)} passed")
    sys.exit(0 if n_ok == len(res) else 1)


def cmd_run(a):
    t0 = time.time()
    games, meta, y0, pk, wp0, K, table, hmap = load_inputs(a)
    tr = games["tranche"]
    query = np.isin(tr, [CODES[t] for t in ("Dpre", "G", "Dpost")])
    print(f"games {len(y0):,}; with WP {np.isfinite(wp0).sum():,}; queried {query.sum():,}", flush=True)
    s, cnt = slot_values(games["heroes"], pk, games["game_ts"], games["first_seen_ts"], y0, wp0, K, table,
                         hmap, query_games=query)
    D = team_diff(s)
    print(f"sweep done in {time.time() - t0:.0f}s", flush=True)
    wp = np.clip(wp0, 1e-6, 1 - 1e-6)
    lo = np.log(wp / (1 - wp))
    fitm = np.isin(tr, [CODES["Dpre"], CODES["G"]]) & np.isfinite(lo)
    if a.combiner:
        b = np.array(json.loads(Path(a.combiner).read_text())["b"])
        src = a.combiner
    else:
        b = fit_combiner(lo[fitm], D[fitm], y0[fitm])
        src = "fit on Dpre+G"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"combiner_{a.tag}.json").write_text(json.dumps({"b": b.tolist(), "source": src,
                                                            "n_fit": int(fitm.sum())}))
    print(f"combiner b = {np.round(b, 5).tolist()} ({src})", flush=True)
    eta = b[0] + b[1] * lo + b[2] * D
    for t in ("Dpre", "G", "Dpost"):
        m = tr == CODES[t]
        np.savez(out / f"eta_P3_{a.tag}_{t}.npz", replay_id=games["replay_id"][m], eta=eta[m], D=D[m],
                 lo=lo[m], private_slots=(pk[m] < 0).sum(1), nph=cnt[m])
        e, yy = eta[m], y0[m]
        ll = np.logaddexp(0, -np.where(yy > 0, e, -e)).mean()
        ll0 = np.logaddexp(0, -np.where(yy > 0, lo[m], -lo[m])).mean()
        print(f"{t}: P3 log loss {ll:.5f}, population WP {ll0:.5f}, acc {((e > 0) == (yy > 0)).mean():.4f}",
              flush=True)
    print(f"done in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("fixtures", "run"):
        p = sub.add_parser(nm)
        p.add_argument("--data", required=True)
        p.add_argument("--meta", required=True)
        p.add_argument("--wp", required=True)
        p.add_argument("--kernels", required=True)
        p.add_argument("--p3-heroes", required=True)
        if nm == "run":
            p.add_argument("--tag", required=True)
            p.add_argument("--out", required=True)
            p.add_argument("--combiner")
    a = ap.parse_args()
    {"fixtures": cmd_fixtures, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
