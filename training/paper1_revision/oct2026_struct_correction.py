"""
Structure correction for the four oct2026 evaluators (expert-study v6 pool
labels), the method of overfit2026/judges_v2.py applied to the namespace's
own evaluators:

    p*(team0 wins) = sigmoid( logit p_J(team0, team1) + beta_J . (s(team0) - s(team1)) )

s(team) = [no_healer, no_frontline, stack] (overfit2026.structure). beta_J is
fitted by logistic regression with J's own symmetrized prediction as a fixed
offset (J unchanged; no intercept, so the correction is antisymmetric in the
teams and symmetrized scores stay symmetrized), on real games the evaluators
never trained on:
  Storm League, game_date 2026-09-01 .. 2026-09-27, builds 2.55.17.97771 and
  2.55.17.98025 only (the pool's meta; nothing from builds released after
  2026-09-27), site-scheme tiers (low/mid/high), excluding every replay that
  appears in the expert pool.
Games are split by a salted replay-id hash: the FIT half estimates beta, the
CHECK half reports calibration (predicted vs observed win rate of degenerate
and normal teams; log-loss; slope) before and after correction.
J in {naive, herostrength, enriched, consensus}; the consensus is the mean
of the three uncorrected evaluators and gets its own beta. (The augmented
evaluator was dropped with synthetic augmentation, Max 2026-10-01.)

  python3 paper1_revision/oct2026_struct_correction.py fit
Hooks for oct2026_pool_judges.py (py:<this file>:<func>):
  naive_sc, herostrength_sc, enriched_sc, consensus_sc
Env: as oct2026_refresh.base_env (RERUN_NS=oct2026, deploy stats, hook).
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(TRAINING_DIR)
sys.path.insert(0, TRAINING_DIR)

import numpy as np

OUT = os.path.join(HERE, "results", "expert_v6", "struct_correction.json")
EVALS = ["naive", "herostrength", "enriched"]   # v6 roster: no synthetic augmentation
BUILDS = ("2.55.17.97771", "2.55.17.98025")
_CACHE = {}
# The pool whose real games are excluded from the fit (v6: the review copy).
POOL_PATH = os.environ.get("RATING_ITEMS_PATH", os.path.join(REPO, "data", "rating-items.json"))


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def base_scores(rows):
    from paper1_revision.oct2026_pool_judges import ns_judge
    out = {}
    for e in EVALS:
        if ("judge", e) not in _CACHE:
            _CACHE[("judge", e)] = ns_judge(e)
        out[e] = np.asarray(_CACHE[("judge", e)](rows), float)
    out["consensus"] = np.mean([out[e] for e in EVALS], 0)
    return out


def struct_diff(rows):
    from overfit2026.structure import struct_matrix
    return struct_matrix([r[0] for r in rows]) - struct_matrix([r[1] for r in rows])


def fit_offset_logistic(z, X, y, iters=50, l2=1e-6):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(z + X @ b)))
        g = X.T @ (y - p) - l2 * b
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-(z + X @ b)))
    H = (X * (p * (1 - p))[:, None]).T @ X
    return b, np.sqrt(np.diag(np.linalg.inv(H + 1e-12 * np.eye(len(b)))))


def load_games():
    import psycopg2
    url = os.environ.get("DATABASE_URL")
    if not url:
        for line in open(os.path.join(REPO, ".env")):
            if line.startswith("DATABASE_URL="):
                url = line.split("=", 1)[1].strip().strip('"')
    pool = json.load(open(POOL_PATH))
    in_pool = {int(it["provenance"]["replayId"]) for it in pool["items"]
               if it["provenance"].get("replayId")}
    conn = psycopg2.connect(url)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute("""SELECT replay_id, game_map, skill_tier, team0_heroes, team1_heroes, winner
                   FROM replay_draft_data
                   WHERE game_date >= '2026-09-01' AND game_date < '2026-09-28'
                     AND game_version = ANY(%s) AND skill_tier IN ('low','mid','high')
                   ORDER BY replay_id""", (list(BUILDS),))
    games = []
    for rid, gm, st, t0, t1, w in cur.fetchall():
        t0 = json.loads(t0) if isinstance(t0, str) else t0
        t1 = json.loads(t1) if isinstance(t1, str) else t1
        if (not t0 or not t1 or len(t0) != 5 or len(t1) != 5 or len(set(t0) | set(t1)) != 10
                or w not in (0, 1) or rid in in_pool):
            continue
        games.append((rid, gm, st, list(t0), list(t1), w))
    conn.close()
    return games, len(in_pool), pool.get("seed")


def calib(p, y, S):
    """Team-level calibration on both sides: degenerate vs normal teams."""
    p = np.asarray(p)
    pt = np.concatenate([p, 1 - p])
    yt = np.concatenate([y, 1 - y])
    St = np.concatenate([S[0], S[1]])
    deg = St.max(1) > 0
    res = {}
    for name, m in (("degenerate", deg), ("normal", ~deg),
                    ("no_healer", St[:, 0] > 0), ("no_frontline", St[:, 1] > 0), ("stack", St[:, 2] > 0)):
        if m.sum():
            res[name] = {"n_teams": int(m.sum()), "pred_wr": float(pt[m].mean()),
                         "obs_wr": float(yt[m].mean()), "gap_pp": float(100 * (yt[m].mean() - pt[m].mean())),
                         "se_pp": float(100 * np.sqrt(yt[m].var() / m.sum()))}
    from paper1_revision.train_wp import fit_slope
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    res["all"] = {"n_games": int(len(y)), "acc": float(np.mean((p > .5) == (y > .5))),
                  "ll": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
                  "slope": fit_slope(pc, y)}
    return res


def cmd_fit():
    from overfit2026.data import splitmix64
    from overfit2026.structure import struct_matrix
    games, n_pool, seed = load_games()
    rows = [(g[3], g[4], g[1], g[2]) for g in games]
    y = np.array([1.0 if g[5] == 0 else 0.0 for g in games])
    fit = np.array([splitmix64(g[0] * 7 + 4242) & 1 for g in games]) == 0
    base = base_scores(rows)
    D = struct_diff(rows)
    S = (struct_matrix([r[0] for r in rows]), struct_matrix([r[1] for r in rows]))
    Sc = (S[0][~fit], S[1][~fit])
    out = {"games": len(games), "fit_games": int(fit.sum()), "check_games": int((~fit).sum()),
           "excluded_pool_replays": n_pool, "pool_seed": seed, "pool_path": os.path.relpath(POOL_PATH, REPO),
           "pool_sha256": __import__("hashlib").sha256(open(POOL_PATH, "rb").read()).hexdigest(), "builds": list(BUILDS),
           "dates": ["2026-09-01", "2026-09-27"], "beta": {}, "beta_se": {}, "check": {}}
    for j, p in base.items():
        b, se = fit_offset_logistic(_logit(p[fit]), D[fit], y[fit])
        out["beta"][j] = b.tolist()
        out["beta_se"][j] = se.tolist()
        pc = 1 / (1 + np.exp(-(_logit(p[~fit]) + D[~fit] @ b)))
        out["check"][j] = {"uncorrected": calib(p[~fit], y[~fit], Sc), "corrected": calib(pc, y[~fit], Sc)}
        u, c = out["check"][j]["uncorrected"], out["check"][j]["corrected"]
        print(f"{j:13s} beta={np.round(b, 3)} (se {np.round(se, 3)}) | degenerate gap "
              f"{u['degenerate']['gap_pp']:+.2f} -> {c['degenerate']['gap_pp']:+.2f}pp | normal "
              f"{u['normal']['gap_pp']:+.2f} -> {c['normal']['gap_pp']:+.2f}pp | ll "
              f"{u['all']['ll']:.4f} -> {c['all']['ll']:.4f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)


def _corrected(name):
    def f(rows):
        if "params" not in _CACHE:
            _CACHE["params"] = json.load(open(OUT))
        b = np.array(_CACHE["params"]["beta"][name])
        key = ("base", id(rows))
        if _CACHE.get("base_rows") is not rows:
            _CACHE["base_rows"] = rows
            _CACHE["base"] = base_scores(rows)
        p = _CACHE["base"][name]
        return 1 / (1 + np.exp(-(_logit(p) + struct_diff(rows) @ b)))
    return f


naive_sc = _corrected("naive")
herostrength_sc = _corrected("herostrength")
enriched_sc = _corrected("enriched")
consensus_sc = _corrected("consensus")

if __name__ == "__main__":
    if sys.argv[1] == "fit":
        cmd_fit()
