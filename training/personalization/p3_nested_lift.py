"""
P3 experiment #2 — nested-model lift (the personalization signal check).

Question: does knowing who is playing improve win prediction beyond the
drift-aware population model, and does residual cleaning beat raw player
win rates?

Windows (all dates by game date; every game has full player labels):
  E  estimation  2024-04-01 .. training cutoff (build <= 2.55.14.95918)
  V1 fit         first half (by date) of the post-cutoff snapshot games
  V2 test        second half; strictly later than everything used to fit
Personal statistics come from E only (static, strictly past). Combiner
weights are fit on V1 and scored on V2.

Per player (blizz_id) and per player x hero, from E:
  raw   WR_ph - 0.5 (games >= 1; the naive estimate)
  resid residual r = y_team - wp_team against the stage-1 drift-aware WP,
        shrunk by empirical Bayes:
          player      s_p  = sum r / (n_p + k_p)
          player-hero s_ph = s_p + sum (r - s_p) / (n_ph + k_ph)
        with k = sigma^2 / tau^2 from method-of-moments variance
        components (sigma^2 = mean wp(1 - wp)).
Team features are sums over the five players, differenced team 0 minus
team 1. Players unseen in E contribute 0.

Models (logistic, V1 fit, V2 test):
  M0 pop          logit(wp0)
  M1 +raw         M0 + d_raw
  M2 +resid_p     M0 + d_resid_player
  M3 +resid_ph    M0 + d_resid_player_hero
  M4 +raw+resid   M0 + d_raw + d_resid_player_hero
  D  diagnostic   M0 + d_mmr (player_mmr is as-of-HP-parse and may contain
                  the game's own result; never a baseline)
Metrics on V2: log loss, Brier, accuracy, AUC, calibration (ECE, slope),
with replay-level bootstrap CIs on differences from M0. Subsets: all V2
games; games where all ten players have >= 100 E games ("dense").

Usage:
  python3 personalization/p3_nested_lift.py fetch
  python3 personalization/p3_nested_lift.py analyze
"""
import os
import sys
import json
import argparse

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
WP = os.path.join(CACHE, "wp_drift.npz")
PLAYERS = os.path.join(CACHE, "players_2024q2.npz")
RESULTS = os.environ.get("P3_RESULTS") or os.path.join(HERE, "results")
E_START = "2024-04-01"
SNAPSHOT_BOUND = 63653039


def fetch():
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor(name="p3")
    cur.itersize = 200000
    cur.execute(f"""
        SELECT p.replay_id, p.blizz_id, p.hero, p.team, p.party, p.player_mmr
        FROM replay_players p JOIN replay_draft_data d USING (replay_id)
        WHERE d.game_date >= '{E_START}' AND d.replay_id <= {SNAPSHOT_BOUND}""")
    heroes = {}
    rid, bid, hid, team, party, mmr = [], [], [], [], [], []
    for r, b, h, t, pa, m in cur:
        rid.append(r)
        bid.append(b)
        hid.append(heroes.setdefault(h, len(heroes)))
        team.append(t)
        party.append(pa or 0)
        mmr.append(np.nan if m is None else m)
        if len(rid) % 2000000 == 0:
            print(f"  {len(rid):,} rows", flush=True)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(PLAYERS, replay_ids=np.array(rid, np.int64),
             blizz_ids=np.array(bid, np.int64), hero=np.array(hid, np.int16),
             team=np.array(team, np.int8), party=np.array(party, np.int64),
             mmr=np.array(mmr, np.float32),
             hero_names=np.array(sorted(heroes, key=heroes.get)))
    print(f"wrote {PLAYERS} ({len(rid):,} rows, {len(heroes)} heroes)")


# ---------------------------------------------------------------- analysis

def load():
    w = np.load(WP)
    p = np.load(PLAYERS)
    g_rid = w["replay_ids"]
    order = np.argsort(g_rid)
    g_rid = g_rid[order]
    gi = np.searchsorted(g_rid, p["replay_ids"])
    ok = (gi < len(g_rid)) & (g_rid[np.minimum(gi, len(g_rid) - 1)] == p["replay_ids"])
    print(f"player rows matched to scored games: {ok.mean():.4f}")
    games = {k: w[k][order] for k in ("date_days", "build_idx", "y", "wp0", "in_sample")}
    games["replay_ids"] = g_rid
    rows = {k: p[k][ok] for k in ("blizz_ids", "hero", "team", "party", "mmr")}
    rows["g"] = gi[ok]
    # player identity is (region, blizz_id): region from the aligned hero_mmr rows
    from p3_keys import player_keys
    m = np.load(os.path.join(CACHE, "hero_mmr_2024q2.npz"))
    ka = np.lexsort((p["blizz_ids"], p["replay_ids"]))
    kb = np.lexsort((m["blizz_ids"], m["replay_ids"]))
    assert np.array_equal(p["replay_ids"][ka], m["replay_ids"][kb])
    assert np.array_equal(p["blizz_ids"][ka], m["blizz_ids"][kb])
    inv = np.empty_like(ka)
    inv[ka] = np.arange(len(ka))
    rows["player_key"] = player_keys(m["region"][kb[inv]][ok], rows["blizz_ids"], label="nested lift")
    return games, rows


def eb_k(sum_r, n, sigma2):
    """Method-of-moments shrinkage constant k = sigma^2 / tau^2 for
    group means of residuals (groups with n >= 20 to keep it stable)."""
    m = n >= 20
    means = sum_r[m] / n[m]
    tau2 = np.var(means) - np.mean(sigma2 / n[m])
    tau2 = max(tau2, 1e-6)
    return float(sigma2 / tau2), float(np.sqrt(tau2))


def personal_features(games, rows, e_mask_game):
    """Per-row personal scores from estimation-window games only."""
    g = rows["g"]
    team = rows["team"].astype(np.int64)
    y_team = np.where(team == 0, games["y"][g], 1 - games["y"][g])
    wp_team = np.where(team == 0, games["wp0"][g], 1 - games["wp0"][g])
    r = y_team - wp_team
    in_e = e_mask_game[g]
    sigma2 = float(np.mean((wp_team * (1 - wp_team))[in_e]))

    pid, pinv = np.unique(rows["player_key"], return_inverse=True)
    nh = int(rows["hero"].max()) + 1
    ph = pinv.astype(np.int64) * nh + rows["hero"]

    n_p = np.bincount(pinv[in_e], minlength=len(pid)).astype(float)
    sr_p = np.bincount(pinv[in_e], weights=r[in_e], minlength=len(pid))
    k_p, tau_p = eb_k(sr_p, n_p, sigma2)
    s_p = sr_p / (n_p + k_p)

    phu, phinv = np.unique(ph, return_inverse=True)
    n_ph = np.bincount(phinv[in_e], minlength=len(phu)).astype(float)
    w_ph = np.bincount(phinv[in_e], weights=y_team[in_e], minlength=len(phu))
    dev = r - s_p[pinv]
    sd_ph = np.bincount(phinv[in_e], weights=dev[in_e], minlength=len(phu))
    k_ph, tau_ph = eb_k(sd_ph, n_ph, sigma2)
    s_ph = s_p[pinv] + (sd_ph / (n_ph + k_ph))[phinv]

    raw = np.where(n_ph[phinv] > 0, w_ph[phinv] / np.maximum(n_ph[phinv], 1) - 0.5, 0.0)
    meta = {"sigma2": sigma2, "k_player": k_p, "tau_player_pp": 100 * tau_p,
            "k_player_hero": k_ph, "tau_player_hero_pp": 100 * tau_ph,
            "players_in_E": int((n_p > 0).sum()),
            "mean_resid_E": float(r[in_e].mean())}
    return {"raw": raw, "resid_p": s_p[pinv], "resid_ph": s_ph,
            "n_p_E": n_p[pinv]}, meta


def team_diff(values, rows, n_games):
    sign = np.where(rows["team"] == 0, 1.0, -1.0)
    return np.bincount(rows["g"], weights=sign * values, minlength=n_games)


def fit_logistic(X, y, iters=100, l2=1e-6):
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
    return w


def predict(w, X):
    return 1 / (1 + np.exp(-(np.column_stack([np.ones(len(X)), X]) @ w)))


def metrics(p, y):
    eps = 1e-7
    ll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    brier = np.mean((p - y) ** 2)
    acc = np.mean((p > 0.5) == (y == 1))
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    npos = y.sum()
    auc = (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    lo = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    slope = fit_logistic(lo[:, None], y)[1]
    return {"logloss": ll, "brier": brier, "acc": acc, "auc": auc,
            "ece": ece, "cal_slope": slope}


def analyze():
    games, rows = load()
    n = len(games["y"])
    days = games["date_days"]
    import datetime
    e0 = (datetime.date.fromisoformat(E_START) - datetime.date(1970, 1, 1)).days
    in_e = games["in_sample"] & (days >= e0)
    post = ~games["in_sample"]
    med = np.median(days[post])
    v1, v2 = post & (days < med), post & (days >= med)
    feats, meta = personal_features(games, rows, in_e)
    print(json.dumps(meta, indent=1))

    D = {k: team_diff(v, rows, n) for k, v in feats.items() if k != "n_p_E"}
    mmr = np.nan_to_num(rows["mmr"], nan=np.nanmean(rows["mmr"])) / 100.0
    D["mmr"] = team_diff(mmr, rows, n)
    # coverage: players per game with a row, and dense games
    cnt = np.bincount(rows["g"], minlength=n)
    dense_cnt = np.bincount(rows["g"], weights=(feats["n_p_E"] >= 100), minlength=n)
    full = cnt == 10
    dense = full & (dense_cnt == 10)
    lo = np.log(games["wp0"] / (1 - games["wp0"]))
    y = games["y"]
    specs = {"M0 pop": [], "M1 +raw": ["raw"], "M2 +resid_p": ["resid_p"],
             "M3 +resid_ph": ["resid_ph"], "M4 +raw+resid_ph": ["raw", "resid_ph"],
             "D  +mmr (diagnostic)": ["mmr"]}
    fit_m, test_m = v1 & full, v2 & full
    out = {"meta": meta, "n": {"E_games": int(in_e.sum()), "V1": int(fit_m.sum()),
                               "V2": int(test_m.sum()), "V2_dense": int((test_m & dense).sum())},
           "models": {}}
    rng = np.random.RandomState(0)
    idx_all = np.flatnonzero(test_m)
    boots = [rng.choice(idx_all, len(idx_all)) for _ in range(200)]
    base_pred = None
    for name, cols in specs.items():
        Xf = np.column_stack([lo] + [D[c] for c in cols])
        w = fit_logistic(Xf[fit_m], y[fit_m])
        p = predict(w, Xf)
        if base_pred is None:
            base_pred = p
        res = {"coef": w.tolist(),
               "all": metrics(p[test_m], y[test_m]),
               "dense": metrics(p[test_m & dense], y[test_m & dense])}
        eps = 1e-7
        ll = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        ll0 = -(y * np.log(base_pred + eps) + (1 - y) * np.log(1 - base_pred + eps))
        diffs = [float((ll0[b] - ll[b]).mean()) for b in boots]
        res["logloss_gain_vs_M0"] = {"est": float((ll0[test_m] - ll[test_m]).mean()),
                                     "ci95": [float(np.percentile(diffs, 2.5)),
                                              float(np.percentile(diffs, 97.5))]}
        out["models"][name] = res
        a, d = res["all"], res["dense"]
        print(f"{name:22s} acc {a['acc']:.4f} ll {a['logloss']:.5f} auc {a['auc']:.4f} "
              f"ece {a['ece']:.4f} slope {a['cal_slope']:.3f} | dense acc {d['acc']:.4f} "
              f"ll {d['logloss']:.5f} | gain {res['logloss_gain_vs_M0']['est']:+.5f} "
              f"{res['logloss_gain_vs_M0']['ci95']}", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "p3_nested_lift.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["n"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["fetch", "analyze"])
    args = ap.parse_args()
    fetch() if args.stage == "fetch" else analyze()


if __name__ == "__main__":
    main()
