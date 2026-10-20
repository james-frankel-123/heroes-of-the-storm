"""
P3 tournament judges. Nothing here shares data with a drafter:
  population part   paper 1's composition-corrected consensus (mean of gN,
                    gN-naive, RN, QM2026; overfit2026.judges_v2), trained on
                    post-snapshot or Quick Match games, after every drafter's
                    population data (2026-05-22) or outside ranked drafts
  personal part     the skill model refit on the OTHER half of players
                    (full_ms_flip), with evidence from the judge window only
                    (2026-07-01 to 09-11, after every lobby): state_judge
  combined judge    J = sigmoid(logit p_cons + c2 (S_0 - S_1) + c3 (O_0 - O_1)),
                    c fit on judge-window games with the consensus as a fixed
                    offset (no intercept: antisymmetric in the teams), using
                    as-of judge-window skill (judgefit)
Judge-free scores (realized outcomes of later games, no fitted model):
  hero WR / synergy / counter of each side from the judge-window games (the
  drift paper's truth metrics); and the personal record: each player's mean
  residual r = y - WP_pop on the assigned hero in his judge-window games,
  with heroes assigned to players by their earlier play counts (outcome-free).

Stages (3090, run_rcf_full env):
  fit      -> $P3_RESULTS/p3t/judge.json
  score    every finished pair -> $P3_CACHE/p3t/scores/<A>__<B>.npz (also the real drafts: real.npz)
"""
import argparse
import gzip
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("P1_TIERS", "site")
import numpy as np

import p3_hs_core as C
import p3t_data as TD

RES = os.path.join(C.RESULTS, "p3t")
_LS = os.environ.get("P3T_LOBBYSET", "")
SDIR = os.path.join(TD.OUT, ("scores_split" if os.environ.get("P3T_SPLIT") == "1" else "scores") + (f"_{_LS}" if _LS else ""))
os.makedirs(SDIR, exist_ok=True)
PERMS = np.array(list(itertools.permutations(range(5))), np.int64)
MIN_G = 30


def window_games():
    from shared import MAPS
    games = json.load(gzip.open(os.path.join(C.CACHE, "x_post_games.json.gz"), "rt"))
    out = []
    for g in games:
        if g["game_version"] == TD.SEALED:
            continue
        day = C.day_of(time.strftime("%Y-%m-%d", time.gmtime(g["ts"])))
        if not (TD.J0 <= day <= TD.J1):
            continue
        tier = TD.site_tier(g.get("league_tier"), g.get("avg_mmr"))
        if tier == "unknown" or len(g["team0_heroes"]) != 5 or len(g["team1_heroes"]) != 5 or g["game_map"] not in MAPS:
            continue
        out.append((int(g["replay_id"]), tuple(g["team0_heroes"]), tuple(g["team1_heroes"]), g["game_map"], tier,
                    int(g["winner"])))
    return out


def consensus(rows, nproc=8):
    from overfit2026 import judges_v2 as J
    out = {}
    for i in range(0, len(rows), 20000):
        part = rows[i:i + 20000]
        base = J.score_base(part, judges=J.CONSENSUS, nproc=nproc)
        cor = J.correct(base, part)
        for k, v in list(base.items()):
            out.setdefault(k + "_raw", []).append(v)
        for k, v in cor.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def fit_offset_logistic(off, X, y, iters=50):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(off + X @ w)))
        g = X.T @ (y - p)
        Hm = (X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(Hm, g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-(off + X @ w)))
    cov = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X)
    return w, np.sqrt(np.diag(cov))


SPLIT = os.environ.get("P3T_SPLIT") == "1"   # split-history robustness: judge state splitj, judge_split.json
JNAME = "judge_split.json" if SPLIT else "judge.json"
JSTATE = "splitj" if SPLIT else "judge"


def fit():
    import p3_x_common as XC
    t0 = time.time()
    z = np.load(os.path.join(TD.OUT, "judgefit_split.npz" if SPLIT else "judgefit.npz"))
    gw = window_games()
    rid_meta = {g[0]: g for g in gw}
    rid = z["replay_id"]
    ok = np.isin(rid, np.fromiter(rid_meta.keys(), np.int64))
    g_u, inv, cnt = np.unique(z["g"][ok], return_inverse=True, return_counts=True)
    full = cnt[inv] == 10
    idx = np.flatnonzero(ok)[full]
    g_u, inv = np.unique(z["g"][idx], return_inverse=True)
    sign = np.where(z["team"][idx] == 0, 1.0, -1.0)
    dS = np.bincount(inv, weights=sign * z["pred"][idx])
    dO = np.bincount(inv, weights=sign * z["off"][idx])
    first = np.zeros(len(g_u), np.int64)
    first[inv[::-1]] = idx[::-1]
    y = np.where(z["team"][first] == 0, z["y"][first], 1 - z["y"][first])
    rows = []
    for r in rid[first]:
        m = rid_meta[int(r)]
        rows.append((m[1], m[2], m[3], m[4]))
    yg = np.array([1.0 - rid_meta[int(r)][5] for r in rid[first]])     # winner 0 -> team 0 won
    assert np.mean(yg == y) > 0.999, np.mean(yg == y)
    print(f"judge-window games with all ten slots: {len(rows):,} ({time.time() - t0:.0f}s)", flush=True)
    cons = consensus(rows)
    pc = np.clip(cons["consensus"], 1e-6, 1 - 1e-6)
    off = np.log(pc / (1 - pc))
    X = np.column_stack([dS, dO])
    w, se = fit_offset_logistic(off, X, y)
    p1 = 1 / (1 + np.exp(-(off + X @ w)))
    ll0 = XC.logloss(pc, y)
    ll1 = XC.logloss(p1, y)
    d = ll0 - ll1
    out = {"games": int(len(y)), "c2_skill": float(w[0]), "c2_se": float(se[0]), "c3_offrole": float(w[1]),
           "c3_se": float(se[1]), "ll_consensus": float(ll0.mean()), "ll_judge": float(ll1.mean()),
           "lift_x1000": float(1000 * d.mean()), "lift_se_x1000": float(1000 * d.std(ddof=1) / np.sqrt(len(d))),
           "acc_consensus": float(((pc > .5) == (y == 1)).mean()), "acc_judge": float(((p1 > .5) == (y == 1)).mean()),
           "consensus_slope": float(fit_offset_logistic(np.zeros(len(y)), off[:, None], y)[0][0]),
           "params": "full_ms_flip", "window": ["2026-07-01", "2026-09-11"]}
    for k in ("gN", "gN_naive", "RN", "QM2026"):
        pk = np.clip(cons[k], 1e-6, 1 - 1e-6)
        out[f"ll_{k}"] = float(XC.logloss(pk, y).mean())
    out["evidence"] = "even days outside the tournament window (split-history robustness)" if SPLIT else "judge window only"
    json.dump(out, open(os.path.join(RES, JNAME), "w"), indent=1)
    np.savez(os.path.join(TD.OUT, "judge_window_cons.npz"), rid=rid[first], y=y, cons=pc, dS=dS, dO=dO)
    print(json.dumps(out, indent=1), flush=True)


# ------------------------------------------------------------------ truth (judge-free, population)

class Truth:
    """Hero WR, pair synergy and counter win rates per tier on judge-window games."""

    def __init__(self):
        from shared import HEROES
        hi = {h: i for i, h in enumerate(HEROES)}
        T = {"low": 0, "mid": 1, "high": 2}
        hw = np.zeros((3, 90, 2))
        sy = np.zeros((3, 90, 90, 2))
        co = np.zeros((3, 90, 90, 2))
        for rid, t0, t1, gm, tier, win in window_games():
            ti = T[tier]
            for team, won in ((t0, win == 0), (t1, win == 1)):
                ix = [hi[h] for h in team]
                for a in ix:
                    hw[ti, a] += (won, 1)
                for a, b in itertools.combinations(ix, 2):
                    sy[ti, a, b] += (won, 1)
                    sy[ti, b, a] += (won, 1)
            a0 = [hi[h] for h in t0]
            a1 = [hi[h] for h in t1]
            for a in a0:
                for b in a1:
                    co[ti, a, b] += (win == 0, 1)
                    co[ti, b, a] += (win == 1, 1)
        self.wr = np.where(hw[..., 1] >= MIN_G, 100 * hw[..., 0] / np.maximum(hw[..., 1], 1), np.nan)
        self.sy = np.where(sy[..., 1] >= MIN_G, 100 * sy[..., 0] / np.maximum(sy[..., 1], 1), np.nan)
        self.co = np.where(co[..., 1] >= MIN_G, 100 * co[..., 0] / np.maximum(co[..., 1], 1), np.nan)

    def side(self, own, opp, ti):
        w = self.wr[ti]
        hero = np.nanmean(w[own]) if np.isfinite(w[own]).any() else np.nan
        syn = [self.sy[ti, a, b] - (50 + (w[a] - 50) + (w[b] - 50)) for a, b in itertools.combinations(own, 2)]
        cou = [self.co[ti, a, b] - (w[a] + (100 - w[b]) - 50) for a in own for b in opp]
        f = lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
        return hero, f(np.array(syn)), f(np.array(cou))


# ------------------------------------------------------------------ scoring

def best_assign(s5, o5, heroes, c2, c3):
    """s5, o5 (5, 90) of one team, heroes (5,): best c2*S + c3*O assignment."""
    sv = s5[np.arange(5)[:, None], np.array(heroes)[None, :]]          # player i, hero j
    ov = o5[np.arange(5)[:, None], np.array(heroes)[None, :]]
    i = np.arange(5)[None, :]
    sp = sv[i, PERMS].sum(1)
    op = ov[i, PERMS].sum(1)
    b = int(np.argmax(c2 * sp + c3 * op))
    return sp[b], op[b], PERMS[b]


def score_drafts(LB, acts, judge, truth):
    """acts (n, 16) kernel order for the tournament lobbies. Returns per-draft
    arrays from kernel team 0's side."""
    from shared import HEROES, MAPS, SKILL_TIERS, is_degenerate
    n = len(acts)
    teams = [[acts[i, k] for k in range(16) if SKILL_PICK[k] and SKILL_TEAM[k] == t] for i in range(n) for t in (0, 1)]
    rows = [([HEROES[h] for h in teams[2 * i]], [HEROES[h] for h in teams[2 * i + 1]], MAPS[LB.map[i]],
             SKILL_TIERS[LB.tier[i]]) for i in range(n)]
    cons = consensus(rows)
    out = {k: v for k, v in cons.items()}
    js, jn = LB.state[JSTATE]["s"], LB.state[JSTATE]["off"].astype(np.float64)
    rs, rn = LB.state["judge"]["rec_sum"], LB.state["judge"]["rec_n"]
    ncur = LB.state["cur"]["n"]
    c2, c3 = judge["c2_skill"], judge["c3_offrole"]
    dS, dO = np.zeros(n), np.zeros(n)
    rec_mean = np.full((n, 2), np.nan)
    rec_cov = np.zeros((n, 2))
    rec_pool = np.full((n, 2), np.nan)
    degen = np.zeros((n, 2), bool)
    healer = np.zeros((n, 2), bool)
    offc = np.zeros((n, 2))
    tr = np.full((n, 2, 3), np.nan)
    for i in range(n):
        for t in (0, 1):
            hs = teams[2 * i + t]
            sl = slice(5 * t, 5 * t + 5)
            S, O, _ = best_assign(js[i, sl], jn[i, sl], hs, c2, c3)
            sgn = 1 if t == 0 else -1
            dS[i] += sgn * S
            dO[i] += sgn * O
            offc[i, t] = O
            # outcome-free assignment for the judge-free record: most earlier games first
            _, _, perm = best_assign(np.log1p(ncur[i, sl]), np.zeros((5, 90)), hs, 1.0, 0.0)
            vals, cnts = [], []
            for p_i in range(5):
                h = hs[perm[p_i]]
                c = rn[i, 5 * t + p_i, h]
                if c > 0:
                    vals.append(rs[i, 5 * t + p_i, h] / c)
                    cnts.append(c)
            rec_cov[i, t] = len(vals) / 5
            if vals:
                rec_mean[i, t] = float(np.mean(vals))
                rec_pool[i, t] = float(np.sum(np.array(vals) * cnts) / np.sum(cnts))
            names = [HEROES[h] for h in hs]
            degen[i, t] = bool(is_degenerate(names))
            from shared import HERO_ROLE_FINE
            healer[i, t] = any(HERO_ROLE_FINE[x] == "healer" for x in names)
            tr[i, t] = truth.side(hs, teams[2 * i + 1 - t], LB.tier[i])
    pc = np.clip(cons["consensus"], 1e-6, 1 - 1e-6)
    out["judge_personal"] = 1 / (1 + np.exp(-(np.log(pc / (1 - pc)) + c2 * dS + c3 * dO)))
    out.update(dS_judge=dS, dO_judge=dO, rec_mean=rec_mean, rec_pool=rec_pool, rec_cov=rec_cov, degen=degen,
               healer=healer, offrole=offc, truth=tr)
    return out


SKILL_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
SKILL_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]


def own_values(LB, acts, sysname):
    """The drafter's own valuation of the final draft (model-internal):
    its vintage WP and, if aware, V with its player state; team 0's side."""
    import p3t_systems as SY
    from shared import HEROES, MAPS, SKILL_TIERS
    spec = SY.SYSTEMS[sysname]
    v = SY.vintage(spec["vint"])
    n = len(acts)
    teams = [[acts[i, k] for k in range(16) if SKILL_PICK[k] and SKILL_TEAM[k] == t] for i in range(n) for t in (0, 1)]
    rows = [([HEROES[h] for h in teams[2 * i]], [HEROES[h] for h in teams[2 * i + 1]], MAPS[LB.map[i]],
             SKILL_TIERS[LB.tier[i]]) for i in range(n)]
    wp = v.wp_drafts(rows)
    b = v.spec.get("combiner") or [0, 1, 0, 0]
    st = LB.state[spec.get("state", SY.vintage_state(spec["vint"]))]
    dS = np.zeros(n)
    dO = np.zeros(n)
    for i in range(n):
        for t in (0, 1):
            S, O, _ = best_assign(st["s"][i, 5 * t:5 * t + 5], st["off"][i, 5 * t:5 * t + 5].astype(float),
                                  teams[2 * i + t], b[2], b[3])
            dS[i] += (1 if t == 0 else -1) * S
            dO[i] += (1 if t == 0 else -1) * O
    lo = np.log(np.clip(wp, 1e-6, 1 - 1e-6) / np.clip(1 - wp, 1e-6, 1 - 1e-6))
    V = 1 / (1 + np.exp(-(b[0] + b[1] * lo + b[2] * dS + b[3] * dO)))
    return wp, V


def score_all():
    import p3t_tourney as TT
    LB = TT.Lobbies()
    judge = json.load(open(os.path.join(RES, JNAME)))
    truth = Truth()
    files = sorted(f for f in os.listdir(TT.TDIR) if "__" in f and f.endswith(".npz") and ".tmp" not in f)
    todo = [("real", LB.real_acts, None, None)]
    for f in files:
        z = np.load(os.path.join(TT.TDIR, f))
        todo.append((f[:-4], z["acts"], str(z["a"]), str(z["b"])))
    for name, acts, a, b in todo:
        p = os.path.join(SDIR, f"{name}.npz")
        if os.path.exists(p):
            # recompute scores written under another judge fit (e.g. before a judge refit)
            old = np.load(p)
            # and scores older than their drafts (a rerun pair)
            src = os.path.join(TT.TDIR, f"{name}.npz")
            fresh = not os.path.exists(src) or os.path.getmtime(src) <= os.path.getmtime(p)
            if fresh and "judge_c2" in old.files and abs(float(old["judge_c2"]) - judge["c2_skill"]) < 1e-9:
                continue
        t0 = time.time()
        sc = score_drafts(LB, acts, judge, truth)
        if a is not None:
            for side, s_ in (("a", a), ("b", b)):
                wp, V = own_values(LB, acts, s_)
                sc[f"own_wp_{side}"], sc[f"own_V_{side}"] = wp, V
        sc["judge_c2"] = np.array(judge["c2_skill"])
        np.savez(p, **sc)
        print(f"scored {name} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "score"])
    a = ap.parse_args()
    if a.stage == "fit":
        fit()
    else:
        score_all()


if __name__ == "__main__":
    main()
