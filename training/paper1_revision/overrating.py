"""
How much does the drafter's win-probability model overrate rare team compositions? Real data
only. Step 1 of the hidden-worlds plan: these numbers set the off-support dial ranges.

Baseline (all stages): a main-effects logistic fit on the corpus the WP model was trained on:
  logit P(team0 wins) = s(team0) - s(team1),  s(T) = sum_{h in T} [w_h + V_{map,h} + U_{tier,h}]
(ridge prior sd 1 on w, 0.25 on V and U; no intercept, so it is antisymmetric like the
symmetrized WP). It is the best additive "hero strength on this map at this tier" model; every
synergy, counter and composition effect is outside it. The interaction claim of a value p is
I = logit p - logit b. A logistic GLM of the real result on I with logit b as offset,
  y ~ Bernoulli(sigmoid(logit b + alpha + beta I)),
gives beta = 1 when the model's interaction claims are right on average, beta < 1 when they
are overrated (beta = 0.6: 40% of the claimed non-additive value is not there). Robustness
baseline: the paper-1 'herostrength' WP net (hero_wr + team_avg_wr features only).

Stages (from training/, p1site env; outputs site/results/overrating/, caches site/cache/overrating/):
  base       main-effects baselines on the current train split and on the s4m / s1y vintage
             train splits
  slope      A. NODRIFT games (as ood_frontier.certify): per own-min3 bin (certify bins, opponent
             min3 >= 25 and all), bias, GLM alpha/beta (SE clustered by game), and the top decile
             of I within each bin; for the agent's leaf (enriched leak-free WP, deploy stats) and
             the gN judge (frontier form: v2 structural + binned support correction). Then the
             own/cross split: Ibar(T) = mean of I(T vs R_k) over K random human reference teams
             (shared by both teams of a game); own = Ibar(team) - Ibar(opp), cross = I - own; one
             game-level GLM with per-bin alpha_k and beta_own,k and a beta_cross split by whether
             the game holds a team below the floor. Finally the implied overvaluation on the teams
             the search picks: constrained_mcts and the floored variants against the eight
             tournament opponents, over = p - sigmoid(fitted correct logit), in win-prob points.
  backtest   B. stale vintages s4m (cutoff 2.55.14.95918) and s1y (2.55.10.94189): triple counts
             and baseline from the vintage's own train split, the vintage's enriched WP with its
             own deploy stats, scored on every later game (snapshot games on later builds,
             post-snapshot backfill on later builds, T97); bias and slope by EARLY min3 bin, split
             by whether the team's rarest triple has >= 25 teams in the later games, with the
             early-common bins as the drift control.
  qm         C. Quick Match games (overfit2026/site/cache/qm_games_site.jsonl, dated >= 2025-06-01,
             same era as the snapshot): how wide QM compositions are against the ranked corpus,
             and the agent leaf's bias/slope there by min3 bin.
  report     site/results/overrating/OVERRATING.md
"""
import os
import sys
import json
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core

OUT = os.path.join(core.RESULTS, "overrating")
CACHE = os.path.join(core.CACHE, "overrating")
NPROC = int(os.environ.get("OVR_NPROC", "24"))
EDGES = [0, 1, 5, 10, 25, 50, 100, 200, 400, 10 ** 12]     # certify bins
JEDGES = [0, 5, 10, 25, 50, 100, 200, 400, 10 ** 12]       # joint GLM (0-1 merged into 0-5)
FLOOR = 25
K_REF = 6
OPP8 = ["mcts", "constrained_mcts", "enriched", "constrained_greedy", "k_oof", "gourdeau", "gourdeau_disc", "gd"]
SEARCH = ["constrained_mcts", "constrained_mcts_sf", "constrained_mcts_sf10", "constrained_mcts_sf50",
          "constrained_mcts_ft", "constrained_mcts_ft_sf", "mcts"]
QM_FROM = "2025-06-01"
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)


def clip(p):
    return np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)


def logit(p):
    p = clip(p)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def hidx(teams):
    from shared import HERO_TO_IDX
    return np.array([sorted(HERO_TO_IDX[h] for h in t) for t in teams], np.int64)


def build_tuple(b):
    return tuple(int(x) for x in b.split("."))


# ── main-effects baseline ────────────────────────────────────────────────
class MainEffects:
    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.w, self.V, self.U = z["w"], z["V"], z["U"]
        self.maps = {m: i for i, m in enumerate(z["maps"].tolist())}
        self.tiers = {t: i for i, t in enumerate(z["tiers"].tolist())}

    def s(self, A, maps, tiers):
        mi = np.array([self.maps.get(m, len(self.maps)) for m in maps])
        ti = np.array([self.tiers.get(t, len(self.tiers)) for t in tiers])
        return (self.w[A] + self.V[mi[:, None], A] + self.U[ti[:, None], A]).sum(1)

    def logit(self, A0, A1, maps, tiers):
        return self.s(A0, maps, tiers) - self.s(A1, maps, tiers)


def fit_main(games, path, sd_w=1.0, sd_mt=0.25):
    import torch
    from shared import HEROES
    H = len(HEROES)
    maps = sorted({g[2] for g in games})
    tiers = sorted({g[1] for g in games})
    mi = {m: i for i, m in enumerate(maps)}
    ti = {t: i for i, t in enumerate(tiers)}
    A0 = torch.tensor(hidx([g[3] for g in games]))
    A1 = torch.tensor(hidx([g[4] for g in games]))
    M = torch.tensor([mi[g[2]] for g in games])[:, None]
    T = torch.tensor([ti[g[1]] for g in games])[:, None]
    y = torch.tensor([1.0 if g[6] == 0 else 0.0 for g in games], dtype=torch.float64)
    w = torch.zeros(H, dtype=torch.float64, requires_grad=True)
    V = torch.zeros(len(maps) + 1, H, dtype=torch.float64, requires_grad=True)
    U = torch.zeros(len(tiers) + 1, H, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, V, U], lr=1, max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = (w[A0] + V[M, A0] + U[T, A0]).sum(1) - (w[A1] + V[M, A1] + U[T, A1]).sum(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="sum")
        loss = loss + 0.5 * (w ** 2).sum() / sd_w ** 2 + 0.5 * ((V ** 2).sum() + (U ** 2).sum()) / sd_mt ** 2
        loss.backward()
        return loss
    t0 = time.time()
    for _ in range(3):
        loss = opt.step(closure)
    with torch.no_grad():
        z = (w[A0] + V[M, A0] + U[T, A0]).sum(1) - (w[A1] + V[M, A1] + U[T, A1]).sum(1)
        ll = torch.nn.functional.binary_cross_entropy_with_logits(z, y).item()
    np.savez(path, w=w.detach().numpy(), V=V.detach().numpy(), U=U.detach().numpy(),
             maps=np.array(maps), tiers=np.array(tiers))
    print(f"main effects {os.path.basename(path)}: {len(games):,} games, train log-loss {ll:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return {"games": len(games), "train_logloss": ll}


def base_path(name):
    return os.path.join(CACHE, f"main_{name}.npz")


_SPLIT = {}


def split_games():
    if not _SPLIT:
        _SPLIT.update(core.split_games())
    return _SPLIT


def vintage_train(name):
    from paper1_revision import vintage
    return vintage.keep(split_games()["train"], name)


def stage_base():
    info = {}
    if not os.path.exists(base_path("current")):
        info["current"] = fit_main(split_games()["train"], base_path("current"))
    for v in ("s4m", "s1y"):
        if not os.path.exists(base_path(v)):
            info[v] = fit_main(vintage_train(v), base_path(v))
    if info:
        p = os.path.join(CACHE, "main_info.json")
        old = json.load(open(p)) if os.path.exists(p) else {}
        old.update(info)
        json.dump(old, open(p, "w"), indent=1)


# ── values ───────────────────────────────────────────────────────────────
def load_wp(model_dir, name, seed=None):
    import torch
    from sweep_enriched_wp import WinProbEnrichedModel
    meta = json.load(open(os.path.join(model_dir, f"{name}.json")))
    seed = seed or meta["selected_seed"]
    cols = core.group_cols(meta["spec"]["groups"])
    m = WinProbEnrichedModel(len(cols), meta["spec"]["arch"], dropout=0.3)
    m.load_state_dict(torch.load(os.path.join(model_dir, f"{name}_s{seed}.pt"), map_location="cpu", weights_only=True))
    m.eval()
    return m, cols


def stats_at(stats_dir):
    from overfit2026 import feats
    st = feats.stats_from_json(os.path.join(stats_dir, "deploy.json"), compositions=False)
    st.comp_data = core.comps_from_json(os.path.join(stats_dir, "deploy_compositions.json"))
    return st


def wp_values(rows, st, models, chunk=400000):
    """rows (team0, team1, map, tier) -> {name: symmetrized P(team0 wins)} for each (name, model, cols)."""
    import torch
    torch.set_num_threads(8)
    out = {n: [] for n, _, _ in models}
    for i in range(0, len(rows), chunk):
        Xf, Xs = core.featurize(rows[i:i + chunk], st, nproc=NPROC)
        with torch.no_grad():
            for n, m, cols in models:
                a = m(torch.tensor(Xf[:, cols])).view(-1).numpy()
                b = m(torch.tensor(Xs[:, cols])).view(-1).numpy()
                out[n].append(0.5 * (a + 1 - b))
        del Xf, Xs
    return {n: np.concatenate(v) for n, v in out.items()}


def gn_values(rows, A0, A1, C, chunk=400000):
    """gN judge as the frontier scores it: v1 gN, v2 structural correction, binned support offset."""
    from overfit2026 import judges_v2
    from paper1_revision import support_mitigation as sm
    jc = json.load(open(os.path.join(sm.OUT, "judge_corr.json")))
    wb = np.array(jc["judges"]["gN"]["w_binned"])
    raw, cor = [], []
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        base = judges_v2.score_base(r, judges=["gN"], nproc=NPROC)
        v2 = judges_v2.correct(base, r, judges=["gN"])["gN"]
        m0, m1 = C.metrics(A0[i:i + chunk]), C.metrics(A1[i:i + chunk])
        raw.append(np.asarray(base["gN"], float))
        cor.append(sig(logit(v2) + (sm.bin_feats(m0, jc["edges"]) - sm.bin_feats(m1, jc["edges"])) @ wb))
    return {"gN_raw": np.concatenate(raw), "gN": np.concatenate(cor)}


def ref_pool(n=3000, seed=7):
    from paper1_revision import judge_ood_calibration as joc
    tr = split_games()["train"]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tr), n, replace=False)
    return np.array([sorted(joc.idx_teams([tr[i]], 3 + (j % 2))[0]) for j, i in enumerate(pick)], np.int64)


def ref_rows(A0, A1, maps, tiers, pool, K, seed=11):
    """For each game, K reference teams disjoint from both teams; rows (team, ref) for team0 then team1."""
    from shared import HEROES
    rng = np.random.default_rng(seed)
    n = len(A0)
    R = np.zeros((n, K), np.int64)
    used = np.zeros((n, len(HEROES)), bool)
    used[np.arange(n)[:, None], A0] = True
    used[np.arange(n)[:, None], A1] = True
    for k in range(K):
        idx = rng.integers(0, len(pool), n)
        for _ in range(200):
            bad = used[np.arange(n)[:, None], pool[idx]].any(1)
            if not bad.any():
                break
            idx[bad] = rng.integers(0, len(pool), int(bad.sum()))
        R[:, k] = idx
    H = [HEROES[i] for i in range(len(HEROES))]
    names = lambda a: [H[j] for j in a]
    rows, TA, TR = [], [], []
    for side, A in ((0, A0), (1, A1)):
        for k in range(K):
            for g in range(n):
                rows.append((names(A[g]), names(pool[R[g, k]]), maps[g], tiers[g]))
            TA.append(A)
            TR.append(pool[R[:, k]])
    return rows, np.concatenate(TA), np.concatenate(TR)


# ── regression helpers ───────────────────────────────────────────────────
def glm(y, off, X, cl=None, iters=50, ridge=1e-9):
    """Logistic GLM with offset; SE clustered by cl (or HC0). -> coef, se."""
    X = np.asarray(X, float)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sig(off + X @ w)
        g = X.T @ (y - p) - ridge * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(w))
        step = np.linalg.solve(Hm, g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    p = sig(off + X @ w)
    Hi = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(w)))
    sc = X * (y - p)[:, None]
    if cl is not None:
        u = np.unique(cl, return_inverse=True)[1]
        S = np.zeros((u.max() + 1, X.shape[1]))
        np.add.at(S, u, sc)
        sc = S
    return w, np.sqrt(np.diag(Hi @ (sc.T @ sc) @ Hi))


def cmean(x, gid):
    """mean and game-clustered SE, in points."""
    mu = x.mean()
    u = np.unique(gid, return_inverse=True)[1]
    return [float(100 * mu), float(100 * np.sqrt((np.bincount(u, x - mu) ** 2).sum()) / len(x))]


def bin_table(y, p, b, own, opp, gid, edges=EDGES, opp_floor=FLOOR, min_n=30):
    """Per own-min3 bin: bias, GLM slope on I = logit p - logit b, top-decile-of-I numbers."""
    I = logit(p) - logit(b)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (own >= lo) & (own < hi)
        if opp_floor:
            sel &= opp >= opp_floor
        n = int(sel.sum())
        rec = {"min3_lo": lo, "min3_hi": hi if hi < 10 ** 12 else None, "n": n}
        if n >= min_n:
            rec.update(_cell(y[sel], p[sel], b[sel], I[sel], gid[sel]))
            q = np.percentile(I[sel], 90)
            t = I[sel] >= q
            if t.sum() >= min_n:
                rec["top_decile"] = _cell(y[sel][t], p[sel][t], b[sel][t], I[sel][t], gid[sel][t])
        out.append(rec)
    return out


def pooled(y, p, b, own, opp, gid, edges=EDGES, opp_floor=FLOOR):
    """Below vs at/above the floor, pooled over bins; top decile taken within each fine bin (as the
    per-bin table), then concatenated. beta_diff: I x below in one GLM; the top-decile difference SE
    treats the two sides as independent (they share few games)."""
    I = logit(p) - logit(b)
    keep = opp >= opp_floor if opp_floor else np.ones(len(y), bool)
    top = np.zeros(len(y), bool)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = keep & (own >= lo) & (own < hi)
        if sel.sum() >= 30:
            top |= sel & (I >= np.percentile(I[sel], 90))
    below = own < FLOOR
    out = {}
    for tag, sel in (("below_floor", keep & below), ("in_support", keep & ~below)):
        out[tag] = _cell(y[sel], p[sel], b[sel], I[sel], gid[sel])
        t = sel & top
        out[tag]["top_decile"] = _cell(y[t], p[t], b[t], I[t], gid[t])
    k = keep
    bl = below[k].astype(float)
    w, se = glm(y[k], logit(b[k]), np.column_stack([np.ones(k.sum()), bl, I[k], I[k] * bl]), gid[k])
    out["beta_diff_below_minus_in"] = [float(w[3]), float(se[3])]
    a, c = out["below_floor"]["top_decile"]["overval_p_minus_y"], out["in_support"]["top_decile"]["overval_p_minus_y"]
    out["topdec_overval_diff_below_minus_in"] = [a[0] - c[0], float(np.hypot(a[1], c[1]))]
    a, c = out["below_floor"]["overval_p_minus_y"], out["in_support"]["overval_p_minus_y"]
    out["overval_diff_below_minus_in"] = [a[0] - c[0], float(np.hypot(a[1], c[1]))]
    return out


def _cell(y, p, b, I, gid):
    w, se = glm(y, logit(b), np.column_stack([np.ones(len(y)), I]), gid)
    return {"n": int(len(y)), "bias_y_minus_p": cmean(y - p, gid), "claimed_pts": cmean(p - b, gid),
            "realized_pts": cmean(y - b, gid), "overval_p_minus_y": cmean(p - y, gid),
            "alpha": [float(w[0]), float(se[0])], "beta": [float(w[1]), float(se[1])],
            "I_sd": float(I.std())}


def team_rows(y0, p0, b0, m0, m1):
    y = np.concatenate([y0, 1 - y0])
    p = np.concatenate([p0, 1 - p0])
    b = np.concatenate([b0, 1 - b0])
    own = np.concatenate([m0, m1])
    opp = np.concatenate([m1, m0])
    gid = np.concatenate([np.arange(len(y0))] * 2)
    return y, p, b, own, opp, gid


def jbin(m):
    e = np.asarray(JEDGES)
    return np.clip(np.searchsorted(e, m, side="right") - 1, 0, len(e) - 2)


def joint_design(I, Ib0, Ib1, k0, k1):
    """Game-level design: alpha_k (top bin dropped), beta_own,k, beta_cross split by floor."""
    nb = len(JEDGES) - 1
    cols, names = [], []
    for k in range(nb - 1):
        cols.append((k0 == k).astype(float) - (k1 == k).astype(float))
        names.append(f"alpha_{JEDGES[k]}")
    for k in range(nb):
        cols.append(Ib0 * (k0 == k) - Ib1 * (k1 == k))
        names.append(f"beta_own_{JEDGES[k]}")
    cross = I - (Ib0 - Ib1)
    rare = (np.minimum(k0, k1) <= JEDGES.index(FLOOR) - 1)
    cols += [cross * rare, cross * ~rare]
    names += ["beta_cross_game_below_floor", "beta_cross_game_in_support"]
    return np.column_stack(cols), names


# ── stage A ──────────────────────────────────────────────────────────────
def nodrift():
    games = core.gold_games("NODRIFT")
    rows = [(list(g[3]), list(g[4]), g[2], g[1]) for g in games]
    y0 = np.array([1.0 if g[6] == 0 else 0.0 for g in games])
    return games, rows, y0


def search_drafts():
    from paper1_revision import ood_frontier as of
    seen, out = set(), []
    for a, b, r in of.records():
        if not ((a in SEARCH and b in OPP8 and a != b) or (b in SEARCH and a in OPP8 and a != b)):
            continue
        k = (a, b, r["draft"], r["game_map"], tuple(r["team0"]["picks"]), tuple(r["team1"]["picks"]))
        if k not in seen:
            seen.add(k)
            out.append((a, b, r))
    return out


def values_cached(tag, rows, A0, A1, C, st, models, want_gn=True):
    p = os.path.join(CACHE, f"vals_{tag}.npz")
    if os.path.exists(p):
        z = np.load(p)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    v = wp_values(rows, st, models)
    if want_gn:
        v.update(gn_values(rows, A0, A1, C))
    np.savez(p, **v)
    print(f"values {tag}: {len(rows):,} rows ({time.time() - t0:.0f}s)", flush=True)
    return v


def stage_slope():
    from paper1_revision import support_mitigation as sm
    C = sm.sup_corpus()
    st = core.load_stats("deploy")
    E = MainEffects(base_path("current"))
    models = [("agent", *load_wp(core.MODEL_DIR, "enriched")), ("herostrength", *load_wp(core.MODEL_DIR, "herostrength"))]
    games, rows, y0 = nodrift()
    A0, A1 = hidx([g[3] for g in games]), hidx([g[4] for g in games])
    maps, tiers = [g[2] for g in games], [g[1] for g in games]
    m0, m1 = C.metrics(A0)["min3"], C.metrics(A1)["min3"]
    V = values_cached("nodrift", rows, A0, A1, C, st, models)
    b0 = sig(E.logit(A0, A1, maps, tiers))
    res = {"games": len(games), "baseline": "main-effects logistic (hero + hero x map + hero x tier), current train split",
           "logloss": {k: float(-np.mean(y0 * np.log(clip(v)) + (1 - y0) * np.log(1 - clip(v))))
                       for k, v in list(V.items()) + [("main_effects", b0)]}}
    # A1: per-bin team rows
    res["bins"] = {}
    for val, base, tag in (("agent", b0, "agent"), ("agent", V["herostrength"], "agent_vs_herostrength_base"),
                           ("gN", b0, "gN"), ("gN_raw", b0, "gN_raw")):
        y, p, b, own, opp, gid = team_rows(y0, V[val], base, m0, m1)
        res["bins"][tag] = {"opp_in_support": bin_table(y, p, b, own, opp, gid),
                            "all_opp": bin_table(y, p, b, own, opp, gid, opp_floor=0),
                            "pooled": pooled(y, p, b, own, opp, gid)}
    # A2: own/cross split with reference opponents
    pool = ref_pool()
    rp = os.path.join(CACHE, "refvals_nodrift.npz")
    if os.path.exists(rp):
        z = np.load(rp)
        RV = {k: z[k] for k in z.files}
    else:
        rrows, TA, TR = ref_rows(A0, A1, maps, tiers, pool, K_REF)
        RV = values_cached("nodrift_ref", rrows, TA, TR, C, st, models[:1])
        bl = E.logit(TA, TR, maps * (2 * K_REF), tiers * (2 * K_REF))
        RV = {k: (logit(v) - bl).reshape(2, K_REF, len(games)).mean(1) for k, v in RV.items() if k in ("agent", "gN")}
        np.savez(rp, **RV)
    k0, k1 = jbin(m0), jbin(m1)
    res["joint"] = {}
    for val in ("agent", "gN"):
        I = logit(V[val]) - logit(b0)
        X, names = joint_design(I, RV[val][0], RV[val][1], k0, k1)
        w, se = glm(y0, logit(b0), np.column_stack([np.ones(len(y0)), X]))
        res["joint"][val] = {"intercept": [float(w[0]), float(se[0])],
                             **{n: [float(a), float(s)] for n, a, s in zip(names, w[1:], se[1:])},
                             "own_sd": float(np.std(RV[val][0] - RV[val][1])),
                             "cross_sd": float(np.std(I - (RV[val][0] - RV[val][1])))}
    # A3: implied overvaluation on search-chosen teams
    res["search"] = search_implied(res["joint"], C, st, E, models[:1], pool)
    json.dump(res, open(os.path.join(OUT, "slope.json"), "w"), indent=1)
    print_slope(res)


def search_implied(J, C, st, E, models, pool):
    recs = search_drafts()
    print(f"search drafts: {len(recs):,}", flush=True)
    rows = [(r["team0"]["picks"], r["team1"]["picks"], r["game_map"], r["tier"]) for _, _, r in recs]
    A0, A1 = hidx([x[0] for x in rows]), hidx([x[1] for x in rows])
    maps, tiers = [x[2] for x in rows], [x[3] for x in rows]
    V = values_cached("search", rows, A0, A1, C, st, models)
    rp = os.path.join(CACHE, "refvals_search.npz")
    if os.path.exists(rp):
        z = np.load(rp)
        RV = {k: z[k] for k in z.files}
    else:
        rrows, TA, TR = ref_rows(A0, A1, maps, tiers, pool, K_REF, seed=12)
        RV = values_cached("search_ref", rrows, TA, TR, C, st, models)
        bl = E.logit(TA, TR, maps * (2 * K_REF), tiers * (2 * K_REF))
        RV = {k: (logit(v) - bl).reshape(2, K_REF, len(rows)).mean(1) for k, v in RV.items() if k in ("agent", "gN")}
        np.savez(rp, **RV)
    b0 = sig(E.logit(A0, A1, maps, tiers))
    m0, m1 = C.metrics(A0)["min3"], C.metrics(A1)["min3"]
    k0, k1 = jbin(m0), jbin(m1)
    S0 = np.array([a for a, _, _ in recs])
    S1 = np.array([b for _, b, _ in recs])
    out = {}
    for val in ("agent", "gN"):
        I = logit(V[val]) - logit(b0)
        X, names = joint_design(I, RV[val][0], RV[val][1], k0, k1)
        coef = np.array([J[val][n][0] for n in names])
        na = len(JEDGES) - 2
        z_full = logit(b0) + X @ coef
        z_int = logit(b0) + X[:, na:] @ coef[na:]                 # interaction shrinkage only, no bin alphas
        over_full = V[val] - sig(z_full)                           # P(team0) overvaluation
        over_int = V[val] - sig(z_int)
        claim_own = sig(logit(b0) + RV[val][0] - RV[val][1]) - b0  # team0's own-interaction claim (pts of P)
        for s in SEARCH:
            row = {}
            per = {k: [] for k in ("p", "over_full", "over_int", "claim", "claim_own", "below", "over_below_full",
                                   "over_inb_full")}
            for o in OPP8:
                if o == s or o == "constrained_mcts":  # the common opponent set (frontier_excl_constrained_mcts)
                    continue
                f = (S0 == s) & (S1 == o)
                g = (S0 == o) & (S1 == s)
                if not (f.any() or g.any()):
                    continue
                own = np.concatenate([m0[f], m1[g]])
                cat = lambda x: np.concatenate([x[f], -x[g]])
                per["p"].append(np.concatenate([V[val][f], 1 - V[val][g]]).mean())
                per["over_full"].append(cat(over_full).mean())
                per["over_int"].append(cat(over_int).mean())
                per["claim"].append(np.concatenate([V[val][f] - b0[f], b0[g] - V[val][g]]).mean())
                per["claim_own"].append(cat(claim_own).mean())
                per["below"].append((own < FLOOR).mean())
                ob = cat(over_full)
                per["over_below_full"].append(ob[own < FLOOR].mean() if (own < FLOOR).any() else np.nan)
                per["over_inb_full"].append(ob[own >= FLOOR].mean() if (own >= FLOOR).any() else np.nan)
            if not per["p"]:
                continue
            row = {k: float(np.nanmean(v)) * (1 if k == "below" else 100) for k, v in per.items()}
            row["opponents"] = len(per["p"])
            row["corrected_p"] = row["p"] - row["over_full"]
            out.setdefault(val, {})[s] = row
    return out


def print_slope(res):
    for tag in ("agent", "agent_vs_herostrength_base", "gN"):
        print(f"\n== {tag} (opponent in support) ==")
        print(f"{'bin':>9s} {'n':>7s} {'bias':>12s} {'beta':>14s} {'claim':>6s} {'real':>6s} | top-dec: {'beta':>14s} {'claim':>6s} {'over':>12s}")
        for r in res["bins"][tag]["opp_in_support"]:
            if "beta" not in r:
                continue
            t = r.get("top_decile", {})
            tb = f"{t['beta'][0]:.2f}±{t['beta'][1]:.2f}" if t else ""
            print(f"{r['min3_lo']:>4}-{str(r['min3_hi']):>4} {r['n']:7d} {r['bias_y_minus_p'][0]:+5.2f}±{r['bias_y_minus_p'][1]:4.2f} "
                  f"{r['beta'][0]:6.2f}±{r['beta'][1]:5.2f} {r['claimed_pts'][0]:+6.2f} {r['realized_pts'][0]:+6.2f} | "
                  f"{tb:>14s} {t.get('claimed_pts', [0])[0]:+6.2f} {t.get('overval_p_minus_y', [0, 0])[0]:+5.2f}±{t.get('overval_p_minus_y', [0, 0])[1]:4.2f}")
    for tag in ("agent", "agent_vs_herostrength_base", "gN"):
        P = res["bins"][tag]["pooled"]
        print(f"pooled {tag}: " + "  ".join(f"{k} beta {P[k]['beta'][0]:.2f}±{P[k]['beta'][1]:.2f} over {P[k]['overval_p_minus_y'][0]:+.2f} "
              f"top-dec over {P[k]['top_decile']['overval_p_minus_y'][0]:+.2f}±{P[k]['top_decile']['overval_p_minus_y'][1]:.2f}"
              for k in ("below_floor", "in_support")) + f"  beta diff {P['beta_diff_below_minus_in'][0]:+.2f}±{P['beta_diff_below_minus_in'][1]:.2f}"
              f"  top-dec over diff {P['topdec_overval_diff_below_minus_in'][0]:+.2f}±{P['topdec_overval_diff_below_minus_in'][1]:.2f}")
    for val, J in res["joint"].items():
        print(f"\n== joint own/cross GLM: {val} ==")
        for k, v in J.items():
            print(f"  {k:32s} {v[0]:+.3f} ± {v[1]:.3f}" if isinstance(v, list) else f"  {k:32s} {v:.3f}")
    for val, S in res["search"].items():
        print(f"\n== implied overvaluation on search teams: {val} ==")
        for s, r in S.items():
            print(f"  {s:24s} below {r['below']:.2f}  p {r['p']:.2f}  claim {r['claim']:+.2f} (own {r['claim_own']:+.2f})  "
                  f"over {r['over_full']:+.2f} (interaction-only {r['over_int']:+.2f}; below-floor teams {r['over_below_full']:+.2f}, "
                  f"in-support {r['over_inb_full']:+.2f})  corrected {r['corrected_p']:.2f}")


# ── stage B ──────────────────────────────────────────────────────────────
def late_games(v):
    from overfit2026 import data as odata
    from paper1_revision import vintage
    cut = build_tuple(vintage.spec(v)["cutoff"])
    snap = odata.load_snapshot()
    b, _ = vintage.build_info([g[0] for g in snap])
    ci = vintage.cutoff_idx(v)
    out = [(g, "snapshot") for g, x in zip(snap, b) if x > ci]
    for bld, gs in odata.load_backfill().items():
        if build_tuple(bld) > cut:
            out += [(g, "backfill") for g in gs]
    out += [(g, "T97") for g in odata.load_future()["2.55.16.97039"]]
    return out


def stage_backtest():
    from paper1_revision import team_support as ts
    from paper1_revision import judge_ood_calibration as joc
    from paper1_revision import vintage
    res = {}
    for v in ("s4m", "s1y"):
        vdir = os.path.join(HERE, "site", "vint", v)
        early = vintage_train(v)
        Ce = ts.Corpus(np.concatenate([joc.idx_teams(early, 3), joc.idx_teams(early, 4)]))
        LG = late_games(v)
        games = [g for g, _ in LG]
        src = np.array([s for _, s in LG])
        Cl = ts.Corpus(np.concatenate([joc.idx_teams(games, 3), joc.idx_teams(games, 4)]))
        rows = [(list(g[3]), list(g[4]), g[2], g[1]) for g in games]
        y0 = np.array([1.0 if g[6] == 0 else 0.0 for g in games])
        A0, A1 = hidx([g[3] for g in games]), hidx([g[4] for g in games])
        p = os.path.join(CACHE, f"vals_late_{v}.npz")
        if os.path.exists(p):
            p0 = np.load(p)["agent"]
        else:
            t0 = time.time()
            p0 = wp_values(rows, stats_at(os.path.join(vdir, "cache", "stats")),
                           [("agent", *load_wp(os.path.join(vdir, "models"), "enriched"))])["agent"]
            np.savez(p, agent=p0)
            print(f"vintage {v}: {len(rows):,} late games ({time.time() - t0:.0f}s)", flush=True)
        E = MainEffects(base_path(v))
        b0 = sig(E.logit(A0, A1, [g[2] for g in games], [g[1] for g in games]))
        e0, e1 = Ce.metrics(A0)["min3"], Ce.metrics(A1)["min3"]
        l0, l1 = Cl.metrics(A0)["min3"], Cl.metrics(A1)["min3"]
        y, pp, b, own, opp, gid = team_rows(y0, p0, b0, e0, e1)
        late_own = np.concatenate([l0, l1])
        r = {"early_games": len(early), "late_games": len(games),
             "late_by_source": {s: int((src == s).sum()) for s in np.unique(src)},
             "cutoff": vintage.spec(v)["cutoff"],
             "early_min3_below_25_share_late_teams": float((own < FLOOR).mean()),
             "bins": {"opp_in_support": bin_table(y, pp, b, own, opp, gid),
                      "all_opp": bin_table(y, pp, b, own, opp, gid, opp_floor=0)},
             "pooled": pooled(y, pp, b, own, opp, gid)}
        # "moved into support": the team's rarest triple, leaving out its own game, occurs in the later
        # games at least as often per team as the floor does in the early corpus (25 per early-corpus size)
        thr_late = FLOOR * Cl.n / Ce.n
        r["late_support_threshold"] = float(thr_late)
        sup = (late_own - 1) >= thr_late
        for tag, sel in (("late_supported", sup), ("late_still_rare", ~sup)):
            r["bins"][tag] = bin_table(y[sel], pp[sel], b[sel], own[sel], opp[sel], gid[sel])
        # drift control: rare-early bins minus the common-early (>= 400) bin, same games
        base = [x for x in r["bins"]["opp_in_support"] if x["min3_lo"] == 400][0]
        r["drift_control_bias_400plus"] = base.get("bias_y_minus_p")
        r["drift_control_beta_400plus"] = base.get("beta")
        res[v] = r
        print(f"\n== backtest {v}: {len(games):,} late games, early corpus {len(early):,} ==")
        for tag in ("opp_in_support", "late_supported", "late_still_rare"):
            print(f"  [{tag}]")
            for x in r["bins"][tag]:
                if "beta" in x:
                    t = x.get("top_decile", {})
                    print(f"   {x['min3_lo']:>4}-{str(x['min3_hi']):>4} n {x['n']:7d} bias {x['bias_y_minus_p'][0]:+5.2f}±{x['bias_y_minus_p'][1]:.2f} "
                          f"beta {x['beta'][0]:.2f}±{x['beta'][1]:.2f}  top-dec over {t.get('overval_p_minus_y', [np.nan, np.nan])[0]:+.2f}"
                          f"±{t.get('overval_p_minus_y', [np.nan, np.nan])[1]:.2f}")
    json.dump(res, open(os.path.join(OUT, "backtest.json"), "w"), indent=1)


# ── stage C ──────────────────────────────────────────────────────────────
def stage_qm():
    from shared import HERO_TO_IDX
    from paper1_revision import support_mitigation as sm
    path = os.path.join(TRAINING_DIR, "overfit2026", "site", "cache", "qm_games_site.jsonl")
    games = []
    for line in open(path):
        d = json.loads(line)
        if (d.get("game_date") or "") < QM_FROM or d.get("winner") not in (0, 1):
            continue
        t0, t1 = d["team0_heroes"], d["team1_heroes"]
        if any(h not in HERO_TO_IDX for h in t0 + t1) or len(set(t0 + t1)) != 10:
            continue
        games.append((d["replay_id"], d["skill_tier"], d["game_map"], tuple(t0), tuple(t1), (), d["winner"]))
    C = sm.sup_corpus()
    A0, A1 = hidx([g[3] for g in games]), hidx([g[4] for g in games])
    m0, m1 = C.metrics(A0)["min3"], C.metrics(A1)["min3"]
    gnd = core.gold_games("NODRIFT")
    n0 = C.metrics(hidx([g[3] for g in gnd]))["min3"]
    n1 = C.metrics(hidx([g[4] for g in gnd]))["min3"]
    qm_t, rk_t = np.concatenate([m0, m1]), np.concatenate([n0, n1])
    res = {"games": len(games), "from": QM_FROM,
           "width": {k: {"below_25": float((x < 25).mean()), "below_5": float((x < 5).mean()),
                         "below_50": float((x < 50).mean()), "median_min3": float(np.median(x))}
                     for k, x in (("qm", qm_t), ("ranked_nodrift", rk_t))}}
    rows = [(list(g[3]), list(g[4]), g[2], g[1]) for g in games]
    y0 = np.array([1.0 if g[6] == 0 else 0.0 for g in games])
    p = os.path.join(CACHE, "vals_qm.npz")
    if os.path.exists(p):
        p0 = np.load(p)["agent"]
    else:
        p0 = wp_values(rows, core.load_stats("deploy"), [("agent", *load_wp(core.MODEL_DIR, "enriched"))])["agent"]
        np.savez(p, agent=p0)
    E = MainEffects(base_path("current"))
    b0 = sig(E.logit(A0, A1, [g[2] for g in games], [g[1] for g in games]))
    w, se = glm(y0, np.zeros(len(y0)), np.column_stack([np.ones(len(y0)), logit(p0)]))
    res["overall_calibration_slope"] = [float(w[1]), float(se[1])]
    res["logloss"] = {k: float(-np.mean(y0 * np.log(clip(v)) + (1 - y0) * np.log(1 - clip(v))))
                      for k, v in (("agent", p0), ("main_effects", b0), ("coin", np.full(len(y0), 0.5)))}
    y, pp, b, own, opp, gid = team_rows(y0, p0, b0, m0, m1)
    res["bins"] = {"opp_in_support": bin_table(y, pp, b, own, opp, gid),
                   "all_opp": bin_table(y, pp, b, own, opp, gid, opp_floor=0)}
    res["pooled"] = pooled(y, pp, b, own, opp, gid)
    json.dump(res, open(os.path.join(OUT, "qm.json"), "w"), indent=1)
    print(json.dumps(res["width"], indent=1), res["overall_calibration_slope"], res["logloss"])
    for x in res["bins"]["opp_in_support"]:
        if "beta" in x:
            print(f"   {x['min3_lo']:>4}-{str(x['min3_hi']):>4} n {x['n']:7d} bias {x['bias_y_minus_p'][0]:+5.2f}±{x['bias_y_minus_p'][1]:.2f} "
                  f"beta {x['beta'][0]:.2f}±{x['beta'][1]:.2f}")


# ── report ───────────────────────────────────────────────────────────────
def _f(x, d=2, sign=True):
    if not x:
        return ""
    return (f"{x[0]:+.{d}f}" if sign else f"{x[0]:.{d}f}") + f" ± {x[1]:.{d}f}"


def _bins_md(bins, top=True):
    h = "| own min3 | n | bias y−p (pts) | alpha | beta | claimed (pts) | realized (pts) |"
    if top:
        h += " top-decile beta | top-decile claimed | top-decile overval p−y |"
    L = [h, "|" + "---|" * (h.count("|") - 1)]
    for r in bins:
        if "beta" not in r:
            continue
        hi = r["min3_hi"] if r["min3_hi"] is not None else "∞"
        row = (f"| {r['min3_lo']}–{hi} | {r['n']:,} | {_f(r['bias_y_minus_p'])} | {_f(r['alpha'])} | "
               f"{_f(r['beta'], sign=False)} | {r['claimed_pts'][0]:+.2f} | {_f(r['realized_pts'])} |")
        if top:
            t = r.get("top_decile")
            row += (f" {_f(t['beta'], sign=False)} | {t['claimed_pts'][0]:+.2f} | {_f(t['overval_p_minus_y'])} |" if t else " | | |")
        L.append(row)
    return "\n".join(L)


def _pooled_md(P):
    L = ["| own min3 | n | bias y−p | beta | overval p−y | top-decile n | top-decile beta | top-decile claimed | top-decile overval p−y |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, lab in (("below_floor", "< 25"), ("in_support", "≥ 25")):
        r, t = P[k], P[k]["top_decile"]
        L.append(f"| {lab} | {r['n']:,} | {_f(r['bias_y_minus_p'])} | {_f(r['beta'], sign=False)} | {_f(r['overval_p_minus_y'])} | "
                 f"{t['n']:,} | {_f(t['beta'], sign=False)} | {t['claimed_pts'][0]:+.2f} | {_f(t['overval_p_minus_y'])} |")
    L += ["", f"Below minus in-support: beta {_f(P['beta_diff_below_minus_in'])}; overvaluation {_f(P['overval_diff_below_minus_in'])} pts; "
          f"top-decile overvaluation {_f(P['topdec_overval_diff_below_minus_in'])} pts."]
    return "\n".join(L)


def stage_report():
    S = json.load(open(os.path.join(OUT, "slope.json")))
    B = json.load(open(os.path.join(OUT, "backtest.json")))
    Q = json.load(open(os.path.join(OUT, "qm.json")))
    L = ["# How much does the WP model overrate rare compositions? (real data)", "",
         "Generated by `paper1_revision/overrating.py report` from slope.json, backtest.json, qm.json. "
         "Points are win-probability points. Floor = 25 (min3, training-split teams).", "",
         BOTTOM_LINE, "",
         "## Method", "",
         "Baseline b: main-effects logistic (hero + hero×map + hero×tier, ridge), fit on the corpus the model was "
         "trained on. Interaction claim I = logit p − logit b. GLM y ~ Bernoulli(σ(logit b + α + β·I)); β = 1 means "
         "the non-additive part is right on average, β < 1 overrated. 'claimed' = mean(p − b), 'realized' = mean(y − b), "
         "'top decile' = the 10% of teams with the largest I in the bin (what an argmax would pick). "
         "SEs clustered by game. Bins by the team's own rarest-triple count; rows require the opponent at min3 ≥ 25 "
         "(as certify) unless stated.", "",
         f"Held-out log-loss on NODRIFT: " + ", ".join(f"{k} {v:.4f}" for k, v in S["logloss"].items()), "",
         "## A. Selection-aware slope, NODRIFT (260,539 post-snapshot games)", "",
         "### Agent leaf (enriched leak-free WP, deploy statistics)", "", _pooled_md(S["bins"]["agent"]["pooled"]), "",
         _bins_md(S["bins"]["agent"]["opp_in_support"]), "",
         "### Same, herostrength WP net as the baseline (robustness)", "",
         _pooled_md(S["bins"]["agent_vs_herostrength_base"]["pooled"]), "",
         _bins_md(S["bins"]["agent_vs_herostrength_base"]["opp_in_support"]), "",
         "### gN judge (frontier form: v2 structural + binned support correction)", "",
         _pooled_md(S["bins"]["gN"]["pooled"]), "", _bins_md(S["bins"]["gN"]["opp_in_support"]), "",
         "### Own-team vs cross-team (one game-level GLM per value; own = Ibar(team) − Ibar(opp) over "
         f"{K_REF} shared random human reference opponents, cross = I − own)", "",
         "| coefficient | agent | gN |", "|---|---|---|"]
    for k in S["joint"]["agent"]:
        a, g = S["joint"]["agent"][k], S["joint"]["gN"].get(k)
        if isinstance(a, list):
            L.append(f"| {k} | {_f(a, 3)} | {_f(g, 3)} |")
        else:
            L.append(f"| {k} | {a:.3f} | {g:.3f} |")
    L += ["", "### Implied overvaluation on the search's own teams (common opponent set: OPP8 minus constrained_mcts)", "",
          "over = value − σ(fitted correct logit) from the joint GLM, averaged per opponent then over opponents; "
          "'interaction only' drops the bin intercepts α_k. Points.", "",
          "| value | strategy | share < 25 | value p | claimed I (p−b) | own claim | overval | interaction-only | on own<25 teams | on own≥25 teams | corrected p |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for val, SS in S["search"].items():
        for s, r in SS.items():
            L.append(f"| {val} | {s} | {r['below']:.2f} | {r['p']:.1f} | {r['claim']:+.2f} | {r['claim_own']:+.2f} | "
                     f"{r['over_full']:+.2f} | {r['over_int']:+.2f} | {r['over_below_full']:+.2f} | {r['over_inb_full']:+.2f} | "
                     f"{r['corrected_p']:.1f} |")
    L += ["", "## B. Temporal backtest (stale vintages)", ""]
    for v, r in B.items():
        L += [f"### {v}: cutoff {r['cutoff']}, early corpus {r['early_games']:,} games, {r['late_games']:,} later games "
              f"({', '.join(f'{k} {n:,}' for k, n in r['late_by_source'].items())}); "
              f"{100 * r['early_min3_below_25_share_late_teams']:.1f}% of later teams were below 25 early", "",
              "Bins by EARLY min3; the vintage's own WP and baseline.", "",
              "All later teams, pooled:", "", _pooled_md(r["pooled"]), "", _bins_md(r["bins"]["opp_in_support"]), "",
              f"Teams that moved into support: rarest triple (own game left out) in ≥ {r['late_support_threshold']:.1f} "
              "later teams, the floor's rate per corpus team:", "",
              _bins_md(r["bins"]["late_supported"]), "",
              "Teams still below that rate later:", "", _bins_md(r["bins"]["late_still_rare"]), ""]
    L += ["## C. Quick Match", "",
          f"{Q['games']:,} QM games dated ≥ {Q['from']} (site tiers). Share of teams by min3 against the ranked training corpus:", "",
          "| set | < 5 | < 25 | < 50 | median min3 |", "|---|---|---|---|---|"]
    for k, w in Q["width"].items():
        L.append(f"| {k} | {100 * w['below_5']:.1f}% | {100 * w['below_25']:.1f}% | {100 * w['below_50']:.1f}% | {w['median_min3']:.0f} |")
    L += ["", f"Overall calibration slope of the agent leaf on QM (logit): {_f(Q['overall_calibration_slope'], 3, False)}; "
          f"log-loss " + ", ".join(f"{k} {v:.4f}" for k, v in Q["logloss"].items()), "",
          _pooled_md(Q["pooled"]), "", _bins_md(Q["bins"]["opp_in_support"]), ""]
    open(os.path.join(OUT, "OVERRATING.md"), "w").write("\n".join(L))
    print("wrote", os.path.join(OUT, "OVERRATING.md"))


BOTTOM_LINE = """## Bottom line

**Rare teams are not overrated more than common ones. The extra overvaluation below the floor is about 0 pts, with
a 95% upper bound of about 2 pts. Break-even is about 2 pts in both papers.** Paper 1: the floor costs 1.27 pts
(consensus, corrected) or 1.42 pts (gN) on the common opponent set, with 63% of teams below the floor, so it pays
only if those teams are overvalued by 2.0–2.3 pts more than their floored replacements. Paper 2: 0.73 pp with 37%
below the floor, also about 2 pts. Nothing measured on real data reaches that.

- **Slope (NODRIFT, agent leaf).** The model's non-additive claims are about 10% too large everywhere. Slope below
  the floor: 0.92 ± 0.04. In support: 0.87 ± 0.02. The difference is +0.05 ± 0.05.
- **What an argmax picks.** Take the top decile of I in each bin, the teams the model likes most beyond main effects.
  They are overvalued by +1.45 ± 0.95 pts below the floor and +1.44 ± 0.23 pts in support. The difference is
  +0.01 ± 0.98 pts. This overrating is a winner's curse that does not depend on support, and the floor does not
  avoid it.
- **The search's own teams.** constrained_mcts's teams claim only about +2 pts of interaction on average, against
  about +10 pts for the human top decile. 70% of its below-floor teams sit at min3 10–24 and none at 0. The joint
  own/cross fit gives an implied overvaluation of −0.4 pts on its below-floor teams and −0.6 pts on its in-support
  teams. Even a below-floor slope 2 SE low (0.84) would cost at most about 0.5 pts.
- **Temporal backtest (the only direct off-support check).** The stale models judged teams that were below 25 in
  their own data, using later real games. Those teams were not overrated, and if anything did slightly better than
  predicted:
  - s4m: slope 1.00 ± 0.04 against 0.94 ± 0.02 in support; bias +0.27 ± 0.26 pts.
  - s1y: slope 1.08 ± 0.02 against 1.00 ± 0.01; bias +0.48 ± 0.13 pts.
  - Top-decile difference: −1.2 ± 0.9 pts (s4m) and −0.7 ± 0.4 pts (s1y).
- **Quick Match.** QM compositions are much wider: 23.6% of teams are below 25, against 5.4% of ranked held-out
  teams. The slope is again flat in support (0.88 ± 0.03 below, 0.84 ± 0.03 in support). There is a level bias of
  −1.0 ± 0.3 pts on below-floor teams. This sits inside an overall domain shift: the calibration slope on QM is 0.81.
- **Judge.** gN is attenuated, not inflated: slope 1.52 below the floor and 1.61 in support. Refitting gN's
  interaction scale raises the judged floor cost on the common set from 1.42 to 1.76 pts. So the frontier's floor
  cost is not a judge artifact of rare-team overrating.

**Dial ranges for the worlds:**
- Interaction attenuation: about 0.85–0.9, flat in support. The real-data bound below the floor is 0.84–1.1.
- Level shift for rare compositions: 0 ± 1 pts in ranked (certify: −1.8 ± 1.2 at min3 1–4), and −1 pt in QM.
- Winner's curse on top-claimed teams: about 1.5 pts at every support level; 3–4 pts against the herostrength
  baseline.

A world in which search-chosen rare teams are overvalued by more than about 2 pts beyond in-support teams is about
2 SE outside these data. The direct evidence is for min3 ≥ 5 (93% of the search's below-floor teams). Data are thin
at 1–4 (1,832 human team rows) and absent at 0, where the search puts none of its teams.

**Caveats.**
- All measured teams are human-chosen. The top decile of I stands in for argmax selection, and is conservative
  because its claims are larger than the search's.
- The absolute overvaluation depends on the baseline. The support difference does not: it is about 0 against both
  baselines.
- In the pooled in-support row, bias and overvaluation are 0 by construction, because both teams of a game are
  rows.
- Backtest confounds:
  - Drift: early-common teams show −0.7 and −1.35 pts.
  - Adoption selection: teams that later moved into support did much better than predicted, so read the still-rare
    split.
- QM is blind-pick, matchmade play, and the model is miscalibrated there overall.
- gN is 0.010 nats better than the agent on NODRIFT (log-loss 0.668 against 0.678). Its statistics may carry outcome
  information for this period, and its v2 structural coefficients may have been fit on NODRIFT-era games. Read its
  slope for direction only.
- The implied overvaluation for search teams extrapolates the human-data fit to the search's teams.
- The paper-2 personal term is not measured here."""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["base", "slope", "backtest", "qm", "report", "all"])
    a = ap.parse_args()
    if a.cmd in ("base", "all"):
        stage_base()
    if a.cmd in ("slope", "all"):
        stage_slope()
    if a.cmd in ("backtest", "all"):
        stage_backtest()
    if a.cmd in ("qm", "all"):
        stage_qm()
    if a.cmd in ("report", "all"):
        stage_report()
