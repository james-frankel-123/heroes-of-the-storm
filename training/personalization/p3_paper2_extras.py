"""
Paper 2 extras on the definitive build (def1027 chosen model; results/P3_PAPER2_EXTRAS.md).

  queries   state-writer query files for OOT slots at player lags L days:
            the player state as of day - L (L = 1 is the model as run), all
            heroes, unfiltered, the drafter fit (half 0). One file per lag.
  score     everything else, from the pinned arm files and the lag answers:
            draft-only (population WP) accuracy on the post-freeze set; the
            cost of a lagged player state (the draft side, population WP and
            the at-game MMR term unchanged: the combiner of the chosen arm
            with the lagged team difference); the one-year breakdown by what
            the stale state is missing (new players, new heroes, known pairs);
            gains by region and by tier.

Usage (from training/, def env of def1027, single-threaded BLAS):
  python3 personalization/p3_paper2_extras.py queries
  python3 personalization/p3_rcf_states.py answer <qdir> <adir>      (one per lag file)
  python3 personalization/p3_paper2_extras.py score
"""
import gzip
import json
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "results", "def_def1027", "rcf", "p3c")
QD = os.path.join(HERE, "cache_def_def1027", "lag_q")
AD = os.path.join(HERE, "cache_def_def1027", "lag_a")
LAGS = (1, 7, 30, 120, 365)
OUT = os.path.join(HERE, "results", "p3_paper2_extras.json")


def ll_of(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def queries():
    z = np.load(os.path.join(D, "chosen__merged.npz"))
    s = z["s_win"] == 3
    key = z["s_key"][s].astype(np.int64)
    day = z["s_day"][s].astype(np.int64)
    os.makedirs(QD, exist_ok=True)
    for L in LAGS:
        u = np.unique(np.stack([key, day - L], 1), axis=0)
        n = len(u)
        np.savez(os.path.join(QD, f"lag{L:03d}.npz"), player_key=u[:, 0], as_of_day=u[:, 1].astype(np.int32),
                 ev_lo=np.full(n, -1, np.int32), ev_parity=np.zeros(n, np.int8), exclude_lo=np.full(n, -1, np.int32),
                 exclude_hi=np.full(n, -1, np.int32), half=np.zeros(n, np.int8), hero=np.full(n, -1, np.int16))
        print(f"lag {L}: {n:,} player-days", flush=True)


def block_ci(d, block, rng, n=2000):
    u, inv = np.unique(block, return_inverse=True)
    S = np.bincount(inv, d)
    N = np.bincount(inv).astype(float)
    bs = [S[i].sum() / N[i].sum() for i in (rng.integers(0, len(u), len(u)) for _ in range(n))]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def score():
    import p3_hs_core as C
    from shared import HEROES
    rng = np.random.default_rng(0)
    out = {}
    # 1. draft-only accuracy on the post-freeze set
    pf = os.path.join(HERE, "results", "def_pf1027", "rcf", "p3c")
    A = {k: np.load(os.path.join(pf, f"{k}__merged.npz")) for k in ("chosen", "full_ms_layer", "today")}
    m = A["chosen"]["g_win"] == 4
    y = A["chosen"]["g_y"][m]
    acc = lambda p: float(((p > 0.5) == (y == 1)).mean())
    out["postfreeze"] = {"games": int(m.sum()), "draft_only_acc": acc(A["chosen"]["g_p0"][m]),
                         "chosen_acc": acc(A["chosen"]["g_p"][m]), "previous_acc": acc(A["full_ms_layer"]["g_p"][m]),
                         "today_acc": acc(A["today"]["g_p"][m]),
                         "draft_only_gain_x1000_vs_coin": float(1000 * (np.log(2) - ll_of(A["chosen"]["g_p0"][m], y)).mean())}
    # 2-3. lagged player state (OOT)
    z = np.load(os.path.join(D, "chosen__merged.npz"))
    s = np.flatnonzero(z["s_win"] == 3)
    key, day, hero = z["s_key"][s].astype(np.int64), z["s_day"][s].astype(np.int64), z["s_hero"][s].astype(np.int64)
    names = [str(h) for h in np.load(os.path.join(HERE, "cache_def_def1027", "x_slots_ext.npz"), mmap_mode="r")["hero_names"]]
    to_sh = np.array([HEROES.index(h) for h in names])[hero]
    g_rid = z["g_replay_id"]
    gm = z["g_win"] == 3
    gpos = {int(r): i for i, r in enumerate(g_rid)}
    gi = np.array([gpos.get(int(r), -1) for r in z["s_replay_id"][s]])
    assert (gi >= 0).all()
    sign = np.where(z["s_team"][s] == 0, 1.0, -1.0)
    ng = len(g_rid)
    cf = np.asarray(z["coef"], np.float64)
    lg = lambda p: np.log(p / (1 - p))
    p_cur = z["g_p"]
    y = z["g_y"].astype(np.float64)
    D_cur = np.bincount(gi, weights=sign * z["s_pred"][s], minlength=ng)
    assert np.abs(D_cur[gm] - z["g_D"][gm]).max() < 1e-5, "team-difference convention"
    mmr_term = lg(np.clip(p_cur, 1e-12, 1 - 1e-12)) - cf[0] - cf[1] * z["g_lo"] - cf[2] * z["g_D"]
    ll0 = ll_of(z["g_p0"], y)
    llc = ll_of(p_cur, y)
    week = z["g_day"] // 7

    def game_p(Dv):
        return 1 / (1 + np.exp(-(cf[0] + cf[1] * z["g_lo"] + cf[2] * Dv + mmr_term)))

    out["lags"] = {}
    stale = {}
    for L in LAGS:
        a = np.load(os.path.join(AD, f"lag{L:03d}.npz"))
        ak = a["player_key"].astype(np.int64) * 100000 + a["as_of_day"].astype(np.int64)
        o = np.argsort(ak)
        j = o[np.searchsorted(ak[o], key * 100000 + day - L)]
        assert np.array_equal(ak[j], key * 100000 + day - L)
        sv = a["s"][j, to_sh].astype(np.float64)
        nh = a["n"][j, to_sh].astype(np.float64)
        ntot = a["n"][j].sum(1).astype(np.float64)
        stale[L] = (sv, nh, ntot)
        if L == 1:
            dm = float(np.abs(sv - z["s_pred"][s]).max())
            out["lag1_reproduces_arm_max_abs"] = dm
            assert dm < 1e-6, f"lag 1 does not reproduce the arm ({dm:.2e})"
        Dl = np.bincount(gi, weights=sign * sv, minlength=ng)
        pl = game_p(Dl)
        lll = ll_of(pl, y)
        cost = 1000 * (lll - llc)[gm]
        out["lags"][str(L)] = {"gain_x1000": float(1000 * (ll0 - lll)[gm].mean()),
                               "cost_x1000": float(cost.mean()), "cost_se": float(cost.std(ddof=1) / np.sqrt(gm.sum())),
                               "cost_week_ci": block_ci(cost, week[gm], rng),
                               "acc": float(((pl[gm] > .5) == (y[gm] == 1)).mean())}
    out["current"] = {"gain_x1000": float(1000 * (ll0 - llc)[gm].mean()),
                      "acc": float(((p_cur[gm] > .5) == (y[gm] == 1)).mean()), "games": int(gm.sum())}
    # one-year breakdown: replace only one category of slots by its stale state
    sv, nh, ntot = stale[365]
    cats = {"new player": ntot == 0, "new hero": (ntot > 0) & (nh == 0), "known pair": nh > 0}
    out["stale365_breakdown"] = {}
    for nm, c in cats.items():
        mix = np.where(c, sv, z["s_pred"][s])
        pl = game_p(np.bincount(gi, weights=sign * mix, minlength=ng))
        cost = 1000 * (ll_of(pl, y) - llc)[gm]
        out["stale365_breakdown"][nm] = {"slot_share": float(c.mean()), "cost_x1000": float(cost.mean()),
                                        "cost_se": float(cost.std(ddof=1) / np.sqrt(gm.sum()))}
    # region and tier (V2 and OOT, chosen vs previous and today)
    import p3_c_tests as T
    R = {k: np.load(os.path.join(D, f"{k}__merged.npz")) for k in ("chosen", "full_ms_layer", "today")}
    for k in R:
        assert np.array_equal(R[k]["g_replay_id"], g_rid)
    reg = np.full(ng, -1)
    sg = np.load(os.path.join(HERE, "cache", "x_side_games.npz"))
    rmap = dict(zip(sg["replay_ids"].tolist(), sg["region"].tolist()))
    for g_ in json.load(gzip.open(os.path.join(HERE, "cache_def_def1027", "x_post_games.json.gz")
                                  if os.path.exists(os.path.join(HERE, "cache_def_def1027", "x_post_games.json.gz"))
                                  else os.path.join(HERE, "cache", "x_post_games.json.gz"), "rt")):
        if g_.get("region") is not None:
            rmap[int(g_["replay_id"])] = int(g_["region"])
    reg = np.array([rmap.get(int(r), -1) for r in g_rid])
    tier = T.real_tier(g_rid)
    LL = {k: ll_of(R[k]["g_p"], y) for k in R}
    RN = {1: "Americas", 2: "Europe", 3: "Asia", 5: "China"}
    TN = {1: "Bronze", 2: "Silver", 3: "Gold", 4: "Platinum", 5: "Diamond", 6: "Master"}
    for lab, grp, nmap in (("region", reg, RN), ("tier", tier, TN)):
        out[lab] = {}
        for wk, wn in ((2, "V2"), (3, "OOT")):
            for gv in sorted(set(grp[z["g_win"] == wk].tolist())):
                mm = (z["g_win"] == wk) & (grp == gv)
                if mm.sum() < 500:
                    continue
                r_ = {"games": int(mm.sum()),
                      "chosen_gain_x1000": float(1000 * (ll0 - LL["chosen"])[mm].mean()),
                      "previous_gain_x1000": float(1000 * (ll0 - LL["full_ms_layer"])[mm].mean()),
                      "today_gain_x1000": float(1000 * (ll0 - LL["today"])[mm].mean())}
                d_ = 1000 * (LL["full_ms_layer"] - LL["chosen"])[mm]
                r_["chosen_minus_previous"] = float(d_.mean())
                r_["chosen_minus_previous_se"] = float(d_.std(ddof=1) / np.sqrt(mm.sum()))
                r_["chosen_acc"] = float(((R["chosen"]["g_p"][mm] > .5) == (y[mm] == 1)).mean())
                r_["draft_only_acc"] = float(((z["g_p0"][mm] > .5) == (y[mm] == 1)).mean())
                out[lab][f"{wn} {nmap.get(int(gv), str(gv))}"] = r_
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"queries": queries, "score": score}[sys.argv[1]]()
