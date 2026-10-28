"""
P3 tournament: real-game backtests (realized outcomes, OOT 2026-05-23 to 09-11).

(a) agree   Natural agreement. At every real pick state of sampled OOT lobbies
            (all ten players with 20+ earlier games), the current player-aware
            search drafter (aMCTS-cur) and its player-blind twin (pMCTS-cur)
            each recommend a hero for the picking team (decide-only search,
            same seed). A real pick "matches" if it equals the recommendation.
            Analysis: does the picking team win more when it matched? Logistic
            and linear models of the team outcome on match indicators, with
            the draft-only WP (paper-1 leak-free WP of the real final draft)
            and the MMR difference as controls; then adding the skill model's
            own predicted gap (the remainder is what the drafter adds beyond
            its value function). Pick-level rows, SEs clustered two-way by
            player and by game.
(b) forced  Forced switches. For every pick slot of every OOT lobby, the
            player's best hero (highest as-of estimate among heroes with 30+
            earlier games) and main (most played, 30+) are "taken" if banned
            before his pick (by either team), picked earlier by a teammate, or
            picked earlier by an opponent. Within player: realized r = y - WP_pop
            in taken vs other games, against the model's predicted r (its
            as-of estimate of the hero he actually played), weighted
            1/(1/n_taken + 1/n_other), player-cluster bootstrap.

Run (3090, run_rcf_full env): python personalization/p3t_backtest.py agree --games 20000   (GPU)
                              python personalization/p3t_backtest.py agree_analyze
                              python personalization/p3t_backtest.py forced
"""
import argparse
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
os.makedirs(RES, exist_ok=True)
PICK_STEPS = [4, 5, 6, 7, 8, 11, 12, 13, 14, 15]


def kernel_lobbies(gi):
    """Kernel-labeled tensors for OOT lobbies gi (state cur)."""
    import p3_mcts_core as M
    from shared import MAPS, SKILL_TIERS
    import p3t_systems as SY
    L = TD.load_lobbies()
    z = np.load(os.path.join(TD.OUT, "state_cur.npz"))
    n = len(gi)
    first = L["first_team"][gi].astype(np.int64)
    acts = L["steps"][gi][:, :, 0].astype(np.int64)
    slot_of = np.full((n, 16), -1, np.int32)
    src = np.zeros((n, 10), np.int64)
    for i in range(n):
        cnt, j = [0, 0], 0
        for k in range(16):
            h, ty, tm, r = L["steps"][gi[i]][k]
            if ty == 1:
                kt = tm ^ first[i]
                sl = 5 * kt + cnt[kt]
                cnt[kt] += 1
                slot_of[i, k] = sl
                src[i, sl] = j
                j += 1
    rows = (gi * 10)[:, None] + src
    s = z["s"][rows.ravel()].astype(np.float32).reshape(n, 10, 90)
    off = z["off"][rows.ravel()].reshape(n, 10, 90)
    pool = z["n"][rows.ravel()].reshape(n, 10, 90) > 0
    team_pool = np.stack([pool[:, :5].any(1), pool[:, 5:].any(1)], 1)
    pool_bits = np.stack([np.stack([M.pool_bits(pool[i, q]) for q in range(10)]) for i in range(n)])
    return {"acts": acts, "first": first, "slot_of": slot_of, "s": s, "off": off, "team_pool": team_pool,
            "pool_bits": pool_bits, "map": np.array([MAPS.index(str(L["map"][g])) for g in gi], np.int32),
            "tier": np.array([SKILL_TIERS.index(str(L["tier"][g])) for g in gi], np.int32), "src": src}


def agree(n_games, device=0, vint="cur_bc"):
    import p3t_systems as SY
    import torch
    torch.set_num_threads(2)
    out_p = os.path.join(TD.OUT, f"agree_{vint}_{n_games}.npz" if vint != "cur" else f"agree_{n_games}.npz")
    L = TD.load_lobbies()
    z = np.load(os.path.join(TD.OUT, "state_cur.npz"))
    from shared import MAPS
    rich = (z["n"].astype(np.float64).sum(1) >= 20).reshape(-1, 10).all(1)
    rich &= np.isin(L["map"], MAPS)
    cand = np.flatnonzero(rich)
    rng = np.random.RandomState(31)
    gi = np.sort(rng.choice(cand, min(n_games, len(cand)), replace=False))
    v = SY.vintage(vint)
    part = out_p + ".partial.npz"
    done = dict(np.load(part)) if os.path.exists(part) else {}
    rec = {"aware": done.get("aware", np.full((len(gi), 16), -1, np.int64)),
           "blind": done.get("blind", np.full((len(gi), 16), -1, np.int64))}
    t0 = time.time()
    for c0 in range(0, len(gi), 2000):
        if (rec["aware"][c0:c0 + 2000][:, PICK_STEPS] >= 0).all():
            continue
        g = gi[c0:c0 + 2000]
        K = kernel_lobbies(g)
        for k in PICK_STEPS:
            ctx = {"acts": K["acts"], "k": k, "map": K["map"], "tier": K["tier"], "team_pool": K["team_pool"],
                   "pool_bits": K["pool_bits"], "slot_of": K["slot_of"], "s": K["s"], "off": K["off"],
                   "seed_i": np.arange(c0, c0 + len(g)), "kseed": 555 + k}
            rec["aware"][c0:c0 + len(g), k] = SY.act_mcts(v, dict(ctx), 1, device=device)
            rec["blind"][c0:c0 + len(g), k] = SY.act_mcts(v, dict(ctx), 0, device=device)
        np.savez(part, gi=gi, **rec)
        print(f"  {c0 + len(g)}/{len(gi)} games ({time.time() - t0:.0f}s)", flush=True)
    np.savez(out_p, gi=gi, **rec)
    os.remove(part)
    print(f"agree done ({time.time() - t0:.0f}s)")


def cluster2_se(X, e, g1, g2):
    """Two-way cluster-robust covariance (Cameron-Gelbach-Miller) for OLS."""
    XtX = np.linalg.inv(X.T @ X)

    def meat(g):
        _, inv = np.unique(g, return_inverse=True)
        S = np.zeros((inv.max() + 1, X.shape[1]))
        np.add.at(S, inv, X * e[:, None])
        return S.T @ S
    inter = g1.astype(np.int64) * 10 ** 9 + g2.astype(np.int64)
    M = meat(g1) + meat(g2) - meat(inter)
    V = XtX @ M @ XtX
    w, U = np.linalg.eigh(V)
    V = (U * np.maximum(w, 0)) @ U.T
    return np.sqrt(np.diag(V))


def causal_mmr(d, rows):
    """Causal player MMR (P3_MMR_AT_GAME: latest stamp parsed before the game
    started) for slot rows; NaN where unknown. The slot table's player_mmr is
    the post-game rating (it contains the game's own result). Keyed by (game,
    region-qualified player) without collisions (2026-10-28: the former
    replay_id << 40 | blizz_id overflowed int64 and dropped the region)."""
    import p3_mmr_feats as MF
    key = np.asarray(d["player_keys"])[d["pid"][rows]].astype(np.int64)
    pm, _ = MF.stamps_at(C.CACHE, d["replay_id"][rows].astype(np.int64), key)
    return pm


def agree_analyze(n_games, vint="cur_bc"):
    """Pick-level linear probability models of the picking team's win."""
    import p3_x_common as X
    import p3t_systems as SY
    from shared import HEROES
    z = np.load(os.path.join(TD.OUT, f"agree_{vint}_{n_games}.npz" if vint != "cur" else f"agree_{n_games}.npz"))
    gi = z["gi"]
    L = TD.load_lobbies()
    d = X.load_ext()
    st = {"s": np.load(os.path.join(TD.OUT, "state_cur.npz"))["s"]}
    # draft-only WP of the real final drafts (paper-1 cur WP), team 0 = real team 0
    v = SY.vintage("cur")
    rows = []
    for g in gi:
        t = L["steps"][g]
        rows.append(([HEROES[h] for h, ty, tm, r in t if ty == 1 and tm == 0],
                     [HEROES[h] for h, ty, tm, r in t if ty == 1 and tm == 1], str(L["map"][g]), str(L["tier"][g])))
    wp0 = np.concatenate([v.wp_drafts(rows[i:i + 5000]) for i in range(0, len(rows), 5000)])
    comb = v.spec["combiner"]
    R = {k: [] for k in ("y", "lo", "dmmr", "m_aw", "m_bl", "m_pers", "dS", "pid", "game", "aw_eq_bl")}
    cm = causal_mmr(d, L["slot_rows"][gi].ravel()).reshape(len(gi), 10)
    for i, g in enumerate(gi):
        steps = L["steps"][g]
        prow = L["slot_rows"][g]
        mmr = cm[i]
        team = steps[:, 2][steps[:, 1] == 1]
        y0 = 1.0 if L["winner"][g] == 0 else 0.0
        mm = [np.nanmean(np.where(team == t, mmr, np.nan)) for t in (0, 1)]
        # skill model's predicted gap of the real draft (drafter's own state, real assignment)
        j = 0
        sp = np.zeros(10)
        for k in range(16):
            if steps[k][1] == 1:
                sp[j] = st["s"][g * 10 + j][steps[k][0]]
                j += 1
        dS0 = sp[team == 0].sum() - sp[team == 1].sum()
        j = 0
        for k in range(16):
            h, ty, tm, r = steps[k]
            if ty != 1:
                continue
            sg = 1 if tm == 0 else -1
            p0 = wp0[i] if tm == 0 else 1 - wp0[i]
            R["y"].append(y0 if tm == 0 else 1 - y0)
            R["lo"].append(np.log(p0 / (1 - p0)))
            R["dmmr"].append(sg * (mm[0] - mm[1]) / 100.0)
            a_, b_ = z["aware"][i, k], z["blind"][i, k]
            R["m_aw"].append(float(h == a_))
            R["m_bl"].append(float(h == b_))
            R["m_pers"].append(float(h == a_ and a_ != b_))
            R["aw_eq_bl"].append(float(a_ == b_))
            R["dS"].append(sg * dS0)
            R["pid"].append(int(d["pid"][prow[j]]))
            R["game"].append(int(g))
            j += 1
    R = {k: np.array(v, float) for k, v in R.items()}
    ok = np.isfinite(R["dmmr"]) & np.isfinite(R["lo"])
    R = {k: v[ok] for k, v in R.items()}
    out = {"games": int(len(gi)), "picks": int(len(R["y"])),
           "match_rate_aware": float(R["m_aw"].mean()), "match_rate_blind": float(R["m_bl"].mean()),
           "aware_equals_blind": float(R["aw_eq_bl"].mean()), "match_rate_aware_only": float(R["m_pers"].mean())}
    models = {
        "raw: matched aware": ["m_aw"],
        "matched aware + WP + MMR": ["m_aw", "lo", "dmmr"],
        "matched aware, matched blind + WP + MMR": ["m_aw", "m_bl", "lo", "dmmr"],
        "matched aware only (aware != blind) + WP + MMR": ["m_pers", "m_bl", "lo", "dmmr"],
        "matched aware + WP + MMR + predicted skill gap": ["m_aw", "lo", "dmmr", "dS"],
        "matched aware only + matched blind + WP + MMR + predicted gap": ["m_pers", "m_bl", "lo", "dmmr", "dS"],
    }
    out["models"] = {}
    for nm, cols in models.items():
        Xm = np.column_stack([np.ones(len(R["y"]))] + [R[c] for c in cols])
        beta = np.linalg.lstsq(Xm, R["y"], rcond=None)[0]
        e = R["y"] - Xm @ beta
        se = cluster2_se(Xm, e, R["pid"].astype(np.int64), R["game"].astype(np.int64))
        out["models"][nm] = {c: [round(100 * float(b), 2), round(100 * float(s), 2)]
                             for c, b, s in zip(["const"] + cols, beta, se)}
    out["note"] = "linear probability model of the picking team's win, coefficients in pp (SE two-way clustered by player and game); lo and dS per unit"
    out["drafter"] = {"cur": "aMCTS-cur vs pMCTS-cur (F prior)", "cur_bc": "aMCTS-bc vs pMCTS-bc (behavioral prior)"}.get(vint, vint)
    out["mmr"] = "causal player MMR (mmr_at_game.npz player_mmr_causal), team mean over players with a known value"
    json.dump(out, open(os.path.join(RES, f"backtest_agree_{vint}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def forced():
    import p3_x_common as X
    L = TD.load_lobbies()
    d = X.load_ext()
    z = np.load(os.path.join(TD.OUT, "state_cur.npz"))
    s = z["s"]
    n = z["n"]
    from shared import MAPS
    G = len(L["replay_id"])
    okmap = np.isin(L["map"], MAPS)
    recs = []
    for g in range(G):
        if not okmap[g]:
            continue
        steps = L["steps"][g]
        prow = L["slot_rows"][g]
        j = 0
        for k in range(16):
            h, ty, tm, r = steps[k]
            if ty != 1:
                continue
            row = g * 10 + j
            nn = n[row].astype(np.int64)
            ss = s[row].astype(np.float32)
            elig = nn >= 30
            if elig.any():
                best = int(np.flatnonzero(elig)[np.argmax(ss[elig])])
                main = int(np.argmax(np.where(elig, nn, -1)))
                prev = steps[:k]
                info = []
                for x in (best, main):
                    ban = any(pp[0] == x and pp[1] == 0 for pp in prev)
                    mate = any(pp[0] == x and pp[1] == 1 and pp[2] == tm for pp in prev)
                    enemy = any(pp[0] == x and pp[1] == 1 and pp[2] != tm for pp in prev)
                    info += [ban, mate, enemy]
                recs.append((int(d["pid"][prow[j]]), float(d["r"][prow[j]]), float(ss[h]), float(ss[best]),
                             float(ss[main]), int(h == best), int(h == main), *info, int(k)))
            j += 1
    A = np.array(recs, float)
    pid = A[:, 0].astype(np.int64)
    r, pred, s_best, s_main = A[:, 1], A[:, 2], A[:, 3], A[:, 4]
    names = ["best_banned", "best_by_teammate", "best_by_enemy", "main_banned", "main_by_teammate", "main_by_enemy"]
    flags = {nm: A[:, 7 + i].astype(bool) for i, nm in enumerate(names)}
    flags["best_taken_any"] = flags["best_banned"] | flags["best_by_teammate"] | flags["best_by_enemy"]
    flags["main_taken_any"] = flags["main_banned"] | flags["main_by_teammate"] | flags["main_by_enemy"]
    out = {"slots": int(len(A)), "players": int(len(np.unique(pid))),
           "share_playing_best": float(A[:, 5].mean()), "share_playing_main": float(A[:, 6].mean())}

    def within(flag, val):
        u, inv = np.unique(pid, return_inverse=True)
        n1 = np.bincount(inv, weights=flag)
        n0 = np.bincount(inv, weights=~flag)
        m1 = np.bincount(inv, weights=val * flag) / np.maximum(n1, 1)
        m0 = np.bincount(inv, weights=val * ~flag) / np.maximum(n0, 1)
        okp = (n1 > 0) & (n0 > 0)
        w = np.where(okp, 1 / (1 / np.maximum(n1, 1) + 1 / np.maximum(n0, 1)), 0)
        return okp, w, m1 - m0

    rng = np.random.default_rng(0)
    for nm, f in flags.items():
        okp, w, dr = within(f, r)
        _, _, dp = within(f, pred)
        sb = s_best if nm.startswith("best") else s_main
        ww = w[okp]
        est = lambda x: float((ww * x[okp]).sum() / ww.sum())
        bs_r, bs_p = [], []
        idx = np.flatnonzero(okp)
        for _ in range(300):
            ii = idx[rng.integers(0, len(idx), len(idx))]
            bs_r.append((w[ii] * dr[ii]).sum() / w[ii].sum())
            bs_p.append((w[ii] * (dr[ii] - dp[ii])).sum() / w[ii].sum())
        out[nm] = {"share_of_slots": float(f.mean()), "players": int(okp.sum()),
                   "realized_pp": round(100 * est(dr), 2), "realized_se": round(100 * float(np.std(bs_r)), 2),
                   "predicted_pp": round(100 * est(dp), 2),
                   "realized_minus_predicted_pp": round(100 * est(dr - dp), 2),
                   "realized_minus_predicted_se": round(100 * float(np.std(bs_p)), 2),
                   "model_cost_vs_lost_hero_pp": round(100 * float(np.mean((sb - pred)[f])), 2)}
        print(nm, json.dumps(out[nm]), flush=True)
    json.dump(out, open(os.path.join(RES, "backtest_forced.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["agree", "agree_analyze", "forced"])
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--vintage", default="cur_bc")
    a = ap.parse_args()
    if a.stage == "agree":
        agree(a.games, a.device, a.vintage)
    elif a.stage == "agree_analyze":
        agree_analyze(a.games, a.vintage)
    else:
        forced()


if __name__ == "__main__":
    main()
