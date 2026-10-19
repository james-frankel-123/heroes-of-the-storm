"""
P3 tournament sanity gates (reconciliation with paper 1).

bench    Paper 1's MCTS benchmark protocol through this tournament's code path:
         1,000 configurations (mid tier, random map and first-pick side), the
         player-blind F search (pMCTS-cur, seeds pooled) drafts one side against
         GD sampled from the 5-model pool, no pools, no structural mask, 200
         sims, root temperature 1 (visit-proportional sampling), as in paper 1's
         Table (F: gN .662, RN .641, QM26 .669). Also the same with this
         tournament's settings (400 sims, argmax root, mask) to see what moves.
realcal  Do the judges predict REAL outcomes of real drafts? Every real draft of
         the tournament-window lobbies (60,500 games, 2026-05-23 to 06-30) is
         scored by CONS and by J (consensus + the judge's personal term with the
         same best assignment used for drafted teams); log loss, accuracy,
         calibration slope and AUC against the real winner. Also on the 300
         tournament lobbies alone.
pools    How often real players play a hero outside their history pool, and how
         large the team pools are (does the pool rule handicap the drafters
         against real drafts?).
Output: $P3_RESULTS/p3t/sanity_<stage>.json
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


def bench(n=1000, device=0):
    import torch
    import p3t_systems as SY
    import p3_mcts_core as M
    import p3t_judge as JG
    from shared import HEROES, MAPS, SKILL_TIERS, is_degenerate
    torch.set_num_threads(2)
    rng = np.random.RandomState(1234)
    mp = rng.randint(len(MAPS), size=n).astype(np.int32)
    tr = np.full(n, 1, np.int32)
    side = rng.randint(2, size=n)                      # the searching side's kernel team
    v = SY.vintage("cur")
    out = {}
    for label, sims, temp, mask in (("paper1_protocol", 200, 1.0, False), ("tournament_settings", 400, 0.0, True)):
        acts = np.full((n, 16), -1, np.int64)
        team_pool = np.ones((n, 2, 90), bool)
        pool_bits = np.stack([np.stack([M.pool_bits(np.ones(90, bool)) for _ in range(10)]) for _ in range(n)])
        slot_of = np.full((n, 16), -1, np.int32)
        cnt = [0, 0]
        for k in range(16):
            if SY.IS_PICK[k]:
                t = SY.DRAFT_TEAM[k]
                slot_of[:, k] = 5 * t + cnt[t]
                cnt[t] += 1
        SY.SIMS = sims
        for k in range(16):
            t = SY.DRAFT_TEAM[k]
            ours = side == t
            if ours.any():
                ii = np.flatnonzero(ours)
                ctx = {"acts": acts[ii], "k": k, "map": mp[ii], "tier": tr[ii], "team_pool": team_pool[ii],
                       "pool_bits": pool_bits[ii], "slot_of": slot_of[ii], "s": np.zeros((len(ii), 10, 90), np.float32),
                       "off": np.zeros((len(ii), 10, 90), np.float32), "seed_i": ii.astype(np.int64), "kseed": 31 + k}
                acts[ii, k] = _search_pick(SY, M, v, ctx, sims, temp, mask, rng, device)
            if (~ours).any():
                ii = np.flatnonzero(~ours)
                X = SY.features(acts[ii], k, mp[ii], tr[ii], t)
                m = SY.legal_mask(acts[ii], k, team_pool[ii])
                p = SY.gd_probs(v, X[:, :289], m)
                u = rng.rand(len(ii), 1)
                acts[ii, k] = np.minimum((np.cumsum(p, 1) < u).sum(1), 89)
                bad = ~m[np.arange(len(ii)), acts[ii, k]]
                acts[ii[bad], k] = p[bad].argmax(1)
        teams = [[acts[i, k] for k in range(16) if SY.IS_PICK[k] and SY.DRAFT_TEAM[k] == t] for i in range(n) for t in (0, 1)]
        rows = [([HEROES[h] for h in teams[2 * i]], [HEROES[h] for h in teams[2 * i + 1]], MAPS[mp[i]], SKILL_TIERS[tr[i]])
                for i in range(n)]
        cons = JG.consensus(rows)
        res = {}
        for key in ("gN", "gN_naive", "RN", "QM2026", "consensus", "gN_raw", "RN_raw", "QM2026_raw"):
            p0 = cons[key]
            res[key] = round(float(np.mean(np.where(side == 0, p0, 1 - p0))), 4)
        deg = np.mean([is_degenerate([HEROES[h] for h in teams[2 * i + side[i]]]) for i in range(n)])
        res["degenerate_pct"] = round(100 * float(deg), 1)
        out[label] = res
        print(label, res, flush=True)
    out["paper1_table_F"] = {"gN": 0.662, "RN": 0.641, "QM2026": 0.669, "degenerate_pct": 4.5}
    json.dump(out, open(os.path.join(RES, "sanity_bench.json"), "w"), indent=1)


def _search_pick(SY, M, v, ctx, sims, temp, mask, rng, device):
    k = ctx["k"]
    B = len(ctx["acts"])
    cfg = np.full((B, M.CFG_LEN), -1, np.int32)
    cfg[:, 0], cfg[:, 1], cfg[:, 2] = ctx["map"], ctx["tier"], SY.DRAFT_TEAM[k]
    cfg[:, 3], cfg[:, 4], cfg[:, 5], cfg[:, 6], cfg[:, 7] = 0, k, 0, 0, 0
    cfg[:, 8:24] = np.where(np.arange(16)[None] < k, ctx["acts"], -1)
    cfg[:, 24:40] = ctx["slot_of"]
    cfg[:, 40:56] = -1
    cfg[:, 56:M.CFG_LEN] = 0
    cfg[:, M.CFG_ASSIGN] = 1
    kmod = SY.kernel_module()
    wf, wo, lut = v.wcfg()
    out = np.empty(B, np.int64)
    seeds = ctx["seed_i"] % v.n_seeds
    coefs = np.array([0, 1, 0, 0, 0], np.float32)
    for i in np.unique(seeds):
        sel = np.flatnonzero(seeds == i)
        pf, po = v.policy_flat(int(i))
        gf, go = v.gd_flat(int(i))
        eng = kmod.PersonalEngine(pf, gf, wf, po, go, wo, lut, max_concurrent=len(sel), device_id=device)
        r = eng.run(np.ascontiguousarray(cfg[sel]), np.zeros((len(sel), 10, 90), np.float32),
                    np.zeros((len(sel), 10, 90), np.float32), np.zeros((len(sel), 10, 90), np.float32),
                    np.ascontiguousarray(ctx["pool_bits"][sel]), coefs, sims, 2.0, int(ctx["kseed"] + 1000 * i),
                    0.0, 0.3, 0.0, 1, search_mode=1)
        del eng
        pol = r[5][:, 0, :]
        legal = SY.legal_mask(ctx["acts"][sel], k, ctx["team_pool"][sel])
        allowed = SY.struct_mask(ctx["acts"][sel], k, legal) if mask else legal
        pol = np.where(allowed, pol, 0.0)
        if temp > 0:
            p = pol / np.maximum(pol.sum(1, keepdims=True), 1e-9)
            u = rng.rand(len(sel), 1)
            pick = np.minimum((np.cumsum(p, 1) < u).sum(1), 89)
            bad = ~allowed[np.arange(len(sel)), pick]
            pick[bad] = pol[bad].argmax(1)
            out[sel] = pick
        else:
            out[sel] = pol.argmax(1)
    return out


def realcal():
    import p3_x_common as XC
    import p3t_judge as JG
    from shared import HEROES
    L = TD.load_lobbies()
    win = np.flatnonzero(L["day"] <= TD.LOBBY_END)
    from shared import MAPS
    win = win[np.isin(L["map"][win], MAPS)]
    jz = np.load(os.path.join(TD.OUT, "state_judge.npz"))
    gpos = {int(g): i for i, g in enumerate(jz["gsel"])}
    js, jo = jz["s"], jz["off"]
    judge = json.load(open(os.path.join(RES, "judge.json")))
    c2, c3 = judge["c2_skill"], judge["c3_offrole"]
    rows, dS, dO = [], [], []
    for g in win:
        st = L["steps"][g]
        t0 = [h for h, ty, tm, r in st if ty == 1 and tm == 0]
        t1 = [h for h, ty, tm, r in st if ty == 1 and tm == 1]
        rows.append(([HEROES[h] for h in t0], [HEROES[h] for h in t1], str(L["map"][g]), str(L["tier"][g])))
        base = gpos[int(g)] * 10
        teamv = st[:, 2][st[:, 1] == 1]
        p0 = [j for j in range(10) if teamv[j] == 0]
        p1 = [j for j in range(10) if teamv[j] == 1]
        S0, O0, _ = JG.best_assign(js[base + np.array(p0)].astype(np.float64), jo[base + np.array(p0)].astype(np.float64), t0, c2, c3)
        S1, O1, _ = JG.best_assign(js[base + np.array(p1)].astype(np.float64), jo[base + np.array(p1)].astype(np.float64), t1, c2, c3)
        dS.append(S0 - S1)
        dO.append(O0 - O1)
    t0_ = time.time()
    cons = JG.consensus(rows)
    pc = np.clip(cons["consensus"], 1e-6, 1 - 1e-6)
    pj = 1 / (1 + np.exp(-(np.log(pc / (1 - pc)) + c2 * np.array(dS) + c3 * np.array(dO))))
    y = (L["winner"][win] == 0).astype(float)
    tl = set(np.load(os.path.join(TD.OUT, "tourney", "lobbies.npz"))["gi"].tolist())
    intour = np.array([int(g) in tl for g in win])

    def stats(p, m):
        p, yy = p[m], y[m]
        lo = np.log(p / (1 - p))
        slope = JG.fit_offset_logistic(np.zeros(len(yy)), lo[:, None], yy)[0][0]
        r = np.empty(len(p))
        r[np.argsort(p, kind="mergesort")] = np.arange(1, len(p) + 1)
        npos = yy.sum()
        auc = (r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(yy) - npos))
        return {"games": int(m.sum()), "log_loss": round(float(XC.logloss(p, yy).mean()), 5),
                "acc": round(float(((p > .5) == (yy == 1)).mean()), 4), "slope": round(float(slope), 3),
                "auc": round(float(auc), 4)}
    out = {}
    for nm, m in (("all_window", np.ones(len(win), bool)), ("tournament_300", intour)):
        out[nm] = {"CONS": stats(pc, m), "J": stats(pj, m), "coin_log_loss": round(float(np.log(2)), 5)}
    json.dump(out, open(os.path.join(RES, "sanity_realcal.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def pools():
    L = TD.load_lobbies()
    z = np.load(os.path.join(TD.OUT, "state_cur.npz"))
    n = z["n"]
    win = np.flatnonzero(L["day"] <= TD.LOBBY_END)
    tl = np.load(os.path.join(TD.OUT, "tourney", "lobbies.npz"))["gi"]
    out = {}
    for nm, gs in (("window", win), ("tournament_300", tl)):
        own_new, team_out, pool_size = [], [], []
        for g in gs:
            st = L["steps"][g]
            hs = st[:, 0][st[:, 1] == 1]
            tm = st[:, 2][st[:, 1] == 1]
            nn = n[g * 10:(g + 1) * 10]
            for j in range(10):
                own_new.append(nn[j, hs[j]] == 0)
            for t in (0, 1):
                tp = (nn[tm == t] > 0).any(0)
                pool_size.append(int(tp.sum()))
                for j in np.flatnonzero(tm == t):
                    team_out.append(not tp[hs[j]])
        out[nm] = {"real_picks_new_to_the_player_pct": round(100 * float(np.mean(own_new)), 2),
                   "real_picks_outside_the_team_pool_pct": round(100 * float(np.mean(team_out)), 3),
                   "team_pool_median": float(np.median(pool_size)), "team_pool_p10": float(np.percentile(pool_size, 10))}
    json.dump(out, open(os.path.join(RES, "sanity_pools.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["bench", "realcal", "pools"])
    a = ap.parse_args()
    {"bench": bench, "realcal": realcal, "pools": pools}[a.stage]()


if __name__ == "__main__":
    main()
