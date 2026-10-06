"""
P3 extensions, ban redo, step 1: diagnose the one-step ban harness
(p3_x_draft.py sim) by tracing one-trick opponents through it.

Checks, on the 5,000 V2 lobbies of the drafter runs (all ten players with 50+
earlier games):
  A. Who are the one-tricks: top-hero share of games through the previous day.
  B. Does a ban remove the main? Forced-ban rollouts through the harness's own
     functions (p3_dr_drafter._policy_sample / _apply) for one-trick lobbies:
     share of final drafts in which the one-trick plays the banned main.
  C. How likely is the one-trick to pick his main when it is available, under
     the harness's opponent model (GD policy restricted to the pool) vs the
     imitation model vs his own history? Measured at his real pick state.
  D. Replacement scoring: s(main) vs s of the replacement he gets (expected
     under the harness's GD policy and under imitation), and the off-pool /
     low-experience part of that.
  E. Pools: share of slots whose pool contains the game's own hero only
     because it was added (no earlier games on it); pool sizes.
  F. Shrinkage: one-trick main cells, posterior mean vs raw precision-weighted
     mean residual (after the experience offset), by games on the main.
  G. Information timing in decide_ban.

Usage (from training/): python3 personalization/p3_x_ban_diag.py
Output: results/p3_x_ban_diag.json
"""
import os
import sys
import gzip
import json
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from p3_heroes import NUM_HEROES
import p3_hs_core as C

RUNS = os.path.join(C.CACHE, "dr_runs.pkl.gz")


def main():
    t0 = time.time()
    import torch
    torch.set_num_threads(4)
    import p3_dr_core as D
    import p3_dr_imitation as I
    import p3_dr_drafter as DR
    rng = np.random.RandomState(0)
    d = C.load_slots()
    meta = C.hero_meta(d["hero_names"])
    L = D.load_lobbies()
    T = D.load_personal()
    z = pickle.load(gzip.open(RUNS))
    gis = [l["gi"] for l in z["lobbies"]]
    out = {"lobbies": len(gis)}
    # ---------------- A/E: slots, shares, pools
    rows, main, share, n_main, npl, pool_sz, own_added, gi_of, team_of, step_of = ([] for _ in range(10))
    for gi in gis:
        for k, (h, ty, tm, row) in enumerate(L["steps"][gi]):
            if ty != 1:
                continue
            p = T["pos"][int(row)]
            n = T["n"][p]
            rows.append(int(row))
            main.append(int(np.argmax(n)))
            share.append(n.max() / max(n.sum(), 1))
            n_main.append(n.max())
            npl.append(n.sum())
            pool_sz.append(int(T["pool"][p].sum()))
            own_added.append(bool(n[h] == 0))
            gi_of.append(gi)
            team_of.append(tm)
            step_of.append(k)
    rows, main, share = np.array(rows), np.array(main), np.array(share)
    n_main, npl, pool_sz = np.array(n_main), np.array(npl), np.array(pool_sz)
    own_added = np.array(own_added)
    ot = share >= 0.5
    real_hero = d["hero"][rows]
    out["A_slots"] = {"slots": int(len(rows)), "one_trick_share_ge_50pct": float(ot.mean()),
                      "specialist_25_50": float(((share >= 0.25) & (share < 0.5)).mean()),
                      "lobbies_with_a_one_trick": float(np.mean([ot[np.array(gi_of) == g].any() for g in gis[:2000]])),
                      "one_tricks_playing_main_in_this_game": float((real_hero[ot] == main[ot]).mean()),
                      "specialists_playing_main": float((real_hero[(share >= .25) & (share < .5)] ==
                                                         main[(share >= .25) & (share < .5)]).mean())}
    out["E_pools"] = {"median_pool_size": float(np.median(pool_sz)),
                      "median_pool_size_one_tricks": float(np.median(pool_sz[ot])),
                      "share_slots_own_hero_added_to_pool (no earlier games)": float(own_added.mean()),
                      "same, one-tricks": float(own_added[ot].mean())}
    print(json.dumps(out), flush=True)

    # ---------------- C/D: pick probabilities and replacement at the real pick state
    gd = D.GDPolicy(d["hero_names"])
    iw = np.load(I.OUT)["w"]
    sel = np.flatnonzero(ot)
    sel = sel[rng.permutation(len(sel))[:4000]]
    rec = I.recency_features(d, rows[sel])
    items = []
    for j, i in enumerate(sel):
        gi = gi_of[i]
        st = L["steps"][gi]
        mo, to = D.one_hots(str(L["map"][gi]), str(L["tier"][gi]))
        t0_ = np.zeros(NUM_HEROES, np.float32)
        t1_ = np.zeros(NUM_HEROES, np.float32)
        bans = np.zeros(NUM_HEROES, np.float32)
        for k, (h, ty, tm, row) in enumerate(st):
            if k == step_of[i]:
                break
            if ty == 1:
                (t0_ if tm == 0 else t1_)[h] = 1
            else:
                bans[h] = 1
        items.append((t0_, t1_, bans, mo, to, step_of[i], (t0_ + t1_ + bans) == 0))
    res = {"p_main_gd_pool": [], "p_main_imit": [], "p_main_imit_no_own_added": [], "main_avail": [],
           "s_main": [], "s_repl_gd": [], "s_repl_imit": [], "mu_main": [], "mu_repl_imit": [],
           "p_main_history": []}
    for s0 in range(0, len(items), 2048):
        ch = items[s0:s0 + 2048]
        idx = sel[s0:s0 + 2048]
        avail = np.array([c[6] for c in ch])
        lp = gd.logprobs(np.array([c[0] for c in ch]), np.array([c[1] for c in ch]), np.array([c[2] for c in ch]),
                         np.array([c[3] for c in ch]), np.array([c[4] for c in ch]),
                         np.array([c[5] for c in ch], np.float32), np.ones(len(ch), np.float32), avail)
        for j, (i, l_) in enumerate(zip(idx, lp)):
            p = T["pos"][int(rows[i])]
            pool = T["pool"][p]
            n = T["n"][p]
            m = main[i]
            av = avail[j]
            res["main_avail"].append(bool(av[m]))
            if not av[m]:
                continue
            # harness opponent: GD restricted to pool & available
            a1 = av & pool
            pg = np.where(a1, np.exp(l_), 0)
            pg /= pg.sum()
            # imitation (as in the drafter's "imit" opponent), avail restricted
            jj = s0 + j
            fake = {"row": np.array([rows[i]]), "recpos": np.array([jj]), "lp": l_[None].astype(np.float32)}
            Xf = I.feature_tensor(fake, T, rec, meta)[0]
            u = (Xf * iw).sum(-1)
            u = np.where(av, u, -1e9)
            pi = np.exp(u - u.max())
            pi /= pi.sum()
            s = T["s"][p]
            res["p_main_gd_pool"].append(pg[m])
            res["p_main_imit"].append(pi[m])
            res["p_main_history"].append(n[m] / n.sum())
            res["s_main"].append(s[m])
            q = pg.copy()
            q[m] = 0
            if q.sum() <= 0:  # pool exhausted: the harness falls back to any free hero
                q = np.where(av, np.exp(l_), 0)
                q[m] = 0
            res["s_repl_gd"].append((q * s).sum() / q.sum())
            q = pi.copy()
            q[m] = 0
            q /= q.sum()
            res["s_repl_imit"].append((q * s).sum())
            nb_ = len(C.NPH_EDGES)
            bp = np.searchsorted(C.NP_EDGES, n.sum(), side="right") - 1
            bh = np.searchsorted(C.NPH_EDGES, n, side="right") - 1
            table = np.load(os.path.join(C.CACHE, "hs_kernels.npz"))["experience_table"]
            mu = table[bp * nb_ + bh]
            res["mu_main"].append(mu[m])
            res["mu_repl_imit"].append((q * mu).sum())
    R = {k: np.array(v, float) for k, v in res.items()}
    out["C_pick_probability_of_main_when_available"] = {
        "one_trick_slots": int(len(R["p_main_imit"])),
        "main_available_at_their_pick": float(R["main_avail"].mean()),
        "harness GD-on-pool": float(R["p_main_gd_pool"].mean()),
        "imitation model": float(R["p_main_imit"].mean()),
        "own history share": float(R["p_main_history"].mean())}
    out["D_replacement"] = {
        "s_main_pp": float(100 * R["s_main"].mean()),
        "s_replacement_under_GD_pp": float(100 * R["s_repl_gd"].mean()),
        "s_replacement_under_imitation_pp": float(100 * R["s_repl_imit"].mean()),
        "of which experience offset: main / imitation replacement (pp)": [float(100 * R["mu_main"].mean()),
                                                                           float(100 * R["mu_repl_imit"].mean())],
        "loss_if_main_banned_given_he_would_pick_it_pp": float(100 * (R["s_main"] - R["s_repl_imit"]).mean()),
        "expected_personal_loss_from_ban_harness_pp": float(100 * (R["p_main_gd_pool"] * (R["s_main"] - R["s_repl_gd"])).mean()),
        "expected_personal_loss_from_ban_imitation_pp": float(100 * (R["p_main_imit"] * (R["s_main"] - R["s_repl_imit"])).mean())}
    print(json.dumps(out["C_pick_probability_of_main_when_available"]), json.dumps(out["D_replacement"]), flush=True)

    # ---------------- B: does a forced ban remove the main? (harness functions)
    DR._init({"hero_names": d["hero_names"], "b": D.combiner(), "iw0": 1.0})
    from p3_x_draft import ZeroDict
    w = np.load(C.WP)
    bidx_all = w["build_idx"][np.argsort(w["replay_ids"])]
    ip = ZeroDict()
    played_main, n_tr, p_main_nob = 0, 0, []
    ots = sel[:150]
    for i in ots:
        gi = gi_of[i]
        lob = DR.lobby_payload(L, T, gi, ip, 1 - team_of[i], bidx_all)
        m = main[i]
        for ban in (True, False):
            rr = np.random.RandomState(int(i))
            state = {"t0": {}, "t1": {}, "bans": set(), "taken": np.zeros(NUM_HEROES, bool)}
            if ban:
                state["bans"].add(m)
                state["taken"][m] = True
            states = DR._rollout([state] * 8, 0, lob, rr, "gd", 1 - team_of[i])
            got = [(s["t0"] if team_of[i] == 0 else s["t1"]).get(int(rows[i])) == m for s in states]
            if ban:
                played_main += sum(got)
                n_tr += len(got)
            else:
                p_main_nob.append(np.mean(got))
    out["B_forced_ban"] = {"rollouts": n_tr, "one_trick_plays_banned_main": played_main,
                           "one_trick_plays_main_when_not_banned (harness rollouts)": float(np.mean(p_main_nob))}
    print(out["B_forced_ban"], flush=True)

    # ---------------- F: shrinkage of one-trick mains (V2 start, static state)
    kz = np.load(os.path.join(C.CACHE, "hs_kernels.npz"))
    from p3_hs_fit import KERNELS
    bases = {k[6:]: kz[k] for k in kz.files if k.startswith("basis_")}
    K = C.kernel_from([bases[b] for b in KERNELS["+CF rank 2"]], kz["theta_+CF rank 2"])
    P = np.load(os.path.join(C.CACHE, "x_predall.npz"))
    post = ~d["in_sample"]
    _, fr = np.unique(d["g"][post], return_index=True)
    med = np.median(d["day"][post][fr])
    hist = d["day"] < med
    S, Pm, N, _, _ = C.static_state(d, hist, d["n_players"], NUM_HEROES, P["r_adj"].astype(np.float64))
    tot = N.sum(1)
    mainh = N.argmax(1)
    shr = N.max(1) / np.maximum(tot, 1)
    pl = np.flatnonzero((shr >= 0.5) & (tot >= 50))
    mm, vv = C.gp_query(np.ascontiguousarray(S[pl]), np.ascontiguousarray(Pm[pl]), mainh[pl], K)
    raw = S[pl, mainh[pl]] / Pm[pl, mainh[pl]]
    nm = N[pl, mainh[pl]]
    # overall level of the player on his other heroes
    oth_S = S[pl].sum(1) - S[pl, mainh[pl]]
    oth_P = Pm[pl].sum(1) - Pm[pl, mainh[pl]]
    F = {}
    for a, b in ((50, 100), (100, 300), (300, 1000), (1000, 10 ** 6)):
        k_ = (nm >= a) & (nm < b)
        if k_.sum() < 30:
            continue
        noise = np.mean(1 / Pm[pl[k_], mainh[pl[k_]]])
        F[f"{a}-{b} games on main"] = {
            "players": int(k_.sum()), "raw_mean_pp": float(100 * raw[k_].mean()),
            "posterior_mean_pp": float(100 * mm[k_].mean()),
            "slope_posterior_on_raw": float(np.cov(mm[k_], raw[k_])[0, 1] / raw[k_].var()),
            "expected_slope_if_prior_only_shrinks (var_true / var_raw)": float(
                max(raw[k_].var() - noise, 0) / raw[k_].var()),
            "raw_sd_pp": float(100 * raw[k_].std()), "noise_sd_pp": float(100 * np.sqrt(noise)),
            "other_heroes_raw_pp": float(100 * np.mean(oth_S[k_] / np.maximum(oth_P[k_], 1e-9)))}
    out["F_shrinkage_one_trick_mains"] = F
    print(json.dumps(F), flush=True)
    out["G_information_timing"] = (
        "decide_ban knows all ten identities, pools and personal terms at every ban step (available "
        "in the real lobby). At the second ban phase, opponents who already picked are still in the "
        "candidate list via their pools; their pick is fixed in the state, so it cannot change the value. "
        "Opponents' pools include the hero they really played in this game (audit leak).")
    with open(os.path.join(C.RESULTS, "p3_x_ban_diag.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
