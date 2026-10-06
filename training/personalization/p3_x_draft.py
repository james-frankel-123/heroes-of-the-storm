"""
P3 extensions, task 5: alternatives in personalized drafting, on the
one-step harness of p3_dr_drafter.py (imported, not modified; the MCTS lane
in cuda_personal/ is not touched). Same value model:

  V = sigmoid(b0 + b1 logit WP_pop + b2 (S_0 - S_1) + b3 (O_0 - O_1))

and the same held-out V2 lobbies (all ten players with 50+ earlier games):
the 1,000 "full" lobbies and the realized-check lobbies stored in
cache/dr_runs.pkl.gz.

Subcommands
  sim     (a) personalized bans and (b) teammates-only drafting, simulated.
          (a) The controlled team's three bans are chosen by the personalized
              value V (which knows the opponents' personal strengths) or by
              WP_pop, or the real bans are kept. Ban candidates: top 10 by
              the GD model's ban probability plus top 10 by the strongest
              opponent's personal term. 8 rollouts per candidate. All picks
              for both teams come from the pool-restricted GD policy, so only
              the bans differ. 4 trajectories per lobby and mode.
          (b) The controlled team drafts with full information, with
              teammates-only information (opponents' identities unknown: in
              its rollouts opponents pick from unrestricted GD, and its value
              sets the opponents' S and O to 0), or by WP_pop. The opponent
              really is the identified team (pool-restricted GD) and final
              drafts are scored with the full-information V.
          Realized checks at the real states of the realized lobbies: each
          real ban (and each real pick, teammates-only ranking) is placed in
          the drafters' rankings; team residual y - WP_pop by agreement.
  assign  (c) role assignment given the picks: for each real team, the
          player-to-hero mapping of its five heroes that maximizes
          b2 S + b3 O vs the real one; realized check of the assignment
          component (real S minus the mean over all 120 mappings).
  combine (d) imitation + outcome: combined score V + lambda * imitation
          log-probability at the real states; lambda chosen on one half of
          the lobbies (realized agreement slope), reported on the other.
  chem    (e) premade chemistry: pair-specific over-performance of premade
          pairs beyond the skill model (split-half), familiarity, and "joint
          comfort" (the pair on a hero pair it has played together before);
          game-level V1 -> V2 test of adding them to the value function.

Run (from training/):
  OMP_NUM_THREADS=1 nice -n 19 taskset -c 48-63 python3 personalization/p3_x_draft.py sim --procs 4
  python3 personalization/p3_x_draft.py assign | combine | chem | analyze
Outputs: cache/x_draft_sim.pkl.gz, results/p3_x_draft.json
"""
import os
import sys
import gzip
import json
import time
import pickle
import argparse

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from p3_heroes import NUM_HEROES, HKEY
import p3_hs_core as C

RUNS = os.path.join(C.CACHE, "dr_runs.pkl.gz")
SIM = os.path.join(C.CACHE, "x_draft_sim.pkl.gz")
OUTJ = os.path.join(C.RESULTS, "p3_x_draft.json")
R_BAN = 8
M_TRAJ = 4


def save_part(key, val):
    out = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
    out[key] = val
    with open(OUTJ, "w") as f:
        json.dump(out, f, indent=1, default=float)


def ci_mean(x, rng, n=1000):
    x = np.asarray(x, float)
    bs = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


# ------------------------------------------------------------------ sim workers

def _empty():
    return {"t0": {}, "t1": {}, "bans": set(), "taken": np.zeros(NUM_HEROES, bool)}


def _ban(state, hero):
    s = {"t0": dict(state["t0"]), "t1": dict(state["t1"]), "bans": set(state["bans"]),
         "taken": state["taken"].copy()}
    if not s["taken"][hero]:
        s["bans"].add(hero)
        s["taken"][hero] = True
    return s


def _opp_rows(lob, tm):
    return [row for (h, ty, t, row) in lob["steps"] if ty == 1 and t != tm]


def decide_ban(state, k, lob, rng):
    import p3_dr_drafter as DR
    W = DR._W
    h_real, ty, tm, _ = lob["steps"][k]
    free = ~state["taken"]
    t0 = np.zeros((1, NUM_HEROES), np.float32)
    t1 = np.zeros((1, NUM_HEROES), np.float32)
    bans = np.zeros((1, NUM_HEROES), np.float32)
    t0[0, list(state["t0"].values())] = 1
    t1[0, list(state["t1"].values())] = 1
    bans[0, list(state["bans"])] = 1
    lp = W["gd"].logprobs(t0, t1, bans, lob["mo"][None], lob["to"][None], np.array([k], np.float32),
                          np.zeros(1, np.float32), free[None])[0]
    cand_f = np.flatnonzero(free)
    a = cand_f[np.argsort(-lp[cand_f])[:10]]
    strength = np.full(NUM_HEROES, -1.0)
    for row in _opp_rows(lob, tm):
        strength = np.maximum(strength, np.where(lob["pool"][row], lob["s"][row], -1.0))
    b_ = cand_f[np.argsort(-strength[cand_f])[:10]]
    cand = np.union1d(a, b_)
    if free[h_real] and h_real not in cand:
        cand = np.r_[cand, h_real]
    starts = []
    for c in cand:
        starts += [_ban(state, int(c))] * R_BAN
    finals = DR._rollout(starts, k + 1, lob, rng, "gd", tm)
    wp, V = DR._value([(f["t0"], f["t1"]) for f in finals], lob, W["b"])
    if tm == 1:
        wp, V = 1 - wp, 1 - V
    wp = wp.reshape(len(cand), R_BAN).mean(1)
    V = V.reshape(len(cand), R_BAN).mean(1)
    return {"cand": cand, "wp": wp, "V": V, "pop": int(cand[np.argmax(wp)]),
            "pers": int(cand[np.argmax(V)]), "team": tm, "step": k,
            "real": int(h_real), "gd_top": int(a[0])}


def degraded_lob(lob, tm):
    """What a drafter for team tm knows when the opponents are unidentified."""
    ld = dict(lob)
    opp = set(_opp_rows(lob, tm))
    ld["s"] = {r: (np.zeros(NUM_HEROES, np.float32) if r in opp else v) for r, v in lob["s"].items()}
    ld["off"] = {r: (np.zeros(NUM_HEROES) if r in opp else v) for r, v in lob["off"].items()}
    ld["pool"] = {r: (np.ones(NUM_HEROES, bool) if r in opp else v) for r, v in lob["pool"].items()}
    if "team_pool" in lob:
        ld["team_pool"] = {t: (np.ones(NUM_HEROES, bool) if t != tm else v) for t, v in lob["team_pool"].items()}
    return ld


def run_bans(lob, seed, mode):
    """Bans for the controlled team by mode (pers/pop/real); everything else
    behavioral. Separate RNG streams for decisions and for the trajectory."""
    import p3_dr_drafter as DR
    c = lob["ctrl"]
    rng_env = np.random.RandomState(seed)
    rng_dec = np.random.RandomState(seed + 99991)
    state = _empty()
    decs = []
    for k in range(16):
        h, ty, tm, row = lob["steps"][k]
        if ty == 0:
            if tm == c and mode != "real":
                dd = decide_ban(state, k, lob, rng_dec)
                decs.append(dd)
                state = _ban(state, dd[mode])
            else:
                state = _ban(state, h)
        else:
            hero = int(DR._policy_sample([state], k, lob, rng_env, "gd")[0])
            state = DR._apply(state, k, hero, lob)
    wp, V = DR._value([(state["t0"], state["t1"])], lob, DR._W["b"])
    wp, V = float(wp[0]), float(V[0])
    if c == 1:
        wp, V = 1 - wp, 1 - V
    return {"wp": wp, "V": V, "bans": sorted(state["bans"]), "decisions": decs}


def run_picks(lob, seed, mode):
    """Controlled team drafts with mode full/deg/pop; opponent is GD in its
    real pools; real bans replayed. Separate RNG streams."""
    import p3_dr_drafter as DR
    c = lob["ctrl"]
    rng_env = np.random.RandomState(seed)
    rng_dec = np.random.RandomState(seed + 99991)
    ldeg = degraded_lob(lob, c)
    state = _empty()
    decs = []
    for k in range(16):
        h, ty, tm, row = lob["steps"][k]
        if ty == 0:
            state = DR._apply(state, k, h, lob)
            continue
        if tm == c:
            if mode == "deg":
                dd = DR._decide(state, k, ldeg, rng_dec, "gd")
                hero = dd["pers"]
            else:
                dd = DR._decide(state, k, lob, rng_dec, "gd")
                hero = dd["pers" if mode == "full" else "pop"]
            decs.append({"cand": dd["cand"], "V": dd["V"], "wp": dd["wp"], "choice": hero})
        else:
            hero = int(DR._policy_sample([state], k, lob, rng_env, "gd")[0])
        state = DR._apply(state, k, hero, lob)
    wp, V = DR._value([(state["t0"], state["t1"])], lob, DR._W["b"])
    wp, V = float(wp[0]), float(V[0])
    if c == 1:
        wp, V = 1 - wp, 1 - V
    return {"wp": wp, "V": V, "decisions": decs}


def work_full(args):
    lob, seed = args
    out = {"gi": lob["gi"], "ctrl": lob["ctrl"]}
    for j in range(M_TRAJ):
        for m in ("pers", "pop", "real"):
            out[f"ban|{m}|{j}"] = run_bans(lob, seed + 17 * j, m)
    for m in ("full", "deg", "pop"):
        out[f"pick|{m}"] = run_picks(lob, seed, m)
    return out


def work_real(args):
    """Real states: rank the real bans (pers/pop ban values) and the real
    picks under the teammates-only drafter."""
    import p3_dr_drafter as DR
    lob, seed = args
    rng = np.random.RandomState(seed + 3)
    state = _empty()
    bans, picks = [], []
    for k in range(16):
        h, ty, tm, row = lob["steps"][k]
        if ty == 0:
            if not state["taken"][h]:
                dd = decide_ban(state, k, lob, rng)
                bans.append(dd)
            state = _ban(state, h)
        else:
            st = dict(state)
            st["force"] = h
            dd = DR._decide(st, k, degraded_lob(lob, tm), rng, "gd")
            dd["actual"] = int(h)
            picks.append({"cand": dd["cand"], "V": dd["V"], "wp": dd["wp"], "actual": int(h), "team": tm})
            state = DR._apply(state, k, h, lob)
    return {"gi": lob["gi"], "bans": bans, "picks_deg": picks}


class ZeroDict(dict):
    def __missing__(self, key):
        return np.zeros(NUM_HEROES, np.float32)


def sim(procs, n_real):
    t0 = time.time()
    import p3_dr_core as D
    import p3_dr_drafter as DR
    z = pickle.load(gzip.open(RUNS))
    d = C.load_slots()
    L = D.load_lobbies()
    T = D.load_personal()
    w = np.load(C.WP)
    bidx_all = w["build_idx"][np.argsort(w["replay_ids"])]
    b = D.combiner()
    full = [l for l in z["lobbies"] if "b|pers" in l]
    ctrl = {l["gi"]: l["ctrl"] for l in full}
    real_g = [l["gi"] for l in z["lobbies"]][:n_real]
    ip = ZeroDict()
    tasks = [(DR.lobby_payload(L, T, gi, ip, ctrl[gi], bidx_all), int(1000 + gi)) for gi in ctrl]
    rtasks = [(DR.lobby_payload(L, T, gi, ip, 0, bidx_all), int(1000 + gi)) for gi in real_g]
    shared = {"hero_names": d["hero_names"], "b": b, "iw0": 1.0}
    print(f"{len(tasks)} full lobbies, {len(rtasks)} realized lobbies ({time.time() - t0:.0f}s)", flush=True)
    import multiprocessing as mp
    res, rres = [], []
    with mp.get_context("fork").Pool(procs, initializer=DR._init, initargs=(shared,)) as pool:
        for i, r in enumerate(pool.imap_unordered(work_full, tasks, chunksize=2)):
            res.append(r)
            if (i + 1) % 100 == 0:
                print(f"  full {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
        for i, r in enumerate(pool.imap_unordered(work_real, rtasks, chunksize=4)):
            rres.append(r)
            if (i + 1) % 500 == 0:
                print(f"  real {i + 1}/{len(rtasks)} ({time.time() - t0:.0f}s)", flush=True)
    # deterministic order: imap_unordered returns results in completion order
    res.sort(key=lambda r: r["gi"])
    rres.sort(key=lambda r: r["gi"])
    with gzip.open(SIM + ".tmp", "wb") as f:
        pickle.dump({"full": res, "real": rres}, f)
    os.replace(SIM + ".tmp", SIM)
    print(f"sim done {time.time() - t0:.0f}s", flush=True)


def realized_quintiles(A, rng, col, res_col):
    qs = np.percentile(A[:, col], [20, 40, 60, 80])
    b = np.searchsorted(qs, A[:, col])
    top, bot = A[b == 4, res_col], A[b == 0, res_col]
    diffs = [top[rng.randint(0, len(top), len(top))].mean() - bot[rng.randint(0, len(bot), len(bot))].mean()
             for _ in range(1000)]
    return {"quintile_residual_pp": [float(100 * A[b == q, res_col].mean()) for q in range(5)],
            "top_minus_bottom_pp": [float(100 * (top.mean() - bot.mean())), float(100 * np.percentile(diffs, 2.5)),
                                    float(100 * np.percentile(diffs, 97.5))]}


def analyze_sim():
    rng = np.random.RandomState(0)
    import p3_dr_core as D
    z = pickle.load(gzip.open(SIM))
    L = D.load_lobbies()
    w = np.load(C.WP)
    o = np.argsort(w["replay_ids"])
    y_all, wp_all = w["y"][o], w["wp0"][o]
    out = {"n_full": len(z["full"]), "n_real": len(z["real"])}
    # (a) bans: final drafts
    F = z["full"]
    ban = {}
    for a_, b_ in (("pers", "pop"), ("pers", "real"), ("pop", "real")):
        dV = [np.mean([l[f"ban|{a_}|{j}"]["V"] - l[f"ban|{b_}|{j}"]["V"] for j in range(M_TRAJ)]) for l in F]
        dW = [np.mean([l[f"ban|{a_}|{j}"]["wp"] - l[f"ban|{b_}|{j}"]["wp"] for j in range(M_TRAJ)]) for l in F]
        ban[f"{a_} bans - {b_} bans"] = {"dV_pp": ci_mean(100 * np.array(dV), rng),
                                         "dWPpop_pp": ci_mean(100 * np.array(dW), rng)}
    decs = [dd for l in F for j in range(M_TRAJ) for dd in l[f"ban|pers|{j}"]["decisions"]]
    diff = np.array([dd["pers"] != dd["pop"] for dd in decs])
    gainV = np.array([dd["V"][dd["cand"] == dd["pers"]][0] - dd["V"][dd["cand"] == dd["pop"]][0] for dd in decs])
    costW = np.array([dd["wp"][dd["cand"] == dd["pers"]][0] - dd["wp"][dd["cand"] == dd["pop"]][0] for dd in decs])
    ban["decisions"] = {"n": int(len(decs)), "share_differ": float(diff.mean()),
                        "dV_pp_when_differ": ci_mean(100 * gainV[diff], rng),
                        "dWPpop_pp_when_differ": ci_mean(100 * costW[diff], rng),
                        "value_spread_pp_between_best_and_median_ban": float(
                            100 * np.median([dd["V"].max() - np.median(dd["V"]) for dd in decs]))}
    # does the personalized ban target the opponents' strongest heroes?
    out["bans"] = ban
    # realized: agreement of real bans with the pers / pop ban rankings
    recs = []
    for l in z["real"]:
        g = L["g"][l["gi"]]
        for t in (0, 1):
            pp, pq = [], []
            for dd in l["bans"]:
                if dd["team"] != t:
                    continue
                i = np.flatnonzero(dd["cand"] == dd["real"])
                nc = len(dd["cand"])
                if len(i) == 0 or nc < 2:
                    continue
                i = int(i[0])
                pp.append(np.sum(dd["V"] < dd["V"][i]) / (nc - 1))
                pq.append(np.sum(dd["wp"] < dd["wp"][i]) / (nc - 1))
            if not pp:
                continue
            yt = y_all[g] if t == 0 else 1 - y_all[g]
            wt = wp_all[g] if t == 0 else 1 - wp_all[g]
            pdiff = np.mean(pp) - np.mean(pq)
            recs.append((np.mean(pp), np.mean(pq), pdiff, yt - wt))
    A = np.array(recs)
    out["bans_realized"] = {"teams": int(len(A)),
                            "personalized ban agreement": realized_quintiles(A, rng, 0, 3),
                            "population ban agreement": realized_quintiles(A, rng, 1, 3),
                            "personal minus population agreement": realized_quintiles(A, rng, 2, 3)}
    # (b) teammates-only
    pk = {}
    for a_, b_ in (("full", "pop"), ("deg", "pop"), ("full", "deg")):
        dV = np.array([l[f"pick|{a_}"]["V"] - l[f"pick|{b_}"]["V"] for l in F])
        dW = np.array([l[f"pick|{a_}"]["wp"] - l[f"pick|{b_}"]["wp"] for l in F])
        pk[f"{a_} - {b_}"] = {"dV_pp": ci_mean(100 * dV, rng), "dWPpop_pp": ci_mean(100 * dW, rng)}
    pk["share_of_full_value_kept_by_teammates_only"] = float(
        pk["deg - pop"]["dV_pp"][0] / pk["full - pop"]["dV_pp"][0])
    rs = np.array([pk["deg - pop"]["dV_pp"][0] / pk["full - pop"]["dV_pp"][0]])
    # bootstrap the ratio by lobby
    dfull = np.array([l["pick|full"]["V"] - l["pick|pop"]["V"] for l in F])
    ddeg = np.array([l["pick|deg"]["V"] - l["pick|pop"]["V"] for l in F])
    bs = []
    for _ in range(1000):
        i = rng.randint(0, len(F), len(F))
        bs.append(ddeg[i].mean() / dfull[i].mean())
    pk["share_kept_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out["teammates_only"] = pk
    # realized: real picks ranked by the teammates-only drafter
    Dr = pickle.load(gzip.open(RUNS))
    realmap = {l["gi"]: l["real"]["decisions"] for l in Dr["lobbies"]}
    recs = []
    for l in z["real"]:
        g = L["g"][l["gi"]]
        full_dec = realmap[l["gi"]]
        for t in (0, 1):
            pdg, pfl = [], []
            for dd, df in zip(l["picks_deg"], full_dec):
                if dd["team"] != t:
                    continue
                nc = len(dd["cand"])
                if nc < 2:
                    continue
                i = int(np.flatnonzero(dd["cand"] == dd["actual"])[0])
                pdg.append(np.sum(dd["V"] < dd["V"][i]) / (nc - 1))
                j = int(np.flatnonzero(df["cand"] == df["actual"])[0])
                pfl.append(np.sum(df["V"] < df["V"][j]) / (len(df["cand"]) - 1))
            yt = y_all[g] if t == 0 else 1 - y_all[g]
            wt = wp_all[g] if t == 0 else 1 - wp_all[g]
            recs.append((np.mean(pdg), np.mean(pfl), yt - wt))
    A = np.array(recs)
    out["teammates_only_realized"] = {"teams": int(len(A)),
                                      "teammates-only drafter agreement": realized_quintiles(A, rng, 0, 2),
                                      "full-information drafter agreement": realized_quintiles(A, rng, 1, 2)}
    save_part("sim", out)
    print(json.dumps(out, default=float)[:3000])


# ------------------------------------------------------------------ (c) assignment

def assign():
    import itertools
    import p3_dr_core as D
    rng = np.random.RandomState(0)
    d = C.load_slots()
    L = D.load_lobbies()
    T = D.load_personal()
    b = D.combiner()
    post = ~d["in_sample"]
    _, fr = np.unique(d["g"][post], return_index=True)
    med = np.median(d["day"][post][fr])
    w = np.load(C.WP)
    o = np.argsort(w["replay_ids"])
    y_all, wp_all = w["y"][o], w["wp0"][o]
    perms = np.array(list(itertools.permutations(range(5))))
    recs = []
    for gi in range(len(L["replay_id"])):
        st = L["steps"][gi]
        g = L["g"][gi]
        for t in (0, 1):
            rows = [int(s[3]) for s in st if s[1] == 1 and s[2] == t]
            heroes = [int(s[0]) for s in st if s[1] == 1 and s[2] == t]
            if len(rows) != 5:
                continue
            pos = [T["pos"][r] for r in rows]
            est = min(T["n_p"][p] for p in pos)
            M = np.array([[b[2] * T["s"][p][h] + b[3] * T["off"][p][h] for h in heroes] for p in pos])
            vals = M[np.arange(5)[None, :], perms].sum(1)  # player i gets heroes[perm[i]]
            real = M[np.arange(5), np.arange(5)].sum()
            Sm = np.array([[T["s"][p][h] for h in heroes] for p in pos])
            s_real = Sm[np.arange(5), np.arange(5)].sum()
            s_perm_mean = Sm[np.arange(5)[None, :], perms].sum(1).mean()
            yt = y_all[g] if t == 0 else 1 - y_all[g]
            wt = wp_all[g] if t == 0 else 1 - wp_all[g]
            recs.append((L["day"][gi] >= med, est, real, vals.max(), vals.mean(), s_real, s_perm_mean,
                         yt - wt, float(np.isclose(real, vals.max())), (vals > real + 1e-9).sum()))
    A = np.array(recs, float)
    out = {}
    for nm, m in (("all post-cutoff teams", np.ones(len(A), bool)),
                  ("V2 teams, all ten 50+ games", (A[:, 0] == 1) & (A[:, 1] >= 50))):
        a = A[m]
        gap = a[:, 3] - a[:, 2]  # logit units
        rnd = a[:, 4] - a[:, 2]
        # assignment component of the real S, controlling for the heroes chosen
        comp = a[:, 5] - a[:, 6]
        X = np.column_stack([np.ones(len(a)), comp, a[:, 6]])
        beta = np.linalg.lstsq(X, a[:, 7], rcond=None)[0]
        bs = []
        for _ in range(500):
            i = rng.randint(0, len(a), len(a))
            bs.append(np.linalg.lstsq(X[i], a[i, 7], rcond=None)[0][1])
        qs = np.percentile(comp, [20, 40, 60, 80])
        qb = np.searchsorted(qs, comp)
        out[nm] = {"teams": int(len(a)),
                   "real_is_best_mapping": float(a[:, 8].mean()),
                   "mean_rank_of_real_mapping_of_120": float(1 + a[:, 9].mean()),
                   "predicted_gain_best_vs_real_pp": float(100 * 0.25 * gap.mean()),
                   "predicted_gain_real_vs_random_mapping_pp": float(-100 * 0.25 * rnd.mean()),
                   "share_gap_over_2pp": float((0.25 * gap > 0.02).mean()),
                   "realized_slope_of_assignment_component": [float(beta[1]), float(np.percentile(bs, 2.5)),
                                                              float(np.percentile(bs, 97.5))],
                   "model_slope_prob_units": float(0.25 * b[2]),
                   "residual_by_assignment_component_quintile_pp": [float(100 * a[qb == q, 7].mean())
                                                                    for q in range(5)],
                   "assignment_component_sd_pp": float(100 * comp.std())}
    save_part("assign", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------ (d) combined

def combine():
    import p3_dr_core as D
    import p3_dr_imitation as I
    rng = np.random.RandomState(0)
    z = pickle.load(gzip.open(RUNS))
    d = C.load_slots()
    meta = C.hero_meta(d["hero_names"])
    L = D.load_lobbies()
    T = D.load_personal()
    gd = D.GDPolicy(d["hero_names"])
    iw = np.load(I.OUT)["w"]
    w = np.load(C.WP)
    o = np.argsort(w["replay_ids"])
    y_all, wp_all = w["y"][o], w["wp0"][o]
    items, rows_needed = [], set()
    for l in z["lobbies"]:
        st = L["steps"][l["gi"]]
        mo, to = D.one_hots(str(L["map"][l["gi"]]), str(L["tier"][l["gi"]]))
        t0 = np.zeros(NUM_HEROES, np.float32)
        t1 = np.zeros(NUM_HEROES, np.float32)
        bans = np.zeros(NUM_HEROES, np.float32)
        di = 0
        for k, (h, ty, tm, row) in enumerate(st):
            if ty == 1:
                dd = l["real"]["decisions"][di]
                di += 1
                items.append((t0.copy(), t1.copy(), bans.copy(), mo, to, k, (t0 + t1 + bans) == 0, row, h, dd,
                              l["gi"], tm))
                rows_needed.add(int(row))
                (t0 if tm == 0 else t1)[h] = 1
            else:
                bans[h] = 1
    rows_needed = np.array(sorted(rows_needed))
    rec = I.recency_features(d, rows_needed)
    fake = {"row": rows_needed, "recpos": np.arange(len(rows_needed)),
            "lp": np.zeros((len(rows_needed), NUM_HEROES), np.float32)}
    Xp = I.feature_tensor(fake, T, rec, meta)
    ipers = {int(r): (Xp[i, :, 1:] * iw[1:]).sum(-1) for i, r in enumerate(rows_needed)}
    per_pick = []
    for s in range(0, len(items), 4096):
        ch = items[s:s + 4096]
        lp = gd.logprobs(np.array([c[0] for c in ch]), np.array([c[1] for c in ch]), np.array([c[2] for c in ch]),
                         np.array([c[3] for c in ch]), np.array([c[4] for c in ch]),
                         np.array([c[5] for c in ch], np.float32), np.ones(len(ch), np.float32),
                         np.array([c[6] for c in ch]))
        for c, l_ in zip(ch, lp):
            dd = c[9]
            u = iw[0] * l_ + ipers[int(c[7])]
            uc = u[dd["cand"]]
            uc = uc - uc.max()
            logp = uc - np.log(np.exp(uc).sum())  # imitation log-prob within the candidate set
            g = L["g"][c[10]]
            yt = y_all[g] if c[11] == 0 else 1 - y_all[g]
            wt = wp_all[g] if c[11] == 0 else 1 - wp_all[g]
            per_pick.append({"gi": c[10], "team": c[11], "cand": dd["cand"], "V": dd["V"], "wp": dd["wp"],
                             "logp": logp, "actual": int(c[8]), "res": yt - wt})
    lobbies = np.array(sorted({p["gi"] for p in per_pick}))
    half = set(rng.choice(lobbies, len(lobbies) // 2, replace=False))

    def score(lam, sel_half):
        recs = {}
        top_rows = []
        for p in per_pick:
            if (p["gi"] in half) != sel_half:
                continue
            sc = p["V"] + lam * p["logp"]
            nc = len(p["cand"])
            if nc < 2:
                continue
            i = int(np.flatnonzero(p["cand"] == p["actual"])[0])
            key = (p["gi"], p["team"])
            recs.setdefault(key, [[], p["res"]])[0].append(np.sum(sc < sc[i]) / (nc - 1))
            top_rows.append((int(p["cand"][np.argmax(sc)]) == p["actual"], p["res"]))
        A = np.array([(np.mean(v[0]), v[1]) for v in recs.values()])
        X = np.column_stack([np.ones(len(A)), A[:, 0] - A[:, 0].mean()])
        slope = np.linalg.lstsq(X, A[:, 1], rcond=None)[0][1]
        return A, slope, np.array(top_rows, float)

    lams = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0, 1e3]
    tune = {}
    for lam in lams:
        _, sl, tr = score(lam, True)
        tune[str(lam)] = {"slope_pp": float(100 * sl), "top1_match": float(tr[:, 0].mean()),
                          "residual_when_top1_matched_pp": float(100 * tr[tr[:, 0] == 1, 1].mean())}
    best = max(lams, key=lambda l_: tune[str(l_)]["slope_pp"])
    best_m = max(lams, key=lambda l_: tune[str(l_)]["residual_when_top1_matched_pp"])
    # team-level: both percentiles jointly (which signal carries independent outcome information?)
    recs = {}
    for p in per_pick:
        nc = len(p["cand"])
        if nc < 2:
            continue
        i = int(np.flatnonzero(p["cand"] == p["actual"])[0])
        key = (p["gi"], p["team"])
        r_ = recs.setdefault(key, [[], [], [], p["res"]])
        r_[0].append(np.sum(p["V"] < p["V"][i]) / (nc - 1))
        r_[1].append(np.sum(p["logp"] < p["logp"][i]) / (nc - 1))
        pdv = p["V"] - p["wp"]
        r_[2].append(np.sum(pdv < pdv[i]) / (nc - 1))
    A = np.array([(np.mean(a), np.mean(b), np.mean(c), r) for a, b, c, r in recs.values()])
    joint = {}
    for nm, cols in (("V and imitation", [0, 1]), ("personal component (V - WP) and imitation", [2, 1])):
        Xj = np.column_stack([np.ones(len(A))] + [A[:, c] - A[:, c].mean() for c in cols])
        beta = np.linalg.lstsq(Xj, A[:, 3], rcond=None)[0]
        bs = []
        for _ in range(500):
            ii = rng.randint(0, len(A), len(A))
            bs.append(np.linalg.lstsq(Xj[ii], A[ii, 3], rcond=None)[0][1:])
        bs = np.array(bs)
        joint[nm] = {"coef_pp_per_unit": [float(100 * beta[1]), float(100 * beta[2])],
                     "ci_first": [float(100 * np.percentile(bs[:, 0], 2.5)), float(100 * np.percentile(bs[:, 0], 97.5))],
                     "ci_second": [float(100 * np.percentile(bs[:, 1], 2.5)), float(100 * np.percentile(bs[:, 1], 97.5))],
                     "corr_between_agreements": float(np.corrcoef(A[:, cols[0]], A[:, cols[1]])[0, 1])}
    test = {}
    for lam in sorted({0.0, best, best_m, 1e3}):
        A, sl, tr = score(lam, False)
        m = tr[:, 0] == 1
        test[str(lam)] = {"agreement": realized_quintiles(np.column_stack([A[:, 0], A[:, 1]]), rng, 0, 1),
                          "slope_pp": float(100 * sl),
                          "top1_match_rate": float(tr[:, 0].mean()),
                          "residual_when_real_pick_is_top1_pp": ci_mean(100 * tr[m, 1], rng),
                          "residual_otherwise_pp": ci_mean(100 * tr[~m, 1], rng)}
    out = {"picks": len(per_pick), "lambda_grid_tuning_half": tune, "chosen_lambda": best,
           "chosen_lambda_by_matched_residual": best_m, "team_level_joint_regression": joint,
           "test_half": test,
           "note": "lambda 0 = outcome (personalized V) ranking; 1e3 = imitation ranking; V is a probability, "
                   "logp a natural-log imitation probability within the candidate set"}
    save_part("combine", out)
    print(json.dumps(out, indent=1, default=float))


# ------------------------------------------------------------------ (e) chemistry

def chem():
    import p3_x_common as X
    rng = np.random.RandomState(0)
    d = C.load_slots()
    P = np.load(os.path.join(C.CACHE, "x_predall.npz"))
    s = (P["mu"] + P["m_cf2"]).astype(np.float64)
    n = len(s)
    n_games, y, wp0, cnt, gday = X.game_arrays(d)
    t = d["team"].astype(np.int64)
    ts = np.zeros((n_games, 2))
    np.add.at(ts, (d["g"], t), s)
    # team-level residual after the skill model, from each slot's side
    e_team = d["r"] - (ts[d["g"], t] - ts[d["g"], 1 - t])
    # premade pairs: slots of the same game, team and nonzero party
    party = d["party"]
    pm = party != 0
    idx = np.flatnonzero(pm)
    key = d["g"][idx] * 4 + t[idx] * 2
    o = np.lexsort((d["pid"][idx], party[idx], key))
    idx = idx[o]
    gk = d["g"][idx] * 2 + t[idx]
    pk = party[idx]
    pairs = []
    i = 0
    N = len(idx)
    while i < N:
        j = i
        while j < N and gk[j] == gk[i] and pk[j] == pk[i]:
            j += 1
        mem = idx[i:j]
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                pairs.append((mem[a], mem[b], j - i))
        i = j
    pairs = np.array(pairs, np.int64)
    ra, rb, psize = pairs[:, 0], pairs[:, 1], pairs[:, 2]
    pa, pb = d["pid"][ra], d["pid"][rb]
    lo_, hi_ = np.minimum(pa, pb), np.maximum(pa, pb)
    pkey = lo_ * (1 << 22) + hi_
    ev = e_team[ra]
    day = d["day"][ra]
    rid = d["replay_id"][ra]
    out = {"premade_pair_games": int(len(pairs)), "distinct_pairs": int(len(np.unique(pkey)))}
    # familiarity: joint games before this one (exact order)
    oo = np.lexsort((rid, day, pkey))
    kk = pkey[oo]
    st = np.flatnonzero(np.r_[True, kk[1:] != kk[:-1]])
    kpos = np.arange(len(kk)) - np.repeat(st, np.diff(np.r_[st, len(kk)]))
    fam = np.empty(len(kk), np.int64)
    fam[oo] = kpos
    fam_tab = {}
    for a, b in ((0, 0), (1, 4), (5, 19), (20, 49), (50, 199), (200, 10 ** 6)):
        m = (fam >= a) & (fam <= b) & (psize == 2)
        if m.sum() > 200:
            fam_tab[f"{a}-{b}"] = {"pair_games": int(m.sum()), "e_pp": ci_mean(100 * ev[m], rng, 300)}
    out["duo_residual_by_previous_joint_games"] = fam_tab
    # pair-specific chemistry variance (split half, duos with 10+ joint games)
    odd = kpos % 2 == 0
    evo = ev[oo]
    u, inv = np.unique(kk, return_inverse=True)
    S1 = np.bincount(inv, weights=evo * odd)
    S2 = np.bincount(inv, weights=evo * ~odd)
    N1 = np.bincount(inv, weights=odd.astype(float))
    N2 = np.bincount(inv, weights=(~odd).astype(float))
    ok = (N1 + N2 >= 10)
    xa, xb = S1[ok] / N1[ok], S2[ok] / N2[ok]
    wgt = (N1[ok] + N2[ok]) / (1 + (N1[ok] + N2[ok]) / 50)
    cov = float((wgt * xa * xb).sum() / wgt.sum())
    mean_ = float((wgt * (xa + xb) / 2).sum() / wgt.sum())
    bs = []
    for _ in range(300):
        i = rng.randint(0, ok.sum(), ok.sum())
        bs.append((wgt[i] * xa[i] * xb[i]).sum() / wgt[i].sum() - ((wgt[i] * (xa[i] + xb[i]) / 2).sum() / wgt[i].sum()) ** 2)
    # net of each member's own level away from this partner: ind_i = mean e_team
    # over i's slots without j (all of i's slots minus the joint ones)
    npl = int(d["n_players"])
    S_i = np.bincount(d["pid"], weights=e_team, minlength=npl)
    N_i = np.bincount(d["pid"], minlength=npl).astype(float)
    Tt = np.bincount(inv, weights=evo)
    Nt = np.bincount(inv).astype(float)
    pi_ = u // (1 << 22)
    pj_ = u % (1 << 22)
    vbar = float(np.mean(d["v"]))
    ind_i = (S_i[pi_] - Tt) / np.maximum(N_i[pi_] - Nt, 1)
    ind_j = (S_i[pj_] - Tt) / np.maximum(N_i[pj_] - Nt, 1)
    ok_n = ok & (N_i[pi_] - Nt >= 30) & (N_i[pj_] - Nt >= 30)
    corr_n = vbar / (N_i[pi_] - Nt) + vbar / (N_i[pj_] - Nt)
    # time split: first half vs second half of the pair's joint games
    half_t = kpos >= np.repeat(np.diff(np.r_[st, len(kk)]) // 2, np.diff(np.r_[st, len(kk)]))
    T1 = np.bincount(inv, weights=evo * ~half_t)
    T2 = np.bincount(inv, weights=evo * half_t)
    M1 = np.bincount(inv, weights=(~half_t).astype(float))
    M2 = np.bincount(inv, weights=half_t.astype(float))
    net_var = {}
    for lab, A1, B1, A2, B2 in (("alternating games", S1, N1, S2, N2), ("first vs second half in time", T1, M1, T2, M2)):
        a1 = A1[ok_n] / B1[ok_n] - ind_i[ok_n] - ind_j[ok_n]
        a2 = A2[ok_n] / B2[ok_n] - ind_i[ok_n] - ind_j[ok_n]
        ww = (B1[ok_n] + B2[ok_n]) / (1 + (B1[ok_n] + B2[ok_n]) / 50)
        mean_n = (ww * (a1 + a2) / 2).sum() / ww.sum()
        cv = (ww * (a1 * a2 - corr_n[ok_n])).sum() / ww.sum() - mean_n ** 2
        bsn = []
        cn_ok = corr_n[ok_n]
        for _ in range(300):
            i = rng.randint(0, ok_n.sum(), ok_n.sum())
            m_ = (ww[i] * (a1[i] + a2[i]) / 2).sum() / ww[i].sum()
            bsn.append((ww[i] * (a1[i] * a2[i] - cn_ok[i])).sum() / ww[i].sum() - m_ ** 2)
        net_var[lab] = {"pairs": int(ok_n.sum()), "mean_net_pair_effect_pp": float(100 * mean_n),
                        "pair_specific_var_pp2": float(1e4 * cv),
                        "var_ci_pp2": [float(1e4 * np.percentile(bsn, 2.5)), float(1e4 * np.percentile(bsn, 97.5))],
                        "pair_specific_sd_pp": float(100 * np.sqrt(max(cv, 0)))}
    out["pair_chemistry_net_of_members"] = net_var
    out["pair_chemistry"] = {"pairs_10plus_joint_games": int(ok.sum()),
                             "mean_pair_effect_pp": 100 * mean_,
                             "pair_specific_sd_pp": float(100 * np.sqrt(max(cov - mean_ ** 2, 0))),
                             "var_ci_pp2": [float(1e4 * np.percentile(bs, 2.5)), float(1e4 * np.percentile(bs, 97.5))]}
    # joint comfort: this hero pair played together by this premade pair before
    hp = np.minimum(d["hero"][ra], d["hero"][rb]) * HKEY + np.maximum(d["hero"][ra], d["hero"][rb])
    ck = pkey * 16384 + hp
    o3 = np.lexsort((rid, day, ck))
    c3 = ck[o3]
    st3 = np.flatnonzero(np.r_[True, c3[1:] != c3[:-1]])
    jpos = np.arange(len(c3)) - np.repeat(st3, np.diff(np.r_[st3, len(c3)]))
    jc = np.empty(len(c3), np.int64)
    jc[o3] = jpos
    jt = {}
    for a, b in ((0, 0), (1, 2), (3, 9), (10, 10 ** 6)):
        m = (jc >= a) & (jc <= b) & (fam >= 10)
        if m.sum() > 200:
            jt[f"{a}-{b}"] = {"pair_games": int(m.sum()), "e_pp": ci_mean(100 * ev[m], rng, 300)}
    out["pair_residual_by_previous_games_on_this_hero_pair (pairs with 10+ joint games)"] = jt
    # game level: add causal chemistry terms to the value function (V1 fit, V2 test)
    post = ~d["in_sample"]
    _, fr = np.unique(d["g"][post], return_index=True)
    med = np.median(d["day"][post][fr])
    cntp = np.bincount(d["g"][post], minlength=n_games)
    lo = np.log(wp0 / (1 - wp0))
    fit_m = (cntp == 10) & (gday < med)
    test_m = (cntp == 10) & (gday >= med)
    sign = lambda tt: np.where(tt == 0, 1.0, -1.0)
    D0 = X.team_diff(s, d["g"], d["team"], n_games)
    # party size terms (as P3_VALIDITY)
    gid = d["g"] * 2 + t
    pkey2 = gid * (1 << 20) + (party % (1 << 20))
    u2, inv2, c2 = np.unique(np.where(pm, pkey2, -1 - np.arange(n)), return_inverse=True, return_counts=True)
    size = c2[inv2]
    Dparty = X.team_diff(((size >= 3) & pm).astype(float), d["g"], d["team"], n_games)
    # causal pair chemistry: shrunk mean of the pair's e over earlier days
    o4 = np.lexsort((rid, day, pkey))
    k4 = pkey[o4]
    d4 = day[o4].astype(np.int64)
    cs = np.r_[0.0, np.cumsum(ev[o4])]
    fd = np.searchsorted(k4 * 100000 + d4, k4 * 100000 + d4, side="left")
    fk = np.searchsorted(k4, k4, side="left")
    sm = cs[fd] - cs[fk]
    cn = (fd - fk).astype(float)
    var_pair = max(net_var["first vs second half in time"]["pair_specific_sd_pp"] / 100, 0.003) ** 2
    kshr = 0.24 / var_pair
    chem_est = np.empty(len(ra))
    chem_est[o4] = sm / (cn + kshr)
    fam_ge = np.empty(len(ra))
    fam_ge[o4] = np.log1p(cn)
    # joint comfort (causal, earlier days): pair played this hero pair before
    _, ckr = np.unique(ck, return_inverse=True)
    o5 = np.lexsort((rid, day, ckr))
    k5 = ckr[o5].astype(np.int64)
    d5 = day[o5].astype(np.int64)
    fd5 = np.searchsorted(k5 * 100000 + d5, k5 * 100000 + d5, side="left")
    fk5 = np.searchsorted(k5, k5, side="left")
    jce = np.empty(len(ra))
    jce[o5] = (fd5 - fk5) > 0
    pg = d["g"][ra]
    sg = sign(t[ra])
    Dchem = np.bincount(pg, weights=sg * chem_est, minlength=n_games)
    Dfam = np.bincount(pg, weights=sg * fam_ge, minlength=n_games)
    Djc = np.bincount(pg, weights=sg * jce, minlength=n_games)
    w0 = C.fit_logistic(np.column_stack([lo, D0])[fit_m], y[fit_m])
    l0 = X.logloss(C.predict(w0, np.column_stack([lo, D0])), y)
    idx = np.flatnonzero(test_m)
    gl = {}
    for nm, cols in (("+ party (3+ stack count)", [Dparty]), ("+ pair chemistry (causal, shrunk)", [Dchem]),
                     ("+ pair familiarity (log joint games)", [Dfam]),
                     ("+ joint comfort (hero pair played together before)", [Djc]),
                     ("+ all chemistry terms", [Dparty, Dchem, Dfam, Djc])):
        Xm = np.column_stack([lo, D0] + cols)
        wv = C.fit_logistic(Xm[fit_m], y[fit_m])
        l1 = X.logloss(C.predict(wv, Xm), y)
        g_, ci = X.boot_gain(l0, l1, idx)
        gl[nm] = {"gain_over_skill_model": g_, "ci": ci, "coef": wv.tolist()}
    out["game_level_V2"] = gl
    save_part("chem", out)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--real", type=int, default=4000)
    a = ap.parse_args()
    if a.cmd == "sim":
        sim(a.procs, a.real)
    elif a.cmd == "analyze":
        analyze_sim()
    elif a.cmd == "assign":
        assign()
    elif a.cmd == "combine":
        combine()
    elif a.cmd == "chem":
        chem()
