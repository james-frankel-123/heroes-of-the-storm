"""
Audit fixes for P3_EXTENSIONS section 5(b) and 5(d) (P3-01, P3-02, P3-24).

Both sections ran on the pre-audit one-step harness (pool includes the real
hero, real bans replayed) and both used realized agreement as evidence. Here
they are rerun under the fixed protocol of p3_fix_drafters (pre-day pools, GD
bans, lag-1 offsets, refit combiner, lag-1 imitation features) and every
realized check is reported as calibration: realized team residual, the skill
model's own predicted gap for the same real draft, and the remainder
(p3_fix_realized.summarize; game bootstrap).

  b_sim       (b) teammates-only drafter: 1,000 simulated lobbies (full /
              teammates-only / population drafter for one team, GD
              opponents in their pools, GD bans), plus teammates-only
              rankings at the real states of the fixed one-step realized
              lobbies
  b_analyze   value kept by the teammates-only drafter; realized calibration
  d           (d) imitation and outcome: joint regression of the realized
              team residual on the personal-component agreement, the
              imitation agreement and the predicted gap; lambda tuning on
              half the lobbies, test on the other half, now with the
              predicted gap and remainder next to every realized number

Run (from training/), after p3_fix_drafters.py tables and onestep:
  OMP_NUM_THREADS=1 nice -n 19 taskset -c 48-63 python3 personalization/p3_fix_ext5.py b_sim --procs 4
  OMP_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_fix_ext5.py b_analyze
  OMP_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_fix_ext5.py d
Outputs: cache/fix/x_draft_sim_b.pkl.gz, results/fix/p3_fix_ext5.json
"""
import os
import sys
import gzip
import json
import time
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from p3_heroes import NUM_HEROES
import p3_hs_core as C
import p3_fix_drafters as FD

SIMB = os.path.join(FD.FIX_CACHE, "x_draft_sim_b.pkl.gz")
OUTJ = os.path.join(FD.FIX_RESULTS, "p3_fix_ext5.json")
_DR = None


def save_part(key, val):
    out = {}
    if os.path.exists(OUTJ):
        with open(OUTJ) as f:
            out = json.load(f)
    out[key] = val
    with open(OUTJ, "w") as f:
        json.dump(out, f, indent=1, default=float)


def run_picks_b(lob, seed, mode):
    """p3_x_draft.run_picks with GD-sampled bans (no real bans replayed)."""
    import p3_x_draft as XD
    DR = _DR
    c = lob["ctrl"]
    rng_env = np.random.RandomState(seed)
    rng_dec = np.random.RandomState(seed + 99991)
    ldeg = XD.degraded_lob(lob, c)
    state = XD._empty()
    decs = []
    for k in range(16):
        h, ty, tm, row = lob["steps"][k]
        if ty == 0:
            state = DR._apply_ban(state, int(DR._ban_sample([state], k, lob, rng_env)[0]))
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


def work_full_b(args):
    lob, seed = args
    out = {"gi": lob["gi"], "ctrl": lob["ctrl"]}
    for m in ("full", "deg", "pop"):
        out[f"pick|{m}"] = run_picks_b(lob, seed, m)
    return out


def work_real_b(args):
    """Real states: rank the real picks with the teammates-only drafter."""
    import p3_x_draft as XD
    DR = _DR
    lob, seed = args
    rng = np.random.RandomState(seed + 3)
    state = XD._empty()
    picks = []
    for k in range(16):
        h, ty, tm, row = lob["steps"][k]
        if ty == 0:
            state = XD._ban(state, h)
        else:
            st = dict(state)
            st["force"] = h
            dd = DR._decide(st, k, XD.degraded_lob(lob, tm), rng, "gd")
            picks.append({"cand": dd["cand"], "V": dd["V"], "wp": dd["wp"], "actual": int(h), "team": tm})
            state = DR._apply(state, k, h, lob)
    return {"gi": lob["gi"], "picks_deg": picks}


def b_sim(procs):
    global _DR
    t0 = time.time()
    FD.patch_common()
    _DR = FD.patch_onestep()
    DR = _DR
    import p3_dr_core as D
    import p3_dr_imitation as I
    z = pickle.load(gzip.open(DR.OUT))
    d = C.load_slots()
    L = D.load_lobbies()
    T = D.load_personal()
    w = np.load(C.WP)
    bidx_all = w["build_idx"][np.argsort(w["replay_ids"])]
    b = D.combiner()
    full = [l for l in z["lobbies"] if "b|pers" in l]
    ctrl = {l["gi"]: l["ctrl"] for l in full}
    real_g = [l["gi"] for l in z["lobbies"]]
    iw = np.load(I.OUT)["w"]

    class ZeroDict(dict):
        def __missing__(self, key):
            return np.zeros(NUM_HEROES, np.float32)
    ip = ZeroDict()
    tasks = [(DR.lobby_payload(L, T, gi, ip, ctrl[gi], bidx_all), int(1000 + gi)) for gi in ctrl]
    rtasks = [(DR.lobby_payload(L, T, gi, ip, 0, bidx_all), int(1000 + gi)) for gi in real_g]
    shared = {"hero_names": d["hero_names"], "b": b, "iw0": float(iw[0])}
    print(f"{len(tasks)} full lobbies, {len(rtasks)} realized lobbies ({time.time() - t0:.0f}s)", flush=True)
    import multiprocessing as mp
    res, rres = [], []
    with mp.get_context("fork").Pool(procs, initializer=DR._init, initargs=(shared,)) as pool:
        for i, r in enumerate(pool.imap_unordered(work_full_b, tasks, chunksize=2)):
            res.append(r)
            if (i + 1) % 100 == 0:
                print(f"  full {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
        for i, r in enumerate(pool.imap_unordered(work_real_b, rtasks, chunksize=4)):
            rres.append(r)
            if (i + 1) % 500 == 0:
                print(f"  real {i + 1}/{len(rtasks)} ({time.time() - t0:.0f}s)", flush=True)
    with gzip.open(SIMB, "wb") as f:
        pickle.dump({"full": res, "real": rres}, f)
    print(f"sim done {time.time() - t0:.0f}s", flush=True)


def pct(v, cand, actual):
    i = int(np.flatnonzero(cand == actual)[0])
    return np.sum(v < v[i]) / (len(cand) - 1)


def b_analyze():
    import p3_fix_realized as FR
    rng = np.random.RandomState(0)
    z = pickle.load(gzip.open(SIMB))
    F = z["full"]
    out = {"n_full": len(F), "n_real": len(z["real"])}
    pk = {}
    for a_, b_ in (("full", "pop"), ("deg", "pop"), ("full", "deg")):
        dV = np.array([l[f"pick|{a_}"]["V"] - l[f"pick|{b_}"]["V"] for l in F])
        dW = np.array([l[f"pick|{a_}"]["wp"] - l[f"pick|{b_}"]["wp"] for l in F])
        bs = [dV[rng.randint(0, len(dV), len(dV))].mean() for _ in range(1000)]
        pk[f"{a_} - {b_}"] = {"dV_pp": [float(100 * dV.mean()), float(100 * np.percentile(bs, 2.5)),
                                        float(100 * np.percentile(bs, 97.5))], "dWP_pp": float(100 * dW.mean())}
    dfull = np.array([l["pick|full"]["V"] - l["pick|pop"]["V"] for l in F])
    ddeg = np.array([l["pick|deg"]["V"] - l["pick|pop"]["V"] for l in F])
    bs = []
    for _ in range(1000):
        i = rng.randint(0, len(F), len(F))
        bs.append(ddeg[i].mean() / dfull[i].mean())
    pk["share_kept"] = [float(ddeg.mean() / dfull.mean()), float(np.percentile(bs, 2.5)),
                        float(np.percentile(bs, 97.5))]
    out["teammates_only_sim"] = pk
    X = FR.Ctx()
    Dr = pickle.load(gzip.open(os.path.join(FD.FIX_CACHE, "dr_runs.pkl.gz")))
    realmap = {l["gi"]: l["real"]["decisions"] for l in Dr["lobbies"]}
    recs = []
    for l in z["real"]:
        gi = l["gi"]
        (pg0, pg1), (r0, r1) = X.team_pred_gap(gi)
        full_dec = realmap[gi]
        for t, pg, rr in ((0, pg0, r0), (1, pg1, r1)):
            pdg, pfl = [], []
            for dd, df in zip(l["picks_deg"], full_dec):
                if dd["team"] != t or len(dd["cand"]) < 2 or len(df["cand"]) < 2:
                    continue
                pdg.append(pct(dd["V"], dd["cand"], dd["actual"]))
                pfl.append(pct(df["V"], df["cand"], df["actual"]))
            if len(pdg) >= 3:
                recs.append((gi, t, np.mean(pdg), np.mean(pfl), rr, pg))
    s = FR.summarize(recs, rng)
    out["teammates_only_realized_calibration"] = {
        "teams": s["teams"], "games": s["games"],
        "teammates-only drafter agreement": s["personalized agreement"],
        "full-information drafter agreement": s["population agreement"]}
    save_part("b", out)
    print(json.dumps(out, indent=1, default=float)[:4000])


def d_combine():
    FD.patch_common()
    import p3_dr_core as D
    import p3_dr_imitation as I
    import p3_fix_realized as FR
    rng = np.random.RandomState(0)
    z = pickle.load(gzip.open(os.path.join(FD.FIX_CACHE, "dr_runs.pkl.gz")))
    d = C.load_slots()
    meta = C.hero_meta(d["hero_names"])
    L = D.load_lobbies()
    T = D.load_personal()
    gd = D.GDPolicy(d["hero_names"])
    iw = np.load(I.OUT)["w"]
    X = FR.Ctx()
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
    gaps = {}
    per_pick = []
    for s0 in range(0, len(items), 4096):
        ch = items[s0:s0 + 4096]
        lp = gd.logprobs(np.array([c[0] for c in ch]), np.array([c[1] for c in ch]), np.array([c[2] for c in ch]),
                         np.array([c[3] for c in ch]), np.array([c[4] for c in ch]),
                         np.array([c[5] for c in ch], np.float32), np.ones(len(ch), np.float32),
                         np.array([c[6] for c in ch]))
        for c, l_ in zip(ch, lp):
            dd = c[9]
            u = iw[0] * l_ + ipers[int(c[7])]
            uc = u[dd["cand"]]
            uc = uc - uc.max()
            logp = uc - np.log(np.exp(uc).sum())
            gi, tm = c[10], c[11]
            if gi not in gaps:
                gaps[gi] = X.team_pred_gap(gi)
            (pg, rr) = gaps[gi]
            per_pick.append({"gi": gi, "team": tm, "cand": dd["cand"], "V": dd["V"], "wp": dd["wp"],
                             "logp": logp, "actual": int(c[8]), "res": rr[tm], "pred": pg[tm]})
    lobbies = np.array(sorted({p["gi"] for p in per_pick}))
    half = set(rng.choice(lobbies, len(lobbies) // 2, replace=False))

    def team_table(lam=None, sel_half=None):
        recs = {}
        top_rows = []
        for p in per_pick:
            if sel_half is not None and (p["gi"] in half) != sel_half:
                continue
            nc = len(p["cand"])
            if nc < 2:
                continue
            i = int(np.flatnonzero(p["cand"] == p["actual"])[0])
            key = (p["gi"], p["team"])
            r_ = recs.setdefault(key, [[], [], [], [], p["res"], p["pred"]])
            pdv = p["V"] - p["wp"]
            r_[0].append(np.sum(pdv < pdv[i]) / (nc - 1))
            r_[1].append(np.sum(p["logp"] < p["logp"][i]) / (nc - 1))
            r_[2].append(np.sum(p["V"] < p["V"][i]) / (nc - 1))
            if lam is not None:
                sc = p["V"] + lam * p["logp"]
                r_[3].append(np.sum(sc < sc[i]) / (nc - 1))
                top_rows.append((int(p["cand"][np.argmax(sc)]) == p["actual"], p["res"], p["pred"], p["gi"]))
        A = np.array([(k[0], k[1], np.mean(a), np.mean(b), np.mean(c), np.mean(e) if e else np.nan, r, g)
                      for k, (a, b, c, e, r, g) in recs.items()])
        return A, np.array(top_rows, float)

    def game_boot(gi, nb=500):
        ug, inv = np.unique(gi, return_inverse=True)
        return [np.bincount(rng.randint(0, len(ug), len(ug)), minlength=len(ug))[inv].astype(float)
                for _ in range(nb)]

    A, _ = team_table()
    W = game_boot(A[:, 0])
    res, pred = A[:, 6], A[:, 7]
    joint = {}
    for nm, cols in (("V and imitation", [4, 3]),
                     ("personal component (V - WP) and imitation", [2, 3]),
                     ("personal component, imitation and the predicted gap", [2, 3, 7]),
                     ("imitation and the predicted gap", [3, 7])):
        Xj = np.column_stack([np.ones(len(A))] + [A[:, c] - A[:, c].mean() for c in cols])
        beta = np.linalg.lstsq(Xj, res, rcond=None)[0]
        bs = []
        for w in W:
            sw = np.sqrt(w)
            bs.append(np.linalg.lstsq(Xj * sw[:, None], res * sw, rcond=None)[0][1:])
        bs = np.array(bs)
        joint[nm] = {"coef": [float(beta[j + 1]) for j in range(len(cols))],
                     "ci": [[float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]
                            for j in range(len(cols))],
                     "units": "agreement terms: residual per unit agreement (x100 = pp); predicted gap: slope"}
    joint["corr personal component vs imitation"] = float(np.corrcoef(A[:, 2], A[:, 3])[0, 1])
    joint["corr imitation vs predicted gap"] = float(np.corrcoef(A[:, 3], pred)[0, 1])
    recs_imit = [(a[0], a[1], a[3], a[2], a[6], a[7]) for a in A]
    calib = FR.summarize(recs_imit, rng)
    calib = {"imitation agreement": calib["personalized agreement"],
             "personal-component agreement": calib["population agreement"], "teams": calib["teams"]}
    lams = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0, 1e3]
    tune = {}
    for lam in lams:
        _, tr = team_table(lam, True)
        m = tr[:, 0] == 1
        tune[str(lam)] = {"top1_match": float(m.mean()),
                          "residual_when_matched_pp": float(100 * tr[m, 1].mean()),
                          "remainder_when_matched_pp": float(100 * (tr[m, 1] - tr[m, 2]).mean())}
    best_m = max(lams, key=lambda l_: tune[str(l_)]["remainder_when_matched_pp"])
    test = {}
    for lam in sorted({0.0, 0.002, best_m, 1e3}):
        At, tr = team_table(lam, False)
        m = tr[:, 0] == 1
        Wg = game_boot(tr[:, 3], 300)

        def ci(v, sel):
            vals = [np.average(v[sel], weights=w[sel]) for w in Wg]
            return [float(100 * v[sel].mean()), float(100 * np.percentile(vals, 2.5)),
                    float(100 * np.percentile(vals, 97.5))]
        rem = tr[:, 1] - tr[:, 2]
        test[str(lam)] = {"top1_match_rate": float(m.mean()),
                          "residual_when_matched_pp": ci(tr[:, 1], m),
                          "predicted_gap_when_matched_pp": ci(tr[:, 2], m),
                          "remainder_when_matched_pp": ci(rem, m),
                          "remainder_otherwise_pp": ci(rem, ~m)}
    out = {"picks": len(per_pick), "teams": int(len(A)), "team_level_joint_regression": joint,
           "calibration_by_quintile": calib, "lambda_tuning_half": tune,
           "lambda_chosen_by_matched_remainder": best_m, "test_half": test}
    save_part("d", out)
    print(json.dumps(out, indent=1, default=float)[:6000])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["b_sim", "b_analyze", "d"])
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    if a.what == "b_sim":
        b_sim(a.procs)
    elif a.what == "b_analyze":
        b_analyze()
    else:
        d_combine()


if __name__ == "__main__":
    main()
