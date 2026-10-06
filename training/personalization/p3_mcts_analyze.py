"""
P3 personalized MCTS: analysis, mirrors p3_dr_analyze for the MCTS runs
and compares with the one-step drafter (cache/dr_runs.pkl.gz).

Metrics (MCTS at the main sims level unless stated):
  decisions      paired decide-only searches at the personalized a-gd
                 trajectory's states: share of differing picks; value gap
                 from the personalized tree's root Q (dV) and the population
                 tree's root Q (dWP), when both picks were visited
  final drafts   personalized vs population MCTS drafter, controlled team,
                 paired by lobby (same GD model and seed): V and WP_pop
  collapse       effective hero pool and top-1 / top-3 shares per player
  meta           hero pick shares, self-play and a-gd; risers/fallers
  off-role       share of picks off-role for the receiving player
  realized       team residual (won - WP_pop of the real game) by agreement
                 of the real picks with the drafter's root visit ranking
  imitation      imitation top-1 vs MCTS top-1 at the real states
Sims curve: predicted whole-draft gain (a-gd) and realized agreement on the
same 2,000-lobby subset at every sims level, plus the one-step reference.

Run (from training/):
  OMP_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_mcts_analyze.py
Outputs: results/p3_mcts_analyze.json, results/fig_mcts_sims_curve.png,
results/fig_mcts_meta.png
"""
import os
import sys
import gzip
import json
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from p3_heroes import NUM_HEROES
import p3_hs_core as C
import p3_dr_core as D
import p3_mcts_core as M

MAIN = 400
LEVELS = [25, 100, 400, 1500]
SUB = 2000


def load(stage, s):
    p = __import__("p3_mcts_core").result_path(f"mcts_{stage}_s{s}.pkl.gz")
    return pickle.load(gzip.open(p)) if os.path.exists(p) else None


def ci_mean(x, rng, n=1000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    bs = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def eff(c):
    c = np.asarray(c, float)
    c = c[c > 0]
    p = c / c.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


class Ctx:
    def __init__(self):
        self.d = C.load_slots()
        self.names = np.array([str(x) for x in self.d["hero_names"]])
        self.meta = C.hero_meta(self.d["hero_names"])
        self.T = D.load_personal()
        self.L = D.load_lobbies()
        gd = D.GDPolicy(self.d["hero_names"])
        self.gd = gd
        self.to_sh = gd.to_shared
        self.from_sh = np.argsort(self.to_sh)
        b = D.combiner()
        self.b = np.array([0.0, b[1], b[2], b[3]])
        w = np.load(C.WP)
        o = np.argsort(w["replay_ids"])
        self.y_all, self.wp_all = w["y"][o], w["wp0"][o]

    def slot_heroes(self, lob, acts):
        """-> list of (kernel team, row, my hero idx) for the 10 picks."""
        out = []
        for k in range(16):
            if M.IS_PICK[k]:
                sl = int(lob["slot_of"][k])
                out.append((0 if sl < 5 else 1, int(lob["rows"][sl]), int(self.from_sh[acts[k]])))
        return out

    def value(self, lob, acts, wp_t0):
        S, O = [0.0, 0.0], [0.0, 0.0]
        for tm, r, h in self.slot_heroes(lob, acts):
            p = self.T["pos"][r]
            S[tm] += float(self.T["s"][p][h])
            O[tm] += float(self.T["off"][p][h])
        wp = min(max(wp_t0, 1e-6), 1 - 1e-6)
        z = self.b[1] * np.log(wp / (1 - wp)) + self.b[2] * (S[0] - S[1]) + self.b[3] * (O[0] - O[1])
        return 1 / (1 + np.exp(-z))


def final_drafts(X, full, opp, rng):
    dV, dW, vp, vq = [], [], [], []
    for lob, rp, rq in zip(full["lobbies"], full[f"a-{opp}|pers"], full[f"a-{opp}|pop"]):
        kc = lob["kctrl"]
        Vp = X.value(lob, rp["acts"], rp["wp_t0"])
        Vq = X.value(lob, rq["acts"], rq["wp_t0"])
        Wp, Wq = rp["wp_t0"], rq["wp_t0"]
        if kc == 1:
            Vp, Vq, Wp, Wq = 1 - Vp, 1 - Vq, 1 - Wp, 1 - Wq
        dV.append(Vp - Vq)
        dW.append(Wp - Wq)
        vp.append(Vp)
        vq.append(Vq)
    return {"dV_pp": ci_mean(100 * np.array(dV), rng), "dWPpop_pp": ci_mean(100 * np.array(dW), rng),
            "mean_V_pers": float(np.mean(vp)), "mean_V_pop": float(np.mean(vq)), "lobbies": len(dV)}


def realized(X, R, rng, gis_subset=None):
    """Team residual by agreement of real picks with the root visit ranking."""
    lobs = R["lobbies"]
    per = {}
    for li, k, pers, pol, q in R["decisions"]:
        lob = lobs[li]
        if gis_subset is not None and lob["gi"] not in gis_subset:
            continue
        actual = int(lob["acts_real"][k])
        sl = int(lob["slot_of"][k])
        row = int(lob["rows"][sl])
        pool = X.T["pool"][X.T["pos"][row]]
        taken = set(int(x) for x in lob["acts_real"][:k])
        cand = [int(X.to_sh[h]) for h in np.flatnonzero(pool) if int(X.to_sh[h]) not in taken]
        if actual not in cand:
            cand.append(actual)
        if len(cand) < 2:
            continue
        key = np.array([pol[c] + 1e-3 * q[c] for c in cand])
        ka = pol[actual] + 1e-3 * q[actual]
        pct = ((key < ka).sum() + 0.5 * ((key == ka).sum() - 1)) / (len(cand) - 1)
        kt = M.DRAFT_TEAM[k]
        per.setdefault((li, kt), {}).setdefault(k, {})[pers] = pct
    recs = []
    for (li, kt), dd in per.items():
        lob = lobs[li]
        pp = [v[1] for v in dd.values() if 1 in v and 0 in v]
        pq = [v[0] for v in dd.values() if 1 in v and 0 in v]
        if len(pp) < 3:
            continue
        g = X.L["g"][lob["gi"]]
        real_team = kt ^ lob["first"]
        yt = X.y_all[g] if real_team == 0 else 1 - X.y_all[g]
        wt = X.wp_all[g] if real_team == 0 else 1 - X.wp_all[g]
        recs.append((np.mean(pp), np.mean(pq), np.mean(pp) - np.mean(pq), yt - wt))
    A = np.array(recs)
    out = {"teams": int(len(A))}
    for j, nm in ((0, "personalized agreement"), (1, "population agreement"), (2, "personal component")):
        qs = np.percentile(A[:, j], [20, 40, 60, 80])
        b = np.searchsorted(qs, A[:, j])
        out[nm] = {"quintile_resid_pp": [float(100 * A[b == q, 3].mean()) for q in range(5)]}
        top, bot = A[b == 4, 3], A[b == 0, 3]
        diffs = [top[rng.randint(0, len(top), len(top))].mean() - bot[rng.randint(0, len(bot), len(bot))].mean()
                 for _ in range(1000)]
        out[nm]["top_minus_bottom_pp"] = [float(100 * (top.mean() - bot.mean())), float(100 * np.percentile(diffs, 2.5)),
                                          float(100 * np.percentile(diffs, 97.5))]
    X_ = np.column_stack([np.ones(len(A)), A[:, 0] - A[:, 0].mean(), A[:, 1] - A[:, 1].mean()])
    beta = np.linalg.lstsq(X_, A[:, 3], rcond=None)[0]
    bs = []
    for _ in range(500):
        ii = rng.randint(0, len(A), len(A))
        bs.append(np.linalg.lstsq(X_[ii], A[ii, 3], rcond=None)[0][1])
    out["pers_slope_controlling_pop_pp"] = [float(100 * beta[1]), float(100 * np.percentile(bs, 2.5)),
                                            float(100 * np.percentile(bs, 97.5))]
    return out


def onestep_realized(X, gis_subset, rng):
    z = pickle.load(gzip.open(os.path.join(C.CACHE, "dr_runs.pkl.gz" if os.environ.get("P3_LOBBY_SET", "all10") == "all10" else f"dr_runs_{os.environ['P3_LOBBY_SET']}.pkl.gz")))
    recs = []
    for l in z["lobbies"]:
        if l["gi"] not in gis_subset:
            continue
        g = X.L["g"][l["gi"]]
        for t in (0, 1):
            pp = []
            for dd in l["real"]["decisions"]:
                if dd["team"] != t or len(dd["cand"]) < 2:
                    continue
                i = int(np.flatnonzero(dd["cand"] == dd["actual"])[0])
                pp.append(np.sum(dd["V"] < dd["V"][i]) / (len(dd["cand"]) - 1))
            yt = X.y_all[g] if t == 0 else 1 - X.y_all[g]
            wt = X.wp_all[g] if t == 0 else 1 - X.wp_all[g]
            recs.append((np.mean(pp), yt - wt))
    A = np.array(recs)
    qs = np.percentile(A[:, 0], [20, 40, 60, 80])
    b = np.searchsorted(qs, A[:, 0])
    top, bot = A[b == 4, 1], A[b == 0, 1]
    diffs = [top[rng.randint(0, len(top), len(top))].mean() - bot[rng.randint(0, len(bot), len(bot))].mean()
             for _ in range(1000)]
    return [float(100 * (top.mean() - bot.mean())), float(100 * np.percentile(diffs, 2.5)),
            float(100 * np.percentile(diffs, 97.5))]


def main():
    rng = np.random.RandomState(0)
    X = Ctx()
    out = {"main_sims": MAIN}
    full = load("full", MAIN)
    # ---------- decisions (paired, same state)
    pr = {}
    for li, k, pers, pol, q in full["paired"]:
        pr.setdefault((li, k), {})[pers] = (pol, q)
    diff, dV, dW, vis = [], [], [], []
    for (li, k), dd in pr.items():
        if 1 not in dd or 0 not in dd:
            continue
        (pp, qp), (pq, qq) = dd[1], dd[0]
        a, b_ = int(np.argmax(pp)), int(np.argmax(pq))
        diff.append(a != b_)
        if a != b_:
            ok = pp[b_] > 0 and pq[a] > 0
            vis.append(ok)
            if ok:
                dV.append(qp[a] - qp[b_])
                dW.append(qq[a] - qq[b_])
    out["decisions"] = {"decisions": len(diff), "share_differ": float(np.mean(diff)),
                        "share_both_picks_visited_in_both_trees": float(np.mean(vis)),
                        "dV_pp_when_differ (personal tree Q)": ci_mean(100 * np.array(dV), rng),
                        "dWPpop_pp_when_differ (population tree Q)": ci_mean(100 * np.array(dW), rng)}
    print("decisions", json.dumps(out["decisions"]), flush=True)
    out["final_drafts"] = {f"vs {o}": final_drafts(X, full, o, rng) for o in ("gd", "imit")}
    print("final", json.dumps(out["final_drafts"]), flush=True)
    # ---------- meta and off-role
    def counts(key, which=None):
        c = np.zeros(NUM_HEROES)
        for lob, r in zip(full["lobbies"], full[key]):
            for tm, row, h in X.slot_heroes(lob, r["acts"]):
                if which is None or tm == lob["kctrl"]:
                    c[h] += 1
        return c
    creal = np.zeros(NUM_HEROES)
    for lob in full["lobbies"]:
        for tm, row, h in X.slot_heroes(lob, lob["acts_real"]):
            creal[h] += 1
    cp, cq = counts("b|pers"), counts("b|pop")
    sh = lambda c: c / c.sum()
    z1 = pickle.load(gzip.open(os.path.join(C.CACHE, "dr_runs.pkl.gz" if os.environ.get("P3_LOBBY_SET", "all10") == "all10" else f"dr_runs_{os.environ['P3_LOBBY_SET']}.pkl.gz")))
    c1p, c1q = np.zeros(NUM_HEROES), np.zeros(NUM_HEROES)
    for l in z1["lobbies"]:
        if "b|pers" in l:
            for t in ("t0", "t1"):
                for h in l["b|pers"][t].values():
                    c1p[h] += 1
                for h in l["b|pop"][t].values():
                    c1q[h] += 1
    out["meta"] = {"effective_heroes": {"MCTS personalized self-play": eff(cp), "MCTS population self-play": eff(cq),
                                        "one-step personalized self-play": eff(c1p),
                                        "one-step population self-play": eff(c1q), "real": eff(creal)},
                   "top10_share": {"MCTS personalized": float(np.sort(sh(cp))[-10:].sum()),
                                   "MCTS population": float(np.sort(sh(cq))[-10:].sum()),
                                   "real": float(np.sort(sh(creal))[-10:].sum())},
                   "corr_with_real": {"MCTS personalized": float(np.corrcoef(sh(cp), sh(creal))[0, 1]),
                                      "MCTS population": float(np.corrcoef(sh(cq), sh(creal))[0, 1]),
                                      "one-step personalized": float(np.corrcoef(sh(c1p), sh(creal))[0, 1]),
                                      "one-step population": float(np.corrcoef(sh(c1q), sh(creal))[0, 1])},
                   "corr_MCTS_pers_vs_pop": float(np.corrcoef(sh(cp), sh(cq))[0, 1]),
                   "role_shares": {nm: {C.BLIZZ_ROLES[k]: float(sh(c)[X.meta["blizz"] == k].sum()) for k in range(6)}
                                   for nm, c in (("MCTS personalized", cp), ("MCTS population", cq), ("real", creal))}}
    lr = np.log2((sh(cp) + 1e-3) / (sh(cq) + 1e-3))
    tab = lambda i: {"hero": X.names[i], "pers_pct": float(100 * sh(cp)[i]), "pop_pct": float(100 * sh(cq)[i]),
                     "real_pct": float(100 * sh(creal)[i])}
    out["meta"]["risers"] = [tab(i) for i in np.argsort(-lr)[:12]]
    out["meta"]["fallers"] = [tab(i) for i in np.argsort(lr)[:12]]
    out["meta"]["top_pers"] = [tab(i) for i in np.argsort(-cp)[:10]]
    out["meta"]["top_pop"] = [tab(i) for i in np.argsort(-cq)[:10]]
    print("meta", json.dumps({k: v for k, v in out["meta"].items() if k in ("effective_heroes", "corr_with_real", "top10_share")}), flush=True)

    def offrole(key):
        o, n = 0, 0
        for lob, r in zip(full["lobbies"], full[key]):
            for tm, row, h in X.slot_heroes(lob, r["acts"]):
                o += X.T["off"][X.T["pos"][row]][h]
                n += 1
        return float(o / n)
    out["off_role"] = {k: offrole(k) for k in ("b|pers", "b|pop", "a-gd|pers", "a-gd|pop")}
    o, n = 0, 0
    for lob in full["lobbies"]:
        for tm, row, h in X.slot_heroes(lob, lob["acts_real"]):
            o += X.T["off"][X.T["pos"][row]][h]
            n += 1
    out["off_role"]["real"] = float(o / n)
    print("off-role", out["off_role"], flush=True)
    # ---------- collapse
    col = load("collapse", MAIN)
    if col is not None:
        per = {}
        for p, k, pers, h in col["decisions"]:
            per.setdefault(p, {0: [], 1: []})[pers].append(int(X.from_sh[h]))
        rows = []
        for p, v in per.items():
            nn = X.T["n"][X.T["pos"][int(p)]]
            top1, top3 = int(np.argmax(nn)), set(np.argsort(-nn)[:3])
            rows.append({"pers_eff": eff(np.bincount(v[1], minlength=NUM_HEROES)), "pop_eff": eff(np.bincount(v[0], minlength=NUM_HEROES)),
                         "pers_top1": np.mean([h == top1 for h in v[1]]), "pop_top1": np.mean([h == top1 for h in v[0]]),
                         "pers_top3": np.mean([h in top3 for h in v[1]]), "pop_top3": np.mean([h in top3 for h in v[0]])})
        out["collapse"] = {k: [float(np.median([r[k] for r in rows])), float(np.mean([r[k] for r in rows]))]
                           for k in rows[0]}
        print("collapse", out["collapse"], flush=True)
    # ---------- realized, all lobbies at MAIN
    R = load("real", MAIN)
    out["realized_all"] = realized(X, R, rng)
    print("realized all", json.dumps(out["realized_all"]), flush=True)
    # ---------- sims curve on the common subset
    gis_sub = set(int(x) for x in np.r_[z1["full_set"], z1["extra_set"]][:SUB])
    curve = {"one-step": {"realized_top_minus_bottom_pp": onestep_realized(X, gis_sub, rng)}}
    fd1 = {}
    for l in z1["lobbies"]:
        if "a-gd|pers" in l:
            c = l["ctrl"]
            vp, vq = l["a-gd|pers"]["V"], l["a-gd|pop"]["V"]
            if c == 1:
                vp, vq = 1 - vp, 1 - vq
            fd1.setdefault("dV", []).append(vp - vq)
    curve["one-step"]["dV_pp"] = ci_mean(100 * np.array(fd1["dV"]), rng)
    for s in LEVELS:
        e = {}
        f = load("full", s) if s == MAIN else load("curve", s)
        if f is not None:
            fd = final_drafts(X, f, "gd", rng)
            e["dV_pp"], e["dWPpop_pp"], e["mean_V_pers"] = fd["dV_pp"], fd["dWPpop_pp"], fd["mean_V_pers"]
        r = load("real", s)
        if r is not None:
            rr = realized(X, r, rng, gis_sub)
            e["realized_top_minus_bottom_pp"] = rr["personalized agreement"]["top_minus_bottom_pp"]
            e["realized_personal_component_pp"] = rr["personal component"]["top_minus_bottom_pp"]
            e["realized_pop_agreement_pp"] = rr["population agreement"]["top_minus_bottom_pp"]
            e["teams"] = rr["teams"]
        curve[f"MCTS {s}"] = e
        print("curve", s, json.dumps(e), flush=True)
    out["sims_curve"] = curve
    # ---------- imitation vs outcome at the real states (MAIN)
    import p3_dr_imitation as I
    iw = np.load(I.OUT)["w"]
    lobs = R["lobbies"]
    rows_needed = sorted(set(int(r) for lob in lobs for r in lob["rows"]))
    rec = I.recency_features(X.d, np.array(rows_needed))
    fake = {"row": np.array(rows_needed), "recpos": np.arange(len(rows_needed)),
            "lp": np.zeros((len(rows_needed), NUM_HEROES), np.float32)}
    Xp = I.feature_tensor(fake, X.T, rec, X.meta)
    ipers = {r: (Xp[i, :, 1:] * iw[1:]).sum(-1) for i, r in enumerate(rows_needed)}
    dec = {}
    for li, k, pers, pol, q in R["decisions"]:
        dec.setdefault((li, k), {})[pers] = int(X.from_sh[int(np.argmax(pol))])
    items = []
    for (li, k), dd in dec.items():
        if 1 not in dd or 0 not in dd:
            continue
        lob = lobs[li]
        acts = [int(X.from_sh[a]) for a in lob["acts_real"]]
        t = np.zeros((2, NUM_HEROES), np.float32)
        bans = np.zeros(NUM_HEROES, np.float32)
        for j in range(k):
            if M.IS_PICK[j]:
                t[M.DRAFT_TEAM[j] ^ lob["first"], acts[j]] = 1
            else:
                bans[acts[j]] = 1
        gi = lob["gi"]
        mo, to = D.one_hots(str(X.L["map"][gi]), str(X.L["tier"][gi]))
        sl = int(lob["slot_of"][k])
        items.append((t[0], t[1], bans, mo, to, k, (t[0] + t[1] + bans) == 0, int(lob["rows"][sl]), acts[k], dd[1], dd[0],
                      gi, M.DRAFT_TEAM[k] ^ lob["first"]))
    cls = []
    for s0 in range(0, len(items), 4096):
        ch = items[s0:s0 + 4096]
        lp = X.gd.logprobs(np.array([c[0] for c in ch]), np.array([c[1] for c in ch]), np.array([c[2] for c in ch]),
                           np.array([c[3] for c in ch]), np.array([c[4] for c in ch]),
                           np.array([c[5] for c in ch], np.float32), np.ones(len(ch), np.float32),
                           np.array([c[6] for c in ch]))
        for c, l_ in zip(ch, lp):
            pool = X.T["pool"][X.T["pos"][c[7]]] & c[6]
            u = np.where(pool if pool.any() else c[6], iw[0] * l_ + ipers[c[7]], -1e9)
            g = X.L["g"][c[11]]
            yt = X.y_all[g] if c[12] == 0 else 1 - X.y_all[g]
            wt = X.wp_all[g] if c[12] == 0 else 1 - X.wp_all[g]
            cls.append((c[8], int(np.argmax(u)), c[9], c[10], yt - wt))
    Cc = np.array(cls, float)
    act, it, pe, po, res_ = Cc.T
    groups = {"matches imitation only": (act == it) & (act != pe), "matches personalized MCTS only": (act == pe) & (act != it),
              "matches both": (act == pe) & (act == it), "matches neither": (act != pe) & (act != it)}
    out["imitation_vs_outcome"] = {"picks": int(len(Cc)),
                                   "real_equals_imitation_top1": float((act == it).mean()),
                                   "real_equals_MCTS_personalized_top1": float((act == pe).mean()),
                                   "real_equals_MCTS_population_top1": float((act == po).mean()),
                                   "imitation_equals_MCTS_personalized": float((it == pe).mean()),
                                   "MCTS_personalized_equals_population": float((pe == po).mean()),
                                   "team_residual_pp": {k: {"picks": int(m.sum()), "resid": ci_mean(100 * res_[m], rng)}
                                                        for k, m in groups.items()}}
    print("imitation", json.dumps(out["imitation_vs_outcome"]), flush=True)
    # ---------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    xs = [s for s in LEVELS if "dV_pp" in curve.get(f"MCTS {s}", {})]
    ys = [curve[f"MCTS {s}"]["dV_pp"] for s in xs]
    axs[0].errorbar(xs, [y[0] for y in ys], yerr=[[y[0] - y[1] for y in ys], [y[2] - y[0] for y in ys]], fmt="o-",
                    capsize=3, label="MCTS personalized minus population (V)")
    o1 = curve["one-step"]["dV_pp"]
    axs[0].axhline(o1[0], color="#e15759", ls="--", label=f"one-step ({o1[0]:.1f}pp)")
    axs[0].set_xscale("log")
    axs[0].set_xlabel("simulations per decision")
    axs[0].set_ylabel("predicted whole-draft gain (pp of V), vs GD opponent")
    axs[0].set_title("Predicted whole-draft gain by search size (1,000 lobbies)", fontsize=10)
    axs[0].legend(fontsize=8)
    xs = [s for s in LEVELS if "realized_top_minus_bottom_pp" in curve.get(f"MCTS {s}", {})]
    for key, lab, col in (("realized_top_minus_bottom_pp", "personalized agreement", "#4e79a7"),
                          ("realized_personal_component_pp", "personal component", "#59a14f"),
                          ("realized_pop_agreement_pp", "population agreement", "#999999")):
        ys = [curve[f"MCTS {s}"][key] for s in xs]
        axs[1].errorbar(xs, [y[0] for y in ys], yerr=[[y[0] - y[1] for y in ys], [y[2] - y[0] for y in ys]],
                        fmt="o-", capsize=3, color=col, label=lab)
    o1 = curve["one-step"]["realized_top_minus_bottom_pp"]
    axs[1].axhline(o1[0], color="#e15759", ls="--", label=f"one-step personalized ({o1[0]:.1f}pp)")
    axs[1].axhline(0, color="#ccc", lw=0.5)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("simulations per decision")
    axs[1].set_ylabel("realized residual, top minus bottom agreement quintile (pp)")
    axs[1].set_title(f"Realized value of agreeing with the drafter ({SUB} held-out lobbies)", fontsize=10)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.RESULTS, "fig_mcts_sims_curve.png"), dpi=130)
    plt.close(fig)
    role_col = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#9c755f"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, (xa, ya, xl) in zip(axs, ((sh(cq), sh(cp), "MCTS population self-play"),
                                      (sh(creal), sh(cp), "real drafts (same lobbies)"))):
        for i in range(NUM_HEROES):
            ax.scatter(100 * xa[i], 100 * ya[i], color=role_col[X.meta["blizz"][i]], s=18)
        lim = 100 * max(xa.max(), ya.max()) * 1.05
        ax.plot([0, lim], [0, lim], color="#999", lw=0.7)
        lab = set(np.argsort(-np.abs(np.log((ya + 1e-3) / (xa + 1e-3))))[:14]) | set(np.argsort(-ya)[:8])
        for i in lab:
            ax.annotate(X.names[i], (100 * xa[i], 100 * ya[i]), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel(f"{xl}: pick share (%)")
        ax.set_ylabel(f"MCTS personalized self-play: pick share (%)")
        ax.set_title(f"MCTS ({MAIN} sims) emergent meta, {len(full['lobbies'])} lobbies", fontsize=10)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8) for c in role_col]
    fig.legend(handles, C.BLIZZ_ROLES, loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(C.RESULTS, "fig_mcts_meta.png"), dpi=120)
    with open(os.path.join(C.RESULTS, "p3_mcts_analyze.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("done")


if __name__ == "__main__":
    main()
