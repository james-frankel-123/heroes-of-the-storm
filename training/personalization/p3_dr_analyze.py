"""
P3 items 1-2: analysis of the drafter runs (cache/dr_runs.pkl.gz).

1. Decisions: how often the personalized pick differs from the population
   pick at the same state, and by how much (personal-model value V and
   population WP), for the a-gd, a-imit and self-play trajectories.
2. Final drafts: personalized vs population drafter, paired by lobby.
3. Collapse: per-player effective hero pool (exp entropy of the heroes
   chosen for him across 30 contexts) for both drafters vs his real play
   distribution; share of choices on his most-played heroes.
4. Emergent meta: hero pick shares under personalized self-play, population
   self-play and the real drafts of the same 1,000 lobbies; risers/fallers;
   figures.
5. Off-role: share of picks that are off-role (fine role < 10%) for the
   player they go to, and the mean role share of assigned players.
6. Realized value (no model in between): in the real drafts of 5,724 held-
   out lobbies, each real pick's percentile in the personalized and in the
   population drafter's ranking at the real state; team residual win rate
   y - WP_pop (the population model's prediction of the real game) by
   agreement quintile.
7. Imitation vs outcome: imitation top-1 vs personalized/population top-1 at
   the real states; when the real pick matches one but not the other, the
   population-model residual of the picking team.

Run (from training/):
  OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_dr_analyze.py
Outputs: results/p3_dr_analyze.json, results/fig_dr_meta.png, results/fig_dr_collapse.png
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

RUNS = os.path.join(C.CACHE, "dr_runs.pkl.gz")


def ci_mean(x, rng, n=1000):
    x = np.asarray(x, float)
    bs = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def entropy_eff(counts):
    c = np.asarray(counts, float)
    c = c[c > 0]
    p = c / c.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    rng = np.random.RandomState(0)
    z = pickle.load(gzip.open(RUNS))
    d = C.load_slots()
    names = np.array([str(x) for x in d["hero_names"]])
    meta = C.hero_meta(d["hero_names"])
    T = D.load_personal()
    L = D.load_lobbies()
    full = [l for l in z["lobbies"] if "b|pers" in l]
    alll = z["lobbies"]
    out = {"n_full_lobbies": len(full), "n_realized_lobbies": len(alll)}

    # ---------- 1. decisions
    dec = {}
    for key in ("a-gd|pers", "a-imit|pers", "b|pers", "a-gd|pop", "b|pop"):
        rows = []
        for l in full:
            for dd in l[key]["decisions"]:
                i_p = int(np.flatnonzero(dd["cand"] == dd["pers"])[0])
                i_q = int(np.flatnonzero(dd["cand"] == dd["pop"])[0])
                rows.append((dd["pers"] != dd["pop"], dd["V"][i_p] - dd["V"][i_q], dd["wp"][i_p] - dd["wp"][i_q],
                             len(dd["cand"])))
        a = np.array(rows, float)
        dv, dw = a[:, 1], a[:, 2]
        diff = a[:, 0] == 1
        dec[key] = {"decisions": int(len(a)), "share_differ": float(diff.mean()),
                    "mean_candidates": float(a[:, 3].mean()),
                    "dV_pp_when_differ": ci_mean(100 * dv[diff], rng),
                    "dWPpop_pp_when_differ": ci_mean(100 * dw[diff], rng),
                    "dV_pp_all": ci_mean(100 * dv, rng),
                    "share_dV_over_1pp": float((dv > 0.01).mean()),
                    "share_dV_over_3pp": float((dv > 0.03).mean())}
        print("decisions", key, json.dumps(dec[key]), flush=True)
    out["decisions"] = dec

    # ---------- 2. final drafts (controlled team perspective)
    fin = {}
    for opp in ("gd", "imit"):
        dV, dW = [], []
        for l in full:
            c = l["ctrl"]
            vp, vq = l[f"a-{opp}|pers"]["V"], l[f"a-{opp}|pop"]["V"]
            wp_, wq = l[f"a-{opp}|pers"]["wp"], l[f"a-{opp}|pop"]["wp"]
            if c == 1:
                vp, vq, wp_, wq = 1 - vp, 1 - vq, 1 - wp_, 1 - wq
            dV.append(vp - vq)
            dW.append(wp_ - wq)
        fin[f"vs {opp}"] = {"dV_pp": ci_mean(100 * np.array(dV), rng), "dWPpop_pp": ci_mean(100 * np.array(dW), rng)}
        # controlled team value vs the real team's value in the same lobby
        vr = [(l["real"]["V"] if l["ctrl"] == 0 else 1 - l["real"]["V"]) for l in full]
        vpers = [(l[f"a-{opp}|pers"]["V"] if l["ctrl"] == 0 else 1 - l[f"a-{opp}|pers"]["V"]) for l in full]
        vpop = [(l[f"a-{opp}|pop"]["V"] if l["ctrl"] == 0 else 1 - l[f"a-{opp}|pop"]["V"]) for l in full]
        fin[f"vs {opp}"]["mean_V_pers_drafter"] = float(np.mean(vpers))
        fin[f"vs {opp}"]["mean_V_pop_drafter"] = float(np.mean(vpop))
        fin[f"vs {opp}"]["mean_V_real_draft"] = float(np.mean(vr))
    out["final_drafts"] = fin
    print("final", json.dumps(fin), flush=True)

    # ---------- 3. collapse
    per = {}
    for r in z["collapse"]:
        per.setdefault(r["player"], {"pers": [], "pop": []})
        per[r["player"]]["pers"].append(r["pers"])
        per[r["player"]]["pop"].append(r["pop"])
    col = []
    for p, v in per.items():
        n = T["n"][T["pos"][int(p)]]
        top3 = set(np.argsort(-n)[:3])
        top1 = int(np.argmax(n))
        pool_size = int((n > 0).sum())
        # real effective pool: exp entropy of play counts, and of a sample of
        # 30 games drawn from them (matches the 30-context resolution)
        sim = [entropy_eff(np.bincount(rng.choice(NUM_HEROES, 30, p=n / n.sum()), minlength=NUM_HEROES)) for _ in range(20)]
        col.append({"pers_eff": entropy_eff(np.bincount(v["pers"], minlength=NUM_HEROES)),
                    "pop_eff": entropy_eff(np.bincount(v["pop"], minlength=NUM_HEROES)),
                    "real_eff_all": entropy_eff(n), "real_eff_30": float(np.mean(sim)),
                    "pers_distinct": len(set(v["pers"])), "pop_distinct": len(set(v["pop"])),
                    "pool": pool_size,
                    "pers_top1": float(np.mean([h == top1 for h in v["pers"]])),
                    "pop_top1": float(np.mean([h == top1 for h in v["pop"]])),
                    "pers_top3": float(np.mean([h in top3 for h in v["pers"]])),
                    "pop_top3": float(np.mean([h in top3 for h in v["pop"]])),
                    "real_top1": float(n[top1] / n.sum()),
                    "real_top3": float(n[list(top3)].sum() / n.sum())})
    ca = {k: [float(np.median([c[k] for c in col])), float(np.mean([c[k] for c in col]))] for k in col[0]}
    out["collapse"] = {"players": len(col), "contexts_per_player": int(np.mean([len(v["pers"]) for v in per.values()])),
                       "median_mean": ca}
    print("collapse", json.dumps(out["collapse"]), flush=True)

    # ---------- 4. emergent meta
    def counts(key, which=None):
        c = np.zeros(NUM_HEROES)
        for l in full:
            dr = l[key]
            teams = ("t0", "t1") if which is None else (("t0",) if l["ctrl"] == 0 else ("t1",))
            for t in teams:
                for h in dr[t].values():
                    c[h] += 1
        return c
    cp, cq, cr = counts("b|pers"), counts("b|pop"), counts("real")
    ap_, aq_ = counts("a-gd|pers", "ctrl"), counts("a-gd|pop", "ctrl")
    ar = np.zeros(NUM_HEROES)
    for l in full:
        for h in (l["real"]["t0"] if l["ctrl"] == 0 else l["real"]["t1"]).values():
            ar[h] += 1
    sh = lambda c: c / c.sum()
    out["meta"] = {"effective_heroes": {"self-play personalized": entropy_eff(cp), "self-play population": entropy_eff(cq),
                                        "real drafts (same lobbies)": entropy_eff(cr),
                                        "vs GD personalized (controlled team)": entropy_eff(ap_),
                                        "vs GD population (controlled team)": entropy_eff(aq_),
                                        "real (controlled team)": entropy_eff(ar)},
                   "distinct_heroes": {"self-play personalized": int((cp > 0).sum()),
                                       "self-play population": int((cq > 0).sum()), "real": int((cr > 0).sum())},
                   "top10_share": {"self-play personalized": float(np.sort(sh(cp))[-10:].sum()),
                                   "self-play population": float(np.sort(sh(cq))[-10:].sum()),
                                   "real": float(np.sort(sh(cr))[-10:].sum())}}
    lr = np.log2((sh(cp) + 1e-3) / (sh(cq) + 1e-3))
    order = np.argsort(-lr)
    tab = lambda i: {"hero": names[i], "role": C.BLIZZ_ROLES[meta["blizz"][i]],
                     "pers_share_pct": float(100 * sh(cp)[i]), "pop_share_pct": float(100 * sh(cq)[i]),
                     "real_share_pct": float(100 * sh(cr)[i])}
    out["meta"]["risers_vs_population"] = [tab(i) for i in order[:12]]
    out["meta"]["fallers_vs_population"] = [tab(i) for i in order[::-1][:12]]
    out["meta"]["top_population_selfplay"] = [tab(i) for i in np.argsort(-cq)[:12]]
    out["meta"]["top_personalized_selfplay"] = [tab(i) for i in np.argsort(-cp)[:12]]
    out["meta"]["top_real"] = [tab(i) for i in np.argsort(-cr)[:12]]
    out["meta"]["corr_share"] = {"pers vs real": float(np.corrcoef(sh(cp), sh(cr))[0, 1]),
                                 "pop vs real": float(np.corrcoef(sh(cq), sh(cr))[0, 1]),
                                 "pers vs pop": float(np.corrcoef(sh(cp), sh(cq))[0, 1])}
    role_sh = {}
    for nm, c in (("self-play personalized", cp), ("self-play population", cq), ("real", cr)):
        role_sh[nm] = {C.BLIZZ_ROLES[k]: float(sh(c)[meta["blizz"] == k].sum()) for k in range(6)}
    out["meta"]["role_shares"] = role_sh
    print("meta", json.dumps({k: v for k, v in out["meta"].items() if k in ("effective_heroes", "top10_share",
                                                                          "corr_share", "role_shares")}), flush=True)

    # ---------- 5. off-role
    def offrole(key):
        o, rs, n = 0, [], 0
        fine = meta["fine"]
        for l in full:
            for t in ("t0", "t1"):
                for r, h in l[key][t].items():
                    p = T["pos"][int(r)]
                    o += T["off"][p][h]
                    nn = T["n"][p]
                    rs.append(nn[fine == fine[h]].sum() / max(nn.sum(), 1))
                    n += 1
        return {"off_role_share": float(o / n), "mean_role_share_of_assigned_player": float(np.mean(rs)),
                "share_role_share_under_10pct": float(np.mean(np.array(rs) < 0.10))}
    out["off_role"] = {k: offrole(k) for k in ("b|pers", "b|pop", "a-gd|pers", "a-gd|pop", "real")}
    print("off-role", json.dumps(out["off_role"]), flush=True)

    # ---------- 6. realized value
    w = np.load(C.WP)
    o = np.argsort(w["replay_ids"])
    y_all, wp_all = w["y"][o], w["wp0"][o]
    recs = []
    for l in alll:
        g = L["g"][l["gi"]]
        for t in (0, 1):
            pp, pq, pd = [], [], []
            for dd in l["real"]["decisions"]:
                if dd["team"] != t or not np.isin(dd["actual"], dd["cand"]):
                    continue  # real hero outside the drafter's candidates: no rank
                i = int(np.flatnonzero(dd["cand"] == dd["actual"])[0])
                nc = len(dd["cand"])
                if nc < 2:
                    continue
                pp.append((np.sum(dd["V"] < dd["V"][i])) / (nc - 1))
                pq.append((np.sum(dd["wp"] < dd["wp"][i])) / (nc - 1))
                pd_ = dd["V"] - dd["wp"]
                pd.append((np.sum(pd_ < pd_[i])) / (nc - 1))
            yt = y_all[g] if t == 0 else 1 - y_all[g]
            wt = wp_all[g] if t == 0 else 1 - wp_all[g]
            recs.append((np.mean(pp), np.mean(pq), np.mean(pd), yt - wt, yt, wt))
    A = np.array(recs)
    rv = {"teams": int(len(A))}
    for j, nm in ((0, "personalized drafter agreement"), (1, "population drafter agreement"),
                  (2, "personal-component agreement (V - WP ranking)")):
        qs = np.percentile(A[:, j], [20, 40, 60, 80])
        b = np.searchsorted(qs, A[:, j])
        rv[nm] = {f"quintile {q + 1}": {"mean_agreement": float(A[b == q, j].mean()),
                                        "residual_win_rate_pp": ci_mean(100 * A[b == q, 3], rng)} for q in range(5)}
        top, bot = A[b == 4, 3], A[b == 0, 3]
        diffs = [top[rng.randint(0, len(top), len(top))].mean() - bot[rng.randint(0, len(bot), len(bot))].mean()
                 for _ in range(1000)]
        rv[nm]["top_minus_bottom_pp"] = [float(100 * (top.mean() - bot.mean())), float(100 * np.percentile(diffs, 2.5)),
                                         float(100 * np.percentile(diffs, 97.5))]
        X = np.column_stack([np.ones(len(A)), A[:, j] - A[:, j].mean()])
        beta = np.linalg.lstsq(X, A[:, 3], rcond=None)[0]
        rv[nm]["slope_pp_per_unit_agreement"] = float(100 * beta[1])
    # personal agreement controlling for population agreement
    X = np.column_stack([np.ones(len(A)), A[:, 2] - A[:, 2].mean(), A[:, 1] - A[:, 1].mean()])
    beta = np.linalg.lstsq(X, A[:, 3], rcond=None)[0]
    bs = []
    for _ in range(500):
        ii = rng.randint(0, len(A), len(A))
        bs.append(np.linalg.lstsq(X[ii], A[ii, 3], rcond=None)[0][1])
    rv["personal_component_slope_controlling_population_pp"] = [float(100 * beta[1]), float(100 * np.percentile(bs, 2.5)),
                                                                float(100 * np.percentile(bs, 97.5))]
    out["realized"] = rv
    print("realized", json.dumps({k: (v.get("top_minus_bottom_pp") if isinstance(v, dict) else v) for k, v in rv.items()}), flush=True)

    # ---------- 7. imitation vs outcome at the real states
    import p3_dr_imitation as I
    gd = D.GDPolicy(d["hero_names"])
    iw = np.load(I.OUT)["w"]
    rows_needed, items = set(), []
    for l in alll:
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
                items.append((t0.copy(), t1.copy(), bans.copy(), mo, to, k, (t0 + t1 + bans) == 0, row, h, dd, l["gi"], tm))
                rows_needed.add(int(row))
                (t0 if tm == 0 else t1)[h] = 1
            else:
                bans[h] = 1
    rows_needed = np.array(sorted(rows_needed))
    rec = I.recency_features(d, rows_needed)
    fake = {"row": rows_needed, "recpos": np.arange(len(rows_needed)), "lp": np.zeros((len(rows_needed), NUM_HEROES), np.float32)}
    Xp = I.feature_tensor(fake, T, rec, meta)
    ipers = {int(r): (Xp[i, :, 1:] * iw[1:]).sum(-1) for i, r in enumerate(rows_needed)}
    cls = []
    for s in range(0, len(items), 4096):
        ch = items[s:s + 4096]
        lp = gd.logprobs(np.array([c[0] for c in ch]), np.array([c[1] for c in ch]), np.array([c[2] for c in ch]),
                         np.array([c[3] for c in ch]), np.array([c[4] for c in ch]),
                         np.array([c[5] for c in ch], np.float32), np.ones(len(ch), np.float32),
                         np.array([c[6] for c in ch]))
        for c, l_ in zip(ch, lp):
            dd = c[9]
            u = iw[0] * l_ + ipers[int(c[7])]
            u = np.where(c[6], u, -1e9)
            imit_top = int(np.argmax(u))
            # imitation's best among the drafter's candidate set, for a like-for-like comparison
            imit_top_c = int(dd["cand"][np.argmax(u[dd["cand"]])])
            g = L["g"][c[10]]
            yt = y_all[g] if c[11] == 0 else 1 - y_all[g]
            wt = wp_all[g] if c[11] == 0 else 1 - wp_all[g]
            cls.append((c[8], imit_top, imit_top_c, dd["pers"], dd["pop"], yt - wt))
    Cc = np.array(cls)
    actual, it, itc, pe, po, res_ = Cc.T
    res_ = res_.astype(float)
    im = {"picks": int(len(Cc)),
          "real_pick_equals_imitation_top1": float((actual == it).mean()),
          "real_pick_equals_personalized_top1": float((actual == pe).mean()),
          "real_pick_equals_population_top1": float((actual == po).mean()),
          "imitation_top1_equals_personalized_top1 (same candidate set)": float((itc == pe).mean()),
          "imitation_top1_equals_population_top1 (same candidate set)": float((itc == po).mean()),
          "personalized_top1_equals_population_top1": float((pe == po).mean())}
    groups = {"matches imitation only": (actual == itc) & (actual != pe),
              "matches personalized only": (actual == pe) & (actual != itc),
              "matches both": (actual == pe) & (actual == itc),
              "matches neither": (actual != pe) & (actual != itc)}
    im["team_residual_by_pick_class_pp"] = {k: {"picks": int(m.sum()), "residual": ci_mean(100 * res_[m], rng)}
                                            for k, m in groups.items()}
    out["imitation_vs_outcome"] = im
    print("imitation vs outcome", json.dumps(im), flush=True)

    # ---------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    role_col = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#9c755f"]
    fig, axs = plt.subplots(1, 3, figsize=(22, 7.5))
    for ax, (xa, ya, xl, yl) in zip(axs[:2], ((sh(cq), sh(cp), "population self-play", "personalized self-play"),
                                              (sh(cr), sh(cp), "real drafts (same lobbies)", "personalized self-play"))):
        for i in range(NUM_HEROES):
            ax.scatter(100 * xa[i], 100 * ya[i], color=role_col[meta["blizz"][i]], s=18)
        lim = 100 * max(xa.max(), ya.max()) * 1.05
        ax.plot([0, lim], [0, lim], color="#999", lw=0.7)
        lab = set(np.argsort(-np.abs(np.log((ya + 1e-3) / (xa + 1e-3))))[:14]) | set(np.argsort(-ya)[:8]) | set(np.argsort(-xa)[:8])
        for i in lab:
            ax.annotate(names[i], (100 * xa[i], 100 * ya[i]), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel(f"{xl}: pick share (%)")
        ax.set_ylabel(f"{yl}: pick share (%)")
        ax.set_title(f"Hero pick shares, {len(full)} held-out lobbies", fontsize=10)
    x = np.arange(6)
    for j, (nm, c) in enumerate((("real", cr), ("population self-play", cq), ("personalized self-play", cp))):
        axs[2].bar(x + (j - 1) * 0.27, [100 * sh(c)[meta["blizz"] == k].sum() for k in range(6)], 0.27, label=nm)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(C.BLIZZ_ROLES, rotation=20)
    axs[2].set_ylabel("share of picks (%)")
    axs[2].set_title("Role shares of picks", fontsize=10)
    axs[2].legend(fontsize=8)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8) for c in role_col]
    fig.legend(handles, C.BLIZZ_ROLES, loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(C.RESULTS, "fig_dr_meta.png"), dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.hist([c["real_eff_30"] for c in col], bins=25, alpha=0.5, label="real play (30 sampled games)")
    ax.hist([c["pop_eff"] for c in col], bins=25, alpha=0.5, label="population drafter (30 contexts)")
    ax.hist([c["pers_eff"] for c in col], bins=25, alpha=0.5, label="personalized drafter (30 contexts)")
    ax.set_xlabel("effective hero pool per player (exp entropy)")
    ax.set_ylabel("players")
    ax.set_title(f"Collapse check, {len(col)} players", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.RESULTS, "fig_dr_collapse.png"), dpi=120)
    with open(os.path.join(C.RESULTS, "p3_dr_analyze.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("done")


if __name__ == "__main__":
    main()
