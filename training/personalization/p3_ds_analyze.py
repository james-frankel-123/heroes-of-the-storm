"""
P3 distillation, stage 5: compare drafters on the P3_DRAFTER metrics.

Drafters (personalized unless noted):
  greedy distilled     argmax of the distilled prior, no search
  greedy BC            argmax of the BC prior (non-personal baseline)
  MCTS BC prior        personalized MCTS, BC prior (p3_mcts_drafter)
  MCTS distilled prior personalized MCTS, distilled prior (p3_ds_mcts)
  MCTS population      population MCTS, BC prior
  one-step             one-step personalized search (p3_dr_drafter)
Metrics: whole-draft personal value V and WP_pop vs the paired non-personal
drafter (same lobbies and opponent seeds), realized agreement (residual of
real teams by agreement quintile, same 2,000-lobby subset for all),
collapse, self-play meta breadth, off-role rate.

Run (from training/):
  OMP_NUM_THREADS=4 nice -n 19 taskset -c 48-63 python3 personalization/p3_ds_analyze.py
Outputs: results/p3_ds_analyze.json, results/fig_ds_meta.png
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
from p3_mcts_analyze import Ctx, final_drafts, realized, eff, ci_mean, load, SUB


def lp(name):
    p = __import__("p3_mcts_core").result_path(name)
    return pickle.load(gzip.open(p)) if os.path.exists(p) else None


def meta_offrole(X, full, key):
    c = np.zeros(NUM_HEROES)
    o, n = 0, 0
    for lob, r in zip(full["lobbies"], full[key]):
        for tm, row, h in X.slot_heroes(lob, r["acts"]):
            c[h] += 1
            o += X.T["off"][X.T["pos"][row]][h]
            n += 1
    return c, o / n


def collapse_stats(X, dec, which):
    per = {}
    for d_ in dec:
        p, k, pers, h = d_
        if pers != which:
            continue
        per.setdefault(p, []).append(int(X.from_sh[h]))
    rows = []
    for p, v in per.items():
        nn = X.T["n"][X.T["pos"][int(p)]]
        top3 = set(np.argsort(-nn)[:3])
        rows.append((eff(np.bincount(v, minlength=NUM_HEROES)), np.mean([h in top3 for h in v])))
    a = np.array(rows)
    return {"effective_pool_median": float(np.median(a[:, 0])), "top3_share_mean": float(a[:, 1].mean()),
            "players": len(a)}


def merge_real(pers_res, pop_res, n):
    """pers decisions from one run + pop decisions from another, same lobby order."""
    lobs = pers_res["lobbies"][:n]
    gi_p = [l["gi"] for l in pop_res["lobbies"][:n]]
    assert gi_p == [l["gi"] for l in lobs]
    dec = [d_ for d_ in pers_res["decisions"] if d_[0] < n and d_[2] == 1] + \
          [d_ for d_ in pop_res["decisions"] if d_[0] < n and d_[2] == 0]
    return {"lobbies": lobs, "decisions": dec}


def main():
    rng = np.random.RandomState(0)
    X = Ctx()
    out = {}
    z1 = pickle.load(gzip.open(os.path.join(C.CACHE, "dr_runs.pkl.gz")))
    gis_sub = set(int(x) for x in np.r_[z1["full_set"], z1["extra_set"]][:SUB])
    g_full, g_real, g_col = lp("dsgreedy_full.pkl.gz"), lp("dsgreedy_real.pkl.gz"), lp("dsgreedy_collapse.pkl.gz")
    m_full, m_real, m_col = load("full", 400), load("real", 400), load("collapse", 400)
    d_full, d_real, d_col = lp("dsmcts_full_s400.pkl.gz"), lp("dsmcts_real_s400.pkl.gz"), lp("dsmcts_collapse_s400.pkl.gz")
    d_real100 = lp("dsmcts_real_s100.pkl.gz")
    m_real100 = load("real", 100)
    # combined dicts for paired comparisons
    dm = {"lobbies": d_full["lobbies"]}
    for k in ("a-gd", "a-imit", "b"):
        dm[f"{k}|pers"] = d_full[f"{k}|pers"]
        dm[f"{k}|pop"] = m_full[f"{k}|pop"]
    dm_vs_bc = {"lobbies": d_full["lobbies"]}
    for k in ("a-gd", "a-imit"):
        dm_vs_bc[f"{k}|pers"] = d_full[f"{k}|pers"]
        dm_vs_bc[f"{k}|pop"] = m_full[f"{k}|pers"]
    assert [l["gi"] for l in d_full["lobbies"]] == [l["gi"] for l in m_full["lobbies"]]
    tab = {}
    for name, full, real, col, which in (
            ("greedy BC (non-personal)", None, g_real, g_col, 0),
            ("greedy distilled", g_full, g_real, g_col, 1),
            ("MCTS population (BC prior)", None, m_real, m_col, 0),
            ("MCTS personalized, BC prior", m_full, m_real, m_col, 1),
            ("MCTS personalized, distilled prior", dm, merge_real(d_real, m_real, SUB) if d_real else None, d_col, 1)):
        e = {}
        if full is not None:
            for opp in ("gd", "imit"):
                e[f"vs {opp}"] = final_drafts(X, full, opp, rng)
        if real is not None:
            rr = realized(X, real, rng, gis_sub)
            e["realized_subset"] = {k: rr[k]["top_minus_bottom_pp"] for k in
                                    ("personalized agreement", "population agreement", "personal component")}
            e["realized_subset"]["teams"] = rr["teams"]
        if col is not None:
            e["collapse"] = collapse_stats(X, col["decisions"], which)
        tab[name] = e
        print(name, json.dumps(e), flush=True)
    # distilled MCTS vs BC-prior MCTS directly
    tab["MCTS distilled prior minus MCTS BC prior (both personalized)"] = {
        f"vs {o}": final_drafts(X, dm_vs_bc, o, rng) for o in ("gd", "imit")}
    # greedy real on all lobbies
    rr = realized(X, g_real, rng)
    tab["greedy distilled"]["realized_all"] = rr["personalized agreement"]["top_minus_bottom_pp"]
    tab["greedy BC (non-personal)"]["realized_all"] = rr["population agreement"]["top_minus_bottom_pp"]
    tab["greedy distilled"]["realized_all_personal_component"] = rr["personal component"]["top_minus_bottom_pp"]
    if d_real100 is not None and m_real100 is not None:
        rr = realized(X, merge_real(d_real100, m_real100, SUB), rng, gis_sub)
        tab["MCTS personalized, distilled prior"]["realized_subset_100sims"] = rr["personalized agreement"]["top_minus_bottom_pp"]
    out["drafters"] = tab
    # meta breadth and off-role (self-play and vs GD)
    creal = np.zeros(NUM_HEROES)
    for lob in m_full["lobbies"]:
        for tm, row, h in X.slot_heroes(lob, lob["acts_real"]):
            creal[h] += 1
    sh = lambda c: c / c.sum()
    meta = {"real": {"effective_heroes": eff(creal), "top10_share": float(np.sort(sh(creal))[-10:].sum())}}
    shares = {"real": sh(creal)}
    for nm, full, key in (("greedy BC self-play", g_full, "b|pop"), ("greedy distilled self-play", g_full, "b|pers"),
                          ("MCTS population self-play", m_full, "b|pop"),
                          ("MCTS personalized BC prior self-play", m_full, "b|pers"),
                          ("MCTS personalized distilled prior self-play", d_full, "b|pers"),
                          ("greedy distilled vs GD", g_full, "a-gd|pers"),
                          ("MCTS distilled prior vs GD", d_full, "a-gd|pers"),
                          ("MCTS BC prior vs GD", m_full, "a-gd|pers")):
        c, o = meta_offrole(X, full, key)
        shares[nm] = sh(c)
        meta[nm] = {"effective_heroes": eff(c), "top10_share": float(np.sort(sh(c))[-10:].sum()),
                    "corr_with_real": float(np.corrcoef(sh(c), sh(creal))[0, 1]), "off_role": float(o)}
    out["meta"] = meta
    for k, v in meta.items():
        print("meta", k, json.dumps(v), flush=True)
    a, b = shares["MCTS personalized distilled prior self-play"], shares["MCTS personalized BC prior self-play"]
    lr = np.log2((a + 1e-3) / (b + 1e-3))
    t = lambda i: {"hero": X.names[i], "distilled_pct": float(100 * a[i]), "bc_prior_pct": float(100 * b[i]),
                   "real_pct": float(100 * shares["real"][i])}
    out["risers_distilled_vs_bc_prior"] = [t(i) for i in np.argsort(-lr)[:12]]
    out["fallers_distilled_vs_bc_prior"] = [t(i) for i in np.argsort(lr)[:12]]
    out["top_distilled_selfplay"] = [t(i) for i in np.argsort(-a)[:10]]
    # figure: pick-share scatter distilled vs BC prior, and breadth bars
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    role_col = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#9c755f"]
    fig, axs = plt.subplots(1, 3, figsize=(22, 7))
    for ax, (xa, xl) in zip(axs[:2], ((b, "MCTS personalized, BC prior: self-play pick share (%)"),
                                      (shares["real"], "real drafts (same lobbies): pick share (%)"))):
        for i in range(NUM_HEROES):
            ax.scatter(100 * xa[i], 100 * a[i], color=role_col[X.meta["blizz"][i]], s=18)
        lim = 100 * max(xa.max(), a.max()) * 1.05
        ax.plot([0, lim], [0, lim], color="#999", lw=0.7)
        lab = set(np.argsort(-np.abs(np.log((a + 1e-3) / (xa + 1e-3))))[:14]) | set(np.argsort(-a)[:8])
        for i in lab:
            ax.annotate(X.names[i], (100 * xa[i], 100 * a[i]), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel(xl)
        ax.set_ylabel("MCTS personalized, distilled prior: self-play pick share (%)")
    names = ["real", "greedy BC self-play", "greedy distilled self-play", "MCTS population self-play",
             "MCTS personalized BC prior self-play", "MCTS personalized distilled prior self-play"]
    axs[2].barh(range(len(names)), [meta[n]["effective_heroes"] for n in names], color="#4e79a7")
    axs[2].set_yticks(range(len(names)))
    axs[2].set_yticklabels(names, fontsize=8)
    axs[2].set_xlabel("effective number of heroes picked (exp entropy)")
    axs[2].set_title("Meta breadth, 1,000 held-out lobbies", fontsize=10)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8) for c in role_col]
    fig.legend(handles, C.BLIZZ_ROLES, loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(C.RESULTS, "fig_ds_meta.png"), dpi=120)
    with open(os.path.join(C.RESULTS, "p3_ds_analyze.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("done")


if __name__ == "__main__":
    main()
