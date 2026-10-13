"""Baseline comparison table (WAVE0_BRIEF item 7) -> paper/hypothesis_system/results/baselines.md

  python -m training.scilearn.baselines.report

Reads per-game predictions of every baseline (A0, K0, E from the harness's remote jobs;
N from fits/neural_n) and scores them on the same games: Dpre, G, Dpost, O2 onset slices
and O5 rare compositions on Dpost. Paired differences against N use the week x player
cluster bootstrap. Harness and coordinator only: never copied into a proposer view.
"""
import json

import numpy as np
from scipy.special import logit

from training.scilearn import config as C
from training.scilearn import data as D
from training.scilearn import evaluator as E
from training.scilearn import tranches as T
from training.scilearn.fit import nll

OUT = C.DOCS / "results"
JOBS = C.PRIVATE / "jobs"
NEURAL = C.PRIVATE / "fits" / "neural_n"


def sources():
    """model -> {tranche: (replay_id, eta)}; each entry notes its fit tranches."""
    src = {}
    reg = C.PRIVATE / "baseline_jobs.json"
    jobs = json.loads(reg.read_text()) if reg.exists() else {}
    for label, j in jobs.items():
        d = JOBS / j["job"] / "out"
        for tr in j["eval"]:
            f = d / f"eta_{j['model']}_{tr}.npz"
            if f.exists():
                z = np.load(f)
                src.setdefault(label, {})[tr] = (z["replay_id"], z["eta"])
    for m, lab in (("n_drift", "N (drift-aware MLP)"), ("p1_site_enriched", "P1 site enriched MLP")):
        for tr in ("Dpre", "G", "Dpost"):
            f = NEURAL / f"{m}_{tr}.npz"
            if f.exists():
                z = np.load(f)
                p = np.clip(z["p"].astype(float), 1e-6, 1 - 1e-6)
                src.setdefault(lab, {})[tr] = (z["replay_id"], logit(p))
    return src, jobs


def rare_mask(g, meta, S):
    roles = np.asarray(meta["hero_roles"])
    def codes(h):
        r = roles[h]
        return (np.stack([(r == j).sum(1) for j in range(6)], 1) * (6 ** np.arange(6))).sum(1)
    sc = np.concatenate([codes(S["heroes"][:, :5]), codes(S["heroes"][:, 5:])])
    u, c = np.unique(sc, return_counts=True)
    share = dict(zip(u.tolist(), (c / c.sum()).tolist()))
    a, b = codes(g["heroes"][:, :5]), codes(g["heroes"][:, 5:])
    return np.array([share.get(int(x), 0) <= 0.002 or share.get(int(y), 0) <= 0.002 for x, y in zip(a, b)])


def main():
    src, jobs = sources()
    g_all, meta, y_all, pl = D.load(tranches=["S", "Dpre", "G", "Dpost"], labels=True, players=True)
    S = {k: v[g_all["tranche"] == T.CODES["S"]] for k, v in g_all.items()}
    pos = {int(r): i for i, r in enumerate(g_all["replay_id"])}
    res = {"tranches": {}, "o2": {}, "o5": {}, "fits": jobs}
    n_label = "N (drift-aware MLP)"
    lines = []
    for tr in ("Dpre", "G", "Dpost"):
        m = g_all["tranche"] == T.CODES[tr]
        ids = g_all["replay_id"][m]
        y = y_all[m].astype(float)
        blocks = E.clusters(g_all["game_ts"][m], pl["player_key"][m])
        etas = {}
        for lab, d in src.items():
            if tr not in d:
                continue
            rid, eta = d[tr]
            order = np.argsort(rid)
            idx = np.searchsorted(rid[order], ids)
            if not np.array_equal(rid[order][np.clip(idx, 0, len(rid) - 1)], ids):
                continue
            etas[lab] = eta[order][idx]
        res["tranches"][tr] = {}
        for lab, eta in etas.items():
            mt = E.metrics(eta, y)
            ll = nll(eta, y)
            se, _ = E.block_se(ll, blocks)
            mt["logloss_se"] = se
            if lab != n_label and n_label in etas:
                pg = E.paired_gain(etas[n_label], eta, y, blocks)
                mt["vs_N_logloss_diff"] = -pg["gain"]      # positive = worse than N
                mt["vs_N_se"] = pg["se"]
                mt["vs_N_acc_pp"] = 100 * (mt["acc"] - float(((etas[n_label] > 0) == (y > 0)).mean()))
            res["tranches"][tr][lab] = mt
        if tr == "Dpost":
            rare = rare_mask({k: v[m] for k, v in g_all.items()}, meta, S)
            res["o5"] = {"n": int(rare.sum()), "share": float(rare.mean()),
                         "models": {lab: E.metrics(eta[rare], y[rare]) for lab, eta in etas.items()}}
            b = g_all["build"][m]
            ts = g_all["game_ts"][m]
            for bb in np.unique(b):
                bi = np.flatnonzero(b == bb)
                bi = bi[np.argsort(ts[bi], kind="stable")]
                for k in (1000, 10000):
                    sl = bi[:k]
                    res["o2"][f"{meta['builds'][bb]} first {k // 1000}K"] = {
                        lab: {"n": int(len(sl)), "logloss": float(nll(eta[sl], y[sl]).mean()),
                              "acc": float(((eta[sl] > 0) == (y[sl] > 0)).mean())} for lab, eta in etas.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baselines.json").write_text(json.dumps(res, indent=1, default=float))
    write_md(res, jobs)
    print(json.dumps({tr: {k: round(v["logloss"], 5) for k, v in d.items()} for tr, d in res["tranches"].items()}, indent=1))


def write_md(res, jobs):
    L = ["# Baselines: LLM-free additive models and the frozen neural WP", "",
         "Generated by `training/scilearn/baselines/report.py`. Harness and coordinator only; never shown to proposer sessions.",
         "Every model is scored on the same games. Log loss is the gate metric. \"vs N\" is the paired log-loss difference against the drift-aware MLP (positive = worse than N), with a week x player-cluster bootstrap SE.", ""]
    for tr in ("Dpost", "Dpre", "G"):
        d = res["tranches"].get(tr, {})
        if not d:
            continue
        n = next(iter(d.values()))["n"]
        L += [f"## {tr} ({n:,} games)", "",
              "| Model | Fit on | Log loss (SE) | Acc. | vs N log loss (SE) | vs N acc. | Slope | Intercept |",
              "|---|---|---|---|---|---|---|---|"]
        for lab, mt in sorted(d.items(), key=lambda kv: kv[1]["logloss"]):
            fit_on = jobs.get(lab, {}).get("fit_on", "see note")
            vs = f"{mt['vs_N_logloss_diff']:+.5f} ({mt['vs_N_se']:.5f})" if "vs_N_logloss_diff" in mt else "ref"
            va = f"{mt['vs_N_acc_pp']:+.2f}pp" if "vs_N_acc_pp" in mt else "ref"
            L.append(f"| {lab} | {fit_on} | {mt['logloss']:.5f} ({mt['logloss_se']:.5f}) | {100 * mt['acc']:.2f}% | {vs} | {va} | {mt['cal_slope']:.3f} | {mt['cal_intercept']:+.3f} |")
        L.append("")
    if res.get("o5"):
        o5 = res["o5"]
        L += [f"## O5 rare compositions on Dpost ({o5['n']:,} games, {100 * o5['share']:.1f}%)", "",
              "A game is rare if either team's role multiset holds at most 0.2% of S teams.", "",
              "| Model | Log loss | Acc. | Slope |", "|---|---|---|---|"]
        for lab, mt in sorted(o5["models"].items(), key=lambda kv: kv[1]["logloss"]):
            L.append(f"| {lab} | {mt['logloss']:.5f} | {100 * mt['acc']:.2f}% | {mt['cal_slope']:.3f} |")
        L.append("")
    if res.get("o2"):
        labs = sorted({lab for v in res["o2"].values() for lab in v})
        L += ["## O2 patch onset on Dpost (log loss / accuracy)", "",
              "| Slice | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs)]
        for sl, v in res["o2"].items():
            L.append(f"| {sl} | " + " | ".join(f"{v[l]['logloss']:.4f} / {100 * v[l]['acc']:.1f}%" if l in v else "-" for l in labs) + " |")
        L.append("")
    L += ["## Notes", "",
          "- N is the population MLP only (no P3 personal layer). Its weights stop at 2.55.14.95918, so every tranche here is out of time for it. P1 site enriched is in sample on Dpre (see `baselines_neural.md`).",
          "- A0, K0 and E are fit by the harness on the remote workers. \"S\" fits score Dpre, G and Dpost; \"S+Dpre+G\" fits score Dpost only. Fits never read a scored row (checked by `evaluator.check_disjoint`).",
          "- O1 (2.57), O4 (Xal'atath) and the C-based comparisons wait for the day-14 C batch. O3 (leave one hero out) and the S rolling-origin backtests are not run yet.", ""]
    (OUT / "baselines.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
