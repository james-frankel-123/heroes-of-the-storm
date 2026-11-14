"""
Exploitation frontier: how far each drafter leaves the support of human teams, against how
strong it is once the judges' support bias is corrected. Compares the support floor (one
threshold sweep on the MCTS-trained policy, plus the floor-trained MCTS when it exists) with
the generic offline-RL conservatism knobs: CQL alpha, IQL expectile/temperature, MCQ, and the
ensemble lower confidence bound at search time.

Strength: mean win probability against the eight learned tournament strategies (mcts,
constrained_mcts, enriched, constrained_greedy, k_oof, gourdeau, gourdeau_disc, gd), both
seats, 200 drafts per ordered pair; an entrant in that set is scored against the other seven.
Judged raw and with the binned support correction (support_mitigation.judge_corr.json); the
difference raw - corrected is the share of the score that rides on the judges' support bias.
SE: per-draft variance within pairs, propagated to the mean over opponents.

Support: the strategy's final teams against the training-split corpus: share with a rarest
triple below the 5th percentile of held-out human teams (25), median rarest triple, mean pair
affinity (PMI), share seen exactly.

certify: the agent's OWN leaf value (enriched leak-free WP, deploy statistics) on the
post-snapshot NODRIFT games, which no judge correction or agent was fit on: gap y - p by the
team's rarest-triple bin, netted against the value's overall calibration curve, SE clustered
by game. The certified threshold is the lowest bin edge from which every bin up has at least
MIN_N teams and |gap| + 2 SE <= EPS: below it the value function's error is either measured
too large or not measurable from real games.

Subcommands (from training/, p1site env):
  tour --pool N [--td-only]  floor-sweep pairs (support_mitigation workers) then the untoured CQL/IQL
                    checkpoints (td_tour workers), skipping finished pairs
  certify           site/results/ood_frontier/certify.json
  score             site/results/ood_frontier/frontier.json (+ table on stdout)
"""
import os
import sys
import json
import glob
import argparse
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core

OUT = os.path.join(core.RESULTS, "ood_frontier")
OPP8 = ["mcts", "constrained_mcts", "enriched", "constrained_greedy", "k_oof", "gourdeau", "gourdeau_disc", "gd"]
SWEEP = [5, 10, 50, 100, 200]  # 25 is constrained_mcts_sf
TD_SWEEP = ["cql_td_a0.5", "cql_td_a2.0", "cql_td_a0.1", "cql_td_a5.0", "iql_td_t0.9_b3.0", "iql_td_t0.7_b10.0", "iql_td_t0.7_b30.0",
            "iql_td_t0.9_b30.0"]
DIRS = ["tournament", "wr_baseline/tournament", "masked_k/tournament", "td/*/tournament", "mcq_real/tournament",
        "search_agents/tournament", "support_mitigation/tournament", "support_mitigation/floor_search/tournament", "ood_frontier/tournament"]
EPS, MIN_N = 1.0, 500
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))


# ── tournament ───────────────────────────────────────────────────────────
def sweep_pairs():
    out = []
    for t in SWEEP:
        v = f"constrained_mcts_sf{t}"
        out += [f"{v}__{o}" for o in OPP8] + [f"{o}__{v}" for o in OPP8]
    from paper1_revision import support_mitigation as sm
    for v in sm.FT_NAMES:  # floor-trained MCTS policy, without and with the floor at play
        out += [f"{v}__{o}" for o in OPP8] + [f"{o}__{v}" for o in OPP8]
    out += ["constrained_mcts_ft_sf__constrained_mcts_sf", "constrained_mcts_sf__constrained_mcts_ft_sf",
            "constrained_mcts_ft_sf__constrained_wr_sf", "constrained_wr_sf__constrained_mcts_ft_sf"]
    return out


def tour(procs, n=200, td_only=False):
    from concurrent.futures import ThreadPoolExecutor
    from paper1_revision import support_mitigation as sm
    sm.triple_counts(); sm.sup_corpus()
    logs = os.path.join(core.LOGS, "ood_frontier")
    os.makedirs(logs, exist_ok=True)
    todo = [] if td_only else [p for p in sweep_pairs() if not os.path.exists(sm.path(p))]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run(p):
        with open(os.path.join(logs, f"{p}.log"), "w") as log:
            r = subprocess.run([sys.executable, "-u", sm.__file__, "pair", p, "--drafts", str(n)],
                               cwd=TRAINING_DIR, stdout=log, stderr=subprocess.STDOUT, env=env)
        print(p, "rc", r.returncode, flush=True)
    print(f"floor sweep: {len(todo)} pairs", flush=True)
    with ThreadPoolExecutor(procs) as ex:
        list(ex.map(run, todo))
    from rerun2026 import common
    for name in TD_SWEEP:
        if not os.path.exists(os.path.join(common.MODELS_DIR, "td", f"{name}.pt")):
            print("missing checkpoint", name, flush=True)
            continue
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "td_tour.py"), "--pool", str(procs),
                            "--drafts", str(n)], cwd=TRAINING_DIR, env=dict(env, TD_NAME=name))
        print(name, "rc", r.returncode, flush=True)


# ── certified threshold from the agent's own value ───────────────────────
def certify():
    import torch
    from paper1_revision import train_wp
    from paper1_revision import support_mitigation as sm
    from shared import HERO_TO_IDX
    C = sm.sup_corpus()
    games = core.gold_games("NODRIFT")
    rows = [(list(g[3]), list(g[4]), g[2], g[1]) for g in games]
    y0 = np.array([1.0 if g[6] == 0 else 0.0 for g in games])
    Xf, Xs = core.featurize(rows, core.load_stats("deploy"), nproc=int(os.environ.get("P1R_NPROC", "4")))
    m, cols = train_wp.load("enriched")
    with torch.no_grad():
        a = m(torch.tensor(Xf[:, cols])).view(-1).numpy()
        b = m(torch.tensor(Xs[:, cols])).view(-1).numpy()
    p0 = 0.5 * (a + 1 - b)
    A0 = np.array([sorted(HERO_TO_IDX[h] for h in g[3]) for g in games], np.int64)
    A1 = np.array([sorted(HERO_TO_IDX[h] for h in g[4]) for g in games], np.int64)
    m0, m1 = C.metrics(A0), C.metrics(A1)
    # team rows: each game gives both teams (team's own p and result)
    y = np.concatenate([y0, 1 - y0])
    p = np.concatenate([p0, 1 - p0])
    own = np.concatenate([m0["min3"], m1["min3"]])
    opp = np.concatenate([m1["min3"], m0["min3"]])
    gid = np.concatenate([np.arange(len(y0))] * 2)
    e = np.unique(np.percentile(p, np.linspace(0, 100, 51)))
    bb = np.clip(np.searchsorted(e, p, side="right") - 1, 0, len(e) - 2)
    calib = np.bincount(bb, y - p, len(e) - 1) / np.maximum(np.bincount(bb, minlength=len(e) - 1), 1)
    net = y - p - calib[bb]
    edges = [0, 1, 5, 10, 25, 50, 100, 200, 400, 10 ** 12]
    p5 = float(np.percentile(np.concatenate([m0["min3"], m1["min3"]]), 5))

    def g(x, sel):
        u = np.unique(gid[sel], return_inverse=True)[1]
        mu = x[sel].mean()
        return float(100 * mu), float(100 * np.sqrt((np.bincount(u, x[sel] - mu) ** 2).sum()) / sel.sum())
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (own >= lo) & (own < hi) & (opp >= 25)  # opponent inside support: the gap is ours alone
        n = int(sel.sum())
        rec = {"min3_lo": lo, "min3_hi": hi if hi < 10 ** 12 else None, "n": n}
        if n >= 30:
            rec["raw"], rec["net"] = g(y - p, sel), g(net, sel)
            rec["ok"] = bool(n >= MIN_N and abs(rec["net"][0]) + 2 * rec["net"][1] <= EPS)
        else:
            rec["ok"] = False
        bins.append(rec)
    thr = None
    for i in range(len(bins)):
        if all(b["ok"] for b in bins[i:]):
            thr = bins[i]["min3_lo"]
            break
    res = {"games": int(len(y0)), "eps_pts": EPS, "min_n": MIN_N, "human_min3_p5": p5, "bins": bins,
           "certified_threshold": thr}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "certify.json"), "w"), indent=1)
    for b in bins:
        print(b)
    print("certified threshold:", thr, "human p5:", p5)


# ── scoring ──────────────────────────────────────────────────────────────
def records():
    out = []
    for d in DIRS:
        for f in sorted(glob.glob(os.path.join(core.RESULTS, d, "*.json"))):
            R = json.load(open(f))
            if "records" not in R:
                continue
            for r in R["records"]:
                out.append((R["team0_strategy"], R["team1_strategy"], r))
    return out


def score():
    from paper1_revision import judge_ood_calibration as joc
    from paper1_revision import support_mitigation as sm
    from shared import HERO_TO_IDX
    recs = records()
    # keep drafts with at least one OPP8 side (all frontier rows are against OPP8)
    recs = [x for x in recs if x[0] in OPP8 or x[1] in OPP8]
    seen = set()
    uniq = []
    for a, b, r in recs:  # the same pair file can sit in two result dirs
        k = (a, b, r["draft"], r["game_map"], tuple(r["team0"]["picks"]), tuple(r["team1"]["picks"]))
        if k not in seen:
            seen.add(k)
            uniq.append((a, b, r))
    recs = uniq
    print(f"{len(recs):,} drafts", flush=True)
    games = [(None, r["tier"], r["game_map"], r["team0"]["picks"], r["team1"]["picks"]) for _, _, r in recs]
    sc = joc.judge_scores(games)
    C = sm.sup_corpus()
    A = lambda side: np.array([sorted(HERO_TO_IDX[h] for h in r[side]["picks"]) for _, _, r in recs], np.int64)
    m0, m1 = C.metrics(A("team0")), C.metrics(A("team1"))
    jc = json.load(open(os.path.join(sm.OUT, "judge_corr.json")))
    P = {}
    for j in joc.JUDGES:
        wb = np.array(jc["judges"][j]["w_binned"])
        P[(j, "raw")] = np.asarray(sc[j], float)
        P[(j, "corrected")] = sig(logit(sc[j]) + (sm.bin_feats(m0, jc["edges"]) - sm.bin_feats(m1, jc["edges"])) @ wb)
    S0 = np.array([a for a, _, _ in recs])
    S1 = np.array([b for _, b, _ in recs])
    names = sorted(set(S0) | set(S1))
    thr = None
    cp = os.path.join(OUT, "certify.json")
    if os.path.exists(cp):
        thr = json.load(open(cp))["certified_threshold"]
    excl = [x for x in os.environ.get("FRONTIER_EXCLUDE", "").split(",") if x]
    res = {"opponents": [o for o in OPP8 if o not in excl], "excluded": excl, "strategies": {}}
    for s in names:
        # an entrant in OPP8 never plays itself; FRONTIER_EXCLUDE drops opponents for everyone, so a
        # variant and its base are scored on the same opponent set
        opps = [o for o in OPP8 if o != s and o not in excl]
        row = {"opponents_covered": 0}
        for (j, form), v in P.items():
            means, varm = [], []
            for o in opps:
                x = np.concatenate([v[(S0 == s) & (S1 == o)], 1 - v[(S0 == o) & (S1 == s)]])
                if len(x):
                    means.append(x.mean())
                    varm.append(x.var(ddof=1) / len(x) if len(x) > 1 else 0.0)
            if len(means) == len(opps):
                row[f"{j}/{form}"] = [float(np.mean(means)), float(np.sqrt(np.sum(varm)) / len(means))]
            row["opponents_covered"] = len(means)
        if row["opponents_covered"] < len(opps):
            continue
        sel0, sel1 = S0 == s, S1 == s
        mm = {k: np.concatenate([np.asarray(m0[k])[sel0], np.asarray(m1[k])[sel1]]) for k in ("min3", "pmi", "n5")}
        row["support"] = {"teams": int(len(mm["min3"])), "min3_below_25": float((mm["min3"] < 25).mean()),
                          "min3_median": float(np.median(mm["min3"])), "pmi_mean": float(mm["pmi"].mean()),
                          "seen_exact": float((mm["n5"] > 0).mean())}
        if thr is not None:
            row["support"]["certified_share"] = float((mm["min3"] >= thr).mean())
        for j in joc.JUDGES:
            row[f"{j}/riding_on_bias_pts"] = 100 * (row[f"{j}/raw"][0] - row[f"{j}/corrected"][0])
        res["strategies"][s] = row
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "frontier" + ("_excl_" + "_".join(excl) if excl else "") + ".json"), "w"), indent=1)
    print(f"{'strategy':24s} {'<25':>5s} {'min3md':>6s} {'pmi':>6s} | {'cons raw':>8s} {'cons corr':>9s} {'gN corr':>8s} {'RN corr':>8s} {'bias':>5s}")
    for s, r in sorted(res["strategies"].items(), key=lambda kv: -kv[1]["consensus/corrected"][0]):
        u = r["support"]
        print(f"{s:24s} {u['min3_below_25']:5.2f} {u['min3_median']:6.0f} {u['pmi_mean']:6.3f} | "
              f"{r['consensus/raw'][0]:.3f}    {r['consensus/corrected'][0]:.3f}±{r['consensus/corrected'][1]:.3f} "
              f"{r['gN/corrected'][0]:.3f}    {r['RN/corrected'][0]:.3f}    {r['consensus/riding_on_bias_pts']:+.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tour", "certify", "score"])
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--drafts", type=int, default=200)
    ap.add_argument("--td-only", action="store_true")
    a = ap.parse_args()
    if a.cmd == "tour":
        tour(a.pool, a.drafts, a.td_only)
    elif a.cmd == "certify":
        certify()
    else:
        score()
