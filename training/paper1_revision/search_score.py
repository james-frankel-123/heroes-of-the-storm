"""
Scoring for the decision-time search agents (search_eval.py outputs).

  score_bench       kernel-benchmark mirror vs the kernel's own F_oof_s0 bench
                    files (results/mcts_bench/new__F_oof_s0__T{0,1}.json):
                    v1 judges (bench_mcts.score_drafts) and the composition-
                    corrected v2 judges exactly as comp_rescore.mcts() (gN =
                    own-composition rebuild, judges_v2.correct)
  score_tournament  standings of the new agents over their ordered pairs
                    (score_tournament.pair_stats: config-clustered SEs),
                    v1 (score_tournament.score_rows) and v2 (comp_rescore
                    tournament(): rebuilt gN + judges_v2.correct); head to head
                    vs mcts and constrained_mcts; a 13-strategy table (the 11 +
                    mcts_search + mcts_lcb) from the stored scores of the
                    existing 110 pairs plus the new pairs
  score_rich        Table IV metrics (greedy_evals.aggregate) + v1/v2 refs
Outputs: results/search_agents/{bench_validation,tournament_standings,rich}.json
"""
import os
import sys
import json
import glob
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core
from paper1_revision import search_eval as se

import numpy as np

JUDGES = ["gN", "gN_naive", "RN", "R17", "QM2026", "QM2021"]
CONS = ["gN", "gN_naive", "RN", "QM2026"]
OUT = se.OUT


def v1_v2(base, rows):
    """comp_rescore.v1_v2 with the rebuilt gN already in base."""
    from overfit2026 import judges_v2
    v1 = {j: np.asarray(base[j], float) for j in JUDGES if j in base}
    v1["consensus"] = np.mean([v1[j] for j in CONS], 0)
    v2 = judges_v2.correct(v1, rows, judges=[j for j in JUDGES if j in v1])
    return {"v1": v1, "v2": v2}


def rebuilt_gn(rows):
    from overfit2026 import judges_v2
    return judges_v2.score_base(rows, judges=["gN"])["gN"]


def mse(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))]


# ── bench ───────────────────────────────────────────────────────────────

def _bench_drafts(value, ck, temp, sub="bench"):
    pat = se.bench_path(value, ck, temp, 0).replace("part0", "part*")
    ps = sorted(glob.glob(pat.replace(os.sep + "bench" + os.sep, os.sep + sub + os.sep)))
    d = []
    for p in ps:
        d += json.load(open(p))["drafts"]
    d.sort(key=lambda x: x["i"])
    return d, ps


def score_bench():
    from paper1_revision.bench_mcts import score_drafts, diversity
    res = {}
    kern = {}
    for T in (0, 1):
        p = os.path.join(core.RESULTS, "mcts_bench", f"new__F_oof_s0__T{T}.json")
        k = json.load(open(p))
        rows = [(x["our"], x["opp"], x["map"], "mid") for x in k["drafts"]]
        base = dict(k["per_draft"])
        base["gN"] = rebuilt_gn(rows)
        kern[T] = {"sc": v1_v2(base, rows), "degen": np.asarray(k["per_draft"]["degen"]),
                   "healer": np.asarray(k["per_draft"]["healer"]), "div": k["diversity"],
                   "proxy_own": np.asarray(k["per_draft"]["proxy_own"]), "n": len(rows)}
    for sub, value, T in (("bench", "enriched", 0), ("bench", "enriched", 1), ("bench", "lcb:1", 0),
                          ("bench_seq", "enriched", 0), ("bench_seq", "enriched", 1)):
        drafts, ps = _bench_drafts(value, 0, float(T), sub)
        if len(drafts) < 1000:
            print(f"{sub} {value} T{T}: {len(drafts)} drafts (incomplete)", flush=True)
            if not drafts:
                continue
        rows = [(x["our"], x["opp"], x["map"], "mid") for x in drafts]
        sc = score_drafts(drafts)
        base = dict(sc)
        base["gN"] = rebuilt_gn(rows)
        S = v1_v2(base, rows)
        K = kern[T]
        idx = np.array([x["i"] for x in drafts])
        r = {"n": len(drafts), "degen": mse(sc["degen"]), "healer": mse(sc["healer"]),
             "proxy_own(enriched leaf)": mse(sc["proxy_rev"]), "diversity": diversity(drafts),
             "kernel": {"degen": mse(K["degen"]), "healer": mse(K["healer"]),
                        "proxy_own": mse(K["proxy_own"]), "diversity": K["div"]},
             "secs": sum(json.load(open(p))["secs"] for p in ps)}
        for v in ("v1", "v2"):
            r[v] = {}
            for j in list(S[v]):
                x, y = S[v][j], K["sc"][v][j][idx]
                r[v][j] = {"python": mse(x), "kernel": mse(y), "diff_paired": mse(x - y)}
        r["degen_diff_paired"] = mse(sc["degen"] - K["degen"][idx])
        res[f"{sub}|{value}|T{T}"] = r
        print(value, T, {j: [round(t, 4) for t in r["v2"][j]["python"]] + [round(t, 4) for t in r["v2"][j]["kernel"]]
                         for j in ("gN", "RN", "QM2026", "consensus")}, "deg", r["degen"], r["kernel"]["degen"],
              flush=True)
    json.dump(res, open(os.path.join(OUT, "bench_validation.json"), "w"), indent=1)


# ── tournament ──────────────────────────────────────────────────────────

def _pairs_new():
    out = {}
    for a, b in se.pairs():
        p = se.pair_path(a, b)
        if os.path.exists(p):
            out[(a, b)] = json.load(open(p))["records"]
    return out


def _scores_new(pairs):
    """v1 judges (score_tournament.score_rows) + rebuilt gN for the new pairs, cached."""
    from paper1_revision.score_tournament import score_rows
    cache = os.path.join(OUT, "tournament_scores_new.npz")
    keys = sorted(pairs)
    rows = [(r["team0"]["picks"], r["team1"]["picks"], r["game_map"], r["tier"])
            for k in keys for r in pairs[k]]
    tag = np.array(["%s__%s" % k for k in keys])
    if os.path.exists(cache):
        z = dict(np.load(cache))
        if list(z["pairs"]) == list(tag):
            return keys, rows, z
    sc = score_rows(rows)
    sc["gN_v1stored"] = sc["gN"]
    sc["gN"] = rebuilt_gn(rows)
    sc["pairs"] = tag
    np.savez(cache, **sc)
    return keys, rows, sc


def _existing():
    """Stored scores of the 110 existing pairs, gN replaced by comp_rescore's rebuilt gN."""
    from paper1_revision.tournament import STRATEGIES
    from paper1_revision.score_tournament import load_pair
    sc = dict(np.load(os.path.join(core.RESULTS, "tournament_scores.npz")))
    rows, idx, recs = [], {}, {}
    for a, b in itertools.permutations(STRATEGIES, 2):
        d = load_pair(a, b)
        rr = d["records"]
        idx[(a, b)] = (len(rows), len(rows) + len(rr))
        rows += [(r["team0"]["picks"], r["team1"]["picks"], r["game_map"], r["tier"]) for r in rr]
        recs[(a, b)] = rr
    cache = os.path.join(OUT, "tournament_existing_gn.npz")   # comp_rescore's regn("tournament")
    if os.path.exists(cache) and len(np.load(cache)["gN"]) == len(rows):
        g = np.load(cache)["gN"]
    else:
        g = rebuilt_gn(rows)
        os.makedirs(OUT, exist_ok=True)
        np.savez(cache, gN=g)
    assert len(sc["gN"]) == len(rows) == len(g)
    sc["gN"] = g
    return rows, idx, recs, sc


def _standings(entries, names):
    """entries: {(a, b): {judge: (mu, var)}} -> standings over names (pairs among names only)."""
    J = list(next(iter(entries.values())))
    per = {s: {j: [] for j in J} for s in names}
    for (a, b), r in entries.items():
        if a not in names or b not in names:
            continue
        for j in J:
            mu, var = r[j]
            per[a][j].append((mu, var))
            per[b][j].append((1 - mu, var))
    st = {}
    for s in names:
        st[s] = {"n_pairs": len(per[s]["consensus"])}
        for j in J:
            mus = np.array([x[0] for x in per[s][j]])
            vs = np.array([x[1] for x in per[s][j]])
            st[s][j] = {"mean": float(mus.mean()), "se": float(np.sqrt(vs.sum()) / len(vs)),
                        "wins": int((mus > 0.5).sum())}
    return st


def score_tournament():
    from paper1_revision.score_tournament import pair_stats
    from paper1_revision.tournament import STRATEGIES
    pairs = _pairs_new()
    keys, rows, sc = _scores_new(pairs)
    print(f"{len(keys)} new pairs, {len(rows):,} drafts", flush=True)
    S = v1_v2(sc, rows)
    for v in ("v1", "v2"):
        S[v]["QM2021"] = S[v].get("QM2021", sc["QM2021"])
    erows, eidx, erecs, esc = _existing()
    ES = v1_v2(esc, erows)
    TJ = JUDGES + ["consensus"]
    out = {"note": "v1 = uncorrected judges with comp_rescore's rebuilt gN; v2 = composition-corrected "
                   "(judges_v2.correct), the comp_rescore.json tournament convention",
           "n_pairs_new": len(keys)}
    off = 0
    ent = {"v1": {}, "v2": {}}
    sides = {}
    for k in keys:
        rr = pairs[k]
        i0, i1 = off, off + len(rr)
        off = i1
        cfg = [(r["game_map"], r["tier"]) for r in rr]
        for v in ("v1", "v2"):
            ent[v][k] = {j: pair_stats(S[v][j][i0:i1], cfg) for j in TJ}
        for s, key in ((k[0], "team0"), (k[1], "team1")):
            m = sides.setdefault(s, {"healer": [], "degen": [], "synergy": [], "counter": [], "picks": []})
            for r in rr:
                m["healer"].append(float(r[key]["has_healer"]))
                m["degen"].append(float(r[key]["is_degen"]))
                m["synergy"].append(r[key]["synergy"])
                m["counter"].append(r[key]["counter"])
                m["picks"] += r[key]["picks"]
    eent = {"v1": {}, "v2": {}}
    for k, (i0, i1) in eidx.items():
        cfg = [(r["game_map"], r["tier"]) for r in erecs[k]]
        for v in ("v1", "v2"):
            eent[v][k] = {j: pair_stats(ES[v][j][i0:i1], cfg) for j in TJ}
    for v in ("v1", "v2"):
        allent = {**eent[v], **ent[v]}
        res = {}
        for ag in se.NEW:
            if not any(ag in k for k in ent[v]):
                continue
            res[ag] = {
                "vs_11": _standings({k: e for k, e in ent[v].items() if ag in k
                                     and (k[0] in STRATEGIES or k[1] in STRATEGIES)},
                                    STRATEGIES + [ag])[ag],
                "vs_11_plus_new": _standings({k: e for k, e in ent[v].items() if ag in k
                                              and all(x in STRATEGIES + se.PRIMARY + [ag] for x in k)},
                                             STRATEGIES + list(dict.fromkeys(se.PRIMARY + [ag])))[ag]}
            hh = {}
            for b in ["mcts", "constrained_mcts", "k_oof"] + [x for x in se.NEW if x != ag]:
                if (ag, b) in ent[v] and (b, ag) in ent[v]:
                    hh[b] = {}
                    for j in TJ:
                        m1, v1_ = ent[v][(ag, b)][j]
                        m2, v2_ = ent[v][(b, ag)][j]
                        hh[b][j] = [0.5 * (m1 + 1 - m2), 0.5 * float(np.sqrt(v1_ + v2_))]
            res[ag]["h2h"] = hh
            res[ag]["pairs"] = {"%s__%s" % k: {j: [e[j][0], float(np.sqrt(e[j][1]))] for j in TJ}
                                for k, e in ent[v].items() if ag in k}
        names13 = STRATEGIES + [x for x in se.PRIMARY if any(x in k for k in ent[v])]
        st13 = _standings({k: e for k, e in allent.items() if k[0] in names13 and k[1] in names13}, names13)
        out[v] = {"agents": res, "standings13": st13,
                  "order13": sorted(names13, key=lambda s: -st13[s]["consensus"]["mean"])}
    from collections import Counter
    dm = {}
    for s, m in sides.items():
        if s not in se.NEW:
            continue
        c = Counter(m.pop("picks"))
        tot = sum(c.values())
        p = np.array(list(c.values())) / tot
        dm[s] = {kk: float(np.mean(vv)) for kk, vv in m.items()}
        dm[s].update({"distinct": len(c), "entropy": float(-(p * np.log2(p)).sum()),
                      "top10": float(sum(vv for _, vv in c.most_common(10)) / tot)})
    out["draft_metrics"] = dm
    json.dump(out, open(os.path.join(OUT, "tournament_standings.json"), "w"), indent=1)
    for v in ("v1", "v2"):
        print(f"== {v}")
        for ag, r in out[v]["agents"].items():
            print(ag, {j: (round(r["vs_11_plus_new"][j]["mean"], 4), round(r["vs_11_plus_new"][j]["se"], 4),
                           r["vs_11_plus_new"][j]["wins"]) for j in ("consensus", "gN", "RN", "QM2026", "QM2021")})
            print("   h2h", {b: {j: [round(t, 4) for t in x[j]] for j in ("consensus",)} for b, x in r["h2h"].items()})
        print("  13:", [(s, round(out[v]["standings13"][s]["consensus"]["mean"], 4)) for s in out[v]["order13"]])


# ── rich ────────────────────────────────────────────────────────────────

def score_rich():
    import torch
    from collections import Counter
    from experiment_rich_evaluation import counter_responsiveness, synergy_exploitation
    from rerun2026.phase3_benchmarks import load_gd_models
    from paper1_revision.score_tournament import score_rows
    hp = core.load_stats("hp")
    gd = load_gd_models(torch.device("cpu"))
    res = {}
    for ag in se.NEW:
        per_seed = {}
        drafts = []
        for s in range(5):
            p = se.rich_path(ag, s)
            if os.path.exists(p):
                d = json.load(open(p))
                per_seed[s] = (len(drafts), len(drafts) + len(d))
                drafts += d
        if not drafts:
            continue
        c = Counter(h for d in drafts for h in d["our"])
        tot = sum(c.values())
        pr = np.array(list(c.values())) / tot
        agree = total = 0
        for d in drafts:
            for st in d["our_pick_steps"]:
                s_t = torch.tensor(st["state"], dtype=torch.float32).unsqueeze(0)
                m_t = torch.tensor(st["mask"], dtype=torch.float32).unsqueeze(0)
                votes = Counter()
                with torch.no_grad():
                    for g in gd:
                        votes[g(s_t, m_t).argmax(dim=1).item()] += 1
                agree += st["hero_idx"] == votes.most_common(1)[0][0]
                total += 1
        # how often the search's pick differs from the policy-head argmax of the
        # same checkpoint (eval seed s plays checkpoint s), picks only
        from paper1_revision import search_actor as sa
        same = tot_p = 0
        for s_, (i0, i1) in per_seed.items():
            pol = sa.PolicyNP(sa.f_oof_paths([s_])[0])
            for d in drafts[i0:i1]:
                for st in d["our_pick_steps"]:
                    x = np.asarray(st["state"], np.float32)
                    lg = pol.logits(x[:289])
                    lg = np.where(np.asarray(st["mask"]) > 0, lg, -np.inf)
                    same += int(np.argmax(lg)) == st["hero_idx"]
                    tot_p += 1
        rows = [(d["our"], d["opp"], d["map"], d["tier"]) for d in drafts]
        sc = score_rows(rows)
        base = dict(sc)
        base["gN"] = rebuilt_gn(rows)
        S = v1_v2(base, rows)
        ctr = np.array([counter_responsiveness(d["our"], d["opp"], hp, d["tier"]) for d in drafts])
        syn = np.array([synergy_exploitation(d["our"], hp, d["tier"]) for d in drafts])
        deg = np.array([float(d["is_degen"]) for d in drafts])
        hea = np.array([float(d["has_healer"]) for d in drafts])

        def seed_se(x):
            m = [float(np.mean(x[i0:i1])) for i0, i1 in per_seed.values()]
            return float(np.std(m, ddof=1) / np.sqrt(len(m))) if len(m) > 1 else None
        res[ag] = {"n": len(drafts), "eval_seeds": sorted(per_seed),
                   "healer": 100 * hea.mean(), "healer_se_seeds": 100 * (seed_se(hea) or 0),
                   "degen": 100 * deg.mean(), "degen_se_seeds": 100 * (seed_se(deg) or 0),
                   "counter": float(ctr.mean()), "synergy": float(syn.mean()),
                   "distinct": len(c), "entropy": float(-(pr * np.log2(pr)).sum()),
                   "top10": 100 * sum(v for _, v in c.most_common(10)) / tot,
                   "gd_similarity": 100 * agree / max(total, 1),
                   "policy_argmax_agreement": 100 * same / max(tot_p, 1),
                   "refs": {k: float(np.mean(v)) for k, v in sc.items()},
                   "refs_v1": {k: [float(v.mean()), seed_se(v)] for k, v in S["v1"].items()},
                   "refs_v2": {k: [float(v.mean()), seed_se(v)] for k, v in S["v2"].items()}}
        print(ag, {k: v for k, v in res[ag].items() if not k.startswith("refs")}, flush=True)
        print("   v2", {k: [round(t, 4) for t in v] for k, v in res[ag]["refs_v2"].items()}, flush=True)
    json.dump(res, open(os.path.join(OUT, "rich.json"), "w"), indent=1)
