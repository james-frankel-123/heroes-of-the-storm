"""
Team-composition support of the P3 drafters, on paper 1's yardstick: every arm's final teams
(both seats, against the other arms of p3_ood_audit.ARMS, headline tournament files) scored
against paper 1's training-split corpus (support_mitigation.sup_corpus): share with a rarest
triple below 25 / 10 / 5, median rarest triple, mean pair PMI, share seen exactly. Then the
selected design's composition-judge scores (consensus, gN, RN) against five arms, raw and with
paper 1's binned support correction (support_mitigation judge_corr.json).

Stages (default `base` reproduces the original output; the others add keys to the same JSON):
  base      the support table and the support-corrected judge scores (keys support, corrected_vs)
  boot      lobby-bootstrap SEs of every share / median / PMI and of the selected design minus
            pMCTS-cur, Real and aMCTS-pp (key boot)
  ablation  every arm in the tourney files on two common opponent sets, so arms are comparable:
            (A) both seats against aMCTS-pp (the only opponent of the design-study rows), and
            (B) both seats against pMCTS-cur, GD and Real; support next to the judge-free paired
            record, J and CONS against the same opponent (key ablation)
  corpus    time-matched human corpora from the slot table (outcome-free team counts only):
            Storm League teams 2026-05-23..06-30 (the lobby window), 05-23..09-11, and
            02-11..05-22, not the sealed build, no sealed id, not the 300 headline games
            (cached under --work)
  matched   every arm's support on the matched corpora, at the raw floor 25, at the
            size-equivalent floor, and at the matched corpus's own human 5th percentile; paper 1's
            corpus binomially thinned to each matched size as the like-for-like reference (key matched)
  residual  real games in the judge window (2026-07-01..09-11): the residual y - WP_pop - P,
            P = the drafter's personal term, by the real team's min3 and PMI on paper 1's corpus and
            on the lobby-window corpus; game-clustered SEs (key residual)
  regret    for each selected-design team below 25: the pick step where its rarest triple first
            went below 25, whether a legal hero kept every triple >= 25, and the cost of the best
            such hero by the drafter's own leaf V (vintage WP, b2 S + b3 O on the cur state) (key regret)

Usage (from training/, paper 1 site env, see paper1_revision/support_mitigation.py):
  python3 personalization/p3_team_support.py [stage ...] --hl <copy of cache_def_def1027_headline/p3t>
      --work <p3_ood_audit work dir with cells.npz>
-> personalization/results/p3_team_support.json. Aggregates only, no player keys.
"""
import os
import sys
import json
import time
import argparse
from itertools import combinations

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import numpy as np
from paper1_revision import support_mitigation as sm
from paper1_revision import team_support as ts

SP = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
ST = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
PK = {t: [k for k in range(16) if SP[k] and ST[k] == t] for t in (0, 1)}
SEL = "x-combo1-dr-chance"
ARMS = [SEL, "pMCTS-cur", "aMCTS-pp", "aMCTS-cur", "a1step-cur", "p1step-cur", "DraftRec", "DraftRec-pv", "GD", "Real"]
CORR_OPPS = ["pMCTS-cur", "aMCTS-pp", "GD", "DraftRec", "Real"]
CONTRASTS = ["pMCTS-cur", "Real", "aMCTS-pp"]
PANEL_B = ["pMCTS-cur", "GD", "Real"]
FLOOR = 25
BOOT = 1000
H = 90
OUT_DEF = os.path.join(HERE, "results", "p3_team_support.json")
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sg = lambda z: 1 / (1 + np.exp(-z))

# matched corpora: (first day, last day) ISO; outcome-free team counts from the slot table
CORPORA = {"win": ("2026-05-23", "2026-06-30"), "post": ("2026-05-23", "2026-09-11"), "pre": ("2026-02-11", "2026-05-22")}
SEALED_BUILD = "2.55.17.98025"


def teams(hl, a, b, t):
    acts = np.load(os.path.join(hl, "tourney", f"{a}__{b}.npz"))["acts"]
    A = np.sort(acts[:, PK[t]], 1).astype(np.int64)
    assert (A >= 0).all() and (A < 90).all() and all(len(set(r)) == 5 for r in A)
    return A


def has(hl, a, b):
    return os.path.exists(os.path.join(hl, "tourney", f"{a}__{b}.npz"))


def arm_teams(hl, arm, opps):
    """The arm's final teams in both seats against each opponent in opps: teams (N, 5), lobby (N,)."""
    T, L = [], []
    for b in opps:
        if b == arm:
            continue
        for (x, y, t) in ((arm, b, 0), (b, arm, 1)):
            if has(hl, x, y):
                A = teams(hl, x, y, t)
                T.append(A)
                L.append(np.arange(len(A)))
    return np.concatenate(T), np.concatenate(L)


def load_json(out):
    return json.load(open(out)) if os.path.exists(out) else {}


def save_key(out, key, val):
    j = load_json(out)
    j[key] = val
    json.dump(j, open(out, "w"), indent=1)


# ------------------------------------------------------------------ base (the original stage)

def base(a):
    hl, out = a.hl, a.out
    C = sm.sup_corpus()
    res = {"support": {}, "corrected_vs": {}}
    for arm in ARMS:
        T = [teams(hl, arm, b, 0) for b in ARMS if b != arm and has(hl, arm, b)]
        T += [teams(hl, b, arm, 1) for b in ARMS if b != arm and has(hl, b, arm)]
        m = C.metrics(np.concatenate(T))
        mn = np.asarray(m["min3"])
        res["support"][arm] = {"teams": int(len(mn)), "lt25": float((mn < 25).mean()), "lt10": float((mn < 10).mean()),
                               "lt5": float((mn < 5).mean()), "min3_median": float(np.median(mn)),
                               "pmi_mean": float(np.mean(m["pmi"])), "seen_exact": float((np.asarray(m["n5"]) > 0).mean())}
        print(arm, res["support"][arm], flush=True)
    jc = json.load(open(os.path.join(sm.OUT, "judge_corr.json")))
    for o in CORR_OPPS:
        row = {}
        for j in ("consensus", "gN", "RN"):
            raw, cor = [], []
            for x, y, first in ((SEL, o, True), (o, SEL, False)):
                p = np.asarray(np.load(os.path.join(hl, "scores", f"{x}__{y}.npz"))[j], float)
                m0, m1 = C.metrics(teams(hl, x, y, 0)), C.metrics(teams(hl, x, y, 1))
                pc = sg(lg(p) + (sm.bin_feats(m0, jc["edges"]) - sm.bin_feats(m1, jc["edges"])) @ np.array(jc["judges"][j]["w_binned"]))
                raw.append(p if first else 1 - p)
                cor.append(pc if first else 1 - pc)
            raw, cor = np.concatenate(raw), np.concatenate(cor)
            row[j] = {"raw": float(raw.mean()), "corrected": float(cor.mean()), "se": float(cor.std(ddof=1) / np.sqrt(len(cor)))}
        res["corrected_vs"][o] = row
        print(o, row, flush=True)
    j = load_json(out)
    j.update(res)
    json.dump(j, open(out, "w"), indent=1)


# ------------------------------------------------------------------ lobby bootstrap helpers

def boot_weights(n, seed=0, B=BOOT):
    """(B, n) lobby resample counts (each row a bootstrap draw of n lobbies)."""
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, n, (B, n))
    W = np.zeros((B, n))
    for b in range(B):
        W[b] = np.bincount(ix[b], minlength=n)
    return W


def boot_mean(v, L, W):
    """Bootstrap draws of the team-level mean of v (teams clustered in lobbies L)."""
    n = W.shape[1]
    s = np.bincount(L, weights=v.astype(float), minlength=n)
    c = np.bincount(L, minlength=n).astype(float)
    return (W @ s) / (W @ c)


def boot_median(v, L, W):
    o = np.argsort(v, kind="stable")
    vs, Ls = v[o], L[o]
    out = np.empty(len(W))
    for b in range(len(W)):
        w = W[b, Ls]
        cw = np.cumsum(w)
        out[b] = vs[np.searchsorted(cw, 0.5 * cw[-1])]
    return out


def stats_of(m):
    mn = np.asarray(m["min3"])
    return {"lt25": (mn < 25).astype(float), "lt10": (mn < 10).astype(float), "lt5": (mn < 5).astype(float),
            "pmi": np.asarray(m["pmi"], float), "seen_exact": (np.asarray(m["n5"]) > 0).astype(float), "min3": mn.astype(float)}


def summarize_boot(st, L, W):
    r = {}
    for k in ("lt25", "lt10", "lt5", "pmi", "seen_exact"):
        r[k] = round(float(st[k].mean()), 4)
        r[k + "_se"] = round(float(np.std(boot_mean(st[k], L, W))), 4)
    r["min3_median"] = float(np.median(st["min3"]))
    r["min3_median_se"] = round(float(np.std(boot_median(st["min3"], L, W))), 2)
    r["teams"] = int(len(L))
    return r


def diff_boot(sa, La, sb, Lb, W):
    d = {}
    for k in ("lt25", "lt10", "lt5", "pmi", "seen_exact"):
        bs = boot_mean(sa[k], La, W) - boot_mean(sb[k], Lb, W)
        d[k] = [round(float(sa[k].mean() - sb[k].mean()), 4), round(float(np.std(bs)), 4)]
    bs = boot_median(sa["min3"], La, W) - boot_median(sb["min3"], Lb, W)
    d["min3_median"] = [float(np.median(sa["min3"]) - np.median(sb["min3"])), round(float(np.std(bs)), 2)]
    return d


def human_teams(a):
    """The humans' actual final teams in the 300 headline lobbies (kernel team labels)."""
    z = np.load(os.path.join(a.work, "cells.npz"))
    rh = z["real_h"].astype(np.int64)
    A = np.concatenate([np.sort(rh[:, :5], 1), np.sort(rh[:, 5:], 1)])
    L = np.concatenate([np.arange(len(rh))] * 2)
    return A, L


# ------------------------------------------------------------------ 1a: bootstrap SEs

def boot(a):
    C = sm.sup_corpus()
    n = 300
    W = boot_weights(n)
    S, R = {}, {}
    for arm in ARMS:
        T, L = arm_teams(a.hl, arm, ARMS)
        S[arm] = (stats_of(C.metrics(T)), L)
        R[arm] = summarize_boot(S[arm][0], L, W)
        print(arm, R[arm], flush=True)
    A, L = human_teams(a)
    S["humans_actual"] = (stats_of(C.metrics(A)), L)
    R["humans_actual"] = summarize_boot(S["humans_actual"][0], L, W)
    D = {f"{SEL} - {o}": diff_boot(*S[SEL], *S[o], W) for o in CONTRASTS + ["humans_actual"]}
    D["Real - humans_actual"] = diff_boot(*S["Real"], *S["humans_actual"], W)
    print(json.dumps(D, indent=1))
    # the Real arm replays the human's action when it is still legal and falls back to GD otherwise
    # (p3t_systems.act_real): split its teams by whether any of its own picks was a fallback
    fbk = []
    for b in ARMS:
        for (x, y, t) in (("Real", b, 0), (b, "Real", 1)):
            if b != "Real" and has(a.hl, x, y):
                fb = np.load(os.path.join(a.hl, "tourney", f"{x}__{y}.npz"))["fallback"]
                fbk.append(fb[:, PK[t]].astype(bool).sum(1))
    fbk = np.concatenate(fbk)
    st, L = S["Real"]
    assert len(fbk) == len(L)
    split = {"teams_with_fallback_pick": round(float((fbk > 0).mean()), 4),
             "mean_fallback_picks": round(float(fbk.mean()), 3)}
    for lab, m in (("no_fallback", fbk == 0), ("any_fallback", fbk > 0)):
        split[lab] = summarize_boot({k: v[m] for k, v in st.items()}, L[m], W)
    print("Real by fallback", split, flush=True)
    save_key(a.out, "boot", {"what": "lobby-bootstrap SEs (1000 draws of the 300 headline lobbies; arm differences "
                                     "on the same draws). Opponents: each arm in both seats against the other arms of "
                                     "ARMS. humans_actual: the humans' own final teams in the 300 lobbies (600 teams). "
                                     "Real_by_fallback: the Real arm replays each human action while it is legal and "
                                     "falls back to GD when the opponent arm has taken or banned it. "
                                     "Differences are [point, se].", "arms": R, "differences": D,
                             "Real_by_fallback": split})


# ------------------------------------------------------------------ 1b: ablation arms on common opponents

def all_arms(hl):
    fs = [f[:-4].split("__") for f in os.listdir(os.path.join(hl, "tourney")) if "__" in f]
    return sorted({x for p in fs for x in p})


def paired_record(hl, A, B, W):
    """Judge-free paired record of A vs B (p3_ood_audit part 4 / p3t_analyze.paired_record), in pp, with
    lobby-bootstrap SE; None if the score files are missing."""
    pa, pb = os.path.join(hl, "scores", f"{A}__{B}.npz"), os.path.join(hl, "scores", f"{B}__{A}.npz")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return None
    sa, sb = np.load(pa), np.load(pb)
    RA = np.concatenate([sa["rec_slot"][:, :5], sb["rec_slot"][:, 5:]], 1)
    RB = np.concatenate([sb["rec_slot"][:, :5], sa["rec_slot"][:, 5:]], 1)
    both = np.isfinite(RA) & np.isfinite(RB)
    d = np.where(both, RA - RB, 0.0)
    cnt = both.sum(1)
    lob = cnt > 0
    per = np.where(lob, d.sum(1) / np.maximum(cnt, 1), 0.0)
    pt = per[lob].mean()
    bs = (W @ (per * lob)) / (W @ lob.astype(float))
    win = {}
    for key in ("judge_personal", "consensus"):
        v = np.concatenate([sa[key], 1 - sb[key]])
        L = np.r_[np.arange(300), np.arange(300)]
        win[key] = [round(100 * float(v.mean()), 2), round(100 * float(np.std(boot_mean(v, L, W))), 2)]
    return {"record_pp": round(100 * float(pt), 2), "record_se": round(100 * float(np.std(bs)), 2),
            "coverage_both": round(float(both.mean()), 3), "J_pct": win["judge_personal"], "CONS_pct": win["consensus"]}


def ablation(a):
    C = sm.sup_corpus()
    W = boot_weights(300)
    arms = all_arms(a.hl)
    out = {"what": "support of each arm's final teams against a common opponent set, with the judge-free paired "
                   "record (pp per covered player; lobby-bootstrap SE, no seed term, so tighter than P3_TOURNAMENT's "
                   "seed-widened CIs), J and CONS win % of the arm, all against the same opponent(s). Support is on "
                   "paper 1's corpus. Set A: both seats vs aMCTS-pp (every design-study row has only this opponent). "
                   "Set B: both seats vs each of pMCTS-cur, GD, Real (arms that played all three); record and J are vs "
                   "pMCTS-cur.", "arms_present": arms, "A_vs_aMCTS-pp": {}, "B_vs_pMCTS_GD_Real": {}}
    for arm in arms:
        if arm == "aMCTS-pp" or not (has(a.hl, arm, "aMCTS-pp") and has(a.hl, "aMCTS-pp", arm)):
            continue
        T, L = arm_teams(a.hl, arm, ["aMCTS-pp"])
        r = summarize_boot(stats_of(C.metrics(T)), L, W)
        rec = paired_record(a.hl, arm, "aMCTS-pp", W)
        if rec:
            r.update(rec)
        out["A_vs_aMCTS-pp"][arm] = r
        print("A", arm, r.get("lt25"), r.get("record_pp"), flush=True)
    for arm in arms:
        if not all(has(a.hl, arm, o) and has(a.hl, o, arm) for o in PANEL_B if o != arm):
            continue
        T, L = arm_teams(a.hl, arm, PANEL_B)
        r = summarize_boot(stats_of(C.metrics(T)), L, W)
        if arm != "pMCTS-cur":
            rec = paired_record(a.hl, arm, "pMCTS-cur", W)
            if rec:
                r.update(rec)
        out["B_vs_pMCTS_GD_Real"][arm] = r
        print("B", arm, r.get("lt25"), r.get("record_pp"), flush=True)
    # is there a support-vs-record frontier? rank correlation across arms
    for key in ("A_vs_aMCTS-pp", "B_vs_pMCTS_GD_Real"):
        rows = [(v["lt25"], v["pmi"], v["record_pp"], v["J_pct"][0]) for v in out[key].values() if "record_pp" in v]
        if len(rows) >= 4:
            x = np.array(rows)
            rk = lambda u: np.argsort(np.argsort(u)).astype(float)
            sp = lambda i, j: float(np.corrcoef(rk(x[:, i]), rk(x[:, j]))[0, 1])
            out[key + "_spearman"] = {"n_arms": len(rows), "lt25_vs_record": round(sp(0, 2), 3),
                                      "pmi_vs_record": round(sp(1, 2), 3), "lt25_vs_J": round(sp(0, 3), 3)}
    save_key(a.out, "ablation", out)


# ------------------------------------------------------------------ 1c: time-matched corpora

class LightCorpus:
    """min3, PMI and exact-team counts (team_support.Corpus without the 4-subsets)."""

    def __init__(self, c3, c2, c1, n, k5=None):
        self.c3, self.c2, self.c1, self.n, self.k5 = c3, c2, c1, int(n), k5

    @classmethod
    def from_teams(cls, A):
        c3 = np.bincount(np.concatenate([ts.key(c, A) for c in combinations(range(5), 3)]), minlength=H ** 3)
        c2 = np.bincount(np.concatenate([ts.key(c, A) for c in combinations(range(5), 2)]), minlength=H ** 2)
        c1 = np.bincount(A.ravel(), minlength=H)
        return cls(c3, c2, c1, len(A), np.unique(ts.key(range(5), A), return_counts=True))

    @classmethod
    def from_p1(cls):
        C = sm.sup_corpus()
        return cls(C.c3, C.c2, C.c1, C.n, C.k5)

    def metrics(self, A, c3=None):
        c3 = self.c3 if c3 is None else c3
        s3 = np.stack([c3[ts.key(c, A)] for c in combinations(range(5), 3)], 1)
        s2 = np.stack([self.c2[ts.key(c, A)] for c in combinations(range(5), 2)], 1)
        p = self.c1 / self.n
        expect = np.stack([self.n * p[A[:, x]] * p[A[:, y]] for x, y in combinations(range(5), 2)], 1)
        out = {"min3": s3.min(1), "pmi": np.log((s2 + 0.5) / expect).mean(1)}
        if self.k5 is not None:
            out["n5"] = ts.Corpus._lookup(self.k5, ts.key(range(5), A))
        return out


def slot_table(cols):
    import p3_hs_core as HC
    z = np.load(os.path.join(HERE, "cache_def_def1027", "x_slots_ext.npz"), mmap_mode="r")
    return {k: np.asarray(z[k]) for k in cols}, HC


def corpus_path(a, name):
    return os.path.join(a.work, f"ts_corpus_{name}.npz")


def window_teams(d, HC, lo, hi, excl_rids):
    """Sorted 5-hero teams (shared index) of slot-table games in [lo, hi]: post-or-snapshot SL rows, not the sealed
    build, no sealed id, not in excl_rids. Outcome-free (no y, no wp)."""
    import p3_sealed
    from shared import HEROES
    vs = [str(x) for x in d["versions"]]
    sb = vs.index(SEALED_BUILD) if SEALED_BUILD in vs else -999
    m = (d["day"] >= HC.day_of(lo)) & (d["day"] <= HC.day_of(hi)) & (d["version"] != sb)
    w = np.flatnonzero(m)
    rid = d["replay_id"][w].astype(np.int64)
    keep = ~np.isin(rid, p3_sealed.sealed_ids()) & ~np.isin(rid, excl_rids)
    w, rid = w[keep], rid[keep]
    sh = np.array([HEROES.index(str(x)) for x in d["hero_names"]], np.int64)
    hero = sh[d["hero"][w]]
    team = d["team"][w].astype(np.int64)
    o = np.lexsort((hero, team, rid))
    g = rid[o] * 2 + team[o]
    hero = hero[o]
    ug, st, cnt = np.unique(g, return_index=True, return_counts=True)
    ok = cnt == 5
    idx = st[ok][:, None] + np.arange(5)[None]
    A = hero[idx]
    A = A[(np.diff(A, axis=1) > 0).all(1)]
    return A, int(ok.sum()), int((~ok).sum()), len(np.unique(rid))


def corpus(a):
    d, HC = slot_table(["replay_id", "hero", "team", "day", "version", "versions", "hero_names"])
    L = np.load(os.path.join(a.hl, "lobbies.npz"))
    gi = np.load(os.path.join(a.hl, "tourney", "lobbies.npz"))["gi"]
    excl = L["replay_id"][gi].astype(np.int64)
    meta = {}
    for name, (lo, hi) in CORPORA.items():
        A, nt, bad, ng = window_teams(d, HC, lo, hi, excl)
        lc = LightCorpus.from_teams(A)
        # leave-one-out min3 of the corpus's own teams (each team counts once in each of its triples)
        mo = lc.metrics(A)
        loo = mo["min3"] - 1
        np.savez(corpus_path(a, name), c3=lc.c3.astype(np.int32), c2=lc.c2.astype(np.int64), c1=lc.c1, n=lc.n,
                 k5k=lc.k5[0], k5c=lc.k5[1], loo_min3=loo.astype(np.int32), pmi=mo["pmi"].astype(np.float32))
        meta[name] = {"window": [lo, hi], "teams": int(lc.n), "games": ng, "incomplete_teams_dropped": bad}
        print(name, meta[name], flush=True)
    json.dump(meta, open(os.path.join(a.work, "ts_corpus_meta.json"), "w"), indent=1)


def load_corpus(a, name):
    z = np.load(corpus_path(a, name))
    return LightCorpus(z["c3"], z["c2"], z["c1"], z["n"], (z["k5k"], z["k5c"])), z


def matched(a):
    P1 = LightCorpus.from_p1()
    W = boot_weights(300)
    meta = json.load(open(os.path.join(a.work, "ts_corpus_meta.json")))
    sets = {arm: arm_teams(a.hl, arm, ARMS) for arm in ARMS}
    sets["humans_actual"] = human_teams(a)
    rng = np.random.default_rng(1)
    out = {"what": "support on time-matched human corpora (Storm League slot-table teams; outcome-free counts; "
                   "no sealed build or id; the 300 headline games excluded). Floors: raw 25; 'eq' = 25 x N_matched / "
                   "N_paper1 (the same rate per team as 25 on paper 1's corpus); 'hp5' = the 5th percentile of the "
                   "matched corpus's own teams' leave-one-out min3 (paper 1's 25 is the p5 of its held-out humans). "
                   "p1_thin = paper 1's triple counts binomially thinned to the matched corpus size (5 draws), "
                   "at the eq floor: the like-for-like reference for corpus size (gap closed = 1 - gap on the matched corpus / gap "
                   "on thinned paper 1, both at eq). hp5 is degenerate (0) on the small lobby-window corpus. Shares as "
                   "fractions, SE = lobby bootstrap.", "corpora": meta, "p1_teams": P1.n, "by_corpus": {}}
    p1m = {k: P1.metrics(v[0]) for k, v in sets.items()}
    for name in CORPORA:
        lc, z = load_corpus(a, name)
        ratio = lc.n / P1.n
        eq = FLOOR * ratio
        hp5 = float(np.percentile(z["loo_min3"], 5))
        thins = [rng.binomial(P1.c3, ratio) for _ in range(5)]
        rows = {}
        for k, (T, Lb) in sets.items():
            m = lc.metrics(T)
            mn = m["min3"].astype(float)
            r = {"lt25": round(float((mn < FLOOR).mean()), 4),
                 "lt25_se": round(float(np.std(boot_mean((mn < FLOOR).astype(float), Lb, W))), 4),
                 "lt_eq": round(float((mn < eq).mean()), 4),
                 "lt_eq_se": round(float(np.std(boot_mean((mn < eq).astype(float), Lb, W))), 4),
                 "lt_hp5": round(float((mn < hp5).mean()), 4),
                 "lt_hp5_se": round(float(np.std(boot_mean((mn < hp5).astype(float), Lb, W))), 4),
                 "min3_median": float(np.median(mn)),
                 "pmi": round(float(m["pmi"].mean()), 4),
                 "pmi_se": round(float(np.std(boot_mean(m["pmi"], Lb, W))), 4),
                 "seen_exact": round(float((m["n5"] > 0).mean()), 4)}
            th = [float((P1.metrics(T, c3=c)["min3"] < eq).mean()) for c in thins]
            r["p1_thin_lt_eq"] = round(float(np.mean(th)), 4)
            r["p1_lt25"] = round(float((p1m[k]["min3"] < FLOOR).mean()), 4)
            r["p1_pmi"] = round(float(p1m[k]["pmi"].mean()), 4)
            rows[k] = r
        # gaps (selected design minus humans / Real arm) on paper 1's corpus vs the matched corpus, with SEs
        gaps = {}
        for ref in ("Real", "humans_actual", "pMCTS-cur"):
            (Ts, Ls), (Tr, Lr) = sets[SEL], sets[ref]
            ms, mr = lc.metrics(Ts), lc.metrics(Tr)
            g = {}
            th_s = np.mean([P1.metrics(Ts, c3=c)["min3"] < eq for c in thins], 0)
            th_r = np.mean([P1.metrics(Tr, c3=c)["min3"] < eq for c in thins], 0)
            for lab, fs, fr in (("pmi_matched", ms["pmi"], mr["pmi"]),
                                ("lt_eq_p1thin", th_s, th_r),
                                ("pmi_p1", p1m[SEL]["pmi"], p1m[ref]["pmi"]),
                                ("lt_hp5_matched", (ms["min3"] < hp5).astype(float), (mr["min3"] < hp5).astype(float)),
                                ("lt_eq_matched", (ms["min3"] < eq).astype(float), (mr["min3"] < eq).astype(float)),
                                ("lt25_p1", (p1m[SEL]["min3"] < FLOOR).astype(float), (p1m[ref]["min3"] < FLOOR).astype(float))):
                bs = boot_mean(fs, Ls, W) - boot_mean(fr, Lr, W)
                g[lab] = [round(float(fs.mean() - fr.mean()), 4), round(float(np.std(bs)), 4)]
            g["pmi_gap_closed_share"] = round(1 - g["pmi_matched"][0] / g["pmi_p1"][0], 3) if g["pmi_p1"][0] else None
            g["lt_eq_gap_closed_share"] = round(1 - g["lt_eq_matched"][0] / g["lt_eq_p1thin"][0], 3) if g["lt_eq_p1thin"][0] else None
            gaps[f"{SEL} - {ref}"] = g
        out["by_corpus"][name] = {"teams": lc.n, "eq_floor": round(eq, 3), "hp5_floor": hp5, "arms": rows, "gaps": gaps}
        print(name, "eq", round(eq, 2), "hp5", hp5, {k: (v["lt25"], v["lt_eq"], v["lt_hp5"], v["p1_thin_lt_eq"], v["pmi"])
                                                     for k, v in rows.items()}, flush=True)
        print(json.dumps(gaps, indent=1), flush=True)
    save_key(a.out, "matched", out)


# ------------------------------------------------------------------ 2: personal residual vs team support

def wls(X, y, g):
    """OLS y = X b, SEs clustered by g."""
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    u = y - X @ b
    _, gi = np.unique(g, return_inverse=True)
    S = np.zeros((gi.max() + 1, X.shape[1]))
    np.add.at(S, gi, X * u[:, None])
    Ai = np.linalg.inv(XtX)
    G = gi.max() + 1
    V = Ai @ (S.T @ S) @ Ai * G / max(G - 1, 1)
    return b, np.sqrt(np.diag(V))


def residual(a):
    import p3_ood_audit as OA
    import p3_sealed
    from shared import HEROES
    t0 = time.time()
    d, HC = slot_table(["replay_id", "pid", "hero", "team", "y", "wp", "day", "post", "version", "versions",
                        "hero_names", "player_keys"])
    vs = [str(x) for x in d["versions"]]
    sb = vs.index(SEALED_BUILD) if SEALED_BUILD in vs else -999
    m = d["post"].astype(bool) & (d["version"] != sb) & (d["day"] >= OA.J0) & (d["day"] <= OA.J1)
    w = np.flatnonzero(m)
    w = w[~np.isin(d["replay_id"][w].astype(np.int64), p3_sealed.sealed_ids())]
    rid = d["replay_id"][w].astype(np.int64)
    o = np.lexsort((d["team"][w], rid))
    w, rid = w[o], rid[o]
    ug, st, cnt = np.unique(rid, return_index=True, return_counts=True)
    ok = cnt == 10
    idx = st[ok][:, None] + np.arange(10)[None]
    rows = w[idx]                                                     # (G, 10): team 0 rows then team 1 rows
    team = d["team"][rows]
    good = (team[:, :5] == team[:, :1]).all(1) & (team[:, 5:] == team[:, 5:6]).all(1) & (team[:, 0] != team[:, 5])
    rows = rows[good]
    G = len(rows)
    print(f"judge window: {G:,} complete games ({time.time() - t0:.0f}s)", flush=True)
    sh = np.array([HEROES.index(str(x)) for x in d["hero_names"]], np.int64)
    hero = sh[d["hero"][rows]]
    y = d["y"][rows].astype(float)
    wp = d["wp"][rows].astype(float)
    assert np.allclose(wp[:, :5], wp[:, :1]) and np.allclose(wp[:, 5:], wp[:, 5:6])
    # drafter state as of 2026-06-30 (p3_ood_audit prep): every tournament-window player
    z = np.load(os.path.join(a.work, "cells.npz"))
    import p3_hs_core as HCC
    jq = np.load(os.path.join(HCC.CACHE, "judge_q", "p3t_data_states__3.npz"))
    bkeys = jq["player_key"].astype(np.int64)
    keys = np.asarray(d["player_keys"]).astype(np.int64)[d["pid"][rows]]
    ob = np.argsort(bkeys)
    j = np.minimum(np.searchsorted(bkeys[ob], keys), len(bkeys) - 1)
    known = bkeys[ob][j] == keys
    brow = ob[j]
    s = np.where(known, z["b_s"][brow, hero], 0.0)
    off = np.where(known, OA.off_role_shared(z["b_n"])[brow, hero], False).astype(float)
    nn = np.where(known, z["b_n"][brow, hero], 0.0)
    comb = json.load(open(os.path.join(OA.HL_RES, "p3t_vintages.json")))["cur"]["combiner"]
    b0, b1, b2, b3 = comb
    allk = known.all(1)
    print(f"all 10 players known: {allk.mean():.3f} of games", flush=True)
    # team rows: t = 0 (first five) and 1
    T = np.concatenate([np.sort(hero[:, :5], 1), np.sort(hero[:, 5:], 1)])
    yt = np.r_[y[:, 0], y[:, 5]]
    wpt = np.r_[wp[:, 0], wp[:, 5]]
    S0, S1 = s[:, :5].sum(1), s[:, 5:].sum(1)
    O0, O1 = off[:, :5].sum(1), off[:, 5:].sum(1)
    dS = np.r_[S0 - S1, S1 - S0]
    dO = np.r_[O0 - O1, O1 - O0]
    lo = lg(wpt)
    P = sg(b0 + b1 * lo + b2 * dS + b3 * dO) - sg(b0 + b1 * lo)
    r = yt - wpt
    e = r - P
    gid = np.r_[np.arange(G), np.arange(G)]
    keep = np.r_[allk, allk]
    thin = np.r_[(nn[:, :5] < 5).sum(1), (nn[:, 5:] < 5).sum(1)]
    corp = {"p1": LightCorpus.from_p1(), "win": load_corpus(a, "win")[0]}
    BINS = {"p1": [(0, 4, "0-4"), (5, 24, "5-24"), (25, 99, "25-99"), (100, 499, "100-499"), (500, 10 ** 12, "500+")],
            "win": [(0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 22, "5-22"), (23, 10 ** 12, "23+")]}
    out = {"what": "real Storm League games in the judge window (2026-07-01..09-11; post-snapshot, not the sealed build, "
                   "no sealed id), one row per team (two per game), SEs clustered by game. r = y - WP_pop (slot-table "
                   "causal WP). P = the drafter's personal term: sig(b0 + b1 logit WP + b2 dS + b3 dO) - sig(b0 + b1 "
                   "logit WP), dS/dO = team minus opponent sums of s(p, h) and off-role flags on the drafter state as "
                   "of 2026-06-30 (fixed before the window), the actual player-hero cells. e = r - P. All in WP points. "
                   "Team support: min3 / PMI of the real team on paper 1's corpus (p1) and on the lobby-window corpus "
                   "(win, 2026-05-23..06-30; the judge window is not in it). 'slope_P' = slope of r on P within the bin "
                   "(1 = the personal term is right on average; below 1 = over-trusted).",
           "unknown_players": "players outside the tournament-window state set (60,335 players) get s = 0 and no "
                              "off-role flag, i.e. no personal term; 'all_games' uses every game this way, 'all_known' "
                              "only games with all ten players in the state set",
           "combiner": comb, "games": int(G), "games_all_known": int(allk.sum()),
           "player_slots_known": round(float(known.mean()), 4), "by_corpus": {}}
    for cname, lc in corp.items():
        mt = lc.metrics(T)
        mn = mt["min3"]
        pm = mt["pmi"]
        res = {}
        for sample, km in (("all_known", keep), ("all_games", np.ones_like(keep))):
            g = gid[km]
            R = {"rows": int(km.sum())}
            bins = {}
            for lo_, hi_, lab in BINS[cname]:
                mm = (mn >= lo_) & (mn <= hi_)
                k2 = km & mm
                if k2.sum() < 200:
                    bins[lab] = {"rows": int(k2.sum())}
                    continue
                gg = gid[k2]
                X1 = np.ones((k2.sum(), 1))
                be, se_ = wls(X1, e[k2], gg)
                br, sr = wls(X1, r[k2], gg)
                bp, spp = wls(X1, P[k2], gg)
                bs_, ss_ = wls(np.column_stack([np.ones(k2.sum()), P[k2]]), r[k2], gg)
                bins[lab] = {"rows": int(k2.sum()), "share": round(float(k2.sum() / km.sum()), 4),
                             "e_pp": [round(100 * be[0], 3), round(100 * se_[0], 3)],
                             "r_pp": [round(100 * br[0], 3), round(100 * sr[0], 3)],
                             "P_pp": [round(100 * bp[0], 3), round(100 * spp[0], 3)],
                             "slope_P": [round(bs_[1], 3), round(ss_[1], 3)],
                             "mean_thin_cells": round(float(thin[k2].mean()), 3)}
            R["by_min3"] = bins
            # linear trends in log1p(min3), and with the opponent's support as a control
            xm = np.log1p(mn[km].astype(float))
            xo = np.log1p(np.r_[mn[G:], mn[:G]][km].astype(float))
            for lab, yv in (("e", e[km]), ("r", r[km])):
                b_, s_ = wls(np.column_stack([np.ones(km.sum()), xm]), yv, g)
                b2_, s2_ = wls(np.column_stack([np.ones(km.sum()), xm, xo]), yv, g)
                R[f"{lab}_slope_log1p_min3"] = [round(100 * b_[1], 3), round(100 * s_[1], 3)]
                R[f"{lab}_slope_log1p_min3_opp_ctrl"] = [round(100 * b2_[1], 3), round(100 * s2_[1], 3)]
            # interaction: does the personal term's slope fall on rare teams?
            Pk, rk = P[km], r[km]
            rare = (mn[km] < (25 if cname == "p1" else 2)).astype(float)
            bi, si = wls(np.column_stack([np.ones(km.sum()), Pk, rare, Pk * rare]), rk, g)
            R["slope_P_interaction_rare"] = {"rare_def": "min3 < 25" if cname == "p1" else "min3 < 2 (about the same rate)",
                                             "slope_P_common": [round(bi[1], 3), round(si[1], 3)],
                                             "slope_P_rare_minus_common": [round(bi[3], 3), round(si[3], 3)],
                                             "rare_share": round(float(rare.mean()), 4)}
            # PMI quintiles
            pk = pm[km]
            qs = np.percentile(pk, [20, 40, 60, 80])
            qb = np.searchsorted(qs, pk)
            pb = {}
            for q in range(5):
                k2 = qb == q
                gg = g[k2]
                be, se_ = wls(np.ones((k2.sum(), 1)), e[km][k2], gg)
                br, sr = wls(np.ones((k2.sum(), 1)), r[km][k2], gg)
                bs_, ss_ = wls(np.column_stack([np.ones(k2.sum()), P[km][k2]]), r[km][k2], gg)
                pb[f"q{q + 1}"] = {"pmi_mean": round(float(pk[k2].mean()), 3), "e_pp": [round(100 * be[0], 3), round(100 * se_[0], 3)],
                                   "r_pp": [round(100 * br[0], 3), round(100 * sr[0], 3)],
                                   "slope_P": [round(bs_[1], 3), round(ss_[1], 3)]}
            R["by_pmi_quintile"] = pb
            for lab, yv in (("e", e[km]), ("r", r[km])):
                b_, s_ = wls(np.column_stack([np.ones(km.sum()), pk]), yv, g)
                R[f"{lab}_slope_pmi"] = [round(100 * b_[1], 3), round(100 * s_[1], 3)]
            res[sample] = R
        out["by_corpus"][cname] = res
        print(cname, json.dumps(res["all_known"], indent=None)[:3000], flush=True)
    save_key(a.out, "residual", out)


# ------------------------------------------------------------------ 3: regret audit

def regret(a):
    import p3_ood_audit as OA
    # the headline run's vintages (its combiner); the checkout default p3t_vintages.json differs in the combiner only
    os.environ.setdefault("P3T_VINTAGES", os.path.join(OA.HL_RES, "p3t_vintages.json"))
    import p3t_systems as SY
    from shared import HEROES
    C3 = sm.triple_counts()
    P1 = LightCorpus.from_p1()
    z = np.load(os.path.join(a.work, "cells.npz"))
    s_all, off_all, n_all = z["cur_s"], z["cur_off"].astype(float), z["cur_n"]
    maps, tiers = z["map"], z["tier"]
    pool = n_all > 0
    team_pool = np.stack([pool[:, :5].any(1), pool[:, 5:].any(1)], 1)          # (300, 2, H)
    comb = json.load(open(os.path.join(OA.HL_RES, "p3t_vintages.json")))["cur"]["combiner"]
    b0, b1, b2, b3 = comb
    v = SY.vintage(SY.SYSTEMS[SEL]["vint"])
    assert list(v.spec["combiner"]) == list(comb)

    def tri_ok(h, ours):
        return all(C3[(t[0] * H + t[1]) * H + t[2]] >= FLOOR for x, y in combinations(ours, 2)
                   for t in [sorted((x, y, h))])

    def min3_of(team):
        return min(C3[(t[0] * H + t[1]) * H + t[2]] for t in combinations(sorted(team), 3))

    def team_value(i, acts_list, t):
        """Own leaf V of team t for each full draft in acts_list (same lobby i): vintage WP + personal terms."""
        rows, dS, dO, thin = [], [], [], []
        for ac in acts_list:
            t0 = [int(ac[k]) for k in PK[0]]
            t1 = [int(ac[k]) for k in PK[1]]
            rows.append(([HEROES[h] for h in t0], [HEROES[h] for h in t1], str(maps[i]), str(tiers[i])))
            Sx, Ox, nth = [], [], 0
            for tt, hs in ((0, t0), (1, t1)):
                sl = slice(5 * tt, 5 * tt + 5)
                S_, O_, perm = OA.best_assign(s_all[i, sl], off_all[i, sl], hs, b2, b3)
                Sx.append(S_)
                Ox.append(O_)
                if tt == t:
                    nth = int(sum(n_all[i, 5 * tt + p, hs[perm[p]]] < 5 for p in range(5)))
            dS.append(Sx[0] - Sx[1])
            dO.append(Ox[0] - Ox[1])
            thin.append(nth)
        wp = v.wp_drafts(rows)
        dS, dO = np.array(dS), np.array(dO)
        V0 = sg(b0 + b1 * lg(wp) + b2 * dS + b3 * dO)
        sgn = 1 if t == 0 else -1
        Vt = V0 if t == 0 else 1 - V0
        wpt = wp if t == 0 else 1 - wp
        return Vt, wpt, sgn * dS, sgn * dO, np.array(thin)

    recs, vchk = [], []
    own_V = {}
    for opp in ARMS:
        if opp == SEL:
            continue
        for (x, y_, t) in ((SEL, opp, 0), (opp, SEL, 1)):
            f = np.load(os.path.join(a.hl, "tourney", f"{x}__{y_}.npz"))
            if t == 0:
                ov = np.load(os.path.join(a.hl, "scores", f"{x}__{y_}.npz"))["own_V_a"]
                own_V.update({(opp, i): ov[i] for i in range(len(ov))})
            acts_all, fb = f["acts"], f["fallback"]
            for i in range(len(acts_all)):
                acts = acts_all[i].astype(np.int64)
                own = [int(acts[k]) for k in PK[t]]
                if min3_of(own) >= FLOOR:
                    continue
                # the pick step where the team's rarest triple first went below the floor
                kstar, prior = None, None
                for j, k in enumerate(PK[t]):
                    if j >= 2 and not tri_ok(own[j], own[:j]):
                        kstar, prior = k, own[:j]
                        break
                assert kstar is not None
                legal = SY.legal_mask(acts[None], kstar, team_pool[i][None])
                smask = SY.struct_mask(acts[None], kstar, legal)[0]
                in_mask = bool(smask[acts[kstar]])
                cand = [h for h in np.flatnonzero(smask) if h != acts[kstar]]
                feas_step = [h for h in cand if tri_ok(h, prior)]
                later = set(int(u) for u in acts)
                feas_free = [h for h in feas_step if h not in later]
                any_free = [h for h in cand if h not in later]
                subs = []
                for h in any_free:
                    ac = acts.copy()
                    ac[kstar] = h
                    subs.append(ac)
                Vs, wps, dSs, dOs, ths = team_value(i, [acts] + subs, t)
                newteams = [[int(ac[k]) for k in PK[t]] for ac in subs]
                fs_ = set(feas_free)
                stepok = np.array([h in fs_ for h in any_free], bool)
                full = stepok & np.array([min3_of(tm) >= FLOOR for tm in newteams], bool)
                if t == 0:
                    vchk.append(abs(float(Vs[0]) - float(own_V[(opp, i)])))
                r = {"opp": opp, "seat": t, "lobby": i, "step": kstar, "pick_no": PK[t].index(kstar) + 1,
                     "actual_in_mask": in_mask, "fallback": bool(fb[i, kstar]), "n_legal": len(cand) + 1,
                     "n_feas_step": len(feas_step), "n_feas_free": len(feas_free), "n_feas_full": int(full.sum()),
                     "V": float(Vs[0]), "wp": float(wps[0]), "dS": float(dSs[0]), "thin": int(ths[0]),
                     "min3": int(min3_of(own))}
                vbest_any = float(Vs[1:].max()) if len(subs) else -np.inf
                r["actual_is_hindsight_best"] = bool(Vs[0] >= vbest_any)
                if full.any():
                    # hindsight cost of the floor: best V with any free hero (or the actual) minus best V in support
                    r["floor_cost"] = float(max(Vs[0], vbest_any) - Vs[1:][full].max())
                for lab, msk in (("step", stepok), ("full", full), ("any", np.ones(len(subs), bool))):
                    if msk.any():
                        jj = 1 + int(np.argmax(np.where(msk, Vs[1:], -np.inf)))
                        r[f"best_{lab}"] = {"dV": float(Vs[jj] - Vs[0]), "dwp": float(wps[jj] - wps[0]),
                                            "b2dS": float(b2 * (dSs[jj] - dSs[0])), "dthin": int(ths[jj] - ths[0]),
                                            "min3_new": int(min3_of(newteams[jj - 1]))}
                recs.append(r)
        print(opp, len(recs), f"{time.time():.0f}", flush=True)
    n_sel = 0
    for opp in ARMS:
        if opp != SEL:
            n_sel += 600
    R = recs
    W = boot_weights(300)

    def share(key_fn):
        v = np.array([key_fn(r) for r in R], float)
        L = np.array([r["lobby"] for r in R])
        return [round(float(v.mean()), 4), round(float(np.std(boot_mean(v, L, W))), 4)]

    def med(path):
        v = np.array([r[path[0]][path[1]] for r in R if path[0] in r], float)
        L = np.array([r["lobby"] for r in R if path[0] in r])
        if not len(v):
            return None
        return {"n": int(len(v)), "median": round(float(np.median(v)), 5), "median_se": round(float(np.std(boot_median(v, L, W))), 5),
                "mean": round(float(v.mean()), 5), "mean_se": round(float(np.std(boot_mean(v, L, W))), 5),
                "share_gt0": round(float((v > 0).mean()), 4)}

    out = {"what": "each selected-design team below 25 (paper 1's corpus) in the headline drafts (both seats vs the other "
                   "nine arms, 5,400 teams): the pick step k* where its rarest triple first fell below 25; alternatives = "
                   "heroes in the drafter's own action set at k* (team-pool legal mask and paper 1's structural mask, "
                   "p3t_systems), other than its pick. feasible_step: every triple with its earlier picks >= 25. "
                   "feasible_full: also unused later in the draft and the substituted final team has min3 >= 25 (rest of "
                   "the draft held fixed). any: every free hero in the action set. These are hindsight comparisons (the rest of the "
                   "draft is the one that happened), so the best of ~30 alternatives often beats the searched pick; "
                   "floor_cost = max(V actual, V best any) - V best full, the hindsight price of staying in support. "
                   "Cost of the best feasible hero by the drafter's own leaf V (vintage WP + "
                   "b2 S + b3 O, best of 120 assignments, cur state): dV = V(substituted) - V(actual), in probability; "
                   "dwp = population-WP part; b2dS = change in b2 x (S_ours - S_theirs) (logit units); dthin = change in "
                   "the team's cells with n < 5 under the drafter's assignment. SEs: lobby bootstrap.",
           "teams_total": n_sel, "teams_below_floor": len(R), "share_below_floor": round(len(R) / n_sel, 4),
           "pick_no_of_first_breach": {str(k): int(sum(r["pick_no"] == k for r in R)) for k in (3, 4, 5)},
           "actual_pick_in_mask": share(lambda r: r["actual_in_mask"]),
           "share_with_feasible_step": share(lambda r: r["n_feas_step"] > 0),
           "share_with_feasible_full": share(lambda r: r["n_feas_full"] > 0),
           "median_n_legal": float(np.median([r["n_legal"] for r in R])),
           "median_n_feasible_step": float(np.median([r["n_feas_step"] for r in R])),
           "best_step": {k: med(("best_step", k)) for k in ("dV", "dwp", "b2dS", "dthin")},
           "best_full": {k: med(("best_full", k)) for k in ("dV", "dwp", "b2dS", "dthin")},
           "best_any": {k: med(("best_any", k)) for k in ("dV", "dwp", "b2dS", "dthin")},
           "actual_is_hindsight_best": share(lambda r: r["actual_is_hindsight_best"]),
           "floor_cost": (lambda v, L: {"n": int(len(v)), "median": round(float(np.median(v)), 5),
                                        "median_se": round(float(np.std(boot_median(v, L, W))), 5),
                                        "mean": round(float(v.mean()), 5), "mean_se": round(float(np.std(boot_mean(v, L, W))), 5),
                                        "share_zero": round(float((v <= 1e-12).mean()), 4),
                                        "q75": round(float(np.percentile(v, 75)), 5), "q90": round(float(np.percentile(v, 90)), 5)})(
               np.array([r["floor_cost"] for r in R if "floor_cost" in r]),
               np.array([r["lobby"] for r in R if "floor_cost" in r])),
           "V_check_max_abs_diff_vs_score_files": float(max(vchk)) if vchk else None,
           "best_full_adds_thin_cell": share(lambda r: r.get("best_full", {}).get("dthin", 0) > 0),
           "actual_team_thin_cells_mean": round(float(np.mean([r["thin"] for r in R])), 3)}
    # cost quantiles of the best full fix
    dv = np.array([r["best_full"]["dV"] for r in R if "best_full" in r])
    if len(dv):
        out["best_full_dV_quantiles"] = {str(q): round(float(np.percentile(dv, q)), 5) for q in (10, 25, 50, 75, 90)}
        out["best_full_cost_lt_0p5pp"] = round(float((dv > -0.005).mean()), 4)
        out["best_full_cost_lt_1pp"] = round(float((dv > -0.01).mean()), 4)
    print(json.dumps(out, indent=1), flush=True)
    np.save(os.path.join(a.work, "ts_regret_records.npy"), np.array(R, dtype=object), allow_pickle=True)
    save_key(a.out, "regret", out)


STAGES = ["base", "boot", "ablation", "corpus", "matched", "residual", "regret"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", default=["base"], choices=STAGES + ["all"])
    ap.add_argument("--hl", required=True)
    ap.add_argument("--work", default=os.environ.get("P3OOD_WORK"), help="p3_ood_audit work dir (cells.npz); corpora cached here")
    ap.add_argument("--out", default=OUT_DEF)
    a = ap.parse_args()
    assert sm.OUT.endswith(os.path.join("site", "results", "support_mitigation")), sm.OUT
    st = STAGES if "all" in a.stages else a.stages
    for s in st:
        t0 = time.time()
        assert s == "base" or a.work, "--work is required for every stage but base"
        globals()[s](a)
        print(f"stage {s} done ({time.time() - t0:.0f}s)", flush=True)
