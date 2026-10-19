"""
P3 tournament analysis: head-to-heads, standings, Bradley-Terry, seed
effects, safety, judge-free scores, compute cost, drift.

Inputs: $P3_CACHE/p3t/tourney/<A>__<B>.npz (drafts) and
$P3_CACHE/p3t/scores/<A>__<B>.npz (p3t_judge.py score).
Every quantity is from the side of the first-named system, averaged over
both seatings where both ordered pairs exist:
  h2h(X, Y) = 0.5 * (m(X seat 0 vs Y) + 1 - m(Y seat 0 vs X))   for win probabilities
  diff(X, Y) = 0.5 * (d(X, Y) - d(Y, X))                          for side metrics (X side minus Y side)
CIs: lobby-cluster bootstrap (the same 300 lobbies in every pair, resampled
jointly), 2,000 draws. Seed SE: crossed random effects of the two sides'
prior seeds by method of moments on cell means (drift paper w8_inference);
for paired seeds (both sides use seed i mod 5) the SE of the mean over seed
cells. The reported CI uses the larger of the two SEs.

Labels: J = player-aware judge (consensus + personal term, model-judged);
CONS = population consensus (model-judged); judge-free = realized outcomes
of later games; OWN = the drafter's own value (model-internal).
Output: $P3_RESULTS/p3t/analysis.json and tables printed as markdown.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("P1_TIERS", "site")
import numpy as np

import p3_hs_core as C
import p3t_data as TD

_LS = os.environ.get("P3T_LOBBYSET", "")
TDIR = os.path.join(TD.OUT, "tourney" + (f"_{_LS}" if _LS else ""))
SPLIT = os.environ.get("P3T_SPLIT") == "1"
SDIR = os.path.join(TD.OUT, ("scores_split" if SPLIT else "scores") + (f"_{_LS}" if _LS else ""))
RES = os.path.join(C.RESULTS, "p3t")
B = 2000
WIN = ["judge_personal", "judge_corr", "consensus", "gN", "gN_naive", "RN", "QM2026"]
SIDE = {"rec_mean": "personal realized record (pp, judge-free)", "rec_pool": "personal realized record, game-weighted",
        "truth_hero": "future hero WR", "truth_syn": "future synergy", "truth_ctr": "future counter",
        "degen": "degenerate", "healer": "has healer", "offrole": "off-role players", "rec_cov": "record coverage"}


def n_seeds(sysname):
    import p3t_systems as SY
    s = SY.SYSTEMS[sysname]
    if s["kind"] in ("mcts", "policy"):
        return SY.vintage(s["vint"]).n_seeds
    return 1


def attenuation():
    """lambda: slope of the judge's personal sum on the drafter's, real teams
    (p3t_diag). The judge sees only 2.5 months of later games, so its skill
    estimates are shrunk harder than the drafter's; a calibrated judge then
    credits about lambda of a true personal difference."""
    p = os.path.join(RES, "diag_split.json" if SPLIT else "diag.json")
    return json.load(open(p))["real"]["slope_SJ_on_SD"] if os.path.exists(p) else None


def load():
    P = {}
    lam = attenuation()
    judge = json.load(open(os.path.join(RES, "judge_split.json" if SPLIT else "judge.json")))
    for f in sorted(os.listdir(SDIR)):
        if "__" not in f or not f.endswith(".npz"):
            continue
        a, b = f[:-4].split("__")
        z = np.load(os.path.join(SDIR, f))
        t = np.load(os.path.join(TDIR, f))
        e = {k: z[k] for k in z.files}
        tr = e.pop("truth")
        for j, nm in enumerate(("truth_hero", "truth_syn", "truth_ctr")):
            e[nm] = tr[:, :, j]
        if lam:
            # attenuation-corrected J (model-based sensitivity, not a judge): the
            # personal skill part divided by lambda
            pc = np.clip(e["consensus"], 1e-6, 1 - 1e-6)
            e["judge_corr"] = 1 / (1 + np.exp(-(np.log(pc / (1 - pc)) + judge["c2_skill"] * e["dS_judge"] / lam
                                                 + judge["c3_offrole"] * e["dO_judge"])))
        else:
            e["judge_corr"] = e["judge_personal"]
        e["secs"] = t["secs"]
        e["fallback"] = t["fallback"]
        e["acts"] = t["acts"]
        P[(a, b)] = e
    return P


def side_vals(e, key):
    """(n, 2) side values; win metrics are P(seat 0 wins)."""
    v = e[key].astype(float)
    return v


def pair_win(P, a, b, key, idx=None):
    """h2h of a over b on judge `key` (mean over lobbies idx)."""
    out = []
    if (a, b) in P:
        x = P[(a, b)][key]
        out.append(x if idx is None else x[idx])
    if (b, a) in P:
        x = 1 - P[(b, a)][key]
        out.append(x if idx is None else x[idx])
    if not out:
        return None
    return np.mean([o.mean() for o in out])


def pair_diff(P, a, b, key, idx=None):
    out = []
    for (x, y), sg in (((a, b), 1), ((b, a), -1)):
        if (x, y) in P:
            v = P[(x, y)][key].astype(float)
            d = (v[:, 0] - v[:, 1]) * sg
            d = d if idx is None else d[idx]
            out.append(np.nanmean(d))
    return np.mean(out) if out else None


def pair_signed(P, a, b, key, idx=None):
    """seat-averaged per-draft quantity that is team0-minus-team1 (sign flips with the seat)."""
    out = []
    for (x, y), sg in (((a, b), 1), ((b, a), -1)):
        if (x, y) in P:
            v = P[(x, y)][key].astype(float) * sg
            out.append(np.nanmean(v if idx is None else v[idx]))
    return np.mean(out) if out else None


def side_mean(P, s, key, idx=None):
    """mean of system s's own side over all its drafts."""
    vals = []
    for (x, y), e in P.items():
        v = e[key].astype(float)
        if idx is not None:
            v = v[idx]
        if x == s:
            vals.append(v[:, 0])
        if y == s:
            vals.append(v[:, 1])
    return float(np.nanmean(np.concatenate(vals))) if vals else None


def seed_se(P, a, b, key, n):
    """Crossed seed random effects SE for the h2h of a over b."""
    na, nb = n_seeds(a), n_seeds(b)
    i = np.arange(n)
    cells = {}
    for (x, y), sg in (((a, b), 1), ((b, a), -1)):
        if (x, y) not in P:
            continue
        v = P[(x, y)][key]
        v = v if sg == 1 else 1 - v
        for j in range(n):
            cells.setdefault((j % na, j % nb), []).append(v[j])
    if na == 1 and nb == 1:
        return 0.0
    keys = sorted(cells)
    m = np.array([np.mean(cells[k]) for k in keys])
    if na == nb and all(k[0] == k[1] for k in keys):
        return float(np.std(m, ddof=1) / np.sqrt(len(m)))
    A = sorted({k[0] for k in keys})
    Bs = sorted({k[1] for k in keys})
    M = np.full((len(A), len(Bs)), np.nan)
    for k, mv in zip(keys, m):
        M[A.index(k[0]), Bs.index(k[1])] = mv
    if np.isnan(M).any() or (len(A) < 2 and len(Bs) < 2):
        return float(np.std(m, ddof=1) / np.sqrt(len(m))) if len(m) > 1 else 0.0
    g = M.mean()
    ra = M.mean(1) - g
    cb = M.mean(0) - g
    res = M - g - ra[:, None] - cb[None, :]
    nA, nB = len(A), len(Bs)
    ms_int = (res ** 2).sum() / max((nA - 1) * (nB - 1), 1)
    sa2 = max((nB * (ra ** 2).sum() / max(nA - 1, 1) - ms_int) / nB, 0) if nA > 1 else 0
    sb2 = max((nA * (cb ** 2).sum() / max(nB - 1, 1) - ms_int) / nA, 0) if nB > 1 else 0
    se2 = sa2 / nA + sb2 / nB + ms_int / (nA * nB)
    return float(np.sqrt(se2))


def bt_fit(P, systems, key, idx=None, iters=200):
    """Bradley-Terry with a seat (first-pick) term on fractional outcomes:
    logit P(seat-0 system wins) = th_a - th_b + h. th_GD = 0."""
    import torch
    S = {s: i for i, s in enumerate(systems)}
    pa, pb, y, w = [], [], [], []
    for (a, b), e in P.items():
        if a in S and b in S:
            v = e[key] if idx is None else e[key][idx]
            pa.append(S[a])
            pb.append(S[b])
            y.append(float(np.mean(v)))
            w.append(len(v))
    pa, pb = torch.tensor(pa), torch.tensor(pb)
    y, w = torch.tensor(y, dtype=torch.float64), torch.tensor(w, dtype=torch.float64)
    th = torch.zeros(len(systems), dtype=torch.float64, requires_grad=True)
    h = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([th, h], max_iter=iters, line_search_fn="strong_wolfe")
    ref = S.get("GD", 0)

    def closure():
        opt.zero_grad()
        t = th - th[ref]
        z = t[pa] - t[pb] + h
        p = torch.sigmoid(z).clamp(1e-9, 1 - 1e-9)
        loss = -(w * (y * torch.log(p) + (1 - y) * torch.log(1 - p))).sum()
        loss.backward()
        return loss
    opt.step(closure)
    t = (th - th[ref]).detach().numpy()
    return {s: float(t[S[s]]) for s in systems}, float(h.detach())


def boot_idx(n, b, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(b)]


def ci(stat_fn, idxs, point):
    bs = np.array([stat_fn(ix) for ix in idxs], float)
    se = float(np.nanstd(bs))
    return point, se


def analyze(contrasts_extra=()):
    import p3t_systems as SY
    import p3t_tourney as TT
    P = load()
    if not P:
        raise SystemExit("no scored pairs")
    n = len(next(iter(P.values()))["consensus"])
    idxs = boot_idx(n, B)
    systems = sorted({s for p in P for s in p})
    out = {"lobbies": n, "pairs": len(P), "systems": systems, "attenuation_lambda": attenuation()}
    # --- head-to-heads of interest
    H = [("aMCTS-cur", "pMCTS-cur"), ("a1step-cur", "p1step-cur"), ("aMCTS-cur", "a1step-cur"),
         ("pMCTS-cur", "pPol-cur"), ("aMCTS-cur", "Real"), ("pMCTS-cur", "Real"), ("aMCTS-cur", "GD"),
         ("pMCTS-cur", "GD"), ("a1step-cur", "Real"),
         ("pMCTS-cur", "pMCTS-s4m"), ("pMCTS-cur", "pMCTS-s1y"),
         ("pMCTS-s4m-rs", "pMCTS-s4m"), ("pMCTS-s1y-rs", "pMCTS-s1y"),
         ("pMCTS-s4m-ft", "pMCTS-s4m"), ("pMCTS-s1y-ft", "pMCTS-s1y"),
         ("aMCTS-cur", "aMCTS-s4m"), ("aMCTS-cur", "aMCTS-s1y"),
         ("aMCTS-cur", "aMCTS-cur@s4m"), ("aMCTS-cur", "aMCTS-cur@s1y"),
         ("aMCTS-s4m", "pMCTS-s4m"), ("aMCTS-s1y", "pMCTS-s1y"),
         ("pPol-cur", "pPol-s4m"), ("pPol-cur", "pPol-s1y"),
         ("a1step-cur", "a1step-s1y"), ("p1step-cur", "p1step-s1y"), ("a1step-s1y", "p1step-s1y"),
         ("aMCTS-bc", "pMCTS-bc"), ("aMCTS-bc", "aMCTS-cur"), ("a1step-cur", "aMCTS-bc"), ("a1step-cur", "pMCTS-cur"),
         ("aMCTS-bc", "pMCTS-cur"), ("pMCTS-bc", "pMCTS-cur"), ("a1step-cur", "GD"), ("aMCTS-bc", "Real"),
         ("p1step-cur", "Real"), ("Real", "GD"), ("pPol-cur", "Real"),
         ("aMCTS-pp", "aMCTS-cur"), ("aMCTS-pp", "pMCTS-cur"), ("aMCTS-pp", "aMCTS-bc"), ("aMCTS-pp", "Real"),
         ("aMCTS-sp", "aMCTS-cur"), ("aMCTS-sp", "pMCTS-cur"), ("aMCTS-sp", "aMCTS-pp"), ("aMCTS-sp", "Real"),
         ("aPol-pp", "pPol-cur"), ("aPol-sp", "pPol-cur"), ("aMCTS-vn", "aMCTS-bc"), ("aMCTS-vn", "pMCTS-bc"),
         ("DraftRec", "aMCTS-bc"), ("DraftRec", "a1step-cur"), ("DraftRec", "GD"), ("DraftRec", "Real"),
         ("DraftRec", "pMCTS-cur"), ("aMCTS-bc@split", "pMCTS-bc"), ("a1step@split", "p1step-cur"),
         ("aMCTS-bc@split", "aMCTS-bc"), ("a1step@split", "a1step-cur")] + list(contrasts_extra)
    # every unordered pair that was played (both seatings), listed after the named contrasts
    seen = {frozenset(x) for x in H}
    for (x, y) in sorted(P):
        if frozenset((x, y)) not in seen and (y, x) in P:
            seen.add(frozenset((x, y)))
            H.append((x, y))
    h2h = {}
    for a, b in H:
        if not ((a, b) in P or (b, a) in P):
            continue
        r = {}
        for key in WIN:
            pt = pair_win(P, a, b, key)
            _, se_b = ci(lambda ix: pair_win(P, a, b, key, ix), idxs[:500], pt)
            se_s = seed_se(P, a, b, key, n)
            se = max(se_b, se_s)
            r[key] = [round(100 * pt, 2), round(100 * (pt - 1.96 * se), 2), round(100 * (pt + 1.96 * se), 2),
                      round(100 * se_b, 2), round(100 * se_s, 2)]
        for key in ("dS_judge",):
            # personal part of J: the judge's personal-sum difference (logit units = c2 * dS), seat-averaged
            pt = pair_signed(P, a, b, key)
            _, se_b = ci(lambda ix: pair_signed(P, a, b, key, ix), idxs[:500], pt)
            r["dS_judge_pp"] = [round(100 * pt, 2), round(100 * (pt - 1.96 * se_b), 2), round(100 * (pt + 1.96 * se_b), 2)]
        for key in ("rec_mean", "rec_pool", "truth_hero", "truth_syn", "truth_ctr", "degen", "offrole"):
            pt = pair_diff(P, a, b, key)
            _, se_b = ci(lambda ix: pair_diff(P, a, b, key, ix), idxs[:500], pt)
            sc = 100 if key in ("rec_mean", "rec_pool", "degen") else 1
            r["diff_" + key] = [round(sc * pt, 3), round(sc * (pt - 1.96 * se_b), 3), round(sc * (pt + 1.96 * se_b), 3)]
        for side_key in ("own_V", "own_wp"):
            # model-internal: each system's own valuation of its own side, seat-averaged
            for who in (a, b):
                vals = []
                for (x, y) in ((a, b), (b, a)):
                    if (x, y) not in P:
                        continue
                    e = P[(x, y)]
                    if who == x:
                        vals.append(e[f"{side_key}_a"])
                    else:
                        vals.append(1 - e[f"{side_key}_b"])
                r[f"{side_key}[{who}]"] = round(100 * float(np.mean(np.concatenate(vals))), 2)
        h2h[f"{a} vs {b}"] = r
    out["h2h"] = h2h
    # --- standings (core round robin), safety, cost
    core = [s for s in TT.CORE if s in systems]
    st = {}
    for s in systems:
        opp = [o for o in core if o != s and ((s, o) in P or (o, s) in P)]
        row = {}
        for key in ("judge_personal", "consensus"):
            vals = [pair_win(P, s, o, key) for o in opp]
            row[key] = round(100 * float(np.mean(vals)), 2) if vals else None
        for key in ("degen", "healer", "offrole", "rec_mean", "rec_cov", "truth_hero"):
            v = side_mean(P, s, key)
            row[key] = None if v is None else round(v * (100 if key in ("degen", "healer", "rec_mean", "rec_cov") else 1), 3)
        secs, nd = 0.0, 0
        for (x, y), e in P.items():
            if x == s:
                secs += float(e["secs"][0])
                nd += n
            if y == s:
                secs += float(e["secs"][1])
                nd += n
        row["sec_per_draft"] = round(secs / max(nd, 1), 3)
        if s == "Real":
            fb = [e["fallback"][:, [k for k in range(16) if SY.DRAFT_TEAM[k] == (0 if x == s else 1)]].mean()
                  for (x, y), e in P.items() if s in (x, y)]
            row["real_fallback_rate"] = round(100 * float(np.mean(fb)), 2)
        row["opponents_in_core"] = len(opp)
        st[s] = row
    out["standings"] = st
    # --- Bradley-Terry
    bt = {}
    for key in ("judge_personal", "consensus"):
        th, h = bt_fit(P, systems, key)
        bsv = []
        for ix in idxs[:200]:
            t_, _ = bt_fit(P, systems, key, ix)
            bsv.append([t_[s] for s in systems])
        bsv = np.array(bsv)
        lo, hi = np.percentile(bsv, [2.5, 97.5], axis=0)
        bt[key] = {"seat_first_pick": round(h, 3),
                   "theta": {s: [round(th[s], 3), round(float(l), 3), round(float(u), 3)] for s, l, u in zip(systems, lo, hi)}}
    out["bt"] = bt
    if _LS == "tune":
        # selection score of the exploration study: mean J (and CONS) win probability against the tune opponents
        tune = {}
        for x in systems:
            if x in TT.TUNE_OPP:
                continue
            row = {}
            for key in ("judge_personal", "consensus", "judge_corr"):
                vals = [pair_win(P, x, o, key) for o in TT.TUNE_OPP if (x, o) in P or (o, x) in P]
                if len(vals) == len(TT.TUNE_OPP):
                    row[key] = round(100 * float(np.mean(vals)), 2)
                    row[key + "_by_opp"] = {o: round(100 * float(v), 2) for o, v in zip(TT.TUNE_OPP, vals)}
            if row:
                tune[x] = row
        out["tune"] = dict(sorted(tune.items(), key=lambda kv: -kv[1].get("judge_personal", 0)))
    name = ("analysis_split" if SPLIT else "analysis") + (f"_{_LS}" if _LS else "")
    json.dump(out, open(os.path.join(RES, name + ".json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    analyze()


if __name__ == "__main__":
    main()
