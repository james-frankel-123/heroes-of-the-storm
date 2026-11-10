"""
Audits of the judges' support bias that need only the saved per-team arrays
(judge_ood_calibration_teams.npz) and the per-player skill file
(overfit2026/cache/comp_causal_games.npz). Reviewer-suggested (Fable 5.2, Astra), 2026-11-10.

  transport   gap by judge-predicted probability within the low-affinity teams: agent teams are
              rare AND rated high, human rare teams are mostly rated low
  forms       five-fold cross-validated log loss of correction forms on the 1M leave-one-out
              training games (none / linear / binned / binned + support-dependent slope), and an
              extrapolation test: fit without the rarest 1% (triple or affinity), predict into it
  selection   on the games that have prior-day per-player skill, the support offset with and
              without skill controls (team skill difference, skill by pick position, party and
              never-played-hero counts), same games throughout

Usage (from training/): python3 paper1_revision/support_audit.py
Output: paper1_revision/site/results/support_audit.json
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.dirname(HERE)
sys.path.insert(0, T)
from paper1_revision.support_mitigation import lin_feats, bin_feats, fit_offset, logit, sig, QS

RES = os.path.join(HERE, "site", "results")
JUDGES = ["consensus", "gN", "RN", "QM2026"]


def load():
    z = dict(np.load(os.path.join(RES, "judge_ood_calibration_teams.npz"), allow_pickle=True))
    out = {}
    for s in ("train_loo", "heldout"):
        sel = z["set"] == s
        rid, team = z["rid"][sel], z["team"][sel]
        o0, o1 = np.where(team == 0)[0], np.where(team == 1)[0]
        o0, o1 = o0[np.argsort(rid[o0], kind="stable")], o1[np.argsort(rid[o1], kind="stable")]
        g = lambda k, o: z[k][sel][o]
        out[s] = {"rid": g("rid", o0), "y": g("y", o0), "p": {j: g("p_" + j, o0) for j in JUDGES},
                  "m0": {k: g("m_" + k, o0) for k in ("min3", "min2", "pmi", "dist", "n5")},
                  "m1": {k: g("m_" + k, o1) for k in ("min3", "min2", "pmi", "dist", "n5")}}
    return out


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def transport(D):
    """Team-level gap in the low-affinity cell, by the judge's predicted p for that team."""
    tr = D["train_loo"]
    res = {}
    allpmi = np.concatenate([tr["m0"]["pmi"], tr["m1"]["pmi"]])
    allm3 = np.concatenate([tr["m0"]["min3"], tr["m1"]["min3"]])
    p5_pmi, p5_m3 = np.percentile(allpmi, 5), np.percentile(allm3, 5)
    for j in JUDGES:
        p = np.concatenate([tr["p"][j], 1 - tr["p"][j]])
        y = np.concatenate([tr["y"], 1 - tr["y"]])
        rows = {}
        for name, sel in (("pmi_bottom5", allpmi <= p5_pmi), ("triple_bottom5", allm3 <= p5_m3),
                          ("both_bottom5", (allpmi <= p5_pmi) & (allm3 <= p5_m3)), ("all", np.ones(len(p), bool))):
            edges = [0, .40, .45, .50, .55, .60, 1.0]
            b = np.clip(np.searchsorted(edges, p[sel], side="right") - 1, 0, len(edges) - 2)
            yy, pp = y[sel], p[sel]
            rows[name] = [{"p_bin": f"{edges[i]:.2f}-{edges[i+1]:.2f}", "n": int((b == i).sum()),
                           "pred": float(pp[b == i].mean()) if (b == i).any() else None,
                           "gap": float((yy[b == i] - pp[b == i]).mean()) if (b == i).sum() > 100 else None,
                           "se": float((yy[b == i] - pp[b == i]).std() / np.sqrt((b == i).sum())) if (b == i).sum() > 100 else None}
                          for i in range(len(edges) - 1)]
        res[j] = rows
    return res


def design(m0, m1, form, edges):
    if form == "linear":
        return lin_feats(m0) - lin_feats(m1)
    return bin_feats(m0, edges) - bin_feats(m1, edges)


def fit_predict(o_tr, X_tr, y_tr, o_te, X_te, slope=None):
    """slope: optional symmetric support term multiplying logit p (both teams' low-support share)."""
    if slope is None:
        w = fit_offset(o_tr, X_tr, y_tr)
        return sig(o_te + X_te @ w)
    S_tr, S_te = slope
    Xa = np.concatenate([X_tr, (o_tr[:, None] * S_tr)], 1)
    Xb = np.concatenate([X_te, (o_te[:, None] * S_te)], 1)
    w = fit_offset(o_tr, Xa, y_tr)
    return sig(o_te + Xb @ w)


def forms(D):
    tr = D["train_loo"]
    n = len(tr["y"])
    allm = {k: np.concatenate([tr["m0"][k], tr["m1"][k]]) for k in ("min3", "pmi")}
    edges = {k: np.unique(np.percentile(allm[k], QS)).tolist() for k in ("min3", "pmi")}
    fold = (tr["rid"].astype(np.int64) * 2654435761 % 5)
    lowshare = lambda m0, m1: np.stack([(m0["pmi"] <= edges["pmi"][3]).astype(float) + (m1["pmi"] <= edges["pmi"][3]),
                                        (m0["min3"] <= edges["min3"][3]).astype(float) + (m1["min3"] <= edges["min3"][3])], 1)
    res = {"cv_ll": {}, "extrapolation": {}}
    for j in JUDGES:
        o = logit(tr["p"][j])
        Xl, Xb = design(tr["m0"], tr["m1"], "linear", edges), design(tr["m0"], tr["m1"], "binned", edges)
        S = lowshare(tr["m0"], tr["m1"])
        pred = {k: np.zeros(n) for k in ("none", "linear", "binned", "binned_slope")}
        for f in range(5):
            te, trn = fold == f, fold != f
            pred["none"][te] = sig(o[te])
            pred["linear"][te] = fit_predict(o[trn], Xl[trn], tr["y"][trn], o[te], Xl[te])
            pred["binned"][te] = fit_predict(o[trn], Xb[trn], tr["y"][trn], o[te], Xb[te])
            pred["binned_slope"][te] = fit_predict(o[trn], Xb[trn], tr["y"][trn], o[te], Xb[te], slope=(S[trn], S[te]))
        res["cv_ll"][j] = {k: ll(v, tr["y"]) for k, v in pred.items()}
        # tail gap (team0 in rarest 1% by affinity / triple) under each CV prediction
        for name, k, q in (("pmi_rarest1", "pmi", 1), ("triple_rarest1", "min3", 1)):
            tail = tr["m0"][k] <= np.percentile(allm[k], q)
            res["cv_ll"][j][f"gap_{name}"] = {f: float((tr["y"][tail] - v[tail]).mean()) for f, v in pred.items()}
        # extrapolation: fit on games where neither team is in the rarest 1%, predict games where team0 is
        ex = {}
        for name, k in (("pmi", "pmi"), ("triple", "min3")):
            cut = np.percentile(allm[k], 1)
            inner = (tr["m0"][k] > cut) & (tr["m1"][k] > cut)
            tail = (tr["m0"][k] <= cut) & (tr["m1"][k] > cut)
            for form, X in (("linear", Xl),):
                pt = fit_predict(o[inner], X[inner], tr["y"][inner], o[tail], X[tail])
                pf = fit_predict(o, X, tr["y"], o[tail], X[tail])
                ex[f"{name}_{form}"] = {"n_tail_games": int(tail.sum()), "gap_none": float((tr["y"][tail] - sig(o[tail])).mean()),
                                        "gap_fit_without_tail": float((tr["y"][tail] - pt).mean()),
                                        "gap_fit_with_tail": float((tr["y"][tail] - pf).mean())}
        res["extrapolation"][j] = ex
        print(j, {k: round(v, 5) for k, v in res["cv_ll"][j].items() if not k.startswith("gap")}, flush=True)
    return res


def selection(D):
    zc = np.load(os.path.join(T, "overfit2026", "cache", "comp_causal_games.npz"))
    crid = zc["rid"]
    res = {}
    for s in ("train_loo", "heldout"):
        d = D[s]
        idx = np.searchsorted(np.sort(crid), d["rid"])
        order = np.argsort(crid)
        sc = np.sort(crid)
        idx = np.clip(idx, 0, len(sc) - 1)
        hit = sc[idx] == d["rid"]
        ci = order[idx[hit]]
        ok_rank = zc["rank_ok"][ci].astype(bool) if "rank_ok" in zc.files else np.ones(hit.sum(), bool)
        res[s] = {"games_with_skill": int(hit.sum()), "of": int(len(hit))}
        if hit.sum() < 1000:
            continue
        # controls in team0 orientation of OUR games: comp_causal stores g (which team is team0?) -> align by y
        y_c = zc["y"][ci]
        flip = (y_c != d["y"][hit])  # if the stored outcome is the other orientation, swap team columns
        def oriented(a):
            a = np.asarray(a)[ci]
            if a.ndim == 1:
                return a
            out = a.copy()
            out[flip] = a[flip][:, ::-1] if a.ndim == 2 else a[flip][:, ::-1, ...]
            return out
        sk = oriented(zc["skill_clean"])
        pos = oriented(zc["skill_pos"])
        party = oriented(zc["partied"])
        never = oriented(zc["never_played_slots"])
        C1 = np.stack([sk[:, 0] - sk[:, 1]], 1)
        C2 = np.concatenate([C1, (pos[:, 0] - pos[:, 1]).reshape(len(pos), -1)], 1)
        C3 = np.concatenate([C2, (party[:, 0] - party[:, 1])[:, None], (never[:, 0] - never[:, 1])[:, None]], 1)
        m0 = {k: v[hit] for k, v in d["m0"].items()}
        m1 = {k: v[hit] for k, v in d["m1"].items()}
        X = lin_feats(m0) - lin_feats(m1)
        y = d["y"][hit]
        res[s]["flip_share"] = float(flip.mean())
        allpmi = np.concatenate([m0["pmi"], m1["pmi"]])
        tail = m0["pmi"] <= np.percentile(allpmi, 5)
        for j in ("consensus", "RN", "gN"):
            o = logit(d["p"][j][hit])
            r = {}
            for name, C in (("M0", None), ("M1_skill", C1), ("M2_pos", C2), ("M3_party_never", C3)):
                Xa = X if C is None else np.concatenate([X, C / (C.std(0) + 1e-9)], 1)
                w = fit_offset(o, Xa, y)
                # implied support effect on team0-low-affinity games: p with vs without the support terms
                ws = w[:X.shape[1]]
                eff = sig(o + Xa @ w) - sig(o + Xa @ w - X @ ws)
                r[name] = {"w_support": ws.tolist(), "support_effect_pmi_bottom5_pts": float(100 * eff[tail].mean()),
                           "raw_gap_pmi_bottom5_pts": float(100 * (y[tail] - sig(o[tail])).mean()), "n_tail": int(tail.sum())}
            res[s][j] = r
            print(s, j, {k: round(v["support_effect_pmi_bottom5_pts"], 2) for k, v in r.items()}, flush=True)
    return res


def main():
    D = load()
    out = {"transport": transport(D)}
    print("transport done", flush=True)
    out["forms"] = forms(D)
    out["selection"] = selection(D)
    json.dump(out, open(os.path.join(RES, "support_audit.json"), "w"), indent=1)
    print("wrote support_audit.json")


if __name__ == "__main__":
    main()
