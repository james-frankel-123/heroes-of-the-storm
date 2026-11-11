"""
Follow-up audits of the judges' support bias (Fable 5.2 audit, 2026-11-11, checks 2-3).

  transport_net  team-level gap y - p in support cells, NETTED against the judge's own
                 calibration curve (50 quantile bins of p over all teams): what is left is
                 the support-specific error, not the judge's general over/under-confidence.
                 Cells: own team rare (bottom 5% affinity or rarest triple) with the
                 opponent rare or not (two-sided), and, for own-rare / opponent-common,
                 by predicted p (agent teams are rare AND rated high). SE clustered by game.
  core_fe        the four-hero-core check (core_check.py) with a fifth-hero fixed effect
                 (two-way demeaning: core and fifth hero), on all completions and on
                 role-valid completions only (constrained_search._valid_final), so the
                 slope is not a fifth-hero strength effect or a role-duplicate effect.

Usage (from training/, p1site env): python3 paper1_revision/support_audit2.py
Output: paper1_revision/site/results/support_audit2.json
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.dirname(HERE)
sys.path.insert(0, T)
from paper1_revision import core_check as cc

RES = os.path.join(HERE, "site", "results")
JUDGES = ["consensus", "gN", "gN_naive", "RN", "QM2026"]


def paired(z, sel):
    """Team rows of the selected set with the opponent row's index."""
    rid, team = z["rid"][sel], z["team"][sel]
    o = np.lexsort((team, rid))
    assert (rid[o][0::2] == rid[o][1::2]).all()
    opp = np.empty(len(o), np.int64)
    opp[o[0::2]], opp[o[1::2]] = o[1::2], o[0::2]
    return opp


def gse(x, g):
    u = np.unique(g, return_inverse=True)[1]
    mu = x.mean()
    return float(mu), float(np.sqrt((np.bincount(u, x - mu) ** 2).sum()) / len(x))


def transport_net():
    z = np.load(os.path.join(RES, "judge_ood_calibration_teams.npz"), allow_pickle=True)
    sel = z["set"] == "train_loo"
    opp = paired(z, sel)
    y, rid = z["y"][sel].astype(float), z["rid"][sel]
    pmi, m3 = z["m_pmi"][sel], z["m_min3"][sel]
    rare = {"pmi": pmi <= np.percentile(pmi, 5), "triple": m3 <= np.percentile(m3, 5)}
    out = {}
    for j in JUDGES:
        p = z["p_" + j][sel].astype(float)
        e = np.unique(np.percentile(p, np.linspace(0, 100, 51)))
        b = np.clip(np.searchsorted(e, p, side="right") - 1, 0, len(e) - 2)
        calib = np.bincount(b, y - p, len(e) - 1) / np.maximum(np.bincount(b, minlength=len(e) - 1), 1)
        net = y - p - calib[b]
        r = {}
        for k, own in rare.items():
            oth = own[opp]
            for cell, m in (("own_rare_opp_common", own & ~oth), ("both_rare", own & oth),
                            ("own_common_opp_rare", ~own & oth)):
                r[f"{k}/{cell}"] = {"n": int(m.sum()), "raw": gse(100 * (y - p)[m], rid[m]),
                                    "net": gse(100 * net[m], rid[m]), "pred": float(p[m].mean())}
            m = own & ~oth
            for lo, hi in ((0, .45), (.45, .5), (.5, .55), (.55, .6), (.6, 1.01)):
                mm = m & (p >= lo) & (p < hi)
                if mm.sum() > 200:
                    r[f"{k}/own_rare_opp_common/p{lo:.2f}-{hi:.2f}"] = {
                        "n": int(mm.sum()), "raw": gse(100 * (y - p)[mm], rid[mm]), "net": gse(100 * net[mm], rid[mm])}
        out[j] = r
        print(j, {c: (round(v["raw"][0], 2), round(v["net"][0], 2)) for c, v in r.items() if "/p" not in c[-12:] and c.count("/") == 1},
              flush=True)
    return out


def twoway(x, g1, g2, n1, n2, it=30):
    x = x.astype(np.float64).copy()
    c1, c2 = np.bincount(g1, minlength=n1), np.bincount(g2, minlength=n2)
    for _ in range(it):
        x -= (np.bincount(g1, x, n1) / c1)[g1]
        x -= (np.bincount(g2, x, n2) / np.maximum(c2, 1))[g2]
    return x


def core_fe():
    from rerun2026.constrained_search import HERO_BLIZZ_IDX, BLIZZ_ROLES, _valid_final
    d, A = cc.load()
    ck, fifth, row = cc.long_cores(A)
    R = HERO_BLIZZ_IDX[A]
    valid_team = np.array([_valid_final(tuple(np.bincount(r, minlength=len(BLIZZ_ROLES)))) for r in R])
    u, g = np.unique(ck, return_inverse=True)
    cnt = np.bincount(g)
    nf = np.bincount(np.unique(g.astype(np.int64) * 90 + fifth) // 90, minlength=len(u))
    out = {"valid_team_share": float(valid_team.mean())}
    for name, extra in (("all", np.ones(len(g), bool)), ("role_valid", valid_team[row])):
        m = (cnt >= cc.MIN_CORE)[g] & (nf >= 2)[g] & extra
        # recount after the role filter
        gm = np.unique(g[m], return_inverse=True)[1]
        keep = np.bincount(gm) >= cc.MIN_CORE
        m2 = np.zeros(len(g), bool)
        m2[np.flatnonzero(m)[keep[gm]]] = True
        g1 = np.unique(g[m2], return_inverse=True)[1]
        g2 = fifth[m2]
        n1, n2 = g1.max() + 1, 90
        r = row[m2]
        _, rid_c = np.unique(d["rid"][r], return_inverse=True)
        res = {"cores": int(n1), "obs": int(m2.sum())}
        for fe, f in (("core", lambda x: cc.demean(x, g1, n1)), ("core+fifth", lambda x: twoway(x, g1, g2, n1, n2))):
            yv = f(d["y"][r].astype(float))
            for sname, sraw in (("pmi", d["m_pmi"]), ("log_min3", np.log1p(d["m_min3"]))):
                s = f(sraw[r].astype(float))
                sd = float(s.std())
                rr = {"sd": sd, "raw_slope_per_sd": [x * sd * 100 for x in cc.slope(yv, s, (g1, rid_c))]}
                for j in JUDGES:
                    p = f(d["p"][j][r].astype(float))
                    rr[j] = {"judged": [x * sd * 100 for x in cc.slope(p, s, (g1, rid_c))],
                             "gap": [x * sd * 100 for x in cc.slope(yv - p, s, (g1, rid_c))]}
                res[f"{fe}/{sname}"] = rr
                print(name, fe, sname, "actual %.2f" % rr["raw_slope_per_sd"][0],
                      {j: "%.2f±%.2f" % tuple(rr[j]["gap"]) for j in JUDGES}, flush=True)
        out[name] = res
    return out


def main():
    out = {"transport_net": transport_net(), "core_fe": core_fe()}
    json.dump(out, open(os.path.join(RES, "support_audit2.json"), "w"), indent=1)
    print("wrote support_audit2.json")


if __name__ == "__main__":
    main()
