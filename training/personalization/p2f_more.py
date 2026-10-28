"""
Paper 2 narrative facts, round 2: the player-model ladder in edge points
(item 7), the scoreboard normalization cards in edge points (item 9) and
Korea by history depth (item 11).

Inputs (scored arms; player keys only, gitignored caches):
  --arms dir with A2S.npz, full_ms.npz, full_ms_layer.npz and the ladder arms
     L_offset.npz, L_nopool.npz (p3_c_tests.predict with the A2S settings and
     the kernel replaced: offset = no skill state, nopool = each player x hero
     cell shrunk on its own, kernel t I with the phase-1 fit t = (3.88pp)^2)
  results/rcf/p3c/sb_*__merged.npz  (normalization cards, oct05 panel)
  cache/x_side_games.npz, cache/x_post_games.json.gz  (region)

Every arm scores the same V2 (72,474) and OOT (286,016) games with
logit p = c0 + c1 logit(WP) + c2 D, combiner fit on V1. The ladder's
"overall level only" rung re-scores A2S with its player+hero posterior
(the player's overall level plus a hero term, no learned hero traits),
combiner refit on V1 the same way.
Accuracy differences are paired by game; SE = sd of the per-game
difference in correctness / sqrt(games).

Usage (from training/): nice -n 19 taskset -c 48-51 python3 personalization/p2f_more.py --arms <dir>
Output: results/p2f/more.json
"""
import argparse
import gzip
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "p2f")
REGION = {1: "Americas", 2: "Europe", 3: "Korea"}


def load(p):
    z = np.load(p)
    return {k: z[k] for k in z.files if k != "meta"}


def ll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logistic(X, y, iters=50):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p)
        Hm = (X * (p * (1 - p))[:, None]).T @ X
        w += np.linalg.solve(Hm + 1e-9 * np.eye(len(w)), g)
    return w


def game_D(a, slot_value):
    gi = np.minimum(np.searchsorted(a["g_replay_id"], a["s_replay_id"]), len(a["g_replay_id"]) - 1)
    ok = a["g_replay_id"][gi] == a["s_replay_id"]
    D = np.zeros(len(a["g_replay_id"]))
    np.add.at(D, gi[ok], (np.where(a["s_team"] == 0, 1.0, -1.0) * slot_value)[ok])
    return D


def rescore(a, slot_value):
    """Combiner refit on V1 games (as p3_c_tests: games with all ten slots)."""
    D = game_D(a, slot_value)
    fit = a["g_win"] == 1
    w = fit_logistic(np.column_stack([a["g_lo"], D])[fit], a["g_y"][fit])
    return 1 / (1 + np.exp(-(w[0] + w[1] * a["g_lo"] + w[2] * D)))


def summ(a, p, m):
    y = a["g_y"][m]
    return {"games": int(m.sum()), "acc": float(((p[m] > 0.5) == (y == 1)).mean()),
            "edge_pts": float(100 * ((p[m] > 0.5) == (y == 1)).mean() - 50),
            "ll_gain_x1000": float(1000 * (ll(a["g_p0"][m], y) - ll(p[m], y)).mean())}


def paired(a, p1, b, p0, m):
    """accuracy difference (points) of p1 (arm a) minus p0 (arm b) on the same games."""
    assert (a["g_replay_id"] == b["g_replay_id"]).all()
    y = a["g_y"][m]
    c1 = ((p1[m] > 0.5) == (y == 1)).astype(float)
    c0 = ((p0[m] > 0.5) == (y == 1)).astype(float)
    d = c1 - c0
    l = 1000 * (ll(p0[m], y) - ll(p1[m], y))
    return {"acc_diff_pts": float(100 * d.mean()), "se_pts": float(100 * d.std() / np.sqrt(len(d))),
            "ll_diff_x1000": float(l.mean()), "ll_se": float(l.std() / np.sqrt(len(l)))}


def regions():
    sg = np.load(os.path.join(HERE, "cache", "x_side_games.npz"))
    rid = [sg["replay_ids"]]
    reg = [sg["region"].astype(np.int64)]
    post = json.load(gzip.open(os.path.join(HERE, "cache", "x_post_games.json.gz"), "rt"))
    rid.append(np.array([g["replay_id"] for g in post], np.int64))
    reg.append(np.array([-1 if g["region"] is None else g["region"] for g in post], np.int64))
    rid, reg = np.concatenate(rid), np.concatenate(reg)
    o = np.argsort(rid)
    return rid[o], reg[o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    a_ = ap.parse_args()
    out = {"method": __doc__.split("Every arm")[1].split("Usage")[0].strip()}
    A = {k: load(os.path.join(a_.arms, f"{k}.npz")) for k in ("A2S", "full_ms", "full_ms_layer", "L_offset", "L_nopool")
         if os.path.exists(os.path.join(a_.arms, f"{k}.npz"))}
    ref = A["A2S"]
    P = {}
    P["draft only"] = (ref, ref["g_p0"])
    if "L_offset" in A:
        P["+ experience (games played, no results)"] = (A["L_offset"], A["L_offset"]["g_p"])
    if "L_nopool" in A:
        P["+ win records, sample-size weighted, no borrowing"] = (A["L_nopool"], A["L_nopool"]["g_p"])
    P["+ borrowing: player's overall level"] = (ref, rescore(ref, (ref["s_mu"] + ref["s_mean_ph"]).astype(np.float64)))
    P["+ borrowing: learned hero traits (win-only model, A2S)"] = (ref, ref["g_p"])
    P["+ scoreboard performance"] = (A["full_ms"], A["full_ms"]["g_p"])
    P["+ per-player learning speed and drift layer (main model)"] = (A["full_ms_layer"], A["full_ms_layer"]["g_p"])
    lad = {}
    names = list(P)
    for wn, wv in (("V2", 2), ("OOT", 3)):
        m = ref["g_win"] == wv
        rows = {}
        for i, nm in enumerate(names):
            arm, p = P[nm]
            rows[nm] = summ(arm, p, m)
            if i > 0:
                parm, pp = P[names[i - 1]]
                rows[nm]["step_vs_previous"] = paired(arm, p, parm, pp, m)
        lad[wn] = rows
    out["ladder"] = lad
    for wn in lad:
        print(wn)
        for nm, r in lad[wn].items():
            s = r.get("step_vs_previous", {})
            print(f"  {nm:60s} {100 * r['acc']:.2f}% edge {r['edge_pts']:.2f} ll {r['ll_gain_x1000']:.1f} "
                  f"step {s.get('acc_diff_pts', 0):+.2f} ({s.get('se_pts', 0):.2f})")

    # ---- item 9: normalization cards
    cd = os.path.join(HERE, "results", "rcf", "p3c")
    cards = {}
    base = load(os.path.join(cd, "sb_base__merged.npz"))
    for f in sorted(os.listdir(cd)):
        if not f.startswith("sb_"):
            continue
        c = load(os.path.join(cd, f))
        nm = f[3:].split("__")[0]
        cards[nm] = {}
        for wn, wv in (("V2", 2), ("OOT", 3)):
            m = base["g_win"] == wv
            cards[nm][wn] = {"acc": float(((c["g_p"][m] > 0.5) == (c["g_y"][m] == 1)).mean()),
                             "vs_base": paired(c, c["g_p"], base, base["g_p"], m)}
    # the first export (met2_hs2) for scale
    for extra in ("met2_hs2", "today"):
        c = load(os.path.join(cd, f"{extra}__merged.npz"))
        cards[extra] = {}
        for wn, wv in (("V2", 2), ("OOT", 3)):
            m = base["g_win"] == wv
            cards[extra][wn] = {"acc": float(((c["g_p"][m] > 0.5) == (c["g_y"][m] == 1)).mean()),
                                "vs_base": paired(c, c["g_p"], base, base["g_p"], m)}
    out["cards"] = cards
    for nm, v in cards.items():
        print("card", nm, {w: (round(100 * x["acc"], 2), round(x["vs_base"]["acc_diff_pts"], 2), round(x["vs_base"]["se_pts"], 2),
                               round(x["vs_base"]["ll_diff_x1000"], 2)) for w, x in v.items()})

    # ---- item 11: Korea by history depth (main model and win-only)
    rk, rg = regions()
    kor = {}
    for armn in ("full_ms_layer", "A2S"):
        M = A[armn]
        j = np.minimum(np.searchsorted(rk, M["g_replay_id"]), len(rk) - 1)
        greg = np.where(rk[j] == M["g_replay_id"], rg[j], -1)
        gi = np.minimum(np.searchsorted(M["g_replay_id"], M["s_replay_id"]), len(M["g_replay_id"]) - 1)
        ng = len(M["g_replay_id"])
        # history depth of a game: median over its ten players of games seen (full history, as of the day before)
        npl = M["s_np"].astype(np.float64)
        depth = np.full(ng, np.nan)
        o = np.argsort(gi, kind="stable")
        gs, vs = gi[o], npl[o]
        st = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
        en = np.r_[st[1:], len(gs)]
        for s, e in zip(st, en):
            depth[gs[s]] = np.median(vs[s:e])
        pulled = np.bincount(gi, weights=M["s_pulled"].astype(float), minlength=ng)
        heavy = np.bincount(gi, weights=(M["s_nh"] >= 50).astype(float), minlength=ng)
        res = {}
        edges = [0, 100, 300, 1000, 3000, 1e12]
        labs = ["<100", "100-299", "300-999", "1,000-2,999", "3,000+"]
        for wn, wv in (("V2", 2), ("OOT", 3), ("V2+OOT", None)):
            base_m = (M["g_win"] >= 2) if wv is None else (M["g_win"] == wv)
            r = {}
            for rv, rn in REGION.items():
                m = base_m & (greg == rv)
                y = M["g_y"][m]
                acc0 = ((M["g_p0"][m] > 0.5) == (y == 1))
                acc1 = ((M["g_p"][m] > 0.5) == (y == 1))
                r[rn] = {"games": int(m.sum()), "median_depth": float(np.median(depth[m])),
                         "mean_pulled_players": float(pulled[m].mean()),
                         "mean_slots_with_50plus_on_hero": float(heavy[m].mean()),
                         "gain_pts": float(100 * (acc1.mean() - acc0.mean())),
                         "ll_gain_x1000": float(1000 * (ll(M["g_p0"][m], y) - ll(M["g_p"][m], y)).mean()),
                         "by_depth": {}}
                for (lo, hi), lb in zip(zip(edges[:-1], edges[1:]), labs):
                    mm = m & (depth >= lo) & (depth < hi)
                    if mm.sum() < 100:
                        continue
                    yy = M["g_y"][mm]
                    d = ((M["g_p"][mm] > 0.5) == (yy == 1)).astype(float) - ((M["g_p0"][mm] > 0.5) == (yy == 1)).astype(float)
                    lg = 1000 * (ll(M["g_p0"][mm], yy) - ll(M["g_p"][mm], yy))
                    r[rn]["by_depth"][lb] = {"games": int(mm.sum()), "share": float(mm.sum() / m.sum()),
                                             "gain_pts": float(100 * d.mean()), "se_pts": float(100 * d.std() / np.sqrt(len(d))),
                                             "ll_gain_x1000": float(lg.mean()), "ll_se": float(lg.std() / np.sqrt(len(lg)))}
            # Americas and Europe reweighted to Korea's depth mix
            kd = r["Korea"]["by_depth"]
            for rn in ("Americas", "Europe"):
                bd = r[rn]["by_depth"]
                common = [k for k in kd if k in bd]
                wsum = sum(kd[k]["share"] for k in common)
                r[rn]["gain_pts_at_korea_depth_mix"] = sum(kd[k]["share"] * bd[k]["gain_pts"] for k in common) / wsum
                r[rn]["ll_gain_at_korea_depth_mix"] = sum(kd[k]["share"] * bd[k]["ll_gain_x1000"] for k in common) / wsum
            res[wn] = r
        kor[armn] = res
        print(armn, json.dumps({w: {rn: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items() if k != "by_depth"}
                                    for rn, x in r.items()} for w, r in res.items()}, indent=0)[:3000])
    out["korea"] = kor
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "more.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
