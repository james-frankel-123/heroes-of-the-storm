"""
P3 tournament: value functions of the drafters and their prediction quality.

combfit   per vintage, the drafter's combiner
            V = sigmoid(b0 + b1 logit WP_v + b2 (S_0 - S_1) + b3 (O_0 - O_1))
          fit on real games the vintage's WP never trained on (paper-1 test
          and validation splits, plus snapshot-era games uploaded after the
          snapshot), all ten slots in the slot table, in the window FIT_WIN[v]
          before the vintage's cutoff. S uses the as-of skill state (offset +
          posterior + layer, state through the day before), O the off-role
          indicator of the played hero.
predeval  prediction quality at staleness on the tournament-window lobbies
          (real drafts, real outcomes, 2026-05-23 to 06-30): log loss and
          accuracy of WP_v alone, and of the full value with the vintage's
          player state (frozen at the cutoff) or the current state.

Run (3090, env of remote/run_rcf_full.sh):
  python personalization/p3t_value.py combfit --vintages cur[,s4m,s1y]
  python personalization/p3t_value.py predeval
Outputs: $P3_RESULTS/p3t/{combiner,predeval}.json; combiners are copied into
p3t_vintages.json by hand (they are small and part of the record).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("P1_TIERS", "site")   # paper-1 artifacts of the site-tier rebuild
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import p3_hs_core as C
import p3t_data as TD

FIT_WIN = {"cur": ("2026-02-11", "2026-05-22"), "s4m": ("2025-11-01", "2026-02-10"),
           "s1y": ("2025-03-01", "2025-06-02")}
RES = os.path.join(C.RESULTS, "p3t")
os.makedirs(RES, exist_ok=True)


def game_meta():
    """replay_id -> (tier, map) for snapshot and backfill games (site tiers)."""
    os.environ.setdefault("P1_TIERS", "site")
    from overfit2026 import data as odata
    out = {}
    for g in odata.load_snapshot():
        out[g[0]] = (g[1], g[2])
    for b, gs in odata.load_backfill().items():
        for g in gs:
            out[g[0]] = (g[1], g[2])
    return out


def heldout_ids(vname):
    """Replay ids never in the vintage WP's training split: the paper-1
    test and validation splits (the vintage's own val is a subset of it)."""
    from paper1_revision import core
    return core.test_ids(), core.is_val


def wp_real(vint, d, q, inv, ngames, first, meta_rid):
    """WP_v (team 0 wins) of the real drafts of the games of slot rows q (inv: game index per row)."""
    from shared import HEROES
    sh = TD.to_shared(d["hero_names"])
    t0 = [[] for _ in range(ngames)]
    t1 = [[] for _ in range(ngames)]
    for r, i in zip(q, inv):
        (t0 if d["team"][r] == 0 else t1)[i].append(HEROES[sh[d["hero"][r]]])
    rows = []
    for i, r0 in enumerate(first):
        tier, gm = meta_rid[int(d["replay_id"][r0])]
        rows.append((t0[i], t1[i], gm, tier))
    out = np.empty(len(rows))
    for i in range(0, len(rows), 5000):
        out[i:i + 5000] = vint.wp_drafts(rows[i:i + 5000])
    return out


def slot_skill(d, aug, par, mets, mw, lay, q):
    """As-of s and off-role at the played hero for slot rows q."""
    res = TD.walk(aug, par, mets, mw, d["pid"][q].astype(np.int64), d["day"][q].astype(np.int64),
                  q_hero=d["hero"][q].astype(np.int64), want_all=False)
    hq = d["hero"][q]
    ns = res["n_slot"]
    n_p = ns.sum(1)
    nb = len(C.NPH_EDGES)
    bp = np.searchsorted(C.NP_EDGES, n_p, side="right") - 1
    bh = np.searchsorted(C.NPH_EDGES, ns[np.arange(len(q)), hq], side="right") - 1
    s = aug["table"][bp * nb + bh] + res["mean"]
    s = s + lay.at(d["pid"][q].astype(np.int64), d["day"][q].astype(np.int64), res["n_all"])[np.arange(len(q)), hq]
    off = TD.off_role(d, res["n_all"])[np.arange(len(q)), hq]
    return s, off


def combfit(vintages):
    import p3t_systems as SY
    t0 = time.time()
    par, d, aug, mets, mw = TD.load_model()
    lay = TD.Layer(d, aug, TD.layer_meta())
    meta_rid = game_meta()
    test, is_val = heldout_ids(None)
    out = json.load(open(os.path.join(RES, "combiner.json"))) if os.path.exists(os.path.join(RES, "combiner.json")) else {}
    for v in vintages:
        lo_d, hi_d = (C.day_of(x) for x in FIT_WIN[v])
        sel = (d["day"] >= lo_d) & (d["day"] <= hi_d)
        rid = d["replay_id"]
        cand = np.flatnonzero(sel)
        keep = np.array([(int(r) in test) or is_val(int(r)) or bool(p) for r, p in zip(rid[cand], d["post"][cand])])
        q = cand[keep]
        q = q[np.isin(rid[q], np.fromiter(meta_rid.keys(), np.int64))]
        cnt = np.bincount(d["g"][q], minlength=int(d["g"].max()) + 1)
        q = q[cnt[d["g"][q]] == 10]
        s, off = slot_skill(d, aug, par, mets, mw, lay, q)
        sign = np.where(d["team"][q] == 0, 1.0, -1.0)
        gi, inv = np.unique(d["g"][q], return_inverse=True)
        dS = np.bincount(inv, weights=sign * s)
        dO = np.bincount(inv, weights=sign * off)
        first = np.zeros(len(gi), np.int64)
        first[inv[::-1]] = q[::-1]
        y = np.where(d["team"][first] == 0, d["y"][first], 1 - d["y"][first])
        vint = SY.vintage(v)
        wp = wp_real(vint, d, q, inv, len(gi), first, meta_rid)
        lo = np.log(np.clip(wp, 1e-6, 1 - 1e-6) / np.clip(1 - wp, 1e-6, 1 - 1e-6))
        X = np.column_stack([lo, dS, dO])
        w = C.fit_logistic(X, y)
        w0 = C.fit_logistic(lo[:, None], y)
        import p3_x_common as XC
        ll = XC.logloss(C.predict(w, X), y)
        ll0 = XC.logloss(C.predict(w0, lo[:, None]), y)
        llc = XC.logloss(np.full(len(y), 0.5), y)
        # bootstrap SE of the coefficients (games)
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(100):
            ii = rng.integers(0, len(y), len(y))
            bs.append(C.fit_logistic(X[ii], y[ii]))
        out[v] = {"window": FIT_WIN[v], "games": int(len(y)), "coef": [float(x) for x in w],
                  "coef_se": [float(x) for x in np.std(bs, 0)],
                  "wp_only_coef": [float(x) for x in w0],
                  "ll_gain_full_vs_wp_x1000": float(1000 * (ll0 - ll).mean()),
                  "ll_gain_wp_vs_coin_x1000": float(1000 * (llc - ll0).mean()),
                  "wp_acc": float(((wp > 0.5) == (y == 1)).mean())}
        print(f"combiner {v}: {json.dumps(out[v])} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(out, open(os.path.join(RES, "combiner.json"), "w"), indent=1)
        write_combiner(v, out[v])


def write_combiner(v, fit):
    """Copy a fitted combiner into p3t_vintages.json (b0 set to 0: a side
    constant that never changes a decision) for the vintage and the arms that
    share its player side (cur -> cur_bc)."""
    import p3t_systems as SY
    vs = json.load(open(SY.VFILE))
    b = [0.0] + list(fit["coef"][1:])
    for name, spec in vs.items():
        if name == v or (name.startswith(v + "_") and name.endswith("_bc")):
            spec["combiner"] = b
            spec["combiner_note"] = (f"p3t_value combfit: {fit['games']:,} held-out games {fit['window'][0]}..{fit['window'][1]}; "
                                     f"b0 set to 0 (fitted {fit['coef'][0]:.4f})")
    json.dump(vs, open(SY.VFILE, "w"), indent=1)


def predeval():
    """Prediction quality at staleness on the tournament-window lobbies."""
    import p3t_systems as SY
    import p3_x_common as XC
    from shared import HEROES
    L = TD.load_lobbies()
    g = np.flatnonzero(L["day"] <= TD.LOBBY_END)
    st = {v: np.load(os.path.join(TD.OUT, f"state_{v}.npz")) for v in ("cur", "s4m", "s1y")}
    assert (st["s4m"]["gsel"] == g).all()
    rows = []
    for gi in g:
        t = L["steps"][gi]
        t0 = [HEROES[h] for h, ty, tm, r in t if ty == 1 and tm == 0]
        t1 = [HEROES[h] for h, ty, tm, r in t if ty == 1 and tm == 1]
        rows.append((t0, t1, str(L["map"][gi]), str(L["tier"][gi])))
    y = (L["winner"][g] == 0).astype(float)
    comb = json.load(open(os.path.join(RES, "combiner.json")))
    # played-hero s and off per slot, team sign
    team = L["steps"][g][:, :, 2][L["steps"][g][:, :, 1] == 1].reshape(len(g), 10)
    hero = L["steps"][g][:, :, 0][L["steps"][g][:, :, 1] == 1].reshape(len(g), 10)
    sign = np.where(team == 0, 1.0, -1.0)
    out = {"games": int(len(g))}
    wps = {}
    for v in ("cur", "s4m", "s1y"):
        vint = SY.vintage(v)
        wp = np.concatenate([vint.wp_drafts(rows[i:i + 5000]) for i in range(0, len(rows), 5000)])
        wps[v] = wp
        lo = np.log(np.clip(wp, 1e-6, 1 - 1e-6) / np.clip(1 - wp, 1e-6, 1 - 1e-6))
        b = comb[v]["coef"]
        res = {"wp_ll": float(XC.logloss(wp, y).mean()), "wp_acc": float(((wp > 0.5) == (y == 1)).mean())}
        for sv in ("cur", v):
            # rows of state files: cur covers all lobbies (gsel = all), stale only g
            s_arr = st[sv]["s"].astype(np.float32)
            o_arr = st[sv]["off"]
            base = (g[:, None] * 10 + np.arange(10)[None]) if sv == "cur" else (np.arange(len(g))[:, None] * 10 + np.arange(10)[None])
            sp = s_arr[base.ravel(), hero.ravel()].reshape(len(g), 10)
            op = o_arr[base.ravel(), hero.ravel()].reshape(len(g), 10)
            dS = (sign * sp).sum(1)
            dO = (sign * op).sum(1)
            p = 1 / (1 + np.exp(-(b[0] + b[1] * lo + b[2] * dS + b[3] * dO)))
            res[f"V_ll_state_{sv}"] = float(XC.logloss(p, y).mean())
            res[f"V_acc_state_{sv}"] = float(((p > 0.5) == (y == 1)).mean())
        out[v] = res
    out["coin_ll"] = float(np.log(2))
    json.dump(out, open(os.path.join(RES, "predeval.json"), "w"), indent=1)
    np.savez(os.path.join(TD.OUT, "predeval_wp.npz"), g=g, y=y, **{f"wp_{k}": v for k, v in wps.items()})
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["combfit", "predeval"])
    ap.add_argument("--vintages", default="cur")
    a = ap.parse_args()
    if a.stage == "combfit":
        combfit(a.vintages.split(","))
    else:
        predeval()


if __name__ == "__main__":
    main()
