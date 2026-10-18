"""
Site export for personalization (docs/personalization-site-design.md).

The website runs a TypeScript port of the skill posterior
(src/lib/personal/model.ts). This script writes what the port needs and the
reference outputs the port is tested against. Nothing here is per-player
data except the de-identified fixture (no ids, no battletags, days shifted).

Stages (from training/):
  tables    on the 3090, with the full_1017 env (remote/run_rcf_full.sh env):
            P3_CACHE=cache_full P3_RESULTS=results/rcf_full
            writes $P3_RESULTS/site/site_tables.json: the experience-offset
            table (slot counts, as the full_ms arms), the hero x quarter
            win-rate proxy for rows without a draft, and the layer and tail
            metadata of the full_ms arms. Checks the offset table against the
            arm's own per-slot offsets.
  artifact  local: params (results/rcf_full/params/full_{ms,imp2}.npz),
            site_tables.json and the cover JSONs -> src/lib/personal/model-artifact.json
  fixture   local: reference outputs for de-identified players
            -> src/lib/personal/__tests__/fixtures/reference.json

Live semantics (the port and reference() below implement the same thing):
  rows      every Storm League game the player has: hero, win, day
  counts    n_p, n_ph at a row = the player's games (all rows) on EARLIER days
  r         y - proxy(hero, quarter), v = proxy (1 - proxy)
  r_adj     r - offset(n_p, n_ph)
  state     rows with day < as_of (the request day): S, P, N per hero
  metrics   corpus scoreboard rows (a_shrink card) reduced to the K-space
            evidence (Lam_m, eta_m), exact for a fixed model (see evidence())
  imp2      posterior with play preference z(N); used when there is no
            corpus metric evidence, and for the never-played ranking
  main      full_ms posterior with metric evidence; displayed through the
            volume card (P3_TAILS) plus the volume arm's temporal layer
  band      80% = mean +- 1.2816 sqrt(c var_gauss + k0^2), with c, k0 the
            V1-fit display scale of the model (main: layer arm)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SITE_DIR = os.path.join(ROOT, "src", "lib", "personal")
H = 90
Z80 = 1.2815515655446004


# ------------------------------------------------------------------ remote tables

def tables(_a):
    import p3_c_merged as M
    import p3_hs_core as C
    import p3_x_common as X
    from p3_heroes import HKEY
    d = X.load_ext()
    e_mask = d["in_sample"] & (d["day"] >= C.day_of(C.E_START))
    pid = d["pid"].astype(np.int64)
    np_s = M.lag1_key_sums(d["day"], pid).astype(np.int64)
    nph_s = M.lag1_key_sums(d["day"], pid * HKEY + d["hero"]).astype(np.int64)
    table = C.fit_experience(d["r"], np_s, nph_s, e_mask)
    # proxy table exactly as p3_c_merged.wp_proxy("hero_era")
    Hn = len(d["hero_names"])
    m = d["in_sample"]
    hero, y = d["hero"][m], d["y"][m].astype(np.float64)
    q = M.quarter(d["day"][m])
    sh = np.bincount(hero, weights=y, minlength=Hn)
    ch = np.bincount(hero, minlength=Hn)
    base = (sh + 0.5 * M.PROXY_PRIOR) / (ch + M.PROXY_PRIOR)
    q0, q1 = int(q.min()), int(q.max())
    nq = q1 - q0 + 1
    cell = hero * nq + (q - q0)
    s = np.bincount(cell, weights=y, minlength=Hn * nq).reshape(Hn, nq)
    c = np.bincount(cell, minlength=Hn * nq).reshape(Hn, nq)
    tab = (s + M.PROXY_PRIOR * base[:, None]) / (c + M.PROXY_PRIOR)
    # check: the arm's own offsets
    res = os.environ["P3_RESULTS"]
    z = np.load(os.path.join(res, "rcf", "p3c", "full_ms__merged.npz"))
    mu = table[C.exp_bins(z["s_np_slot"].astype(np.int64), z["s_nh_slot"].astype(np.int64))]
    err = float(np.abs(mu - z["s_mu"]).max())
    print(f"offset table check vs full_ms s_mu: max abs err {err:.2e} over {len(mu):,} slots", flush=True)
    assert err < 1e-6
    meta = {}
    for arm in ("full_ms", "full_ms_layer", "full_ms_tail_vol", "full_imp2"):
        za = np.load(os.path.join(res, "rcf", "p3c", f"{arm}__merged.npz"))
        mt = json.loads(str(za["meta"]))
        meta[arm] = {"layer": mt.get("layer"), "tail": mt.get("tail"), "summary": mt.get("summary"),
                     "coef": za["coef"].tolist()}
    out = {"hero_names": [str(h) for h in d["hero_names"]], "np_edges": C.NP_EDGES.tolist(),
           "nph_edges": C.NPH_EDGES.tolist(), "exp_table": table.tolist(),
           "proxy": {"q0": q0, "table": tab.tolist(), "base": base.tolist(), "clip": [0.2, 0.8]},
           "arms": meta, "offset_check_max_err": err}
    os.makedirs(os.path.join(res, "site"), exist_ok=True)
    p = os.path.join(res, "site", "site_tables.json")
    with open(p, "w") as f:
        json.dump(out, f)
    print(f"wrote {p}", flush=True)


# ------------------------------------------------------------------ model pieces (numpy reference)

def quarter_of_day(day):
    import p3_c_merged as M
    return M.quarter(np.asarray(day, np.int64))


def proxy_of(T, hero, day):
    """p3_c_merged.wp_proxy for rows (hero, day)."""
    tab = np.asarray(T["proxy"]["table"])
    base = np.asarray(T["proxy"]["base"])
    q0 = T["proxy"]["q0"]
    nq = tab.shape[1]
    qe = quarter_of_day(day)
    qi = np.clip(qe - q0, 0, nq - 1)
    out = tab[hero, qi]
    out = np.where(qe < q0, base[hero], out)
    return np.clip(out, 0.2, 0.8)


def exp_offset(T, n_p, n_ph):
    npe = np.asarray(T["np_edges"])
    nphe = np.asarray(T["nph_edges"])
    a = np.searchsorted(npe, n_p, side="right") - 1
    b = np.searchsorted(nphe, n_ph, side="right") - 1
    return np.asarray(T["exp_table"])[a * len(nphe) + b]


def lag1_counts(day, hero):
    """Games overall and on the row's hero on EARLIER days, per row."""
    import p3_c_merged as M
    day = np.asarray(day, np.int64)
    hero = np.asarray(hero, np.int64)
    n_p = M.lag1_key_sums(day, np.zeros(len(day), np.int64))
    n_ph = M.lag1_key_sums(day, hero)
    return n_p.astype(np.int64), n_ph.astype(np.int64)


def evidence(par, X, Cm):
    """Metric evidence in K space for one player: Lam_m (K, K) and eta_m (K,)
    such that the posterior's Lam += Lam_m and eta += eta_m. Exact: metrics
    enter p3_rcf.posterior only through these two sums. X, Cm: (H, M)."""
    Cm = Cm.astype(np.float64)
    X = X.astype(np.float64)
    wm = Cm / (par.rho2[None, :] * Cm + par.eps2[None, :])                    # (H, M)
    xb = np.where(Cm > 0, X / np.maximum(Cm, 1e-9), 0.0) - par.c.T             # (H, M)
    G = np.transpose(par.G, (1, 0, 2))                                          # (H, M, K)
    Lam = np.einsum("hm,hmk,hml->kl", wm, G, G)
    eta = np.einsum("hm,hmk->k", wm * xb, G)
    return Lam, eta


def state_arrays(T, rows, as_of):
    """S, P, N (H,) from rows with day < as_of, plus per-row r, v, r_adj."""
    hero, y, day = rows["hero"], rows["y"].astype(np.float64), rows["day"]
    n_p, n_ph = lag1_counts(day, hero)
    pr = proxy_of(T, hero, day)
    r = y - pr
    v = pr * (1 - pr)
    r_adj = r - exp_offset(T, n_p, n_ph)
    m = day < as_of
    S = np.bincount(hero[m], weights=r_adj[m] / v[m], minlength=H)
    P = np.bincount(hero[m], weights=1.0 / v[m], minlength=H)
    N = np.bincount(hero[m], minlength=H).astype(np.float64)
    return S, P, N, r, v, r_adj


def reference(T, ms, imp, consts, rows, X=None, Cm=None, as_of=None):
    """Everything the site shows for one player, computed with the research
    functions (p3_rcf.posterior, p3_rcf_tails.post). rows: dict of arrays
    hero (model index), y (0/1), day (int days since epoch), sorted by day."""
    import p3_rcf as R
    import p3_rcf_tails as TL
    if as_of is None:
        as_of = int(rows["day"].max()) + 1 if len(rows["day"]) else 0
    S, P, N, r, v, r_adj = state_arrays(T, rows, as_of)
    hall = np.arange(H)
    one = lambda a: np.broadcast_to(a, (H,) + a.shape).copy()
    n_tot = float(N.sum())
    off = exp_offset(T, np.full(H, int(n_tot)), N.astype(np.int64))
    # imp2: every hero (cards when no metric evidence; never-played ranking)
    mi, vi = R.posterior(imp, S[None], P[None], hall[:1], N=N[None], want_all=True)
    mi, vi = mi[0], vi[0]
    out = {"as_of": as_of, "n_games": n_tot, "N": N.tolist(), "offset": off.tolist(),
           "imp2": {"mean": mi.tolist(), "var": vi.tolist()}}
    if X is None:
        return out
    # main (full_ms): Gaussian posterior, then the volume card from the parts
    kw = {"X": one(X), "Cm": one(Cm)}
    mg, vg = R.posterior(ms, one(S), one(P), hall, **kw)
    S0, P0 = one(S), one(P)
    S0[hall, hall] = 0.0
    P0[hall, hall] = 0.0
    m0, v0 = R.posterior(ms, S0, P0, hall, **kw)
    vp = consts["vol_params"]
    ctx = {"bin": TL.vol_bin(N, np.full(H, n_tot))}
    mv, vv = TL.post("vol", vp, m0, np.maximum(v0, 1e-10), S, P, ctx)
    # temporal layer (volume arm): walk the player's games by day; u = r -
    # (offset at the row's counts + volume-card mean at the row's hero, state
    # through the previous day); metric evidence is the current summary
    lay = consts["vol_layer"]
    day, hero = rows["day"], rows["hero"]
    n_p, n_ph = lag1_counts(day, hero)
    u = np.full(len(day), np.nan)
    early = np.zeros(len(day), bool)
    for t in np.unique(day[day < as_of]):
        g = np.flatnonzero(day == t)
        St, Pt, Nt, _, _, _ = state_arrays(T, rows, t)
        hq = hero[g]
        k = len(g)
        kwt = {"X": np.broadcast_to(X, (k,) + X.shape).copy(), "Cm": np.broadcast_to(Cm, (k,) + Cm.shape).copy()}
        Sq, Pq = np.tile(St, (k, 1)), np.tile(Pt, (k, 1))
        ar = np.arange(k)
        Sh, Ph = Sq[ar, hq].copy(), Pq[ar, hq].copy()
        Sq[ar, hq] = 0.0
        Pq[ar, hq] = 0.0
        a0, b0 = R.posterior(ms, Sq, Pq, hq, **kwt)
        mt, _ = TL.post("vol", vp, a0, np.maximum(b0, 1e-10), Sh, Ph,
                        {"bin": TL.vol_bin(Nt[hq], np.full(k, Nt.sum()))})
        u[g] = r[g] - (exp_offset(T, n_p[g], n_ph[g]) + mt)
        early[g] = Nt[hq] < 5
    ok = (day < as_of) & np.isfinite(u)
    lam = np.log(2) / lay["hl_d"]
    w = np.exp(-lam * (as_of - day[ok]))
    rho = float((w * u[ok]).sum() / (w.sum() + lay["kd"]))
    e = early[ok]
    nu = float(np.where(e, u[ok], 0.0).sum() / (e.sum() + lay["ke"]))
    shift = lay["drift_scale"] * rho + lay["learn_scale"] * nu * (N < 5)
    out["main"] = {"gauss_mean": mg.tolist(), "gauss_var": vg.tolist(), "m0": m0.tolist(), "v0": v0.tolist(),
                   "vol_mean": mv.tolist(), "vol_var": vv.tolist(), "rho": rho, "nu": nu,
                   "shift": shift.tolist(), "u": [None if not np.isfinite(x) else float(x) for x in u]}
    return out


# ------------------------------------------------------------------ artifact

def load_inputs():
    import p3_rcf as R
    res = os.path.join(HERE, "results", "rcf_full")
    T = json.load(open(os.path.join(res, "site", "site_tables.json")))
    ms = R.Params.load(os.path.join(res, "params", "full_ms.npz"))
    imp = R.Params.load(os.path.join(res, "params", "full_imp2.npz"))
    cov_ms = json.load(open(os.path.join(res, "cover_full_ms_layer__merged.json")))
    cov_imp = json.load(open(os.path.join(res, "cover_full_imp2__merged.json")))
    vol = T["arms"]["full_ms_tail_vol"]
    consts = {"vol_params": vol["tail"]["params"],
              "vol_layer": {k: vol["layer"][k] for k in ("hl_d", "kd", "ke", "drift_scale", "learn_scale")},
              "main_layer": {k: T["arms"]["full_ms_layer"]["layer"][k]
                             for k in ("hl_d", "kd", "ke", "drift_scale", "learn_scale")},
              "scale_main": {"c": cov_ms["scale_c"], "k0": cov_ms["scale_k0_sd_pp"] / 100},
              "scale_imp2": {"c": cov_imp["scale_c"], "k0": cov_imp["scale_k0_sd_pp"] / 100}}
    return T, ms, imp, consts, cov_ms, cov_imp


def r6(x):
    return np.asarray(x, np.float64).tolist()


def artifact(_a):
    T, ms, imp, consts, cov_ms, cov_imp = load_inputs()
    assert ms.W is None and ms.G is not None and imp.W is not None and imp.G is None
    out = {
        "version": "full_1017-2026-10-18",
        "source": "training/personalization/p3_site_export.py artifact; models full_ms (main) and full_imp2 "
                  "on the full_1017 panel (results/P3_TAILS.md)",
        "heroNames": T["hero_names"],
        "npEdges": T["np_edges"], "nphEdges": T["nph_edges"], "expTable": r6(T["exp_table"]),
        "proxy": {"q0": T["proxy"]["q0"], "table": r6(T["proxy"]["table"]), "base": r6(T["proxy"]["base"]),
                  "clip": T["proxy"]["clip"]},
        "main": {"U": r6(ms.U), "s2": r6(ms.s2), "scale": consts["scale_main"],
                 "layer": consts["vol_layer"], "gaussLayer": consts["main_layer"],
                 "volParams": r6(consts["vol_params"]), "volEdges": [0, 1, 10, 30, 100], "volSpecialist": 0.2,
                 "combiner": T["arms"]["full_ms_layer"]["coef"]},
        "imp2": {"U": r6(imp.U), "s2": r6(imp.s2), "W": r6(imp.W), "zmean": r6(imp.zmean), "zsd": r6(imp.zsd),
                 "scale": consts["scale_imp2"]},
        "grid": {"lo": -10, "hi": 10, "n": 401},
        "validation": {
            # results/rcf_full/p3_c_tests_main.txt, "all" rows: share of held-out games whose winner the
            # draft WP plus all ten players' estimates called correctly (A2S = the old win-only model)
            "accuracy": {"v2": {"a2s": 0.6035, "imp2": 0.6079, "main": 0.6200},
                         "oot": {"a2s": 0.6014, "imp2": 0.6044, "main": 0.6166}},
            "cover80": {"main": round(cov_ms["cover80_scaled"], 3), "imp2": round(cov_imp["cover80_scaled"], 3)},
        },
    }
    os.makedirs(SITE_DIR, exist_ok=True)
    p = os.path.join(SITE_DIR, "model-artifact.json")
    with open(p, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {p} ({os.path.getsize(p) / 1024:.0f} KB)")


# ------------------------------------------------------------------ fixture

def _shards():
    import glob
    import gzip
    for f in sorted(glob.glob(os.path.join(HERE, "cache", "mmr_history", "shards", "*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    break


def fixture(a):
    """De-identified reference cases: a few real Heroes Profile histories
    (no ids, no battletags, days shifted by a random per-player offset that
    keeps the quarter grid; only the last 600 games) with their corpus
    metric evidence, plus synthetic edge cases. Expected outputs come from
    reference()."""
    import datetime
    T, ms, imp, consts, _, _ = load_inputs()
    names = T["hero_names"]
    hidx = {n: i for i, n in enumerate(names)}
    rng = np.random.default_rng(a.seed)
    # corpus scoreboard (a_shrink card) by blizz_id
    sb = np.load(os.path.join(HERE, "cache", "rcf", "sb_a_shrink.npz"))
    sb_bl = sb["blizz_id"]
    want = {"thin": (30, 80), "mid": (300, 700), "heavy": (1500, 4000)}
    picked = {}
    cand = 0
    for rec in _shards():
        if len(picked) == len(want):
            break
        n = len(rec["rows"])
        for lab, (lo, hi) in want.items():
            if lab in picked or not (lo <= n <= hi):
                continue
            cand += 1
            if rng.random() > 0.02:
                continue
            bl = rec["blizz_id"]
            if (sb_bl == bl).sum() < 10:
                continue
            picked[lab] = rec
    cases = []
    for lab, rec in picked.items():
        cols = rec["cols"]
        ci = {c: i for i, c in enumerate(cols)}
        rows = [(hidx.get(r[ci["hero"]]), int(str(r[ci["winner"]]).lower() in ("1", "true")),
                 r[ci["game_date"]], int(r[ci["replayID"]])) for r in rec["rows"]]
        rows = [x for x in rows if x[0] is not None and x[2]]
        day = np.array([(datetime.date.fromisoformat(x[2][:10]) - datetime.date(1970, 1, 1)).days for x in rows])
        o = np.argsort(day, kind="stable")
        hero = np.array([rows[i][0] for i in o], np.int64)
        y = np.array([rows[i][1] for i in o], np.int64)
        rid = np.array([rows[i][3] for i in o], np.int64)
        day = day[o]
        keep = slice(max(0, len(day) - 600), None)
        hero, y, day, rid = hero[keep], y[keep], day[keep], rid[keep]
        # corpus metric sums for this player (all corpus rows: the live summary)
        msk = sb_bl == rec["blizz_id"]
        Zp = sb["Z"][msk].astype(np.float64)
        Wp = sb["W"][msk].astype(np.float64)
        rid_sb = sb["replay_id"][msk]
        # hero of each corpus row from the player's HP history (same replay ids)
        h_of = {int(r): int(h) for r, h in zip(rid, hero)}
        hh = np.array([h_of.get(int(r), -1) for r in rid_sb])
        g = hh >= 0
        X = np.zeros((H, Zp.shape[1]))
        Cm = np.zeros((H, Zp.shape[1]))
        for k in range(Zp.shape[1]):
            x = Zp[g, k]
            ok = ~np.isnan(x)
            X[:, k] = np.bincount(hh[g][ok], weights=(Wp[g] * x)[ok], minlength=H)
            Cm[:, k] = np.bincount(hh[g][ok], weights=Wp[g][ok], minlength=H)
        # de-identify: shift every day by one random offset (the expected
        # outputs are computed on the shifted rows)
        day = day + int(rng.integers(-60, 61))
        as_of = int(day.max()) + 1
        ref = reference(T, ms, imp, consts, {"hero": hero, "y": y, "day": day}, X, Cm, as_of)
        Lm, em = evidence(ms, X, Cm)
        cases.append({"label": f"real_{lab}", "rows": {"hero": hero.tolist(), "y": y.tolist(), "day": day.tolist()},
                      "metric": {"Lam": Lm.tolist(), "eta": em.tolist(), "n_rows": int(g.sum())},
                      "as_of": as_of, "expected": ref})
        print(f"case real_{lab}: {len(hero)} games, {int(g.sum())} corpus scoreboard rows", flush=True)
    # synthetic edge cases
    d0 = int((np.datetime64("2026-08-01") - np.datetime64("1970-01-01")).astype(int))
    syn = {
        "empty": {"hero": [], "y": [], "day": []},
        "one_game": {"hero": [hidx["Ana"]], "y": [1], "day": [d0]},
        "one_trick": {"hero": [hidx["Murky"]] * 220, "y": (rng.random(220) < 0.6).astype(int).tolist(),
                      "day": sorted((d0 - rng.integers(0, 400, 220)).tolist())},
        "same_day_burst": {"hero": rng.integers(0, H, 40).tolist(), "y": rng.integers(0, 2, 40).tolist(),
                           "day": [d0] * 20 + [d0 + 1] * 20},
        "old_games": {"hero": rng.integers(0, H, 60).tolist(), "y": rng.integers(0, 2, 60).tolist(),
                      "day": sorted((d0 - 3000 - rng.integers(0, 300, 60)).tolist())},
    }
    for lab, rr in syn.items():
        rows = {k: np.asarray(v, np.int64) for k, v in rr.items()}
        as_of = int(rows["day"].max()) + 1 if len(rows["day"]) else d0
        withm = lab in ("one_trick", "same_day_burst")
        X = Cm = None
        if withm:
            X = rng.normal(0, 0.4, (H, 15)) * 5
            Cm = np.zeros((H, 15))
            for h in np.unique(rows["hero"]):
                Cm[h] = (rows["hero"] == h).sum() * 0.5
            X = X * (Cm > 0)
        ref = reference(T, ms, imp, consts, rows, X, Cm, as_of)
        case = {"label": f"syn_{lab}", "rows": {k: v.tolist() for k, v in rows.items()}, "as_of": as_of,
                "expected": ref}
        if withm:
            Lm, em = evidence(ms, X, Cm)
            case["metric"] = {"Lam": Lm.tolist(), "eta": em.tolist(), "n_rows": int(Cm.sum())}
        cases.append(case)
    p = os.path.join(SITE_DIR, "__tests__", "fixtures", "reference.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump({"note": "De-identified reference cases for src/lib/personal/model.ts (no ids, no battletags; "
                           "real cases keep only the last 600 games with days shifted). Generated by "
                           "training/personalization/p3_site_export.py fixture.",
                   "cases": cases}, f, separators=(",", ":"))
    print(f"wrote {p} ({os.path.getsize(p) / 1024:.0f} KB), {len(cases)} cases")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tables", "artifact", "fixture"])
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    {"tables": tables, "artifact": artifact, "fixture": fixture}[a.stage](a)


if __name__ == "__main__":
    main()
