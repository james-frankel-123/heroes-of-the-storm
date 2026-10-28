"""
P3 full-history rerun tests (plan step 6, P3_FULL_HISTORY_PLAN.md): arms
A0 / A1 / A2 / A2-seen scored on the same games, and the tests Max asked for
(R1 calibration by observed record, shrinkage, R3 subset lifts).

predict  one arm on its cache ($P3_CACHE: the window cache for A0, cache_full
         for A1): the extended slot table (x_slots_ext.npz), the arm's
         "+CF rank 2" kernel, the experience table refit on E with the arm's
         counts, the causal online state (lag 1 day), the logistic combiner
         (logit WP + team difference of the slot estimates) fit on V1, scored
         on V2 and OOT, as p3_x_oot.py does. With --history merged the extra
         rows of p3_c_merged.py (HP rows not in the slot table) enter the
         counts and the state; --rule seen keeps only rows uploaded by then.
         --refit also refits the kernel log-variances on the merged E state.
         Writes $P3_RESULTS/p3c/<arm>.npz (slot and game arrays; no
         battletags, player keys only, gitignored results cache).
report   pairs the arms on replay_id and writes
         $P3_RESULTS/p3_c_tests_<name>.json / .txt.

Usage (from training/):
  P3_CACHE=.../cache python3 personalization/p3_c_tests.py predict --arm A0 --rows .../cache/merged_rows_oct05.npz
  P3_CACHE=.../cache python3 personalization/p3_c_tests.py predict --arm A2w --history merged --rows ...
  python3 personalization/p3_c_tests.py report --arms A0,A1,A2,A2seen --ref A2 --name early
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C

BEST = "+CF rank 2"
NH_BINS = [(1, 9), (10, 19), (20, 49), (50, 99), (100, 10 ** 9)]
NH_LABELS = ["1-9", "10-19", "20-49", "50-99", "100+"]
REC_EDGES = [-1e9, -0.05, 0.0, 0.05, 0.10, 1e9]
REC_LABELS = ["< -5pp", "-5 to 0", "0 to +5", "+5 to +10", "> +10"]
WIN = {1: "V1", 2: "V2", 3: "OOT", 4: "PF"}
PF_START = "2026-09-28"   # post-freeze games (2.57): window 4 when P3_PF_WINDOW=1


def out_dir():
    d = os.path.join(C.RESULTS, "p3c")
    os.makedirs(d, exist_ok=True)
    return d


# ------------------------------------------------------------------ predict

def windows(d):
    import p3_x_common as X
    day, post = d["day"], d["post"]
    snap_post = (~post) & (~d["in_sample"])
    _, fr = np.unique(d["g"][snap_post], return_index=True)
    med = np.median(day[snap_post][fr])
    w = np.zeros(len(day), np.int8)
    w[snap_post & (day < med)] = 1
    w[snap_post & (day >= med)] = 2
    w[post & (day > X.SNAP_LAST_DAY)] = 3
    if os.environ.get("P3_PF_WINDOW") == "1":
        # post-freeze confirmation build only: games from 2026-09-28 form window 4
        w[post & (day >= C.day_of(PF_START))] = 4
    return w, float(med)


def refit_kernel(d, ex, res_table, proxy, Ks_bases, theta0, V, exp_counts="merged"):
    """Refit the log-variances of "+CF rank 2" (the five structural bases
    and the rank-2 factor term V V^T as one basis) on the merged E state."""
    import p3_c_merged as M
    from p3_hs_fit import STRUCT
    H = len(d["hero_names"])
    n = len(d["pid"])
    from p3_heroes import HKEY
    wc = None if exp_counts == "merged" else np.r_[np.ones(n), np.zeros(len(ex["pid"]))]
    np_s = M.lag1_key_sums(np.r_[d["day"], ex["end_day"]], np.r_[d["pid"], ex["pid"]].astype(np.int64), wc)
    kph = np.r_[d["pid"], ex["pid"]].astype(np.int64) * HKEY + np.r_[d["hero"], ex["hero"]]
    nph_s = M.lag1_key_sums(np.r_[d["day"], ex["end_day"]], kph, wc)
    tb = res_table
    r_adj_s = d["r"] - tb[C.exp_bins(np_s[:n].astype(np.int64), nph_s[:n].astype(np.int64))]
    pr = M.wp_proxy(d, ex, proxy)
    r_adj_e = (ex["y"] - pr) - tb[C.exp_bins(np_s[n:].astype(np.int64), nph_s[n:].astype(np.int64))]
    e_last = int(d["day"][d["in_sample"]].max())
    S, P = M.ex_static_state(d, ex, e_last, d["n_players"], H, r_adj_s, r_adj_e, pr * (1 - pr))
    rng = np.random.RandomState(7)
    half = rng.rand(int(d["n_players"])) < 0.5
    fit_p = half & (P.sum(1) > 0)
    # the factor term as a unit-diagonal basis with its variance as a free
    # weight (fit_kernel bounds variances to [1e-9, 0.1])
    F = V @ V.T
    s = float(np.mean(np.diag(F)))
    B = [Ks_bases[b] for b in STRUCT] + [F / s]
    th0 = np.r_[theta0[:len(STRUCT)], np.log(s)]
    th, ll = C.fit_kernel(np.ascontiguousarray(S[fit_p]), np.ascontiguousarray(P[fit_p]), B, th0)
    return C.kernel_from(B, th), th


def predict(a):
    import p3_c_merged as M
    import p3_x_common as X
    t0 = time.time()
    d = X.load_ext()
    Ks, _ = X.kernels()
    w, med = windows(d)
    q = w > 0
    ex, pulled_pid = M.load_rows(a.rows, d) if a.rows else (None, np.zeros(0, np.int64))
    use = ex if a.history == "merged" else None
    if use is not None and a.since:
        keep = use["end_day"] >= C.day_of(a.since)
        use = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v) for k, v in use.items()}
    print(f"[{a.arm}] slots {len(d['pid']):,}, queries {q.sum():,} (V1 {(w == 1).sum():,}, V2 {(w == 2).sum():,}, "
          f"OOT {(w == 3).sum():,}), split day {med}; extra rows {0 if use is None else len(use['pid']):,} "
          f"rule {a.rule} proxy {a.proxy}", flush=True)
    K = Ks[BEST]
    res = M.merged_predict(d, use, K, q, rule=a.rule, proxy=a.proxy, extra_K=[Ks["player+hero"]],
                           ex_weight=a.ex_weight, exp_counts=a.exp_counts,
                           half_life=a.half_life)
    theta = None
    if a.refit and use is not None:
        from p3_hs_fit import KOUT
        kz = np.load(KOUT)
        bases = {k[6:]: kz[k] for k in kz.files if k.startswith("basis_")}
        K2, theta = refit_kernel(d, use, res["table"], a.proxy, bases, kz["theta_" + BEST], kz["V_cf2"],
                                 a.exp_counts)
        print(f"[{a.arm}] refit log-variances {np.round(theta, 3).tolist()} "
              f"(was {np.round(kz['theta_' + BEST], 3).tolist()})", flush=True)
        res = M.merged_predict(d, use, K2, q, rule=a.rule, proxy=a.proxy, extra_K=[Ks["player+hero"]],
                               ex_weight=a.ex_weight, exp_counts=a.exp_counts,
                           half_life=a.half_life)
        K = K2
    print(f"[{a.arm}] state done {time.time() - t0:.0f}s", flush=True)
    qi = res["qidx"]
    # game level
    n_games, y, wp0, cnt, gday = X.game_arrays(d)
    lo = np.log(wp0 / (1 - wp0))
    pred_rows = np.zeros(len(d["pid"]))
    pred_rows[qi] = res["pred"]
    D = X.team_diff(pred_rows, d["g"], d["team"], n_games)
    gw = np.zeros(n_games, np.int8)
    gw[d["g"][qi]] = w[qi]
    full = cnt == 10
    fit = (gw == 1) & full
    wc = C.fit_logistic(np.column_stack([lo, D])[fit], y[fit])
    w0 = C.fit_logistic(lo[fit][:, None], y[fit])
    p = C.predict(wc, np.column_stack([lo, D]))
    p0 = C.predict(w0, lo[:, None])
    gsel = (gw > 0) & full
    grid = np.zeros(n_games, np.int64)
    grid[d["g"]] = d["replay_id"]
    pulled = np.isin(d["pid"], pulled_pid)
    n_pull = np.bincount(d["g"], weights=pulled, minlength=n_games)
    out = {
        # slots (query rows)
        "s_replay_id": d["replay_id"][qi], "s_key": d["player_keys"][d["pid"][qi]], "s_team": d["team"][qi],
        "s_hero": d["hero"][qi].astype(np.int16), "s_day": d["day"][qi], "s_win": w[qi],
        "s_r": d["r"][qi].astype(np.float32), "s_wp": d["wp"][qi].astype(np.float32),
        "s_pred": res["pred"].astype(np.float32), "s_mean": res["mean"].astype(np.float32),
        "s_mean_ph": res["extra_means"][0].astype(np.float32),
        "s_var": res["var"].astype(np.float32), "s_kdiag": res["kdiag"].astype(np.float32),
        "s_mu": res["mu"].astype(np.float32), "s_nh": res["nh"], "s_np": res["np"],
        "s_nh_slot": res["nh_slot"], "s_np_slot": res["np_slot"], "s_rec": res["rec"],
        "s_rec_slot": res["rec_slot"], "s_pulled": pulled[qi],
        # games
        "g_replay_id": grid[gsel], "g_win": gw[gsel], "g_y": y[gsel], "g_lo": lo[gsel], "g_D": D[gsel],
        "g_p": p[gsel], "g_p0": p0[gsel], "g_day": gday[gsel], "g_all10_pulled": (n_pull[gsel] == 10),
        "coef": wc, "coef0": w0, "table": res["table"], "kernel_hero_sd_pp": 100 * np.sqrt(np.diag(K)),
    }
    if theta is not None:
        out["theta_refit"] = theta
    ll = X.logloss(p, y)
    ll0 = X.logloss(p0, y)
    summ = {}
    for k, nm in WIN.items():
        m = (gw == k) & full
        if not m.any():
            continue
        summ[nm] = {"games": int(m.sum()), "gain": float((ll0[m] - ll[m]).mean()),
                    "acc": float(((p[m] > 0.5) == (y[m] == 1)).mean()), "ll": float(ll[m].mean())}
    print(f"[{a.arm}] {json.dumps(summ)}", flush=True)
    path = os.path.join(out_dir(), f"{a.arm}.npz")
    np.savez(path, **out, meta=np.array(json.dumps({
        "arm": a.arm, "cache": C.CACHE, "history": a.history, "rule": a.rule, "proxy": a.proxy, "rows": a.rows,
        "refit": bool(a.refit), "exp_counts": a.exp_counts, "half_life": a.half_life, "ex_weight": a.ex_weight, "since": a.since, "split_day": med,
        "n_extra": int(res["n_extra"]), "summary": summ, "seconds": round(time.time() - t0)})))
    print(f"[{a.arm}] wrote {path} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ report

def load_arm(name):
    z = np.load(os.path.join(out_dir(), f"{name}.npz"))
    a = {k: z[k] for k in z.files}
    a["meta"] = json.loads(str(a["meta"]))
    return a


def ll_of(p, y):
    eps = 1e-7
    return -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


SEALED_BUILD = "2.55.17.98025"   # formerly sealed build: inside OOT (reported separately, C13)


def game_version(rids):
    """Build string per replay id (slot table of the current cache)."""
    z = np.load(os.path.join(C.CACHE, "x_slots_ext.npz"), mmap_mode="r")
    r, v = np.asarray(z["replay_id"]), np.asarray(z["version"])
    u, i = np.unique(r, return_index=True)
    names = np.array([str(x) for x in z["versions"]])
    j = np.searchsorted(u, rids)
    j = np.minimum(j, len(u) - 1)
    assert np.array_equal(u[j], rids), "game_version: replay ids missing from the slot table"
    return names[v[i][j]]


def real_tier(rids):
    """Real league tier per game (1 Bronze .. 5 Diamond, 6 Master, -2 unknown):
    x_tiers.npz for snapshot games, x_post_games.json.gz for post games."""
    import gzip
    out = np.full(len(rids), -2)
    p = os.path.join(C.CACHE, "x_tiers.npz")
    if os.path.exists(p):
        z = np.load(p)
        o = np.argsort(z["replay_ids"])
        s = z["replay_ids"][o]
        j = np.minimum(np.searchsorted(s, rids), len(s) - 1)
        ok = s[j] == rids
        lt = z["league_tier"][o][j].astype(int)
        out = np.where(ok, np.where(lt < 0, 6, lt - 1), out)
    p = os.path.join(C.CACHE, "x_post_games.json.gz")
    if os.path.exists(p):
        games = json.load(gzip.open(p, "rt"))
        lmap = {g["replay_id"]: (6 if g["league_tier"] is None else g["league_tier"] - 1) for g in games}
        for i in np.flatnonzero(out == -2):
            out[i] = lmap.get(int(rids[i]), -2)
    return out


def week_boot(diff, week, n=400, seed=0):
    """Week-block bootstrap 95% CI of the mean of diff."""
    rng = np.random.default_rng(seed)
    uw, inv = np.unique(week, return_inverse=True)
    s = np.bincount(inv, weights=diff)
    c = np.bincount(inv)
    bs = []
    for _ in range(n):
        k = rng.integers(0, len(uw), len(uw))
        bs.append(s[k].sum() / max(c[k].sum(), 1))
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def paired(a, b):
    """Mean of a - b and its SE (games independent)."""
    dlt = a - b
    return float(dlt.mean()), float(dlt.std(ddof=1) / np.sqrt(max(len(dlt), 2)))


def align_games(arms):
    """Common games (replay_id) across arms, in one order."""
    common = None
    for a in arms.values():
        r = a["g_replay_id"]
        common = r if common is None else np.intersect1d(common, r)
    out = {}
    for nm, a in arms.items():
        o = np.argsort(a["g_replay_id"])
        j = np.searchsorted(a["g_replay_id"][o], common)
        out[nm] = o[j]
    return common, out


def slot_index(a, rids, keys):
    """Row of each (replay_id, key) in arm a's slot arrays (-1 if absent)."""
    comp = a["s_replay_id"].astype(np.int64) * (1 << 24) + (a["s_key"] % (1 << 24))
    o = np.argsort(comp)
    q = rids.astype(np.int64) * (1 << 24) + (keys % (1 << 24))
    j = np.minimum(np.searchsorted(comp[o], q), len(o) - 1)
    hit = (comp[o][j] == q) & (a["s_key"][o][j] == keys)
    return np.where(hit, o[j], -1)


def calib_table(r, pred, nh, rec, week, label):
    """R1: per (games on hero, record to date) bin, realized vs predicted mean
    residual on the scored slot, week-block bootstrap CI of the gap."""
    rows = []
    recm = np.where(nh > 0, rec / np.maximum(nh, 1), np.nan)
    rb = np.searchsorted(REC_EDGES, recm, side="right") - 1
    for (lo_, hi_), nl in zip(NH_BINS, NH_LABELS):
        for k, rl in enumerate(REC_LABELS):
            m = (nh >= lo_) & (nh <= hi_) & (rb == k)
            if m.sum() < 200:
                continue
            gap = r[m] - pred[m]
            rows.append({"arm": label, "games_on_hero": nl, "record": rl, "slots": int(m.sum()),
                         "record_mean_pp": float(100 * np.nanmean(recm[m])),
                         "realized_pp": float(100 * r[m].mean()), "predicted_pp": float(100 * pred[m].mean()),
                         "gap_pp": float(100 * gap.mean()),
                         "gap_ci_pp": [100 * x for x in week_boot(gap, week[m], n=200)]})
    return rows


def report(a):
    names = a.arms.split(",")
    arms = {n: load_arm(n) for n in names}
    ref = a.ref or names[-1]
    common, gidx = align_games(arms)
    A = {n: {k[2:]: v[gidx[n]] for k, v in arm.items() if k.startswith("g_")} for n, arm in arms.items()}
    y = A[names[0]]["y"]
    for n in names:
        assert np.array_equal(A[n]["y"], y), f"outcomes differ for {n}"
    win = A[names[0]]["win"]
    week = A[names[0]]["day"] // 7
    LL = {n: ll_of(A[n]["p"], y) for n in names}
    ACC = {n: ((A[n]["p"] > 0.5) == (y == 1)).astype(float) for n in names}
    LL0 = ll_of(A[names[0]]["p0"], y)
    ACC0 = ((A[names[0]]["p0"] > 0.5) == (y == 1)).astype(float)
    pulled10 = A[ref]["all10_pulled"]
    tier = real_tier(common)
    out = {"arms": {n: arms[n]["meta"] for n in names}, "ref": ref, "games_common": int(len(common))}
    lines = [f"# P3 full-history tests ({a.name}); arms {names}; reference {ref}", ""]

    # per-game slot features from the reference arm (proven, deep)
    R = arms[ref]
    gr = R["s_replay_id"]
    gpos = np.searchsorted(common, gr)
    ok = (gpos < len(common)) & (common[np.minimum(gpos, len(common) - 1)] == gr)
    proven_slot = (R["s_nh"] >= 50) & (R["s_mean"] > 0.05)
    n_proven = np.bincount(gpos[ok], weights=proven_slot[ok], minlength=len(common))
    min_np = np.full(len(common), np.inf)
    np.minimum.at(min_np, gpos[ok], R["s_np"][ok])
    sets = {}
    for wk, wn in ((2, "V2"), (3, "OOT")):
        base = win == wk
        sets[f"{wn} all"] = base
        sets[f"{wn} all-10 pulled"] = base & pulled10
        sets[f"{wn} 1+ proven"] = base & (n_proven >= 1)
        sets[f"{wn} 3+ proven"] = base & (n_proven >= 3)
        sets[f"{wn} deep (all 10 with 500+ games)"] = base & (min_np >= 500)
        sets[f"{wn} Diamond"] = base & (tier == 5)
        sets[f"{wn} Master"] = base & (tier == 6)
        sets[f"{wn} Diamond+Master, all-10 pulled"] = base & (tier >= 5) & pulled10
    # OOT by period and with / without the formerly sealed build (C13)
    ver = game_version(common)
    sealed = ver == SEALED_BUILD
    split = C.day_of("2026-09-12")
    gday = A[names[0]]["day"]
    oot = win == 3
    sets["OOT before 2026-09-12"] = oot & (gday < split)
    sets["OOT from 2026-09-12"] = oot & (gday >= split)
    sets["OOT from 2026-09-12, without " + SEALED_BUILD] = oot & (gday >= split) & ~sealed
    sets["OOT without " + SEALED_BUILD] = oot & ~sealed
    sets["OOT " + SEALED_BUILD + " only"] = oot & sealed
    out["oot_sealed_build_games"] = int((oot & sealed).sum())
    out["sets"] = {}
    lines.append("## Game-level lift (log-loss gain over M0 population WP, x1000; acc in %)")
    lines.append("set | games | " + " | ".join(f"{n} gain (SE)" for n in names) + " | "
                 + " | ".join(f"{n} acc" for n in names) + " | " + " | ".join(
                     f"{names[i]}-{names[j]} LL (SE) [week CI]" for i, j in pairs_of(names)))
    for sn, m in sets.items():
        idx = np.flatnonzero(m)
        if len(idx) < 300:
            continue
        rec = {"games": int(len(idx)), "arms": {}, "paired": {}}
        for n in names:
            g, se = paired(LL0[idx], LL[n][idx])
            ag, ase = paired(ACC[n][idx], ACC0[idx])
            rec["arms"][n] = {"gain": g, "gain_se": se, "ll": float(LL[n][idx].mean()),
                              "acc": float(ACC[n][idx].mean()), "acc_gain": ag, "acc_gain_se": ase}
        rec["M0"] = {"ll": float(LL0[idx].mean()), "acc": float(ACC0[idx].mean())}
        for i, j in pairs_of(names):
            hi_, lo_ = names[i], names[j]
            g, se = paired(LL[lo_][idx], LL[hi_][idx])
            ag, ase = paired(ACC[hi_][idx], ACC[lo_][idx])
            rec["paired"][f"{hi_}-{lo_}"] = {"ll_gain": g, "se": se,
                                            "week_ci": week_boot(LL[lo_][idx] - LL[hi_][idx], week[idx]),
                                            "acc_gain": ag, "acc_se": ase}
        out["sets"][sn] = rec
        lines.append(f"{sn} | {len(idx):,} | " + " | ".join(
            f"{1000 * rec['arms'][n]['gain']:+.2f} ({1000 * rec['arms'][n]['gain_se']:.2f})" for n in names) + " | "
            + " | ".join(f"{100 * rec['arms'][n]['acc']:.2f}" for n in names) + " | " + " | ".join(
            "{:+.2f} ({:.2f}) [{:+.2f},{:+.2f}]".format(1000 * v["ll_gain"], 1000 * v["se"], 1000 * v["week_ci"][0],
                                                       1000 * v["week_ci"][1]) for v in rec["paired"].values()))
    lines.append("")

    # R1 calibration by observed record (V2 and OOT slots)
    out["R1"] = {}
    lines.append("## R1 calibration by observed record (slot residual r = y - WP; pp)")
    for n in names:
        s = arms[n]
        for wk, wn in ((2, "V2"), (3, "OOT"), (0, "V2+OOT")):
            m = (s["s_win"] == wk) if wk else (s["s_win"] >= 2)
            wkk = s["s_day"][m] // 7
            rows = calib_table(s["s_r"][m], s["s_pred"][m], s["s_nh"][m], s["s_rec"][m], wkk, n)
            out["R1"][f"{n} {wn} own history"] = rows
    # each arm's predictions on the reference arm's bins (full history)
    for n in names:
        s = arms[n]
        j = slot_index(s, R["s_replay_id"], R["s_key"])
        okj = j >= 0
        for wk, wn in ((2, "V2"), (3, "OOT"), (0, "V2+OOT")):
            m = okj & ((R["s_win"] == wk) if wk else (R["s_win"] >= 2))
            rows = calib_table(R["s_r"][m], s["s_pred"][j[m]], R["s_nh"][m], R["s_rec"][m], R["s_day"][m] // 7, n)
            out["R1"][f"{n} {wn} on {ref} bins"] = rows
    for key in [k for k in out["R1"] if k.endswith(f"V2+OOT on {ref} bins")]:
        lines.append(f"### {key}")
        lines.append("games on hero | record | slots | record pp | realized pp | predicted pp | gap pp [CI]")
        for r_ in out["R1"][key]:
            lines.append(f"{r_['games_on_hero']} | {r_['record']} | {r_['slots']:,} | {r_['record_mean_pp']:+.1f} | "
                         f"{r_['realized_pp']:+.2f} | {r_['predicted_pp']:+.2f} | {r_['gap_pp']:+.2f} "
                         f"[{r_['gap_ci_pp'][0]:+.2f},{r_['gap_ci_pp'][1]:+.2f}]")
        lines.append("")

    # shrinkage on heavy-history cells (reference arm's counts)
    out["shrinkage"] = {}
    lines.append("## Shrinkage (slots of V2+OOT; prior-weight share = posterior var / prior var at the hero)")
    lines.append("cells (by ref games on hero) | arm | slots | median prior share | median |GP mean| pp | "
                 "slope of GP mean on record | median games on hero seen")
    for nl, (lo_, hi_) in (("20-49", (20, 49)), ("50-99", (50, 99)), ("100+", (100, 10 ** 9)),
                           ("100+, record > +5pp", (100, 10 ** 9))):
        sel = (R["s_win"] >= 2) & (R["s_nh"] >= lo_) & (R["s_nh"] <= hi_)
        if "record" in nl:
            sel &= R["s_rec"] / np.maximum(R["s_nh"], 1) > 0.05
        for n in names:
            s = arms[n]
            j = slot_index(s, R["s_replay_id"][sel], R["s_key"][sel])
            okj = j >= 0
            jj = j[okj]
            share = s["s_var"][jj] / s["s_kdiag"][jj]
            recm = R["s_rec"][sel][okj] / np.maximum(R["s_nh"][sel][okj], 1)
            mm = s["s_mean"][jj]
            slope = float(np.cov(mm, recm)[0, 1] / max(recm.var(), 1e-12))
            rec_ = {"slots": int(okj.sum()), "median_prior_share": float(np.median(share)),
                    "median_abs_mean_pp": float(100 * np.median(np.abs(mm))), "slope_mean_on_record": slope,
                    "median_nh_seen": float(np.median(s["s_nh"][jj]))}
            out["shrinkage"][f"{nl} | {n}"] = rec_
            lines.append(f"{nl} | {n} | {rec_['slots']:,} | {rec_['median_prior_share']:.3f} | "
                         f"{rec_['median_abs_mean_pp']:.2f} | {slope:.3f} | {rec_['median_nh_seen']:.0f}")
    lines.append("")
    out["kernel_hero_sd_pp"] = {n: float(np.median(arms[n].get("kernel_hero_sd_pp", np.nan))) for n in names}
    lines.append("prior sd at a hero (median over heroes, pp): " + json.dumps(
        {k: round(v, 2) for k, v in out["kernel_hero_sd_pp"].items()}))
    # coverage of history: what the reference arm adds
    m = R["s_win"] >= 2
    out["history_depth"] = {"V2+OOT slots": int(m.sum()),
                            "median games before, slot rows only": float(np.median(R["s_np_slot"][m])),
                            "median games before, with extras": float(np.median(R["s_np"][m])),
                            "median games on hero, slot rows only": float(np.median(R["s_nh_slot"][m])),
                            "median games on hero, with extras": float(np.median(R["s_nh"][m])),
                            "share 50+ on hero, slot rows only": float((R["s_nh_slot"][m] >= 50).mean()),
                            "share 50+ on hero, with extras": float((R["s_nh"][m] >= 50).mean()),
                            "pulled slot share": float(R["s_pulled"][m].mean())}
    lines.append("history depth (reference arm): " + json.dumps(out["history_depth"]))
    path = os.path.join(C.RESULTS, f"p3_c_tests_{a.name}")
    with open(path + ".json", "w") as f:
        json.dump(out, f, indent=1)
    with open(path + ".txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def pairs_of(names):
    """Comparisons: each later arm against each earlier one."""
    return [(j, i) for j in range(len(names)) for i in range(j)]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--arm", required=True)
    p.add_argument("--history", choices=["slots", "merged"], default="slots")
    p.add_argument("--rows", help="merged_rows_<store>.npz for this cache (pulled flags; extras if merged)")
    p.add_argument("--rule", choices=["game", "seen"], default="game")
    p.add_argument("--proxy", choices=["hero_era", "half"], default="hero_era")
    p.add_argument("--refit", action="store_true")
    p.add_argument("--ex-weight", type=float, default=1.0)
    p.add_argument("--exp-counts", choices=["merged", "slots"], default="merged",
                   help="counts for the experience offset (slots: HP rows feed the skill state only)")
    p.add_argument("--half-life", type=float, help="recency half-life (days) on the skill state")
    p.add_argument("--since", help="only extra rows from this date (YYYY-MM-DD)")
    r = sub.add_parser("report")
    r.add_argument("--arms", required=True)
    r.add_argument("--ref")
    r.add_argument("--name", required=True)
    a = ap.parse_args()
    predict(a) if a.cmd == "predict" else report(a)


if __name__ == "__main__":
    main()
