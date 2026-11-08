"""
Judge-free paired record (paper 2, Table V and the pooled 771-lobby figure) with player-aware uncertainty
and coverage diagnostics. Read-only on every existing artifact; writes only its own outputs.

The paired record (p3t_analyze.paired_record): for players covered under both arms, the player's
judge-window record r = y - WP_pop on the hero arm A's draft gives him (heroes assigned by earlier play
counts, p3t_judge.score_drafts) minus the same on arm B's; mean over covered players per lobby, then over
lobbies. Players recur across lobbies (and across the three draws), and a player's later games are reused
every time he recurs, so a lobby bootstrap alone treats correlated slots as independent.

Two stages:
  extract  (worker that holds the namespaces: the 3080; numpy only, light):
           ssh windows-3080-wsl 'source ~/hots/env.sh; cd ~/hots/repo/training && \
             nice -n 19 taskset -c 0-3 python - extract --cache <ns cache> --out <npz>' < p3t_paired_robust.py
           Per lobby and kernel slot: player id, assigned hero under each arm of each pair, the judge-window
           residual and raw records on it (recomputed from the slot table and checked against the scored
           rec_slot), earlier games, judge-window games, as-of rank, party, slot MMR, tier.
  analyze  (main box, nice 19, cores 44-47):
           python training/personalization/p3t_paired_robust.py analyze --data <dir with extract npz>
           -> results/P3T_PAIRED_ROBUST.{md,json}

CIs (normal, point +- 1.96 SE, as p3t_analyze): (a) lobby bootstrap, the paper's (500 draws, seed 0, so it
reproduces the paper) and the same at 2,000 draws; (b) player-cluster bootstrap (multinomial weights on
distinct players, shared across lobbies and draws); (c) two-way lobby x player pigeonhole bootstrap
(Owen 2007: product of independent lobby and player multinomial weights; conservative). The JSON also
has the max of the two one-way SEs. Pooled figure: the paper's inverse-variance fixed-effect pool of the
three draws (paper, replicate A, loose draw's 171 lobbies meeting the paper's rule), weights fixed at the
paper's lobby-bootstrap SEs; its SE comes from a joint bootstrap (lobbies within draw, players across all
draws, since the same players recur in the three draws).
"""
import argparse
import json
import os
import sys

import numpy as np

SEL = "x-combo1-dr-chance"
ROWS = [("pMCTS-cur", "The earlier search drafter"), ("Real", "Real drafts"), ("GD", "Imitation drafter"),
        ("DraftRec", "DraftRec, pick head"), ("DraftRec-pv", "DraftRec, pick-and-win rule"),
        ("aMCTS-pp", "Player-prior search")]
POOL_OPP = ["pMCTS-cur", "Real", "GD", "DraftRec"]
DRAWS = {"paper": "", "A_replicate": "_rep50", "loose": "_loose"}
SKILL_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
SKILL_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() and __file__ != "<stdin>" else None


# ------------------------------------------------------------------ extract (worker)

def extract(cache, out, pairs_opp):
    os.environ["P3_CACHE"] = cache
    os.environ.setdefault("P1_TIERS", "site")
    tr = os.getcwd()                                   # run from training/
    sys.path.insert(0, os.path.join(tr, "personalization"))
    sys.path.insert(0, tr)
    import itertools
    import p3t_data as TD
    perms = np.array(list(itertools.permutations(range(5))), np.int64)
    P = os.path.join(cache, "p3t")
    L = np.load(os.path.join(P, "lobbies.npz"))
    gi = np.load(os.path.join(P, "tourney", "lobbies.npz"))["gi"]
    n = len(gi)
    day, first, steps, slot_rows = L["day"], L["first_team"], L["steps"], L["slot_rows"]
    tier = L["tier"][gi].astype(str)
    win = np.flatnonzero(day <= TD.LOBBY_END)
    wpos = {int(g): i for i, g in enumerate(win)}
    # kernel slot -> pick index j (p3t_tourney.Lobbies)
    src = np.zeros((n, 10), np.int64)
    for i in range(n):
        cnt = [0, 0]
        j = 0
        for k in range(16):
            h, ty, tm, r = steps[gi[i]][k]
            if ty == 1:
                kt = int(tm ^ first[gi[i]])
                sl = 5 * kt + cnt[kt]
                cnt[kt] += 1
                src[i, sl] = j
                j += 1
    cur = np.load(os.path.join(P, "state_cur.npz"))
    jd = np.load(os.path.join(P, "state_judge.npz"))
    rc = ((gi * 10)[:, None] + src).ravel()
    rj = (np.array([wpos[int(g)] * 10 for g in gi])[:, None] + src).ravel()
    ncur = cur["n"][rc].astype(np.float32).reshape(n, 10, 90)     # float32 as p3t_tourney.Lobbies (tie-breaks)
    rsum = jd["rec_sum"][rj].astype(np.float64).reshape(n, 10, 90)
    rn = jd["rec_n"][rj].astype(np.float64).reshape(n, 10, 90)
    del cur, jd
    rows = np.take_along_axis(slot_rows[gi], src, 1)          # slot-table row of each kernel slot
    X = np.load(os.path.join(cache, "x_slots_ext.npz"))
    pid_all = X["pid"]
    pid = pid_all[rows]
    mmr = X["player_mmr"][rows]
    sh = TD.to_shared(X["hero_names"])
    vs = [str(v) for v in X["versions"]]
    sealed = vs.index(TD.SEALED) if TD.SEALED in vs else -999
    d_day = X["day"]
    m = (d_day >= TD.J0) & (d_day <= TD.J1) & X["post"] & (X["version"] != sealed)
    m &= np.isin(pid_all, np.unique(pid))
    wi = np.flatnonzero(m)
    key = pid_all[wi].astype(np.int64) * 90 + sh[X["hero"][wi]]
    uk, inv = np.unique(key, return_inverse=True)
    ysum = np.bincount(inv, weights=X["y"][wi].astype(np.float64))
    wsum = np.bincount(inv, weights=X["wp"][wi].astype(np.float64))
    cnt = np.bincount(inv)
    del X
    lr = np.load(os.path.join(P, "lobby_ranks.npz"))
    rank = np.take_along_axis(lr["rank"][gi], src, 1)
    party = np.take_along_axis(lr["party"][gi], src, 1)
    res = dict(gi=gi, pid=pid, tier=tier, mmr=mmr, rank=rank, party=party, ncur_tot=ncur.astype(np.float64).sum(2),
               jn_tot=rn.sum(2))
    chk = {}
    for o in pairs_opp:
        for a, b in ((SEL, o), (o, SEL)):
            f = f"{a}__{b}"
            acts = np.load(os.path.join(P, "tourney", f + ".npz"))["acts"]
            sc = np.load(os.path.join(P, "scores", f + ".npz"))
            hero = np.zeros((n, 10), np.int64)
            for i in range(n):
                for t in (0, 1):
                    hs = np.array([acts[i, k] for k in range(16) if SKILL_PICK[k] and SKILL_TEAM[k] == t])
                    sv = np.log1p(ncur[i, 5 * t:5 * t + 5])[np.arange(5)[:, None], hs[None, :]]
                    sp = sv[np.arange(5)[None, :], perms].sum(1)
                    pm = perms[int(np.argmax(sp))]
                    hero[i, 5 * t:5 * t + 5] = hs[pm]
            ii = np.arange(n)[:, None]
            jj = np.arange(10)[None, :]
            c = rn[ii, jj, hero]
            res_rec = np.where(c > 0, rsum[ii, jj, hero] / np.maximum(c, 1), np.nan)
            k2 = pid.astype(np.int64) * 90 + hero
            pos = np.searchsorted(uk, k2)
            pos = np.minimum(pos, len(uk) - 1)
            hit = uk[pos] == k2
            n2 = np.where(hit, cnt[pos], 0)
            raw = np.where(hit & (n2 > 0), ysum[pos] / np.maximum(n2, 1), np.nan)
            res2 = np.where(hit & (n2 > 0), (ysum[pos] - wsum[pos]) / np.maximum(n2, 1), np.nan)
            rs = sc["rec_slot"]
            chk[f] = {"rec_slot_vs_state": float(np.nanmax(np.abs(rs - res_rec))),
                      "slots_differ": int(((np.abs(rs - res_rec) > 1e-5) | (np.isfinite(rs) != np.isfinite(res_rec))).sum()),
                      "nan_agree": bool((np.isfinite(rs) == np.isfinite(res_rec)).all()),
                      "count_agree": bool((n2 == c).all()),
                      "slot_table_vs_state": float(np.nanmax(np.abs(res2 - res_rec)))}
            res[f"hero|{f}"] = hero
            res[f"rec|{f}"] = rs
            res[f"raw|{f}"] = raw
            res[f"n|{f}"] = c
            res[f"nearlier|{f}"] = ncur[ii, jj, hero]
    res["checks"] = np.array(json.dumps(chk))
    np.savez(out, **res)
    print(json.dumps(chk, indent=1))


# ------------------------------------------------------------------ analyze (main box)

def arms(z, o):
    """(A, B) per (lobby, slot) for SEL and opp o: residual and raw records, counts, earlier games."""
    ab, ba = f"{SEL}__{o}", f"{o}__{SEL}"
    out = {}
    for k in ("rec", "raw", "n", "nearlier", "hero"):
        x, y = z[f"{k}|{ab}"], z[f"{k}|{ba}"]
        out[k + "_A"] = np.concatenate([x[:, :5], y[:, 5:]], 1).astype(np.float64)
        out[k + "_B"] = np.concatenate([y[:, :5], x[:, 5:]], 1).astype(np.float64)
    return out


def stat(d, wl, ws):
    """Weighted paired record: d (n, 10) with NaN off the common coverage; wl (..., n) lobby weights;
    ws (..., n, 10) slot (player) weights. Returns (...,)."""
    f = np.isfinite(d)
    dd = np.where(f, d, 0.0)
    W = ws * f
    sw = W.sum(-1)
    per = np.where(sw > 0, (W * dd).sum(-1) / np.where(sw > 0, sw, 1), 0.0)
    L = wl * (sw > 0)
    return (L * per).sum(-1) / L.sum(-1)


def mult(rng, B, n):
    """multinomial resampling counts, rows = draws (the same draws as rng.integers(0, n, n) per row)."""
    out = np.zeros((B, n))
    for b in range(B):
        out[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return out


class Draw:
    def __init__(self, z, sub=None):
        self.z = z
        self.sub = np.arange(len(z["gi"])) if sub is None else sub
        self.pid = z["pid"][self.sub]
        self.n = len(self.sub)

    def d(self, o, kind="rec"):
        a = arms(self.z, o)
        A, Bv = a[kind + "_A"][self.sub], a[kind + "_B"][self.sub]
        # common coverage is defined by the residual record (the same games for raw)
        both = np.isfinite(a["rec_A"][self.sub]) & np.isfinite(a["rec_B"][self.sub])
        return np.where(both, A - Bv, np.nan), a, both


def weights(draws, B, seed, kind):
    """Per draw: lobby weights (B, n) and slot weights (B, n, 10); player weights shared across draws."""
    rng = np.random.default_rng(seed)
    allp = np.unique(np.concatenate([D.pid.ravel() for D in draws]))
    out = []
    if kind in ("player", "twoway"):
        pw = mult(np.random.default_rng(seed + 1), B, len(allp))
    for D in draws:
        wl = mult(rng, B, D.n) if kind in ("lobby", "twoway") else np.ones((B, D.n))
        if kind in ("player", "twoway"):
            ws = pw[:, np.searchsorted(allp, D.pid)]
        elif kind == "slot":
            # no clustering at all (each lobby-slot independent): the iid term of the multiway formula
            ws = mult(rng, B, D.n * 10).reshape(B, D.n, 10)
        else:
            ws = np.ones((1, D.n, 10))
        out.append((wl, ws))
    return out


def se_of(draws, dlist, B, seed, kind, comb=None):
    W = weights(draws, B, seed, kind)
    th = np.stack([stat(d, wl, ws) for d, (wl, ws) in zip(dlist, W)], 1)   # (B, k)
    v = th[:, 0] if comb is None else th @ comb
    return float(np.nanstd(v)), v


def cgm(se_l, se_p, se_s):
    """Cameron-Gelbach-Miller two-way variance V_lobby + V_player - V_iid (floored at the larger one-way)."""
    return float(np.sqrt(max(se_l ** 2 + se_p ** 2 - se_s ** 2, se_l ** 2, se_p ** 2)))


def all_se(draws, ds, B, seed, comb=None):
    S = {k: se_of(draws, ds, B, seed, k, comb)[0] for k in ("lobby", "player", "twoway", "slot")}
    S["cgm"] = cgm(S["lobby"], S["player"], S["slot"])
    return S


def paper_lobby_se(D, d):
    """exactly p3t_analyze: boot_idx(n, 2000, seed 0)[:500], nanstd of the statistic."""
    rng = np.random.default_rng(0)
    idxs = [rng.integers(0, D.n, D.n) for _ in range(2000)][:500]
    bs = []
    for ix in idxs:
        dd = d[ix]
        with np.errstate(invalid="ignore"):
            per = np.nanmean(dd, 1)
        bs.append(np.nanmean(per))
    return float(np.nanstd(bs))


def fmt(pt, se):
    return [round(100 * pt, 2), round(100 * (pt - 1.96 * se), 2), round(100 * (pt + 1.96 * se), 2)]


def summarize_cov(D, o):
    """covered-under-both slots vs the rest: medians/means of player descriptors."""
    z = D.z
    _, a, both = D.d(o)
    s = D.sub
    feats = {"earlier games (all heroes)": z["ncur_tot"][s], "judge-window games (all heroes)": z["jn_tot"][s],
             "earlier games on the selected design's hero": a["nearlier_A"][s],
             "earlier games on the comparator's hero": a["nearlier_B"][s],
             "as-of rank (MMR)": z["rank"][s].astype(np.float64), "slot MMR": z["mmr"][s].astype(np.float64)}
    party = z["party"][s]
    psize = np.ones(party.shape)
    for i in range(party.shape[0]):
        for t in (0, 1):
            p = party[i, 5 * t:5 * t + 5]
            for j in range(5):
                if p[j] != 0:
                    psize[i, 5 * t + j] = (p == p[j]).sum()
    feats["party size"] = psize
    tier = np.repeat(z["tier"][s][:, None], 10, 1)
    out = {}
    for k, v in feats.items():
        v = np.where(v <= 0, np.nan, v) if k in ("as-of rank (MMR)", "slot MMR") else v
        c, u = v[both], v[~both]
        out[k] = {"covered_median": float(np.nanmedian(c)), "uncovered_median": float(np.nanmedian(u)),
                  "covered_mean": float(np.nanmean(c)), "uncovered_mean": float(np.nanmean(u)),
                  "std_mean_diff": float((np.nanmean(c) - np.nanmean(u)) / np.nanstd(v))}
    out["solo share"] = {"covered": float((psize[both] == 1).mean()), "uncovered": float((psize[~both] == 1).mean())}
    out["tier share covered/uncovered"] = {t: [float((tier[both] == t).mean()), float((tier[~both] == t).mean())]
                                           for t in np.unique(tier)}
    return out


def analyze(data, B):
    Z = {k: np.load(os.path.join(data, f"robust_{k}.npz")) for k in DRAWS}
    chk = {k: json.loads(str(Z[k]["checks"])) for k in Z}
    loose_sub = np.flatnonzero((Z["loose"]["ncur_tot"] >= 50).all(1))
    Dp = Draw(Z["paper"])
    out = {"B": B, "checks": chk, "rows": {}, "pooled": {}, "coverage_compare": {}, "recurrence": {}}
    # recurrence
    for k, D in (("paper", Dp), ("A_replicate", Draw(Z["A_replicate"])), ("loose_171", Draw(Z["loose"], loose_sub))):
        u, c = np.unique(D.pid, return_counts=True)
        out["recurrence"][k] = {"slots": int(D.pid.size), "players": int(len(u)), "players_2plus": int((c > 1).sum()),
                                "slots_of_2plus": int(c[c > 1].sum()), "max": int(c.max())}
    allp = [Dp.pid, Z["A_replicate"]["pid"], Z["loose"]["pid"][loose_sub]]
    u_all, c_all = np.unique(np.concatenate([p.ravel() for p in allp]), return_counts=True)
    out["recurrence"]["pooled_771"] = {"slots": int(c_all.sum()), "players": int(len(u_all)),
                                       "players_2plus": int((c_all > 1).sum()), "slots_of_2plus": int(c_all[c_all > 1].sum())}
    sets = [set(np.unique(p).tolist()) for p in allp]
    out["recurrence"]["players_shared_between_draws"] = {"paper&A": len(sets[0] & sets[1]), "paper&loose171": len(sets[0] & sets[2]),
                                                         "A&loose171": len(sets[1] & sets[2])}
    for o, label in ROWS:
        d, a, both = Dp.d(o)
        pt = float(stat(d, np.ones(Dp.n), np.ones((Dp.n, 10))))
        se_paper = paper_lobby_se(Dp, d)
        S = all_se([Dp], [d], B, 11)
        se_l, se_p, se_2, se_s, se_cgm = S["lobby"], S["player"], S["twoway"], S["slot"], S["cgm"]
        draw_raw, _, _ = Dp.d(o, "raw")
        pt_raw = float(stat(draw_raw, np.ones(Dp.n), np.ones((Dp.n, 10))))
        SR = all_se([Dp], [draw_raw], B, 11)
        # raw records on each arm's own coverage (not paired)
        out["rows"][label] = {
            "opp": o, "point": round(100 * pt, 3),
            "lobby_paper500": fmt(pt, se_paper), "lobby": fmt(pt, se_l), "player": fmt(pt, se_p), "twoway": fmt(pt, se_2),
            "max_oneway": fmt(pt, max(se_l, se_p)), "cgm": fmt(pt, se_cgm),
            "se": {"lobby_paper500": se_paper, "lobby": se_l, "player": se_p, "twoway": se_2, "slot_iid": se_s,
                   "cgm": se_cgm},
            "se_ratio_twoway_over_lobby": round(se_2 / se_paper, 3),
            "raw": fmt(pt_raw, SR["lobby"]), "raw_cgm": fmt(pt_raw, SR["cgm"]), "raw_twoway": fmt(pt_raw, SR["twoway"]),
            "wp_gap": round(100 * (pt_raw - pt), 3),
            "coverage": {"sel": round(100 * float(np.isfinite(a["rec_A"]).mean()), 2),
                         "opp": round(100 * float(np.isfinite(a["rec_B"]).mean()), 2),
                         "both": round(100 * float(both.mean()), 2)},
            "games_per_covered_slot": {"sel": float(np.nanmedian(np.where(both, a["n_A"], np.nan))),
                                       "opp": float(np.nanmedian(np.where(both, a["n_B"], np.nan)))},
        }
        out["coverage_compare"][label] = summarize_cov(Dp, o)
        # reuse of the same later games: covered slots whose (player, hero under A, hero under B) recurs
        tri = np.stack([Dp.pid[both], a["hero_A"][both], a["hero_B"][both]], 1).astype(np.int64)
        _, ct = np.unique(tri, axis=0, return_counts=True)
        out["rows"][label]["reuse"] = {"covered_slots": int(both.sum()), "distinct_player_hero_pairs": int(len(ct)),
                                       "slots_in_repeated_triples": int(ct[ct > 1].sum()),
                                       "distinct_players_covered": int(len(np.unique(Dp.pid[both])))}
        print(label, out["rows"][label]["point"], out["rows"][label]["lobby_paper500"], out["rows"][label]["player"],
              out["rows"][label]["twoway"], flush=True)
    # pooled 771
    Ds = [Dp, Draw(Z["A_replicate"]), Draw(Z["loose"], loose_sub)]
    for o in POOL_OPP:
        ds = [D.d(o)[0] for D in Ds]
        dr = [D.d(o, "raw")[0] for D in Ds]
        pts = np.array([float(stat(d, np.ones(D.n), np.ones((D.n, 10)))) for d, D in zip(ds, Ds)])
        sep = np.array([paper_lobby_se(D, d) for d, D in zip(ds, Ds)])
        w = (1 / sep ** 2) / (1 / sep ** 2).sum()
        pt = float(w @ pts)
        r = {"draws": {k: fmt(p, s) for k, p, s in zip(("paper", "A", "loose171"), pts, sep)},
             "weights": [round(float(x), 3) for x in w], "point": round(100 * pt, 3),
             "paper_iv_pool": fmt(pt, float(np.sqrt(1 / (1 / sep ** 2).sum())))}
        ses = all_se(Ds, ds, B, 21, w)
        for kind, se in ses.items():
            r[kind] = fmt(pt, se)
        r["se"] = ses
        for kind in ("lobby", "player", "twoway"):
            ind = np.array([se_of([D], [d], B, 31 + j, kind)[0] for j, (D, d) in enumerate(zip(Ds, ds))])
            wi = 1 / ind ** 2
            r[kind + "_iv_independent_draws"] = fmt(float(wi @ pts / wi.sum()), float(np.sqrt(1 / wi.sum())))
        ptr = np.array([float(stat(d, np.ones(D.n), np.ones((D.n, 10)))) for d, D in zip(dr, Ds)])
        sr = all_se(Ds, dr, B, 21, w)
        r["raw"] = fmt(float(w @ ptr), sr["lobby"])
        r["raw_cgm"] = fmt(float(w @ ptr), sr["cgm"])
        r["raw_twoway"] = fmt(float(w @ ptr), sr["twoway"])
        # one more view: all 771 lobbies as one set (each lobby weight 1)
        dcat = np.concatenate(ds)
        r["flat_771_mean"] = round(100 * float(stat(dcat, np.ones(len(dcat)), np.ones(dcat.shape))), 3)
        r["coverage_both"] = [round(100 * float(np.isfinite(d).mean()), 1) for d in ds]
        out["pooled"][o] = r
        print("pooled", o, r["point"], r["paper_iv_pool"], r["player"], r["twoway"], flush=True)
    return out


def write_md(out, path):
    L = []
    L.append("# Paper 2 paired record: player-clustered and two-way CIs, raw differences, coverage (2026-11-08)\n")
    L.append("Script: `training/personalization/p3t_paired_robust.py` (extract on the 3080 from the def1027 headline, "
             "rep50 and loose namespaces; analyze on the main box). Read-only on every existing artifact.\n")
    L.append(f"Points in pp, selected design (`{SEL}`) minus comparator; 95% CI = point +- 1.96 SE. "
             f"Bootstrap draws: {out['B']} (lobby/player/two-way); the paper's lobby CI is reproduced with its own 500 draws, seed 0. "
             "Two-way CGM = Cameron-Gelbach-Miller multiway variance V_lobby + V_player - V_iid (V_iid from an unclustered slot "
             "bootstrap; floored at the larger one-way), the headline two-way interval. Two-way pigeonhole = product of independent "
             "lobby and player multinomial weights (Owen 2007), which double-counts the iid term and is conservative. "
             "Raw = realized win rate on the assigned hero, no WP subtraction, same covered players.\n")
    L.append("## Table V rows (300 headline lobbies)\n")
    L.append("| vs | point | lobby CI (paper, 500) | lobby CI (2k) | player CI | two-way CGM | two-way pigeonhole | SE x vs paper (player / CGM) | raw diff (CGM CI) | coverage sel / opp / both |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    f3 = lambda v: f"{v[0]:+.1f} ({v[1]:.1f}, {v[2]:.1f})"
    f2 = lambda v: f"({v[1]:.1f}, {v[2]:.1f})"
    for lab, r in out["rows"].items():
        c = r["coverage"]
        L.append(f"| {lab} | {r['point']:+.2f} | {f2(r['lobby_paper500'])} | {f2(r['lobby'])} | {f2(r['player'])} | "
                 f"**{f2(r['cgm'])}** | {f2(r['twoway'])} | {r['se']['player'] / r['se']['lobby_paper500']:.2f} / "
                 f"{r['se']['cgm'] / r['se']['lobby_paper500']:.2f} | {f3(r['raw_cgm'])} | "
                 f"{c['sel']:.0f}% / {c['opp']:.0f}% / {c['both']:.0f}% |")
    cross = [lab for lab, r in out["rows"].items() if r["cgm"][1] <= 0 < r["lobby_paper500"][1]]
    crossp = [lab for lab, r in out["rows"].items() if r["twoway"][1] <= 0 < r["lobby_paper500"][1]]
    ratio = [r["se"]["cgm"] / r["se"]["lobby_paper500"] for r in out["rows"].values()]
    rep = [100 * r["reuse"]["slots_in_repeated_triples"] / r["reuse"]["covered_slots"] for r in out["rows"].values()]
    L.append(f"\n**Reading.** Player-cluster SEs are within {100 * max(abs(r['se']['player'] / r['se']['lobby_paper500'] - 1) for r in out['rows'].values()):.0f}% "
             f"of the paper's lobby SEs, and both are close to the unclustered slot SE: a recurring player rarely gets the same "
             f"hero pair again (repeated (player, hero A, hero B) triples are {min(rep):.0f}-{max(rep):.0f}% of covered slots), so reuse of later games adds "
             f"little correlation. The two-way CGM SE is {min(ratio):.2f}-{max(ratio):.2f}x the paper's; intervals newly reaching 0 "
             f"under CGM: {cross or 'none'}. The pigeonhole bound counts the slot-level variance two extra times (here about "
             f"sqrt(3)x the one-way SE) and would put {crossp or 'none'} at or below 0; it is reported as an upper bound only.\n")
    L.append("\n## Pooled figure (paper + A + loose's 171 = 771 lobbies)\n")
    L.append("Fixed-effect IV pool with the paper's weights; the SE from a joint bootstrap (lobbies within draw, players "
             "shared across draws). `iv-indep` re-pools with each scheme's per-draw SEs assuming independent draws.\n")
    L.append("| vs | point | paper IV CI | lobby (joint) | player (joint) | two-way CGM (joint) | two-way pigeonhole (joint) | raw diff (CGM CI) | per-draw (paper / A / loose171) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for o, r in out["pooled"].items():
        dd = " / ".join(f"{v[0]:+.1f}" for v in r["draws"].values())
        L.append(f"| {o} | {r['point']:+.2f} | {f2(r['paper_iv_pool'])} | {f2(r['lobby'])} | {f2(r['player'])} | "
                 f"**{f2(r['cgm'])}** | {f2(r['twoway'])} | {f3(r['raw_cgm'])} | {dd} |")
    L.append("\n## Player recurrence\n")
    for k, v in out["recurrence"].items():
        L.append(f"- {k}: {v}")
    L.append("\nReuse of the same later games among covered slots (headline):\n")
    for lab, r in out["rows"].items():
        u = r["reuse"]
        L.append(f"- {lab}: {u['covered_slots']} covered slots, {u['distinct_players_covered']} distinct players, "
                 f"{u['slots_in_repeated_triples']} slots repeat a (player, hero A, hero B) triple")
    L.append("\n## Covered (both arms) vs not, headline lobbies\n")
    L.append("Medians (covered / uncovered) and the standardized mean difference.\n")
    first = next(iter(out["coverage_compare"].values()))
    keys = [k for k in first if isinstance(first[k], dict) and "covered_median" in first[k]]
    L.append("| vs | " + " | ".join(keys) + " | solo share |")
    L.append("|---|" + "---|" * (len(keys) + 1))
    for lab, cc in out["coverage_compare"].items():
        cells = []
        for k in keys:
            v = cc[k]
            cells.append(f"{v['covered_median']:.0f} / {v['uncovered_median']:.0f} (d {v['std_mean_diff']:+.2f})")
        s = cc["solo share"]
        L.append(f"| {lab} | " + " | ".join(cells) + f" | {100 * s['covered']:.0f}% / {100 * s['uncovered']:.0f}% |")
    L.append("\nTier shares covered/uncovered by row are in the JSON.\n")
    L.append("## Extraction checks\n")
    bad = {d: {f: v for f, v in c.items() if not (v["rec_slot_vs_state"] < 1e-5 and v["nan_agree"] and v["count_agree"])}
           for d, c in out["checks"].items()}
    L.append(f"Recomputed per-slot records match the scored `rec_slot` in every pair file and draw: "
             f"{all(not v for v in bad.values())}; slot-table residual vs state max abs diff: "
             f"{max(c['slot_table_vs_state'] for cc in out['checks'].values() for c in cc.values()):.2e}.\n")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "analyze"])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--opps", default=",".join(o for o, _ in ROWS))
    ap.add_argument("--data")
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    if a.stage == "extract":
        extract(a.cache, a.out, a.opps.split(","))
        return
    res = os.path.join(HERE, "results")
    out = analyze(a.data, a.B)
    json.dump(out, open(os.path.join(res, "P3T_PAIRED_ROBUST.json"), "w"), indent=1)
    write_md(out, os.path.join(res, "P3T_PAIRED_ROBUST.md"))


if __name__ == "__main__":
    main()
