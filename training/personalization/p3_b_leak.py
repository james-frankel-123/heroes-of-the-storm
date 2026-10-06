"""
P3 review item 1, leak checks for the joint combiner (p3_b_joint.py).

(a) Lag 7. Every non-status term is rebuilt from games of days <= t - 7:
    skill state and counts (online_predict lag 7, offset at lag-7 counts),
    main share and forced off main, fine-role familiarity, EWMA-100
    residual, rust (days since hero / any game), hero MMR (the latest causal
    at-game value from a game of day <= t - 7, same fallbacks), nowcast
    (cell mean over days <= t - 7). The same-day term is dropped; status and
    party are unchanged. The joint fit is then compared with its own lag-7
    headline (the headline itself went from +0.0142 at lag 1 to +0.0112 at
    lag 7). A real effect shrinks gently; a leak collapses or stays flat.
    The lag-L walks are checked against the lag-1 features at L = 1.
(b) Shuffle. The joint terms (everything except logit WP, the headline s
    and the game-level party terms) are permuted as whole slot vectors across
    slots of the same (day, tier) cell; combiner refit on V1, scored on V2
    and OOT. The joint gain over the headline should drop to about 0.
(c) Older history. Every term, and s, is taken as of the player's game two
    games back in play order (that game's own lag-1 features, so history up
    to the day before it); no same-day term. Should shrink a little, like
    lag 7.
All variants, and the phase-1 and C-skill joint fits, are scored on ONE
common game set: complete games where every variant has features for all
ten slots, including the C skill of cache/b7_onetrick_preds.npz.

Same windows, sealed build and bootstrap as p3_b_joint (sealed build scored
only with --final).

Usage (from training/): python personalization/p3_b_leak.py [--sample 0.1]
Output: results/p3_b_leak.json (features of the extended query set cached as
cache/b13_leak_feats.npz)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from numba import njit

import p3_hs_core as C
import p3_x_common as X
import p3_fix_counts as F
import p3_b_joint as J
from p3_heroes import NUM_HEROES, HKEY

log = J.log


# Every builder below takes a per-row cutoff day: history = the player's rows
# with day <= cut[i]. Lag L is cut = day - L (L = 1 reproduces p3_b_joint);
# "older history" is cut = (day of the game two games back) - 1.

@njit(cache=True)
def _main_walk_cut(starts, ends, day, hero, cut, o_tot, o_top, o_main, o_nown):
    """Rows of each player in play order; cut nondecreasing within player."""
    for p in range(starts.shape[0]):
        cnt = np.zeros(NUM_HEROES)
        a = starts[p]
        for i in range(starts[p], ends[p]):
            while a < ends[p] and day[a] <= cut[i]:
                cnt[hero[a]] += 1.0
                a += 1
            tot = 0.0
            mx = 0.0
            mh = -1
            for h in range(NUM_HEROES):
                tot += cnt[h]
                if cnt[h] > mx:
                    mx = cnt[h]
                    mh = h
            o_tot[i] = tot
            o_top[i] = mx
            o_main[i] = mh
            o_nown[i] = cnt[hero[i]]


@njit(cache=True)
def _mom_walk_cut(starts, ends, day, hero, r, cut, o_ew, o_dsh, o_dsa):
    al = 1.0 - np.exp(-np.log(2.0) / 100.0)
    for p in range(starts.shape[0]):
        ew = 0.0
        w = 0.0
        last_h = np.full(NUM_HEROES, -1)
        last_any = -1
        a = starts[p]
        for i in range(starts[p], ends[p]):
            while a < ends[p] and day[a] <= cut[i]:
                ew = (1 - al) * ew + al * r[a]
                w = (1 - al) * w + al
                last_h[hero[a]] = day[a]
                last_any = day[a]
                a += 1
            o_ew[i] = ew / w if w > 0 else 0.0
            h = hero[i]
            o_dsh[i] = day[i] - last_h[h] if last_h[h] >= 0 else -1.0
            o_dsa[i] = day[i] - last_any if last_any >= 0 else -1.0


def play_order(dv):
    """Rows sorted by (pid, day, start time, replay_id), as for p2."""
    o = np.lexsort((dv["replay_id"], dv["t_start"], dv["day"], dv["pid"]))
    pid = dv["pid"][o]
    brk = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    return o, brk.astype(np.int64), np.r_[brk[1:], len(pid)].astype(np.int64)


def key_counts_cut(dv, key, cut):
    """Rows with the same key and day <= cut[i]."""
    day = dv["day"].astype(np.int64)
    o = np.lexsort((day, key))
    k, dd = key[o], day[o]
    gid = np.cumsum(np.r_[True, k[1:] != k[:-1]]) - 1
    B = np.int64(1_000_000)
    comp = gid.astype(np.int64) * B + dd
    gstart = np.searchsorted(comp, gid.astype(np.int64) * B, side="left")
    c = np.maximum(cut[o].astype(np.int64), -1)
    pos = np.searchsorted(comp, gid.astype(np.int64) * B + c, side="right")
    out = np.empty(len(o), np.int64)
    out[o] = np.maximum(pos - gstart, 0)
    return out


def last_value_cut(dv, values, key, cut):
    """Latest non-NaN value of the same key from a row with day <= cut[i]."""
    n = len(values)
    day = dv["day"].astype(np.int64)
    o = np.lexsort((dv["replay_id"], day, key))
    k, dd, vs = key[o], day[o], values[o]
    gid = np.cumsum(np.r_[True, k[1:] != k[:-1]]) - 1
    B = np.int64(1_000_000)
    comp = gid.astype(np.int64) * B + dd
    gstart = np.searchsorted(comp, gid.astype(np.int64) * B, side="left")
    idx = np.where(~np.isnan(vs), np.arange(n), -1)
    run = np.maximum.accumulate(idx)
    pos = np.searchsorted(comp, gid.astype(np.int64) * B + np.maximum(cut[o].astype(np.int64), -1), side="right") - 1
    okp = pos >= gstart
    j = np.where(okp, run[np.maximum(pos, 0)], -1)
    good = okp & (j >= gstart)
    tmp = np.full(n, np.nan)
    tmp[good] = vs[j[good]]
    out = np.empty(n)
    out[o] = tmp
    return out


def cell_mean_cut(dv, r, key, cut, k_shrink):
    """Shrunken mean of r over rows of the same cell with day <= cut[i]."""
    day = dv["day"].astype(np.int64)
    o = np.lexsort((dv["replay_id"], day, key))
    ck, dy = key[o], day[o]
    cs = np.r_[0.0, np.cumsum(r[o])]
    gid = np.cumsum(np.r_[True, ck[1:] != ck[:-1]]) - 1
    B = np.int64(1_000_000)
    comp = gid.astype(np.int64) * B + dy
    gstart = np.searchsorted(comp, gid.astype(np.int64) * B, side="left")
    pos = np.searchsorted(comp, gid.astype(np.int64) * B + np.maximum(cut[o].astype(np.int64), -1), side="right")
    pos = np.maximum(pos, gstart)
    nb = (pos - gstart).astype(float)
    out = np.empty(len(o))
    out[o] = (cs[pos] - cs[gstart]) / (nb + k_shrink)
    return out


def online_cut(dv, add, q, cut, K, r_adj):
    """GP mean and counts for query rows with state from add rows of day <=
    cut (query copies appended with day = cut + 1, lag 1)."""
    qi = np.flatnonzero(q)
    o = np.argsort(cut[qi], kind="stable")
    qs = qi[o]
    d2 = {"day": np.r_[dv["day"], (cut[qs] + 1).astype(dv["day"].dtype)], "pid": np.r_[dv["pid"], dv["pid"][qs]],
          "hero": np.r_[dv["hero"], dv["hero"][qs]], "v": np.r_[dv["v"], dv["v"][qs]],
          "n_players": dv["n_players"]}
    n = len(dv["pid"])
    add2 = np.r_[add, np.zeros(len(qs), bool)]
    q2 = np.r_[np.zeros(n, bool), np.ones(len(qs), bool)]
    _, m, _, Nh, Np, _ = X.online_predict(d2, add2, q2, K, np.r_[r_adj, np.zeros(len(qs))])
    out = {k: np.full(n, np.nan) for k in ("m", "Nh", "Np")}
    out["m"][qs], out["Nh"][qs], out["Np"][qs] = m, Nh, Np
    return out


def build_asof(d, rows, Fz1, cut_of, label):
    """All terms as of a per-row cutoff day (cut_of(dv) -> cut per view
    row). s = lag-1 table at the as-of counts + GP mean on the as-of state;
    status = status table minus phase-1 table at the as-of counts; party
    unchanged; no same-day term."""
    t0 = time.time()
    n = len(d["pid"])
    Ks, _ = X.kernels()
    K = Ks[J.KNAME]
    meta = C.hero_meta(d["hero_names"])
    e_mask, n_p, n_ph, table1, r_adj1 = F.prepare_l1(d)
    import p3_b_demean as DM
    import p3_hero_level_causal as HL
    t_end, t_start, post_draft = J.game_times(d)
    d["t_start"] = t_start
    dr = J.draft_info(d, post_draft)
    d["pick_rank"] = dr["rank"]
    status = HL.account_status(d, J.hl_arrays(d, "ext", t_start))
    nb = len(C.NP_EDGES) * len(C.NPH_EDGES)
    key4 = status * nb + C.exp_bins(n_p, n_ph)
    table4 = np.bincount(key4[e_mask], weights=d["r"][e_mask], minlength=4 * nb) / \
        (np.bincount(key4[e_mask], minlength=4 * nb) + 200.0)
    build, _ = DM.row_builds(d)
    ckey = build * NUM_HEROES + d["hero"]
    tau2 = DM.estimate_tau2(r_adj1, d["v"], ckey, e_mask)[0]
    k_now = float(np.mean(d["v"])) / (2.0 * tau2)
    M = J.causal_mmr(d)
    Fz = {}
    for view, qm, add in (("snapshot", rows["V1"] | rows["V2"], ~d["post"]), ("all", rows["OOT"], np.ones(n, bool))):
        dv = J.subset(d, add)
        sel = qm[add]
        dest = np.flatnonzero(add)[sel]
        cut = cut_of(dv)
        g = online_cut(dv, np.ones(len(dv["pid"]), bool), sel, cut, K, r_adj1[add])
        bl = C.exp_bins(np.nan_to_num(g["Np"]).astype(np.int64), np.nan_to_num(g["Nh"]).astype(np.int64))
        vals = {"s": table1[bl] + g["m"], "status": table4[status[add] * nb + bl] - table1[bl]}
        o, st, en = play_order(dv)
        nn = len(o)
        tot, top, nown, ew, dsh, dsa = (np.zeros(nn) for _ in range(6))
        mh = np.zeros(nn, np.int64)
        _main_walk_cut(st, en, dv["day"][o].astype(np.int64), dv["hero"][o].astype(np.int64),
                       cut[o].astype(np.int64), tot, top, mh, nown)
        _mom_walk_cut(st, en, dv["day"][o].astype(np.int64), dv["hero"][o].astype(np.int64),
                      dv["r"][o].astype(np.float64), cut[o].astype(np.int64), ew, dsh, dsa)
        un = {}
        for k_, a_ in (("tot", tot), ("top", top), ("main", mh), ("nown", nown), ("ew", ew), ("dsh", dsh), ("dsa", dsa)):
            b_ = np.empty_like(a_)
            b_[o] = a_
            un[k_] = b_
        # main-share terms (same definitions as p3_b_joint.main_terms)
        hero, gg, rank, main = dv["hero"], dv["g"], dv["pick_rank"], un["main"]
        ms = un["top"] / np.maximum(un["tot"], 1)
        is_main = (hero == main) & (main >= 0)
        bb = np.zeros(nn, bool)
        for k_ in range(6):
            bb |= (dr["ban_h"][gg, k_] == main) & (main >= 0) & (~dr["ban_second"][gg, k_] | (rank >= 5))
        gk = gg.astype(np.int64) * HKEY + hero
        og = np.argsort(gk)
        sgk, srank = gk[og], rank[og]
        mk = gg.astype(np.int64) * HKEY + np.maximum(main, 0)
        jj = np.minimum(np.searchsorted(sgk, mk), len(sgk) - 1)
        tb = (sgk[jj] == mk) & ~is_main & (main >= 0) & (srank[jj] >= 0) & (rank >= 0) & (srank[jj] < rank)
        unav = (un["tot"] >= 30) & ~is_main & (bb | tb)
        vals.update({"share_hero": un["nown"] / np.maximum(un["tot"], 1), "is_main": is_main.astype(float),
                     "offmain_ms": ms * (~is_main), "forced_ms": ms * unav, "ewma100": un["ew"],
                     "rust_hero": np.log1p(np.minimum(np.maximum(un["dsh"], 0), 365)) * (un["dsh"] >= 0),
                     "log_gap_any": np.log1p(np.minimum(np.maximum(un["dsa"], 0), 365)) * (un["dsa"] >= 0)})
        fine = meta["fine"][hero]
        npc = key_counts_cut(dv, dv["pid"], cut)
        npf = key_counts_cut(dv, dv["pid"] * 16 + fine, cut)
        base = np.bincount(fine[e_mask[add]], minlength=len(meta["fine_names"])) / max(e_mask[add].sum(), 1)
        vals["offrole_fine"] = ((npc >= 50) & (npf / np.maximum(npc, 1) < 0.10)).astype(float)
        vals["log_share_fine"] = np.log((npf + 10 * base[fine]) / (npc + 10))
        role = meta["blizz"][hero]
        Mv = {k_: M[k_][add] for k_ in ("player", "role", "hero")}
        vals["hero_mmr"] = J.hero_mmr_term({
            "player": last_value_cut(dv, Mv["player"], dv["pid"], cut),
            "role": last_value_cut(dv, Mv["role"], dv["pid"] * 8 + role, cut),
            "hero": last_value_cut(dv, Mv["hero"], dv["pid"] * HKEY + hero, cut)})
        vals["nowcast"] = cell_mean_cut(dv, r_adj1[add], ckey[add], cut, k_now)
        for k_, v_ in vals.items():
            Fz.setdefault(k_, np.full(n, np.nan))[dest] = np.asarray(v_, np.float64)[sel]
        del dv
        log(f"{label} features ({view}): {time.time() - t0:.0f}s")
    Fz["sd_mean"] = np.zeros(n)
    Fz["sd_logk"] = np.zeros(n)
    checks = {}
    if Fz1 is not None:
        q = rows["V1"] | rows["V2"] | rows["OOT"]
        for k_ in ("s", "status", "share_hero", "forced_ms", "offrole_fine", "ewma100", "rust_hero", "nowcast"):
            if k_ in Fz1:
                checks[k_] = float(np.nanmax(np.abs(Fz[k_][q] - Fz1[k_][q])))
    return Fz, checks


def cut_older(dv):
    """Cutoff = day of the player's game two games back (play order) - 1;
    no such game: no history."""
    o, st, en = play_order(dv)
    pid = dv["pid"][o]
    p2 = np.r_[-1, -1, np.arange(len(o) - 2)]
    ok = (p2 >= 0) & (pid[np.maximum(p2, 0)] == pid)
    day = dv["day"][o].astype(np.int64)
    c = np.where(ok, day[np.maximum(p2, 0)] - 1, int(dv["day"].min()) - 2)
    out = np.empty(len(o), np.int64)
    out[o] = c
    return out


def prev2_index(d, t_start):
    """Row of the same player's game two games back in play order (-1 if none)."""
    o = np.lexsort((d["replay_id"], t_start, d["day"], d["pid"]))
    pid = d["pid"][o]
    p2 = np.r_[-1, -1, np.arange(len(o) - 2)]
    ok = (p2 >= 0) & (pid[np.maximum(p2, 0)] == pid)
    out = np.full(len(o), -1)
    out[o] = np.where(ok, o[np.maximum(p2, 0)], -1)
    return out


def shuffle_terms(d, Fz, cols, qmask, seed=0):
    """Permute the term vectors across query slots within (day, tier) cells."""
    import p3_b_demean as DM
    tier = DM.row_tiers(d)
    rng = np.random.RandomState(seed)
    q = np.flatnonzero(qmask)
    cell = d["day"][q].astype(np.int64) * 8 + tier[q]
    o = np.lexsort((rng.rand(len(q)), cell))
    srt = q[np.argsort(cell, kind="stable")]  # rows grouped by cell, original order
    shuf = q[o]  # rows grouped by cell, random order within cell
    out = dict(Fz)
    for c in cols:
        a = Fz[c].copy()
        a[srt] = Fz[c][shuf]
        out[c] = a
    return out


class Scorer:
    """Fits on V1 and paired gains on the common games."""

    def __init__(self, d, games, gd):
        self.d, self.games, self.gd = d, games, gd
        self.tests = [k for k in games if k != "V1"]
        clus = J.clusters_first_player(d, gd[0])
        week = (gd[4] // 7).astype(np.int64)
        self.boots = {k: J.Boot(np.flatnonzero(games[k]), clus, week) for k in self.tests}
        self.ll = {}
        self.info = {}

    def fit(self, name, Fz, groups, ridge=True):
        des = J.Design(self.d, Fz, self.games, self.gd)
        lam = J.choose_lambda(des, groups, self.games["V1"])[0] if (ridge and groups) else 0.0
        w, Hm, Xm, cols = des.fit(groups, lam, self.games["V1"])
        p = J.predict(w, Xm)
        self.ll[name] = X.logloss(p, des.y)
        self.info[name] = {"lambda": lam, "b_s": float(w[2]), "b_logitWP": float(w[1]),
                           "metrics": {t: C.game_metrics(p[self.games[t]], des.y[self.games[t]]) for t in self.tests}}
        if "WP alone" not in self.ll:
            w0 = C.fit_logistic(des.lo[self.games["V1"]][:, None], des.y[self.games["V1"]])
            self.ll["WP alone"] = X.logloss(C.predict(w0, des.lo[:, None]), des.y)

    def gain(self, ref, new):
        return {t: b.gain(self.ll[ref], self.ll[new]) for t, b in self.boots.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=float, default=None)
    ap.add_argument("--lag", type=int, default=7)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--check-l1", action="store_true", help="rebuild at lag 1 and compare with p3_b_joint")
    a = ap.parse_args()
    t0 = time.time()
    J.FEATS = os.path.join(C.CACHE, "b13_leak_feats.npz")
    d = X.load_ext()
    med, rows, games, gd = J.windows(d, a.sample, a.final)
    n, n_games = len(d["pid"]), gd[0]
    Fz1 = J.load_features(d, a.sample)
    if Fz1 is None:
        Fz1, _ = J.build_features(d, rows, a.sample)
    J.add_nowcast(d, Fz1, rows)
    G_all = dict(J.GROUPS)
    G_nosd = [g for g in G_all if g != "sameday"]
    term_cols = [c for g, cs in G_all.items() if g != "party" for c in cs]
    # variants
    need = ["s"] + term_cols
    Fz1 = {k: Fz1[k] for k in need}
    checks = {}
    if a.check_l1:
        _, checks = build_asof(d, rows, Fz1, lambda dv: dv["day"].astype(np.int64) - 1, "lag 1 (check)")
        log("as-of builder at lag 1 vs p3_b_joint features, max abs diff:", json.dumps(checks))
    Fz7, _ = build_asof(d, rows, None, lambda dv: dv["day"].astype(np.int64) - a.lag, f"lag {a.lag}")
    Fz7["status"] = Fz1["status"]  # the lag check keeps the status term as at lag 1
    Fold, _ = build_asof(d, rows, None, cut_older, "older history")
    Fsh = shuffle_terms(d, Fz1, term_cols, rows["V1"] | rows["V2"] | rows["OOT"])
    FC = dict(Fz1)
    out_c = J.use_skill_C(d, FC)
    FC.pop("s_phase1", None)
    # common game set: every variant has s and every term on all ten slots
    ok = np.ones(n, bool)
    for Fv, cols in ((Fz1, ["s"] + term_cols), (FC, ["s"]), (Fold, ["s"] + [c for g in G_nosd if g != "party" for c in G_all[g]]),
                     (Fz7, ["s"] + [c for g in G_nosd if g != "party" for c in G_all[g]])):
        for c in cols:
            ok &= np.isfinite(Fv[c])
    good = np.bincount(d["g"][ok], minlength=n_games) == 10
    cg = {k: v & good for k, v in games.items()}
    out = {"skill_C_check": out_c, "lag": a.lag, "sample": a.sample, "final": a.final, "lag_check_L1_max_abs_diff": checks,
           "games_before": {k: int(v.sum()) for k, v in games.items()},
           "games_common": {k: int(v.sum()) for k, v in cg.items()}}
    log("common games:", json.dumps(out["games_common"]))
    sc = Scorer(d, cg, gd)
    # same-day term absent from every variant except the lag-1 full joint
    for nm, Fv, grps in (("lag1 headline", Fz1, []), ("lag1 status", Fz1, ["status"]),
                         ("lag1 joint (all terms)", Fz1, list(G_all)), ("lag1 joint (no same-day)", Fz1, G_nosd),
                         ("C headline", FC, []), ("C status", FC, ["status"]),
                         ("C joint (all terms)", FC, list(G_all)), ("C joint (no same-day)", FC, G_nosd),
                         (f"lag{a.lag} headline", Fz7, []), (f"lag{a.lag} status", Fz7, ["status"]),
                         (f"lag{a.lag} joint (no same-day)", Fz7, G_nosd),
                         ("shuffle joint (all terms)", Fsh, list(G_all)),
                         ("shuffle joint (no same-day)", Fsh, G_nosd),
                         ("older headline", Fold, []), ("older status", Fold, ["status"]),
                         ("older joint (no same-day)", Fold, G_nosd)):
        sc.fit(nm, Fv, grps)
        log(f"fit {nm} ({time.time() - t0:.0f}s)")
    comps = {}
    for v in ("lag1", "C", f"lag{a.lag}", "older"):
        comps[f"{v} headline vs WP"] = ("WP alone", f"{v} headline")
        comps[f"{v} status vs headline"] = (f"{v} headline", f"{v} status")
        comps[f"{v} joint (no same-day) vs headline"] = (f"{v} headline", f"{v} joint (no same-day)")
        comps[f"{v} joint (no same-day) vs status"] = (f"{v} status", f"{v} joint (no same-day)")
        comps[f"{v} joint (no same-day) vs lag1 headline"] = ("lag1 headline", f"{v} joint (no same-day)")
    for v in ("lag1", "C"):
        comps[f"{v} joint (all terms) vs headline"] = (f"{v} headline", f"{v} joint (all terms)")
        comps[f"{v} joint (all terms) vs status"] = (f"{v} status", f"{v} joint (all terms)")
    comps["shuffle joint (all terms) vs lag1 headline"] = ("lag1 headline", "shuffle joint (all terms)")
    comps["shuffle joint (no same-day) vs lag1 headline"] = ("lag1 headline", "shuffle joint (no same-day)")
    out["comparisons"] = {}
    for k, (ref, new) in comps.items():
        g = sc.gain(ref, new)
        out["comparisons"][k] = g
        log(f"  {k:52s} " + "  ".join(f"{t} {J.fmt(v)}" for t, v in g.items()))
    out["fits"] = sc.info
    log("b_s:", {k: round(v["b_s"], 2) for k, v in sc.info.items()})
    name = "p3_b_leak" + ("_final" if a.final else "") + ("" if a.sample is None else f"_sample{a.sample}") + ".json"
    with open(os.path.join(C.RESULTS, name), "w") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {os.path.join(C.RESULTS, name)}; done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
