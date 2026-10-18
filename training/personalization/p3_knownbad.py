"""
P3 tails, part 3: "known-bad hero" behavior (descriptive, as-of).

For every V1 and V2 slot (snapshot games, which carry the pick order and
bans) the rich-CF model's as-of estimate for all 90 heroes of the player:
est_h = experience offset (slot-row counts) + posterior mean (state on days
before the game). Definitions, all as-of:
  known-bad hero  30+ earlier games on it (full history) and est < -3pp
  main            the hero with the most earlier games (30+)
  best            the hero with the highest est among heroes with 30+ games
  available       not banned before the player's pick and not picked earlier
                  in the game by anyone
Draft situations of the slot: main banned before the pick (by either team,
by the enemy), main picked earlier by another player, all of the player's
three most-played heroes unavailable, the team's last pick, role fill (the
picked hero's Blizzard role is under 10% of the player's earlier games).

Reports: how often known-bad heroes are queued, by situation; the realized
residual r = y - WP on those picks against the model; the expected cost
against the best available alternative; and the within-player effect of a
main ban on the chance of a known-bad pick. Clustered by player (bootstrap).

Usage (3090, from training/): bash personalization/remote/run_rcf_full.sh is
for p3_rcf.py; run this with the same environment:
  P3_CACHE=... P3_RESULTS=... P3_RCF_ROWS=... python personalization/p3_knownbad.py --tag full_ms
Output: $P3_RESULTS/rcf/knownbad_<tag>.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C
import p3_rcf as R

H = 90
PICK_STEP = np.array([4, 5, 6, 7, 8, 11, 12, 13, 14, 15])


def boot(fn, groups, n=300, seed=0):
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    vals = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[inv].astype(float)
        vals.append(fn(w))
    return float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full_ms")
    ap.add_argument("--hist", default="merged")
    ap.add_argument("--relative", action="store_true",
                    help="known-bad = 3pp+ below the player's own mean over heroes with 30+ games (hero-specific)")
    a = ap.parse_args()
    t0 = time.time()
    import p3_c_tests as T
    import p3_c_merged as M
    from p3_heroes import HKEY
    par = R.get_params(a.tag)
    d, aug = R.load_aug(a.hist)
    w, med = T.windows(d)
    names = [str(x) for x in d["hero_names"]]
    hidx = {h: i for i, h in enumerate(names)}
    # pick order and bans (snapshot games)
    po = np.load(os.path.join(C.CACHE, "pickorder_2024q2.npz"))
    po_h = np.array([hidx.get(str(h), -1) for h in po["hero"]])
    po_key = po["replay_ids"].astype(np.int64) * 128 + po_h
    o = np.argsort(po_key)
    po_key, po_rank, po_rid, po_hh = po_key[o], po["pick_rank"][o], po["replay_ids"][o], po_h[o]
    qi = np.flatnonzero((w == 1) | (w == 2))
    k = d["replay_id"][qi].astype(np.int64) * 128 + d["hero"][qi]
    j = np.minimum(np.searchsorted(po_key, k), len(po_key) - 1)
    has = po_key[j] == k
    qi = qi[has]
    rank = po_rank[j[has]].astype(np.int64)
    print(f"V1+V2 slots with a pick order: {len(qi):,}", flush=True)
    bz = np.load(os.path.join(C.CACHE, "x_bans.npz"))
    bo = np.argsort(bz["replay_ids"])
    brid = bz["replay_ids"][bo]
    jb = np.minimum(np.searchsorted(brid, d["replay_id"][qi]), len(brid) - 1)
    hb = brid[jb] == d["replay_id"][qi]
    ban_h = np.where(hb[:, None], bz["hero"][bo][jb], -1)
    ban_t = np.where(hb[:, None], bz["team"][bo][jb], -1)
    ban_s = np.where(hb[:, None], bz["pick_number"][bo][jb], 99)
    # the walk with all-hero posteriors
    mets, mw = R.aug_metrics(aug, par.meta.get("sb", "legacy")) if par.G is not None else (None, None)
    res = R.online_walk(aug, qi, par, metrics=mets, mweights=mw, want_all=True)
    # slot-row lag-1 counts per hero at the query (for the experience offset)
    day = d["day"]
    npl = int(d["n_players"])
    Ns = np.zeros((npl, H), np.float32)
    nsq = np.empty((len(qi), H), np.float32)
    oq = np.argsort(day[qi], kind="stable")
    ap_ = 0
    qd = day[qi][oq]
    ud, qs = np.unique(qd, return_index=True)
    qe = np.r_[qs[1:], len(qi)]
    for t, s_, e_ in zip(ud, qs, qe):
        e = np.searchsorted(day, t - 1, side="right")
        if e > ap_:
            np.add.at(Ns, (d["pid"][ap_:e], d["hero"][ap_:e]), 1.0)
            ap_ = e
        nsq[oq[s_:e_]] = Ns[d["pid"][qi[oq[s_:e_]]]]
    table = aug["table"]
    npt = nsq.sum(1)
    nb = len(C.NPH_EDGES)
    bp = np.searchsorted(C.NP_EDGES, npt, side="right") - 1
    bh = np.searchsorted(C.NPH_EDGES, nsq, side="right") - 1
    est = table[bp[:, None] * nb + bh] + res["mean_all"]
    nall = res["n_all"]
    n = len(qi)
    ar = np.arange(n)
    hq = d["hero"][qi]
    pid = d["pid"][qi]
    team = d["team"][qi]
    r = d["r"][qi]
    deep = nall >= 30
    if a.relative:
        own = np.where(deep, est, 0).sum(1) / np.maximum(deep.sum(1), 1)
        bad = deep & (est - own[:, None] < -0.03) & (deep.sum(1) >= 3)[:, None]
    else:
        bad = deep & (est < -0.03)
    pick_bad = bad[ar, hq]
    has_bad = bad.any(1)
    main = np.where(deep.any(1), np.argmax(np.where(deep, nall, -1), 1), -1)
    # availability at the player's pick
    step = PICK_STEP[np.clip(rank, 0, 9)]
    unavail = np.zeros((n, H), bool)
    for c in range(6):
        hb_ = ban_h[:, c]
        ok = (hb_ >= 0) & (ban_s[:, c] < step)
        unavail[ar[ok], hb_[ok]] = True
    banned_main = (main >= 0) & unavail[ar, np.maximum(main, 0)]
    enemy_ban_main = np.zeros(n, bool)
    for c in range(6):
        ok = (ban_h[:, c] == main) & (main >= 0) & (ban_s[:, c] < step) & (ban_t[:, c] != team)
        enemy_ban_main |= ok
    # heroes picked earlier in the game by anyone
    g_rid = d["replay_id"][qi].astype(np.int64)
    lo_ = np.searchsorted(po_rid, g_rid, side="left")
    for off in range(10):
        jj = lo_ + off
        ok = (jj < len(po_rid))
        ok[ok] &= (po_rid[jj[ok]] == g_rid[ok])
        ok[ok] &= po_rank[jj[ok]] < rank[ok]
        hh = po_hh[np.minimum(jj, len(po_hh) - 1)]
        ok &= hh >= 0
        unavail[ar[ok], hh[ok]] = True
    taken_main = (main >= 0) & unavail[ar, np.maximum(main, 0)] & ~banned_main
    top3 = np.argsort(-nall, 1)[:, :3]
    top3_out = unavail[ar[:, None], top3].all(1) & (nall[ar[:, None], top3] >= 10).all(1)
    # team's last pick
    gkey = g_rid * 2 + team
    gmax = {}
    ok_ord = np.lexsort((rank, gkey))
    gk_s, rk_s = gkey[ok_ord], rank[ok_ord]
    last = np.empty(n, bool)
    brk = np.r_[np.flatnonzero(gk_s[1:] != gk_s[:-1]), n - 1]
    last_rank = np.repeat(rk_s[brk], np.diff(np.r_[-1, brk]))
    last[ok_ord] = rk_s == last_rank
    # role fill
    meta = C.hero_meta(d["hero_names"])
    role = meta["blizz"]
    role_cnt = np.zeros((n, 6), np.float32)
    for rr in range(6):
        role_cnt[:, rr] = nall[:, role == rr].sum(1)
    tot = nall.sum(1)
    role_fill = (tot >= 50) & (role_cnt[ar, role[hq]] / np.maximum(tot, 1) < 0.10)
    # best available alternative among deep heroes
    alt = np.where(deep & ~unavail, est, -np.inf)
    alt[ar, hq] = -np.inf
    best_alt = alt.max(1)
    est_pick = est[ar, hq]
    out = {"tag": a.tag, "slots": int(n), "players": int(len(np.unique(pid))),
           "definition": ("30+ earlier games on the hero and estimate 3pp+ below the player's mean over their "
                          "30+ game heroes (3+ such heroes)" if a.relative else
                          "30+ earlier games on the hero (full history) and estimate < -3pp") + ", as of the day before"}

    def rate(mask, sel=None, groups=pid):
        sel = np.ones(n, bool) if sel is None else sel
        m_ = mask[sel].astype(float)
        g_ = groups[sel]
        if sel.sum() == 0:
            return None
        return {"n": int(sel.sum()), "rate": float(m_.mean()),
                "se": boot(lambda ww: np.average(m_, weights=ww), g_, n=200)}
    out["share_of_slots_known_bad_pick"] = rate(pick_bad)
    out["share_of_slots_player_has_known_bad"] = rate(has_bad)
    out["known_bad_pick_rate_given_player_has_one"] = rate(pick_bad, has_bad)
    out["mean_known_bad_heroes_when_any"] = float(bad.sum(1)[has_bad].mean())
    sit = {"main banned before the pick": banned_main, "main banned by the enemy": enemy_ban_main,
           "main picked earlier by another player": taken_main, "three most-played heroes unavailable": top3_out,
           "team's last pick": last, "role fill (<10% of games in the role)": role_fill}
    out["by_situation"] = {}
    for nm, m_ in sit.items():
        out["by_situation"][nm] = {"with": rate(pick_bad, has_bad & m_), "without": rate(pick_bad, has_bad & ~m_),
                                   "share_of_known_bad_picks_in_situation": float(m_[pick_bad].mean())}
    # realized cost on known-bad picks
    kb = pick_bad
    out["realized"] = {
        "known_bad_picks": int(kb.sum()),
        "realized_r_pp": 100 * float(r[kb].mean()),
        "realized_r_se_pp": 100 * boot(lambda ww: np.average(r[kb], weights=ww), pid[kb]),
        "predicted_est_pp": 100 * float(est_pick[kb].mean()),
        "realized_minus_predicted_pp": 100 * float((r[kb] - est_pick[kb]).mean()),
        "same_players_other_picks_r_pp": 100 * float(r[np.isin(pid, pid[kb]) & ~kb].mean()),
        "best_available_alternative_est_pp": 100 * float(best_alt[kb & np.isfinite(best_alt)].mean()),
        "expected_cost_vs_best_available_pp": 100 * float((best_alt - est_pick)[kb & np.isfinite(best_alt)].mean()),
        "share_with_a_better_available_deep_hero": float((best_alt > est_pick)[kb].mean()),
    }
    # main-ban effect, within player: players with a known-bad hero and a main,
    # with games both with and without a main ban before their pick
    elig = has_bad & (main >= 0)
    up, inv = np.unique(pid[elig], return_inverse=True)
    mb = banned_main[elig]
    pb = pick_bad[elig].astype(float)
    n1 = np.bincount(inv, weights=mb, minlength=len(up))
    n0 = np.bincount(inv, weights=~mb, minlength=len(up))
    s1 = np.bincount(inv, weights=pb * mb, minlength=len(up))
    s0 = np.bincount(inv, weights=pb * ~mb, minlength=len(up))
    both = (n1 > 0) & (n0 > 0)
    wts = 1 / (1 / np.maximum(n1, 1) + 1 / np.maximum(n0, 1))
    diff = s1 / np.maximum(n1, 1) - s0 / np.maximum(n0, 1)
    rng = np.random.default_rng(1)
    bs = []
    ib = np.flatnonzero(both)
    for _ in range(400):
        kk = rng.choice(ib, len(ib))
        bs.append(np.average(diff[kk], weights=wts[kk]))
    out["main_ban_effect_within_player"] = {
        "players": int(both.sum()), "slots_main_banned": int(n1[both].sum()), "slots_not": int(n0[both].sum()),
        "rate_main_banned": float(s1[both].sum() / n1[both].sum()), "rate_not_banned": float(s0[both].sum() / n0[both].sum()),
        "within_player_diff": float(np.average(diff[both], weights=wts[both])), "se": float(np.std(bs))}
    # realized residual of the player in games where the main was banned (within player)
    rr_ = r[elig]
    t1 = np.bincount(inv, weights=rr_ * mb, minlength=len(up))
    t0_ = np.bincount(inv, weights=rr_ * ~mb, minlength=len(up))
    dr = t1 / np.maximum(n1, 1) - t0_ / np.maximum(n0, 1)
    bs = [np.average(dr[kk], weights=wts[kk]) for kk in (rng.choice(ib, len(ib)) for _ in range(400))]
    out["main_ban_effect_within_player"]["realized_r_diff_pp"] = 100 * float(np.average(dr[both], weights=wts[both]))
    out["main_ban_effect_within_player"]["realized_r_diff_se_pp"] = 100 * float(np.std(bs))
    # how known-bad picks distribute over pick rank and roles
    out["known_bad_pick_rate_by_pick_rank"] = {int(k_): float(pick_bad[has_bad & (rank == k_)].mean())
                                               for k_ in range(10) if (has_bad & (rank == k_)).sum() > 1000}
    out["known_bad_pick_rate_by_role"] = {C.BLIZZ_ROLES[rr]: float(pick_bad[has_bad & (role[hq] == rr)].mean())
                                          for rr in range(6)}
    out["seconds"] = round(time.time() - t0)
    p = os.path.join(R.RCF_RES, f"knownbad_{a.tag}{'_rel' if a.relative else ''}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
