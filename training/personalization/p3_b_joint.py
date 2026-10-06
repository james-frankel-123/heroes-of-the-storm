"""
P3 review item 1 (REVIEW section 2.2): one joint combiner with every term
that was tested alone on top of the headline skill model.

Game-level model (logistic, fit on V1 games only):
  logit P(team 0 wins) = b0 + b1 logit(WP_pop) + b2 D[s] + sum_k c_k D[f_k]
where D[x] is the team-0 minus team-1 sum of a per-slot value and s is the
headline personal term (lag-1 experience offset with the table refit on
lag-1 counts in E, plus the "+CF rank 2" GP posterior mean, online, state
from games of days <= t - 1). Terms f_k, grouped as tested alone:
  status   new-account status: the experience table split by causal
           account status (p3_hero_level_causal.account_status, fit on E as
           in p3_x_newacct "4 statuses") minus the phase-1 table, at the
           row's counts (offset correction; the GP state is not rebuilt, so
           the term does not compete with s)
  main     main-share terms (p3_x_ban_model slot design): share of the
           player's games on this hero, [hero is the main], main share x
           [hero is not the main], main share x [main unavailable: banned or
           taken in this game] (forced off main)
  role     fine-role familiarity (p3_ph_role): off-role indicator (50+
           games, fine-role share < 10%) and log smoothed fine-role share
  ewma100  EWMA residual (y - WP), half-life 100 games (p3_sd_eval)
  rust     log(1 + days since this hero was last played) for played heroes,
           log(1 + days since the player's last game) (p3_x_learn)
  party    partied players and largest party, team differences (p3_val_validity)
  mmr      causal hero MMR at game time / 100 (cache/mmr_at_game.npz; missing
           falls back to role, player, then the mean; p3_fix_mmr)
  nowcast  (build, hero) nowcast of p3_b_nowcast / p3_b_demean: shrunken mean
           of r_adj over the same build and hero on earlier days of that
           build (lag 1), k = noise / (m tau2_E), default m = 2 (chosen on a
           V1-internal split by p3_b_nowcast.py); computed on the snapshot
           rows for V1/V2 and on all rows for OOT
  sameday  strict same-day form: earlier games of the same day that ended
           before this game started and have a lower replay_id (the strict
           rule of p3_sd_order); shrunk mean of their residuals r - s and
           log(1 + their count). Same-day information, so the joint model
           is reported with and without it.
Every count, share, EWMA and day gap uses games of earlier days only (lag 1
day), except sameday (strict rule) and the causal MMR (latest stamp parsed
before the game started). Party, bans and picks are draft-time information
of the game itself, as is the WP.

Protocol (as p3_x_oot): V1/V2 rows use state and features from snapshot rows
only; OOT rows (post-snapshot, day > 2026-05-22) use every row up to day - 1.
Combiner fit on V1, scored on V2 and OOT. Term columns are standardized on
V1 and carry a ridge penalty (logit(WP) and s are unpenalized); the penalty
is chosen on V1 alone, by a time split inside V1 (first two thirds of V1 days
fit, last third validate). Reported: gains over the headline and over WP
alone, each group's solo gain and leave-one-out marginal gain in the joint
fit, coefficients, calibration slope and ECE, the skill coefficient refit on
V2 and on OOT (gate 3.4 to 4.0), replay-bootstrap CIs (raw and widened 1.3x)
and a player-cluster bootstrap (first team-0 player of the game).

Skill term: --skill phase1 (headline) or --skill C (the concentration-mean
variant of p3_b_onetrick.py, read from cache/b7_onetrick_preds.npz: same
offset, GP mean with the one-trick prior mean). The status term stays
s4 - s(phase-1) in both runs.

Usage (from training/): python personalization/p3_b_joint.py [--sample 0.1] [--skill C] [--final]
Outputs: results/p3_b_joint.json; cache/b13_joint_feats.npz (per-row
features of the query rows, reused by p3_b_form.py)
"""
import argparse
import gzip
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
import p3_hero_level_causal as HL
from p3_heroes import NUM_HEROES, HKEY
from p3_mmr_at_game import match
from p3_sd_eval import momentum_features

KNAME = "+CF rank 2"
FEATS = os.path.join(C.CACHE, "b13_joint_feats.npz")
HL_FALLBACK = os.path.join(C.CACHE, "b13_hl_{}.npz")
GROUPS = {
    "status": ["status"],
    "nowcast": ["nowcast"],
    "main": ["share_hero", "is_main", "offmain_ms", "forced_ms"],
    "role": ["offrole_fine", "log_share_fine"],
    "ewma100": ["ewma100"],
    "rust": ["rust_hero", "log_gap_any"],
    "party": ["partied", "largest_party"],
    "mmr": ["hero_mmr"],
    "sameday": ["sd_mean", "sd_logk"],
}
LAG1_GROUPS = [g for g in GROUPS if g != "sameday"]
LAMBDAS = [0.0, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ data

def subset(d, mask):
    n = len(d["pid"])
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v)
            for k, v in d.items()}


SEALED_BUILD = "2.55.17.98025"


def windows(d, sample=None, final=False):
    """Row masks V1, V2, OOT and game masks (complete games only). "OOT" is
    the post-snapshot games before the sealed build 2.55.17.98025; the sealed
    games enter a game mask only with final=True (their rows still get
    features; no outcome of theirs is used otherwise)."""
    day, post = d["day"], d["post"]
    snap_post = (~post) & (~d["in_sample"])
    _, fr = np.unique(d["g"][snap_post], return_index=True)
    med = float(np.median(day[snap_post][fr]))
    rows = {"V1": snap_post & (day < med), "V2": snap_post & (day >= med),
            "OOT": post & (day > X.SNAP_LAST_DAY)}
    n_games, y, wp0, cnt, gday = X.game_arrays(d)
    keep = np.ones(n_games, bool)
    if sample is not None and sample < 1:
        keep = (np.arange(n_games, dtype=np.int64) * 2654435761 % 1000) < int(1000 * sample)
    vers = [str(v) for v in d["versions"]]
    gver = np.full(n_games, -1)
    gver[d["g"]] = d["version"]
    sealed = (gver == vers.index(SEALED_BUILD)) if SEALED_BUILD in vers else np.zeros(n_games, bool)
    games = {}
    for k, m in rows.items():
        gm = np.zeros(n_games, bool)
        gm[d["g"][m]] = True
        games[k] = gm & (cnt == 10) & keep
        rows[k] = m & keep[d["g"]]
    if final:
        games["OOT 98025 (sealed)"] = games["OOT"] & sealed
        games["OOT all builds"] = games["OOT"].copy()
    games["OOT"] = games["OOT"] & ~sealed
    log(f"sealed build {SEALED_BUILD}: {int((sealed & (cnt == 10)).sum()):,} games "
        f"({'scored, final run' if final else 'not scored'})")
    return med, rows, games, (n_games, y, wp0, cnt, gday)


def game_times(d):
    """End and start time (epoch s) per row: snapshot games from
    gametime_2024q2.npz, post-snapshot games from x_post_games.json.gz.
    Also returns the draft of post games from their draft_order: bans
    (hero index, pick_number) per game and picks (replay_id, hero, rank
    0..9 among the game's picks)."""
    gt = np.load(os.path.join(C.CACHE, "gametime_2024q2.npz"))
    rid_s = gt["replay_ids"]
    o = np.argsort(rid_s)
    rid_s, ts_s, gl_s = rid_s[o], gt["ts"][o].astype(np.float64), gt["game_length"][o].astype(np.float64)
    names = {str(h): i for i, h in enumerate(d["hero_names"])}
    with gzip.open(os.path.join(C.CACHE, "x_post_games.json.gz"), "rt") as f:
        games = json.load(f)
    rid_p = np.array([g["replay_id"] for g in games], np.int64)
    ts_p = np.array([g["ts"] if g["ts"] is not None else np.nan for g in games], np.float64)
    gl_p = np.array([g["game_length"] if g["game_length"] is not None else 0 for g in games], np.float64)
    bans_p = np.full((len(games), 6), -1, np.int64)
    bpn_p = np.full((len(games), 6), -1, np.int64)
    pk_r, pk_h, pk_k = [], [], []
    for i, g in enumerate(games):
        do = sorted(g.get("draft_order") or [], key=lambda e: e[2])
        j = 0
        for e in do:
            if e[1] == 0 and j < 6:
                bans_p[i, j] = names.get(str(e[0]), -1)
                bpn_p[i, j] = e[2]
                j += 1
        for k, e in enumerate([e for e in do if e[1] == 1]):
            pk_r.append(g["replay_id"])
            pk_h.append(names.get(str(e[0]), -1))
            pk_k.append(k)
    del games
    o = np.argsort(rid_p)
    rid_p, ts_p, gl_p, bans_p, bpn_p = rid_p[o], ts_p[o], gl_p[o], bans_p[o], bpn_p[o]
    post_draft = {"replay_ids": rid_p, "ban_hero": bans_p, "ban_pn": bpn_p,
                  "pick_rid": np.array(pk_r, np.int64), "pick_hero": np.array(pk_h, np.int64),
                  "pick_rank": np.array(pk_k, np.int64)}
    rid = d["replay_id"]
    t_end = np.full(len(rid), np.nan)
    gl = np.zeros(len(rid))
    for r_, t_, l_ in ((rid_s, ts_s, gl_s), (rid_p, ts_p, gl_p)):
        j = np.minimum(np.searchsorted(r_, rid), len(r_) - 1)
        ok = (r_[j] == rid) & np.isnan(t_end)
        t_end[ok] = t_[j[ok]]
        gl[ok] = l_[j[ok]]
    log(f"game times matched for {np.isfinite(t_end).mean():.4f} of rows")
    t_end = np.where(np.isfinite(t_end), t_end, d["day"].astype(np.float64) * 86400 + 43200)
    t_start = t_end - np.where(gl > 0, gl, 1200.0)
    return t_end, t_start, post_draft


def draft_info(d, post_draft):
    """Bans per game index ((n_games, 6) hero, and whether the ban is in the
    second ban phase: pick_number >= 5, the p3_x_ban_nat rule) and each
    slot's pick rank (0..9 among the game's picks, -1 unknown).
    Snapshot games: x_bans.npz and pickorder_2024q2.npz; post games: their
    draft_order."""
    n_games = int(d["g"].max()) + 1
    grid = np.zeros(n_games, np.int64)
    grid[d["g"]] = d["replay_id"]
    ban_h = np.full((n_games, 6), -1, np.int64)
    ban_pn = np.full((n_games, 6), -1, np.int64)
    B = np.load(os.path.join(C.CACHE, "x_bans.npz"))
    for r_, h_, k_ in ((B["replay_ids"], B["hero"].astype(np.int64), B["pick_number"].astype(np.int64)),
                       (post_draft["replay_ids"], post_draft["ban_hero"], post_draft["ban_pn"])):
        o = np.argsort(r_)
        rs = r_[o]
        j = np.minimum(np.searchsorted(rs, grid), len(rs) - 1)
        ok = (rs[j] == grid) & (ban_h < 0).all(1)
        ban_h[ok] = h_[o][j[ok]]
        ban_pn[ok] = k_[o][j[ok]]
    po = np.load(os.path.join(C.CACHE, "pickorder_2024q2.npz"))
    names = {str(h): i for i, h in enumerate(d["hero_names"])}
    ph = np.array([names.get(str(h), -1) for h in po["hero"]], np.int64)
    pkey = np.r_[po["replay_ids"].astype(np.int64) * HKEY + ph,
                 post_draft["pick_rid"] * HKEY + post_draft["pick_hero"]]
    prank = np.r_[po["pick_rank"].astype(np.int64), post_draft["pick_rank"]]
    pkey, ui = np.unique(pkey, return_index=True)
    prank = prank[ui]
    key = d["replay_id"].astype(np.int64) * HKEY + d["hero"]
    j = np.minimum(np.searchsorted(pkey, key), len(pkey) - 1)
    rank = np.where(pkey[j] == key, prank[j], -1)
    log(f"games with ban data {(ban_h >= 0).any(1).mean():.4f}; slots with a pick rank {(rank >= 0).mean():.4f}")
    return {"ban_h": ban_h, "ban_second": ban_pn >= 5, "rank": rank}


# ------------------------------------------------------------------ features

@njit(cache=True)
def _main_walk(starts, ends, day, hero, o_tot, o_top, o_main, o_nown):
    """Per row, from the player's games on earlier days: total games, games
    on the most played hero, that hero (-1 if none), games on this hero."""
    for p in range(starts.shape[0]):
        cnt = np.zeros(NUM_HEROES)
        i = starts[p]
        while i < ends[p]:
            j = i
            while j < ends[p] and day[j] == day[i]:
                j += 1
            tot = 0.0
            mx = 0.0
            mh = -1
            for h in range(NUM_HEROES):
                tot += cnt[h]
                if cnt[h] > mx:
                    mx = cnt[h]
                    mh = h
            for q in range(i, j):
                o_tot[q] = tot
                o_top[q] = mx
                o_main[q] = mh
                o_nown[q] = cnt[hero[q]]
            for q in range(i, j):
                cnt[hero[q]] += 1.0
            i = j


@njit(cache=True)
def _sameday_walk(starts, ends, day, t_end, t_start, rid, e, valid, o_sum, o_k):
    """Strict rule: an earlier same-day game is visible to game q only if it
    ended before q started and has a lower replay_id."""
    for p in range(starts.shape[0]):
        i = starts[p]
        while i < ends[p]:
            j = i
            while j < ends[p] and day[j] == day[i]:
                j += 1
            for q in range(i, j):
                s = 0.0
                k = 0
                for r in range(i, j):
                    if r != q and valid[r] and t_end[r] < t_start[q] and rid[r] < rid[q]:
                        s += e[r]
                        k += 1
                o_sum[q] = s
                o_k[q] = k
            i = j


def player_blocks(dv, *second):
    keys = tuple(second[::-1]) + (dv["day"], dv["pid"])
    o = np.lexsort(keys)
    pid = dv["pid"][o]
    brk = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    ends = np.r_[brk[1:], len(pid)]
    return o, brk.astype(np.int64), ends.astype(np.int64)


def main_terms(dv, dr):
    """Main-share terms from earlier-day history. Forced off main: the main
    (30+ earlier games) was unavailable in the real draft prefix at the
    player's own pick: banned in the first ban phase, banned in the second
    phase when the player picks 6th or later (pick rank >= 5), or picked by
    another player earlier in the draft. The p3_x_ban_nat flag (main banned
    or taken anywhere in the draft) is kept as _unavailable_any for the
    comparison."""
    o, st, en = player_blocks(dv, dv["replay_id"])
    n = len(o)
    tot, top, nown = np.zeros(n), np.zeros(n), np.zeros(n)
    mh = np.zeros(n, np.int64)
    _main_walk(st, en, dv["day"][o].astype(np.int64), dv["hero"][o].astype(np.int64), tot, top, mh, nown)
    res = {}
    for k, a in (("tot", tot), ("top", top), ("main", mh), ("nown", nown)):
        b = np.empty_like(a)
        b[o] = a
        res[k] = b
    hero, g = dv["hero"], dv["g"]
    main = res["main"]
    ms = res["top"] / np.maximum(res["tot"], 1)
    is_main = (hero == main) & (main >= 0)
    rank = dv["pick_rank"]
    banned_any = np.zeros(n, bool)
    banned_before = np.zeros(n, bool)
    for k in range(6):
        hit = (dr["ban_h"][g, k] == main) & (main >= 0)
        banned_any |= hit
        banned_before |= hit & (~dr["ban_second"][g, k] | (rank >= 5))
    gk = g.astype(np.int64) * HKEY + hero
    og = np.argsort(gk)
    sgk = gk[og]
    srank = rank[og]
    mk = g.astype(np.int64) * HKEY + np.maximum(main, 0)
    jj = np.minimum(np.searchsorted(sgk, mk), len(sgk) - 1)
    taken = (sgk[jj] == mk) & ~is_main & (main >= 0)
    taken_before = taken & (srank[jj] >= 0) & (rank >= 0) & (srank[jj] < rank)
    elig = (res["tot"] >= 30) & ~is_main
    unavailable = elig & (banned_before | taken_before)
    return {"share_hero": res["nown"] / np.maximum(res["tot"], 1), "is_main": is_main.astype(float),
            "offmain_ms": ms * (~is_main), "forced_ms": ms * unavailable,
            "_unavailable": unavailable, "_unavailable_any": elig & (banned_any | taken),
            "_ms": ms, "_tot": res["tot"]}


def role_terms(dv, meta, e_mask):
    fine = meta["fine"][dv["hero"]]
    n_p = F.lag1_key_counts(dv, dv["pid"])
    n_pf = F.lag1_key_counts(dv, dv["pid"] * 16 + fine)
    share_f = n_pf / np.maximum(n_p, 1)
    base = np.bincount(fine[e_mask], minlength=len(meta["fine_names"])) / max(e_mask.sum(), 1)
    sm = (n_pf + 10 * base[fine]) / (n_p + 10)
    return {"offrole_fine": ((n_p >= 50) & (share_f < 0.10)).astype(float), "log_share_fine": np.log(sm),
            "_n_p": n_p}


def momentum_terms(dv):
    mf = momentum_features(dv)
    dsh, dsa = mf["days since hero"], mf["days since any"]
    return {"ewma100": mf["EWMA resid hl100"], "_ewma10": mf["EWMA resid hl10"],
            "_ewma30": mf["EWMA resid hl30"],
            "rust_hero": np.log1p(np.minimum(np.maximum(dsh, 0), 365)) * (dsh >= 0),
            "log_gap_any": np.log1p(np.minimum(np.maximum(dsa, 0), 365)) * (dsa >= 0),
            "_days_since_any": dsa}


def sameday_terms(dv, s, t_end, t_start):
    e = dv["r"] - s
    valid = np.isfinite(e)
    o, st, en = player_blocks(dv, t_end, dv["replay_id"])
    n = len(o)
    ssum, k = np.zeros(n), np.zeros(n)
    _sameday_walk(st, en, dv["day"][o].astype(np.int64), t_end[o], t_start[o],
                  dv["replay_id"][o].astype(np.int64), np.where(valid, e, 0.0)[o], valid[o], ssum, k)
    a, b = np.empty(n), np.empty(n)
    a[o], b[o] = ssum, k
    return {"sd_mean": a / (b + 3.0), "sd_logk": np.log1p(b), "_sd_k": b}


def party_terms(d, n_games):
    party = d["party"]
    gteam = d["g"].astype(np.int64) * 2 + d["team"]
    nz = party != 0
    pk = np.where(nz, (gteam << 32) ^ (party.astype(np.int64) % 2147483647), -1)
    sz = np.ones(len(party), np.int64)
    _, inv, c = np.unique(pk[nz], return_inverse=True, return_counts=True)
    sz[nz] = c[inv]
    sign = np.where(d["team"] == 0, 1.0, -1.0)
    partied = np.bincount(d["g"], weights=sign * (sz >= 2), minlength=n_games)
    big = np.zeros(2 * n_games)
    np.maximum.at(big, gteam, sz.astype(float))
    return {"partied": partied, "largest_party": big[0::2] - big[1::2]}


def causal_mmr(d):
    """Causal at-game MMR (player, role, hero) per row, NaN if unknown."""
    z = np.load(os.path.join(C.CACHE, "mmr_at_game.npz"))
    zkey = z["region"].astype(np.int64) << 40 | z["blizz_ids"]
    idx = match(d["player_keys"][d["pid"]], d["replay_id"], zkey, z["replay_ids"])
    ok = idx >= 0
    gi = np.maximum(idx, 0)
    out = {k: np.where(ok, z[f"{k}_mmr_causal"][gi].astype(np.float64), np.nan)
           for k in ("player", "role", "hero")}
    out["player_age_s"] = np.where(ok, z["player_mmr_causal_age_s"][gi], -1)
    log(f"mmr_at_game rows matched {ok.mean():.4f}; causal player MMR known "
        f"{np.isfinite(out['player']).mean():.4f}, hero {np.isfinite(out['hero']).mean():.4f}")
    return out


def hero_mmr_term(M):
    gm = np.nanmean(M["player"])
    pl = np.where(np.isnan(M["player"]), gm, M["player"])
    ro = np.where(np.isnan(M["role"]), pl, M["role"])
    return np.where(np.isnan(M["hero"]), ro, M["hero"]) / 100.0


def hl_arrays(d, which, t_start):
    """Causal account-level arrays for the status: the owner's
    hero_level_causal_*.npz when it is aligned to d, else built here from
    cache/export/<tag>/stamps.npz with p3_hero_level_causal's functions
    (cached as b13_hl_*.npz)."""
    try:
        return HL.load(d, which)
    except (AssertionError, FileNotFoundError, KeyError) as ex:
        log(f"hero_level_causal_{which}: {str(ex)[:120]}; building an aligned copy")
    path = HL_FALLBACK.format(which)
    if os.path.exists(path):
        z = np.load(path)
        if np.array_equal(z["replay_id"], d["replay_id"]) and np.array_equal(z["pid"], d["pid"]):
            return {k: z[k].astype(np.float64) for k in ("acct_lo", "acct_hi")}
    exp_dir = os.path.join(C.CACHE, "export")
    tags = sorted(t for t in os.listdir(exp_dir) if os.path.exists(os.path.join(exp_dir, t, "stamps.npz")))
    z = np.load(os.path.join(exp_dir, tags[-1], "stamps.npz"))
    from p3_keys import player_keys
    skey = player_keys(z["region"], z["blizz_ids"], label="stamps")
    pkey = d["player_keys"][d["pid"]]
    j = match(pkey, d["replay_id"], skey, z["replay_ids"])
    start = z["end_ts"] - np.where(z["game_length"] > 0, z["game_length"], 3600)
    q_start = np.where(j >= 0, start[np.maximum(j, 0)], t_start.astype(np.int64))
    lo = HL.running_max_before(skey, z["parse_ts"], z["lo"].astype(np.float64), pkey, q_start)
    hi = HL.running_max_before(skey, z["parse_ts"], z["hi"].astype(np.float64), pkey, q_start)
    log(f"built causal account levels from export {tags[-1]}: own stamp matched {np.mean(j >= 0):.4f}, "
        f"level known {np.isfinite(lo).mean():.4f}")
    np.savez(path, replay_id=d["replay_id"], pid=d["pid"], acct_lo=lo.astype(np.float32),
             acct_hi=hi.astype(np.float32))
    return {"acct_lo": lo, "acct_hi": hi}


# ------------------------------------------------------------------ skill terms

def skill_terms(d, rows, K, status):
    """Headline s = table1[bin] + GP mean, and s4 (status-split table and GP
    state rebuilt on its residuals), for V rows (snapshot state) and OOT rows
    (all rows), as in p3_x_oot / p3_x_newacct."""
    e_mask, n_p, n_ph, table1, r_adj1 = F.prepare_l1(d)
    b = C.exp_bins(n_p, n_ph)
    nb = len(C.NP_EDGES) * len(C.NPH_EDGES)
    key = status * nb + b
    ssum = np.bincount(key[e_mask], weights=d["r"][e_mask], minlength=4 * nb)
    scnt = np.bincount(key[e_mask], minlength=4 * nb)
    table4 = ssum / (scnt + 200.0)
    r_adj4 = d["r"] - table4[key]
    n = len(d["pid"])
    out = {k: np.full(n, np.nan) for k in ("s", "s4", "mu", "mu4", "gp_var", "Nh", "Np")}
    for q, add in ((rows["V1"] | rows["V2"], ~d["post"]), (rows["OOT"], np.ones(n, bool))):
        qi, m, v, Nh, Np, _ = X.online_predict(d, add, q, K, r_adj1)
        bl = C.exp_bins(Np.astype(np.int64), Nh.astype(np.int64))
        out["mu"][qi] = table1[bl]
        out["s"][qi] = table1[bl] + m
        out["gp_var"][qi] = v
        out["Nh"][qi], out["Np"][qi] = Nh, Np
        qi4, m4, _, _, _, _ = X.online_predict(d, add, q, K, r_adj4)
        assert np.array_equal(qi, qi4)
        out["mu4"][qi] = table4[status[qi] * nb + bl]
        out["s4"][qi] = out["mu4"][qi] + m4
    # offset correction only: s4 - s is nearly collinear with s and splits
    # the skill coefficient; s4 - s is kept as _status_full for reference
    out["status"] = out["mu4"] - out["mu"]
    out["_status_full"] = out["s4"] - out["s"]
    tabs = {"table1_pp": (100 * table1).round(2).tolist(),
            "table4_pp": (100 * table4.reshape(4, -1)).round(2).tolist()}
    return out, tabs


# ------------------------------------------------------------------ build all

def build_features(d, rows, sample=None):
    """Per-row features for the query rows (V1, V2, OOT); NaN elsewhere."""
    t0 = time.time()
    n = len(d["pid"])
    Ks, _ = X.kernels()
    K = Ks[KNAME]
    meta = C.hero_meta(d["hero_names"])
    t_end, t_start, post_draft = game_times(d)
    hl = hl_arrays(d, "ext", t_start)
    status = HL.account_status(d, hl)
    Fz, tabs = skill_terms(d, rows, K, status)
    Fz["acct_status"] = status.astype(float)
    log(f"skill terms: {time.time() - t0:.0f}s")
    dr = draft_info(d, post_draft)
    d["pick_rank"] = dr["rank"]
    e_mask = d["in_sample"] & (d["day"] >= C.day_of(C.E_START))
    qmask = rows["V1"] | rows["V2"] | rows["OOT"]
    for view, qm in (("snapshot", rows["V1"] | rows["V2"]), ("all", rows["OOT"])):
        vm = ~d["post"] if view == "snapshot" else np.ones(n, bool)
        dv = subset(d, vm)
        sel = qm[vm]
        dest = np.flatnonzero(vm)[sel]
        parts = [main_terms(dv, dr),role_terms(dv, meta, e_mask[vm]), momentum_terms(dv),
                 sameday_terms(dv, Fz["s"][vm], t_end[vm], t_start[vm])]
        for p in parts:
            for k, v in p.items():
                a = Fz.setdefault(k, np.full(n, np.nan))
                a[dest] = np.asarray(v, np.float64)[sel]
        del dv, parts
        log(f"walk features ({view} view): {time.time() - t0:.0f}s")
    M = causal_mmr(d)
    Fz["hero_mmr"] = np.where(qmask, hero_mmr_term(M), np.nan)
    Fz["player_mmr_causal"] = np.where(qmask, M["player"], np.nan)
    Fz["player_mmr_age_s"] = np.where(qmask, M["player_age_s"], -1).astype(np.float64)
    Fz["t_start"] = t_start
    qi = np.flatnonzero(qmask)
    save = {k: v[qi].astype(np.float32) for k, v in Fz.items() if k != "t_start"}
    save["t_start"] = t_start[qi]
    np.savez(FEATS + ".tmp.npz", rows=qi, replay_id=d["replay_id"][qi], pid=d["pid"][qi],
             sample=np.float64(-1 if sample is None else sample), **save)
    os.replace(FEATS + ".tmp.npz", FEATS)
    log(f"features built and cached ({time.time() - t0:.0f}s)")
    return Fz, tabs


def load_features(d, sample=None):
    if not os.path.exists(FEATS):
        return None
    z = np.load(FEATS)
    if float(z["sample"]) != (-1 if sample is None else sample):
        return None
    qi = z["rows"]
    if not (np.array_equal(z["replay_id"], d["replay_id"][qi]) and np.array_equal(z["pid"], d["pid"][qi])):
        return None
    n = len(d["pid"])
    out = {}
    for k in z.files:
        if k in ("rows", "replay_id", "pid", "sample"):
            continue
        a = np.full(n, np.nan)
        a[qi] = z[k]
        out[k] = a
    return out


# ------------------------------------------------------------------ combiner

def fit_ridge(Xm, y, pen, iters=60):
    """Logistic regression with a per-column L2 penalty (no penalty on the
    intercept)."""
    X1 = np.column_stack([np.ones(len(Xm)), Xm])
    P = np.diag(np.r_[0.0, pen]) + 1e-9 * np.eye(X1.shape[1])
    w = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ w))
        g = X1.T @ (y - p) - P @ w
        Hm = (X1 * (p * (1 - p))[:, None]).T @ X1 + P
        step = np.linalg.solve(Hm, g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w, Hm


def predict(w, Xm):
    return 1 / (1 + np.exp(-(np.column_stack([np.ones(len(Xm)), Xm]) @ w)))


class Design:
    """Game-level design: logit WP, D[s], then standardized term columns."""

    def __init__(self, d, Fz, games, gd):
        n_games, y, wp0, cnt, gday = gd
        self.y, self.gday = y, gday
        self.lo = np.log(wp0 / (1 - wp0))
        fill = lambda a: np.where(np.isfinite(a), a, 0.0)
        self.D = {"s": X.team_diff(fill(Fz["s"]), d["g"], d["team"], n_games)}
        for grp, cols in GROUPS.items():
            for c in cols:
                if grp == "party":
                    continue
                self.D[c] = X.team_diff(fill(Fz[c]), d["g"], d["team"], n_games)
        self.D.update(party_terms(d, n_games))
        fit = games["V1"]
        self.sd = {k: float(v[fit].std()) or 1.0 for k, v in self.D.items()}

    def matrix(self, groups, skill=True):
        cols = (["s"] if skill else []) + [c for g in groups for c in GROUPS[g]]
        Xm = np.column_stack([self.lo] + [self.D[c] if c == "s" else self.D[c] / self.sd[c] for c in cols])
        pen_mask = np.array([0.0] + [0.0 if c == "s" else 1.0 for c in cols])
        return Xm, pen_mask, cols

    def fit(self, groups, lam, mask, skill=True):
        Xm, pm, cols = self.matrix(groups, skill)
        w, Hm = fit_ridge(Xm[mask], self.y[mask], lam * pm)
        return w, Hm, Xm, cols

    def coef_table(self, w, Hm, cols):
        se = np.sqrt(np.diag(np.linalg.inv(Hm)))
        names = ["intercept", "logit WP"] + cols
        scale = [1.0, 1.0] + [1.0 if c == "s" else 1.0 / self.sd[c] for c in cols]
        return {nm: {"coef": float(w[i] * scale[i]), "se_x1.3": float(1.3 * se[i] * scale[i]),
                     "coef_std": float(w[i])} for i, nm in enumerate(names)}


def choose_lambda(des, groups, fit_games):
    """Ridge penalty by a time split inside V1 only."""
    days = des.gday[fit_games]
    cut = np.percentile(days, 200 / 3)
    a = fit_games & (des.gday <= cut)
    b = fit_games & (des.gday > cut)
    res = {}
    for lam in LAMBDAS:
        w, _, Xm, _ = des.fit(groups, lam, a)
        res[lam] = float(X.logloss(predict(w, Xm[b]), des.y[b]).mean())
    best = min(res, key=res.get)
    return best, {str(k): v for k, v in res.items()}, float(cut)


def clusters_first_player(d, n_games):
    first = np.full(n_games, -1, np.int64)
    t0 = np.flatnonzero(d["team"] == 0)[::-1]
    first[d["g"][t0]] = d["pid"][t0]
    return first


class Boot:
    """Paired per-game log-loss differences, resampled three ways: by game
    (replay), by player cluster (first team-0 player of the game) and by
    calendar week (block bootstrap, a sensitivity check)."""

    def __init__(self, idx, clus, week, B=200, seed=0):
        rng = np.random.RandomState(seed)
        self.idx = idx
        n = len(idx)
        self.W = {"replay": [np.bincount(rng.randint(0, n, n), minlength=n).astype(float) for _ in range(B)]}
        for nm, c in (("player_cluster", clus), ("week_block", week)):
            u, inv = np.unique(c[idx], return_inverse=True)
            self.W[nm] = [np.bincount(rng.randint(0, len(u), len(u)), minlength=len(u))[inv].astype(float)
                          for _ in range(B)]
        self.n_weeks = int(len(np.unique(week[idx])))

    def gain(self, ll_ref, ll_new):
        dd = (ll_ref - ll_new)[self.idx]
        pt = float(dd.mean())
        out = {"gain": pt}
        for nm, Ws in self.W.items():
            b = np.array([np.average(dd, weights=w) for w in Ws])
            lo, hi = np.percentile(b, [2.5, 97.5])
            out[f"ci_{nm}"] = [float(lo), float(hi)]
            if nm == "replay":
                out["ci_replay_x1.3"] = [pt - 1.3 * (pt - lo), pt + 1.3 * (hi - pt)]
        return out

    def ratio(self, num_ref, num_new, den_ref, den_new):
        """Ratio of two paired gains with a replay-bootstrap CI."""
        a = (num_ref - num_new)[self.idx]
        b = (den_ref - den_new)[self.idx]
        pt = float(a.mean() / b.mean())
        bs = [np.average(a, weights=w) / np.average(b, weights=w) for w in self.W["replay"]]
        return {"ratio": pt, "ci_replay": [float(x) for x in np.percentile(bs, [2.5, 97.5])]}


def fmt(r):
    return f"{r['gain']:+.5f} [{r['ci_replay_x1.3'][0]:+.5f},{r['ci_replay_x1.3'][1]:+.5f}]"


# presentation order for the nested ablation (status first, as asked; the
# same-day term last since it uses a different visibility rule)
NEST = ["status", "nowcast", "main", "ewma100", "mmr", "role", "rust", "party", "sameday"]


def evaluate(d, Fz, games, gd):
    des = Design(d, Fz, games, gd)
    y = des.y
    fit = games["V1"]
    tests = [k for k in games if k != "V1"]
    out = {"games": {k: int(v.sum()) for k, v in games.items()}}
    clus = clusters_first_player(d, gd[0])
    week = (gd[4] // 7).astype(np.int64)
    boots = {k: Boot(np.flatnonzero(games[k]), clus, week) for k in tests}
    out["weeks_per_test_window"] = {k: b.n_weeks for k, b in boots.items()}
    lls, preds, fits = {}, {}, {}

    def run(name, groups, lam, skill=True):
        w, Hm, Xm, cols = des.fit(groups, lam, fit, skill)
        p = predict(w, Xm)
        lls[name], preds[name], fits[name] = X.logloss(p, y), p, (w, Hm, Xm, cols, groups, lam)

    w0 = C.fit_logistic(des.lo[fit][:, None], y[fit])
    p0 = C.predict(w0, des.lo[:, None])
    lls["WP alone"], preds["WP alone"] = X.logloss(p0, y), p0
    run("headline", [], 0.0)
    lam_all, cv_all, cut = choose_lambda(des, list(GROUPS), fit)
    lam_l1, cv_l1, _ = choose_lambda(des, LAG1_GROUPS, fit)
    out["ridge"] = {"V1_internal_split_day": cut, "lambda_joint": lam_all, "cv_joint_V1_late_logloss": cv_all,
                    "lambda_joint_lag1": lam_l1, "cv_joint_lag1_V1_late_logloss": cv_l1}
    log("ridge:", json.dumps(out["ridge"]))
    run("joint (all terms)", list(GROUPS), lam_all)
    run("joint (lag-1 terms only)", LAG1_GROUPS, lam_l1)
    run("joint (all terms), no ridge", list(GROUPS), 0.0)
    run("joint (all terms) without skill s", list(GROUPS), lam_all, skill=False)
    for g in GROUPS:
        run(f"solo: headline + {g}", [g], 0.0)
        run(f"LOO: joint (all terms) without {g}", [x for x in GROUPS if x != g], lam_all)
        if g in LAG1_GROUPS:
            run(f"LOO: joint (lag-1) without {g}", [x for x in LAG1_GROUPS if x != g], lam_l1)
        if g != "status":
            run(f"status + {g}", ["status", g], 0.0)
    for k in range(1, len(NEST) + 1):
        run(f"nested {k}: headline + " + " + ".join(NEST[:k]), NEST[:k], lam_all if k > 2 else 0.0)

    res = {}
    for nm in lls:
        if nm == "WP alone":
            continue
        r = {}
        for t, bt in boots.items():
            r[t] = {"vs WP alone": bt.gain(lls["WP alone"], lls[nm])}
            if nm != "headline":
                r[t]["vs headline"] = bt.gain(lls["headline"], lls[nm])
            if nm.startswith("LOO: joint (all"):
                r[t]["marginal (joint minus this)"] = bt.gain(lls[nm], lls["joint (all terms)"])
            if nm.startswith("LOO: joint (lag-1"):
                r[t]["marginal (joint minus this)"] = bt.gain(lls[nm], lls["joint (lag-1 terms only)"])
            if nm.startswith("status + "):
                r[t]["vs headline + status"] = bt.gain(lls["solo: headline + status"], lls[nm])
            if nm.startswith("nested "):
                k = int(nm.split()[1].rstrip(":"))
                prev = "headline" if k == 1 else next(x for x in lls if x.startswith(f"nested {k - 1}:"))
                r[t]["increment over previous step"] = bt.gain(lls[prev], lls[nm])
            m = games[t]
            r[t]["metrics"] = C.game_metrics(preds[nm][m], y[m])
        if nm in fits:
            r["V1 coef"] = {"s": float(fits[nm][0][2]) if fits[nm][3][:1] == ["s"] else None,
                            "logit WP": float(fits[nm][0][1])}
        res[nm] = r
        t1 = "V2"
        t2 = "OOT"
        key = "vs headline" if nm != "headline" else "vs WP alone"
        log(f"  {nm:62s} V2 {key} {fmt(r[t1][key])}  OOT {fmt(r[t2][key])}  "
            f"V2 slope {r[t1]['metrics']['cal_slope']:.3f} ece {r[t1]['metrics']['ece']:.4f}"
            + (f" b_s {r['V1 coef']['s']:.2f}" if r.get("V1 coef", {}).get("s") is not None else ""))
    out["models"] = res
    out["metrics_WP_alone"] = {t: C.game_metrics(p0[games[t]], y[games[t]]) for t in tests}
    # coefficients and the refit gate (skill coefficient in 3.4 to 4.0)
    out["coef"] = {}
    gate = {}
    for nm in ("headline", "joint (all terms)", "joint (lag-1 terms only)", "joint (all terms), no ridge"):
        w, Hm, Xm, cols, groups, lam = fits[nm]
        out["coef"][nm] = des.coef_table(w, Hm, cols)
        gate[nm] = {"V1 fit": {"s": float(w[2]), "logit WP": float(w[1])}}
        for t in tests:
            wr, _ = fit_ridge(Xm[games[t]], y[games[t]], lam * des.matrix(groups)[1])
            gate[nm][f"refit on {t}"] = {"s": float(wr[2]), "logit WP": float(wr[1])}
    out["skill_coefficient_gate_3.4_to_4.0"] = gate
    log("gate:", json.dumps(gate))
    for nm in ("joint (all terms)", "joint (lag-1 terms only)"):
        log(nm, {k: round(v["coef"], 4) for k, v in out["coef"][nm].items()})
    return out


def add_nowcast(d, Fz, rows, mult=2.0):
    """(build, hero) nowcast per query row (p3_b_demean.causal_cell_demean on
    r_adj with the lag-1 table; tau2 by moments on E)."""
    import p3_b_demean as DM
    _, n_p, n_ph, table, r_adj = F.prepare_l1(d)
    e_mask = d["in_sample"] & (d["day"] >= C.day_of(C.E_START))
    build, _ = DM.row_builds(d)
    key = build * NUM_HEROES + d["hero"]
    tau2 = DM.estimate_tau2(r_adj, d["v"], key, e_mask)[0]
    noise = float(np.mean(d["v"]))
    n = len(d["pid"])
    now = np.full(n, np.nan)
    for qm, vm in ((rows["V1"] | rows["V2"], ~d["post"]), (rows["OOT"], np.ones(n, bool))):
        dv = subset(d, vm)
        _, nv, _ = DM.causal_cell_demean(dv, r_adj[vm], key[vm], mult * tau2, lag=1, noise=noise)
        dest = np.flatnonzero(vm)[qm[vm]]
        now[dest] = nv[qm[vm]]
    Fz["nowcast"] = now
    log(f"nowcast: tau2_E {1e4 * tau2:.3f} pp^2, m {mult}, k {noise / (mult * tau2):.0f} games")
    return {"tau2_E_pp2": 1e4 * tau2, "mult": mult, "k_games": noise / (mult * tau2)}


def use_skill_C(d, Fz):
    """Replace s by the C concentration-mean skill of p3_b_onetrick.py."""
    P = np.load(os.path.join(C.CACHE, "b7_onetrick_preds.npz"))
    names = [str(x) for x in P["names"]]
    s = np.full(len(d["pid"]), np.nan)
    s[P["qidx"]] = P["mu"].astype(np.float64) + P["m_" + str(names.index("C concentration mean"))]
    q = np.isfinite(Fz["s"])
    cov = float(np.isfinite(s[q]).mean())
    sp = np.full(len(d["pid"]), np.nan)
    sp[P["qidx"]] = P["mu"].astype(np.float64) + P["m_" + str(names.index("phase-1 kernel"))]
    both = q & np.isfinite(sp)
    info = {"coverage_of_query_rows": cov,
            "phase1_in_b7_vs_here_max_abs_diff": float(np.abs(sp[both] - Fz["s"][both]).max()),
            "phase1_in_b7_vs_here_mean_abs_diff": float(np.abs(sp[both] - Fz["s"][both]).mean())}
    log("skill C:", json.dumps(info))
    Fz["s_phase1"] = Fz["s"].copy()
    Fz["s"] = np.where(np.isfinite(s), s, np.nan)
    return info


def diagnostics(d, Fz, rows):
    """Feature summaries on V2 rows (sanity checks)."""
    m = rows["V2"]
    out = {}
    for k, v in Fz.items():
        if k.startswith("_") or k in ("t_start",):
            continue
        a = v[m]
        out[k] = {"mean": float(np.nanmean(a)), "sd": float(np.nanstd(a)), "nan_share": float(np.isnan(a).mean())}
    u, ua = Fz["_unavailable"][m] > 0, Fz["_unavailable_any"][m] > 0
    out["forced_off_main_V2"] = {"draft_prefix_flag_share": float(u.mean()),
                                 "anywhere_in_draft_flag_share (p3_x_ban_nat rule)": float(ua.mean()),
                                 "agreement": float((u == ua).mean()),
                                 "prefix_flag_given_anywhere_flag": float(u[ua].mean()) if ua.any() else None}
    out["acct_status_share_V2"] = {str(k): float(np.mean(Fz["acct_status"][m] == k)) for k in range(4)}
    out["sameday_visible_share_V2"] = float(np.mean(Fz["_sd_k"][m] > 0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=float, default=None, help="fraction of window games (debug)")
    ap.add_argument("--rebuild", action="store_true", help="ignore cached features")
    ap.add_argument("--skill", choices=["phase1", "C"], default="phase1")
    ap.add_argument("--nowcast-mult", type=float, default=2.0)
    ap.add_argument("--final", action="store_true",
                    help="also score the sealed build 2.55.17.98025 (final run only)")
    a = ap.parse_args()
    t0 = time.time()
    d = X.load_ext()
    med, rows, games, gd = windows(d, a.sample, a.final)
    log(f"rows {len(d['pid']):,}; split day {med}; window games {[int(v.sum()) for v in games.values()]}")
    Fz = None if a.rebuild else load_features(d, a.sample)
    tabs = None
    if Fz is None:
        Fz, tabs = build_features(d, rows, a.sample)
    else:
        log("loaded cached features")
    now_info = add_nowcast(d, Fz, rows, a.nowcast_mult)
    skill_info = use_skill_C(d, Fz) if a.skill == "C" else None
    if skill_info is not None:
        games = {k: v & (np.bincount(d["g"][np.isfinite(Fz["s"])], minlength=gd[0]) == 10)
                 for k, v in games.items()}
    out = {"skill": a.skill, "skill_C_check": skill_info, "nowcast": now_info, "split_day": med, "sample": a.sample, "final": a.final, "sealed_build": SEALED_BUILD, "groups": GROUPS, "diagnostics": diagnostics(d, Fz, rows)}
    if tabs:
        out["experience_tables"] = tabs
    out.update(evaluate(d, Fz, games, gd))
    name = "p3_b_joint" + ("_skillC" if a.skill == "C" else "") + ("_final" if a.final else "") + ("" if a.sample is None else f"_sample{a.sample}") + ".json"
    with open(os.path.join(C.RESULTS, name), "w") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {os.path.join(C.RESULTS, name)}; done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
