"""
P3 rich CF: scoreboard normalization cards (results/P3_RICH_CF.md, part 2).

Every card turns the scoreboard export (p3_export.py scoreboard, every corpus
player-game, no battletags) into one standardized value per player-game and
metric, for EVERY export row: window slots, the pre-2024 corpus rows that A1
adds, and the corpus copies of Heroes Profile history rows. Population
statistics are fit on export rows played 2024-04-01 .. 2026-02-10 (the E
window, before the WP training cutoff) only and applied to all rows.

Cards (each also writes a per-game evidence weight, 1 unless card e):
  base       met2_hs2's normalization (p3_rcf_data.py) on the full export:
             rate = value / minutes; t = asinh(rate / median positive rate);
             z = (t - mean) / sd within (hero, map, length bin, outcome), length
             bins 0-14 / 14-17 / 17-20 / 20-24 / 24+ min; cells with < 50 E rows
             fall back to (hero, length bin, outcome); clip +-5; minus the
             (hero, calendar quarter) mean of E rows (quarters outside E use
             the nearest E quarter); NaN where the metric is uninformative
             for the hero (< 5% nonzero E rows or mean cell sd <= 0.02)
  a_shrink   as base, but the (hero, map, bin, outcome) mean and variance are
             shrunk toward (hero, bin, outcome) with 100 pseudo-games instead
             of the hard 50-row fallback
  a_tier     a_shrink, then a (hero, map, tier group, bin, outcome) level
             shrunk toward it with 100 pseudo-games; tier groups
             Bronze-Silver / Gold / Platinum / Diamond / Master
  b_len      totals, not rates: t = asinh(value / median positive value);
             expected mean and sd as smooth functions of game length within
             (hero, outcome) (1-minute bins 6..40, Gaussian kernel 1.5 min,
             count weighted), plus a (hero, map, outcome) mean shift shrunk
             with 100 pseudo-games
  c_share    the player's share of the team total of the metric (team totals
             from the same game's ten rows), then base standardization
  c_opp      base t minus the mean t of opponents in the same Blizzard role
             (NaN if none), then base standardization
  d_rank     normal scores: Phi^-1 of the E-row rank of t within (hero, map,
             bin, outcome) (cells >= 200 E rows, else (hero, bin, outcome))
  d_wins     base with clip +-2.5 instead of +-5
  e_weight   base z, and each game's metric evidence weighted by how
             informative games of its kill margin are (weights fit on E
             rows; see weights())

Metrics only describe the row's own game. p3_rcf adds a game's row to the
state of games on later days only, so a game's own scoreboard never enters
its prediction.

Usage (from training/):
  nice -n 19 taskset -c 48-51 python3 personalization/p3_rcf_cards.py --tag rcf1017 --cards base,a_shrink,...
Output: cache/rcf/sb_<card>.npz: replay_id, blizz_id (export row order),
Z (float16, rows x M), W (float16 evidence weight), metrics.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C
from p3_rcf_data import METRICS, LEN_EDGES, RCF

E_LAST = C.day_of("2026-02-10")
K_SHRINK = 100.0
# team-share card (P3_PLAY_BY_PLAY): the player's share of the team total of these stats
SHARE_METRICS = ["hero_damage", "damage_taken", "healing", "siege_damage", "experience_contribution", "takedowns"]
CARDS = ["base", "a_shrink", "a_tier", "b_len", "c_share", "c_opp", "d_rank", "d_wins", "e_weight"]


def card_path(card):
    return os.path.join(RCF, f"sb_{card}.npz")


def read_export(tag, chunk=1_000_000):
    """Stream scoreboard.csv.gz with the csv module in chunks (worker venvs
    have no pandas; whole-file row lists do not fit in 20 GB). Strings are
    coded on the fly (hero names, maps); empty numeric fields become NaN."""
    import csv
    import gzip
    p = os.path.join(C.CACHE, "export", tag, "scoreboard.csv.gz")
    t0 = time.time()
    cols = ["replay_id", "blizz_id", "hero", "team", "winner", "league_tier", "end_ts", "game_map",
            "game_length"] + METRICS + [m for m in SHARE_METRICS if m not in METRICS]
    hero_code, map_code = {}, {}
    parts = {k: [] for k in cols}

    def fnum(x):
        return float(x) if x != "" else np.nan

    def flush(buf):
        T = list(zip(*buf))
        parts["replay_id"].append(np.array(T[0], np.int64))
        parts["blizz_id"].append(np.array(T[1], np.int64))
        parts["hero"].append(np.array([hero_code.setdefault(h, len(hero_code)) for h in T[2]], np.int32))
        parts["team"].append(np.array(T[3], np.int64))
        parts["winner"].append(np.array(T[4], np.int64))
        parts["league_tier"].append(np.array([fnum(x) for x in T[5]], np.float64))
        parts["end_ts"].append(np.array(T[6], np.int64))
        parts["game_map"].append(np.array([map_code.setdefault(x if x != "" else "?", len(map_code)) for x in T[7]],
                                          np.int32))
        parts["game_length"].append(np.array([fnum(x) for x in T[8]], np.float64))
        for k, m in enumerate(cols[9:]):
            parts[m].append(np.array([fnum(x) for x in T[9 + k]], np.float32))
    with gzip.open(p, "rt", newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        ix = [hdr.index(c) for c in cols]
        buf = []
        for r in rd:
            buf.append([r[k] for k in ix])
            if len(buf) >= chunk:
                flush(buf)
                buf = []
        if buf:
            flush(buf)
    out = {k: np.concatenate(v) for k, v in parts.items()}
    inv_h = np.array(sorted(hero_code, key=hero_code.get), object)
    inv_m = np.array(sorted(map_code, key=map_code.get), object)
    out["hero"] = inv_h[out["hero"]]
    out["game_map"] = inv_m[out["game_map"]]
    print(f"read and parsed {len(out['replay_id']):,} rows ({time.time() - t0:.0f}s)", flush=True)
    return out


def load(tag):
    df = read_export(tag)
    names = list(np.load(os.path.join(C.CACHE, "x_slots_ext.npz"))["hero_names"])
    hidx = {h: i for i, h in enumerate(names)}
    hero = np.array([hidx.get(h, -1) for h in df["hero"]], np.int64)
    keep = hero >= 0
    print(f"unknown hero rows dropped {(~keep).sum():,}", flush=True)
    df = {k: v[keep] for k, v in df.items()}
    hero = hero[keep]
    R = {"replay_id": df["replay_id"], "blizz_id": df["blizz_id"],
         "hero": hero, "team": df["team"], "win": df["winner"],
         "day": (df["end_ts"] // 86400).astype(np.int64),
         "glen": df["game_length"] / 60.0}
    lt = df["league_tier"]
    real = np.where(np.isnan(lt), 6, lt - 1)              # NULL = Master (live DB scheme)
    R["tier"] = np.select([real <= 2, real == 3, real == 4, real == 5], [0, 1, 2, 3], 4).astype(np.int64)
    _, R["map"] = np.unique(df["game_map"].astype(str), return_inverse=True)
    R["V"] = {m: df[m].astype(np.float64) for m in dict.fromkeys(METRICS + SHARE_METRICS)}
    R["H"] = len(names)
    R["fit"] = (R["day"] >= C.day_of(C.E_START)) & (R["day"] <= E_LAST) & (R["glen"] > 3)
    _, R["g"] = np.unique(R["replay_id"], return_inverse=True)
    return R


def moments(cell, t, fit, size):
    c = np.bincount(cell[fit], minlength=size).astype(np.float64)
    s1 = np.bincount(cell[fit], weights=t[fit], minlength=size)
    s2 = np.bincount(cell[fit], weights=t[fit] ** 2, minlength=size)
    m = s1 / np.maximum(c, 1)
    v = np.maximum(s2 / np.maximum(c, 1) - m ** 2, 0)
    return c, m, v


def shrink(c, m, v, pm, pv, k=K_SHRINK):
    """Cell moments shrunk toward parent moments (pm, pv per cell) with k pseudo-rows."""
    mu = (c * m + k * pm) / (c + k)
    var = (c * (v + (m - mu) ** 2) + k * (pv + (pm - mu) ** 2)) / (c + k)
    return mu, var


def quarter_center(R, z, good):
    q = R["day"].astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) // 3
    fq = q[R["fit"]]
    qlo, qhi = int(fq.min()), int(fq.max())
    nq = qhi - qlo + 1
    H = R["H"]
    hq = R["hero"] * nq + (np.clip(q, qlo, qhi) - qlo)
    g = good & R["fit"]
    cq = np.bincount(hq[g], minlength=H * nq)
    sq = np.bincount(hq[g], weights=z[g], minlength=H * nq)
    return z - (sq / np.maximum(cq, 1))[hq]


def base_t(R, m, totals=False):
    v = R["V"][m]
    miss = np.isnan(v) | (R["glen"] <= 3)
    v = np.where(np.isnan(v), 0.0, v)
    x = v if totals else v / np.maximum(R["glen"], 1.0)
    pos = x[R["fit"] & ~miss & (x > 0)]
    s = float(np.median(pos)) if len(pos) else 1.0
    return np.arcsinh(x / s), miss, x


def informative(R, x, miss, sd):
    H = R["H"]
    f = R["fit"] & ~miss
    n = np.maximum(np.bincount(R["hero"][f], minlength=H), 1)
    nz = np.bincount(R["hero"][f], weights=(x[f] > 0), minlength=H) / n
    sdh = np.bincount(R["hero"][f], weights=sd[f], minlength=H) / n
    return (nz >= 0.05) & (sdh > 0.02)


def standardize(R, t, miss, x, mode="hard", clip=5.0, tier=False):
    """(hero, map, length bin, outcome) standardization of t."""
    H = R["H"]
    nmap = int(R["map"].max()) + 1
    nb = len(LEN_EDGES)
    lb = np.searchsorted(LEN_EDGES, R["glen"], side="right") - 1
    w = R["win"]
    cb = (R["hero"] * nb + lb) * 2 + w
    ca = ((R["hero"] * nmap + R["map"]) * nb + lb) * 2 + w
    fit = R["fit"] & ~miss
    cB, mB, vB = moments(cb, t, fit, H * nb * 2)
    cA, mA, vA = moments(ca, t, fit, H * nmap * nb * 2)
    if mode == "hard":
        useA = cA[ca] >= 50
        mu = np.where(useA, mA[ca], mB[cb])
        sd = np.sqrt(np.where(useA, vA[ca], vB[cb]))
        bad = (cB[cb] < 50) & ~useA
    else:
        mu, var = shrink(cA[ca], mA[ca], vA[ca], mB[cb], vB[cb])
        bad = cB[cb] < 50
        if tier:
            ct = (((R["hero"] * nmap + R["map"]) * 5 + R["tier"]) * nb + lb) * 2 + w
            cT, mT, vT = moments(ct, t, fit, H * nmap * 5 * nb * 2)
            mu, var = shrink(cT[ct], mT[ct], vT[ct], mu, var)
        sd = np.sqrt(var)
    z = (t - mu) / np.maximum(sd, 1e-3)
    inf = informative(R, x, miss, sd)
    bad = bad | miss | ~inf[R["hero"]] | (sd < 1e-3)
    z = np.clip(z, -clip, clip)
    z = quarter_center(R, z, ~bad)
    z[bad] = np.nan
    return z


def card_len(R, m):
    """b_len: totals with a smooth expected mean and sd by game length."""
    t, miss, x = base_t(R, m, totals=True)
    H = R["H"]
    grid = np.arange(6, 41)
    L = np.clip(np.rint(R["glen"]), 6, 40).astype(np.int64) - 6
    nL = len(grid)
    hw = R["hero"] * 2 + R["win"]
    cell = hw * nL + L
    fit = R["fit"] & ~miss
    c, mu, var = moments(cell, t, fit, H * 2 * nL)
    c, mu, var = c.reshape(-1, nL), mu.reshape(-1, nL), var.reshape(-1, nL)
    Kmat = np.exp(-0.5 * ((grid[:, None] - grid[None, :]) / 1.5) ** 2)
    wsum = c @ Kmat
    ms = (c * mu) @ Kmat / np.maximum(wsum, 1e-9)
    s2 = (c * (var + mu ** 2)) @ Kmat / np.maximum(wsum, 1e-9) - ms ** 2
    e_mu = ms.ravel()[cell]
    e_sd = np.sqrt(np.maximum(s2.ravel()[cell], 1e-6))
    r = (t - e_mu) / e_sd
    nmap = int(R["map"].max()) + 1
    cm = (R["hero"] * nmap + R["map"]) * 2 + R["win"]
    cc, mm, _ = moments(cm, r, fit, H * nmap * 2)
    shift = cc / (cc + K_SHRINK) * mm
    z = np.clip(r - shift[cm], -5, 5)
    inf = informative(R, x, miss, e_sd)
    bad = miss | ~inf[R["hero"]] | (wsum.ravel()[cell] < 50)
    z = quarter_center(R, z, ~bad)
    z[bad] = np.nan
    return z


def card_share(R, m, mode="hard"):
    v = R["V"][m]
    miss = np.isnan(v) | (R["glen"] <= 3)
    v0 = np.where(np.isnan(v), 0.0, v)
    gt = R["g"] * 2 + R["team"]
    tot = np.bincount(gt, weights=v0, minlength=int(gt.max()) + 1)[gt]
    n10 = np.bincount(R["g"], minlength=int(R["g"].max()) + 1)[R["g"]]
    share = np.where(tot > 0, v0 / np.maximum(tot, 1e-9), np.nan)
    miss = miss | np.isnan(share) | (n10 != 10)
    share = np.where(np.isnan(share), 0.0, share)
    return standardize(R, share, miss, v0, mode=mode)


def card_opp(R, m, roles):
    t, miss, x = base_t(R, m)
    role = roles[R["hero"]]
    # opponents in the same role: sums of t by (game, team, role)
    key = (R["g"] * 2 + R["team"]) * 6 + role
    okt = ~miss
    S = np.bincount(key, weights=np.where(okt, t, 0), minlength=int(key.max()) + 7)
    N = np.bincount(key, weights=okt.astype(float), minlength=int(key.max()) + 7)
    okey = (R["g"] * 2 + (1 - R["team"])) * 6 + role
    no = N[okey]
    d = np.where(no > 0, t - S[okey] / np.maximum(no, 1), np.nan)
    miss = miss | (no == 0)
    d = np.where(np.isnan(d), 0.0, d)
    return standardize(R, d, miss, x)


def card_rank(R, m):
    from scipy.special import ndtri
    t, miss, x = base_t(R, m)
    H = R["H"]
    nmap = int(R["map"].max()) + 1
    nb = len(LEN_EDGES)
    lb = np.searchsorted(LEN_EDGES, R["glen"], side="right") - 1
    cb = (R["hero"] * nb + lb) * 2 + R["win"]
    ca = ((R["hero"] * nmap + R["map"]) * nb + lb) * 2 + R["win"]
    fit = R["fit"] & ~miss
    nA = np.bincount(ca[fit], minlength=H * nmap * nb * 2)
    cell = np.where(nA[ca] >= 200, ca, H * nmap * nb * 2 + cb)        # two id spaces
    tt = np.clip(t, 0, 99.0)
    keys_fit = np.sort(cell[fit] * 100.0 + tt[fit])
    q = cell * 100.0 + tt
    lo = np.searchsorted(keys_fit, cell * 100.0, side="left")
    hi = np.searchsorted(keys_fit, cell * 100.0 + 99.999, side="right")
    r_lo = np.searchsorted(keys_fit, q, side="left")
    r_hi = np.searchsorted(keys_fit, q, side="right")
    n = hi - lo
    u = ((r_lo + r_hi) / 2.0 - lo + 0.5) / np.maximum(n + 1, 1)
    z = ndtri(np.clip(u, 1e-4, 1 - 1e-4))
    sd_proxy = np.where(n >= 2, 1.0, 0.0)
    inf = informative(R, x, miss, np.ones(len(t)))
    nB = np.bincount(cb[fit], minlength=H * nb * 2)
    bad = miss | ~inf[R["hero"]] | (nB[cb] < 50) | (sd_proxy == 0)
    z = quarter_center(R, z, ~bad)
    z[bad] = np.nan
    return z


def weights(R, Z):
    """Card e: evidence weight per game by kill margin. For E rows, a game's
    informativeness about the player's stable level is the slope of the
    player's leave-one-out mean z on the game's z (cov / var of the game);
    it is computed per quintile of the kill margin |K0 - K1| / (K0 + K1 + 1)
    and per metric, averaged over metrics, and normalized to mean 1 over E
    games. Returns (weight per row, table)."""
    k = np.where(np.isnan(R["V"]["kills"]), 0, R["V"]["kills"])
    gt = R["g"] * 2 + R["team"]
    kt = np.bincount(gt, weights=k, minlength=int(gt.max()) + 2)
    k0, k1 = kt[R["g"] * 2], kt[R["g"] * 2 + 1]
    margin = np.abs(k0 - k1) / (k0 + k1 + 1)
    fit = R["fit"]
    edges = np.quantile(margin[fit], [0.2, 0.4, 0.6, 0.8])
    qb = np.searchsorted(edges, margin, side="right")
    key = R["blizz_id"] * 128 + R["hero"]
    slopes = []
    for j in range(Z.shape[1]):
        z = Z[:, j].astype(np.float64)
        ok = fit & ~np.isnan(z)
        u, inv = np.unique(key[ok], return_inverse=True)
        s = np.bincount(inv, weights=z[ok])
        n = np.bincount(inv).astype(float)
        loo = (s[inv] - z[ok]) / np.maximum(n[inv] - 1, 1)
        has = n[inv] >= 6
        sl = []
        for b in range(5):
            mb = has & (qb[ok] == b)
            zz, ll = z[ok][mb], loo[mb]
            cov = np.mean((zz - zz.mean()) * (ll - ll.mean()))
            sl.append(cov / zz.var())
        slopes.append(np.array(sl) / np.mean(sl))
    tab = np.mean(slopes, 0)
    w = tab[qb]
    w = w / w[fit].mean()
    return w.astype(np.float16), {"margin_quintile_edges": edges.tolist(), "weight_by_quintile": tab.tolist(),
                                  "per_metric": {m: s.round(3).tolist() for m, s in zip(METRICS, slopes)}}


def build(tag, cards):
    t0 = time.time()
    R = load(tag)
    roles = C.hero_meta(np.load(os.path.join(C.CACHE, "x_slots_ext.npz"))["hero_names"])["blizz"]
    os.makedirs(RCF, exist_ok=True)
    for card in cards:
        mets = SHARE_METRICS if card == "share6" else METRICS
        Z = np.full((len(R["hero"]), len(mets)), np.nan, np.float32)
        for k, m in enumerate(mets):
            if card == "share6":
                # shares standardized like a_shrink: (hero, map, length, outcome) shrunk to (hero, length, outcome)
                z = card_share(R, m, mode="shrink")
            elif card in ("base", "e_weight", "d_wins", "a_shrink", "a_tier"):
                t, miss, x = base_t(R, m)
                z = standardize(R, t, miss, x, mode="hard" if card in ("base", "e_weight", "d_wins") else "shrink",
                                clip=2.5 if card == "d_wins" else 5.0, tier=card == "a_tier")
            elif card == "b_len":
                z = card_len(R, m)
            elif card == "c_share":
                z = card_share(R, m)
            elif card == "c_opp":
                z = card_opp(R, m, roles)
            elif card == "d_rank":
                z = card_rank(R, m)
            Z[:, k] = z
        W = np.ones(len(Z), np.float16)
        info = {"rows": int(len(Z)), "rows_any_value": int((~np.isnan(Z)).any(1).sum()),
                "e_sd": {m: float(np.nanstd(Z[R["fit"], k])) for k, m in enumerate(mets)}}
        if card == "e_weight":
            W, info["weights"] = weights(R, Z)
            print(f"  [e_weight] weight by kill-margin quintile {np.round(info['weights']['weight_by_quintile'], 3)}",
                  flush=True)
        np.savez(card_path(card), replay_id=R["replay_id"], blizz_id=R["blizz_id"], Z=Z.astype(np.float16), W=W,
                 metrics=np.array(mets), info=np.array(json.dumps(info)))
        print(f"[{card}] wrote {card_path(card)}; rows with a value {info['rows_any_value']:,} "
              f"({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rcf1017")
    ap.add_argument("--cards", default=",".join(CARDS))
    a = ap.parse_args()
    build(a.tag, a.cards.split(","))


if __name__ == "__main__":
    main()
