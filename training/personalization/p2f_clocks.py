"""
Paper 2 narrative facts, question 2, rigorous version: the game's clock and
the players' clock measured with ONE estimator on ONE table.

Data: the full corpus slot table of the full-history panel (cache_full
x_slots_ext.npz: 22.4M slots, every Storm League game 2021-12-07 to
2026-09-27 with all ten players labeled), with the draft-only WP of each
game (d2c_cumprev, statistics through the previous build). Tiers: real
league tier (stored - 1, NULL = Master) from the pinned snapshot and the
post-snapshot export, grouped Bronze-Silver / Gold-Platinum / Diamond-Master.

Windows. Window A is 90 days; window B is 90 days starting h days after A
ends (h = 91, 182, 365, 730). A new A starts every 30 days and the pairs are
pooled. Each cell is estimated separately from the games played inside A and
inside B. Nothing carries over: a player who does not play a hero in B has no
B value for it and enters only the selection check.

Cells and values, in points of win probability:
  game side    hero x tier group: x = 100 (win rate - 0.5), sampling
               variance p(1 - p) / n (binomial, p = the cell's own rate)
  player side  player x hero: x = 100 mean(y - WP), the player's results on the
               hero beyond what the draft predicts ("whole cell"); and the
               same minus the player's own mean over all heroes in the window
               ("hero-specific"). Sampling variance: within-cell sample
               variance / n (residual-based, primary; one-game cells use
               wp(1 - wp)), or sum wp(1 - wp) / n^2 (model-based).

Estimator (measurement-error model, method of moments with inverse-variance
weights): each cell's estimate = true effect + sampling error of known
variance e. For cells observed in both windows, with weights
w = 1 / sqrt((eA + tA^2)(eB + tB^2)) with e the model noise (tA^2, tB^2 from an equal-weight first pass):
  tA^2 = [sum w (xA - mA)^2 - sum w eA] / (W - sum w^2 / W), same for B,
  cov  = sum w (xA - mA)(xB - mB) / (W - sum w^2 / W)  (errors independent across windows),
  r    = cov / sqrt(tA^2 tB^2), the true-effect (disattenuated) correlation.
Moments are pooled over window pairs (each pair centered on its own means).
95% CIs: 200 bootstrap draws over players (200 hash groups) or over heroes.

Secondary: sensitivity to a minimum number of games per cell in each window
(1, 5, 10, 20, 50, 100), and on the game side to a minimum pick rate
(0.25% to 5% of the tier group's games, both windows). Selection checks:
player cells active in A that stop in B (share, their A values against
continuers', and an intent-to-treat bound that sets a stopper's B value to the
population mean); game cells split by whether the patch notes list a balance
change or rework for the hero between the two windows.

Usage (from training/): nice -n 19 taskset -c 48-51 python3 personalization/p2f_clocks.py --slots <x_slots_ext.npz>
Output: results/p2f/clocks.json
"""
import argparse
import gzip
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results", "p2f")
W = 90
STEP = 30
HORIZONS = {"3 months": 91, "6 months": 182, "1 year": 365, "2 years": 730}
NB = 200
NGP = 200
MINS = (1, 2, 5, 10, 20, 50, 100)
# player cells need 2+ games in a window for the residual-based noise (a sample
# variance); one-game cells carry almost no weight and their noise must come
# from the WP model, which is 2.5% off per game (larger than the skill variance)
NMIN = {"game": 1, "player": 2}
PICKS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05)
TG = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
TG_NAMES = ["Bronze-Silver", "Gold-Platinum", "Diamond-Master"]


def tiers_for(rid):
    z = np.load(os.path.join(HERE, "cache", "p2f_snapshot_games.npz"))
    keys, tier = [z["replay_id"]], [z["tier"].astype(np.int64)]
    post = json.load(gzip.open(os.path.join(HERE, "cache", "x_post_games.json.gz"), "rt"))
    keys.append(np.array([g["replay_id"] for g in post], np.int64))
    tier.append(np.array([6 if g["league_tier"] is None else g["league_tier"] - 1 for g in post], np.int64))
    keys, tier = np.concatenate(keys), np.concatenate(tier)
    o = np.argsort(keys)
    keys, tier = keys[o], tier[o]
    j = np.minimum(np.searchsorted(keys, rid), len(keys) - 1)
    t = np.where(keys[j] == rid, tier[j], -1)
    lut = np.full(8, -1)
    for k, v in TG.items():
        lut[k] = v
    return np.where(t >= 0, lut[np.clip(t, 0, 7)], -1)


def changed_heroes(hero_names):
    """(publish day, set of hero indices with a balance or rework section) per Live patch."""
    d = json.load(open(os.path.join(TRAINING, "drift_v2", "patch_notes_ground_truth.json")))
    pos = {h: i for i, h in enumerate(hero_names)}
    out = []
    for p in d["patches"]:
        if p.get("patch_type") == "PTR":
            continue
        ch = set(p.get("heroes_changed") or []) - set(p.get("heroes_bugfix_only") or [])
        day = int(np.datetime64(p["published"], "D").astype(np.int64))
        out.append((day, {pos[h] for h in ch if h in pos}))
    return out


# ------------------------------------------------------------------ windows

def game_window(S, lo, hi):
    m = (S["day"] >= lo) & (S["day"] < hi) & (S["tg"] >= 0)
    cell = S["tg"][m] * S["H"] + S["hero"][m]
    nc = 3 * S["H"]
    n = np.bincount(cell, minlength=nc).astype(np.float64)
    wins = np.bincount(cell, weights=S["y"][m], minlength=nc)
    games = np.bincount(S["tg"][m], minlength=3).astype(np.float64) / 10.0
    p = np.where(n > 0, wins / np.maximum(n, 1), 0.5)
    pr = n / np.maximum(games[np.arange(nc) // S["H"]], 1)
    return {"cid": np.flatnonzero(n > 0), "n": n[n > 0], "x": 100 * (p[n > 0] - 0.5),
            "e": 1e4 * p[n > 0] * (1 - p[n > 0]) / n[n > 0], "pick": pr[n > 0]}


def player_window(S, lo, hi):
    m = np.flatnonzero((S["day"] >= lo) & (S["day"] < hi))
    pl, he = S["pid"][m], S["hero"][m]
    cid = pl * S["H"] + he
    u, inv = np.unique(cid, return_inverse=True)
    r, v = S["r"][m], S["v"][m]
    n = np.bincount(inv).astype(np.float64)
    s1 = np.bincount(inv, weights=r)
    s2 = np.bincount(inv, weights=r * r)
    sv = np.bincount(inv, weights=v)
    x = s1 / n
    ss = np.maximum(s2 - n * x * x, 0)
    e_emp = np.where(n > 1, ss / np.maximum(n - 1, 1) / n, sv / n ** 2)
    e_mod = sv / n ** 2
    pu, pinv = np.unique(u // S["H"], return_inverse=True)
    N = np.bincount(pinv, weights=n)
    R = np.bincount(pinv, weights=s1)
    xs = x - (R / N)[pinv]
    out = {"cid": u, "n": n, "x": 100 * x, "xs": 100 * xs}
    for nm, e in (("emp", e_emp), ("mod", e_mod)):
        sum_n2e = np.bincount(pinv, weights=n * n * e)
        Np = N[pinv]
        out["e_" + nm] = 1e4 * e
        out["es_" + nm] = np.where(Np > n, 1e4 * (e + sum_n2e[pinv] / Np ** 2 - 2 * n * e / Np), np.nan)
    out["xs"] = np.where(N[pinv] > n, out["xs"], np.nan)
    return out


# ------------------------------------------------------------------ estimator

FIELDS = ("W", "W2", "wa", "wb", "waa", "wbb", "wab", "wea", "web")


def sums(xa, ea, xb, eb, w, grp, G):
    cols = (w, w * w, w * xa, w * xb, w * xa * xa, w * xb * xb, w * xa * xb, w * ea, w * eb)
    return np.stack([np.bincount(grp, weights=c, minlength=G) for c in cols], 1)


def moments(T):
    """T: (pairs, 9) summed over groups -> pooled (tA2, tB2, cov) numerators and denominator."""
    Wt, W2, wa, wb, waa, wbb, wab, wea, web = T.T
    ok = Wt > 0
    Wt, W2, wa, wb, waa, wbb, wab, wea, web = (z[ok] for z in (Wt, W2, wa, wb, waa, wbb, wab, wea, web))
    den = (Wt - W2 / Wt).sum()
    ta = (waa - wa * wa / Wt - wea).sum() / den
    tb = (wbb - wb * wb / Wt - web).sum() / den
    cv = (wab - wa * wb / Wt).sum() / den
    return ta, tb, cv


def corr_ci(P, rng):
    """P: (pairs, G, 9). Point estimate and bootstrap CI over groups."""
    G = P.shape[1]
    ta, tb, cv = moments(P.sum(1))
    r = cv / np.sqrt(ta * tb) if ta > 0 and tb > 0 else np.nan
    bs = []
    for _ in range(NB):
        c = rng.multinomial(G, np.full(G, 1.0 / G)).astype(np.float64)
        a, b, v = moments(np.einsum("pgk,g->pk", P, c))
        bs.append(v / np.sqrt(a * b) if a > 0 and b > 0 else np.nan)
    bs = np.array(bs)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]) if np.isfinite(bs).sum() > 20 else (np.nan, np.nan)
    return {"r": float(r), "ci": [float(lo), float(hi)], "true_sd_A_pts": float(np.sqrt(max(ta, 0))),
            "true_sd_B_pts": float(np.sqrt(max(tb, 0))), "true_cov": float(cv),
            "carried_sd_pts": float(np.sqrt(max(cv, 0))), "boot_nan": int((~np.isfinite(bs)).sum())}


# ------------------------------------------------------------------ driver

def run_side(S, side, rng, log):
    d0, d1 = int(S["day"].min()), int(S["day"].max()) + 1
    starts = np.arange(d0, d1, STEP)
    win = (lambda lo: game_window(S, lo, lo + W)) if side == "game" else (lambda lo: player_window(S, lo, lo + W))
    G = 90 if side == "game" else NGP
    # (name, value, noise subtracted, noise used for the weights). Weights always come
    # from the model noise: the residual-based noise is itself noisy at small n
    # (two wins give a sample variance of 0) and correlated with the value.
    measures = [("hero x tier", "x", "e", "e")] if side == "game" else [
        ("whole cell, residual-based noise", "x", "e_emp", "e_mod"), ("whole cell, model noise", "x", "e_mod", "e_mod"),
        ("hero-specific, residual-based noise", "xs", "es_emp", "es_mod"), ("hero-specific, model noise", "xs", "es_mod", "es_mod")]
    cache = {}

    def get(lo):
        if lo not in cache:
            cache[lo] = win(lo)
        return cache[lo]

    patches = changed_heroes(S["hero_names"]) if side == "game" else None
    res = {}
    for hname, h in HORIZONS.items():
        pairs = []
        for s in starts:
            sb = s + W + h
            if sb + W > d1:
                break
            for old in [t for t in cache if t < s]:
                cache.pop(old)
            a, b = get(int(s)), get(int(sb))
            com, ia, ib = np.intersect1d(a["cid"], b["cid"], assume_unique=True, return_indices=True)
            onlya = np.setdiff1d(np.arange(len(a["cid"])), ia, assume_unique=True)
            pairs.append((s, sb, a, b, com, ia, ib, onlya))
        out = {"window_pairs": len(pairs)}
        for mname, xk, ek, wk in measures:
            # first pass: equal weights, all cells in both windows -> t^2
            P0 = np.zeros((len(pairs), G, 9))
            for i, (s, sb, a, b, com, ia, ib, _) in enumerate(pairs):
                grp = (com % S["H"]) if side == "game" else ((com // S["H"]) * 2654435761 % 2 ** 32) % G
                ok = (a["n"][ia] >= NMIN[side]) & (b["n"][ib] >= NMIN[side]) & np.isfinite(a[xk][ia]) & np.isfinite(b[xk][ib])
                P0[i] = sums(a[xk][ia][ok], a[ek][ia][ok], b[xk][ib][ok], b[ek][ib][ok], np.ones(ok.sum()), grp[ok], G)
            ta0, tb0, _ = moments(P0.sum(1))
            ta0, tb0 = max(ta0, 1e-6), max(tb0, 1e-6)
            sens = {}
            filters = [("min games", k, 0.0) for k in MINS if k >= NMIN[side]]
            if side == "game":
                filters += [("min pick rate", 1, pr) for pr in PICKS[1:]]
            for fkind, kmin, pmin in filters:
                P = np.zeros((len(pairs), G, 9))
                for i, (s, sb, a, b, com, ia, ib, _) in enumerate(pairs):
                    sel = (a["n"][ia] >= kmin) & (b["n"][ib] >= kmin) & np.isfinite(a[xk][ia]) & np.isfinite(b[xk][ib])
                    if side == "game":
                        sel &= (a["pick"][ia] >= pmin) & (b["pick"][ib] >= pmin)
                    grp = (com % S["H"]) if side == "game" else ((com // S["H"]) * 2654435761 % 2 ** 32) % G
                    ea, eb = a[ek][ia][sel], b[ek][ib][sel]
                    w = 1.0 / np.sqrt((a[wk][ia][sel] + ta0) * (b[wk][ib][sel] + tb0))
                    P[i] = sums(a[xk][ia][sel], ea, b[xk][ib][sel], eb, w, grp[sel], G)
                key = f"{fkind} {kmin if fkind == 'min games' else f'{100 * pmin:g}%'}"
                sens[key] = corr_ci(P, rng)
            pk = f"min games {NMIN[side]}"
            out[mname] = {"primary (all cells with play in both windows)": sens[pk], "sensitivity": sens}
            log(f"{side} {hname} {mname}: r {sens[pk]['r']:.3f} {np.round(sens[pk]['ci'], 3)} | "
                + " ".join(f"{k.split()[-1]}:{v['r']:.2f}" for k, v in sens.items()))
        # cell counts and selection checks
        n_both = sum(len(p[4]) for p in pairs)
        n_a = sum(len(p[2]["cid"]) for p in pairs)
        out["cells_in_both"] = int(n_both)
        out["cells_in_A"] = int(n_a)
        if side == "player":
            out["selection"] = selection_player(pairs, S)
        else:
            out["rework_split"] = rework_split(pairs, S, patches, rng)
        res[hname] = out
    return res


def selection_player(pairs, S):
    """Share of A cells that continue into B; A values of continuers vs stoppers;
    intent-to-treat bound with a stopper's B value set to the B mean (equal weights,
    residual-based noise, whole cell and hero-specific)."""
    out = {}
    for xk, ek, lab in (("x", "e_emp", "whole cell"), ("xs", "es_emp", "hero-specific")):
        num_both = den_both = 0.0
        taA_all = taA_both = tbB = 0.0
        nA_all = nA_both = 0
        cont_a, stop_a, cont_n, stop_n = [], [], 0, 0
        for s, sb, a, b, com, ia, ib, onlya in pairs:
            okA = (a["n"] >= 2) & np.isfinite(a[xk])
            okB = (b["n"] >= 2) & np.isfinite(b[xk])
            keep = okA[ia] & okB[ib]
            onlya = onlya[okA[onlya]]             # stoppers: 2+ games in A, none in B
            ia, ib = ia[keep], ib[keep]           # continuers: 2+ games in both
            xa_b, xb_b = a[xk][ia], b[xk][ib]
            allA = np.flatnonzero(okA)
            ma = a[xk][allA].mean()
            mb = xb_b.mean()
            num_both += ((xa_b - xa_b.mean()) * (xb_b - mb)).sum()
            den_both += len(ia)
            taA_both += ((xa_b - xa_b.mean()) ** 2 - a[ek][ia]).sum()
            tbB += ((xb_b - mb) ** 2 - b[ek][ib]).sum()
            taA_all += ((a[xk][allA] - ma) ** 2 - a[ek][allA]).sum()
            nA_all += len(allA)
            nA_both += len(ia)
            cont_a.append(a[xk][ia].sum())
            stop_a.append(a[xk][onlya].sum())
            cont_n += len(ia)
            stop_n += len(onlya)
        tA_b, tB, tA_all = taA_both / nA_both, tbB / nA_both, taA_all / nA_all
        cov_both = num_both / den_both
        r_both = cov_both / np.sqrt(tA_b * tB) if tA_b > 0 and tB > 0 else np.nan
        # ITT: stoppers have B = mean, so they add zero covariance but count in the denominator
        cov_itt = num_both / nA_all
        r_itt = cov_itt / np.sqrt(tA_all * tB) if tA_all > 0 and tB > 0 else np.nan
        out[lab] = {"share_of_A_cells_continuing": nA_both / nA_all,
                    "r_both_active_equal_weights": float(r_both), "r_intent_to_treat_bound": float(r_itt),
                    "true_sd_A_continuers_pts": float(np.sqrt(max(tA_b, 0))),
                    "true_sd_A_all_pts": float(np.sqrt(max(tA_all, 0))),
                    "mean_A_value_continuers_pts": float(sum(cont_a) / cont_n),
                    "mean_A_value_stoppers_pts": float(sum(stop_a) / stop_n)}
    return out


def rework_split(pairs, S, patches, rng):
    """Hero x tier correlation for heroes with vs without a listed balance change
    or rework published between the end of A and the end of B (the period after
    A's games). Bootstrap over heroes."""
    G = 90
    res = {}
    for lab in ("no listed change", "changed once", "changed 2+ times"):
        P = np.zeros((len(pairs), G, 9))
        for i, (s, sb, a, b, com, ia, ib, _) in enumerate(pairs):
            cnt = np.zeros(S["H"], np.int64)
            for day, hs in patches:
                if s + W <= day < sb + W:
                    for hh in hs:
                        cnt[hh] += 1
            hero = com % S["H"]
            c = cnt[hero]
            sel = (c == 0) if lab == "no listed change" else ((c == 1) if lab == "changed once" else (c >= 2))
            if sel.sum() == 0:
                continue
            w = np.ones(sel.sum())
            P[i] = sums(a["x"][ia][sel], a["e"][ia][sel], b["x"][ib][sel], b["e"][ib][sel], w, hero[sel], G)
        if P[:, :, 0].sum() < 20:
            res[lab] = None
            continue
        res[lab] = corr_ci(P, rng)
        res[lab]["cell_pairs"] = int(P[:, :, 0].sum())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", required=True)
    ap.add_argument("--side", default="both", choices=["game", "player", "both"])
    a_ = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    logf = open(os.path.join(OUT, "clocks.log"), "a")

    def log(msg):
        line = f"[{time.time() - t0:5.0f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    z = np.load(a_.slots)
    wp = z["wp"].astype(np.float64)
    S = {"pid": z["pid"].astype(np.int64), "hero": z["hero"].astype(np.int64), "day": z["day"].astype(np.int64),
         "y": z["y"].astype(np.float64), "r": z["y"].astype(np.float64) - wp, "v": wp * (1 - wp),
         "H": int(len(z["hero_names"])), "hero_names": [str(h) for h in z["hero_names"]]}
    S["tg"] = tiers_for(z["replay_id"].astype(np.int64))
    log(f"slots {len(wp):,}, {np.datetime64(int(S['day'].min()), 'D')} to {np.datetime64(int(S['day'].max()), 'D')}, "
        f"tier known {np.mean(S['tg'] >= 0):.3f}")
    path = os.path.join(OUT, "clocks.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    out["method"] = __doc__.split("Usage")[0].strip()
    rng = np.random.default_rng(0)
    for side in (["game", "player"] if a_.side == "both" else [a_.side]):
        out[side] = run_side(S, side, rng, log)
        json.dump(out, open(path, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
