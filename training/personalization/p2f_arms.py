"""
Paper 2 narrative facts from the scored arms of the full-history panel
(full_1017): questions 3 (shrinkage persistence), 5 (regions) and 6 (top
tiers and specialists).

Inputs (copied from the 3090, gitignored results cache; player keys only):
  A2S.npz            win-only CF model on full histories (p3_c_tests arm A2S)
  full_ms_layer.npz  the main model (metrics + layer), p3_rcf.py online arm
  cache/x_side_games.npz, cache/x_tiers.npz, cache/x_post_games.json.gz
                     region and league tier per game (snapshot and OOT)

Both arms score the same games: logit p = c0 + c1 logit(WP) + c2 D, with D the
team difference of the slot estimates s_pred, combiner fit on V1. Test games
are V2 (72,474) and OOT (286,016), all ten players labeled.

Q3, persistence of a player's record on a hero (A2S slots of V2 + OOT):
  x = record to date on the hero = mean of (y - WP) on corpus games and
  (y - hero-quarter win rate) on Heroes Profile rows, over the full history
  through the previous day; n = games on the hero in that history. The
  outcome is the scored game's residual r = y - WP (WP sees the draft and the
  lobby's MMR). Per n bin, the OLS slope of r on x is the share of a past
  record that shows up in the next game ("keep"). Shrinkage formula:
  keep = tau^2 / (tau^2 + sigma^2 / n). tau^2 is measured in each bin as
  var(x) - mean(sigma^2 / n) (method of moments, sigma^2 = 0.245), so the
  formula's prediction uses no outcome after the record.
Q5, by region (1 Americas, 2 Europe, 3 Korea; region of the game).
Q6, by real league tier, plus the top 1% of games by lobby MMR, and
  counterfactual concentration: keep s_pred only on a chosen set of slots
  (others 0 = population expectation), recompute p with the same combiner,
  and compare the log-loss gain with the full model's.

Usage (from training/): nice -n 19 taskset -c 48-51 python3 personalization/p2f_arms.py --arms <dir>
Output: results/p2f/arms.json
"""
import argparse
import gzip
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "p2f")
CACHE = os.path.join(HERE, "cache")
SIG2 = 0.245
NBINS = [(30, 59), (60, 119), (120, 249), (250, 10 ** 9)]
NLAB = ["30-59", "60-119", "120-249", "250+"]
TIERS = {1: "Bronze", 2: "Silver", 3: "Gold", 4: "Platinum", 5: "Diamond", 6: "Master"}
REGION = {1: "Americas", 2: "Europe", 3: "Korea"}
NG = 20


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files if k != "meta"}


def game_meta():
    sg = np.load(os.path.join(CACHE, "x_side_games.npz"))
    xt = np.load(os.path.join(CACHE, "x_tiers.npz"))
    o1, o2 = np.argsort(sg["replay_ids"]), np.argsort(xt["replay_ids"])
    assert (sg["replay_ids"][o1] == xt["replay_ids"][o2]).all()
    lt = xt["league_tier"][o2].astype(np.int64)
    rid = [sg["replay_ids"][o1]]
    reg = [sg["region"][o1].astype(np.int64)]
    tier = [np.where(lt <= 0, 6, lt - 1)]   # stored = real + 1; NULL (stored -1 here) = Master
    mmr = [xt["avg_mmr"][o2].astype(np.float64)]
    post = json.load(gzip.open(os.path.join(CACHE, "x_post_games.json.gz"), "rt"))
    rid.append(np.array([g["replay_id"] for g in post], np.int64))
    reg.append(np.array([-1 if g["region"] is None else g["region"] for g in post], np.int64))
    tier.append(np.array([6 if g["league_tier"] is None else g["league_tier"] - 1 for g in post], np.int64))
    mmr.append(np.array([np.nan if g["avg_mmr"] is None else g["avg_mmr"] for g in post], np.float64))
    rid, reg, tier, mmr = (np.concatenate(v) for v in (rid, reg, tier, mmr))
    o = np.argsort(rid)
    return rid[o], reg[o], tier[o], mmr[o]


def lookup(keys, rid, *cols):
    j = np.minimum(np.searchsorted(keys, rid), len(keys) - 1)
    ok = keys[j] == rid
    return [np.where(ok, c[j], -1 if c.dtype.kind == "i" else np.nan) for c in cols]


def ll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def game_index(a):
    gi = np.searchsorted(a["g_replay_id"], a["s_replay_id"])
    gi = np.minimum(gi, len(a["g_replay_id"]) - 1)
    ok = a["g_replay_id"][gi] == a["s_replay_id"]
    return gi, ok


def counterfactual(a, keep_slot, gi, ok):
    """Game probabilities with s_pred kept only on keep_slot (others 0)."""
    sp = np.where(keep_slot, a["s_pred"].astype(np.float64), 0.0)
    sgn = np.where(a["s_team"] == 0, 1.0, -1.0)
    D = np.zeros(len(a["g_replay_id"]))
    np.add.at(D, gi[ok], (sgn * sp)[ok])
    c = a["coef"]
    return 1 / (1 + np.exp(-(c[0] + c[1] * a["g_lo"] + c[2] * D)))


def gsummary(a, m, p=None):
    y = a["g_y"][m]
    p = a["g_p"][m] if p is None else p[m]
    p0 = a["g_p0"][m]
    return {"games": int(m.sum()), "acc_draft": float(((p0 > 0.5) == (y == 1)).mean()),
            "acc_full": float(((p > 0.5) == (y == 1)).mean()),
            "ll_gain_x1000": float(1000 * (ll(p0, y) - ll(p, y)).mean())}


def persistence(x, r, n, grp):
    """Slope of r on x with a jackknife SE over player groups, and the
    method-of-moments shrinkage prediction."""
    def slope(xx, rr):
        xc = xx - xx.mean()
        return float((xc * (rr - rr.mean())).sum() / (xc * xc).sum())
    b = slope(x, r)
    jk = np.array([slope(x[grp != g], r[grp != g]) for g in range(NG)])
    se = float(np.sqrt((NG - 1) / NG * ((jk - jk.mean()) ** 2).sum()))
    noise = SIG2 / n
    tau2 = float(x.var() - noise.mean())
    pred = float(np.mean(tau2 / (tau2 + noise))) if tau2 > 0 else 0.0
    return {"slots": int(len(x)), "median_n": float(np.median(n)), "slope_keep": b, "se": se,
            "record_sd_pp": float(100 * x.std()), "tau_pp": float(100 * np.sqrt(max(tau2, 0))),
            "formula_keep_measured_tau": pred, "formula_keep_k150": float(np.mean(n / (n + 150.0))),
            "persistent_sd_pp": float(100 * np.sqrt(max(np.cov(x, r)[0, 1], 0)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    a_ = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    keys, reg, tier, mmr = game_meta()
    out = {"method": __doc__.split("Both arms")[1].split("Usage")[0].strip()}

    # ---------------------------------------------------------------- Q3
    A = load(os.path.join(a_.arms, "A2S.npz"))
    q = A["s_win"] >= 2
    n = A["s_nh"].astype(np.float64)
    x = np.where(n > 0, A["s_rec"] / np.maximum(n, 1), np.nan)
    r = A["s_r"].astype(np.float64)
    grp = ((A["s_key"] * 2654435761) % 2 ** 32) % NG
    s_reg_s, s_tier_s, _ = lookup(keys, A["s_replay_id"], reg, tier, mmr)
    q3 = {}
    for (lo, hi), lab in zip(NBINS, NLAB):
        m = q & (n >= lo) & (n <= hi)
        q3[lab] = persistence(x[m], r[m], n[m], grp[m])
    # direct worked example: records of +10 to +14pp at about 80 and about 230 games
    ex = {}
    for lab, (lo, hi) in {"70-90 games": (70, 90), "200-260 games": (200, 260),
                          "60-119 games": (60, 119), "120-249 games": (120, 249)}.items():
        for rl, (r0, r1) in {"+10 to +14pp": (0.10, 0.14), "-14 to -10pp": (-0.14, -0.10)}.items():
            m = q & (n >= lo) & (n <= hi) & (x >= r0) & (x < r1)
            if m.sum() < 200:
                continue
            ex[f"{lab}, record {rl}"] = {"slots": int(m.sum()), "record_pp": float(100 * x[m].mean()),
                                        "realized_pp": float(100 * r[m].mean()),
                                        "realized_se_pp": float(100 * r[m].std() / np.sqrt(m.sum())),
                                        "keep": float(r[m].mean() / x[m].mean())}
    # by tier, n >= 30
    q3t = {}
    for t, tn in TIERS.items():
        m = q & (n >= 30) & (s_tier_s == t)
        if m.sum() > 1000:
            q3t[tn] = persistence(x[m], r[m], n[m], grp[m])
    # fitted k for keep(n) = n / (n + k) on the bin slopes
    ks = np.arange(10, 1000, 1.0)
    sl = np.array([q3[l]["slope_keep"] for l in NLAB])
    mn = np.array([q3[l]["median_n"] for l in NLAB])
    w = 1 / np.array([q3[l]["se"] for l in NLAB]) ** 2
    sse = [(w * (sl - mn / (mn + k)) ** 2).sum() for k in ks]
    kfit = float(ks[int(np.argmin(sse))])
    out["q3"] = {"by_games_on_hero": q3, "worked_examples": ex, "by_tier_n30plus": q3t, "k_fit": kfit,
                 "keep_at_80_fit": 80 / (80 + kfit), "keep_at_230_fit": 230 / (230 + kfit),
                 "keep_at_80_k150": 80 / 230, "keep_at_230_k150": 230 / 380}
    print(json.dumps(out["q3"], indent=1)[:4000], flush=True)
    del A

    # ---------------------------------------------------------------- Q5 and Q6, main model
    for arm in ("full_ms_layer", "A2S"):
        M = load(os.path.join(a_.arms, f"{arm}.npz"))
        g_reg, g_tier, g_mmr = lookup(keys, M["g_replay_id"], reg, tier, mmr)
        res = {}
        for wn, wv in (("V2", 2), ("OOT", 3)):
            base = M["g_win"] == wv
            res[wn] = {"all": gsummary(M, base)}
            for rv, rn in REGION.items():
                res[wn][f"region {rn}"] = gsummary(M, base & (g_reg == rv))
            res[wn]["region unknown"] = int((base & ~np.isin(g_reg, list(REGION))).sum())
            for t, tn in TIERS.items():
                res[wn][f"tier {tn}"] = gsummary(M, base & (g_tier == t))
            cut = np.nanpercentile(g_mmr[base], 99)
            res[wn]["top 1% lobby MMR"] = gsummary(M, base & (g_mmr >= cut))
            res[wn]["top 1% lobby MMR"]["mmr_cut"] = float(cut)
            # spread of the personal signal by tier: sd of D (pp) and of slot estimates
            gi, ok = game_index(M)
            sq = M["s_win"] == wv
            s_tier = np.where(ok, g_tier[gi], -1)
            s_mmr = np.where(ok, g_mmr[gi], np.nan)
            spread = {}
            for t, tn in list(TIERS.items()) + [(99, "top 1% lobby MMR")]:
                gm = base & ((g_mmr >= cut) if t == 99 else (g_tier == t))
                sm = sq & ((s_mmr >= cut) if t == 99 else (s_tier == t))
                spread[tn] = {"sd_team_diff_pp": float(100 * M["g_D"][gm].std()),
                              "sd_slot_estimate_pp": float(100 * M["s_pred"][sm].std()),
                              "p90_abs_slot_estimate_pp": float(100 * np.percentile(np.abs(M["s_pred"][sm]), 90))}
            res[wn]["spread"] = spread
            # concentration: keep the estimate only on chosen slots
            full_gain = res[wn]["all"]["ll_gain_x1000"]
            nh = M["s_nh"].astype(np.float64)
            npl = M["s_np"].astype(np.float64)
            share = np.where(npl >= 100, nh / np.maximum(npl, 1), 0.0)
            conc = {}
            sets = {}
            for pct in (1, 5, 20):
                cn = np.percentile(nh[sq], 100 - pct)
                cs = np.percentile(share[sq], 100 - pct)
                ca = np.percentile(np.abs(M["s_pred"][sq]), 100 - pct)
                sets[f"top {pct}% of slots by games on the hero (>= {cn:.0f})"] = sq & (nh >= cn)
                sets[f"top {pct}% of slots by share of the player's games on the hero (>= {100 * cs:.0f}%)"] = sq & (share >= cs) & (share > 0)
                sets[f"top {pct}% of slots by size of the personal estimate (>= {100 * ca:.1f}pp)"] = sq & (np.abs(M["s_pred"]) >= ca)
            sets["specialist: 100+ games on the hero and 20%+ of the player's games"] = sq & (nh >= 100) & (nh >= 0.2 * npl)
            for nm, keep in sets.items():
                p_only = counterfactual(M, keep, gi, ok)
                p_wo = counterfactual(M, sq & ~keep, gi, ok)
                g_only = gsummary(M, base, p_only)["ll_gain_x1000"]
                g_wo = gsummary(M, base, p_wo)["ll_gain_x1000"]
                hit = np.zeros(len(base), bool)
                hit[gi[ok & keep]] = True
                conc[nm] = {"share_of_slots": float(keep.sum() / sq.sum()), "share_of_games_touched": float((hit & base).sum() / base.sum()),
                            "gain_only_these_x1000": g_only, "share_of_full_gain_only_these": g_only / full_gain,
                            "gain_without_these_x1000": g_wo, "share_lost_without_these": 1 - g_wo / full_gain,
                            "mean_realized_r_pp": float(100 * M["s_r"][keep].mean()),
                            "mean_estimate_pp": float(100 * M["s_pred"][keep].mean())}
            res[wn]["concentration"] = conc
        out[arm] = res
        print(arm, json.dumps({w: {k: v for k, v in res[w].items() if k.startswith(("all", "region"))} for w in res}, indent=1)[:3000], flush=True)
    json.dump(out, open(os.path.join(OUT, "arms.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
