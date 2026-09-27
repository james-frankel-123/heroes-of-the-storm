"""
PHASE D2(a): SIGNAL DECAY — how fast does each statistic class rot?

For every ordered pair of sizable builds (N, N+k) correlate the stat vector
measured at build N against the realized ("ground truth") values at build
N+k, per signal class:

  hero_wr        per-hero win rate (pooled tiers, >= MIN_HERO games/build)
  hero_pickrate  per-hero pick rate (meta drift; all heroes)
  pair_with_raw  teammate-pair WR (>= MIN_PAIR games in both builds)
  pair_with_net  same, net of the two heroes' solo WRs (the feature the
                 enriched extractor actually uses: raw - expected)
  pair_against_raw / pair_against_net   cross-team analogues
  comp_wr        role-composition WR (>= MIN_COMP games in both builds)

Small samples attenuate correlations, so alongside raw (games-weighted)
Pearson r we report a disattenuated r: each build's measurement reliability
is estimated as (observed variance - mean binomial sampling variance) /
observed variance, and r_adj = r / sqrt(rel_N * rel_N+k). An exponential
decay r(dt) = exp(-dt / tau) is fitted per signal over day-lags to give a
half-life (tau * ln 2).

Pure computation from patch_stats/patch_counts.pkl.gz — no training, no DB.

Output: results/signal_decay.json + results/SIGNAL_DECAY.md summary table.
"""
import os
import sys
import gzip
import json
import pickle
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

import numpy as np

MIN_HERO = 300
MIN_PAIR = 100
MIN_PAIR_NET = 300   # net-of-solo-WR pair signals are noise-dominated below this
MIN_COMP = 300
MIN_RELIABILITY = 0.15   # below this, disattenuated r is meaningless -> None


def pool_tiers(by_tier):
    """Merge the per-tier count cells of one build into a single cell."""
    cell = {"games": 0, "hero": defaultdict(lambda: [0, 0]),
            "with": defaultdict(lambda: [0, 0]),
            "against": defaultdict(lambda: [0, 0]),
            "comp": defaultdict(lambda: [0, 0])}
    for t, c in by_tier.items():
        cell["games"] += c["games"]
        for k in ("hero", "with", "against", "comp"):
            d = cell[k]
            for key, (g, w) in c[k].items():
                e = d[key]
                e[0] += g
                e[1] += w
    return cell


def signals_for(cell):
    """signal name -> {key: (value, games)}."""
    total = cell["games"]
    hero_wr, hero_pr = {}, {}
    for h, (g, w) in cell["hero"].items():
        hero_pr[h] = (100.0 * g / total, g)
        if g >= MIN_HERO:
            hero_wr[h] = (100.0 * w / g, g)

    def solo(h):
        v = hero_wr.get(h)
        return v[0] if v else None

    pw_raw, pw_net, pa_raw, pa_net = {}, {}, {}, {}
    for (a, b), (g, w) in cell["with"].items():
        if g < MIN_PAIR:
            continue
        wr = 100.0 * w / g
        pw_raw[(a, b)] = (wr, g)
        sa, sb = solo(a), solo(b)
        if g >= MIN_PAIR_NET and sa is not None and sb is not None:
            pw_net[(a, b)] = (wr - (50 + (sa - 50) + (sb - 50)), g)
    for (a, b), (g, wa) in cell["against"].items():
        if g < MIN_PAIR:
            continue
        wr = 100.0 * wa / g
        pa_raw[(a, b)] = (wr, g)
        sa, sb = solo(a), solo(b)
        if g >= MIN_PAIR_NET and sa is not None and sb is not None:
            pa_net[(a, b)] = (wr - (sa + (100 - sb) - 50), g)
    comp = {k: (100.0 * w / g, g) for k, (g, w) in cell["comp"].items()
            if g >= MIN_COMP}
    return {"hero_wr": hero_wr, "hero_pickrate": hero_pr,
            "pair_with_raw": pw_raw, "pair_with_net": pw_net,
            "pair_against_raw": pa_raw, "pair_against_net": pa_net,
            "comp_wr": comp}


def _wmean(v, w):
    return float(np.sum(v * w) / np.sum(w))


def weighted_corr(x, y, w):
    mx, my = _wmean(x, w), _wmean(y, w)
    cov = _wmean((x - mx) * (y - my), w)
    vx, vy = _wmean((x - mx) ** 2, w), _wmean((y - my) ** 2, w)
    if vx <= 0 or vy <= 0:
        return 0.0
    return float(cov / np.sqrt(vx * vy))


def reliability(vals, games, is_rate=True):
    """Fraction of observed variance that is signal, not binomial noise.
    Sampling var of a WR-like % with g games ~ 2500/g (p~0.5 upper bound);
    for pick rates use p*(1-p) at the observed rate."""
    v = np.asarray(vals, dtype=np.float64)
    g = np.asarray(games, dtype=np.float64)
    w = g
    mv = _wmean(v, w)
    var_obs = _wmean((v - mv) ** 2, w)
    if var_obs <= 0:
        return 0.0
    if is_rate:
        p = np.clip(v / 100.0, 1e-4, 1 - 1e-4)
        se2 = p * (1 - p) * 1e4 / g
    else:
        se2 = 2500.0 / g
    rel = (var_obs - _wmean(se2, w)) / var_obs
    return float(max(0.0, min(1.0, rel)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()
    common.setup()

    with gzip.open(common.COUNTS_PKL, "rb") as f:
        data = pickle.load(f)
    builds, per_build = data["builds"], data["per_build"]

    # median game date per build (for day lags)
    ids, bidx, days, _ = common.load_sidecar()
    med_day = {}
    for bi in np.unique(bidx):
        med_day[int(bi)] = float(np.median(days[bidx == bi]))

    sizable = []
    for bi in sorted(per_build):
        cell = pool_tiers(per_build[bi])
        if cell["games"] >= common.SIZABLE_GAMES:
            sizable.append((bi, cell))
    print(f"{len(sizable)} sizable builds (>= {common.SIZABLE_GAMES} games)")

    sigs = [(bi, signals_for(cell), cell["games"]) for bi, cell in sizable]
    signal_names = list(sigs[0][1].keys())

    pair_records = []
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            bi, si, gi = sigs[i]
            bj, sj, gj = sigs[j]
            rec = {"build_n": builds[bi], "build_nk": builds[bj],
                   "lag_builds": j - i,
                   "lag_days": round(med_day[bj] - med_day[bi], 1)}
            for name in signal_names:
                # net signals subtract a same-build WR expectation, so their
                # residual noise isn't cleanly binomial; use the WR bound.
                keys = sorted(set(si[name]) & set(sj[name]))
                if len(keys) < 8:
                    rec[name] = None
                    continue
                x = np.array([si[name][k][0] for k in keys])
                y = np.array([sj[name][k][0] for k in keys])
                gx = np.array([si[name][k][1] for k in keys], dtype=np.float64)
                gy = np.array([sj[name][k][1] for k in keys], dtype=np.float64)
                w = np.minimum(gx, gy)
                r = weighted_corr(x, y, w)
                is_rate = not name.endswith("_net")
                rel_i = reliability(x, gx, is_rate)
                rel_j = reliability(y, gy, is_rate)
                if min(rel_i, rel_j) >= MIN_RELIABILITY:
                    r_adj = float(np.clip(r / np.sqrt(rel_i * rel_j), -1, 1))
                else:
                    r_adj = None   # measurement noise dominates; disattenuation unstable
                rec[name] = {"r": round(r, 4),
                             "r_adj": round(r_adj, 4) if r_adj is not None else None,
                             "rel_min": round(min(rel_i, rel_j), 3),
                             "n": len(keys),
                             "mae": round(float(_wmean(np.abs(x - y), w)), 3)}
            pair_records.append(rec)

    # aggregate per build-lag
    by_lag = {}
    for name in signal_names:
        agg = defaultdict(list)
        for rec in pair_records:
            if rec[name]:
                agg[rec["lag_builds"]].append(rec[name])
        by_lag[name] = {
            str(k): {"r_mean": round(float(np.mean([e["r"] for e in v])), 4),
                     "r_adj_mean": round(float(np.mean(
                         [e["r_adj"] for e in v if e["r_adj"] is not None] or [0])), 4),
                     "n_pairs": len(v)}
            for k, v in sorted(agg.items())}

    # decay fit over day lags: r(dt) = floor + amp * exp(-dt/tau).
    # Correlations plateau well above zero (a stable "structural" component),
    # so a pure exponential is misspecified; the interesting quantity is the
    # half-life of the DECAYING component plus the floor it decays to.
    half_lives = {}
    for name in signal_names:
        dts, rs = [], []
        for rec in pair_records:
            e = rec[name]
            rv = e["r_adj"] if (e and e["r_adj"] is not None) else (e["r"] if e else None)
            if rv is not None:
                dts.append(rec["lag_days"])
                rs.append(rv)
        if len(dts) >= 10:
            dts, rs = np.array(dts, dtype=np.float64), np.array(rs, dtype=np.float64)
            used = "r_adj" if any(rec[name] and rec[name]["r_adj"] is not None
                                  for rec in pair_records) else "r"
            # grid over tau; for fixed tau the model is linear in (floor, amp)
            best = None
            for tau in np.geomspace(10, 5000, 80):
                e = np.exp(-dts / tau)
                A = np.vstack([np.ones_like(e), e]).T
                (floor, amp), res, _, _ = np.linalg.lstsq(A, rs, rcond=None)
                if floor < 0 or amp < 0:
                    continue
                sse = float(np.sum((A @ np.array([floor, amp]) - rs) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, float(floor), float(amp), float(tau))
            if best:
                sse, floor, amp, tau = best
                half_lives[name] = {
                    "model": "floor + amp*exp(-dt/tau), tau grid + LS",
                    "fit_on": used,
                    "floor": round(floor, 3), "amp": round(amp, 3),
                    "tau_days": round(tau, 1),
                    "half_life_days": round(tau * np.log(2), 1),
                    "sse": round(sse, 4), "n_points": len(dts)}
            else:
                half_lives[name] = {"error": "no valid fit", "n_points": len(dts)}
        else:
            half_lives[name] = None

    out = {"config": {"sizable_games": common.SIZABLE_GAMES,
                      "min_hero": MIN_HERO, "min_pair": MIN_PAIR,
                      "min_comp": MIN_COMP,
                      "sizable_builds": [builds[bi] for bi, _ in sizable]},
           "half_lives": half_lives, "by_lag_builds": by_lag,
           "pairs": pair_records}
    common.write_json(os.path.join(common.RESULTS_DIR, "signal_decay.json"), out)

    # markdown summary
    lines = ["# Signal decay (D2a)", "",
             "Games-weighted Pearson r between build-N stats and build-N+k stats,",
             "averaged over all sizable-build pairs at each lag; r_adj corrects for",
             "binomial sampling attenuation. half-life from exp fit over day-lags.",
             "", "| signal | r@1 | r_adj@1 | r@3 | r_adj@3 | r@8 | r_adj@8 | r@16 | r_adj@16 | half-life (d) | floor |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in signal_names:
        cells = []
        for lag in ("1", "3", "8", "16"):
            e = by_lag[name].get(lag)
            cells += ([f"{e['r_mean']:.3f}",
                       f"{e['r_adj_mean']:.3f}" if e["r_adj_mean"] else "-"]
                      if e else ["-", "-"])
        hl = half_lives[name]
        ok = hl and hl.get("half_life_days")
        hl_s = f"{hl['half_life_days']}" if ok else "-"
        fl_s = f"{hl['floor']:.2f}" if ok else "-"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {hl_s} | {fl_s} |")
    md = os.path.join(common.RESULTS_DIR, "SIGNAL_DECAY.md")
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {md}")
    for line in lines[5:]:
        print(line)


if __name__ == "__main__":
    main()
