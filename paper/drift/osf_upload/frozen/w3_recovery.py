"""
W3(a): POST-PATCH RECOVERY — how many games into a build until patch-local
rolling estimates are trustworthy, per signal class?

For every sizable build, games are walked in chronological order
(date_days, replay_id). At log-spaced game-count checkpoints the rolling
(games-so-far) estimate of each signal class is compared against

  vs_final  the build's END-OF-BUILD values (the spec's definition: "within
            X of the build's final values"; rolling games are a subset of
            final games, so this converges to 0 mechanically)
  vs_rest   the value measured on the REMAINING games only (disjoint;
            measures how well early-build stats predict the rest — the
            honest deployment question)

Error metric = games-weighted mean |rolling - reference| in pp, over keys
that clear per-class reliability minimums, plus the expected MAE from pure
binomial sampling under stationarity (to show whether recovery is
sampling-limited rather than drift-limited).

Signal classes (pooled tiers, matching signal_decay.py):
  hero_wr, hero_pickrate, pair_with (raw WR), pair_against (raw WR), comp_wr

Outputs:
  results/w3_recovery.json      curves + per-build crossings + the
                                hybrid window answer consumed by W3(d)
  results/W3_RECOVERY.md        recovery-time table + curves
  daily_hero_counts.pkl.gz      per-build per-day hero pick/win/ban counts
                                (pooled + per-tier + per-map), consumed by
                                w3_changepoints.py and w3_heterogeneity.py

Usage:
    python drift2026/w3_recovery.py [--workers N]
"""
import os
import sys
import gzip
import json
import time
import pickle
import argparse
import multiprocessing as mp
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common
from drift2026.build_patch_stats import comp_key

import numpy as np

NUM_WORKERS = min(mp.cpu_count(), 28)
DAILY_CACHE = os.path.join(common.DRIFT_DIR, "daily_hero_counts.pkl.gz")

CHECKPOINTS = [500, 1000, 2000, 3000, 5000, 8000, 12000, 20000,
               30000, 50000, 80000, 120000]

# per-class config: (final-games min, rolling min, rest min, threshold X pp)
CLASSES = {
    "hero_wr":       dict(min_final=300, min_roll=30, min_rest=100, X=1.0),
    "hero_pickrate": dict(min_final=0,   min_roll=0,  min_rest=0,   X=0.25),
    "pair_with":     dict(min_final=100, min_roll=30, min_rest=50,  X=2.0),
    "pair_against":  dict(min_final=100, min_roll=30, min_rest=50,  X=2.0),
    "comp_wr":       dict(min_final=300, min_roll=30, min_rest=100, X=0.5),
}


def _wmae(diffs, w):
    w = np.asarray(w, dtype=np.float64)
    return float(np.sum(np.abs(diffs) * w) / np.sum(w))


def _class_errors(name, roll, final, n_roll, n_final):
    """roll/final: {key: [g, w]} (for pickrate w unused). Returns dict with
    mae_vs_final, mae_vs_rest, expected MAEs, coverage."""
    cfg = CLASSES[name]
    is_rate = name == "hero_pickrate"
    dv_f, dv_r, wf, wr_, ef, er = [], [], [], [], [], []
    covered = total_w = 0.0
    for k, (gf, wfin) in final.items():
        if gf < cfg["min_final"]:
            continue
        total_w += gf
        gr, wroll = roll.get(k, (0, 0))
        if is_rate:
            v_roll = 100.0 * gr / n_roll if n_roll else 0.0
            v_fin = 100.0 * gf / n_final
            p = max(v_fin / 100.0, 1e-4)
            se_f = np.sqrt(p * (1 - p) * abs(1.0 / max(n_roll, 1) - 1.0 / n_final))
            dv_f.append(v_roll - v_fin)
            wf.append(gf)
            ef.append(se_f * 100)
            n_rest = n_final - n_roll
            if n_rest > 0:
                v_rest = 100.0 * (gf - gr) / n_rest
                dv_r.append(v_roll - v_rest)
                wr_.append(gf)
                er.append(np.sqrt(p * (1 - p) * (1.0 / max(n_roll, 1) + 1.0 / n_rest)) * 100)
            covered += gf
            continue
        if gr < cfg["min_roll"]:
            continue
        covered += gf
        v_roll = 100.0 * wroll / gr
        v_fin = 100.0 * wfin / gf
        p = min(max(v_fin / 100.0, 0.05), 0.95)
        dv_f.append(v_roll - v_fin)
        wf.append(gf)
        ef.append(np.sqrt(p * (1 - p) * abs(1.0 / gr - 1.0 / gf)) * 100)
        g_rest, w_rest = gf - gr, wfin - wroll
        if g_rest >= cfg["min_rest"]:
            dv_r.append(v_roll - 100.0 * w_rest / g_rest)
            wr_.append(min(gr, g_rest))
            er.append(np.sqrt(p * (1 - p) * (1.0 / gr + 1.0 / g_rest)) * 100)
    out = {"n_keys": len(dv_f),
           "coverage": round(covered / total_w, 3) if total_w else None}
    if dv_f:
        out["mae_vs_final"] = round(_wmae(np.array(dv_f), wf), 3)
        out["exp_mae_vs_final"] = round(_wmae(np.array(ef) * np.sqrt(2 / np.pi) / 1.0,
                                              wf), 3)
    if dv_r:
        out["mae_vs_rest"] = round(_wmae(np.array(dv_r), wr_), 3)
        out["exp_mae_vs_rest"] = round(_wmae(np.array(er) * np.sqrt(2 / np.pi),
                                             wr_), 3)
    return out


def walk_build(args):
    """One build: chronological walk. Returns (bidx, checkpoint_records,
    daily_cache_entry). checkpoint_records only for sizable builds."""
    bidx, rows, sizable = args
    hero = defaultdict(lambda: [0, 0])
    pw = defaultdict(lambda: [0, 0])
    pa = defaultdict(lambda: [0, 0])
    comp = defaultdict(lambda: [0, 0])
    daily = {}   # day -> {"games", "hero", "bans", "tier_hero", "map_hero"}
    snaps = []   # (n, copies) at checkpoints
    cps = [c for c in CHECKPOINTS if c < len(rows)] if sizable else []
    ci = 0
    n = 0
    for d in rows:
        day, tier, gmap, t0, t1, bans, winner = d
        n += 1
        de = daily.get(day)
        if de is None:
            de = daily[day] = {"games": 0, "hero": defaultdict(lambda: [0, 0]),
                               "bans": defaultdict(int),
                               "tier_hero": defaultdict(lambda: defaultdict(lambda: [0, 0])),
                               "map_hero": defaultdict(lambda: defaultdict(lambda: [0, 0]))}
        de["games"] += 1
        for ti, team in ((0, t0), (1, t1)):
            won = 1 if winner == ti else 0
            th = de["tier_hero"][tier]
            mh = de["map_hero"][gmap]
            for h in team:
                e = hero[h]
                e[0] += 1
                e[1] += won
                e = de["hero"].setdefault(h, [0, 0])
                e[0] += 1
                e[1] += won
                e = th[h]
                e[0] += 1
                e[1] += won
                e = mh[h]
                e[0] += 1
                e[1] += won
            for i in range(5):
                for j in range(i + 1, 5):
                    a, b = team[i], team[j]
                    if a > b:
                        a, b = b, a
                    e = pw[(a, b)]
                    e[0] += 1
                    e[1] += won
            e = comp[comp_key(team)]
            e[0] += 1
            e[1] += won
        t0won = 1 if winner == 0 else 0
        for ha in t0:
            for hb in t1:
                if ha < hb:
                    e = pa[(ha, hb)]
                    e[0] += 1
                    e[1] += t0won
                else:
                    e = pa[(hb, ha)]
                    e[0] += 1
                    e[1] += 1 - t0won
        for h in set(bans):
            de["bans"][h] += 1
        if ci < len(cps) and n == cps[ci]:
            snaps.append((n, {k: dict((kk, list(vv)) for kk, vv in d2.items())
                              for k, d2 in (("hero", hero), ("pw", pw),
                                            ("pa", pa), ("comp", comp))}))
            ci += 1

    finals = {"hero": hero, "pw": pw, "pa": pa, "comp": comp}
    records = []
    for cn, s in snaps:
        rec = {"games": cn}
        rec["hero_wr"] = _class_errors("hero_wr", s["hero"], hero, cn, n)
        rec["hero_pickrate"] = _class_errors("hero_pickrate", s["hero"], hero, cn, n)
        rec["pair_with"] = _class_errors("pair_with", s["pw"], pw, cn, n)
        rec["pair_against"] = _class_errors("pair_against", s["pa"], pa, cn, n)
        rec["comp_wr"] = _class_errors("comp_wr", s["comp"], comp, cn, n)
        records.append(rec)

    # plain-dict daily cache entry
    plain = {}
    for day, de in daily.items():
        plain[day] = {
            "games": de["games"],
            "hero": {h: list(v) for h, v in de["hero"].items()},
            "bans": dict(de["bans"]),
            "tier_hero": {t: {h: list(v) for h, v in d2.items()}
                          for t, d2 in de["tier_hero"].items()},
            "map_hero": {m: {h: list(v) for h, v in d2.items()}
                         for m, d2 in de["map_hero"].items()},
        }
    return bidx, n, records, plain


def crossing(curve, X, min_cov=0.5):
    """First games count where MAE <= X, log-interpolated between
    checkpoints. curve: list of (games, mae, coverage). Checkpoints whose
    rolling estimates cover < min_cov of the (games-weighted) final keys are
    excluded — early low-coverage points measure only the most popular keys
    and systematically understate the error. None if never."""
    pts = [(g, c) for g, c, cov in curve
           if c is not None and (cov is None or cov >= min_cov)]
    for i, (g, c) in enumerate(pts):
        if c <= X:
            if i == 0:
                return g
            g0, c0 = pts[i - 1]
            if c0 == c:
                return g
            f = (c0 - X) / (c0 - c)
            return float(np.exp(np.log(g0) + f * (np.log(g) - np.log(g0))))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--from-json", action="store_true",
                    help="re-aggregate from results/w3_recovery.json "
                         "per_build_curves (no data walk, no cache rewrite)")
    args = ap.parse_args()
    common.setup()

    if args.from_json:
        with open(os.path.join(common.RESULTS_DIR, "w3_recovery.json")) as f:
            prev = json.load(f)
        builds = common.load_patch_index()["builds"]
        per_build_records = {builds.index(b): recs
                             for b, recs in prev["per_build_curves"].items()}
        return finalize(per_build_records, builds)

    rows, builds = common.load_data_with_patches()
    by_build = defaultdict(list)
    for r in rows:
        by_build[r["build_idx"]].append(
            (r["date_days"], r["replay_id"], r["skill_tier"], r["game_map"],
             tuple(r["team0_heroes"]), tuple(r["team1_heroes"]),
             tuple(r["team0_bans"]) + tuple(r["team1_bans"]), r["winner"]))
    del rows
    for bi in by_build:
        by_build[bi].sort(key=lambda t: (t[0], t[1]))

    tasks = []
    for bi in sorted(by_build):
        n = len(by_build[bi])
        slim = [(t[0], t[2], t[3], t[4], t[5], t[6], t[7]) for t in by_build[bi]]
        tasks.append((bi, slim, n >= common.SIZABLE_GAMES))
    tasks.sort(key=lambda t: -len(t[1]))
    print(f"{len(tasks)} builds ({sum(1 for t in tasks if t[2])} sizable)")

    t0 = time.time()
    per_build_records, daily_cache, build_games = {}, {}, {}
    with mp.Pool(args.workers) as pool:
        for i, (bidx, n, records, plain) in enumerate(
                pool.imap_unordered(walk_build, tasks, chunksize=1)):
            if records:
                per_build_records[bidx] = records
            daily_cache[bidx] = plain
            build_games[bidx] = n
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                print(f"  {i+1}/{len(tasks)} builds ({time.time()-t0:.0f}s)",
                      flush=True)

    with gzip.open(DAILY_CACHE, "wb") as f:
        pickle.dump({"builds": builds, "per_build": daily_cache,
                     "build_games": build_games}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {DAILY_CACHE} ({os.path.getsize(DAILY_CACHE)/1e6:.1f} MB)")
    return finalize(per_build_records, builds)


def finalize(per_build_records, builds):
    """Aggregate curves + crossings -> results JSON + markdown."""
    classes = list(CLASSES)
    curves_med = {c: {} for c in classes}     # class -> games -> median MAE
    for c in classes:
        for cp in CHECKPOINTS:
            vals_f, vals_r, exps_f = [], [], []
            for bidx, recs in per_build_records.items():
                for rec in recs:
                    if rec["games"] == cp and rec[c].get("mae_vs_final") is not None:
                        vals_f.append(rec[c]["mae_vs_final"])
                        exps_f.append(rec[c].get("exp_mae_vs_final"))
                        if rec[c].get("mae_vs_rest") is not None:
                            vals_r.append(rec[c]["mae_vs_rest"])
            if vals_f:
                curves_med[c][cp] = {
                    "n_builds": len(vals_f),
                    "mae_vs_final_med": round(float(np.median(vals_f)), 3),
                    "exp_vs_final_med": round(float(np.median(
                        [e for e in exps_f if e is not None] or [np.nan])), 3),
                    "mae_vs_rest_med": round(float(np.median(vals_r)), 3)
                    if vals_r else None,
                }

    recovery = {}
    for c in classes:
        X = CLASSES[c]["X"]
        per_b_f, per_b_r = {}, {}
        for bidx, recs in per_build_records.items():
            cur_f = [(r["games"], r[c].get("mae_vs_final"),
                      r[c].get("coverage")) for r in recs]
            cur_r = [(r["games"], r[c].get("mae_vs_rest"),
                      r[c].get("coverage")) for r in recs]
            xf = crossing(cur_f, X)
            xr = crossing(cur_r, X)
            per_b_f[builds[bidx]] = round(xf) if xf else None
            per_b_r[builds[bidx]] = round(xr) if xr else None
        got_f = [v for v in per_b_f.values() if v]
        got_r = [v for v in per_b_r.values() if v]
        recovery[c] = {
            "threshold_pp": X,
            "vs_final": {"median_games": int(np.median(got_f)) if got_f else None,
                         "p75_games": int(np.percentile(got_f, 75)) if got_f else None,
                         "n_reached": len(got_f),
                         "n_builds": len(per_b_f),
                         "per_build": per_b_f},
            "vs_rest": {"median_games": int(np.median(got_r)) if got_r else None,
                        "p75_games": int(np.percentile(got_r, 75)) if got_r else None,
                        "n_reached": len(got_r),
                        "per_build": per_b_r},
        }

    # W3(d) hybrid window: games of recent history needed for a trustworthy
    # hero-WR table (1pp, vs final; p75 across builds = conservative).
    hw = recovery["hero_wr"]["vs_final"]["p75_games"] or 30000
    out = {"config": {"checkpoints": CHECKPOINTS,
                      "classes": {c: CLASSES[c] for c in classes},
                      "sizable_games": common.SIZABLE_GAMES},
           "hybrid_window_games": int(hw),
           "recovery": recovery,
           "curves_median": curves_med,
           "per_build_curves": {builds[b]: r
                                for b, r in per_build_records.items()}}
    common.write_json(os.path.join(common.RESULTS_DIR, "w3_recovery.json"), out)

    # ── markdown ──
    lines = [
        "# W3(a) — Post-patch recovery: when can you trust patch-local stats?",
        "",
        "Within each sizable build (n=%d), rolling estimates at game-count"
        % len(per_build_records),
        "checkpoints vs the build's final values (spec definition) and vs the",
        "disjoint remainder (deployment-honest). Games-weighted MAE in pp over",
        "keys clearing per-class game minimums; 'expected' = MAE from binomial",
        "sampling alone under within-build stationarity.",
        "",
        "## Recovery thresholds (games into the build until MAE <= X)",
        "",
        "| signal class | X (pp) | median games (vs final) | p75 | builds "
        "reaching | median games (vs rest) |",
        "|---|---|---|---|---|---|",
    ]
    for c in classes:
        r = recovery[c]
        f, rr = r["vs_final"], r["vs_rest"]
        fm = f"{f['median_games']:,}" if f["median_games"] else "never"
        fp = f"{f['p75_games']:,}" if f["p75_games"] else "-"
        rm = f"{rr['median_games']:,}" if rr["median_games"] else "never"
        lines.append(f"| {c} | {r['threshold_pp']} | {fm} | {fp} | "
                     f"{f['n_reached']}/{f['n_builds']} | {rm} |")
    lines += [
        "",
        "## Median MAE curves (vs final / expected-from-sampling / vs rest)",
        "",
        "| games | " + " | ".join(classes) + " |",
        "|---|" + "---|" * len(classes),
    ]
    for cp in CHECKPOINTS:
        cells = []
        for c in classes:
            e = curves_med[c].get(cp)
            cells.append(f"{e['mae_vs_final_med']:.2f} / "
                         f"{e['exp_vs_final_med']:.2f} / " +
                         (f"{e['mae_vs_rest_med']:.2f}"
                          if e["mae_vs_rest_med"] is not None else "-")
                         if e else "-")
        lines.append(f"| {cp:,} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"**W3(d) hybrid window answer: {hw:,} games** of recent history for "
        "the hero-WR family (p75 of the 1pp vs-final crossing across sizable "
        "builds).",
        "",
        "Reading notes: vs-final converges to 0 mechanically (rolling games "
        "are a subset); vs-rest is bounded below by ~sqrt(2)x the one-sided "
        "sampling noise. Where observed MAE tracks the expected-from-sampling "
        "column, recovery is sampling-limited, not drift-limited — i.e. the "
        "estimate is trustworthy as soon as it is statistically stable.",
    ]
    md = os.path.join(common.RESULTS_DIR, "W3_RECOVERY.md")
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {md}")
    for line in lines[8:16]:
        print(line)


if __name__ == "__main__":
    main()
