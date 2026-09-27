"""
W3(b): HERO-LEVEL CHANGEPOINT DETECTION at build boundaries + validation
against official patch notes.

For every boundary between consecutive SIZABLE builds (prev -> new; tiny
hotfix builds in between are folded into the boundary), per hero:

  boundary-adjacent tests (full builds on both sides):
    WR      two-proportion z on hero win rate (requires >= MIN_G games/side)
    pick    two-proportion z on pick rate (games with hero / build games)
    ban     two-proportion z on ban rate
  multiplicity: Benjamini-Hochberg within (boundary x test family).

  detection latency (game-indexed, evaluated at day boundaries of the new
  build, for heroes the full-build test flags):
    z_naive  first day where the two-prop z vs the prev build has p < 0.05
             (repeated looks -> anti-conservative; reported as the naive
             "flag at 95%" number the spec asks for)
    z_bonf   same with p < 0.05 / n_days (look-corrected)
    sprt     two one-sided Wald SPRTs on the hero's game outcomes with
             H0: p = p0 (prev-build WR), H1: p = p0 +/- DELTA; flag when
             either LLR crosses ln(19) (~95%). Game-indexed and
             anytime-valid up to the day-granularity of the cache.
  latencies are reported in BUILD games elapsed (deployment clock) and
  hero games elapsed.

VALIDATION: if patch_notes_ground_truth.json exists (fetched by the
patch-notes agent; see its "scriptable"/"notes" fields), each boundary's
ground-truth changed-hero set = union of patch-note hero lists mapped to any
build in (prev, new]. Precision/recall/F1 per detector variant, micro-
averaged over validatable boundaries. Skipped gracefully if the file is
missing or empty.

Inputs: daily_hero_counts.pkl.gz (from w3_recovery.py).
Outputs: results/w3_changepoints.json + results/W3_CHANGEPOINTS.md.
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
from drift2026.w3_recovery import DAILY_CACHE

import math

import numpy as np


def _p_two_sided(z):
    return math.erfc(abs(z) / math.sqrt(2.0))

GT_PATH = os.path.join(common.DRIFT_DIR, "patch_notes_ground_truth.json")
MIN_G = 200          # min hero games per side for the WR test
DELTA = 0.03         # SPRT alternative: |WR shift| = 3pp
SPRT_H = np.log(19)  # ~95%
EFF_WR = 1.5         # effect floors (pp) for the *_eff variant
EFF_RATE = 0.5

VARIANTS = ["wr_p05", "wr_p05_eff", "wr_bh", "pickban_bh", "pickban_eff2",
            "any_bh", "any_bh_eff"]


def two_prop(x0, n0, x1, n1):
    """Two-sided two-proportion z-test. Returns (delta_pp, z, p)."""
    if n0 == 0 or n1 == 0:
        return None
    p0, p1 = x0 / n0, x1 / n1
    p = (x0 + x1) / (n0 + n1)
    se = np.sqrt(p * (1 - p) * (1 / n0 + 1 / n1))
    if se == 0:
        return (100 * (p1 - p0), 0.0, 1.0)
    z = (p1 - p0) / se
    return (100 * (p1 - p0), float(z), float(_p_two_sided(z)))


def bh_flags(pvals, q=0.05):
    """{key: p} -> set of keys significant at BH FDR q."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    thresh_i = -1
    for i, (k, p) in enumerate(items):
        if p <= (i + 1) / m * q:
            thresh_i = i
    return {k for k, _ in items[:thresh_i + 1]}


def build_totals(daily):
    """Per-build totals from the daily cache: games, hero {h:[g,w]},
    bans {h:n}."""
    games = sum(d["games"] for d in daily.values())
    hero = defaultdict(lambda: [0, 0])
    bans = defaultdict(int)
    for d in daily.values():
        for h, (g, w) in d["hero"].items():
            hero[h][0] += g
            hero[h][1] += w
        for h, n in d["bans"].items():
            bans[h] += n
    return games, dict(hero), dict(bans)


def latency_walk(daily_new, hero, g0, w0, n_prev):
    """Day-cumulative walk of the new build for one hero. Returns dict of
    latencies (build games / hero games) or None per method."""
    days = sorted(daily_new)
    n_days = len(days)
    p0 = w0 / g0
    p0c = min(max(p0, 1e-4), 1 - 1e-4)
    pa_hi = min(p0c + DELTA, 1 - 1e-4)
    pa_lo = max(p0c - DELTA, 1e-4)
    lr_hi = (np.log(pa_hi / p0c), np.log((1 - pa_hi) / (1 - p0c)))
    lr_lo = (np.log(pa_lo / p0c), np.log((1 - pa_lo) / (1 - p0c)))
    cg = cw = cb = 0   # cumulative hero games, wins, build games
    out = {}
    for day in days:
        de = daily_new[day]
        cb += de["games"]
        g, w = de["hero"].get(hero, (0, 0))
        cg += g
        cw += w
        if cg < 30:
            continue
        r = two_prop(w0, g0, cw, cg)
        if r is None:
            continue
        _, _, p = r
        if "z_naive" not in out and p < 0.05:
            out["z_naive"] = (cb, cg)
        if "z_bonf" not in out and p < 0.05 / n_days:
            out["z_bonf"] = (cb, cg)
        if "sprt" not in out:
            llr_hi = cw * lr_hi[0] + (cg - cw) * lr_hi[1]
            llr_lo = cw * lr_lo[0] + (cg - cw) * lr_lo[1]
            if max(llr_hi, llr_lo) > SPRT_H:
                out["sprt"] = (cb, cg)
        if len(out) == 3:
            break
    return out


def load_ground_truth(builds):
    if not os.path.exists(GT_PATH):
        return None, "patch_notes_ground_truth.json missing"
    with open(GT_PATH) as f:
        gt = json.load(f)
    if not gt.get("patches"):
        return None, f"ground truth empty (scriptable={gt.get('scriptable')}; " \
                     f"{gt.get('notes', '')[:200]})"
    per_build = {}
    for p in gt["patches"]:
        b = p.get("mapped_build")
        if b in builds:
            # zero-hero patches are known "no balance change" truth
            per_build.setdefault(b, set()).update(p.get("heroes_changed") or [])
    # maintenance builds with no published notes anywhere = no balance changes
    for b in gt.get("corpus_builds_without_patch_notes", []):
        if b in builds:
            per_build.setdefault(b, set())
    n_empty = sum(1 for s in per_build.values() if not s)
    meta = (f"source: {gt.get('source', '?')[:110]}...; fetch: "
            f"{gt.get('fetch_method', '?')[:110]}...; {len(gt['patches'])} "
            f"patches; {len(per_build)} corpus builds with known truth "
            f"({n_empty} with zero balance changes); bugfix-only/ARAM-only "
            "hero mentions excluded from truth")
    return per_build, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()
    common.setup()

    with gzip.open(DAILY_CACHE, "rb") as f:
        cache = pickle.load(f)
    builds, per_build, build_games = (cache["builds"], cache["per_build"],
                                      cache["build_games"])
    order = sorted(per_build)
    sizable = [bi for bi in order if build_games[bi] >= common.SIZABLE_GAMES]
    print(f"{len(sizable)} sizable builds -> {len(sizable)-1} boundaries")

    truth_by_build, gt_meta = load_ground_truth(set(builds))
    print(f"ground truth: {gt_meta}")

    totals = {bi: build_totals(per_build[bi]) for bi in order}

    boundaries = []
    for k in range(1, len(sizable)):
        prev, new = sizable[k - 1], sizable[k]
        inter = [bi for bi in order if prev < bi <= new]
        n0, hero0, bans0 = totals[prev]
        n1, hero1, bans1 = totals[new]

        p_wr, p_pick, p_ban, effects = {}, {}, {}, {}
        heroes = set(hero0) | set(hero1)
        for h in heroes:
            g0, w0 = hero0.get(h, (0, 0))
            g1, w1 = hero1.get(h, (0, 0))
            e = {"prev_games": g0, "new_games": g1}
            if g0 >= MIN_G and g1 >= MIN_G:
                d, z, p = two_prop(w0, g0, w1, g1)
                p_wr[h] = p
                e.update(d_wr_pp=round(d, 2), p_wr=round(p, 5))
            d, z, p = two_prop(g0, n0, g1, n1)
            p_pick[h] = p
            e.update(d_pick_pp=round(d, 2), p_pick=round(p, 5))
            d, z, p = two_prop(bans0.get(h, 0), n0, bans1.get(h, 0), n1)
            p_ban[h] = p
            e.update(d_ban_pp=round(d, 2), p_ban=round(p, 5))
            effects[h] = e

        wr_bh = bh_flags(p_wr)
        pick_bh = bh_flags(p_pick) | bh_flags(p_ban)
        det = {
            "wr_p05": {h for h, p in p_wr.items() if p < 0.05},
            "wr_bh": wr_bh,
            "pickban_bh": pick_bh,
            "any_bh": wr_bh | pick_bh,
        }
        det["any_bh_eff"] = {
            h for h in det["any_bh"]
            if abs(effects[h].get("d_wr_pp", 0)) >= EFF_WR
            or abs(effects[h].get("d_pick_pp", 0)) >= EFF_RATE
            or abs(effects[h].get("d_ban_pp", 0)) >= EFF_RATE}
        # effect-floored variants: pick/ban tests are hugely overpowered at
        # build-level n, so significance alone mostly measures continuous
        # meta adaptation; floors trade recall for precision.
        det["wr_p05_eff"] = {h for h in det["wr_p05"]
                             if abs(effects[h].get("d_wr_pp", 0)) >= EFF_WR}
        det["pickban_eff2"] = {
            h for h in pick_bh
            if abs(effects[h].get("d_pick_pp", 0)) >= 2.0
            or abs(effects[h].get("d_ban_pp", 0)) >= 2.0}

        # latency for WR-flagged heroes
        latencies = {}
        for h in sorted(det["wr_p05"]):
            g0, w0 = hero0[h]
            lat = latency_walk(per_build[new], h, g0, w0, n0)
            if lat:
                latencies[h] = {m: {"build_games": bg, "hero_games": hg}
                                for m, (bg, hg) in lat.items()}

        rec = {
            "prev_build": builds[prev], "new_build": builds[new],
            "intermediate_builds": [builds[bi] for bi in inter],
            "prev_games": n0, "new_games": n1,
            "n_wr_testable": len(p_wr),
            "detected": {v: sorted(det[v]) for v in VARIANTS},
            "latency_wr": latencies,
            "effects_flagged": {h: effects[h]
                                for h in sorted(det["any_bh"])},
        }

        if truth_by_build is not None:
            t = set()
            known = []
            for bi in inter:
                b = builds[bi]
                if b in truth_by_build:
                    t |= truth_by_build[b]
                    known.append(b)
            # validatable only when EVERY folded build has known truth
            # (zero-change patches/maintenance builds count as known-empty)
            validatable = len(known) == len(inter)
            rec["truth"] = {"heroes": sorted(t), "known_builds": known,
                            "validatable": validatable}
            if validatable:
                rec["pr"] = {}
                for v in VARIANTS:
                    d = det[v]
                    tp = len(d & t)
                    rec["pr"][v] = {
                        "tp": tp, "fp": len(d - t), "fn": len(t - d),
                        "precision": round(tp / len(d), 3) if d else None,
                        "recall": round(tp / len(t), 3) if t else None}
        boundaries.append(rec)

    # ── aggregates ──
    agg = {v: {"detected": sum(len(b["detected"][v]) for b in boundaries)}
           for v in VARIANTS}
    lat_all = defaultdict(list)
    for b in boundaries:
        for h, lm in b["latency_wr"].items():
            for m, v in lm.items():
                lat_all[m].append(v["build_games"])
    lat_summary = {m: {"median_build_games": int(np.median(v)),
                       "p75_build_games": int(np.percentile(v, 75)),
                       "n": len(v)}
                   for m, v in lat_all.items()}

    val_summary = None
    if truth_by_build is not None:
        vb = [b for b in boundaries if b.get("truth", {}).get("validatable")]
        if vb:
            val_summary = {"n_boundaries": len(vb), "variants": {}}
            for v in VARIANTS:
                tp = sum(b["pr"][v]["tp"] for b in vb)
                fp = sum(b["pr"][v]["fp"] for b in vb)
                fn = sum(b["pr"][v]["fn"] for b in vb)
                prec = tp / (tp + fp) if tp + fp else None
                rec_ = tp / (tp + fn) if tp + fn else None
                f1 = (2 * prec * rec_ / (prec + rec_)
                      if prec and rec_ else None)
                val_summary["variants"][v] = {
                    "tp": tp, "fp": fp, "fn": fn,
                    "precision": round(prec, 3) if prec is not None else None,
                    "recall": round(rec_, 3) if rec_ is not None else None,
                    "f1": round(f1, 3) if f1 else None}
            # latency on true positives (any_bh & truth, wr-flagged)
            lat_tp = []
            for b in vb:
                t = set(b["truth"]["heroes"])
                for h, lm in b["latency_wr"].items():
                    if h in t and "z_naive" in lm:
                        lat_tp.append(lm["z_naive"]["build_games"])
            if lat_tp:
                val_summary["tp_latency_z_naive"] = {
                    "median_build_games": int(np.median(lat_tp)),
                    "n": len(lat_tp)}

    out = {"config": {"min_g_wr": MIN_G, "sprt_delta": DELTA,
                      "effect_floors_pp": {"wr": EFF_WR, "rate": EFF_RATE}},
           "ground_truth": gt_meta,
           "boundaries": boundaries,
           "detected_totals": agg,
           "latency_summary": lat_summary,
           "validation": val_summary}
    common.write_json(os.path.join(common.RESULTS_DIR, "w3_changepoints.json"),
                      out)

    # ── markdown ──
    nb = len(boundaries)
    lines = [
        "# W3(b) — Hero-level changepoint detection at build boundaries",
        "",
        f"{nb} boundaries between consecutive sizable builds (hotfix builds "
        "folded in). Per hero: two-proportion tests on WR (>= "
        f"{MIN_G} games/side), pick rate and ban rate; BH FDR within boundary "
        "x family. Detection latency = games into the new build until the "
        "sequential test flags (z at day boundaries; SPRT with |shift|="
        f"{DELTA*100:.0f}pp alternative, ~95% threshold).",
        "",
        "## Detected changed-hero counts per boundary",
        "",
        "| boundary (new build) | games | " + " | ".join(VARIANTS) + " |"
        + (" truth | P / R (any_bh_eff) |" if truth_by_build else ""),
        "|---|---|" + "---|" * len(VARIANTS)
        + ("---|---|" if truth_by_build else ""),
    ]
    for b in boundaries:
        row = (f"| {b['new_build']} | {b['new_games']:,} | " +
               " | ".join(str(len(b["detected"][v])) for v in VARIANTS) + " |")
        if truth_by_build:
            if b.get("pr"):
                pr = b["pr"]["any_bh_eff"]
                row += (f" {len(b['truth']['heroes'])} | "
                        f"{pr['precision'] if pr['precision'] is not None else '-'}"
                        f" / {pr['recall'] if pr['recall'] is not None else '-'} |")
            else:
                row += " n/a | n/a |"
        lines.append(row)

    lines += ["", "## Detection latency (WR-flagged heroes, games into the "
              "new build)", "",
              "| method | median build-games to flag | p75 | n heroes |",
              "|---|---|---|---|"]
    for m in ("z_naive", "z_bonf", "sprt"):
        s = lat_summary.get(m)
        lines.append(f"| {m} | {s['median_build_games']:,} | "
                     f"{s['p75_build_games']:,} | {s['n']} |" if s
                     else f"| {m} | - | - | 0 |")

    if val_summary:
        lines += ["", "## Validation vs official patch notes", "",
                  f"Ground truth: {gt_meta}", "",
                  f"{val_summary['n_boundaries']} validatable boundaries "
                  "(every folded build has known truth; zero-change "
                  "maintenance builds count as known-empty). Micro-averaged:",
                  "",
                  "| variant | TP | FP | FN | precision | recall | F1 |",
                  "|---|---|---|---|---|---|---|"]
        for v in VARIANTS:
            s = val_summary["variants"][v]
            lines.append(f"| {v} | {s['tp']} | {s['fp']} | {s['fn']} | "
                         f"{s['precision']} | {s['recall']} | {s['f1']} |")
        if val_summary.get("tp_latency_z_naive"):
            tl = val_summary["tp_latency_z_naive"]
            lines += ["", f"- Median latency to flag a patch-note-listed hero "
                      f"(naive z): {tl['median_build_games']:,} build games "
                      f"(n={tl['n']})."]
        lines += ["", "Caveats: patch notes are not a perfect oracle — "
                  "indirect effects (nerfing a counter shifts a hero's WR "
                  "without any note) create principled false positives, and "
                  "small note-level tweaks (e.g. a talent no one picks) are "
                  "principled false negatives. Precision here is therefore a "
                  "lower bound on detector quality."]
    else:
        lines += ["", "## Validation", "",
                  f"SKIPPED: {gt_meta}. Detection results stand alone; rerun "
                  "after providing patch_notes_ground_truth.json."]

    md = os.path.join(common.RESULTS_DIR, "W3_CHANGEPOINTS.md")
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
