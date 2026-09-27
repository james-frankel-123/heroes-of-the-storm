"""
Q7: exponentially-decayed aggregate statistics (stats-side recency decay).

For every 2.55 build N and half-life HL, the decayed aggregate is

    sum over builds b <= N of  w_b * counts_b,
    w_b = 0.5 ** (delta_days / HL),
    delta_days = days between build b's max_date and build N's max_date
                 (dates from patch_index.json build_meta; clamped >= 0)

computed incrementally (scale running cells by 0.5**(gap/HL), then add the
new build's counts — exact, since the weights compose multiplicatively).
Raw per-build counts come from patch_stats/patch_counts.pkl.gz (build once by
build_patch_stats.py); nothing is recounted.

Weighted [games, wins] are FRACTIONAL; build_patch_stats.derive_stats_file
consumes them as-is (pair_min=10, matching the cumulative kind), so
    patch_stats/decayed90/<build>.json.gz
    patch_stats/decayed365/<build>.json.gz
are drop-in for common.load_patch_stats. NOTE the storage/consumer thresholds
(hero >= 20 games, pairwise >= 10 stored / >= 30 used, map >= 5 stored /
>= 50 used) now apply to DECAYED effective counts: under HL=90 old data
decays away and estimates genuinely lose reliability — that effective-
sample-size collapse is part of what the Q7 arm measures.

SHRINKAGE CELL (the 2x2's fourth corner): patch_stats/decayed90k100/<build>
= decayed90 stats count-shrunk toward the same-build CUMULATIVE stats with
k=100 pseudo-games, mirroring the W2a blend (build_causal_local_features):

    wr = (g_dec * wr_dec + k * wr_cum) / (g_dec + k)          per statistic
    pick/ban = (total_dec * rate_dec + k * rate_cum) / (total_dec + k)

Availability mirrors W2a's convention (thresholds gate like the cumulative
arm): the stored `games` field is the CUMULATIVE count, the VALUE is the
blend. The feature pass then applies the usual `_prev` shift, so a row in
build N sees decayed-through-(N-1) shrunk toward cumulative-through-(N-1) —
exactly the cumprev prior.

SANITY (--check): for HL=inf the incremental weights are all 1, so the
derived files must reproduce patch_stats/cumulative/ bit-for-bit up to
float-vs-int `games` formatting. --check builds kind "decayedinf" and diffs
every build's file against the cumulative kind.

Usage:
    python drift2026/build_decayed_stats.py            # decayed90/365 + k100
    python drift2026/build_decayed_stats.py --check    # HL=inf == cumulative
"""
import os
import sys
import json
import gzip
import time
import pickle
import argparse
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common
from drift2026.build_patch_stats import derive_stats_file

HALF_LIVES = [90.0, 365.0]
K_SHRINK = 100.0
SHRINK_HL = 90.0            # the 2x2's shrunk corner uses the HL=90 decay


def date_days(iso):
    return datetime.date.fromisoformat(iso).toordinal()


def new_cell():
    return {"games": 0.0, "bans": {}, "hero": {}, "hmap": {},
            "with": {}, "against": {}, "comp": {}}


def scale_cells(cells, f):
    for cell in cells.values():
        cell["games"] *= f
        for h in cell["bans"]:
            cell["bans"][h] *= f
        for k in ("hero", "hmap", "with", "against", "comp"):
            for e in cell[k].values():
                e[0] *= f
                e[1] *= f


def add_counts(cells, per_tier):
    """Add one build's raw count cells (patch_counts format) into the
    running cells."""
    for tier, src in per_tier.items():
        dst = cells.get(tier)
        if dst is None:
            dst = cells[tier] = new_cell()
        dst["games"] += src["games"]
        bans = dst["bans"]
        for h, n in src["bans"].items():
            bans[h] = bans.get(h, 0.0) + n
        for k in ("hero", "hmap", "with", "against", "comp"):
            d = dst[k]
            for key, (g, w) in src[k].items():
                e = d.get(key)
                if e is None:
                    d[key] = [float(g), float(w)]
                else:
                    e[0] += g
                    e[1] += w


def write_shrunk_file(build, dec, cum, kind, k=K_SHRINK, pair_min=10):
    """decayed stats shrunk toward same-build cumulative with k pseudo-games.
    Same JSON schema as derive_stats_file; games field = cumulative count
    (availability gates like the cumulative arm), win_rate = the blend."""
    hero_stats, hero_map_stats, pairwise_stats, comp_stats = [], [], [], []
    for tier, ccell in cum.items():
        total_c = ccell["games"]
        if total_c == 0:
            continue
        dcell = dec.get(tier) or new_cell()
        total_d = dcell["games"]

        def blend(dec_entry, g_c, w_c):
            wr_c = 100.0 * w_c / g_c
            if dec_entry is None:
                g_d, wr_d = 0.0, wr_c
            else:
                g_d = dec_entry[0]
                wr_d = 100.0 * dec_entry[1] / g_d if g_d > 0 else wr_c
            return (g_d * wr_d + k * wr_c) / (g_d + k)

        for h, (g_c, w_c) in ccell["hero"].items():
            if g_c < 20:
                continue
            wr = blend(dcell["hero"].get(h), g_c, w_c)
            pr_c = 100.0 * g_c / total_c
            br_c = 100.0 * ccell["bans"].get(h, 0) / total_c
            if total_d > 0:
                g_d = dcell["hero"].get(h, (0.0, 0.0))[0]
                pr_d = 100.0 * g_d / total_d
                br_d = 100.0 * dcell["bans"].get(h, 0.0) / total_d
                den = total_d + k
                pick = (total_d * pr_d + k * pr_c) / den
                ban = (total_d * br_d + k * br_c) / den
            else:
                pick, ban = pr_c, br_c
            hero_stats.append({
                "hero": h, "tier": tier, "games": g_c,
                "win_rate": round(wr, 3), "pick_rate": round(pick, 3),
                "ban_rate": round(ban, 3)})
        for (m, h), (g_c, w_c) in ccell["hmap"].items():
            if g_c < 5:
                continue
            hero_map_stats.append({
                "hero": h, "map": m, "tier": tier, "games": g_c,
                "win_rate": round(blend(dcell["hmap"].get((m, h)), g_c, w_c), 3)})
        for (a, b), (g_c, w_c) in ccell["with"].items():
            if g_c < pair_min:
                continue
            wr = round(blend(dcell["with"].get((a, b)), g_c, w_c), 3)
            pairwise_stats.append({"hero_a": a, "hero_b": b, "tier": tier,
                                   "relationship": "with", "win_rate": wr,
                                   "games": g_c})
            pairwise_stats.append({"hero_a": b, "hero_b": a, "tier": tier,
                                   "relationship": "with", "win_rate": wr,
                                   "games": g_c})
        for (a, b), (g_c, wa_c) in ccell["against"].items():
            if g_c < pair_min:
                continue
            wr_a = round(blend(dcell["against"].get((a, b)), g_c, wa_c), 3)
            pairwise_stats.append({"hero_a": a, "hero_b": b, "tier": tier,
                                   "relationship": "against", "win_rate": wr_a,
                                   "games": g_c})
            pairwise_stats.append({"hero_a": b, "hero_b": a, "tier": tier,
                                   "relationship": "against",
                                   "win_rate": round(100.0 - wr_a, 3),
                                   "games": g_c})
        for ck, (g_c, w_c) in ccell["comp"].items():
            if g_c < 5:
                continue
            comp_stats.append({"roles": ck, "tier": tier, "games": g_c,
                               "win_rate": round(blend(dcell["comp"].get(ck),
                                                       g_c, w_c), 3)})

    path = common.stats_path(kind, build)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump({
            "_meta": {"build": build, "kind": kind, "k_shrink": k,
                      "half_life": SHRINK_HL,
                      "games_cum": {t: c["games"] for t, c in cum.items()},
                      "games_decayed": {t: round(c["games"], 2)
                                        for t, c in dec.items()},
                      "source": "pinned snapshot 2026-05-22 (own corpus)"},
            "hero_stats": hero_stats, "hero_map_stats": hero_map_stats,
            "pairwise_stats": pairwise_stats, "comp_stats": comp_stats,
        }, f)


def run_half_life(hl, kind, present, builds, meta, per_build,
                  shrink_kind=None, force=False):
    """One incremental pass over the builds. hl=None -> weight 1 (the
    --check arm). shrink_kind: also write the k100-shrunk files (needs the
    running cumulative cells, maintained alongside)."""
    dec, cum = {}, {}
    prev_day = None
    t0 = time.time()
    for bidx in present:
        build = builds[bidx]
        day = date_days(meta[build]["max_date"])
        if prev_day is not None and hl is not None:
            gap = max(0, day - prev_day)
            if gap:
                scale_cells(dec, 0.5 ** (gap / hl))
        prev_day = day
        add_counts(dec, per_build[bidx])
        if force or not os.path.exists(common.stats_path(kind, build)):
            derive_stats_file(build, dec, kind, pair_min=10)
        if shrink_kind:
            add_counts(cum, per_build[bidx])
            if force or not os.path.exists(common.stats_path(shrink_kind, build)):
                write_shrunk_file(build, dec, cum, shrink_kind)
    print(f"  {kind}{' + ' + shrink_kind if shrink_kind else ''}: "
          f"{len(present)} builds in {time.time() - t0:.0f}s", flush=True)


def check_inf(present, builds):
    """decayedinf files must equal cumulative files (games int-vs-float aside)."""
    def norm(raw):
        out = {}
        for sec in ("hero_stats", "hero_map_stats", "pairwise_stats",
                    "comp_stats"):
            rows = []
            for r in raw[sec]:
                r = dict(r)
                r["games"] = round(float(r["games"]), 3)
                rows.append(tuple(sorted(r.items())))
            out[sec] = sorted(rows)
        return out

    bad = 0
    for bidx in present:
        build = builds[bidx]
        with gzip.open(common.stats_path("decayedinf", build), "rt") as f:
            a = norm(json.load(f))
        with gzip.open(common.stats_path("cumulative", build), "rt") as f:
            b = norm(json.load(f))
        ok = a == b
        bad += not ok
        if not ok:
            for sec in a:
                if a[sec] != b[sec]:
                    da = set(a[sec]) - set(b[sec])
                    db = set(b[sec]) - set(a[sec])
                    print(f"  MISMATCH {build} {sec}: "
                          f"{len(da)} only-decayedinf, {len(db)} only-cumulative")
                    for x in list(da)[:2]:
                        print(f"    inf : {x}")
                    for x in list(db)[:2]:
                        print(f"    cum : {x}")
    print(f"HL=inf convergence check: {len(present) - bad}/{len(present)} "
          f"builds identical to cumulative")
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="build kind 'decayedinf' (weight 1) and verify it "
                         "reproduces the cumulative files")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    common.setup()

    idx = common.load_patch_index()
    builds, meta = idx["builds"], idx["build_meta"]
    with gzip.open(common.COUNTS_PKL, "rb") as f:
        counts = pickle.load(f)
    assert counts["builds"] == builds
    per_build = counts["per_build"]
    present = sorted(per_build)
    print(f"{len(present)} builds with counts "
          f"({builds[present[0]]} .. {builds[present[-1]]})")

    if args.check:
        run_half_life(None, "decayedinf", present, builds, meta, per_build,
                      force=args.force)
        ok = check_inf(present, builds)
        sys.exit(0 if ok else 1)

    for hl in HALF_LIVES:
        kind = f"decayed{int(hl)}"
        shrink = f"decayed{int(hl)}k{int(K_SHRINK)}" if hl == SHRINK_HL else None
        run_half_life(hl, kind, present, builds, meta, per_build,
                      shrink_kind=shrink, force=args.force)


if __name__ == "__main__":
    main()
