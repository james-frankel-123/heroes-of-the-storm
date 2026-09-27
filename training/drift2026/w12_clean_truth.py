"""
W12 — out-of-sample judge-free rescoring of every head-to-head file.

Problem fixed: the maintained drafters (d2c_cumprev and the decayed90
champion) use statistics through builds[-2] = 2.55.16.96881, and the W6/W7/W8/
W10 judge-free truth merged ALL seven post-cutoff builds, so ~131K of the
~144K scoring games were also inside the maintained agent's own inputs.

Clean truth sets (no drafting agent's statistics touch any of these games):
  T_97039   build 2.55.16.97039, every game now in the DB (the snapshot held
            only its first ~week)
  T_255_17  all 2.55.17.* builds (shipped 2026-07-20 onward, after the
            snapshot; nothing in the paper has touched them)
  T_clean   both merged
plus T_orig (the stored future_truth_stats) to verify exact reproduction.

Counts use build_patch_stats.count_chunk and common.stats_from_counts, so the
thresholds match the original truth object exactly.

Usage:
  python3 drift2026/w12_clean_truth.py fetch     # DB -> patch_stats/clean_truth_counts.pkl.gz
  python3 drift2026/w12_clean_truth.py rescore   # -> results/w12_clean_truth.{json,md}
"""
import os
import sys
import json
import gzip
import pickle
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

common.setup()

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COUNTS = os.path.join(common.STATS_DIR, "clean_truth_counts.pkl.gz")
OUT_JSON = os.path.join(common.RESULTS_DIR, "w12_clean_truth.json")
OUT_MD = os.path.join(common.RESULTS_DIR, "W12_CLEAN_TRUTH.md")

# (suffix, label) in paper order
FILES = [
    ("", "maintained vs stale-4mo (W6, 5x5 seeds)"),
    ("_stale1yr", "maintained vs stale-1yr (W7)"),
    ("_stale2yr", "maintained vs stale-2yr (W7)"),
    ("_w8_volmatch", "volume-matched maintained vs stale-2yr (W8a)"),
    ("_w10_degen", "maintained vs stale-4mo (W10, 15x15 seeds)"),
    ("_w8_champ4mo", "maintained-d90 vs stale-4mo (W8b)"),
    ("_w8_champ1yr", "maintained-d90 vs stale-1yr (W8b)"),
    ("_w8_champ2yr", "maintained-d90 vs stale-2yr (W8b)"),
    ("_champ_vs_cumprev", "maintained-d90 vs maintained (W11)"),
]


def fetch():
    import psycopg2
    from drift2026.build_patch_stats import count_chunk, merge_cell, _new_cell
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("SET statement_timeout='20min'")
    cur.execute("""
        SELECT game_version, skill_tier, game_map, team0_heroes, team1_heroes,
               team0_bans, team1_bans, winner, replay_id, game_date
        FROM replay_draft_data
        WHERE game_version = '2.55.16.97039' OR game_version LIKE '2.55.17.%'""")
    rows = cur.fetchall()
    print(f"{len(rows):,} rows fetched")
    builds = sorted({r[0] for r in rows}, key=common.build_sort_key)
    bidx = {b: i for i, b in enumerate(builds)}

    def lst(x):
        return json.loads(x) if isinstance(x, str) else x
    slim, meta = [], {b: {"games": 0, "in_snapshot": 0, "min_date": None,
                          "max_date": None} for b in builds}
    for v, tier, gmap, t0, t1, b0, b1, w, rid, gd in rows:
        slim.append((bidx[v], tier, gmap, tuple(lst(t0)), tuple(lst(t1)),
                     tuple(lst(b0)) + tuple(lst(b1)), w))
        m = meta[v]
        m["games"] += 1
        m["in_snapshot"] += int(rid <= common.SNAPSHOT_BOUND)
        d = gd.date().isoformat()
        m["min_date"] = min(m["min_date"] or d, d)
        m["max_date"] = max(m["max_date"] or d, d)
    per_build = {}
    size = 20000
    for i in range(0, len(slim), size):
        for (bi, tier), cell in count_chunk(slim[i:i + size]).items():
            dst = per_build.setdefault(bi, {}).setdefault(tier, None)
            if dst is None:
                dst = per_build[bi][tier] = _new_cell()
            merge_cell(dst, cell)
    for by_tier in per_build.values():
        for tier, cell in by_tier.items():
            plain = {"games": cell["games"], "bans": dict(cell["bans"])}
            for k in ("hero", "hmap", "with", "against", "comp"):
                plain[k] = {kk: list(vv) for kk, vv in cell[k].items()}
            by_tier[tier] = plain
    with gzip.open(COUNTS, "wb") as f:
        pickle.dump({"builds": builds, "per_build": per_build, "meta": meta},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    for b in builds:
        print(b, meta[b])
    print(f"wrote {COUNTS}")


def truth_sets():
    from drift2026.build_patch_stats import merge_cell, _new_cell
    from drift2026.phase_w4_mcts import future_truth_stats
    with gzip.open(COUNTS, "rb") as f:
        data = pickle.load(f)
    builds, per_build = data["builds"], data["per_build"]

    def merged(pred):
        out = {}
        for bi, by_tier in per_build.items():
            if not pred(builds[bi]):
                continue
            for tier, cell in by_tier.items():
                dst = out.get(tier)
                if dst is None:
                    dst = out[tier] = _new_cell()
                merge_cell(dst, cell)
        games = sum(c["games"] for c in out.values())
        return common.stats_from_counts(out, pair_min=10), games

    sets = {"T_orig": (future_truth_stats(), None)}
    sets["T_97039"] = merged(lambda b: b == "2.55.16.97039")
    sets["T_255_17"] = merged(lambda b: b.startswith("2.55.17."))
    sets["T_clean"] = merged(lambda b: True)
    return sets, data["meta"]


def judgefree(records, truth):
    """Per-draft paired (maintained - opponent) deltas; mirrors
    w8_inference.judgefree_deltas with an injected truth object, plus the
    number of scorable counter/synergy cells per draft."""
    from shared import is_degenerate

    def ctr_d(ha, hb, tier):
        r = truth.get_counter(ha, hb, tier)
        if r is None:
            return None
        return r - (truth.get_hero_wr(ha, tier)
                    + (100 - truth.get_hero_wr(hb, tier)) - 50)

    def syn_d(ha, hb, tier):
        r = truth.get_synergy(ha, hb, tier)
        if r is None:
            return None
        return r - (50 + (truth.get_hero_wr(ha, tier) - 50)
                    + (truth.get_hero_wr(hb, tier) - 50))

    def side(own, opp, tier):
        cd = [x for o in opp for h in own
              for x in [ctr_d(h, o, tier)] if x is not None]
        sy = [x for j, h1 in enumerate(own) for h2 in own[j + 1:]
              for x in [syn_d(h1, h2, tier)] if x is not None]
        hw = [truth.get_hero_wr(h, tier) for h in own]
        hw = [v for v in hw if v is not None]
        return (np.mean(cd) if cd else 0.0, np.mean(sy) if sy else 0.0,
                np.mean(hw) if hw else np.nan, float(is_degenerate(own)))

    out = {"counter": [], "synergy": [], "future_hero_wr": [], "degen": []}
    for r in records:
        m = side(r["maintained"], r["frozen"], r["tier"])
        f = side(r["frozen"], r["maintained"], r["tier"])
        out["counter"].append(m[0] - f[0])
        out["synergy"].append(m[1] - f[1])
        out["future_hero_wr"].append(m[2] - f[2])
        out["degen"].append(m[3] - f[3])
    return {k: np.array(v) for k, v in out.items()}


def rescore():
    from drift2026.w8_inference import attach_labels, crossed_re
    sets, meta = truth_sets()
    result = {"truth_meta": meta,
              "truth_games": {k: g for k, (_, g) in sets.items()},
              "files": {}}
    for suffix, label in FILES:
        path = os.path.join(common.RESULTS_DIR, f"w6_head2head{suffix}.json")
        res = json.load(open(path))
        recs = attach_labels(res)
        sa = np.array([r["sa"] for r in recs])
        sb = np.array([r["sb"] for r in recs])
        entry = {"label": label, "n_drafts": len(recs), "by_truth": {}}
        for tname, (truth, _) in sets.items():
            d = judgefree(recs, truth)
            row = {}
            for k, v in d.items():
                ok = ~np.isnan(v)
                cr = crossed_re(v[ok], sa[ok], sb[ok])
                row[k] = {"est": cr["est"], "se": cr["se"],
                          "z": cr["est"] / cr["se"] if cr["se"] > 0 else None}
            entry["by_truth"][tname] = row
        result["files"][suffix or "_w6"] = entry
        o, c = entry["by_truth"]["T_orig"], entry["by_truth"]["T_clean"]
        print(f"{label}: hwr orig {o['future_hero_wr']['est']:+.3f} -> clean "
              f"{c['future_hero_wr']['est']:+.3f} ± {c['future_hero_wr']['se']:.3f}",
              flush=True)
    common.write_json(OUT_JSON, result)

    lines = ["# W12 out-of-sample judge-free rescoring", "",
             "Truth sets: T_orig = stored future truth (7 post-cutoff builds, "
             "6 of which overlap the maintained agents' statistics); "
             "T_97039 = build 2.55.16.97039, all games; T_255_17 = all 2.55.17 "
             "builds; T_clean = both. No drafting agent's statistics include any "
             "T_97039/T_255_17 game.", "",
             "Games: " + ", ".join(f"{k} {v:,}" for k, v in
                                   result["truth_games"].items() if v), "",
             "Paired deltas (maintained-side minus opponent), crossed seed "
             "random effects, est ± se (z).", ""]
    for metric, unit in (("future_hero_wr", "pp"), ("synergy", ""),
                         ("counter", ""), ("degen", "fraction")):
        lines += [f"## {metric} {unit}", "",
                  "| matchup | T_orig | T_97039 | T_255_17 | T_clean |",
                  "|---|---|---|---|---|"]
        for key, e in result["files"].items():
            cells = []
            for t in ("T_orig", "T_97039", "T_255_17", "T_clean"):
                r = e["by_truth"][t][metric]
                cells.append(f"{r['est']:+.3f} ± {r['se']:.3f} ({r['z']:.1f})"
                             if r["z"] is not None else f"{r['est']:+.3f}")
            lines.append(f"| {e['label']} | " + " | ".join(cells) + " |")
        lines.append("")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["fetch", "rescore"])
    args = ap.parse_args()
    fetch() if args.stage == "fetch" else rescore()


if __name__ == "__main__":
    main()
