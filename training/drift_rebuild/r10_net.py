"""
R10 — the W13 "Net" score (team win-probability gap implied by realized hero
WR, synergy and counter scores, with weights fitted on held-out real games)
for every rebuild head-to-head file, plus the paper's files for reference.

Reuses drift2026/w13_counter_dig.py's Scorer / stats_for /
real_game_analysis unchanged. The clean-truth games are fetched with an
explicit build list (2.55.16.97039 and the four 2.55.17 builds of W12) and a
game-date bound of 2026-09-27, so no build released after the
pre-registration date and no later game enters.

Usage: python3 drift_rebuild/r10_net.py
Output: drift_rebuild/results/r10_net.json
"""
import os
import sys
import json
import glob
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_common as rb  # noqa: E402

from drift2026 import common  # noqa: E402

common.setup()
import numpy as np  # noqa: E402
from drift2026.w13_counter_dig import Scorer, stats_for, real_game_analysis  # noqa: E402
from drift2026.w8_inference import attach_labels, crossed_re  # noqa: E402

BUILDS = ("2.55.16.97039", "2.55.17.97605", "2.55.17.97650",
          "2.55.17.97771", "2.55.17.98025")
PAPER = {"paper_M_vs_U": "w6_head2head.json",
         "paper_M_vs_S1": "w6_head2head_stale1yr.json",
         "paper_M_vs_S2": "w6_head2head_stale2yr.json",
         "paper_M_vs_U_15x15": "w6_head2head_w10_degen.json"}


def fetch():
    import psycopg2
    cache = os.path.join(rb.CACHE_DIR, "r10_clean_games.json")
    if os.path.exists(cache):
        return [tuple(tuple(x) if isinstance(x, list) else x for x in g)
                for g in json.load(open(cache))]
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute("""
        SELECT replay_id, skill_tier, game_map, team0_heroes, team1_heroes,
               team0_bans, team1_bans, winner
        FROM replay_draft_data
        WHERE game_version = ANY(%s) AND game_date < '2026-09-28'""", (list(BUILDS),))

    def lst(x):
        return json.loads(x) if isinstance(x, str) else x
    g = [(rid, tier, gm, tuple(lst(t0)), tuple(lst(t1)),
          tuple(lst(b0)) + tuple(lst(b1)), w)
         for rid, tier, gm, t0, t1, b0, b1, w in cur.fetchall()]
    json.dump(g, open(cache, "w"))
    return g


def net_for(recs, scorer, coef):
    vals, sa, sb = [], [], []
    for r in recs:
        d = scorer.diff(tuple(r["maintained"]), tuple(r["frozen"]), r["tier"])
        if np.isnan(d["hwr"]):
            continue
        z = coef[1] * d["hwr"] + coef[2] * d["syn"] + coef[3] * d["ctr"]
        vals.append(100 * (1 / (1 + math.exp(-z)) - 0.5))
        sa.append(r["sa"])
        sb.append(r["sb"])
    cr = crossed_re(np.array(vals), np.array(sa), np.array(sb))
    return cr["est"], cr["se"]


def main():
    games = fetch()
    print(f"{len(games):,} clean games")
    files = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join(rb.RESULTS_DIR, "h2h", "*.json"))}
    for k, f in PAPER.items():
        files[k] = os.path.join(common.RESULTS_DIR, f)
    recs = {}
    for k, p in files.items():
        res = json.load(open(p))
        recs[k] = (res["records"] if res["records"] and "sa" in res["records"][0]
                   else attach_labels(res))
    out = {k: {} for k in files}
    even = [g for g in games if g[0] % 2 == 0]
    odd = [g for g in games if g[0] % 2 == 1]
    for name, sh, oh in (("even_stats", even, odd), ("odd_stats", odd, even)):
        sc = Scorer(stats_for(sh))
        gm = real_game_analysis(sc, oh)
        coef = np.array(gm["logit_full"]["coef"])
        print(name, "coef", np.round(coef, 4).tolist(), flush=True)
        for k in files:
            out[k][name] = net_for(recs[k], sc, coef)
    for k, v in out.items():
        est = np.mean([v[s][0] for s in ("even_stats", "odd_stats")])
        zs = [v[s][0] / v[s][1] for s in ("even_stats", "odd_stats")]
        v["mean_est_pp"] = float(est)
        v["z_by_split"] = zs
        print(f"{k:28s} net {est:+.2f}pp  z by split {zs[0]:.1f}, {zs[1]:.1f}")
    common.write_json(os.path.join(rb.RESULTS_DIR, "r10_net.json"), out)


if __name__ == "__main__":
    main()
