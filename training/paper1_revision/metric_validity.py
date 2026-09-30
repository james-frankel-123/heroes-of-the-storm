"""
Supplement metric-validity check (audit A5), redone without self-inclusion.

The submission scored the 38,981 test replays with the external 2026-05-19
statistics, which contain those replays' own outcomes. Here the same metrics
(synergy_exploitation, counter_responsiveness) and the same strength-adjusted
decile curve (gen_fig_metric_validation.curve_adjusted) are computed on:

  hp   / test      as published (self-inclusion present), reproduction check
  own  / test      own-corpus deploy statistics (train split only)
  hp   / NODRIFT   external statistics on 291,837 post-snapshot games, which
                   were uploaded after the aggregates were pulled
  own  / NODRIFT   own deploy statistics on the same games

Reported per curve: top-minus-bottom decile spread (pp), bottom and top
decile win rates, count of increasing adjacent steps (of 9), and the Spearman
of the per-team metric with the outcome inside strata. Output:
results/metric_validity.json (+ cached per-team arrays).
"""
import os
import sys
import json
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from paper1_revision import core

import numpy as np

FIG = "/home/max/heroes-of-the-storm/paper/paper 1/scripts/gen_fig_metric_validation.py"


def figmod():
    spec = importlib.util.spec_from_file_location("gfmv", FIG)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _chunk(args):
    games, sd = args
    from sweep_enriched_wp import StatsCache
    from experiment_rich_evaluation import counter_responsiveness, synergy_exploitation
    from shared import SKILL_TIERS
    st = object.__new__(StatsCache)
    for k, v in sd.items():
        setattr(st, k, v)
    out = []
    for g in games:
        rid, tier, gm, t0, t1, bans, w = g
        for side, (a, b) in enumerate(((list(t0), list(t1)), (list(t1), list(t0)))):
            out.append((synergy_exploitation(a, st, tier), counter_responsiveness(a, b, st, tier),
                        1.0 if w == side else 0.0,
                        float(np.mean([st.get_hero_wr(h, tier) for h in a])),
                        float(np.mean([st.get_hero_wr(h, tier) for h in b])),
                        SKILL_TIERS.index(tier)))
    return out


def arrays(games, stats_name, tag):
    path = os.path.join(core.CACHE, f"metric_validity_{tag}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    import multiprocessing as mp
    sd = core.stats_dict(core.load_stats(stats_name))
    parts = [(games[i:i + 5000], sd) for i in range(0, len(games), 5000)]
    with mp.get_context("fork").Pool(16) as pool:
        res = [r for part in pool.map(_chunk, parts) for r in part]
    A = np.array(res, dtype=np.float64)
    d = {"syn": A[:, 0], "ctr": A[:, 1], "win": A[:, 2], "twr": A[:, 3], "owr": A[:, 4],
         "tid": A[:, 5].astype(int)}
    np.savez(path, **d)
    return d


def summarize(curve):
    y, ci = curve
    return {"curve": [round(float(v), 2) for v in y], "ci": [round(float(v), 2) for v in ci],
            "bottom": float(y[0]), "top": float(y[-1]), "spread_pp": float(y[-1] - y[0]),
            "increasing_steps": int(np.sum(np.diff(y) > 0))}


def main():
    F = figmod()
    sp = core.split_games()
    sets = {"test": sp["test"], "NODRIFT": core.gold_games("NODRIFT")}
    out = {}
    for gname, games in sets.items():
        for sname in ("hp", "deploy"):
            tag = f"{sname}_{gname}"
            A = arrays(games, sname, tag)
            r = {"n_team_obs": int(len(A["win"]))}
            for met in ("syn", "ctr"):
                r[met] = {"adjusted": summarize(F.curve_adjusted(A[met], A["win"], A["tid"],
                                                                 A["twr"], A["owr"])),
                          "raw": summarize(F.curve_raw(A[met], A["win"]))}
            out[tag] = r
            print(tag, {m: (round(r[m]["adjusted"]["spread_pp"], 2),
                            r[m]["adjusted"]["increasing_steps"]) for m in ("syn", "ctr")},
                  flush=True)
    json.dump(out, open(os.path.join(core.RESULTS, "metric_validity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
