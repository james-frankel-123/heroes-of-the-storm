"""
W13 — why do stale agents score better on counters? And what do the three
judge-free metrics add up to?

Uses the W12 clean truth games (2.55.16.97039 + 2.55.17.*; no drafting
agent's statistics touch them), split by replay_id parity:
  stats half   (even ids) -> truth statistics
  outcome half (odd ids)  -> real games scored with those statistics
so no game is scored by statistics that contain its own outcome.

Questions:
  Q1 Metric bias. The counter residual is additive in win-rate space:
       ctr(a,b) = WR(a vs b) - (WR_a + (100 - WR_b) - 50).
     If matchups compress toward 50 relative to that additive baseline, a
     team of strong heroes gets a negative counter score mechanically. We
     measure, in real games, the slope of team counter score on team
     hero-WR difference, and rescore the head-to-heads with a log-odds
     baseline and with the hero-strength trend regressed out.
  Q2 Validity and weights. In real games (outcome half), logistic
     regression of the result on (d_hero_wr, d_synergy, d_counter): do the
     counter and synergy scores predict wins beyond hero WR, and at what
     weight?
  Q3 Net. Apply the fitted weights to every head-to-head draft: the
     predicted win probability of the maintained side, from realized
     statistics only, with crossed seed random effects.
Both split directions are run; the second is reported as a robustness row.

Usage: python3 drift2026/w13_counter_dig.py
Output: results/w13_counter_dig.{json,md}
"""
import os
import sys
import json
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

common.setup()

import numpy as np

from drift2026.w12_clean_truth import FILES

OUT_JSON = os.path.join(common.RESULTS_DIR, "w13_counter_dig.json")
OUT_MD = os.path.join(common.RESULTS_DIR, "W13_COUNTER_DIG.md")


def fetch_games():
    import psycopg2
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("SET statement_timeout='20min'")
    cur.execute("""
        SELECT replay_id, skill_tier, game_map, team0_heroes, team1_heroes,
               team0_bans, team1_bans, winner
        FROM replay_draft_data
        WHERE game_version = '2.55.16.97039' OR game_version LIKE '2.55.17.%'""")

    def lst(x):
        return json.loads(x) if isinstance(x, str) else x
    return [(rid, tier, gmap, tuple(lst(t0)), tuple(lst(t1)),
             tuple(lst(b0)) + tuple(lst(b1)), w)
            for rid, tier, gmap, t0, t1, b0, b1, w in cur.fetchall()]


def stats_for(games):
    from drift2026.build_patch_stats import count_chunk, merge_cell, _new_cell
    merged = {}
    for i in range(0, len(games), 20000):
        chunk = [(0, t, m, a, b, bans, w) for _, t, m, a, b, bans, w
                 in games[i:i + 20000]]
        for (_, tier), cell in count_chunk(chunk).items():
            dst = merged.get(tier)
            if dst is None:
                dst = merged[tier] = _new_cell()
            merge_cell(dst, cell)
    return common.stats_from_counts(merged, pair_min=10)


def logit(p):
    p = min(max(p / 100.0, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


class Scorer:
    """Team-level judge-free features against a truth object. Mirrors
    w6_judgefree/w8_inference exactly for hwr/syn/ctr, plus a log-odds
    counter variant."""

    def __init__(self, truth):
        self.t = truth

    def hwr(self, team, tier):
        v = [self.t.get_hero_wr(h, tier) for h in team]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else np.nan

    def syn(self, team, tier):
        t = self.t
        out = []
        for j, a in enumerate(team):
            for b in team[j + 1:]:
                r = t.get_synergy(a, b, tier)
                if r is not None:
                    out.append(r - (50 + (t.get_hero_wr(a, tier) - 50)
                                    + (t.get_hero_wr(b, tier) - 50)))
        return float(np.mean(out)) if out else 0.0

    def ctr(self, own, opp, tier):
        t = self.t
        add, lo = [], []
        for a in own:
            for b in opp:
                r = t.get_counter(a, b, tier)
                if r is None:
                    continue
                wa, wb = t.get_hero_wr(a, tier), t.get_hero_wr(b, tier)
                add.append(r - (wa + (100 - wb) - 50))
                lo.append(logit(r) - (logit(wa) - logit(wb)))
        return ((float(np.mean(add)) if add else 0.0),
                (float(np.mean(lo)) if lo else 0.0))

    def diff(self, a, b, tier):
        ca, la = self.ctr(a, b, tier)
        cb, lb = self.ctr(b, a, tier)
        return {"hwr": self.hwr(a, tier) - self.hwr(b, tier),
                "syn": self.syn(a, tier) - self.syn(b, tier),
                "ctr": ca - cb, "ctr_logit": la - lb}


def fit_logistic(X, y, iters=50):
    """Plain Newton-Raphson logistic regression with intercept; returns
    (coef incl. intercept, se)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ w))
        g = X1.T @ (y - p)
        H = (X1 * (p * (1 - p))[:, None]).T @ X1
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X1 @ w))
    H = (X1 * (p * (1 - p))[:, None]).T @ X1
    return w, np.sqrt(np.diag(np.linalg.inv(H)))


def real_game_analysis(scorer, games):
    rows = []
    for _, tier, _, t0, t1, _, w in games:
        d = scorer.diff(t0, t1, tier)
        if np.isnan(d["hwr"]):
            continue
        rows.append((d["hwr"], d["syn"], d["ctr"], d["ctr_logit"],
                     1.0 if w == 0 else 0.0))
    A = np.array(rows)
    hwr, syn, ctr, ctrl, y = A.T
    slope = float(np.polyfit(hwr, ctr, 1)[0])
    slope_l = float(np.polyfit(hwr, ctrl, 1)[0])
    coef, se = fit_logistic(np.column_stack([hwr, syn, ctr]), y)
    coef_h, se_h = fit_logistic(hwr[:, None], y)
    return {
        "n_games": len(y),
        "corr_hwr_ctr": float(np.corrcoef(hwr, ctr)[0, 1]),
        "slope_ctr_on_hwr": slope,
        "corr_hwr_ctr_logit": float(np.corrcoef(hwr, ctrl)[0, 1]),
        "slope_ctr_logit_on_hwr": slope_l,
        "logit_full": {"names": ["intercept", "hwr", "syn", "ctr"],
                       "coef": coef.tolist(), "se": se.tolist()},
        "logit_hwr_only": {"coef": coef_h.tolist(), "se": se_h.tolist()},
    }


def h2h_analysis(scorer, gm):
    from drift2026.w8_inference import attach_labels, crossed_re
    coef = np.array(gm["logit_full"]["coef"])
    slope = gm["slope_ctr_on_hwr"]
    out = {}
    for suffix, label in FILES:
        path = os.path.join(common.RESULTS_DIR, f"w6_head2head{suffix}.json")
        recs = attach_labels(json.load(open(path)))
        vals = {k: [] for k in ("hwr", "syn", "ctr", "ctr_logit",
                                "ctr_detrended", "net_wp_pp")}
        sa, sb = [], []
        for r in recs:
            d = scorer.diff(tuple(r["maintained"]), tuple(r["frozen"]), r["tier"])
            if np.isnan(d["hwr"]):
                continue
            for k in ("hwr", "syn", "ctr", "ctr_logit"):
                vals[k].append(d[k])
            vals["ctr_detrended"].append(d["ctr"] - slope * d["hwr"])
            # Net: symmetric predicted P(maintained wins) from realized-stat
            # features only; the intercept (blue/red side) is dropped because
            # side orderings are balanced.
            z = coef[1] * d["hwr"] + coef[2] * d["syn"] + coef[3] * d["ctr"]
            vals["net_wp_pp"].append(100 * (1 / (1 + math.exp(-z)) - 0.5))
            sa.append(r["sa"])
            sb.append(r["sb"])
        sa, sb = np.array(sa), np.array(sb)
        row = {"label": label}
        for k, v in vals.items():
            cr = crossed_re(np.array(v), sa, sb)
            row[k] = {"est": cr["est"], "se": cr["se"],
                      "z": cr["est"] / cr["se"] if cr["se"] > 0 else None}
        out[suffix or "_w6"] = row
        print(f"{label}: ctr {row['ctr']['est']:+.3f} ({row['ctr']['z']:.1f}) "
              f"detrended {row['ctr_detrended']['est']:+.3f} "
              f"({row['ctr_detrended']['z']:.1f}) net {row['net_wp_pp']['est']:+.2f}pp "
              f"({row['net_wp_pp']['z']:.1f})", flush=True)
    return out


def main():
    games = fetch_games()
    print(f"{len(games):,} clean games")
    even = [g for g in games if g[0] % 2 == 0]
    odd = [g for g in games if g[0] % 2 == 1]
    result = {}
    for name, stats_half, outcome_half in (("even_stats_odd_outcomes", even, odd),
                                           ("odd_stats_even_outcomes", odd, even)):
        scorer = Scorer(stats_for(stats_half))
        gm = real_game_analysis(scorer, outcome_half)
        print(name, json.dumps(gm, indent=1), flush=True)
        result[name] = {"real_games": gm, "h2h": h2h_analysis(scorer, gm)}
    common.write_json(OUT_JSON, result)

    lines = ["# W13 counter dig: metric bias, validity, net realized-outcome score",
             "", "Clean truth games (2.55.16.97039 + 2.55.17.*) split by replay_id "
             "parity; statistics from one half, outcomes/scoring on the other.", ""]
    for name, r in result.items():
        g = r["real_games"]
        c, s = g["logit_full"]["coef"], g["logit_full"]["se"]
        lines += [f"## {name}", "",
                  f"Real games: {g['n_games']:,}. corr(d_hwr, d_ctr) = "
                  f"{g['corr_hwr_ctr']:+.3f}, slope {g['slope_ctr_on_hwr']:+.3f}; "
                  f"log-odds counter: corr {g['corr_hwr_ctr_logit']:+.3f}, "
                  f"slope {g['slope_ctr_logit_on_hwr']:+.4f}.", "",
                  "Logistic regression of the real result on team differences "
                  "(per unit): " + ", ".join(
                      f"{n} {c[i]:+.4f} ± {s[i]:.4f}" for i, n in
                      enumerate(g["logit_full"]["names"])), "",
                  "| matchup | counter | counter (log-odds) | counter detrended | "
                  "net realized WP (pp) |", "|---|---|---|---|---|"]
        for e in r["h2h"].values():
            f = lambda m: (f"{e[m]['est']:+.3f} ± {e[m]['se']:.3f} "
                           f"({e[m]['z']:.1f})")
            lines.append(f"| {e['label']} | {f('ctr')} | {f('ctr_logit')} | "
                         f"{f('ctr_detrended')} | {f('net_wp_pp')} |")
        lines.append("")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
