"""
WORLDS.md from the results JSONs (site/results). python3 worlds/report.py
"""
import os
import sys
import json
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as WD  # noqa: E402

R = WD.RESULTS


def J(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, d=2, sign=False):
    return "" if x is None else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def fit_section(L):
    b = J("fit_base.json")
    hp = J("fit_hp.json")
    if hp:
        L += ["## World model fit", "", "Base-model hyperparameters on the val split (sS = sK pair prior SD, r = CP rank, sE embedding prior SD):", "",
              "| sS = sK | r | sE | val nll | train nll |", "|---|---|---|---|---|"]
        for g in sorted(hp["grid"], key=lambda d: d["val_nll"]):
            L.append(f"| {g['sP']} | {g['r']} | {g['sE']} | {g['val_nll']:.5f} | {g['train_nll']:.5f} |")
        L.append("")
    if b:
        L += [f"Chosen: {json.dumps(b['cfg'])}. Held-out nll: val {b['val_nll']:.5f}, test {b['test_nll']:.5f}, "
              f"NODRIFT {b['NODRIFT_nll']:.5f}.", "",
              "| block | median posterior SD (logit) | median abs MAP |", "|---|---|---|"]
        for k in b["post_sd_median"]:
            L.append(f"| {k} | {b['post_sd_median'][k]:.4f} | {b['map_abs_median'][k]:.4f} |")
        L.append("")
    ly = J("fit_layers.json")
    if ly:
        L += ["Higher-order layers (prior SD s by diag-Laplace evidence; 95% profile range):", "",
              "| layer | cells seen | s (logit) | s 95% range | val nll base -> with layer | NODRIFT nll base -> with layer |",
              "|---|---|---|---|---|---|"]
        for k, v in ly.items():
            L.append(f"| {k} | {v['cells_seen']:,} | {v['s']:.4f} | {v['s_lo95']:.4f}-{v['s_hi95']:.4f} | "
                     f"{v['val_nll_base']:.5f} -> {v['val_nll_with_layer']:.5f} | {v['NODRIFT_nll_base']:.5f} -> {v['NODRIFT_nll_with_layer']:.5f} |")
        L.append("")
    d = J("drift.json")
    if d:
        L += ["Drift (family e), move SDs in logit, T17 net of the NODRIFT control: " + json.dumps({k: round(v, 4) for k, v in d["sizes"].items()}), ""]
    for name in ("player_fit_fullhistory.json", "player_fit2.json"):
        q = J(name)
        if q:
            L += [f"Player family ({name}): {q['games']:,} games; " + (f"SDs {json.dumps(q['sd'])}" if "sd" in q else f"s_alpha {q['s_a']}, s_eps {q['s_e']}")
                  + f"; nll pop {q['nll_pop']:.5f} -> with player term {q['nll_with_player']:.5f}.", ""]
    p = J("player_fit.json")
    if p:
        L += [f"Player family (f): {p['games']:,} games, {p['players']:,} players, {p['cells']:,} cells; s_alpha {p['s_a']}, "
              f"s_eps {p['s_e']}; familiarity curve (pts vs 101+ games, bins {p['nbins']}): "
              + ", ".join(f"{x:+.1f}" for x in p["gamma_pts_rel_101plus"])
              + f". nll pop {p['nll_pop']:.5f} -> with player term {p['nll_with_player']:.5f}.", ""]


def checks_section(L):
    for path in sorted(glob.glob(os.path.join(R, "check_step1_*.json"))):
        c = json.load(open(path))
        L += [f"### Step-1 statistics, family {c['family']}, dials {c['dial']} ({c['worlds']} worlds)", "",
              "| statistic | real | worlds mean ± SD | worlds 5-95% | real percentile |", "|---|---|---|---|---|"]
        for k, s in c["stats"].items():
            L.append(f"| {k} | {s['real']:+.3f} | {s['world_mean']:+.3f} ± {s['world_sd']:.3f} | "
                     f"[{s['world_p05']:+.3f}, {s['world_p95']:+.3f}] | {s['real_percentile']:.0f} |")
        L.append("")
    ll = J("check_logloss.json")
    if ll:
        L += ["### Held-out log-loss (NODRIFT) and world spread", "",
              f"Agent (enriched WP) {ll['agent_nll']:.5f}; base world model alone {ll['base_only_nll']:.5f}.", "",
              "| family | world-mean nll | bin (game's lower min3) | n | agent nll | world nll | SD over worlds (pts) |",
              "|---|---|---|---|---|---|---|"]
        for fam, r in ll["families"].items():
            for b in r["by_game_min_min3"]:
                L.append(f"| {fam} | {r['world_mean_nll']:.5f} | {b['lo']}-{b['hi'] or ''} | {b['n']:,} | {b['agent_nll']:.5f} | "
                         f"{b['world_nll']:.5f} | {b['worlds_p_sd_pts']:.2f} |")
        L.append("")
    pr = J("check_pairs.json")
    if pr:
        L += ["### Pair-effect SDs (counter_checks estimator; real.json: counter 1.09, synergy 1.72 pts)", "",
              "| family | counter SD true (pts) | synergy SD true (pts) |", "|---|---|---|"]
        for fam, s in pr["summary"].items():
            L.append(f"| {fam} | {s['counter']['sd_true_pts_mean']:.2f} ± {s['counter']['sd_true_pts_sd']:.2f} | "
                     f"{s['synergy']['sd_true_pts_mean']:.2f} ± {s['synergy']['sd_true_pts_sd']:.2f} |")
        L.append("")
    for nm_ in ("check_player_slope_fullhistory.json", "check_player_slope.json", "check_player_slope2.json"):
      ps = J(nm_)
      if ps:
        L += [f"### Player family ({nm_}): slope of r on the drafter's personal term by min3 bin (judge window)", "",
              "| sample | bin | real | worlds mean ± SD | worlds 5-95% |", "|---|---|---|---|---|"]
        for smp, B in ps["worlds_summary"].items():
            for lab, s in B.items():
                L.append(f"| {smp} | {lab} | {s['real']:.3f} | {s['mean']:.3f} ± {s['sd']:.3f} | [{s['p05']:.3f}, {s['p95']:.3f}] |")
        L.append("")
    for path in sorted(glob.glob(os.path.join(R, "check_allowed_*.json"))):
        c = json.load(open(path))
        ok = [x for x in c["cells"] if x["allowed"]]
        L += [f"Family {c['family']}: {len(ok)} of {len(c['cells'])} dial cells allowed by step 1: "
              + "; ".join(f"k={x['k']} d={x['delta']} a={x['a']}" for x in ok[:40]), ""]


KEY_P1 = ["constrained_mcts_sf - constrained_mcts", "constrained_mcts_ft - constrained_mcts",
          "constrained_mcts_ft_sf - constrained_mcts", "constrained_mcts_ft_sf - constrained_mcts_ft",
          "search_sf - constrained_mcts", "search_ft_sf - constrained_mcts", "mcts - constrained_mcts",
          "mcts_lcb - mcts_search", "mcq_t0.5 - constrained_mcts", "cql_enr_a2.0 - constrained_mcts",
          "iql_td_t0.7_b10.0 - constrained_mcts"]


def c_fmt(c):
    return f"{c['mean']:+.2f} [{c['lo95']:+.2f}, {c['hi95']:+.2f}] | {c['p_gt0']:.2f} | {c['worst10_mean']:+.2f}"


def rescore_section(L):
    r = J("rescore_p1.json")
    if r and "families" in r:
        L += ["## Rescoring: paper 1", "",
              "Per-world paired differences X - Y in true win-probability points over the common opponent set "
              "(OPP8 minus constrained_mcts). Columns: mean [2.5%, 97.5% over worlds] | P(X > Y) | mean of the worst 10% of worlds.", ""]
        for fam, F in r["families"].items():
            tags = [t for t in F["contrasts"] if not t.startswith("{")] or list(F["contrasts"])[:5]
            for tag in tags:
                L += [f"### Family {fam}, dials {tag} ({F['worlds']} worlds)", "", "| contrast | mean [95%] | P(X>Y) | worst 10% |", "|---|---|---|---|"]
                for k in KEY_P1:
                    if k in F["contrasts"][tag]:
                        L.append(f"| {k} | {c_fmt(F['contrasts'][tag][k])} |")
                L.append("")
            if "breakeven" in F:
                L += [f"Break-even (family {fam}): mean paired difference by full-grade shift delta (points; columns), "
                      "per plane; 'allowed' = deltas step 1 allows on that plane.", ""]
                for k, rows in F["breakeven"].items():
                    L += [f"**{k}**", "", "| plane | " + " | ".join(f"d={x['delta']:+g}" for x in rows[0]["by_delta"]) + " | break-even d | allowed d |",
                          "|---|" + "---|" * (len(rows[0]["by_delta"]) + 2)]
                    for row in rows:
                        be = row["breakeven_delta"]
                        L.append(f"| {row['plane']}={row['value']:g} | " + " | ".join(f"{x['mean_pts']:+.2f}" for x in row["by_delta"])
                                 + f" | {'none' if be is None else f'{be:+.1f}'} | {','.join(f'{d:g}' for d in row.get('allowed_deltas') or [])} |")
                    L.append("")
        s = r.get("support_lt25", {})
        if s:
            L += ["Share of teams below min3 25 per strategy: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(s.items())), ""]
    p = J("rescore_p2.json")
    if p:
        L += ["## Rescoring: paper 2", "", "Arm minus the selected design (x-combo1-dr-chance), mean over the common opponents, "
              "true win-probability points: mean [95% over worlds] | P(arm > selected) | worst 10% of worlds.", ""]
        for lset, fams in p.items():
            for fam, F in fams.items():
                for tag, T in F["arms"].items():
                    if tag.startswith("{") and fam != "f":
                        continue
                    L += [f"### {lset} lobbies, family {fam}, {tag} ({F['worlds']} worlds)", "", "| arm | value | minus selected | P(>) | worst 10% |", "|---|---|---|---|---|"]
                    for a, v in T.items():
                        m = v.get("minus_sel")
                        L.append(f"| {a} | {v['value_pts']:.2f} | " + (f"{m['mean']:+.2f} [{m['lo95']:+.2f}, {m['hi95']:+.2f}] | {m['p_gt0']:.2f} | {m['worst10_mean']:+.2f}" if m else " | | ") + " |")
                    L.append("")


def main():
    L = ["# Hidden worlds", "", "Generated by `worlds/report.py` from `worlds/site/results/*.json`.", ""]
    fit_section(L)
    L += ["## Plausibility checks", ""]
    checks_section(L)
    rescore_section(L)
    rc = J("check_recovery.json")
    if rc:
        L += ["### Recovery (family a: refit on one world's outcomes, coverage of the truth on probe games)", "",
              "| world | probes | min3 bin | n | cover 50% | cover 90% | z mean | z SD | post SD (pts) | abs err (pts) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for run in rc["runs"]:
            for b in run["bins"]:
                L.append(f"| {run['world']} | {b['probes']} | {b['min3_lo']}-{b['min3_hi'] or ''} | {b['n']} | {b['cover50']:.2f} | {b['cover90']:.2f} | "
                         f"{b['z_mean']:+.2f} | {b['z_sd']:.2f} | {b['post_sd_pts']:.2f} | {b['abs_err_pts']:.2f} |")
        L.append("")
    open(os.path.join(R, "WORLDS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
