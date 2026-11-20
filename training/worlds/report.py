"""
WORLDS.md from the results JSONs (site/results). python3 worlds/report.py
With --paper1: WORLDS_P1.md, the hero-generic subset for paper 1 (no player families, no paper-2 arms).
"""
import os
import sys
import json
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as WD  # noqa: E402

R = WD.RESULTS
P1 = "--paper1" in sys.argv
PLAYER_FAMS = ("f", "fc")


def keep(fam):
    return not (P1 and fam in PLAYER_FAMS)


def keep_pair(k):
    return all(keep(x) for x in k.split("~"))


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
        q = None if P1 else J(name)
        if q:
            L += [f"Player family ({name}): {q['games']:,} games; " + (f"SDs {json.dumps(q['sd'])}" if "sd" in q else f"s_alpha {q['s_a']}, s_eps {q['s_e']}")
                  + f"; nll pop {q['nll_pop']:.5f} -> with player term {q['nll_with_player']:.5f}.", ""]
    p = None if P1 else J("player_fit.json")
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
      ps = None if P1 else J(nm_)
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
            if not keep(fam):
                continue
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
    p = None if P1 else J("rescore_p2.json")
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


def cf(c, d=2):
    return f"{c['mean']:+.{d}f} [{c['lo95']:+.{d}f}, {c['hi95']:+.{d}f}]"


def step4_section(L):
    t1 = J("truthleaf_p1.json")
    if t1:
        L += ["## Truth-leaf search: paper 1", "", "Same floor_search code and 200 sims, leaf = world truth; opponents mcts, "
              "constrained_greedy, gd; 28 drafts per seat. Regret = truth-leaf counterpart minus agent (true pts).", "",
              "| setting | worlds | regret search_c (paired) | search_sf (paired) | search_ft_sf (paired) | constrained_mcts | "
              "constrained_mcts_sf | constrained_mcts_ft_sf | floor TL sf - TL c | Rehgar first-pick share TL c / real c |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        rc = t1["concentration"]["real_leaf"]["search_c"]["first_top_share"]
        for k, S in t1["settings"].items():
            R = S["regret_pts"]
            g = lambda n: cf(R[n]) if n in R else ""
            L.append(f"| {k} | {S['worlds']} | {g('search_c (real leaf, same seeds)')} | {g('search_sf (real leaf, same seeds)')} | "
                     f"{g('search_ft_sf (real leaf, same seeds)')} | {g('constrained_mcts')} | {g('constrained_mcts_sf')} | "
                     f"{g('constrained_mcts_ft_sf')} | {cf(S['floor_true_effect_pts']['TL:search_sf - TL:search_c'])} | "
                     f"{S['concentration_mean']['search_c']['first_top_share']:.2f} / {rc:.2f} |")
        L.append("")
    for name, opp in (("truthleaf_p2.json", "pMCTS-cur"), ("truthleaf_p2_aMCTS-pp.json", "aMCTS-pp")):
        t2 = None if P1 else J(name)
        if not t2:
            continue
        L += [f"## Truth-leaf search: paper 2 vs {opp}", "", "Support-kernel search, leaf = world truth (worlds/cuda_truth); "
              "300 headline lobbies, both seats. True win % and regret (TL counterpart minus arm).", "",
              "| family | worlds | TL-sel | selected | s-floor25 | s-pcal-floor25 | regret selected | regret s-floor25 | regret s-pcal-floor25 | floor TL-f25 - TL-sel |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, S in t2["settings"].items():
            v, R = S["value_pts"], S["regret"]
            L.append(f"| {k} | {S['worlds']} | {v['TL-sel']:.1f} | {v['x-combo1-dr-chance']:.1f} | {v['s-floor25']:.1f} | "
                     f"{v['s-pcal-floor25']:.1f} | {cf(R['x-combo1-dr-chance (vs TL-sel)'])} | {cf(R['s-floor25 (vs TL-f25)'])} | "
                     f"{cf(R['s-pcal-floor25 (vs TL-f25)'])} | {cf(S['floor_true_effect'])} |")
        L.append("")
    rt = J("retrain_p1.json")
    if rt:
        S = rt["summary"]
        L += ["## Retrained arm (paper 1, family a default)", "", f"Worlds {rt['worlds']}: outcomes redrawn, stats + features + "
              "enriched WP refit (seed 42), search with the refit leaf, scored by truth.", "",
              "| agent | true value | claimed | claimed - true | refit - real leaf | regret refit vs TL | regret real vs TL |", "|---|---|---|---|---|---|---|"]
        for a in ("search_c", "search_sf"):
            x = S[a]
            L.append(f"| {a} | {x['refit_true_pts']:.2f} | {x['refit_claimed_pts']:.2f} | {cf(x['winners_curse_claimed_minus_true'])} | "
                     f"{cf(x['refit_minus_real_leaf'])} | {cf(x['regret_refit_vs_truth_leaf'])} | {cf(x['regret_real_vs_truth_leaf'])} |")
        L += [f"| floor refit sf - c | {cf(S['floor_refit_sf_minus_c'])} | | | | | |", ""]


def explore_section(L):
    e = J("explore.json")
    if e:
        L += ["## Exploration (step 3)", "", f"{e['rounds']} rounds per run, {e['organic_per_round']} organic games per round, "
              "learner = family-a class; role filter + truth clip [0.25, 0.75]. Regret minus exploit: true pts per round, mean [95% over worlds].", "",
              "| family | c | policy | SD x | regret | minus exploit | followed | played explorations | true cost mean | P(cost > 2 pp) | below-floor plays |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in e["cells"]:
            if not keep(c["family"]):
                continue
            m = c["minus_exploit"]
            L.append(f"| {c['family']} | {c['compliance']} | {c['policy']} | {c['sd_mult']} | {c['mean_regret_pts']:.2f} | "
                     f"{cf(m, 3) if m else ''} | {c['followed_share']:.4f} | {c['played_explorations']:.1f} | {c['cost_mean_pts']:.2f} | "
                     f"{c['p_cost_gt_2pp']:.2f} | {c['below_floor_played']:.1f} |")
        L.append("")
    v = J("explore_voi.json")
    if v:
        L += ["### Value-of-information map", "", "Spearman of pair VOI across families: " +
              ", ".join(f"{k} {x:.2f}" for k, x in v["spearman_pair_voi"].items() if keep_pair(k)) + "; across worlds: " +
              ", ".join(f"{k} {x:.2f}" for k, x in v.get("spearman_across_worlds", {}).items() if keep(k)), ""]
        for f, t in v["top_pairs"].items():
            if not keep(f):
                continue
            L.append(f"- {f}: " + ", ".join(f"{a}+{b}" for a, b, _ in t[:8]))
        P = v["projection"]
        L += ["", "### Traffic scenarios (target-pair games per week)", "",
              "| mode | c | drafts/week | games/week | weeks to 25 | weeks to 600 | speed vs both |", "|---|---|---|---|---|---|---|"]
        for r in P["table"]:
            L.append(f"| {r['mode']} | {r['compliance']} | {r['drafts_per_week']} | {r['pair_games_per_week']:.2f} | "
                     f"{r['weeks_for_25_games']:.1f} | {r['weeks_for_600_games']:.1f} | {r['speed_vs_both']:.2f} |")
        L += ["", "### Directed push, 2,000 drafts/week at c = 20%", "",
              "| family | mode, horizon | followed games | pairs into support | expected true gain (pts per draft, upper bound) |", "|---|---|---|---|---|"]
        for f, R in P["push_2000_c20"].items():
            if not keep(f):
                continue
            for k, x in R.items():
                L.append(f"| {f} | {k} | {x['followed_games']:.0f} | {x['pairs_into_support']} | {x['expected_true_gain_pts_per_draft']:.3f} |")
        L.append("")


def findings(L):
    L += ["## Findings (numbers only)", ""]
    b, ll, pr = J("fit_base.json"), J("check_logloss.json"), J("check_pairs.json")
    if b and ll:
        L.append(f"- World model NODRIFT nll {b['NODRIFT_nll']:.4f} vs enriched WP {ll['agent_nll']:.4f}; higher-order layer SDs <= 0.005 logit (cross 0.002); hero drift SD 0.074 logit (T17).")
    if pr:
        a = pr["summary"]["a"]
        L.append(f"- Pair SD true: counter {a['counter']['sd_true_pts_mean']:.2f} / synergy {a['synergy']['sd_true_pts_mean']:.2f} pts (real 1.09 / 1.72).")
    s1 = J("check_step1_a_default.json")
    if s1:
        st = s1["stats"]
        L.append(f"- Step-1 stats in worlds (a default): below-floor beta {st['below_beta']['world_mean']:.2f} (real {st['below_beta']['real']:.2f}); "
                 f"cert 1-4 {st['cert_1']['world_mean']:+.2f} (real {st['cert_1']['real']:+.2f}, pct {st['cert_1']['real_percentile']:.0f}); allowed delta at a = 1: -3..0 pts.")
    r1 = J("rescore_p1.json")
    if r1:
        C = r1["families"]["a"]["contrasts"]["default"]
        L.append(f"- Paper 1 rescoring (a default, 1,000 worlds): sf - cmcts {cf(C['constrained_mcts_sf - constrained_mcts'])}, "
                 f"ft_sf - cmcts {cf(C['constrained_mcts_ft_sf - constrained_mcts'])}, search_ft_sf - cmcts {cf(C['search_ft_sf - constrained_mcts'])} pts; no break-even on the delta grid.")
        for fam in ("f", "fc"):
            if fam in r1["families"] and keep(fam):
                C = r1["families"][fam]["contrasts"]['{"assign": "best"}']
                L.append(f"- Paper 1 under {'f-out' if fam == 'f' else 'f-cov'} (best assignment): sf - cmcts {cf(C['constrained_mcts_sf - constrained_mcts'])}, "
                         f"search_ft_sf - cmcts {cf(C['search_ft_sf - constrained_mcts'])}.")
    r2 = None if P1 else J("rescore_p2.json")
    if r2:
        for lset in ("headline", "tune"):
            for fam, tag in (("a", "default"), ("f", '{"assign": "best"}'), ("fc", '{"assign": "best"}')):
                T = r2.get(lset, {}).get(fam, {}).get("arms", {}).get(tag)
                if T:
                    L.append(f"- Paper 2 {lset} {fam}: s-floor25 {cf(T['s-floor25']['minus_sel'])}, s-pcal-floor25 {cf(T['s-pcal-floor25']['minus_sel'])} pts vs selected.")
    ps, pc = J("check_player_slope2.json"), J("check_player_slope_cov.json")
    if ps and pc and not P1:
        L.append(f"- Personal-term slope (25-99 bin): real {ps['worlds_summary']['all_games']['25-99']['real']:.2f}; f-out {ps['worlds_summary']['all_games']['25-99']['mean']:.2f}; "
                 f"f-cov {pc['worlds_summary']['all_games']['25-99']['mean']:.2f} (half B).")
    t1 = J("truthleaf_p1.json")
    if t1:
        S = t1["settings"]["a/default"]
        L.append(f"- Paper 1 truth-leaf (a default): regret constrained_mcts {cf(S['regret_pts']['constrained_mcts'])}, search_c {cf(S['regret_pts']['search_c (real leaf, same seeds)'])}; "
                 f"floor under truth leaf {cf(S['floor_true_effect_pts']['TL:search_sf - TL:search_c'])}; Rehgar first-pick share {S['concentration_mean']['search_c']['first_top_share']:.2f} (TL) vs "
                 f"{t1['concentration']['real_leaf']['search_c']['first_top_share']:.2f} (real leaf).")
    for name, opp in (("truthleaf_p2.json", "pMCTS-cur"), ("truthleaf_p2_aMCTS-pp.json", "aMCTS-pp")):
        t2 = None if P1 else J(name)
        if t2:
            parts = [f"{k}: selected {t2['settings'][k]['regret']['x-combo1-dr-chance (vs TL-sel)']['mean']:.1f}, s-pcal-floor25 "
                     f"{t2['settings'][k]['regret']['s-pcal-floor25 (vs TL-f25)']['mean']:.1f}, floor {t2['settings'][k]['floor_true_effect']['mean']:+.2f}"
                     for k in t2["settings"]]
            L.append(f"- Paper 2 truth-leaf regret vs {opp} (pts): " + "; ".join(parts) + ".")
    rt = J("retrain_p1.json")
    if rt:
        S = rt["summary"]
        L.append(f"- Retrained arm ({len(rt['worlds'])} worlds): claimed - true search_c {cf(S['search_c']['winners_curse_claimed_minus_true'])}, "
                 f"refit - real leaf {cf(S['search_c']['refit_minus_real_leaf'])}; floor {cf(S['floor_refit_sf_minus_c'])}.")
    rc = J("check_recovery.json")
    if rc:
        h = [b_["cover90"] for r in rc["runs"] for b_ in r["bins"] if b_["probes"] == "human"]
        s = [b_["cover90"] for r in rc["runs"] for b_ in r["bins"] if b_["probes"] == "search"]
        zs = [b_["z_sd"] for r in rc["runs"] for b_ in r["bins"] if b_["probes"] == "search"]
        L.append(f"- Recovery (worlds {[r['world'] for r in rc['runs']]}): 90% coverage human {min(h):.2f}-{max(h):.2f}; search-chosen {min(s):.2f}-{max(s):.2f}, z SD {min(zs):.2f}-{max(zs):.2f}.")
    e = J("explore.json")
    if e:
        x = [c for c in e["cells"] if keep(c["family"]) and c["compliance"] == 0.2 and c["policy"] != "exploit" and c["minus_exploit"]]
        L.append(f"- Exploration (c = 20%): every policy minus exploit within {min(c_['minus_exploit']['mean'] for c_ in x):+.3f}..{max(c_['minus_exploit']['mean'] for c_ in x):+.3f} pts per round; "
                 f"followed share {min(c_['followed_share'] for c_ in x):.4f}-{max(c_['followed_share'] for c_ in x):.4f}; P(cost > 2 pp) {min(c_['p_cost_gt_2pp'] for c_ in x):.2f}-{max(c_['p_cost_gt_2pp'] for c_ in x):.2f}.")
    v = J("explore_voi.json")
    if v:
        P = v["projection"]
        r = {(t["mode"], t["compliance"], t["drafts_per_week"]): t for t in P["table"]}
        L.append(f"- Weeks to 25 games for one pair at 2,000 drafts/week, c = 20%: both {r[('both', 0.2, 2000)]['weeks_for_25_games']:.1f}, "
                 f"lobby {r[('lobby', 0.2, 2000)]['weeks_for_25_games']:.2f}, single {r[('single', 0.2, 2000)]['weeks_for_25_games']:.1f}; at 25 drafts/week both {r[('both', 0.2, 25)]['weeks_for_25_games']:.0f}.")
        g = P["push_2000_c20"]
        L.append("- 2,000/week push, 12 months, both-follow: expected true gain per draft (upper bound) " +
                 ", ".join(f"{f} {g[f]['both_12_months']['expected_true_gain_pts_per_draft']:.2f}" for f in g if keep(f)) + " pts.")
        L.append("- VOI rank stability: " + ", ".join(f"{k} {x:.2f}" for k, x in v["spearman_pair_voi"].items() if keep_pair(k)) + ".")
    L.append("")


def main():
    L = ["# Hidden worlds" + (": hero-generic worlds" if P1 else ""), "",
         "Generated by `worlds/report.py" + (" --paper1" if P1 else "") + "` from `worlds/site/results/*.json`.", ""]
    findings(L)
    fit_section(L)
    L += ["## Plausibility checks", ""]
    checks_section(L)
    rescore_section(L)
    step4_section(L)
    explore_section(L)
    rc = J("check_recovery.json")
    if rc:
        L += ["### Recovery (family a: refit on one world's outcomes, coverage of the truth on probe games)", "",
              "| world | probes | min3 bin | n | cover 50% | cover 90% | z mean | z SD | post SD (pts) | abs err (pts) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for run in rc["runs"]:
            for b in run["bins"]:
                L.append(f"| {run['world']} | {b['probes']} | {b['min3_lo']}-{b['min3_hi'] or ''} | {b['n']} | {b['cover50']:.2f} | {b['cover90']:.2f} | "
                         f"{b['z_mean']:+.2f} | {b['z_sd']:.2f} | {b['post_sd_pts']:.2f} | {b['abs_err_pts']:.2f} |")
        L.append("")
    open(os.path.join(R, "WORLDS_P1.md" if P1 else "WORLDS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
