"""
P3 tournament: render the result JSONs (results/p3t/*.json) as markdown
tables in results/P3_TOURNAMENT_TABLES.md, so the full-data rerun regenerates
every table with one command. Run locally after the pipeline fetched the JSONs:
  python3 training/personalization/p3t_report.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results", "p3t")
OUT = os.path.join(HERE, "results", "P3_TOURNAMENT_TABLES.md")


def j(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ci(v):
    return f"{v[0]:.1f} ({v[1]:.1f}, {v[2]:.1f})" if v else ""


MAIN = ["x-combo1-dr-chance", "x-combo1", "x-combo3", "x-combo1-dr", "x-pp-dr", "aMCTS-pp", "aMCTS-cur", "pMCTS-cur", "a1step-cur",
        "p1step-cur", "DraftRec", "DraftRec-pv", "Real", "GD"]


def cell(A, a, b, m):
    """Row a's win % against column b (both seatings), 95% CI; from either key order."""
    v = A["h2h"].get(f"{a} vs {b}")
    if v is not None and m in v:
        x = v[m]
        return x[0], x[1], x[2]
    v = A["h2h"].get(f"{b} vs {a}")
    if v is not None and m in v:
        x = v[m]
        return 100 - x[0], 100 - x[2], 100 - x[1]
    return None


def matrix(A):
    """The complete head-to-head matrix among the main systems, under J and CONS."""
    sysn = [s for s in MAIN if s in A["systems"]]
    L = []
    for m, title in (("judge_personal", "J (player-aware judge)"), ("consensus", "CONS (paper 1's consensus judges)"),
                     ("judge_cal", "J_cal (J with calibrated parts; sensitivity)")):
        L += ["", f"## Head-to-head matrix, {title}", "",
              "Cell = the row system's mean win % against the column system on the 300 tournament lobbies, both "
              "seatings, 95% CI (lobby bootstrap, widened for seed effects). Every cell is a played match; "
              "an empty cell was not played.", "",
              "| row vs column | " + " | ".join(sysn) + " | mean |", "|---" * (len(sysn) + 2) + "|"]
        for a in sysn:
            row, vals = [], []
            for b in sysn:
                if a == b:
                    row.append("")
                    continue
                c = cell(A, a, b, m)
                row.append(f"{c[0]:.1f} ({c[1]:.1f}, {c[2]:.1f})" if c else "")
                if c:
                    vals.append(c[0])
            mean = f"{sum(vals) / len(vals):.1f}" if vals and len(vals) == len(sysn) - 1 else ""
            L.append(f"| {a} | " + " | ".join(row) + f" | {mean} |")
    return L


def validation():
    """Paper-1 style validation tables (p3t_extra.py, p3t_qm.py, the sims curve and repeated runs)."""
    L = []
    W, Bh = j("wpseeds.json"), j("behavior.json")
    if W and Bh:
        L += ["", "## WP-seed rescoring and behavior (all tournament drafts of each system)", "",
              f"WP by seed: mean P(win) of the system's drafts against its opponents under paper 1's enriched WP, seeds "
              f"{', '.join(map(str, W['seeds']))} (the drafters' leaf is seed {W['leaf_seed']}); LCB = mean over drafts of the "
              "minimum over seeds. Opponent sets differ by system, so compare spread and LCB gap, not levels. GD similarity: share "
              "of the system's picks equal to GD's argmax at that state. Resilience: paper 1's, judge-window statistics.", "",
              "| system | WP s42 | WP s123 | WP s777 | seed spread pp | LCB | distinct heroes | pick entropy (bits) | top-10 share "
              "| GD similarity, picks | GD log-lik, picks | GD similarity, bans | resilience |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s_ in sorted(W["systems"], key=lambda x: -W["systems"][x]["wp_by_seed"][str(W["leaf_seed"])]):
            w, b = W["systems"][s_], Bh.get(s_, {})
            ws = w["wp_by_seed"]
            L.append(f"| {s_} | {ws['42']} | {ws['123']} | {ws['777']} | {w['spread_pp']} | {w['lcb_mean_min_over_seeds']} | "
                     f"{b.get('distinct_heroes')} | {b.get('pick_entropy_bits')} | {b.get('top10_share')} | "
                     f"{b.get('gd_similarity_picks')} | {b.get('gd_loglik_picks')} | {b.get('gd_similarity_bans')} | {b.get('resilience')} |")
    P = j("probes.json")
    if P:
        L += ["", "## Personal probes on the 300 real drafts", "",
              "| probe | lobbies | J change pp | share J ranks below real | CONS change pp | share CONS ranks below real | population WR change pp |",
              "|---|---|---|---|---|---|---|"]
        for k, v in P.items():
            if isinstance(v, dict):
                L.append(f"| {k} | {v['lobbies_with_probe']} | {v['judge_personal']['mean_change_pp']} | "
                         f"{v['judge_personal']['share_ranked_below_real']} | {v['consensus']['mean_change_pp']} | "
                         f"{v['consensus']['share_ranked_below_real']} | {v['population_wr_change_pp']} |")
    Qs = [(k, j(f"qm_check_k{k}.json")) for k in (0, 1)]
    if any(q for _, q in Qs):
        L += ["", "## Quick Match check of the personal term", "",
              "| games | known SL players per game | a (QM2026 logit) | c2 (SE) | c3 (SE) | SL judge c2 | log loss QM2026 | + personal (refit) | + SL coefficients |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k, q in Qs:
            if q:
                f = q["fit"]
                L.append(f"| {q['games_used']:,} ({'at least one per team' if k else 'at least one'}) | {q['known_players_per_game_mean']} | "
                         f"{f['a_QM2026'][0]} | {f['c2_skill'][0]} ({f['c2_skill'][1]}) | {f['c3_offrole'][0]} ({f['c3_offrole'][1]}) | "
                         f"{q['sl_judge']['c2_skill']} | {q['log_loss']['QM2026 alone (refit scale)']} | "
                         f"{q['log_loss']['QM2026 + personal (refit)']} | {q['log_loss']['QM2026 (refit scale) + SL judge' + chr(39) + 's personal coefficients']} |")
    T = j("analysis_tune.json")
    if T and any("x-sel-s100" in k for k in T["h2h"]):
        L += ["", "## Search-budget curve (tune lobbies, against pMCTS-cur at 400 sims)", "",
              "| sims | selected design, J | selected design, CONS | blind search, J | blind search, CONS |", "|---|---|---|---|---|"]
        def tc(a, m):
            v = T["h2h"].get(f"{a} vs pMCTS-cur")
            if v:
                return v[m][:3]
            v = T["h2h"].get(f"pMCTS-cur vs {a}")
            return [100 - v[m][0], 100 - v[m][2], 100 - v[m][1]] if v else None
        for n in (100, 200, 400, 800, 1600):
            sel = "x-combo1-dr-chance" if n == 400 else f"x-sel-s{n}"
            L.append(f"| {n} | {ci(tc(sel, 'judge_personal'))} | {ci(tc(sel, 'consensus'))} | "
                     f"{ci(tc(f'x-blind-s{n}', 'judge_personal'))} | {ci(tc(f'x-blind-s{n}', 'consensus'))} |")
    reps = [j(f"analysis_rep{n}.json") for n in (1, 2, 3)]
    A = j("analysis.json")
    if A and all(reps):
        import statistics
        L += ["", "## Repeated runs: the search's own noise (three extra kernel-seed offsets, same lobbies)", "",
              "| matchup | J, original | J, reruns | SD of the four J values | CONS SD |", "|---|---|---|---|---|"]
        for k in reps[0]["h2h"]:
            if k not in A["h2h"]:
                continue
            vals = [A["h2h"][k]["judge_personal"][0]] + [r["h2h"][k]["judge_personal"][0] for r in reps]
            cv = [A["h2h"][k]["consensus"][0]] + [r["h2h"][k]["consensus"][0] for r in reps]
            L.append(f"| {k} | {vals[0]:.2f} | {', '.join(f'{x:.2f}' for x in vals[1:])} | {statistics.stdev(vals):.2f} | {statistics.stdev(cv):.2f} |")
    return L


def main():
    L = ["# P3 tournament tables (generated by p3t_report.py)", ""]
    A = j("analysis.json")
    if A:
        L += [f"Lobbies: {A['lobbies']}; ordered pairs played: {A['pairs']}.", "",
              "## Head-to-head (row system's side, both seatings)", "",
              "J = player-aware judge (model-judged, primary). CONS = population consensus (model-judged). "
              "Personal part = judge's personal-sum difference (pp of team skill sum). Record = judge-free realized "
              "record per assigned player (pp). Future hero WR = judge-free. 95% CIs (lobby bootstrap, widened for seed effects).", "",
              f"J corrected = attenuation-corrected J (personal part divided by lambda = {A.get('attenuation_lambda')}; a model-based sensitivity).", "",
              "J_cal = J with the two parts calibrated on held-out judge-window games (1.17 logit p_cons + 1.00 personal term; a sensitivity check).", "",
              "| matchup | J win % | J_cal | J corrected | CONS win % | personal part (pp) | realized record diff (pp) | future hero WR diff | off-role diff |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k, v in A["h2h"].items():
            L.append(f"| {k} | {ci(v['judge_personal'])} | {ci(v.get('judge_cal'))} | {ci(v.get('judge_corr'))} | {ci(v['consensus'])} | {ci(v.get('dS_judge_pp'))} | "
                     f"{ci(v.get('diff_rec_mean'))} | {ci(v.get('diff_truth_hero'))} | {ci(v.get('diff_offrole'))} |")
        L += ["", "## Bradley-Terry strengths (logit, GD = 0) and standings", "",
              "| system | BT under J | BT under CONS | standing J % | standing CONS % | degenerate % | healer % | off-role players | GPU+CPU s per draft |",
              "|---|---|---|---|---|---|---|---|---|"]
        order = sorted(A["systems"], key=lambda s: -A["bt"]["judge_personal"]["theta"][s][0])
        for s in order:
            tj = A["bt"]["judge_personal"]["theta"][s]
            tc = A["bt"]["consensus"]["theta"][s]
            st = A["standings"][s]
            L.append(f"| {s} | {tj[0]:.3f} ({tj[1]:.3f}, {tj[2]:.3f}) | {tc[0]:.3f} ({tc[1]:.3f}, {tc[2]:.3f}) | "
                     f"{st['judge_personal']} | {st['consensus']} | {st['degen']} | {st['healer']} | {st['offrole']} | {st['sec_per_draft']} |")
        L.append(f"\nSeat (first pick) term: J {A['bt']['judge_personal']['seat_first_pick']}, CONS {A['bt']['consensus']['seat_first_pick']}.")
        L += matrix(A)
    if A:
        L += ["", "## Drift: drafters vs the prediction model at the same staleness", "",
              "Drafter loss = current twin's head-to-head win % over the stale version minus 50 (pp). Edge lost = "
              "BT logit gap current minus stale, as a share of the current drafter's BT edge over GD. "
              "Prediction lift lost = share of the log-loss lift over a coin flip lost at that staleness (tournament-window games).", "",
              "| family | staleness | judge | drafter loss pp (CI) | edge lost % |", "|---|---|---|---|---|"]
        fam = [("pMCTS", "pMCTS-cur", {"4m": "pMCTS-s4m", "1y": "pMCTS-s1y"}),
               ("aMCTS", "aMCTS-cur", {"4m": "aMCTS-s4m", "1y": "aMCTS-s1y"}),
               ("pPol", "pPol-cur", {"4m": "pPol-s4m", "1y": "pPol-s1y"}),
               ("p1step", "p1step-cur", {"1y": "p1step-s1y"}),
               ("a1step", "a1step-cur", {"1y": "a1step-s1y"}),
               ("aMCTS player state only", "aMCTS-cur", {"4m": "aMCTS-cur@s4m", "1y": "aMCTS-cur@s1y"}),
               ("aMCTS-pp player state only", "aMCTS-pp", {"4m": "aMCTS-pp@s4m", "1y": "aMCTS-pp@s1y"}),
               ("pMCTS refreshed statistics", "pMCTS-cur", {"4m": "pMCTS-s4m-rs", "1y": "pMCTS-s1y-rs"}),
               ("pMCTS fine-tuned", "pMCTS-cur", {"4m": "pMCTS-s4m-ft", "1y": "pMCTS-s1y-ft"})]
        for name, cur, st in fam:
            for lag, stale in st.items():
                hk = f"{cur} vs {stale}"
                if hk not in A["h2h"]:
                    continue
                for key, lab in (("consensus", "CONS"), ("judge_personal", "J")):
                    v = A["h2h"][hk][key]
                    th = A["bt"][key]["theta"]
                    edge = th[cur][0] - th.get("GD", [0])[0]
                    lost = (th[cur][0] - th[stale][0]) / edge if stale in th and abs(edge) > 1e-6 else float("nan")
                    L.append(f"| {name} | {lag} | {lab} | {v[0] - 50:.1f} ({v[1] - 50:.1f}, {v[2] - 50:.1f}) | {100 * lost:.0f} |")
        P_ = j("predeval.json")
        if P_:
            coin = P_["coin_ll"]
            lift = lambda ll: coin - ll
            for lag, v in (("4m", "s4m"), ("1y", "s1y")):
                wl = 1 - lift(P_[v]["wp_ll"]) / lift(P_["cur"]["wp_ll"])
                vl = 1 - lift(P_[v][f"V_ll_state_{v}"]) / lift(P_["cur"]["V_ll_state_cur"])
                L.append(f"| prediction: WP only | {lag} | log loss | | {100 * wl:.0f} (lift lost) |")
                L.append(f"| prediction: WP + player state | {lag} | log loss | | {100 * vl:.0f} (lift lost) |")
    if A:
        def get(a_, b_):
            v = A["h2h"].get(f"{a_} vs {b_}")
            if v:
                return v, False
            v = A["h2h"].get(f"{b_} vs {a_}")
            return v, True

        def w(v, rev, key):
            x = v[key]
            return (100 - x[0], 100 - x[2], 100 - x[1]) if rev else tuple(x[:3])
        T = j("analysis_tune.json") or {}
        TM = j("timing.json") or {}
        L += ["", "## Exploration design study: each component against aMCTS-pp (tournament lobbies)", "",
              "J and CONS: the row's win % against aMCTS-pp, both seats, 95% CI. Tune J: the selection score on the tune lobbies "
              "(mean J against pMCTS-cur and aMCTS-pp; aMCTS-pp itself scores about 54.8).", "",
              "| design | J vs aMCTS-pp | CONS vs aMCTS-pp | tune J |", "|---|---|---|---|"]
        abl = ["x-combo3", "x-combo1", "x-pp-ban3", "x-pp-opp2", "x-pp-as1", "x-pp-vn2", "x-pp-apos", "x-pp-s1600", "x-pp-c4",
               "x-pp-c1", "x-pp-b05", "x-pp-b2", "x-pp-a03", "x-pp-a0", "x-pp-opt05", "x-pp-opt1", "x-rich-search",
               "x-rich-bcow", "x-rich-onestep", "aPol-rich-search", "aPol-rich-bcow", "aPol-rich-onestep", "x-pp"]
        for x in abl:
            v, rev = get(x, "aMCTS-pp")
            if not v:
                continue
            tj = T.get("tune", {}).get(x, {}).get("judge_personal", "")
            L.append(f"| {x} | {ci(w(v, rev, 'judge_personal'))} | {ci(w(v, rev, 'consensus'))} | {tj} |")
    S_ = j("analysis_split.json")
    if S_:
        L += ["", "## Split-history judge (drafter state from odd days, judge from even days; valid contrasts only)", "",
              f"Attenuation slope of this judge on real teams: {S_.get('attenuation_lambda')}.", "",
              "| matchup | J split-history | J window (same drafts) | CONS |", "|---|---|---|---|"]
        for a_, b_ in (("x-combo3-split", "pMCTS-cur"), ("x-combo3-split", "pMCTS-bc"), ("x-pp-split", "pMCTS-cur"),
                       ("x-pp-split", "pMCTS-bc"), ("aMCTS-bc@split", "pMCTS-bc"), ("a1step@split", "p1step-cur")):
            vs_ = S_["h2h"].get(f"{a_} vs {b_}")
            rs = False
            if not vs_:
                vs_, rs = S_["h2h"].get(f"{b_} vs {a_}"), True
            vw_, rw = get(a_, b_) if A else (None, False)
            if vs_:
                L.append(f"| {a_} vs {b_} | {ci(w(vs_, rs, 'judge_personal'))} | {ci(w(vw_, rw, 'judge_personal')) if vw_ else ''} | "
                         f"{ci(w(vs_, rs, 'consensus'))} |")
    D = j("diag.json")
    if D:
        L += ["", "## Diagnostics: drafter's own personal sum vs the judge's", "",
              f"Real teams: S_D {D['real']['S_D_pp']} pp, S_J {D['real']['S_J_pp']} pp, slope of S_J on S_D {D['real']['slope_SJ_on_SD']} (corr {D['real']['corr']}).", "",
              "| system | S_D (drafter state) pp | S_J (judge state) pp | S_D minus real | S_J minus real | slope S_J on S_D |",
              "|---|---|---|---|---|---|"]
        for s, v in sorted(D["systems"].items(), key=lambda x: -x[1]["S_J_pp"]):
            L.append(f"| {s} | {v['S_D_pp']} | {v['S_J_pp']} | {v['S_D_minus_real_pp']} | {v['S_J_minus_real_pp']} | {v['slope_SJ_on_SD']} |")
    P = j("predeval.json")
    if P:
        coin = P["coin_ll"]
        L += ["", "## Prediction quality at staleness (tournament-window games, real drafts and outcomes)", "",
              "| vintage | WP log-loss lift x1000 | WP acc | value lift, current player state | value lift, player state of the vintage | value acc (vintage state) |",
              "|---|---|---|---|---|---|"]
        for v in ("cur", "s4m", "s1y"):
            r = P[v]
            vs = r.get(f"V_ll_state_{v}", r["V_ll_state_cur"])
            va = r.get(f"V_acc_state_{v}", r["V_acc_state_cur"])
            L.append(f"| {v} | {1000 * (coin - r['wp_ll']):.1f} | {100 * r['wp_acc']:.1f} | {1000 * (coin - r['V_ll_state_cur']):.1f} | "
                     f"{1000 * (coin - vs):.1f} | {100 * va:.1f} |")
    for name, title in (("backtest_agree_cur_bc.json", "Backtest (a), behavioral-prior search drafter"),
                        ("backtest_agree_cur.json", "Backtest (a), paper-1 F-prior search drafter")):
        B = j(name)
        if B:
            L += ["", f"## {title}", "", f"Games {B['games']:,}, picks {B['picks']:,}; real pick = aware recommendation "
                  f"{100 * B['match_rate_aware']:.1f}%, = blind {100 * B['match_rate_blind']:.1f}%, aware = blind "
                  f"{100 * B['aware_equals_blind']:.1f}%, aware-only matches {100 * B['match_rate_aware_only']:.2f}%.", "",
                  "| model | coefficients, pp (SE) |", "|---|---|"]
            for k, v in B["models"].items():
                L.append(f"| {k} | " + "; ".join(f"{c} {x[0]} ({x[1]})" for c, x in v.items() if c != "const") + " |")
    F = j("backtest_forced.json")
    if F:
        L += ["", "## Backtest (b): forced switches (within player, realized vs predicted)", "",
              "| situation | share of slots | players | realized pp (SE) | predicted pp | realized minus predicted (SE) |",
              "|---|---|---|---|---|---|"]
        for k, v in F.items():
            if isinstance(v, dict):
                L.append(f"| {k} | {100 * v['share_of_slots']:.1f}% | {v['players']:,} | {v['realized_pp']} ({v['realized_se']}) | "
                         f"{v['predicted_pp']} | {v['realized_minus_predicted_pp']} ({v['realized_minus_predicted_se']}) |")
    L += validation()
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
