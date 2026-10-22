"""
P3 party drafter: capture ratios and position diagnostics (reads p3t_party.py outputs).

  capture   gain(m) / gain(5), where gain = the party team's J (or J_cal, CONS) with
            the aware drafter minus the generic party, both seats pooled; CI by a
            lobby bootstrap that resamples the same lobbies for numerator and
            denominator
  position  for solo users at each of the team's five pick turns: how often the
            player's own best hero (largest drafter skill s among heroes he has
            played) is still free at his turn in the generic draft; the drafter
            skill of the hero he ends up on (aware vs generic); the share of the
            opponents' picks made before his turn; the judge-window counter score
            of his pick against the opponents who pick after him
Output: results p3t/party_report.json
Usage (GPU host or any host with the party files): python personalization/p3t_party_report.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("P1_TIERS", "site")
import numpy as np

import p3_hs_core as C
import p3t_party as PT


def load_views():
    import p3t_judge as JG
    judge = json.load(open(os.path.join(JG.RES, JG.JNAME)))
    out = {}
    sd = os.path.join(PT.OUT, "scores")
    for f in sorted(os.listdir(sd)):
        if not f.endswith(".npz"):
            continue
        z = np.load(os.path.join(PT.OUT, f))
        out[f[:-4]] = (PT.team_view(dict(np.load(os.path.join(sd, f))), int(z["t"]), judge), z)
    return out


def capture(V, B=2000):
    res = {}
    rng = np.random.RandomState(1)
    gains = {}
    for m in (1, 2, 3, 4, 5):
        parts = {met: [] for met in ("J", "J_cal", "CONS", "record")}
        for t in (0, 1):
            a, g = V.get(f"eff_m{m}_t{t}__aware"), V.get(f"eff_m{m}_t{t}__generic")
            if a is None or g is None:
                break
            for met in parts:
                parts[met].append(a[0][met] - g[0][met])
        else:
            gains[m] = {met: np.concatenate(v) for met, v in parts.items()}
    if 5 not in gains:
        return res
    n = len(gains[5]["J"])
    idx = [rng.randint(0, n, n) for _ in range(B)]
    for m, gm in gains.items():
        res[m] = {}
        for met in ("J", "J_cal", "CONS"):
            num, den = gm[met], gains[5][met]
            r = num.mean() / den.mean()
            bs = [num[i].mean() / den[i].mean() for i in idx]
            res[m][met] = [round(float(r), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    return res


def positions(V):
    import p3t_judge as JG
    import p3t_tourney as TT
    LB = TT.Lobbies()
    truth = JG.Truth()
    s_cur = LB.state["cur"]["s"].astype(np.float64)
    n_cur = LB.state["cur"]["n"]
    res = {}
    for t in (0, 1):
        for q in range(5):
            key = f"grid_m1_{q}_t{t}"
            if f"{key}__generic" not in V:
                key = f"pos_m1_p{q}_t{t}"
            if f"{key}__generic" not in V or f"{key}__aware" not in V:
                continue
            g, a = V[f"{key}__generic"][1], V[f"{key}__aware"][1]
            n = len(g["acts"])
            party = g["party"]
            sl = np.argmax(party, 1)                        # the solo player's slot
            step = PT.PICK_STEPS[PT.TEAM_TURNS[t][q]]
            ar = np.arange(n)
            pool = n_cur[ar, sl] > 0                        # heroes he has played
            sk = np.where(pool, s_cur[ar, sl], -np.inf)
            best = sk.argmax(1)
            before = g["acts"][:, :step]
            free_best = np.array([best[i] not in set(before[i].tolist()) for i in range(n)])
            # the hero he ends on: his own pick at his turn
            hg, ha = g["acts"][:, step], a["acts"][:, step]
            opp_before = sum(1 for k in range(step) if PT.IS_PICK[k] and PT.DRAFT_TEAM[k] != t)
            ctr = []
            for arm_acts in (a["acts"], g["acts"]):
                cs = []
                for i in range(n):
                    ti = LB.tier[i]
                    w = truth.wr[ti]
                    h = arm_acts[i, step]
                    later = [arm_acts[i, k] for k in range(step + 1, 16) if PT.IS_PICK[k] and PT.DRAFT_TEAM[k] != t]
                    d = [truth.co[ti, h, b] - (w[h] + (100 - w[b]) - 50) for b in later]
                    d = np.array(d, float)
                    cs.append(np.nanmean(d) if np.isfinite(d).any() else np.nan)
                ctr.append(np.nanmean(cs))
            res[key] = {"draft_step": int(step), "opponent_picks_before": int(opp_before),
                        "best_hero_free_at_turn_generic": round(float(free_best.mean()), 3),
                        "skill_of_pick_aware": round(float(100 * s_cur[ar, sl, ha].mean()), 2),
                        "skill_of_pick_generic": round(float(100 * s_cur[ar, sl, hg].mean()), 2),
                        "pick_is_his_best_hero_aware": round(float((ha == best).mean()), 3),
                        "pick_is_his_best_hero_generic": round(float((hg == best).mean()), 3),
                        "counter_vs_later_opponents_aware": round(float(ctr[0]), 3),
                        "counter_vs_later_opponents_generic": round(float(ctr[1]), 3),
                        "holds_bans": bool(q == 0)}
    return res


def main():
    V = load_views()
    out = {"capture": capture(V), "positions_solo": positions(V)}
    os.makedirs(os.path.join(C.RESULTS, "p3t"), exist_ok=True)
    json.dump(out, open(os.path.join(C.RESULTS, "p3t", PT.RES_NAME + "_report.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:4000], flush=True)


if __name__ == "__main__":
    main()
