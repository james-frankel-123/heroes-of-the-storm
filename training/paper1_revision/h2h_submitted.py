"""
Audit A3 fix for the SUBMITTED agents: constrained vs unconstrained MCTS head
to head with the SAME checkpoint on both sides (rerun2026/mcts_runs/
J_800sim_s9; the submission's chain omitted --mcts-run, so its "mcts" side was
L_800sim_4M_s0). Both orderings, 200 drafts each, phase3b map/tier
stratification, policy argmax (no search), scored by the independent
references. Output: results/h2h_submitted.json
"""
import os
import sys
import json
import random

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core

import numpy as np
import torch

CKPT = os.path.join(TRAINING_DIR, "rerun2026", "mcts_runs", "J_800sim_s9", "draft_policy.pt")


def main():
    from rerun2026.constrained_search import constrain, make_mcts_policy_strategy, load_mcts_policy
    from rerun2026.phase3_benchmarks import load_gd_models
    from shared import HEROES, NUM_HEROES, MAPS, SKILL_TIERS, is_degenerate
    from train_draft_policy import DraftState, DRAFT_ORDER
    from paper1_revision.score_tournament import score_rows, CONSENSUS
    torch.set_num_threads(4)
    gd = load_gd_models(torch.device("cpu"))
    net = load_mcts_policy(CKPT)
    plain = make_mcts_policy_strategy(net)
    cons = constrain(make_mcts_policy_strategy(net))
    out = {}
    for a_name, b_name, a, b in (("constrained", "plain", cons, plain), ("plain", "constrained", plain, cons)):
        random.seed(7)
        torch.manual_seed(7)
        rows, deg = [], {"team0": 0, "team1": 0}
        for i in range(200):
            gm, tier = MAPS[i % len(MAPS)], SKILL_TIERS[(i // len(MAPS)) % len(SKILL_TIERS)]
            st = DraftState(gm, tier, our_team=0)
            while not st.is_terminal():
                team, typ = DRAFT_ORDER[st.step]
                st.our_team = team
                f = a if team == 0 else b
                st.apply_action(f(st, team, typ, gm, tier, gd, torch.device("cpu")), team, typ)
            t0 = [HEROES[j] for j in range(NUM_HEROES) if st.team0_picks[j] > 0]
            t1 = [HEROES[j] for j in range(NUM_HEROES) if st.team1_picks[j] > 0]
            deg["team0"] += is_degenerate(t0)
            deg["team1"] += is_degenerate(t1)
            rows.append((t0, t1, gm, tier))
        sc = score_rows(rows)
        sc["consensus"] = np.mean([sc[k] for k in CONSENSUS], 0)
        out[f"{a_name}__{b_name}"] = {"team0_p": {k: float(np.mean(v)) for k, v in sc.items()},
                                      "degen": deg, "distinct_drafts": len({tuple(sorted(r[0])) + tuple(sorted(r[1])) for r in rows})}
        print(a_name, "vs", b_name, {k: round(float(np.mean(v)), 4) for k, v in sc.items()}, deg, flush=True)
    # constrained agent's mean P(win) over both orderings
    c = {k: 0.5 * (out["constrained__plain"]["team0_p"][k] + 1 - out["plain__constrained"]["team0_p"][k])
         for k in out["constrained__plain"]["team0_p"]}
    out["constrained_mean_p"] = c
    print("constrained mean P(win) vs plain, same checkpoint:", {k: round(v, 4) for k, v in c.items()})
    json.dump(out, open(os.path.join(core.RESULTS, "h2h_submitted.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
