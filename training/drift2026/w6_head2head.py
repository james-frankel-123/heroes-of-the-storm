"""
W6 — head-to-head: drift-maintained MCTS vs frozen-stats MCTS.

The headline experiment Max asked for (2026-07-13): what is drift-aware
maintenance WORTH in win probability, measured the same way paper 1 settles
arguments — direct head-to-head drafting, scored by the four paper-1 WP
evaluators (all trained on the full pinned snapshot, i.e. future-inclusive,
and none trained by this experiment's arms)?

Arms (both trained by phase_w4_mcts, 200 sims, 300K episodes, 5 seeds):
  - d2c_cumprev  — value function trained with per-build cumulative-refresh
                   stats (the drift-maintained system)
  - d2b_allhist  — value function trained on all history with cutoff-frozen
                   stats (the "deploy and forget" system)

Protocol: every (cumprev_seed, allhist_seed) pairing (5x5=25), both first-pick
orderings, N drafts each, stratified maps x tiers — phase3b tournament loop
verbatim (policy nets act greedily via their policy heads, as in the paper-1
tournament's MCTS entrant). Consensus = mean of the four evaluators'
symmetrized P(win).

Usage: python3 drift2026/w6_head2head.py [--drafts-per-cell 40]
Output: drift2026/results/w6_head2head.json + W6_HEAD2HEAD.md
"""
import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rerun2026 import common

common.setup()

import numpy as np
import torch

from shared import is_degenerate, NUM_HEROES, HEROES, HERO_ROLE_FINE, MAPS, SKILL_TIERS
from rerun2026.phase3b_roundrobin import load_wp_evaluators, score_terminal
from train_draft_policy import AlphaZeroDraftNet, DraftState, DRAFT_ORDER

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "mcts_runs")
N_SEEDS = 5


def load_policy(arm, seed, prefix="w4"):
    """prefix may be comma-separated (e.g. "w4,w8"): first existing run wins
    (W8b pools w4_d2b_allhist_s0..4 with w8_d2b_allhist_s5..14)."""
    for p in prefix.split(","):
        ckpt = os.path.join(RUNS, f"{p}_{arm}_s{seed}", "draft_policy.pt")
        if os.path.exists(ckpt):
            break
    net = AlphaZeroDraftNet(size="base", policy_head_type="linear")
    net.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
    net.eval()
    return net


def make_actor(net):
    def act(state):
        x = torch.tensor(state.to_numpy(), dtype=torch.float32).unsqueeze(0)
        m = state.valid_mask(torch.device("cpu"))
        with torch.no_grad():
            logits, _ = net(x, m)
        return logits.argmax(dim=1).item()
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts-per-cell", type=int, default=40)
    ap.add_argument("--arm-a", default="d2c_cumprev", help="maintained side")
    ap.add_argument("--arm-a-prefix", default="w4")
    ap.add_argument("--arm-b", default="d2b_allhist", help="stale side")
    ap.add_argument("--arm-b-prefix", default="w4")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--n-seeds-a", type=int, default=N_SEEDS)
    ap.add_argument("--n-seeds-b", type=int, default=N_SEEDS)
    ap.add_argument("--stagger-configs", action="store_true",
                    help="offset the (map, tier) config index by cell rank so "
                         "a small drafts-per-cell still covers all 42 map x "
                         "tier combos across the grid (identical configs for "
                         "both orderings and both sides; default preserves "
                         "the stored W6 protocol exactly)")
    args = ap.parse_args()
    global ARM_A, ARM_B, OUT_JSON, OUT_MD
    ARM_A, ARM_B = args.arm_a, args.arm_b
    OUT_JSON = os.path.join(HERE, "results", f"w6_head2head{args.out_suffix}.json")
    OUT_MD = os.path.join(HERE, "results", f"W6_HEAD2HEAD{args.out_suffix.upper()}.md")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = common.stats_cache()
    evaluators = load_wp_evaluators(device)
    healers = set(h for h, r in HERO_ROLE_FINE.items() if r == "healer")

    na, nb = args.n_seeds_a, args.n_seeds_b
    nets_a = [load_policy(ARM_A, s, args.arm_a_prefix) for s in range(na)]
    nets_b = [load_policy(ARM_B, s, args.arm_b_prefix) for s in range(nb)]
    print(f"loaded {na}+{nb} policies; "
          f"{na*nb*2*args.drafts_per_cell} drafts total", flush=True)

    per_draft = []       # consensus sym WP for the MAINTAINED side
    per_pairing = {}     # (sa, sb) -> list of maintained-side WPs
    records = []         # terminal drafts, for out-of-family (QM) rescoring
    side_stats = {ARM_A: {"healer": [], "degen": []},
                  ARM_B: {"healer": [], "degen": []}}

    for sa in range(na):
        for sb in range(nb):
            for order in (0, 1):  # 0: maintained first-pick, 1: frozen first-pick
                random.seed(7000 + sa * 100 + sb * 10 + order)
                torch.manual_seed(7000 + sa * 100 + sb * 10 + order)
                act_first = make_actor(nets_a[sa]) if order == 0 else make_actor(nets_b[sb])
                act_second = make_actor(nets_b[sb]) if order == 0 else make_actor(nets_a[sa])
                base = ((sa * nb + sb) * args.drafts_per_cell
                        if args.stagger_configs else 0)
                for i in range(args.drafts_per_cell):
                    ci = base + i
                    game_map = MAPS[ci % len(MAPS)]
                    tier = SKILL_TIERS[(ci // len(MAPS)) % len(SKILL_TIERS)]
                    state = DraftState(game_map, tier, our_team=0)
                    while not state.is_terminal():
                        step_team, step_type = DRAFT_ORDER[state.step]
                        state.our_team = step_team
                        actor = act_first if step_team == 0 else act_second
                        state.apply_action(actor(state), step_team, step_type)
                    t0h = [HEROES[j] for j in range(NUM_HEROES) if state.team0_picks[j] > 0]
                    t1h = [HEROES[j] for j in range(NUM_HEROES) if state.team1_picks[j] > 0]
                    wp = score_terminal(t0h, t1h, game_map, tier, evaluators, stats, device)
                    cons_t0 = float(np.mean([v["sym"] for v in wp.values()]))
                    maintained_wp = cons_t0 if order == 0 else 1.0 - cons_t0
                    per_draft.append(maintained_wp)
                    per_pairing.setdefault(f"{sa}_{sb}", []).append(maintained_wp)
                    mh = t0h if order == 0 else t1h
                    fh = t1h if order == 0 else t0h
                    records.append({"maintained": mh, "frozen": fh,
                                    "game_map": game_map, "tier": tier,
                                    "consensus_maintained": maintained_wp,
                                    "sa": sa, "sb": sb, "order": order})
                    side_stats[ARM_A]["healer"].append(any(h in healers for h in mh))
                    side_stats[ARM_A]["degen"].append(is_degenerate(mh))
                    side_stats[ARM_B]["healer"].append(any(h in healers for h in fh))
                    side_stats[ARM_B]["degen"].append(is_degenerate(fh))
            print(f"pairing s{sa} vs s{sb} done "
                  f"(running maintained WP {np.mean(per_draft):.4f})", flush=True)

    arr = np.array(per_draft)
    pairing_means = np.array([np.mean(v) for v in per_pairing.values()])
    result = {
        "arms": {"maintained": ARM_A, "frozen": ARM_B},
        "n_seeds": {"a": na, "b": nb},
        "drafts_per_cell": args.drafts_per_cell,
        "stagger_configs": bool(args.stagger_configs),
        "n_drafts": len(arr),
        "maintained_consensus_wp": float(arr.mean()),
        "se_across_drafts": float(arr.std(ddof=1) / np.sqrt(len(arr))),
        "se_across_pairings": float(pairing_means.std(ddof=1) / np.sqrt(len(pairing_means))),
        "pairing_means": {k: float(np.mean(v)) for k, v in per_pairing.items()},
        "pairings_won": int((pairing_means > 0.5).sum()),
        "n_pairings": len(pairing_means),
        "win_share_drafts": float((arr > 0.5).mean()),
        "side_rates": {arm: {k: float(np.mean(v) * 100) for k, v in d.items()}
                       for arm, d in side_stats.items()},
        "records": records,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    lines = [
        "# W6 — head-to-head: drift-maintained vs frozen MCTS",
        "",
        f"{na}x{nb} seed pairings x 2 orderings x "
        f"{args.drafts_per_cell} drafts = {len(arr)} drafts; consensus of the "
        "four paper-1 evaluators, symmetrized; maps x tiers stratified.",
        "",
        f"- **Maintained ({ARM_A}) consensus WP vs frozen ({ARM_B}): "
        f"{arr.mean():.4f}** (SE {result['se_across_drafts']:.4f} across "
        f"drafts, {result['se_across_pairings']:.4f} across pairings)",
        f"- Seed pairings won: {result['pairings_won']}/{result['n_pairings']}",
        f"- Draft-level win share (consensus > 0.5): {100*result['win_share_drafts']:.1f}%",
        f"- Maintained side: {result['side_rates'][ARM_A]['healer']:.1f}% healer, "
        f"{result['side_rates'][ARM_A]['degen']:.1f}% degen | Frozen side: "
        f"{result['side_rates'][ARM_B]['healer']:.1f}% healer, "
        f"{result['side_rates'][ARM_B]['degen']:.1f}% degen",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
