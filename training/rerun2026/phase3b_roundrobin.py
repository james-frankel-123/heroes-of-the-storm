"""
Phase 3b: head-to-head round-robin tournament between draft strategies.

Strategies (9): MCTS best policy (historical checkpoint), CQL alpha=1.0
(naive), CQL alpha=2.0 (enriched), GD behavioral cloning, enriched greedy,
enriched+aug greedy, MCQ tau=0.5, Gourdeau WP greedy (gourdeau_wp.pt +
GD rollouts, the phase3 rich-eval strategy), Gourdeau discriminator
(gourdeau_discriminator.pt, min-P(generated) at every step).

Every ORDERED pair (A, B), A != B, plays N drafts with A on team 0 (first
pick) and B on team 1 — both first-pick assignments are therefore covered by
the two orderings of each pair. Maps/tiers are stratified: draft i uses
MAPS[i % 14], SKILL_TIERS[(i // 14) % 3].

Per terminal draft we record, for BOTH sides: composition metrics,
interaction metrics (counter responsiveness / synergy / temporal resilience
+ counter quality from experiment_draft_quality.py), and the terminal state
scored by ALL WP variants (naive, hero-strength, enriched, augmented; raw
P(team0 wins) plus the team-swap symmetrized value). The adjudication metric
for the paper is TBD (see MANIFEST.md) — everything needed to compute any
candidate metric is stored.

Usage:
    python rerun2026/phase3b_roundrobin.py --dry-run
    python rerun2026/phase3b_roundrobin.py                    # pool over pairs
    python rerun2026/phase3b_roundrobin.py --pair enriched__gd --drafts 200
    python rerun2026/phase3b_roundrobin.py --aggregate-only
"""
import os
import sys
import json
import time
import random
import argparse
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rerun2026 import common
from rerun2026.common import MODELS_DIR, RESULTS_DIR, Job

common.setup()

import numpy as np
import torch

from shared import (is_degenerate, NUM_HEROES, HEROES, HERO_ROLE_FINE,
                    MAPS, SKILL_TIERS)
from sweep_enriched_wp import (WinProbEnrichedModel, FEATURE_GROUPS,
                               compute_group_indices, extract_features)
from experiment_synthetic_augmentation import ENRICHED_GROUPS
from train_draft_policy import DraftState, DRAFT_ORDER

STRATEGIES = ["mcts", "cql_naive_a1.0", "cql_enr_a2.0", "gd",
              "enriched", "enriched_aug", "mcq_t0.5",
              # tournament extension (2026-07): Gourdeau WP greedy + Gourdeau
              # discriminator (min P(generated)). Appended so existing pair
              # files/indices are untouched.
              "gourdeau", "gourdeau_disc"]

WP_EVALUATORS = {  # name -> (checkpoint, arch, groups preset)
    "naive": ("wp_naive.pt", [256, 128], []),
    "herostrength": ("wp_herostrength.pt", [256, 128], ["hero_wr", "team_avg_wr"]),
    "enriched": ("wp_enriched_256.pt", [256, 128], ENRICHED_GROUPS),
    "augmented": ("wp_aug_v2_512.pt", [512, 256, 128], ENRICHED_GROUPS),
}

DEFAULT_MCTS_RUNS = ["L_800sim_4M_s0", "J_800sim_s0", "E_1M_s0", "B_fullwp_s0"]


def mpath(*p):
    return os.path.join(MODELS_DIR, *p)


def rpath(*p):
    path = os.path.join(RESULTS_DIR, *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _cols(groups):
    gi = compute_group_indices()
    cols = []
    for g in groups:
        s, e = gi[g]
        cols.extend(range(s, e))
    return cols


def _find_mcts_checkpoint(preferred):
    runs = [preferred] if preferred else []
    runs += [r for r in DEFAULT_MCTS_RUNS if r not in runs]
    for run in runs:
        for base in (common.MCTS_RUNS_DIR, os.path.join(common.TRAINING_DIR, "mcts_runs")):
            p = os.path.join(base, run, "draft_policy.pt")
            if os.path.exists(p):
                return run, p
    raise FileNotFoundError(
        "No historical MCTS checkpoint found under training/mcts_runs/ "
        f"(tried {runs}). Pass --mcts-run, or rerun MCTS training (phase 4, "
        "see MANIFEST.md).")


def build_actor(name, device, stats, group_indices, gd_models, mcts_run):
    """Return act(state, team, step_type, game_map, tier) -> hero_idx.
    Reuses the strategy factories from experiment_rich_evaluation.py; for
    factories that read state.our_team, the caller sets it to the acting
    team before each call."""
    from experiment_rich_evaluation import (
        make_gd_strategy, make_cql_strategy, make_cql_enriched_strategy,
        make_wp_greedy_strategy)

    if name == "gd":
        inner = make_gd_strategy()
    elif name == "cql_naive_a1.0":
        inner = make_cql_strategy(mpath("cql", "_cql_temp_a1.0.pt"), device)
    elif name == "cql_enr_a2.0":
        inner = make_cql_enriched_strategy(mpath("cql", "_cql_enriched_a2.0.pt"),
                                           device, stats, group_indices,
                                           ENRICHED_GROUPS)
    elif name == "mcq_t0.5":
        inner = make_cql_strategy(mpath("mcq", "_mcq_temp_t0.5.pt"), device)
    elif name == "enriched":
        m = WinProbEnrichedModel(283, [256, 128], dropout=0.3).to(device)
        m.load_state_dict(torch.load(mpath("wp_enriched_256.pt"),
                                     weights_only=True, map_location=device))
        m.eval()
        inner = make_wp_greedy_strategy(m, ENRICHED_GROUPS, stats,
                                        group_indices, device)
    elif name == "enriched_aug":
        m = WinProbEnrichedModel(283, [512, 256, 128], dropout=0.3).to(device)
        m.load_state_dict(torch.load(mpath("wp_aug_v2_512.pt"),
                                     weights_only=True, map_location=device))
        m.eval()
        inner = make_wp_greedy_strategy(m, ENRICHED_GROUPS, stats,
                                        group_indices, device)
    elif name == "gourdeau":
        # Gourdeau WP greedy (phase3 rich-eval strategy, verbatim factory)
        from experiment_rich_evaluation import make_gourdeau_greedy_strategy
        from train_gourdeau_baseline import GourdeauWPModel
        gm = GourdeauWPModel().to(device)
        gm.load_state_dict(torch.load(mpath("gourdeau_wp.pt"),
                                      weights_only=True, map_location=device))
        gm.eval()
        inner = make_gourdeau_greedy_strategy(gm, device)
    elif name == "gourdeau_disc":
        # Gourdeau discriminator: minimize P(generated) at every step
        # (train_gourdeau_discriminator rich-stage strategy, verbatim factory)
        from rerun2026.train_gourdeau_discriminator import (make_disc_strategy,
                                                            load_model)
        inner = make_disc_strategy(load_model(device), device)
    elif name == "mcts":
        from train_draft_policy import AlphaZeroDraftNet
        run, ckpt = _find_mcts_checkpoint(mcts_run)
        net = AlphaZeroDraftNet()
        sd = torch.load(ckpt, weights_only=True, map_location="cpu")
        if any(k.startswith("res_block1.") for k in sd):
            sd = {k.replace("res_block1.", "res_blocks.0.")
                   .replace("res_block2.", "res_blocks.1.")
                   .replace("res_block3.", "res_blocks.2."): v
                  for k, v in sd.items()}
        net.load_state_dict(sd)
        net.eval()
        print(f"MCTS policy: {run}")

        def act(state, team, step_type, game_map, tier):
            x = torch.tensor(state.to_numpy(), dtype=torch.float32).unsqueeze(0)
            m = state.valid_mask(torch.device("cpu"))
            with torch.no_grad():
                logits, _ = net(x, m)
            return logits.argmax(dim=1).item()
        return act
    else:
        raise ValueError(name)

    def act(state, team, step_type, game_map, tier):
        return inner(state, team, step_type, game_map, tier, gd_models, device)
    return act


def load_wp_evaluators(device):
    out = {}
    for name, (fn, arch, groups) in WP_EVALUATORS.items():
        cols = _cols(groups)
        m = WinProbEnrichedModel(197 + len(cols), arch, dropout=0.3).to(device)
        m.load_state_dict(torch.load(mpath(fn), weights_only=True,
                                     map_location=device))
        m.eval()
        out[name] = (m, cols)
    return out


def score_terminal(t0h, t1h, game_map, tier, evaluators, stats, device):
    """P(team0 wins) per WP variant, raw + team-swap symmetrized."""
    all_mask = [True] * len(FEATURE_GROUPS)
    scores = {}
    for name, (model, cols) in evaluators.items():
        def run(a, b):
            d = {"team0_heroes": a, "team1_heroes": b, "game_map": game_map,
                 "skill_tier": tier, "winner": 0}
            base, enr = extract_features(d, stats, all_mask)
            x = np.concatenate([base, enr[cols]]) if cols else base
            with torch.no_grad():
                return model(torch.tensor(x, dtype=torch.float32
                                          ).unsqueeze(0).to(device)).item()
        raw = run(t0h, t1h)
        swapped = run(t1h, t0h)
        scores[name] = {"raw": raw, "sym": (raw + (1.0 - swapped)) / 2.0}
    return scores


def side_metrics(pick_steps, opp_pick_steps, stats, tier):
    """Composition + interaction + temporal metrics for one side.
    pick_steps: [(hero, step)] for this side, opp likewise."""
    from experiment_draft_quality import (draft_resilience, draft_counter_quality,
                                          incremental_synergy)
    from experiment_rich_evaluation import (counter_responsiveness,
                                            synergy_exploitation)
    heroes = [h for h, _ in pick_steps]
    opp_heroes = [h for h, _ in opp_pick_steps]
    labeled = ([(h, "ours", s) for h, s in pick_steps] +
               [(h, "theirs", s) for h, s in opp_pick_steps])
    healer_heroes = set(h for h, r in HERO_ROLE_FINE.items() if r == "healer")
    res = draft_resilience(labeled, stats, tier)
    ctr = draft_counter_quality(labeled, stats, tier)
    syn = incremental_synergy(labeled, stats, tier)
    return {
        "picks": heroes,
        "has_healer": any(h in healer_heroes for h in heroes),
        "is_degen": is_degenerate(heroes),
        "counter": float(counter_responsiveness(heroes, opp_heroes, stats, tier)),
        "synergy": float(synergy_exploitation(heroes, stats, tier)),
        "resilience_avg": float(res["avg_resilience"]),
        "resilience_early": float(res["early_pick_resilience"]),
        "resilience_late": float(res["late_pick_resilience"]),
        "counter_avg": float(ctr["avg_counter"]),
        "counter_late": float(ctr["late_counter"]),
        "team_synergy": float(syn["team_synergy"]),
        "incremental_synergy": float(syn["avg_incremental_synergy"]),
    }


def run_pair(a_name, b_name, n_drafts, device, stats, group_indices,
             gd_models, evaluators, mcts_run):
    act_a = build_actor(a_name, device, stats, group_indices, gd_models, mcts_run)
    act_b = build_actor(b_name, device, stats, group_indices, gd_models, mcts_run)

    pair_idx = STRATEGIES.index(a_name) * len(STRATEGIES) + STRATEGIES.index(b_name)
    random.seed(1000 + pair_idx)
    torch.manual_seed(1000 + pair_idx)

    records = []
    for i in range(n_drafts):
        game_map = MAPS[i % len(MAPS)]
        tier = SKILL_TIERS[(i // len(MAPS)) % len(SKILL_TIERS)]
        state = DraftState(game_map, tier, our_team=0)
        picks0, picks1 = [], []
        while not state.is_terminal():
            step_team, step_type = DRAFT_ORDER[state.step]
            step_num = state.step
            # greedy/CQL factories read state.our_team for perspective flips
            state.our_team = step_team
            actor = act_a if step_team == 0 else act_b
            hero_idx = actor(state, step_team, step_type, game_map, tier)
            if step_type == "pick":
                (picks0 if step_team == 0 else picks1).append(
                    (HEROES[hero_idx], step_num))
            state.apply_action(hero_idx, step_team, step_type)

        t0h = [HEROES[j] for j in range(NUM_HEROES) if state.team0_picks[j] > 0]
        t1h = [HEROES[j] for j in range(NUM_HEROES) if state.team1_picks[j] > 0]
        records.append({
            "draft": i, "game_map": game_map, "tier": tier,
            "team0": side_metrics(picks0, picks1, stats, tier),
            "team1": side_metrics(picks1, picks0, stats, tier),
            "wp": score_terminal(t0h, t1h, game_map, tier, evaluators,
                                 stats, device),
            "bans": [HEROES[j] for j in range(NUM_HEROES) if state.bans[j] > 0],
        })
        if (i + 1) % 25 == 0:
            print(f"  [{a_name} vs {b_name}] {i+1}/{n_drafts}", flush=True)

    return {"team0_strategy": a_name, "team1_strategy": b_name,
            "n_drafts": n_drafts, "records": records}


def task_pair(args):
    a_name, b_name = args.pair.split("__")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = common.stats_cache()
    group_indices = compute_group_indices()
    gd_models = []
    from train_generic_draft import GenericDraftModel
    for i in range(5):
        gd = GenericDraftModel()
        gd.load_state_dict(torch.load(mpath(f"generic_draft_{i}.pt"),
                                      weights_only=True, map_location="cpu"))
        gd.eval()
        gd_models.append(gd)
    evaluators = load_wp_evaluators(device)
    out = run_pair(a_name, b_name, args.drafts, device, stats, group_indices,
                   gd_models, evaluators, args.mcts_run)
    path = rpath("roundrobin", f"{args.pair}.json")
    with open(path, "w") as f:
        json.dump(out, f, default=str)
    print(f"Saved {path}")


def task_aggregate():
    d = os.path.join(RESULTS_DIR, "roundrobin")
    pairs = {}
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json"):
                with open(os.path.join(d, fn)) as f:
                    pairs[fn[:-5]] = json.load(f)
    # Summary: per ordered pair, mean WP-by-evaluator for team0 (=first pick)
    # and composition/interaction means per side. Adjudication metric TBD.
    summary = {}
    for key, p in pairs.items():
        recs = p["records"]
        n = len(recs)
        s = {"n": n}
        for ev in WP_EVALUATORS:
            s[f"team0_wp_{ev}_sym"] = float(np.mean(
                [r["wp"][ev]["sym"] for r in recs]))
        for side in ("team0", "team1"):
            s[f"{side}_healer"] = sum(r[side]["has_healer"] for r in recs) / n * 100
            s[f"{side}_degen"] = sum(r[side]["is_degen"] for r in recs) / n * 100
            s[f"{side}_counter"] = float(np.mean([r[side]["counter"] for r in recs]))
            s[f"{side}_synergy"] = float(np.mean([r[side]["synergy"] for r in recs]))
            s[f"{side}_resil_late"] = float(np.mean(
                [r[side]["resilience_late"] for r in recs]))
        summary[key] = s
    out = {"strategies": STRATEGIES,
           "adjudication_metric": "TBD — raw per-draft records retained in "
                                  "results/roundrobin/*.json",
           "pair_summaries": summary}
    path = rpath("roundrobin_summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {path} ({len(summary)} pairs aggregated)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--drafts", type=int, default=200)
    parser.add_argument("--pair", default=None,
                        help="run a single ordered pair: A__B")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--mcts-run", default=None,
                        help="historical mcts_runs/<name> for the MCTS strategy")
    args = parser.parse_args()

    if args.aggregate_only:
        task_aggregate()
        return
    if args.pair:
        if args.dry_run:
            print(f"dry run: pair {args.pair}, {args.drafts} drafts")
            return
        task_pair(args)
        return

    # orchestrate: one pool job per ordered pair
    me = os.path.abspath(__file__)
    jobs = []
    greedy = {"enriched", "enriched_aug", "gourdeau"}
    for a, b in itertools.permutations(STRATEGIES, 2):
        weight = "heavy" if (a in greedy or b in greedy) else "light"
        extra = ["--mcts-run", args.mcts_run] if args.mcts_run else []
        jobs.append(Job(f"p3b_{a}__{b}",
                        [me, "--pair", f"{a}__{b}", "--drafts", str(args.drafts)]
                        + extra,
                        [rpath("roundrobin", f"{a}__{b}.json")], weight=weight))
    print(f"Phase 3b: {len(jobs)} ordered pairs x {args.drafts} drafts")
    failed = common.run_pool(jobs, dry_run=args.dry_run, force=args.force,
                             only=args.only)
    if args.dry_run:
        return
    task_aggregate()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
