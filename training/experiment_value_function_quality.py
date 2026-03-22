"""
Experiment: WP Model Quality as MCTS Value Function.

Demonstrates that enriched WP models produce better draft policies than naive ones,
and the improvement comes from compositional reasoning, not aggregate accuracy.

See EXPERIMENT.md for full design.

Usage:
    set -a && source .env && set +a
    python3 -u training/experiment_value_function_quality.py --drafts 500
    python3 -u training/experiment_value_function_quality.py --drafts 50   # quick test
    python3 -u training/experiment_value_function_quality.py --drafts 500 --include-mcts
"""
import os
import sys
import json
import random
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS,
    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
    load_replay_data, split_data,
    HERO_ROLE_FINE, FINE_ROLE_NAMES,
)
from sweep_enriched_wp import (
    StatsCache, WinProbEnrichedModel, FEATURE_GROUPS, FEATURE_GROUP_DIMS,
    compute_group_indices, precompute_all_features, extract_features,
)
from train_draft_policy import (
    AlphaZeroDraftNet, DraftState, DRAFT_ORDER, STATE_DIM,
)
from train_generic_draft import GenericDraftModel
from test_wp_sanity import TESTS, run_tests

# ── Model configs ──

MODEL_CONFIGS = {
    "naive": {
        "groups": [],
        "description": "Base features only (multi-hot hero IDs + map + tier)",
    },
    "herostrength": {
        "groups": ["hero_wr", "team_avg_wr"],
        "description": "Per-hero stats, no team interactions",
    },
    "enriched": {
        "groups": ["role_counts", "team_avg_wr", "map_delta", "pairwise_counters",
                   "pairwise_synergies", "counter_detail", "meta_strength", "draft_diversity"],
        "description": "Full enriched: roles, pairwise, meta, diversity",
    },
}

TRAIN_HP = {
    "hidden_dims": [256, 128],
    "dropout": 0.3,
    "lr": 5e-4,
    "weight_decay": 5e-3,
    "patience": 25,
    "max_epochs": 200,
}


# ── WP Model Training ──

def train_wp_model(name, groups, train_base, train_enriched, train_labels,
                   test_base, test_enriched, test_labels, group_indices, device):
    """Train a WP model variant. Returns (best_acc, best_loss, model_path)."""
    cols = []
    for g in groups:
        s, e = group_indices[g]
        cols.extend(range(s, e))
    total_dim = 197 + len(cols)

    if cols:
        trX = torch.cat([train_base, train_enriched[:, cols]], dim=1).to(device)
        teX = torch.cat([test_base, test_enriched[:, cols]], dim=1).to(device)
    else:
        trX = train_base.to(device)
        teX = test_base.to(device)
    trY = train_labels.to(device)
    teY = test_labels.to(device)

    best_across_seeds = {"acc": 0, "loss": float("inf"), "seed": 0}
    all_accs = []

    for seed in [42, 123, 777]:
        torch.manual_seed(seed)
        model = WinProbEnrichedModel(total_dim, TRAIN_HP["hidden_dims"],
                                     dropout=TRAIN_HP["dropout"]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=TRAIN_HP["lr"],
                                weight_decay=TRAIN_HP["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
        crit = nn.BCELoss()

        best_loss = float("inf")
        best_acc = 0.0
        patience = 0
        save_path = os.path.join(os.path.dirname(__file__), f"wp_experiment_{name}.pt")

        for epoch in range(TRAIN_HP["max_epochs"]):
            model.train()
            perm = torch.randperm(len(trX), device=device)
            for i in range(0, len(trX), 1024):
                idx = perm[i:i + 1024]
                pred = model(trX[idx])
                loss = crit(pred, trY[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                te_pred = model(teX)
                te_loss = crit(te_pred, teY).item()
                te_acc = ((te_pred > 0.5).float() == teY).float().mean().item() * 100

            if te_loss < best_loss:
                best_loss = te_loss
                best_acc = te_acc
                patience = 0
                if best_acc > best_across_seeds["acc"]:
                    best_across_seeds = {"acc": best_acc, "loss": best_loss, "seed": seed}
                    torch.save(model.state_dict(), save_path)
            else:
                patience += 1
                if patience >= TRAIN_HP["patience"]:
                    break

        all_accs.append(best_acc)

    avg_acc = np.mean(all_accs)
    print(f"  {name}: {avg_acc:.2f}% avg (seeds: {[f'{a:.2f}' for a in all_accs]}), "
          f"best={best_across_seeds['acc']:.2f}% (seed {best_across_seeds['seed']})")

    return {
        "name": name,
        "groups": groups,
        "avg_acc": avg_acc,
        "best_acc": best_across_seeds["acc"],
        "best_loss": best_across_seeds["loss"],
        "all_accs": all_accs,
        "input_dim": total_dim,
        "path": save_path,
    }


# ── Greedy Draft ──

def evaluate_terminal_with_model(wp_model, state, feature_groups, stats_cache,
                                  group_indices, device):
    """Score a terminal draft state with a specific WP model variant."""
    t0_heroes = [HEROES[i] for i in range(NUM_HEROES) if state.team0_picks[i] > 0]
    t1_heroes = [HEROES[i] for i in range(NUM_HEROES) if state.team1_picks[i] > 0]
    d = {"team0_heroes": t0_heroes, "team1_heroes": t1_heroes,
         "game_map": state.game_map, "skill_tier": state.skill_tier, "winner": 0}
    all_mask = [True] * len(FEATURE_GROUPS)
    base, enriched = extract_features(d, stats_cache, all_mask)
    cols = []
    for g in feature_groups:
        s, e = group_indices[g]
        cols.extend(range(s, e))
    if cols:
        x = np.concatenate([base, enriched[cols]])
    else:
        x = base
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        wp = wp_model(x_t).item()
    return wp if state.our_team == 0 else 1.0 - wp


def greedy_pick_with_model(state, wp_model, wp_groups, gd_models, stats_cache,
                           group_indices, device, config_idx, step_num, n_rollouts=1):
    """Greedy pick using batched GD rollouts.

    Instead of evaluating each hero candidate sequentially, we batch ALL candidates
    through each rollout step simultaneously. For ~84 valid heroes and ~10 remaining
    steps, this turns 840 sequential Python ops into 10 batched GPU forward passes.
    """
    valid = state.valid_mask_np()
    current_team, action_type = DRAFT_ORDER[state.step]
    valid_heroes = [i for i in range(NUM_HEROES) if valid[i] > 0]
    B = len(valid_heroes)
    if B == 0:
        return 0, 0.0, []

    gd_model = gd_models[0]  # use first GD model for batched inference
    gd_device = next(gd_model.parameters()).device

    # Build base state as numpy (289 dims for GD — without our_team)
    base_np = state.to_numpy()[:-1]  # strip our_team to get 289 dims

    # Create B parallel states, each with one candidate hero applied
    # State layout: team0_picks(90) + team1_picks(90) + bans(90) + map(14) + tier(3) + step(1) + type(1)
    states = np.tile(base_np, (B, 1)).astype(np.float32)  # (B, 289)
    taken_sets = []  # track taken heroes per candidate

    for bi, hero_idx in enumerate(valid_heroes):
        # Apply this candidate hero to the state
        if action_type == 'ban':
            states[bi, 2 * NUM_HEROES + hero_idx] = 1.0  # bans slot
        elif current_team == 0:
            states[bi, hero_idx] = 1.0  # team0_picks
        else:
            states[bi, NUM_HEROES + hero_idx] = 1.0  # team1_picks
        # Update step/type for next step
        next_step = state.step + 1
        if next_step < 16:
            states[bi, -2] = next_step / 15.0
            states[bi, -1] = 0.0 if DRAFT_ORDER[next_step][1] == 'ban' else 1.0
        # Track taken heroes
        taken = set(state.taken)
        taken.add(hero_idx)
        taken_sets.append(taken)

    # Now roll out remaining draft steps with batched GD
    current_step = state.step + 1
    states_t = torch.from_numpy(states).to(gd_device)

    with torch.no_grad():
        while current_step < 16:
            # Build masks: 1 where hero is NOT taken
            masks = torch.ones(B, NUM_HEROES, device=gd_device)
            for bi in range(B):
                for idx in taken_sets[bi]:
                    masks[bi, idx] = 0.0

            # Batched GD forward pass
            logits = gd_model(states_t, masks)
            probs = F.softmax(logits / 1.0, dim=1)
            actions = torch.multinomial(probs, 1).squeeze(1)  # (B,)

            # Apply actions to all states
            team_cs, action_type_cs = DRAFT_ORDER[current_step]
            actions_np = actions.cpu().numpy()
            for bi in range(B):
                a = int(actions_np[bi])
                taken_sets[bi].add(a)
                if action_type_cs == 'ban':
                    states_t[bi, 2 * NUM_HEROES + a] = 1.0
                elif team_cs == 0:
                    states_t[bi, a] = 1.0
                else:
                    states_t[bi, NUM_HEROES + a] = 1.0

            # Update step/type for next
            current_step += 1
            if current_step < 16:
                next_type = 0.0 if DRAFT_ORDER[current_step][1] == 'ban' else 1.0
                states_t[:, -2] = current_step / 15.0
                states_t[:, -1] = next_type

    # Evaluate all terminal states with the WP model
    # Need to build enriched features for each terminal state
    scores = []
    for bi, hero_idx in enumerate(valid_heroes):
        # Reconstruct hero names from terminal state
        t0_heroes = [HEROES[i] for i in range(NUM_HEROES) if states_t[bi, i].item() > 0.5]
        t1_heroes = [HEROES[i] for i in range(NUM_HEROES) if states_t[bi, NUM_HEROES + i].item() > 0.5]
        d = {"team0_heroes": t0_heroes, "team1_heroes": t1_heroes,
             "game_map": state.game_map, "skill_tier": state.skill_tier,
             "winner": 0, "avg_mmr": None}
        all_mask = [True] * len(FEATURE_GROUPS)
        base, enriched = extract_features(d, stats_cache, all_mask)
        cols = []
        for g in wp_groups:
            s, e = group_indices[g]
            cols.extend(range(s, e))
        if cols:
            x = np.concatenate([base, enriched[cols]])
        else:
            x = base
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            wp = wp_model(x_t).item()
        # Flip for team perspective
        if state.our_team == 1:
            wp = 1.0 - wp
        scores.append((hero_idx, wp))

    scores.sort(key=lambda x: -x[1])
    chosen = scores[0]
    top3 = [{"hero": HEROES[idx], "score": sc} for idx, sc in scores[:3]]

    return chosen[0], chosen[1], top3


def analyze_composition(state, our_team):
    """Extract composition metrics from a terminal draft state."""
    picks = state.team0_picks if our_team == 0 else state.team1_picks
    heroes = [HEROES[i] for i in range(NUM_HEROES) if picks[i] > 0]
    role_counts = {}
    for h in heroes:
        r = HERO_ROLE_FINE.get(h, "unknown")
        role_counts[r] = role_counts.get(r, 0) + 1
    return {
        "heroes": heroes,
        "roles": dict(role_counts),
        "has_healer": "healer" in role_counts,
        "has_tank": "tank" in role_counts,
        "has_frontline": "tank" in role_counts or "bruiser" in role_counts,
        "has_ranged_damage": "ranged_aa" in role_counts or "ranged_mage" in role_counts,
        "num_distinct_roles": len(role_counts),
        "is_absurd": (
            "healer" not in role_counts
            or ("tank" not in role_counts and "bruiser" not in role_counts)
            or any(v >= 3 for v in role_counts.values())
            or ("ranged_aa" not in role_counts and "ranged_mage" not in role_counts)
        ),
    }


def run_single_draft(config_idx, game_map, skill_tier, our_team,
                     model_name, wp_model, wp_groups,
                     all_wp_models, all_wp_groups,
                     gd_models, stats_cache, group_indices, device):
    """Run one greedy draft and return the full record."""
    state = DraftState(game_map, skill_tier, our_team=our_team)
    gd_temp = 1.0
    steps = []

    while not state.is_terminal():
        team, action_type = DRAFT_ORDER[state.step]

        if team == our_team:
            # Our turn: greedy pick
            hero_idx, score, top3 = greedy_pick_with_model(
                state, wp_model, wp_groups, gd_models, stats_cache,
                group_indices, device, config_idx, state.step,
            )
            # Record current context
            t0h = [HEROES[i] for i in range(NUM_HEROES) if state.team0_picks[i] > 0]
            t1h = [HEROES[i] for i in range(NUM_HEROES) if state.team1_picks[i] > 0]
            steps.append({
                "step": state.step,
                "action_type": action_type,
                "chosen_hero": HEROES[hero_idx],
                "chosen_score": score,
                "top3": top3,
                "current_our_picks": t0h if our_team == 0 else t1h,
                "current_opp_picks": t1h if our_team == 0 else t0h,
            })
            state.apply_action(hero_idx, team, action_type)
        else:
            # Opponent: GD sample (seeded for reproducibility)
            seed = config_idx * 10000 + state.step * 100 + 99
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            cpu = torch.device("cpu")
            gd = random.choice(gd_models)
            x = state.to_tensor_gd(cpu)
            mask = state.valid_mask(cpu)
            with torch.no_grad():
                logits = gd(x, mask)
                probs = F.softmax(logits / gd_temp, dim=1)
                action = torch.multinomial(probs, 1).item()
            state.apply_action(action, team, action_type)

    # Cross-evaluate terminal state with all three WP models
    cross_eval = {}
    for eval_name, (eval_model, eval_groups) in zip(
        all_wp_models.keys(),
        zip(all_wp_models.values(), all_wp_groups.values())
    ):
        cross_eval[f"wp_by_{eval_name}"] = evaluate_terminal_with_model(
            eval_model, state, eval_groups, stats_cache, group_indices, device
        )

    # Composition analysis
    comp = analyze_composition(state, our_team)

    # Opponent picks
    opp_picks_vec = state.team1_picks if our_team == 0 else state.team0_picks
    opp_heroes = [HEROES[i] for i in range(NUM_HEROES) if opp_picks_vec[i] > 0]

    # Our bans and opp bans
    our_bans = [s["chosen_hero"] for s in steps if s["action_type"] == "ban"]
    # Note: opponent bans are implicit in the GD model's choices

    return {
        "wp_model": model_name,
        "game_map": game_map,
        "skill_tier": skill_tier,
        "our_team": our_team,
        "our_picks": comp["heroes"],
        "opp_picks": opp_heroes,
        "our_bans": our_bans,
        "steps": steps,
        **cross_eval,
        **{f"comp_{k}": v for k, v in comp.items() if k != "heroes"},
    }


# ── GPU Worker ──

def gpu_worker(gpu_id, config_batch, wp_sds, wp_groups_map, gd_sds, stats_data,
               group_indices, result_queue):
    """Worker: runs assigned draft configs for all three strategies on one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")

    # Reconstruct stats cache
    stats_cache = StatsCache.__new__(StatsCache)
    stats_cache.hero_wr = stats_data["hero_wr"]
    stats_cache.hero_map_wr = stats_data["hero_map_wr"]
    stats_cache.pairwise = stats_data["pairwise"]
    stats_cache.hero_meta = stats_data["hero_meta"]

    # Reconstruct GD models
    gd_models = []
    for sd in gd_sds:
        gd = GenericDraftModel()
        gd.load_state_dict(sd)
        gd.eval()
        gd_models.append(gd)

    # Reconstruct all three WP models
    all_wp_models = {}
    all_wp_groups = {}
    for name, sd in wp_sds.items():
        groups = wp_groups_map[name]
        cols = []
        for g in groups:
            s, e = group_indices[g]
            cols.extend(range(s, e))
        dim = 197 + len(cols)
        model = WinProbEnrichedModel(dim, TRAIN_HP["hidden_dims"],
                                     dropout=TRAIN_HP["dropout"]).to(device)
        model.load_state_dict(sd)
        model.eval()
        all_wp_models[name] = model
        all_wp_groups[name] = groups

    total = len(config_batch)
    for ci, (config_idx, game_map, skill_tier, our_team) in enumerate(config_batch):
        # Run all three strategies on this config
        for model_name in ["naive", "herostrength", "enriched"]:
            record = run_single_draft(
                config_idx, game_map, skill_tier, our_team,
                model_name, all_wp_models[model_name], all_wp_groups[model_name],
                all_wp_models, all_wp_groups,
                gd_models, stats_cache, group_indices, device,
            )
            result_queue.put(record)

        if (ci + 1) % 10 == 0:
            print(f"  [GPU {gpu_id}] {ci+1}/{total} configs done")


# ── Analysis ──

def print_analysis(records, wp_model_info, sanity_results):
    """Print all analysis tables."""
    out_dir = os.path.join(os.path.dirname(__file__), "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 70)
    p("3a. WP MODEL ACCURACY")
    p("=" * 70)
    p(f"{'Model':<20} {'Avg Acc':>8} {'Best Acc':>9} {'Input Dim':>10} {'Groups'}")
    p("-" * 70)
    for name in ["naive", "herostrength", "enriched"]:
        info = wp_model_info[name]
        g_str = ", ".join(info["groups"][:4]) + ("..." if len(info["groups"]) > 4 else "") if info["groups"] else "(base only)"
        p(f"{name:<20} {info['avg_acc']:>7.2f}% {info['best_acc']:>8.2f}% {info['input_dim']:>10} {g_str}")
    p()

    # 3b. Cross-evaluation
    p("=" * 70)
    p("3b. CROSS-EVALUATION MATRIX")
    p("=" * 70)
    drafters = ["naive", "herostrength", "enriched"]
    by_drafter = {d: [r for r in records if r["wp_model"] == d] for d in drafters}

    p(f"{'Drafter':<20} {'by Naive':>10} {'by HeroStr':>10} {'by Enriched':>10}")
    p("-" * 50)
    for drafter in drafters:
        recs = by_drafter[drafter]
        n = recs[0]["wp_by_naive"] if recs else 0
        h = recs[0]["wp_by_herostrength"] if recs else 0
        e = recs[0]["wp_by_enriched"] if recs else 0
        avg_n = np.mean([r["wp_by_naive"] for r in recs])
        avg_h = np.mean([r["wp_by_herostrength"] for r in recs])
        avg_e = np.mean([r["wp_by_enriched"] for r in recs])
        p(f"{drafter:<20} {avg_n:>9.4f} {avg_h:>10.4f} {avg_e:>10.4f}")
    p()

    # 3c. Composition quality
    p("=" * 70)
    p("3c. COMPOSITION QUALITY")
    p("=" * 70)
    p(f"{'Drafter':<20} {'Healer%':>8} {'Tank%':>6} {'Front%':>7} {'Ranged%':>8} {'Roles':>6} {'Absurd%':>8}")
    p("-" * 60)
    for drafter in drafters:
        recs = by_drafter[drafter]
        n = len(recs)
        healer = sum(1 for r in recs if r["comp_has_healer"]) / n * 100
        tank = sum(1 for r in recs if r["comp_has_tank"]) / n * 100
        front = sum(1 for r in recs if r["comp_has_frontline"]) / n * 100
        ranged = sum(1 for r in recs if r["comp_has_ranged_damage"]) / n * 100
        roles = np.mean([r["comp_num_distinct_roles"] for r in recs])
        absurd = sum(1 for r in recs if r["comp_is_absurd"]) / n * 100
        p(f"{drafter:<20} {healer:>7.1f}% {tank:>5.1f}% {front:>6.1f}% {ranged:>7.1f}% {roles:>5.1f} {absurd:>7.1f}%")
    p()

    # 3d. Pick divergence by phase
    p("=" * 70)
    p("3d. PICK DIVERGENCE BY DRAFT PHASE")
    p("=" * 70)
    ban_steps = {0, 1, 2, 3, 9, 10}
    early_pick_steps = {4, 5, 6, 7, 8}
    late_pick_steps = {11, 12, 13, 14, 15}

    # Group records by config_idx (same map/tier/side)
    by_config = {}
    for r in records:
        key = (r["game_map"], r["skill_tier"], r["our_team"])
        by_config.setdefault(key, {})[r["wp_model"]] = r

    phases = [("Bans", ban_steps), ("Early picks", early_pick_steps), ("Late picks", late_pick_steps)]
    p(f"{'Phase':<15} {'Naive vs Enriched':>20} {'HeroStr vs Enriched':>22}")
    p("-" * 57)
    for phase_name, phase_steps in phases:
        ne_agree = 0
        he_agree = 0
        total = 0
        for key, drafter_recs in by_config.items():
            if "naive" not in drafter_recs or "enriched" not in drafter_recs:
                continue
            n_steps = {s["step"]: s["chosen_hero"] for s in drafter_recs["naive"]["steps"]}
            e_steps = {s["step"]: s["chosen_hero"] for s in drafter_recs["enriched"]["steps"]}
            h_steps = {s["step"]: s["chosen_hero"] for s in drafter_recs.get("herostrength", {}).get("steps", [])}
            for step in phase_steps:
                if step in n_steps and step in e_steps:
                    total += 1
                    if n_steps[step] == e_steps[step]:
                        ne_agree += 1
                    if step in h_steps and h_steps[step] == e_steps[step]:
                        he_agree += 1
        if total > 0:
            p(f"{phase_name:<15} {ne_agree/total:>19.1%} {he_agree/total:>21.1%}")
    p()

    # 3e. Disagreement deep-dive
    p("=" * 70)
    p("3e. DISAGREEMENT DEEP-DIVE")
    p("=" * 70)
    naive_recs = by_drafter["naive"]
    naive_recs_sorted = sorted(naive_recs,
                                key=lambda r: r["wp_by_naive"] - r["wp_by_enriched"],
                                reverse=True)
    top50 = naive_recs_sorted[:50]

    reasons = {"no_healer": 0, "no_frontline": 0, "role_stacking": 0,
               "high_wr_bad_comp": 0}
    for r in top50:
        roles = r["comp_roles"]
        if not r["comp_has_healer"]:
            reasons["no_healer"] += 1
        if not r["comp_has_frontline"]:
            reasons["no_frontline"] += 1
        if any(v >= 3 for v in roles.values()):
            reasons["role_stacking"] += 1
        if r["comp_is_absurd"]:
            reasons["high_wr_bad_comp"] += 1

    p("Top 50 disagreements (naive scores high, enriched scores low):")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        p(f"  {reason}: {count}/50")
    p()

    # Print 10 examples
    p("Example drafts:")
    for i, r in enumerate(top50[:10]):
        gap = r["wp_by_naive"] - r["wp_by_enriched"]
        p(f"  #{i+1}: {r['game_map']}, {r['skill_tier']} tier, team {r['our_team']}")
        p(f"    Our team: {', '.join(r['our_picks'])}")
        p(f"    Opponent: {', '.join(r['opp_picks'])}")
        p(f"    WP naive={r['wp_by_naive']:.3f} enriched={r['wp_by_enriched']:.3f} (gap={gap:+.3f})")
        issues = []
        if not r["comp_has_healer"]:
            issues.append("no healer")
        if not r["comp_has_frontline"]:
            issues.append("no frontline")
        if any(v >= 3 for v in r["comp_roles"].values()):
            stacked = [k for k, v in r["comp_roles"].items() if v >= 3]
            issues.append(f"role stacking ({', '.join(stacked)})")
        p(f"    Issues: {', '.join(issues) if issues else 'none identified'}")
        # Show a late-pick decision
        late_steps = [s for s in r["steps"] if s["action_type"] == "pick"]
        if late_steps:
            last = late_steps[-1]
            p(f"    Last pick: chose {last['chosen_hero']} ({last['chosen_score']:.3f})")
            alts = [f"{t['hero']} ({t['score']:.3f})" for t in last["top3"][1:3]]
            p(f"      Over: {', '.join(alts)}")
        p()

    # 3f. Sanity tests
    p("=" * 70)
    p("3f. SANITY TEST RESULTS")
    p("=" * 70)
    p(f"{'Model':<20} {'Absurd':>7} {'Trap':>5} {'Normal':>7} {'Symmetry':>9} {'Total':>6}")
    p("-" * 55)
    for name, sr in sanity_results.items():
        p(f"{name:<20} {sr['absurd']}/8 {sr['trap']}/5 {sr['normal']}/5 {sr['symmetry']}/3 {sr['total']}/21")
    p()

    # 3h. Per-map analysis
    p("=" * 70)
    p("3h. PER-MAP COMPOSITION ANALYSIS")
    p("=" * 70)
    two_lane = {"Battlefield of Eternity", "Braxis Holdout", "Hanamura Temple"}
    for map_group, map_names in [("Two-lane maps", two_lane), ("Three-lane maps", set(MAPS) - two_lane)]:
        p(f"\n{map_group}:")
        for drafter in drafters:
            recs = [r for r in by_drafter[drafter] if r["game_map"] in map_names]
            if not recs:
                continue
            n = len(recs)
            healer = sum(1 for r in recs if r["comp_has_healer"]) / n * 100
            absurd = sum(1 for r in recs if r["comp_is_absurd"]) / n * 100
            p(f"  {drafter:<18} healer={healer:.0f}% absurd={absurd:.0f}% (n={n})")
    p()

    # Save
    with open(os.path.join(out_dir, "analysis_summary.txt"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_dir, "draft_records.json"), "w") as f:
        json.dump(records, f, indent=2)
    with open(os.path.join(out_dir, "wp_model_comparison.json"), "w") as f:
        json.dump(wp_model_info, f, indent=2)
    with open(os.path.join(out_dir, "sanity_test_results.json"), "w") as f:
        json.dump(sanity_results, f, indent=2)
    with open(os.path.join(out_dir, "disagreement_cases.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "steps"} for r in top50], f, indent=2)

    print(f"\nResults saved to {out_dir}/")


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", type=int, default=500)
    parser.add_argument("--skip-wp-training", action="store_true")
    parser.add_argument("--include-mcts", action="store_true")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU IDs")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
    else:
        gpu_ids = list(range(num_gpus)) if num_gpus > 0 else [0]
    print(f"GPUs: {gpu_ids}")

    # Load data
    print("Loading data...")
    data = load_replay_data()
    train_data, test_data = split_data(data)
    stats = StatsCache()

    print("Precomputing features...")
    train_base, train_enriched, train_labels = precompute_all_features(train_data, stats)
    test_base, test_enriched, test_labels = precompute_all_features(test_data, stats)
    group_indices = compute_group_indices()
    print(f"  Train: {train_base.shape[0]}, enriched dims: {train_enriched.shape[1]}")

    # ── Part 1: Train/load WP models ──
    print("\n" + "=" * 70)
    print("PART 1: WP MODEL TRAINING")
    print("=" * 70)

    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    wp_model_info = {}

    for name, cfg in MODEL_CONFIGS.items():
        path = os.path.join(os.path.dirname(__file__), f"wp_experiment_{name}.pt")
        if args.skip_wp_training and os.path.exists(path):
            # Load existing — compute dims to verify
            cols = []
            for g in cfg["groups"]:
                s, e = group_indices[g]
                cols.extend(range(s, e))
            dim = 197 + len(cols)
            print(f"  {name}: loading from {path} (dim={dim})")
            wp_model_info[name] = {
                "name": name, "groups": cfg["groups"],
                "avg_acc": 0, "best_acc": 0, "best_loss": 0,
                "all_accs": [], "input_dim": dim, "path": path,
            }
        else:
            info = train_wp_model(
                name, cfg["groups"],
                train_base, train_enriched, train_labels,
                test_base, test_enriched, test_labels,
                group_indices, device,
            )
            wp_model_info[name] = info

    # ── Sanity tests ──
    print("\nRunning sanity tests...")
    sanity_results = {}
    for name, cfg in MODEL_CONFIGS.items():
        groups = cfg["groups"]
        cols = []
        for g in groups:
            s, e = group_indices[g]
            cols.extend(range(s, e))
        dim = 197 + len(cols)
        model = WinProbEnrichedModel(dim, TRAIN_HP["hidden_dims"],
                                     dropout=TRAIN_HP["dropout"])
        model.load_state_dict(torch.load(wp_model_info[name]["path"],
                                         weights_only=True, map_location="cpu"))
        model.eval()
        all_mask = [True] * len(FEATURE_GROUPS)

        def make_eval_fn(m, c):
            def eval_fn(t0h, t1h, game_map="Cursed Hollow", tier="mid"):
                d = {"team0_heroes": t0h, "team1_heroes": t1h,
                     "game_map": game_map, "skill_tier": tier, "winner": 0}
                base, enriched = extract_features(d, stats, all_mask)
                enriched_sel = enriched[c] if c else np.array([], dtype=np.float32)
                x = np.concatenate([base, enriched_sel]) if len(enriched_sel) > 0 else base
                with torch.no_grad():
                    return m(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()
            return eval_fn

        eval_fn = make_eval_fn(model, cols)
        passed, total, results_list = run_tests(eval_fn, verbose=False)

        # Count by category
        cats = {"absurd": 0, "trap": 0, "normal": 0, "symmetry": 0}
        cat_totals = {"absurd": 0, "trap": 0, "normal": 0, "symmetry": 0}
        for t, p in zip(TESTS, results_list):
            cat = t.get("category", "")
            if cat in cats:
                cat_totals[cat] += 1
                if p:
                    cats[cat] += 1

        sanity_results[name] = {
            "total": f"{passed}/{total}",
            "absurd": f"{cats['absurd']}",
            "trap": f"{cats['trap']}",
            "normal": f"{cats['normal']}",
            "symmetry": f"{cats['symmetry']}",
        }
        print(f"  {name}: {passed}/{total}")

    # ── Part 2: Greedy benchmark ──
    print("\n" + "=" * 70)
    print("PART 2: GREEDY DRAFT BENCHMARK")
    print("=" * 70)

    random.seed(42)
    configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
               for i in range(args.drafts)]

    # Prepare model state dicts and stats for workers
    wp_sds = {}
    wp_groups_map = {}
    for name, cfg in MODEL_CONFIGS.items():
        groups = cfg["groups"]
        cols = []
        for g in groups:
            s, e = group_indices[g]
            cols.extend(range(s, e))
        dim = 197 + len(cols)
        model = WinProbEnrichedModel(dim, TRAIN_HP["hidden_dims"],
                                     dropout=TRAIN_HP["dropout"])
        model.load_state_dict(torch.load(wp_model_info[name]["path"],
                                         weights_only=True, map_location="cpu"))
        wp_sds[name] = model.state_dict()
        wp_groups_map[name] = groups

    from train_draft_policy import load_pretrained_models
    _, gd_models_loaded = load_pretrained_models(torch.device("cpu"))
    gd_sds = [gd.state_dict() for gd in gd_models_loaded]

    stats_data = {
        "hero_wr": stats.hero_wr,
        "hero_map_wr": stats.hero_map_wr,
        "pairwise": stats.pairwise,
        "hero_meta": stats.hero_meta,
    }

    # Distribute configs across GPUs
    mp.set_start_method("spawn", force=True)
    gpu_batches = [[] for _ in gpu_ids]
    for i, cfg in enumerate(configs):
        gpu_batches[i % len(gpu_ids)].append(cfg)

    print(f"Running {args.drafts} drafts × 3 strategies = {args.drafts * 3} greedy drafts")
    for i, gid in enumerate(gpu_ids):
        print(f"  GPU {gid}: {len(gpu_batches[i])} configs")

    t_start = time.time()
    result_queue = mp.Queue()
    processes = []
    for i, gid in enumerate(gpu_ids):
        if gpu_batches[i]:
            p = mp.Process(target=gpu_worker, args=(
                gid, gpu_batches[i], wp_sds, wp_groups_map, gd_sds,
                stats_data, group_indices, result_queue,
            ))
            p.start()
            processes.append(p)

    # Drain queue WHILE waiting for processes (avoids deadlock when queue is full)
    records = []
    while any(p.is_alive() for p in processes) or not result_queue.empty():
        try:
            record = result_queue.get(timeout=1)
            records.append(record)
        except Exception:
            pass

    for p in processes:
        p.join(timeout=10)

    elapsed = time.time() - t_start
    print(f"\nBenchmark complete: {len(records)} drafts in {elapsed/60:.1f} min")

    # ── Part 3: Analysis ──
    print("\n" + "=" * 70)
    print("PART 3: ANALYSIS")
    print("=" * 70)
    print_analysis(records, wp_model_info, sanity_results)


if __name__ == "__main__":
    main()
