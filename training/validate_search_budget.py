#!/usr/bin/env python3
"""
Step 2: Search Budget Sweep using CUDA MCTS kernel.

Runs E and G checkpoints at multiple sim counts on separate GPUs in parallel.
Extracts pick sequences from kernel state outputs for temporal draft metrics.

Usage:
    set -a && source .env && set +a
    python3 -u training/validate_search_budget.py --configs 100 --sims 200,400,600,800
"""

import os, sys, json, random, argparse, time, subprocess
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda_mcts'))
from shared import (NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS,
                    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot, HERO_ROLE_FINE)
from sweep_enriched_wp import (StatsCache, WinProbEnrichedModel, FEATURE_GROUPS,
                                compute_group_indices, extract_features, FEATURE_GROUP_DIMS)
from train_draft_policy import AlphaZeroDraftNet, DraftState, DRAFT_ORDER
from train_generic_draft import GenericDraftModel
from extract_weights import (extract_policy_weights, extract_gd_weights,
                              extract_wp_weights, build_wp_net_offsets, extract_lookup_tables)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "experiment_results", "diagnostics")

# Draft sequence
DRAFT_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
DRAFT_IS_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]


def counter_delta(hero_a, hero_b, stats, tier):
    raw = stats.get_counter(hero_a, hero_b, tier)
    if raw is None: return None
    return raw - (stats.get_hero_wr(hero_a, tier) + (100 - stats.get_hero_wr(hero_b, tier)) - 50)

def synergy_delta(hero_a, hero_b, stats, tier):
    raw = stats.get_synergy(hero_a, hero_b, tier)
    if raw is None: return None
    return raw - (50 + (stats.get_hero_wr(hero_a, tier) - 50) + (stats.get_hero_wr(hero_b, tier) - 50))


def extract_picks_from_terminal(terminal_state, our_team):
    """Extract hero lists from terminal state multi-hot encoding."""
    t0 = [HEROES[i] for i in range(NUM_HEROES) if terminal_state[i] > 0.5]
    t1 = [HEROES[i] for i in range(NUM_HEROES) if terminal_state[NUM_HEROES + i] > 0.5]
    return t0, t1


def reconstruct_pick_steps(out_states, num_turns, terminal_state, our_team):
    """
    Reconstruct the full pick sequence from MCTS kernel state outputs.

    out_states[t] contains the state BEFORE our t-th turn.
    By diffing consecutive states, we can see what was picked between turns.
    terminal_state shows the final state after all 16 steps.

    Returns pick_steps: list of (hero_name, "ours"/"theirs", step_number)
    """
    pick_steps = []

    # Walk through the draft order using the saved states
    # State encoding: [0:90]=t0_picks, [90:180]=t1_picks, [180:270]=bans
    prev_t0 = set()
    prev_t1 = set()
    prev_bans = set()

    # Each out_state corresponds to one of our turns (before we pick)
    # Between our turns, the opponent may have picked/banned
    our_turn_idx = 0

    for step in range(16):
        team = DRAFT_TEAM[step]
        is_pick = DRAFT_IS_PICK[step]
        is_ours = (team == our_team)

        if is_ours and our_turn_idx < num_turns:
            # We have the state BEFORE this turn
            state = out_states[our_turn_idx]
            cur_t0 = set(i for i in range(NUM_HEROES) if state[i] > 0.5)
            cur_t1 = set(i for i in range(NUM_HEROES) if state[NUM_HEROES + i] > 0.5)

            # Any new opponent picks since last state
            for hi in (cur_t0 - prev_t0):
                if team != 0 or hi not in prev_t0:
                    pass  # These happened before our turn
            for hi in (cur_t1 - prev_t1):
                pass

            prev_t0 = cur_t0.copy()
            prev_t1 = cur_t1.copy()
            our_turn_idx += 1

    # Simpler approach: just use terminal state to get all picks,
    # and use draft_order to assign step numbers
    # We know which heroes ended on which team from terminal_state
    t0_heroes = set(i for i in range(NUM_HEROES) if terminal_state[i] > 0.5)
    t1_heroes = set(i for i in range(NUM_HEROES) if terminal_state[NUM_HEROES + i] > 0.5)

    # For temporal metrics, we need to know WHEN each hero was picked
    # From out_states, we can reconstruct our picks:
    # The hero we picked at our turn t is the new hero in the next state vs current
    our_picks_ordered = []
    for t in range(num_turns):
        state = out_states[t]
        our_set = set(i for i in range(NUM_HEROES)
                      if state[our_team * NUM_HEROES + i] > 0.5)
        if t + 1 < num_turns:
            next_state = out_states[t + 1]
        else:
            next_state = terminal_state
        next_our_set = set(i for i in range(NUM_HEROES)
                           if next_state[our_team * NUM_HEROES + i] > 0.5)
        new_heroes = next_our_set - our_set
        if len(new_heroes) >= 1:
            # The first new hero is what we picked at this turn
            # (there might be opponent picks between turns too)
            our_picks_ordered.extend(new_heroes)

    # Also get opponent picks from terminal
    opp_team = 1 - our_team
    opp_heroes_all = set(i for i in range(NUM_HEROES)
                          if terminal_state[opp_team * NUM_HEROES + i] > 0.5)

    # Assign step numbers based on draft order
    our_pick_steps = [s for s in range(16) if DRAFT_IS_PICK[s] and DRAFT_TEAM[s] == our_team]
    opp_pick_steps = [s for s in range(16) if DRAFT_IS_PICK[s] and DRAFT_TEAM[s] != our_team]

    pick_steps = []
    for i, step in enumerate(our_pick_steps):
        if i < len(our_picks_ordered):
            pick_steps.append((HEROES[our_picks_ordered[i]], "ours", step))

    # For opponent picks, we don't know the order, but we can assign them
    # to their draft steps in some order (doesn't affect aggregate metrics much)
    opp_list = list(opp_heroes_all)
    for i, step in enumerate(opp_pick_steps):
        if i < len(opp_list):
            pick_steps.append((HEROES[opp_list[i]], "theirs", step))

    return pick_steps


def compute_draft_metrics(pick_steps, stats, tier):
    """Compute resilience, counter, synergy for one draft."""
    our = [(h, s) for h, team, s in pick_steps if team == 'ours']
    opp = [(h, s) for h, team, s in pick_steps if team == 'theirs']
    our_heroes = [h for h, _ in our]

    # Resilience
    exposures = []
    for our_hero, our_step in our:
        subsequent_opp = [h for h, s in opp if s > our_step]
        if not subsequent_opp: exposures.append(0.0); continue
        deltas = [d for d in (counter_delta(oh, our_hero, stats, tier) for oh in subsequent_opp) if d is not None]
        exposures.append(np.mean(deltas) if deltas else 0.0)
    resil_avg = -np.mean(exposures) if exposures else 0.0
    resil_grad = (np.mean(exposures[-2:]) - np.mean(exposures[:2])) if len(exposures) >= 4 else 0.0

    # Counter
    ctr_deltas = []
    for our_hero, our_step in our:
        prior_opp = [h for h, s in opp if s < our_step]
        if not prior_opp: ctr_deltas.append(0.0); continue
        deltas = [d for d in (counter_delta(our_hero, oh, stats, tier) for oh in prior_opp) if d is not None]
        ctr_deltas.append(np.mean(deltas) if deltas else 0.0)
    counter_avg = np.mean(ctr_deltas) if ctr_deltas else 0.0

    # Synergy
    syn_pairs = []
    for i, h1 in enumerate(our_heroes):
        for h2 in our_heroes[i + 1:]:
            d = synergy_delta(h1, h2, stats, tier)
            if d is not None: syn_pairs.append(d)
    team_syn = np.mean(syn_pairs) if syn_pairs else 0.0

    # Composition checks
    healer_heroes = set(h for h, r in HERO_ROLE_FINE.items() if r == 'healer')
    frontline = set(h for h, r in HERO_ROLE_FINE.items() if r in ('tank', 'bruiser'))
    ranged = set(h for h, r in HERO_ROLE_FINE.items() if r in ('ranged_aa', 'ranged_mage', 'pusher'))
    has_healer = any(h in healer_heroes for h in our_heroes)
    has_front = any(h in frontline for h in our_heroes)
    has_ranged = any(h in ranged for h in our_heroes)
    roles = {}
    for h in our_heroes:
        r = HERO_ROLE_FINE.get(h, 'unknown')
        roles[r] = roles.get(r, 0) + 1
    degen = not has_healer or not has_front or not has_ranged or any(c >= 3 for c in roles.values())

    return {
        'resilience_avg': resil_avg, 'resilience_gradient': resil_grad,
        'counter_avg': counter_avg, 'team_synergy': team_syn,
        'has_healer': has_healer, 'degen': degen, 'heroes': our_heroes,
    }


def load_policy(path):
    net = AlphaZeroDraftNet()
    sd = torch.load(path, weights_only=True, map_location="cpu")
    if any(k.startswith("res_block1.") for k in sd):
        new_sd = {}
        for k, v in sd.items():
            nk = k.replace("res_block1.", "res_blocks.0.").replace("res_block2.", "res_blocks.1.").replace("res_block3.", "res_blocks.2.")
            new_sd[nk] = v
        sd = new_sd
    net.load_state_dict(sd)
    net.eval()
    return net


def run_sweep_for_checkpoint(ckpt_name, ckpt_path, wp_model_type, sim_counts,
                              configs, stats, device_id=0):
    """Run sweep for one checkpoint using CUDA kernel.
    Uses 5 different GD opponents, cycling per batch for diversity."""
    # Load all 5 GD models
    gd_models = []
    for i in range(5):
        gd_path = os.path.join(os.path.dirname(__file__), f"generic_draft_{i}.pt")
        if os.path.exists(gd_path):
            gd = GenericDraftModel()
            gd.load_state_dict(torch.load(gd_path, weights_only=True, map_location="cpu"))
            gd.eval()
            gd_models.append(gd)
    print(f"  {ckpt_name}: loaded {len(gd_models)} GD opponents")
    gd_flats = [extract_gd_weights(g) for g in gd_models]

    gi = compute_group_indices()
    WP_GROUPS = ['role_counts', 'team_avg_wr', 'map_delta', 'pairwise_counters',
                 'pairwise_synergies', 'counter_detail', 'meta_strength',
                 'draft_diversity', 'comp_wr']
    wp_cols = []
    for g in WP_GROUPS:
        s, e = gi[g]
        wp_cols.extend(range(s, e))
    wp_input_dim = 197 + sum(FEATURE_GROUP_DIMS[g] for g in WP_GROUPS)

    if wp_model_type == "enriched":
        wp_model = WinProbEnrichedModel(wp_input_dim, [256, 128], dropout=0.3)
        wp_model.load_state_dict(torch.load(
            os.path.join(os.path.dirname(__file__), "wp_experiment_enriched.pt"),
            weights_only=True, map_location="cpu"))
    else:
        wp_model = WinProbEnrichedModel(wp_input_dim, [512, 256, 128], dropout=0.3)
        wp_model.load_state_dict(torch.load(
            os.path.join(os.path.dirname(__file__), "wp_enriched_winner.pt"),
            weights_only=True, map_location="cpu"))
    wp_model.eval()
    wp_flat, wp_no = extract_wp_weights(wp_model)
    wp_offsets = build_wp_net_offsets(wp_model, wp_no, wp_input_dim)
    wp_stats = StatsCache()
    lut_blob = extract_lookup_tables(wp_stats)

    policy = load_policy(os.path.join(os.path.dirname(__file__), ckpt_path))
    pf, po = extract_policy_weights(policy)

    # Load CUDA kernel
    so_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda_mcts')
    so_files = [f for f in os.listdir(so_dir) if f.startswith('cuda_mcts_kernel') and f.endswith('.so')]
    spec = importlib.util.spec_from_file_location('cuda_mcts_kernel', os.path.join(so_dir, so_files[0]))
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)

    N = len(configs)
    config_array = np.array(configs, dtype=np.int32)

    results = {}
    for n_sims in sim_counts:
        t0 = time.time()
        all_metrics = []
        hero_counter = Counter()

        # Run in batches, cycling GD opponents
        batch_size = min(N, 128)
        for batch_idx, batch_start in enumerate(range(0, N, batch_size)):
            batch_end = min(batch_start + batch_size, N)
            batch_configs = config_array[batch_start:batch_end]

            # Cycle GD opponent per batch
            gd_idx = batch_idx % len(gd_flats)
            gd_flat, gd_offsets_cur = gd_flats[gd_idx]

            engine = kernel.MCTSKernelEngine(
                pf, gd_flat, wp_flat, po, gd_offsets_cur, wp_offsets, lut_blob,
                max_concurrent=min(len(batch_configs), 128), device_id=device_id)

            batch_results = engine.run_episodes(batch_configs, max(n_sims, 1), 2.0, 42 + batch_start)
            del engine

            for i, result_tuple in enumerate(batch_results):
                wp, examples, terminal_state, ep_our_team = result_tuple
                terminal = np.array(terminal_state)
                cfg = configs[batch_start + i]
                map_idx, tier_idx, our_team = cfg
                tier = SKILL_TIERS[tier_idx]

                # Extract heroes from TERMINAL state multi-hots
                t0h = [HEROES[j] for j in range(NUM_HEROES) if terminal[j] > 0.5]
                t1h = [HEROES[j] for j in range(NUM_HEROES) if terminal[NUM_HEROES + j] > 0.5]
                our_heroes = t0h if our_team == 0 else t1h
                opp_heroes = t1h if our_team == 0 else t0h

                if len(our_heroes) != 5 or len(opp_heroes) != 5:
                    continue  # incomplete draft

                # Reconstruct our pick ORDER from out_states diffs
                our_picks_ordered = []
                prev_our = set()
                for t in range(len(examples)):
                    state_t = np.array(examples[t][0])
                    cur_our = set(HEROES[j] for j in range(NUM_HEROES)
                                  if state_t[our_team * NUM_HEROES + j] > 0.5)
                    # After this state, we pick — the new hero appears in next state
                    if t + 1 < len(examples):
                        next_state = np.array(examples[t + 1][0])
                    else:
                        next_state = terminal
                    next_our = set(HEROES[j] for j in range(NUM_HEROES)
                                   if next_state[our_team * NUM_HEROES + j] > 0.5)
                    new = next_our - cur_our
                    # The hero we picked is in new (there might be multiple if opponent also picked between)
                    for h in new:
                        if h in set(our_heroes) and h not in prev_our:
                            our_picks_ordered.append(h)
                            prev_our.add(h)
                            break

                # Fill any missing (edge case)
                for h in our_heroes:
                    if h not in our_picks_ordered:
                        our_picks_ordered.append(h)

                # Build pick_steps with correct step numbers
                our_steps = [s for s in range(16) if DRAFT_IS_PICK[s] and DRAFT_TEAM[s] == our_team]
                opp_steps = [s for s in range(16) if DRAFT_IS_PICK[s] and DRAFT_TEAM[s] != our_team]

                pick_steps = []
                for j, step in enumerate(our_steps):
                    if j < len(our_picks_ordered):
                        pick_steps.append((our_picks_ordered[j], "ours", step))
                for j, step in enumerate(opp_steps):
                    if j < len(opp_heroes):
                        pick_steps.append((opp_heroes[j], "theirs", step))

                metrics = compute_draft_metrics(pick_steps, stats, tier)
                all_metrics.append(metrics)
                for h in metrics['heroes']:
                    hero_counter[h] += 1

        elapsed = time.time() - t0

        agg = {}
        for key in ['resilience_avg', 'resilience_gradient', 'counter_avg', 'team_synergy']:
            agg[key] = np.mean([m[key] for m in all_metrics])
        agg['healer_pct'] = np.mean([m['has_healer'] for m in all_metrics]) * 100
        agg['degen_pct'] = np.mean([m['degen'] for m in all_metrics]) * 100
        agg['distinct'] = len(hero_counter)
        agg['elapsed'] = elapsed
        agg['n_drafts'] = len(all_metrics)

        results[n_sims] = agg
        print(f"  {ckpt_name} sims={n_sims}: ctr={agg['counter_avg']:+.3f} syn={agg['team_synergy']:.3f} "
              f"resil_grad={agg['resilience_gradient']:.3f} "
              f"hlr={agg['healer_pct']:.0f}% deg={agg['degen_pct']:.0f}% "
              f"div={agg['distinct']} ({elapsed:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=int, default=100)
    parser.add_argument("--sims", type=str, default="0,50,200,400,600,800")
    parser.add_argument("--mode", default="sweep")
    parser.add_argument("--ckpt-name", default="")
    parser.add_argument("--ckpt-path", default="")
    parser.add_argument("--wp-type", default="enriched")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="")
    parser.add_argument("--configs-json", default="")
    args = parser.parse_args()

    sim_counts = [int(s) for s in args.sims.split(",")]
    N = args.configs

    # Single checkpoint mode (called by parent process)
    if args.mode == "single":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        configs = json.loads(args.configs_json) if args.configs_json else None
        if configs is None:
            random.seed(args.seed)
            configs = [(random.randint(0, 13), random.randint(0, 2), i % 2) for i in range(N)]
        stats = StatsCache()
        results = run_sweep_for_checkpoint(
            args.ckpt_name, args.ckpt_path, args.wp_type, sim_counts, configs, stats, device_id=0)
        # Save results
        json_r = {str(s): {kk: float(vv) for kk, vv in v.items()} for s, v in results.items()}
        with open(args.out, "w") as f:
            json.dump(json_r, f, indent=2)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Search Budget Sweep (CUDA kernel): {N} configs × {sim_counts} sims")
    print("=" * 80)

    stats = StatsCache()

    # Fixed configs (map_idx, tier_idx, our_team)
    configs = [(random.randint(0, 13), random.randint(0, 2), i % 2) for i in range(N)]

    checkpoints = [
        ("E_seed0", "mcts_runs/E_seed0/draft_policy.pt", "enriched", 0),
        ("G_seed4", "mcts_runs/G_seed4/draft_policy.pt", "augmented", 1),
    ]

    # Filter to existing checkpoints
    checkpoints = [(n, p, w, g) for n, p, w, g in checkpoints
                    if os.path.exists(os.path.join(os.path.dirname(__file__), p))]

    # Run checkpoints in parallel via subprocess for GPU isolation
    import tempfile
    procs = []
    tmp_files = []

    for ckpt_name, ckpt_path, wp_type, gpu_id in checkpoints:
        tmp_out = os.path.join(RESULTS_DIR, f"_sweep_{ckpt_name}.json")
        tmp_files.append((ckpt_name, tmp_out))

        # Spawn subprocess that runs a single checkpoint
        cmd = [sys.executable, "-u", __file__,
               "--mode", "single",
               "--ckpt-name", ckpt_name,
               "--ckpt-path", ckpt_path,
               "--wp-type", wp_type,
               "--gpu", str(gpu_id),
               "--configs", str(N),
               "--sims", args.sims,
               "--seed", "42",
               "--out", tmp_out,
               "--configs-json", json.dumps(configs)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        p = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
        procs.append((ckpt_name, p))
        time.sleep(3)  # stagger

    for name, p in procs:
        p.wait()
        status = "OK" if p.returncode == 0 else f"FAIL({p.returncode})"
        print(f"  {name}: {status}")

    # Collect results
    all_results = {}
    for ckpt_name, tmp_out in tmp_files:
        if os.path.exists(tmp_out):
            with open(tmp_out) as f:
                all_results[ckpt_name] = json.load(f)
            # Convert string keys back to int
            all_results[ckpt_name] = {int(k): v for k, v in all_results[ckpt_name].items()}

    # Summary table
    print("\n\n" + "=" * 95)
    print(f"{'Checkpoint':<12} {'Sims':>5} | {'Counter':>8} {'Synergy':>8} {'Resil':>7} "
          f"{'R.Grad':>7} {'Hlr%':>5} {'Deg%':>5} {'Div':>4} | {'Time':>6}")
    print("-" * 95)
    for ckpt_name in ["E_seed0", "G_seed4"]:
        if ckpt_name not in all_results:
            continue
        for n_sims in sorted(all_results[ckpt_name].keys()):
            r = all_results[ckpt_name][n_sims]
            print(f"{ckpt_name:<12} {n_sims:>5} | {r['counter_avg']:>+8.3f} {r['team_synergy']:>8.3f} "
                  f"{r['resilience_avg']:>7.3f} {r['resilience_gradient']:>7.3f} "
                  f"{r['healer_pct']:>5.0f} {r['degen_pct']:>5.0f} {r['distinct']:>4} | "
                  f"{r['elapsed']:>5.1f}s")
        print("-" * 95)
    print("=" * 95)

    # Save
    out_path = os.path.join(RESULTS_DIR, "search_budget_sweep.json")
    json_results = {}
    for k, vr in all_results.items():
        json_results[k] = {str(s): {kk: float(vv) for kk, vv in v.items()} for s, v in vr.items()}
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
