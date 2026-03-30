"""
BEAR-style draft agent: Q-learning + behavior policy constraint.

Uses the existing Generic Draft models as the behavior policy π_β,
trains a Q-function on offline data, and at inference blends:
    score = Q(s,a) + β * log π_β(a|s)

This constrains the policy to stay near human drafting behavior while
biasing toward higher-value actions.

Usage:
    set -a && source .env && set +a
    python3 -u training/experiment_bear_draft.py --sweep --drafts 200
"""
import os
import sys
import json
import random
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    is_degenerate,
    NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS,
    map_to_one_hot, tier_to_one_hot,
    load_replay_data, split_data,
    HERO_ROLE_FINE,
)
from train_draft_policy import DraftState, DRAFT_ORDER
from train_generic_draft import GenericDraftModel
from experiment_cql_draft import (
    CQLDraftAgent, CQLDataset, replay_to_transitions, RESULTS_DIR,
)


def train_q_only(train_transitions, test_transitions, lr=3e-4, epochs=50,
                 batch_size=2048, device=None, seed=42):
    """Train a standard Q-network (no CQL penalty) with MC returns."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = CQLDataset(train_transitions)
    test_ds = CQLDataset(test_transitions)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = CQLDraftAgent().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    save_path = os.path.join(RESULTS_DIR, "_bear_q_temp.pt")
    patience = 0

    for epoch in range(epochs):
        model.train()
        for states, actions, outcomes, masks in train_loader:
            states, actions, outcomes, masks = (
                states.to(device), actions.to(device),
                outcomes.to(device), masks.to(device),
            )
            q = model.q_values(states, masks)
            q_taken = q.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = F.binary_cross_entropy(q_taken, outcomes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0
        test_n = 0
        with torch.no_grad():
            for states, actions, outcomes, masks in test_loader:
                states, actions, outcomes, masks = (
                    states.to(device), actions.to(device),
                    outcomes.to(device), masks.to(device),
                )
                q = model.q_values(states, masks)
                q_taken = q.gather(1, actions.unsqueeze(1)).squeeze(1)
                test_loss += F.binary_cross_entropy(q_taken, outcomes, reduction='sum').item()
                test_n += len(states)

        avg_loss = test_loss / test_n
        if epoch % 10 == 0:
            print(f"  Q epoch {epoch+1}: test_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))
    model.eval()
    return model


def run_bear_drafts(q_model, gd_models, device, draft_configs, beta=1.0):
    """Run drafts using score = Q(s,a) + β * log π_β(a|s)."""
    healer_heroes = set(h for h, r in HERO_ROLE_FINE.items() if r == "healer")
    tank_heroes = set(h for h, r in HERO_ROLE_FINE.items() if r == "tank")
    bruiser_heroes = set(h for h, r in HERO_ROLE_FINE.items() if r == "bruiser")
    ranged_heroes = set(h for h, r in HERO_ROLE_FINE.items()
                        if r in ("ranged_aa", "ranged_mage", "pusher"))
    frontline = tank_heroes | bruiser_heroes

    results = []
    for di, (_, game_map, tier, our_team) in enumerate(draft_configs):
        state = DraftState(game_map, tier, our_team=our_team)

        while not state.is_terminal():
            team, action_type = DRAFT_ORDER[state.step]

            if team == our_team:
                step_type = 0.0 if action_type == "ban" else 1.0
                step_num = state.step / 15.0
                s = np.concatenate([
                    state.team0_picks, state.team1_picks, state.bans,
                    map_to_one_hot(game_map), tier_to_one_hot(tier),
                    [step_num, step_type],
                ])
                mask_np = state.valid_mask_np()
                s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
                m_t = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0).to(device)

                # Q-values
                with torch.no_grad():
                    q = q_model(s_t, m_t).squeeze(0).cpu()  # raw logits

                # Behavior policy log-probs (average over GD ensemble)
                s_cpu = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                m_cpu = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)
                log_probs = torch.zeros(NUM_HEROES)
                for gd in gd_models:
                    with torch.no_grad():
                        logits = gd(s_cpu, m_cpu).squeeze(0)
                        lp = F.log_softmax(logits, dim=0)
                        log_probs += lp
                log_probs /= len(gd_models)

                # Blended score
                score = q + beta * log_probs
                # Mask invalid
                score[mask_np < 0.5] = float('-inf')
                action = score.argmax().item()
                state.apply_action(action, team, action_type)
            else:
                gd = random.choice(gd_models)
                x = state.to_tensor_gd(torch.device("cpu"))
                mask = state.valid_mask(torch.device("cpu"))
                with torch.no_grad():
                    logits = gd(x, mask)
                    probs = F.softmax(logits / 1.0, dim=1)
                    action = torch.multinomial(probs, 1).item()
                state.apply_action(action, team, action_type)

        our_picks_vec = state.team0_picks if our_team == 0 else state.team1_picks
        our_heroes = [HEROES[i] for i in range(NUM_HEROES) if our_picks_vec[i] > 0]
        has_healer = any(h in healer_heroes for h in our_heroes)
        has_frontline = any(h in frontline for h in our_heroes)
        has_ranged = any(h in ranged_heroes for h in our_heroes)
        role_counts = {}
        for h in our_heroes:
            r = HERO_ROLE_FINE.get(h, "unknown")
            role_counts[r] = role_counts.get(r, 0) + 1
        # Stacking check moved to is_degenerate()
        is_degen = is_degenerate(our_heroes)

        results.append({"has_healer": has_healer, "has_frontline": has_frontline,
                        "has_ranged": has_ranged, "is_degen": is_degen})

        if (di + 1) % 50 == 0:
            n = di + 1
            hr = sum(1 for r in results if r["has_healer"]) / n * 100
            dr = sum(1 for r in results if r["is_degen"]) / n * 100
            print(f"  {n}/{len(draft_configs)}: healer={hr:.1f}% degen={dr:.1f}%")

    n = len(results)
    return {
        "healer_rate": sum(1 for r in results if r["has_healer"]) / n * 100,
        "frontline_rate": sum(1 for r in results if r["has_frontline"]) / n * 100,
        "ranged_rate": sum(1 for r in results if r["has_ranged"]) / n * 100,
        "degen_rate": sum(1 for r in results if r["is_degen"]) / n * 100,
        "n_drafts": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--drafts", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    data = load_replay_data()
    train_data, test_data = split_data(data)

    print("Building transitions...")
    t0 = time.time()
    train_transitions = []
    for d in train_data:
        t = replay_to_transitions(d)
        if t:
            train_transitions.extend(t)
    test_transitions = []
    for d in test_data:
        t = replay_to_transitions(d)
        if t:
            test_transitions.extend(t)
    print(f"  {len(train_transitions):,} train, {len(test_transitions):,} test in {time.time()-t0:.1f}s")

    # Load GD models
    gd_models = []
    for i in range(5):
        gd_path = os.path.join(os.path.dirname(__file__), f"generic_draft_{i}.pt")
        if not os.path.exists(gd_path):
            gd_path = os.path.join(os.path.dirname(__file__), "generic_draft.pt")
        gd = GenericDraftModel()
        gd.load_state_dict(torch.load(gd_path, weights_only=True, map_location="cpu"))
        gd.cpu().eval()
        gd_models.append(gd)

    # Train Q-network (no CQL penalty)
    print("\nTraining Q-network (no CQL)...")
    q_model = train_q_only(train_transitions, test_transitions,
                           epochs=args.epochs, device=device)

    # Also run pure GD (behavior policy only) as baseline
    print("\nPure GD baseline (β→∞, no Q)...")
    gd_only_results = run_bear_drafts(q_model, gd_models, device,
                                       [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i%2)
                                        for i in range(args.drafts)],
                                       beta=1000.0)  # huge beta = pure behavior policy
    print(f"  GD only: healer={gd_only_results['healer_rate']:.1f}% degen={gd_only_results['degen_rate']:.1f}%")

    # Draft configs
    random.seed(42)
    draft_configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
                     for i in range(args.drafts)]

    # Sweep betas
    betas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0] if args.sweep else [args.beta]
    # Also include Q-only (beta=0) and GD-only (high beta)
    all_results = []

    for beta in betas:
        print(f"\n{'='*50}")
        print(f"BEAR β={beta}")
        print(f"{'='*50}")
        dr = run_bear_drafts(q_model, gd_models, device, draft_configs, beta=beta)
        print(f"  Healer: {dr['healer_rate']:.1f}%  Degen: {dr['degen_rate']:.1f}%")
        all_results.append({"beta": beta, **dr})

    # Summary
    print(f"\n{'='*70}")
    print("BEAR-STYLE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Beta':>8} {'Healer%':>8} {'Degen%':>8} {'Front%':>8} {'Ranged%':>8}")
    print("-" * 45)
    print(f"{'GD only':>8} {gd_only_results['healer_rate']:>8.1f} {gd_only_results['degen_rate']:>8.1f} "
          f"{gd_only_results['frontline_rate']:>8.1f} {gd_only_results['ranged_rate']:>8.1f}")
    print("-" * 45)
    for r in all_results:
        print(f"{r['beta']:>8.1f} {r['healer_rate']:>8.1f} {r['degen_rate']:>8.1f} "
              f"{r['frontline_rate']:>8.1f} {r['ranged_rate']:>8.1f}")

    # Baselines
    print("-" * 45)
    print(f"{'naive':>8} {'49.0':>8} {'75.8':>8} {'95.0':>8} {'65.2':>8}  (WP greedy)")
    print(f"{'enrich':>8} {'66.5':>8} {'57.7':>8} {'96.0':>8} {'73.3':>8}  (WP greedy)")
    print(f"{'e+synth':>8} {'94.5':>8} {'26.5':>8} {'93.0':>8} {'88.0':>8}  (WP greedy)")

    save_path = os.path.join(RESULTS_DIR, "bear_results.json")
    with open(save_path, "w") as f:
        json.dump({"gd_only": gd_only_results, "bear": all_results}, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
