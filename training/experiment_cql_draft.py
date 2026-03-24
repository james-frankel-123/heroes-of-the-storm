"""
Experiment: Conservative Q-Learning for Draft Selection.

Tests whether algorithmic pessimism (CQL) can achieve comparable draft quality
to feature engineering + synthetic augmentation, using only naive multi-hot features.

Key question: Can CQL with multi-hot inputs match the 94.5% healer / 26.5% degen
achieved by enriched features + synthetic augmentation?

Usage:
    set -a && source .env && set +a
    python3 -u training/experiment_cql_draft.py --features naive --alpha 1.0
    python3 -u training/experiment_cql_draft.py --sweep
    python3 -u training/experiment_cql_draft.py --features enriched --alpha 1.0
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
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS,
    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
    load_replay_data, split_data,
    HERO_ROLE_FINE,
)
from train_draft_policy import DraftState, DRAFT_ORDER

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "experiment_results", "cql")

# ── Dataset: replay → transitions ──

def replay_to_transitions(replay):
    """Convert a replay into (state, action, outcome, valid_mask, team) tuples.
    Uses Monte Carlo returns: Q-target = game outcome for all steps.
    Two perspectives per replay (team-swap).
    """
    draft_order = replay.get("draft_order")
    if not draft_order or len(draft_order) != 16:
        return []

    winner = replay["winner"]  # 0 or 1
    game_map = replay["game_map"]
    skill_tier = replay.get("skill_tier", "mid")

    team0_picks = np.zeros(NUM_HEROES, dtype=np.float32)
    team1_picks = np.zeros(NUM_HEROES, dtype=np.float32)
    bans = np.zeros(NUM_HEROES, dtype=np.float32)
    taken = set()

    transitions = []
    for step_idx, step in enumerate(draft_order):
        hero = step["hero"]
        hero_idx = HERO_TO_IDX.get(hero)
        if hero_idx is None:
            return []

        step_type = float(step["type"])  # 0=ban, 1=pick
        step_num = step_idx / 15.0

        # State before action
        m = map_to_one_hot(game_map)
        t = tier_to_one_hot(skill_tier)
        state = np.concatenate([
            team0_picks.copy(), team1_picks.copy(), bans.copy(),
            m, t, [step_num, step_type],
        ])

        # Valid mask
        valid = np.ones(NUM_HEROES, dtype=np.float32)
        for idx in taken:
            valid[idx] = 0.0

        # Determine acting team
        slot = step.get("player_slot", 0)
        acting_team = 0 if slot <= 4 else 1

        # MC return: did this team's side win?
        outcome_for_actor = 1.0 if winner == acting_team else 0.0

        transitions.append({
            "state": state,
            "action": hero_idx,
            "outcome": outcome_for_actor,
            "valid_mask": valid,
            "team": acting_team,
        })

        # Update state
        taken.add(hero_idx)
        if step_type == 0:
            bans[hero_idx] = 1.0
        else:
            if acting_team == 0:
                team0_picks[hero_idx] = 1.0
            else:
                team1_picks[hero_idx] = 1.0

    # Team-swap augmentation: flip team0/team1, flip outcomes
    team0_picks2 = np.zeros(NUM_HEROES, dtype=np.float32)
    team1_picks2 = np.zeros(NUM_HEROES, dtype=np.float32)
    bans2 = np.zeros(NUM_HEROES, dtype=np.float32)
    taken2 = set()

    for step_idx, step in enumerate(draft_order):
        hero = step["hero"]
        hero_idx = HERO_TO_IDX[hero]
        step_type = float(step["type"])
        step_num = step_idx / 15.0

        m = map_to_one_hot(game_map)
        t = tier_to_one_hot(skill_tier)
        # Swapped: team1 in slot 0, team0 in slot 1
        state = np.concatenate([
            team1_picks2.copy(), team0_picks2.copy(), bans2.copy(),
            m, t, [step_num, step_type],
        ])

        valid = np.ones(NUM_HEROES, dtype=np.float32)
        for idx in taken2:
            valid[idx] = 0.0

        slot = step.get("player_slot", 0)
        acting_team = 0 if slot <= 4 else 1
        # In swapped view, team 0 becomes team 1 and vice versa
        swapped_team = 1 - acting_team
        outcome_swapped = 1.0 if winner != acting_team else 0.0

        transitions.append({
            "state": state,
            "action": hero_idx,
            "outcome": outcome_swapped,
            "valid_mask": valid,
            "team": swapped_team,
        })

        taken2.add(hero_idx)
        if step_type == 0:
            bans2[hero_idx] = 1.0
        else:
            if acting_team == 0:
                # In original, team0 picked → in swapped view, goes to team1 slot
                team1_picks2[hero_idx] = 1.0
            else:
                team0_picks2[hero_idx] = 1.0

    return transitions


class CQLDataset(Dataset):
    def __init__(self, transitions):
        self.states = torch.tensor(np.array([t["state"] for t in transitions]), dtype=torch.float32)
        self.actions = torch.tensor([t["action"] for t in transitions], dtype=torch.long)
        self.outcomes = torch.tensor([t["outcome"] for t in transitions], dtype=torch.float32)
        self.masks = torch.tensor(np.array([t["valid_mask"] for t in transitions]), dtype=torch.float32)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.outcomes[idx], self.masks[idx]


# ── CQL Agent ──

class CQLDraftAgent(nn.Module):
    def __init__(self, input_dim=289):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_HEROES),
        )

    def forward(self, state, valid_mask=None):
        q = self.net(state)
        if valid_mask is not None:
            q = q + (1 - valid_mask) * (-1e9)
        return q

    def q_values(self, state, valid_mask=None):
        """Bounded Q-values in [0,1] via sigmoid for training."""
        q = self.net(state)
        if valid_mask is not None:
            q = q + (1 - valid_mask) * (-1e9)
        return torch.sigmoid(q)


def train_cql(train_transitions, test_transitions, alpha=1.0, lr=3e-4,
              epochs=50, batch_size=2048, device=None, seed=42):
    """Train CQL agent. Returns (model, metrics)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = CQLDataset(train_transitions)
    test_ds = CQLDataset(test_transitions)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = CQLDraftAgent().to(device)
    target_model = CQLDraftAgent().to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tau = 0.01  # soft target update rate

    best_test_loss = float('inf')
    save_path = os.path.join(RESULTS_DIR, f"_cql_temp_a{alpha}.pt")
    patience = 0

    for epoch in range(epochs):
        model.train()
        total_bellman = 0
        total_cql = 0
        total_n = 0

        for states, actions, outcomes, masks in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            outcomes = outcomes.to(device)
            masks = masks.to(device)

            # Bounded Q-values for taken actions (MC target = outcome in [0,1])
            q_bounded = model.q_values(states, masks)
            q_taken = q_bounded.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Bellman loss: BCE since both Q and target are in [0,1]
            bellman_loss = F.binary_cross_entropy(q_taken, outcomes)

            # CQL penalty on raw logits (before sigmoid) for proper logsumexp
            q_raw = model(states, masks)
            logsumexp_q = torch.logsumexp(q_raw, dim=1).mean()
            data_q = q_raw.gather(1, actions.unsqueeze(1)).squeeze(1).mean()
            cql_penalty = logsumexp_q - data_q

            loss = bellman_loss + alpha * cql_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Soft target update
            with torch.no_grad():
                for p, tp in zip(model.parameters(), target_model.parameters()):
                    tp.data.mul_(1 - tau).add_(p.data * tau)

            total_bellman += bellman_loss.item() * len(states)
            total_cql += cql_penalty.item() * len(states)
            total_n += len(states)

        # Eval
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for states, actions, outcomes, masks in test_loader:
                states = states.to(device)
                actions = actions.to(device)
                outcomes = outcomes.to(device)
                masks = masks.to(device)
                q_bounded = model.q_values(states, masks)
                q_taken = q_bounded.gather(1, actions.unsqueeze(1)).squeeze(1)
                test_loss += F.binary_cross_entropy(q_taken, outcomes, reduction='sum').item()
                # "Accuracy": does argmax Q match the actual action?
                predicted = q_bounded.argmax(dim=1)
                test_correct += (predicted == actions).sum().item()
                test_total += len(states)

        avg_test_loss = test_loss / test_total
        test_acc = test_correct / test_total * 100

        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1}: bellman={total_bellman/total_n:.4f} "
                  f"cql={total_cql/total_n:.4f} test_loss={avg_test_loss:.4f} "
                  f"action_match={test_acc:.1f}%")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), save_path)
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))
    model.eval()
    return model, {"test_loss": best_test_loss, "action_match": test_acc}


# ── Draft evaluation ──

def run_cql_drafts(model, device, draft_configs, gd_models):
    """Run greedy drafts using CQL agent (argmax Q at each step)."""
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
                # Our turn: argmax Q
                step_type = 0.0 if action_type == "ban" else 1.0
                step_num = state.step / 15.0
                s = np.concatenate([
                    state.team0_picks, state.team1_picks, state.bans,
                    map_to_one_hot(game_map), tier_to_one_hot(tier),
                    [step_num, step_type],
                ])
                mask = state.valid_mask_np()
                s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
                m_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q = model(s_t, m_t).squeeze(0).cpu()
                action = q.argmax().item()
                state.apply_action(action, team, action_type)
            else:
                # Opponent: GD sample
                gd = random.choice(gd_models)
                x = state.to_tensor_gd(torch.device("cpu"))
                mask = state.valid_mask(torch.device("cpu"))
                with torch.no_grad():
                    logits = gd(x, mask)
                    probs = F.softmax(logits / 1.0, dim=1)
                    action = torch.multinomial(probs, 1).item()
                state.apply_action(action, team, action_type)

        # Analyze our team's composition
        our_picks_vec = state.team0_picks if our_team == 0 else state.team1_picks
        our_heroes = [HEROES[i] for i in range(NUM_HEROES) if our_picks_vec[i] > 0]

        has_healer = any(h in healer_heroes for h in our_heroes)
        has_frontline = any(h in frontline for h in our_heroes)
        has_ranged = any(h in ranged_heroes for h in our_heroes)
        role_counts = {}
        for h in our_heroes:
            r = HERO_ROLE_FINE.get(h, "unknown")
            role_counts[r] = role_counts.get(r, 0) + 1
        has_stacking = any(c >= 3 for c in role_counts.values())
        is_degen = not has_healer or not has_frontline or not has_ranged or has_stacking

        results.append({
            "heroes": our_heroes,
            "has_healer": has_healer,
            "has_frontline": has_frontline,
            "has_ranged": has_ranged,
            "is_degen": is_degen,
        })

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


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="naive", choices=["naive", "enriched"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--drafts", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading replay data...")
    data = load_replay_data()
    train_data, test_data = split_data(data)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Build transition dataset
    print("Building transition dataset...")
    t0 = time.time()
    train_transitions = []
    skipped = 0
    for d in train_data:
        t = replay_to_transitions(d)
        if t:
            train_transitions.extend(t)
        else:
            skipped += 1
    test_transitions = []
    for d in test_data:
        t = replay_to_transitions(d)
        if t:
            test_transitions.extend(t)

    print(f"  Train transitions: {len(train_transitions):,} ({skipped} replays skipped)")
    print(f"  Test transitions: {len(test_transitions):,}")
    print(f"  Built in {time.time()-t0:.1f}s")

    # Load GD models for opponent simulation
    from train_generic_draft import GenericDraftModel
    gd_models = []
    for i in range(5):
        gd_path = os.path.join(os.path.dirname(__file__), f"generic_draft_{i}.pt")
        if not os.path.exists(gd_path):
            gd_path = os.path.join(os.path.dirname(__file__), "generic_draft.pt")
        gd = GenericDraftModel()
        gd.load_state_dict(torch.load(gd_path, weights_only=True, map_location="cpu"))
        gd.cpu().eval()
        gd_models.append(gd)
    print(f"Loaded {len(gd_models)} GD models")

    # Draft configs
    random.seed(42)
    draft_configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
                     for i in range(args.drafts)]

    # Sweep or single config
    if args.sweep:
        alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
    else:
        alphas = [args.alpha]

    all_results = []
    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"CQL alpha={alpha}, features={args.features}")
        print(f"{'='*60}")

        model, metrics = train_cql(
            train_transitions, test_transitions,
            alpha=alpha, lr=3e-4, epochs=args.epochs,
            batch_size=2048, device=device,
        )
        print(f"  Training done: {metrics}")

        print(f"\nRunning {args.drafts} greedy drafts...")
        draft_results = run_cql_drafts(model, device, draft_configs, gd_models)
        print(f"\n  Healer: {draft_results['healer_rate']:.1f}%")
        print(f"  Frontline: {draft_results['frontline_rate']:.1f}%")
        print(f"  Ranged: {draft_results['ranged_rate']:.1f}%")
        print(f"  Degenerate: {draft_results['degen_rate']:.1f}%")

        all_results.append({
            "alpha": alpha,
            "features": args.features,
            "metrics": metrics,
            "draft_results": draft_results,
        })

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Alpha':>8} {'Action%':>8} {'Healer%':>8} {'Degen%':>8} {'Front%':>8} {'Ranged%':>8}")
    print("-" * 55)

    # Reference baselines
    print(f"{'---':>8} {'---':>8} {'49.0':>8} {'75.8':>8} {'95.0':>8} {'65.2':>8}  (naive WP)")
    print(f"{'---':>8} {'---':>8} {'66.5':>8} {'57.7':>8} {'96.0':>8} {'73.3':>8}  (enriched WP)")
    print(f"{'---':>8} {'---':>8} {'94.5':>8} {'26.5':>8} {'93.0':>8} {'88.0':>8}  (enriched+synth)")
    print("-" * 55)

    for r in all_results:
        d = r["draft_results"]
        print(f"{r['alpha']:>8.1f} {r['metrics']['action_match']:>7.1f}% "
              f"{d['healer_rate']:>8.1f} {d['degen_rate']:>8.1f} "
              f"{d['frontline_rate']:>8.1f} {d['ranged_rate']:>8.1f}  (CQL)")

    # Save
    save_path = os.path.join(RESULTS_DIR, f"cql_{args.features}_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
