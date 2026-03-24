"""
CQL with enriched features (283-dim state).

Same CQL objective as experiment_cql_draft.py, but the state includes the
enriched features (role_counts, pairwise stats, comp_wr, etc.) that the
supervised WP model uses. Tests whether CQL + compositional features can
discover context-aware drafting, not just behavioral cloning.

Usage:
    set -a && source .env && set +a
    python3 -u training/experiment_cql_enriched.py --sweep --drafts 200
"""
import os
import sys
import json
import random
import argparse
import time
from collections import Counter
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
from sweep_enriched_wp import (
    StatsCache, FEATURE_GROUPS, compute_group_indices, extract_features,
)
from train_draft_policy import DraftState, DRAFT_ORDER
from train_generic_draft import GenericDraftModel
from experiment_cql_draft import CQLDraftAgent
from experiment_synthetic_augmentation import ENRICHED_GROUPS
from experiment_rich_evaluation import (
    counter_responsiveness, synergy_exploitation, draft_diversity,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "experiment_results", "cql")


def get_enriched_cols(group_indices):
    cols = []
    for g in ENRICHED_GROUPS:
        s, e = group_indices[g]
        cols.extend(range(s, e))
    return cols


def build_enriched_state(t0_heroes, t1_heroes, game_map, tier, step_num, step_type,
                         stats, group_indices, enriched_cols, all_mask):
    """Build 289 + 86 enriched = 375-dim state for a draft step.
    Base 289: t0_multi_hot(90) + t1_multi_hot(90) + bans(90) + map(14) + tier(3) + step(1) + type(1)
    Actually we use the same base as extract_features (197 dim) + enriched (86) + step info.

    Wait - the WP model uses 197-dim base (t0+t1+map+tier, no bans/step).
    CQL needs bans and step info too. So we concatenate:
      - base_197 from extract_features (t0+t1+map+tier multi-hots)
      - enriched_86 from extract_features
      - bans(90) + step_num(1) + step_type(1) = 92 extra dims
    Total: 197 + 86 + 92 = 375 dims

    Actually, let's keep it simpler. Use the standard 289-dim CQL state
    (which includes bans and step info) PLUS the 86 enriched features.
    Total: 289 + 86 = 375.
    """
    # Standard CQL state components are already in the caller.
    # Here we just compute the enriched features from current picks.
    d = {
        "team0_heroes": t0_heroes,
        "team1_heroes": t1_heroes,
        "game_map": game_map,
        "skill_tier": tier,
        "winner": 0,
    }
    _, enriched_all = extract_features(d, stats, all_mask)
    return enriched_all[enriched_cols]


def replay_to_enriched_transitions(replay, stats, group_indices, enriched_cols, all_mask):
    """Convert replay to transitions with enriched features appended."""
    draft_order = replay.get("draft_order")
    if not draft_order or len(draft_order) != 16:
        return []

    winner = replay["winner"]
    game_map = replay["game_map"]
    skill_tier = replay.get("skill_tier", "mid")

    team0_picks_mh = np.zeros(NUM_HEROES, dtype=np.float32)
    team1_picks_mh = np.zeros(NUM_HEROES, dtype=np.float32)
    bans = np.zeros(NUM_HEROES, dtype=np.float32)
    taken = set()
    team0_heroes = []
    team1_heroes = []

    transitions = []
    for step_idx, step in enumerate(draft_order):
        hero = step["hero"]
        hero_idx = HERO_TO_IDX.get(hero)
        if hero_idx is None:
            return []

        step_type = float(step["type"])
        step_num = step_idx / 15.0

        # Build base state (289 dims)
        m = map_to_one_hot(game_map)
        t = tier_to_one_hot(skill_tier)
        base_state = np.concatenate([
            team0_picks_mh.copy(), team1_picks_mh.copy(), bans.copy(),
            m, t, [step_num, step_type],
        ])

        # Enriched features from current picks
        enriched = build_enriched_state(
            team0_heroes.copy(), team1_heroes.copy(),
            game_map, skill_tier, step_num, step_type,
            stats, group_indices, enriched_cols, all_mask,
        )

        state = np.concatenate([base_state, enriched])

        # Valid mask
        valid = np.ones(NUM_HEROES, dtype=np.float32)
        for idx in taken:
            valid[idx] = 0.0

        slot = step.get("player_slot", 0)
        acting_team = 0 if slot <= 4 else 1
        outcome = 1.0 if winner == acting_team else 0.0

        transitions.append({
            "state": state,
            "action": hero_idx,
            "outcome": outcome,
            "valid_mask": valid,
        })

        # Update state
        taken.add(hero_idx)
        if step_type == 0:
            bans[hero_idx] = 1.0
        else:
            if acting_team == 0:
                team0_picks_mh[hero_idx] = 1.0
                team0_heroes.append(hero)
            else:
                team1_picks_mh[hero_idx] = 1.0
                team1_heroes.append(hero)

    # Team-swap augmentation
    team0_picks_mh2 = np.zeros(NUM_HEROES, dtype=np.float32)
    team1_picks_mh2 = np.zeros(NUM_HEROES, dtype=np.float32)
    bans2 = np.zeros(NUM_HEROES, dtype=np.float32)
    taken2 = set()
    team0_heroes2 = []
    team1_heroes2 = []

    for step_idx, step in enumerate(draft_order):
        hero = step["hero"]
        hero_idx = HERO_TO_IDX[hero]
        step_type = float(step["type"])
        step_num = step_idx / 15.0

        m = map_to_one_hot(game_map)
        t = tier_to_one_hot(skill_tier)
        # Swapped teams in base
        base_state = np.concatenate([
            team1_picks_mh2.copy(), team0_picks_mh2.copy(), bans2.copy(),
            m, t, [step_num, step_type],
        ])
        # Swapped teams in enriched
        enriched = build_enriched_state(
            team1_heroes2.copy(), team0_heroes2.copy(),
            game_map, skill_tier, step_num, step_type,
            stats, group_indices, enriched_cols, all_mask,
        )
        state = np.concatenate([base_state, enriched])

        valid = np.ones(NUM_HEROES, dtype=np.float32)
        for idx in taken2:
            valid[idx] = 0.0

        slot = step.get("player_slot", 0)
        acting_team = 0 if slot <= 4 else 1
        outcome = 1.0 if winner != acting_team else 0.0

        transitions.append({
            "state": state,
            "action": hero_idx,
            "outcome": outcome,
            "valid_mask": valid,
        })

        taken2.add(hero_idx)
        if step_type == 0:
            bans2[hero_idx] = 1.0
        else:
            if acting_team == 0:
                team1_picks_mh2[hero_idx] = 1.0
                team1_heroes2.append(hero)
            else:
                team0_picks_mh2[hero_idx] = 1.0
                team0_heroes2.append(hero)

    return transitions


class EnrichedCQLDataset(Dataset):
    def __init__(self, transitions):
        self.states = torch.tensor(np.array([t["state"] for t in transitions]), dtype=torch.float32)
        self.actions = torch.tensor([t["action"] for t in transitions], dtype=torch.long)
        self.outcomes = torch.tensor([t["outcome"] for t in transitions], dtype=torch.float32)
        self.masks = torch.tensor(np.array([t["valid_mask"] for t in transitions]), dtype=torch.float32)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.outcomes[idx], self.masks[idx]


def train_enriched_cql(train_transitions, test_transitions, input_dim, alpha=1.0,
                       lr=3e-4, epochs=50, batch_size=2048, device=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = EnrichedCQLDataset(train_transitions)
    test_ds = EnrichedCQLDataset(test_transitions)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = CQLDraftAgent(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    save_path = os.path.join(RESULTS_DIR, f"_cql_enriched_a{alpha}.pt")
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        total_bellman = 0
        total_cql = 0
        total_n = 0

        for states, actions, outcomes, masks in train_loader:
            states, actions, outcomes, masks = (
                states.to(device), actions.to(device),
                outcomes.to(device), masks.to(device),
            )
            q_bounded = model.q_values(states, masks)
            q_taken = q_bounded.gather(1, actions.unsqueeze(1)).squeeze(1)
            bellman_loss = F.binary_cross_entropy(q_taken, outcomes)

            q_raw = model(states, masks)
            logsumexp_q = torch.logsumexp(q_raw, dim=1).mean()
            data_q = q_raw.gather(1, actions.unsqueeze(1)).squeeze(1).mean()
            cql_penalty = logsumexp_q - data_q

            loss = bellman_loss + alpha * cql_penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_bellman += bellman_loss.item() * len(states)
            total_cql += cql_penalty.item() * len(states)
            total_n += len(states)

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
                qt = q.gather(1, actions.unsqueeze(1)).squeeze(1)
                test_loss += F.binary_cross_entropy(qt, outcomes, reduction='sum').item()
                test_n += len(states)

        avg_loss = test_loss / test_n
        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1}: bellman={total_bellman/total_n:.4f} "
                  f"cql={total_cql/total_n:.4f} test={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= 10:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))
    model.eval()
    return model, save_path


def run_enriched_cql_drafts(model, input_dim, stats, group_indices, enriched_cols,
                            all_mask, device, draft_configs, gd_models):
    healer_heroes = set(h for h, r in HERO_ROLE_FINE.items() if r == "healer")
    tank_heroes = set(h for h, r in HERO_ROLE_FINE.items() if r == "tank")
    bruiser_heroes = set(h for h, r in HERO_ROLE_FINE.items() if r == "bruiser")
    ranged_heroes = set(h for h, r in HERO_ROLE_FINE.items()
                        if r in ("ranged_aa", "ranged_mage", "pusher"))
    frontline = tank_heroes | bruiser_heroes

    results = []
    for di, (_, game_map, tier, our_team) in enumerate(draft_configs):
        state = DraftState(game_map, tier, our_team=our_team)
        team0_heroes = []
        team1_heroes = []

        while not state.is_terminal():
            team, action_type = DRAFT_ORDER[state.step]

            if team == our_team:
                step_type = 0.0 if action_type == "ban" else 1.0
                step_num = state.step / 15.0
                base = np.concatenate([
                    state.team0_picks, state.team1_picks, state.bans,
                    map_to_one_hot(game_map), tier_to_one_hot(tier),
                    [step_num, step_type],
                ])
                enriched = build_enriched_state(
                    team0_heroes, team1_heroes, game_map, tier,
                    step_num, step_type, stats, group_indices, enriched_cols, all_mask,
                )
                s = np.concatenate([base, enriched])
                mask = state.valid_mask_np()
                s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
                m_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q = model(s_t, m_t).squeeze(0).cpu()
                action = q.argmax().item()

                if action_type == "pick":
                    if team == 0:
                        team0_heroes.append(HEROES[action])
                    else:
                        team1_heroes.append(HEROES[action])
                state.apply_action(action, team, action_type)
            else:
                gd = random.choice(gd_models)
                x = state.to_tensor_gd(torch.device("cpu"))
                mask = state.valid_mask(torch.device("cpu"))
                with torch.no_grad():
                    logits = gd(x, mask)
                    probs = F.softmax(logits / 1.0, dim=1)
                    action = torch.multinomial(probs, 1).item()

                if action_type == "pick":
                    if team == 0:
                        team0_heroes.append(HEROES[action])
                    else:
                        team1_heroes.append(HEROES[action])
                state.apply_action(action, team, action_type)

        our_vec = state.team0_picks if our_team == 0 else state.team1_picks
        opp_vec = state.team1_picks if our_team == 0 else state.team0_picks
        our_picks = [HEROES[i] for i in range(NUM_HEROES) if our_vec[i] > 0]
        opp_picks = [HEROES[i] for i in range(NUM_HEROES) if opp_vec[i] > 0]

        has_healer = any(h in healer_heroes for h in our_picks)
        has_frontline = any(h in frontline for h in our_picks)
        has_ranged = any(h in ranged_heroes for h in our_picks)
        role_counts = {}
        for h in our_picks:
            r = HERO_ROLE_FINE.get(h, "unknown")
            role_counts[r] = role_counts.get(r, 0) + 1
        has_stacking = any(c >= 3 for c in role_counts.values())
        is_degen = not has_healer or not has_frontline or not has_ranged or has_stacking

        results.append({
            "our_picks": our_picks, "opp_picks": opp_picks,
            "game_map": game_map, "tier": tier,
            "has_healer": has_healer, "is_degen": is_degen,
            "has_frontline": has_frontline, "has_ranged": has_ranged,
        })

        if (di + 1) % 50 == 0:
            n = di + 1
            hr = sum(1 for r in results if r["has_healer"]) / n * 100
            dr = sum(1 for r in results if r["is_degen"]) / n * 100
            print(f"  {n}/{len(draft_configs)}: healer={hr:.1f}% degen={dr:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--drafts", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    data = load_replay_data()
    train_data, test_data = split_data(data)
    stats = StatsCache()
    group_indices = compute_group_indices()
    enriched_cols = get_enriched_cols(group_indices)
    all_mask = [True] * len(FEATURE_GROUPS)
    enriched_dim = len(enriched_cols)
    input_dim = 289 + enriched_dim  # 289 base + 86 enriched = 375
    print(f"Input dim: {input_dim} (289 base + {enriched_dim} enriched)")

    # Build transitions
    print("Building enriched transitions...")
    t0 = time.time()
    train_trans = []
    skipped = 0
    for d in train_data:
        t = replay_to_enriched_transitions(d, stats, group_indices, enriched_cols, all_mask)
        if t:
            train_trans.extend(t)
        else:
            skipped += 1
    test_trans = []
    for d in test_data:
        t = replay_to_enriched_transitions(d, stats, group_indices, enriched_cols, all_mask)
        if t:
            test_trans.extend(t)
    print(f"  Train: {len(train_trans):,}, Test: {len(test_trans):,} ({skipped} skipped) in {time.time()-t0:.1f}s")

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

    random.seed(42)
    draft_configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
                     for i in range(args.drafts)]

    alphas = [0.1, 0.5, 1.0, 2.0, 5.0] if args.sweep else [args.alpha]
    all_results = []

    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"CQL ENRICHED alpha={alpha}")
        print(f"{'='*60}")

        model, path = train_enriched_cql(
            train_trans, test_trans, input_dim, alpha=alpha,
            epochs=args.epochs, device=device,
        )

        print(f"\nRunning {args.drafts} drafts...")
        random.seed(42)
        drafts = run_enriched_cql_drafts(
            model, input_dim, stats, group_indices, enriched_cols, all_mask,
            device, draft_configs, gd_models,
        )

        n = len(drafts)
        healer = sum(1 for d in drafts if d["has_healer"]) / n * 100
        degen = sum(1 for d in drafts if d["is_degen"]) / n * 100
        front = sum(1 for d in drafts if d["has_frontline"]) / n * 100
        ranged = sum(1 for d in drafts if d["has_ranged"]) / n * 100

        counters = [counter_responsiveness(d["our_picks"], d["opp_picks"], stats, d["tier"])
                    for d in drafts]
        synergies = [synergy_exploitation(d["our_picks"], stats, d["tier"])
                     for d in drafts]
        div = draft_diversity(drafts)

        print(f"\n  Healer: {healer:.1f}%  Degen: {degen:.1f}%")
        print(f"  Counter: {np.mean(counters):.2f}  Synergy: {np.mean(synergies):.2f}")
        print(f"  Diversity: {div['distinct_heroes']} heroes, entropy={div['entropy']}")

        all_results.append({
            "alpha": alpha,
            "healer": healer, "degen": degen,
            "frontline": front, "ranged": ranged,
            "counter": round(float(np.mean(counters)), 2),
            "synergy": round(float(np.mean(synergies)), 2),
            **div,
        })

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*90}")
    print("CQL ENRICHED SUMMARY")
    print(f"{'='*90}")
    print(f"{'Alpha':>8} {'Heal%':>7} {'Deg%':>6} {'Counter':>8} {'Synergy':>8} "
          f"{'Distinct':>8} {'Entropy':>8} {'Top10%':>7}")
    print("-" * 70)
    # Reference baselines
    print(f"{'GD':>8} {'99.5':>7} {'1.0':>6} {'-0.09':>8} {'-0.05':>8} {'40':>8} {'4.24':>8} {'75.3':>7}")
    print(f"{'CQL 1.0':>8} {'93.5':>7} {'11.0':>6} {'-0.10':>8} {'-0.14':>8} {'44':>8} {'4.42':>8} {'70.7':>7}")
    print(f"{'Enr WP':>8} {'77.0':>7} {'42.5':>6} {'+0.14':>8} {'+1.03':>8} {'86':>8} {'5.96':>8} {'32.0':>7}")
    print(f"{'E+aug':>8} {'96.0':>7} {'13.5':>6} {'+0.06':>8} {'+0.80':>8} {'90':>8} {'5.97':>8} {'32.8':>7}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['alpha']:>8.1f} {r['healer']:>7.1f} {r['degen']:>6.1f} "
              f"{r['counter']:>8.2f} {r['synergy']:>8.2f} "
              f"{r['distinct_heroes']:>8} {r['entropy']:>8.2f} {r['top10_concentration']:>7.1f}")

    save_path = os.path.join(RESULTS_DIR, "cql_enriched_sweep.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
