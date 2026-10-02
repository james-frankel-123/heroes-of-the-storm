"""
Experiment: Mildly Conservative Q-Learning (MCQ) for Draft Selection.

Tests whether targeted pessimism avoids CQL's collapse to behavioral cloning.
MCQ only penalizes Q-values above a threshold, preserving learned preferences
for well-represented state-action pairs.

Variant 2: BC-regularized CQL adds a behavioral cloning term to anchor
Q-values near the data distribution without suppressing all exploration.

Key question: Does MCQ produce positive counter/synergy while maintaining
low degenerate rate?

Usage:
    set -a && source .env && set +a
    python3 -u training/experiment_mcq_draft.py --sweep
    python3 -u training/experiment_mcq_draft.py --method mcq --threshold 0.6
    python3 -u training/experiment_mcq_draft.py --method bc_cql --bc-weight 0.5
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
    is_degenerate,
    NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS,
    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
    load_replay_data, split_data,
    HERO_ROLE_FINE,
)
from train_draft_policy import DraftState, DRAFT_ORDER
from experiment_cql_draft import CQLDraftAgent, replay_to_transitions, CQLDataset
from sweep_enriched_wp import StatsCache

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "experiment_results", "mcq")


def train_mcq(train_transitions, test_transitions, threshold=0.6,
              alpha=1.0, lr=3e-4, epochs=50, batch_size=2048, device=None, seed=42):
    """Train MCQ agent. Penalizes only Q-values exceeding threshold."""
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
    tau = 0.01

    best_test_loss = float('inf')
    save_path = os.path.join(RESULTS_DIR, f"_mcq_temp_t{threshold}.pt")
    patience = 0

    for epoch in range(epochs):
        model.train()
        total_bellman = 0
        total_mcq = 0
        total_n = 0

        for states, actions, outcomes, masks in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            outcomes = outcomes.to(device)
            masks = masks.to(device)

            q_bounded = model.q_values(states, masks)
            q_taken = q_bounded.gather(1, actions.unsqueeze(1)).squeeze(1)
            bellman_loss = F.binary_cross_entropy(q_taken, outcomes)

            # MCQ penalty: only penalize Q-values above threshold
            q_all = q_bounded  # [B, NUM_HEROES]
            excess = F.relu(q_all - threshold)  # zero for Q below threshold
            mcq_penalty = excess.sum(dim=1).mean() - F.relu(q_taken - threshold).mean()

            loss = bellman_loss + alpha * mcq_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for p, tp in zip(model.parameters(), target_model.parameters()):
                    tp.data.mul_(1 - tau).add_(p.data * tau)

            total_bellman += bellman_loss.item() * len(states)
            total_mcq += mcq_penalty.item() * len(states)
            total_n += len(states)

        model.eval()
        test_loss = 0
        test_total = 0
        test_correct = 0
        with torch.no_grad():
            for states, actions, outcomes, masks in test_loader:
                states = states.to(device)
                actions = actions.to(device)
                outcomes = outcomes.to(device)
                masks = masks.to(device)
                q_bounded = model.q_values(states, masks)
                q_taken = q_bounded.gather(1, actions.unsqueeze(1)).squeeze(1)
                test_loss += F.binary_cross_entropy(q_taken, outcomes, reduction='sum').item()
                predicted = q_bounded.argmax(dim=1)
                test_correct += (predicted == actions).sum().item()
                test_total += len(states)

        avg_test_loss = test_loss / test_total
        test_acc = test_correct / test_total * 100

        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1}: bellman={total_bellman/total_n:.4f} "
                  f"mcq={total_mcq/total_n:.4f} test_loss={avg_test_loss:.4f} "
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


def train_bc_cql(train_transitions, test_transitions, alpha=1.0, bc_weight=0.5,
                 lr=3e-4, epochs=50, batch_size=2048, device=None, seed=42):
    """Train BC-regularized CQL: CQL penalty + behavioral cloning loss."""
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
    tau = 0.01

    best_test_loss = float('inf')
    save_path = os.path.join(RESULTS_DIR, f"_bc_cql_temp_bc{bc_weight}.pt")
    patience = 0

    for epoch in range(epochs):
        model.train()
        total_bellman = 0
        total_cql = 0
        total_bc = 0
        total_n = 0

        for states, actions, outcomes, masks in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            outcomes = outcomes.to(device)
            masks = masks.to(device)

            q_bounded = model.q_values(states, masks)
            q_taken = q_bounded.gather(1, actions.unsqueeze(1)).squeeze(1)
            bellman_loss = F.binary_cross_entropy(q_taken, outcomes)

            # Standard CQL penalty
            q_raw = model(states, masks)
            logsumexp_q = torch.logsumexp(q_raw, dim=1).mean()
            data_q = q_raw.gather(1, actions.unsqueeze(1)).squeeze(1).mean()
            cql_penalty = logsumexp_q - data_q

            # BC loss: cross-entropy on Q-value distribution vs actual actions
            log_probs = F.log_softmax(q_raw, dim=1)
            bc_loss = F.nll_loss(log_probs, actions)

            loss = bellman_loss + alpha * cql_penalty + bc_weight * bc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for p, tp in zip(model.parameters(), target_model.parameters()):
                    tp.data.mul_(1 - tau).add_(p.data * tau)

            total_bellman += bellman_loss.item() * len(states)
            total_cql += cql_penalty.item() * len(states)
            total_bc += bc_loss.item() * len(states)
            total_n += len(states)

        model.eval()
        test_loss = 0
        test_total = 0
        test_correct = 0
        with torch.no_grad():
            for states, actions, outcomes, masks in test_loader:
                states = states.to(device)
                actions = actions.to(device)
                outcomes = outcomes.to(device)
                masks = masks.to(device)
                q_bounded = model.q_values(states, masks)
                q_taken = q_bounded.gather(1, actions.unsqueeze(1)).squeeze(1)
                test_loss += F.binary_cross_entropy(q_taken, outcomes, reduction='sum').item()
                predicted = q_bounded.argmax(dim=1)
                test_correct += (predicted == actions).sum().item()
                test_total += len(states)

        avg_test_loss = test_loss / test_total
        test_acc = test_correct / test_total * 100

        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1}: bellman={total_bellman/total_n:.4f} "
                  f"cql={total_cql/total_n:.4f} bc={total_bc/total_n:.4f} "
                  f"test_loss={avg_test_loss:.4f} action_match={test_acc:.1f}%")

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


# ── Rich evaluation (Table VII metrics) ──

def counter_responsiveness(our_heroes, opp_heroes, stats, tier):
    deltas = []
    for our_h in our_heroes:
        our_wr = stats.get_hero_wr(our_h, tier)
        for opp_h in opp_heroes:
            raw = stats.get_counter(our_h, opp_h, tier)
            if raw is None:
                continue
            opp_wr = stats.get_hero_wr(opp_h, tier)
            expected = our_wr + (100 - opp_wr) - 50
            deltas.append(raw - expected)
    return np.mean(deltas) if deltas else 0.0


def synergy_exploitation(our_heroes, stats, tier):
    deltas = []
    for i, h1 in enumerate(our_heroes):
        wr1 = stats.get_hero_wr(h1, tier)
        for h2 in our_heroes[i+1:]:
            raw = stats.get_synergy(h1, h2, tier)
            if raw is None:
                continue
            wr2 = stats.get_hero_wr(h2, tier)
            expected = 50 + (wr1 - 50) + (wr2 - 50)
            deltas.append(raw - expected)
    return np.mean(deltas) if deltas else 0.0


def draft_diversity(drafts):
    all_heroes = []
    for d in drafts:
        all_heroes.extend(d["our_picks"])
    counts = Counter(all_heroes)
    distinct = len(counts)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    sorted_counts = sorted(counts.values(), reverse=True)
    top10 = sum(sorted_counts[:10]) / total * 100
    return {
        "distinct_heroes": distinct,
        "entropy": round(entropy, 2),
        "top10_concentration": round(top10, 1),
    }


def run_drafts_rich(model, device, draft_configs, gd_models, stats):
    """Run drafts with CQL/MCQ model and collect rich data for metrics."""
    from train_generic_draft import GenericDraftModel

    healer_heroes = set(h for h, r in HERO_ROLE_FINE.items() if r == "healer")

    all_drafts = []
    for di, (_, game_map, tier, our_team) in enumerate(draft_configs):
        state = DraftState(game_map, tier, our_team=our_team)
        steps = []

        while not state.is_terminal():
            step_team, step_type = DRAFT_ORDER[state.step]

            state_vec = np.concatenate([
                state.team0_picks.copy(), state.team1_picks.copy(), state.bans.copy(),
                map_to_one_hot(game_map), tier_to_one_hot(tier),
                [state.step / 15.0, 0.0 if step_type == "ban" else 1.0],
            ])
            mask_vec = state.valid_mask_np()

            if step_team == our_team:
                s_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(device)
                m_t = torch.tensor(mask_vec, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q = model(s_t, m_t).squeeze(0).cpu()
                hero_idx = q.argmax().item()
                steps.append({
                    "step": state.step, "team": step_team, "type": step_type,
                    "hero_idx": hero_idx, "state": state_vec, "mask": mask_vec,
                    "is_ours": True,
                })
                state.apply_action(hero_idx, step_team, step_type)
            else:
                gd = random.choice(gd_models)
                x = state.to_tensor_gd(torch.device("cpu"))
                mask = state.valid_mask(torch.device("cpu"))
                with torch.no_grad():
                    logits = gd(x, mask)
                    probs = F.softmax(logits / 1.0, dim=1)
                    hero_idx = torch.multinomial(probs, 1).item()
                steps.append({
                    "step": state.step, "team": step_team, "type": step_type,
                    "hero_idx": hero_idx, "state": state_vec, "mask": mask_vec,
                    "is_ours": False,
                })
                state.apply_action(hero_idx, step_team, step_type)

        our_vec = state.team0_picks if our_team == 0 else state.team1_picks
        opp_vec = state.team1_picks if our_team == 0 else state.team0_picks
        our_picks = [HEROES[i] for i in range(NUM_HEROES) if our_vec[i] > 0]
        opp_picks = [HEROES[i] for i in range(NUM_HEROES) if opp_vec[i] > 0]

        all_drafts.append({
            "game_map": game_map, "tier": tier, "our_team": our_team,
            "our_picks": our_picks, "opp_picks": opp_picks,
            "steps": steps,
            "has_healer": any(h in healer_heroes for h in our_picks),
            "is_degen": is_degenerate(our_picks),
        })

        if (di + 1) % 50 == 0:
            n = di + 1
            hr = sum(1 for d in all_drafts if d["has_healer"]) / n * 100
            dr = sum(1 for d in all_drafts if d["is_degen"]) / n * 100
            print(f"    {n}/{len(draft_configs)}: healer={hr:.1f}% degen={dr:.1f}%")

    return all_drafts


def compute_rich_metrics(drafts, gd_models, stats):
    """Compute full Table VII metrics from draft data."""
    n = len(drafts)
    healer_rate = sum(1 for d in drafts if d["has_healer"]) / n * 100
    degen_rate = sum(1 for d in drafts if d["is_degen"]) / n * 100

    counters = [counter_responsiveness(d["our_picks"], d["opp_picks"], stats, d["tier"])
                for d in drafts]
    avg_counter = np.mean(counters)

    synergies = [synergy_exploitation(d["our_picks"], stats, d["tier"])
                 for d in drafts]
    avg_synergy = np.mean(synergies)

    div = draft_diversity(drafts)

    # GD agreement
    gd_agree = 0
    gd_total = 0
    for draft in drafts:
        for step in draft["steps"]:
            if not step["is_ours"] or step["type"] == "ban":
                continue
            s_t = torch.tensor(step["state"], dtype=torch.float32).unsqueeze(0)
            m_t = torch.tensor(step["mask"], dtype=torch.float32).unsqueeze(0)
            votes = Counter()
            for gd in gd_models:
                with torch.no_grad():
                    logits = gd(s_t, m_t)
                    votes[logits.argmax(dim=1).item()] += 1
            gd_consensus = votes.most_common(1)[0][0]
            if step["hero_idx"] == gd_consensus:
                gd_agree += 1
            gd_total += 1
    gd_sim = gd_agree / gd_total * 100 if gd_total > 0 else 0

    return {
        "healer_rate": round(healer_rate, 1),
        "degen_rate": round(degen_rate, 1),
        "counter": round(float(avg_counter), 2),
        "synergy": round(float(avg_synergy), 2),
        **div,
        "gd_similarity": round(gd_sim, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="mcq", choices=["mcq", "bc_cql"])
    parser.add_argument("--threshold", type=float, default=0.6, help="MCQ threshold")
    parser.add_argument("--bc-weight", type=float, default=0.5, help="BC regularization weight")
    parser.add_argument("--alpha", type=float, default=1.0, help="CQL/MCQ penalty weight")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--drafts", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading replay data...")
    data = load_replay_data()
    train_data, test_data = split_data(data)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

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

    # Load GD models
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

    # Load stats for rich metrics
    stats = StatsCache()

    random.seed(42)
    draft_configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
                     for i in range(args.drafts)]

    # Build sweep configs
    if args.sweep:
        configs = []
        for threshold in [0.4, 0.5, 0.6, 0.7, 0.8]:
            configs.append({"method": "mcq", "threshold": threshold, "alpha": 1.0, "bc_weight": 0})
        for bc_weight in [0.1, 0.5, 1.0, 2.0]:
            configs.append({"method": "bc_cql", "threshold": 0, "alpha": 1.0, "bc_weight": bc_weight})
    elif args.method == "mcq":
        configs = [{"method": "mcq", "threshold": args.threshold, "alpha": args.alpha, "bc_weight": 0}]
    else:
        configs = [{"method": "bc_cql", "threshold": 0, "alpha": args.alpha, "bc_weight": args.bc_weight}]

    all_results = []
    for ci, cfg in enumerate(configs):
        method = cfg["method"]
        name = f"MCQ τ={cfg['threshold']}" if method == "mcq" else f"BC-CQL β={cfg['bc_weight']}"
        print(f"\n{'='*60}")
        print(f"  [{ci+1}/{len(configs)}] {name} (α={cfg['alpha']})")
        print(f"{'='*60}")

        if method == "mcq":
            model, metrics = train_mcq(
                train_transitions, test_transitions,
                threshold=cfg["threshold"], alpha=cfg["alpha"],
                lr=3e-4, epochs=args.epochs, batch_size=2048, device=device,
            )
        else:
            model, metrics = train_bc_cql(
                train_transitions, test_transitions,
                alpha=cfg["alpha"], bc_weight=cfg["bc_weight"],
                lr=3e-4, epochs=args.epochs, batch_size=2048, device=device,
            )

        print(f"  Training done: {metrics}")
        print(f"\n  Running {args.drafts} drafts with rich evaluation...")
        random.seed(42)
        drafts = run_drafts_rich(model, device, draft_configs, gd_models, stats)
        rich = compute_rich_metrics(drafts, gd_models, stats)
        print(f"  Rich metrics: {rich}")

        all_results.append({
            "name": name,
            "config": cfg,
            "train_metrics": metrics,
            "rich_metrics": rich,
        })

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*110}")
    print("COMPREHENSIVE EVALUATION (Table VII format)")
    print(f"{'='*110}")
    print(f"{'Method':<18} {'Heal%':>6} {'Deg%':>6} {'Counter':>8} {'Synergy':>8} "
          f"{'Distinct':>8} {'Entropy':>8} {'Top10%':>7} {'GD%':>6} {'ActMtch':>8}")
    print("-" * 110)

    # Reference baselines from existing results
    print(f"{'GD baseline':<18} {'97.5':>6} {'8.0':>6} {'+0.32':>8} {'+0.14':>8} "
          f"{'57':>8} {'4.93':>8} {'38.3':>7} {'100':>6} {'---':>8}")
    print(f"{'CQL α=1.0':<18} {'91.0':>6} {'9.0':>6} {'-0.18':>8} {'-0.21':>8} "
          f"{'31':>8} {'4.29':>8} {'54.2':>7} {'62':>6} {'---':>8}")
    print(f"{'Enrich+aug':<18} {'94.5':>6} {'26.5':>6} {'+0.41':>8} {'+0.18':>8} "
          f"{'62':>8} {'5.14':>8} {'33.2':>7} {'55':>6} {'---':>8}")
    print("-" * 110)

    for r in all_results:
        m = r["rich_metrics"]
        tm = r["train_metrics"]
        counter_str = f"{m['counter']:+.2f}" if m['counter'] != 0 else "+0.00"
        synergy_str = f"{m['synergy']:+.2f}" if m['synergy'] != 0 else "+0.00"
        print(f"{r['name']:<18} {m['healer_rate']:>6.1f} {m['degen_rate']:>6.1f} "
              f"{counter_str:>8} {synergy_str:>8} "
              f"{m['distinct_heroes']:>8} {m['entropy']:>8.2f} "
              f"{m['top10_concentration']:>7.1f} {m['gd_similarity']:>6.1f} "
              f"{tm['action_match']:>7.1f}%")

    # Save
    save_path = os.path.join(RESULTS_DIR, "mcq_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
