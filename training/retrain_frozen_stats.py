#!/usr/bin/env python3
"""Retrain all models that depend on heroesprofile stats using frozen snapshot.

Retrains:
  1. Enriched WP (256→128) → wp_experiment_enriched.pt
  2. Enriched+aug WP (512→256→128) → wp_enriched_winner.pt
  3. CQL enriched α=0.5, 2.0 → experiment_results/cql/_cql_enriched_a{alpha}.pt

Usage:
    set -a && source .env && set +a
    python3 -u training/retrain_frozen_stats.py
"""
import os, sys, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import (
    NUM_HEROES, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS,
    load_replay_data, split_data, HERO_ROLE_FINE,
)
from sweep_enriched_wp import (
    StatsCache, WinProbEnrichedModel, FEATURE_GROUPS, FEATURE_GROUP_DIMS,
    compute_group_indices, precompute_all_features, extract_features,
)
from experiment_synthetic_augmentation import ENRICHED_GROUPS, generate_synthetic_data_v2
from experiment_cql_draft import CQLDraftAgent, CQLDataset
from experiment_cql_enriched import replay_to_enriched_transitions, get_enriched_cols

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Enriched feature columns
_gi = compute_group_indices()
WP_COLS = []
for g in ENRICHED_GROUPS:
    s, e = _gi[g]
    WP_COLS.extend(range(s, e))
WP_INPUT_DIM = 197 + sum(FEATURE_GROUP_DIMS[g] for g in ENRICHED_GROUPS)
ALL_MASK = [True] * len(FEATURE_GROUPS)


def train_wp_model(model, train_X, test_X, train_y, test_y, name, device,
                   lr=5e-4, epochs=200, patience=25):
    criterion = nn.BCELoss()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
    best_loss, best_acc, best_state = float('inf'), 0, None
    pat = 0
    n = len(train_X)
    for ep in range(epochs):
        model.train()
        pm = torch.randperm(n, device=device)
        for i in range(0, n, 4096):
            idx = pm[i:i + 4096]
            loss = criterion(model(train_X[idx]), train_y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            tp = model(test_X)
            tl = criterion(tp, test_y).item()
            ta = ((tp > 0.5).float() == test_y).float().mean().item() * 100
        if tl < best_loss:
            best_loss, best_acc = tl, ta
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                break
        if (ep + 1) % 20 == 0:
            print(f'  {name} ep{ep+1}: acc={ta:.2f}% best={best_acc:.2f}%')
    model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


def train_cql_enriched(train_transitions, test_transitions, input_dim, alpha,
                       device, lr=3e-4, epochs=50, batch_size=2048, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds = CQLDataset(train_transitions)
    test_ds = CQLDataset(test_transitions)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = CQLDraftAgent(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_test_loss = float('inf')
    save_dir = os.path.join(SCRIPT_DIR, "experiment_results", "cql")
    save_path = os.path.join(save_dir, f"_cql_enriched_a{alpha}.pt")
    patience = 0

    for epoch in range(epochs):
        model.train()
        total_bellman = total_cql = total_n = 0
        for states, actions, outcomes, masks in train_loader:
            states, actions = states.to(device), actions.to(device)
            outcomes, masks = outcomes.to(device), masks.to(device)
            q_bounded = model.q_values(states, masks)
            q_taken = q_bounded.gather(1, actions.unsqueeze(1)).squeeze(1)
            bellman_loss = F.binary_cross_entropy(q_taken, outcomes)
            q_raw = model(states, masks)
            logsumexp_q = torch.logsumexp(q_raw, dim=1).mean()
            data_q = q_raw.gather(1, actions.unsqueeze(1)).squeeze(1).mean()
            cql_penalty = logsumexp_q - data_q
            loss = bellman_loss + alpha * cql_penalty
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_bellman += bellman_loss.item() * len(states)
            total_cql += cql_penalty.item() * len(states)
            total_n += len(states)

        model.eval()
        test_loss = test_total = 0
        with torch.no_grad():
            for states, actions, outcomes, masks in test_loader:
                states, actions = states.to(device), actions.to(device)
                outcomes, masks = outcomes.to(device), masks.to(device)
                q_bounded = model.q_values(states, masks)
                q_taken = q_bounded.gather(1, actions.unsqueeze(1)).squeeze(1)
                test_loss += F.binary_cross_entropy(q_taken, outcomes, reduction='sum').item()
                test_total += len(states)

        avg_test_loss = test_loss / test_total
        if epoch % 5 == 0:
            print(f"  CQL enriched a={alpha} ep{epoch+1}: bellman={total_bellman/total_n:.4f} "
                  f"cql={total_cql/total_n:.4f} test={avg_test_loss:.4f}")
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
    return model, best_test_loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading replay data...")
    data = load_replay_data()
    train_data, test_data = split_data(data, test_frac=0.15)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    print("Loading frozen stats...")
    stats = StatsCache()

    # ═══════════════════════════════════════════════════════════════
    # 1. Enriched WP (256→128)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  ENRICHED WP (256→128)")
    print("="*60)
    cache_dir = os.path.join(SCRIPT_DIR, "feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()
    train_base, train_enriched, train_labels = precompute_all_features(
        train_data, stats, cache_path=os.path.join(cache_dir, "train_features.npz"))
    test_base, test_enriched, test_labels = precompute_all_features(
        test_data, stats, cache_path=os.path.join(cache_dir, "test_features.npz"))
    print(f"  Features computed in {time.time()-t0:.1f}s")

    train_X_e = torch.cat([train_base, train_enriched[:, WP_COLS]], dim=1).to(device)
    test_X_e = torch.cat([test_base, test_enriched[:, WP_COLS]], dim=1).to(device)
    train_y = train_labels.to(device)
    test_y = test_labels.to(device)

    model_enr, acc_enr = train_wp_model(
        WinProbEnrichedModel(WP_INPUT_DIM, [256, 128], dropout=0.3),
        train_X_e, test_X_e, train_y, test_y, 'Enriched-256', device)
    save_path = os.path.join(SCRIPT_DIR, 'wp_experiment_enriched.pt')
    torch.save(model_enr.cpu().state_dict(), save_path)
    print(f"Enriched WP: {acc_enr:.2f}% → {save_path}")

    # ═══════════════════════════════════════════════════════════════
    # 2. Enriched+aug WP (512→256→128)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  ENRICHED+AUG WP (512→256→128)")
    print("="*60)
    comp_data = json.load(open(os.path.join(SCRIPT_DIR, '..', 'src', 'lib', 'data', 'compositions.json')))
    syn, _ = generate_synthetic_data_v2(train_data, comp_data, stats, unseen_wr=10.0, unseen_volume=100)
    sb, se, sl = [], [], []
    for d in syn:
        try:
            b, e = extract_features(d, stats, ALL_MASK)
            sb.append(b); se.append(e[WP_COLS]); sl.append(float(d['winner'] == 0))
        except:
            continue
    sX = torch.cat([torch.tensor(np.array(sb, dtype=np.float32)),
                    torch.tensor(np.array(se, dtype=np.float32))], dim=1).to(device)
    sy = torch.tensor(np.array(sl, dtype=np.float32)).to(device)
    aX = torch.cat([train_X_e, sX])
    ay = torch.cat([train_y, sy])
    pm = torch.randperm(len(aX)); aX = aX[pm]; ay = ay[pm]
    print(f"  Training data: {len(train_X_e)} real + {len(sX)} synthetic = {len(aX)} total")

    model_aug, acc_aug = train_wp_model(
        WinProbEnrichedModel(WP_INPUT_DIM, [512, 256, 128], dropout=0.3),
        aX, test_X_e, ay, test_y, 'Aug-512', device)
    save_path = os.path.join(SCRIPT_DIR, 'wp_enriched_winner.pt')
    torch.save(model_aug.cpu().state_dict(), save_path)
    print(f"Enriched+aug WP: {acc_aug:.2f}% → {save_path}")

    # ═══════════════════════════════════════════════════════════════
    # 3. CQL enriched (α=0.5, 2.0)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  CQL ENRICHED")
    print("="*60)
    group_indices = compute_group_indices()
    enriched_cols = get_enriched_cols(group_indices)
    enriched_dim = len(enriched_cols)
    all_mask = [True] * len(FEATURE_GROUPS)
    input_dim = 289 + enriched_dim

    print(f"Building enriched CQL transitions (input_dim={input_dim})...")
    t0 = time.time()
    train_transitions, test_transitions = [], []
    for d in train_data:
        t = replay_to_enriched_transitions(d, stats, group_indices, enriched_cols, all_mask)
        if t: train_transitions.extend(t)
    for d in test_data:
        t = replay_to_enriched_transitions(d, stats, group_indices, enriched_cols, all_mask)
        if t: test_transitions.extend(t)
    print(f"  {len(train_transitions):,} train, {len(test_transitions):,} test in {time.time()-t0:.0f}s")

    for alpha in [0.5, 2.0]:
        print(f"\n  Training CQL enriched α={alpha}...")
        model, test_loss = train_cql_enriched(
            train_transitions, test_transitions, input_dim, alpha, device)
        print(f"  CQL enriched α={alpha}: test_loss={test_loss:.4f}")

    print("\n" + "="*60)
    print("  ALL RETRAINING COMPLETE")
    print("="*60)
    print(f"  Enriched WP (256→128): {acc_enr:.2f}%")
    print(f"  Enriched+aug WP (512→256→128): {acc_aug:.2f}%")
    print(f"  CQL enriched α=0.5, 2.0: retrained")


if __name__ == '__main__':
    main()
