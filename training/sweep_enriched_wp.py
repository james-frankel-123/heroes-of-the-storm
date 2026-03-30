"""
Enriched Win Probability model sweep.

Tests all 1024 subsets of 10 feature groups × 2 MLP sizes × 2 LRs = 4096 variants.
Each feature group can be toggled on/off independently to find which combinations
of input features best predict win probability.

Usage:
    set -a && source .env && set +a
    python training/sweep_enriched_wp.py                       # full sweep (4096 variants)
    python training/sweep_enriched_wp.py --variants 32         # quick test
    python training/sweep_enriched_wp.py --workers-per-gpu 64  # tune concurrency
"""
import os
import sys
import csv
import json
import itertools
import argparse
import time
import filelock
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    NUM_HEROES, NUM_MAPS, NUM_TIERS, HEROES, HERO_TO_IDX, MAPS, SKILL_TIERS,
    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
    load_replay_data, split_data,
    HERO_ROLE_FINE, FINE_ROLE_NAMES, FINE_ROLE_TO_IDX, TWO_LANE_MAPS,
)
from archive.hero_capabilities import HERO_CAPABILITIES, CAPABILITY_DIMS

INPUT_DIM_BASE = NUM_HEROES * 2 + NUM_MAPS + NUM_TIERS  # 197

# ── Feature group definitions ──

FEATURE_GROUPS = [
    "map_type",          # 1 feature
    "role_counts",       # 18 features (9 per team)
    "hero_wr",           # 10 features
    "team_avg_wr",       # 2 features
    "hero_map_wr",       # 10 features
    "map_delta",         # 2 features
    "pairwise_counters", # 2 features
    "pairwise_synergies",# 2 features
    "counter_detail",    # 50 features
    "synergy_detail",    # 20 features
    "capabilities",      # 32 features (16 dims × 2 teams)
    "meta_strength",     # 4 features (avg pick_rate + avg ban_rate per team)
    "draft_diversity",   # 2 features (std dev of hero WRs within each team)
    "avg_mmr",           # 1 feature (normalized match MMR)
    "comp_wr",           # 4 features (composition WR + log_games per team from HP data)
]

NUM_CAPABILITY_DIMS = len(CAPABILITY_DIMS)

FEATURE_GROUP_DIMS = {
    "map_type": 1,
    "role_counts": 18,  # 9 roles × 2 teams
    "hero_wr": 10,
    "team_avg_wr": 2,
    "hero_map_wr": 10,
    "map_delta": 2,
    "pairwise_counters": 2,
    "pairwise_synergies": 2,
    "counter_detail": 50,
    "synergy_detail": 20,
    "capabilities": NUM_CAPABILITY_DIMS * 2,  # per team
    "meta_strength": 4,   # avg pick_rate + avg ban_rate per team
    "draft_diversity": 2, # std dev of hero WRs per team
    "avg_mmr": 1,         # normalized match MMR
    "comp_wr": 4,         # SL composition WR + log_games per team
}

# ── Stats cache ──

class StatsCache:
    """Preloaded stats from the database."""
    def __init__(self):
        import psycopg2
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
            if os.path.exists(env_path):
                for line in open(env_path):
                    if line.startswith("DATABASE_URL="):
                        db_url = line.split("=", 1)[1].strip().strip('"')
                        break
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        # Hero WR by tier: {tier: {hero: wr}}
        self.hero_wr = {}
        # Hero pick/ban rates: {tier: {hero: (pick_rate, ban_rate)}}
        self.hero_meta = {}
        cur.execute("SELECT hero, win_rate, pick_rate, ban_rate, games, skill_tier FROM hero_stats_aggregate")
        for hero, wr, pr, br, games, tier in cur.fetchall():
            self.hero_wr.setdefault(tier, {})[hero] = wr
            self.hero_meta.setdefault(tier, {})[hero] = (pr or 0, br or 0)

        # Hero-map WR: {tier: {map: {hero: (wr, games)}}}
        self.hero_map_wr = {}
        cur.execute("SELECT hero, map, win_rate, games, skill_tier FROM hero_map_stats_aggregate")
        for hero, map_name, wr, games, tier in cur.fetchall():
            self.hero_map_wr.setdefault(tier, {}).setdefault(map_name, {})[hero] = (wr, games)

        # Pairwise: {tier: {rel: {heroA: {heroB: (wr, games)}}}}
        self.pairwise = {}
        cur.execute("SELECT hero_a, hero_b, relationship, win_rate, games, skill_tier FROM hero_pairwise_stats")
        for ha, hb, rel, wr, games, tier in cur.fetchall():
            self.pairwise.setdefault(tier, {}).setdefault(rel, {}).setdefault(ha, {})[hb] = (wr, games)

        cur.close()
        conn.close()

        # Load composition data from JSON file
        # {tier: [{roles: [...], winRate: float, games: int}, ...]}
        self.comp_data = {}  # {tier: {role_key: (wr, games)}}
        comp_path = os.path.join(os.path.dirname(__file__), "..", "src", "lib", "data", "compositions.json")
        if os.path.exists(comp_path):
            import json as _json
            raw = _json.load(open(comp_path))
            for tier, comps in raw.items():
                tier_map = {}
                for c in comps:
                    key = ",".join(sorted(c["roles"]))
                    tier_map[key] = (c["winRate"], c["games"])
                self.comp_data[tier] = tier_map

    def get_comp_wr(self, heroes, tier):
        """Look up composition WR from HP data given a list of hero names.
        Returns (winRate, games) or (33.0, 0) if not found (unknown comp = bad)."""
        roles = sorted([HERO_ROLE_FINE.get(h, "unknown") for h in heroes])
        # Map fine roles to official HP roles
        hp_role_map = {
            "tank": "Tank", "bruiser": "Bruiser", "healer": "Healer",
            "ranged_aa": "Ranged Assassin", "ranged_mage": "Ranged Assassin",
            "melee_assassin": "Melee Assassin", "support_utility": "Support",
            "varian": "Bruiser", "pusher": "Ranged Assassin", "unknown": "Ranged Assassin",
        }
        hp_roles = sorted([hp_role_map.get(r, "Ranged Assassin") for r in roles])
        key = ",".join(hp_roles)
        tier_data = self.comp_data.get(tier, {})
        if key in tier_data:
            return tier_data[key]
        # Fallback: try without tier
        for t in ["mid", "high", "low"]:
            if t in self.comp_data and key in self.comp_data[t]:
                return self.comp_data[t][key]
        return (33.0, 0)  # unknown comp = assert 33% WR per your spec

    def get_hero_wr(self, hero, tier):
        return self.hero_wr.get(tier, {}).get(hero, 50.0)

    def get_hero_map_wr(self, hero, game_map, tier):
        data = self.hero_map_wr.get(tier, {}).get(game_map, {}).get(hero)
        if data and data[1] >= 50:
            return data[0]
        return self.get_hero_wr(hero, tier)

    def get_counter(self, heroA, heroB, tier):
        data = self.pairwise.get(tier, {}).get("against", {}).get(heroA, {}).get(heroB)
        if data and data[1] >= 30:
            return data[0]
        return None

    def get_synergy(self, heroA, heroB, tier):
        data = self.pairwise.get(tier, {}).get("with", {}).get(heroA, {}).get(heroB)
        if data and data[1] >= 30:
            return data[0]
        return None


# ── Feature extraction ──

def extract_features(d, stats, groups_mask):
    """Extract all enabled feature groups for one sample.
    Returns (base_features, enriched_features) as numpy arrays.
    d: replay dict with team0_heroes, team1_heroes, game_map, skill_tier, winner
    groups_mask: list of bools, one per FEATURE_GROUPS entry
    """
    t0_heroes = d["team0_heroes"]
    t1_heroes = d["team1_heroes"]
    game_map = d["game_map"]
    tier = d["skill_tier"]

    # Base features (always on)
    t0_mh = heroes_to_multi_hot(t0_heroes)
    t1_mh = heroes_to_multi_hot(t1_heroes)
    map_oh = map_to_one_hot(game_map)
    tier_oh = tier_to_one_hot(tier)
    base = np.concatenate([t0_mh, t1_mh, map_oh, tier_oh])

    enriched_parts = []

    # Sort hero indices for deterministic ordering
    t0_sorted = sorted([HERO_TO_IDX.get(h, 0) for h in t0_heroes])
    t1_sorted = sorted([HERO_TO_IDX.get(h, 0) for h in t1_heroes])

    # Pre-compute hero WRs (needed by multiple groups)
    t0_wrs = [stats.get_hero_wr(h, tier) for h in t0_heroes]
    t1_wrs = [stats.get_hero_wr(h, tier) for h in t1_heroes]

    for i, group in enumerate(FEATURE_GROUPS):
        if not groups_mask[i]:
            continue

        if group == "map_type":
            enriched_parts.append(np.array([1.0 if game_map in TWO_LANE_MAPS else 0.0], dtype=np.float32))

        elif group == "role_counts":
            n_roles = len(FINE_ROLE_NAMES)
            t0_roles = np.zeros(n_roles, dtype=np.float32)
            t1_roles = np.zeros(n_roles, dtype=np.float32)
            for h in t0_heroes:
                role = HERO_ROLE_FINE.get(h)
                if role:
                    t0_roles[FINE_ROLE_TO_IDX[role]] += 1
            for h in t1_heroes:
                role = HERO_ROLE_FINE.get(h)
                if role:
                    t1_roles[FINE_ROLE_TO_IDX[role]] += 1
            enriched_parts.append(np.concatenate([t0_roles, t1_roles]))

        elif group == "hero_wr":
            # Per-hero WR, sorted by hero index within each team
            t0_wr_sorted = [stats.get_hero_wr(HEROES[idx], tier) for idx in t0_sorted]
            t1_wr_sorted = [stats.get_hero_wr(HEROES[idx], tier) for idx in t1_sorted]
            # Pad to 5 if needed
            while len(t0_wr_sorted) < 5: t0_wr_sorted.append(50.0)
            while len(t1_wr_sorted) < 5: t1_wr_sorted.append(50.0)
            enriched_parts.append(np.array(t0_wr_sorted[:5] + t1_wr_sorted[:5], dtype=np.float32))

        elif group == "team_avg_wr":
            t0_avg = np.mean(t0_wrs) if t0_wrs else 50.0
            t1_avg = np.mean(t1_wrs) if t1_wrs else 50.0
            enriched_parts.append(np.array([t0_avg, t1_avg], dtype=np.float32))

        elif group == "hero_map_wr":
            t0_map_wr = [stats.get_hero_map_wr(HEROES[idx], game_map, tier) for idx in t0_sorted]
            t1_map_wr = [stats.get_hero_map_wr(HEROES[idx], game_map, tier) for idx in t1_sorted]
            while len(t0_map_wr) < 5: t0_map_wr.append(50.0)
            while len(t1_map_wr) < 5: t1_map_wr.append(50.0)
            enriched_parts.append(np.array(t0_map_wr[:5] + t1_map_wr[:5], dtype=np.float32))

        elif group == "map_delta":
            t0_delta = sum(stats.get_hero_map_wr(h, game_map, tier) - stats.get_hero_wr(h, tier)
                          for h in t0_heroes)
            t1_delta = sum(stats.get_hero_map_wr(h, game_map, tier) - stats.get_hero_wr(h, tier)
                          for h in t1_heroes)
            enriched_parts.append(np.array([t0_delta, t1_delta], dtype=np.float32))

        elif group == "pairwise_counters":
            # Average normalized counter delta: each team vs the other
            t0_counter = _avg_counter_delta(t0_heroes, t1_heroes, t0_wrs, t1_wrs, stats, tier)
            t1_counter = _avg_counter_delta(t1_heroes, t0_heroes, t1_wrs, t0_wrs, stats, tier)
            enriched_parts.append(np.array([t0_counter, t1_counter], dtype=np.float32))

        elif group == "pairwise_synergies":
            t0_syn = _avg_synergy_delta(t0_heroes, t0_wrs, stats, tier)
            t1_syn = _avg_synergy_delta(t1_heroes, t1_wrs, stats, tier)
            enriched_parts.append(np.array([t0_syn, t1_syn], dtype=np.float32))

        elif group == "counter_detail":
            # 5×5×2 = 50 features: every cross-team pair for both teams
            detail = []
            for our_h, our_wr in zip(t0_heroes, t0_wrs):
                for their_h, their_wr in zip(t1_heroes, t1_wrs):
                    detail.append(_normalized_counter(our_h, their_h, our_wr, their_wr, stats, tier))
            for their_h, their_wr in zip(t1_heroes, t1_wrs):
                for our_h, our_wr in zip(t0_heroes, t0_wrs):
                    detail.append(_normalized_counter(their_h, our_h, their_wr, our_wr, stats, tier))
            while len(detail) < 50: detail.append(0.0)
            enriched_parts.append(np.array(detail[:50], dtype=np.float32))

        elif group == "synergy_detail":
            # C(5,2)×2 = 20 features
            detail = []
            for j in range(len(t0_heroes)):
                for k in range(j+1, len(t0_heroes)):
                    detail.append(_normalized_synergy(t0_heroes[j], t0_heroes[k],
                                                      t0_wrs[j], t0_wrs[k], stats, tier))
            while len(detail) < 10: detail.append(0.0)
            for j in range(len(t1_heroes)):
                for k in range(j+1, len(t1_heroes)):
                    detail.append(_normalized_synergy(t1_heroes[j], t1_heroes[k],
                                                      t1_wrs[j], t1_wrs[k], stats, tier))
            while len(detail) < 20: detail.append(0.0)
            enriched_parts.append(np.array(detail[:20], dtype=np.float32))

        elif group == "capabilities":
            # Sum each capability dimension per team
            t0_caps = np.zeros(NUM_CAPABILITY_DIMS, dtype=np.float32)
            t1_caps = np.zeros(NUM_CAPABILITY_DIMS, dtype=np.float32)
            for h in t0_heroes:
                caps = HERO_CAPABILITIES.get(h, {})
                for ci, dim in enumerate(CAPABILITY_DIMS):
                    t0_caps[ci] += caps.get(dim, 0)
            for h in t1_heroes:
                caps = HERO_CAPABILITIES.get(h, {})
                for ci, dim in enumerate(CAPABILITY_DIMS):
                    t1_caps[ci] += caps.get(dim, 0)
            enriched_parts.append(np.concatenate([t0_caps, t1_caps]))

        elif group == "meta_strength":
            # Avg pick_rate and ban_rate per team — how meta is this composition?
            t0_pr, t0_br, t1_pr, t1_br = 0.0, 0.0, 0.0, 0.0
            tier_meta = stats.hero_meta.get(tier, {})
            for h in t0_heroes:
                pr, br = tier_meta.get(h, (0, 0))
                t0_pr += pr; t0_br += br
            for h in t1_heroes:
                pr, br = tier_meta.get(h, (0, 0))
                t1_pr += pr; t1_br += br
            n0 = max(len(t0_heroes), 1)
            n1 = max(len(t1_heroes), 1)
            enriched_parts.append(np.array([t0_pr/n0, t0_br/n0, t1_pr/n1, t1_br/n1], dtype=np.float32))

        elif group == "draft_diversity":
            # Std dev of hero WRs per team — diverse WR spread = risky draft
            t0_wr_list = [stats.get_hero_wr(h, tier) for h in t0_heroes]
            t1_wr_list = [stats.get_hero_wr(h, tier) for h in t1_heroes]
            t0_std = float(np.std(t0_wr_list)) if len(t0_wr_list) > 1 else 0.0
            t1_std = float(np.std(t1_wr_list)) if len(t1_wr_list) > 1 else 0.0
            enriched_parts.append(np.array([t0_std, t1_std], dtype=np.float32))

        elif group == "avg_mmr":
            # Normalized match MMR: (avg_mmr - 2000) / 1000
            raw_mmr = d.get("avg_mmr") or 2600  # default to median if missing
            normalized = (float(raw_mmr) - 2000.0) / 1000.0
            enriched_parts.append(np.array([normalized], dtype=np.float32))

        elif group == "comp_wr":
            # Look up composition WR from Heroes Profile data
            # 4 features: t0_comp_wr, t0_log_games, t1_comp_wr, t1_log_games
            t0_wr, t0_games = stats.get_comp_wr(t0_heroes, tier)
            t1_wr, t1_games = stats.get_comp_wr(t1_heroes, tier)
            enriched_parts.append(np.array([
                (t0_wr - 50.0) / 10.0,  # normalize around 50%, scale to ~[-3, 1]
                np.log1p(t0_games) / 15.0,  # log games normalized (log(1.9M) ≈ 14.5)
                (t1_wr - 50.0) / 10.0,
                np.log1p(t1_games) / 15.0,
            ], dtype=np.float32))

    if enriched_parts:
        enriched = np.concatenate(enriched_parts)
    else:
        enriched = np.array([], dtype=np.float32)

    return base, enriched


def _normalized_counter(our_hero, their_hero, our_wr, their_wr, stats, tier):
    raw = stats.get_counter(our_hero, their_hero, tier)
    if raw is None:
        return 0.0
    expected = our_wr + (100 - their_wr) - 50
    return raw - expected


def _normalized_synergy(heroA, heroB, wrA, wrB, stats, tier):
    raw = stats.get_synergy(heroA, heroB, tier)
    if raw is None:
        return 0.0
    expected = 50 + (wrA - 50) + (wrB - 50)
    return raw - expected


def _avg_counter_delta(our_heroes, their_heroes, our_wrs, their_wrs, stats, tier):
    deltas = []
    for our_h, our_wr in zip(our_heroes, our_wrs):
        for their_h, their_wr in zip(their_heroes, their_wrs):
            d = _normalized_counter(our_h, their_h, our_wr, their_wr, stats, tier)
            deltas.append(d)
    return np.mean(deltas) if deltas else 0.0


def _avg_synergy_delta(heroes, wrs, stats, tier):
    deltas = []
    for j in range(len(heroes)):
        for k in range(j+1, len(heroes)):
            d = _normalized_synergy(heroes[j], heroes[k], wrs[j], wrs[k], stats, tier)
            deltas.append(d)
    return np.mean(deltas) if deltas else 0.0


# ── Dataset ──

def precompute_all_features(data, stats):
    """Precompute all feature groups for all samples (both teams + augmented swap).
    Returns: base_tensor, enriched_tensor, labels_tensor
    All features for all groups are computed; specific groups are selected at training time
    by indexing into the enriched tensor.
    """
    all_groups_mask = [True] * len(FEATURE_GROUPS)
    bases = []
    enricheds = []
    labels = []

    for d in data:
        base, enriched = extract_features(d, stats, all_groups_mask)
        y = float(d["winner"] == 0)
        # Original
        bases.append(base)
        enricheds.append(enriched)
        labels.append(y)
        # Augmented: swap teams
        base_swap, enriched_swap = _swap_features(base, enriched)
        bases.append(base_swap)
        enricheds.append(enriched_swap)
        labels.append(1.0 - y)

    return (torch.tensor(np.array(bases, dtype=np.float32)),
            torch.tensor(np.array(enricheds, dtype=np.float32)),
            torch.tensor(np.array(labels, dtype=np.float32)))


def _swap_features(base, enriched):
    """Swap team0↔team1 in all features."""
    base_swap = base.copy()
    # Swap multi-hot vectors: first 90 ↔ next 90
    base_swap[:NUM_HEROES], base_swap[NUM_HEROES:2*NUM_HEROES] = \
        base[NUM_HEROES:2*NUM_HEROES].copy(), base[:NUM_HEROES].copy()

    enriched_swap = enriched.copy()
    offset = 0
    for group in FEATURE_GROUPS:
        dim = FEATURE_GROUP_DIMS[group]
        if group == "map_type":
            pass  # symmetric
        elif group == "role_counts":
            # Swap n_roles + n_roles
            n_roles = len(FINE_ROLE_NAMES)
            enriched_swap[offset:offset+n_roles], enriched_swap[offset+n_roles:offset+2*n_roles] = \
                enriched[offset+n_roles:offset+2*n_roles].copy(), enriched[offset:offset+n_roles].copy()
        elif group in ("hero_wr", "hero_map_wr"):
            # Swap 5+5
            enriched_swap[offset:offset+5], enriched_swap[offset+5:offset+10] = \
                enriched[offset+5:offset+10].copy(), enriched[offset:offset+5].copy()
        elif group in ("team_avg_wr", "map_delta", "pairwise_counters", "pairwise_synergies"):
            # Swap 2 values
            enriched_swap[offset], enriched_swap[offset+1] = enriched[offset+1], enriched[offset]
        elif group == "counter_detail":
            # Swap first 25 ↔ last 25
            enriched_swap[offset:offset+25], enriched_swap[offset+25:offset+50] = \
                enriched[offset+25:offset+50].copy(), enriched[offset:offset+25].copy()
        elif group == "synergy_detail":
            # Swap first 10 ↔ last 10
            enriched_swap[offset:offset+10], enriched_swap[offset+10:offset+20] = \
                enriched[offset+10:offset+20].copy(), enriched[offset:offset+10].copy()
        elif group == "capabilities":
            # Swap n_caps + n_caps
            nc = NUM_CAPABILITY_DIMS
            enriched_swap[offset:offset+nc], enriched_swap[offset+nc:offset+2*nc] = \
                enriched[offset+nc:offset+2*nc].copy(), enriched[offset:offset+nc].copy()
        elif group == "meta_strength":
            # Swap (pr0, br0, pr1, br1) → (pr1, br1, pr0, br0)
            enriched_swap[offset:offset+2], enriched_swap[offset+2:offset+4] = \
                enriched[offset+2:offset+4].copy(), enriched[offset:offset+2].copy()
        elif group == "draft_diversity":
            enriched_swap[offset], enriched_swap[offset+1] = enriched[offset+1], enriched[offset]
        elif group == "avg_mmr":
            pass  # symmetric — same value for both teams
        elif group == "comp_wr":
            # Swap t0 (wr, games) ↔ t1 (wr, games)
            enriched_swap[offset:offset+2], enriched_swap[offset+2:offset+4] = \
                enriched[offset+2:offset+4].copy(), enriched[offset:offset+2].copy()
        offset += dim

    return base_swap, enriched_swap


def compute_group_indices():
    """For each feature group, compute the start:end index range in the enriched tensor."""
    indices = {}
    offset = 0
    for group in FEATURE_GROUPS:
        dim = FEATURE_GROUP_DIMS[group]
        indices[group] = (offset, offset + dim)
        offset += dim
    return indices


# ── Model ──

class WinProbEnrichedModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.15):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.extend([nn.Linear(in_dim, 1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Worker ──

def train_worker(variant_queue, base_tensor, enriched_tensor, labels_tensor,
                 test_base, test_enriched, test_labels,
                 group_indices, csv_path, gpu_id):
    """Worker process: pulls variants from queue, trains each to completion."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    while True:
        try:
            variant = variant_queue.get_nowait()
        except Exception:
            break

        variant_id = variant['id']
        groups_mask = variant['groups_mask']
        hidden_dims = variant['hidden_dims']
        lr = variant['lr']

        # Compute input dimension
        extra_dim = sum(FEATURE_GROUP_DIMS[FEATURE_GROUPS[i]]
                       for i in range(len(FEATURE_GROUPS)) if groups_mask[i])
        total_dim = INPUT_DIM_BASE + extra_dim

        # Select feature columns
        enriched_cols = []
        for i, group in enumerate(FEATURE_GROUPS):
            if groups_mask[i]:
                start, end = group_indices[group]
                enriched_cols.extend(range(start, end))

        if enriched_cols:
            train_X = torch.cat([base_tensor, enriched_tensor[:, enriched_cols]], dim=1).to(device)
            test_X = torch.cat([test_base, test_enriched[:, enriched_cols]], dim=1).to(device)
        else:
            train_X = base_tensor.to(device)
            test_X = test_base.to(device)

        train_y = labels_tensor.to(device)
        test_y = test_labels.to(device)

        model = WinProbEnrichedModel(total_dim, hidden_dims, dropout=0.15).to(device)
        params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
        criterion = nn.BCELoss()

        best_test_loss = float('inf')
        best_test_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        batch_size = 1024
        n_train = len(train_X)
        n_test = len(test_X)

        for epoch in range(200):
            # Train
            model.train()
            perm = torch.randperm(n_train, device=device)
            for i in range(0, n_train, batch_size):
                idx = perm[i:i+batch_size]
                pred = model(train_X[idx])
                loss = criterion(pred, train_y[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # Eval
            model.eval()
            with torch.no_grad():
                test_pred = model(test_X)
                test_loss = criterion(test_pred, test_y).item()
                test_acc = ((test_pred > 0.5).float() == test_y).float().mean().item() * 100

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_acc = test_acc
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break

        # Build group name
        active_groups = [FEATURE_GROUPS[i] for i in range(len(FEATURE_GROUPS)) if groups_mask[i]]
        name = "+".join(active_groups) if active_groups else "base_only"
        name += f"_h{'_'.join(str(d) for d in hidden_dims)}_lr{lr}"

        # Write result
        lock = filelock.FileLock(csv_path + ".lock")
        with lock:
            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['variant_id', 'name', 'groups', 'hidden_dims', 'lr',
                                    'best_test_acc', 'best_test_loss', 'epochs_trained',
                                    'params', 'input_dim'])
                writer.writerow([variant_id, name, json.dumps(active_groups),
                               json.dumps(hidden_dims), lr,
                               f"{best_test_acc:.4f}", f"{best_test_loss:.6f}",
                               best_epoch, params, total_dim])

        if variant_id % 100 == 0:
            print(f"[GPU {gpu_id}] v{variant_id}: {name[:50]} → {best_test_acc:.2f}%")


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=int, default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=256)
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        num_gpus = 1
    print(f"GPUs: {num_gpus}, Workers per GPU: {args.workers_per_gpu}")

    # Load data and stats
    print("Loading replay data...")
    data = load_replay_data()
    train_data, test_data = split_data(data)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    print("Loading stats from DB...")
    stats = StatsCache()
    print(f"  Hero WR tiers: {list(stats.hero_wr.keys())}")

    # Precompute all features
    print("Precomputing features (all groups, with augmentation)...")
    t0 = time.time()
    train_base, train_enriched, train_labels = precompute_all_features(train_data, stats)
    test_base, test_enriched, test_labels = precompute_all_features(test_data, stats)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Train: {train_base.shape} base + {train_enriched.shape} enriched")
    print(f"  Test: {test_base.shape} base + {test_enriched.shape} enriched")

    group_indices = compute_group_indices()

    # Share tensors across processes
    train_base.share_memory_()
    train_enriched.share_memory_()
    train_labels.share_memory_()
    test_base.share_memory_()
    test_enriched.share_memory_()
    test_labels.share_memory_()

    # Build variant list: all 2^10 feature subsets × 2 MLP sizes × 2 LRs
    mlp_sizes = [[256, 128], [512, 256, 128]]
    lrs = [1e-3, 5e-4]

    variants = []
    vid = 0
    for mask_int in range(1024):  # 2^10 subsets
        groups_mask = [(mask_int >> i) & 1 == 1 for i in range(10)]
        for hidden_dims in mlp_sizes:
            for lr in lrs:
                variants.append({
                    'id': vid,
                    'groups_mask': groups_mask,
                    'hidden_dims': hidden_dims,
                    'lr': lr,
                })
                vid += 1

    # Also add base-only (no enriched features) as explicit baseline
    for hidden_dims in mlp_sizes:
        for lr in lrs:
            variants.append({
                'id': vid,
                'groups_mask': [False] * 10,
                'hidden_dims': hidden_dims,
                'lr': lr,
            })
            vid += 1

    total = len(variants)
    if args.variants:
        variants = variants[:args.variants]
    print(f"Total variants: {total}, running: {len(variants)}")

    csv_path = os.path.join(os.path.dirname(__file__), "win_prob_sweep_results.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Create shared work queue
    mp.set_start_method('spawn', force=True)
    variant_queue = mp.Queue()
    for v in variants:
        variant_queue.put(v)

    # Launch workers
    total_workers = num_gpus * args.workers_per_gpu
    print(f"Launching {total_workers} workers ({args.workers_per_gpu} per GPU)...")
    t_start = time.time()

    processes = []
    for gpu_id in range(num_gpus):
        for w in range(args.workers_per_gpu):
            p = mp.Process(target=train_worker, args=(
                variant_queue, train_base, train_enriched, train_labels,
                test_base, test_enriched, test_labels,
                group_indices, csv_path, gpu_id,
            ))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - t_start
    print(f"\nSweep complete in {elapsed/60:.1f} min")

    # ── Analysis ──
    if not os.path.exists(csv_path):
        print("No results!")
        return

    with open(csv_path) as f:
        results = list(csv.DictReader(f))

    results.sort(key=lambda r: -float(r['best_test_acc']))

    print(f"\n{'='*90}")
    print("TOP 30")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'Acc':>7} {'Loss':>10} {'Ep':>4} {'InDim':>6} {'Params':>9} {'Groups'}")
    print(f"{'-'*90}")
    for i, r in enumerate(results[:30]):
        groups = json.loads(r['groups'])
        g_str = "+".join(g[:8] for g in groups) if groups else "base_only"
        print(f"{i+1:<5} {float(r['best_test_acc']):>6.2f}% {float(r['best_test_loss']):>10.6f} "
              f"{r['epochs_trained']:>4} {r['input_dim']:>6} {int(r['params']):>9,} {g_str}")

    print(f"\nBOTTOM 10")
    for i, r in enumerate(results[-10:]):
        groups = json.loads(r['groups'])
        g_str = "+".join(g[:8] for g in groups) if groups else "base_only"
        print(f"  {float(r['best_test_acc']):>6.2f}% {g_str}")

    # ── Per-feature-group marginal analysis ──
    print(f"\n{'='*90}")
    print("FEATURE GROUP MARGINAL ANALYSIS")
    print(f"{'='*90}")
    print(f"{'Group':<25} {'With':>8} {'Without':>8} {'Delta':>8} {'Impact'}")
    print(f"{'-'*60}")

    deltas = []
    for i, group in enumerate(FEATURE_GROUPS):
        with_group = [float(r['best_test_acc']) for r in results
                      if group in json.loads(r['groups'])]
        without_group = [float(r['best_test_acc']) for r in results
                        if group not in json.loads(r['groups'])]
        avg_with = np.mean(with_group) if with_group else 0
        avg_without = np.mean(without_group) if without_group else 0
        delta = avg_with - avg_without
        deltas.append((group, avg_with, avg_without, delta))

    deltas.sort(key=lambda x: -x[3])
    for group, avg_with, avg_without, delta in deltas:
        bar = "+" * max(0, int(delta * 20)) + "-" * max(0, int(-delta * 20))
        print(f"{group:<25} {avg_with:>7.2f}% {avg_without:>7.2f}% {delta:>+7.3f}% {bar}")

    # ── Best MLP / LR ──
    print(f"\n{'='*90}")
    print("BEST MLP SIZE AND LR (marginal averages)")
    from collections import defaultdict
    mlp_accs = defaultdict(list)
    lr_accs = defaultdict(list)
    for r in results:
        mlp_accs[r['hidden_dims']].append(float(r['best_test_acc']))
        lr_accs[r['lr']].append(float(r['best_test_acc']))

    print("MLP size:")
    for k in sorted(mlp_accs.keys(), key=lambda k: -np.mean(mlp_accs[k])):
        print(f"  {k}: {np.mean(mlp_accs[k]):.2f}%")
    print("Learning rate:")
    for k in sorted(lr_accs.keys(), key=lambda k: -np.mean(lr_accs[k])):
        print(f"  {k}: {np.mean(lr_accs[k]):.2f}%")


if __name__ == "__main__":
    main()
