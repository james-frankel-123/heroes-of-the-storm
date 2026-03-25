"""
Batched MCTS with virtual loss for draft selection.

Key optimizations over the sequential mcts_search:
1. Virtual loss parallel MCTS: collect K=32 leaves per batch, single GPU call
2. DraftStateFast: integer bitsets instead of numpy arrays for ~0 cost clone
3. Cached opponent distributions: GD called once per opponent node, not per simulation
4. Pre-allocated numpy buffers for batch construction
"""
import numpy as np
import random
import torch
import torch.nn.functional as F

NUM_HEROES = 90
NUM_MAPS = 14
NUM_TIERS = 3
STATE_DIM = NUM_HEROES * 3 + NUM_MAPS + NUM_TIERS + 2 + 1  # 290

from shared import (
    MAPS, SKILL_TIERS, HERO_TO_IDX, HEROES,
    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
)
from train_draft_policy import DRAFT_ORDER


# ── DraftStateFast ──────────────────────────────────────────────────

class DraftStateFast:
    """Compact draft state using Python ints as bitsets. ~0 cost clone."""
    __slots__ = ['t0_picks', 't1_picks', 'bans', 'taken_mask',
                 'game_map_idx', 'tier_idx', 'step', 'our_team']

    def __init__(self, game_map_idx: int, tier_idx: int, our_team: int = 0):
        self.t0_picks = 0
        self.t1_picks = 0
        self.bans = 0
        self.taken_mask = 0
        self.game_map_idx = game_map_idx
        self.tier_idx = tier_idx
        self.step = 0
        self.our_team = our_team

    def clone(self):
        s = DraftStateFast.__new__(DraftStateFast)
        s.t0_picks = self.t0_picks
        s.t1_picks = self.t1_picks
        s.bans = self.bans
        s.taken_mask = self.taken_mask
        s.game_map_idx = self.game_map_idx
        s.tier_idx = self.tier_idx
        s.step = self.step
        s.our_team = self.our_team
        return s

    def apply_action(self, hero_idx: int, team: int, action_type: str):
        bit = 1 << hero_idx
        self.taken_mask |= bit
        if action_type == 'ban':
            self.bans |= bit
        elif team == 0:
            self.t0_picks |= bit
        else:
            self.t1_picks |= bit
        self.step += 1

    def is_terminal(self) -> bool:
        return self.step >= 16

    def current_team(self) -> int:
        if self.step >= 16:
            return -1
        return DRAFT_ORDER[self.step][0]

    def current_action_type(self) -> str:
        if self.step >= 16:
            return 'pick'
        return DRAFT_ORDER[self.step][1]

    def valid_mask_np(self) -> np.ndarray:
        mask = np.ones(NUM_HEROES, dtype=np.float32)
        taken = self.taken_mask
        i = 0
        while taken:
            if taken & 1:
                mask[i] = 0.0
            taken >>= 1
            i += 1
        return mask

    def to_numpy(self) -> np.ndarray:
        """Convert to 290-dim float array for network input."""
        out = np.zeros(STATE_DIM, dtype=np.float32)
        for i in range(NUM_HEROES):
            if self.t0_picks & (1 << i):
                out[i] = 1.0
            if self.t1_picks & (1 << i):
                out[NUM_HEROES + i] = 1.0
            if self.bans & (1 << i):
                out[2 * NUM_HEROES + i] = 1.0
        out[3 * NUM_HEROES + self.game_map_idx] = 1.0
        out[3 * NUM_HEROES + NUM_MAPS + self.tier_idx] = 1.0
        if self.step < 16:
            out[-3] = self.step / 15.0
            out[-2] = 0.0 if DRAFT_ORDER[self.step][1] == 'ban' else 1.0
        else:
            out[-3] = 1.0
            out[-2] = 1.0
        out[-1] = float(self.our_team)
        return out

    def to_numpy_gd(self) -> np.ndarray:
        """289-dim array for GD model (no our_team)."""
        full = self.to_numpy()
        return full[:-1]

    def hero_lists(self):
        """Return (team0_heroes, team1_heroes) as lists of hero names."""
        t0 = [HEROES[i] for i in range(NUM_HEROES) if self.t0_picks & (1 << i)]
        t1 = [HEROES[i] for i in range(NUM_HEROES) if self.t1_picks & (1 << i)]
        return t0, t1


def state_from_strings(game_map: str, skill_tier: str, our_team: int = 0):
    """Create DraftStateFast from string map/tier names."""
    map_idx = MAPS.index(game_map) if game_map in MAPS else 0
    tier_idx = SKILL_TIERS.index(skill_tier) if skill_tier in SKILL_TIERS else 1
    return DraftStateFast(map_idx, tier_idx, our_team)


# ── MCTSNode ────────────────────────────────────────────────────────

class MCTSNodeBatched:
    """MCTS node with virtual loss support and cached opponent distributions."""
    __slots__ = ['parent', 'action', 'children', 'visit_count', 'value_sum',
                 'prior', 'is_expanded', 'cached_opponent_dist']

    def __init__(self, parent=None, action=-1, prior=0.0):
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False
        self.cached_opponent_dist = None

    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        exploration = c_puct * self.prior * (parent_visits ** 0.5) / (1 + self.visit_count)
        return self.q_value() + exploration


# ── Batched MCTS Search ────────────────────────────────────────────

def mcts_search_batched(
    root_state: DraftStateFast,
    batch_predict_fn,  # (states_np, masks_np) -> (priors_np, values_np) — batched
    gd_predict_fn,     # (state_np, mask_np) -> logits_np — single, for opponent
    num_simulations: int = 200,
    batch_size: int = 32,
    c_puct: float = 2.0,
    virtual_loss: float = 3.0,
) -> np.ndarray:
    """
    Run batched MCTS with virtual loss.

    batch_predict_fn: takes (N, STATE_DIM) states and (N, NUM_HEROES) masks,
        returns (N, NUM_HEROES) priors and (N,) symmetrized values.
    gd_predict_fn: takes single (289,) state and (90,) mask,
        returns (90,) logits for opponent sampling.

    Returns (NUM_HEROES,) visit distribution.
    """
    our_team = root_state.our_team
    root = MCTSNodeBatched()

    # Pre-allocate buffers for batch construction (2x for symmetrization)
    state_buffer = np.zeros((batch_size * 2, STATE_DIM), dtype=np.float32)
    mask_buffer = np.zeros((batch_size * 2, NUM_HEROES), dtype=np.float32)

    # Expand root
    root_np = root_state.to_numpy()
    root_mask = root_state.valid_mask_np()
    state_buffer[0] = root_np
    state_buffer[1] = root_np.copy()
    state_buffer[1, -1] = 1.0 - state_buffer[1, -1]  # swapped perspective
    mask_buffer[0] = root_mask
    mask_buffer[1] = root_mask
    priors, values = batch_predict_fn(state_buffer[:2], mask_buffer[:2])
    root_priors = priors[0] * root_mask
    psum = root_priors.sum()
    if psum > 0:
        root_priors /= psum
    root.is_expanded = True
    for a in range(NUM_HEROES):
        if root_mask[a] > 0 and root_priors[a] > 0:
            root.children[a] = MCTSNodeBatched(parent=root, action=a, prior=root_priors[a])

    remaining = num_simulations
    while remaining > 0:
        k = min(batch_size, remaining)
        leaves = []  # (node, scratch_state, path) tuples
        needs_eval = []  # indices into leaves that need network evaluation

        for i in range(k):
            node = root
            scratch = root_state.clone()
            path = [root]

            # SELECT with virtual loss
            while node.is_expanded and not scratch.is_terminal():
                step_team = scratch.current_team()
                step_type = scratch.current_action_type()

                if step_team == our_team:
                    # UCB selection among children
                    if not node.children:
                        break
                    best_child = None
                    best_score = -1e9
                    for child in node.children.values():
                        score = child.ucb_score(node.visit_count, c_puct)
                        if score > best_score:
                            best_score = score
                            best_child = child

                    if best_child is None:
                        break

                    # Apply virtual loss
                    best_child.visit_count += int(virtual_loss)
                    best_child.value_sum -= virtual_loss

                    scratch.apply_action(best_child.action, step_team, step_type)
                    node = best_child
                    path.append(node)
                else:
                    # Opponent: sample from cached distribution
                    if node.cached_opponent_dist is None:
                        gd_state = scratch.to_numpy_gd()
                        gd_mask = scratch.valid_mask_np()
                        logits = gd_predict_fn(gd_state, gd_mask)
                        # Softmax with temperature 1.0
                        logits = logits - logits.max()
                        exp_logits = np.exp(logits) * gd_mask
                        total = exp_logits.sum()
                        if total > 0:
                            node.cached_opponent_dist = exp_logits / total
                        else:
                            node.cached_opponent_dist = gd_mask / gd_mask.sum()

                    dist = node.cached_opponent_dist
                    action = np.random.choice(NUM_HEROES, p=dist)
                    scratch.apply_action(action, step_team, step_type)
                    # Opponent moves don't create tree nodes; stay at same node

            leaves.append((node, scratch, path))

            if scratch.is_terminal() or not node.is_expanded:
                needs_eval.append(i)

        # BATCH EVALUATE all leaves that need it
        if needs_eval:
            n_eval = len(needs_eval)
            # Fill buffers: normal + swapped perspective
            for j, idx in enumerate(needs_eval):
                _, scratch, _ = leaves[idx]
                s_np = scratch.to_numpy()
                m_np = scratch.valid_mask_np()
                state_buffer[j] = s_np
                state_buffer[n_eval + j] = s_np.copy()
                state_buffer[n_eval + j, -1] = 1.0 - s_np[-1]  # flip our_team
                mask_buffer[j] = m_np
                mask_buffer[n_eval + j] = m_np

            # Single batched GPU call (2*n_eval samples)
            all_priors, all_values = batch_predict_fn(
                state_buffer[:2 * n_eval], mask_buffer[:2 * n_eval]
            )

            # Symmetrize values
            values_normal = all_values[:n_eval]
            values_swapped = all_values[n_eval:]
            values_sym = (values_normal + (1.0 - values_swapped)) / 2.0

            # EXPAND + BACKPROP
            for j, idx in enumerate(needs_eval):
                node, scratch, path = leaves[idx]
                value = values_sym[j]

                if not scratch.is_terminal() and not node.is_expanded:
                    priors_j = all_priors[j] * mask_buffer[j]
                    psum = priors_j.sum()
                    if psum > 0:
                        priors_j /= psum
                    node.is_expanded = True
                    for a in range(NUM_HEROES):
                        if mask_buffer[j, a] > 0 and priors_j[a] > 0:
                            node.children[a] = MCTSNodeBatched(
                                parent=node, action=a, prior=priors_j[a]
                            )

                # Undo virtual loss and apply real update
                for n in path:
                    n.visit_count += 1 - int(virtual_loss)  # undo VL +VL, apply +1
                    n.value_sum += value + virtual_loss      # undo VL -VL, apply +value
        else:
            # All leaves were already expanded (rare) — just undo virtual loss
            for i in range(k):
                _, _, path = leaves[i]
                for n in path:
                    n.visit_count -= int(virtual_loss)
                    n.value_sum += virtual_loss

        remaining -= k

    # Extract visit distribution
    visits = np.zeros(NUM_HEROES, dtype=np.float32)
    for a, child in root.children.items():
        visits[a] = max(0, child.visit_count)
    total = visits.sum()
    if total > 0:
        visits /= total
    return visits


# ── Full Episode Simulation ─────────────────────────────────────────

def simulate_episode_batched(
    game_map: str,
    skill_tier: str,
    our_team: int,
    batch_predict_fn,   # for policy network (batched)
    gd_predict_fn,      # for opponent (single)
    wp_eval_fn,         # (t0_heroes, t1_heroes, game_map, tier) -> float (team0 WP)
    num_simulations: int = 200,
    batch_size: int = 32,
):
    """
    Run a full draft episode using batched MCTS.
    Returns (win_prob, training_examples) matching the old interface.
    """
    state = state_from_strings(game_map, skill_tier, our_team)
    training_examples = []
    gd_temperature = random.choice([0.5, 0.8, 1.0, 1.2, 1.5])

    while not state.is_terminal():
        team, action_type = DRAFT_ORDER[state.step]

        if team == our_team:
            state_features = state.to_numpy()
            valid = state.valid_mask_np()

            visit_dist = mcts_search_batched(
                state, batch_predict_fn, gd_predict_fn,
                num_simulations=num_simulations,
                batch_size=batch_size,
            )

            training_examples.append((state_features, visit_dist, valid))
            action = np.random.choice(NUM_HEROES, p=visit_dist) if visit_dist.sum() > 0 else 0
            state.apply_action(action, team, action_type)
        else:
            gd_state = state.to_numpy_gd()
            gd_mask = state.valid_mask_np()
            logits = gd_predict_fn(gd_state, gd_mask)
            # Sample with temperature
            logits = logits / gd_temperature
            logits = logits - logits.max()
            exp_l = np.exp(logits) * gd_mask
            total = exp_l.sum()
            if total > 0:
                probs = exp_l / total
            else:
                probs = gd_mask / gd_mask.sum()
            action = np.random.choice(NUM_HEROES, p=probs)
            state.apply_action(action, team, action_type)

    # Terminal: evaluate with WP model (symmetrized by the caller)
    t0h, t1h = state.hero_lists()
    wp_t0 = wp_eval_fn(t0h, t1h, game_map, skill_tier)
    win_prob = wp_t0 if our_team == 0 else 1.0 - wp_t0

    return win_prob, training_examples
