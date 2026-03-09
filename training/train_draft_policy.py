"""
Draft Policy Model — RL-trained policy that maximizes win probability.

Uses Q-Learning where:
- State: current draft state (heroes picked/banned + map + tier)
- Action: pick or ban a hero (masked to valid options)
- Reward: final win probability from the Win Probability model (only at game end)
- Opponent: Generic Draft model (provides realistic opponent moves)

The policy plays as team 0 against the Generic Draft model (team 1),
alternating moves according to the standard Storm League draft order.

No discount factor — only the final win probability matters.

Usage:
    export DATABASE_URL=...
    python training/train_draft_policy.py
"""
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    NUM_HEROES, NUM_MAPS, NUM_TIERS, HEROES, MAPS, SKILL_TIERS,
    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
    load_replay_data, split_data,
)

# Same input dims as generic draft
STATE_DIM = NUM_HEROES * 3 + NUM_MAPS + NUM_TIERS + 2  # 289

# Standard Storm League draft order (16 steps)
# Format: (team, action_type) where team 0/1, action_type 'ban'/'pick'
# Team 0 bans first
DRAFT_ORDER = [
    (0, 'ban'), (1, 'ban'),  # Bans 1-2
    (0, 'ban'), (1, 'ban'),  # Bans 3-4
    (0, 'pick'), (1, 'pick'), (1, 'pick'), (0, 'pick'),  # Picks 1-4
    (0, 'ban'), (1, 'ban'),  # Bans 5-6
    (1, 'pick'), (0, 'pick'), (0, 'pick'), (1, 'pick'),  # Picks 5-8
    (1, 'pick'), (0, 'pick'),  # Picks 9-10
]


class DraftState:
    """Mutable draft state for simulation."""
    def __init__(self, game_map: str, skill_tier: str):
        self.team0_picks = np.zeros(NUM_HEROES, dtype=np.float32)
        self.team1_picks = np.zeros(NUM_HEROES, dtype=np.float32)
        self.bans = np.zeros(NUM_HEROES, dtype=np.float32)
        self.taken = set()
        self.game_map = game_map
        self.skill_tier = skill_tier
        self.step = 0

    def to_tensor(self, step_type: float) -> torch.Tensor:
        map_vec = map_to_one_hot(self.game_map)
        tier_vec = tier_to_one_hot(self.skill_tier)
        step_norm = self.step / 15.0
        x = np.concatenate([
            self.team0_picks, self.team1_picks, self.bans,
            map_vec, tier_vec,
            [step_norm, step_type],
        ])
        return torch.from_numpy(x).unsqueeze(0)

    def valid_mask(self) -> torch.Tensor:
        mask = np.ones(NUM_HEROES, dtype=np.float32)
        for idx in self.taken:
            mask[idx] = 0.0
        return torch.from_numpy(mask).unsqueeze(0)

    def apply_action(self, hero_idx: int, team: int, action_type: str):
        self.taken.add(hero_idx)
        if action_type == 'ban':
            self.bans[hero_idx] = 1.0
        elif team == 0:
            self.team0_picks[hero_idx] = 1.0
        else:
            self.team1_picks[hero_idx] = 1.0
        self.step += 1


class PolicyNetwork(nn.Module):
    """Q-network: state → Q-values per hero action."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_HEROES),
        )

    def forward(self, x, mask=None):
        q = self.net(x)
        if mask is not None:
            q = q + (1 - mask) * (-1e9)
        return q


def load_pretrained_models(device):
    """Load the pre-trained Win Probability and Generic Draft models."""
    from train_win_probability import WinProbModel, INPUT_DIM as WP_INPUT_DIM
    from train_generic_draft import GenericDraftModel

    wp_path = os.path.join(os.path.dirname(__file__), "win_probability.pt")
    gd_path = os.path.join(os.path.dirname(__file__), "generic_draft.pt")

    if not os.path.exists(wp_path) or not os.path.exists(gd_path):
        raise FileNotFoundError(
            "Pre-trained models not found. Train win_probability and generic_draft first."
        )

    wp_model = WinProbModel().to(device)
    wp_model.load_state_dict(torch.load(wp_path, weights_only=True, map_location=device))
    wp_model.eval()

    gd_model = GenericDraftModel().to(device)
    gd_model.load_state_dict(torch.load(gd_path, weights_only=True, map_location=device))
    gd_model.eval()

    return wp_model, gd_model


def evaluate_win_prob(wp_model, state: DraftState, device) -> float:
    """Get win probability for team 0 from current state."""
    from shared import NUM_HEROES as NH
    t0 = torch.from_numpy(state.team0_picks).unsqueeze(0).to(device)
    t1 = torch.from_numpy(state.team1_picks).unsqueeze(0).to(device)
    m = torch.from_numpy(map_to_one_hot(state.game_map)).unsqueeze(0).to(device)
    t = torch.from_numpy(tier_to_one_hot(state.skill_tier)).unsqueeze(0).to(device)
    x = torch.cat([t0, t1, m, t], dim=1)
    with torch.no_grad():
        return wp_model(x).item()


def opponent_pick(gd_model, state: DraftState, step_type: float, device) -> int:
    """Use Generic Draft model to pick for the opponent (team 1)."""
    x = state.to_tensor(step_type).to(device)
    mask = state.valid_mask().to(device)
    with torch.no_grad():
        logits = gd_model(x, mask)
        # Sample from distribution (with temperature) for diversity
        probs = F.softmax(logits / 0.8, dim=1)
        action = torch.multinomial(probs, 1).item()
    return action


def simulate_draft(
    policy: PolicyNetwork,
    gd_model,
    wp_model,
    game_map: str,
    skill_tier: str,
    device,
    epsilon: float = 0.1,
) -> tuple[float, list[tuple[torch.Tensor, int, torch.Tensor]]]:
    """
    Simulate a full draft. Policy controls team 0, Generic Draft controls team 1.
    Returns (reward, trajectory) where trajectory is [(state, action, mask), ...].
    """
    state = DraftState(game_map, skill_tier)
    trajectory = []

    for team, action_type in DRAFT_ORDER:
        step_type = 0.0 if action_type == 'ban' else 1.0

        if team == 0:
            # Policy's turn
            x = state.to_tensor(step_type).to(device)
            mask = state.valid_mask().to(device)

            if random.random() < epsilon:
                # Explore: random valid action
                valid_indices = [i for i in range(NUM_HEROES) if i not in state.taken]
                action = random.choice(valid_indices)
            else:
                with torch.no_grad():
                    q = policy(x, mask)
                    action = q.argmax(dim=1).item()

            trajectory.append((x.detach(), action, mask.detach()))
            state.apply_action(action, team, action_type)
        else:
            # Opponent's turn (Generic Draft model)
            action = opponent_pick(gd_model, state, step_type, device)
            state.apply_action(action, team, action_type)

    # Reward = win probability for team 0
    reward = evaluate_win_prob(wp_model, state, device)
    return reward, trajectory


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading pre-trained models...")
    wp_model, gd_model = load_pretrained_models(device)

    policy = PolicyNetwork().to(device)
    target_net = PolicyNetwork().to(device)
    target_net.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)

    # Load test data for evaluation
    print("Loading replay data for evaluation...")
    all_data = load_replay_data(limit=1000)
    _, test_data = split_data(all_data)
    test_maps = [d["game_map"] for d in test_data[:50]]
    test_tiers = [d["skill_tier"] for d in test_data[:50]]

    # Replay buffer
    buffer = []
    BUFFER_SIZE = 50_000
    BATCH_SIZE = 64
    GAMMA = 1.0  # No discount — only final reward matters
    TARGET_UPDATE = 100
    NUM_EPISODES = 10_000
    EVAL_EVERY = 500

    best_eval_reward = 0.0
    epsilon = 0.3

    print(f"Training for {NUM_EPISODES} episodes...")
    for episode in range(NUM_EPISODES):
        # Random map and tier for each episode
        game_map = random.choice(MAPS)
        skill_tier = random.choice(SKILL_TIERS)

        # Decay epsilon
        epsilon = max(0.05, 0.3 - (episode / NUM_EPISODES) * 0.25)

        reward, trajectory = simulate_draft(
            policy, gd_model, wp_model, game_map, skill_tier, device, epsilon
        )

        # Store transitions in buffer
        # Since reward only comes at the end, assign reward to last action
        # and 0 to all others (Q-learning will propagate backwards)
        for i, (s, a, m) in enumerate(trajectory):
            r = reward if i == len(trajectory) - 1 else 0.0
            is_terminal = i == len(trajectory) - 1
            # Next state is the state after the next policy action (skip opponent)
            if i + 1 < len(trajectory):
                next_s, _, next_m = trajectory[i + 1]
            else:
                next_s, next_m = s, m  # dummy, won't be used (terminal)

            buffer.append((s, a, r, next_s, next_m, is_terminal))
            if len(buffer) > BUFFER_SIZE:
                buffer.pop(0)

        # Train from buffer
        if len(buffer) >= BATCH_SIZE:
            batch = random.sample(buffer, BATCH_SIZE)
            states = torch.cat([b[0] for b in batch]).to(device)
            actions = torch.tensor([b[1] for b in batch], dtype=torch.long).to(device)
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(device)
            next_states = torch.cat([b[3] for b in batch]).to(device)
            next_masks = torch.cat([b[4] for b in batch]).to(device)
            terminals = torch.tensor([b[5] for b in batch], dtype=torch.bool).to(device)

            # Current Q values
            q_values = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Target Q values
            with torch.no_grad():
                next_q = target_net(next_states, next_masks).max(dim=1).values
                next_q[terminals] = 0.0
                target_q = rewards + GAMMA * next_q

            loss = F.mse_loss(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network
        if (episode + 1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy.state_dict())

        # Evaluate
        if (episode + 1) % EVAL_EVERY == 0:
            eval_rewards = []
            policy.eval()
            for m, t in zip(test_maps, test_tiers):
                r, _ = simulate_draft(policy, gd_model, wp_model, m, t, device, epsilon=0)
                eval_rewards.append(r)
            policy.train()
            avg_reward = np.mean(eval_rewards)
            print(f"Episode {episode+1}: avg_win_prob={avg_reward:.4f} epsilon={epsilon:.3f} buffer={len(buffer)}")

            if avg_reward > best_eval_reward:
                best_eval_reward = avg_reward
                torch.save(policy.state_dict(), os.path.join(os.path.dirname(__file__), "draft_policy.pt"))

    # Export to ONNX
    policy.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "draft_policy.pt"),
                                      weights_only=True))
    policy.eval()
    dummy_x = torch.randn(1, STATE_DIM)
    dummy_mask = torch.ones(1, NUM_HEROES)
    onnx_path = os.path.join(os.path.dirname(__file__), "draft_policy.onnx")
    torch.onnx.export(
        policy, (dummy_x, dummy_mask), onnx_path,
        input_names=["state", "valid_mask"],
        output_names=["q_values"],
        dynamic_axes={
            "state": {0: "batch"},
            "valid_mask": {0: "batch"},
            "q_values": {0: "batch"},
        },
    )
    print(f"Exported ONNX model to {onnx_path}")
    print(f"Best eval win probability: {best_eval_reward:.4f}")


if __name__ == "__main__":
    train()
