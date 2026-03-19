"""
Draft Policy Model — AlphaZero-style MCTS + neural network.

Architecture:
- Shared residual backbone: 289 → 512 (2 residual blocks) → 256 → 128
- Policy head: 128 → 90 (softmax, masked to legal actions)
- Value head: 128 → 64 → 1 (tanh, scaled to [0, 1])

Training:
- Network plays as team 0 using MCTS (200-400 simulations per move)
- Opponent (team 1) sampled from pool of 3-5 Generic Draft models
- Win Probability model evaluates terminal states
- MCTS visit count distribution → policy target
- Final win probability → value target
- Replay buffer of last 50k-100k drafts

Usage:
    export DATABASE_URL=...
    python training/train_draft_policy.py
"""
import os
import sys
import math
import random
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    NUM_HEROES, NUM_MAPS, NUM_TIERS, HEROES, MAPS, SKILL_TIERS,
    heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot,
    load_replay_data, split_data, embed_onnx_weights,
    optimize_onnx, quantize_onnx, verify_quantized_model,
)

STATE_DIM = NUM_HEROES * 3 + NUM_MAPS + NUM_TIERS + 2 + 1  # 290 (last = our_team indicator)

# Standard Storm League draft order (16 steps)
# Standard Storm League draft order (16 steps)
# Spec: A ban 1, B ban 1, A ban 1, B ban 1,
#        A pick 1, B pick 2, A pick 2,
#        B ban 1, A ban 1,
#        B pick 2, A pick 2, B pick 1
DRAFT_ORDER = [
    (0, 'ban'), (1, 'ban'),
    (0, 'ban'), (1, 'ban'),
    (0, 'pick'), (1, 'pick'), (1, 'pick'), (0, 'pick'), (0, 'pick'),
    (1, 'ban'), (0, 'ban'),
    (1, 'pick'), (1, 'pick'), (0, 'pick'), (0, 'pick'), (1, 'pick'),
]


# ── Draft State ──────────────────────────────────────────────────────

class DraftState:
    """Mutable draft state for simulation."""
    def __init__(self, game_map: str, skill_tier: str, our_team: int = 0):
        self.team0_picks = np.zeros(NUM_HEROES, dtype=np.float32)
        self.team1_picks = np.zeros(NUM_HEROES, dtype=np.float32)
        self.bans = np.zeros(NUM_HEROES, dtype=np.float32)
        self.taken = set()
        self.game_map = game_map
        self.skill_tier = skill_tier
        self.step = 0
        self.our_team = our_team  # 0 or 1 — whose perspective

    def clone(self):
        s = DraftState(self.game_map, self.skill_tier, self.our_team)
        s.team0_picks = self.team0_picks.copy()
        s.team1_picks = self.team1_picks.copy()
        s.bans = self.bans.copy()
        s.taken = set(self.taken)
        s.step = self.step
        return s

    def to_numpy(self) -> np.ndarray:
        step_type = 0.0 if DRAFT_ORDER[self.step][1] == 'ban' else 1.0
        map_vec = map_to_one_hot(self.game_map)
        tier_vec = tier_to_one_hot(self.skill_tier)
        step_norm = self.step / 15.0
        return np.concatenate([
            self.team0_picks, self.team1_picks, self.bans,
            map_vec, tier_vec,
            np.array([step_norm, step_type, float(self.our_team)], dtype=np.float32),
        ])

    def to_tensor(self, device) -> torch.Tensor:
        return torch.from_numpy(self.to_numpy()).unsqueeze(0).float().to(device)

    def to_tensor_gd(self, device) -> torch.Tensor:
        """State tensor without our_team indicator (289 dims) for GD/WP models."""
        return torch.from_numpy(self.to_numpy()[:-1]).unsqueeze(0).float().to(device)

    def valid_mask_np(self) -> np.ndarray:
        mask = np.ones(NUM_HEROES, dtype=np.float32)
        for idx in self.taken:
            mask[idx] = 0.0
        return mask

    def valid_mask(self, device) -> torch.Tensor:
        return torch.from_numpy(self.valid_mask_np()).unsqueeze(0).to(device)

    def apply_action(self, hero_idx: int, team: int, action_type: str):
        self.taken.add(hero_idx)
        if action_type == 'ban':
            self.bans[hero_idx] = 1.0
        elif team == 0:
            self.team0_picks[hero_idx] = 1.0
        else:
            self.team1_picks[hero_idx] = 1.0
        self.step += 1

    def is_terminal(self) -> bool:
        return self.step >= 16

    def current_team(self) -> int:
        return DRAFT_ORDER[self.step][0]

    def current_action_type(self) -> str:
        return DRAFT_ORDER[self.step][1]


# ── Residual Network ────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out = F.relu(out + residual)
        return out


class AlphaZeroDraftNet(nn.Module):
    """
    Shared residual backbone with policy and value heads.
    290 → 768 (3 residual blocks) → 512 → 256 → heads
    ~4M params total.
    """
    def __init__(self):
        super().__init__()
        # Shared backbone: wider + deeper
        self.input_fc = nn.Linear(STATE_DIM, 768)
        self.input_bn = nn.BatchNorm1d(768)
        self.res_block1 = ResidualBlock(768)
        self.res_block2 = ResidualBlock(768)
        self.res_block3 = ResidualBlock(768)
        self.compress1 = nn.Linear(768, 512)
        self.compress1_bn = nn.BatchNorm1d(512)
        self.compress2 = nn.Linear(512, 256)
        self.compress2_bn = nn.BatchNorm1d(256)

        # Policy head: prior probabilities over hero actions
        self.policy_head = nn.Linear(256, NUM_HEROES)

        # Value head: estimated win probability for team 0
        # 256 → 128 → 64 → 1 with ReLU + tanh→[0,1]
        self.value_fc1 = nn.Linear(256, 128)
        self.value_fc2 = nn.Linear(128, 64)
        self.value_out = nn.Linear(64, 1)

    def forward(self, x, mask=None):
        # Shared backbone
        h = F.relu(self.input_bn(self.input_fc(x)))
        h = self.res_block1(h)
        h = self.res_block2(h)
        h = self.res_block3(h)
        h = F.relu(self.compress1_bn(self.compress1(h)))
        h = F.relu(self.compress2_bn(self.compress2(h)))

        # Policy head
        policy_logits = self.policy_head(h)
        if mask is not None:
            policy_logits = policy_logits + (1 - mask) * (-1e9)

        # Value head (tanh scaled to [0, 1])
        v = F.relu(self.value_fc1(h))
        v = F.relu(self.value_fc2(v))
        value = torch.tanh(self.value_out(v)) * 0.5 + 0.5  # map [-1,1] → [0,1]

        return policy_logits, value.squeeze(-1)

    def predict(self, state: DraftState, device) -> tuple[np.ndarray, float]:
        """Get policy priors and value for a single state."""
        self.eval()
        x = state.to_tensor(device)
        mask = state.valid_mask(device)
        with torch.no_grad():
            logits, value = self(x, mask)
            priors = F.softmax(logits, dim=1).cpu().numpy()[0]
        return priors, value.item()


# ── MCTS ─────────────────────────────────────────────────────────────

class MCTSNode:
    __slots__ = ['state', 'parent', 'action', 'children', 'visit_count',
                 'value_sum', 'prior', 'is_expanded']

    def __init__(self, state: DraftState, parent=None, action: int = -1, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}  # action → MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False

    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float = 2.0) -> float:
        if self.parent is None:
            return 0.0
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value() + exploration


def mcts_search(
    root_state: DraftState,
    network: AlphaZeroDraftNet,
    wp_model,
    gd_models: list,
    gd_temperature: float,
    device,
    num_simulations: int = 200,
    c_puct: float = 2.0,
) -> np.ndarray:
    """
    Run MCTS from root_state for root_state.our_team's decision.
    The our_team indicator is embedded in the state encoding, so the network
    naturally outputs policy/value from the correct perspective.
    Returns visit count distribution over actions (normalized).
    """
    our_team = root_state.our_team
    root = MCTSNode(root_state)

    # Expand root — network sees our_team in the state encoding
    priors, _ = network.predict(root_state, device)
    valid = root_state.valid_mask_np()
    priors = priors * valid
    prior_sum = priors.sum()
    if prior_sum > 0:
        priors /= prior_sum
    root.is_expanded = True
    for a in range(NUM_HEROES):
        if valid[a] > 0:
            root.children[a] = MCTSNode(None, parent=root, action=a, prior=priors[a])

    for _ in range(num_simulations):
        node = root
        scratch_state = root_state.clone()

        # Selection: traverse tree using UCB
        while node.is_expanded and not scratch_state.is_terminal():
            if scratch_state.current_team() == our_team:
                # Our turn: select by UCB
                best_score = -float('inf')
                best_child = None
                for child in node.children.values():
                    score = child.ucb_score(c_puct)
                    if score > best_score:
                        best_score = score
                        best_child = child
                if best_child is None:
                    break
                team, action_type = DRAFT_ORDER[scratch_state.step]
                scratch_state.apply_action(best_child.action, team, action_type)
                node = best_child
            else:
                # Opponent turn: sample from a random Generic Draft model
                gd_model = random.choice(gd_models)
                x = scratch_state.to_tensor_gd(device)  # 289 dims for GD model
                mask = scratch_state.valid_mask(device)
                with torch.no_grad():
                    logits = gd_model(x, mask)
                    probs = F.softmax(logits / gd_temperature, dim=1)
                    opp_action = torch.multinomial(probs, 1).item()
                team, action_type = DRAFT_ORDER[scratch_state.step]
                scratch_state.apply_action(opp_action, team, action_type)
                # Opponent nodes are pass-through (not tracked in tree)

        if scratch_state.is_terminal():
            # Terminal: evaluate with Win Probability model from our_team's perspective
            value = _evaluate_wp(wp_model, scratch_state, device)
        else:
            # Expansion: expand this node with network predictions
            # Network output is already from our_team's perspective (embedded in state)
            if not node.is_expanded and scratch_state.current_team() == our_team:
                priors_leaf, value = network.predict(scratch_state, device)
                valid_leaf = scratch_state.valid_mask_np()
                priors_leaf = priors_leaf * valid_leaf
                prior_sum = priors_leaf.sum()
                if prior_sum > 0:
                    priors_leaf /= prior_sum
                node.state = scratch_state
                node.is_expanded = True
                for a in range(NUM_HEROES):
                    if valid_leaf[a] > 0:
                        node.children[a] = MCTSNode(
                            None, parent=node, action=a, prior=priors_leaf[a]
                        )
            else:
                # Leaf at opponent's turn or already expanded — use network value
                _, value = network.predict(scratch_state, device)

        # Backpropagation
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    # Build visit count distribution
    visits = np.zeros(NUM_HEROES, dtype=np.float32)
    for action, child in root.children.items():
        visits[action] = child.visit_count
    visit_sum = visits.sum()
    if visit_sum > 0:
        visits /= visit_sum
    return visits


def _evaluate_wp(wp_model, state: DraftState, device) -> float:
    """Get win probability from state.our_team's perspective using the WP model."""
    t0 = torch.from_numpy(state.team0_picks).unsqueeze(0).to(device)
    t1 = torch.from_numpy(state.team1_picks).unsqueeze(0).to(device)
    m = torch.from_numpy(map_to_one_hot(state.game_map)).unsqueeze(0).to(device)
    t = torch.from_numpy(tier_to_one_hot(state.skill_tier)).unsqueeze(0).to(device)
    x = torch.cat([t0, t1, m, t], dim=1)
    with torch.no_grad():
        wp = wp_model(x).item()
    # WP model always outputs P(team 0 wins); flip for team 1
    return wp if state.our_team == 0 else 1.0 - wp


# ── Training ─────────────────────────────────────────────────────────

def load_pretrained_models(device):
    """Load Win Probability model and all Generic Draft model variants."""
    from train_win_probability import WinProbModel
    from train_generic_draft import GenericDraftModel

    wp_path = os.path.join(os.path.dirname(__file__), "win_probability.pt")
    if not os.path.exists(wp_path):
        raise FileNotFoundError("Win Probability model not found. Train it first.")

    wp_model = WinProbModel().to(device)
    wp_model.load_state_dict(torch.load(wp_path, weights_only=True, map_location=device))
    wp_model.eval()

    # Load all Generic Draft variants
    gd_models = []
    for i in range(10):  # try up to 10
        gd_path = os.path.join(os.path.dirname(__file__), f"generic_draft_{i}.pt")
        if not os.path.exists(gd_path):
            break
        gd = GenericDraftModel().to(device)
        gd.load_state_dict(torch.load(gd_path, weights_only=True, map_location=device))
        gd.eval()
        gd_models.append(gd)

    # Fallback to single model
    if not gd_models:
        gd_path = os.path.join(os.path.dirname(__file__), "generic_draft.pt")
        if not os.path.exists(gd_path):
            raise FileNotFoundError("No Generic Draft models found. Train them first.")
        gd = GenericDraftModel().to(device)
        gd.load_state_dict(torch.load(gd_path, weights_only=True, map_location=device))
        gd.eval()
        gd_models.append(gd)

    print(f"Loaded {len(gd_models)} Generic Draft model(s)")
    return wp_model, gd_models


def bootstrap_from_generic_draft(network: AlphaZeroDraftNet, device):
    """Bootstrap policy head weights from a trained Generic Draft model."""
    from train_generic_draft import GenericDraftModel, INPUT_DIM as GD_INPUT_DIM

    gd_path = os.path.join(os.path.dirname(__file__), "generic_draft_0.pt")
    if not os.path.exists(gd_path):
        gd_path = os.path.join(os.path.dirname(__file__), "generic_draft.pt")
    if not os.path.exists(gd_path):
        print("No Generic Draft model to bootstrap from — starting from scratch")
        return

    gd = GenericDraftModel()
    gd.load_state_dict(torch.load(gd_path, weights_only=True, map_location=device))

    # Copy what we can from GD's final layer to the policy head
    with torch.no_grad():
        gd_final = gd.net[-1]  # Linear(128, NUM_HEROES)
        policy_in = network.policy_head.in_features
        gd_in = gd_final.in_features
        if policy_in == gd_in:
            network.policy_head.weight.copy_(gd_final.weight)
            network.policy_head.bias.copy_(gd_final.bias)
            print("Bootstrapped policy head from Generic Draft model (full copy)")
        else:
            # Dimensions differ — copy bias and partial weights
            network.policy_head.bias.copy_(gd_final.bias)
            copy_dim = min(policy_in, gd_in)
            network.policy_head.weight[:, :copy_dim].copy_(gd_final.weight[:, :copy_dim])
            print(f"Bootstrapped policy head (partial: {copy_dim}/{policy_in} input dims + bias)")


def pretrain_value_head(network: AlphaZeroDraftNet, device):
    """Briefly pre-train the value head on replay data with win/loss outcomes."""
    print("Pre-training value head on replay data...")
    data = load_replay_data()  # use all available replay data
    if len(data) < 100:
        print("Not enough data to pre-train value head")
        return

    train_data, _ = split_data(data)

    # Prepare data — each replay produces two examples: one from each team's perspective
    X_list, y_list = [], []
    for d in train_data:
        t0 = heroes_to_multi_hot(d["team0_heroes"])
        t1 = heroes_to_multi_hot(d["team1_heroes"])
        bans = np.zeros(NUM_HEROES, dtype=np.float32)
        for h in d.get("team0_bans", []) + d.get("team1_bans", []):
            from shared import HERO_TO_IDX
            idx = HERO_TO_IDX.get(h)
            if idx is not None:
                bans[idx] = 1.0
        m = map_to_one_hot(d["game_map"])
        t = tier_to_one_hot(d["skill_tier"])
        team0_won = float(d["winner"] == 0)
        # our_team=0 perspective
        x0 = np.concatenate([t0, t1, bans, m, t, [1.0, 1.0, 0.0]])
        X_list.append(x0)
        y_list.append(team0_won)
        # our_team=1 perspective (same board, opposite value target)
        x1 = np.concatenate([t0, t1, bans, m, t, [1.0, 1.0, 1.0]])
        X_list.append(x1)
        y_list.append(1.0 - team0_won)

    X = torch.tensor(np.array(X_list, dtype=np.float32)).to(device)
    y = torch.tensor(np.array(y_list, dtype=np.float32)).to(device)

    # Only train value head parameters (freeze backbone and policy)
    value_params = (list(network.value_fc1.parameters()) +
                    list(network.value_fc2.parameters()) +
                    list(network.value_out.parameters()))
    optimizer = torch.optim.Adam(value_params, lr=1e-3)

    network.train()
    batch_size = 1024
    for epoch in range(30):
        perm = torch.randperm(len(X))
        total_loss = 0
        n_batches = 0
        for i in range(0, len(X), batch_size):
            idx = perm[i:i+batch_size]
            bx, by = X[idx], y[idx]
            _, value_pred = network(bx)
            loss = F.mse_loss(value_pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0:
            print(f"  Value pre-train epoch {epoch+1}: loss={total_loss/n_batches:.4f}")

    print("Value head pre-training complete")


def simulate_draft_with_mcts(
    network: AlphaZeroDraftNet,
    wp_model,
    gd_models: list,
    game_map: str,
    skill_tier: str,
    device,
    our_team: int = 0,
    num_simulations: int = 200,
) -> tuple[float, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Simulate a full draft using MCTS for our_team, Generic Draft for opponent.
    our_team is randomly 0 or 1 so the model learns both sides.
    The our_team indicator is embedded in the state encoding so the network
    knows whose perspective to evaluate from.
    Returns (win_prob, training_examples) where win_prob is from our_team's
    perspective, and each example is (state_features, mcts_policy_target, valid_mask).
    """
    state = DraftState(game_map, skill_tier, our_team=our_team)
    training_examples = []
    gd_temperature = random.choice([0.5, 0.8, 1.0, 1.2, 1.5])

    while not state.is_terminal():
        team, action_type = DRAFT_ORDER[state.step]

        if team == our_team:
            # Our turn: MCTS
            state_features = state.to_numpy()
            valid = state.valid_mask_np()

            visit_dist = mcts_search(
                state, network, wp_model, gd_models, gd_temperature,
                device, num_simulations=num_simulations,
            )

            training_examples.append((state_features, visit_dist, valid))

            # Select action: sample proportional to visit counts
            action = np.random.choice(NUM_HEROES, p=visit_dist) if visit_dist.sum() > 0 else 0
            state.apply_action(action, team, action_type)
        else:
            # Opponent: sample from Generic Draft model
            gd_model = random.choice(gd_models)
            x = state.to_tensor_gd(device)  # 289 dims for GD model
            mask = state.valid_mask(device)
            with torch.no_grad():
                logits = gd_model(x, mask)
                probs = F.softmax(logits / gd_temperature, dim=1)
                action = torch.multinomial(probs, 1).item()
            state.apply_action(action, team, action_type)

    # Final evaluation from our_team's perspective
    win_prob = _evaluate_wp(wp_model, state, device)
    return win_prob, training_examples


def _run_episode(args):
    """Worker function for parallel episode generation."""
    net_state_dict, wp_state_dict, gd_state_dicts, game_map, skill_tier, num_sims = args
    device = torch.device('cpu')

    # Reconstruct models in worker (can't pickle ONNX sessions or models)
    from train_generic_draft import GenericDraftModel
    from train_win_probability import WinProbModel

    network = AlphaZeroDraftNet()
    network.load_state_dict(net_state_dict)
    network.eval()

    wp_model = WinProbModel()
    wp_model.load_state_dict(wp_state_dict)
    wp_model.eval()

    gd_models = []
    for sd in gd_state_dicts:
        gd = GenericDraftModel()
        gd.load_state_dict(sd)
        gd.eval()
        gd_models.append(gd)

    # Randomly play as team 0 or team 1. The our_team indicator is
    # embedded in the state encoding (dim 290) so the network knows
    # whose perspective to evaluate from.
    our_team = random.randint(0, 1)

    win_prob, examples = simulate_draft_with_mcts(
        network, wp_model, gd_models, game_map, skill_tier,
        device, our_team=our_team, num_simulations=num_sims,
    )
    return win_prob, examples


def train():
    # All on CPU — GPU has no speedup for this network size at batch 512
    device = torch.device("cpu")
    print(f"Device: {device}")

    # Detect parallelism
    NUM_WORKERS = min(58, os.cpu_count() or 1)
    print(f"Workers: {NUM_WORKERS}")

    print("Loading pre-trained models...")
    wp_model, gd_models = load_pretrained_models(device)

    # Initialize network — resume from checkpoint if available
    network = AlphaZeroDraftNet().to(device)
    print(f"Network params: {sum(p.numel() for p in network.parameters()):,}")

    checkpoint_path = os.path.join(os.path.dirname(__file__), "draft_policy.pt")
    if os.path.exists(checkpoint_path):
        try:
            network.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
            print(f"Resumed from checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Could not resume (architecture mismatch?): {e}")
            print("Starting fresh with bootstrap + pretrain")
            bootstrap_from_generic_draft(network, device)
            pretrain_value_head(network, device)
    else:
        bootstrap_from_generic_draft(network, device)
        pretrain_value_head(network, device)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150000, eta_min=1e-5)

    # Replay buffer: (state, mcts_policy, valid_mask, value_target)
    buffer = []
    BUFFER_SIZE = 150_000
    BATCH_SIZE = 512
    NUM_EPISODES = 300_000
    MCTS_SIMULATIONS = 200
    EVAL_EVERY = 500
    EVAL_DRAFTS = 200
    PARALLEL_BATCH = NUM_WORKERS  # episodes per parallel batch

    best_eval_wp = 0.0
    save_dir = os.path.dirname(__file__)

    # Pre-extract state dicts for workers (CPU)
    wp_sd = wp_model.state_dict()
    gd_sds = [gd.state_dict() for gd in gd_models]

    print(f"Training for {NUM_EPISODES} episodes with {MCTS_SIMULATIONS} MCTS sims each "
          f"({PARALLEL_BATCH} parallel)...")

    # Initialize wandb
    if HAS_WANDB:
        wandb.init(
            project="hots-draft-policy",
            config={
                "num_episodes": NUM_EPISODES,
                "mcts_simulations": MCTS_SIMULATIONS,
                "buffer_size": BUFFER_SIZE,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "network_params": sum(p.numel() for p in network.parameters()),
                "num_gd_opponents": len(gd_models),
                "replay_data_count": 275_000,
                "architecture": "290→768(3res)→512→256→heads (last input=our_team)",
            },
        )
        print(f"wandb run: {wandb.run.url}")

    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    train_start = time.time()
    episode = 0
    while episode < NUM_EPISODES:
        # Generate a batch of episodes in parallel on CPU
        batch_size = min(PARALLEL_BATCH, NUM_EPISODES - episode)
        # Move network to CPU for serialization to workers
        net_sd = network.state_dict()

        worker_args = [
            (net_sd, wp_sd, gd_sds,
             random.choice(MAPS), random.choice(SKILL_TIERS),
             MCTS_SIMULATIONS)
            for _ in range(batch_size)
        ]

        with mp.Pool(NUM_WORKERS) as pool:
            results = pool.map(_run_episode, worker_args)

        # Collect results into buffer
        last_wp = 0.0
        for win_prob, examples in results:
            for state_feat, mcts_policy, valid in examples:
                buffer.append((state_feat, mcts_policy, valid, win_prob))
                if len(buffer) > BUFFER_SIZE:
                    buffer.pop(0)
            last_wp = win_prob
            episode += 1

        # Train on buffer on GPU (multiple gradient steps per batch of episodes)
        if len(buffer) >= BATCH_SIZE:
            network.train()
            num_train_steps = max(1, batch_size // 2)
            for _ in range(num_train_steps):
                batch = random.sample(buffer, BATCH_SIZE)
                states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(device)
                target_policies = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32).to(device)
                masks = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32).to(device)
                target_values = torch.tensor(np.array([b[3] for b in batch], dtype=np.float32)).to(device)

                pred_logits, pred_values = network(states, masks)
                pred_log_probs = F.log_softmax(pred_logits, dim=1)

                policy_loss = -(target_policies * pred_log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(pred_values, target_values)

                loss = policy_loss + value_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        # Logging
        if episode % 58 < PARALLEL_BATCH or episode >= NUM_EPISODES:
            elapsed = time.time() - train_start
            eps_per_sec = episode / elapsed if elapsed > 0 else 0
            eta_h = (NUM_EPISODES - episode) / eps_per_sec / 3600 if eps_per_sec > 0 else 0
            print(f"Episode {episode}: wp={last_wp:.4f} buffer={len(buffer)} "
                  f"lr={scheduler.get_last_lr()[0]:.6f} "
                  f"[{eps_per_sec:.1f} ep/s, ETA {eta_h:.1f}h]")

            if HAS_WANDB and wandb.run:
                wandb.log({
                    "episode": episode,
                    "last_wp": last_wp,
                    "buffer_size": len(buffer),
                    "lr": scheduler.get_last_lr()[0],
                    "episodes_per_sec": eps_per_sec,
                }, step=episode)

        # Evaluate
        if episode % EVAL_EVERY < PARALLEL_BATCH or episode >= NUM_EPISODES:
            network.eval()
            net_sd_eval = network.state_dict()
            eval_args = [
                (net_sd_eval, wp_sd, gd_sds,
                 random.choice(MAPS), random.choice(SKILL_TIERS),
                 MCTS_SIMULATIONS // 2)
                for _ in range(EVAL_DRAFTS)
            ]
            with mp.Pool(NUM_WORKERS) as pool:
                eval_results = pool.map(_run_episode, eval_args)
            eval_wps = [r[0] for r in eval_results]
            avg_wp = np.mean(eval_wps)
            std_wp = np.std(eval_wps)
            win_rate = np.mean([1.0 if w > 0.5 else 0.0 for w in eval_wps])
            print(f"\n  EVAL @ {episode}: avg_wp={avg_wp:.4f} +/- {std_wp:.4f} "
                  f"win_rate={win_rate:.1%} (vs {len(gd_models)} opponents)")

            if HAS_WANDB and wandb.run:
                wandb.log({
                    "eval/avg_wp": avg_wp,
                    "eval/std_wp": std_wp,
                    "eval/win_rate": win_rate,
                    "eval/best_wp": max(best_eval_wp, avg_wp),
                }, step=episode)

            if avg_wp > best_eval_wp:
                best_eval_wp = avg_wp
                torch.save(network.state_dict(), os.path.join(save_dir, "draft_policy.pt"))
                print(f"  New best! Saved draft_policy.pt")
            print()

    # Export to ONNX
    print("Exporting to ONNX...")
    best_path = os.path.join(save_dir, "draft_policy.pt")
    if os.path.exists(best_path):
        network.load_state_dict(torch.load(best_path, weights_only=True, map_location="cpu"))
    network.cpu().eval()

    dummy_x = torch.randn(1, STATE_DIM)  # 290 features (last = our_team)
    dummy_mask = torch.ones(1, NUM_HEROES)
    onnx_path = os.path.join(save_dir, "draft_policy.onnx")

    torch.onnx.export(
        network, (dummy_x, dummy_mask), onnx_path,
        input_names=["state", "valid_mask"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "state": {0: "batch"},
            "valid_mask": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
    )
    embed_onnx_weights(onnx_path)
    print(f"Exported ONNX model to {onnx_path}")
    print(f"Model size: {os.path.getsize(onnx_path) / 1024:.1f} KB")

    # Optimize + quantize
    print("Optimizing ONNX graph...")
    optimize_onnx(onnx_path)

    print("Quantizing to INT8...")
    calib_data = load_replay_data(limit=2000)
    quant_path = quantize_onnx(onnx_path, calib_data, model_type="policy")

    print("Verifying quantization...")
    verify_quantized_model(onnx_path, quant_path, calib_data, model_type="policy")

    print(f"Best eval win probability: {best_eval_wp:.4f}")
    print(f"Total training time: {(time.time() - train_start) / 3600:.1f}h")

    if HAS_WANDB and wandb.run:
        wandb.log({"final/best_eval_wp": best_eval_wp, "final/model_size_kb": os.path.getsize(onnx_path) / 1024})
        wandb.finish()


if __name__ == "__main__":
    train()

