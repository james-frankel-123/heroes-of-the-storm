"""
Gourdeau & Archambault draft DISCRIMINATOR (IEEE ToG 2021), random-negative
variant (their 'NN' column, which beat their GAN), adapted to the rerun2026
corpus. This is the paper's discriminative manifold model, NOT the Siamese WP
estimator from the HairyBlob repo (that one is train_gourdeau_baseline.py).

Encoding (paper Sec. III-A):
  DraftState: 180 dims = 2 x 90 heroes. First half = team A (the team that
  drafts FIRST), second half = team B. Pick -> +1 in the acting team's half,
  ban -> -1 in the acting team's half. States built incrementally over the
  16-step Storm League draft (3 bans/team; the paper's HGC data had 2-3).
  Per-step acting team is taken from the replay itself (pick: membership in
  team{0,1}_heroes; ban: membership in team{0,1}_bans - the draft_order
  player_slot field is 1/2 for bans, not a real slot). Replays are normalized
  so the first-drafting team occupies the first half; in simulation the
  canonical DRAFT_ORDER has team 0 acting first, so team 0 = team A.
  Input = concat(DraftState 180, map one-hot 14, tier one-hot 3) = 197.

Positives/negatives (paper Sec. III-B, random-sampling variant):
  Positive = every real post-action state X^{t+1} (16 per replay).
  Negative = for each real X^t, ONE uniformly random legal alternative action
  (same step type and team, hero untaken, != the real hero) -> X_g^{t+1}.
  1:1 balance. Train negatives are REGENERATED EVERY EPOCH (closest to the
  paper, where the sampler draws fresh negatives per iteration); the held-out
  negatives use a fixed seed so early stopping / metrics are deterministic.

Architecture (paper Sec. III-C): MLP 197 -> 1024 -> 512 -> 128 -> 1 sigmoid,
  leaky-ReLU hidden activations, dropout 0.4, Adam lr=1e-4, BCE with the
  paper's soft labels (0.05 real / 0.95 generated), separate real and
  generated mini-batches (their GAN-style convention). Output = P(generated).

Documented deviations from the paper:
  * Tier one-hot (3) replaces the patch one-hot: our snapshot is a single
    major patch (2.55), so skill tier is the heterogeneity conditioner.
  * The paper trained 2e8 iterations at batch 100 on 7,630 pro games; we have
    ~1.9M replays, so we train epochs at batch 8192 with early stopping on
    held-out real-vs-generated accuracy (patience 5, cap 20 epochs).
  * L1 regularization: the paper says "all weights are L1-regularized" without
    a coefficient; we use 1e-7 on weight matrices (not biases). A pilot at
    1e-5 was pathological with Adam at lr 1e-4: the constant-sign L1 gradient
    dominates Adam's per-weight normalized updates (batch-averaged BCE
    gradients are far smaller), shrinking ~90% of weights to ~0 and capping
    held-out accuracy at ~53% vs ~61% for 1e-7/0 (which are indistinguishable
    from each other) under otherwise identical settings.
  * 90 heroes / 14 maps (vs. their 82 heroes); 2% replay-level test split
    (rerun2026 convention; paper used 5%).

Data: rerun2026 dataset of record (pinned 2026-05-22 snapshot, patch-2.55
filtered via common.load_data()), replay-level split test_frac=0.02 seed=42.
A compact per-replay cache (16 x hero/team/type + map/tier, ~100 MB) goes to
rerun2026/feature_cache/; full 197-dim int8 sample tensors are materialized
on the GPU (~5 GB per polarity), so no state memmap is needed.

Evaluation:
  1. Paper metrics: (a) held-out real-vs-generated accuracy, (b) fraction of
     uniformly random VALID END-POINT drafts classified generated, (c) top-n
     containment of the real pick at held-out pick steps (rank all legal
     candidate next-states by P(real) desc). Paper pro-HotS references:
     84.3% / 99.9994% / top-3 30.4%.
  2. rerun2026 framework: a drafting strategy that at every step (picks AND
     bans) plays the legal action minimizing P(generated), run through
     phase3_benchmarks.task_rich_eval verbatim (5 seeds x 1000 drafts vs GD
     opponents, Table VII metrics) and appended to
     results/rich_evaluation_results.json under 'gourdeau_discriminator'.
  3. The phase3 sanity suite assumes win-probability semantics, so as a
     qualitative analogue we report P(generated) for the 6 degenerate
     compositions of the sanity suite (vs the STANDARD opponent team,
     Cursed Hollow / mid, no bans - note real endpoints contain 6 bans, so
     these states are slightly off-manifold in absolute terms; a plausible
     non-degenerate control comp is scored the same way for reference).

Usage:
  python rerun2026/train_gourdeau_discriminator.py --dry-run
  CUDA_VISIBLE_DEVICES=3 python rerun2026/train_gourdeau_discriminator.py
  (stages skip themselves when their outputs exist; --force reruns)
"""
import os
import sys
import json
import time
import argparse

# GPU 3 only (shared with the running MCTS campaign; do not touch GPUs 0-2)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rerun2026 import common
from rerun2026.common import MODELS_DIR, RESULTS_DIR, CACHE_DIR

common.setup()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared import (NUM_HEROES, NUM_MAPS, NUM_TIERS, HEROES, HERO_TO_IDX,
                    MAPS, MAP_TO_IDX, SKILL_TIERS, TIER_TO_IDX,
                    map_to_one_hot, tier_to_one_hot)

STATE_DIM = 2 * NUM_HEROES                    # 180
INPUT_DIM = STATE_DIM + NUM_MAPS + NUM_TIERS  # 197

CACHE_TRAIN = os.path.join(CACHE_DIR, "gourdeau_disc_train.npz")
CACHE_TEST = os.path.join(CACHE_DIR, "gourdeau_disc_test.npz")
MODEL_PATH = os.path.join(MODELS_DIR, "gourdeau_discriminator.pt")
RESULT_PATH = os.path.join(RESULTS_DIR, "gourdeau_discriminator.json")
RICH_SLUG = "gourdeau_discriminator"
RICH_PATH = os.path.join(RESULTS_DIR, "rich_eval", f"{RICH_SLUG}.json")
RICH_MERGED = os.path.join(RESULTS_DIR, "rich_evaluation_results.json")

TEST_NEG_SEED = 42        # fixed negatives for the held-out set
L1_COEF = 1e-7            # 1e-5 blocks learning with Adam@1e-4 (see docstring)
LR = 1e-4
BATCH = 8192
MAX_EPOCHS = 20
PATIENCE = 5
LABEL_REAL, LABEL_GEN = 0.05, 0.95   # paper's soft labels; output = P(generated)

# Canonical simulation draft order (team 0 = first drafter = "team A")
from train_draft_policy import DRAFT_ORDER  # noqa: E402
DRAFT_TEAM = [t for t, _ in DRAFT_ORDER]
DRAFT_IS_PICK = [1 if a == "pick" else 0 for _, a in DRAFT_ORDER]

PAPER_REFERENCE_HOTS_NN = {  # paper Table II, NN column (pro HotS)
    "random_drafts": 0.999994, "pro_drafts": 0.843,
    "top10": 0.624, "top5": 0.435, "top3": 0.304, "top1": 0.131,
}

# Plausible (non-degenerate) control comp, disjoint from STANDARD
CONTROL_COMP = ["Johanna", "Malfurion", "Raynor", "Greymane", "Dehaka"]


# ═══════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════

class GourdeauDiscriminator(nn.Module):
    """Paper Sec. III-C: 3 hidden layers {1024,512,128}, leaky ReLU,
    heavy dropout (0.4), sigmoid output = P(generated)."""

    def __init__(self, input_dim=INPUT_DIM, dropout=0.4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.leaky_relu(self.fc1(x)))
        x = self.dropout(F.leaky_relu(self.fc2(x)))
        x = self.dropout(F.leaky_relu(self.fc3(x)))
        return torch.sigmoid(self.out(x)).squeeze(-1)

    def l1_penalty(self):
        return sum(m.weight.abs().sum()
                   for m in (self.fc1, self.fc2, self.fc3, self.out))


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device):
    model = GourdeauDiscriminator().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True,
                                     map_location=device))
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════
# Stage: compact cache
# ═══════════════════════════════════════════════════════════════

def replay_to_compact(r):
    """(heroes[16], team_rel[16], is_pick[16], map_idx, tier_idx) or None.
    team_rel: 0 = team A (the team acting at step 0), 1 = team B."""
    do = r.get("draft_order")
    if not do or len(do) != 16:
        return None
    t0b = set(r.get("team0_bans") or [])
    t1b = set(r.get("team1_bans") or [])
    t0h = set(r.get("team0_heroes") or [])
    t1h = set(r.get("team1_heroes") or [])
    heroes, teams, picks = [], [], []
    for s in do:
        h = HERO_TO_IDX.get(s["hero"])
        if h is None:
            return None
        pick = str(s["type"]) == "1"
        name = s["hero"]
        if pick:
            in0, in1 = name in t0h, name in t1h
        else:
            in0, in1 = name in t0b, name in t1b
        if in0 == in1:  # missing or ambiguous team attribution
            return None
        heroes.append(h)
        teams.append(0 if in0 else 1)
        picks.append(1 if pick else 0)
    if len(set(heroes)) != 16:
        return None
    # sanity: 3 bans + 5 picks per team
    for tm in (0, 1):
        nb = sum(1 for t, p in zip(teams, picks) if t == tm and not p)
        npk = sum(1 for t, p in zip(teams, picks) if t == tm and p)
        if nb != 3 or npk != 5:
            return None
    a_team = teams[0]  # first drafter = "team A" = first half of the vector
    team_rel = [0 if t == a_team else 1 for t in teams]
    map_idx = MAP_TO_IDX.get(r.get("game_map"), -1)
    tier_idx = TIER_TO_IDX.get(r.get("skill_tier") or "mid", 1)
    return heroes, team_rel, picks, map_idx, tier_idx


def build_cache(force=False):
    if not force and os.path.exists(CACHE_TRAIN) and os.path.exists(CACHE_TEST):
        print("cache: exists, skipping")
        return
    train_data, test_data = common.load_split()  # test_frac=0.02, seed=42
    for path, data in ((CACHE_TRAIN, train_data), (CACHE_TEST, test_data)):
        heroes, teams, picks, mapi, tieri = [], [], [], [], []
        dropped = 0
        for r in data:
            c = replay_to_compact(r)
            if c is None:
                dropped += 1
                continue
            heroes.append(c[0])
            teams.append(c[1])
            picks.append(c[2])
            mapi.append(c[3])
            tieri.append(c[4])
        np.savez(path,
                 heroes=np.array(heroes, dtype=np.uint8),
                 team_rel=np.array(teams, dtype=np.uint8),
                 is_pick=np.array(picks, dtype=np.uint8),
                 map_idx=np.array(mapi, dtype=np.int8),
                 tier_idx=np.array(tieri, dtype=np.int8))
        print(f"  {path}: {len(heroes):,} replays kept, {dropped:,} dropped "
              f"({dropped / max(1, len(data)) * 100:.1f}%)")


class Compact:
    """Compact per-replay draft arrays, moved to the training device."""

    def __init__(self, path, device):
        z = np.load(path)
        self.n = len(z["heroes"])
        self.heroes = torch.from_numpy(z["heroes"].astype(np.int64)).to(device)
        self.team = torch.from_numpy(z["team_rel"].astype(np.int64)).to(device)
        self.pick = torch.from_numpy(z["is_pick"].astype(np.int64)).to(device)
        self.map_idx = torch.from_numpy(z["map_idx"].astype(np.int64)).to(device)
        self.tier_idx = torch.from_numpy(z["tier_idx"].astype(np.int64)).to(device)
        self.device = device

    def maptier(self, lo, hi):
        """(C, 17) int8 map+tier one-hots for replays lo:hi."""
        C = hi - lo
        mt = torch.zeros((C, NUM_MAPS + NUM_TIERS), dtype=torch.int8,
                         device=self.device)
        ar = torch.arange(C, device=self.device)
        mi = self.map_idx[lo:hi]
        ok = mi >= 0
        mt[ar[ok], mi[ok]] = 1
        mt[ar, NUM_MAPS + self.tier_idx[lo:hi]] = 1
        return mt


def gen_states(cmp, want_pos, want_neg, seed=0, chunk=131072):
    """Materialize int8 sample tensors on the device.

    Returns (pos, neg): each (n_replays*16, 197) int8 or None. Sample layout:
    row i*16+t = state of replay i after step t (positive) / after replacing
    step t's action with a uniformly random legal alternative (negative)."""
    dev = cmp.device
    N = cmp.n
    pos = torch.empty((N * 16, INPUT_DIM), dtype=torch.int8, device=dev) \
        if want_pos else None
    neg = torch.empty((N * 16, INPUT_DIM), dtype=torch.int8, device=dev) \
        if want_neg else None
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    posv = pos.view(N, 16, INPUT_DIM) if want_pos else None
    negv = neg.view(N, 16, INPUT_DIM) if want_neg else None
    for lo in range(0, N, chunk):
        hi = min(N, lo + chunk)
        C = hi - lo
        hs, tm, pk = cmp.heroes[lo:hi], cmp.team[lo:hi], cmp.pick[lo:hi]
        mt = cmp.maptier(lo, hi)
        S = torch.zeros((C, STATE_DIM), dtype=torch.int8, device=dev)
        taken = torch.zeros((C, NUM_HEROES), dtype=torch.bool, device=dev)
        ar = torch.arange(C, device=dev)
        for t in range(16):
            h, team, sign = hs[:, t], tm[:, t], (pk[:, t] * 2 - 1).to(torch.int8)
            if want_neg:
                r = torch.rand((C, NUM_HEROES), generator=g, device=dev)
                r[taken] = -1.0          # already picked/banned: illegal
                r[ar, h] = -1.0          # must differ from the real action
                alt = r.argmax(dim=1)    # uniform over legal alternatives
                ns = S.clone()
                ns[ar, team * NUM_HEROES + alt] += sign
                negv[lo:hi, t, :STATE_DIM] = ns
                negv[lo:hi, t, STATE_DIM:] = mt
            S[ar, team * NUM_HEROES + h] += sign
            taken[ar, h] = True
            if want_pos:
                posv[lo:hi, t, :STATE_DIM] = S
                posv[lo:hi, t, STATE_DIM:] = mt
    return pos, neg


# ═══════════════════════════════════════════════════════════════
# Stage: train
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(model, X, bs=65536):
    model.eval()
    out = torch.empty(len(X), device=X.device)
    for i in range(0, len(X), bs):
        out[i:i + bs] = model(X[i:i + bs].float())
    return out


@torch.no_grad()
def heldout_accuracy(model, test_pos, test_neg):
    p_pos = predict(model, test_pos)
    p_neg = predict(model, test_neg)
    acc_real = (p_pos < 0.5).float().mean().item()
    acc_gen = (p_neg > 0.5).float().mean().item()
    return (acc_real + acc_gen) / 2, acc_real, acc_gen


def train(args):
    if os.path.exists(MODEL_PATH) and not args.force:
        print(f"train: {MODEL_PATH} exists, skipping")
        return None
    device = _device()
    t0 = time.time()
    tr = Compact(CACHE_TRAIN, device)
    te = Compact(CACHE_TEST, device)
    print(f"train: {tr.n:,} train / {te.n:,} test replays "
          f"({tr.n * 16:,} positives + as many negatives per epoch)")
    pos, _ = gen_states(tr, want_pos=True, want_neg=False)
    test_pos, test_neg = gen_states(te, want_pos=True, want_neg=True,
                                    seed=TEST_NEG_SEED)
    print(f"  positives materialized in {time.time() - t0:.0f}s "
          f"({pos.numel() / 1e9:.1f} GB int8 on {device})")

    torch.manual_seed(42)
    model = GourdeauDiscriminator().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(pos)
    y_real = torch.full((BATCH,), LABEL_REAL, device=device)
    y_gen = torch.full((BATCH,), LABEL_GEN, device=device)

    best = {"acc": -1.0, "epoch": -1, "state": None}
    history = []
    bad_epochs = 0
    for epoch in range(args.epochs):
        ep0 = time.time()
        _, neg = gen_states(tr, want_pos=False, want_neg=True, seed=1000 + epoch)
        perm_p = torch.randperm(n, device=device)
        perm_n = torch.randperm(n, device=device)
        model.train()
        tot_loss, nb = 0.0, 0
        for i in range(0, n - BATCH + 1, BATCH):
            # paper convention: separate mini-batches for real and generated
            for X, idx, y in ((pos, perm_p, y_real), (neg, perm_n, y_gen)):
                xb = X[idx[i:i + BATCH]].float()
                loss = F.binary_cross_entropy(model(xb), y) \
                    + L1_COEF * model.l1_penalty()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot_loss += loss.item()
                nb += 1
        acc, acc_r, acc_g = heldout_accuracy(model, test_pos, test_neg)
        history.append({"epoch": epoch, "loss": tot_loss / nb, "test_acc": acc,
                        "acc_real": acc_r, "acc_generated": acc_g,
                        "seconds": time.time() - ep0})
        print(f"  ep{epoch}: loss={tot_loss / nb:.4f} test_acc={acc * 100:.2f}% "
              f"(real {acc_r * 100:.2f}% / gen {acc_g * 100:.2f}%) "
              f"[{time.time() - ep0:.0f}s]", flush=True)
        if acc > best["acc"]:
            best = {"acc": acc, "epoch": epoch,
                    "state": {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  early stop at epoch {epoch} "
                      f"(best {best['acc'] * 100:.2f}% @ ep{best['epoch']})")
                break
        del neg
        torch.cuda.empty_cache()

    torch.save(best["state"], MODEL_PATH)
    minutes = (time.time() - t0) / 60
    common.write_meta("gourdeau_discriminator", {
        "kind": "gourdeau_discriminator", "input_dim": INPUT_DIM,
        "arch": [1024, 512, 128], "dropout": 0.4, "l1": L1_COEF, "lr": LR,
        "batch": BATCH, "labels": [LABEL_REAL, LABEL_GEN],
        "n_train_replays": tr.n, "n_test_replays": te.n,
        "n_train_samples_per_epoch": tr.n * 32,
        "best_test_acc": best["acc"], "best_epoch": best["epoch"],
        "history": history, "minutes": minutes, "out": MODEL_PATH,
    })
    print(f"train: best acc {best['acc'] * 100:.2f}% -> {MODEL_PATH} "
          f"({minutes:.1f} min)")
    return best["acc"]


# ═══════════════════════════════════════════════════════════════
# Stage: paper metrics
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_random_endpoints(model, device, n=100000, chunk=20000, seed=7):
    """Fraction of uniformly random VALID complete drafts (legal 16-step
    sequences under the canonical draft order, uniform map/tier) classified
    as generated (P(generated) > 0.5)."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    team_t = torch.tensor(DRAFT_TEAM, device=device)
    sign_t = torch.tensor([2 * p - 1 for p in DRAFT_IS_PICK], device=device,
                          dtype=torch.int8)
    n_gen = 0
    for lo in range(0, n, chunk):
        C = min(chunk, n - lo)
        # uniform sequential legal choice == uniform random 16-permutation prefix
        order = torch.argsort(torch.rand((C, NUM_HEROES), generator=g,
                                         device=device), dim=1)[:, :16]
        S = torch.zeros((C, STATE_DIM), dtype=torch.int8, device=device)
        ar = torch.arange(C, device=device)
        for t in range(16):
            S[ar, team_t[t] * NUM_HEROES + order[:, t]] += sign_t[t]
        mt = torch.zeros((C, NUM_MAPS + NUM_TIERS), dtype=torch.int8,
                         device=device)
        mt[ar, torch.randint(0, NUM_MAPS, (C,), generator=g, device=device)] = 1
        mt[ar, NUM_MAPS + torch.randint(0, NUM_TIERS, (C,), generator=g,
                                        device=device)] = 1
        X = torch.cat([S, mt], dim=1)
        n_gen += (predict(model, X) > 0.5).sum().item()
    return n_gen / n


@torch.no_grad()
def eval_topn(model, cmp, chunk=2048):
    """Top-n containment at held-out PICK steps: rank all legal candidate
    next-states by P(real) descending (== P(generated) ascending; rank =
    1 + #candidates strictly better) and check the real pick's rank."""
    dev = cmp.device
    tops = {1: 0, 3: 0, 5: 0, 10: 0}
    total = 0
    idx90 = torch.arange(NUM_HEROES, device=dev)
    for lo in range(0, cmp.n, chunk):
        hi = min(cmp.n, lo + chunk)
        C = hi - lo
        hs, tm, pk = cmp.heroes[lo:hi], cmp.team[lo:hi], cmp.pick[lo:hi]
        mt = cmp.maptier(lo, hi).float()
        S = torch.zeros((C, STATE_DIM), dtype=torch.int8, device=dev)
        taken = torch.zeros((C, NUM_HEROES), dtype=torch.bool, device=dev)
        ar = torch.arange(C, device=dev)
        for t in range(16):
            h, team = hs[:, t], tm[:, t]
            is_pick = pk[:, t].bool()
            if is_pick.any():
                base = torch.cat([S.float(), mt], dim=1)          # (C,197)
                X = base.unsqueeze(1).repeat(1, NUM_HEROES, 1)    # (C,90,197)
                cols = team[:, None] * NUM_HEROES + idx90[None, :]
                X[ar[:, None], idx90[None, :], cols] += 1.0       # pick delta
                p = model(X.view(-1, INPUT_DIM)).view(C, NUM_HEROES)
                p[taken] = float("inf")                           # illegal
                true_p = p[ar, h]
                rank = (p < true_p[:, None]).sum(dim=1) + 1
                rank = rank[is_pick]
                total += len(rank)
                for k in tops:
                    tops[k] += (rank <= k).sum().item()
            S[ar, team * NUM_HEROES + h] += (pk[:, t] * 2 - 1).to(torch.int8)
            taken[ar, h] = True
    return {f"top{k}": v / total for k, v in tops.items()} | \
        {"n_pick_steps": total}


def eval_their_metrics(args):
    device = _device()
    model = load_model(device)
    te = Compact(CACHE_TEST, device)
    test_pos, test_neg = gen_states(te, want_pos=True, want_neg=True,
                                    seed=TEST_NEG_SEED)
    acc, acc_real, acc_gen = heldout_accuracy(model, test_pos, test_neg)
    del test_pos, test_neg
    torch.cuda.empty_cache()
    frac_gen = eval_random_endpoints(model, device, n=args.endpoints)
    topn = eval_topn(model, te)
    out = {
        "real_vs_generated_accuracy": acc,
        "acc_on_real_states": acc_real,       # paper's 'pro drafts' analogue
        "acc_on_generated_states": acc_gen,
        "random_endpoints_frac_generated": frac_gen,
        "random_endpoints_n": args.endpoints,
        "topn_containment": topn,
        "n_test_replays": te.n,
        "paper_reference_hots_nn": PAPER_REFERENCE_HOTS_NN,
    }
    print(f"their-metrics: acc={acc * 100:.2f}% "
          f"(real {acc_real * 100:.2f}%, gen {acc_gen * 100:.2f}%) "
          f"random-endpoints={frac_gen * 100:.4f}% "
          f"top1/3/5/10={topn['top1']:.3f}/{topn['top3']:.3f}/"
          f"{topn['top5']:.3f}/{topn['top10']:.3f}")
    return out


# ═══════════════════════════════════════════════════════════════
# Stage: degenerate compositions (sanity-suite analogue)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_degen(args):
    from experiment_synthetic_augmentation import DEGEN_COMPS, STANDARD
    device = _device()
    model = load_model(device)

    def score(team_a, team_b, game_map="Cursed Hollow", tier="mid"):
        s = np.zeros(INPUT_DIM, dtype=np.float32)
        for h in team_a:
            s[HERO_TO_IDX[h]] = 1.0
        for h in team_b:
            s[NUM_HEROES + HERO_TO_IDX[h]] = 1.0
        s[STATE_DIM:STATE_DIM + NUM_MAPS] = map_to_one_hot(game_map)
        s[STATE_DIM + NUM_MAPS:] = tier_to_one_hot(tier)
        return model(torch.tensor(s, device=device).unsqueeze(0)).item()

    out = {name: score(comp, STANDARD) for name, comp in DEGEN_COMPS.items()}
    out["control (plausible comp)"] = score(CONTROL_COMP, STANDARD)
    out["_note"] = ("P(generated) for endpoint states with 10 picks and NO "
                    "bans (mirroring the WP sanity-suite inputs); real "
                    "endpoints carry 6 bans, so absolute values skew toward "
                    "'generated' - compare against the control row.")
    for k, v in out.items():
        if not k.startswith("_"):
            print(f"  P(generated) {k}: {v:.4f}")
    return out


# ═══════════════════════════════════════════════════════════════
# Stage: rich eval (phase3 machinery, verbatim)
# ═══════════════════════════════════════════════════════════════

def make_disc_strategy(model, device):
    """Strategy for experiment_rich_evaluation.run_drafts_with_strategy:
    at every one of OUR steps (picks and bans) play the legal action whose
    resulting DraftState minimizes P(generated).

    DraftState does not team-attribute bans, so the closure mirrors the
    draft: between two of our calls every new hero belongs to the opponent
    (our own actions are applied to the mirror when we choose them), which
    makes team attribution of the shared ban vector unambiguous."""
    tr = {"step": None}

    def reset():
        tr["signed"] = np.zeros(STATE_DIM, dtype=np.float32)
        tr["t0"] = np.zeros(NUM_HEROES, dtype=np.float32)
        tr["t1"] = np.zeros(NUM_HEROES, dtype=np.float32)
        tr["bans"] = np.zeros(NUM_HEROES, dtype=np.float32)

    def strategy(state, team, step_type, game_map, tier, gd_models, dev):
        if tr["step"] is None or state.step <= tr["step"]:
            reset()  # new draft
        # sync opponent actions since our last call
        opp = 1 - state.our_team
        for new in np.where(state.team0_picks - tr["t0"] > 0)[0]:
            tr["signed"][0 * NUM_HEROES + new] = 1.0
        for new in np.where(state.team1_picks - tr["t1"] > 0)[0]:
            tr["signed"][1 * NUM_HEROES + new] = 1.0
        for new in np.where(state.bans - tr["bans"] > 0)[0]:
            tr["signed"][opp * NUM_HEROES + new] = -1.0
        tr["t0"] = state.team0_picks.copy()
        tr["t1"] = state.team1_picks.copy()
        tr["bans"] = state.bans.copy()

        sign = 1.0 if step_type == "pick" else -1.0
        valid = np.where(state.valid_mask_np() > 0)[0]
        base = np.concatenate([tr["signed"], map_to_one_hot(game_map),
                               tier_to_one_hot(tier)])
        X = np.tile(base, (len(valid), 1))
        X[np.arange(len(valid)), team * NUM_HEROES + valid] += sign
        with torch.no_grad():
            p = model(torch.tensor(X, dtype=torch.float32, device=device))
        hero = int(valid[p.argmin().item()])

        # apply our action to the mirror
        tr["signed"][team * NUM_HEROES + hero] += sign
        if step_type == "pick":
            (tr["t0"] if team == 0 else tr["t1"])[hero] = 1.0
        else:
            tr["bans"][hero] = 1.0
        tr["step"] = state.step
        return hero

    return strategy


def run_rich_eval(args):
    """phase3_benchmarks.task_rich_eval verbatim, with our strategy injected."""
    from rerun2026 import phase3_benchmarks as p3

    orig = p3._rich_strategy

    def patched(name, device, stats, group_indices):
        if name == RICH_SLUG:
            return "Gourdeau discriminator", \
                make_disc_strategy(load_model(device), device)
        return orig(name, device, stats, group_indices)

    p3._rich_strategy = patched
    try:
        ns = argparse.Namespace(strategy=RICH_SLUG, seeds=args.seeds,
                                drafts=args.drafts)
        p3.task_rich_eval(ns)
    finally:
        p3._rich_strategy = orig

    with open(RICH_PATH) as f:
        row = json.load(f)
    # append to the merged Table VII JSON without disturbing existing keys
    merged = {}
    if os.path.exists(RICH_MERGED):
        with open(RICH_MERGED) as f:
            merged = json.load(f)
    merged[RICH_SLUG] = row
    with open(RICH_MERGED, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"rich-eval row appended to {RICH_MERGED} (keys: {list(merged)})")
    return row


# ═══════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════

def load_results():
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH) as f:
            return json.load(f)
    return {}


def save_results(res):
    with open(RESULT_PATH, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"Saved {RESULT_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Gourdeau discriminator: build/train/eval (see docstring)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stage", default="all",
                        choices=["all", "cache", "train", "their-eval",
                                 "degen", "rich"])
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--drafts", type=int, default=1000)
    parser.add_argument("--endpoints", type=int, default=100000)
    parser.add_argument("--limit", type=int, default=None,
                        help="replay cap (smoke tests)")
    args = parser.parse_args()
    if args.limit:
        # smoke test: never pollute the real cache/model/result paths
        common.REPLAY_LIMIT = args.limit
        suffix = f".limit{args.limit}"
        g = globals()
        g["CACHE_TRAIN"] = CACHE_TRAIN + suffix + ".npz"
        g["CACHE_TEST"] = CACHE_TEST + suffix + ".npz"
        g["MODEL_PATH"] = MODEL_PATH + suffix + ".pt"
        g["RESULT_PATH"] = RESULT_PATH + suffix + ".json"
        print(f"--limit {args.limit}: outputs suffixed with {suffix} "
              "(rich stage still writes the real phase3 paths; avoid "
              "--stage rich/all with --limit unless intended)")

    stages = ["cache", "train", "their-eval", "degen", "rich"] \
        if args.stage == "all" else [args.stage]

    if args.dry_run:
        print(f"dry run. device={_device()} "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        res = load_results()
        for s, out, done in (
                ("cache", f"{CACHE_TRAIN} + {CACHE_TEST}",
                 os.path.exists(CACHE_TRAIN) and os.path.exists(CACHE_TEST)),
                ("train", MODEL_PATH, os.path.exists(MODEL_PATH)),
                ("their-eval", f"{RESULT_PATH}[their_metrics]",
                 "their_metrics" in res),
                ("degen", f"{RESULT_PATH}[degenerate_comps_p_generated]",
                 "degenerate_comps_p_generated" in res),
                ("rich", f"{RICH_PATH} + {RICH_MERGED}[{RICH_SLUG}]",
                 os.path.exists(RICH_PATH))):
            flag = "SKIP (done)" if done and not args.force else "RUN "
            mark = flag if s in stages else "----"
            print(f"  [{mark:<11}] {s:<11} -> {out}")
        return

    t_all = time.time()
    res = load_results()
    res.setdefault("config", {
        "model": "MLP 197-1024-512-128-1, leaky ReLU, dropout 0.4, sigmoid",
        "output": "P(generated)", "l1": L1_COEF, "lr": LR, "batch": BATCH,
        "labels_soft": [LABEL_REAL, LABEL_GEN],
        "negatives": "1 uniformly random legal same-team same-type "
                     "alternative action per real state, regenerated each "
                     "epoch (paper's random-sampling 'NN' variant)",
        "encoding": "180d signed DraftState (team A = first drafter) + "
                    "14d map + 3d tier (tier replaces the paper's patch "
                    "one-hot; single-major-patch snapshot)",
        "data": "rerun2026 pinned 2.55 snapshot, replay-level 2% test split",
        "early_stopping": f"held-out real-vs-generated acc, patience "
                          f"{PATIENCE}, cap {MAX_EPOCHS} epochs",
    })
    res.setdefault("wall_time_minutes", {})

    for stage in stages:
        t0 = time.time()
        if stage == "cache":
            build_cache(force=args.force)
        elif stage == "train":
            train(args)
        elif stage == "their-eval":
            if "their_metrics" in res and not args.force:
                print("their-eval: already in results, skipping")
                continue
            res["their_metrics"] = eval_their_metrics(args)
        elif stage == "degen":
            if "degenerate_comps_p_generated" in res and not args.force:
                print("degen: already in results, skipping")
                continue
            res["degenerate_comps_p_generated"] = eval_degen(args)
        elif stage == "rich":
            if os.path.exists(RICH_PATH) and not args.force:
                print(f"rich: {RICH_PATH} exists, reusing")
                with open(RICH_PATH) as f:
                    row = json.load(f)
                merged = {}
                if os.path.exists(RICH_MERGED):
                    with open(RICH_MERGED) as f:
                        merged = json.load(f)
                if RICH_SLUG not in merged:
                    merged[RICH_SLUG] = row
                    with open(RICH_MERGED, "w") as f:
                        json.dump(merged, f, indent=2)
                res["rich_eval"] = row
            else:
                res["rich_eval"] = run_rich_eval(args)
        res["wall_time_minutes"][stage] = round((time.time() - t0) / 60, 2)
        save_results(res)

    res["wall_time_minutes"]["total_this_run"] = round((time.time() - t_all) / 60, 2)
    save_results(res)


if __name__ == "__main__":
    main()
