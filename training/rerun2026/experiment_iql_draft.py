"""
IQL baseline for draft selection (reviewer request): Implicit Q-Learning
(Kostrikov, Nair & Levine, 2022) adapted to our discrete terminal-reward
setting, matching the CQL setup exactly:

  - state: the 289-d naive multi-hot state (team0/team1/bans/map/tier/step),
    read from the rerun2026 phase0 CQL caches (cql_naive_train/test,
    replay-level split, seed 42);
  - Q: 90-way head, sigmoid-bounded to [0,1], trained with BCE against
    Monte Carlo returns (gamma=1, terminal outcome propagated to every step).
    NOTE: bootstrapping is absent BY CONSTRUCTION — the reward is terminal-only
    and the MC target replaces r + gamma*V(s'), exactly as in the CQL runs
    (experiment_cql_draft.train_cql), so IQL's usual TD backup through V
    degenerates to the same MC regression. What remains distinctively IQL is
    (a) the expectile value baseline V_psi and (b) advantage-weighted policy
    extraction, i.e. the machinery designed to avoid over-conservatism;
  - V_psi: expectile regression of V(s) toward Q(s, a_data).detach(), expectile
    tau in {0.7, 0.8, 0.9};
  - policy: 90-way, advantage-weighted regression (AWR):
    E[ exp(beta * (Q(s,a) - V(s))) * -log pi(a|s) ], beta in {1, 3}, weights
    clamped at 100 (standard IQL clip). The policy network uses the exact
    CQLDraftAgent architecture (289 -> 512 -> 256 -> 128 -> 90), so every
    phase3/3b evaluation factory (make_cql_strategy) loads it verbatim;
  - budget/protocol as the phase1 CQL jobs: Adam lr 3e-4 (per network),
    batch 2048, up to 50 epochs, early stopping (patience 10) on the held-out
    Q BCE loss — the exact criterion train_cql used — seed 42, DataLoader
    4 workers over the memmap caches (MemmapCQLDataset).

Grid: 3 tau x 2 beta = 6 cells, each an independent job (Q retrained per cell
for self-containedness, matching the one-job-per-config CQL protocol).

Evaluation:
  screen     1 seed x 200 drafts per cell (rich metrics) -> selection of the
             "best" cell (max aug_wp = cross-model WP by the augmented
             evaluator, Table VII's quality column) + one contrast cell
             (max aug_wp among cells differing in BOTH tau and beta).
  rich-eval  5 seeds x 1000 drafts, exact phase3 rich-eval machinery ->
             results/rich_eval/iql_*.json, appended to
             results/rich_evaluation_results.json under iql_* keys.
  pair       tournament entrant vs {constrained_mcts, mcts, enriched, gd,
             cql_enr_a2.0}, both orderings x 200 drafts, exact phase3b/3c
             machinery -> results/iql/roundrobin/*.json, aggregated into
             results/iql_tournament.json.

Usage:
    python rerun2026/experiment_iql_draft.py --task train --tau 0.7 --beta 1.0
    python rerun2026/experiment_iql_draft.py --train-all          # pool, 6 cells
    python rerun2026/experiment_iql_draft.py --task screen
    python rerun2026/experiment_iql_draft.py --task rich-eval --cell t0.7_b1.0
    python rerun2026/experiment_iql_draft.py --task pair --pair iql__gd
    python rerun2026/experiment_iql_draft.py --eval-all           # pool: screen -> rich + pairs
    python rerun2026/experiment_iql_draft.py --task aggregate
"""
import os
import sys
import json
import time
import random
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rerun2026 import common
from rerun2026.common import MODELS_DIR, RESULTS_DIR, Job, CQL_NAIVE_TRAIN, CQL_NAIVE_TEST

common.setup()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

TAUS = [0.7, 0.8, 0.9]
BETAS = [1.0, 3.0]
CELLS = [(t, b) for t in TAUS for b in BETAS]

IQL_DIR = os.path.join(MODELS_DIR, "iql")
TOURNAMENT_OPPONENTS = ["constrained_mcts", "mcts", "enriched", "gd",
                        "cql_enr_a2.0"]


def cell_slug(tau, beta):
    return f"t{tau}_b{beta}"


def policy_ckpt(tau, beta):
    return os.path.join(IQL_DIR, f"_iql_policy_{cell_slug(tau, beta)}.pt")


def full_ckpt(tau, beta):
    return os.path.join(IQL_DIR, f"_iql_full_{cell_slug(tau, beta)}.pt")


def rpath(*p):
    path = os.path.join(RESULTS_DIR, *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Networks ────────────────────────────────────────────────────────

class ValueNet(nn.Module):
    """V_psi(s): same trunk as CQLDraftAgent, scalar sigmoid output in [0,1]
    (the Q it regresses toward is sigmoid-bounded)."""

    def __init__(self, input_dim=289):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        return torch.sigmoid(self.net(state)).squeeze(-1)


def expectile_loss(diff, tau):
    """L2_tau(u) = |tau - 1(u<0)| * u^2, u = target - V(s)."""
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * diff.pow(2)).mean()


# ── Training ────────────────────────────────────────────────────────

def train_iql(tau, beta, epochs=50, batch_size=2048, lr=3e-4, seed=42,
              max_batches=None):
    from experiment_cql_draft import CQLDraftAgent
    from rerun2026.train_jobs import MemmapCQLDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _device()
    os.makedirs(IQL_DIR, exist_ok=True)

    train_ds = MemmapCQLDataset(CQL_NAIVE_TRAIN)
    test_ds = MemmapCQLDataset(CQL_NAIVE_TEST)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)
    print(f"IQL tau={tau} beta={beta}: {len(train_ds):,} train / "
          f"{len(test_ds):,} test transitions")

    q_net = CQLDraftAgent().to(device)
    v_net = ValueNet().to(device)
    # policy = exact CQLDraftAgent architecture so make_cql_strategy loads it
    pi_net = CQLDraftAgent().to(device)

    q_opt = torch.optim.Adam(q_net.parameters(), lr=lr)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=lr)
    pi_opt = torch.optim.Adam(pi_net.parameters(), lr=lr)

    best_q_loss = float("inf")
    best_metrics = None
    patience = 0
    save_full = full_ckpt(tau, beta)
    save_policy = policy_ckpt(tau, beta)

    for epoch in range(epochs):
        q_net.train(); v_net.train(); pi_net.train()
        tot = Counter()
        n_tot = 0
        for bi, (states, actions, outcomes, masks) in enumerate(train_loader):
            if max_batches and bi >= max_batches:
                break
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            outcomes = outcomes.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # 1) Q <- MC returns (identical to the CQL Bellman term; no
            #    bootstrapping by construction, see module docstring)
            q_bounded = q_net.q_values(states, masks)
            q_taken = q_bounded.gather(1, actions.unsqueeze(1)).squeeze(1)
            q_loss = F.binary_cross_entropy(q_taken, outcomes)
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()

            # 2) V <- expectile regression toward Q(s, a_data)
            with torch.no_grad():
                q_target = q_net.q_values(states, masks).gather(
                    1, actions.unsqueeze(1)).squeeze(1)
            v = v_net(states)
            v_loss = expectile_loss(q_target - v, tau)
            v_opt.zero_grad()
            v_loss.backward()
            v_opt.step()

            # 3) policy <- AWR with weights exp(beta * A), A = Q - V.
            #    ~3e-5 of transitions carry a data action that is invalid
            #    under the state-derived mask (duplicate-hero anomalies in
            #    raw draft records; CQL absorbed them via the BCE clamp) —
            #    their AWR weight is zeroed so the masked logits (-1e9)
            #    cannot distort the loss.
            with torch.no_grad():
                adv = q_target - v_net(states)
                w = torch.exp(beta * adv).clamp(max=100.0)
                w = w * (masks.gather(1, actions.unsqueeze(1))
                         .squeeze(1) > 0.5).float()
            logits = pi_net(states, masks)
            nll = F.cross_entropy(logits, actions, reduction="none")
            pi_loss = (w * nll).mean()
            pi_opt.zero_grad()
            pi_loss.backward()
            pi_opt.step()

            bs = len(states)
            tot["q"] += q_loss.item() * bs
            tot["v"] += v_loss.item() * bs
            tot["pi"] += pi_loss.item() * bs
            tot["w_mean"] += w.mean().item() * bs
            n_tot += bs

        # ── Eval ──
        q_net.eval(); v_net.eval(); pi_net.eval()
        te = Counter()
        n_te = 0
        q_match = pi_match = 0
        with torch.no_grad():
            for bi, (states, actions, outcomes, masks) in enumerate(test_loader):
                if max_batches and bi >= max(1, max_batches // 4):
                    break
                states = states.to(device, non_blocking=True)
                actions = actions.to(device, non_blocking=True)
                outcomes = outcomes.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                q_bounded = q_net.q_values(states, masks)
                q_taken = q_bounded.gather(1, actions.unsqueeze(1)).squeeze(1)
                te["q"] += F.binary_cross_entropy(
                    q_taken, outcomes, reduction="sum").item()
                v = v_net(states)
                te["v"] += expectile_loss(q_taken - v, tau).item() * len(states)
                adv = q_taken - v
                w = torch.exp(beta * adv).clamp(max=100.0)
                w = w * (masks.gather(1, actions.unsqueeze(1))
                         .squeeze(1) > 0.5).float()
                logits = pi_net(states, masks)
                nll = F.cross_entropy(logits, actions, reduction="none")
                te["pi"] += (w * nll).sum().item()
                q_match += (q_bounded.argmax(dim=1) == actions).sum().item()
                pi_match += (logits.argmax(dim=1) == actions).sum().item()
                n_te += len(states)

        q_test = te["q"] / n_te
        metrics = {
            "epoch": epoch + 1,
            "q_test_loss": q_test,
            "v_test_loss": te["v"] / n_te,
            "pi_test_loss": te["pi"] / n_te,
            "q_action_match": q_match / n_te * 100,
            "pi_action_match": pi_match / n_te * 100,
        }
        print(f"  Epoch {epoch+1}: q={tot['q']/n_tot:.4f} v={tot['v']/n_tot:.5f} "
              f"pi={tot['pi']/n_tot:.4f} w_mean={tot['w_mean']/n_tot:.3f} | "
              f"q_test={q_test:.4f} pi_match={metrics['pi_action_match']:.1f}% "
              f"q_match={metrics['q_action_match']:.1f}%", flush=True)

        if q_test < best_q_loss:
            best_q_loss = q_test
            best_metrics = metrics
            torch.save({"q": q_net.state_dict(), "v": v_net.state_dict(),
                        "policy": pi_net.state_dict(),
                        "tau": tau, "beta": beta, "metrics": metrics},
                       save_full)
            torch.save(pi_net.state_dict(), save_policy)
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    common.write_meta(f"iql_{cell_slug(tau, beta)}", {
        "kind": "iql", "tau": tau, "beta": beta,
        "protocol": "CQL phase1 protocol (Adam 3e-4, batch 2048, <=50 epochs, "
                    "early stop patience 10 on held-out Q BCE)",
        "metrics": best_metrics, "out": save_policy, "full": save_full})
    print(f"IQL {cell_slug(tau, beta)}: best {best_metrics}")


# ── Shared rich-metric runner (screen = 1x200, rich-eval = 5x1000) ──

def _rich_run(strategy_fn, label, slug, n_seeds, n_drafts, device, stats):
    """Exact phase3 task_rich_eval computation (metrics identical), shared by
    the screening pass and the full rich evaluation."""
    from rerun2026.phase3_benchmarks import load_gd_models, load_wp
    from sweep_enriched_wp import FEATURE_GROUPS, extract_features
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    from experiment_rich_evaluation import (
        run_drafts_with_strategy, counter_responsiveness, synergy_exploitation,
        draft_diversity, map_adaptation)
    from shared import HEROES, NUM_HEROES

    gd_models = load_gd_models(torch.device("cpu"))
    random.seed(42)
    from shared import MAPS, SKILL_TIERS
    draft_configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
                     for i in range(n_drafts)]

    combined, seed_batches = [], []
    for seed in range(n_seeds):
        random.seed(seed)
        torch.manual_seed(seed)
        drafts = run_drafts_with_strategy(strategy_fn, draft_configs,
                                          gd_models, stats, device)
        combined.extend(drafts)
        seed_batches.append(drafts)
        print(f"  [{label}] seed {seed} done", flush=True)

    n = len(combined)
    healer_rate = sum(d["has_healer"] for d in combined) / n * 100
    degen_rate = sum(d["is_degen"] for d in combined) / n * 100
    counters = [counter_responsiveness(d["our_picks"], d["opp_picks"], stats,
                                       d["tier"]) for d in combined]
    synergies = [synergy_exploitation(d["our_picks"], stats, d["tier"])
                 for d in combined]
    div = draft_diversity(combined)

    gd_agree = gd_total = 0
    for draft in combined:
        for step in draft["steps"]:
            if not step["is_ours"] or step["type"] == "ban":
                continue
            s_t = torch.tensor(step["state"], dtype=torch.float32).unsqueeze(0)
            m_t = torch.tensor(step["mask"], dtype=torch.float32).unsqueeze(0)
            votes = Counter()
            for gd in gd_models:
                with torch.no_grad():
                    votes[gd(s_t, m_t).argmax(dim=1).item()] += 1
            gd_agree += step["hero_idx"] == votes.most_common(1)[0][0]
            gd_total += 1

    # Optional augmented-WP cross score: skipped (and not written) when the
    # namespace has no augmented model (paper-1 site-tier rebuild).
    aug_wp = None
    wps = []
    have_aug = os.path.exists(os.path.join(MODELS_DIR, "wp_aug_v2_512.pt"))
    if have_aug:
        aug_model, aug_cols = load_wp("wp_aug_v2_512.pt", [512, 256, 128],
                                      ENRICHED_GROUPS, device)
        all_mask = [True] * len(FEATURE_GROUPS)
    for d in (combined if have_aug else []):
        t0h = [HEROES[i] for i in range(NUM_HEROES) if d["terminal_t0"][i] > 0]
        t1h = [HEROES[i] for i in range(NUM_HEROES) if d["terminal_t1"][i] > 0]
        rec = {"team0_heroes": t0h, "team1_heroes": t1h,
               "game_map": d["game_map"], "skill_tier": d["tier"], "winner": 0}
        base, enr = extract_features(rec, stats, all_mask)
        x = np.concatenate([base, enr[aug_cols]])
        with torch.no_grad():
            wp = aug_model(torch.tensor(x, dtype=torch.float32
                                        ).unsqueeze(0).to(device)).item()
        wps.append(1 - wp if d["our_team"] == 1 else wp)
    if have_aug:
        aug_wp = float(np.mean(wps))

    seed_metrics = []
    for batch in seed_batches:
        bn = len(batch)
        seed_metrics.append({
            "degen": sum(d["is_degen"] for d in batch) / bn * 100,
            "counter": float(np.mean([counter_responsiveness(
                d["our_picks"], d["opp_picks"], stats, d["tier"])
                for d in batch])),
            "synergy": float(np.mean([synergy_exploitation(
                d["our_picks"], stats, d["tier"]) for d in batch])),
        })

    return {
        "strategy": label, "slug": slug,
        "seeds": n_seeds, "drafts_per_seed": n_drafts,
        "metrics": {
            "healer_rate": round(healer_rate, 1),
            "degen_rate": round(degen_rate, 1),
            "counter": round(float(np.mean(counters)), 2),
            "synergy": round(float(np.mean(synergies)), 2),
            **div,
            "gd_similarity": round(gd_agree / gd_total * 100, 1) if gd_total else 0,
            **({} if aug_wp is None else {"aug_wp": round(aug_wp, 3)}),
            "map_adapt": map_adaptation(combined),
        },
        "seed_metrics": seed_metrics,
        "degen_std": float(np.std([m["degen"] for m in seed_metrics])),
        "counter_std": float(np.std([m["counter"] for m in seed_metrics])),
        "synergy_std": float(np.std([m["synergy"] for m in seed_metrics])),
    }


def _iql_strategy(tau, beta, device):
    """Policy argmax via the verbatim phase3 factory (the checkpoint is a
    CQLDraftAgent state dict, so make_cql_strategy applies unchanged)."""
    from experiment_rich_evaluation import make_cql_strategy
    return make_cql_strategy(policy_ckpt(tau, beta), device)


# ── Tasks ───────────────────────────────────────────────────────────

def task_screen(args):
    """1 seed x 200 drafts per cell -> screening.json + cell selection."""
    device = _device()
    stats = common.stats_cache()
    out = {"criterion": "best = max aug_wp (augmented-WP cross-model score); "
                        "contrast = max aug_wp among cells differing in both "
                        "tau and beta from the best",
           "cells": {}}
    for tau, beta in CELLS:
        slug = cell_slug(tau, beta)
        if not os.path.exists(policy_ckpt(tau, beta)):
            print(f"  missing checkpoint for {slug}, skipping")
            continue
        strategy = _iql_strategy(tau, beta, device)
        r = _rich_run(strategy, f"IQL {slug} (screen)", f"iql_{slug}",
                      1, args.drafts, device, stats)
        meta_path = os.path.join(MODELS_DIR, "meta", f"iql_{slug}.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                r["train_metrics"] = json.load(f).get("metrics")
        out["cells"][slug] = r
        print(f"  {slug}: aug_wp={r['metrics']['aug_wp']} "
              f"healer={r['metrics']['healer_rate']} "
              f"degen={r['metrics']['degen_rate']} "
              f"gd_sim={r['metrics']['gd_similarity']} "
              f"synergy={r['metrics']['synergy']}", flush=True)

    cells = out["cells"]
    if cells:
        best = max(cells, key=lambda k: cells[k]["metrics"]["aug_wp"])
        bt, bb = best.split("_")
        others = {k: v for k, v in cells.items()
                  if not k.startswith(bt) and not k.endswith(bb)}
        contrast = (max(others, key=lambda k: others[k]["metrics"]["aug_wp"])
                    if others else
                    max((k for k in cells if k != best),
                        key=lambda k: cells[k]["metrics"]["aug_wp"]))
        out["best_cell"] = best
        out["contrast_cell"] = contrast
        print(f"Selected: best={best}, contrast={contrast}")

    path = rpath("iql", "screening.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved {path}")


def _selected_cells():
    path = os.path.join(RESULTS_DIR, "iql", "screening.json")
    with open(path) as f:
        s = json.load(f)
    return s["best_cell"], s["contrast_cell"]


def task_rich_eval(args):
    """5 seeds x 1000 drafts for one cell, phase3 machinery."""
    slug = args.cell
    tau, beta = (float(x[1:]) for x in slug.split("_"))
    device = _device()
    stats = common.stats_cache()
    strategy = _iql_strategy(tau, beta, device)
    r = _rich_run(strategy, f"IQL t={tau} b={beta}", f"iql_{slug}",
                  args.seeds, args.drafts, device, stats)
    path = rpath("rich_eval", f"iql_{slug}.json")
    with open(path, "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(f"Saved {path}")
    print(json.dumps(r["metrics"], indent=2, default=str))


def merge_rich_rows():
    """Append iql_* rows into results/rich_evaluation_results.json without
    disturbing existing keys."""
    main_path = os.path.join(RESULTS_DIR, "rich_evaluation_results.json")
    with open(main_path) as f:
        merged = json.load(f)
    d = os.path.join(RESULTS_DIR, "rich_eval")
    added = []
    for fn in sorted(os.listdir(d)):
        if fn.startswith("iql_") and fn.endswith(".json"):
            with open(os.path.join(d, fn)) as f:
                merged[fn[:-5]] = json.load(f)
            added.append(fn[:-5])
    with open(main_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Appended {added} to {main_path} "
          f"({len(merged)} keys total)")


def task_pair(args):
    """One ordered tournament pair, phase3c machinery with an 'iql' actor."""
    from rerun2026 import phase3b_roundrobin as p3b
    from rerun2026 import constrained_search as p3c
    from rerun2026.phase3_benchmarks import load_gd_models
    from shared import HEROES, NUM_HEROES, MAPS, SKILL_TIERS
    from sweep_enriched_wp import compute_group_indices
    from train_draft_policy import DraftState, DRAFT_ORDER

    best, _ = _selected_cells()
    tau, beta = (float(x[1:]) for x in best.split("_"))

    a_name, b_name = args.pair.split("__")
    device = _device()
    stats = common.stats_cache()
    group_indices = compute_group_indices()
    gd_models = load_gd_models(torch.device("cpu"))
    evaluators = p3b.load_wp_evaluators(device)

    def build(name):
        if name == "iql":
            inner = _iql_strategy(tau, beta, device)

            def act(state, team, step_type, game_map, tier):
                return inner(state, team, step_type, game_map, tier,
                             gd_models, device)
            return act
        return p3c.build_actor_ext(name, device, stats, group_indices,
                                   gd_models, args.mcts_run)

    act_a, act_b = build(a_name), build(b_name)

    order = ["iql"] + TOURNAMENT_OPPONENTS
    pair_idx = order.index(a_name) * len(order) + order.index(b_name)
    random.seed(5000 + pair_idx)
    torch.manual_seed(5000 + pair_idx)

    records = []
    for i in range(args.drafts):
        game_map = MAPS[i % len(MAPS)]
        tier = SKILL_TIERS[(i // len(MAPS)) % len(SKILL_TIERS)]
        state = DraftState(game_map, tier, our_team=0)
        picks0, picks1 = [], []
        while not state.is_terminal():
            step_team, step_type = DRAFT_ORDER[state.step]
            step_num = state.step
            state.our_team = step_team
            actor = act_a if step_team == 0 else act_b
            hero_idx = actor(state, step_team, step_type, game_map, tier)
            if step_type == "pick":
                (picks0 if step_team == 0 else picks1).append(
                    (HEROES[hero_idx], step_num))
            state.apply_action(hero_idx, step_team, step_type)

        t0h = [HEROES[j] for j in range(NUM_HEROES) if state.team0_picks[j] > 0]
        t1h = [HEROES[j] for j in range(NUM_HEROES) if state.team1_picks[j] > 0]
        records.append({
            "draft": i, "game_map": game_map, "tier": tier,
            "team0": p3b.side_metrics(picks0, picks1, stats, tier),
            "team1": p3b.side_metrics(picks1, picks0, stats, tier),
            "wp": p3b.score_terminal(t0h, t1h, game_map, tier, evaluators,
                                     stats, device),
            "bans": [HEROES[j] for j in range(NUM_HEROES) if state.bans[j] > 0],
        })
        if (i + 1) % 25 == 0:
            print(f"  [{a_name} vs {b_name}] {i+1}/{args.drafts}", flush=True)

    out = {"team0_strategy": a_name, "team1_strategy": b_name,
           "iql_cell": best, "n_drafts": args.drafts, "records": records}
    path = rpath("iql", "roundrobin", f"{args.pair}.json")
    with open(path, "w") as f:
        json.dump(out, f, default=str)
    print(f"Saved {path}")


def task_aggregate(args):
    from rerun2026 import phase3b_roundrobin as p3b

    # grid summary from training metas
    grid = {}
    for tau, beta in CELLS:
        meta_path = os.path.join(MODELS_DIR, "meta",
                                 f"iql_{cell_slug(tau, beta)}.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                grid[cell_slug(tau, beta)] = json.load(f).get("metrics")

    screening = {}
    spath = os.path.join(RESULTS_DIR, "iql", "screening.json")
    if os.path.exists(spath):
        with open(spath) as f:
            screening = json.load(f)

    out = {"grid_train_metrics": grid,
           "screening_criterion": screening.get("criterion"),
           "best_cell": screening.get("best_cell"),
           "contrast_cell": screening.get("contrast_cell"),
           "opponents": TOURNAMENT_OPPONENTS,
           "mcts_note": "mcts = phase3b default historical checkpoint "
                        "(training/mcts_runs, same resolution as the "
                        "constrained tournament); constrained_mcts = "
                        "rerun2026 J_800sim_s9 + role-constraint mask",
           "pairs": {}, "matchup_summary": {}}

    d = os.path.join(RESULTS_DIR, "iql", "roundrobin")
    evs = list(p3b.WP_EVALUATORS)
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(d, fn)) as f:
                p = json.load(f)
            recs = p["records"]
            n = len(recs)
            s = {"n": n}
            for ev in evs:
                s[f"team0_wp_{ev}_sym"] = float(np.mean(
                    [r["wp"][ev]["sym"] for r in recs]))
            s["team0_wp_consensus"] = float(np.mean(
                [s[f"team0_wp_{ev}_sym"] for ev in evs]))
            for side in ("team0", "team1"):
                s[f"{side}_healer"] = sum(r[side]["has_healer"]
                                          for r in recs) / n * 100
                s[f"{side}_degen"] = sum(r[side]["is_degen"]
                                         for r in recs) / n * 100
                s[f"{side}_synergy"] = float(np.mean(
                    [r[side]["synergy"] for r in recs]))
                s[f"{side}_counter"] = float(np.mean(
                    [r[side]["counter"] for r in recs]))
            out["pairs"][fn[:-5]] = s

    wps, wins, total, per_opp = [], 0, 0, {}
    for key, s in out["pairs"].items():
        a, b = key.split("__")
        if a == "iql":
            wp, opp = s["team0_wp_consensus"], b
        elif b == "iql":
            wp, opp = 1.0 - s["team0_wp_consensus"], a
        else:
            continue
        wps.append(wp)
        wins += wp > 0.5
        total += 1
        per_opp.setdefault(opp, []).append(wp)
    if total:
        out["matchup_summary"] = {
            "ordered_matchups": total, "won": wins,
            "consensus_wp": float(np.mean(wps)),
            "per_opponent_consensus_wp": {k: float(np.mean(v))
                                          for k, v in per_opp.items()},
        }

    path = rpath("iql_tournament.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {path}")
    if out["matchup_summary"]:
        m = out["matchup_summary"]
        print(f"IQL ({out.get('best_cell')}): {m['won']}/{m['ordered_matchups']} "
              f"ordered matchups won, consensus WP {m['consensus_wp']:.3f}")
        for k, v in m["per_opponent_consensus_wp"].items():
            print(f"  vs {k}: {v:.3f}")

    merge_rich_rows()


# ── Orchestrators ───────────────────────────────────────────────────

def me():
    return os.path.abspath(__file__)


def train_jobs():
    jobs = []
    for tau, beta in CELLS:
        jobs.append(Job(f"iql_train_{cell_slug(tau, beta)}",
                        [me(), "--task", "train", "--tau", str(tau),
                         "--beta", str(beta)],
                        [policy_ckpt(tau, beta)], weight="heavy"))
    return jobs


def eval_jobs(args):
    """screen must have run already (selection feeds rich-eval + pairs)."""
    best, contrast = _selected_cells()
    jobs = []
    for slug in dict.fromkeys([best, contrast]):
        jobs.append(Job(f"iql_rich_{slug}",
                        [me(), "--task", "rich-eval", "--cell", slug],
                        [rpath("rich_eval", f"iql_{slug}.json")],
                        weight="light"))
    heavy_opps = {"enriched"}
    for opp in TOURNAMENT_OPPONENTS:
        for pair in (f"iql__{opp}", f"{opp}__iql"):
            jobs.append(Job(f"iql_pair_{pair}",
                            [me(), "--task", "pair", "--pair", pair,
                             "--drafts", str(args.drafts or 200)],
                            [rpath("iql", "roundrobin", f"{pair}.json")],
                            weight="heavy" if opp in heavy_opps else "light"))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None,
                        choices=[None, "train", "screen", "rich-eval", "pair",
                                 "aggregate"])
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--eval-all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="smoke tests only: cap batches per epoch")
    parser.add_argument("--cell", default=None, help="rich-eval cell slug")
    parser.add_argument("--pair", default=None, help="ordered pair A__B")
    parser.add_argument("--drafts", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--mcts-run", default=None)
    args = parser.parse_args()

    defaults = {"screen": 200, "rich-eval": 1000, "pair": 200}
    if args.drafts is None:
        args.drafts = defaults.get(args.task, 200)

    if args.train_all:
        failed = common.run_pool(train_jobs(), dry_run=args.dry_run,
                                 force=args.force, only=args.only)
        if failed:
            sys.exit(1)
        return
    if args.eval_all:
        if not os.path.exists(os.path.join(RESULTS_DIR, "iql", "screening.json")) \
                or args.force:
            task_screen(argparse.Namespace(drafts=200))
        failed = common.run_pool(eval_jobs(args), dry_run=args.dry_run,
                                 force=args.force, only=args.only)
        if args.dry_run:
            return
        task_aggregate(args)
        if failed:
            sys.exit(1)
        return

    if args.dry_run:
        print(f"dry run: task={args.task} {vars(args)}")
        return

    t0 = time.time()
    if args.task == "train":
        train_iql(args.tau, args.beta, epochs=args.epochs,
                  max_batches=args.max_batches)
    else:
        fn = {"screen": task_screen, "rich-eval": task_rich_eval,
              "pair": task_pair, "aggregate": task_aggregate}[args.task]
        fn(args)
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
