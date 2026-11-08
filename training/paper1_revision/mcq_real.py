"""
Mildly Conservative Q-learning (MCQ; Lyu et al., NeurIPS 2022), discrete port,
and the unpenalized control, for paper 1's offline-RL comparison.

The paper's earlier "MCQ" rows (experiment_mcq_draft.py) apply a ReLU
threshold penalty; they are now reported as TCQL. This script trains MCQ as
published, adapted to discrete actions:

  Q = sigmoid(z) over the 90 heroes (CQLDraftAgent, 512-256-128, as every
  offline-RL row). Targets are Monte Carlo returns (game outcome), as for CQL.
  in-distribution term:   BCE(Q(s, a_data), y)
  OOD term (MCQ):         for N_PI actions a' sampled from the learned policy
                          (Boltzmann over z, temperature 1, legal actions),
                          pseudo-target  y' = max_{a ~ mu(.|s), N_MU samples} Q_target(s, a)
                          with mu the behavior model (mean of the 5 GD
                          behavioral-cloning models, renormalized over legal
                          actions) and Q_target a Polyak-averaged copy;
                          loss BCE(Q(s, a'), y')
  total = lam * in + (1 - lam) * OOD          (Lyu et al. weight lambda)

Selection as for every offline-RL row: early stopping on held-out BCE of the
data actions (the p1val validation split), patience 10, at most 50 epochs.
Tournament variant (fixed before any evaluation): the lam with the lowest
held-out BCE.

Control: the same network and data with no penalty (CQL code path, alpha = 0),
trained via rerun2026/train_jobs.py cql_naive --alpha 0.0.

Usage (from training/, with the p1site env of site_rebuild.P1SITE_ENV):
  python3 paper1_revision/mcq_real.py train --lam 0.7
  python3 paper1_revision/mcq_real.py eval  [names...]   # rich evaluation, 5 x 1000 drafts
"""
import os
import sys
import json
import time
import random
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, os.path.join(TRAINING_DIR, "rerun2026"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

N_PI = 10      # policy samples per state for the OOD term
N_MU = 10      # behavior samples per state for the pseudo-target
POLYAK = 0.005
MAX_STEPS = int(os.environ.get("MCQ_MAX_STEPS", "0"))   # smoke tests only


def out_dir():
    from rerun2026 import common
    d = os.path.join(common.MODELS_DIR, "mcq_real")
    os.makedirs(d, exist_ok=True)
    return d


def results_path():
    from paper1_revision import core
    d = os.path.join(core.RESULTS, "mcq_real")
    os.makedirs(d, exist_ok=True)
    return d


def behavior_probs(gd_models, s, m):
    with torch.no_grad():
        p = torch.stack([torch.softmax(g(s, m), dim=1) for g in gd_models]).mean(0)
        p = p * m
        return p / p.sum(1, keepdim=True).clamp_min(1e-12)


def train(lam, epochs=50, batch_size=2048, lr=3e-4, seed=42):
    from rerun2026 import common
    from rerun2026.train_jobs import MemmapCQLDataset
    from rerun2026.phase3_benchmarks import load_gd_models
    from experiment_cql_draft import CQLDraftAgent
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = DataLoader(MemmapCQLDataset(common.CQL_NAIVE_TRAIN), batch_size=batch_size, shuffle=True,
                    num_workers=6, pin_memory=True, persistent_workers=True)
    te = DataLoader(MemmapCQLDataset(common.CQL_NAIVE_TEST), batch_size=batch_size * 2, shuffle=False,
                    num_workers=4, pin_memory=True)
    gd = load_gd_models(dev)
    model, target = CQLDraftAgent().to(dev), CQLDraftAgent().to(dev)
    target.load_state_dict(model.state_dict())
    target.eval()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    path = os.path.join(out_dir(), f"mcq_real_l{lam}.pt")
    best, patience, hist = float("inf"), 0, []
    for ep in range(epochs):
        model.train()
        t0, li, lo, n = time.time(), 0.0, 0.0, 0
        for step, (s, a, y, m) in enumerate(tr):
            if MAX_STEPS and step >= MAX_STEPS:
                break
            s, a, y, m = s.to(dev, non_blocking=True), a.to(dev), y.to(dev), m.to(dev)
            z = model(s, m)
            q = torch.sigmoid(z)
            l_in = F.binary_cross_entropy(q.gather(1, a[:, None]).squeeze(1), y)
            with torch.no_grad():
                mu = behavior_probs(gd, s, m)
                a_mu = torch.multinomial(mu, N_MU, replacement=True)
                qt = torch.sigmoid(target(s, m))
                y_ood = qt.gather(1, a_mu).max(1).values                    # [B]
                pi = torch.softmax(z.detach(), dim=1)
                a_pi = torch.multinomial(pi, N_PI, replacement=True)         # [B, N_PI]
            q_pi = q.gather(1, a_pi).clamp(1e-6, 1 - 1e-6)
            l_ood = F.binary_cross_entropy(q_pi, y_ood[:, None].expand_as(q_pi))
            loss = lam * l_in + (1 - lam) * l_ood
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for p, tp in zip(model.parameters(), target.parameters()):
                    tp.mul_(1 - POLYAK).add_(p, alpha=POLYAK)
            li += l_in.item() * len(s)
            lo += l_ood.item() * len(s)
            n += len(s)
        model.eval()
        tl, tn, match = 0.0, 0, 0
        with torch.no_grad():
            for step, (s, a, y, m) in enumerate(te):
                if MAX_STEPS and step >= MAX_STEPS:
                    break
                s, a, y, m = s.to(dev), a.to(dev), y.to(dev), m.to(dev)
                z = model(s, m)
                tl += F.binary_cross_entropy(torch.sigmoid(z).gather(1, a[:, None]).squeeze(1), y,
                                             reduction="sum").item()
                match += (z.argmax(1) == a).sum().item()
                tn += len(s)
        tl /= tn
        hist.append({"epoch": ep + 1, "train_in": li / n, "train_ood": lo / n, "val_bce": tl,
                     "val_action_match": match / tn, "secs": time.time() - t0})
        print(json.dumps(hist[-1]), flush=True)
        if tl < best:
            best, patience = tl, 0
            torch.save(model.state_dict(), path)
        else:
            patience += 1
            if patience >= 10:
                break
    common.write_meta(f"mcq_real_l{lam}", {"kind": "mcq_real", "lam": lam, "n_pi": N_PI, "n_mu": N_MU,
                                          "polyak": POLYAK, "best_val_bce": best, "history": hist,
                                          "out": path})


def ckpt(name):
    from rerun2026 import common
    if name.startswith("mcq_real_l"):
        return os.path.join(common.MODELS_DIR, "mcq_real", f"{name}.pt")
    if name == "qreg_a0":
        return os.path.join(common.MODELS_DIR, "cql", "_cql_temp_a0.0.pt")
    raise ValueError(name)


def rich_eval(names):
    """Rich evaluation as for the Table IV rows (paper1_revision/deferred.py job_cql_eval)."""
    from collections import Counter
    from experiment_rich_evaluation import (make_cql_strategy, run_drafts_with_strategy,
                                            counter_responsiveness, synergy_exploitation)
    from rerun2026.phase3_benchmarks import load_gd_models
    from shared import MAPS, SKILL_TIERS
    from paper1_revision import core
    from paper1_revision.score_tournament import score_rows
    torch.set_num_threads(int(os.environ.get("MCQ_EVAL_THREADS", "4")))
    dev = torch.device("cpu")
    gd = load_gd_models(dev)
    hp, deploy = core.load_stats("hp"), core.load_stats("deploy")
    for name in names:
        outp = os.path.join(results_path(), f"rich_{name}.json")
        if os.path.exists(outp):
            continue
        fn = make_cql_strategy(ckpt(name), dev)
        random.seed(42)
        cfgs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2) for i in range(1000)]
        drafts = []
        for seed in range(5):
            random.seed(seed)
            torch.manual_seed(seed)
            drafts += run_drafts_with_strategy(fn, cfgs, gd, deploy, dev)
        c = Counter(h for d in drafts for h in d["our_picks"])
        tot = sum(c.values())
        pr = np.array(list(c.values())) / tot
        agree = total = 0
        for d in drafts:
            for st in d["steps"]:
                if not st["is_ours"] or st["type"] == "ban":
                    continue
                s_t = torch.tensor(st["state"], dtype=torch.float32).unsqueeze(0)
                m_t = torch.tensor(st["mask"], dtype=torch.float32).unsqueeze(0)
                votes = Counter()
                with torch.no_grad():
                    for g in gd:
                        votes[g(s_t, m_t).argmax(dim=1).item()] += 1
                agree += st["hero_idx"] == votes.most_common(1)[0][0]
                total += 1
        rows = [(d["our_picks"], d["opp_picks"], d["game_map"], d["tier"]) for d in drafts]
        sc = score_rows(rows)
        out = {"name": name, "n": len(drafts),
               "healer": 100 * np.mean([d["has_healer"] for d in drafts]),
               "degen": 100 * np.mean([d["is_degen"] for d in drafts]),
               "counter": float(np.mean([counter_responsiveness(d["our_picks"], d["opp_picks"], hp, d["tier"])
                                         for d in drafts])),
               "synergy": float(np.mean([synergy_exploitation(d["our_picks"], hp, d["tier"]) for d in drafts])),
               "distinct": len(c), "entropy": float(-(pr * np.log2(pr)).sum()),
               "top10": 100 * sum(v for _, v in c.most_common(10)) / tot,
               "gd_similarity": 100 * agree / max(total, 1),
               "refs": {k: float(np.mean(v)) for k, v in sc.items()}}
        json.dump(out, open(outp, "w"), indent=1)
        print(json.dumps({k: v for k, v in out.items() if k != "refs"}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["train", "eval"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args()
    if a.task == "train":
        train(a.lam, epochs=a.epochs)
    else:
        rich_eval(a.names or ["mcq_real_l0.5", "mcq_real_l0.7", "mcq_real_l0.9", "qreg_a0"])


if __name__ == "__main__":
    main()
