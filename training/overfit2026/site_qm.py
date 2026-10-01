"""
Quick Match judges on site-scheme tiers (paper-1 rebuild).

The QM export (qm2026/data/qm_games.jsonl) carries skill tiers written with
the pre-2026-09-30 mapping. This script reads each game's stored league_tier
and avg_mmr from qm_games (read-only), relabels with the site rule
(overfit2026.data.site_tier; unknown = no rank and no MMR, dropped), and
retrains both QM judges with their original recipes (qm2026/judge_2026.py and
qm2026/train_qm_wp.py: naive features, team-swap augmentation, AdamW 1e-3,
cosine 30, <=30 epochs, patience 5 on the judge's own 10% hold-out):
  QM2026  games dated >= 2025-07-01 (seed 20260721)
  QM2021  the original v0 pool: the first 91,365 games of the export
          (replay ids up to 41,565,287; seed of train_qm_wp)
Outputs (P1_TIERS=site namespace): overfit2026/site/cache/qm_games_site.jsonl,
overfit2026/site/models/qm_wp_2026.pt, qm_wp_v0.pt (+ .json).

Usage (from training/): P1_TIERS=site nice -n 19 taskset -c 48-53 python3 overfit2026/site_qm.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np

from overfit2026 import data

SRC = os.path.join(TRAINING_DIR, "qm2026", "data", "qm_games.jsonl")
V0_POOL = 91365


def relabeled():
    out = data.art(HERE, "cache", "qm_games_site.jsonl")
    if os.path.exists(out):
        return [json.loads(l) for l in open(out)]
    import psycopg2
    for line in open(os.path.join(TRAINING_DIR, "..", ".env")):
        if line.startswith("DATABASE_URL=") and "DATABASE_URL" not in os.environ:
            os.environ["DATABASE_URL"] = line.split("=", 1)[1].strip().strip('"')
    games = [json.loads(l) for l in open(SRC)]
    for i, g in enumerate(games):
        g["_order"] = i
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute("SELECT replay_id, league_tier, avg_mmr FROM qm_games WHERE replay_id = ANY(%s)",
                ([g["replay_id"] for g in games],))
    lt = {r: (t, m) for r, t, m in cur.fetchall()}
    keep = []
    for g in games:
        t, m = lt[g["replay_id"]]
        g["skill_tier"] = data.site_tier(t, m)
        keep.append(g)
    with open(out, "w") as f:
        for g in keep:
            f.write(json.dumps(g) + "\n")
    return keep


def train(games, seed, device):
    import torch
    import torch.nn as nn
    from qm2026.train_qm_wp import MLP, featurize
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    idx = rng.permutation(len(games))
    test_ids = set(idx[:len(games) // 10].tolist())

    def build(split):
        X, y = [], []
        for i, g in enumerate(games):
            if (i in test_ids) != (split == "test"):
                continue
            X.append(featurize(g["team0_heroes"], g["team1_heroes"], g["game_map"], g["skill_tier"]))
            y.append(1.0 if g["winner"] == 0 else 0.0)
            if split == "train":
                X.append(featurize(g["team1_heroes"], g["team0_heroes"], g["game_map"], g["skill_tier"]))
                y.append(1.0 - y[-1])
        return np.stack(X), np.array(y, dtype=np.float32)
    Xtr, ytr = build("train")
    Xte, yte = build("test")
    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    loss_fn = nn.BCELoss()
    Xtr_t, ytr_t = torch.tensor(Xtr, device=device), torch.tensor(ytr, device=device)
    Xte_t = torch.tensor(Xte, device=device)
    best_acc, best_state, patience = 0.0, None, 0
    for epoch in range(30):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(perm), 4096):
            b = perm[i:i + 4096]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pte = model(Xte_t).cpu().numpy()
        acc = float(((pte > 0.5) == (yte > 0.5)).mean())
        if acc > best_acc:
            best_acc, best_state, patience = acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if patience >= 5:
            break
    model.load_state_dict(best_state)
    return model.cpu().eval(), best_acc


def main():
    import torch
    torch.set_num_threads(4)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    games = relabeled()
    from collections import Counter
    print("tiers", Counter(g["skill_tier"] for g in games), flush=True)
    pools = {"qm_wp_2026": ([g for g in games if (g.get("game_date") or "") >= "2025-07-01"
                             and g["skill_tier"] != "unknown"], 20260721),
             "qm_wp_v0": ([g for g in games if g["_order"] < V0_POOL and g["skill_tier"] != "unknown"],
                          20260711)}
    for name, (pool, seed) in pools.items():
        path = data.art(HERE, "models", f"{name}.pt")
        if os.path.exists(path):
            continue
        model, acc = train(pool, seed, dev)
        torch.save(model.state_dict(), path)
        json.dump({"n_games": len(pool), "seed": seed, "test_acc": acc,
                   "tiers": "site scheme, unknown excluded"}, open(path[:-3] + ".json", "w"), indent=1)
        print(f"{name}: {len(pool):,} games, own hold-out acc {acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
