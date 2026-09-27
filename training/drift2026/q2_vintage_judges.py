"""
Q2 — Vintage judge matrix (WORK_QUEUE_PREWRITING.md item Q2).

Trains NAIVE-feature WP judges (hero multi-hot x2 + map + tier one-hots; the
exact qm2026/train_qm_wp.py MLP recipe — no aggregate stats, era-portable) on
ranked pinned-snapshot windows at 5 vintages: train-through cutoffs
2022-07-31, 2023-07-31, 2024-07-31, 2025-07-31, and the W2 cutoff build
2.55.14.95918 (2026-02). Every judge trains on the SAME game count (the
smallest window's volume), sampled recent-first below its cutoff, so judges
differ only in era. 10% replay-level holdout per judge for accuracy.

Then scores the 2000 saved W6 head-to-head drafts (results/w6_head2head.json)
with every judge + the existing QM-2021 judge (qm2026/results/qm_wp_v0.pt):
symmetrized P(maintained side wins).

Outputs:
  results/W6_VINTAGE_MATRIX.json + .md
  models/vintage_judges/wp_<vintage>.pt (+ .meta.json)

Usage (from training/): CUDA_VISIBLE_DEVICES=1 python3 drift2026/q2_vintage_judges.py
"""
import os
import sys
import json
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np
import torch
import torch.nn as nn

from drift2026 import common
from qm2026.train_qm_wp import MLP, featurize, sym_wp, DIM  # exact recipe reuse

SEED = 20260713
JUDGES_DIR = os.path.join(common.MODELS_DIR, "vintage_judges")
W6_PATH = os.path.join(common.RESULTS_DIR, "w6_head2head.json")
QM_CKPT = os.path.join(TRAINING_DIR, "qm2026", "results", "qm_wp_v0.pt")
OUT_JSON = os.path.join(common.RESULTS_DIR, "W6_VINTAGE_MATRIX.json")
OUT_MD = os.path.join(common.RESULTS_DIR, "W6_VINTAGE_MATRIX.md")

EPOCH = datetime.date(1970, 1, 1)


def days(y, m, d):
    return (datetime.date(y, m, d) - EPOCH).days


def day_str(dd):
    return (EPOCH + datetime.timedelta(days=int(dd))).isoformat()


# (name, kind, value): date cutoffs are inclusive game_date <= value;
# the build cutoff is build_idx <= idx(TRAIN_CUTOFF_BUILD).
VINTAGES = [
    ("2022-07", "date", days(2022, 7, 31)),
    ("2023-07", "date", days(2023, 7, 31)),
    ("2024-07", "date", days(2024, 7, 31)),
    ("2025-07", "date", days(2025, 7, 31)),
    ("2026-build", "build", None),
]


def train_judge(name, rows, device, cap):
    """rows: eligible replays (already era-filtered). Samples the `cap` most
    recent (recent-first below the cutoff), splits 90/10 replay-level, trains
    the qm2026 MLP recipe. Returns (model, meta)."""
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    rows = sorted(rows, key=lambda r: (r["date_days"], r["replay_id"]),
                  reverse=True)[:cap]
    d_lo, d_hi = min(r["date_days"] for r in rows), max(r["date_days"] for r in rows)
    print(f"\n=== judge {name}: {len(rows)} games, window "
          f"{day_str(d_lo)} .. {day_str(d_hi)} ===")

    idx = rng.permutation(len(rows))
    n_test = len(rows) // 10
    test_ids = set(idx[:n_test].tolist())

    def build(split):
        X, y = [], []
        for i, g in enumerate(rows):
            if (i in test_ids) != (split == "test"):
                continue
            X.append(featurize(g["team0_heroes"], g["team1_heroes"],
                               g["game_map"], g["skill_tier"]))
            y.append(1.0 if g["winner"] == 0 else 0.0)
            if split == "train":  # team-swap augmentation
                X.append(featurize(g["team1_heroes"], g["team0_heroes"],
                                   g["game_map"], g["skill_tier"]))
                y.append(1.0 - y[-1])
        return np.stack(X), np.array(y, dtype=np.float32)

    Xtr, ytr = build("train")
    Xte, yte = build("test")
    print(f"train {len(Xtr)} (augmented), test {len(Xte)}")

    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    loss_fn = nn.BCELoss()
    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    Xte_t = torch.tensor(Xte, device=device)

    best_acc, best_state, patience = 0.0, None, 0
    for epoch in range(30):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        tot = 0.0
        for i in range(0, len(perm), 4096):
            b = perm[i:i + 4096]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            pte = model(Xte_t).cpu().numpy()
        acc = float(((pte > 0.5) == (yte > 0.5)).mean())
        if acc > best_acc:
            best_acc, best_state, patience = acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        print(f"epoch {epoch}: loss {tot/len(perm):.4f} test acc {acc:.4f}")
        if patience >= 5:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pte = model(Xte_t).cpu().numpy()
    order = np.argsort(pte)
    ranks = np.empty(len(pte)); ranks[order] = np.arange(len(pte))
    pos = yte > 0.5
    auc = float((ranks[pos].mean() - (pos.sum() - 1) / 2) / (~pos).sum())
    print(f"judge {name}: best held-out acc {best_acc:.4f}, AUC {auc:.4f}")

    meta = {
        "vintage": name, "n_games": len(rows), "n_train_aug": int(len(Xtr)),
        "n_test": int(len(Xte)), "window_min_date": day_str(d_lo),
        "window_max_date": day_str(d_hi), "holdout_acc": best_acc, "auc": auc,
        "recipe": "qm2026/train_qm_wp.py MLP, naive features, team-swap aug, "
                  "AdamW 1e-3, 30ep cosine, patience 5", "seed": SEED,
    }
    return model, meta


def score_judge(model, device, records):
    """Symmetrized P(maintained wins) for every W6 record."""
    wps = np.array([
        sym_wp(model, device, r["maintained"], r["frozen"],
               r["game_map"], r["tier"])
        for r in records
    ])
    return {
        "maintained_wp_mean": float(wps.mean()),
        "se": float(wps.std(ddof=1) / np.sqrt(len(wps))),
        "win_share": float((wps > 0.5).mean()),
        "n": len(wps),
    }, wps


def main():
    common.setup()
    os.makedirs(JUDGES_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, DIM={DIM}")

    with open(W6_PATH) as f:
        w6 = json.load(f)
    records = w6["records"]
    assert len(records) == 2000, len(records)

    rows, builds = common.load_data_with_patches()
    cut_idx = common.cutoff_idx(builds)

    # sanity: print one featurization
    r0 = rows[0]
    x0 = featurize(r0["team0_heroes"], r0["team1_heroes"], r0["game_map"], r0["skill_tier"])
    print(f"sample featurization: nnz={int(x0.sum())} (expect 12), dim={len(x0)}, "
          f"map={r0['game_map']}, tier={r0['skill_tier']}")

    # eligible pools per vintage
    pools = {}
    for name, kind, val in VINTAGES:
        if kind == "date":
            pool = [r for r in rows if r["date_days"] <= val]
        else:
            pool = [r for r in rows if r["build_idx"] <= cut_idx]
        pools[name] = pool
        print(f"vintage {name}: eligible pool {len(pool)} games")
    cap = min(len(p) for p in pools.values())
    print(f"volume cap (smallest window): {cap} games per judge")

    matrix = {}
    per_record = {}
    for name, _, _ in [v for v in VINTAGES]:
        model, meta = train_judge(name, pools[name], device, cap)
        torch.save(model.state_dict(), os.path.join(JUDGES_DIR, f"wp_{name}.pt"))
        with open(os.path.join(JUDGES_DIR, f"wp_{name}.meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        scores, wps = score_judge(model, device, records)
        matrix[name] = {**meta, **scores}
        per_record[name] = wps.tolist()
        print(f"judge {name}: maintained WP {scores['maintained_wp_mean']:.4f} "
              f"± {scores['se']:.4f}, win share {scores['win_share']:.3f}")
        del model
        torch.cuda.empty_cache()

    # QM-2021 judge (existing checkpoint, same architecture)
    qm = MLP().to(device)
    qm.load_state_dict(torch.load(QM_CKPT, map_location=device))
    qm.eval()
    qm_meta = json.load(open(os.path.join(TRAINING_DIR, "qm2026", "results", "qm_wp_v0.json")))
    scores, wps = score_judge(qm, device, records)
    matrix["QM-2021"] = {
        "vintage": "QM-2021", "n_games": qm_meta["n_games"],
        "holdout_acc": qm_meta["test_acc"], "auc": qm_meta["auc"],
        "window_min_date": "2021 QM corpus", "window_max_date": "",
        **scores,
    }
    per_record["QM-2021"] = wps.tolist()
    prev = w6.get("qm_judge", {})
    print(f"QM-2021: maintained WP {scores['maintained_wp_mean']:.4f} "
          f"(w6 stored {prev.get('qm_maintained_wp')}), "
          f"win share {scores['win_share']:.3f} (stored {prev.get('win_share')})")

    order = ["QM-2021"] + [v[0] for v in VINTAGES]
    payload = {
        "spec": "WORK_QUEUE_PREWRITING.md Q2 — vintage judge matrix",
        "date": datetime.date.today().isoformat(),
        "n_records": len(records),
        "volume_cap_per_judge": cap,
        "consensus_reference": w6.get("maintained_consensus_wp"),
        "judges": {k: matrix[k] for k in order},
        "per_record_maintained_wp": per_record,
    }
    common.write_json(OUT_JSON, payload)

    lines = [
        "# W6 vintage judge matrix (Q2)",
        "",
        f"Naive-feature WP judges (qm2026 recipe: hero multi-hot ×2 + map + tier"
        f" one-hots), each trained on the {cap:,} most recent ranked games below"
        f" its cutoff (volume-matched to the smallest window), 10% replay-level"
        f" holdout. Scored on the 2,000 saved W6 head-to-head drafts:"
        f" symmetrized P(maintained agent's comp wins).",
        "",
        f"Consensus reference (existing 4-evaluator W6 number): "
        f"{w6.get('maintained_consensus_wp'):.4f}.",
        "",
        "| Judge vintage | Train window | Games | Held-out acc | AUC | Maintained WP | SE | Win share |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in order:
        m = matrix[k]
        win = (f"{m['window_min_date']} .. {m['window_max_date']}"
               if m.get("window_max_date") else m.get("window_min_date", ""))
        lines.append(
            f"| {k} | {win} | {m['n_games']:,} | {m['holdout_acc']:.4f} | "
            f"{m['auc']:.4f} | {m['maintained_wp_mean']:.4f} | "
            f"{m['se']:.4f} | {m['win_share']:.3f} |")
    lines += [
        "",
        f"Judges saved under `drift2026/models/vintage_judges/`. "
        f"Seed {SEED}; recipe identical across vintages (only the era of the "
        f"training window differs).",
        "",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_MD}")

    print("\n=== VINTAGE MATRIX ===")
    for k in order:
        m = matrix[k]
        print(f"  {k:12s} acc {m['holdout_acc']:.4f}  maintained WP "
              f"{m['maintained_wp_mean']:.4f} ± {m['se']:.4f}  "
              f"win share {m['win_share']:.3f}")


if __name__ == "__main__":
    main()
