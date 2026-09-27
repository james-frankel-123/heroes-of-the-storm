"""
Q8 — warm-start finetune row for the W2c retrain-policy grid.

One new W2c-comparable policy row: "warm-start finetune every 3 builds +
stats refresh". At each retrain point of the existing K=3 cadence
(positions 11,14,...,41 in the 2.55 build order; 11 retrain points),
instead of a cold full retrain we initialize from the PREVIOUS checkpoint
in the chain (starting from the cold C0 model w2c_cut08_s42) and finetune
for 5 epochs at 10% of the original LR (5e-5 vs 5e-4) on the trailing
6-build window (regime=window --window 6), same cumulative_prev causal
features and seed 42 as the cold K=3 row.

Evaluation is identical to summarize_w2.w2c: per-build accuracy curves ->
games-weighted accuracy over the 36 simulated builds, regret vs the
always-retrain oracle (weights/matrix/oracle read from the existing
results/w2c_policies.json; existing results files are NOT modified).

The chain is sequential by construction (each checkpoint initializes the
next), so trainings run one at a time on a single GPU (~11 short jobs).

Usage:
    python drift2026/q8_finetune_row.py --dry-run
    python drift2026/q8_finetune_row.py [--gpu 1] [--skip-train]
"""
import os
import sys
import json
import glob
import time
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

TRAIN = os.path.join(common.DRIFT_DIR, "train_drift_wp.py")
R = common.RESULTS_DIR
SUBDIR = "w2c_ft"

C0_POS = 8            # keep in sync with phase_w2.py / summarize_w2.py
K = 3
FT_EPOCHS = 5
FT_LR = 5e-5          # 10% of the protocol 5e-4
FT_WINDOW = 6         # trailing builds

W7_GLOB = os.path.join(common.DRIFT_DIR, "mcts_runs", "w7_*", "draft_policy.pt")
W7_NEEDED = 10


def builds_255():
    return [b for b in common.load_patch_index()["builds"]
            if b.startswith("2.55")]


def retrain_points(builds):
    last = len(builds) - 1   # 44
    points = list(range(C0_POS, last, K))
    # actual retrains exclude the initial C0 deployment (matches
    # summarize_w2's retrain count: C0_POS < p < last simulated build)
    return points


def wait_for_w7_gate(poll_s=300):
    """W7 MCTS campaign owns the GPUs; block until its 10 checkpoints exist."""
    while True:
        n = len(glob.glob(W7_GLOB))
        if n >= W7_NEEDED:
            print(f"W7 gate open ({n}/{W7_NEEDED} checkpoints)")
            return
        print(f"W7 gate: {n}/{W7_NEEDED} checkpoints — waiting {poll_s}s",
              flush=True)
        time.sleep(poll_s)


def run_chain(builds, points, gpu, dry_run=False):
    prev_ckpt = os.path.join(common.MODELS_DIR, "w2c_cut08_s42.pt")
    assert os.path.exists(prev_ckpt), prev_ckpt
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    waited = False
    for p in points:
        if p == C0_POS:
            continue   # initial deployment model is the cold C0 model
        name = f"w2c_ft_cut{p:02d}_s42"
        out_json = os.path.join(R, SUBDIR, f"{name}.json")
        out_pt = os.path.join(common.MODELS_DIR, f"{name}.pt")
        cmd = [sys.executable, TRAIN, "--name", name,
               "--features", "cumulative_prev", "--regime", "window",
               "--window", str(FT_WINDOW), "--seed", "42",
               "--cutoff-build", builds[p],
               "--init-from", prev_ckpt,
               "--lr", str(FT_LR), "--max-epochs", str(FT_EPOCHS),
               "--skip-sanity", "--results-subdir", SUBDIR]
        if os.path.exists(out_json) and os.path.exists(out_pt):
            print(f"SKIP {name} (exists)")
        elif dry_run:
            print("DRY  " + " ".join(cmd))
        else:
            if not waited:
                wait_for_w7_gate()
                waited = True
            log = os.path.join(common.LOGS_DIR, f"{name}.log")
            print(f"RUN  {name} (cutoff={builds[p]}, init={os.path.basename(prev_ckpt)})",
                  flush=True)
            with open(log, "w") as lf:
                rc = subprocess.call(cmd, env=env, stdout=lf,
                                     stderr=subprocess.STDOUT)
            if rc != 0:
                print(f"FAIL {name} (rc={rc}) — see {log}")
                sys.exit(1)
        prev_ckpt = out_pt


def summarize(builds, points):
    with open(os.path.join(R, "w2c_policies.json")) as f:
        pol = json.load(f)
    matrix, games = pol["matrix"], pol["games_per_build"]
    oracle = pol["oracle_weighted_acc"]
    bpos = {b: i for i, b in enumerate(builds)}
    Ns = sorted(bpos[b] for b in games)          # 9..44

    def wavg(curve):
        num = sum(games[builds[n]] * curve[n] for n in Ns)
        return num / sum(games[builds[n]] for n in Ns)

    # cold K=3 cross-check straight from the matrix
    cold_curve = {}
    for n in Ns:
        csel = max(p for p in points if p < n)
        cold_curve[n] = matrix[builds[csel]][builds[n]]
    cold_acc = wavg(cold_curve)

    # finetune curve: builds served by C0 use the cold C0 model (chain start);
    # afterwards each segment is served by the finetuned checkpoint at csel
    ft_per_cut, ft_meta = {}, {}
    for p in points:
        if p == C0_POS:
            continue
        with open(os.path.join(R, SUBDIR, f"w2c_ft_cut{p:02d}_s42.json")) as f:
            r = json.load(f)
        ft_per_cut[p] = {bpos[b]: d["acc"]
                         for b, d in r["test_acc_per_build"].items()}
        ft_meta[p] = {"cutoff_build": r["cutoff_build"], "epochs": r["epochs"],
                      "n_train": r["n_train"], "val_acc": r["val_acc"],
                      "minutes": r["minutes"]}
    ft_curve = {}
    for n in Ns:
        csel = max(p for p in points if p < n)
        ft_curve[n] = (matrix[builds[csel]][builds[n]] if csel == C0_POS
                       else ft_per_cut[csel][n])
    ft_acc = wavg(ft_curve)
    n_retrains = len([p for p in points if C0_POS < p < Ns[-1]])

    # per-segment cold-vs-finetune deltas (builds each checkpoint serves)
    seg = []
    for p in points:
        if p == C0_POS:
            continue
        served = [n for n in Ns if max(q for q in points if q < n) == p]
        if not served:
            continue
        g = sum(games[builds[n]] for n in served)
        cold_s = sum(games[builds[n]] * matrix[builds[p]][builds[n]]
                     for n in served) / g
        ft_s = sum(games[builds[n]] * ft_per_cut[p][n] for n in served) / g
        seg.append({"retrain_point": p, "cutoff_build": builds[p],
                    "builds_served": len(served), "games": g,
                    "cold_acc": round(cold_s, 3), "ft_acc": round(ft_s, 3),
                    "delta_pp": round(ft_s - cold_s, 3), **ft_meta[p]})

    row = {"policy": f"warm-start finetune every K={K} builds + refresh "
                     f"({FT_EPOCHS} ep @ lr {FT_LR:g}, trailing "
                     f"{FT_WINDOW}-build window)",
           "retrains": n_retrains,
           "weighted_acc": round(ft_acc, 3),
           "regret_pp": round(oracle - ft_acc, 3)}
    refs = {r["policy"]: r for r in pol["policies"]}
    payload = {
        "row": row,
        "oracle_weighted_acc": oracle,
        "cold_k3_crosscheck": {"weighted_acc": round(cold_acc, 3),
                               "published": refs.get(
                                   "retrain every K=3 builds + refresh")},
        "reference_rows": {k: refs[k] for k in (
            "retrain every K=3 builds + refresh",
            "never retrain + stats refresh") if k in refs},
        "config": {"k": K, "epochs": FT_EPOCHS, "lr": FT_LR,
                   "window_builds": FT_WINDOW, "seed": 42,
                   "features": "cumulative_prev", "chain_init":
                   "w2c_cut08_s42.pt (cold C0 model)"},
        "segments": seg,
        "curve": {builds[n]: ft_curve[n] for n in Ns},
    }
    common.write_json(os.path.join(R, "w2c_finetune_row.json"), payload)

    cold = refs["retrain every K=3 builds + refresh"]
    refresh = refs["never retrain + stats refresh"]
    lines = [
        "# W2c addendum — warm-start finetune row (Q8)",
        "",
        "Same deployment simulation and evaluation as W2C_SUMMARY.md (weights,",
        "accuracy matrix and oracle read from results/w2c_policies.json). New",
        f"policy: at each retrain point of the K={K} cadence, instead of a cold",
        "full retrain, initialize from the previous checkpoint in the chain",
        f"(chain starts at the cold C0 model w2c_cut08_s42) and finetune for",
        f"{FT_EPOCHS} epochs at lr {FT_LR:g} (10% of the protocol 5e-4) on the "
        f"trailing {FT_WINDOW}-build",
        "window, cumulative_prev causal features, seed 42, best-val-loss",
        "checkpoint selection within the finetune epochs.",
        "",
        "| policy | retrains | weighted acc % | regret vs oracle (pp) |",
        "|---|---|---|---|",
        f"| oracle (retrain every build) | 35 | {oracle:.3f} | +0.000 |",
        f"| retrain every K=3 builds + refresh (cold) | {cold['retrains']} | "
        f"{cold['weighted_acc']:.3f} | {cold['regret_pp']:+.3f} |",
        f"| **{row['policy']}** | {row['retrains']} | "
        f"{row['weighted_acc']:.3f} | {row['regret_pp']:+.3f} |",
        f"| never retrain + stats refresh | 0 | {refresh['weighted_acc']:.3f} | "
        f"{refresh['regret_pp']:+.3f} |",
        "",
        f"Cold K=3 cross-check recomputed from the matrix: {cold_acc:.3f} "
        f"(published {cold['weighted_acc']:.3f}).",
        "",
        "## Per-retrain-point comparison (games-weighted over the builds each "
        "checkpoint serves)",
        "",
        "| retrain point | cutoff build | builds served | games | cold acc | "
        "finetune acc | delta (pp) | ft epochs | ft minutes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in seg:
        lines.append(f"| {s['retrain_point']} | {s['cutoff_build']} | "
                     f"{s['builds_served']} | {s['games']:,} | "
                     f"{s['cold_acc']:.3f} | {s['ft_acc']:.3f} | "
                     f"{s['delta_pp']:+.3f} | {s['epochs']} | {s['minutes']} |")
    tot_min = sum(s["minutes"] for s in seg)
    lines += [
        "",
        f"Total finetune compute: {tot_min:.1f} min across "
        f"{len(seg)} sequential jobs (vs {cold['retrains']} full cold "
        "retrains for the K=3 row).",
    ]
    path = os.path.join(R, "W2C_FINETUNE_ROW.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")
    print(f"  {row['policy']}: acc={row['weighted_acc']:.3f} "
          f"regret={row['regret_pp']:+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    common.setup()
    builds = builds_255()
    assert builds[C0_POS] == "2.55.3.89754"
    points = retrain_points(builds)
    print(f"K={K} retrain points: {points}")

    if not args.skip_train:
        run_chain(builds, points, args.gpu, dry_run=args.dry_run)
    if not args.dry_run:
        summarize(builds, points)


if __name__ == "__main__":
    main()
