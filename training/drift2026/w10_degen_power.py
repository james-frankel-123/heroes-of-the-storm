"""
W10 — fully-powered cumprev-vs-stale-4mo degenerate-rate contrast.

Trains 10 additional d2c_cumprev MCTS seeds (s5..s14, exact W4 recipe,
prefix w8_ so w6_head2head's comma-prefix pooling finds them), then runs
the 15x15 head-to-head vs d2b_allhist (whose seeds 5-14 already exist from
W8b), judge-free scoring, and crossed-RE inference on the degen delta.
Completion gating: the pool driver's own return (synchronous), never file
existence (see COAUTHOR_SUMMARY correction log).

Usage:  python3 drift2026/w10_degen_power.py          # everything
"""
import os
import sys
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

common.setup()

import numpy as np

from drift2026.phase_w4_mcts import (build_exclude_ids, cell_stats_build,
                                     MCTS_RUNS_DIR, RERUN_MODELS, WORKER,
                                     SIMS, EPISODES)

HERE = os.path.dirname(os.path.abspath(__file__))


def stage_train():
    exclude_path = build_exclude_ids()
    vf = os.path.join(common.MODELS_DIR, "w4_vf_d2c_cumprev.pt")
    jobs = []
    for seed in range(5, 15):
        name = f"w8_d2c_cumprev_s{seed}"
        save_dir = os.path.join(MCTS_RUNS_DIR, name)
        os.makedirs(save_dir, exist_ok=True)
        env = {
            "MCTS_SAVE_DIR": save_dir,
            "MCTS_WP_MODEL": "enriched_full",
            "MCTS_WP_PATH": vf,
            "MCTS_GD_PATH": os.path.join(RERUN_MODELS, "generic_draft_0.pt"),
            "MCTS_NUM_EPISODES": str(EPISODES),
            "MCTS_NUM_SIMS": str(SIMS),
            "MCTS_BATCH_EPISODES": "128",
            "MCTS_FRESH": "1",
            "MCTS_POLICY_HEAD": "linear",
            "MCTS_NET_SIZE": "base",
            "MCTS_STATS_BUILD": cell_stats_build("d2c_cumprev"),
            "MCTS_EXCLUDE_IDS": exclude_path,
            "WANDB_RUN_NAME": f"drift2026_{name}",
            "WANDB_MODE": "offline",
        }
        jobs.append(common.Job(f"w10_mcts_{name}", [WORKER],
                               [os.path.join(save_dir, "draft_policy.pt")],
                               weight="light", env=env))
    if common.run_pool(jobs):
        sys.exit(1)
    print("W10 TRAINING COMPLETE (pool returned OK)")


def stage_h2h_and_inference():
    subprocess.run([sys.executable, os.path.join(HERE, "w6_head2head.py"),
                    "--arm-a", "d2c_cumprev", "--arm-a-prefix", "w4,w8",
                    "--n-seeds-a", "15",
                    "--arm-b", "d2b_allhist", "--arm-b-prefix", "w4,w8",
                    "--n-seeds-b", "15",
                    "--drafts-per-cell", "10", "--stagger-configs",
                    "--out-suffix", "_w10_degen"],
                   check=True, cwd=os.path.dirname(HERE))
    subprocess.run([sys.executable, os.path.join(HERE, "w6_judgefree.py"),
                    "--suffix", "_w10_degen"], check=True,
                   cwd=os.path.dirname(HERE))

    # Crossed-RE on the paired degen delta, using seed labels in records.
    from shared import is_degenerate
    res = json.load(open(os.path.join(HERE, "results",
                                      "w6_head2head_w10_degen.json")))
    na = nb = 15
    counts = np.zeros((na, nb)); ddeg_sum = np.zeros((na, nb))
    deg_m = np.zeros((na, nb)); deg_f = np.zeros((na, nb))
    for r in res["records"]:
        sa, sb = r["sa"], r["sb"]
        dm = float(is_degenerate(r["maintained"]))
        df = float(is_degenerate(r["frozen"]))
        counts[sa, sb] += 1
        ddeg_sum[sa, sb] += dm - df
        deg_m[sa, sb] += dm
        deg_f[sa, sb] += df
    P = ddeg_sum / counts
    grand = P.mean()
    rowm, colm = P.mean(axis=1), P.mean(axis=0)
    msa = nb * ((rowm - grand) ** 2).sum() / (na - 1)
    msb = na * ((colm - grand) ** 2).sum() / (nb - 1)
    mse = ((P - rowm[:, None] - colm[None, :] + grand) ** 2).sum() / ((na - 1) * (nb - 1))
    # within-pairing sampling var of the mean delta
    m = counts.mean()
    within = 0.5 / m  # conservative Bernoulli-difference bound
    s2a = max((msa - mse) / nb, 0); s2b = max((msb - mse) / na, 0)
    s2e = max(mse - within, 0)
    se = np.sqrt(s2a / na + s2b / nb + s2e / (na * nb) + within / (na * nb))
    out = {
        "n_drafts": int(counts.sum()),
        "degen_maintained_pct": float(deg_m.sum() / counts.sum() * 100),
        "degen_stale_pct": float(deg_f.sum() / counts.sum() * 100),
        "paired_delta_pp": float(grand * 100),
        "se_pp": float(se * 100),
        "z": float(grand / se),
    }
    with open(os.path.join(HERE, "results", "w10_degen_inference.json"), "w") as f:
        json.dump(out, f, indent=2)
    lines = [
        "# W10 — fully-powered cumprev-vs-stale-4mo degen contrast (15x15)",
        "",
        f"- drafts: {out['n_drafts']} (15x15 pairings, both orderings, "
        "stagger-configs)",
        f"- maintained (cumprev) degen: {out['degen_maintained_pct']:.1f}%",
        f"- stale-4mo degen: {out['degen_stale_pct']:.1f}%",
        f"- paired delta: {out['paired_delta_pp']:+.1f} pp ± "
        f"{out['se_pp']:.1f} (crossed-RE), **z = {out['z']:.1f}**",
    ]
    with open(os.path.join(HERE, "results", "W10_DEGEN_POWER.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    stage_train()
    stage_h2h_and_inference()
