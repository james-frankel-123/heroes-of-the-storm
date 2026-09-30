"""
R6 — MCTS draft agents for the rebuild (W4 recipe, unchanged:
train_mcts_worker.py, 200 sims, 300K episodes, linear head, base net,
batch 128). What changes per arm is only the value function, the kernel's
statistics (MCTS_STATS_BUILD), the opponent/bootstrap model (MCTS_GD_PATH)
and the value-pretraining exclusion list.

Value-function seeds are NOT selected on the test window (audit A5): agent
seed i uses VF seed VF_SEEDS[i % 3], so every VF seed carries two agent seeds
and the crossed-seed inference charges VF variance to the agent seed.

Arms:
  uoof_rg   leak-free unmaintained VF, cutoff stats, paper-1 GD (rerun2026)
            -> changes ONLY the VF leak relative to the paper's unmaintained
  mref_rg   maintained (cumprev) VF, stats through 2.55.16.96881, paper-1 GD
            -> the paper's maintained agent, with all three VF seeds
  mcut_rg   maintained (cumprev) VF, stats frozen at the cutoff, paper-1 GD
  mref_gc   maintained (cumprev) VF, stats through 2.55.16.96881, cutoff GD
  mcut_gc   maintained (cumprev) VF, stats frozen at the cutoff, cutoff GD
            -> the maintained recipe trained at the cutoff and then frozen
  uoof_gc   leak-free unmaintained VF, cutoff stats, cutoff GD
  s1oof_era leak-free 1-yr-stale VF, stats at 2.55.9.93613, GD of that era
  s2oof_era leak-free 2-yr-stale VF, stats at 2.55.4.91418, GD of that era

Usage:
  python3 drift_rebuild/r6_mcts.py --write-jobs drift_rebuild/jobs_mcts.txt --arms uoof_rg
  python3 drift_rebuild/run_queue.py drift_rebuild/jobs_mcts.txt --gpu --parallel 3
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_common as rb  # noqa: E402

import torch  # noqa: E402

TR = rb.TRAINING_DIR
D2 = os.path.join(TR, "drift2026")
WORKER = os.path.join(TR, "train_mcts_worker.py")
RERUN_GD = os.path.join(TR, "rerun2026", "models", "generic_draft_0.pt")
CUT_GD = os.path.join(D2, "models", "gd_cutoff", "generic_draft_0.pt")
EXCL_2026 = os.path.join(D2, "w4_exclude_ids.json")
CUT = "2.55.14.95918"
REF = "2.55.16.96881"
VF_SEEDS = [42, 123, 777]
N_AGENT_SEEDS = 6


def era_gd(b):
    return os.path.join(rb.MODELS_DIR, f"gd_era_{b}", "generic_draft_0.pt")


def era_excl(b):
    return os.path.join(rb.MODELS_DIR, f"exclude_after_{b}.json")


def vf_src(kind, seed):
    if kind == "cumprev":
        return os.path.join(D2, "models", f"d2c_cumprev_s{seed}.pt")
    return os.path.join(rb.MODELS_DIR, f"r2_{kind}_oof_s{seed}.pt")


ARMS = {  # arm -> (vf kind, stats build, gd path, exclude path)
    "uoof_rg": ("allhist", CUT, RERUN_GD, EXCL_2026),
    "mref_rg": ("cumprev", REF, RERUN_GD, EXCL_2026),
    "mcut_rg": ("cumprev", CUT, RERUN_GD, EXCL_2026),
    # the paper's maintained agent used VF seed 123 only; this arm holds that
    # VF fixed and freezes the kernel statistics at the cutoff, so it differs
    # from the paper's maintained agent ONLY in the statistics it trained on
    "mcut123_rg": ("cumprev", CUT, RERUN_GD, EXCL_2026),
    "mref_gc": ("cumprev", REF, CUT_GD, EXCL_2026),
    "mcut_gc": ("cumprev", CUT, CUT_GD, EXCL_2026),
    "uoof_gc": ("allhist", CUT, CUT_GD, EXCL_2026),
    "s1oof_era": ("stale1yr", "2.55.9.93613", era_gd("2.55.9.93613"),
                  era_excl("2.55.9.93613")),
    "s2oof_era": ("stale2yr", "2.55.4.91418", era_gd("2.55.4.91418"),
                  era_excl("2.55.4.91418")),
}


def export_vf(kind, seed):
    out = os.path.join(rb.MODELS_DIR, f"r6_vf_{kind}_s{seed}.pt")
    if not os.path.exists(out):
        ck = torch.load(vf_src(kind, seed), weights_only=True, map_location="cpu")
        assert not ck.get("embed")
        torch.save(ck["state_dict"], out)
    return out


def write_jobs(arms, path, n_seeds):
    lines = []
    for arm in arms:
        kind, sb, gd, ex = ARMS[arm]
        for i in range(n_seeds):
            vseed = 123 if arm.startswith("mcut123") else VF_SEEDS[i % 3]
            vf = export_vf(kind, vseed)
            name = f"r6_{arm}_s{i}"
            save = os.path.join(rb.RUNS_DIR, name)
            env = {"MCTS_SAVE_DIR": save, "MCTS_WP_MODEL": "enriched_full",
                   "MCTS_WP_PATH": vf, "MCTS_GD_PATH": gd,
                   "MCTS_NUM_EPISODES": "300000", "MCTS_NUM_SIMS": "200",
                   "MCTS_BATCH_EPISODES": "128", "MCTS_FRESH": "1",
                   "MCTS_POLICY_HEAD": "linear", "MCTS_NET_SIZE": "base",
                   "MCTS_STATS_BUILD": sb, "MCTS_EXCLUDE_IDS": ex,
                   "WANDB_RUN_NAME": f"drift_rebuild_{name}",
                   "WANDB_MODE": "offline"}
            envs = " ".join(f"{k}={v}" for k, v in env.items())
            lines.append(f"{os.path.join(save, 'draft_policy.pt')}\t"
                         f"{os.path.join(rb.LOGS_DIR, name + '.log')}\t"
                         f"mkdir -p {save} && env {envs} python3 -u {WORKER}")
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"appended {len(lines)} jobs to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-jobs", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--n-seeds", type=int, default=N_AGENT_SEEDS)
    a = ap.parse_args()
    write_jobs(a.arms.split(","), a.write_jobs, a.n_seeds)
