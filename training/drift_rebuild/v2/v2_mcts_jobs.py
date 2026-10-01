"""
v2 MCTS arm specifications (prepared, NOT launched: training waits for the
fixed-kernel build). Writes drift_v2/mcts_jobs.json: one entry per agent with
its environment and command, for local or remote (pause-robust) execution.

Design (v2, site tiers): every arm uses an opponent model of its own era,
so no agent's opponent model saw the test period:
  M      maintained: cumulative-previous-build VF, kernel stats through the
         last completed build (2.55.16.96881), opponent gd_cutoff
  Mcut   maintained VF, kernel stats frozen at the cutoff, opponent gd_cutoff
  Md90   decayed-90 shrunk VF (q7_decayed90k100), decayed90k100 stats through
         96881, opponent gd_cutoff
  U      unmaintained: out-of-fold cutoff VF (r2_allhist_oof), cutoff stats,
         opponent gd_cutoff
  S1/S2  out-of-fold VFs at 2.55.9.93613 / 2.55.4.91418, stats and opponent
         of that era
  Mvol   maintained recipe on S2's training-row count (w8 volmatch VF),
         stats through 96881, opponent gd_cutoff
6 agents per arm, two per value-function seed (42, 123, 777); W4 recipe
(200 sims, 300K episodes, linear head, base net, batch 128).
Pause-robust: MCTS_CKPT_EVERY_SEC=540 (resume_state.pt at most 9 minutes
apart), MCTS_FRESH=0 so a restarted job resumes; exit code 75 = paused.

Usage (v2 env): python3 drift_rebuild/v2/run.py drift_rebuild/v2/v2_mcts_jobs.py
"""
import json
import os

import v2env
import torch

from drift2026 import common

T = v2env.TRAINING
M = common.MODELS_DIR
REF, CUT = "2.55.16.96881", "2.55.14.95918"
EXCL = {CUT: os.path.join(T, "drift2026", "w4_exclude_ids.json"),
        "2.55.9.93613": os.path.join(T, "drift_rebuild", "models", "exclude_after_2.55.9.93613.json"),
        "2.55.4.91418": os.path.join(T, "drift_rebuild", "models", "exclude_after_2.55.4.91418.json")}
GD = {CUT: os.path.join(M, "gd_cutoff", "generic_draft_0.pt"),
      "2.55.9.93613": os.path.join(M, "gd_era_2.55.9.93613", "generic_draft_0.pt"),
      "2.55.4.91418": os.path.join(M, "gd_era_2.55.4.91418", "generic_draft_0.pt")}
ARMS = {  # arm -> (vf stem, stats build, stats kind, era)
    "M": ("d2c_cumprev", REF, "cumulative", CUT),
    "Mcut": ("d2c_cumprev", CUT, "cumulative", CUT),
    "Md90": ("q7_decayed90k100", REF, "decayed90k100", CUT),
    "U": ("r2_allhist_oof", CUT, "cumulative", CUT),
    "S1": ("r2_stale1yr_oof", "2.55.9.93613", "cumulative", "2.55.9.93613"),
    "S2": ("r2_stale2yr_oof", "2.55.4.91418", "cumulative", "2.55.4.91418"),
    "Mvol": ("w8_volmatch", REF, "cumulative", CUT),
}
VF_SEEDS = [42, 123, 777]


def export(stem, seed):
    out = os.path.join(M, f"mcts_vf_{stem}_s{seed}.pt")
    src = os.path.join(M, f"{stem}_s{seed}.pt")
    if not os.path.exists(out) and os.path.exists(src):
        ck = torch.load(src, weights_only=True, map_location="cpu")
        torch.save(ck["state_dict"], out)
    return out


def main():
    jobs = []
    for arm, (stem, sb, kind, era) in ARMS.items():
        for i in range(6):
            vf = export(stem, VF_SEEDS[i % 3])
            name = f"v2_{arm}_s{i}"
            save = os.path.join(common.DRIFT_DIR, "mcts_runs", name)
            env = {"MCTS_SAVE_DIR": save, "MCTS_WP_MODEL": "enriched_full", "MCTS_WP_PATH": vf,
                   "MCTS_GD_PATH": GD[era], "MCTS_NUM_EPISODES": "300000", "MCTS_NUM_SIMS": "200",
                   "MCTS_BATCH_EPISODES": "128", "MCTS_FRESH": "0", "MCTS_POLICY_HEAD": "linear",
                   "MCTS_NET_SIZE": "base", "MCTS_STATS_BUILD": sb, "MCTS_STATS_KIND": kind,
                   "MCTS_EXCLUDE_IDS": EXCL[era], "MCTS_CKPT_EVERY_SEC": "540",
                   "WANDB_RUN_NAME": f"drift_v2_{name}", "WANDB_MODE": "offline"}
            jobs.append({"name": name, "arm": arm, "vf_seed": VF_SEEDS[i % 3], "env": env,
                         "done_file": os.path.join(save, "draft_policy.pt"),
                         "vf_exists": os.path.exists(vf), "gd_exists": os.path.exists(GD[era]),
                         "cmd": "python3 -u drift_rebuild/v2/run.py train_mcts_worker.py"})
    path = os.path.join(common.DRIFT_DIR, "mcts_jobs.json")
    json.dump(jobs, open(path, "w"), indent=1)
    print(f"{len(jobs)} agents -> {path}; missing VF: "
          f"{sorted({j['arm'] for j in jobs if not j['vf_exists']})}; missing GD: "
          f"{sorted({j['arm'] for j in jobs if not j['gd_exists']})}")


if __name__ == "__main__":
    main()
