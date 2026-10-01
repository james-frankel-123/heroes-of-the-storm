"""
MCTS self-play with a leak-free value function: runs the UNCHANGED paper
worker (training/train_mcts_worker.py) with

  leaf WP          paper1_revision/models/enriched_s<sel>.pt (283-d, 256x128,
                   trained on out-of-fold statistics; seed chosen by val loss)
  kernel LUTs      own-corpus deploy statistics (train split only), including
                   the own-corpus role-composition table (patched into
                   StatsCache._load_compositions; the worker file is untouched)
  value pretrain   snapshot minus pre-2.55, minus the paper test set, minus the
                   revision's validation split
  opponent         rerun2026/models/generic_draft_0.pt (behavior only), as in
                   the submission

Usage: python3 paper1_revision/train_mcts.py <config> <seed> [--gpu 0] [--wp spec]
  configs: B (200 sims, 300K episodes), F (400), J (800), E (200 sims, 1M),
           K (400 sims, 300K, naive leaf WP: the no-features agent)
  --wp enriched_leak trains against the in-sample-statistics control WP
  (run names <config>_leak_s<seed>)
"""
import os
import sys
import json
import runpy
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)

CONFIGS = {"B": (200, 300000), "F": (400, 300000), "I": (600, 300000),
           "K": (400, 300000),       # no-features agent: naive (hero-identity) leaf WP
           "J": (800, 300000), "E": (200, 1000000)}


GD_PATH = os.path.join(TRAINING_DIR, "rerun2026", "models", "generic_draft_0.pt")


def exclude_path():
    sys.path.insert(0, TRAINING_DIR)
    from paper1_revision import core
    p = os.path.join(core.CACHE, "mcts_pretrain_exclude.json")
    if not os.path.exists(p):
        ids = set(json.load(open(os.path.join(TRAINING_DIR, "rerun2026", "pre255_exclude_ids.json"))))
        ids |= core.test_ids()
        ids |= {g[0] for g in core.split_games()["val"]}
        from overfit2026 import data as odata
        if odata.TIER_SCHEME == "site":   # unranked games are not in the site-tier corpus
            ids |= {r for r, t in odata.site_tiers().items() if t == "unknown"}
        json.dump(sorted(ids), open(p, "w"))
    return p


def worker():
    """Child entry: patch compositions, then run the worker as __main__."""
    sys.path.insert(0, TRAINING_DIR)
    os.chdir(TRAINING_DIR)
    import sweep_enriched_wp as swp
    comp = os.environ["P1R_COMP_PATH"]

    def _load_compositions(self):
        raw = json.load(open(comp))
        self.comp_data = {t: {",".join(sorted(c["roles"])): (c["winRate"], c["games"])
                              for c in cs} for t, cs in raw.items()}
        print(f"compositions: own-corpus table {comp}")
    swp.StatsCache._load_compositions = _load_compositions
    # Python's `random` (used by the worker for opponent and map sampling) was
    # never seeded in the runs reported in the paper (audit P1-C17); seed it
    # from the run seed for runs launched from 2026-10-01 on.
    import random
    random.seed(int(os.environ.get("P1R_RUN_SEED", "0")))
    sys.argv = [os.path.join(TRAINING_DIR, "train_mcts_worker.py")]
    runpy.run_path(sys.argv[0], run_name="__main__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("seed", type=int)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--wp", default=None,
                    help="leaf WP spec in paper1_revision/train_wp.py (default: enriched; "
                         "naive for config K; enriched_leak = in-sample statistics control)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the run's saved checkpoint (the worker saves at each new best eval)")
    a = ap.parse_args()
    sys.path.insert(0, TRAINING_DIR)
    from paper1_revision import core
    import subprocess
    sims, episodes = CONFIGS[a.config]
    from overfit2026 import data as odata
    global GD_PATH
    if odata.TIER_SCHEME == "site":
        # site-tier rebuild: GD pool of rerun2026 namespace p1site; value
        # pretraining reads the site-tier snapshot (same rows)
        GD_PATH = os.path.join(TRAINING_DIR, "rerun2026", "ns", "p1site", "models", "generic_draft_0.pt")
        # value pretraining reads only teams, bans, map, tier and winner, so the
        # lean copy (site_lite_snapshot.py; same rows, same order) gives the
        # same pretraining at a fraction of the memory
        snap = os.path.join(TRAINING_DIR, "snapshots", "replay_snapshot_2026-05-22_1956753_p1site.json")
        lite = snap[:-5] + "_lite.json"
        os.environ["REPLAY_SNAPSHOT_PATH"] = lite if os.path.exists(lite) else snap
    if a.wp is None:
        a.wp = "naive" if a.config == "K" else "enriched"
    meta = json.load(open(os.path.join(core.MODEL_DIR, f"{a.wp}.json")))
    wp = os.path.join(core.MODEL_DIR, f"{a.wp}_s{meta['selected_seed']}.pt")
    wp_type = "true_base" if meta["input_dim"] == 197 else "enriched_full"
    tag = {"enriched": "oof", "naive": "oof", "enriched_leak": "leak"}[a.wp]
    name = f"{a.config}_{tag}_s{a.seed}"
    save = os.path.join(core.MCTS_RUNS, name)
    os.makedirs(save, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(a.gpu),
        "MCTS_SAVE_DIR": save,
        "MCTS_WP_MODEL": wp_type,
        "MCTS_WP_PATH": wp,
        "WP_STATS_PATH": core.stats_path("deploy"),
        "P1R_COMP_PATH": core.comp_path("deploy"),
        "MCTS_GD_PATH": GD_PATH,
        "MCTS_NUM_EPISODES": str(episodes),
        "MCTS_NUM_SIMS": str(sims),
        "MCTS_BATCH_EPISODES": "128",
        "MCTS_FRESH": "0" if a.resume else "1",
        "MCTS_POLICY_HEAD": "linear",
        "MCTS_NET_SIZE": "base",
        "MCTS_EXCLUDE_IDS": exclude_path(),
        "REPLAY_SNAPSHOT": "1",
        "MCTS_SEARCH_MODE": os.environ.get("MCTS_SEARCH_MODE", "chance"),
        "WANDB_MODE": "disabled",
        "WANDB_RUN_NAME": f"paper1_revision_{name}",
        "PYTHONHASHSEED": str(a.seed),
        "P1R_RUN_SEED": str(a.seed),
    })
    json.dump({"config": a.config, "sims": sims, "episodes": episodes, "seed": a.seed,
               "wp": wp, "stats": core.stats_path("deploy")},
              open(os.path.join(save, "run_meta.json" if not a.resume else "run_meta_resume.json"), "w"),
              indent=1)
    log = open(os.path.join(core.LOGS, f"mcts_{name}.log"), "a" if a.resume else "w")
    p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--child"],
                         cwd=TRAINING_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
    sys.exit(p.wait())


if __name__ == "__main__":
    if sys.argv[1:] == ["--child"]:
        worker()
    else:
        main()
