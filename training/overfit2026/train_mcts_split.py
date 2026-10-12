"""
Launch the UNCHANGED paper MCTS self-play worker (training/train_mcts_worker.py)
against a split-A proxy, so the full paper pipeline (self-play training +
best-by-proxy checkpoint selection) is replicated with a value function that
has never seen half B.

  proxy weights   overfit2026/models/<proxy>.pt   (283-d, 256x128, like wp_enriched_256)
  LUT statistics  WP_STATS_PATH=overfit2026/cache/split_stats/A8.json
  value-head pretraining data: snapshot minus pre-2.55 minus every half-B replay
  opponent / bootstrap: rerun2026/models/generic_draft_0.pt (behavior only)

P1_TIERS=site (site skill tiers; everything under overfit2026/site/):
  proxy weights   site/models/<proxy>.pt, LUTs site/cache/split_stats/<stats>.json,
                  composition table = the pinned Heroes Profile table the
                  proxies' features use (feats.PIN_HP_COMPS), patched into the
                  worker by paper1_revision/train_mcts.py --child (which also
                  seeds Python's random from the run seed)
  value pretrain  lite p1site snapshot minus pre-2.55, half B, the paper's
                  test replays and unknown-tier replays (exclude_path())
  opponent / bootstrap: the half-A GD (split_gd.py), so no half-B game enters
  search          chance-node kernel; refuses a kernel whose composition
                  fallback is not mid_high_low
  runs / logs     site/mcts_runs/<name>, site/logs/mcts_<name>.log
  pause/resume    run under hotsjob (kind mcts: MCTS_PAUSE_FILE and
                  MCTS_CKPT_EVERY_SEC come from the runner); --resume
                  continues from <save>/resume_state.pt (MCTS_FRESH=0)

Usage: python3 overfit2026/train_mcts_split.py <proxy> <sims> <seed_tag> <gpu> [episodes] [stats] [--resume]
  The paper's 8 runs: pA8_leak / pA8_oof x 200 / 800 sims x s0 / s1, 300K episodes.
"""
import os
import sys
import json
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

RUNS = [(px, sims, tag) for sims in (200, 800) for px in ("pA8_leak", "pA8_oof")
        for tag in ("s0", "s1")]


def run_name(proxy, sims, tag):
    return f"{proxy}_{sims}sim_{tag}"


def exclude_path():
    """Site value-pretraining exclusions: pre-2.55, half B, paper test, unknown tier."""
    from overfit2026 import data, split
    from paper1_revision import core
    p = data.art(HERE, "cache", "exclude_pre255_B_test_unknown.json")
    if not os.path.exists(p):
        ids = set(json.load(open(os.path.join(TRAINING_DIR, "rerun2026", "pre255_exclude_ids.json"))))
        if not os.path.exists(core.TEST_IDS):
            raise SystemExit(f"missing {core.TEST_IDS}")
        ids |= core.test_ids()
        for rid, t in data.site_tiers().items():
            if t == "unknown" or split.side_of(rid) == "B":
                ids.add(rid)
        json.dump(sorted(ids), open(p + ".tmp", "w"))
        os.replace(p + ".tmp", p)
    return p


def site_env(proxy, sims, tag, stats, episodes, resume):
    from overfit2026 import data, feats, split_gd
    from paper1_revision.train_mcts import kernel_fallback_order
    name = run_name(proxy, sims, tag)
    order = kernel_fallback_order()
    if order != "mid_high_low":
        raise SystemExit(f"cuda_mcts kernel composition fallback is {order}; rebuild with "
                         "HOTS_COMP_FALLBACK_MID_HIGH_LOW=1 (remote_workers/build_ext.sh)")
    gd = split_gd.model_path(0)
    wp = data.art(HERE, "models", f"{proxy}.pt")
    st = data.art(HERE, "cache", "split_stats", f"{stats}.json")
    for p in (gd, wp, st):
        if not os.path.exists(p):
            raise SystemExit(f"missing input {p}")
    snap = os.path.join(TRAINING_DIR, "snapshots", "replay_snapshot_2026-05-22_1956753_p1site.json")
    lite = snap[:-5] + "_lite.json"
    seed = int(tag.lstrip("s")) if tag.lstrip("s").isdigit() else 0
    return name, data.art(HERE, "mcts_runs", name), data.art(HERE, "logs", f"mcts_{name}.log"), {
        "MCTS_WP_MODEL": "enriched_full",
        "MCTS_WP_PATH": wp,
        "WP_STATS_PATH": st,
        "P1R_COMP_PATH": feats.PIN_HP_COMPS,
        "MCTS_GD_PATH": gd,
        "MCTS_NUM_EPISODES": str(episodes),
        "MCTS_NUM_SIMS": str(sims),
        "MCTS_BATCH_EPISODES": "128",
        "MCTS_FRESH": "0" if resume else "1",
        "MCTS_POLICY_HEAD": "linear",
        "MCTS_NET_SIZE": "base",
        "MCTS_EXCLUDE_IDS": exclude_path(),
        "REPLAY_SNAPSHOT": "1",
        "REPLAY_SNAPSHOT_PATH": lite if os.path.exists(lite) else snap,
        "MCTS_SEARCH_MODE": "chance",
        "WANDB_MODE": "disabled",
        "WANDB_RUN_NAME": f"overfit2026_site_{name}",
        "PYTHONHASHSEED": str(seed),
        "P1R_RUN_SEED": str(seed),
    }


def main():
    argv = [x for x in sys.argv[1:] if x != "--resume"]
    resume = len(argv) != len(sys.argv) - 1
    proxy, sims, tag, gpu = argv[0], int(argv[1]), argv[2], argv[3]
    episodes = int(argv[4]) if len(argv) > 4 else 300000
    stats = argv[5] if len(argv) > 5 else "A8"
    from overfit2026 import data
    if data.TIER_SCHEME == "site":
        name, save, logp, extra = site_env(proxy, sims, tag, stats, episodes, resume)
        env = os.environ.copy()
        env.update(extra)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["MCTS_SAVE_DIR"] = save
        json.dump({"proxy": proxy, "sims": sims, "tag": tag, "episodes": episodes, "stats": stats,
                   "env": extra}, open(os.path.join(save, "run_meta_resume.json" if resume
                                                    else "run_meta.json"), "w"), indent=1)
        log = open(logp, "a" if resume else "w")
        p = subprocess.Popen([sys.executable, "-u",
                              os.path.join(TRAINING_DIR, "paper1_revision", "train_mcts.py"), "--child"],
                             cwd=TRAINING_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
        sys.exit(p.wait())
    name = f"{proxy}_{sims}sim_{tag}"
    save = os.path.join(HERE, "mcts_runs", name)
    os.makedirs(save, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "MCTS_SAVE_DIR": save,
        "MCTS_WP_MODEL": "enriched_full",
        "MCTS_WP_PATH": os.path.join(HERE, "models", f"{proxy}.pt"),
        "WP_STATS_PATH": os.path.join(HERE, "cache", "split_stats", f"{stats}.json"),
        "MCTS_GD_PATH": os.path.join(TRAINING_DIR, "rerun2026", "models", "generic_draft_0.pt"),
        "MCTS_NUM_EPISODES": str(episodes),
        "MCTS_NUM_SIMS": str(sims),
        "MCTS_BATCH_EPISODES": "128",
        "MCTS_FRESH": "0" if resume else "1",
        "MCTS_POLICY_HEAD": "linear",
        "MCTS_NET_SIZE": "base",
        "MCTS_EXCLUDE_IDS": os.path.join(HERE, "cache", "exclude_pre255_and_B.json"),
        "REPLAY_SNAPSHOT": "1",
        "WANDB_MODE": "disabled",
        "WANDB_RUN_NAME": f"overfit2026_{name}",
    })
    log = open(os.path.join(HERE, "logs", f"mcts_{name}.log"), "a" if resume else "w")
    p = subprocess.Popen([sys.executable, "-u", os.path.join(TRAINING_DIR, "train_mcts_worker.py")],
                         cwd=TRAINING_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
    sys.exit(p.wait())


if __name__ == "__main__":
    main()
