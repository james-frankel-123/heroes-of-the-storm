"""
oct2026 expert-study refresh driver (namespace rerun2026/ns/oct2026).

Rebuilds every tournament strategy on a fresh 2026-09-01 snapshot of the
relabeled DB (site-scheme tiers) with the paper-1 revision's leak-free
machinery, then replays the tournament. Differences from the September (v5)
run are listed in DIFFERENCES below and in docs.

A dependency-aware scheduler under the resource cap: at most 2 concurrent
GPU jobs of this lane (all on GPU_ID) and never more than 4 HotS GPU processes
machine-wide; one CPU stage at a time (pools of 4); everything under
nice 19 / cores 48-63. Each job is skipped when its output exists, so the
driver can be re-run after any interruption.

Usage (from training/): python3 paper1_revision/oct2026_refresh.py [--gpu 3]
"""
import os
import sys
import json
import time
import glob
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
RERUN = os.path.join(TRAINING_DIR, "rerun2026")
NS = "oct2026"
NS_DIR = os.path.join(RERUN, "ns", NS)
MCTS_RUN = "F_400sim_s0"
PHASE1 = ["gd_0", "gd_1", "gd_2", "gd_3", "gd_4", "cql_naive_a1.0", "cql_enr_a2.0",
          "mcq_t0.5", "gourdeau_wp"]
DIFFERENCES = [
    "snapshot: relabeled site-scheme tiers (low = Bronze+Silver, mid = Gold+Platinum, "
    "high = Diamond+Master); 'unknown'-tier rows excluded",
    "statistics: decayed90 recipe unchanged, but computed from training rows only and "
    "out of fold for training rows (5 hash folds); own-corpus role-composition table "
    "replaces the external Heroes Profile compositions.json everywhere",
    "WP models (4 evaluators / greedy value functions): early stopping and seed choice on "
    "a validation subset of training rows (September: test set)",
    "enriched CQL transitions: out-of-fold features",
    "ensemble members: trained on the out-of-fold caches",
    "MCTS policy: F_400sim (400 sims, 300K episodes), seed 0, instead of J_800sim seed 9",
    "constrained-search pairs: the 'mcts' side uses the same F_400sim_s0 checkpoint as "
    "constrained_mcts (September: a stale July checkpoint, L_800sim_4M_s0, through a "
    "missing --mcts-run)",
]


def snapshot():
    p = sorted(glob.glob(os.path.join(TRAINING_DIR, "snapshots",
                                      "replay_snapshot_2026-09-01_sitetiers_*.json")))
    p = [x for x in p if not x.endswith(".meta.json")]
    return p[-1] if p else None


def base_env(gpu=None):
    sd = os.path.join(NS_DIR, "feature_cache", "stats")
    env = dict(os.environ)
    env.update({
        "RERUN_NS": NS, "REPLAY_SNAPSHOT_PATH": snapshot() or "",
        "WP_STATS_PATH": os.path.join(sd, "deploy.json"),
        "P1R_COMP_PATH": os.path.join(sd, "deploy_compositions.json"),
        "P1R_MCTS_CKPT": os.path.join(NS_DIR, "mcts_runs", MCTS_RUN, "draft_policy.pt"),
        "PYTHONPATH": os.path.join(HERE, "oct2026_site"),
        "PYTHON_CPU_COUNT": "4", "P1R_NPROC": "4",
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "NUMBA_NUM_THREADS": "2",
        "RERUN_SLOW_GPU": "-1", "WANDB_MODE": "disabled", "REPLAY_SNAPSHOT": "1",
        "CUDA_VISIBLE_DEVICES": str(gpu) if gpu is not None else "",
    })
    return env


def out(*p):
    return os.path.join(NS_DIR, *p)


def _listing_env():
    env = base_env()
    if not os.path.exists(env["WP_STATS_PATH"]):   # setup() asserts the file exists
        env["WP_STATS_PATH"] = os.path.join(TRAINING_DIR, "snapshots", "stats_decayed90_2026-09-01.json")
    return env


def phase1_jobs():
    env = _listing_env()
    code = ("import sys,json; sys.path.insert(0, %r); from rerun2026 import phase1_models as p; "
            "print(json.dumps({j.name: [j.argv, j.outputs] for j in p.build_jobs() if j.name in %r}))"
            % (TRAINING_DIR, PHASE1))
    r = subprocess.run([sys.executable, "-c", code], cwd=TRAINING_DIR, env=env,
                       capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def ens_members():
    env = _listing_env()
    code = ("import sys,json; sys.path.insert(0, %r); from rerun2026 import ensemble_uncertainty as e; "
            "print(json.dumps([[m[0], e.model_path(m[0])] for m in e.ROSTER]))" % TRAINING_DIR)
    r = subprocess.run([sys.executable, "-c", code], cwd=TRAINING_DIR, env=env,
                       capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def build_jobs():
    J = []

    def add(name, argv, outputs, deps=(), gpu=False, extra_env=None):
        J.append({"name": name, "argv": argv, "outputs": outputs, "deps": list(deps),
                  "gpu": gpu, "env": extra_env or {}})
    me = os.path.join(HERE, "oct2026_data.py")
    add("stats", [me, "stats"], [out("feature_cache", "stats", "split.json")],
        extra_env={"WP_STATS_PATH": os.path.join(TRAINING_DIR, "snapshots",
                                                 "stats_decayed90_2026-09-01.json")})
    add("features", [me, "features"], [out("feature_cache", "full_train_features.npz"),
                                       out("feature_cache", "cql_enriched_train", "meta.json")],
        deps=["stats"])
    p0 = os.path.join(RERUN, "phase0_features.py")
    add("phase0_cql", [p0, "--only", "cql"], [out("feature_cache", "cql_naive_train", "meta.json")],
        deps=["features"])
    add("phase0_gd", [p0, "--only", "gd"], [out("feature_cache", "gd_train", "meta.json")],
        deps=["features"])
    add("wp", [os.path.join(HERE, "oct2026_wp.py")],
        [out("models", f"{n}.pt") for n in ("wp_naive", "wp_herostrength", "wp_enriched_256",
                                             "wp_aug_v2_512")], deps=["features"], gpu=True)
    for name, (argv, outs) in phase1_jobs().items():
        dep = ["phase0_gd"] if name.startswith("gd_") else (
            ["phase0_cql"] if name in ("cql_naive_a1.0", "mcq_t0.5") else ["features"])
        add(name, argv, outs, deps=dep, gpu=True)
    disc = os.path.join(RERUN, "train_gourdeau_discriminator.py")
    add("disc_cache", [disc, "--stage", "cache"], [out("feature_cache", "gourdeau_disc_train.npz")],
        deps=["features"])
    add("disc_train", [disc, "--stage", "train"], [out("models", "gourdeau_discriminator.pt")],
        deps=["disc_cache"], gpu=True)
    for m, path in ens_members():
        add(f"ens_{m}", [os.path.join(RERUN, "ensemble_uncertainty.py"), "--task", "train",
                         "--member", m], [path], deps=["features"], gpu=True)
    add("mcts_prep", [os.path.join(HERE, "oct2026_refresh.py"), "--mcts-prep"],
        [out("feature_cache", "mcts_pretrain_exclude.json")], deps=["stats"])
    add("mcts", [os.path.join(HERE, "oct2026_refresh.py"), "--mcts-train"],
        [out("logs", f"mcts_{MCTS_RUN}.done")], deps=["wp", "gd_0", "mcts_prep"], gpu=True)
    gate = ["mcts", "wp", "disc_train"] + PHASE1
    add("phase3b", [os.path.join(RERUN, "phase3b_roundrobin.py"), "--mcts-run", MCTS_RUN],
        [out("results", "roundrobin_summary.json")], deps=gate,
        extra_env={"RERUN_GPU_IDS": "9,9,9,9", "CUDA_VISIBLE_DEVICES": ""})
    add("constrained", [os.path.join(HERE, "oct2026_refresh.py"), "--constrained-pairs"],
        [out("results", "constrained", "pairs.done")], deps=gate)
    return J


def done(job):
    return all(os.path.exists(p) for p in job["outputs"])


def my_gpu_procs(running):
    return sum(1 for j in running.values() if j["gpu"])


def hots_gpu_procs():
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        n = 0
        for pid in r.stdout.split():
            try:
                cmd = open(f"/proc/{pid.strip()}/cmdline", "rb").read().decode(errors="ignore")
                if "heroes-of-the-storm" in cmd or "train_mcts_worker" in cmd or "rerun2026" in cmd:
                    n += 1
            except OSError:
                pass
        return n
    except Exception:
        return 99


def run(gpu_id):
    jobs = build_jobs()
    byname = {j["name"]: j for j in jobs}
    running = {}
    log = open(os.path.join(NS_DIR, "refresh.log"), "a")

    def say(msg):
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        log.flush()
        print(msg, flush=True)
    say(f"=== oct2026 refresh start; snapshot={snapshot()}")
    failed = set()
    while True:
        for name, j in list(running.items()):
            rc = j["proc"].poll()
            if rc is None:
                continue
            j["logf"].close()
            del running[name]
            if rc == 0 and done(j):
                say(f"DONE {name}")
            else:
                failed.add(name)
                say(f"FAIL {name} rc={rc}")
        pending = [j for j in jobs if j["name"] not in running and j["name"] not in failed
                   and not done(j)]
        if not pending and not running:
            break
        ready = [j for j in pending if all(done(byname[d]) for d in j["deps"])]
        # critical path first: MCTS (longest), then the WP models and GD
        # opponents it depends on, ensemble members last (needed only at pool time)
        prio = lambda j: (0 if j["name"] == "mcts" else 1 if j["name"] in ("wp", "gd_0") else
                          2 if j["name"].startswith("gd_") else 5 if j["name"].startswith("ens_") else 3)
        ready.sort(key=prio)
        if not ready and not running:
            say(f"stuck: failed={sorted(failed)} pending={[j['name'] for j in pending]}")
            break
        for j in ready:
            if j["gpu"]:
                if my_gpu_procs(running) >= 2 or hots_gpu_procs() >= 4:
                    continue
            else:
                if sum(1 for r in running.values() if not r["gpu"]) >= 1:
                    continue
            env = base_env(gpu_id if j["gpu"] else None)
            env.update(j["env"])
            os.makedirs(out("logs"), exist_ok=True)
            lf = open(out("logs", f"refresh_{j['name']}.log"), "a")
            argv = ["nice", "-n", "19", "taskset", "-c", "48-63", sys.executable, "-u"] + j["argv"]
            j["proc"] = subprocess.Popen(argv, cwd=TRAINING_DIR, env=env, stdout=lf,
                                         stderr=subprocess.STDOUT)
            j["logf"] = lf
            running[j["name"]] = j
            say(f"START {j['name']} ({'gpu ' + str(gpu_id) if j['gpu'] else 'cpu'})")
            time.sleep(20)
        time.sleep(60)
    say(f"=== end; failed={sorted(failed)}")


def mcts_prep():
    sys.path.insert(0, TRAINING_DIR)
    from rerun2026 import common
    common.setup()
    _, test = common.load_split()
    ids = set(json.load(open(common.EXCLUDE_IDS_PATH)))
    ids |= {int(r["replay_id"]) for r in test}
    json.dump(sorted(ids), open(out("feature_cache", "mcts_pretrain_exclude.json"), "w"))
    print(f"exclude ids: {len(ids):,}")


def mcts_train():
    sims, eps = 400, 300000
    save = out("mcts_runs", MCTS_RUN)
    os.makedirs(save, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "MCTS_SAVE_DIR": save, "MCTS_WP_MODEL": "enriched_full",
        "MCTS_WP_PATH": out("models", "wp_enriched_256.pt"),
        "MCTS_GD_PATH": out("models", "generic_draft_0.pt"),
        "MCTS_NUM_EPISODES": str(eps), "MCTS_NUM_SIMS": str(sims), "MCTS_BATCH_EPISODES": "128",
        "MCTS_FRESH": "1", "MCTS_POLICY_HEAD": "linear", "MCTS_NET_SIZE": "base",
        "MCTS_EXCLUDE_IDS": out("feature_cache", "mcts_pretrain_exclude.json"),
        "WANDB_RUN_NAME": f"oct2026_{MCTS_RUN}"})
    rc = subprocess.call([sys.executable, "-u", os.path.join(TRAINING_DIR, "train_mcts_worker.py")],
                         cwd=TRAINING_DIR, env=env)
    if rc == 0 and os.path.exists(os.path.join(save, "draft_policy.pt")):
        open(out("logs", f"mcts_{MCTS_RUN}.done"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    sys.exit(rc)


def constrained_pairs():
    """Every constrained pair through constrained_search.py --task pair, with
    --mcts-run forwarded (the September orchestrator dropped it). 5 CPU workers."""
    from concurrent.futures import ThreadPoolExecutor
    code = ("import sys,json; sys.path.insert(0, %r); from rerun2026 import constrained_search as c; "
            "print(json.dumps(c.pair_list()))" % TRAINING_DIR)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-c", code], cwd=TRAINING_DIR, env=env,
                       capture_output=True, text=True)
    pairs = json.loads(r.stdout.strip().splitlines()[-1])
    res_dir = out("results", "constrained", "roundrobin")

    def one(pr):
        a, b = pr
        if os.path.exists(os.path.join(res_dir, f"{a}__{b}.json")):
            return 0
        lf = open(out("logs", f"p3c_{a}__{b}.log"), "w")
        return subprocess.call([sys.executable, "-u", os.path.join(RERUN, "constrained_search.py"),
                                "--task", "pair", "--pair", f"{a}__{b}", "--mcts-run", MCTS_RUN],
                               cwd=TRAINING_DIR, env=env, stdout=lf, stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(5) as ex:
        rcs = list(ex.map(one, pairs))
    if all(rc == 0 for rc in rcs):
        open(out("results", "constrained", "pairs.done"), "w").write(str(len(pairs)))
    sys.exit(0 if all(rc == 0 for rc in rcs) else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--mcts-prep", action="store_true")
    ap.add_argument("--mcts-train", action="store_true")
    ap.add_argument("--constrained-pairs", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.mcts_prep:
        mcts_prep()
    elif a.mcts_train:
        mcts_train()
    elif a.constrained_pairs:
        constrained_pairs()
    elif a.list:
        for j in build_jobs():
            print(("x " if done(j) else "  ") + j["name"], j["deps"], "GPU" if j["gpu"] else "")
    else:
        run(a.gpu)
