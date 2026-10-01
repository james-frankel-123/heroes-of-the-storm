"""
Paper-1 rebuild on site-scheme tiers (low = Bronze+Silver, mid = Gold+Platinum,
high = Diamond+Master; unranked games excluded). Same pinned snapshot, same
replays and split (minus unranked games); every statistic, feature, value
function and judge is rebuilt, and every non-MCTS experiment rerun. All
outputs go to the P1_TIERS=site namespace (paper1_revision/site/...,
overfit2026/site/...), so the legacy artifacts other lanes read are untouched.

Scheduler: dependency-aware, skip-if-output-exists (safe to rerun after any
interruption). At most one CPU job (<= 6 worker processes) and GPU_SLOTS GPU
jobs of this lane at a time, on the GPU with the most free memory among
GPU_IDS, never while HotS already holds MAX_HOTS_GPU_PROCS GPU processes;
everything under nice 19, cores 48-63.

Usage (from training/): nohup python3 paper1_revision/site_rebuild.py [--only a,b] [--list] &
Log: paper1_revision/site/logs/site_rebuild.log (+ one log per job)
"""
import os
import sys
import json
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
OF = os.path.join(TRAINING_DIR, "overfit2026")
SITE = os.path.join(HERE, "site")
OFS = os.path.join(OF, "site")
LOG_DIR = os.path.join(SITE, "logs")
PY = sys.executable
SYS_PY = "/usr/bin/python3"          # has numba (personalization code)
GPU_IDS = [int(x) for x in os.environ.get("SITE_GPU_IDS", "1,2").split(",")]
GPU_SLOTS = int(os.environ.get("SITE_GPU_SLOTS", "2"))
MAX_HOTS_GPU_PROCS = int(os.environ.get("SITE_MAX_HOTS_GPU", "5"))
TABLE1 = ["naive", "herostrength", "enriched", "enriched_leak", "enriched_512", "relational",
          "absolute", "aug_wr0_512", "aug_wr5_512", "aug_wr10_512", "aug_wr50_512"]


def s(*p):
    return os.path.join(SITE, *p)


def o(*p):
    return os.path.join(OFS, *p)


def jobs():
    J = []

    def add(name, argv, outputs, deps=(), gpu=False, py=PY, env=None):
        J.append(dict(name=name, argv=[py] + argv, outputs=outputs, deps=list(deps), gpu=gpu,
                      env=env or {}))
    p1 = lambda f: os.path.join(HERE, f)
    of = lambda f: os.path.join(OF, f)
    add("stats", [p1("core.py"), "stats"], [s("cache", "stats", "deploy.json")])
    add("features", [p1("core.py"), "features", "train_oof", "train_leak", "val", "test",
                     "NODRIFT", "T17"], [s("cache", "feats", "T17.npz")], deps=["stats"])
    add("qm", [of("site_qm.py")], [o("models", "qm_wp_v0.pt")], gpu=True)
    add("gn", [of("comp_gn_rebuild.py")], [o("models", "gN8_naive.pt")], gpu=True)
    add("gn_folds", [of("comp_gn_folds.py")], [o("cache", "comp_gn_oof_N.npz")], deps=["gn"],
        gpu=True)
    add("wp_table1", [p1("train_wp.py"), ",".join(TABLE1)],
        [s("models", f"{n}.json") for n in TABLE1], deps=["features"], gpu=True)
    add("wp_sweep", [p1("train_wp.py"), "sw_*"], [s("models", "sweep_done.flag")],
        deps=["features"], gpu=True)
    add("comp_audit", [of("comp_audit_data.py"), "T17", "N", "SNAP"],
        [o("cache", "comp_audit_SNAP.npz")], deps=["gn", "qm"])
    add("causal_data", [of("comp_causal_data.py")], [o("cache", "comp_causal_games.npz")],
        py=SYS_PY)
    add("comp_causal", [of("comp_causal.py")], [o("results", "comp_causal.json")],
        deps=["comp_audit", "causal_data"])
    add("comp_judges", [of("comp_judges.py")], [o("results", "comp_judges_done.flag")],
        deps=["comp_audit", "gn_folds", "causal_data"])
    add("native_ri", [of("comp_native_ri.py")], [o("results", "comp_native_ri.json")])
    add("judge_cal", [p1("judge_calibration.py")], [s("results", "judge_calibration.json")],
        deps=["gn", "qm", "wp_table1"])
    add("attenuation", [p1("judge_attenuation.py")], [s("results", "judge_attenuation.json")],
        deps=["comp_judges", "wp_table1"])
    add("metric_validity", [p1("metric_validity.py")], [s("results", "metric_validity.json")],
        deps=["features"])
    add("eval_ngs", [p1("eval_ngs.py")], [s("results", "ngs.json")], deps=["wp_table1"])
    # ── B2: rerun2026 baselines retrained in namespace p1site (site tiers,
    # the paper's split, validation-based early stopping) ──
    rr = lambda f: os.path.join(TRAINING_DIR, "rerun2026", f)
    ns = lambda *p: os.path.join(TRAINING_DIR, "rerun2026", "ns", "p1site", *p)
    tj = rr("train_jobs.py")
    E = P1SITE_ENV
    add("p1_snapshot", [p1("site_snapshot.py")], [P1SITE_SNAPSHOT])
    add("p1_phase0", [p1("site_phase0.py")], [ns("feature_cache", "cql_naive_test", "meta.json")],
        deps=["p1_snapshot", "stats"], env=E)
    for i in range(5):
        add(f"gd_{i}", [tj, "gd", "--variant", str(i)], [ns("models", f"generic_draft_{i}.pt")],
            deps=["p1_phase0"], gpu=True, env=E)
    add("gourdeau_wp", [tj, "gourdeau"], [ns("models", "gourdeau_wp.pt")], deps=["p1_snapshot", "stats"],
        gpu=True, env=E)
    for a_ in (0.5, 1.0, 0.1, 2.0, 5.0):
        add(f"cql_naive_a{a_}", [tj, "cql_naive", "--alpha", str(a_)],
            [ns("models", "cql", f"_cql_temp_a{a_}.pt")], deps=["p1_phase0"], gpu=True, env=E)
    for t in (0.5, 0.4, 0.6, 0.7, 0.8):
        add(f"mcq_t{t}", [tj, "mcq", "--threshold", str(t)], [ns("models", "mcq", f"_mcq_temp_t{t}.pt")],
            deps=["p1_phase0"], gpu=True, env=E)
    for b_ in (1.0, 0.1, 0.5, 2.0):
        add(f"bccql_b{b_}", [tj, "bccql", "--bc-weight", str(b_)],
            [ns("models", "mcq", f"_bc_cql_temp_bc{b_}.pt")], deps=["p1_phase0"], gpu=True, env=E)
    for tau, beta in ((0.9, 3.0), (0.7, 1.0), (0.7, 3.0), (0.8, 1.0), (0.8, 3.0), (0.9, 1.0)):
        add(f"iql_t{tau}_b{beta}", [rr("experiment_iql_draft.py"), "--task", "train", "--tau", str(tau),
                                    "--beta", str(beta)],
            [ns("models", "meta", f"iql_t{tau}_b{beta}.json")], deps=["p1_phase0"], gpu=True, env=E)
    add("discriminator", [rr("train_gourdeau_discriminator.py")],
        [ns("models", "gourdeau_discriminator.pt")], deps=[f"gd_{i}" for i in range(5)], gpu=True, env=E)
    for arch in ("512,256,128", "1024,512,256", "256,128,64"):
        for tau in (0.005, 0.001, 0.01, 0.05, 0.1):
            a_s = arch.replace(",", "x")
            add(f"cql_hp_t{tau}_{a_s}", [tj, "cql_hp", "--alpha", "1.0", "--tau", str(tau), "--arch", arch],
                [ns("models", "cql_hyperparams", f"_cql_temp_a1.0_t{tau}_{a_s}.pt")],
                deps=["p1_phase0"], gpu=True, env=E)
    return J


P1SITE_SNAPSHOT = os.path.join(TRAINING_DIR, "snapshots", "replay_snapshot_2026-05-22_1956753_p1site.json")
P1SITE_ENV = {
    "RERUN_NS": "p1site", "RERUN_SPLIT": "p1val", "REPLAY_SNAPSHOT_PATH": P1SITE_SNAPSHOT,
    "WP_STATS_PATH": os.path.join(SITE, "cache", "stats", "deploy.json"),
    "P1R_COMP_PATH": os.path.join(SITE, "cache", "stats", "deploy_compositions.json"),
    "PYTHONPATH": os.path.join(HERE, "oct2026_site"), "RERUN_SLOW_GPU": "-1", "REPLAY_SNAPSHOT": "1",
    "PYTHON_CPU_COUNT": "6",
}


def log(msg):
    os.makedirs(LOG_DIR, exist_ok=True)
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    open(os.path.join(LOG_DIR, "site_rebuild.log"), "a").write(line + "\n")


def done(j):
    return all(os.path.exists(p) for p in j["outputs"])


def hots_gpu_procs():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.split()
    except Exception:
        return 0
    n = 0
    for pid in out:
        try:
            if "heroes-of-the-storm" in open(f"/proc/{pid}/cmdline").read():
                n += 1
        except Exception:
            pass
    return n


def best_gpu():
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    free = {int(a): int(b) for a, b in (l.split(",") for l in r)}
    c = [g for g in GPU_IDS if free.get(g, 0) > 16000]
    return max(c, key=lambda g: free[g]) if c else None


def base_env(gpu=None):
    env = dict(os.environ)
    env.update({"P1_TIERS": "site", "P1R_NPROC": "6", "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4", "NUMBA_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1",
                "WANDB_MODE": "disabled", "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu)})
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    J = jobs()
    if a.only:
        keep = set(a.only.split(","))
        J = [j for j in J if j["name"] in keep]
    if a.list:
        for j in J:
            print(f"{j['name']:16s} {'GPU' if j['gpu'] else 'CPU'} done={done(j)} deps={j['deps']}")
        return
    # flags for jobs whose script writes no single terminal file
    os.makedirs(o("results"), exist_ok=True)
    pj = o("results", "comp_judges.json")
    if not os.path.exists(pj):
        json.dump({"chosen": "realized"}, open(pj, "w"))     # variant fixed in REVISION_NOTES §11
    names = {j["name"]: j for j in J}
    running = {}
    failed = set()
    while True:
        for n, (p, j, fh) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            fh.close()
            del running[n]
            if n == "comp_judges" and rc == 0:
                open(o("results", "comp_judges_done.flag"), "w").write("ok")
            if n == "wp_sweep" and rc == 0:
                open(s("models", "sweep_done.flag"), "w").write("ok")
            if rc == 0 and done(j):
                log(f"done {n}")
            else:
                failed.add(n)
                log(f"FAILED {n} (rc={rc}); see {LOG_DIR}/{n}.log")
        pending = [j for j in J if not done(j) and j["name"] not in running and j["name"] not in failed]
        if not pending and not running:
            log(f"all done; failed: {sorted(failed)}")
            return
        for j in pending:
            if any(d in failed for d in j["deps"]):
                failed.add(j["name"])
                log(f"SKIP {j['name']} (dependency failed)")
                continue
            if any(d in names and not done(names[d]) for d in j["deps"]):
                continue
            if j["gpu"]:
                if sum(1 for _, r, _ in running.values() if r["gpu"]) >= GPU_SLOTS:
                    continue
                if hots_gpu_procs() >= MAX_HOTS_GPU_PROCS:
                    continue
                gpu = best_gpu()
                if gpu is None:
                    continue
            else:
                if sum(1 for _, r, _ in running.values() if not r["gpu"]) >= 1:
                    continue
                gpu = None
            env = base_env(gpu)
            env.update(j["env"])
            fh = open(os.path.join(LOG_DIR, f"{j['name']}.log"), "a")
            cmd = ["nice", "-n", "19", "taskset", "-c", "48-63"] + j["argv"]
            p = subprocess.Popen(cmd, cwd=TRAINING_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT)
            running[j["name"]] = (p, j, fh)
            log(f"start {j['name']} ({'GPU ' + str(gpu) if j['gpu'] else 'CPU'}) pid {p.pid}")
        time.sleep(20)


if __name__ == "__main__":
    main()
