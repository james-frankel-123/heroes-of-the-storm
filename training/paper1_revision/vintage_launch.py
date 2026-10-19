"""
Orchestration for the stale-vintage F drafters (paper1_revision/vintage.py).
Runs on the main box: ssh + rsync only, no compute.

  fetch <vintage>                  copy the vintage's stats, WP models, exclude
                                   list, logs (not the feature npz) and GD from
                                   the 3090 (where they were built) to this box
  push <vintage> <host>            copy the MCTS inputs from this box to a worker
  launch <vintage> <seed> <host>   start the hotsjob MCTS job p1vint_<vintage>_F_s<seed>
  wait-launch <vintage> <seed> <job>[,<job>...]
                                   waiter: polls `hotsjob.py state` of the given
                                   jobs (host:name) every 10 min; when one is
                                   completed and its host runs no other MCTS
                                   job, pushes and launches there, then exits.
                                   A failed job, or an unreachable host for
                                   > 1 h, writes an ALERT line and keeps waiting
                                   on the others; all failed -> exit 1.

Inputs per vintage (paths relative to training/):
  paper1_revision/site/vint/<v>/cache/stats/deploy.json, deploy_compositions.json
  paper1_revision/site/vint/<v>/cache/mcts_pretrain_exclude.json
  paper1_revision/site/vint/<v>/models/enriched.json, enriched_s<sel>.pt
  vintage.VINTAGES[<v>]["gd"]

Usage (from training/): python3 paper1_revision/vintage_launch.py <cmd> ...
"""
import os
import sys
import json
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(TRAINING_DIR)
RW = os.path.join(TRAINING_DIR, "remote_workers")
sys.path.insert(0, RW)
sys.path.insert(0, TRAINING_DIR)
import hosts  # noqa: E402
from paper1_revision import vintage  # noqa: E402

BUILD_HOST = "max-windows-3090"
REMOTE = "/home/max/hots/repo/"
# per-process data limit (ulimit -d) of an MCTS job; measured VmData peak of an
# MCTS run is ~7 GB. The 5090 has 35 GB in all (<= 12 GB per process).
MEM_MAX = {"max-windows-3090": "16G", "3080-gaming-desktop": "16G", "windows-5090-wsl": "12G"}
LOG = os.path.join(HERE, "site", "vint", "vintage_launch.log")
ALERT = os.path.join(HERE, "site", "vint", "ALERT_vintage_launch")


def vdir(v, *p):
    return os.path.join("training", "paper1_revision", "site", "vint", v, *p)


def log(msg):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")


def alert(msg):
    log(f"ALERT {msg}")
    open(ALERT, "a").write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def rsync(args, **kw):
    return subprocess.run(["nice", "-n", "19", "rsync", "-rlt"] + args, cwd=REPO, **kw)


def fetch(v):
    src = f"{BUILD_HOST}:{REMOTE}"
    rp = f"--rsync-path={hosts.rsync_path(BUILD_HOST)}"
    for d in ("cache/stats/", "models/", "logs/"):
        os.makedirs(os.path.join(REPO, vdir(v, d)), exist_ok=True)
        rsync([rp, src + vdir(v, d), vdir(v, d)], check=True)
    for f in ("cache/mcts_pretrain_exclude.json", "cache/split_meta.json"):
        rsync([rp, src + vdir(v, f), vdir(v, f)], check=True)
    gd = os.path.join("training", vintage.spec(v)["gd"])
    gdd = os.path.dirname(gd) + "/"
    os.makedirs(os.path.join(REPO, gdd), exist_ok=True)
    rsync([rp, src + gdd, gdd], check=True)
    log(f"fetched {v} from {BUILD_HOST}")


def inputs(v):
    meta = json.load(open(os.path.join(REPO, vdir(v, "models", "enriched.json"))))
    files = [vdir(v, "cache", "stats", "deploy.json"), vdir(v, "cache", "stats", "deploy_compositions.json"),
             vdir(v, "cache", "mcts_pretrain_exclude.json"), vdir(v, "models", "enriched.json"),
             vdir(v, "models", f"enriched_s{meta['selected_seed']}.pt"),
             os.path.join("training", vintage.spec(v)["gd"])]
    for f in files:
        if not os.path.exists(os.path.join(REPO, f)):
            raise SystemExit(f"missing input {f}")
    return files


def push(v, host):
    lst = "\n".join(inputs(v)) + "\n"
    rsync([f"--rsync-path={hosts.rsync_path(host)}", "--files-from=-", "./", f"{host}:{REMOTE}"],
          input=lst, text=True, check=True)
    log(f"pushed {v} inputs to {host}")


def job_name(v, seed):
    return f"p1vint_{v}_F_s{seed}"


def launch(v, seed, host):
    run = f"F_oof_s{seed}"
    env = dict(os.environ, RUN_NAME=job_name(v, seed),
               HOTSJOB_FLAGS=f"--save-dir {vdir(v, 'mcts_runs', run)[len('training/'):]} "
                             f"--progress-log {vdir(v, 'logs', f'mcts_{run}.log')[len('training/'):]} "
                             f"--mem-max {MEM_MAX[host]}")
    r = subprocess.run([os.path.join(RW, "run_remote.sh"), host, "env", "P1_TIERS=site", f"P1_VINTAGE={v}",
                        "python", "paper1_revision/train_mcts.py", "F", str(seed), "--gpu", "0"],
                       capture_output=True, text=True, cwd=TRAINING_DIR, env=env)
    log(f"launch {job_name(v, seed)} on {host}: rc={r.returncode} {(r.stdout + r.stderr).strip()[-300:]}")
    return r.returncode == 0


def state(host, name):
    """hotsjob state word, or None if the host does not answer."""
    try:
        r = subprocess.run(["ssh", host, hosts.shell(host)], capture_output=True, text=True, timeout=180,
                           input=f"source ~/hots/env.sh >/dev/null 2>&1; "
                                 f"python3 ~/hots/repo/training/remote_workers/hotsjob.py state {name}\n")
    except subprocess.TimeoutExpired:
        return None
    w = r.stdout.strip().splitlines()
    return w[-1].strip() if r.returncode == 0 and w else None


def mcts_running(host):
    """Names of running MCTS jobs on host (None if unreachable)."""
    try:
        r = subprocess.run([os.path.join(RW, "jobs_remote.sh"), host], capture_output=True, text=True,
                           cwd=TRAINING_DIR, timeout=240)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    out = []
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) >= 3 and p[1] in ("running", "pausing", "starting") and p[2] == "mcts":
            out.append(p[0])
    return out


def wait_launch(v, seed, jobs):
    jobs = [j.split(":", 1) for j in jobs.split(",")]
    down_since, failed = {}, set()
    log(f"waiter: {job_name(v, seed)} goes to the first free host of {jobs}")
    while True:
        for host, name in jobs:
            if name in failed:
                continue
            st = state(host, name)
            if st is None:
                t0 = down_since.setdefault(host, time.time())
                if time.time() - t0 > 3600:
                    alert(f"{host} unreachable for {(time.time() - t0) / 3600:.1f} h")
                    down_since[host] = time.time()
                continue
            down_since.pop(host, None)
            if st == "failed":
                alert(f"{name} on {host} failed")
                failed.add(name)
                continue
            if st == "completed":
                busy = mcts_running(host)
                if busy is None or busy:
                    continue
                push(v, host)
                if launch(v, seed, host):
                    log(f"waiter done: {job_name(v, seed)} launched on {host}")
                    return 0
                alert(f"launch of {job_name(v, seed)} on {host} failed")
                return 1
        if len(failed) == len(jobs):
            alert("every watched job failed; nothing launched")
            return 1
        time.sleep(600)


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "fetch":
        fetch(args[0])
    elif cmd == "push":
        push(args[0], args[1])
    elif cmd == "launch":
        sys.exit(0 if launch(args[0], int(args[1]), args[2]) else 1)
    elif cmd == "wait-launch":
        sys.exit(wait_launch(args[0], int(args[1]), args[2]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
