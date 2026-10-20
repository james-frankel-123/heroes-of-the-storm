"""
Waiter for the 2026-10-19 reruns of the P3_DRAFTER / P3_PERSONAL_GD MCTS runs
on the race-fixed P3 kernels (queue: p3_racefix_queue.json, same commands as
p3_mcts_queue.json under new job names). Runs on the main box as a light
polling loop (ssh and rsync only, no compute).

Launch rules
  * windows-5090-wsl: at most one job of this queue at a time (the host runs
    other work too).
  * max-windows-3090, 3080-gaming-desktop: one GPU job of this queue at a
    time, and only while no p1vint_* job there is running, starting, pausing
    or paused (paper-1 vintages have the GPU first).
  * CPU items (training the distilled prior, greedy drafters, analyses) run
    on max-windows-3090, cores 20-27, one at a time, with no GPU.
  * MemAvailable >= PEAK_GB on the host; every item's needs completed.
  * Before every launch the search outputs of every host
    (cache/mcts_v2/chance, no partial files) are copied to the target host
    through this box, so the distilled prior reaches the GPU hosts and the
    analyses see every run.
A job is launched once (a manifest of that name on any host counts as taken);
failures are logged as alerts and not retried. One instance (flock), no pgrep.

Usage (from training/):
  setsid nohup nice -n 19 taskset -c 48-63 python3 personalization/remote/p3_racefix_waiter.py \\
      >> personalization/logs/p3_racefix_waiter.out 2>&1 &
Test: --dry --once
"""
import fcntl
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
TRAINING = os.path.dirname(P3)
sys.path.insert(0, os.path.join(TRAINING, "drift_rebuild", "v2"))
sys.path.insert(0, os.path.join(TRAINING, "remote_workers"))
import drift_mcts_waiter as DW  # noqa: E402  (status(), remote())
import hosts as RH  # noqa: E402

LOGS = os.path.join(P3, "logs")
LOG = os.path.join(LOGS, "p3_racefix_waiter.log")
ALERT = os.path.join(LOGS, "ALERT_p3_racefix")
QUEUE = os.path.join(HERE, "p3_racefix_queue.json")
JOB = os.path.join(HERE, "p3_job.sh")
SCRATCH = os.path.join(P3, "logs", "p3_racefix_gather")
REL = "/home/max/hots/repo/training/personalization/cache/mcts_v2/chance/"
HOSTS = ["windows-5090-wsl", "max-windows-3090", "3080-gaming-desktop"]
GATED = {"max-windows-3090", "3080-gaming-desktop"}   # wait for p1vint_* there
CPU_HOST = "max-windows-3090"
MEM = {"max-windows-3090": "24G", "3080-gaming-desktop": "20G", "windows-5090-wsl": "16G"}
CORES = {"max-windows-3090": "28-35", "3080-gaming-desktop": "0-19", "windows-5090-wsl": "0-7"}
CPU_CORES = "20-27"
PEAK_GB = 14
CYCLE = 240
LIVE = ("running", "starting", "pausing", "paused")
DRY = "--dry" in sys.argv
ONCE = "--once" in sys.argv
if DRY:
    LOG, ALERT = LOG + ".dry", ALERT + ".dry"


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def alert(msg, seen):
    if msg in seen:
        return
    seen.add(msg)
    log(f"ALERT {msg}")
    with open(ALERT, "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def rsync(args):
    r = subprocess.run(["nice", "-n", "19", "rsync", "-rlt", "--update", "-z", "--exclude", "partial_*",
                        "--exclude", "*.tmp"] + args, capture_output=True, text=True, timeout=1800)
    return r.returncode, r.stderr[-200:]


def spread(target):
    """Copy every host's finished search outputs to target, through this box."""
    os.makedirs(SCRATCH, exist_ok=True)
    for h in HOSTS:
        rc, err = rsync([f"--rsync-path={RH.rsync_path(h)}", f"{h}:{REL}", SCRATCH + "/"])
        if rc not in (0, 23, 24):   # 23/24: no outputs on that host yet
            log(f"pull from {h} failed rc={rc}: {err}")
            return False
    if not os.listdir(SCRATCH):
        return True
    if DW.remote(target, f"mkdir -p {REL}") is None:
        log(f"mkdir on {target} did not answer")
        return False
    rc, err = rsync([f"--rsync-path={RH.rsync_path(target)}", SCRATCH + "/", f"{target}:{REL}"])
    if rc != 0:
        log(f"push to {target} failed rc={rc}: {err}")
        return False
    return True


def avail_gb(host):
    out = DW.remote(host, "awk '/MemAvailable/{print \"AVAIL\", $2/1048576}' /proc/meminfo")
    if out is None:
        return None
    for line in out.splitlines():
        if line.startswith("AVAIL"):
            return float(line.split()[1])
    return None


def main():
    os.makedirs(LOGS, exist_ok=True)
    lock = open(os.path.join(LOGS, "p3_racefix_waiter.lock" + (".dry" if DRY else "")), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another p3_racefix_waiter holds the lock")
    if not DRY:
        open(os.path.join(LOGS, "p3_racefix_waiter.pid"), "w").write(f"{os.getpid()}\n")
    log(f"start pid {os.getpid()}")
    seen, failed_launch, said = set(), set(), set()
    while True:
        queue = json.load(open(QUEUE))
        names = {q["name"] for q in queue}
        st = {h: DW.status(h) for h in HOSTS}
        ours = {}
        for h, jobs in st.items():
            for n, j in (jobs or {}).items():
                if n in names:
                    ours[n] = (h, j)
        for n, (h, j) in ours.items():
            if j["status"] in ("failed", "died", "paused"):
                alert(f"{n} on {h} is {j['status']} (log {j['log']})", seen)
        done = {n for n, (h, j) in ours.items() if j["status"] == "completed"}
        if all(n in done or n in failed_launch for n in names):
            log("all race-fix queue items completed")
            return
        todo = [q for q in queue if q["name"] not in ours and q["name"] not in failed_launch
                and all(x in done for x in q["needs"])]
        cpu_busy = any(n in ours and ours[n][1]["status"] in LIVE for n in names
                       if not next(q for q in queue if q["name"] == n)["gpu"])
        for host in HOSTS:
            if not todo:
                break
            jobs = st[host]
            if jobs is None:
                alert(f"{host} unreachable", seen)
                continue
            mine_gpu = [n for n, (h, j) in ours.items() if h == host and j["status"] in LIVE
                        and next(q for q in queue if q["name"] == n)["gpu"]]
            vint = [n for n, j in jobs.items() if n.startswith("p1vint_") and j["status"] in LIVE]
            gpu_ok = not mine_gpu and not (host in GATED and vint)
            if host in GATED and vint and f"vint:{host}" not in said:
                log(f"{host}: waiting for {', '.join(sorted(vint))}")
                said.add(f"vint:{host}")
            if host in GATED and not vint and f"vint:{host}" in said and f"free:{host}" not in said:
                log(f"{host}: no p1vint_* job live any more")
                said.add(f"free:{host}")
            cand = [q for q in todo if host in q["hosts"] and (q["gpu"] and gpu_ok or
                                                                 not q["gpu"] and host == CPU_HOST and not cpu_busy)]
            if not cand:
                continue
            av = avail_gb(host)
            if av is None or av < PEAK_GB:
                log(f"{host}: MemAvailable {av} GB, not launching")
                continue
            pick = cand[0]
            if DRY:
                log(f"[dry] would launch {pick['name']} on {host} (avail {av:.0f} GB)")
                todo.remove(pick)
                continue
            if not spread(host):
                alert(f"spreading outputs to {host} failed; retry next cycle", seen)
                continue
            todo.remove(pick)
            gpu = pick["gpu"]
            env = dict(os.environ, P3_CORES=CORES[host] if gpu else CPU_CORES,
                       P3_THREADS="2" if gpu else "8", P3_MEM=MEM[host] if gpu else "48G")
            r = subprocess.run([JOB, host, pick["name"]] + pick["cmd"].split(), capture_output=True, text=True,
                               cwd=os.path.dirname(TRAINING), env=env, timeout=300)
            tail = (r.stdout.strip().splitlines() or [r.stderr.strip()[-300:]])[-1]
            log(f"launch {pick['name']} on {host} (avail {av:.0f} GB): rc={r.returncode} {tail}")
            if r.returncode != 0:
                alert(f"launch of {pick['name']} on {host} failed: rc={r.returncode} {tail}; not retried", seen)
                failed_launch.add(pick["name"])
            if not gpu:
                cpu_busy = True
        if ONCE:
            return
        time.sleep(CYCLE)


if __name__ == "__main__":
    main()
