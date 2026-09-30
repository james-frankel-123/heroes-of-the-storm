"""
Minimal job queue for the rebuild: runs shell commands with at most
--parallel concurrent jobs, skipping jobs whose output already exists.
GPU jobs are pinned to GPU 1 and launched only while nvidia-smi reports at
least --min-free-gb free on it (other people's jobs share the machine).

Usage: python3 drift_rebuild/run_queue.py jobs.txt --parallel 3 [--gpu]
jobs file: one job per line, "<output path>\t<log path>\t<command>"
"""
import os
import sys
import time
import argparse
import subprocess


def gpu_free_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    free = {}
    for line in out.strip().splitlines():
        i, tot, used = [int(x.strip()) for x in line.split(",")]
        free[i] = (tot - used) / 1024
    return free


def rebuild_gpu_load():
    """Per-GPU count of rebuild jobs (any queue instance), read from /proc:
    top-most python processes whose environment carries a drift_rebuild
    MCTS_SAVE_DIR or the RB_JOB marker (child processes are not counted)."""
    marked = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            if not cmd or b"python" not in cmd[0]:
                continue
            env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
            ppid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
        except Exception:
            continue
        envd = dict(e.split(b"=", 1) for e in env if b"=" in e)
        if b"drift_rebuild" not in envd.get(b"MCTS_SAVE_DIR", b"") and \
                b"RB_JOB" not in envd:
            continue
        marked[int(pid)] = (ppid, envd.get(b"CUDA_VISIBLE_DEVICES", b"").decode())
    load = {}
    for pid, (ppid, g) in marked.items():
        if ppid in marked or not g.isdigit():
            continue
        load[int(g)] = load.get(int(g), 0) + 1
    return load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--gpus", default="1", help="comma list of allowed GPUs")
    ap.add_argument("--per-gpu", type=int, default=3)
    ap.add_argument("--skip-started", action="store_true",
                    help="also skip jobs whose log file already exists")
    args = ap.parse_args()
    jobs = []
    for line in open(args.jobs):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        out, log, cmd = line.split("\t", 2)
        if os.path.exists(out):
            continue
        if args.skip_started and os.path.exists(log):
            continue
        jobs.append((out, log, cmd))
    print(f"{len(jobs)} jobs to run", flush=True)
    running = []
    allowed = [int(g) for g in args.gpus.split(",")]
    while jobs or running:
        running = [(p, o, g) for p, o, g in running if p.poll() is None]
        ext = rebuild_gpu_load() if args.gpu else {}
        n_running = max(len(running), sum(ext.values()))
        if jobs and n_running < args.parallel:
            env = dict(os.environ)
            g = None
            if args.gpu:
                free = gpu_free_gb()
                load = {x: max(ext.get(x, 0),
                               sum(1 for _, _, gg in running if gg == x))
                        for x in allowed}
                cands = [x for x in allowed if free.get(x, 0) >= args.min_free_gb
                         and load[x] < args.per_gpu]
                if not cands:
                    time.sleep(30)
                    continue
                g = max(cands, key=lambda x: (free[x] - 20 * load[x]))
                env["CUDA_VISIBLE_DEVICES"] = str(g)
                env["RB_JOB"] = "1"
            ready = [j for j in jobs if all(
                os.path.exists(t.split("=", 1)[1]) for t in j[2].split()
                if t.startswith(("MCTS_GD_PATH=", "MCTS_WP_PATH="))
                or t.startswith("REQUIRES="))]
            if not ready:
                time.sleep(60)
                continue
            jobs.remove(ready[0])
            out, log, cmd = ready[0]
            os.makedirs(os.path.dirname(log), exist_ok=True)
            p = subprocess.Popen(cmd, shell=True, env=env,
                                 stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                 cwd=os.path.dirname(os.path.dirname(
                                     os.path.abspath(__file__))))
            running.append((p, out, g))
            print(f"[{time.strftime('%H:%M:%S')}] start {out} (gpu {g})", flush=True)
            time.sleep(20 if args.gpu else 2)
            continue
        time.sleep(5)
    print("queue done", flush=True)


if __name__ == "__main__":
    main()
