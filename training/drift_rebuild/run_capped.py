"""
Capped GPU runner (owner rule, 2026-10-01): at most 2 rebuild GPU processes,
launched only while fewer than 4 HotS GPU processes run machine-wide, on the
freest GPU, under nice 19 / cores 48-63 / 3 BLAS threads.
Usage: python3 drift_rebuild/run_capped.py <jobs.txt> [own_max]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_queue import rebuild_gpu_load  # noqa: E402

TRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hots_gpu_procs():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.split()
    n = 0
    for p in out:
        try:
            cmd = open(f"/proc/{p.strip()}/cmdline", "rb").read()
        except Exception:
            continue
        n += b"heroes-of-the-storm" in cmd
    return n


def freest_gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    rows = [tuple(int(x) for x in l.split(",")) for l in out.strip().splitlines()]
    rows = [r for r in rows if r[2] < 80000]
    return min(rows, key=lambda r: (r[1], r[2]))[0]


OWN_MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 2
GAP = int(sys.argv[3]) if len(sys.argv) > 3 else 180   # seconds after each launch
jobs = []
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    out, log, cmd = line.rstrip("\n").split("\t", 2)
    if os.path.exists(out) or os.path.exists(log):
        continue
    jobs.append((out, log, cmd))
print(f"{len(jobs)} jobs", flush=True)
procs = []
for out, log, cmd in jobs:
    while (sum(rebuild_gpu_load().values()) >= 2 or hots_gpu_procs() >= 4
           or sum(p.poll() is None for p in procs) >= OWN_MAX):
        time.sleep(60)
    g = freest_gpu()
    pre = (f"OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 CUDA_VISIBLE_DEVICES={g} "
           "RB_JOB=1 nice -n 19 taskset -c 48-63 python3 -u ")
    cmd = pre + cmd[len("python3 -u "):] if cmd.startswith("python3 -u ") \
        else cmd.replace(" python3 -u ", " " + pre)
    procs.append(subprocess.Popen(cmd, shell=True, cwd=TRAIN, stdout=open(log, "w"),
                                  stderr=subprocess.STDOUT))
    print(f"[{time.strftime('%H:%M')}] start {os.path.basename(log)} gpu {g}", flush=True)
    time.sleep(GAP)
print("all launched", flush=True)
