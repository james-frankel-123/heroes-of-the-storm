"""Sequential, rerun-safe job queue for a remote worker (cwd = training/).
Each line of the jobs file: "<output path>\t<command>". Jobs whose output
exists are skipped, so a paused (killed) queue resumes by rerunning; at most
the job in progress (a value-function training of 1-3 minutes) is lost.
Usage: python drift_rebuild/v2/remote_queue.py drift_v2/jobs_remote_vf.txt"""
import os
import subprocess
import sys
import time

lines = [l.rstrip("\n").split("\t", 1) for l in open(sys.argv[1]) if l.strip()]
todo = [(o, c) for o, c in lines if not os.path.exists(o)]
print(f"{len(todo)}/{len(lines)} jobs to run", flush=True)
for i, (out, cmd) in enumerate(todo):
    t0 = time.time()
    log = os.path.join("drift_v2", "logs", "remote_" + os.path.basename(out).replace(".json", ".log"))
    os.makedirs(os.path.dirname(log), exist_ok=True)
    rc = subprocess.call(cmd, shell=True, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    print(f"[{i+1}/{len(todo)}] rc={rc} {os.path.basename(out)} ({time.time()-t0:.0f}s)", flush=True)
print("QUEUE DONE", flush=True)
