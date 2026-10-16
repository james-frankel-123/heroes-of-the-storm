"""
P3 A2 waiter (main box). Waits for the A1 chain (p3_full on the 3090,
launched by p3_backfill_waiter.py), then:
  1. builds the merged store on A1's export (p3_full_history.py build, nice
     19, cores 48-51) and ships history.npz (no battletags) to the 3090
  2. launches the "merged" chain of p3_pipe.py on cache_full as hotsjob job
     p3_merged (cores 28-31, 48G): extra rows, arms A1, A2, A2-seen and the
     slot-count variants, then the report against A0 (results/oct26_p3c)
Alerts: personalization/logs/ALERT_p3_merged. One instance (flock).
Usage (from the repo root): setsid nohup nice -n 19 python3 \\
    training/personalization/remote/p3_merged_waiter.py >> training/personalization/logs/p3_merged.out 2>&1 &
"""
import fcntl
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "remote_workers"))
import hosts as _rw_hosts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(P3))
LOGS = os.path.join(P3, "logs")
LOG = os.path.join(LOGS, "p3_merged.log")
ALERT = os.path.join(LOGS, "ALERT_p3_merged")
STATE = os.path.join(LOGS, "p3_merged_state.json")
BACKFILL_STATE = os.path.join(LOGS, "p3_backfill_state.json")
H90 = "max-windows-3090"
RBASE = "/home/max/hots/repo/training/personalization"


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def alert(msg):
    log(f"ALERT {msg}")
    with open(ALERT, "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def remote(script, timeout=300):
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=30", H90, _rw_hosts.shell(H90)], input=script,
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout


def job_state(job):
    try:
        _, out = remote(f"source ~/hots/env.sh; python ~/hots/repo/training/remote_workers/hotsjob.py state {job}\n",
                        timeout=120)
        return out.split()[-1] if out.split() else "none"
    except subprocess.TimeoutExpired:
        return None


def main():
    os.makedirs(LOGS, exist_ok=True)
    lock = open(os.path.join(LOGS, "p3_merged.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another merged waiter holds the lock")
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    log(f"start pid {os.getpid()} state {state}")
    while True:
        if "a1_done" not in state:
            bf = json.load(open(BACKFILL_STATE)) if os.path.exists(BACKFILL_STATE) else {}
            st = job_state("p3_full") if bf.get("launched") else None
            log(f"backfill state {bf}; p3_full {st}")
            if st == "completed":
                state["a1_done"] = True
                state["tag"] = bf["tag"]
                json.dump(state, open(STATE, "w"))
            elif st in ("failed", "died"):
                alert(f"p3_full {st}; not launching A2")
                return
        if state.get("a1_done") and "store" not in state:
            tag = state["tag"]
            r = subprocess.run(["nice", "-n", "19", "taskset", "-c", "48-51", "python3",
                                "personalization/p3_full_history.py", "build", "--tag", tag],
                               cwd=os.path.join(REPO, "training"), capture_output=True, text=True)
            log(f"store build {tag}: rc={r.returncode} {r.stdout.strip()[-300:]}")
            if r.returncode != 0:
                alert(f"store build failed: {r.stderr[-300:]}")
                return
            dst = f"{RBASE}/cache_full/full_history/{tag}/"
            remote(f"mkdir -p {dst}\n")
            r = subprocess.run(["nice", "-n", "19", "rsync", "-rlt", f"--rsync-path={_rw_hosts.rsync_path(H90)}",
                                f"training/personalization/cache/full_history/{tag}/history.npz",
                                f"training/personalization/cache/full_history/{tag}/meta.json",
                                f"{H90}:{dst}"], cwd=REPO, capture_output=True, text=True)
            if r.returncode != 0:
                alert(f"store rsync failed: {r.stderr[-300:]}")
                return
            state["store"] = dst + "history.npz"
            json.dump(state, open(STATE, "w"))
        if state.get("store") and "launched" not in state:
            busy = [j for j in ("p3_full", "p3c_drafter2") if job_state(j) in ("running", "pausing", "starting")]
            if busy:
                log(f"waiting for {busy}")
            else:
                subprocess.run([os.path.join(REPO, "training", "remote_workers", "sync.sh"), H90, "code"], cwd=REPO)
                cmd = ["env", f"P3_CACHE={RBASE}/cache_full", f"P3_STORE={state['store']}",
                       f"P3_RESULTS={RBASE}/results/oct26_p3c", f"P3_EXPORT_TAG={state['tag']}",
                       "python", "personalization/p3_pipe.py", "merged"]
                r = subprocess.run([os.path.join(HERE, "p3_job.sh"), H90, "p3_merged"] + cmd, cwd=REPO,
                                   capture_output=True, text=True,
                                   env={**os.environ, "P3_CORES": "28-31", "P3_MEM": "48G"})
                log(f"launch p3_merged: rc={r.returncode} {r.stdout.strip()[-200:]}")
                if r.returncode != 0:
                    alert(f"launch of p3_merged failed: {r.stderr[-300:]}")
                    return
                state["launched"] = True
                json.dump(state, open(STATE, "w"))
        if state.get("launched"):
            st = job_state("p3_merged")
            if st in ("completed", "failed", "died"):
                (log if st == "completed" else alert)(f"p3_merged {st}")
                return
        time.sleep(1800)


if __name__ == "__main__":
    main()
