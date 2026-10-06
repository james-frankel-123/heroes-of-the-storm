"""
P3 backfill waiter (main box; a status query of the local store every 6
hours, otherwise ssh only). When the player refetch is nearly done (fewer
than MIN_LEFT replays left below the cursor) or on DEADLINE, whichever
comes first:
  1. export stamps and games from the local read-only store (p3_export.py,
     builds up to 2026-09-27 only) under tag full_<date>
  2. copy the export to the 3090
  3. once no other P3 chain runs there, launch p3_full on the 3090: parse,
     build the full-history cache (p3_c_history.py), then the full chain
     (p3_pipe.py full) with P3_CACHE=cache_full, P3_RESULTS=results/oct26_full
Alerts: personalization/logs/ALERT_p3_backfill. One instance (flock).
Usage (from the repo root): setsid nohup nice -n 19 python3 \\
    training/personalization/remote/p3_backfill_waiter.py >> training/personalization/logs/p3_backfill.out 2>&1 &
"""
import datetime
import fcntl
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(P3))
LOGS = os.path.join(P3, "logs")
LOG = os.path.join(LOGS, "p3_backfill.log")
ALERT = os.path.join(LOGS, "ALERT_p3_backfill")
STATE = os.path.join(LOGS, "p3_backfill_state.json")
H90 = "max-windows-3090"
MIN_LEFT = 5000
DEADLINE = datetime.date(2026, 10, 16)
CYCLE = 6 * 3600


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def alert(msg):
    log(f"ALERT {msg}")
    with open(ALERT, "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def env():
    e = dict(os.environ)
    for line in open(os.path.join(REPO, ".env")):
        m = re.match(r"^([A-Z_0-9]+)=(.*)$", line.strip())
        if m:
            e[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return e


def remaining():
    r = subprocess.run(["npx", "tsx", "sync/refetch-players.ts", "--status"], cwd=REPO, env=env(),
                       capture_output=True, text=True, timeout=600)
    m = re.search(r"remaining below cursor:\s*([\d,]+)", r.stdout)
    return int(m.group(1).replace(",", "")) if m else None


def remote_state(job):
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=30", H90, "wsl -e bash -s"],
                           input=f"source ~/hots/env.sh; python ~/hots/repo/training/remote_workers/hotsjob.py state {job}\n",
                           capture_output=True, text=True, timeout=120)
        return r.stdout.split()[-1] if r.stdout.split() else "none"
    except subprocess.TimeoutExpired:
        return None


def main():
    os.makedirs(LOGS, exist_ok=True)
    lock = open(os.path.join(LOGS, "p3_backfill.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another backfill waiter holds the lock")
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    log(f"start pid {os.getpid()} state {state}")
    while True:
        if "tag" not in state:
            left = remaining()
            today = datetime.date.today()
            log(f"refetch remaining below cursor: {left}")
            if left is None:
                alert("could not read the refetch status")
            if (left is not None and left < MIN_LEFT) or today >= DEADLINE:
                tag = f"full_{today.strftime('%m%d')}"
                for what in ("stamps", "games"):
                    r = subprocess.run(["nice", "-n", "19", "python3", "training/personalization/p3_export.py", what,
                                        "--tag", tag], cwd=REPO, env=env(), capture_output=True, text=True)
                    log(f"export {what}: rc={r.returncode} {r.stdout.strip()[-300:]}")
                    if r.returncode != 0:
                        alert(f"export {what} failed: {r.stderr[-300:]}")
                        return
                state["tag"] = tag
                state["remaining_at_export"] = left
                json.dump(state, open(STATE, "w"))
        if "tag" in state and "shipped" not in state:
            src = f"training/personalization/cache/export/{state['tag']}/"
            r = subprocess.run(["nice", "-n", "19", "rsync", "-rlt", "--rsync-path=wsl rsync", src,
                                f"{H90}:/home/max/hots/repo/{src}"], cwd=REPO, capture_output=True, text=True)
            if r.returncode != 0:
                alert(f"rsync of the export failed: {r.stderr[-300:]}")
            else:
                state["shipped"] = True
                json.dump(state, open(STATE, "w"))
        if state.get("shipped") and "launched" not in state:
            busy = [j for j in ("p3_fixes", "p3_drafter") if remote_state(j) in ("running", "pausing", "starting")]
            if busy:
                log(f"waiting for {busy} on {H90}")
            else:
                subprocess.run([os.path.join(REPO, "training", "remote_workers", "sync.sh"), H90, "code"], cwd=REPO)
                cmd = ["bash", "personalization/remote/run_full.sh", state["tag"]]
                r = subprocess.run([os.path.join(HERE, "p3_job.sh"), H90, "p3_full"] + cmd, cwd=REPO,
                                   capture_output=True, text=True)
                log(f"launch p3_full: rc={r.returncode} {r.stdout.strip()[-200:]}")
                if r.returncode != 0:
                    alert(f"launch of p3_full failed: {r.stderr[-300:]}")
                    return
                state["launched"] = True
                json.dump(state, open(STATE, "w"))
        if state.get("launched"):
            st = remote_state("p3_full")
            if st in ("completed", "failed", "died"):
                (log if st == "completed" else alert)(f"p3_full {st}")
                return
        time.sleep(CYCLE if "tag" not in state else 1800)


if __name__ == "__main__":
    main()
