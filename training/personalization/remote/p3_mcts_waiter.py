"""
Waiter + scheduler for the personalization (P3) MCTS reruns on the remote
workers. Runs on the main box as a light polling loop (ssh only, no compute).

Order of the GPU queue across lanes: paper 1's option B, then drift v2's 30
agents (drift_rebuild/v2/drift_mcts_waiter.py), then this lane. A P3 search
job launches only when ALL of these hold:
  * paper 1 has handed off the host (paper1_revision/site/logs/HANDOFF_<host>);
  * the drift waiter has launched everything: every agent in
    drift_v2/mcts_remote_order.txt has a manifest on some host or is in the
    drift waiter's launch_failed list;
  * the host has a free MCTS slot: at most 2 MCTS-like jobs on the 3090 and 1
    on the 3080, counting every lane's mcts jobs and this lane's p3mcts_* jobs;
  * no MCTS-like job on the host is paused, and no drift/paper-1 run there is
    still in setup (value pretraining, the memory peak);
  * memmon runs there and MemAvailable >= its --min-free-gb + PEAK_GB;
  * the host is marked ready for P3 (~/hots/p3_ready: data synced, kernels
    built with the mid_high_low fallback, p3_mcts_verify passed there);
  * every job the item needs has completed.
CPU-only items of the queue (training the distilled prior, analyses) run on
the 3090 one at a time and use no MCTS slot.
One launch per host per cycle. A job is never launched twice (a manifest of
that name on any host counts as taken). Alerts (unreachable host, failed or
died job, launch error, memmon down) go to personalization/logs/ALERT_p3_mcts
and the log. One instance at a time (flock), no pgrep.

Queue: personalization/remote/p3_mcts_queue.json, a list of
  {"name": "p3mcts_...", "cmd": "...", "needs": [...], "gpu": true|false,
   "hosts": ["max-windows-3090", "3080-gaming-desktop"]}

Usage (from training/):
  setsid nohup nice -n 19 python3 personalization/remote/p3_mcts_waiter.py \\
      >> personalization/logs/p3_mcts_waiter.out 2>&1 &
Test: --dry --once [--assume-gates]
"""
import fcntl
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
TRAINING = os.path.dirname(P3)
sys.path.insert(0, os.path.join(TRAINING, "drift_rebuild", "v2"))
import drift_mcts_waiter as DW  # noqa: E402  (status(), host_facts(), remote())

LOGS = os.path.join(P3, "logs")
LOG = os.path.join(LOGS, "p3_mcts_waiter.log")
ALERT = os.path.join(LOGS, "ALERT_p3_mcts")
STATE = os.path.join(LOGS, "p3_mcts_waiter_state.json")
QUEUE = os.path.join(HERE, "p3_mcts_queue.json")
JOB = os.path.join(HERE, "p3_job.sh")
DRIFT_ORDER = os.path.join(TRAINING, "drift_v2", "mcts_remote_order.txt")
DRIFT_STATE = os.path.join(TRAINING, "drift_v2", "logs", "drift_mcts_waiter_state.json")
HANDOFF = os.path.join(TRAINING, "paper1_revision", "site", "logs", "HANDOFF_{}")

MAX_SLOTS = {"max-windows-3090": 1, "3080-gaming-desktop": 1}  # 3090: one MCTS job at a time (2026-10-09; two processes time-slice the GPU, no gain)
MEM = {"max-windows-3090": "24G", "3080-gaming-desktop": "20G"}
CORES = {"max-windows-3090": "28-35", "3080-gaming-desktop": "0-19"}
PEAK_GB = 14
CYCLE = 300
DRY = "--dry" in sys.argv
ONCE = "--once" in sys.argv
ASSUME = DRY and "--assume-gates" in sys.argv
if DRY:
    LOG, ALERT, STATE = (x + ".dry" for x in (LOG, ALERT, STATE))


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def alert(msg, key=None, state=None):
    if key is not None and state is not None:
        if key in state["alerted"]:
            return
        state["alerted"].append(key)
        json.dump(state, open(STATE, "w"))
    log(f"ALERT {msg}")
    with open(ALERT, "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def mcts_like(name, job):
    return job["kind"] == "mcts" or name.startswith("p3mcts_")


def drift_all_launched(st):
    order = [l.strip() for l in open(DRIFT_ORDER) if l.strip()]
    failed = set()
    if os.path.exists(DRIFT_STATE):
        failed = set(json.load(open(DRIFT_STATE)).get("launch_failed", []))
    seen = {n for jobs in st.values() for n in jobs}
    pending = [n for n in order if n not in seen and n not in failed]
    return not pending, pending


def ready(host):
    out = DW.remote(host, "test -f ~/hots/p3_ready && echo READY || echo NOTREADY")
    return out is not None and "READY" in out.split()


def main():
    os.makedirs(LOGS, exist_ok=True)
    lock = open(os.path.join(LOGS, "p3_mcts_waiter.lock" + (".dry" if DRY else "")), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another p3_mcts_waiter holds the lock")
    if not DRY:
        open(os.path.join(LOGS, "p3_mcts_waiter.pid"), "w").write(f"{os.getpid()}\n")
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    state.setdefault("alerted", [])
    state.setdefault("launch_failed", [])
    log(f"start pid {os.getpid()}; slots {MAX_SLOTS}; peak {PEAK_GB} GB")
    said = set()
    while True:
        queue = json.load(open(QUEUE))
        names = [q["name"] for q in queue]
        st = {h: DW.status(h) for h in MAX_SLOTS}
        down = [h for h, v in st.items() if v is None]
        if down:
            alert(f"unreachable (ssh/WSL): {', '.join(down)}; no launches this cycle")
            time.sleep(CYCLE)
            continue
        ours = {n: (h, j) for h, jobs in st.items() for n, j in jobs.items() if n in names}
        for n, (h, j) in ours.items():
            if j["status"] in ("failed", "died", "paused"):
                alert(f"{n} on {h} is {j['status']} (log {j['log']})", key=f"{n}:{h}:{j['status']}", state=state)
        done = {n for n, (h, j) in ours.items() if j["status"] == "completed"}
        if all(n in done or n in state["launch_failed"] for n in names):
            log("all P3 queue items completed")
            return
        gate, pend = drift_all_launched(st)
        if not gate and not ASSUME:
            if "drift" not in said:
                log(f"waiting for the drift waiter to launch everything ({len(pend)} pending, e.g. {pend[:3]})")
                said.add("drift")
            time.sleep(CYCLE)
            continue
        todo = [q for q in queue if q["name"] not in ours and q["name"] not in state["launch_failed"]
                and all(x in done for x in q.get("needs", []))]
        cpu_busy = any(n in ours and ours[n][1]["status"] in ("running", "starting", "pausing")
                       for n in names if not next(q for q in queue if q["name"] == n).get("gpu", True))
        for host, cap in MAX_SLOTS.items():
            if not todo:
                break
            if not os.path.exists(HANDOFF.format(host)) and not ASSUME:
                continue
            jobs = st[host]
            if not ASSUME and not ready(host):
                if f"ready:{host}" not in said:
                    log(f"{host}: not marked ready for P3 (~/hots/p3_ready); skipping")
                    said.add(f"ready:{host}")
                continue
            live = {n: j for n, j in jobs.items() if mcts_like(n, j) and j["status"] in ("running", "pausing", "starting")}
            if any(mcts_like(n, j) and j["status"] == "paused" for n, j in jobs.items()):
                continue
            facts = DW.host_facts(host, {n: j for n, j in live.items() if j["kind"] == "mcts"})
            if facts is None:
                alert(f"{host}: facts check did not answer")
                continue
            min_free, avail, setup = facts
            if min_free is None:
                alert(f"{host}: memmon is not running; no P3 launches there", key=f"memmon:{host}", state=state)
                continue
            if setup or avail < min_free + PEAK_GB:
                continue
            cand = [q for q in todo if host in q.get("hosts", list(MAX_SLOTS))]
            gpu_ok = len(live) < cap
            pick = next((q for q in cand if (q.get("gpu", True) and gpu_ok) or
                         (not q.get("gpu", True) and not cpu_busy)), None)
            if pick is None:
                if DRY:
                    log(f"[dry] {host}: nothing to launch (live MCTS {len(live)}/{cap}, cpu busy {cpu_busy}, "
                        f"avail {avail:.0f} GB)")
                continue
            todo.remove(pick)
            name = pick["name"]
            if DRY:
                log(f"[dry] would launch {name} on {host} (avail {avail:.0f} GB, live {len(live)}/{cap})")
                continue
            env = dict(os.environ, P3_CORES=CORES[host], P3_THREADS="2", P3_MEM=MEM[host])
            r = subprocess.run([JOB, host, name] + pick["cmd"].split(), capture_output=True, text=True,
                               cwd=os.path.dirname(TRAINING), env=env, timeout=300)
            tail = (r.stdout.strip().splitlines() or [r.stderr.strip()[-300:]])[-1]
            log(f"launch {name} on {host} (avail {avail:.0f} GB, live {len(live)}/{cap}): rc={r.returncode} {tail}")
            if r.returncode != 0:
                alert(f"launch of {name} on {host} failed: rc={r.returncode} {tail}; not retried")
                state["launch_failed"].append(name)
                json.dump(state, open(STATE, "w"))
            if not pick.get("gpu", True):
                cpu_busy = True
        if ONCE:
            return
        time.sleep(CYCLE)


if __name__ == "__main__":
    main()
