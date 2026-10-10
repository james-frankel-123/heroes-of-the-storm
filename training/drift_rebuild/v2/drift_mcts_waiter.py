"""
Waiter + scheduler for the drift v2 MCTS agents on the remote workers. Runs on
the main box as a light polling loop (ssh only, no compute).

A host is used only after paper 1's option-B scheduler has written
paper1_revision/site/logs/HANDOFF_<host> (every option-B run launched). Then,
every cycle, per host:
  * slots: at most MAX_SLOTS[host] MCTS runs of any lane (3090 2, 3080 1);
  * no launch while an MCTS job there is paused (memmon or the owner) or while
    an MCTS run there is still in setup/value pretraining (the memory peak);
  * memmon must be running there, and MemAvailable must exceed memmon's
    --min-free-gb plus PEAK_GB;
  * one launch per host per cycle, in drift_v2/mcts_remote_order.txt order, via
    launch_mcts_remote.sh (per-process cap MEM_MAX; the fixed-kernel and
    save-dir guards live in drift_rebuild/v2/run.py).
An agent is never launched twice: a v2_* manifest of that name on any host
counts as taken. An unreachable host stops all launches for the cycle.
Alerts (unreachable host, failed/died/paused agent, launch error, memmon down)
go to drift_v2/logs/ALERT_drift_mcts and the log. One instance at a time
(flock on drift_v2/logs/drift_mcts_waiter.lock; no pgrep).

Usage (from training/):
  setsid nohup nice -n 19 python3 drift_rebuild/v2/drift_mcts_waiter.py \
      >> drift_v2/logs/drift_mcts_waiter.out 2>&1 &
Log: drift_v2/logs/drift_mcts_waiter.log   PID: drift_v2/logs/drift_mcts_waiter.pid
Test: --dry --once [--assume-handoff] runs one cycle and logs what it would launch.
"""
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import sys as _rw_sys, os as _rw_os
_rw_sys.path.insert(0, _rw_os.path.join(_rw_os.path.dirname(_rw_os.path.abspath(__file__)), *(['..'] * (2 if 'drift_rebuild' in __file__ or '/remote/' in __file__ else 1)), 'remote_workers'))
import hosts as _rw_hosts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(os.path.dirname(HERE))
DV2 = os.path.join(TRAINING, "drift_v2")
LOGS = os.path.join(DV2, "logs")
LOG = os.path.join(LOGS, "drift_mcts_waiter.log")
ALERT = os.path.join(LOGS, "ALERT_drift_mcts")
STATE = os.path.join(LOGS, "drift_mcts_waiter_state.json")
ORDER = os.path.join(DV2, "mcts_remote_order.txt")
HANDOFF = os.path.join(TRAINING, "paper1_revision", "site", "logs", "HANDOFF_{}")
RW = os.path.join(TRAINING, "remote_workers")
LAUNCH = os.path.join(HERE, "launch_mcts_remote.sh")

MAX_SLOTS = {"max-windows-3090": 1, "3080-gaming-desktop": 1, "windows-5090-wsl": 1}  # one MCTS job per GPU (2026-10-09/10: two processes time-slice the GPU; two runs in one process gave no gain in full training)
# Per-process data cap. Drift agents load the full 2.9 GB snapshot for value
# pretraining; paper-1 runs on that snapshot peaked near 21 GB RSS (the 12G
# paper-1 cap is sized for its 0.6 GB lite snapshot).
MEM_MAX = os.environ.get("DRIFT_MEM_MAX", "32G")
PEAK_GB = 24
CYCLE = 300
DRY = "--dry" in sys.argv
ONCE = "--once" in sys.argv
ASSUME_HANDOFF = DRY and "--assume-handoff" in sys.argv
if DRY:
    LOG, ALERT, STATE = (x + ".dry" for x in (LOG, ALERT, STATE))
PRETRAIN_DONE = r"pre-training complete|pretraining skipped|^Episode |Resumed from"


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def alert(msg, key=None, state=None):
    """Log and append to the ALERT flag file; with key, only once per key."""
    if key is not None and state is not None:
        if key in state["alerted"]:
            return
        state["alerted"].append(key)
        json.dump(state, open(STATE, "w"))
    log(f"ALERT {msg}")
    with open(ALERT, "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def remote(host, script, timeout=180):
    """Run a bash script in host's WSL; stdout, or None if it did not answer."""
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=30", host, _rw_hosts.shell(host)],
                           input=script + "\necho __ok\n", capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return r.stdout if "__ok" in r.stdout else None


def status(host):
    """{name: {status, kind, log, progress}} for every manifest on host, or None."""
    try:
        r = subprocess.run([os.path.join(RW, "jobs_remote.sh"), host], capture_output=True,
                           text=True, cwd=TRAINING, timeout=240)
    except subprocess.TimeoutExpired:
        return None
    text = (r.stdout + r.stderr).replace("\0", "")
    if (r.returncode != 0 or "HCS_E" in text or "Wsl/" in text
            or not re.search(r"^\S+\s+(running|completed|failed|paused|died|pausing|starting)\b",
                             r.stdout, re.M)):
        return None
    jobs, cur = {}, None
    for line in r.stdout.splitlines():
        m = re.match(r"^(\S+)\s+(\w+)\s+(\w+)\s+pgid=", line)
        if m:
            cur = m.group(1)
            jobs[cur] = {"status": m.group(2), "kind": m.group(3), "log": None, "progress": None}
            continue
        m = re.match(r"^\s+(log|progress):\s+(\S+)", line)
        if m and cur:
            jobs[cur][m.group(1)] = m.group(2)
    return jobs


def host_facts(host, live):
    """(memmon_min_free_gb or None if memmon is not running, MemAvailable GB,
    [live MCTS jobs still in setup/value pretraining]) or None."""
    checks = []
    for name, j in live.items():
        f = j["progress"] or j["log"]
        if not f:
            checks.append(f"echo SETUP {name}")
            continue
        # the text after the last '# start' line (whole file if none)
        checks.append(f"awk '/^# start/{{b=\"\"}} {{b=b $0 \"\\n\"}} END{{printf \"%s\", b}}' {f} 2>/dev/null"
                      f" | grep -qE '{PRETRAIN_DONE}' || echo SETUP {name}")
    script = "\n".join([
        "python3 -c \"import json;m=json.load(open('/home/max/hots/jobs/memmon.json'));"
        "print('MEMMON', m['status'], m['cmd'])\" 2>/dev/null || echo MEMMON none",
        "awk '/MemAvailable/{print \"AVAIL\", $2/1048576}' /proc/meminfo",
    ] + checks)
    out = remote(host, script)
    if out is None:
        return None
    min_free, avail, setup = None, 0.0, []
    for line in out.splitlines():
        if line.startswith("MEMMON running"):
            m = re.search(r"--min-free-gb\s+(\d+)", line)
            min_free = float(m.group(1)) if m else 40.0
        elif line.startswith("AVAIL"):
            avail = float(line.split()[1])
        elif line.startswith("SETUP"):
            setup.append(line.split()[1])
    return min_free, avail, setup


def main():
    os.makedirs(LOGS, exist_ok=True)
    lock = open(os.path.join(LOGS, "drift_mcts_waiter.lock" + (".dry" if DRY else "")), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another drift_mcts_waiter holds the lock")
    if not DRY:
        open(os.path.join(LOGS, "drift_mcts_waiter.pid"), "w").write(f"{os.getpid()}\n")
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    state.setdefault("alerted", [])
    state.setdefault("launch_failed", [])
    log(f"start pid {os.getpid()}; slots {MAX_SLOTS}; mem cap {MEM_MAX}")
    waiting_logged = set()
    while True:
        order = [l.strip() for l in open(ORDER) if l.strip()]
        st = {h: status(h) for h in MAX_SLOTS}
        down = [h for h, v in st.items() if v is None]
        seen = state.setdefault("last_seen", {})  # host -> agents last seen there
        for h, jobs in st.items():
            if jobs is not None:
                seen[h] = sorted(n for n in jobs if n in order)
        if down:
            unknown = [h for h in down if h not in seen]
            if unknown:
                # no record of what runs there: an agent could be launched twice
                alert(f"unreachable (ssh/WSL), never seen: {', '.join(unknown)}; no launches this cycle")
                time.sleep(CYCLE)
                continue
            alert(f"unreachable (ssh/WSL): {', '.join(down)}; its last-seen agents count as taken, "
                  f"launches continue elsewhere", key="down:" + ",".join(sorted(down)), state=state)
        json.dump(state, open(STATE, "w"))
        v2 = {n: (h, j) for h, jobs in st.items() if jobs is not None for n, j in jobs.items() if n in order}
        for h in down:
            for n in seen.get(h, []):
                v2.setdefault(n, (h, {"status": "unreachable", "kind": "mcts", "log": "?"}))
        for n, (h, j) in v2.items():
            if j["status"] in ("failed", "died", "paused"):
                alert(f"{n} on {h} is {j['status']} (log {j['log']})", key=f"{n}:{h}:{j['status']}",
                      state=state)
        pending = [n for n in order if n not in v2 and n not in state["launch_failed"]]
        if not pending and all(j["status"] == "completed" for _, j in v2.values()):
            log("all agents completed")
            return
        for host, cap in MAX_SLOTS.items():
            if not pending:
                break
            if host in down:
                continue
            if not os.path.exists(HANDOFF.format(host)) and not ASSUME_HANDOFF:
                if host not in waiting_logged:
                    log(f"{host}: waiting for HANDOFF")
                    waiting_logged.add(host)
                continue
            jobs = st[host]
            live = {n: j for n, j in jobs.items()
                    if j["kind"] == "mcts" and j["status"] in ("running", "pausing", "starting")}
            if len(live) >= cap:
                if DRY:
                    log(f"[dry] {host}: {len(live)}/{cap} MCTS slots busy ({', '.join(live)})")
                continue
            if any(j["kind"] == "mcts" and j["status"] == "paused" for j in jobs.values()):
                if DRY:
                    log(f"[dry] {host}: an MCTS job is paused")
                continue
            facts = host_facts(host, live)
            if facts is None:
                alert(f"{host}: facts check did not answer; no launch this cycle")
                continue
            min_free, avail, setup = facts
            if min_free is None:
                alert(f"{host}: memmon is not running; no drift launches there", key=f"memmon:{host}",
                      state=state)
                continue
            if setup or avail < min_free + PEAK_GB:
                if DRY:
                    log(f"[dry] {host}: hold (setup {setup}, avail {avail:.0f} GB, need {min_free + PEAK_GB:.0f})")
                continue
            name = pending.pop(0)
            if DRY:
                log(f"[dry] would launch {name} on {host} (avail {avail:.0f} GB, live {len(live)}/{cap})")
                continue
            r = subprocess.run([LAUNCH, host, name], capture_output=True, text=True, cwd=TRAINING,
                               env=dict(os.environ, MEM_MAX=MEM_MAX, RUN_NAME=name), timeout=300)
            tail = (r.stdout.strip().splitlines() or [r.stderr.strip()[-300:]])[-1]
            log(f"launch {name} on {host} (avail {avail:.0f} GB, live {len(live)}/{cap}): rc={r.returncode} {tail}")
            if r.returncode != 0 or "WARNING" in r.stdout:
                alert(f"launch of {name} on {host} failed: rc={r.returncode} {tail}; not retried")
                state["launch_failed"].append(name)
                json.dump(state, open(STATE, "w"))
        if ONCE:
            return
        time.sleep(CYCLE)


if __name__ == "__main__":
    main()
