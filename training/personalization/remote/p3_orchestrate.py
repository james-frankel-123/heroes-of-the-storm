"""
P3 orchestrator (main box, ssh only, no compute): runs the October 2026
rerun steps in order, each once, and alerts on failure.

  1. wait for p3_fixes (fixes chain) and p3_gd_trainonly2 (train-window GD
     and BC prior) on the 3090
  2. launch the drafter chain (p3_drafter) on the 3090; wait
  3. MCTS verification on the 3090 (chance and rollfwd, both assign modes
     inside): p3_mcts_verify must print VERIFY OK; then mark the 3090 ready
     (~/hots/p3_ready)
  4. stage the 3080: copy the P3 inputs and derived caches 3090 -> here ->
     3080, sync code, build the kernels, verify there; mark it ready
  5. start the MCTS waiter (p3_mcts_waiter.py) if it is not running
State: personalization/logs/p3_orch_state.json (finished steps).
Alerts: personalization/logs/ALERT_p3_orch. One instance (flock).
Usage (from training/): setsid nohup nice -n 19 python3 personalization/remote/p3_orchestrate.py \\
    >> personalization/logs/p3_orch.out 2>&1 &
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
REPO = os.path.dirname(TRAINING)
LOGS = os.path.join(P3, "logs")
LOG = os.path.join(LOGS, "p3_orch.log")
ALERT = os.path.join(LOGS, "ALERT_p3_orch")
STATE = os.path.join(LOGS, "p3_orch_state.json")
H90, H80 = "max-windows-3090", "3080-gaming-desktop"
JOB = os.path.join(HERE, "p3_job.sh")
STAGE = "/tmp/claude-1000/-home-max-heroes-of-the-storm/a49813e0-7f18-4a49-90f1-0426c355d609/scratchpad/p3_stage_3080"
POLL = 300


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def alert(msg):
    log(f"ALERT {msg}")
    with open(ALERT, "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def remote(host, script, timeout=600):
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=30", host, "wsl -e bash -s"],
                           input=script + "\necho __ok\n", capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return r.stdout if "__ok" in r.stdout else None


def state_of(host, job):
    out = remote(host, f"source ~/hots/env.sh; python ~/hots/repo/training/remote_workers/hotsjob.py state {job}")
    if out is None:
        return None
    w = [x for x in out.split() if x != "__ok"]
    return w[-1] if w else "none"


def wait_job(host, job):
    """Block until the job leaves running; return its final state."""
    while True:
        st = state_of(host, job)
        if st in ("completed", "failed", "died", "none"):
            return st
        if st == "paused":
            log(f"{job} on {host} is paused; waiting")
        time.sleep(POLL)


def launch(host, name, cmd, **env):
    e = dict(os.environ, **env)
    r = subprocess.run([JOB, host, name] + cmd, capture_output=True, text=True, cwd=REPO, env=e, timeout=300)
    log(f"launch {name} on {host}: rc={r.returncode} {(r.stdout.strip().splitlines() or [r.stderr[-200:]])[-1]}")
    return r.returncode == 0


def run_step(state, name, fn):
    if name in state["done"]:
        return True
    log(f"step {name}: start")
    ok = fn()
    if ok:
        state["done"].append(name)
        json.dump(state, open(STATE, "w"))
        log(f"step {name}: done")
    else:
        alert(f"step {name} failed; orchestrator stops (fix, then restart it)")
    return ok


def job_ok(host, name, cmd, grep=None, **env):
    st = state_of(host, name)
    if st in (None, "none") or st == "failed":
        if not launch(host, name, cmd, **env):
            return False
    st = wait_job(host, name)
    if st != "completed":
        alert(f"{name} on {host} ended {st}")
        return False
    if grep:
        out = remote(host, f"grep -c '{grep}' ~/hots/logs/{name}.log")
        if out is None or not out.split()[0].isdigit() or int(out.split()[0]) == 0:
            alert(f"{name} on {host}: '{grep}' not in its log")
            return False
    return True


def verify(host, tag, cores, mem):
    ok = True
    for mode in ("chance", "rollfwd"):
        ok &= job_ok(host, f"p3verify_{tag}_{mode}",
                     ["env", f"P3_SEARCH_MODE={mode}", "P3_ASSIGN=team", "python",
                      "personalization/p3_mcts_verify.py", "--search-mode", mode],
                     grep="VERIFY OK", P3_CORES=cores, P3_THREADS="2", P3_MEM=mem)
    return ok


def stage_3080():
    os.makedirs(STAGE, exist_ok=True)
    src = f"{H90}:/home/max/hots/repo/training/personalization/"
    r1 = subprocess.run(["nice", "-n", "19", "rsync", "-rlt", "--rsync-path=wsl rsync", "-z",
                         "--exclude", "cache/_legacy_pre_oct26/", "--exclude", "cache/mmr_history/",
                         "--exclude", "cache/export/", "--exclude", "cache/mcts_v2/", "--exclude", "cache/fix/",
                         "--exclude", "cache/gd_trainonly/train/", "--exclude", "cache/gd_trainonly/hold/",
                         "--include", "cache/***", "--include", "results/", "--include", "results/oct26/***",
                         "--exclude", "*", src, STAGE + "/"], capture_output=True, text=True)
    if r1.returncode != 0:
        alert(f"rsync from {H90} failed: {r1.stderr[-300:]}")
        return False
    dst = f"{H80}:/home/max/hots/repo/training/personalization/"
    r2 = subprocess.run(["nice", "-n", "19", "rsync", "-rlt", "--rsync-path=wsl rsync", "-z", STAGE + "/", dst],
                        capture_output=True, text=True)
    if r2.returncode != 0:
        alert(f"rsync to {H80} failed: {r2.stderr[-300:]}")
        return False
    # population WP, patch stats and the snapshot-independent inputs of the search hosts
    lst = subprocess.run([os.path.join(HERE, "p3_data_manifest.sh")], capture_output=True, text=True).stdout
    keep = "\n".join(l for l in lst.splitlines() if not l.startswith("training/personalization/cache/"))
    r3 = subprocess.run(["nice", "-n", "19", "rsync", "-rlt", "--rsync-path=wsl rsync", "-z",
                         "--files-from=-", "./", f"{H80}:/home/max/hots/repo/"], input=keep, text=True,
                        capture_output=True, cwd=REPO)
    if r3.returncode != 0:
        alert(f"rsync of WP inputs to {H80} failed: {r3.stderr[-300:]}")
        return False
    subprocess.run([os.path.join(TRAINING, "remote_workers", "sync.sh"), H80, "code"], cwd=REPO,
                   capture_output=True, text=True)
    return job_ok(H80, "p3_build_kernels_3080", ["bash", "personalization/build_p3_kernels.sh"],
                  grep="BUILD OK", P3_CORES="0-3", P3_THREADS="2", P3_MEM="12G")


def main():
    os.makedirs(LOGS, exist_ok=True)
    lock = open(os.path.join(LOGS, "p3_orch.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another orchestrator holds the lock")
    open(os.path.join(LOGS, "p3_orch.pid"), "w").write(f"{os.getpid()}\n")
    state = json.load(open(STATE)) if os.path.exists(STATE) else {"done": []}
    log(f"start pid {os.getpid()}; done so far {state['done']}")
    steps = [
        ("wait_fixes", lambda: wait_job(H90, "p3_fixes") == "completed"),
        ("wait_gd", lambda: wait_job(H90, "p3_gd_trainonly2") == "completed"),
        ("sync_code_3090", lambda: subprocess.run(
            [os.path.join(TRAINING, "remote_workers", "sync.sh"), H90, "code"], cwd=REPO,
            capture_output=True).returncode == 0),
        ("drafter_chain", lambda: job_ok(H90, "p3_drafter", ["python", "personalization/p3_pipe.py", "drafter"],
                                         grep="chain drafter complete")),
        ("verify_3090", lambda: verify(H90, "3090", "32-35", "24G")),
        ("ready_3090", lambda: remote(H90, "touch ~/hots/p3_ready") is not None),
        ("stage_3080", stage_3080),
        ("verify_3080", lambda: verify(H80, "3080", "0-3", "20G")),
        ("ready_3080", lambda: remote(H80, "touch ~/hots/p3_ready") is not None),
        ("start_waiter", lambda: subprocess.Popen(
            ["setsid", "nohup", "nice", "-n", "19", "python3", os.path.join(HERE, "p3_mcts_waiter.py")],
            stdout=open(os.path.join(LOGS, "p3_mcts_waiter.out"), "a"), stderr=subprocess.STDOUT,
            cwd=TRAINING, start_new_session=True) is not None),
    ]
    for name, fn in steps:
        if not run_step(state, name, fn):
            return
    log("all steps done")


if __name__ == "__main__":
    main()
