#!/usr/bin/env python3
"""
Job manager for a HotS remote worker (runs inside WSL; stdlib only).
Manifests: ~/hots/jobs/<name>.json   Logs: ~/hots/logs/<name>.log

  hotsjob.py launch <name> <cmd> [--resume-cmd C] [--save-dir D] [--progress-log L] [--gpu 0]
                                 [--mem-max 24G]
  hotsjob.py pause  <name|all> [--timeout 540]
  hotsjob.py resume <name|all>     (all: paused jobs; a named job may also be failed/died)
  hotsjob.py status [name|all]
  hotsjob.py state  <name>             (just the status word, after the liveness check)
  hotsjob.py finish <name> <rc>        (called by the job's runner shell)

Job kinds
  mcts   cmd runs paper1_revision/train_mcts.py or train_mcts_worker.py. The
         runner sets MCTS_PAUSE_FILE and MCTS_CKPT_EVERY_SEC (default 480 s), so
         the worker writes <save_dir>/resume_state.pt periodically and on pause.
         Pause = create the pause file; the worker checkpoints at the next batch
         boundary and exits 75. A job still in setup (before the self-play loop)
         has nothing to save and is killed at once; it restarts fresh on resume.
         If a job does not exit in time it is killed only when a resume
         checkpoint exists; otherwise it is left running and pause reports failure.
  plain  anything else (benchmarks, tournaments). Pause = SIGTERM to the process
         group (SIGKILL after 30 s); resume re-runs the same command, and those
         scripts skip work whose output files already exist.

Memory cap (--mem-max): the command runs in its own systemd user scope with
MemoryMax=<cap> and MemorySwapMax=0, so a memory blow-up kills only that
command (the kernel OOM-kills inside the scope). The runner shell sits
outside the scope and records the exit code (137 = killed).

A job whose process group is gone with no exit record (the runner itself was
killed, e.g. by a host-wide OOM) is marked failed by every status/state call.
"""
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time

HOTS = os.path.expanduser("~/hots")
JOBS = os.path.join(HOTS, "jobs")
LOGS = os.path.join(HOTS, "logs")
TRAINING = os.path.join(HOTS, "repo", "training")
SELF = os.path.abspath(__file__)
PAUSE_EXIT = 75
DEFAULT_CKPT_SEC = 480
os.makedirs(JOBS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)


def mpath(name):
    return os.path.join(JOBS, f"{name}.json")


def load(name):
    with open(mpath(name)) as f:
        return json.load(f)


def save(m):
    m["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = mpath(m["name"]) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=1)
    os.replace(tmp, mpath(m["name"]))


def all_jobs():
    return sorted(f[:-5] for f in os.listdir(JOBS) if f.endswith(".json"))


def alive(pgid):
    if not pgid:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie leader with no other members still answers killpg; check /proc
    out = subprocess.run(["ps", "-o", "stat=", "-g", str(pgid)], capture_output=True, text=True).stdout
    return any(s and not s.startswith("Z") for s in out.split())


def abspath_t(p):
    return p if os.path.isabs(p) else os.path.join(TRAINING, p)


def infer(cmd):
    """-> (kind, save_dir, progress_log, resume_cmd) for known MCTS entry points."""
    m = re.search(r"paper1_revision/train_mcts\.py\s+(\w+)\s+(\d+)", cmd)
    if m:
        name = f"{m.group(1)}_oof_s{m.group(2)}"
        return ("mcts", f"paper1_revision/mcts_runs/{name}",
                f"paper1_revision/logs/mcts_{name}.log",
                cmd if "--resume" in cmd else cmd + " --resume")
    if "train_mcts_worker.py" in cmd:
        m = re.search(r"MCTS_SAVE_DIR=(\S+)", cmd)
        resume = re.sub(r"MCTS_FRESH=\S+\s*", "", cmd)
        return ("mcts", m.group(1) if m else None, None, "MCTS_FRESH=0 " + resume)
    return ("plain", None, None, cmd)


def ckpt_info(m):
    if not m.get("save_dir"):
        return None
    p = os.path.join(abspath_t(m["save_dir"]), "resume_state.json")
    if not os.path.exists(p) or not os.path.exists(p[:-5] + ".pt"):
        return None
    with open(p) as f:
        info = json.load(f)
    return info


def progress_text(m):
    log = abspath_t(m["progress_log"]) if m.get("progress_log") else m["log"]
    if not os.path.exists(log):
        return ""
    with open(log, errors="replace") as f:
        f.seek(m.get("progress_offset", 0) if os.path.getsize(log) >= m.get("progress_offset", 0) else 0)
        return f.read()


def start(m, cmd):
    """Spawn cmd detached (own session/process group) via a runner shell."""
    env_extra = {"CUDA_VISIBLE_DEVICES": str(m.get("gpu", "0"))}
    if m["kind"] == "mcts":
        env_extra.update({"MCTS_PAUSE_FILE": m["pause_file"],
                          "MCTS_CKPT_EVERY_SEC": str(m.get("ckpt_every_sec", DEFAULT_CKPT_SEC))})
    exports = " ".join(f"export {k}={shlex.quote(v)};" for k, v in env_extra.items())
    run = cmd
    if m.get("mem_max"):
        run = (f"systemd-run --user --scope -q -p MemoryMax={m['mem_max']} "
               f"-p MemorySwapMax=0 bash -c {shlex.quote(cmd)}")
    runner = (f"source {HOTS}/env.sh; {exports} cd {TRAINING}; "
              f"echo \"# start $(date -Is) host=$(hostname) mem_max={m.get('mem_max') or 'none'} cmd: \"{shlex.quote(cmd)}; "
              f"{run}; rc=$?; echo \"# exit=$rc $(date -Is)\"; "
              f"{sys.executable} {SELF} finish {shlex.quote(m['name'])} $rc")
    if m.get("progress_log"):
        pl = abspath_t(m["progress_log"])
        m["progress_offset"] = os.path.getsize(pl) if os.path.exists(pl) else 0
    if os.path.exists(m["pause_file"]):
        os.remove(m["pause_file"])
    logf = open(m["log"], "a")
    p = subprocess.Popen(["bash", "-c", runner], stdout=logf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True, cwd=TRAINING)
    m.update(pgid=p.pid, status="running", started=time.strftime("%Y-%m-%dT%H:%M:%S"),
             current_cmd=cmd, exit_code=None)
    m.setdefault("history", []).append({"t": m["started"], "event": "start", "cmd": cmd})
    save(m)
    time.sleep(2)
    if not alive(p.pid) and load(m["name"])["status"] == "running":
        print(f"WARNING: {m['name']} exited within 2 s, see {m['log']}")
    print(f"{m['name']}: started pgid {p.pid}, log {m['log']}")


def cmd_launch(a):
    name, cmd = a.name, a.cmd
    if os.path.exists(mpath(name)) and load(name)["status"] in ("running", "pausing"):
        sys.exit(f"{name} is already running")
    kind, save_dir, plog, resume_cmd = infer(cmd)
    m = {"name": name, "kind": kind, "cmd": cmd,
         "resume_cmd": a.resume_cmd or resume_cmd,
         "save_dir": a.save_dir or save_dir, "progress_log": a.progress_log or plog,
         "log": os.path.join(LOGS, f"{name}.log"), "gpu": a.gpu,
         "pause_file": os.path.join(JOBS, f"{name}.pause"), "mem_max": a.mem_max,
         "ckpt_every_sec": a.ckpt_every_sec, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if kind == "mcts" and not m["save_dir"]:
        sys.exit("mcts job needs --save-dir (or MCTS_SAVE_DIR=... in the command)")
    start(m, cmd)


def cmd_finish(a):
    m = load(a.name)
    rc = int(a.rc)
    m["exit_code"] = rc
    if rc == PAUSE_EXIT:
        m["status"] = "paused"
    elif rc == 0:
        m["status"] = "completed"
    elif m["status"] == "pausing":
        m["status"] = "paused"
    else:
        m["status"] = "failed"
    ck = ckpt_info(m)
    if ck:
        m["last_checkpoint"] = ck
    m.setdefault("history", []).append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                        "event": f"exit {rc} -> {m['status']}"})
    save(m)


def refresh(m):
    if m["status"] in ("running", "pausing") and not alive(m.get("pgid")):
        m = load(m["name"])  # runner may have just called finish
        if m["status"] in ("running", "pausing"):
            if m["status"] == "pausing":
                m["status"] = "paused"
            else:
                # the runner never called finish: it was killed with the job
                m["status"] = "failed"
                m["failure"] = "process group gone with no exit record (killed, e.g. OOM)"
                m.setdefault("history", []).append(
                    {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": "dead -> failed"})
            save(m)
    ck = ckpt_info(m)
    if ck and ck != m.get("last_checkpoint"):
        m["last_checkpoint"] = ck
        save(m)
    return m


def targets(which):
    names = all_jobs() if which in (None, "all") else [which]
    return [refresh(load(n)) for n in names]


def kill_group(pgid, grace=30):
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    t0 = time.time()
    while alive(pgid) and time.time() - t0 < grace:
        time.sleep(1)
    if alive(pgid):
        os.killpg(pgid, signal.SIGKILL)
        time.sleep(1)


def cmd_pause(a):
    jobs = [m for m in targets(a.name) if m["status"] == "running"]
    if not jobs:
        print("no running jobs")
        return
    t0 = time.time()
    for m in jobs:
        m["status"] = "pausing"
        m.setdefault("history", []).append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": "pause requested"})
        save(m)
        if m["kind"] == "mcts":
            open(m["pause_file"], "w").close()
            if "CUDA kernel engine" not in progress_text(m):
                print(f"{m['name']}: still in setup (no self-play yet), nothing to save: killing")
                m["paused_in_setup"] = True
                save(m)
                kill_group(m["pgid"], grace=10)
        else:
            print(f"{m['name']}: plain job, SIGTERM")
            kill_group(m["pgid"])
    pending = {m["name"] for m in jobs}
    while pending and time.time() - t0 < a.timeout:
        for n in list(pending):
            if not alive(load(n)["pgid"]):
                pending.discard(n)
        time.sleep(2)
    failed = []
    for n in sorted(pending):
        m = load(n)
        ck = ckpt_info(m)
        if ck:
            print(f"{n}: no clean exit after {a.timeout}s; killing (checkpoint @ episode {ck['episode']})")
            kill_group(m["pgid"], grace=15)
        else:
            print(f"{n}: no clean exit and NO checkpoint; left running")
            failed.append(n)
    for m in jobs:
        m = refresh(load(m["name"]))
        if m["status"] == "pausing" and not alive(m["pgid"]):
            m["status"] = "paused"
            save(m)
        ck = m.get("last_checkpoint")
        print(f"{m['name']}: {m['status']}" + (f" (checkpoint @ episode {ck['episode']}, {ck['reason']})"
                                                if ck and not m.get("paused_in_setup") else ""))
    print(f"pause finished in {time.time() - t0:.0f}s")
    if failed:
        sys.exit(1)


def cmd_resume(a):
    # `all` resumes paused jobs; a job named explicitly may also be failed/died
    # (restarted from its last resume checkpoint if it has one).
    ok = ("paused",) if a.name in (None, "all") else ("paused", "failed", "died")
    jobs = [m for m in targets(a.name) if m["status"] in ok]
    if not jobs:
        print("no paused jobs")
    for m in jobs:
        if m["kind"] == "mcts" and not m.pop("paused_in_setup", False) and ckpt_info(m):
            cmd = m["resume_cmd"]
        else:
            cmd = m["cmd"]
        m.setdefault("history", []).append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": "resume"})
        start(m, cmd)


def cmd_status(a):
    for m in targets(a.name):
        ck = m.get("last_checkpoint")
        cks = f"ckpt ep {ck['episode']} ({ck['reason']}, {time.time() - ck['time']:.0f}s ago)" if ck else "no ckpt"
        print(f"{m['name']:<28} {m['status']:<10} {m['kind']:<6} pgid={m.get('pgid')} {cks}\n"
              f"    cmd: {m.get('current_cmd', m['cmd'])}\n    log: {m['log']}"
              + (f"\n    progress: {abspath_t(m['progress_log'])}" if m.get("progress_log") else ""))


def cmd_state(a):
    print(refresh(load(a.name))["status"] if os.path.exists(mpath(a.name)) else "none")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="op", required=True)
    p = sub.add_parser("launch")
    p.add_argument("name")
    p.add_argument("cmd")
    p.add_argument("--resume-cmd")
    p.add_argument("--save-dir")
    p.add_argument("--progress-log")
    p.add_argument("--gpu", default="0")
    p.add_argument("--ckpt-every-sec", type=int, default=DEFAULT_CKPT_SEC)
    p.add_argument("--mem-max", help="cgroup memory cap, e.g. 24G (swap disabled)")
    p = sub.add_parser("pause")
    p.add_argument("name")
    p.add_argument("--timeout", type=int, default=540)
    p = sub.add_parser("resume")
    p.add_argument("name")
    p = sub.add_parser("status")
    p.add_argument("name", nargs="?")
    p = sub.add_parser("state")
    p.add_argument("name")
    p = sub.add_parser("finish")
    p.add_argument("name")
    p.add_argument("rc")
    a = ap.parse_args()
    {"launch": cmd_launch, "pause": cmd_pause, "resume": cmd_resume,
     "status": cmd_status, "state": cmd_state, "finish": cmd_finish}[a.op](a)


if __name__ == "__main__":
    main()
