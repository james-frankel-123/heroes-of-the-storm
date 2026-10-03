#!/usr/bin/env python3
"""
Memory monitor for a remote worker (runs inside WSL; stdlib only). Every
60 s it:
  * appends `date` + `free -g` to /mnt/c/hots_memlog/free.log (Windows side,
    so it survives a WSL hang or restart);
  * records each running hotsjob job's memory (PSS and anonymous memory
    summed over its process group, and the largest per-process VmData, which is what `ulimit -d`
    limits) in ~/hots/memmon_state.json (current + peak) and one line per job
    in /mnt/c/hots_memlog/jobs.log;
  * if MemAvailable < --min-free-gb (default 40), pauses the most recently
    started running job (hotsjob pause: MCTS jobs checkpoint, plain jobs are
    stopped and redo their current step on resume), logs an ALERT line in
    /mnt/c/hots_memlog/alerts.log and ~/hots/logs/memmon_alerts.log, then
    waits --cooldown s before it may pause another.
Jobs whose name starts with "memmon" are never paused.

Usage: python remote_workers/memmon.py [--min-free-gb 40] [--interval 60]
  (launch as a hotsjob plain job named memmon)
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HOTS = os.path.expanduser("~/hots")
JOBS = os.path.join(HOTS, "jobs")
WIN = "/mnt/c/hots_memlog"
HERE = os.path.dirname(os.path.abspath(__file__))
HOTSJOB = os.path.join(HERE, "hotsjob.py")


def meminfo():
    out = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        out[k] = int(v.split()[0]) / (1 << 20)   # GB
    return out


def proc_mem(pid):
    """(pss_gb, anon_gb, vmdata_gb) for a pid, or None when it is gone.
    PSS includes file-backed pages (memmaps, reclaimable); anon does not."""
    try:
        pss = anon = 0.0
        for line in open(f"/proc/{pid}/smaps_rollup"):
            if line.startswith("Pss:"):
                pss = int(line.split()[1]) / (1 << 20)
            elif line.startswith("Anonymous:"):
                anon = int(line.split()[1]) / (1 << 20)
        vd = 0.0
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmData:"):
                vd = int(line.split()[1]) / (1 << 20)
                break
        return pss, anon, vd
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def pgid_members():
    groups = {}
    for d in glob.glob("/proc/[0-9]*"):
        try:
            stat = open(f"{d}/stat").read()
            pgrp = int(stat.rsplit(")", 1)[1].split()[2])
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            continue
        groups.setdefault(pgrp, []).append(int(d.rsplit("/", 1)[1]))
    return groups


def running_jobs():
    jobs = []
    for p in glob.glob(os.path.join(JOBS, "*.json")):
        if os.path.basename(p).startswith("_"):
            continue
        try:
            m = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        if m.get("status") == "running" and m.get("pgid"):
            jobs.append(m)
    return jobs


def append(path, text):
    try:
        with open(path, "a") as f:
            f.write(text)
    except OSError as e:
        print(f"cannot write {path}: {e}", flush=True)


def alert(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} ALERT {msg}\n"
    print(line, end="", flush=True)
    append(os.path.join(WIN, "alerts.log"), line)
    append(os.path.join(HOTS, "logs", "memmon_alerts.log"), line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--cooldown", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(WIN, exist_ok=True)
    state_path = os.path.join(HOTS, "memmon_state.json")   # not in JOBS: hotsjob reads every *.json there
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"peak": {}}
    last_pause = 0.0
    while True:
        t = time.strftime("%Y-%m-%dT%H:%M:%S")
        free = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout
        append(os.path.join(WIN, "free.log"), f"# {t}\n{free}")
        mi = meminfo()
        avail = mi["MemAvailable"]
        groups = pgid_members()
        jobs = running_jobs()
        cur, lines = {}, []
        for m in jobs:
            pss, anon, vmax = 0.0, 0.0, 0.0
            for pid in groups.get(m["pgid"], []):
                r = proc_mem(pid)
                if r:
                    pss += r[0]
                    anon += r[1]
                    vmax = max(vmax, r[2])
            cur[m["name"]] = {"pss_gb": round(pss, 2), "anon_gb": round(anon, 2),
                              "max_proc_vmdata_gb": round(vmax, 2),
                              "mem_max": m.get("mem_max"), "started": m.get("started")}
            pk = state["peak"].setdefault(m["name"], {"pss_gb": 0.0, "max_proc_vmdata_gb": 0.0})
            pk["pss_gb"] = round(max(pk["pss_gb"], pss), 2)
            pk["anon_gb"] = round(max(pk.get("anon_gb", 0.0), anon), 2)
            pk["max_proc_vmdata_gb"] = round(max(pk["max_proc_vmdata_gb"], vmax), 2)
            lines.append(f"{t} {m['name']} pss={pss:.1f}G anon={anon:.1f}G vmdata_max={vmax:.1f}G cap={m.get('mem_max')}")
        state.update(time=t, avail_gb=round(avail, 1), total_gb=round(mi["MemTotal"], 1), current=cur)
        json.dump(state, open(state_path + ".tmp", "w"), indent=1)
        os.replace(state_path + ".tmp", state_path)
        append(os.path.join(WIN, "jobs.log"),
               f"{t} avail={avail:.1f}G\n" + "".join(l + "\n" for l in lines))
        if avail < a.min_free_gb and time.time() - last_pause > a.cooldown:
            cands = sorted((m for m in jobs if not m["name"].startswith("memmon")),
                           key=lambda m: m.get("started", ""))
            if cands:
                victim = cands[-1]["name"]
                alert(f"MemAvailable {avail:.1f} GB < {a.min_free_gb} GB: pausing newest job {victim}")
                r = subprocess.run([sys.executable, HOTSJOB, "pause", victim],
                                   capture_output=True, text=True)
                alert(f"pause {victim}: rc={r.returncode} {r.stdout.strip()[-300:]}")
                last_pause = time.time()
            else:
                alert(f"MemAvailable {avail:.1f} GB < {a.min_free_gb} GB and no job to pause")
                last_pause = time.time()
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
