"""
Remote-only scheduler for the paper-1 MCTS retrains (site tiers, chance-node
kernel). Runs on the main box as a light polling loop (ssh + rsync only, no
compute) and launches pause-aware hotsjob MCTS jobs on the workers.

Runs, in priority order (5 seeds unless noted; coordinator option B):
  B_oof F_oof J_oof        200/400/800 sims, leak-free enriched leaf (core set)
  K_oof                    400 sims, naive leaf (no-features agent)
  E_oof                    200 sims, 1M episodes (3 seeds)
  B_leak F_leak J_leak     in-sample-statistics control leaf (3 seeds)
Slots per host come from paper1_revision/site/mcts_slots.json (edited by hand
to yield to other lanes; read every cycle), e.g. {"max-windows-3090": 2,
"3080-gaming-desktop": 1}. A host gets its inputs pushed once (site WP models
and stats, exclude list, GD pool) before its first launch. A run is never
launched twice: a manifest of that name on any host counts as taken.

Memory (2026-10-03): at most 2 runs per host, launched one at a time: a run
starts only when no other run on that host is in value pretraining; each
run has a per-process data cap (site/mem_caps.json "mcts"). An unreachable
host (ssh or WSL not answering) stops all launches and writes an alert to
site/logs/ALERT_site_mcts_remote.

Handoff: once every run has been launched, site/logs/HANDOFF_<host> says how
many MCTS slots each host has free for the drift lane (updated as runs end).

Usage (from training/): setsid nohup python3 paper1_revision/site_mcts_remote.py &
Log: paper1_revision/site/logs/site_mcts_remote.log
"""
import os
import re
import json
import time
import subprocess
import sys as _rw_sys, os as _rw_os
_rw_sys.path.insert(0, _rw_os.path.join(_rw_os.path.dirname(_rw_os.path.abspath(__file__)), *(['..'] * (2 if 'drift_rebuild' in __file__ or '/remote/' in __file__ else 1)), 'remote_workers'))
import hosts as _rw_hosts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(TRAINING_DIR)
SITE = os.path.join(HERE, "site")
NS = os.path.join(TRAINING_DIR, "rerun2026", "ns", "p1site", "models")
SLOTS = os.path.join(SITE, "mcts_slots.json")
LOG = os.path.join(SITE, "logs", "site_mcts_remote.log")
RW = os.path.join(TRAINING_DIR, "remote_workers")
RUNS = ([("B", "oof", s) for s in range(5)] + [("F", "oof", s) for s in range(5)]
        + [("J", "oof", s) for s in range(5)] + [("K", "oof", s) for s in range(5)]
        + [("E", "oof", s) for s in range(3)] + [("B", "leak", s) for s in range(3)]
        + [("F", "leak", s) for s in range(3)] + [("J", "leak", s) for s in range(3)])


MAX_SLOTS = {"max-windows-3090": 2, "3080-gaming-desktop": 1, "windows-5090-wsl": 2}


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    open(LOG, "a").write(line + "\n")


def ready():
    return (all(os.path.exists(os.path.join(NS, "meta", f"gd_{i}.json")) for i in range(5))
            and os.path.exists(os.path.join(SITE, "models", "enriched.json"))
            and os.path.exists(os.path.join(SITE, "models", "naive.json"))
            and os.path.exists(os.path.join(SITE, "models", "enriched_leak.json")))


ALERT = os.path.join(SITE, "logs", "ALERT_site_mcts_remote")


def alert(msg):
    """Log and leave a flag file the session watcher picks up."""
    log(f"ALERT {msg}")
    open(ALERT, "a").write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def status(host):
    """{run: status} of this lane's MCTS jobs on host, or None when the host
    (or its WSL) does not answer: then nothing is launched or assumed."""
    try:
        r = subprocess.run([os.path.join(RW, "jobs_remote.sh"), host], capture_output=True, text=True,
                           cwd=TRAINING_DIR, timeout=240)
    except subprocess.TimeoutExpired:
        return None
    text = (r.stdout + r.stderr).replace("\0", "")
    # a live WSL lists at least one manifest line (hotsjob prints "no jobs" otherwise)
    if r.returncode != 0 or "HCS_E" in text or "Wsl/" in text or not re.search(r"^\S+\s+(running|completed|failed|paused|died|pausing)\b", r.stdout, re.M):
        return None
    out = {}
    for line in r.stdout.splitlines():
        m = re.match(r"(p1site_[A-Z]_(oof|leak)_s\d)\s+(\w+)", line.strip())
        if m:
            out[m.group(1)] = m.group(3)
    return out


def in_pretraining(host, runs):
    """Runs on host whose value pretraining has not finished (the stagger rule:
    never two value pretrainings at once on a host). Unknown -> treated as in."""
    if not runs:
        return []
    cmd = "; ".join(f"grep -qE 'pre-training complete|^Episode ' ~/hots/repo/training/paper1_revision/site/logs/mcts_{r[len('p1site_'):]}.log 2>/dev/null || echo {r}"
                    for r in runs)
    try:
        r = subprocess.run(["ssh", host, _rw_hosts.shell(host)], input=cmd + "\necho __ok\n", capture_output=True,
                           text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return list(runs)
    if "__ok" not in r.stdout:
        return list(runs)
    return [l.strip() for l in r.stdout.splitlines() if l.strip().startswith("p1site_")]


def drift_mcts_running():
    r = subprocess.run([os.path.join(RW, "jobs_remote.sh"), "3080-gaming-desktop"], capture_output=True,
                       text=True, cwd=TRAINING_DIR)
    return any(re.search(r"\bmcts\b", l) and "running" in l and not l.strip().startswith("p1site_")
               for l in r.stdout.splitlines())


def push(host):
    files = []
    for d, pat in ((os.path.join(SITE, "models"), r"^(enriched|naive|enriched_leak)(_s\d+)?\.(pt|json)$"),
                   (os.path.join(SITE, "cache", "stats"), r"^deploy"),):
        files += [os.path.join(d, f) for f in os.listdir(d) if re.match(pat, f)]
    files += [os.path.join(SITE, "cache", "paper_test_ids.json"),
              os.path.join(SITE, "cache", "mcts_pretrain_exclude.json"),
              os.path.join(TRAINING_DIR, "snapshots", "replay_snapshot_2026-05-22_1956753_p1site_lite.json")]
    files += [os.path.join(NS, f"generic_draft_{i}.pt") for i in range(5)]
    lst = "\n".join(os.path.relpath(f, REPO) for f in files) + "\n"
    r = subprocess.run(["nice", "-n", "19", "rsync", "-rlt", f"--rsync-path={_rw_hosts.rsync_path(host)}", "-z",
                        "--files-from=-", "./", f"{host}:/home/max/hots/repo/"],
                       input=lst, text=True, cwd=REPO, capture_output=True)
    return r.returncode == 0


def mem_cap(cfg):
    """Per-process data limit for an MCTS run (site/mem_caps.json "mcts", sized
    from measured peaks; default 32G)."""
    p = os.path.join(SITE, "mem_caps.json")
    caps = json.load(open(p)) if os.path.exists(p) else {}
    return caps.get("mcts", {}).get(cfg) or caps.get("mcts", {}).get("default") or "32G"


def launch(host, cfg, tag, seed):
    run = f"{cfg}_{tag}_s{seed}"
    wp = " --wp enriched_leak" if tag == "leak" else ""
    env = dict(os.environ, RUN_NAME=f"p1site_{run}",
               HOTSJOB_FLAGS=f"--save-dir paper1_revision/site/mcts_runs/{run} "
                             f"--progress-log paper1_revision/site/logs/mcts_{run}.log "
                             f"--mem-max {mem_cap(cfg)}")
    r = subprocess.run([os.path.join(RW, "run_remote.sh"), host, "env", "P1_TIERS=site", "python",
                        "paper1_revision/train_mcts.py", cfg, str(seed), "--gpu", "0"] + wp.split(),
                       capture_output=True, text=True, cwd=TRAINING_DIR, env=env)
    log(f"launch {run} on {host}: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]}")


def main():
    pushed = set()
    while not ready():
        time.sleep(600)
    log("inputs ready")
    # the site exclude list is built here (light: ids only)
    subprocess.run(["nice", "-n", "19", "python3", "-c",
                    "import sys; sys.path.insert(0, '.'); from paper1_revision import train_mcts; "
                    "train_mcts.exclude_path()"], cwd=TRAINING_DIR, env=dict(os.environ, P1_TIERS="site"))
    while True:
        slots = json.load(open(SLOTS)) if os.path.exists(SLOTS) else {"max-windows-3090": 2}
        st = {h: status(h) for h in slots}
        down = [h for h, v in st.items() if v is None]
        if down:
            # an unreachable host's jobs are unknown: launch nothing anywhere
            # (a run there may be alive) and alert
            alert(f"unreachable (ssh/WSL): {', '.join(down)}; no launches this cycle")
            time.sleep(300)
            continue
        taken = {n for s in st.values() for n in s}
        pending = [r for r in RUNS if f"p1site_{r[0]}_{r[1]}_s{r[2]}" not in taken]
        if not pending and all(v in ("done", "finished") for s in st.values() for v in s.values()):
            log("all runs finished")
            return
        if not pending:
            # handoff to the drift lane: a host whose option-B runs have all been
            # launched gets a marker (with its free slots) once per change
            for host, n in slots.items():
                live = sum(1 for v in st[host].values() if v in ("running", "starting"))
                mk = os.path.join(SITE, "logs", f"HANDOFF_{host}")
                msg = f"no option-B runs pending; {live} still running, {max(0, MAX_SLOTS.get(host, 1) - live)} MCTS slots free"
                if not os.path.exists(mk) or open(mk).read().strip() != msg:
                    open(mk, "w").write(msg + "\n")
                    log(f"HANDOFF {host}: {msg}")
        for host, n in slots.items():
            # at most 2 concurrent MCTS runs on the 3090 (2026-10-03); one on the
            # 3080, whose GPU one agent saturates and whose WSL has 47 GB (2026-10-04)
            n = min(n, MAX_SLOTS.get(host, 1))
            live = [r for r, v in st[host].items() if v in ("running", "starting")]
            running = len(live)
            # stagger: no launch while a run on this host is still in value pretraining
            if running < n and pending and in_pretraining(host, live):
                continue
            while running < n and pending:
                if host not in pushed:
                    if not push(host):
                        log(f"push to {host} failed")
                        break
                    pushed.add(host)
                    log(f"pushed inputs to {host}")
                cfg, tag, seed = pending.pop(0)
                launch(host, cfg, tag, seed)
                running += 1
                break        # one launch per host per cycle; the next waits out its pretraining
        time.sleep(300)


if __name__ == "__main__":
    main()
