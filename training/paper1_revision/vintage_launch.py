"""
Orchestration for the stale-vintage F drafters (paper1_revision/vintage.py).
Runs on the main box: ssh + rsync only, no compute.

  fetch <vintage>                  copy the vintage's stats, WP models, exclude
                                   list, logs (not the feature npz) and GD from
                                   the 3090 (where they were built) to this box
  push <vintage> <host>            copy the MCTS inputs from this box to a worker
  launch <vintage> <seed> <host>   start the hotsjob MCTS job p1vint_<vintage>_F_s<seed>
  queue <vintage>:<seed>[,...]     waiter: launches the listed runs, in list order
                                   among the ready ones, each on the first of
                                   QUEUE_HOSTS with no running MCTS job. A listed
                                   run whose job exists and is paused is resumed,
                                   on its own host only. A host with a paused
                                   p1vint job that is not in the list is held
                                   (someone paused it). A fine-tune run (<base>_ft) is ready
                                   once p1vint_<base>_F_s<seed> has completed and
                                   site/vint/<base>_ft/models/enriched.json exists
                                   on the 3090. Polls `hotsjob.py state` every
                                   10 min; a failed job or a host unreachable for
                                   over 1 h writes an ALERT line. Touch
                                   site/vint/HOLD to stop launches.
  fetch-run <vintage> <seed>       copy a finished run (policy, checkpoint, meta,
                                   log; not resume_state.pt) to this box
  compute                          write site/vint/compute.json (wall and GPU
                                   hours per job, from the runner logs)

Usage (from training/): python3 paper1_revision/vintage_launch.py <cmd> ...
"""
import os
import re
import sys
import json
import time
import datetime
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(TRAINING_DIR)
RW = os.path.join(TRAINING_DIR, "remote_workers")
sys.path.insert(0, RW)
sys.path.insert(0, TRAINING_DIR)
import hosts  # noqa: E402
from paper1_revision import vintage  # noqa: E402

BUILD_HOST = "max-windows-3090"
HOSTS = ["max-windows-3090", "windows-5090-wsl", "3080-gaming-desktop"]
# MCTS training of this lane runs on these GPUs only (the 5090 is kept for
# tournament searches, coordinator 2026-10-19)
QUEUE_HOSTS = ["max-windows-3090", "3080-gaming-desktop"]
REMOTE = "/home/max/hots/repo/"
# per-process data limit (ulimit -d) of an MCTS job; the measured VmData peak of
# an MCTS run is ~7 GB. The 5090 has 35 GB in all (<= 12 GB per process).
MEM_MAX = {"max-windows-3090": "16G", "3080-gaming-desktop": "16G", "windows-5090-wsl": "12G"}
VINT = os.path.join(HERE, "site", "vint")
LOG = os.path.join(VINT, "vintage_launch.log")
ALERT = os.path.join(VINT, "ALERT_vintage_launch")
HOLD = os.path.join(VINT, "HOLD")


def vdir(v, *p):
    return os.path.join("training", "paper1_revision", "site", "vint", v, *p)


def log(msg):
    os.makedirs(VINT, exist_ok=True)
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")


def alert(msg):
    log(f"ALERT {msg}")
    open(ALERT, "a").write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def rsync(args, **kw):
    return subprocess.run(["nice", "-n", "19", "rsync", "-rlt"] + args, cwd=REPO, **kw)


def rsh(host, script, timeout=180):
    """stdout of a bash script on host, or None if it does not answer."""
    try:
        r = subprocess.run(["ssh", host, hosts.shell(host)], capture_output=True, text=True,
                           timeout=timeout, input="source ~/hots/env.sh >/dev/null 2>&1\n" + script
                           + "\necho __ok\n")
    except subprocess.TimeoutExpired:
        return None
    return r.stdout.replace("__ok\n", "") if "__ok" in r.stdout else None


def is_ft(v):
    return vintage.is_ft(v)


def base_of(v):
    return vintage.spec(v)["base"] if is_ft(v) else v


def fetch(v):
    src = f"{BUILD_HOST}:{REMOTE}"
    rp = f"--rsync-path={hosts.rsync_path(BUILD_HOST)}"
    dirs = ("models/", "logs/") if is_ft(v) else ("cache/stats/", "models/", "logs/")
    for d in dirs:
        os.makedirs(os.path.join(REPO, vdir(v, d)), exist_ok=True)
        rsync([rp, src + vdir(v, d), vdir(v, d)], check=True)
    if not is_ft(v):
        for f in ("cache/mcts_pretrain_exclude.json", "cache/split_meta.json"):
            rsync([rp, src + vdir(v, f), vdir(v, f)], check=True)
        gdd = os.path.dirname(os.path.join("training", vintage.spec(v)["gd"])) + "/"
        os.makedirs(os.path.join(REPO, gdd), exist_ok=True)
        rsync([rp, src + gdd, gdd], check=True)
    log(f"fetched {v} from {BUILD_HOST}")


def inputs(v, seed=0):
    meta = json.load(open(os.path.join(REPO, vdir(v, "models", "enriched.json"))))
    files = [vdir(v, "models", "enriched.json"), vdir(v, "models", f"enriched_s{meta['selected_seed']}.pt")]
    if is_ft(v):
        site = os.path.join("training", "paper1_revision", "site")
        files += [os.path.join(site, "cache", "stats", "deploy.json"),
                  os.path.join(site, "cache", "stats", "deploy_compositions.json"),
                  os.path.join(site, "cache", "mcts_pretrain_exclude.json"),
                  os.path.join("training", "rerun2026", "ns", "p1site", "models", "generic_draft_0.pt"),
                  vdir(base_of(v), "mcts_runs", f"F_oof_s{seed}", "draft_policy.pt")]
    else:
        files += [vdir(v, "cache", "stats", "deploy.json"), vdir(v, "cache", "stats", "deploy_compositions.json"),
                  vdir(v, "cache", "mcts_pretrain_exclude.json"), os.path.join("training", vintage.spec(v)["gd"])]
    for f in files:
        if not os.path.exists(os.path.join(REPO, f)):
            raise SystemExit(f"missing input {f}")
    return files


def push(v, host, seed=0):
    lst = "\n".join(inputs(v, seed)) + "\n"
    rsync([f"--rsync-path={hosts.rsync_path(host)}", "--files-from=-", "./", f"{host}:{REMOTE}"],
          input=lst, text=True, check=True)
    log(f"pushed {v} inputs to {host}")


def job_name(v, seed):
    return f"p1vint_{v}_F_s{seed}"


def launch(v, seed, host):
    run = f"F_oof_s{seed}"
    env = dict(os.environ, RUN_NAME=job_name(v, seed),
               HOTSJOB_FLAGS=f"--save-dir {vdir(v, 'mcts_runs', run)[len('training/'):]} "
                             f"--progress-log {vdir(v, 'logs', f'mcts_{run}.log')[len('training/'):]} "
                             f"--mem-max {MEM_MAX[host]}")
    r = subprocess.run([os.path.join(RW, "run_remote.sh"), host, "env", "P1_TIERS=site", f"P1_VINTAGE={v}",
                        "python", "paper1_revision/train_mcts.py", "F", str(seed), "--gpu", "0"],
                       capture_output=True, text=True, cwd=TRAINING_DIR, env=env)
    log(f"launch {job_name(v, seed)} on {host}: rc={r.returncode} {(r.stdout + r.stderr).strip()[-300:]}")
    return r.returncode == 0


def jobs(host):
    """{name: (status, kind)} of every manifest on host (hotsjob status refreshes
    dead process groups to failed), or None if the host does not answer."""
    out = rsh(host, "python3 ~/hots/repo/training/remote_workers/hotsjob.py status", timeout=240)
    if out is None:
        return None
    js = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 3 and p[2] in ("mcts", "plain") and not line.startswith(" "):
            js[p[0]] = (p[1], p[2])
    return js


def fetch_run(v, seed, host):
    run = f"F_oof_s{seed}"
    rp = f"--rsync-path={hosts.rsync_path(host)}"
    os.makedirs(os.path.join(REPO, vdir(v, "mcts_runs", run)), exist_ok=True)
    os.makedirs(os.path.join(REPO, vdir(v, "logs")), exist_ok=True)
    rsync([rp, "--exclude", "resume_state.pt", "--exclude", ".run.lock",
           f"{host}:{REMOTE}{vdir(v, 'mcts_runs', run)}/", vdir(v, "mcts_runs", run) + "/"], check=True)
    rsync([rp, f"{host}:{REMOTE}{vdir(v, 'logs', f'mcts_{run}.log')}", vdir(v, "logs") + "/"], check=True)
    log(f"fetched run {v} {run} from {host}")


def where(name, snap):
    for h, js in snap.items():
        if js and name in js:
            return h, js[name][0]
    return None, "none"


def queue(items):
    items = [(it.split(":")[0], int(it.split(":")[1])) for it in items.split(",")]
    for v, _ in items:
        vintage.spec(v)
    down_since, alerted, fetched = {}, set(), set()
    log(f"queue: {items}")
    while True:
        snap = {h: jobs(h) for h in HOSTS}
        for h, js in snap.items():
            if js is None:
                t0 = down_since.setdefault(h, time.time())
                if time.time() - t0 > 3600:
                    alert(f"{h} unreachable for {(time.time() - t0) / 3600:.1f} h")
                    down_since[h] = time.time()
            else:
                down_since.pop(h, None)
        # failures and finished runs of this lane
        for h, js in snap.items():
            for n, (st, kind) in (js or {}).items():
                if n.startswith("p1vint_") and st == "failed" and n not in alerted:
                    alert(f"{n} on {h} failed")
                    alerted.add(n)
                m = re.match(r"p1vint_(\w+?)_F_s(\d+)$", n)
                if m and st == "completed" and n not in fetched:
                    try:
                        fetch_run(m.group(1), int(m.group(2)), h)
                        fetched.add(n)
                    except subprocess.CalledProcessError as e:
                        log(f"fetch {n} failed: {e}")
        pending = [(v, s) for v, s in items if where(job_name(v, s), snap)[1] in ("none", "paused")]
        if not pending:
            log("queue done: every run launched")
            compute()
            return 0
        if os.path.exists(HOLD):
            time.sleep(600)
            continue
        if any(js is None for js in snap.values()):
            time.sleep(600)       # a host whose jobs are unknown: launch nothing anywhere
            continue
        mine = {job_name(v, s) for v, s in items}
        free = [h for h in QUEUE_HOSTS if not any(
            kind == "mcts" and (st in ("running", "pausing", "starting")
                                or (st == "paused" and n.startswith("p1vint_") and n not in mine))
            for n, (st, kind) in snap[h].items())]
        for v, s in pending:
            if not free:
                break
            jh, jst = where(job_name(v, s), snap)
            if jst == "paused":
                if jh not in free:
                    continue
                free.remove(jh)
                out = rsh(jh, f"python3 ~/hots/repo/training/remote_workers/hotsjob.py resume {job_name(v, s)}")
                log(f"resume {job_name(v, s)} on {jh}: {(out or 'no answer').strip()[-300:]}")
                compute()
                continue
            if is_ft(v):
                bh, bst = where(job_name(base_of(v), s), snap)
                if bst != "completed" or job_name(base_of(v), s) not in fetched:
                    continue
                if rsh(BUILD_HOST, f"test -f {REMOTE}{vdir(v, 'models', 'enriched.json')} && echo yes") != "yes\n":
                    continue
                fetch(v)
            host = free.pop(0)
            try:
                push(v, host, s)
            except (subprocess.CalledProcessError, SystemExit) as e:
                alert(f"push {v} to {host} failed: {e}")
                continue
            if not launch(v, s, host):
                alert(f"launch of {job_name(v, s)} on {host} failed")
            compute()
        time.sleep(600)


def _intervals(text):
    """[(start, end|None)] from a hotsjob runner log's '# start' / '# exit=' lines."""
    out, cur = [], None
    for line in text.splitlines():
        m = re.match(r"# start (\S+)", line)
        if m:
            if cur:
                out.append((cur, None))
            cur = datetime.datetime.fromisoformat(m.group(1))
            continue
        m = re.match(r"# exit=(\S+) (\S+)", line)
        if m and cur:
            out.append((cur, datetime.datetime.fromisoformat(m.group(2)), m.group(1)))
            cur = None
    if cur:
        out.append((cur, None, None))
    return out


def compute():
    """Wall and GPU hours per job of this lane (and the original p1site F
    seeds), from the runner logs ~/hots/logs/<job>.log on every worker."""
    rows = {}
    script = ("cd ~/hots/logs && for f in p1vint_*.log p1site_F_oof_s?.log; do [ -f \"$f\" ] || continue; "
              "echo \"@@ $f\"; grep -E '^# (start|exit)|enriched s[0-9]+: ep=|^after: |Complete\\. ' \"$f\"; done; "
              "cd ~/hots/repo/training/paper1_revision/site && for f in vint/*/logs/mcts_*.log logs/mcts_F_oof_s?.log; do "
              "[ -f \"$f\" ] || continue; echo \"@@ $f\"; grep -E 'Complete\\. |ep/s' \"$f\" | tail -1; done")
    now = datetime.datetime.now(datetime.timezone.utc)
    for h in HOSTS:
        out = rsh(h, script, timeout=240)
        if out is None:
            continue
        for block in out.split("@@ ")[1:]:
            fname, _, body = block.partition("\n")
            fname = fname.strip()
            if fname.endswith(".log") and "/" not in fname:
                name = fname[:-4]
                iv = _intervals(body)
                if not iv:
                    continue
                wall = sum(((e or now) - s).total_seconds() for s, e, *_ in iv) / 3600
                r = {"host": h, "wall_h": round(wall, 3), "running": iv[-1][1] is None,
                     "segments": [[s.isoformat(), e.isoformat() if e else None] for s, e, *_ in iv]}
                if "_F_s" in name or name.startswith("p1site_F_oof") or "gd_" in name:
                    r["gpu_h"] = r["wall_h"]
                    r["kind"] = "mcts" if "_F_" in name else "gd (GPU + 4 CPU procs for samples)"
                elif "cache" in name:
                    wp = [int(x) for x in re.findall(r"enriched s\d+: ep=.*\((\d+)s\)", body)]
                    r["gpu_h"] = round((wp[-1] if wp else 0) / 3600, 3)
                    r["cpu_wall_h"] = round(wall - r["gpu_h"], 3)
                    r["kind"] = "stats + features (CPU, 8 worker procs) + WP 3 seeds (GPU)"
                elif "ftwp" in name:
                    r["gpu_h"] = r["wall_h"]
                    r["kind"] = "WP fine-tune (GPU)"
                rows.setdefault(name, r)
            else:
                m = re.search(r"Complete\. (\d+) episodes in ([\d.]+)h \(([\d.]+) ep/s\)", body)
                if m:
                    rows.setdefault("selfplay:" + fname, {"host": h, "episodes": int(m.group(1)),
                                                           "selfplay_h": float(m.group(2)),
                                                           "ep_per_s": float(m.group(3))})
    rows["_notes"] = {
        "wall_h": "sum of runner start->exit segments (pauses excluded); gpu_h = wall_h for GPU jobs "
                  "(one GPU each). This lane runs one MCTS job per GPU; the original p1site F s1 and s2 shared "
                  "the 3090 with each other (4.1 ep/s each vs 10.2 ep/s alone on the 3080)",
        "p1site_builds": "original site rebuild (site/logs/site_rebuild.log, local box): stats 7.0 min "
                         "CPU, features 9.7 min CPU (6 procs), wp_table1 37.3 min GPU for 11 Table-I specs; "
                         "enriched alone 241 s (wp_table1.log)",
        "s4m_gd": "drift_v2/models/gd_cutoff pre-existing (drift lane), not counted here",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    json.dump(rows, open(os.path.join(VINT, "compute.json"), "w"), indent=1)
    log(f"compute.json: {len(rows) - 1} entries")


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "fetch":
        fetch(args[0])
    elif cmd == "push":
        push(args[0], args[1], int(args[2]) if len(args) > 2 else 0)
    elif cmd == "launch":
        sys.exit(0 if launch(args[0], int(args[1]), args[2]) else 1)
    elif cmd == "queue":
        sys.exit(queue(args[0]))
    elif cmd == "fetch-run":
        h = args[2] if len(args) > 2 else None
        if h is None:
            snap = {x: jobs(x) for x in HOSTS}
            h = where(job_name(args[0], int(args[1])), snap)[0]
        fetch_run(args[0], int(args[1]), h)
    elif cmd == "compute":
        compute()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
