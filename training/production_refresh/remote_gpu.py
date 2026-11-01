"""
Run the refresh's GPU phases on the remote workers (2026-11-01: the main box's
GPUs 0-1 are tennis's; the 5090 and 3080 are shared with cliff, so a run needs
15 minutes' notice through the coordinator).

Enabled by REFRESH_GPU_HOSTS="windows-5090-wsl,3080-gaming-desktop" (refresh.py
dispatches here; the DB-dependent and CPU phases stay on the main box):

  train   wp, lowdata, partial, gd (and refit, after the local gates): one job on REFRESH_TRAIN_HOST (default the
          last host in REFRESH_GPU_HOSTS), `refresh.py --resume <phases>` with
          HOTS_CORPUS_PATH pointing at the shipped corpus dump
  kparity on every host that trains MCTS: build_h91.py --if-stale for that
          host's GPU (HOTS_SOURCE_COMMIT carries the provenance) and the gated
          kernel parity check
  mcts    seed s on host s % len(hosts), one job per seed (`refresh.py
          mcts_run --seed s`, pause-robust: the worker resumes), then the local
          phase mcts collects them (REFRESH_MCTS_EXTERNAL=1: DONE marker +
          draft_policy.pt)

Data: the run directory (stats, compositions, split, feature caches) and the
dump's corpus.pkl.gz are rsynced to the same relative path on the worker
(~/hots/repo/training/production_refresh/<date>); results come back the same
way (models, mcts_s*/, logs), and each remote phase's refresh_meta.json entries
are merged into the local one. Jobs run under the existing tooling
(personalization/remote/p3_job.sh -> run_remote.sh -> hotsjob), nice 19,
pinned cores, a per-process memory cap.

Per host defaults (REFRESH_HOST_<NAME>_CORES / _MEM override):
  windows-5090-wsl     cores 0-15, 11G per process (host RAM is small)
  3080-gaming-desktop  cores 16-19, 30G per process (never during its nightly
                       replay parse 22:40-23:30, cores 4-15)

REFRESH_REMOTE_DRYRUN=1 prints every command instead of running it.
"""
import json
import os
import shlex
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(BASE)
REPO_DIR = os.path.dirname(TRAINING_DIR)
REMOTE_TRAINING = "/home/max/hots/repo/training"
P3_JOB = os.path.join(TRAINING_DIR, "personalization", "remote", "p3_job.sh")
REMOTE_SH = os.path.join(TRAINING_DIR, "remote_workers", "_remote.sh")
TRAIN_PHASES = ("wp", "lowdata", "partial", "gd", "refit")
DEFAULTS = {"windows-5090-wsl": ("0-15", "11G", "4"), "3080-gaming-desktop": ("16-19", "30G", "4")}
DRY = os.environ.get("REFRESH_REMOTE_DRYRUN") == "1"
# rsync excludes: never ship the GD cache or partial caches back and forth
EXCLUDES = ["gd_cache/", "*.tmp", "__pycache__/"]


def hosts():
    return [h for h in os.environ.get("REFRESH_GPU_HOSTS", "").split(",") if h]


def enabled():
    return bool(hosts())


def train_host():
    return os.environ.get("REFRESH_TRAIN_HOST") or hosts()[-1]


def _cfg(host):
    key = host.upper().replace("-", "_")
    c, m, t = DEFAULTS.get(host, ("0-3", "16G", "4"))
    return (os.environ.get(f"REFRESH_HOST_{key}_CORES", c), os.environ.get(f"REFRESH_HOST_{key}_MEM", m),
            os.environ.get(f"REFRESH_HOST_{key}_THREADS", t))


def _sh(cmd, check=True, capture=False):
    """Run a bash command with the remote helpers sourced."""
    full = f"source {shlex.quote(REMOTE_SH)}; {cmd}"
    if DRY:
        print("DRYRUN:", cmd, flush=True)
        return ""
    r = subprocess.run(["bash", "-c", full], cwd=REPO_DIR, check=check, text=True,
                       capture_output=capture)
    return r.stdout if capture else ""


def _log(msg):
    print(f"[remote {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def remote_run_dir(run_dir):
    return os.path.join(REMOTE_TRAINING, os.path.relpath(run_dir, TRAINING_DIR))


def sync_code(host):
    _sh(f"nice -n 19 training/remote_workers/sync.sh {shlex.quote(host)} code | tail -1")


def push(host, run_dir, corpus=None):
    """Ship the run directory (and the corpus dump) to the worker."""
    rd = remote_run_dir(run_dir)
    ex = " ".join(f"--exclude {shlex.quote(e)}" for e in EXCLUDES + ["corpus.pkl.gz"])
    _sh(f"ssh {host} \"$(remote_shell {host})\" <<< 'mkdir -p {rd}'")
    _sh(f"nice -n 19 rsync -rt --rsync-path=\"$(remote_rsync {host})\" {ex} "
        f"{shlex.quote(run_dir)}/ {host}:{shlex.quote(rd)}/")
    if corpus:
        _sh(f"nice -n 19 rsync -t --rsync-path=\"$(remote_rsync {host})\" "
            f"{shlex.quote(corpus)} {host}:{shlex.quote(rd)}/corpus.pkl.gz")


def pull(host, run_dir, subdirs=None):
    """Fetch the worker's results into the local run directory; its
    refresh_meta.json lands in remote_meta_<host>.json for merging."""
    rd = remote_run_dir(run_dir)
    ex = " ".join(f"--exclude {shlex.quote(e)}" for e in EXCLUDES + ["corpus.pkl.gz", "refresh_meta.json"])
    src = ([f"{host}:{shlex.quote(os.path.join(rd, s))}" for s in subdirs] if subdirs is not None
           else [f"{host}:{shlex.quote(rd)}/"])
    for s in src:
        dst = run_dir if subdirs is not None else run_dir + "/"
        _sh(f"nice -n 19 rsync -rt --rsync-path=\"$(remote_rsync {host})\" {ex} {s} {shlex.quote(dst)}")
    _sh(f"nice -n 19 rsync -t --rsync-path=\"$(remote_rsync {host})\" "
        f"{host}:{shlex.quote(os.path.join(rd, 'refresh_meta.json'))} "
        f"{shlex.quote(os.path.join(run_dir, f'remote_meta_{host}.json'))}")
    p = os.path.join(run_dir, f"remote_meta_{host}.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def launch(host, name, argv, gpu=True):
    """One job on the worker (p3_job.sh: hotsjob, nice 19, taskset, memory cap)."""
    cores, mem, threads = _cfg(host)
    for a in argv:
        if " " in a:
            raise ValueError(f"argument with spaces cannot go through p3_job.sh: {a!r}")
    env = f"P3_CORES={cores} P3_MEM={mem} P3_THREADS={threads}"
    _sh(f"{env} {shlex.quote(P3_JOB)} {host} {name} " + " ".join(shlex.quote(a) for a in argv))
    _log(f"launched {name} on {host} (cores {cores}, {mem})")


def wait(host, name, unknown_max=3600, poll=60):
    """Poll the job until it ends; an empty or failed status read is unknown, not an end."""
    if DRY:
        return True
    since = None
    while True:
        st = _sh(f"hotsjob {host} state {name} 2>/dev/null | tail -1", check=False, capture=True).strip()
        if st == "completed":
            _log(f"{name} on {host} completed")
            return True
        if st in ("failed", "died", "paused"):
            _log(f"{name} on {host}: {st}")
            return False
        if st in ("running", "queued", "starting"):
            since = None
        else:
            since = since or time.time()
            if time.time() - since > unknown_max:
                _log(f"{name} on {host}: status unknown for {unknown_max}s; giving up")
                return False
        time.sleep(poll)


def _env_args(run_date, run_dir, extra=None):
    commit = subprocess.run(["git", "log", "-1", "--format=%H", "--", "training/cuda_mcts"], cwd=REPO_DIR,
                            capture_output=True, text=True).stdout.strip() or "unknown"
    e = {"REFRESH_DATE": run_date, "HOTS_CORPUS_PATH": os.path.join(remote_run_dir(run_dir), "corpus.pkl.gz"),
         "HOTS_SOURCE_COMMIT": commit, "REFRESH_REFIT": os.environ.get("REFRESH_REFIT", "0"),
         "HOTS_HERO_SET": os.environ.get("HOTS_HERO_SET", "v2")}
    for k in ("REFRESH_MCTS_EPISODES", "REFRESH_MCTS_SEARCH", "REFRESH_TEST_DAYS"):
        if os.environ.get(k):
            e[k] = os.environ[k]
    e.update(extra or {})
    return ["env"] + [f"{k}={v}" for k, v in e.items()]


def run_train(phases, run_date, run_dir, corpus, meta_update):
    """wp / lowdata / partial / gd as one job on the train host."""
    h = train_host()
    sync_code(h)
    push(h, run_dir, corpus)
    name = f"refresh_{run_date}_train"
    launch(h, name, _env_args(run_date, run_dir) + ["python", "-u", "production_refresh/refresh.py", "--resume"] + list(phases))
    if not wait(h, name):
        sys.exit(f"remote training phases {phases} failed on {h} (log ~/hots/logs/{name}.log)")
    m = pull(h, run_dir)
    meta_update(**{k: v for k, v in m.items() if k != "phases_done"},
                remote_train={"host": h, "phases": list(phases)})
    return m.get("phases_done", [])


def run_kparity_and_mcts(seeds, run_date, run_dir, corpus, meta_update):
    """kparity on each MCTS host (its own kernel build), then one job per seed."""
    hs = hosts()
    by_host = {h: [s for s in seeds if hs[s % len(hs)] == h] for h in hs}
    used = [h for h in hs if by_host[h]]
    res = {}
    for h in used:
        sync_code(h)
        push(h, run_dir, corpus)
    # kernel parity gate per host (build for that GPU, then the check)
    for h in used:
        name = f"refresh_{run_date}_kparity_{h.split('-')[0]}"
        launch(h, name, _env_args(run_date, run_dir) + ["python", "-u", "production_refresh/refresh.py", "kparity"])
        if not wait(h, name):
            sys.exit(f"kernel parity failed or did not finish on {h}: not training MCTS")
        m = pull(h, run_dir, subdirs=[])
        res[h] = m.get("kernel_parity") or {}
        meta_update(**{f"kernel_parity_{h}": m.get("kernel_parity"), f"kernel_{h}": m.get("kernel")})
        if not res[h].get("pass") and not DRY:
            sys.exit(f"KERNEL PARITY FAIL on {h}: {res[h]}")
    # the gate reads kernel_parity: the worst host
    if res and not DRY:
        worst = max(res.values(), key=lambda r: r.get("max_abs_diff") or 0.0)
        meta_update(kernel_parity=dict(worst, pass_all_hosts=all(r.get("pass") for r in res.values()),
                                       hosts=sorted(res)))
    jobs = []
    for h in used:
        for s in by_host[h]:
            name = f"refresh_{run_date}_mcts_s{s}"
            launch(h, name, _env_args(run_date, run_dir) + ["python", "-u", "production_refresh/refresh.py",
                                                            "mcts_run", "--seed", str(s)])
            jobs.append((h, s, name))
    ok = True
    for h, s, name in jobs:
        ok &= wait(h, name, poll=300)
        pull(h, run_dir, subdirs=[f"mcts_s{s}"])
    meta_update(remote_mcts={"hosts": {str(s): h for h, s, _ in jobs}})
    return ok
