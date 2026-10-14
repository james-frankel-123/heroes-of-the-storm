"""Remote dispatch (FRAMING 11.10) on top of training/remote_workers.

  submit(spec, host) -> job_id     ships code + the label extract once, launches via hotsjob
  wait(job_id, host) -> result      polls `hotsjob.py state`, then pulls <job>/out back

3090: taskset 24-35 (our cores only), nice 19, --mem-max 56G. 5090: CPU threads only,
--mem-max 11G. Never the 3080. Jobs are named scilearn_<job_id>. Extracts never include
C, F, R or pool rows; they hold only the tranches a job may fit or score.
"""
import base64
import json
import subprocess
import time
import uuid

import numpy as np

from training.scilearn import config as C
from training.scilearn import data as D
from training.scilearn import tranches as T

RW = C.REPO / "training" / "remote_workers"
ROOT = C.REMOTE_ROOT
JOBS_LOCAL = C.PRIVATE / "jobs"
EXTRACT_TRANCHES = ["S", "Dpre", "G", "Dpost"]


def _shell(host):
    return "bash -s" if host == "windows-5090-wsl" else "wsl -e bash -s"


def _rsync_path(host):
    return "rsync" if host == "windows-5090-wsl" else "wsl rsync"


def rsh(host, script, timeout=600):
    r = subprocess.run(["ssh", host, _shell(host)], input=script, text=True, capture_output=True,
                       timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"{host}: {r.stderr[-2000:]}")
    return r.stdout


def rsync_to(host, src, dest, extra=()):
    subprocess.run(["nice", "-n", "19", "rsync", "-rlt", f"--rsync-path={_rsync_path(host)}", "-z",
                    *extra, str(src), f"{host}:{dest}"], check=True, capture_output=True)


def rsync_from(host, src, dest):
    subprocess.run(["nice", "-n", "19", "rsync", "-rlt", f"--rsync-path={_rsync_path(host)}",
                    f"{host}:{src}", str(dest)], check=True, capture_output=True)


def sync_code(host):
    rsh(host, f"mkdir -p {ROOT}/code/training/scilearn {ROOT}/data {ROOT}/jobs")
    rsync_to(host, str(C.CODE) + "/", f"{ROOT}/code/training/scilearn/",
             ["--exclude=sandbox/", "--exclude=kit/", "--exclude=session_kit/", "--include=*.py",
              "--include=*.json", "--include=*/", "--exclude=*"])


def make_extract(label_mode="real", snap_id=None, kit=True):
    """Build (once) the label extract for S, Dpre, G, Dpost. label_mode: real | shuffled:<key>."""
    snap_id = snap_id or D.latest_snapshot()
    tag = label_mode.replace(":", "_")[:40]
    kp = C.CODE / "kit" / "kit_2.57.json"
    ktag = ("_kit" + __import__("hashlib").sha256(kp.read_bytes()).hexdigest()[:8]) if (kit and kp.exists()) else ""
    name = f"{snap_id}_{tag}{ktag}"
    path = C.PRIVATE / "extracts" / f"{name}.npz"
    if path.exists():
        return name
    path.parent.mkdir(parents=True, exist_ok=True)
    g, meta, y, pl = D.load(snap_id, EXTRACT_TRANCHES, labels=True, players=True)
    if label_mode.startswith("shuffled:"):
        from training.scilearn.evaluator import shuffle_labels
        y = shuffle_labels(y, g["game_ts"], g["map"], g["tier"], g["build"], label_mode.split(":", 1)[1], g["replay_id"])
    arrays = {k: g[k] for k in ("replay_id", "game_ts", "first_seen_ts", "build", "map", "tier", "heroes", "tranche")}
    arrays.update(y0=y, player_key=pl["player_key"], party=pl["party"])
    kmeta = dict(meta)
    if kit:
        karr, axes = D.load_kit(meta)
        if karr is not None:
            arrays["kit"] = karr
            kmeta["kit_axes"] = axes
    np.savez(path, **arrays)
    for k in ("snap_dir",):
        kmeta.pop(k, None)
    path.with_suffix(".json").write_text(json.dumps(kmeta))
    return name


def ship_extract(host, name):
    src = C.PRIVATE / "extracts"
    out = rsh(host, f"ls {ROOT}/data/{name}.npz 2>/dev/null || true")
    if name not in out:
        rsync_to(host, src / f"{name}.npz", f"{ROOT}/data/")
    rsync_to(host, src / f"{name}.json", f"{ROOT}/data/")


def hotsjob(host, *args):
    b64 = base64.b64encode(b"\0".join(a.encode() for a in args) + b"\0").decode()
    script = f"set -e\nsource ~/hots/env.sh\necho '{b64}' | base64 -d | xargs -0 python ~/hots/repo/training/remote_workers/hotsjob.py\n"
    return rsh(host, script)


def submit(spec, host="max-windows-3090", label_mode="real", module="training.scilearn.worker", python="python"):
    if host not in C.REMOTE_HOSTS:
        raise ValueError(f"host {host} not allowed for scilearn")
    hc = C.REMOTE_HOSTS[host]
    job_id = spec.get("job_id") or f"{time.strftime('%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
    spec["job_id"] = job_id
    name = make_extract(label_mode)
    spec["data"], spec["meta"] = f"{name}.npz", f"{name}.json"
    ldir = JOBS_LOCAL / job_id
    ldir.mkdir(parents=True, exist_ok=True)
    (ldir / "spec.json").write_text(json.dumps(spec, indent=1))
    (ldir / "host").write_text(host)
    sync_code(host)
    ship_extract(host, name)
    rsh(host, f"mkdir -p {ROOT}/jobs/{job_id}")
    rsync_to(host, ldir / "spec.json", f"{ROOT}/jobs/{job_id}/")
    th = hc["threads"]
    pin = f"taskset -c {hc['cpus']} " if hc["cpus"] else ""
    cmd = (f"cd {ROOT}/code && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS={th} OPENBLAS_NUM_THREADS={th} "
           f"MKL_NUM_THREADS={th} PYTHONPATH={ROOT}/code {pin}nice -n 19 {python} -u -m {module} "
           f"{ROOT}/jobs/{job_id}/spec.json")
    if host == "windows-5090-wsl":
        cmd = cmd.replace("CUDA_VISIBLE_DEVICES= ", "")  # leave the GPU env alone there
    # Idempotent: a job that is already running (another broker thread or process launched
    # it) is attached to, and a completed one with a result is reused, never relaunched.
    st = state(job_id, host)
    if st in ("running", "pausing"):
        (ldir / "attached").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        return job_id
    if st == "completed" and rsh(host, f"test -f {ROOT}/jobs/{job_id}/out/result.json && echo yes || true").strip() == "yes":
        return job_id
    hotsjob(host, "launch", f"scilearn_{job_id}", cmd, "--mem-max", hc["mem_max"])
    return job_id


def state(job_id, host):
    return hotsjob(host, "state", f"scilearn_{job_id}").strip().splitlines()[-1]


def wait(job_id, host=None, poll=30, timeout=6 * 3600):
    host = host or (JOBS_LOCAL / job_id / "host").read_text().strip()
    t0 = time.time()
    while True:
        st = state(job_id, host)
        if st not in ("running", "pausing"):
            break
        if time.time() - t0 > timeout:
            raise TimeoutError(job_id)
        time.sleep(poll)
    ldir = JOBS_LOCAL / job_id
    (ldir / "out").mkdir(parents=True, exist_ok=True)
    rsync_from(host, f"{ROOT}/jobs/{job_id}/out/", str(ldir / "out") + "/")
    rsync_from(host, f"/home/max/hots/logs/scilearn_{job_id}.log", ldir / "job.log")
    if st != "completed" or not (ldir / "out" / "result.json").exists():
        raise RuntimeError(f"job {job_id} ended {st}; see {ldir / 'job.log'}")
    return json.loads((ldir / "out" / "result.json").read_text())
