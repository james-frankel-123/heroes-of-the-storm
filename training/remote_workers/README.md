# Remote GPU workers (WSL)

Two Windows machines run HotS MCTS training inside WSL (Ubuntu 24.04), reached
over Tailscale SSH from the main box. `ssh <host>` lands in Windows `cmd`, so every
script here ships its commands to WSL on stdin (`ssh <host> "wsl -e bash -s"`),
which avoids cmd.exe quoting.

| host | GPU | driver | torch | CUDA toolkit (nvcc) | cores / RAM |
|---|---|---|---|---|---|
| `max-windows-3090` | RTX 3090 24 GB | 560.94 (CUDA <= 12.6) | 2.10.0+cu126 | 12.6.3 | 36 / 196 GB |
| `3080-gaming-desktop` | RTX 3080 10 GB | 576.52 (CUDA <= 12.9) | 2.10.0+cu128 | 12.8.1 | 20 / 47 GB |

The 3090 uses cu126 because its driver predates CUDA 12.8. After a driver update to
570 or newer it can switch to cu128 by re-running `setup_env.sh cu128` after
deleting `~/hots/venv`.

Everything lives in `~/hots/` inside WSL and touches nothing else:

```
~/hots/env.sh        source this: venv + CUDA_HOME + TORCH_CUDA_ARCH_LIST=8.6 + WANDB_MODE=disabled
~/hots/venv/         Python 3.14 (uv-managed, in ~/hots/.uv-python), torch, numpy, psycopg2-binary, onnx, onnxruntime, ...
~/hots/cuda/<ver>/   CUDA toolkit from NVIDIA redistributable tarballs (no root)
~/hots/bin/uv
~/hots/repo/         code + data synced from this box (same layout as the repo)
~/hots/jobs/         job manifests <name>.json and pause sentinels <name>.pause
~/hots/logs/         job logs <name>.log, setup log
```

No `.env` or `DATABASE_URL` is on the workers: MCTS training, the benchmarks and the
tournaments read only pinned files (snapshot, stats, checkpoints).

## Files in this directory

| file | runs on | purpose |
|---|---|---|
| `setup_env.sh <cu126\|cu128>` | worker | one-time env build (uv, Python 3.14, torch, CUDA toolkit, env.sh) |
| `sync.sh <host> [code\|data\|all]` | this box | rsync code (`code.filter`) and/or data (`data_manifest.sh`) to `~/hots/repo` |
| `build_ext.sh` | worker | build `cuda_mcts` and `overfit2026/cuda_ofit` extensions in place |
| `smoke.py [--gpu]` | worker | import stack, extension load, worker setup path, inputs present; `--gpu` adds a device check |
| `hotsjob.py` | worker | job manager: launch / pause / resume / status / finish, with manifests |
| `run_remote.sh <host> <cmd...>` | this box | launch a detached, pause-aware job |
| `pause_remote.sh <host\|all>` | this box | checkpoint and stop every job (about 10 min or less) |
| `resume_remote.sh <host\|all>` | this box | relaunch paused jobs from their latest checkpoints |
| `jobs_remote.sh <host\|all>` | this box | job status, last checkpoint, log paths |
| `train_mcts_worker_resume.patch` | - | the opt-in pause/checkpoint hunks of `train_mcts_worker.py`, against the committed file (HEAD 16382fb, last changed in 90558ec) |

## Launching a run

```bash
# MCTS self-play (paper1_revision wrapper; config B/F/I/J/E, seed)
RUN_NAME=F_oof_s0 training/remote_workers/run_remote.sh max-windows-3090 \
    python paper1_revision/train_mcts.py F 0 --gpu 0

# the bare worker (set MCTS_SAVE_DIR in the command so pause/resume can find the checkpoint)
RUN_NAME=myrun training/remote_workers/run_remote.sh 3080-gaming-desktop \
    MCTS_SAVE_DIR=/home/max/hots/runs/myrun MCTS_NUM_SIMS=200 python train_mcts_worker.py

# benchmark / tournament (plain jobs: pause = stop, resume = re-run, finished outputs are skipped)
RUN_NAME=bench1 training/remote_workers/run_remote.sh max-windows-3090 python paper1_revision/bench_mcts.py ...
```

What the launcher does on the worker:
- sources `~/hots/env.sh`, `cd ~/hots/repo/training`, sets `CUDA_VISIBLE_DEVICES=0` (`--gpu N` in `HOTSJOB_FLAGS` changes it);
- runs the command under `setsid` (its own process group, survives SSH disconnect);
- writes `~/hots/jobs/<name>.json` with the command, resume command, status, process group, last checkpoint and logs;
- for MCTS jobs, exports `MCTS_PAUSE_FILE=~/hots/jobs/<name>.pause` and `MCTS_CKPT_EVERY_SEC=480`.

Logs: `~/hots/logs/<name>.log` has the runner output and exit code. `train_mcts.py`
writes the training log itself to `~/hots/repo/training/paper1_revision/logs/mcts_<cfg>_oof_s<seed>.log`,
and its outputs to `paper1_revision/mcts_runs/<cfg>_oof_s<seed>/`. `jobs_remote.sh` prints both paths.

```bash
ssh max-windows-3090 "wsl -e bash -s" <<< 'tail -5 ~/hots/repo/training/paper1_revision/logs/mcts_F_oof_s0.log'
```

## Memory cap and failure detection

`--mem-max 26G` (in `HOTSJOB_FLAGS`, or `hotsjob.py launch ... --mem-max 26G`) runs the command
under `ulimit -d` (RLIMIT_DATA). This is a per-process limit on heap and private writable
mappings; CUDA's address-space reservations don't count. A blow-up then fails inside that process
(MemoryError, exit 1, status `failed`) instead of the host OOM killer taking WSL down. The cap also
applies on resume. Production jobs on the 3080 (47 GB) use it. systemd scopes (MemoryMax) are not
an option on these workers: once only scoped processes remain, WSL considers the distro idle and
shuts it down, which killed every job on 2026-10-02.

A job whose process group is gone without an exit record is marked `failed` by `status` and by
`hotsjob.py state <name>`, which prints only the status word. Scripts that wait on a job should
poll `state`, not read the manifest: the manifest still says `running` if the runner itself was
killed. That happened on 2026-10-02, when a host-wide OOM left a chain waiting for 5 h.

### Memory monitor (memmon.py)

Run one per worker as a plain job: `RUN_NAME=memmon run_remote.sh <host> python
remote_workers/memmon.py --min-free-gb 40`. Every minute it appends `free -g` to
`/mnt/c/hots_memlog/free.log` (Windows side, readable after a WSL hang:
`ssh <host> "type C:\hots_memlog\free.log"`), records each running job's PSS, anonymous memory
and largest per-process VmData in `/mnt/c/hots_memlog/jobs.log` and `~/hots/memmon_state.json`
(current and peak; size `--mem-max` from the VmData peak), and, when MemAvailable drops below
the threshold, pauses the most recently started job and writes an ALERT line to
`/mnt/c/hots_memlog/alerts.log`. Paused jobs are not resumed automatically. On 2026-10-03 the
3090's WSL hung with vmmem at 134 GB (too many jobs at once) and every job there died.

## Pause and resume

The owner may ask for the GPUs back with about 15 minutes' notice:

```bash
training/remote_workers/pause_remote.sh all     # prints ALL PAUSED, or PAUSE INCOMPLETE and exits 1
training/remote_workers/resume_remote.sh all    # later
```

How a pause works:
- **MCTS jobs.** `pause` creates the job's pause file. At the next batch boundary,
  `train_mcts_worker.py` writes `<save_dir>/resume_state.pt` and exits with code 75. That
  normally takes one or two batches: about 25 s per 128-episode batch at 800 sims, and
  less at 200 sims. The manifest becomes `paused`.
  - A job still in setup (value pretraining, before self-play starts) has nothing to save,
    so it is killed at once and restarts fresh on resume, losing only its setup time
    (a few minutes).
  - If a job has not exited after 540 s, it is killed only if a resume checkpoint exists.
    Otherwise it is left running and the pause reports failure.
- **Plain jobs** (benchmarks, tournaments) get SIGTERM, then SIGKILL after 30 s. On resume
  they re-run, and the scripts skip pairs and runs whose result files exist, so only the
  item in progress is lost.

`resume` relaunches paused jobs. An MCTS job with a checkpoint uses its resume command:
`train_mcts.py ... --resume`, or `MCTS_FRESH=0` for the bare worker. Its log is appended,
not truncated.

### Drill results (2026-10-01, config B at 200 sims, both hosts)

| step | 3090 | 3080 |
|---|---|---|
| setup (snapshot load + value pretrain) | about 11 min | about 11 min |
| self-play speed | about 17-19 ep/s | about 17-18 ep/s |
| periodic checkpoint (120 s in the drill) | about 360 MB, written in a few seconds | same |
| `pause_remote.sh all` | 6 s, exit 75, GPU memory back to idle | 6-8 s |
| resume | pretraining skipped, about 20 s to the first batch; episode 3584 -> 3712, buffer 28672 -> 29696 | 4096 -> 4224, buffer 32768 -> 33792 |
| pause during setup | killed in 2 s, restarts fresh | same |

### What is lost, and why resume is faithful

The opt-in worker flags (in `training/train_mcts_worker.py`; when unset, behavior is unchanged).
The same hunks are kept as `train_mcts_worker_resume.patch` (applies to the committed file, last changed in 90558ec)
in case the file is rewritten:
- `MCTS_CKPT_EVERY_SEC=N`: every N seconds, at a batch boundary, write `resume_state.pt`.
- `MCTS_PAUSE_FILE=path`: when the file exists, or on SIGTERM, write `resume_state.pt` and exit 75.

`resume_state.pt`, about 300 MB, is written atomically (tmp file, then rename) with a
`resume_state.json` sidecar. It holds:
- the policy network, the Adam optimizer and the cosine scheduler;
- the episode counter, best eval WP, the last eval episode and the weight-sync counter;
- the full 150K-entry replay ring buffer, its fill level and write index;
- the generator's next episode seed;
- the batches already generated but not yet trained on (the generator is parked after
  its in-flight batch and the queue is drained into the state);
- the exact policy weights the CUDA engine holds, which can lag the network by up to
  512 episodes;
- the Python `random`, NumPy, torch CPU and CUDA RNG states.

On resume every one of these is restored before self-play restarts, so the run continues
from the same episode with the same buffer contents and LR schedule.

- **Clean pause:** nothing is lost. The checkpoint is taken after the in-flight batch
  finishes, and the produced batches are carried over.
- **Hard kill** (power loss, `wsl --shutdown`, a forced kill after timeout): at most the
  last `MCTS_CKPT_EVERY_SEC` (480 s, so 8 minutes) of self-play plus training.
- **Not bit-identical:** the original pipeline is itself nondeterministic. The generator
  thread and the eval share Python's `random` without a fixed interleaving, so a resumed
  run is a faithful continuation of the same process but not a byte-for-byte replay.
- **Best-eval files unchanged:** `draft_policy.pt` and `draft_policy_checkpoint.pt` keep
  the historical best-eval semantics. A plain `--resume` without the flags still restarts
  from the best-eval checkpoint (an older rollback behavior), which is why the job manager
  always sets the flags.

## Syncing new code (e.g. the fixed kernel)

```bash
training/remote_workers/sync.sh max-windows-3090 code      # ~1 s, only changed files
training/remote_workers/sync.sh 3080-gaming-desktop code
for h in max-windows-3090 3080-gaming-desktop; do
  ssh $h "wsl -e bash -s" <<< 'bash ~/hots/repo/training/remote_workers/build_ext.sh && source ~/hots/env.sh && cd ~/hots/repo/training && python remote_workers/smoke.py --gpu'
done
```

`sync.sh` copies the working tree, including uncommitted edits, so sync from the state
you mean to run. Rebuild after any change under `cuda_mcts/` or `overfit2026/cuda_ofit/`,
and never rebuild while a job is running on that host.

`sync.sh <host> data` re-sends the data manifest (`data_manifest.sh`, about 4.4 GB,
unchanged files skipped). It covers:
- the pinned snapshot and the top-level `training/*.pt`;
- `paper1_revision/{models,cache/stats*,cache/*.json|npz}` and the policies in
  `paper1_revision/mcts_runs/*/draft_policy.pt`;
- `rerun2026/{models,results}` and the K_truebase, F_400sim and J_800sim_s0 policies;
- `overfit2026/{models,cache top level,results}` and `qm2026/results`.

Add paths in `data_manifest.sh` if a new script needs more.

## Fetching results back

```bash
# one MCTS run (policy, best checkpoint, meta; skip the 300 MB resume state)
nice -n 19 rsync -rlt --rsync-path="wsl rsync" --exclude resume_state.pt \
  max-windows-3090:/home/max/hots/repo/training/paper1_revision/mcts_runs/F_oof_s0/ \
  training/paper1_revision/mcts_runs/F_oof_s0/
# its training log
nice -n 19 rsync -t --rsync-path="wsl rsync" \
  max-windows-3090:/home/max/hots/repo/training/paper1_revision/logs/mcts_F_oof_s0.log training/paper1_revision/logs/
# benchmark / tournament results
nice -n 19 rsync -rlt --rsync-path="wsl rsync" \
  max-windows-3090:/home/max/hots/repo/training/paper1_revision/results/ training/paper1_revision/results/
```

Rsync to a staging directory first if a local run with the same name exists.

## Setup from scratch (already done on both hosts)

```bash
ssh <host> "wsl -e bash -s" < /dev/null  # check connectivity
ssh <host> 'wsl -e bash -lc "mkdir -p ~/hots/logs && cat > ~/hots/setup_env.sh"' < training/remote_workers/setup_env.sh
echo 'bash ~/hots/setup_env.sh cu128 > ~/hots/logs/setup_env.log 2>&1; tail -3 ~/hots/logs/setup_env.log' | ssh <host> "wsl -e bash -s"
training/remote_workers/sync.sh <host> all
echo 'bash ~/hots/repo/training/remote_workers/build_ext.sh' | ssh <host> "wsl -e bash -s"
```

Gotchas:
- `.wslconfig` needs `[general] instanceIdleTimeout=-1` (besides `[wsl2] vmIdleTimeout=-1`).
  Without it, once the SSH session that launched a job ends, WSL treats the distro as idle and
  terminates it, killing the job (seen on the 3090 after the WSL 2.7 restart on 2026-10-03,
  where the systemd user session also fails to start). Check: `ps -o lstart= -p 1` must not change
  between two SSH sessions.
- On the 3080 (driver 576.52), `CUDA_VISIBLE_DEVICES=""` makes WSL's libcuda abort in
  `cuInit` with a "double free". Do not hide the GPU that way; `build_ext.sh` keeps it visible.
- The extensions are built for sm_86 only (`TORCH_CUDA_ARCH_LIST=8.6`).
