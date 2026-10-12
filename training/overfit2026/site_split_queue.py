"""
Site-tier rerun of paper 1's within-snapshot split experiment (supplement
sec:split: Table tab:split, the search-pressure sweep, the self-play cells),
as one pause-safe queue for a remote worker (RTX 5090, 16 threads, 35 GB).

Every step runs under P1_TIERS=site (artifacts under overfit2026/site/...) and
is skipped when its completion marker exists, so the queue can be killed or
paused (hotsjob plain job: SIGTERM) and rerun at any point. GPU steps run one
at a time.

  step       what                                              marker / est. cost (5090)
  stats      split.py stats: A1..A8 (+f0/f1 fold stats), B8     site/cache/split_stats/<n>.json
             (+f0/f1); N/S stats are not used here              CPU ~10-20 min, ~10 GB RAM
  train      split.py train --only pA,gB: 27 side-A proxies     every site/models/<n>.json
             (Table, capacity knobs, 4-seed ensembles) and      ~0.5 GPU-h (legacy log: 30 models
             the 5 side-B judges score.py needs                 in ~25 min), ~8 GB RAM
  heldout    split.py heldout (Table: held-out acc, slope)      site/results/split_heldout.json
                                                                ~0.3 h, ~8 GB
  gd_cache   split_gd.py cache: half-A GD samples, streamed     site/cache/gd_splitA/*/meta.json
             from the full p1site snapshot (needs draft_order)  CPU ~20 min, <6 GB RAM, 14 GB disk
  gd_train   split_gd.py train: one GD on half A                site/models/gd_splitA/gd_0.json
             (epoch-level resume)                               ~3-5 GPU-h (p1site GD: 36 epochs,
                                                                9 h on a 3090 at twice the rows)
  bc_prior   bc_prior.py: distill the half-A GD into the        site/models/bc_prior.pt
             search prior on half-A states                      ~0.3 GPU-h, ~10 GB RAM
  stage_e    stage_e.py run --grid main,deep,size,capacity,lcb  every grid run's JSON in
             (chance-node kernel, half-A GD opponent)           site/results/split_search
                                                                legacy 3.2 GPU-h with the pre-X2
                                                                kernel; expect ~6-12 h; ~12 GB
  mcts       the 8 self-play runs (pA8_leak/pA8_oof x 200/800   draft_policy.pt + "Complete." in
             sims x s0/s1, 300K episodes), one at a time, each  site/logs/mcts_<run>.log
             a pause-aware hotsjob MCTS job                     chance kernel, remote: ~11 h per
                                                                200-sim run, ~34 h per 800-sim run
                                                                (~180 GPU-h total); 7.2 GB peak
  stage_f    stage_f_selfplay.py: benchmark the 8 policies      2 result JSONs per run in
                                                                site/results/selfplay; ~1 GPU-h
  analyze    analyze_split.py (tables, contrasts, figures)      site/results/split_summary.json

RAM figures are estimates (only the MCTS peak is measured); none should pass
~15 GB: the full snapshot is only streamed, never parsed whole, and value
pretraining reads the lite snapshot. The self-play runs go 200-sim first.

Inputs the worker needs (besides the repo): overfit2026/cache/{snapshot,future,
backfill}_games.pkl.gz, backups/replay_draft_skill_tier_20260930.csv.gz, both
p1site snapshots (full + _lite), rerun2026/pre255_exclude_ids.json,
paper1_revision/site/cache/paper_test_ids.json, pins/compositions_hp_f2eb025.json,
overfit2026/site/models/qm_wp_*.pt and site/cache/realized_NODRIFT_s99.pkl, and
cuda_mcts + overfit2026/cuda_ofit built with HOTS_COMP_FALLBACK_MID_HIGH_LOW=1.

Usage (worker, from ~/hots/repo/training):
  python3 remote_workers/hotsjob.py launch split_site_queue \\
      "python3 overfit2026/site_split_queue.py" --mem-max 30G
  python3 overfit2026/site_split_queue.py [--only step,step] [--direct]
  --direct runs the self-play in-process instead of as hotsjob jobs.
Log of each step: overfit2026/site/logs/queue_<step>.log
"""
import os
import sys
import json
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
os.environ["P1_TIERS"] = "site"          # before any overfit2026 import

from overfit2026 import data  # noqa: E402

PY = sys.executable
HOTSJOB = os.path.join(TRAINING_DIR, "remote_workers", "hotsjob.py")
ENV = {"P1_TIERS": "site", "PYTHONUNBUFFERED": "1", "REPLAY_SNAPSHOT": "1",
       "WANDB_MODE": "disabled", "OFIT_FEAT_NPROC": "12", "P1R_NPROC": "8",
       "SPLIT_STATS_NPROC": "6", "SPLIT_GD_NPROC": "8"}
STATS = [f"{s}{e}{f}" for s, es in (("A", (1, 2, 4, 8)), ("B", (8,))) for e in es
         for f in ("", "f0", "f1")]
GRIDS = ["main", "deep", "size", "capacity", "lcb"]
MCTS_MEM = "16G"


def art(*p):
    return data.art(HERE, *p)


def needed_models():
    from overfit2026 import split
    return [n for n in split.spec_table() if n.startswith(("pA", "gB"))]


def mcts_done(run):
    from overfit2026.train_mcts_split import run_name
    n = run_name(*run)
    log = art("logs", f"mcts_{n}.log")
    return (os.path.exists(art("mcts_runs", n, "draft_policy.pt")) and os.path.exists(log)
            and "Complete." in open(log, errors="replace").read())


def markers():
    from overfit2026 import stage_e, split, split_gd
    from overfit2026.train_mcts_split import RUNS, run_name
    return {
        "stats": lambda: all(os.path.exists(split.stats_path(n)) for n in STATS),
        "train": lambda: all(os.path.exists(split.model_path(n)[:-3] + ".json") for n in needed_models()),
        "heldout": lambda: os.path.exists(art("results", "split_heldout.json")),
        "gd_cache": lambda: all(os.path.exists(os.path.join(split_gd.cache_dir(k), "meta.json"))
                                for k in ("gd_train", "gd_test")),
        "gd_train": lambda: all(os.path.exists(split_gd.marker(v)) for v in split_gd.VARIANTS),
        "bc_prior": lambda: os.path.exists(art("models", "bc_prior.pt")),
        "stage_e": lambda: all(os.path.exists(os.path.join(stage_e.OUT_DIR, stage_e.run_name(*g) + ".json"))
                               for gr in GRIDS for g in stage_e.grid(gr)),
        "mcts": lambda: all(mcts_done(r) for r in RUNS),
        "stage_f": lambda: all(os.path.exists(art("results", "selfplay", f"{run_name(*r)}__inf200__T{T}.json"))
                               for r in RUNS for T in (1.0, 0.0)),
        "analyze": lambda: os.path.exists(art("results", "split_summary.json")),
    }


def commands():
    from overfit2026 import split_gd
    o = "overfit2026/"
    return {
        "stats": [[o + "split.py", "stats", "--only", ",".join(STATS)]],
        "train": [[o + "split.py", "train", "--only", "pA,gB"]],
        "heldout": [[o + "split.py", "heldout"]],
        "gd_cache": [[o + "split_gd.py", "cache"]],
        "gd_train": [[o + "split_gd.py", "train", "--variant", str(v)] for v in split_gd.VARIANTS],
        "bc_prior": [[o + "bc_prior.py"]],
        "stage_e": [[o + "stage_e.py", "run", "--grid", ",".join(GRIDS)]],
        "stage_f": [[o + "stage_f_selfplay.py"]],
        "analyze": [[o + "analyze_split.py"]],
    }


def run_step(step, env):
    logp = art("logs", f"queue_{step}.log")
    for argv in commands()[step]:
        with open(logp, "a") as log:
            log.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(argv)}\n")
            log.flush()
            rc = subprocess.call([PY, "-u"] + argv, cwd=TRAINING_DIR, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
        if rc != 0:
            raise SystemExit(f"step {step} failed (rc={rc}), see {logp}")


def hotsjob(*args):
    r = subprocess.run([PY, HOTSJOB] + list(args), capture_output=True, text=True, cwd=TRAINING_DIR)
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]


def run_mcts(run, env, direct):
    """One self-play run as a pause-aware hotsjob MCTS job; returns when complete."""
    from overfit2026.train_mcts_split import run_name
    n = run_name(*run)
    if direct:
        save = art("mcts_runs", n)
        argv = [PY, "-u", os.path.join(HERE, "train_mcts_split.py"), run[0], str(run[1]), run[2], "0"]
        if os.path.exists(os.path.join(save, "resume_state.pt")):
            argv.append("--resume")
        e = dict(env, MCTS_CKPT_EVERY_SEC="480")
        rc = subprocess.call(argv, cwd=TRAINING_DIR, env=e)
        if rc != 0 or not mcts_done(run):
            raise SystemExit(f"self-play {n} failed (rc={rc})")
        return
    job = f"split_site_{n}"
    cmd = (f"P1_TIERS=site PYTHONUNBUFFERED=1 {PY} overfit2026/train_mcts_split.py "
           f"{run[0]} {run[1]} {run[2]} 0")
    retried = False
    while True:
        _, state = hotsjob("state", job)
        if state == "none":
            print(f"launch {job}", flush=True)
            rc, out = hotsjob("launch", job, cmd, "--gpu", "0", "--mem-max", MCTS_MEM)
            if rc != 0:
                raise SystemExit(f"hotsjob launch {job} failed: {out}")
        elif state == "completed":
            if not mcts_done(run):
                raise SystemExit(f"{job} completed without draft_policy.pt / 'Complete.'")
            return
        elif state in ("failed", "died"):
            if retried:
                raise SystemExit(f"{job} {state} twice; see ~/hots/logs/{job}.log")
            print(f"{job} {state}; resuming once", flush=True)
            hotsjob("resume", job)
            retried = True
        time.sleep(60)            # running / pausing / paused: wait (paused = operator's call)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of steps")
    ap.add_argument("--direct", action="store_true", help="self-play without hotsjob")
    a = ap.parse_args()
    env = dict(os.environ)
    env.update(ENV)
    M = markers()
    steps = list(M)
    if a.only:
        steps = [s for s in steps if s in a.only.split(",")]
    from overfit2026.train_mcts_split import RUNS
    for step in steps:
        if M[step]():
            print(f"skip {step} (done)", flush=True)
            continue
        print(f"=== {step} {time.strftime('%m-%d %H:%M')}", flush=True)
        if step == "mcts":
            for r in RUNS:
                if not mcts_done(r):
                    run_mcts(r, env, a.direct)
        else:
            run_step(step, env)
        if not M[step]():
            raise SystemExit(f"step {step} ended without its marker")
    print("SPLIT QUEUE DONE", flush=True)


if __name__ == "__main__":
    main()
