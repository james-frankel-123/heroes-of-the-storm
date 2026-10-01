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

Usage (from training/): setsid nohup python3 paper1_revision/site_mcts_remote.py &
Log: paper1_revision/site/logs/site_mcts_remote.log
"""
import os
import re
import json
import time
import subprocess

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


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    open(LOG, "a").write(line + "\n")


def ready():
    return (all(os.path.exists(os.path.join(NS, "meta", f"gd_{i}.json")) for i in range(5))
            and os.path.exists(os.path.join(SITE, "models", "enriched.json"))
            and os.path.exists(os.path.join(SITE, "models", "naive.json"))
            and os.path.exists(os.path.join(SITE, "models", "enriched_leak.json")))


def status(host):
    r = subprocess.run([os.path.join(RW, "jobs_remote.sh"), host], capture_output=True, text=True,
                       cwd=TRAINING_DIR)
    out = {}
    for line in r.stdout.splitlines():
        m = re.match(r"(p1site_[A-Z]_(oof|leak)_s\d)\s+(\w+)", line.strip())
        if m:
            out[m.group(1)] = m.group(3)
    return out


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
              os.path.join(TRAINING_DIR, "snapshots", "replay_snapshot_2026-05-22_1956753_p1site.json")]
    files += [os.path.join(NS, f"generic_draft_{i}.pt") for i in range(5)]
    lst = "\n".join(os.path.relpath(f, REPO) for f in files) + "\n"
    r = subprocess.run(["nice", "-n", "19", "rsync", "-rlt", "--rsync-path=wsl rsync", "-z",
                        "--files-from=-", "./", f"{host}:/home/max/hots/repo/"],
                       input=lst, text=True, cwd=REPO, capture_output=True)
    return r.returncode == 0


def launch(host, cfg, tag, seed):
    run = f"{cfg}_{tag}_s{seed}"
    wp = " --wp enriched_leak" if tag == "leak" else ""
    env = dict(os.environ, RUN_NAME=f"p1site_{run}",
               HOTSJOB_FLAGS=f"--save-dir paper1_revision/site/mcts_runs/{run} "
                             f"--progress-log paper1_revision/site/logs/mcts_{run}.log")
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
        taken = {n for s in st.values() for n in s}
        pending = [r for r in RUNS if f"p1site_{r[0]}_{r[1]}_s{r[2]}" not in taken]
        if not pending and all(v in ("done", "finished") for s in st.values() for v in s.values()):
            log("all runs finished")
            return
        # 3080: a second slot once the drift lane's MCTS job there has finished
        # (paper 1 outranks drift there; agreed via the coordinator 2026-10-01)
        if slots.get("3080-gaming-desktop") == 1 and not drift_mcts_running():
            slots["3080-gaming-desktop"] = 2
        for host, n in slots.items():
            running = sum(1 for v in st[host].values() if v in ("running", "starting"))
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
        time.sleep(300)


if __name__ == "__main__":
    main()
