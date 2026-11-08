"""
Re-evaluation of the Q-learning baselines after the team-swap fix of the
transition builder (experiment_cql_draft.replay_to_transitions): every
evaluation that reads a CQL / BC-CQL / TCQL / MCQ / IQL checkpoint, in the
original protocol. Each job waits for its model's completion marker and is
skipped when its output exists; `archive` moves the pre-fix outputs aside
(never deletes).

Queues (from ~/hots/repo/training):
  archive   move pre-fix outputs to rerun2026/ns/p1site/archive_swapbug_2026-11-06/results
  rich      Table IV rows (5 x 1000 drafts) + supplement sweeps        (RICH env)
  p1        MCQ / no-penalty rows, enriched CQL eval, TCQL dead units   (P1 env)
  tour N    tournament + search-agent pairs involving a retrained baseline, N procs
"""
import os
import sys
import glob
import time
import shutil
import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from site_b3_queue import RICH, P1, P3, R, S, NS, T, META  # noqa: E402

ARCH = os.path.join(NS, "archive_swapbug_2026-11-06", "results")
RETRAINED = ("cql_naive_a1.0", "cql_enr_a2.0", "mcq_t0.5")
IQL_CELL = os.environ.get("IQL_CELL", "t0.9_b3.0")
DEF = lambda n: S("deferred", f"{n}.json")


def old_outputs():
    out = [R("rich_eval", f"{s}.json") for s in ("cql_a0.5", "cql_a1.0", "bccql_b1.0", "mcq_t0.5")]
    out += glob.glob(R("rich_eval", "iql_*.json")) + [R("cql_naive_5x1000.json")]
    out += glob.glob(R("cql_hyperparams", "*.json")) + glob.glob(R("mcq", "*.json")) + glob.glob(R("iql", "*"))
    out += glob.glob(R("iql_*.json")) + glob.glob(R("screening*.json"))
    out += glob.glob(S("mcq_real", "*")) + [S("mcq_dead_units.json"), DEF("cql_enriched_eval")]
    for name in RETRAINED:
        out += glob.glob(S("tournament", f"*{name}*.json")) + glob.glob(S("search_agents", "tournament", f"*{name}*.json"))
    return sorted({p for p in out if os.path.exists(p)})


def archive():
    for p in old_outputs():
        dst = os.path.join(ARCH, os.path.relpath(p, T))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(p, dst)
        print("archived", os.path.relpath(p, T))


def rich_jobs():
    J = []
    for s, need in (("cql_a1.0", "cql_naive_a1.0"), ("cql_a0.5", "cql_naive_a0.5"),
                    ("bccql_b1.0", "bccql_b1.0"), ("mcq_t0.5", "mcq_t0.5")):
        J.append((f"rich_{s}", [P3, "--task", "rich-eval", "--strategy", s, "--drafts", "1000", "--seeds", "5"],
                  R("rich_eval", f"{s}.json"), RICH, [META(need)]))
    iql = os.path.join(T, "rerun2026", "experiment_iql_draft.py")
    J.append(("rich_iql", [iql, "--task", "rich-eval", "--cell", IQL_CELL], R("rich_eval", f"iql_{IQL_CELL}.json"),
              RICH, [META(f"iql_{IQL_CELL}")]))
    J.append(("cql_basic", [P3, "--task", "cql-basic"], R("cql_naive_5x1000.json"), RICH,
              [META(f"cql_naive_a{a}") for a in ("0.1", "0.5", "1.0", "2.0", "5.0")]))
    for m_ in ("mcq_t0.4", "mcq_t0.6", "mcq_t0.7", "mcq_t0.8", "bccql_b0.1", "bccql_b0.5", "bccql_b1.0", "bccql_b2.0", "mcq_t0.5"):
        J.append((f"mcqeval_{m_}", [P3, "--task", "mcq-eval", "--model", m_], R("mcq", f"{m_}.json"), RICH, [META(m_)]))
    J.append(("iql_screen", [iql, "--task", "screen"], R("iql", "screening.json"), RICH,
              [META(f"iql_t{t}_b{b}") for t in (0.7, 0.8, 0.9) for b in (1.0, 3.0)]))
    for arch in ("512,256,128", "1024,512,256", "256,128,64"):
        for tau in (0.005, 0.001, 0.01, 0.05, 0.1):
            a_s = arch.replace(",", "x")
            J.append((f"cqlhp_{tau}_{a_s}", [P3, "--task", "cql-hp-eval", "--tau", str(tau), "--arch", arch],
                      R("cql_hyperparams", f"a1.0_t{tau}_{a_s}.json"), RICH, [META(f"cql_hp_a1.0_t{tau}_{a_s}")]))
    return J


def p1_jobs():
    here = os.path.dirname(os.path.abspath(__file__))
    mr = os.path.join(here, "mcq_real.py")
    J = [(f"mcqreal_{n}", [mr, "eval", n], S("mcq_real", f"rich_{n}.json"), P1,
          [META(n if n != "qreg_a0" else "cql_naive_a0.0")])
         for n in ("mcq_real_l0.9", "mcq_real_l0.7", "mcq_real_l0.5", "qreg_a0")]
    J.append(("cql_enr_eval", [os.path.join(here, "deferred.py"), "cql_eval"], DEF("cql_enriched_eval"), P1,
              [DEF("cql_enriched_train_a2.0"), DEF("cql_enriched_train_a0.5")]))
    J.append(("mcq_dead", [os.path.join(here, "mcq_dead_units.py")], S("mcq_dead_units.json"), P1, [META("mcq_t0.5")]))
    return J


def wait(needs):
    while not all(os.path.exists(n) for n in needs):
        time.sleep(120)


def run_serial(jobs):
    for name, argv, marker, env, needs in jobs:
        if os.path.exists(marker):
            print(f"skip {name}", flush=True)
            continue
        wait(needs)
        print(f"=== {name} {time.strftime('%H:%M')}", flush=True)
        e = dict(os.environ)
        e.update(env)
        rc = subprocess.call([sys.executable, "-u"] + argv, cwd=T, env=e)
        print(f"{'FAILED' if rc else 'done'} {name} rc={rc} {time.strftime('%H:%M')}", flush=True)
    print("QUEUE DONE", flush=True)


def tour(procs):
    here = os.path.dirname(os.path.abspath(__file__))
    from tournament import STRATEGIES
    need = {"cql_naive_a1.0": [META("cql_naive_a1.0")], "mcq_t0.5": [META("mcq_t0.5")],
            "cql_enr_a2.0": [DEF("cql_enriched_train_a2.0")], "mcq_real": [META("mcq_real_l0.9")]}
    jobs = []
    for a, b in itertools.permutations(STRATEGIES, 2):
        if a in RETRAINED or b in RETRAINED:
            jobs.append((S("tournament", f"{a}__{b}.json"), [os.path.join(here, "tournament.py"), "--pair", f"{a}__{b}"],
                         need.get(a, []) + need.get(b, []), f"tour_{a}__{b}"))
    others = [s for s in STRATEGIES]
    for o in others:
        for a, b in (("mcq_real", o), (o, "mcq_real")):
            jobs.append((S("mcq_real", "tournament", f"{a}__{b}.json"), [os.path.join(here, "mcq_tour.py"), "--pair", f"{a}__{b}"],
                         need["mcq_real"] + need.get(o, []), f"mcq_{a}__{b}"))
    for ag in ("mcts_search", "mcts_lcb", "mcts_lcb2"):
        for r_ in RETRAINED:
            for a, b in ((ag, r_), (r_, ag)):
                jobs.append((S("search_agents", "tournament", f"{a}__{b}.json"),
                             [os.path.join(here, "search_eval.py"), "pair", f"{a}__{b}"], need[r_], f"search_{a}__{b}"))
    logs = S("..", "logs", "swapfix_tour")
    os.makedirs(logs, exist_ok=True)

    def run(j):
        out, argv, needs, name = j
        if os.path.exists(out):
            return
        wait(needs)
        e = dict(os.environ)
        e.update(P1)
        e.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        with open(os.path.join(logs, f"{name}.log"), "w") as fh:
            rc = subprocess.call([sys.executable, "-u"] + argv, cwd=T, env=e, stdout=fh, stderr=subprocess.STDOUT)
        print(f"{name} rc={rc} {time.strftime('%H:%M')}", flush=True)
    # pairs whose models exist first
    jobs.sort(key=lambda j: not all(os.path.exists(n) for n in j[2]))
    with ThreadPoolExecutor(procs) as ex:
        list(ex.map(run, jobs))
    print("TOUR DONE", len(jobs), flush=True)


if __name__ == "__main__":
    q = sys.argv[1]
    if q == "archive":
        archive()
    elif q == "list":
        print("\n".join(old_outputs()))
    elif q == "rich":
        run_serial(rich_jobs())
    elif q == "p1":
        run_serial(p1_jobs())
    elif q == "tour":
        tour(int(sys.argv[2]))
