"""
Non-MCTS stage B3 of the paper-1 site-tier rebuild, for the 3090 (pause-safe:
each job is skipped when its output exists; the job in progress restarts).

Two environments:
  RICH  rerun2026 rich evaluation of the retrained offline-RL baselines
        (namespace p1site, its GD pool). The interaction metrics use the
        community aggregates as the instrument, as in the paper:
        WP_STATS_PATH = frozen HP statistics, compositions from the pinned
        Heroes Profile table.
  P1    paper1_revision scripts on site tiers (P1_TIERS=site) with the
        p1site GD pool and baselines (RERUN_NS=p1site).

Queues (run concurrently, <= ~16 cores in total):
  rich    GD, CQL a=0.5/1.0, BC-CQL b=1, MCQ t=0.5, Gourdeau greedy, IQL
          t=0.9 b=3, discriminator degen + rich
  p1      sanity, cross-evaluation, greedy pool (rich greedy rows, WR sweep
          composition rates) + aggregate, MCQ dead units, Gourdeau eval, NGS
          quartiles, scope sweep, leak-free enriched CQL (the ensemble scores MCTS and
          tournament drafts, so it runs after those)
          (build, train a=2.0 and a=0.5, eval)

Usage (worker, from ~/hots/repo/training): python paper1_revision/site_b3_queue.py <queue>
"""
import os
import sys
import glob
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.dirname(HERE)
SITE = os.path.join(HERE, "site")
NS = os.path.join(T, "rerun2026", "ns", "p1site")
SNAP = os.path.join(T, "snapshots", "replay_snapshot_2026-05-22_1956753_p1site.json")
BASE = {"RERUN_NS": "p1site", "RERUN_SPLIT": "p1val", "REPLAY_SNAPSHOT_PATH": SNAP,
        "RERUN_SLOW_GPU": "-1", "REPLAY_SNAPSHOT": "1", "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "disabled", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
RICH = dict(BASE, WP_STATS_PATH=os.path.join(T, "frozen_stats_2026-05-19.json"),
            P1R_COMP_PATH=os.path.join(T, "pins", "compositions_hp_f2eb025.json"),
            PYTHONPATH=os.path.join(HERE, "oct2026_site"))
P1 = dict(BASE, P1_TIERS="site", P1R_NPROC="6",
          WP_STATS_PATH=os.path.join(SITE, "cache", "stats", "deploy.json"),
          P1R_COMP_PATH=os.path.join(SITE, "cache", "stats", "deploy_compositions.json"),
          PYTHONPATH=os.path.join(HERE, "oct2026_site"))
P3 = os.path.join(T, "rerun2026", "phase3_benchmarks.py")
R = lambda *p: os.path.join(NS, "results", *p)
S = lambda *p: os.path.join(SITE, "results", *p)
D = os.path.join(HERE, "deferred.py")


def queues():
    Q = {"rich": [], "p1": []}
    # the order puts baselines whose training has finished first; each job waits
    # for its model's final meta file (NEEDS)
    for s in ("gd", "mcq_t0.5", "gourdeau", "bccql_b1.0", "cql_a0.5", "cql_a1.0"):
        Q["rich"].append((f"rich_{s}", [P3, "--task", "rich-eval", "--strategy", s, "--drafts", "1000",
                                        "--seeds", "5"], R("rich_eval", f"{s}.json"), RICH))
    Q["rich"].append(("rich_iql", [os.path.join(T, "rerun2026", "experiment_iql_draft.py"), "--task", "rich-eval",
                                   "--cell", "t0.9_b3.0"], R("rich_eval", "iql_t0.9_b3.0.json"), RICH))
    disc = os.path.join(T, "rerun2026", "train_gourdeau_discriminator.py")
    Q["rich"].append(("disc_degen", [disc, "--stage", "degen"], None, RICH))
    Q["rich"].append(("disc_rich", [disc, "--stage", "rich"], R("rich_eval", "gourdeau_discriminator.json"), RICH))
    p = lambda f: os.path.join(HERE, f)
    Q["p1"] += [
        ("sanity", [p("sanity.py")], S("sanity.json"), P1),
        ("crosseval", [p("crosseval.py")], S("crosseval.json"), P1),
        ("greedy_pool", [p("greedy_evals.py"), "pool", "--procs", "6"], None, P1),
        ("greedy_agg", [p("greedy_evals.py"), "aggregate"], S("greedy_rich.json"), P1),
        ("mcq_dead", [p("mcq_dead_units.py")], S("mcq_dead_units.json"), P1),
        ("gourdeau_eval", [p("eval_gourdeau.py")], S("gourdeau_eval.json"), P1),
        ("ngs_quartiles", [D, "ngs_quartiles"], S("deferred", "ngs_quartiles.json"), P1),
        ("cql_build", [D, "cql_build"], None, P1),
        ("cql_train_a2.0", [D, "cql_train_a2.0"], None, P1),
        ("cql_train_a0.5", [D, "cql_train_a0.5"], None, P1),
        ("cql_eval", [D, "cql_eval"], S("deferred", "cql_enriched_eval.json"), P1),
    ]
    return Q


META = lambda n: os.path.join(NS, "models", "meta", f"{n}.json")
NEEDS = {"rich_cql_a0.5": META("cql_naive_a0.5"), "rich_cql_a1.0": META("cql_naive_a1.0"),
         "rich_bccql_b1.0": META("bccql_b1.0"), "rich_mcq_t0.5": META("mcq_t0.5"),
         "rich_iql": META("iql_t0.9_b3.0"),
         }


def main():
    import time
    q = sys.argv[1]
    for name, argv, marker, env in queues()[q]:
        if marker and os.path.exists(marker):
            print(f"skip {name} (done)", flush=True)
            continue
        need = NEEDS.get(name)
        while need and not os.path.exists(need):
            time.sleep(300)
        print(f"=== {name}", flush=True)
        e = dict(os.environ)
        e.update(env)
        rc = subprocess.call([sys.executable, "-u"] + argv, cwd=T, env=e)
        if rc != 0:
            print(f"FAILED {name} rc={rc}", flush=True)     # continue with the rest
    print(f"QUEUE {q} DONE", flush=True)


if __name__ == "__main__":
    main()
