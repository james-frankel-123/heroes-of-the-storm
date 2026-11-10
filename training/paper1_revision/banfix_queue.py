"""
Re-evaluation of every Q-learning baseline after the ban-row fix of the
transition builder (ban rows take the banning team's result; see banfix.py).
Same jobs and protocol as swapfix_queue.py, plus the bootstrapped CQL/IQL
tournament pairs against a retrained baseline. Pre-fix outputs are moved to
rerun2026/ns/p1site/archive_banbug_2026-11-10/results (never deleted).

Queues (from ~/hots/repo/training):
  archive   move pre-fix outputs aside
  list      print what archive would move
  rich | p1 | tour N    as swapfix_queue.py
  td N      bootstrapped CQL/IQL pairs (td_tour.py, skips existing records)
"""
import os
import sys
import glob
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swapfix_queue as sq  # noqa: E402
from site_b3_queue import P1, S, NS, T  # noqa: E402

sq.ARCH = os.path.join(NS, "archive_banbug_2026-11-10", "results")
TD_NAMES = ("cql_td_a1.0", "iql_td_t0.7_b3.0")
_old = sq.old_outputs


def old_outputs():
    out = _old()
    for td in TD_NAMES:
        for name in sq.RETRAINED:
            out += glob.glob(S("td", td, "tournament", f"*{name}*.json"))
        out += [p for p in (S("td", td, "tournament_scores_new.npz"), S("td", td, "tournament_standings.json"))
                if os.path.exists(p)]
    return sorted(set(out))


sq.old_outputs = old_outputs


def td(procs):
    here = os.path.dirname(os.path.abspath(__file__))
    for name in TD_NAMES:
        e = dict(os.environ)
        e.update(P1)
        e.update({"TD_NAME": name, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        rc = subprocess.call([sys.executable, "-u", os.path.join(here, "td_tour.py"), "--pool", str(procs)], cwd=T, env=e)
        print(f"td {name} rc={rc}", flush=True)
    print("TD DONE", flush=True)


if __name__ == "__main__":
    q = sys.argv[1]
    if q == "td":
        td(int(sys.argv[2]))
    else:
        sys.argv = sys.argv[:1] + sys.argv[1:]
        {"archive": sq.archive, "list": lambda: print("\n".join(old_outputs())),
         "rich": lambda: sq.run_serial(sq.rich_jobs()), "p1": lambda: sq.run_serial(sq.p1_jobs()),
         "tour": lambda: sq.tour(int(sys.argv[2]))}[q]()
