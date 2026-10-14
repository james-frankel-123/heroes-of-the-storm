"""Regression test for the wave-1 "harness ERROR" (w1-K-blinded c03, c05, c13).

Root cause: after an accept, several Q2 threads in one broker each called ensure_incumbent,
found the new incumbent missing from the registry, and each launched the same remote job
`inc_<key>`. hotsjob refused the second launch ("scilearn_inc_... is already running"), the
exception escaped finish_q2, and the card ended in status "error".

This test drives the real harness.ensure_incumbent and remote.submit against a fake host
that behaves like hotsjob (refuses a second launch of a running job), from 6 threads at once.

  PYTHONPATH=. python -m training.scilearn.tests.test_incumbent_race
"""
import sys
import tempfile
import threading
import time
from pathlib import Path

from training.scilearn import harness as HN
from training.scilearn import remote as R
from training.scilearn.ledger import Ledger


class FakeHost:
    def __init__(self, run_s=0.5):
        self.jobs, self.launches, self.lock, self.run_s = {}, [], threading.Lock(), run_s

    def hotsjob(self, host, *args):
        op, name = args[0], args[1]
        with self.lock:
            if op == "state":
                return self.jobs.get(name, "none") + "\n"
            if op == "launch":
                if self.jobs.get(name) == "running":
                    raise RuntimeError(f"{host}: {name} is already running")
                self.jobs[name] = "running"
                self.launches.append(name)
                threading.Timer(self.run_s, self._done, args=(name,)).start()
                return ""
        raise ValueError(op)

    def _done(self, name):
        with self.lock:
            self.jobs[name] = "completed"

    def rsh(self, host, script, timeout=600):
        if script.startswith("test -f"):
            name = "scilearn_" + script.split("/jobs/")[1].split("/")[0]
            return "yes\n" if self.jobs.get(name) == "completed" else ""
        return ""

    def wait(self, job_id, host=None, **kw):
        while self.jobs.get(f"scilearn_{job_id}") == "running":
            time.sleep(0.05)
        return {"models": {"INC": {"eval": {"Dpost": {"logloss": 0.68}}}}}


class _NoSlot:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch(tmp, fake):
    saved = {"HN": {k: getattr(HN, k) for k in ("REGISTRY", "LOCKS", "Ledger", "remote_slot")},
             "R": {k: getattr(R, k) for k in ("hotsjob", "rsh", "wait", "JOBS_LOCAL", "make_extract",
                                              "sync_code", "ship_extract", "rsync_to")}}
    HN.REGISTRY = tmp / "incumbents.json"
    HN.LOCKS = tmp / "locks"
    HN.Ledger = lambda name: Ledger(name, root=tmp / "ledger")
    HN.remote_slot = _NoSlot     # two real slots would also allow the double launch
    R.hotsjob, R.rsh, R.wait = fake.hotsjob, fake.rsh, fake.wait
    R.JOBS_LOCAL = tmp / "jobs"
    R.make_extract = lambda label_mode="real", **kw: "fake_extract"
    R.sync_code = R.ship_extract = R.rsync_to = lambda *a, **k: None
    return saved


def _restore(saved):
    for k, v in saved["HN"].items():
        setattr(HN, k, v)
    for k, v in saved["R"].items():
        setattr(R, k, v)


def run():
    res = []
    state = {"session_id": "t", "arm": "blinded", "incumbent": "K0", "snapshot": "snap_test",
             "accepted_terms": [{"type": "multiset", "name": "card_c01", "min_teams": 2000}]}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fake = FakeHost()
        saved = _patch(tmp, fake)
        try:
            out, errs = [], []

            def worker():
                try:
                    out.append(HN.ensure_incumbent(dict(state), "Dpost")["job"])
                except Exception as e:  # the wave-1 failure mode
                    errs.append(str(e))
            th = [threading.Thread(target=worker) for _ in range(6)]
            for t in th:
                t.start()
            for t in th:
                t.join()
            res.append(("6 concurrent ensure_incumbent calls raise nothing", not errs))
            res.append(("exactly one remote launch", len(fake.launches) == 1))
            res.append(("all callers get the same incumbent job", len(set(out)) == 1 and len(out) == 6))
            res.append(("one harness ledger fit record", sum(r["kind"] == "fit" for r in
                                                            Ledger("harness", root=tmp / "ledger").records()) == 1))

            # remote.submit alone: a second submit of a running job attaches instead of failing,
            # and a completed job is not relaunched.
            fake2 = FakeHost(run_s=0.3)
            R.hotsjob, R.rsh, R.wait = fake2.hotsjob, fake2.rsh, fake2.wait
            spec = {"job_id": "q2_x", "kind": "q2"}
            R.submit(dict(spec), "max-windows-3090")
            try:
                R.submit(dict(spec), "max-windows-3090")
                ok = True
            except RuntimeError:
                ok = False
            res.append(("submit of a running job attaches", ok and len(fake2.launches) == 1))
            time.sleep(0.5)
            R.submit(dict(spec), "max-windows-3090")
            res.append(("submit of a completed job is not relaunched", len(fake2.launches) == 1))

            # The pre-fix code path (no key lock) reproduces the error against the same fake host.
            fake3 = FakeHost()
            R.hotsjob, R.rsh, R.wait = fake3.hotsjob, fake3.rsh, fake3.wait
            errs3 = []

            def old_launch():
                try:
                    fake3.hotsjob("max-windows-3090", "launch", "scilearn_inc_dup", "cmd")
                except RuntimeError as e:
                    errs3.append(str(e))
            t1, t2 = threading.Thread(target=old_launch), threading.Thread(target=old_launch)
            t1.start(); t2.start(); t1.join(); t2.join()
            res.append(("control: a blind double launch fails as in wave 1",
                        len(errs3) == 1 and "already running" in errs3[0]))
        finally:
            _restore(saved)
    for name, ok in res:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    n = sum(ok for _, ok in res)
    print(f"{n}/{len(res)} passed")
    return n == len(res)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
