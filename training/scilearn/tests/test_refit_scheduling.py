"""Regression test for the cycle-2 stall (w2-K-blinded, Oct 15 2026, 20:29 to 22:10 UTC).

Root cause: w2-K-blinded accepted p01 at 20:18 and p02 at 20:25. Test threads that had
copied the session state between the two accepts needed an incumbent with p01; threads
started after p02 needed one with p01 and p02. Both refits (about 100 minutes each) launched
at 20:29 and 20:35 and held both 3090 slots until 22:10 and 22:18. No test of any session
could run in that time, and w2-K-b's own refit could start only at 22:12. The K+p01 refit was
already out of date when it launched.

This test drives the real finish_q2, ensure_fit, remote_slot and remote.submit against a
fake host whose jobs take time (refit 1.2 s, test 0.1 s), with two sessions:
  A: one test thread copies the state after the first accept; a second accept lands before
     its refit can launch; a second test thread starts after it.
  B: needs its own refit, and has a short test against an incumbent already fitted.

  PYTHONPATH=. python -m training.scilearn.tests.test_refit_scheduling
"""
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

from training.scilearn import harness as HN
from training.scilearn import remote as R
from training.scilearn.ledger import Ledger

REFIT_S, TEST_S = 1.2, 0.1


class FakeHost:
    def __init__(self):
        self.jobs, self.launches, self.lock = {}, [], threading.Lock()
        self.running, self.max_refits, self.max_total = set(), 0, 0
        self.done_at = {}

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
                self.running.add(name)
                self.max_total = max(self.max_total, len(self.running))
                self.max_refits = max(self.max_refits, sum(n.startswith("scilearn_inc_") for n in self.running))
                dur = REFIT_S if name.startswith("scilearn_inc_") else TEST_S
                threading.Timer(dur, self._done, args=(name,)).start()
                return ""
        raise ValueError(op)

    def _done(self, name):
        with self.lock:
            self.jobs[name] = "completed"
            self.running.discard(name)
            self.done_at[name] = time.time()

    def rsh(self, host, script, timeout=600):
        if script.startswith("test -f"):
            name = "scilearn_" + script.split("/jobs/")[1].split("/")[0]
            return "yes\n" if self.jobs.get(name) == "completed" else ""
        return ""

    def wait(self, job_id, host=None, **kw):
        while self.jobs.get(f"scilearn_{job_id}") == "running":
            time.sleep(0.01)
        return {"models": {"INC": {"eval": {"Dpost": {"logloss": 0.68}}}}, "paired": {}}


class FakeBroker:
    def __init__(self, sid, accepted):
        self.sid, self.dev, self.meta = sid, "Dpost", {}
        self.lock = threading.RLock()
        self.stages = {}
        self.state = {"session_id": sid, "arm": "b", "incumbent": "K0", "snapshot": "snap_test",
                      "accepted_terms": list(accepted), "cards": {}}
        self.gated = {}

    def add_card(self, cid):
        with self.lock:
            self.state["cards"][cid] = {"job": f"q2_{self.sid}_{cid}", "term": {"type": "rule", "name": cid}}

    def accept(self, name):
        with self.lock:
            self.state["accepted_terms"].append({"type": "multiset", "name": name, "min_teams": 2000})


def T(name):
    return {"type": "multiset", "name": name, "min_teams": 2000}


def _patch(tmp, fake, fixed):
    saved = {"HN": {k: getattr(HN, k) for k in ("REGISTRY", "LOCKS", "Ledger", "SLOT_POLL",
                                                 "recent_s_builds", "apply_gate", "key_lock")},
             "R": {k: getattr(R, k) for k in ("hotsjob", "rsh", "wait", "JOBS_LOCAL", "make_extract",
                                              "sync_code", "ship_extract", "rsync_to")}}
    HN.REGISTRY = tmp / "incumbents.json"
    HN.LOCKS = tmp / "locks"
    HN.Ledger = lambda name: Ledger(name, root=tmp / "ledger")
    HN.SLOT_POLL = 0.01
    HN.recent_s_builds = lambda meta: []

    def gate(broker, cid, res, inc):
        broker.gated[cid] = (inc["job"], time.time())
    HN.apply_gate = gate
    R.hotsjob, R.rsh, R.wait = fake.hotsjob, fake.rsh, fake.wait
    R.JOBS_LOCAL = tmp / "jobs"
    R.make_extract = lambda label_mode="real", **kw: "fake_extract"
    R.sync_code = R.ship_extract = R.rsync_to = lambda *a, **k: None
    for st in fixed:     # incumbents that already exist (registered, as after prefit)
        key = HN.incumbent_key(st, "Dpost")
        HN._registry_put(key, {"job": f"inc_{key}", "name": "INC", "dev": "Dpost"})
    return saved


def _restore(saved):
    for k, v in saved["HN"].items():
        setattr(HN, k, v)
    for k, v in saved["R"].items():
        setattr(R, k, v)


def scenario(old_code):
    """Returns the fake host and the brokers after the cycle-2 sequence of events."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fake = FakeHost()
        A = FakeBroker("A", [T("a_c01")])
        B = FakeBroker("B", [T("b_c01")])
        fixed = [json.loads(json.dumps(B.state))]          # B's starting incumbent exists
        saved = _patch(tmp, fake, fixed)
        if old_code:
            # pre-fix scheduling: no refit token and no supersession check
            real_lock = HN.key_lock

            class _maybe:
                def __init__(self, name):
                    self.inner = None if name == "refit_slot" else real_lock(name)

                def __enter__(self):
                    return self.inner.__enter__() if self.inner else self

                def __exit__(self, *exc):
                    return self.inner.__exit__(*exc) if self.inner else False
            HN.key_lock = _maybe
            HN.ensure_incumbent_saved = HN.ensure_incumbent
            HN.ensure_incumbent = lambda st, dev, current=None, on_stage=None: HN.ensure_fit(st, dev, on_stage=on_stage)
        try:
            # another session's refit is running and holds the refit token (old code: a slot)
            blocker = HN.key_lock("refit_slot") if not old_code else HN.remote_slot()
            blocker.__enter__()
            A.accept("a_p01")
            A.add_card("x1")
            th = [threading.Thread(target=HN.finish_q2, args=(A, "x1"))]
            th[0].start()
            time.sleep(0.15)
            stage_seen = A.stages.get("x1", "")
            A.accept("a_p02")                                # second accept, 7 minutes later
            A.add_card("x2")
            th.append(threading.Thread(target=HN.finish_q2, args=(A, "x2")))
            th[-1].start()
            B.accept("b_p01")                                # B accepts too and needs a refit
            B.add_card("y1")
            th.append(threading.Thread(target=HN.finish_q2, args=(B, "y1")))
            th[-1].start()
            time.sleep(0.15)
            blocker.__exit__(None, None, None)
            t_release = time.time()
            # B's short test against its existing incumbent, submitted while refits are pending
            time.sleep(0.2)
            with B.lock:
                saved_terms = B.state["accepted_terms"]
                B.state["accepted_terms"] = saved_terms[:1]
            B.add_card("y0")
            snap = HN._snapshot(B)
            with B.lock:
                B.state["accepted_terms"] = saved_terms
            t_short = time.time()
            short = threading.Thread(target=_short_test, args=(B, "y0", snap))
            short.start()
            for t in th + [short]:
                t.join(timeout=30)
            return fake, A, B, t_short, t_release, stage_seen
        finally:
            if old_code:
                HN.ensure_incumbent = HN.ensure_incumbent_saved
                del HN.ensure_incumbent_saved
            _restore(saved)


def _short_test(broker, cid, st):
    """A Q2 test whose incumbent is already fitted (the state copy predates the accept)."""
    inc = HN.ensure_incumbent(st, "Dpost")
    job = st["cards"][cid]["job"]
    with HN.remote_slot():
        R.submit({"job_id": job, "kind": "q2"}, HN.HOST)
        R.wait(job, HN.HOST)
    broker.gated[cid] = (inc["job"], time.time())


def run():
    res = []
    fake, A, B, t_short, t_rel, stage_seen = scenario(old_code=False)
    refits = [n for n in fake.launches if n.startswith("scilearn_inc_")]
    newest_A = "inc_" + HN.incumbent_key(A.state, "Dpost")
    res.append(("all four tests finish", set(A.gated) == {"x1", "x2"} and set(B.gated) == {"y1", "y0"}))
    res.append(("the superseded K+p01 refit of A is never launched", len(refits) == 2
                and f"scilearn_{newest_A}" in refits))
    res.append(("both of A's tests use A's newest incumbent", A.gated.get("x1", ("",))[0] == newest_A
                and A.gated.get("x2", ("",))[0] == newest_A))
    res.append(("at most one refit holds a slot at a time", fake.max_refits == 1))
    res.append(("at most two remote jobs at once", fake.max_total <= 2))
    lat = B.gated.get("y0", (None, 1e18))[1] - t_short
    res.append((f"a short test is not stuck behind refits ({lat:.2f} s, refit {REFIT_S} s)", lat < REFIT_S / 2))
    res.append(("test-status stage names the refit wait", "incumbent" in stage_seen))

    # control: the pre-fix scheduling on the same events
    fake0, A0, B0, t0_short, _, _ = scenario(old_code=True)
    refits0 = [n for n in fake0.launches if n.startswith("scilearn_inc_")]
    lat0 = B0.gated.get("y0", (None, 1e18))[1] - t0_short
    res.append((f"control: old code launches the superseded refit ({len(refits0)} refits) and "
                f"blocks the short test ({lat0:.2f} s)", len(refits0) == 3 and fake0.max_refits == 2
                and lat0 > REFIT_S / 2))
    for name, ok in res:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    n = sum(ok for _, ok in res)
    print(f"{n}/{len(res)} passed")
    return n == len(res)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
