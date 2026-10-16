"""Draft-only refits of the readable models for the fair comparison
(paper/hypothesis_system/results/fair_comparison.md).

  python -m training.scilearn.compare.refits plan          print the jobs
  python -m training.scilearn.compare.refits run [names]   submit and wait, at most 2 at a time

Each model is refit without its player-history terms (every `decay` term: the 180-day
player-hero experience term, and the incumbents' 4- and 5-day practice terms), on the same
fit tranches as the original. This is a refit, not a zeroed term: the other terms re-absorb
what they can. Jobs run on the 3090 through remote.submit (cores 24-35, nice 19), under
ids fair_*, and are registered in PRIVATE/fair_jobs.json for compare/score.py.
"""
import json
import sys
import threading

from training.scilearn import config as C
from training.scilearn import remote as R

HOST = "max-windows-3090"
REG = C.PRIVATE / "fair_jobs.json"
JOBS = C.PRIVATE / "jobs"
PLAYER_TYPES = {"decay"}
MAX_PARALLEL = 2

# source model specs: (job, model name inside the job)
SOURCES = {
    "A0": ("a0_fitSDG", "A0"),
    "K0": ("inc_b40b452c1e061c38", "INC"),
    "w2-K-b": ("inc_be8151b00daa906a", "INC"),
    "w2-K-blinded": ("inc_1a768a53b59ba3dc", "INC"),
}
FITS = {"SDG": (["S", "Dpre", "G"], ["Dpost"]), "S": (["S"], ["Dpre", "G", "Dpost"])}
PLAN = [  # (name, source, fit); ordered so the two longest fits start first
    ("A0d_SDG", "A0", "SDG"),
    ("wKbld_SDG", "w2-K-blinded", "SDG"),
    ("wKbd_SDG", "w2-K-b", "SDG"),
    ("A0d_S", "A0", "S"),
    ("K0d_SDG", "K0", "SDG"),
    ("K0d_S", "K0", "S"),
]


def draft_only_specs(source):
    job, name = SOURCES[source]
    specs = json.loads((JOBS / job / "out" / f"{name}.json").read_text())["specs"]
    out = [dict(s) for s in specs if s.get("type") not in PLAYER_TYPES]
    for s in out:
        s.pop("_vocab", None)
    return out, [s.get("name") for s in specs if s.get("type") in PLAYER_TYPES]


def job_spec(name, source, fit):
    specs, dropped = draft_only_specs(source)
    fit_tr, ev = FITS[fit]
    return {"job_id": f"fair_{name}", "kind": "fit_eval", "fit_tranches": fit_tr,
            "eval_tranches": ev, "models": {"M": specs}}, dropped


def _register(name, entry):
    import fcntl
    with open(str(REG) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        reg = json.loads(REG.read_text()) if REG.exists() else {}
        reg[name] = entry
        REG.write_text(json.dumps(reg, indent=1))


_lock = threading.Lock()


def run_one(name, source, fit, sem, host=HOST):
    with sem:
        spec, dropped = job_spec(name, source, fit)
        jid = R.submit(spec, host, label_mode="real")
        print(f"submitted {jid} on {host}", flush=True)
        res = R.wait(jid, host, poll=60, timeout=8 * 3600)
        with _lock:
            _register(name, {"job": jid, "host": host, "model": "M", "source": source, "fit": fit,
                             "dropped_terms": dropped, "eval": FITS[fit][1],
                             "p": res["models"]["M"]["p"], "seconds": res["models"]["M"]["seconds"]})
        print(name, {t: round(m["logloss"], 5) for t, m in res["models"]["M"]["eval"].items()}, flush=True)


def main(argv):
    cmd = argv[0] if argv else "plan"
    host, par = HOST, MAX_PARALLEL
    rest = list(argv[1:])
    if rest[:1] == ["--host"]:
        host, rest = rest[1], rest[2:]
        par = 1 if host == "windows-5090-wsl" else MAX_PARALLEL   # 5090: one fit at a time (11 GB)
    want = set(rest)
    plan = [p for p in PLAN if not want or p[0] in want]
    if cmd == "plan":
        for name, src, fit in plan:
            spec, dropped = job_spec(name, src, fit)
            print(name, src, fit, "terms", len(spec["models"]["M"]), "dropped", dropped)
        return
    sem = threading.Semaphore(par)
    ts = [threading.Thread(target=run_one, args=(*p, sem, host)) for p in plan]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


if __name__ == "__main__":
    main(sys.argv[1:])
