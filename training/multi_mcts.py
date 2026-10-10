"""
Several MCTS training runs in one process, one thread each, so their search
kernels run concurrently on one GPU (per-engine CUDA streams, cuda_mcts
kernel_bindings.cpp). Two separate processes only time-slice the GPU; inside
one process the runs fill each other's idle SMs (5090 search benchmark,
2026-10-09: 1 engine 29.2 ep/s total, 2 engines 34.9, 4 engines 37.8).

Each run keeps its own:
  * environment: os.environ is a per-thread view (base environment + the run's
    overrides), inherited by threads the run starts (its generation thread)
  * log file: stdout/stderr of the run's threads go to its log
  * Python `random` and NumPy global RNG (seeded per run; restored per run on
    resume), so one run's resume cannot replay another run's random stream.
    torch's global RNG stays shared (network init and dropout only).
  * pause: SIGTERM (hotsjob pause) creates every run's MCTS_PAUSE_FILE; each
    run writes its resume_state.pt at its next batch boundary. The process exits
    75 once every unfinished run has paused, 0 when all completed, 1 if any failed.

Spec (JSON):
  {"setup": "drift_v2" | "paper1_site" | null,      process-wide setup, same for all runs
   "script": "train_mcts_worker.py",                 run as __main__ in each thread (runpy)
   "runs": [{"name": "...", "log": "path", "env": {"MCTS_SAVE_DIR": ..., "MCTS_PAUSE_FILE": ..., ...}}]}
Every run's env must set MCTS_SAVE_DIR and MCTS_PAUSE_FILE (distinct per run).
Runs that completed earlier (run_status.json "completed" next to the spec) are skipped.

Usage (cwd = training/): python multi_mcts.py <spec.json>
"""
import json
import os
import random as _random_mod
import runpy
import signal
import sys
import threading
import traceback
import types
from collections.abc import MutableMapping

TRAINING = os.path.dirname(os.path.abspath(__file__))
PAUSE_EXIT = 75

_ctx = threading.local()  # .env (dict overrides), .log (file), .rng (random.Random), .nprng (RandomState)


def _cur(attr):
    return getattr(_ctx, attr, None)


class ThreadEnviron(MutableMapping):
    """os.environ replacement: the base environment plus the current run's overrides."""

    def __init__(self, base):
        self._base = base

    def _ov(self):
        return _cur("env")

    def __getitem__(self, k):
        ov = self._ov()
        if ov is not None and k in ov:
            if ov[k] is None:
                raise KeyError(k)
            return ov[k]
        return self._base[k]

    def __setitem__(self, k, v):
        ov = self._ov()
        if ov is not None:
            ov[k] = v
        else:
            self._base[k] = v

    def __delitem__(self, k):
        ov = self._ov()
        if ov is not None:
            if k not in self:
                raise KeyError(k)
            ov[k] = None
        else:
            del self._base[k]

    def _keys(self):
        ov = self._ov() or {}
        keys = [k for k in self._base if ov.get(k, "") is not None]
        keys += [k for k, v in ov.items() if v is not None and k not in self._base]
        return keys

    def __iter__(self):
        return iter(self._keys())

    def __len__(self):
        return len(self._keys())

    def copy(self):
        return {k: self[k] for k in self._keys()}


class _Dispatch:
    """stdout/stderr: the current run's log, else the original stream."""

    def __init__(self, orig):
        self._orig = orig

    def _t(self):
        return _cur("log") or self._orig

    def write(self, s):
        return self._t().write(s)

    def flush(self):
        return self._t().flush()

    def __getattr__(self, k):
        return getattr(self._t(), k)


class _RandomModule(types.ModuleType):
    """`random` replacement: functions of the current run's random.Random, else the real module."""

    def __init__(self, real):
        super().__init__("random")
        self.__dict__["_real"] = real

    def __getattr__(self, k):
        r = _cur("rng")
        if r is not None and hasattr(r, k) and not k.startswith("_"):
            return getattr(r, k)
        return getattr(self.__dict__["_real"], k)


class _NumpyRandom(types.ModuleType):
    """numpy.random replacement: the global-state functions of the current run's
    RandomState (randint, get_state, ...), else the real numpy.random."""

    STATEFUL = {"randint", "rand", "randn", "random", "random_sample", "choice", "shuffle", "permutation",
                "uniform", "normal", "get_state", "set_state", "seed", "integers", "beta", "dirichlet",
                "gamma", "binomial", "poisson", "exponential", "multinomial", "standard_normal"}

    def __init__(self, real):
        super().__init__("numpy.random")
        self.__dict__["_real"] = real

    def __getattr__(self, k):
        r = _cur("nprng")
        if r is not None and k in self.STATEFUL and hasattr(r, k):
            return getattr(r, k)
        return getattr(self.__dict__["_real"], k)


def _install():
    os.environ = ThreadEnviron(os.environ)
    sys.stdout = _Dispatch(sys.stdout)
    sys.stderr = _Dispatch(sys.stderr)
    sys.modules["random"] = _RandomModule(_random_mod)
    import numpy
    numpy.random = _NumpyRandom(numpy.random)
    # threads inherit the creating thread's run context
    orig_init, orig_run = threading.Thread.__init__, threading.Thread.run

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self._mm_ctx = {n: _cur(n) for n in ("env", "log", "rng", "nprng", "name")}

    def run(self):
        for n, v in getattr(self, "_mm_ctx", {}).items():
            setattr(_ctx, n, v)
        return orig_run(self)

    threading.Thread.__init__, threading.Thread.run = init, run


def _load_kernel():
    """Load the one cuda_mcts_kernel build every run shares (MCTS_KERNEL_DIR or
    training/cuda_mcts) and require the composition fallback the research WPs
    were trained with (mid, high, low)."""
    import importlib.util
    d = os.environ.get("MCTS_KERNEL_DIR") or os.path.join(TRAINING, "cuda_mcts")
    so = [f for f in os.listdir(d) if f.startswith("cuda_mcts_kernel.") and f.endswith(".so")]
    if len(so) != 1:
        raise SystemExit(f"expected one cuda_mcts_kernel .so in {d}, found {so}")
    sp = importlib.util.spec_from_file_location("cuda_mcts_kernel", os.path.join(d, so[0]))
    k = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(k)
    sys.modules["cuda_mcts_kernel"] = k
    order = getattr(k, "COMP_FALLBACK_ORDER", None)
    if order != "mid_high_low":
        raise SystemExit(f"kernel composition fallback is {order}; rebuild with HOTS_COMP_FALLBACK_MID_HIGH_LOW=1")
    return order


def _check_save_dir(save, order):
    """Same rule as drift_rebuild/v2/run.py: no resuming from state an old-order
    kernel wrote; mark the directory for this build."""
    marker = os.path.join(save, "comp_fallback_order.txt")
    state = [f for f in ("resume_state.pt", "draft_policy_checkpoint.pt", "draft_policy.pt", "DONE")
             if os.path.exists(os.path.join(save, f))]
    prev = open(marker).read().strip() if os.path.exists(marker) else None
    if state and prev != order:
        raise SystemExit(f"{save} holds {state} from a kernel with fallback {prev or 'unmarked'}")
    os.makedirs(save, exist_ok=True)
    open(marker, "w").write(order + "\n")


def _serialize_setup():
    """Value-head pretraining and the GD bootstrap are CPU/Python-heavy; several in
    one process fight over the GIL (2 at once took ~3 h on the 5090 against ~15 min
    alone). Runs take turns on them; runs already self-playing keep the GPU busy."""
    import train_draft_policy as tdp
    lock = threading.Lock()
    for fn in ("pretrain_value_head", "bootstrap_from_generic_draft"):
        orig = getattr(tdp, fn)

        def wrapped(*a, _orig=orig, _fn=fn, **k):
            with lock:
                print(f"[multi_mcts] {_fn}: start (one run at a time)", flush=True)
                return _orig(*a, **k)
        setattr(tdp, fn, wrapped)


class _Exit(Exception):
    def __init__(self, code):
        self.code = code


def _setup(kind):
    if kind == "drift_v2":
        sys.path.insert(0, os.path.join(TRAINING, "drift_rebuild", "v2"))
        import v2env  # noqa: F401
    elif kind == "paper1_site":
        # the paper-1 child patch (own-corpus compositions); same table for every run
        import sweep_enriched_wp as swp
        comp = os.environ["P1R_COMP_PATH"]

        def _load_compositions(self):
            raw = json.load(open(comp))
            self.comp_data = {t: {",".join(sorted(c["roles"])): (c["winRate"], c["games"]) for c in cs}
                              for t, cs in raw.items()}
        swp.StatsCache._load_compositions = _load_compositions
    elif kind:
        raise SystemExit(f"unknown setup {kind!r}")


def main():
    spec_path = os.path.abspath(sys.argv[1])
    spec = json.load(open(spec_path))
    status_path = os.path.join(os.path.dirname(spec_path), "run_status.json")
    status = json.load(open(status_path)) if os.path.exists(status_path) else {}
    os.chdir(TRAINING)
    sys.path.insert(0, TRAINING)
    os.environ.update({k: str(v) for k, v in spec.get("env", {}).items()})  # shared by every run
    import torch  # noqa: F401  (libtorch before any kernel .so)
    order = _load_kernel()
    _setup(spec.get("setup"))
    _install()
    _serialize_setup()
    script = os.path.join(TRAINING, spec.get("script", "train_mcts_worker.py"))
    runs = [r for r in spec["runs"] if status.get(r["name"]) != "completed"]
    for r in runs:
        # resume from SAVE_DIR/resume_state.pt when it exists, else start fresh
        # (the worker does both with MCTS_FRESH=0), as a hotsjob resume would
        r["env"]["MCTS_FRESH"] = "0"
    for r in runs:
        for k in ("MCTS_SAVE_DIR", "MCTS_PAUSE_FILE"):
            if not r["env"].get(k):
                raise SystemExit(f"run {r['name']}: env needs {k}")
    if len({r["env"]["MCTS_PAUSE_FILE"] for r in runs}) != len(runs):
        raise SystemExit("MCTS_PAUSE_FILE must differ per run")
    for r in runs:
        _check_save_dir(r["env"]["MCTS_SAVE_DIR"], order)
    print(f"[multi_mcts] {len(runs)} runs ({len(spec['runs']) - len(runs)} already completed): "
          f"{', '.join(r['name'] for r in runs)}", flush=True)

    def on_term(*_):
        for r in runs:
            open(r["env"]["MCTS_PAUSE_FILE"], "a").close()
        print("[multi_mcts] SIGTERM: pause requested for every run", flush=True)
    signal.signal(signal.SIGTERM, on_term)

    lock = threading.Lock()

    def run_one(i, r):
        _ctx.name = r["name"]
        _ctx.env = {k: str(v) for k, v in r["env"].items()}
        os.makedirs(os.path.dirname(os.path.abspath(r["log"])), exist_ok=True)
        _ctx.log = open(r["log"], "a", buffering=1)
        seed = int(r["env"].get("P1R_RUN_SEED", r["env"].get("PYTHONHASHSEED", i)))
        _ctx.rng = _random_mod.Random(seed)
        import numpy as np
        _ctx.nprng = np.random.RandomState(seed)
        code = 0

        try:
            runpy.run_path(script, run_name="__main__")
        except _Exit as e:
            code = e.code
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
        except BaseException:
            traceback.print_exc()
            code = 1
        st = {0: "completed", PAUSE_EXIT: "paused"}.get(code, "failed")
        print(f"[multi_mcts] run {r['name']} {st} (code {code})", flush=True)
        _ctx.log.close()
        with lock:
            status[r["name"]] = st
            json.dump(status, open(status_path, "w"), indent=1)

    # signal handlers can only be set from the main thread; the runner owns SIGTERM
    real_signal = signal.signal

    def thread_signal(sig, handler):
        if _cur("name") is not None:
            return None
        return real_signal(sig, handler)
    signal.signal = thread_signal

    # os._exit inside a run thread must not kill the process: route it through the thread
    real_exit = os._exit

    def thread_exit(c=0):
        if _cur("name") is not None:
            raise _Exit(c)
        real_exit(c)
    os._exit = thread_exit

    threads = [threading.Thread(target=run_one, args=(i, r), name=f"run:{r['name']}") for i, r in enumerate(runs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    final = [status.get(r["name"]) for r in runs]
    code = 1 if "failed" in final else (PAUSE_EXIT if "paused" in final else 0)
    print(f"[multi_mcts] done: {dict(zip([r['name'] for r in runs], final))} -> exit {code}", flush=True)
    sys.stdout.flush()
    real_exit(code)


if __name__ == "__main__":
    main()
