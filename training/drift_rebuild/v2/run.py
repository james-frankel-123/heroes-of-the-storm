"""Run any drift script under the v2 environment:
    nice -n 19 taskset -c 48-63 python3 drift_rebuild/v2/run.py <script.py> [args...]
(cwd = training/). See v2env.py."""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2env  # noqa: E402,F401

script = os.path.abspath(sys.argv[1])


def kernel_fallback_order(so_dir):
    """COMP_FALLBACK_ORDER of the cuda_mcts_kernel build in so_dir (None: unreadable).
    Checked in a subprocess so the worker's own import of the .so is the first."""
    import subprocess
    code = ("import importlib.util,os,sys,torch;d=sys.argv[1];"
            "f=[x for x in os.listdir(d) if x.startswith('cuda_mcts_kernel.') and x.endswith('.so')];"
            "assert len(f)==1,f;"
            "s=importlib.util.spec_from_file_location('cuda_mcts_kernel',os.path.join(d,f[0]));"
            "k=importlib.util.module_from_spec(s);s.loader.exec_module(k);"
            "print(getattr(k,'COMP_FALLBACK_ORDER','none'))")
    out = subprocess.run([sys.executable, "-c", code, so_dir], capture_output=True, text=True)
    return out.stdout.strip().splitlines()[-1] if out.returncode == 0 and out.stdout.strip() else None


if os.path.basename(script) == "train_mcts_worker.py":
    # Drift agents use the v1 research kernel, whose composition fallback must
    # follow StatsCache.get_comp_wr (mid, high, low), the order every drift WP
    # was trained on. Build with HOTS_COMP_FALLBACK_MID_HIGH_LOW=1.
    if os.environ.get("HOTS_HERO_SET", "v1") != "v1":
        raise SystemExit("drift MCTS needs HOTS_HERO_SET=v1 (90-hero research kernel)")
    _so_dir = os.path.join(os.path.dirname(script), "cuda_mcts")
    _order = kernel_fallback_order(_so_dir)
    if _order != "mid_high_low":
        raise SystemExit(f"cuda_mcts kernel composition fallback is {_order}; rebuild with "
                         "HOTS_COMP_FALLBACK_MID_HIGH_LOW=1 (remote_workers/build_ext.sh)")
    # A save dir must not carry state from an old-order build: the worker would
    # resume from resume_state.pt or draft_policy_checkpoint.pt. The marker is
    # written only by runs that passed the check above.
    _save = os.environ.get("MCTS_SAVE_DIR")
    if _save:
        _marker = os.path.join(_save, "comp_fallback_order.txt")
        _state = [f for f in ("resume_state.pt", "draft_policy_checkpoint.pt", "draft_policy.pt", "DONE")
                  if os.path.exists(os.path.join(_save, f))]
        _prev = open(_marker).read().strip() if os.path.exists(_marker) else None
        if _state and _prev != _order:
            raise SystemExit(f"{_save} holds {_state} from a kernel with composition fallback "
                             f"{_prev or 'low_mid_high (unmarked)'}; archive the directory and start fresh")
        os.makedirs(_save, exist_ok=True)
        with open(_marker, "w") as _f:
            _f.write(_order + "\n")
    print(f"[v2] kernel composition fallback: {_order}", flush=True)
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
