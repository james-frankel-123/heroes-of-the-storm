"""
Run a P3 tournament stage and record every input file it opens under the
cache and results trees (for the rerun's input manifest).
Usage (from training/): python personalization/p3t_trace.py <log.jsonl> personalization/<stage.py> [args...]
Appends one JSON line per run: {"stage": ..., "args": ..., "read": [...], "written": [...]}.
"""
import json
import os
import runpy
import sys

LOG, SCRIPT, ARGS = sys.argv[1], sys.argv[2], sys.argv[3:]
ROOTS = [os.path.realpath(os.environ[k]) for k in ("P3_CACHE", "P3_RESULTS") if os.environ.get(k)]
READ, WRITTEN = set(), set()


def hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = os.path.realpath(os.fsdecode(args[0]))
    if not any(p.startswith(r) for r in ROOTS):
        return
    mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
    (WRITTEN if any(c in mode for c in "wax+") else READ).add(p)


sys.addaudithook(hook)
sys.argv = [SCRIPT] + ARGS
try:
    runpy.run_path(SCRIPT, run_name="__main__")
finally:
    with open(LOG, "a") as f:
        f.write(json.dumps({"stage": os.path.basename(SCRIPT), "args": ARGS, "read": sorted(READ - WRITTEN),
                            "written": sorted(WRITTEN)}) + "\n")
