"""Run any drift script under the v2 environment:
    nice -n 19 taskset -c 48-63 python3 drift_rebuild/v2/run.py <script.py> [args...]
(cwd = training/). See v2env.py."""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2env  # noqa: E402,F401

script = os.path.abspath(sys.argv[1])
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
