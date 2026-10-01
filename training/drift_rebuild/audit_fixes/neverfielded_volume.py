"""Correlation of each sizable build's never-fielded composition count with
log(build games), from w3_neverfielded.json and the per-build counts.
Output: <results>/neverfielded_volume.json"""
import gzip
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from drift2026 import common  # noqa: E402

n = json.load(open(os.path.join(common.RESULTS_DIR, "w3_neverfielded.json")))["per_build_never_size"]
d = pickle.load(gzip.open(common.COUNTS_PKL))
games = {d["builds"][b]: sum(c["games"] for c in t.values()) for b, t in d["per_build"].items()}
bs = [b for b in n if b in games]
x, y = np.log([games[b] for b in bs]), np.array([n[b] for b in bs])
out = {"n_builds": len(bs), "r_never_vs_log_games": round(float(np.corrcoef(x, y)[0, 1]), 3)}
print(out)
json.dump(out, open(os.path.join(common.RESULTS_DIR, "neverfielded_volume.json"), "w"))
