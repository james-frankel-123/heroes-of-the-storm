"""
Overlap check between the lobby sets that fit drafter components (opponent-model
weights, probe, player-prior targets in pprior_targets.npz, any other *_lobbies.npz)
and the evaluation sets (tournament, tune, party study), by replay ID.

Usage: python3 p3t_overlap_check.py <cache dir>/p3t   (numpy only; read-only)
"""
import os
import sys
import glob
import json
import numpy as np


def main(d):
    L = np.load(os.path.join(d, "lobbies.npz"))
    rid = L["replay_id"]
    ev = {}
    for sub in ("tourney", "tourney_tune", "tourney_party"):
        p = os.path.join(d, sub, "lobbies.npz")
        if os.path.exists(p):
            ev[sub] = set(rid[np.load(p)["gi"]].tolist())
    fit = {}
    for p in sorted(glob.glob(os.path.join(d, "*lobbies*.npz")) + glob.glob(os.path.join(d, "pprior_targets.npz"))):
        name = os.path.basename(p)
        if name == "lobbies.npz":
            continue
        z = np.load(p)
        if "gi" in z.files:
            fit[name] = set(rid[z["gi"]].tolist())
    out = {"cache": d, "eval_sizes": {k: len(v) for k, v in ev.items()},
           "fit_sizes": {k: len(v) for k, v in fit.items()},
           "overlap": {f"{a} & {b}": len(fit[a] & ev[b]) for a in fit for b in ev},
           "eval_pairwise": {f"{a} & {b}": len(ev[a] & ev[b]) for a in ev for b in ev if a < b}}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
