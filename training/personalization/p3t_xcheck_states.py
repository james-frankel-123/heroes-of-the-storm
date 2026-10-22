"""
P3 tournament cross-check (2026-10-21): recompute the drafter state (cur) and
the judge state (judge) for the players of the 300 tournament lobbies on
another machine, and compare them with the state files the 3090 produced
(p3t_data.py states). The 3090 showed single-bit RAM flips, so this checks
the inputs of every drafter and of the judge's personal term.
Run (any host with the personalization caches, run_rcf_full env):
  python personalization/p3t_xcheck_states.py
Output: $P3_RESULTS/p3t/xcheck_states.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("P1_TIERS", "site")
import numpy as np

import p3_hs_core as C
import p3t_data as TD


def main():
    par, d, aug, mets, mw = TD.load_model()
    jpar = TD._jpar("full_ms_flip")
    sh = TD.to_shared(d["hero_names"])
    inv_sh = TD.sh_inv(sh)
    L = TD.load_lobbies()
    lay = TD.Layer(d, aug, TD.layer_meta())
    gi = np.load(os.path.join(TD.OUT, "tourney", "lobbies.npz"))["gi"]
    out = {"lobbies": int(len(gi))}
    for v in ("cur", "judge"):
        z = np.load(os.path.join(TD.OUT, f"state_{v}.npz"))
        pos = {int(g): i for i, g in enumerate(z["gsel"])}
        sel = np.array([pos[int(g)] for g in gi])
        rows_state = (sel[:, None] * 10 + np.arange(10)[None]).ravel()
        rows = L["slot_rows"][gi].ravel()
        qpid = d["pid"][rows].astype(np.int64)
        qday = z["qday"][rows_state].astype(np.int64)
        ukey, inv = np.unique(qpid * 100000 + qday, return_inverse=True)
        upid, uday = ukey // 100000, ukey % 100000
        res = TD.walk(aug, jpar if v == "judge" else par, mets, mw, upid, uday, ev_lo=TD.J0 if v == "judge" else None)
        s = TD.offset(aug["table"], res["n_slot"]) + res["mean_all"]
        if v != "judge":
            s = s + lay.at(upid, uday, res["n_all"])
        dt = np.float32 if v != "cur" else np.float16
        new = {"s": s[inv][:, inv_sh].astype(dt), "sd": np.sqrt(res["var_all"])[inv][:, inv_sh].astype(np.float16),
               "n": res["n_all"][inv][:, inv_sh].astype(np.uint16),
               "off": TD.off_role(d, res["n_all"])[inv][:, inv_sh]}
        r = {}
        for k, a in new.items():
            b = z[k][rows_state]
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            r[k] = {"identical": bool(np.array_equal(a, b)), "cells_differing": int((diff > 0).sum()),
                    "max_abs_diff": float(diff.max()), "cells": int(diff.size)}
        out[v] = r
        print(v, json.dumps(r), flush=True)
    ok = all(x["max_abs_diff"] <= (1e-2 if k in ("s", "sd") else 0) for v in ("cur", "judge") for k, x in out[v].items())
    out["pass"] = bool(ok)
    os.makedirs(os.path.join(C.RESULTS, "p3t"), exist_ok=True)
    json.dump(out, open(os.path.join(C.RESULTS, "p3t", "xcheck_states.json"), "w"), indent=1)
    print("XCHECK STATES", "OK" if ok else "DIFF")


if __name__ == "__main__":
    main()
