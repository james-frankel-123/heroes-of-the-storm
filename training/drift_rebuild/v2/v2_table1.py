"""v2 Table I: leak-free and leaky regimes (3 seeds each) plus the maintained
model; val / future accuracy, future calibration slope (r3_eval_vf.json),
greedy degeneracy (r4_w2b, cutoff-era opponent pool). Writes
<results>/r12_table1.json (same shape gen_figs.fig_scatter reads) and the
Welch tests of Secs. 2-3 to <results>/a2_welch.json."""
import json
import os
import sys

import v2env  # noqa: F401
import numpy as np
from drift2026 import common

sys.path.insert(0, os.path.join(v2env.TRAINING, "drift_rebuild"))
from r8_score import t_sf  # noqa: E402

R = common.RESULTS_DIR
S = [42, 123, 777]
REGS = ["allhist", "win3", "win6", "win12", "decay90", "decay365", "embed"]
r3 = json.load(open(os.path.join(R, "r3_eval_vf.json")))


def ms(v):
    return [float(np.mean(v)), float(np.std(v, ddof=1)), len(v)]


def w2b(tag, s):
    p = os.path.join(R, "r4_w2b", f"{tag}_s{s}_gdcutoff.json")
    return json.load(open(p))["degen_rate"] if os.path.exists(p) else None


def welch(a, b):
    a, b = np.array(a, float), np.array(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return {"t": round(float(t), 2), "df": round(float(df), 2), "p": round(2 * t_sf(abs(t), df), 4)}


rows = {}
for reg in REGS + ["cumprev"]:
    row = {}
    for tag in ("leaky", "oof"):
        if reg == "cumprev":
            stem, w, c = "d2c_cumprev", "d2_d2c_cumprev", "maintained/cumprev"
        elif tag == "leaky":
            stem, w, c = f"d2b_{reg}", f"d2_d2b_{reg}", f"leaky/{reg}"
        else:
            stem, w, c = f"r2_{reg}_oof", f"r2_{reg}_oof", f"oof/{reg}"
        tr = [json.load(open(os.path.join(R, "d2", f"{stem}_s{s}.json"))) for s in S]
        dg = [x for x in (w2b(w, s) for s in S) if x is not None]
        row[tag] = {"val": ms([t["val_acc"] for t in tr]), "future": ms([t["test_acc_future"] for t in tr]),
                    "degen": ms(dg) if len(dg) > 1 else [float("nan"), 0, len(dg)], "degen_seeds": dg,
                    "future_slope": ms([r3[f"{c}/s{s}"]["future"]["calib_slope"] for s in S])}
    rows[reg] = row
out = {"rows": rows}
for tag in ("leaky", "oof"):
    x = np.array([rows[r][tag]["future"][0] for r in REGS])
    y = np.array([rows[r][tag]["degen"][0] for r in REGS])
    if np.isfinite(y).all():
        out[f"corr_{tag}"] = float(np.corrcoef(x, y)[0, 1])
json.dump(out, open(os.path.join(R, "r12_table1.json"), "w"), indent=1)
M = rows["cumprev"]["oof"]["degen_seeds"]
tests = {}
for name, a in (("allhist_oof_vs_maint", rows["allhist"]["oof"]["degen_seeds"]),
                ("embed_oof_vs_maint", rows["embed"]["oof"]["degen_seeds"]),
                ("allhist_leaky_vs_oof", rows["allhist"]["leaky"]["degen_seeds"])):
    b = rows["allhist"]["oof"]["degen_seeds"] if name.endswith("vs_oof") else M
    if len(a) > 1 and len(b) > 1:
        tests[name] = welch(a, b)
for c in ("q7_decayed90", "q7_decayed90k100"):
    a = [x for x in (w2b(f"d2_{c}", s) for s in S) if x is not None]
    if len(a) > 1 and len(M) > 1:
        tests[f"{c}_vs_maint"] = {**welch(a, M), "mean": float(np.mean(a))}
json.dump(tests, open(os.path.join(R, "a2_welch.json"), "w"), indent=1)
for reg in REGS + ["cumprev"]:
    o, l = rows[reg]["oof"], rows[reg]["leaky"]
    print(f"{reg:9s} OOF val {o['val'][0]:.2f} fut {o['future'][0]:.2f} slope {o['future_slope'][0]:.2f} "
          f"degen {o['degen'][0]:.1f}±{o['degen'][1]:.1f} | leaky val {l['val'][0]:.2f} fut {l['future'][0]:.2f} degen {l['degen'][0]:.1f}")
print({k: v for k, v in out.items() if k.startswith("corr")})
print(json.dumps(tests))
