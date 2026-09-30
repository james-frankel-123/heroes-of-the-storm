"""
Collect every revision result into results/SUMMARY.json and print the numbers
the manuscript and REVISION_NOTES quote.

Sections: table1 (WP models), sweep (fractional-factorial main effects),
metric_validity, ngs, mcts (Table VI under references + contrasts),
tournament, crosseval, greedy_rich, wr_sweep.
"""
import os
import sys
import json
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from paper1_revision import core

import numpy as np

R = core.RESULTS


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def table1():
    out = {}
    for p in sorted(glob.glob(os.path.join(core.MODEL_DIR, "*.json"))):
        n = os.path.basename(p)[:-5]
        if n.startswith("sw_"):
            continue
        d = json.load(open(p))
        if "test_mean" not in d:
            continue
        out[n] = {"dim": d["input_dim"], "selected_seed": d["selected_seed"],
                  "test": d["test_mean"], "test_sd": d["test_sd"],
                  "NODRIFT": d["NODRIFT_mean"], "NODRIFT_sd": d["NODRIFT_sd"],
                  "n_train_rows": d["n_train_rows"]}
    return out


def sweep():
    from paper1_revision.train_wp import SWEPT
    runs = []
    for p in glob.glob(os.path.join(core.MODEL_DIR, "sw_*.json")):
        d = json.load(open(p))
        mask = [c == "1" for c in os.path.basename(p)[3:-5]]
        s = d["seeds"]["42"]
        runs.append((mask, s["test"]["acc"], s["NODRIFT"]["acc"], s["NODRIFT"]["ll"],
                     s["NODRIFT"]["slope"]))
    if not runs:
        return None
    M = np.array([r[0] for r in runs])
    Y = {"test_acc": np.array([r[1] for r in runs]), "nodrift_acc": np.array([r[2] for r in runs]),
         "nodrift_ll": np.array([r[3] for r in runs]), "nodrift_slope": np.array([r[4] for r in runs])}
    eff = {}
    for i, g in enumerate(SWEPT):
        eff[g] = {k: float(v[M[:, i]].mean() - v[~M[:, i]].mean()) for k, v in Y.items()}
    # residual SE of a main effect from the saturated-minus-mains residual
    se = {}
    X = np.column_stack([np.ones(len(M))] + [np.where(M[:, i], 1.0, -1.0) for i in range(M.shape[1])])
    for k, v in Y.items():
        b, res, *_ = np.linalg.lstsq(X, v, rcond=None)
        r = v - X @ b
        s2 = (r @ r) / (len(v) - X.shape[1])
        se[k] = float(2 * np.sqrt(s2 / len(v)))     # effect = 2*coef
    full = [r for r in runs if all(r[0])]
    none = [r for r in runs if not any(r[0])]
    return {"n_runs": len(runs), "main_effects": eff, "effect_se": se,
            "acc_range_test": [float(Y["test_acc"].min()), float(Y["test_acc"].max())],
            "acc_range_nodrift": [float(Y["nodrift_acc"].min()), float(Y["nodrift_acc"].max())]}


def mcts():
    B = os.path.join(R, "mcts_bench")
    rows = {}
    for p in glob.glob(os.path.join(B, "*.json")):
        d = json.load(open(p))
        run = d["run"]
        kind, name = run.split(":", 1)
        cfg = name.rsplit("_s", 1)[0]
        key = (f"{'new' if kind == 'new' else 'old'}:{cfg}", d["temp"])
        rows.setdefault(key, []).append(d)
    refs = ["proxy_sub", "proxy_rev", "naive_rev", "gN", "gN_naive", "RN", "R17", "QM2026", "QM2021",
            "degen", "healer", "syn_hp", "ctr_hp", "syn_own", "ctr_own"]
    out = {}
    per_seed = {}
    for (cfg, T), ds in rows.items():
        ds = sorted(ds, key=lambda d: d["run"])
        k = f"{cfg}|T{T:g}"
        per_seed[k] = {r: np.array([d["means"][r] for d in ds]) for r in refs}
        per_seed[k]["_draft"] = {r: np.array([d["per_draft"][r] for d in ds]) for r in refs}
        out[k] = {"n_seeds": len(ds),
                  "seeds": [d["run"] for d in ds],
                  "mean": {r: float(per_seed[k][r].mean()) for r in refs},
                  "se": {r: float(per_seed[k][r].std(ddof=1) / np.sqrt(len(ds))) if len(ds) > 1 else None
                         for r in refs},
                  "distinct": float(np.mean([d["diversity"]["distinct"] for d in ds])),
                  "entropy": float(np.mean([d["diversity"]["entropy_bits"] for d in ds])),
                  "cap_hits": int(sum(d["cap_hits"] for d in ds))}
    contrasts = {}
    pairs = [("new:F_oof", "new:B_oof"), ("new:J_oof", "new:F_oof"), ("new:J_oof", "new:B_oof"),
             ("old:F_400sim", "old:B_fullwp"), ("old:J_800sim", "old:F_400sim"),
             ("old:J_800sim", "old:B_fullwp"), ("old:B_fullwp", "old:K_truebase"),
             ("new:B_oof", "old:K_truebase"), ("new:F_oof", "old:K_truebase"),
             ("new:J_oof", "old:K_truebase"),
             ("new:B_oof", "old:B_fullwp"), ("new:F_oof", "old:F_400sim"),
             ("new:J_oof", "old:J_800sim"), ("old:E_1M", "old:B_fullwp"),
             ("new:F_oof", "old:E_1M"), ("new:J_oof", "old:E_1M")]
    for T in (1.0, 0.0):
        for a, b in pairs:
            ka, kb = f"{a}|T{T:g}", f"{b}|T{T:g}"
            if ka not in per_seed or kb not in per_seed:
                continue
            c = {}
            for r in refs:
                x, y = per_seed[ka][r], per_seed[kb][r]
                se = float(np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)))
                c[r] = {"diff": float(x.mean() - y.mean()), "se": se,
                        "z": float((x.mean() - y.mean()) / se) if se > 0 else None}
            contrasts[f"{a} - {b}|T{T:g}"] = c
    # argmax vs T=1 within config (paired by seed)
    for cfg in {k.split("|")[0] for k in per_seed}:
        k1, k0 = f"{cfg}|T1", f"{cfg}|T0"
        if k1 in per_seed and k0 in per_seed and len(per_seed[k1]["gN"]) == len(per_seed[k0]["gN"]):
            c = {}
            for r in refs:
                d = per_seed[k0][r] - per_seed[k1][r]
                c[r] = {"diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}
            contrasts[f"{cfg}: T0 - T1"] = c
    return {"configs": out, "contrasts": contrasts}


def main():
    S = {"table1": table1(), "submitted": jload(os.path.join(R, "submitted_models.json")),
         "sweep": sweep(), "metric_validity": jload(os.path.join(R, "metric_validity.json")),
         "ngs": jload(os.path.join(R, "ngs.json")), "mcts": mcts(),
         "tournament": jload(os.path.join(R, "tournament_standings.json")),
         "crosseval": jload(os.path.join(R, "crosseval.json")),
         "greedy_rich": jload(os.path.join(R, "greedy_rich.json")),
         "wr_sweep": jload(os.path.join(R, "wr_sweep.json"))}
    if S["tournament"]:
        S["tournament"] = {k: v for k, v in S["tournament"].items() if k != "pairs"}
    json.dump(S, open(os.path.join(R, "SUMMARY.json"), "w"), indent=1)
    t1 = S["table1"]
    print("TABLE I (3-seed means)")
    for n, d in t1.items():
        print(f"  {n:16s} dim={d['dim']} test acc={100 * d['test']['acc']:.2f}±{100 * d['test_sd']['acc']:.2f} "
              f"ll={d['test']['ll']:.4f} slope={d['test']['slope']:.2f} | NODRIFT acc="
              f"{100 * d['NODRIFT']['acc']:.2f}±{100 * d['NODRIFT_sd']['acc']:.2f} ll={d['NODRIFT']['ll']:.4f} "
              f"slope={d['NODRIFT']['slope']:.2f}")
    if S["sweep"]:
        print("SWEEP main effects (pp)", S["sweep"]["n_runs"], "runs; SE", S["sweep"]["effect_se"])
        for g, e in sorted(S["sweep"]["main_effects"].items(), key=lambda kv: -kv[1]["test_acc"]):
            print(f"  {g:20s} test {100 * e['test_acc']:+.3f}  nodrift {100 * e['nodrift_acc']:+.3f} "
                  f"ll {e['nodrift_ll']:+.5f}")
    m = S["mcts"]
    for k, v in sorted(m["configs"].items()):
        mu = v["mean"]
        print(f"  {k:24s} n={v['n_seeds']} sub={mu['proxy_sub']:.3f} rev={mu['proxy_rev']:.3f} "
              f"gN={mu['gN']:.3f} gNn={mu['gN_naive']:.3f} RN={mu['RN']:.3f} R17={mu['R17']:.3f} "
              f"QM={mu['QM2026']:.3f} deg={100 * mu['degen']:.1f} hlr={100 * mu['healer']:.1f} "
              f"syn={mu['syn_hp']:.2f} dist={v['distinct']:.0f} caps={v['cap_hits']}")
    for k, c in m["contrasts"].items():
        print(f"  {k:40s} " + " ".join(f"{r}={c[r]['diff']:+.4f}({c[r]['se']:.4f})"
                                        for r in ("proxy_rev", "proxy_sub", "gN", "RN", "QM2026", "degen")))


if __name__ == "__main__":
    main()
