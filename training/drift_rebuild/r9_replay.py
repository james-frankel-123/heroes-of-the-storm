"""
R9 — extensions of the W2c deployment replay (drift2026/summarize_w2.py,
C0 = position 8, 36 simulated builds, games-weighted accuracy, regret vs
retrain-every-build), answering audit A3/A4:

  * retrain every K builds WITHOUT refreshing statistics between retrains
    (model trained at p keeps cumulative stats frozen at p until the next
    retrain), K = 3, 6 — the arm the paper lacked;
  * blind refresh schedules on the fixed C0 model (every 3rd build; 14
    evenly spaced refreshes), for comparison with the detector's 14;
  * the leak-free frozen-recipe row: r2_c0frozen_oof_s{42,123,777}
    (trained at C0 on out-of-fold features, deployed with C0 stats).

Accuracy of model m on build n under stats s = cumulative through build s is
computed from raw replays with the paper's extractor; the pipeline is
checked against the stored matrix (s = n-1 reproduces acc[m][n]).

Usage: python3 drift_rebuild/r9_replay.py
Output: drift_rebuild/results/r9_replay.json, R9_REPLAY.md
"""
import os
import sys
import json
import time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_common as rb  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from drift2026 import common  # noqa: E402

import numpy as np  # noqa: E402

C0 = 8
R2 = os.path.join(common.RESULTS_DIR)
_STATS = {}
_ROWS = {}


def builds_255():
    return [b for b in common.load_patch_index()["builds"] if b.startswith("2.55")]


def jload(*p):
    return json.load(open(os.path.join(R2, *p)))


def feat_task(args):
    from sweep_enriched_wp import extract_features, _swap_features, FEATURE_GROUPS
    n, s, lo, hi = args
    st = _STATS[s]
    mask = [True] * len(FEATURE_GROUPS)
    B, E, Y = [], [], []
    for d in _ROWS[n][lo:hi]:
        b, e = extract_features(d, st, mask)
        y = float(d["winner"] == 0)
        bs, es = _swap_features(b, e)
        B += [b, bs]
        E += [e, es]
        Y += [y, 1 - y]
    return (n, s, np.asarray(B, np.float32), np.asarray(E, np.float32),
            np.asarray(Y, np.float32))


def main():
    t0 = time.time()
    common.setup()
    builds = builds_255()
    last = len(builds) - 1
    cutoff_pos = builds.index(common.TRAIN_CUTOFF_BUILD)

    # stored matrix (refresh every build)
    acc, games = {}, {}
    for p in range(C0, last):
        src = (("d2", "d2c_cumprev_s42.json") if p == cutoff_pos
               else ("w2c", f"w2c_cut{p:02d}_s42.json"))
        r = jload(*src)
        acc[p] = {}
        for b, d in r["test_acc_per_build"].items():
            n = builds.index(b)
            acc[p][n] = d["acc"]
            games[n] = d["n_rows"] // 2
    Ns = sorted(n for n in games if n > C0)
    w = np.array([games[n] for n in Ns], float)

    def wavg(curve):
        return float(np.sum(w * np.array([curve[n] for n in Ns])) / w.sum())
    oracle = wavg({n: acc[n - 1][n] for n in Ns})

    # needed (model p, build n, stats s) cells
    need = set()
    for K in (3, 6):
        pts = list(range(C0, last, K))
        for n in Ns:
            p = max(q for q in pts if q < n)
            need.add((p, n, p))
    refresh_scheds = {"blind_every3": list(range(C0 + 3, last + 1, 3))}
    even = np.linspace(C0 + 1, last, 16)[1:-1]
    refresh_scheds["blind_even14"] = sorted({int(round(x)) for x in even})
    for name, R in refresh_scheds.items():
        for n in Ns:
            s = max([C0] + [r for r in R if r <= n - 1])
            need.add((C0, n, s))
    for n in Ns:
        need.add((C0, n, C0))                 # never refresh (check vs 1.063)
    need.add((C0, C0 + 3, C0 + 2))            # check cells: s = n-1
    need.add((20, 25, 24))

    # data
    rows, full_builds = common.load_data_with_patches()
    keys = ("team0_heroes", "team1_heroes", "game_map", "skill_tier", "winner")
    pos_of_bidx = {full_builds.index(b): i for i, b in enumerate(builds)}
    want = {n for _, n, _ in need}
    for r in rows:
        n = pos_of_bidx.get(r["build_idx"])
        if n in want:
            _ROWS.setdefault(n, []).append({k: r[k] for k in keys})
    del rows
    for s in {s for _, _, s in need}:
        _STATS[s] = common.load_patch_stats("cumulative", builds[s])
    ns_pairs = sorted({(n, s) for _, n, s in need})
    tasks = []
    for n, s in ns_pairs:
        L = len(_ROWS[n])
        for lo in range(0, L, 4000):
            tasks.append((n, s, lo, min(L, lo + 4000)))
    print(f"{len(need)} cells, {len(ns_pairs)} (build, stats) pairs, "
          f"{len(tasks)} chunks ({time.time()-t0:.0f}s)", flush=True)
    parts = {}
    with mp.get_context("fork").Pool(48) as pool:
        for n, s, B, E, Y in pool.imap(feat_task, tasks, chunksize=2):
            parts.setdefault((n, s), []).append((B, E, Y))
    print(f"features done ({time.time()-t0:.0f}s)", flush=True)

    import torch
    from sweep_enriched_wp import WinProbEnrichedModel
    from drift2026.train_drift_wp import enriched_cols
    cols = enriched_cols()
    torch.set_num_threads(16)
    models = {}

    def model(p):
        if p not in models:
            name = ("d2c_cumprev_s42" if p == cutoff_pos else f"w2c_cut{p:02d}_s42")
            ck = torch.load(os.path.join(common.MODELS_DIR, f"{name}.pt"),
                            map_location="cpu", weights_only=True)
            m = WinProbEnrichedModel(ck["input_dim"], list(ck["arch"]), ck["dropout"])
            m.load_state_dict(ck["state_dict"])
            m.eval()
            models[p] = m
        return models[p]

    cell = {}
    for (p, n, s) in sorted(need):
        B = np.concatenate([x[0] for x in parts[(n, s)]])
        E = np.concatenate([x[1] for x in parts[(n, s)]])
        Y = np.concatenate([x[2] for x in parts[(n, s)]])
        X = torch.from_numpy(np.concatenate([B, E[:, cols]], 1))
        with torch.no_grad():
            pr = model(p)(X).numpy().ravel()
        cell[(p, n, s)] = float(((pr > 0.5) == (Y > 0.5)).mean() * 100)

    checks = {f"{p}->{n}@{s}": (round(cell[(p, n, s)], 3), acc[p][n])
              for (p, n, s) in [(C0, C0 + 3, C0 + 2), (20, 25, 24)]}
    print("check vs stored matrix (mine, stored):", checks, flush=True)

    table = {"oracle (retrain every build)": (oracle, 35, 35)}
    table["never retrain, never refresh (same C0 model)"] = (
        wavg({n: cell[(C0, n, C0)] for n in Ns}), 0, 0)
    table["never retrain, refresh every build"] = (
        wavg({n: acc[C0][n] for n in Ns}), 0, 35)
    for name, R in refresh_scheds.items():
        cur = {n: cell[(C0, n, max([C0] + [r for r in R if r <= n - 1]))] for n in Ns}
        nref = len([r for r in R if r <= Ns[-1] - 1])
        table[f"never retrain, {name} refresh"] = (wavg(cur), 0, nref)
    for K in (3, 6):
        pts = list(range(C0, last, K))
        cur_ref, cur_noref = {}, {}
        for n in Ns:
            p = max(q for q in pts if q < n)
            cur_ref[n] = acc[p][n]
            cur_noref[n] = cell[(p, n, p)]
        nret = len([p for p in pts if C0 < p < Ns[-1]])
        table[f"retrain every {K}, refresh every build"] = (wavg(cur_ref), nret, 35)
        table[f"retrain every {K}, no refresh between retrains"] = (
            wavg(cur_noref), nret, nret)
    # leak-free frozen recipe (trained at C0 on OOF features, C0 stats)
    for s in common.SEEDS:
        pth = os.path.join(rb.RESULTS_DIR, "d2", f"r2_c0frozen_oof_s{s}.json")
        if os.path.exists(pth):
            r = json.load(open(pth))
            cur = {builds.index(b): d["acc"] for b, d in r["test_acc_per_build"].items()}
            table[f"frozen recipe, leak-free (OOF) s{s}"] = (wavg(cur), 0, 0)
    fr = jload("w2c", "w2c_frozenstats_cut08_s42.json")
    table["frozen recipe, leaky (paper) s42"] = (
        wavg({builds.index(b): d["acc"] for b, d in fr["test_acc_per_build"].items()}), 0, 0)

    out = {"checks": checks, "oracle_acc": oracle,
           "rows": {k: {"weighted_acc": round(a, 3), "regret_pp": round(oracle - a, 3),
                        "retrains": rt, "refreshes": rf}
                    for k, (a, rt, rf) in table.items()},
           "schedules": refresh_scheds}
    common.write_json(os.path.join(rb.RESULTS_DIR, "r9_replay.json"), out)
    lines = ["# R9 deployment replay extensions", "",
             f"Oracle weighted acc {oracle:.3f}. Checks: {checks}", "",
             "| policy | retrains | refreshes | weighted acc | regret (pp) |",
             "|---|---|---|---|---|"]
    for k, v in out["rows"].items():
        lines.append(f"| {k} | {v['retrains']} | {v['refreshes']} | "
                     f"{v['weighted_acc']:.3f} | {v['regret_pp']:+.3f} |")
    open(os.path.join(rb.RESULTS_DIR, "R9_REPLAY.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
