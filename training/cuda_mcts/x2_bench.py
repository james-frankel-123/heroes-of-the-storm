"""
X2 benchmark: fixed (chance-node) search vs the pre-fix kernel, search only.

Setup (paper-1 revision benchmark protocol, overfit2026 harness conventions):
  1,000 draft configurations (seed 20260929, random map and side, mid tier),
  the 5 rerun2026 GD models cycled per batch of 100 (opponent and rollouts),
  c_puct 2.0, leaf value = revision leak-free enriched WP with its own deploy
  statistics, root argmax (T=0) unless stated. Prior = a fixed checkpoint
  (new:F_oof_s<k>, the 400-sim operating point) or the outcome-free
  behavioral prior (bc). Kernel: one build; mode legacy = search_mode 0
  (bit-identical to the pre-fix .so, see x2_record_legacy.py).

  gen    (GPU) write drafts + kernel WP + tree stats per run
  score  (CPU) score every generated run:
           proxy      kernel leaf WP of the final draft (training value fn)
           v2 judges  overfit2026.judges_v2: gN, gN_naive, RN, R17, QM2026,
                      QM2021 with the structural correction, consensus_v2
           RN_struct  gold.StructRealizedIndex on NODRIFT (cached index)
           degen, healer, diversity
  report  aggregate -> x2_results/bench_summary.json

Usage (from training/):
  CUDA_VISIBLE_DEVICES=<gpu> python3 cuda_mcts/x2_bench.py gen --grid main
  python3 cuda_mcts/x2_bench.py score
  python3 cuda_mcts/x2_bench.py report
"""
import os
import sys
import json
import time
import glob
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np

OUT = os.path.join(HERE, "x2_results", "bench")
N_DRAFTS = 1000
SEED = 20260929
BATCH = 100
SIMS = [100, 200, 400, 800, 1600]


def grid(name):
    g = []
    if name == "main":
        for prior in ("new:F_oof_s0", "new:F_oof_s1", "new:F_oof_s2", "bc"):
            for mode in ("legacy", "chance"):
                for s in SIMS:
                    if prior == "bc" and mode == "legacy" and s == 1600:
                        # the pre-fix kernel overflows its 4096-node tree here
                        # (illegal memory access); run the guarded copy instead
                        mode = "legacyg"
                    g.append((prior, mode, s, 0.0, 1.0))
        for prior in ("new:F_oof_s0", "new:F_oof_s1", "new:F_oof_s2"):
            for s in (200, 400, 800):
                g.append((prior, "rollfwd", s, 0.0, 1.0))
            g.append((prior, "chance", 400, 0.0, 0.0))      # no widening
            g.append((prior, "legacy", 400, 1.0, 1.0))      # sampled root
            g.append((prior, "chance", 400, 1.0, 1.0))
    elif name == "selfplay":
        # short self-play checkpoints (x2_selfplay.py), each under both searches
        sp = os.path.join(HERE, "x2_results", "selfplay")
        for run in ("F50k_legacy_s0", "F50k_chance_s0"):
            for mode in ("legacy", "chance"):
                for s in (200, 400):
                    g.append((f"path:{sp}/{run}/draft_policy.pt", mode, s, 0.0, 1.0))
    elif name == "smoke":
        g = [("new:F_oof_s0", "legacy", 100, 0.0, 1.0), ("new:F_oof_s0", "chance", 100, 0.0, 1.0)]
    return g


def run_name(prior, mode, sims, T, pw_k):
    if prior.startswith("path:"):   # .../selfplay/<run>/draft_policy.pt -> sp_<run>
        prior = "sp_" + os.path.basename(os.path.dirname(prior))
    s = f"{prior.replace(':', '_')}__{mode}__s{sims}__T{T:g}"
    if mode == "chance" and pw_k != 1.0:
        s += f"__pw{pw_k:g}"
    return s


def gen(a):
    import x2_common as C
    from shared import HEROES, MAPS
    k = C.load_kernel(a.so)
    kg = None   # overfit2026 guarded copy, for 'legacyg' cells
    os.makedirs(OUT, exist_ok=True)
    cfgs = C.configs(N_DRAFTS, SEED)
    engines = {}
    for prior, mode, sims, T, pw_k in grid(a.grid):
        path = os.path.join(OUT, run_name(prior, mode, sims, T, pw_k) + ".gen.json")
        if os.path.exists(path):
            continue
        t0 = time.time()
        drafts, stats = [], []
        for bi, bs in enumerate(range(0, N_DRAFTS, BATCH)):
            ca = np.ascontiguousarray(cfgs[bs:bs + BATCH])
            if mode == "legacyg":
                if kg is None:
                    kg = C.load_kernel(os.path.join(os.path.dirname(HERE), "overfit2026", "cuda_ofit",
                                                    [f for f in os.listdir(os.path.join(os.path.dirname(HERE), "overfit2026", "cuda_ofit"))
                                                     if f.startswith("ofit_kernel") and f.endswith(".so")][0]),
                                       name="ofit_kernel")
                pf, po = C.policy(prior)
                gf, go = C.gd_flat(bi % 5)
                wf, wo, lut = C.rev_leaf()
                eng = kg.OfitEngine(pf, gf, wf, po, go, wo, lut, max_concurrent=BATCH, device_id=0)
                res = eng.run_episodes(ca, sims, 2.0, SEED + bs, T, 0.3, 0.0, 1, search_mode=0)
                st = np.zeros((len(res), 20), np.int32)
                st[:, 2] = [r[4] for r in res]
                st[:, 4] = [r[5] for r in res]
                del eng
            else:
                if engines.get("prior") != prior:   # keep one prior's 5 engines
                    engines = {"prior": prior}
                if bi % 5 not in engines:
                    engines[bi % 5] = C.make_engine(k, prior, gd_idx=bi % 5, max_concurrent=BATCH)
                eng = engines[bi % 5]
                res = eng.run_episodes(ca, sims, 2.0, SEED + bs, T, 0.3, 0.0,
                                       search_mode=C.MODES[mode], pw_k=pw_k, pw_alpha=0.5)
                st = eng.last_stats()
            for j, r in enumerate(res):
                ts = np.asarray(r[2])
                mi, ti, ou = (int(x) for x in ca[j])
                our, opp = C.teams_from_terminal(ts, ou)
                drafts.append({"our": [HEROES[h] for h in our], "opp": [HEROES[h] for h in opp],
                               "map": MAPS[mi], "side": ou, "kernel_wp": float(r[0])})
            stats.append(st)
        st = np.concatenate(stats)
        sims_tot = max(int(st[:, 1].sum()), 1)
        summ = {"max_nodes": int(st[:, 2].max()), "mean_max_nodes": float(st[:, 2].mean()),
                "max_slots": int(st[:, 3].max()), "cap_hits": int(st[:, 4].sum()),
                "max_eager_nodes": int(st[:, 5].max()),
                "mean_eager_nodes": float(st[:, 5].mean()),
                "leaf_depth_steps": float(st[:, 6].sum() / sims_tot),
                "own_ahead": float(st[:, 7].sum() / sims_tot),
                "max_depth": int(st[:, 11].max()),
                "own_ahead_hist": (st[:, 12:20].sum(0) / sims_tot).round(4).tolist(),
                "policy_fwd_per_sim": float(st[:, 10].sum() / sims_tot),
                "gd_tree_per_sim": float(st[:, 9].sum() / sims_tot)} if mode not in ("legacy", "legacyg") else \
               {"max_nodes": int(st[:, 2].max()), "cap_hits": int(st[:, 4].sum())}
        rec = {"prior": prior, "mode": mode, "sims": sims, "temp": T, "pw_k": pw_k,
               "n": len(drafts), "secs": time.time() - t0, "tree": summ, "drafts": drafts}
        json.dump(rec, open(path, "w"))
        print(f"{os.path.basename(path)}: {rec['secs']:.0f}s wp={np.mean([d['kernel_wp'] for d in drafts]):.4f}",
              flush=True)


_RI = {}


def rn_struct():
    if "s" not in _RI:
        import pickle
        from drift2026 import common as dcommon
        dcommon._bind_statscache_methods()
        p = os.path.join(os.path.dirname(HERE), "overfit2026", "cache", "realized_NODRIFT_struct_s99.pkl")
        with open(p, "rb") as f:
            _RI["s"] = pickle.load(f)
    return _RI["s"]


def score_one(rec):
    from overfit2026 import judges_v2
    from shared import is_degenerate, HERO_ROLE_FINE
    rows = [(tuple(d["our"]), tuple(d["opp"]), d["map"], "mid") for d in rec["drafts"]]
    base = judges_v2.score_base(rows, nproc=int(os.environ.get("P1R_NPROC", "4")))
    v2 = judges_v2.correct(base, rows)
    out = {"proxy": np.array([d["kernel_wp"] for d in rec["drafts"]])}
    for j, v in base.items():
        out[f"{j}_v1"] = np.asarray(v)
    for j, v in v2.items():
        out[f"{j}_v2"] = np.asarray(v)
    ri = rn_struct()
    out["RN_struct"] = np.array([ri.score(o, p, t) for o, p, m, t in rows])
    healers = {h for h, r in HERO_ROLE_FINE.items() if r == "healer"}
    out["degen"] = np.array([float(is_degenerate(list(o))) for o, p, m, t in rows])
    out["healer"] = np.array([float(any(h in healers for h in o)) for o, p, m, t in rows])
    return out


def diversity(drafts):
    from collections import Counter
    c = Counter(h for d in drafts for h in d["our"])
    tot = sum(c.values())
    p = np.array(list(c.values())) / tot
    teams = Counter(tuple(sorted(d["our"])) for d in drafts)
    return {"distinct_heroes": len(c), "entropy_bits": float(-(p * np.log2(p)).sum()),
            "top10_share": float(sum(v for _, v in c.most_common(10)) / tot),
            "distinct_teams": len(teams)}


def score(a):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    while True:
        todo = [p for p in sorted(glob.glob(os.path.join(OUT, "*.gen.json")))
                if not os.path.exists(p.replace(".gen.json", ".score.json"))]
        for p in todo:
            rec = json.load(open(p))
            t0 = time.time()
            sc = score_one(rec)
            res = {k: v for k, v in rec.items() if k != "drafts"}
            res["means"] = {k: float(np.nanmean(v)) for k, v in sc.items()}
            res["per_draft"] = {k: np.round(v, 5).tolist() for k, v in sc.items()}
            res["diversity"] = diversity(rec["drafts"])
            json.dump(res, open(p.replace(".gen.json", ".score.json"), "w"))
            m = res["means"]
            print(f"{os.path.basename(p)[:-9]}: proxy={m['proxy']:.4f} cons_v2={m['consensus_v2']:.4f} "
                  f"gN_v2={m['gN_v2']:.4f} RN_struct={m['RN_struct']:.4f} degen={m['degen']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if not a.wait or (os.path.exists(os.path.join(OUT, "GEN_DONE")) and not todo):
            break
        time.sleep(30)


REFS = ["proxy", "consensus_v2", "gN_v2", "gN_naive_v2", "RN_v2", "R17_v2", "QM2026_v2",
        "QM2021_v2", "RN_struct", "consensus_v1", "degen", "healer"]


def report(a):
    files = sorted(glob.glob(os.path.join(OUT, "*.score.json")))
    runs = [json.load(open(p)) for p in files]
    for r in runs:   # guarded legacy (overflowing pre-fix cells) counts as legacy
        if r["mode"] == "legacyg":
            r["mode"], r["guarded"] = "legacy", True
    for r in runs:
        if "consensus_v1" not in r["per_draft"]:
            r["per_draft"]["consensus_v1"] = np.mean(
                [r["per_draft"][f"{j}_v1"] for j in ("gN", "gN_naive", "RN", "QM2026")], 0).tolist()
    by = {(r["prior"], r["mode"], r["sims"], r["temp"], r["pw_k"]): r for r in runs}
    cells = {}
    for (prior, mode, sims, T, pw), r in by.items():
        grp = "F_oof" if prior.startswith("new:F_oof") else prior
        if prior.startswith("path:"):
            grp = "sp_" + os.path.basename(os.path.dirname(prior))
        cells.setdefault((grp, mode, sims, T, pw), []).append(r)
    summary = {"cells": {}, "contrasts": {}, "sim_curve": {}}
    for key, rs in sorted(cells.items(), key=lambda kv: str(kv[0])):
        name = "|".join(str(x) for x in key)
        d = {"n_runs": len(rs), "priors": [r["prior"] for r in rs]}
        for ref in REFS:
            vals = np.concatenate([np.asarray(r["per_draft"][ref], float) for r in rs])
            d[ref] = float(np.nanmean(vals))
        d["diversity"] = {k: float(np.mean([r["diversity"][k] for r in rs])) for k in rs[0]["diversity"]}
        d["tree"] = rs[0]["tree"]
        d["secs_per_1000"] = float(np.mean([r["secs"] for r in rs]))
        summary["cells"][name] = d
    # paired contrasts (same drafts configs, same prior): chance - legacy at each sims
    for prior in sorted({k[0] for k in by}):
        grp = "F_oof" if prior.startswith("new:F_oof") else prior
        if prior.startswith("path:"):
            grp = "sp_" + os.path.basename(os.path.dirname(prior))
        for s in SIMS:
            for mode in ("chance", "rollfwd"):
                a_, b_ = by.get((prior, mode, s, 0.0, 1.0)), by.get((prior, "legacy", s, 0.0, 1.0))
                if a_ is None or b_ is None:
                    continue
                for ref in REFS:
                    diff = np.asarray(a_["per_draft"][ref]) - np.asarray(b_["per_draft"][ref])
                    summary["contrasts"].setdefault(f"{grp}|{mode}-legacy|s{s}", {}).setdefault(ref, []).append(diff)
    for k, v in summary["contrasts"].items():
        for ref, diffs in v.items():
            # seeds pooled; SE clustered by draft config (diffs are paired per config)
            D = np.mean(diffs, 0)
            v[ref] = [float(D.mean()), float(D.std(ddof=1) / np.sqrt(len(D))), len(diffs)]
    # sim curve contrasts: s vs 400 within mode (paired by config)
    for prior in sorted({k[0] for k in by}):
        grp = "F_oof" if prior.startswith("new:F_oof") else prior
        for mode in ("legacy", "chance"):
            base = by.get((prior, mode, 400, 0.0, 1.0))
            if base is None:
                continue
            for s in SIMS:
                r = by.get((prior, mode, s, 0.0, 1.0))
                if r is None:
                    continue
                for ref in REFS:
                    diff = np.asarray(r["per_draft"][ref]) - np.asarray(base["per_draft"][ref])
                    summary["sim_curve"].setdefault(f"{grp}|{mode}|s{s}-s400", {}).setdefault(ref, []).append(diff)
    for k, v in summary["sim_curve"].items():
        for ref, diffs in v.items():
            D = np.mean(diffs, 0)
            v[ref] = [float(D.mean()), float(D.std(ddof=1) / np.sqrt(len(D))), len(diffs)]
    json.dump(summary, open(os.path.join(HERE, "x2_results", "bench_summary.json"), "w"), indent=1)
    print(f"{'cell':42s} " + " ".join(f"{r[:9]:>9s}" for r in REFS[:9]) + "  degen  teams")
    for name, d in summary["cells"].items():
        print(f"{name:42s} " + " ".join(f"{d[r]:9.4f}" for r in REFS[:9])
              + f"  {d['degen']:.3f}  {d['diversity']['distinct_teams']:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "score", "report"])
    ap.add_argument("--grid", default="main")
    ap.add_argument("--so", default=None)
    ap.add_argument("--wait", action="store_true")
    a = ap.parse_args()
    if a.cmd == "gen":
        gen(a)
        if a.grid in ("main", "selfplay"):
            open(os.path.join(OUT, "GEN_DONE"), "w").write(time.strftime("%F %T"))
    elif a.cmd == "score":
        score(a)
    else:
        report(a)
