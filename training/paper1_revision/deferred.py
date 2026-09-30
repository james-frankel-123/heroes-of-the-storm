"""
Deferred paper-1 revision jobs, queued 2026-09-30 under the owner's resource
cap (run by capped_queue2.sh after the 800-sim J_oof seeds). Each job writes
results/deferred/<job>.json (plus its own cache/models), so it can be folded
into the supplement later. Every job is resumable: finished pieces are skipped.

Jobs (queue order = value to the revision):
  scope        27-config augmentation scope sweep (WR {10,20,30} x volume
               {50,100,200} x scope {unseen, sparse, both}), leak-free: own
               composition table, out-of-fold features, 256x128, seed 42, the
               submission's v1 generator. 5-tank WP + 28-test sanity + held-out
               accuracy/slope per config.                                  [GPU]
  ngs_quartiles  NGS off-meta quartile breakdown for leak-free models.     [CPU]
  rebench_ag   A_partial and G_base (submitted step-conditioned value
               functions), seeds 0-4, via bench_mcts.py (T=1 and T=0).     [GPU]
  concentration  draft-concentration diagnostics for the leak-free F_oof
               agent (and the submitted J_800sim for comparison): per-seed
               diversity, favorite heroes, pair/trio edges in-sample (train
               split) and forward (post-snapshot games), role-composition
               coverage in the own table, exact-team occurrence.           [CPU]
  ensemble     20-member specification-diverse leak-free WP ensemble
               (arch x data half x feature subset), per-draft variance on
               human, agent, tournament, and degenerate draft sets.        [GPU]
  cql_build / cql_train_a2.0 / cql_train_a0.5 / cql_eval
               CQL with enriched features, retrained leak-free (out-of-fold
               transition features; early stopping on the validation split),
               then rich evaluation (5 x 1000 drafts) with deploy statistics
               and independent-reference scores.                     [CPU/GPU]

Usage: python3 paper1_revision/deferred.py <job>
"""
import os
import sys
import json
import time
import random

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
from paper1_revision import core

import numpy as np

OUT = os.path.join(core.RESULTS, "deferred")
os.makedirs(OUT, exist_ok=True)
NPROC = int(os.environ.get("P1R_NPROC", "4"))


def save(name, obj):
    json.dump(obj, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1, default=float)
    print(f"wrote {name}.json", flush=True)


# ── scope sweep ─────────────────────────────────────────────────────────

def job_scope():
    import torch
    from experiment_synthetic_augmentation import (generate_synthetic_data, evaluate_config,
                                                   ENRICHED_GROUPS)
    from paper1_revision import train_wp
    dev = torch.device("cuda")
    cols = core.group_cols(ENRICHED_GROUPS)
    part = os.path.join(OUT, "scope")
    os.makedirs(part, exist_ok=True)
    sp = core.split_games()
    real = [{"team0_heroes": list(g[3]), "team1_heroes": list(g[4])} for g in sp["train"]]
    comp_raw = json.load(open(core.comp_path("deploy")))
    tX, tY = train_wp._tensors(cols, "train_oof", dev)
    vX, vY = train_wp._tensors(cols, "val", dev)
    configs = [(wr, vol, sc) for wr in (10, 20, 30) for vol in (50, 100, 200)
               for sc in ("tier2_only", "both", "tier1_only")]
    for wr, vol, sc in configs:
        name = f"wr{wr}_vol{vol}_{sc}"
        path = os.path.join(part, f"{name}.json")
        if os.path.exists(path):
            continue
        t0 = time.time()
        random.seed(42)
        np.random.seed(42)
        syn, st = generate_synthetic_data(real, comp_raw, unseen_wr=float(wr),
                                          unseen_volume=vol, scope=sc)
        Xf = np.zeros((len(syn), 357), np.float32)
        Xs = np.zeros_like(Xf)
        idx = np.arange(len(syn)) % core.N_FOLDS
        for k in range(core.N_FOLDS):
            ii = np.where(idx == k)[0]
            rows = [(syn[i]["team0_heroes"], syn[i]["team1_heroes"], syn[i]["game_map"],
                     syn[i]["skill_tier"]) for i in ii]
            if len(rows):
                a, b = core.featurize(rows, core.load_stats(f"oof{k}"), nproc=NPROC)
                Xf[ii], Xs[ii] = a, b
        y = np.array([1.0 if r["winner"] == 0 else 0.0 for r in syn], np.float32)
        sX = torch.tensor(np.concatenate([Xf[:, cols], Xs[:, cols]]), device=dev)
        sY = torch.tensor(np.concatenate([y, 1 - y]), device=dev)
        spec = {"groups": ENRICHED_GROUPS, "arch": [256, 128]}
        model, vl, hist = train_wp.train_one(name, spec, 42, dev,
                                             (torch.cat([tX, sX]), torch.cat([tY, sY]), vX, vY))
        s1 = evaluate_config(model.cpu(), list(cols[197:] - 197), core.load_stats("deploy"),
                             torch.device("cpu"))
        model.to(dev)
        res = {"wr": wr, "volume": vol, "scope": sc, "n_synthetic": len(syn), "gen_stats": st,
               "val_loss": vl, "epochs": len(hist),
               "test": train_wp.evaluate(model, cols, "test", dev),
               "NODRIFT": train_wp.evaluate(model, cols, "NODRIFT", dev),
               "sanity_passed": s1["sanity_passed"], "sanity_total": s1["sanity_total"],
               "degen_scores": s1["degen_scores"], "secs": time.time() - t0}
        json.dump(res, open(path, "w"), indent=1, default=float)
        print(name, len(syn), round(s1["degen_scores"]["5 tanks"], 3), s1["sanity_passed"],
              flush=True)
    # baseline: the leak-free enriched model, seed 42 (same architecture, no synthetic data)
    m, c = train_wp.load("enriched", seed=42)
    b = evaluate_config(m, list(c[197:] - 197), core.load_stats("deploy"), torch.device("cpu"))
    out = {"baseline_enriched_s42": {"sanity_passed": b["sanity_passed"],
                                     "degen_scores": b["degen_scores"]},
           "configs": {os.path.basename(p)[:-5]: json.load(open(os.path.join(part, p)))
                       for p in sorted(os.listdir(part))}}
    save("scope_sweep", out)


# ── NGS quartiles ───────────────────────────────────────────────────────

def job_ngs_quartiles():
    import torch
    from paper1_revision.train_wp import load
    d = json.load(open(os.path.join(TRAINING_DIR, "ngs2026", "ngs_drafts.json")))["drafts"]
    sc = json.load(open(os.path.join(TRAINING_DIR, "ngs2026", "meta_analysis.json")))[
        "per_draft_offmeta_score"]
    rows = [(g["team0_heroes"], g["team1_heroes"], g["game_map"], "mid") for g in d]
    y = np.array([1.0 if int(g["winner"]) == 0 else 0.0 for g in d])
    s = np.array([sc[str(g["replay_id"])] for g in d])
    cuts = np.quantile(s, [0.25, 0.5, 0.75])
    q = np.digitize(s, cuts)
    Xf, Xs = core.featurize(rows, core.load_stats("deploy"), nproc=NPROC)
    out = {"n": len(rows), "cuts_bits": cuts.tolist(), "models": {}}
    for n in ("naive", "enriched", "aug_wr10_512"):
        m, cols = load(n)
        with torch.no_grad():
            a = m(torch.tensor(Xf[:, cols])).view(-1).numpy()
            b = m(torch.tensor(Xs[:, cols])).view(-1).numpy()
        p = 0.5 * (a + 1 - b)
        out["models"][n] = {f"Q{k + 1}": {"n": int((q == k).sum()),
                                          "acc": float(np.mean((p[q == k] > .5) == (y[q == k] > .5)))}
                            for k in range(4)}
        out["models"][n]["all"] = float(np.mean((p > .5) == (y > .5)))
        print(n, {k: round(v["acc"], 4) if isinstance(v, dict) else round(v, 4)
                  for k, v in out["models"][n].items()}, flush=True)
    save("ngs_quartiles", out)


# ── A_partial / G_base rebench ──────────────────────────────────────────

def job_rebench_ag():
    import subprocess
    runs = ",".join(f"old:{c}_s{s}" for c in ("A_partial", "G_base") for s in range(5))
    subprocess.run([sys.executable, os.path.join(HERE, "bench_mcts.py"), runs, "--temps", "1,0"],
                   cwd=TRAINING_DIR, check=True)
    B = os.path.join(core.RESULTS, "mcts_bench")
    out = {}
    for c in ("A_partial", "G_base"):
        for T in ("1", "0"):
            ds = [json.load(open(os.path.join(B, f"old__{c}_s{s}__T{T}.json"))) for s in range(5)]
            out[f"{c}|T{T}"] = {k: [float(np.mean([x["means"][k] for x in ds])),
                                    float(np.std([x["means"][k] for x in ds], ddof=1) / np.sqrt(5))]
                                for k in ds[0]["means"]}
    save("rebench_ag", out)


# ── draft concentration ─────────────────────────────────────────────────

def _team_stats(games, heroes_sets):
    """For each hero set S: games where one team contains S -> (n, wins),
    plus per-hero (n, wins) for the independence expectation."""
    from collections import defaultdict
    hero = defaultdict(lambda: [0, 0])
    sets = {S: [0, 0] for S in heroes_sets}
    for g in games:
        for ti, team in ((0, g[3]), (1, g[4])):
            w = 1 if g[6] == ti else 0
            ts = set(team)
            for h in team:
                hero[h][0] += 1
                hero[h][1] += w
            for S in heroes_sets:
                if S <= ts:
                    sets[S][0] += 1
                    sets[S][1] += w
    return hero, sets


def _edges(games, favs, k):
    import itertools
    combos = [frozenset(c) for c in itertools.combinations(favs, k)]
    hero, sets = _team_stats(games, combos)
    rows, tot_n, tot_w, tot_e = [], 0, 0, 0.0
    for S in combos:
        n, w = sets[S]
        if n == 0:
            continue
        wr = {h: 100 * hero[h][1] / hero[h][0] for h in S}
        exp = 50 + sum(v - 50 for v in wr.values())
        obs = 100 * w / n
        se = 100 * np.sqrt(max(obs / 100 * (1 - obs / 100), 1e-6) / n)
        rows.append({"set": sorted(S), "n": n, "obs": obs, "exp": exp, "delta": obs - exp,
                     "z": (obs - exp) / se})
        tot_n += n
        tot_w += w
        tot_e += exp * n
    pooled = {"n": tot_n, "obs": 100 * tot_w / max(tot_n, 1), "exp": tot_e / max(tot_n, 1)}
    pooled["delta"] = pooled["obs"] - pooled["exp"]
    pooled["z"] = pooled["delta"] / (100 * np.sqrt(0.25 / max(tot_n, 1)))
    return rows, pooled


def job_concentration():
    from collections import Counter
    from drift2026.build_patch_stats import comp_key
    B = os.path.join(core.RESULTS, "mcts_bench")
    sp = core.split_games()
    snap = sp["train"] + sp["val"] + sp["test"]
    fwd = core.gold_games("NODRIFT") + core.gold_games("T17")
    comps = core.comps_from_json(core.comp_path("deploy"))
    team_index = Counter(tuple(sorted(t)) for g in snap for t in (g[3], g[4]))
    out = {}
    for label, prefix in (("F_oof", "new__F_oof_s"), ("J_800sim_submitted", "old__J_800sim_s")):
        per_seed, pooled = [], Counter()
        drafts = []
        for s in range(5):
            p = os.path.join(B, f"{prefix}{s}__T1.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))["drafts"]
            c = Counter(h for x in d for h in x["our"])
            tot = sum(c.values())
            pr = np.array(list(c.values())) / tot
            per_seed.append({"seed": s, "distinct": len(c), "entropy": float(-(pr * np.log2(pr)).sum()),
                             "top3": [h for h, _ in c.most_common(3)]})
            pooled.update(c)
            drafts += d
        if not drafts:
            continue
        tot = sum(pooled.values())
        pr = np.array(list(pooled.values())) / tot
        favs = [h for h, _ in pooled.most_common(7)]
        res = {"per_seed": per_seed, "pooled_distinct": len(pooled),
               "pooled_entropy": float(-(pr * np.log2(pr)).sum()),
               "top10_share": sum(v for _, v in pooled.most_common(10)) / tot,
               "favorites": favs}
        for k, nm in ((2, "pairs"), (3, "trios")):
            for gname, games in (("in_sample_train", sp["train"]), ("forward_post_snapshot", fwd)):
                rows, pool = _edges(games, favs, k)
                res[f"{nm}_{gname}"] = {"pooled": pool, "sets": rows}
        teams = Counter(tuple(sorted(x["our"])) for x in drafts)
        res["top10_teams"] = [{"team": list(t), "share": n / len(drafts),
                               "corpus_occurrences": team_index.get(t, 0)} for t, n in teams.most_common(10)]
        tiers = [x.get("tier", "mid") for x in drafts]
        in_tab = [comp_key(x["our"]) in comps.get(t, {}) for x, t in zip(drafts, tiers)]
        big = [comps.get(t, {}).get(comp_key(x["our"]), (0, 0))[1] >= 1000 for x, t in zip(drafts, tiers)]
        res["role_comp_in_own_table"] = float(np.mean(in_tab))
        res["role_comp_ge_1000_games"] = float(np.mean(big))
        out[label] = res
        print(label, favs, res["pairs_in_sample_train"]["pooled"], res["pairs_forward_post_snapshot"]["pooled"],
              flush=True)
    save("concentration", out)


# ── ensemble uncertainty ────────────────────────────────────────────────

def _ensemble_specs():
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    E = list(ENRICHED_GROUPS)
    specs = []
    i = 0
    for arch in ([256, 128], [512, 256, 128]):
        for half in (0, 1):
            for s in (0, 1):
                specs.append({"name": f"ens_full_a{arch[0]}_h{half}_s{s}", "groups": E, "arch": arch,
                              "half": half, "seed": 300 + i})
                i += 1
    for j, g in enumerate(E):
        specs.append({"name": f"ens_minus_{g}", "groups": [x for x in E if x != g], "arch": [256, 128],
                      "half": j % 2, "seed": 300 + i})
        i += 1
    for arch, half in (([256, 128], 0), ([256, 128], 1), ([512, 256, 128], 0)):
        specs.append({"name": f"ens_naive_a{arch[0]}_h{half}", "groups": [], "arch": arch,
                      "half": half, "seed": 300 + i})
        i += 1
    return specs


def job_ensemble():
    import torch
    from overfit2026 import data as odata
    from paper1_revision import train_wp
    from sweep_enriched_wp import WinProbEnrichedModel
    dev = torch.device("cuda")
    mdir = os.path.join(core.MODEL_DIR, "ensemble")
    os.makedirs(mdir, exist_ok=True)
    Xf, Xs, y, rid, tier = core.load_features("train_oof")
    half = np.array([odata.splitmix64(int(r) * 7 + 101) & 1 for r in rid])
    members = []
    for sp in _ensemble_specs():
        path = os.path.join(mdir, sp["name"] + ".pt")
        cols = core.group_cols(sp["groups"])
        if not os.path.exists(path):
            m = half == sp["half"]
            tX = torch.tensor(np.concatenate([Xf[m][:, cols], Xs[m][:, cols]]), device=dev)
            tY = torch.tensor(np.concatenate([y[m], 1 - y[m]]), device=dev)
            vX, vY = train_wp._tensors(cols, "val", dev)
            model, vl, hist = train_wp.train_one(sp["name"], sp, sp["seed"], dev, (tX, tY, vX, vY))
            torch.save(model.state_dict(), path)
            json.dump({"spec": sp, "val_loss": vl, "epochs": len(hist),
                       "test": train_wp.evaluate(model, cols, "test", dev)},
                      open(path[:-3] + ".json", "w"), indent=1, default=float)
            del tX, tY
            torch.cuda.empty_cache()
            print("trained", sp["name"], round(vl, 5), flush=True)
        m = WinProbEnrichedModel(len(cols), sp["arch"], dropout=0.3)
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        members.append((m, cols))
    del Xf, Xs

    def variance(rows):
        A, B = core.featurize(rows, core.load_stats("deploy"), nproc=NPROC)
        P = []
        with torch.no_grad():
            for m, cols in members:
                a = m(torch.tensor(A[:, cols])).view(-1).numpy()
                b = m(torch.tensor(B[:, cols])).view(-1).numpy()
                P.append(0.5 * (a + 1 - b))
        P = np.array(P)
        return P.var(0), P.mean(0)

    rng = np.random.RandomState(0)
    sets = {}
    test = core.split_games()["test"]
    sel = rng.choice(len(test), 2000, replace=False)
    sets["held_out_human"] = [(test[i][3], test[i][4], test[i][2], test[i][1]) for i in sel]
    Bd = os.path.join(core.RESULTS, "mcts_bench")
    for lab, pref in (("mcts_F_oof", "new__F_oof_s"),):
        d = []
        for s in range(5):
            p = os.path.join(Bd, f"{pref}{s}__T0.json")
            if os.path.exists(p):
                d += json.load(open(p))["drafts"]
        if d:
            sel = rng.choice(len(d), min(2000, len(d)), replace=False)
            sets[lab] = [(d[i]["our"], d[i]["opp"], d[i]["map"], "mid") for i in sel]
    from paper1_revision.tournament import STRATEGIES
    from paper1_revision.score_tournament import load_pair
    for s in STRATEGIES:
        rows = []
        for o in STRATEGIES:
            if o == s:
                continue
            for a, b, side in ((s, o, "team0"), (o, s, "team1")):
                pr = load_pair(a, b)
                if pr is None:
                    continue
                for r in pr["records"]:
                    ours = r[side]["picks"]
                    opp = r["team1" if side == "team0" else "team0"]["picks"]
                    rows.append((ours, opp, r["game_map"], r["tier"]))
        if rows:
            sel = rng.choice(len(rows), min(2000, len(rows)), replace=False)
            sets[f"tournament_{s}"] = [rows[i] for i in sel]
    from experiment_synthetic_augmentation import generate_synthetic_data, DEGEN_COMPS, STANDARD
    random.seed(1)
    real = [{"team0_heroes": list(g[3]), "team1_heroes": list(g[4])} for g in test]
    syn, _ = generate_synthetic_data(real, json.load(open(core.comp_path("deploy"))), unseen_wr=10.0,
                                     unseen_volume=5, scope="tier2_only")
    random.shuffle(syn)
    sets["synthetic_degenerate"] = [(r["team0_heroes"], r["team1_heroes"], r["game_map"],
                                     r["skill_tier"]) for r in syn[:500]]
    sets["degenerate_probes"] = [(c, STANDARD, "Cursed Hollow", "mid") for c in DEGEN_COMPS.values()]
    res = {"members": [sp["name"] for sp in _ensemble_specs()], "sets": {}}
    hv, _ = variance(sets["held_out_human"])
    hmed, hp95 = float(np.median(hv)), float(np.percentile(hv, 95))
    for k, rows in sets.items():
        v, mu = variance(rows) if k != "held_out_human" else (hv, None)
        res["sets"][k] = {"n": len(rows), "median_x1e3": 1e3 * float(np.median(v)),
                          "p95_x1e3": 1e3 * float(np.percentile(v, 95)),
                          "ratio_to_human_median": float(np.median(v)) / hmed,
                          "frac_above_human_p95": float(np.mean(v > hp95))}
        print(k, {a: round(b, 3) for a, b in res["sets"][k].items()}, flush=True)
    save("ensemble_uncertainty", res)


# ── CQL with enriched features, leak-free ───────────────────────────────

CQL_DIR = os.path.join(core.CACHE, "cql_enriched_oof")


def job_cql_build():
    from rerun2026 import common
    from rerun2026.phase0_features import _cql_enriched_chunk
    from sweep_enriched_wp import compute_group_indices
    from experiment_cql_enriched import get_enriched_cols
    import multiprocessing as mp
    common.setup()
    rows = common.load_data()
    T = core.test_ids()
    dim = 289 + len(get_enriched_cols(compute_group_indices()))
    fields = [("actions", "int64"), ("outcomes", "float32")]
    val = [r for r in rows if r["replay_id"] not in T and core.is_val(r["replay_id"])]
    train = [r for r in rows if r["replay_id"] not in T and not core.is_val(r["replay_id"])]
    del rows

    def stream(data, stats_of):
        by = {}
        for r in data:
            by.setdefault(stats_of(r), []).append(r)
        with mp.get_context("fork").Pool(NPROC) as pool:
            for sname, rs in sorted(by.items()):
                sd = core.stats_dict(core.load_stats(sname))
                parts = [(rs[i:i + 2000], sd) for i in range(0, len(rs), 2000)]
                for out in pool.imap(_cql_enriched_chunk, parts):
                    yield out
                print(f"  {sname}: {len(rs):,} replays", flush=True)

    for name, data, fn in (("val", val, lambda r: "deploy"),
                           ("train", train, lambda r: f"oof{core.fold_of(r['replay_id'])}")):
        d = os.path.join(CQL_DIR, name)
        if os.path.exists(os.path.join(d, "meta.json")):
            continue
        common.memmap_write(d, stream(data, fn), dim, fields)
    save("cql_enriched_build", {"dim": dim, "train": common.memmap_meta(os.path.join(CQL_DIR, "train")),
                                "val": common.memmap_meta(os.path.join(CQL_DIR, "val"))})


def job_cql_train(alpha):
    import torch
    from rerun2026 import common
    from rerun2026.train_jobs import MemmapCQLDataset
    import experiment_cql_enriched as ece
    ece.RESULTS_DIR = os.path.join(core.MODEL_DIR, "cql_enriched_oof")
    os.makedirs(ece.RESULTS_DIR, exist_ok=True)
    ece.EnrichedCQLDataset = MemmapCQLDataset
    dim = common.memmap_meta(os.path.join(CQL_DIR, "train"))["state_dim"]
    # early stopping on the VALIDATION split (passed where the function expects "test")
    model, path = ece.train_enriched_cql(os.path.join(CQL_DIR, "train"), os.path.join(CQL_DIR, "val"),
                                         dim, alpha=alpha, epochs=50, device=torch.device("cuda"))
    save(f"cql_enriched_train_a{alpha}", {"alpha": alpha, "checkpoint": path, "input_dim": dim})


def job_cql_eval():
    import torch
    from collections import Counter
    from experiment_rich_evaluation import (make_cql_enriched_strategy, run_drafts_with_strategy,
                                            counter_responsiveness, synergy_exploitation)
    from experiment_synthetic_augmentation import ENRICHED_GROUPS
    from sweep_enriched_wp import compute_group_indices
    from rerun2026.phase3_benchmarks import load_gd_models
    from shared import MAPS, SKILL_TIERS
    from paper1_revision.score_tournament import score_rows
    torch.set_num_threads(2)
    dev = torch.device("cpu")
    gd = load_gd_models(dev)
    hp, deploy = core.load_stats("hp"), core.load_stats("deploy")
    out = {}
    for alpha in (2.0, 0.5):
        ck = os.path.join(core.MODEL_DIR, "cql_enriched_oof", f"_cql_enriched_a{alpha}.pt")
        if not os.path.exists(ck):
            continue
        fn = make_cql_enriched_strategy(ck, dev, deploy, compute_group_indices(), ENRICHED_GROUPS)
        random.seed(42)
        cfgs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2) for i in range(1000)]
        drafts = []
        for seed in range(5):
            random.seed(seed)
            torch.manual_seed(seed)
            drafts += run_drafts_with_strategy(fn, cfgs, gd, deploy, dev)
        c = Counter(h for d in drafts for h in d["our_picks"])
        tot = sum(c.values())
        pr = np.array(list(c.values())) / tot
        agree = total = 0
        for d in drafts:
            for st in d["steps"]:
                if not st["is_ours"] or st["type"] == "ban":
                    continue
                s_t = torch.tensor(st["state"], dtype=torch.float32).unsqueeze(0)
                m_t = torch.tensor(st["mask"], dtype=torch.float32).unsqueeze(0)
                votes = Counter()
                with torch.no_grad():
                    for g in gd:
                        votes[g(s_t, m_t).argmax(dim=1).item()] += 1
                agree += st["hero_idx"] == votes.most_common(1)[0][0]
                total += 1
        rows = [(d["our_picks"], d["opp_picks"], d["game_map"], d["tier"]) for d in drafts]
        sc = score_rows(rows)
        out[f"a{alpha}"] = {
            "n": len(drafts), "healer": 100 * np.mean([d["has_healer"] for d in drafts]),
            "degen": 100 * np.mean([d["is_degen"] for d in drafts]),
            "counter": float(np.mean([counter_responsiveness(d["our_picks"], d["opp_picks"], hp, d["tier"])
                                      for d in drafts])),
            "synergy": float(np.mean([synergy_exploitation(d["our_picks"], hp, d["tier"]) for d in drafts])),
            "distinct": len(c), "entropy": float(-(pr * np.log2(pr)).sum()),
            "top10": 100 * sum(v for _, v in c.most_common(10)) / tot,
            "gd_similarity": 100 * agree / max(total, 1),
            "refs": {k: float(np.mean(v)) for k, v in sc.items()}}
        print(alpha, {k: v for k, v in out[f"a{alpha}"].items() if k != "refs"}, flush=True)
    save("cql_enriched_eval", out)


if __name__ == "__main__":
    job = sys.argv[1]
    if job.startswith("cql_train_a"):
        job_cql_train(float(job[len("cql_train_a"):]))
    else:
        {"scope": job_scope, "ngs_quartiles": job_ngs_quartiles, "rebench_ag": job_rebench_ag,
         "concentration": job_concentration, "ensemble": job_ensemble,
         "cql_build": job_cql_build, "cql_eval": job_cql_eval}[job]()
