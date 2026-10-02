"""
Ensemble epistemic-uncertainty study for the WP value model (reviewer note:
vary DATA HALF and FEATURE SUBSET, not just seeds).

Trains a 20-member ensemble of full-draft WP models on the pinned snapshot,
varying ALL of:
  - seed              (distinct per member, 201..220)
  - architecture      (256x128 vs 512x256x128)
  - feature subset    (full enriched; minus-one enriched group, rotating over
                       all 9 groups; naive 197d base)
  - data half         (replay-level halves A/B of the phase0 train cache;
                       swap-augmented row pairs (2i, 2i+1) stay together, so
                       halving is replay-level by construction)

Then computes per-draft ensemble predictive variance of the SYMMETRIZED WP,
p_sym = 0.5 * (p(x) + 1 - p(swap(x))), on four draft sets:
  1. real      2,000 held-out real human drafts (test split of the snapshot)
  2. agent     2,000 J_800sim agent drafts (results/diversity/J_800sim_s*__base.json)
  3. degen     6 degenerate probe comps + 500 synthetic degenerate drafts
               (unseen role tuples, sampled by role like the augmentation
               generator, real-draft opponents)
  4. constrained  500 constrained_mcts drafts (regenerated; rich_eval stores
               only aggregates)

Usage:
  python3 rerun2026/ensemble_uncertainty.py --task pool [--dry-run]   # train all (skip-if-done)
  python3 rerun2026/ensemble_uncertainty.py --task train --member NAME [--smoke]
  python3 rerun2026/ensemble_uncertainty.py --task gen-constrained [--drafts 500]
  python3 rerun2026/ensemble_uncertainty.py --task eval
  python3 rerun2026/ensemble_uncertainty.py --task per-strategy   # after eval

Outputs:
  models/ens_unc/<member>.pt, models/meta/ens_unc_<member>.json
  results/ensemble_uncertainty_constrained_drafts.json
  results/ensemble_uncertainty.json, results/ensemble_uncertainty.md
"""
import os
import sys
import json
import glob
import time
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rerun2026 import common
from rerun2026.common import (MODELS_DIR, RESULTS_DIR, FULL_TRAIN_NPZ,
                              FULL_TEST_NPZ, Job)

common.setup()

import numpy as np
import torch

from sweep_enriched_wp import (WinProbEnrichedModel, FEATURE_GROUPS,
                               compute_group_indices, extract_features,
                               _swap_features)
from experiment_synthetic_augmentation import (
    ENRICHED_GROUPS, DEGEN_COMPS, STANDARD, build_heroes_by_blizz_role,
    generate_all_role_tuples, sample_heroes_for_roles)

ENS_DIR = os.path.join(MODELS_DIR, "ens_unc")
CONSTRAINED_DRAFTS_PATH = os.path.join(
    RESULTS_DIR, "ensemble_uncertainty_constrained_drafts.json")
OUT_JSON = os.path.join(RESULTS_DIR, "ensemble_uncertainty.json")
OUT_MD = os.path.join(RESULTS_DIR, "ensemble_uncertainty.md")

HALF_SEED = 20260709   # replay-level half assignment (fixed for all members)

# ── Ensemble roster ──────────────────────────────────────────────────
# (name, preset, arch, half, seed). Presets: "full" = all 9 ENRICHED_GROUPS,
# "minus:<g>" drops one group, "naive" = base 197d only.
A256, A512 = "256,128", "512,256,128"


def _roster():
    members = [
        ("e_a256_hA_s201", "full", A256, "A", 201),
        ("e_a256_hB_s202", "full", A256, "B", 202),
        ("e_a512_hA_s203", "full", A512, "A", 203),
        ("e_a512_hB_s204", "full", A512, "B", 204),
        ("e_a256_hA_s205", "full", A256, "A", 205),
        ("e_a512_hB_s206", "full", A512, "B", 206),
        ("e_a256_hB_s207", "full", A256, "B", 207),
        ("e_a512_hA_s208", "full", A512, "A", 208),
    ]
    for i, g in enumerate(ENRICHED_GROUPS):
        arch = A256 if i % 2 == 0 else A512
        half = ("A", "B", "B", "A", "A", "B", "B", "A", "A")[i]
        members.append((f"m_{g}_s{209 + i}", f"minus:{g}", arch, half, 209 + i))
    members += [
        ("naive_a256_hB_s218", "naive", A256, "B", 218),
        ("naive_a512_hA_s219", "naive", A512, "A", 219),
        ("naive_a256_hA_s220", "naive", A256, "A", 220),
    ]
    assert len(members) == 20
    return members


ROSTER = _roster()
ROSTER_BY_NAME = {m[0]: m for m in ROSTER}


def preset_groups(preset):
    if preset == "full":
        return list(ENRICHED_GROUPS)
    if preset == "naive":
        return []
    if preset.startswith("minus:"):
        g = preset.split(":", 1)[1]
        assert g in ENRICHED_GROUPS, g
        return [x for x in ENRICHED_GROUPS if x != g]
    raise ValueError(preset)


def cols_for(groups):
    gi = compute_group_indices()
    cols = []
    for g in groups:
        s, e = gi[g]
        cols.extend(range(s, e))
    return cols


def model_path(name):
    return os.path.join(ENS_DIR, f"{name}.pt")


# ── Training (one member per subprocess) ─────────────────────────────

def half_row_indices(n_rows, half):
    """Replay-level half: swap-augmented pairs (2p, 2p+1) stay together."""
    assert n_rows % 2 == 0, "full cache must be swap-augmented pairs"
    n_pairs = n_rows // 2
    rng = np.random.RandomState(HALF_SEED)
    perm = rng.permutation(n_pairs)
    pairs = perm[: n_pairs // 2] if half == "A" else perm[n_pairs // 2:]
    rows = np.sort(np.concatenate([2 * pairs, 2 * pairs + 1]))
    return rows


def task_train(args):
    from retrain_frozen_stats import train_wp_model

    name = args.member
    _, preset, arch_s, half, seed = ROSTER_BY_NAME[name]
    groups = preset_groups(preset)
    cols = cols_for(groups)
    arch = [int(x) for x in arch_s.split(",")]
    dim = 197 + len(cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = np.load(FULL_TRAIN_NPZ)
    te = np.load(FULL_TEST_NPZ)
    rows = half_row_indices(len(tr["labels"]), half)
    if args.smoke:
        rows = rows[:100_000]

    def build(split, sel=None):
        b, e, l = split["bases"], split["enricheds"], split["labels"]
        if sel is not None:
            b, e, l = b[sel], e[sel], l[sel]
        X = np.concatenate([b, e[:, cols]], axis=1) if cols else np.array(b)
        return (torch.tensor(X, dtype=torch.float32).to(device),
                torch.tensor(l, dtype=torch.float32).to(device))

    train_X, train_y = build(tr, rows)
    test_X, test_y = build(te)
    print(f"{name}: preset={preset} arch={arch} half={half} seed={seed} "
          f"dim={dim} train={len(train_X):,} test={len(test_X):,}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = WinProbEnrichedModel(dim, arch, dropout=0.3)
    epochs = 3 if args.smoke else 200
    model, acc = train_wp_model(model, train_X, test_X, train_y, test_y,
                                name, device, epochs=epochs)
    os.makedirs(ENS_DIR, exist_ok=True)
    out = model_path(name)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out)
    common.write_meta(f"ens_unc_{name}", {
        "kind": "wp_ensemble_member", "preset": preset, "groups": groups,
        "arch": arch, "input_dim": dim, "half": half, "half_seed": HALF_SEED,
        "seed": seed, "n_train": len(train_X), "best_acc": acc,
        "smoke": bool(args.smoke), "out": out,
    })
    print(f"{name}: best_acc={acc:.2f}% -> {out}")


def task_pool(args):
    jobs = []
    for name, preset, arch, half, seed in ROSTER:
        jobs.append(Job(
            name=f"ens_unc_{name}",
            argv=["rerun2026/ensemble_uncertainty.py", "--task", "train",
                  "--member", name],
            outputs=[model_path(name)],
            weight="heavy"))
    failed = common.run_pool(jobs, dry_run=args.dry_run, force=args.force,
                             only=args.only)
    if failed:
        sys.exit(1)


# ── Draft sets ───────────────────────────────────────────────────────

def task_gen_constrained(args):
    """Regenerate constrained_mcts drafts (rich_eval stored only aggregates)."""
    if os.path.exists(CONSTRAINED_DRAFTS_PATH) and not args.force:
        print(f"exists: {CONSTRAINED_DRAFTS_PATH} (use --force to regenerate)")
        return
    from rerun2026.constrained_search import build_strategy, FALLBACKS
    from rerun2026.phase3_benchmarks import load_gd_models
    from experiment_rich_evaluation import run_drafts_with_strategy
    from shared import MAPS, SKILL_TIERS, HEROES, NUM_HEROES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = common.stats_cache()
    group_indices = compute_group_indices()
    gd_models = load_gd_models(torch.device("cpu"))
    label, strategy_fn = build_strategy("constrained_mcts", device, stats,
                                        group_indices)
    # Same config protocol as constrained_search.task_rich_eval, seed 0
    random.seed(42)
    draft_configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
                     for i in range(args.drafts)]
    random.seed(0)
    torch.manual_seed(0)
    t0 = time.time()
    drafts = run_drafts_with_strategy(strategy_fn, draft_configs, gd_models,
                                      stats, device)
    print(f"{label}: {len(drafts)} drafts in {(time.time()-t0)/60:.1f} min "
          f"(fallbacks={dict(FALLBACKS)})")
    records = []
    for d in drafts:
        t0h = [HEROES[i] for i in range(NUM_HEROES) if d["terminal_t0"][i] > 0]
        t1h = [HEROES[i] for i in range(NUM_HEROES) if d["terminal_t1"][i] > 0]
        records.append({"team0_heroes": t0h, "team1_heroes": t1h,
                        "game_map": d["game_map"], "skill_tier": d["tier"],
                        "our_team": d["our_team"],
                        "is_degen": int(d["is_degen"])})
    with open(CONSTRAINED_DRAFTS_PATH, "w") as f:
        json.dump({"strategy": "constrained_mcts", "seed": 0,
                   "n": len(records), "records": records}, f)
    print(f"Saved {CONSTRAINED_DRAFTS_PATH}")


def _load_agent_drafts(n=2000):
    files = sorted(glob.glob(os.path.join(
        RESULTS_DIR, "diversity", "J_800sim_s*__base.json")))
    files = [f for f in files if "_ens_" not in os.path.basename(f)]
    recs = []
    for fp in files:
        d = json.load(open(fp))
        for dr in d["drafts"]:
            if dr["side"] == 0:
                t0, t1 = dr["our"], dr["opp"]
            else:
                t0, t1 = dr["opp"], dr["our"]
            recs.append({"team0_heroes": t0, "team1_heroes": t1,
                         "game_map": dr["map"], "skill_tier": "mid",
                         "src": os.path.basename(fp)})
    rng = np.random.RandomState(8)
    idx = rng.permutation(len(recs))[:n]
    return [recs[i] for i in sorted(idx)], len(files)


def _synthetic_degen_drafts(opponent_teams, n=500):
    """Unseen role tuples, sampled by role like the augmentation generator."""
    from shared import MAPS
    comp_path = os.path.join(common.TRAINING_DIR, "..", "src", "lib", "data",
                             "compositions.json")
    comp_data = json.load(open(comp_path))
    heroes_by_role = build_heroes_by_blizz_role()
    all_tuples = generate_all_role_tuples()
    known = set()
    for c in comp_data.get("mid", []):
        known.add(tuple(sorted(c["roles"])))
    unseen = [t for t in all_tuples if t not in known]
    random.seed(9)
    recs = []
    while len(recs) < n:
        role_tuple = random.choice(unseen)
        team = sample_heroes_for_roles(role_tuple, heroes_by_role)
        for _ in range(20):
            opp = random.choice(opponent_teams)
            if not set(team) & set(opp):
                break
        else:
            continue
        recs.append({"team0_heroes": team, "team1_heroes": list(opp),
                     "game_map": random.choice(MAPS), "skill_tier": "mid",
                     "role_tuple": list(role_tuple)})
    return recs, len(unseen)


# ── Ensemble evaluation ──────────────────────────────────────────────

def _extract_set(records, stats):
    """(base, enr, base_swap, enr_swap) float32 stacks for a record list."""
    all_mask = [True] * len(FEATURE_GROUPS)
    B, E, BS, ES = [], [], [], []
    for d in records:
        rec = {"team0_heroes": d["team0_heroes"],
               "team1_heroes": d["team1_heroes"],
               "game_map": d["game_map"], "skill_tier": d["skill_tier"],
               "winner": 0}
        b, e = extract_features(rec, stats, all_mask)
        bs, es = _swap_features(b, e)
        B.append(b); E.append(e); BS.append(bs); ES.append(es)
    return (np.array(B, dtype=np.float32), np.array(E, dtype=np.float32),
            np.array(BS, dtype=np.float32), np.array(ES, dtype=np.float32))


def _member_predict(name, feats, device):
    """Symmetrized WP for one member over one extracted set."""
    _, preset, arch_s, half, seed = ROSTER_BY_NAME[name]
    cols = cols_for(preset_groups(preset))
    arch = [int(x) for x in arch_s.split(",")]
    dim = 197 + len(cols)
    model = WinProbEnrichedModel(dim, arch, dropout=0.3).to(device)
    model.load_state_dict(torch.load(model_path(name), weights_only=True,
                                     map_location=device))
    model.eval()
    B, E, BS, ES = feats
    Xf = np.concatenate([B, E[:, cols]], axis=1) if cols else B
    Xr = np.concatenate([BS, ES[:, cols]], axis=1) if cols else BS
    out = []
    for X in (Xf, Xr):
        ps = []
        with torch.no_grad():
            for i in range(0, len(X), 65536):
                xt = torch.tensor(X[i:i + 65536], dtype=torch.float32,
                                  device=device)
                ps.append(model(xt).cpu().numpy())
        out.append(np.concatenate(ps))
    p_fwd, p_rev = out
    return 0.5 * (p_fwd + (1.0 - p_rev))


def _dist_stats(v):
    q = lambda p: float(np.percentile(v, p))
    return {"n": int(len(v)), "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "p5": q(5), "p25": q(25), "p75": q(75), "p90": q(90),
            "p95": q(95), "p99": q(99), "max": float(np.max(v))}


def task_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = common.stats_cache()

    missing = [m[0] for m in ROSTER if not os.path.exists(model_path(m[0]))]
    if missing:
        raise RuntimeError(f"missing members: {missing} (run --task pool)")

    # ── build the four draft sets ──
    print("Loading pinned snapshot (real held-out drafts)…")
    _, test = common.load_split()
    rng = np.random.RandomState(7)
    real = [test[i] for i in sorted(rng.permutation(len(test))[:2000])]
    real_recs = [{"team0_heroes": d["team0_heroes"],
                  "team1_heroes": d["team1_heroes"],
                  "game_map": d["game_map"], "skill_tier": d["skill_tier"]}
                 for d in real]

    agent_recs, n_agent_files = _load_agent_drafts(2000)
    print(f"agent drafts: {len(agent_recs)} from {n_agent_files} J_800sim dumps")

    probe_recs = [{"team0_heroes": comp, "team1_heroes": STANDARD,
                   "game_map": "Cursed Hollow", "skill_tier": "mid",
                   "probe": pname} for pname, comp in DEGEN_COMPS.items()]
    opp_pool = ([d["team0_heroes"] for d in test]
                + [d["team1_heroes"] for d in test])
    syn_recs, n_unseen = _synthetic_degen_drafts(opp_pool, 500)
    print(f"degenerate: {len(probe_recs)} probes + {len(syn_recs)} synthetic "
          f"(from {n_unseen} unseen mid-tier role tuples)")

    if not os.path.exists(CONSTRAINED_DRAFTS_PATH):
        raise RuntimeError("run --task gen-constrained first")
    cdump = json.load(open(CONSTRAINED_DRAFTS_PATH))
    con_recs = cdump["records"]
    print(f"constrained_mcts drafts: {len(con_recs)} (regenerated, seed "
          f"{cdump['seed']})")

    sets = [("real", real_recs), ("agent", agent_recs),
            ("degen_probes", probe_recs), ("degen_synth", syn_recs),
            ("constrained", con_recs)]

    # ── features once per set, then all members ──
    preds = {}   # set -> [n_members, N] symmetrized WP
    for sname, recs in sets:
        t0 = time.time()
        feats = _extract_set(recs, stats)
        P = np.stack([_member_predict(m[0], feats, device) for m in ROSTER])
        preds[sname] = P
        print(f"  {sname}: {P.shape[1]} drafts x {P.shape[0]} members "
              f"({time.time()-t0:.0f}s)")

    member_meta = []
    for name, preset, arch, half, seed in ROSTER:
        meta_p = os.path.join(MODELS_DIR, "meta", f"ens_unc_{name}.json")
        acc = None
        if os.path.exists(meta_p):
            acc = json.load(open(meta_p)).get("best_acc")
        member_meta.append({"name": name, "preset": preset, "arch": arch,
                            "half": half, "seed": seed, "test_acc": acc})

    subgroups = {
        "all20": [m[0] for m in ROSTER],
        "enriched_only17": [m[0] for m in ROSTER
                            if not m[0].startswith("naive")],
        "full_enriched8": [m[0] for m in ROSTER if m[1] == "full"],
    }
    names = [m[0] for m in ROSTER]

    results = {"members": member_meta, "half_seed": HALF_SEED,
               "n_members": len(ROSTER), "sets": {}, "subgroups": {}}
    variances = {}
    for sname, recs in sets:
        P = preds[sname]
        var = P.var(axis=0, ddof=1)
        variances[sname] = var
        results["sets"][sname] = {
            "variance": _dist_stats(var),
            "std": _dist_stats(np.sqrt(var)),
            "mean_pred": _dist_stats(P.mean(axis=0)),
        }
        for sg, members in subgroups.items():
            ix = [names.index(x) for x in members]
            sv = P[ix].var(axis=0, ddof=1)
            results["subgroups"].setdefault(sg, {})[sname] = _dist_stats(sv)

    # per-probe detail
    results["degen_probe_detail"] = [
        {"probe": r["probe"],
         "mean_wp": float(preds["degen_probes"][:, i].mean()),
         "variance": float(preds["degen_probes"][:, i].var(ddof=1)),
         "std": float(preds["degen_probes"][:, i].std(ddof=1))}
        for i, r in enumerate(probe_recs)]

    # overlap vs real-human variance distribution
    hv = variances["real"]
    h_med, h_p90, h_p95 = (np.median(hv), np.percentile(hv, 90),
                           np.percentile(hv, 95))
    overlap = {}
    for sname in ("agent", "degen_synth", "degen_probes", "constrained"):
        v = variances[sname]
        overlap[sname] = {
            "frac_above_real_median": float((v > h_med).mean()),
            "frac_above_real_p90": float((v > h_p90).mean()),
            "frac_above_real_p95": float((v > h_p95).mean()),
            "median_ratio_vs_real": float(np.median(v) / np.median(hv)),
            "mean_ratio_vs_real": float(np.mean(v) / np.mean(hv)),
            "median_percentile_in_real": float((hv < np.median(v)).mean() * 100),
        }
    results["overlap_vs_real"] = overlap

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_JSON}")
    _write_md(results)


# ── Per-strategy tournament drafts ───────────────────────────────────
# The round-robin dumps store both teams of every adjudicated draft; a
# strategy's draft set is its side of every pairing it played. The
# symmetrized WP is anti-symmetric under a team swap (p_sym -> 1 - p_sym),
# so the per-draft ensemble variance is side-invariant; we still orient
# each record with the strategy's team as team0 so mean predictions are
# from the strategy's perspective.

RR_DIRS = [os.path.join(RESULTS_DIR, "roundrobin"),
           os.path.join(RESULTS_DIR, "constrained", "roundrobin")]
PER_STRATEGY_CAP = 2000
PER_STRATEGY_SEED = 11


def _load_tournament_records():
    """strategy -> oriented {team0/team1_heroes, game_map, skill_tier} list."""
    files = []
    for d in RR_DIRS:
        files.extend(sorted(glob.glob(os.path.join(d, "*.json"))))
    by_strat = {}
    for fp in files:
        d = json.load(open(fp))
        s0, s1 = d["team0_strategy"], d["team1_strategy"]
        for r in d["records"]:
            t0, t1 = r["team0"]["picks"], r["team1"]["picks"]
            base = {"game_map": r["game_map"], "skill_tier": r["tier"]}
            by_strat.setdefault(s0, []).append(
                dict(base, team0_heroes=t0, team1_heroes=t1))
            by_strat.setdefault(s1, []).append(
                dict(base, team0_heroes=t1, team1_heroes=t0))
    return by_strat, len(files)


def task_per_strategy(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = common.stats_cache()

    missing = [m[0] for m in ROSTER if not os.path.exists(model_path(m[0]))]
    if missing:
        raise RuntimeError(f"missing members: {missing} (run --task pool)")
    if not os.path.exists(OUT_JSON):
        raise RuntimeError("run --task eval first (need real-human baseline)")
    results = json.load(open(OUT_JSON))
    hv = results["sets"]["real"]["variance"]
    h_med, h_p95 = hv["median"], hv["p95"]

    by_strat, n_files = _load_tournament_records()
    print(f"{len(by_strat)} strategies from {n_files} round-robin dumps: "
          f"{sorted(by_strat)}")

    per = {}
    for strat in sorted(by_strat):
        recs = by_strat[strat]
        n_avail = len(recs)
        if n_avail > PER_STRATEGY_CAP:
            rng = np.random.RandomState(PER_STRATEGY_SEED)
            idx = sorted(rng.permutation(n_avail)[:PER_STRATEGY_CAP])
            recs = [recs[i] for i in idx]
        t0 = time.time()
        feats = _extract_set(recs, stats)
        P = np.stack([_member_predict(m[0], feats, device) for m in ROSTER])
        var = P.var(axis=0, ddof=1)
        per[strat] = {
            "n_available": n_avail, "n": int(len(recs)),
            "variance": _dist_stats(var),
            "mean_pred": _dist_stats(P.mean(axis=0)),
            "median_ratio_vs_real": float(np.median(var) / h_med),
            "frac_above_real_median": float((var > h_med).mean()),
            "frac_above_real_p95": float((var > h_p95).mean()),
        }
        print(f"  {strat}: n={len(recs)} (of {n_avail}) "
              f"median_var={np.median(var):.5f} "
              f"ratio={per[strat]['median_ratio_vs_real']:.2f} "
              f">p95={per[strat]['frac_above_real_p95']:.3f} "
              f"({time.time()-t0:.0f}s)")

    results["per_strategy"] = {
        "n_roundrobin_files": n_files, "cap": PER_STRATEGY_CAP,
        "subsample_seed": PER_STRATEGY_SEED,
        "real_baseline": {"median": h_med, "p95": h_p95},
        "note": ("Oriented (strategy = team0); ensemble variance of "
                 "symmetrized WP is side-invariant."),
        "strategies": per,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_JSON}")
    _append_per_strategy_md(results)


def _append_per_strategy_md(r):
    ps = r["per_strategy"]
    header = "## Per-strategy tournament draft variance"
    L = [header, ""]
    A = L.append
    A(f"All 11 tournament strategies' drafted teams from the round-robin "
      f"per-draft records ({ps['n_roundrobin_files']} dump files; each "
      f"strategy's side of every pairing it played, subsampled to "
      f"{ps['cap']} drafts, seed {ps['subsample_seed']}). Baseline: "
      f"held-out human median variance {ps['real_baseline']['median']:.5f}, "
      f"p95 {ps['real_baseline']['p95']:.5f}. Sorted by median ratio.\n")
    A("| strategy | n | median var | ratio vs human median | frac > human p95 |")
    A("| --- | --- | --- | --- | --- |")
    rows = sorted(ps["strategies"].items(),
                  key=lambda kv: -kv[1]["median_ratio_vs_real"])
    for name, s in rows:
        A(f"| {name} | {s['n']} | {s['variance']['median']:.5f} | "
          f"{s['median_ratio_vs_real']:.2f} | "
          f"{s['frac_above_real_p95']:.3f} |")
    A("")
    md = open(OUT_MD).read()
    if header in md:
        md = md[:md.index(header)].rstrip() + "\n\n"
    else:
        md = md.rstrip() + "\n\n"
    with open(OUT_MD, "w") as f:
        f.write(md + "\n".join(L))
    print(f"Saved {OUT_MD}")


def _write_md(r):
    L = []
    A = L.append
    A("# Ensemble epistemic uncertainty of the WP value model\n")
    A(f"20-member ensemble; every member differs in seed, and members "
      f"additionally vary architecture (256x128 vs 512x256x128), feature "
      f"subset (8 full-enriched, 9 minus-one-group, 3 naive 197d), and "
      f"replay-level data half (half_seed={r['half_seed']}). Metric: "
      f"per-draft ensemble variance of symmetrized WP.\n")
    A("## Members\n")
    A("| member | preset | arch | half | seed | test acc |")
    A("| --- | --- | --- | --- | --- | --- |")
    for m in r["members"]:
        acc = f"{m['test_acc']:.2f}%" if m["test_acc"] else "?"
        A(f"| {m['name']} | {m['preset']} | {m['arch']} | {m['half']} | "
          f"{m['seed']} | {acc} |")
    A("\n## Per-draft ensemble variance of symmetrized WP\n")
    A("| set | n | mean var | median var | p90 | p95 | p99 | mean std |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for sname, s in r["sets"].items():
        v, sd = s["variance"], s["std"]
        A(f"| {sname} | {v['n']} | {v['mean']:.5f} | {v['median']:.5f} | "
          f"{v['p90']:.5f} | {v['p95']:.5f} | {v['p99']:.5f} | "
          f"{sd['mean']:.4f} |")
    A("\n## Overlap vs. real-human variance distribution\n")
    A("| set | median ratio | mean ratio | frac > real median | "
      "frac > real p90 | frac > real p95 | median pct-in-real |")
    A("| --- | --- | --- | --- | --- | --- | --- |")
    for sname, o in r["overlap_vs_real"].items():
        A(f"| {sname} | {o['median_ratio_vs_real']:.2f} | "
          f"{o['mean_ratio_vs_real']:.2f} | {o['frac_above_real_median']:.3f} | "
          f"{o['frac_above_real_p90']:.3f} | {o['frac_above_real_p95']:.3f} | "
          f"{o['median_percentile_in_real']:.1f} |")
    A("\n## Degenerate probe comps (vs STANDARD, Cursed Hollow, mid)\n")
    A("| probe | ensemble mean WP | variance | std |")
    A("| --- | --- | --- | --- |")
    for p in r["degen_probe_detail"]:
        A(f"| {p['probe']} | {p['mean_wp']:.4f} | {p['variance']:.5f} | "
          f"{p['std']:.4f} |")
    A("\n## Subgroup variance (median per set)\n")
    A("| subgroup | " + " | ".join(r["sets"].keys()) + " |")
    A("| --- |" + " --- |" * len(r["sets"]))
    for sg, per in r["subgroups"].items():
        A(f"| {sg} | " + " | ".join(f"{per[s]['median']:.5f}"
                                    for s in r["sets"]) + " |")
    A("")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L))
    print(f"Saved {OUT_MD}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True,
                   choices=["pool", "train", "gen-constrained", "eval",
                            "per-strategy"])
    p.add_argument("--member")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--only")
    p.add_argument("--drafts", type=int, default=500)
    args = p.parse_args()
    t0 = time.time()
    {"pool": task_pool, "train": task_train,
     "gen-constrained": task_gen_constrained, "eval": task_eval,
     "per-strategy": task_per_strategy}[args.task](args)
    print(f"task {args.task} done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
