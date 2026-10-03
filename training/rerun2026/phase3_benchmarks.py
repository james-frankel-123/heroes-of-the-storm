"""
Phase 3: re-run every evaluation the paper reports, against the phase1
checkpoints (rerun2026/models/) and the frozen stats snapshot. Outputs JSON
under rerun2026/results/.

Tasks (each honors --dry-run):
  crosseval     Tables II/III + 50-largest-gap analysis + pick divergence:
                480 drafts per WP drafter (naive/herostrength/enriched),
                greedy search, cross-scored by all three models. Uses
                archive/experiment_value_function_quality.gpu_worker across
                all GPUs (run standalone, not inside the pool).
  sanity        21-test and 28-test sanity suites + degenerate-comp WP scores
                for every WP variant (incl. Gourdeau + Siamese).
  wr-sweep      Table V / fig_wr_sweep: 5 eval seeds x 1000 greedy drafts per
                augmentation config (mirrors rerun_wr_sweep.py).
  cql-basic     Table tab:cql_basic: CQL naive alpha sweep, 5 seeds x 1000
                drafts (mirrors rerun_cql_naive_sweep.py).
  rich-eval     Table VII / fig_safety_context: 5 seeds x 1000 drafts per
                strategy with full rich metrics (mirrors
                experiment_rich_evaluation.py).
  cql-hp-eval   Table VIII rows: 200-draft rich metrics per CQL hyperparam
                config (mirrors experiment_cql_hyperparams.py evaluation).
  mcq-eval      MCQ/BC-CQL sweep rich metrics (mirrors experiment_mcq_draft.py).
  draft-quality Resilience / counter / synergy temporal metrics (reuses
                experiment_draft_quality.py functions).
  aggregate     Merge per-shard outputs into the final result JSONs.

Default (no --task): runs crosseval standalone, then the remaining tasks on
the GPU pool, then aggregate.
"""
import os
import sys
import json
import time
import random
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rerun2026 import common
from rerun2026.common import MODELS_DIR, RESULTS_DIR, Job

common.setup()

import numpy as np
import torch
import torch.nn.functional as F

from shared import (is_degenerate, NUM_HEROES, HEROES, HERO_TO_IDX, MAPS,
                    SKILL_TIERS, HERO_ROLE_FINE, map_to_one_hot, tier_to_one_hot)
from sweep_enriched_wp import (WinProbEnrichedModel, FEATURE_GROUPS,
                               compute_group_indices, extract_features)
from experiment_synthetic_augmentation import ENRICHED_GROUPS, make_eval_fn
from test_wp_sanity import TESTS, run_tests

N_ORIGINAL_SANITY = 21   # TESTS[:21] = the paper's initial suite; all 28 = expanded

WR_MODELS = {  # wr-sweep config -> phase1 checkpoint
    "no_aug": "wp_enriched_512.pt",
    "wr0": "wp_aug_wr0_vol100_512.pt",
    "wr5": "wp_aug_wr5_vol100_512.pt",
    "wr10": "wp_aug_wr10_vol100_512.pt",
    "wr50": "wp_aug_wr50_vol100_512.pt",
}
CQL_ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0]
CQL_HP_TAUS = [0.001, 0.005, 0.01, 0.05, 0.1]
CQL_HP_ARCHS = ["1024,512,256", "512,256,128", "256,128,64"]
MCQ_MODELS = ([f"mcq_t{t}" for t in [0.4, 0.5, 0.6, 0.7, 0.8]] +
              [f"bccql_b{b}" for b in [0.1, 0.5, 1.0, 2.0]])
RICH_STRATEGIES = ["gd", "cql_a0.5", "cql_a1.0", "cql_enr_a0.5", "cql_enr_a2.0",
                   "mcq_t0.5", "bccql_b1.0", "gourdeau", "enriched", "enriched_aug"]


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mpath(*p):
    return os.path.join(MODELS_DIR, *p)


def rpath(*p):
    path = os.path.join(RESULTS_DIR, *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _cols(groups):
    gi = compute_group_indices()
    cols = []
    for g in groups:
        s, e = gi[g]
        cols.extend(range(s, e))
    return cols


def load_wp(filename, arch, groups, device):
    cols = _cols(groups)
    model = WinProbEnrichedModel(197 + len(cols), arch, dropout=0.3).to(device)
    model.load_state_dict(torch.load(mpath(filename), weights_only=True,
                                     map_location=device))
    model.eval()
    return model, cols


def load_gd_models(device=torch.device("cpu")):
    from train_generic_draft import GenericDraftModel
    models = []
    for i in range(5):
        gd = GenericDraftModel()
        gd.load_state_dict(torch.load(mpath(f"generic_draft_{i}.pt"),
                                      weights_only=True, map_location="cpu"))
        gd.to(device).eval()
        models.append(gd)
    return models


# ═══════════════════════════════════════════════════════════════
# sanity
# ═══════════════════════════════════════════════════════════════

def task_sanity(args):
    from experiment_synthetic_augmentation import DEGEN_COMPS, STANDARD
    device = _device()
    stats = common.stats_cache()

    specs = {
        "naive": ("wp_naive.pt", [256, 128], []),
        "herostrength": ("wp_herostrength.pt", [256, 128],
                         ["hero_wr", "team_avg_wr"]),
        "enriched_256": ("wp_enriched_256.pt", [256, 128], ENRICHED_GROUPS),
        "enriched_512": ("wp_enriched_512.pt", [512, 256, 128], ENRICHED_GROUPS),
        "true_base": ("wp_true_base.pt", [256, 128], []),
        "aug_v2_512": ("wp_aug_v2_512.pt", [512, 256, 128], ENRICHED_GROUPS),
        "aug_v2_256": ("wp_aug_v2_256.pt", [256, 128], ENRICHED_GROUPS),
        "scope_tier1_only": ("wp_scope_tier1_only_256.pt", [256, 128], ENRICHED_GROUPS),
        "scope_tier2_only": ("wp_scope_tier2_only_256.pt", [256, 128], ENRICHED_GROUPS),
        "scope_both": ("wp_scope_both_256.pt", [256, 128], ENRICHED_GROUPS),
    }
    for cfg in WR_MODELS:
        if cfg != "no_aug":
            specs[f"aug_{cfg}"] = (WR_MODELS[cfg], [512, 256, 128], ENRICHED_GROUPS)

    out = {}
    for name, (fn, arch, groups) in specs.items():
        if not os.path.exists(mpath(fn)):
            print(f"  missing {fn}, skipping {name}")
            continue
        model, cols = load_wp(fn, arch, groups, device)
        eval_fn = make_eval_fn(model, cols, stats, device)
        out[name] = _sanity_for(eval_fn)
        print(f"  {name}: {out[name]['passed_28']}/28 "
              f"({out[name]['passed_21']}/21) 5tank={out[name]['degen_scores']['5 tanks']:.3f}")
        del model
        torch.cuda.empty_cache()

    # Gourdeau reimplementation (map-conditioned, no tier)
    if os.path.exists(mpath("gourdeau_wp.pt")):
        from train_gourdeau_baseline import GourdeauWPModel
        gm = GourdeauWPModel().to(device)
        gm.load_state_dict(torch.load(mpath("gourdeau_wp.pt"), weights_only=True,
                                      map_location=device))
        gm.eval()

        def g_eval(t0h, t1h, game_map="Cursed Hollow", tier="mid"):
            from shared import heroes_to_multi_hot
            t0 = torch.tensor([heroes_to_multi_hot(t0h)]).to(device)
            t1 = torch.tensor([heroes_to_multi_hot(t1h)]).to(device)
            m = torch.tensor([map_to_one_hot(game_map)]).to(device)
            with torch.no_grad():
                return gm.predict_wp(t0, t1, m).item()
        out["gourdeau"] = _sanity_for(g_eval)
        print(f"  gourdeau: {out['gourdeau']['passed_28']}/28")

    # Independent Siamese baseline
    if os.path.exists(mpath("wp_independent_siamese.pt")):
        from archive.experiment_independent_baseline import (
            SiameseWinProbModel, make_siamese_eval_fn)
        sm = SiameseWinProbModel().to(device)
        sm.load_state_dict(torch.load(mpath("wp_independent_siamese.pt"),
                                      weights_only=True, map_location=device))
        sm.eval()
        out["siamese"] = _sanity_for(make_siamese_eval_fn(sm, device))
        print(f"  siamese: {out['siamese']['passed_28']}/28")

    path = rpath("sanity_tests.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved {path}")


def _sanity_for(eval_fn):
    from experiment_synthetic_augmentation import DEGEN_COMPS, STANDARD
    passed, total, results_list = run_tests(eval_fn, verbose=False)
    passed_21 = sum(1 for p in results_list[:N_ORIGINAL_SANITY] if p)
    by_cat = {}
    for t, p in zip(TESTS, results_list):
        c = t.get("category", "")
        by_cat.setdefault(c, [0, 0])
        by_cat[c][1] += 1
        if p:
            by_cat[c][0] += 1
    degen = {name: eval_fn(comp, STANDARD, "Cursed Hollow", "mid")
             for name, comp in DEGEN_COMPS.items()}
    return {"passed_28": passed, "passed_21": passed_21,
            "by_category": {c: f"{a}/{b}" for c, (a, b) in by_cat.items()},
            "degen_scores": degen}


# ═══════════════════════════════════════════════════════════════
# crosseval (Tables II & III, divergence, top-50 gap)
# ═══════════════════════════════════════════════════════════════

def task_crosseval(args):
    """480 greedy drafts per drafter, cross-scored by all three WP models.
    Reuses archive/experiment_value_function_quality.gpu_worker unchanged."""
    import torch.multiprocessing as tmp
    from archive.experiment_value_function_quality import gpu_worker, MODEL_CONFIGS

    n_drafts = args.drafts
    num_gpus = common.NUM_GPUS
    stats = common.stats_cache()
    stats_data = common.stats_as_dict(stats)
    group_indices = compute_group_indices()

    wp_files = {"naive": "wp_naive.pt", "herostrength": "wp_herostrength.pt",
                "enriched": "wp_enriched_256.pt"}
    wp_sds, wp_groups_map = {}, {}
    for name, cfg in MODEL_CONFIGS.items():
        wp_sds[name] = torch.load(mpath(wp_files[name]), weights_only=True,
                                  map_location="cpu")
        wp_groups_map[name] = cfg["groups"]
    gd_sds = [torch.load(mpath(f"generic_draft_{i}.pt"), weights_only=True,
                         map_location="cpu") for i in range(5)]

    random.seed(42)
    configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
               for i in range(n_drafts)]

    result_file = rpath("crosseval_records.json")
    for g in range(num_gpus):  # clear stale shards
        shard = result_file + f".gpu{g}"
        if os.path.exists(shard):
            os.remove(shard)

    ctx = tmp.get_context("spawn")
    procs = []
    for g in range(num_gpus):
        batch = configs[g::num_gpus]
        p = ctx.Process(target=gpu_worker, args=(
            g, batch, wp_sds, wp_groups_map, gd_sds, stats_data,
            group_indices, result_file))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    records = []
    for g in range(num_gpus):
        shard = result_file + f".gpu{g}"
        if os.path.exists(shard):
            with open(shard) as f:
                records.extend(json.load(f))
    with open(result_file, "w") as f:
        json.dump(records, f)
    print(f"{len(records)} draft records -> {result_file}")
    _crosseval_aggregate(records, n_drafts)


def _crosseval_aggregate(records, n_drafts):
    import math
    drafters = ["naive", "herostrength", "enriched"]
    by_drafter = {d: [r for r in records if r["wp_model"] == d] for d in drafters}

    matrix = {}
    for d in drafters:
        recs = by_drafter[d]
        matrix[d] = {f"by_{e}": float(np.mean([r[f"wp_by_{e}"] for r in recs]))
                     for e in drafters}

    comp = {}
    for d in drafters:
        recs = by_drafter[d]
        n = len(recs)
        comp[d] = {
            "n": n,
            "healer": sum(r["comp_has_healer"] for r in recs) / n * 100,
            "frontline": sum(r["comp_has_frontline"] for r in recs) / n * 100,
            "ranged": sum(r["comp_has_ranged_damage"] for r in recs) / n * 100,
            "degen": sum(r["comp_is_absurd"] for r in recs) / n * 100,
        }

    # naive vs enriched healer: chi-square (1 dof) + normal-approx 95% CI
    def chi2_2x2(a_yes, a_no, b_yes, b_no):
        n = a_yes + a_no + b_yes + b_no
        row1, row2 = a_yes + a_no, b_yes + b_no
        col1, col2 = a_yes + b_yes, a_no + b_no
        exp = [row1 * col1 / n, row1 * col2 / n, row2 * col1 / n, row2 * col2 / n]
        obs = [a_yes, a_no, b_yes, b_no]
        stat = sum((o - e) ** 2 / e for o, e in zip(obs, exp) if e > 0)
        return stat, math.erfc(math.sqrt(stat / 2))

    n_n, n_e = comp["naive"]["n"], comp["enriched"]["n"]
    h_n = sum(r["comp_has_healer"] for r in by_drafter["naive"])
    h_e = sum(r["comp_has_healer"] for r in by_drafter["enriched"])
    p1, p2 = h_e / n_e, h_n / n_n
    se = math.sqrt(p1 * (1 - p1) / n_e + p2 * (1 - p2) / n_n)
    stat, pval = chi2_2x2(h_n, n_n - h_n, h_e, n_e - h_e)
    healer_stats = {"diff_pp": (p1 - p2) * 100,
                    "ci95": [(p1 - p2 - 1.96 * se) * 100,
                             (p1 - p2 + 1.96 * se) * 100],
                    "chi2": stat, "p_value": pval}

    # Pick divergence by phase (vs enriched)
    ban_steps, early, late = {0, 1, 2, 3, 9, 10}, {4, 5, 6, 7, 8}, {11, 12, 13, 14, 15}
    by_config = {}
    for r in records:
        key = (r["game_map"], r["skill_tier"], r["our_team"])
        by_config.setdefault(key, {})[r["wp_model"]] = r
    divergence = {}
    for phase, steps in (("bans", ban_steps), ("early_picks", early),
                         ("late_picks", late)):
        ne = he = tot = 0
        for drafter_recs in by_config.values():
            if "naive" not in drafter_recs or "enriched" not in drafter_recs:
                continue
            ns = {s["step"]: s["chosen_hero"] for s in drafter_recs["naive"]["steps"]}
            es = {s["step"]: s["chosen_hero"] for s in drafter_recs["enriched"]["steps"]}
            hs = {s["step"]: s["chosen_hero"]
                  for s in drafter_recs.get("herostrength", {}).get("steps", [])}
            for st in steps:
                if st in ns and st in es:
                    tot += 1
                    ne += ns[st] == es[st]
                    he += hs.get(st) == es[st]
        divergence[phase] = {"naive_vs_enriched": ne / tot if tot else None,
                             "herostr_vs_enriched": he / tot if tot else None}

    # Top-50 naive-vs-enriched gap analysis
    naive_sorted = sorted(by_drafter["naive"],
                          key=lambda r: r["wp_by_naive"] - r["wp_by_enriched"],
                          reverse=True)
    top50 = naive_sorted[:50]
    reasons = {"no_healer": 0, "no_frontline": 0, "role_stacking": 0,
               "high_wr_bad_comp": 0}
    for r in top50:
        if not r["comp_has_healer"]:
            reasons["no_healer"] += 1
        if not r["comp_has_frontline"]:
            reasons["no_frontline"] += 1
        if any(v >= 3 for v in r["comp_roles"].values()):
            reasons["role_stacking"] += 1
        if r["comp_is_absurd"]:
            reasons["high_wr_bad_comp"] += 1

    out = {"n_drafts_per_strategy": n_drafts, "matrix": matrix,
           "composition": comp, "healer_naive_vs_enriched": healer_stats,
           "divergence": divergence, "top50_gap_reasons": reasons,
           "top50_cases": [{k: v for k, v in r.items() if k != "steps"}
                           for r in top50]}
    path = rpath("crosseval.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved {path}")
    print("Cross-evaluation matrix:")
    for d in drafters:
        print(f"  {d:<14} " + "  ".join(f"{matrix[d][f'by_{e}']:.4f}" for e in drafters))


# ═══════════════════════════════════════════════════════════════
# wr-sweep (Table V)
# ═══════════════════════════════════════════════════════════════

def task_wr_sweep(args):
    """One augmentation config: sanity eval + 5 seeds x 1000 greedy drafts.
    Mirrors rerun_wr_sweep.py (GD-argmax rollouts and opponents)."""
    from train_draft_policy import DraftState, DRAFT_ORDER
    from experiment_synthetic_augmentation import evaluate_config

    cfg = args.config
    device = _device()
    stats = common.stats_cache()
    model, cols = load_wp(WR_MODELS[cfg], [512, 256, 128], ENRICHED_GROUPS, device)
    gd_models = load_gd_models(device)
    eval_fn = make_eval_fn(model, cols, stats, device)
    stage1 = evaluate_config(model, cols, stats, device)  # 28-test suite

    healer_heroes = set(h for h, r in HERO_ROLE_FINE.items() if r == "healer")
    ranged_heroes = set(h for h, r in HERO_ROLE_FINE.items()
                        if r in ("ranged_aa", "ranged_mage", "pusher"))

    def gd_pick(state):
        gd = random.choice(gd_models)
        x = state.to_tensor_gd(device)
        mask = state.valid_mask(device)
        with torch.no_grad():
            return gd(x, mask).argmax(dim=1).item()

    def greedy_wp_pick(state, step_team, our_team):
        mask_np = state.valid_mask_np()
        best_idx, best_wp = None, -1
        for hero_idx in [i for i in range(NUM_HEROES) if mask_np[i] > 0]:
            s = state.clone()
            s.apply_action(hero_idx, step_team, "pick")
            for rs_team, rs_type in DRAFT_ORDER[s.step:]:
                s.apply_action(gd_pick(s), rs_team, rs_type)
            t0h = [HEROES[i] for i in range(NUM_HEROES) if s.team0_picks[i] > 0]
            t1h = [HEROES[i] for i in range(NUM_HEROES) if s.team1_picks[i] > 0]
            wp = eval_fn(t0h, t1h, state.game_map, state.skill_tier)
            if our_team == 1:
                wp = 1 - wp
            if wp > best_wp:
                best_wp, best_idx = wp, hero_idx
        return best_idx

    seed_results = []
    for seed in range(args.seeds):
        random.seed(seed)
        torch.manual_seed(seed)
        draft_configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
                         for i in range(args.drafts)]
        counts = {"healer": 0, "ranged": 0, "degen": 0}
        for di, (_, game_map, tier, our_team) in enumerate(draft_configs):
            state = DraftState(game_map, tier, our_team=our_team)
            while not state.is_terminal():
                step_team, step_type = DRAFT_ORDER[state.step]
                if step_team == our_team and step_type == "pick":
                    hero_idx = greedy_wp_pick(state, step_team, our_team)
                else:
                    hero_idx = gd_pick(state)
                state.apply_action(hero_idx, step_team, step_type)
            vec = state.team0_picks if our_team == 0 else state.team1_picks
            picks = [HEROES[i] for i in range(NUM_HEROES) if vec[i] > 0]
            counts["healer"] += any(h in healer_heroes for h in picks)
            counts["ranged"] += any(h in ranged_heroes for h in picks)
            counts["degen"] += is_degenerate(picks)
            if (di + 1) % 200 == 0:
                print(f"  [{cfg}] seed {seed} {di+1}/{args.drafts}: "
                      f"healer={counts['healer']/(di+1)*100:.1f}% "
                      f"degen={counts['degen']/(di+1)*100:.1f}%", flush=True)
        seed_results.append({k: v / args.drafts * 100 for k, v in counts.items()}
                            | {"seed": seed})

    result = {
        "name": cfg, "model": WR_MODELS[cfg],
        "accuracy": _meta_acc(WR_MODELS[cfg]),
        "sanity_passed_28": stage1["sanity_passed"],
        "sanity_total": stage1["sanity_total"],
        "degen_scores": stage1["degen_scores"],
        "healer_rate": float(np.mean([s["healer"] for s in seed_results])),
        "ranged_rate": float(np.mean([s["ranged"] for s in seed_results])),
        "degen_rate": float(np.mean([s["degen"] for s in seed_results])),
        "healer_std": float(np.std([s["healer"] for s in seed_results])),
        "degen_std": float(np.std([s["degen"] for s in seed_results])),
        "seed_results": seed_results,
        "total_drafts": args.seeds * args.drafts,
    }
    path = rpath("wr_sweep", f"{cfg}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved {path}")


def _meta_acc(model_file):
    meta = os.path.join(MODELS_DIR, "meta",
                        os.path.basename(model_file).replace(".pt", ".json"))
    if os.path.exists(meta):
        with open(meta) as f:
            return json.load(f).get("best_acc")
    return None


# ═══════════════════════════════════════════════════════════════
# cql-basic (tab:cql_basic)
# ═══════════════════════════════════════════════════════════════

def task_cql_basic(args):
    """CQL naive alpha sweep composition metrics, 5 seeds x 1000 drafts.
    Mirrors rerun_cql_naive_sweep.py against the rerun2026 checkpoints."""
    from train_draft_policy import DraftState, DRAFT_ORDER
    from experiment_cql_draft import CQLDraftAgent

    device = _device()
    gd_models = load_gd_models(torch.device("cpu"))
    healer_heroes = set(h for h, r in HERO_ROLE_FINE.items() if r == "healer")
    tank_heroes = set(h for h, r in HERO_ROLE_FINE.items() if r == "tank")
    bruiser = set(h for h, r in HERO_ROLE_FINE.items() if r == "bruiser")
    ranged = set(h for h, r in HERO_ROLE_FINE.items()
                 if r in ("ranged_aa", "ranged_mage", "pusher"))
    frontline = tank_heroes | bruiser

    random.seed(42)
    draft_configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
                     for i in range(args.drafts)]

    all_results = []
    for alpha in CQL_ALPHAS:
        model = CQLDraftAgent().to(device)
        model.load_state_dict(torch.load(mpath("cql", f"_cql_temp_a{alpha}.pt"),
                                         weights_only=True, map_location=device))
        model.eval()

        def cql_pick(state, step_type):
            s = np.concatenate([
                state.team0_picks, state.team1_picks, state.bans,
                map_to_one_hot(state.game_map), tier_to_one_hot(state.skill_tier),
                [state.step / 15.0, 0.0 if step_type == "ban" else 1.0]])
            s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
            m_t = torch.tensor(state.valid_mask_np(), dtype=torch.float32
                               ).unsqueeze(0).to(device)
            with torch.no_grad():
                return model(s_t, m_t).squeeze(0).argmax().item()

        seed_results = []
        for seed in range(args.seeds):
            random.seed(seed)
            torch.manual_seed(seed)
            counts = Counter()
            for _, game_map, tier, our_team in draft_configs:
                state = DraftState(game_map, tier, our_team=our_team)
                while not state.is_terminal():
                    step_team, step_type = DRAFT_ORDER[state.step]
                    if step_team == our_team:
                        hero_idx = cql_pick(state, step_type)
                    else:
                        gd = random.choice(gd_models)
                        x = state.to_tensor_gd(torch.device("cpu"))
                        mask = state.valid_mask(torch.device("cpu"))
                        with torch.no_grad():
                            probs = F.softmax(gd(x, mask), dim=1)
                            hero_idx = torch.multinomial(probs, 1).item()
                    state.apply_action(hero_idx, step_team, step_type)
                vec = state.team0_picks if our_team == 0 else state.team1_picks
                picks = [HEROES[i] for i in range(NUM_HEROES) if vec[i] > 0]
                counts["healer"] += any(h in healer_heroes for h in picks)
                counts["frontline"] += any(h in frontline for h in picks)
                counts["ranged"] += any(h in ranged for h in picks)
                counts["degen"] += is_degenerate(picks)
            seed_results.append({k: counts[k] / args.drafts * 100
                                 for k in ("healer", "frontline", "ranged", "degen")}
                                | {"seed": seed})
            print(f"  a={alpha} seed {seed}: healer={seed_results[-1]['healer']:.1f}% "
                  f"degen={seed_results[-1]['degen']:.1f}%", flush=True)

        result = {"alpha": alpha, "seed_results": seed_results,
                  "total_drafts": args.seeds * args.drafts}
        for k in ("healer", "frontline", "ranged", "degen"):
            result[f"{k}_rate"] = float(np.mean([s[k] for s in seed_results]))
            result[f"{k}_std"] = float(np.std([s[k] for s in seed_results]))
        all_results.append(result)
        del model
        torch.cuda.empty_cache()

    path = rpath("cql_naive_5x1000.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════
# rich-eval (Table VII)
# ═══════════════════════════════════════════════════════════════

def _rich_strategy(name, device, stats, group_indices):
    """Build one Table VII strategy from rerun2026 checkpoints, reusing the
    factories in experiment_rich_evaluation.py."""
    from experiment_rich_evaluation import (
        make_gd_strategy, make_cql_strategy, make_wp_greedy_strategy,
        make_cql_enriched_strategy, make_gourdeau_greedy_strategy)

    if name == "gd":
        return "GD baseline", make_gd_strategy()
    if name.startswith("cql_a"):
        a = name.split("cql_a")[1]
        return f"CQL a={a}", make_cql_strategy(mpath("cql", f"_cql_temp_a{a}.pt"), device)
    if name.startswith("cql_enr_a"):
        a = name.split("cql_enr_a")[1]
        return f"CQL enr a={a}", make_cql_enriched_strategy(
            mpath("cql", f"_cql_enriched_a{a}.pt"), device, stats,
            group_indices, ENRICHED_GROUPS)
    if name.startswith("mcq_t"):
        t = name.split("mcq_t")[1]
        return f"MCQ t={t}", make_cql_strategy(mpath("mcq", f"_mcq_temp_t{t}.pt"), device)
    if name.startswith("bccql_b"):
        b = name.split("bccql_b")[1]
        return f"BC-CQL b={b}", make_cql_strategy(
            mpath("mcq", f"_bc_cql_temp_bc{b}.pt"), device)
    if name == "gourdeau":
        from train_gourdeau_baseline import GourdeauWPModel
        gm = GourdeauWPModel().to(device)
        gm.load_state_dict(torch.load(mpath("gourdeau_wp.pt"), weights_only=True,
                                      map_location=device))
        gm.eval()
        return "Gourdeau (greedy)", make_gourdeau_greedy_strategy(gm, device)
    if name == "enriched":
        model, _ = load_wp("wp_enriched_256.pt", [256, 128], ENRICHED_GROUPS, device)
        return "Enriched WP", make_wp_greedy_strategy(
            model, ENRICHED_GROUPS, stats, group_indices, device)
    if name == "enriched_aug":
        model, _ = load_wp("wp_aug_v2_512.pt", [512, 256, 128], ENRICHED_GROUPS, device)
        return "Enriched+aug", make_wp_greedy_strategy(
            model, ENRICHED_GROUPS, stats, group_indices, device)
    raise ValueError(name)


def task_rich_eval(args):
    """One strategy: 5 seeds x 1000 drafts + full Table VII metrics."""
    from experiment_rich_evaluation import (
        run_drafts_with_strategy, counter_responsiveness, synergy_exploitation,
        draft_diversity, map_adaptation)

    device = _device()
    stats = common.stats_cache()
    group_indices = compute_group_indices()
    gd_models = load_gd_models(torch.device("cpu"))
    label, strategy_fn = _rich_strategy(args.strategy, device, stats, group_indices)

    random.seed(42)
    draft_configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
                     for i in range(args.drafts)]

    combined, seed_batches = [], []
    for seed in range(args.seeds):
        random.seed(seed)
        torch.manual_seed(seed)
        drafts = run_drafts_with_strategy(strategy_fn, draft_configs, gd_models,
                                          stats, device)
        combined.extend(drafts)
        seed_batches.append(drafts)
        print(f"  [{label}] seed {seed} done", flush=True)

    n = len(combined)
    healer_rate = sum(d["has_healer"] for d in combined) / n * 100
    degen_rate = sum(d["is_degen"] for d in combined) / n * 100
    counters = [counter_responsiveness(d["our_picks"], d["opp_picks"], stats, d["tier"])
                for d in combined]
    synergies = [synergy_exploitation(d["our_picks"], stats, d["tier"])
                 for d in combined]
    div = draft_diversity(combined)

    # GD similarity (consensus of the 5 GD models at our pick steps)
    gd_agree = gd_total = 0
    for draft in combined:
        for step in draft["steps"]:
            if not step["is_ours"] or step["type"] == "ban":
                continue
            s_t = torch.tensor(step["state"], dtype=torch.float32).unsqueeze(0)
            m_t = torch.tensor(step["mask"], dtype=torch.float32).unsqueeze(0)
            votes = Counter()
            for gd in gd_models:
                with torch.no_grad():
                    votes[gd(s_t, m_t).argmax(dim=1).item()] += 1
            gd_agree += step["hero_idx"] == votes.most_common(1)[0][0]
            gd_total += 1

    # Cross-model WP: augmented (v2_512) model scores this strategy's drafts.
    # Optional: skipped when the namespace has no augmented model (the
    # paper-1 site-tier rebuild trains none), and then no aug_wp is written.
    if args.strategy != "enriched_aug" and not os.path.exists(mpath("wp_aug_v2_512.pt")):
        aug_wp = None
    elif args.strategy != "enriched_aug":
        aug_model, aug_cols = load_wp("wp_aug_v2_512.pt", [512, 256, 128],
                                      ENRICHED_GROUPS, device)
        all_mask = [True] * len(FEATURE_GROUPS)
        wps = []
        for d in combined:
            t0h = [HEROES[i] for i in range(NUM_HEROES) if d["terminal_t0"][i] > 0]
            t1h = [HEROES[i] for i in range(NUM_HEROES) if d["terminal_t1"][i] > 0]
            rec = {"team0_heroes": t0h, "team1_heroes": t1h,
                   "game_map": d["game_map"], "skill_tier": d["tier"], "winner": 0}
            base, enr = extract_features(rec, stats, all_mask)
            x = np.concatenate([base, enr[aug_cols]])
            with torch.no_grad():
                wp = aug_model(torch.tensor(x, dtype=torch.float32
                                            ).unsqueeze(0).to(device)).item()
            wps.append(1 - wp if d["our_team"] == 1 else wp)
        aug_wp = float(np.mean(wps))
    else:
        aug_wp = "self"

    seed_metrics = []
    for batch in seed_batches:
        bn = len(batch)
        seed_metrics.append({
            "degen": sum(d["is_degen"] for d in batch) / bn * 100,
            "counter": float(np.mean([counter_responsiveness(
                d["our_picks"], d["opp_picks"], stats, d["tier"]) for d in batch])),
            "synergy": float(np.mean([synergy_exploitation(
                d["our_picks"], stats, d["tier"]) for d in batch])),
        })

    result = {
        "strategy": label, "slug": args.strategy,
        "seeds": args.seeds, "drafts_per_seed": args.drafts,
        "metrics": {
            "healer_rate": round(healer_rate, 1),
            "degen_rate": round(degen_rate, 1),
            "counter": round(float(np.mean(counters)), 2),
            "synergy": round(float(np.mean(synergies)), 2),
            **div,
            "gd_similarity": round(gd_agree / gd_total * 100, 1) if gd_total else 0,
            **({} if aug_wp is None else
               {"aug_wp": round(aug_wp, 3) if isinstance(aug_wp, float) else aug_wp}),
            "map_adapt": map_adaptation(combined),
        },
        "seed_metrics": seed_metrics,
        "degen_std": float(np.std([m["degen"] for m in seed_metrics])),
        "counter_std": float(np.std([m["counter"] for m in seed_metrics])),
        "synergy_std": float(np.std([m["synergy"] for m in seed_metrics])),
    }
    path = rpath("rich_eval", f"{args.strategy}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════
# cql-hp-eval (Table VIII) & mcq-eval
# ═══════════════════════════════════════════════════════════════

def task_cql_hp_eval(args):
    from experiment_cql_hyperparams import (FlexCQLAgent, run_drafts_rich,
                                            compute_rich_metrics)
    device = _device()
    stats = common.stats_cache()
    gd_models = load_gd_models(torch.device("cpu"))
    hidden = tuple(int(x) for x in args.arch.split(","))
    arch_str = "x".join(str(d) for d in hidden)
    model = FlexCQLAgent(hidden_dims=hidden).to(device)
    model.load_state_dict(torch.load(
        mpath("cql_hyperparams", f"_cql_temp_a1.0_t{args.tau}_{arch_str}.pt"),
        weights_only=True, map_location=device))
    model.eval()

    random.seed(42)
    draft_configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
                     for i in range(args.drafts)]
    random.seed(42)
    drafts = run_drafts_rich(model, device, draft_configs, gd_models, stats, hidden)
    rich = compute_rich_metrics(drafts, gd_models, stats)
    result = {"alpha": 1.0, "tau": args.tau, "hidden_dims": list(hidden),
              "drafts": args.drafts, "rich_metrics": rich}
    path = rpath("cql_hyperparams", f"a1.0_t{args.tau}_{arch_str}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved {path}: {rich}")


def task_mcq_eval(args):
    from experiment_cql_draft import CQLDraftAgent
    from experiment_mcq_draft import run_drafts_rich, compute_rich_metrics
    device = _device()
    stats = common.stats_cache()
    gd_models = load_gd_models(torch.device("cpu"))
    name = args.model
    if name.startswith("mcq_t"):
        ckpt = mpath("mcq", f"_mcq_temp_t{name.split('mcq_t')[1]}.pt")
    else:
        ckpt = mpath("mcq", f"_bc_cql_temp_bc{name.split('bccql_b')[1]}.pt")
    model = CQLDraftAgent().to(device)
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
    model.eval()

    random.seed(42)
    draft_configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
                     for i in range(args.drafts)]
    random.seed(42)
    drafts = run_drafts_rich(model, device, draft_configs, gd_models, stats)
    rich = compute_rich_metrics(drafts, gd_models, stats)
    result = {"model": name, "drafts": args.drafts, "rich_metrics": rich}
    path = rpath("mcq", f"{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved {path}: {rich}")


# ═══════════════════════════════════════════════════════════════
# draft-quality (resilience metrics)
# ═══════════════════════════════════════════════════════════════

def task_draft_quality(args):
    """Temporal draft-quality metrics (resilience/counter/synergy), reusing
    experiment_draft_quality.py functions against rerun2026 models."""
    import experiment_draft_quality as edq
    from sweep_enriched_wp import FEATURE_GROUP_DIMS

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    stats = common.stats_cache()
    wp_cols = _cols(ENRICHED_GROUPS)
    all_mask = [True] * len(FEATURE_GROUPS)

    gd_models = load_gd_models(torch.device("cpu"))
    gd = gd_models[0]
    wp_enr, _ = load_wp("wp_enriched_256.pt", [256, 128], ENRICHED_GROUPS,
                        torch.device("cpu"))
    wp_aug, _ = load_wp("wp_aug_v2_512.pt", [512, 256, 128], ENRICHED_GROUPS,
                        torch.device("cpu"))

    strategies = {
        "GD baseline": edq.make_gd_strategy(gd, sample=False),
        "Enriched greedy": edq.make_wp_greedy_strategy(wp_enr, stats, wp_cols, all_mask),
        "Augmented greedy": edq.make_wp_greedy_strategy(wp_aug, stats, wp_cols, all_mask),
    }
    if args.include_mcts:
        from train_draft_policy import AlphaZeroDraftNet
        for run in args.mcts_runs.split(","):
            p = os.path.join(common.TRAINING_DIR, "mcts_runs", run, "draft_policy.pt")
            if not os.path.exists(p):
                print(f"  (historical MCTS checkpoint missing: {p})")
                continue
            net = AlphaZeroDraftNet()
            sd = torch.load(p, weights_only=True, map_location="cpu")
            if any(k.startswith("res_block1.") for k in sd):
                sd = {k.replace("res_block1.", "res_blocks.0.")
                       .replace("res_block2.", "res_blocks.1.")
                       .replace("res_block3.", "res_blocks.2."): v
                      for k, v in sd.items()}
            net.load_state_dict(sd)
            net.eval()
            strategies[f"MCTS {run}"] = edq.make_policy_strategy(net)

    gd_opp = edq.make_gd_opponent(gd)
    configs = [(random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
               for i in range(args.drafts)]

    results = {}
    for strat_name, strat_fn in strategies.items():
        t0 = time.time()
        all_metrics = []
        for game_map, tier, our_team in configs:
            pick_steps, valid_at_step = edq.simulate_draft(
                strat_fn, gd_opp, game_map, tier, our_team, stats)
            all_metrics.append(edq.full_draft_quality(pick_steps, stats, tier,
                                                      valid_at_step))
        agg = {}
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics]
            agg[key] = float(np.mean(vals))
            agg[key + "_std"] = float(np.std(vals))
        results[strat_name] = agg
        print(f"  {strat_name} ({time.time()-t0:.0f}s): "
              f"R_early={agg['resilience_early']:+.3f} "
              f"R_late={agg['resilience_late']:+.3f} "
              f"synergy={agg['team_synergy']:+.3f}", flush=True)

    path = rpath("draft_quality_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════
# aggregate
# ═══════════════════════════════════════════════════════════════

def task_aggregate(args):
    def merge(subdir, out_name, order=None):
        d = os.path.join(RESULTS_DIR, subdir)
        if not os.path.isdir(d):
            print(f"  (no {subdir}/ results yet)")
            return None
        items = {}
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json"):
                with open(os.path.join(d, fn)) as f:
                    items[fn[:-5]] = json.load(f)
        if order:
            items = {k: items[k] for k in order if k in items} | \
                    {k: v for k, v in items.items() if not order or k not in order}
        path = rpath(out_name)
        with open(path, "w") as f:
            json.dump(items, f, indent=2)
        print(f"  merged {len(items)} -> {path}")
        return items

    wr = merge("wr_sweep", "wr_sweep_5x1000.json",
               order=["no_aug", "wr0", "wr5", "wr10", "wr50"])
    rich = merge("rich_eval", "rich_evaluation_results.json",
                 order=RICH_STRATEGIES)
    merge("cql_hyperparams", "cql_hyperparam_results_a1.0.json")
    merge("mcq", "mcq_results.json")

    if wr:
        print("\nTable V (WR sweep):")
        print(f"{'Config':<10} {'Acc%':>7} {'Healer%':>10} {'Ranged%':>8} "
              f"{'Degen%':>10} {'Sanity':>7}")
        for k, r in wr.items():
            acc = r.get("accuracy")
            print(f"{k:<10} {acc if acc else '---':>7} "
                  f"{r['healer_rate']:>5.1f}±{r['healer_std']:<4.1f} "
                  f"{r['ranged_rate']:>7.1f} "
                  f"{r['degen_rate']:>5.1f}±{r['degen_std']:<4.1f} "
                  f"{r['sanity_passed_28']:>4}/28")
    if rich:
        print("\nTable VII (rich evaluation):")
        print(f"{'Strategy':<20} {'Heal%':>6} {'Deg%':>6} {'Ctr':>6} {'Syn':>6} "
              f"{'Dist':>5} {'Ent':>5} {'T10%':>6} {'GD%':>6} {'AugWP':>6}")
        for k, r in rich.items():
            m = r["metrics"]
            aug = m.get("aug_wp", "-")
            print(f"{r['strategy']:<20} {m['healer_rate']:>6.1f} {m['degen_rate']:>6.1f} "
                  f"{m['counter']:>+6.2f} {m['synergy']:>+6.2f} "
                  f"{m['distinct_heroes']:>5} {m['entropy']:>5.2f} "
                  f"{m['top10_concentration']:>6.1f} {m['gd_similarity']:>6.1f} "
                  f"{aug if isinstance(aug, str) else f'{aug:.3f}':>6}")


# ═══════════════════════════════════════════════════════════════
# orchestrator
# ═══════════════════════════════════════════════════════════════

def build_pool_jobs(args):
    me = os.path.abspath(__file__)
    jobs = []

    def add(name, argv, outputs, weight):
        jobs.append(Job(name, [me] + argv, outputs, weight=weight))

    add("p3_sanity", ["--task", "sanity"], [rpath("sanity_tests.json")], "light")
    add("p3_cql_basic", ["--task", "cql-basic"], [rpath("cql_naive_5x1000.json")],
        "light")
    add("p3_draft_quality", ["--task", "draft-quality"],
        [rpath("draft_quality_results.json")], "light")
    for cfg in WR_MODELS:
        add(f"p3_wr_{cfg}", ["--task", "wr-sweep", "--config", cfg],
            [rpath("wr_sweep", f"{cfg}.json")], "heavy")
    for s in RICH_STRATEGIES:
        weight = "heavy" if s in ("enriched", "enriched_aug", "gourdeau") else "light"
        add(f"p3_rich_{s}", ["--task", "rich-eval", "--strategy", s],
            [rpath("rich_eval", f"{s}.json")], weight)
    for arch in CQL_HP_ARCHS:
        arch_str = arch.replace(",", "x")
        for tau in CQL_HP_TAUS:
            add(f"p3_cqlhp_t{tau}_{arch_str}",
                ["--task", "cql-hp-eval", "--tau", str(tau), "--arch", arch],
                [rpath("cql_hyperparams", f"a1.0_t{tau}_{arch_str}.json")], "light")
    for m in MCQ_MODELS:
        add(f"p3_mcqeval_{m}", ["--task", "mcq-eval", "--model", m],
            [rpath("mcq", f"{m}.json")], "light")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None,
                        choices=[None, "sanity", "crosseval", "wr-sweep",
                                 "cql-basic", "rich-eval", "cql-hp-eval",
                                 "mcq-eval", "draft-quality", "aggregate"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--drafts", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--config", default=None, help="wr-sweep config name")
    parser.add_argument("--strategy", default=None, help="rich-eval strategy slug")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--model", default=None, help="mcq-eval model name")
    parser.add_argument("--include-mcts", action="store_true",
                        help="draft-quality: include historical mcts_runs policies")
    parser.add_argument("--mcts-runs", default="E_1M_s0,J_800sim_s0",
                        help="comma-separated historical MCTS run names")
    args = parser.parse_args()

    defaults = {"crosseval": 480, "wr-sweep": 1000, "cql-basic": 1000,
                "rich-eval": 1000, "cql-hp-eval": 200, "mcq-eval": 200,
                "draft-quality": 200}
    if args.drafts is None:
        args.drafts = defaults.get(args.task, 480)

    if args.task is None:
        # full phase 3: crosseval standalone (uses all GPUs internally),
        # then the pool, then aggregate
        jobs = build_pool_jobs(args)
        if args.dry_run:
            print("Phase 3 dry run. Plan:")
            print("  1. crosseval (standalone, all GPUs, 480 drafts x 3 drafters)")
            common.run_pool(jobs, dry_run=True, force=args.force, only=args.only)
            print("  3. aggregate")
            return
        crosseval_out = rpath("crosseval.json")
        if args.force or not os.path.exists(crosseval_out):
            args.drafts = defaults["crosseval"]
            task_crosseval(args)
        failed = common.run_pool(jobs, force=args.force, only=args.only)
        task_aggregate(args)
        if failed:
            sys.exit(1)
        return

    if args.dry_run:
        print(f"dry run: task={args.task} {vars(args)}")
        return

    fn = {"sanity": task_sanity, "crosseval": task_crosseval,
          "wr-sweep": task_wr_sweep, "cql-basic": task_cql_basic,
          "rich-eval": task_rich_eval, "cql-hp-eval": task_cql_hp_eval,
          "mcq-eval": task_mcq_eval, "draft-quality": task_draft_quality,
          "aggregate": task_aggregate}[args.task]
    t0 = time.time()
    fn(args)
    print(f"task {args.task} done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
