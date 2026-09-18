"""
Phase 3c: "queue constrained search" — hard role-constraint masking as a
strategy wrapper, applied to enriched-WP greedy search and to the MCTS
policy (rerun2026/mcts_runs/J_800sim_s9), then evaluated with the exact
phase 3 machinery: rich evaluation (5 seeds x 1000 drafts) and round-robin
tournament pairs against the existing 9 phase3b strategies.

CONSTRAINT RULE (exact):
At each of OUR pick steps, a candidate hero h is masked out iff picking h
would make a structurally valid final composition unreachable. "Reachable"
means: there exists some way to fill the team's remaining pick slots from the
heroes currently available (not picked by either team, not banned; future
opponent removals are NOT modeled) such that the final five-hero team
satisfies all of:
  (1) at least 1 Healer;
  (2) at least 1 frontline hero (Blizzard Tank or Bruiser; Varian counts as
      Bruiser per shared.FINE_TO_BLIZZ_ROLE);
  (3) at least 1 ranged-damage hero (fine roles ranged_aa / ranged_mage /
      pusher, i.e. Blizzard Ranged Assassin);
  (4) at most 2 heroes in each stack-bad Blizzard role — Tank, Melee
      Assassin, Support, Healer (shared.DEGEN_STACK_ROLES).
Roles come from shared.HERO_ROLE_FINE composed with shared.FINE_TO_BLIZZ_ROLE,
so the constraint depends only on Blizzard-role counts, and reachability is
decided EXACTLY by enumerating role-count completions bounded by per-role
pool availability (depth <= 4, 6 roles; memoized).

Relation to the paper's safety metrics: (1)+(2)+(4) imply
shared.is_degenerate(team) == False — we deliberately do not use the
is_degenerate Uther-as-frontline exception, so the constraint is (slightly)
stricter — and (3) additionally enforces the no-ranged-damage criterion from
the paper's degenerate-composition definition. Bans and opponent steps are
never constrained. If NO available pick is feasible (possible only in
constructed states: under constrained play from an empty team, feasibility is
maintained inductively as long as the pool suffices), the wrapper falls back
to the plain availability mask and increments FALLBACKS.

Usage:
  python rerun2026/constrained_search.py --task smoke              # CPU logic check
  python rerun2026/constrained_search.py --dry-run                 # show plan
  python rerun2026/constrained_search.py                           # pool: rich-eval + pairs + aggregate
  python rerun2026/constrained_search.py --task rich-eval --strategy constrained_mcts
  python rerun2026/constrained_search.py --task pair --pair constrained_greedy__gd
  python rerun2026/constrained_search.py --task aggregate
"""
import os
import sys
import json
import time
import random
import argparse
import types
from collections import Counter
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rerun2026 import common
from rerun2026.common import MODELS_DIR, RESULTS_DIR, Job

common.setup()

import numpy as np
import torch

from shared import (is_degenerate, NUM_HEROES, HEROES, HERO_ROLE_FINE,
                    FINE_TO_BLIZZ_ROLE, DEGEN_STACK_ROLES, MAPS, SKILL_TIERS)
from sweep_enriched_wp import (WinProbEnrichedModel, FEATURE_GROUPS,
                               compute_group_indices, extract_features)
from experiment_synthetic_augmentation import ENRICHED_GROUPS
from train_draft_policy import DraftState, DRAFT_ORDER, AlphaZeroDraftNet
from rerun2026 import phase3b_roundrobin as p3b

MCTS_CHECKPOINT = os.path.join(common.MCTS_RUNS_DIR, "J_800sim_s9",
                               "draft_policy.pt")
NEW_STRATEGIES = ["constrained_greedy", "constrained_mcts"]
FALLBACKS = Counter()   # diagnostics: how often the feasibility mask emptied

# ── The constraint ──────────────────────────────────────────────────

BLIZZ_ROLES = ("Tank", "Bruiser", "Healer", "Ranged Assassin",
               "Melee Assassin", "Support")
ROLE_IDX = {r: i for i, r in enumerate(BLIZZ_ROLES)}
ROLE_CAP = tuple(2 if r in DEGEN_STACK_ROLES else 5 for r in BLIZZ_ROLES)
HERO_BLIZZ_IDX = np.array(
    [ROLE_IDX[FINE_TO_BLIZZ_ROLE.get(HERO_ROLE_FINE.get(h, ""),
                                     "Ranged Assassin")] for h in HEROES],
    dtype=np.int64)


def _valid_final(counts):
    """counts: per-BLIZZ_ROLES tuple for a complete 5-hero team."""
    if any(c > cap for c, cap in zip(counts, ROLE_CAP)):
        return False
    healer = counts[ROLE_IDX["Healer"]] >= 1
    front = counts[ROLE_IDX["Tank"]] + counts[ROLE_IDX["Bruiser"]] >= 1
    ranged = counts[ROLE_IDX["Ranged Assassin"]] >= 1
    return healer and front and ranged


@lru_cache(maxsize=None)
def _feasible(cur, avail, k):
    """Can `cur` role counts be completed with k picks from `avail` role
    availability into a _valid_final composition? Exact (exhaustive over
    role-count completions with memoization)."""
    if any(c > cap for c, cap in zip(cur, ROLE_CAP)):
        return False
    if k == 0:
        return _valid_final(cur)
    for i in range(len(BLIZZ_ROLES)):
        if avail[i] == 0 or cur[i] + 1 > ROLE_CAP[i]:
            continue
        c = list(cur); c[i] += 1
        a = list(avail); a[i] -= 1
        # clip availability to remaining depth to keep the memo table small
        a = [min(x, k - 1) for x in a]
        if _feasible(tuple(c), tuple(a), k - 1):
            return True
    return False


def role_constraint_mask(state, team):
    """Constrained pick mask for `team` at the current step: availability
    mask AND'ed with per-candidate reachability of a valid final comp.
    Falls back to the plain availability mask if nothing is feasible."""
    base = state.valid_mask_np()
    team_vec = state.team0_picks if team == 0 else state.team1_picks
    cur = [0] * len(BLIZZ_ROLES)
    for i in range(NUM_HEROES):
        if team_vec[i] > 0.5:
            cur[HERO_BLIZZ_IDX[i]] += 1
    slots_after = 5 - int(round(float(team_vec.sum()))) - 1
    avail = [0] * len(BLIZZ_ROLES)
    for i in range(NUM_HEROES):
        if base[i] > 0.5:
            avail[HERO_BLIZZ_IDX[i]] += 1

    allowed = np.zeros(NUM_HEROES, dtype=np.float32)
    for i in range(NUM_HEROES):
        if base[i] < 0.5:
            continue
        r = HERO_BLIZZ_IDX[i]
        c = list(cur); c[r] += 1
        a = list(avail); a[r] -= 1
        a = [min(x, slots_after) for x in a]
        if _feasible(tuple(c), tuple(a), slots_after):
            allowed[i] = 1.0
    if allowed.sum() < 0.5:
        FALLBACKS["empty_mask"] += 1
        return base.copy()
    return allowed


def _constrained_view(state, cmask):
    """A clone of `state` whose valid_mask()/valid_mask_np() return the
    constrained mask, so any one-shot-mask inner strategy (greedy candidate
    loop, policy argmax, Q argmax) is constrained without modification.
    clone() of the view yields a plain DraftState, so rollout completions
    inside greedy search stay unconstrained (intended: the constraint governs
    only OUR pick choice)."""
    s = state.clone()
    m = cmask.copy()
    s.valid_mask_np = types.MethodType(lambda self: m.copy(), s)
    s.valid_mask = types.MethodType(
        lambda self, device: torch.from_numpy(m).unsqueeze(0).to(device), s)
    return s


def constrain(inner):
    """Generic wrapper: rich-eval strategy signature in and out."""
    def strategy(state, team, step_type, game_map, tier, gd_models, device):
        if step_type != "pick":
            return inner(state, team, step_type, game_map, tier, gd_models,
                         device)
        cmask = role_constraint_mask(state, team)
        view = _constrained_view(state, cmask)
        idx = inner(view, team, step_type, game_map, tier, gd_models, device)
        if cmask[idx] < 0.5:   # inner ignored the mask — should not happen
            FALLBACKS["inner_violation"] += 1
            idx = int(np.argmax(cmask))
        return idx
    return strategy


# ── Inner strategies ────────────────────────────────────────────────

def load_mcts_policy(path=MCTS_CHECKPOINT):
    net = AlphaZeroDraftNet()
    sd = torch.load(path, weights_only=True, map_location="cpu")
    if any(k.startswith("res_block1.") for k in sd):
        sd = {k.replace("res_block1.", "res_blocks.0.")
               .replace("res_block2.", "res_blocks.1.")
               .replace("res_block3.", "res_blocks.2."): v
              for k, v in sd.items()}
    net.load_state_dict(sd)
    net.eval()
    return net


def make_mcts_policy_strategy(net):
    """J_800sim policy argmax (the phase3b 'mcts' actor, rich-eval signature).
    Perspective: our_team indicator = acting team."""
    def strategy(state, team, step_type, game_map, tier, gd_models, device):
        x = state.to_numpy().copy()
        x[-1] = float(team)
        xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        m = state.valid_mask(torch.device("cpu"))
        with torch.no_grad():
            logits, _ = net(xt, m)
        return logits.argmax(dim=1).item()
    return strategy


def build_strategy(name, device, stats, group_indices):
    """Rich-eval-signature strategy for a NEW (constrained) strategy name."""
    from experiment_rich_evaluation import make_wp_greedy_strategy
    if name == "constrained_greedy":
        m = WinProbEnrichedModel(283, [256, 128], dropout=0.3).to(device)
        m.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, "wp_enriched_256.pt"),
            weights_only=True, map_location=device))
        m.eval()
        inner = make_wp_greedy_strategy(m, ENRICHED_GROUPS, stats,
                                        group_indices, device)
        return "Constrained enriched greedy", constrain(inner)
    if name == "constrained_mcts":
        net = load_mcts_policy()
        return "Constrained MCTS (J_800sim_s9)", constrain(
            make_mcts_policy_strategy(net))
    raise ValueError(name)


def rpath(*p):
    path = os.path.join(RESULTS_DIR, "constrained", *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ── rich-eval (mirrors phase3_benchmarks.task_rich_eval) ────────────

def task_rich_eval(args):
    from rerun2026.phase3_benchmarks import load_gd_models, load_wp
    from experiment_rich_evaluation import (
        run_drafts_with_strategy, counter_responsiveness, synergy_exploitation,
        draft_diversity, map_adaptation)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = common.stats_cache()
    group_indices = compute_group_indices()
    gd_models = load_gd_models(torch.device("cpu"))
    label, strategy_fn = build_strategy(args.strategy, device, stats,
                                        group_indices)

    random.seed(42)
    draft_configs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
                     for i in range(args.drafts)]

    combined, seed_batches = [], []
    for seed in range(args.seeds):
        random.seed(seed)
        torch.manual_seed(seed)
        drafts = run_drafts_with_strategy(strategy_fn, draft_configs,
                                          gd_models, stats, device)
        combined.extend(drafts)
        seed_batches.append(drafts)
        print(f"  [{label}] seed {seed} done (fallbacks={dict(FALLBACKS)})",
              flush=True)

    n = len(combined)
    healer_rate = sum(d["has_healer"] for d in combined) / n * 100
    degen_rate = sum(d["is_degen"] for d in combined) / n * 100
    counters = [counter_responsiveness(d["our_picks"], d["opp_picks"], stats,
                                       d["tier"]) for d in combined]
    synergies = [synergy_exploitation(d["our_picks"], stats, d["tier"])
                 for d in combined]
    div = draft_diversity(combined)

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

    seed_metrics = []
    for batch in seed_batches:
        bn = len(batch)
        seed_metrics.append({
            "degen": sum(d["is_degen"] for d in batch) / bn * 100,
            "counter": float(np.mean([counter_responsiveness(
                d["our_picks"], d["opp_picks"], stats, d["tier"])
                for d in batch])),
            "synergy": float(np.mean([synergy_exploitation(
                d["our_picks"], stats, d["tier"]) for d in batch])),
        })

    result = {
        "strategy": label, "slug": args.strategy,
        "seeds": args.seeds, "drafts_per_seed": args.drafts,
        "constraint_fallbacks": dict(FALLBACKS),
        "metrics": {
            "healer_rate": round(healer_rate, 1),
            "degen_rate": round(degen_rate, 1),
            "counter": round(float(np.mean(counters)), 2),
            "synergy": round(float(np.mean(synergies)), 2),
            **div,
            "gd_similarity": round(gd_agree / gd_total * 100, 1) if gd_total else 0,
            "aug_wp": round(aug_wp, 3),
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
    print(json.dumps(result["metrics"], indent=2, default=str))


# ── tournament pairs (mirrors phase3b.run_pair with extended actors) ─

ALL_STRATEGIES = p3b.STRATEGIES + NEW_STRATEGIES


def build_actor_ext(name, device, stats, group_indices, gd_models, mcts_run):
    """phase3b actor signature act(state, team, step_type, game_map, tier)."""
    if name in p3b.STRATEGIES:
        return p3b.build_actor(name, device, stats, group_indices, gd_models,
                               mcts_run)
    _, strat = build_strategy(name, device, stats, group_indices)

    def act(state, team, step_type, game_map, tier):
        return strat(state, team, step_type, game_map, tier, gd_models, device)
    return act


def task_pair(args):
    a_name, b_name = args.pair.split("__")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = common.stats_cache()
    group_indices = compute_group_indices()
    from rerun2026.phase3_benchmarks import load_gd_models
    gd_models = load_gd_models(torch.device("cpu"))
    evaluators = p3b.load_wp_evaluators(device)

    act_a = build_actor_ext(a_name, device, stats, group_indices, gd_models,
                            args.mcts_run)
    act_b = build_actor_ext(b_name, device, stats, group_indices, gd_models,
                            args.mcts_run)

    pair_idx = (ALL_STRATEGIES.index(a_name) * len(ALL_STRATEGIES)
                + ALL_STRATEGIES.index(b_name))
    random.seed(3000 + pair_idx)
    torch.manual_seed(3000 + pair_idx)

    records = []
    for i in range(args.drafts):
        game_map = MAPS[i % len(MAPS)]
        tier = SKILL_TIERS[(i // len(MAPS)) % len(SKILL_TIERS)]
        state = DraftState(game_map, tier, our_team=0)
        picks0, picks1 = [], []
        while not state.is_terminal():
            step_team, step_type = DRAFT_ORDER[state.step]
            step_num = state.step
            state.our_team = step_team
            actor = act_a if step_team == 0 else act_b
            hero_idx = actor(state, step_team, step_type, game_map, tier)
            if step_type == "pick":
                (picks0 if step_team == 0 else picks1).append(
                    (HEROES[hero_idx], step_num))
            state.apply_action(hero_idx, step_team, step_type)

        t0h = [HEROES[j] for j in range(NUM_HEROES) if state.team0_picks[j] > 0]
        t1h = [HEROES[j] for j in range(NUM_HEROES) if state.team1_picks[j] > 0]
        records.append({
            "draft": i, "game_map": game_map, "tier": tier,
            "team0": p3b.side_metrics(picks0, picks1, stats, tier),
            "team1": p3b.side_metrics(picks1, picks0, stats, tier),
            "wp": p3b.score_terminal(t0h, t1h, game_map, tier, evaluators,
                                     stats, device),
            "bans": [HEROES[j] for j in range(NUM_HEROES) if state.bans[j] > 0],
        })
        if (i + 1) % 25 == 0:
            print(f"  [{a_name} vs {b_name}] {i+1}/{args.drafts}", flush=True)

    out = {"team0_strategy": a_name, "team1_strategy": b_name,
           "n_drafts": args.drafts, "constraint_fallbacks": dict(FALLBACKS),
           "records": records}
    path = rpath("roundrobin", f"{args.pair}.json")
    with open(path, "w") as f:
        json.dump(out, f, default=str)
    print(f"Saved {path}")


def pair_list():
    """Ordered pairs: each constrained strategy vs each existing strategy in
    both orders, plus the two constrained strategies against each other."""
    pairs = []
    for c in NEW_STRATEGIES:
        for s in p3b.STRATEGIES:
            pairs.append((c, s))
            pairs.append((s, c))
    pairs.append((NEW_STRATEGIES[0], NEW_STRATEGIES[1]))
    pairs.append((NEW_STRATEGIES[1], NEW_STRATEGIES[0]))
    return pairs


# ── aggregate ───────────────────────────────────────────────────────

def task_aggregate(args):
    out = {"strategies": ALL_STRATEGIES, "rich_eval": {}, "pairs": {},
           "matchups": {}}
    d = os.path.join(RESULTS_DIR, "constrained", "rich_eval")
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json"):
                with open(os.path.join(d, fn)) as f:
                    out["rich_eval"][fn[:-5]] = json.load(f)

    d = os.path.join(RESULTS_DIR, "constrained", "roundrobin")
    evs = list(p3b.WP_EVALUATORS)
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(d, fn)) as f:
                p = json.load(f)
            recs = p["records"]
            n = len(recs)
            s = {"n": n}
            for ev in evs:
                s[f"team0_wp_{ev}_sym"] = float(np.mean(
                    [r["wp"][ev]["sym"] for r in recs]))
            s["team0_wp_consensus"] = float(np.mean(
                [s[f"team0_wp_{ev}_sym"] for ev in evs]))
            for side in ("team0", "team1"):
                s[f"{side}_healer"] = sum(r[side]["has_healer"]
                                          for r in recs) / n * 100
                s[f"{side}_degen"] = sum(r[side]["is_degen"]
                                         for r in recs) / n * 100
                s[f"{side}_synergy"] = float(np.mean(
                    [r[side]["synergy"] for r in recs]))
                s[f"{side}_counter"] = float(np.mean(
                    [r[side]["counter"] for r in recs]))
            out["pairs"][fn[:-5]] = s

    # per constrained strategy: ordered matchups won + mean consensus WP
    for c in NEW_STRATEGIES:
        wps, wins, total = [], 0, 0
        for key, s in out["pairs"].items():
            a, b = key.split("__")
            if c == a:
                wp = s["team0_wp_consensus"]
            elif c == b:
                wp = 1.0 - s["team0_wp_consensus"]
            else:
                continue
            wps.append(wp)
            wins += wp > 0.5
            total += 1
        if total:
            out["matchups"][c] = {"ordered_matchups": total, "won": wins,
                                  "consensus_wp": float(np.mean(wps))}

    path = rpath("constrained_summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {path}")
    for c, m in out["matchups"].items():
        print(f"  {c}: {m['won']}/{m['ordered_matchups']} ordered matchups won, "
              f"consensus WP {m['consensus_wp']:.3f}")


# ── smoke test (CPU, logic only) ────────────────────────────────────

def task_smoke(args):
    """Verify mask behavior without heavy compute:
    1. targeted states: forced-healer / stack-cap / ranged-required / fallback;
    2. inductive invariant: constrained MCTS + a few constrained-greedy drafts
       on CPU always end structurally valid, with zero fallbacks."""
    print("== targeted mask checks ==")
    healers = [h for h, r in HERO_ROLE_FINE.items() if r == "healer"]
    tanks = [h for h, r in HERO_ROLE_FINE.items() if r == "tank"]
    ranged = [h for h, r in HERO_ROLE_FINE.items()
              if r in ("ranged_aa", "ranged_mage", "pusher")]
    idx = {h: i for i, h in enumerate(HEROES)}
    failures = []

    def check(desc, cond):
        print(f"  {'PASS' if cond else 'FAIL'}  {desc}")
        if not cond:
            failures.append(desc)

    # (a) 4 picks, no healer -> only healers allowed
    st = DraftState("Cursed Hollow", "mid", our_team=0)
    for h in [tanks[0], tanks[1], ranged[0], ranged[1]]:
        st.team0_picks[idx[h]] = 1.0
        st.taken.add(idx[h])
    st.step = 14
    m = role_constraint_mask(st, 0)
    allowed = {HEROES[i] for i in range(NUM_HEROES) if m[i] > 0.5}
    check("4 picks no healer -> allowed set == available healers",
          allowed == set(healers) - set())

    # (b) 2 tanks picked -> third tank masked (Tank cap 2)
    st = DraftState("Cursed Hollow", "mid", our_team=0)
    for h in [tanks[0], tanks[1]]:
        st.team0_picks[idx[h]] = 1.0
        st.taken.add(idx[h])
    st.step = 6
    m = role_constraint_mask(st, 0)
    check("2 tanks -> all remaining tanks masked",
          all(m[idx[t]] < 0.5 for t in tanks[2:]))
    check("2 tanks -> healers still allowed", m[idx[healers[0]]] > 0.5)

    # (c) 4 picks with healer+tank but no ranged -> only ranged allowed
    st = DraftState("Cursed Hollow", "mid", our_team=0)
    melee = [h for h, r in HERO_ROLE_FINE.items() if r == "melee_assassin"]
    for h in [healers[0], tanks[0], melee[0], melee[1]]:
        st.team0_picks[idx[h]] = 1.0
        st.taken.add(idx[h])
    st.step = 14
    m = role_constraint_mask(st, 0)
    allowed = {HEROES[i] for i in range(NUM_HEROES) if m[i] > 0.5}
    check("healer+tank+2 melee -> only ranged-damage heroes allowed",
          allowed and allowed <= set(ranged))

    # (d) impossible state (4 picks, no healer, no ranged) -> fallback to base
    st = DraftState("Cursed Hollow", "mid", our_team=0)
    for h in [tanks[0], tanks[1], melee[0], melee[1]]:
        st.team0_picks[idx[h]] = 1.0
        st.taken.add(idx[h])
    st.step = 14
    before = FALLBACKS["empty_mask"]
    m = role_constraint_mask(st, 0)
    check("infeasible state -> falls back to availability mask",
          FALLBACKS["empty_mask"] == before + 1 and m.sum() > 80)

    # (e) empty team, healers exhausted from pool -> first pick already forced?
    st = DraftState("Cursed Hollow", "mid", our_team=0)
    for h in healers[:14]:   # opponent/bans consumed 14 of 16 healers
        st.taken.add(idx[h])
        st.bans[idx[h]] = 1.0
    m = role_constraint_mask(st, 0)
    check("healer-starved pool -> non-healers still fine (2 healers left)",
          m.sum() > 50)

    print("\n== constrained MCTS policy drafts (CPU) ==")
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cpu")
    stats = common.stats_cache()
    gi = compute_group_indices()
    from rerun2026.phase3_benchmarks import load_gd_models
    gd_models = load_gd_models(device)
    from experiment_rich_evaluation import run_drafts_with_strategy
    FALLBACKS.clear()
    _, strat = build_strategy("constrained_mcts", device, stats, gi)
    cfgs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
            for i in range(args.drafts)]
    drafts = run_drafts_with_strategy(strat, cfgs, gd_models, stats, device)
    degen = sum(d["is_degen"] for d in drafts)
    no_ranged = sum(not any(HERO_ROLE_FINE.get(h) in
                            ("ranged_aa", "ranged_mage", "pusher")
                            for h in d["our_picks"]) for d in drafts)
    check(f"constrained MCTS: 0/{len(drafts)} degenerate (got {degen})",
          degen == 0)
    check(f"constrained MCTS: 0/{len(drafts)} without ranged (got {no_ranged})",
          no_ranged == 0)
    check(f"constrained MCTS: no fallbacks (got {dict(FALLBACKS)})",
          sum(FALLBACKS.values()) == 0)

    print("\n== constrained greedy drafts (CPU, slow — few drafts) ==")
    FALLBACKS.clear()
    random.seed(0)
    torch.manual_seed(0)
    _, strat = build_strategy("constrained_greedy", device, stats, gi)
    cfgs = [(i, random.choice(MAPS), random.choice(SKILL_TIERS), i % 2)
            for i in range(max(2, args.drafts // 5))]
    drafts = run_drafts_with_strategy(strat, cfgs, gd_models, stats, device)
    degen = sum(d["is_degen"] for d in drafts)
    check(f"constrained greedy: 0/{len(drafts)} degenerate (got {degen})",
          degen == 0)
    check(f"constrained greedy: no fallbacks (got {dict(FALLBACKS)})",
          sum(FALLBACKS.values()) == 0)

    print(f"\nSmoke: {'ALL PASS' if not failures else f'{len(failures)} FAILURES'}")
    if failures:
        sys.exit(1)


# ── orchestrator ────────────────────────────────────────────────────

def build_pool_jobs(args):
    me = os.path.abspath(__file__)
    jobs = []
    for s in NEW_STRATEGIES:
        jobs.append(Job(f"p3c_rich_{s}", [me, "--task", "rich-eval",
                                          "--strategy", s],
                        [rpath("rich_eval", f"{s}.json")],
                        weight="heavy" if s == "constrained_greedy" else "light"))
    greedy_like = {"enriched", "enriched_aug", "gourdeau", "constrained_greedy"}
    for a, b in pair_list():
        weight = "heavy" if (a in greedy_like or b in greedy_like) else "light"
        jobs.append(Job(f"p3c_{a}__{b}",
                        [me, "--task", "pair", "--pair", f"{a}__{b}",
                         "--drafts", str(args.drafts or 200)],
                        [rpath("roundrobin", f"{a}__{b}.json")], weight=weight))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None,
                        choices=[None, "smoke", "rich-eval", "pair",
                                 "aggregate"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--drafts", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--pair", default=None)
    parser.add_argument("--mcts-run", default=None,
                        help="historical mcts_runs/<name> for the phase3b "
                             "'mcts' baseline (default: phase3b defaults)")
    args = parser.parse_args()

    defaults = {"rich-eval": 1000, "pair": 200, "smoke": 10}
    if args.drafts is None:
        args.drafts = defaults.get(args.task, 200)

    if args.task is None:
        jobs = build_pool_jobs(args)
        print(f"Phase 3c: {len(jobs)} jobs "
              f"(2 rich-eval + {len(pair_list())} tournament pairs)")
        failed = common.run_pool(jobs, dry_run=args.dry_run, force=args.force,
                                 only=args.only)
        if args.dry_run:
            return
        task_aggregate(args)
        if failed:
            sys.exit(1)
        return

    fn = {"smoke": task_smoke, "rich-eval": task_rich_eval,
          "pair": task_pair, "aggregate": task_aggregate}[args.task]
    t0 = time.time()
    fn(args)
    print(f"task {args.task} done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
