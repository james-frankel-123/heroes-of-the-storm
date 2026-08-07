"""
Instrument the drafter's projection sources against real drafts.

Question (Max, 2026-08-07): the site's "proj. final" (policy-net value head
via MCTS child Q) sits ~80% all draft and cliffs to ~50% at the end. Is the
value head's LEVEL actually state-insensitive, and is the partial-draft WP
model good enough to ship as the projection engine instead?

Replays N real drafts step by step and, at every one of the 16 states,
computes:
  V   deployed policy-net value head (symmetrized), production checkpoint
  P   partial-state WP model (283-d enriched features + step embedding)
Terminal references: deployed full WP score of the finished draft
(symmetrized) and the actual game outcome.

Reports per-step: std of each estimate, Pearson r vs terminal full-WP,
AUC vs actual outcome, and the step-15 -> terminal convergence gap.
Plus an enemy-swap probe: same our-side mid-draft state, enemy = strong
meta comp vs degenerate comp; a state-sensitive estimator must move.

Usage: python3 training/instrument_projections.py  (from repo root, .env sourced)
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TRAINING = os.path.dirname(os.path.abspath(__file__))

# Partial-WP checkpoint was trained on the frozen 2026-05-19 stats (its
# default); full WP is the production decayed-stats model. Instantiate both.
from sweep_enriched_wp import StatsCache, extract_features, FEATURE_GROUPS, WinProbEnrichedModel
import sweep_enriched_wp as sew
from train_partial_wp import PartialStateWP, WP_GROUPS
from train_draft_policy import AlphaZeroDraftNet, DraftState, DRAFT_ORDER, STATE_DIM
from shared import HERO_TO_IDX, NUM_HEROES

PROD = os.path.join(TRAINING, "production_refresh", "2026-07-14")
POLICY_PT = os.path.join(PROD, "mcts_s0", "draft_policy.pt")
WP_PT = os.path.join(PROD, "wp_enriched_256.pt")
PARTIAL_PT = os.path.join(TRAINING, "rerun2026", "models", "partial_wp.pt")
PROD_STATS = os.path.join(PROD, "stats_decayed90.json")
N_DRAFTS = 2000
SEED = 20260807

GROUPS_MASK = [g in WP_GROUPS for g in FEATURE_GROUPS]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_models():
    policy = AlphaZeroDraftNet(size="base", policy_head_type="linear")
    policy.load_state_dict(torch.load(POLICY_PT, weights_only=True, map_location="cpu"))
    policy.to(device).eval()
    ckpt = torch.load(PARTIAL_PT, weights_only=True, map_location="cpu")
    partial = PartialStateWP(input_dim=ckpt.get("input_dim", 283),
                             step_embed_dim=ckpt.get("step_embed_dim", 8),
                             hidden=tuple(ckpt.get("hidden", (256, 128))))
    partial.load_state_dict(ckpt["model_state_dict"])
    partial.to(device).eval()
    wp = WinProbEnrichedModel(283, [256, 128], dropout=0.3)
    wp.load_state_dict(torch.load(WP_PT, weights_only=True, map_location="cpu"))
    wp.to(device).eval()
    return policy, partial, wp


def sym_value(policy, state: DraftState) -> float:
    """Symmetrized team0 win estimate from the policy value head."""
    with torch.no_grad():
        s0 = state.clone(); s0.our_team = 0
        s1 = state.clone(); s1.our_team = 1
        x = torch.cat([s0.to_tensor(device), s1.to_tensor(device)])
        mask = torch.ones(2, NUM_HEROES, device=device)
        _, v = policy(x, mask)
        return float(0.5 * (v[0].item() + (1.0 - v[1].item())))


def features_for(row_like, stats):
    f = extract_features(row_like, stats, GROUPS_MASK)
    return np.concatenate(f).astype(np.float32)


def partial_estimate(partial, stats, t0, t1, gmap, tier, step) -> float:
    d = {"team0_heroes": t0, "team1_heroes": t1, "game_map": gmap, "skill_tier": tier}
    dr = {"team0_heroes": t1, "team1_heroes": t0, "game_map": gmap, "skill_tier": tier}
    X = np.stack([features_for(d, stats), features_for(dr, stats)])
    si = torch.tensor([min(step, 15)] * 2, dtype=torch.long, device=device)
    with torch.no_grad():
        p = partial(torch.tensor(X, device=device), si).cpu().numpy()
    return float(0.5 * (p[0] + (1.0 - p[1])))


def wp_final(wp, stats, t0, t1, gmap, tier) -> float:
    d = {"team0_heroes": t0, "team1_heroes": t1, "game_map": gmap, "skill_tier": tier}
    dr = {"team0_heroes": t1, "team1_heroes": t0, "game_map": gmap, "skill_tier": tier}
    X = np.stack([features_for(d, stats), features_for(dr, stats)])
    with torch.no_grad():
        p = wp(torch.tensor(X, device=device)).cpu().numpy()
    return float(0.5 * (p[0] + (1.0 - p[1])))


def replay_steps(row):
    """Yield (step_idx, t0_picks, t1_picks, bans) after each of the 16 actions,
    with team attribution by membership. Returns None if inconsistent."""
    t0s, t1s = set(row["team0_heroes"]), set(row["team1_heroes"])
    b0s, b1s = set(row["team0_bans"]), set(row["team1_bans"])
    order = sorted(row["draft_order"], key=lambda e: e["pick_number"])
    if len(order) != 16 or len(t0s) != 5 or len(t1s) != 5:
        return None
    t0, t1, bans = [], [], []
    out = []
    for i, e in enumerate(order):
        h = e["hero"]
        if h not in HERO_TO_IDX:
            return None
        if e["type"] == "1":
            if h in t0s: t0.append(h)
            elif h in t1s: t1.append(h)
            else: return None
        else:
            if h not in b0s and h not in b1s: return None
            bans.append(h)
        out.append((i, list(t0), list(t1), list(bans)))
    if len(t0) != 5 or len(t1) != 5:
        return None
    return out


def auc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(len(scores))
    pos = labels > 0.5
    if pos.sum() == 0 or (~pos).sum() == 0: return float("nan")
    return float((ranks[pos].mean() - (pos.sum() - 1) / 2) / (~pos).sum())


def main():
    rng = np.random.default_rng(SEED)
    print("loading corpus cache...")
    rows = json.load(open(os.path.join(TRAINING, ".replay_cache.json")))
    rows = [r for r in rows if r.get("draft_order") and len(r["draft_order"]) == 16
            and len(r.get("team0_heroes", [])) == 5 and len(r.get("team1_heroes", [])) == 5
            and r.get("winner") in (0, 1)]
    rows = rows[-200_000:]  # recent era (matches deployed models)
    idx = rng.choice(len(rows), size=min(N_DRAFTS, len(rows)), replace=False)
    sample = [rows[i] for i in idx]
    print(f"{len(sample)} drafts sampled from recent corpus")

    stats_frozen = StatsCache()  # frozen 2026-05-19 (partial ckpt's regime)
    sew.FROZEN_STATS_PATH = PROD_STATS
    stats_prod = StatsCache()    # production decayed90 (deployed WP's regime)

    policy, partial, wp = load_models()

    V = np.full((len(sample), 16), np.nan)
    P = np.full((len(sample), 16), np.nan)
    final_wp = np.full(len(sample), np.nan)
    outcome = np.zeros(len(sample))

    for n, row in enumerate(sample):
        steps = replay_steps(row)
        if steps is None: continue
        gmap, tier = row["game_map"], row["skill_tier"]
        st = DraftState(gmap, tier, our_team=0)
        for (i, t0, t1, bans) in steps:
            for h in t0: st.team0_picks[HERO_TO_IDX[h]] = 1.0
            for h in t1: st.team1_picks[HERO_TO_IDX[h]] = 1.0
            st.team0_picks[:] = 0; st.team1_picks[:] = 0; st.bans[:] = 0
            for h in t0: st.team0_picks[HERO_TO_IDX[h]] = 1.0
            for h in t1: st.team1_picks[HERO_TO_IDX[h]] = 1.0
            for h in bans: st.bans[HERO_TO_IDX[h]] = 1.0
            st.step = min(i + 1, 15)
            V[n, i] = sym_value(policy, st)
            P[n, i] = partial_estimate(partial, stats_frozen, t0, t1, gmap, tier, min(i + 1, 15))
        final_wp[n] = wp_final(wp, stats_prod, row["team0_heroes"], row["team1_heroes"], gmap, tier)
        outcome[n] = 1.0 - row["winner"]  # 1 if team0 won
        if (n + 1) % 250 == 0:
            print(f"  {n+1}/{len(sample)}")

    ok = ~np.isnan(final_wp) & ~np.isnan(V[:, 15]) & ~np.isnan(P[:, 15])
    V, P, final_wp, outcome = V[ok], P[ok], final_wp[ok], outcome[ok]
    print(f"\n{ok.sum()} drafts fully replayed\n")
    print("step | V.std  P.std | r(V,finalWP) r(P,finalWP) | AUC_V  AUC_P")
    results = {"per_step": []}
    for s in [0, 3, 5, 8, 11, 13, 15]:
        rV = float(np.corrcoef(V[:, s], final_wp)[0, 1])
        rP = float(np.corrcoef(P[:, s], final_wp)[0, 1])
        row = {"step": s, "V_std": float(V[:, s].std()), "P_std": float(P[:, s].std()),
               "r_V": rV, "r_P": rP,
               "auc_V": auc(V[:, s], outcome), "auc_P": auc(P[:, s], outcome)}
        results["per_step"].append(row)
        print(f"  {s:2d} | {row['V_std']:.3f}  {row['P_std']:.3f} |    {rV:+.3f}      {rP:+.3f}    | {row['auc_V']:.3f}  {row['auc_P']:.3f}")

    conv_V = float(np.mean(np.abs(V[:, 15] - final_wp)))
    conv_P = float(np.mean(np.abs(P[:, 15] - final_wp)))
    results["convergence"] = {"V_mae_vs_finalWP": conv_V, "P_mae_vs_finalWP": conv_P,
                              "V_mean_at_15": float(V[:, 15].mean()),
                              "P_mean_at_15": float(P[:, 15].mean()),
                              "finalWP_mean": float(final_wp.mean())}
    print(f"\nstep-15 -> terminal: V MAE {conv_V:.3f} (V mean {V[:,15].mean():.3f}), "
          f"P MAE {conv_P:.3f} (P mean {P[:,15].mean():.3f}), finalWP mean {final_wp.mean():.3f}")

    # Enemy-swap probe at step 8: our picks fixed, enemy = strong vs degenerate.
    STRONG = ["Johanna", "Deckard", "Raynor", "Jaina", "Sonya"]
    DEGEN = ["Johanna", "Muradin", "Arthas", "Garrosh", "Diablo"]
    dV, dP = [], []
    probes = 0
    for row in sample:
        if probes >= 300: break
        steps = replay_steps(row)
        if steps is None: continue
        i, t0, t1, bans = steps[8]
        if len(t0) < 2: continue
        gmap, tier = row["game_map"], row["skill_tier"]
        ests = {}
        for name, enemy in (("strong", STRONG), ("degen", DEGEN)):
            if any(h in t0 or h in bans for h in enemy): break
            st = DraftState(gmap, tier, our_team=0)
            for h in t0: st.team0_picks[HERO_TO_IDX[h]] = 1.0
            for h in enemy: st.team1_picks[HERO_TO_IDX[h]] = 1.0
            for h in bans: st.bans[HERO_TO_IDX[h]] = 1.0
            st.step = 15
            ests[name] = (sym_value(policy, st),
                          partial_estimate(partial, stats_frozen, t0, enemy, gmap, tier, 15))
        if len(ests) == 2:
            dV.append(ests["degen"][0] - ests["strong"][0])
            dP.append(ests["degen"][1] - ests["strong"][1])
            probes += 1
    results["enemy_swap"] = {"n": probes,
                             "V_mean_shift": float(np.mean(dV)), "P_mean_shift": float(np.mean(dP))}
    print(f"\nenemy swap (degenerate enemy vs strong enemy, {probes} states): "
          f"our win estimate should RISE. V: {np.mean(dV):+.3f}, P: {np.mean(dP):+.3f}")

    out = os.path.join(TRAINING, "instrument_projections_results.json")
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
