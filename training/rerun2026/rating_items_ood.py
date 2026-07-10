"""
Freeze the OOD ensemble-variance covariate into the expert-rating item pool.

For every machine-pair item in data/rating-items.json (provenance.source ==
'tournament'), computes the per-draft epistemic uncertainty of EACH team using
the 20-member WP ensemble from ensemble_uncertainty.py: the team is paired
against a fixed reference composition (STANDARD, with same-fine-role
substitutes on hero collision) on the item's map/tier, and the covariate is
the ensemble variance of the symmetrized WP across the 20 members.

(The matchup itself has a single side-invariant ensemble variance — the
symmetrized WP is antisymmetric under a team swap — so per-TEAM OOD-ness is
measured against the fixed reference, exactly like the degenerate probe comps
in ensemble_uncertainty.py. This yields two numbers per item.)

Written into provenance (frozen before any rating is collected):
  ood_var_team0, ood_var_team1        per-team ensemble variance vs reference
  ood_var_max, ood_var_mean           item-level aggregates
  ood_var_matchup                     ensemble variance of the ACTUAL matchup
                                      (team0 vs team1) symmetrized WP — a
                                      single side-invariant number per item;
                                      v4 sensitivity covariate for H2
  ood_ref_team0, ood_ref_team1        the exact reference comp used (audit)

GD-likelihood was considered as a second covariate and SKIPPED: the
round-robin records store final picks/bans but not the interleaved pick
sequence, so the generic-draft log-probability of each pick cannot be
reconstructed without re-simulating draft order (risk of silent
misalignment). Noted in the prereg.

Usage:
  python3 rerun2026/rating_items_ood.py            # updates data/rating-items.json in place
  python3 rerun2026/rating_items_ood.py --dry-run  # compute + summarize only
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rerun2026 import common

common.setup()

import numpy as np
import torch

from shared import HERO_ROLE_FINE
from rerun2026.ensemble_uncertainty import (ROSTER, _extract_set,
                                            _member_predict, model_path)
from experiment_synthetic_augmentation import STANDARD

ITEMS_PATH = os.path.join(common.TRAINING_DIR, "..", "data", "rating-items.json")

# Same-fine-role substitutes for each STANDARD slot, tried in order when the
# reference hero collides with the evaluated team. Deterministic.
SUBSTITUTES = {
    "Muradin": ["Johanna", "E.T.C.", "Diablo", "Stitches", "Garrosh"],       # tank
    "Brightwing": ["Malfurion", "Rehgar", "Uther", "Li Li", "Stukov"],       # healer
    "Valla": ["Raynor", "Falstad", "Greymane", "Tychus", "Cassia"],          # ranged_aa
    "Sonya": ["Thrall", "Leoric", "Artanis", "Dehaka", "Imperius"],          # bruiser
    "Jaina": ["Li-Ming", "Kael'thas", "Gul'dan", "Orphea", "Junkrat"],       # ranged_mage
}


def reference_for(team):
    """STANDARD comp with same-fine-role substitutes on collision."""
    used = set(team)
    ref = []
    for slot in STANDARD:
        pick = slot
        if pick in used or pick in ref:
            for sub in SUBSTITUTES[slot]:
                if sub not in used and sub not in ref:
                    pick = sub
                    break
            else:
                raise RuntimeError(f"no substitute for {slot} vs team {team}")
        assert HERO_ROLE_FINE[pick] == HERO_ROLE_FINE[slot]
        ref.append(pick)
    return ref


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    missing = [m[0] for m in ROSTER if not os.path.exists(model_path(m[0]))]
    if missing:
        raise RuntimeError(f"missing ensemble members: {missing}")

    with open(ITEMS_PATH) as f:
        pool = json.load(f)
    machine = [it for it in pool["items"]
               if it["provenance"].get("source") == "tournament"]
    print(f"pool seed {pool['seed']}: {len(pool['items'])} items, "
          f"{len(machine)} machine pairs")

    # Three records per item: team0 vs reference, team1 vs reference (the
    # per-team probes), and team0 vs team1 (the actual matchup).
    records, refs = [], []
    for it in machine:
        for side in (0, 1):
            team = it["teams"][f"team{side}"]
            ref = reference_for(team)
            refs.append(ref)
            records.append({"team0_heroes": team, "team1_heroes": ref,
                            "game_map": it["map"], "skill_tier": it["tier"]})
        refs.append(None)
        records.append({"team0_heroes": it["teams"]["team0"],
                        "team1_heroes": it["teams"]["team1"],
                        "game_map": it["map"], "skill_tier": it["tier"]})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = common.stats_cache()
    feats = _extract_set(records, stats)
    P = np.stack([_member_predict(m[0], feats, device) for m in ROSTER])
    var = P.var(axis=0, ddof=1)  # [3 * n_machine]
    print(f"predicted {P.shape[1]} probe/matchup records x {P.shape[0]} members")

    for i, it in enumerate(machine):
        v0, v1 = float(var[3 * i]), float(var[3 * i + 1])
        vm = float(var[3 * i + 2])
        prov = it["provenance"]
        prov["ood_var_team0"] = v0
        prov["ood_var_team1"] = v1
        prov["ood_var_max"] = max(v0, v1)
        prov["ood_var_mean"] = 0.5 * (v0 + v1)
        prov["ood_var_matchup"] = vm
        prov["ood_ref_team0"] = refs[3 * i]
        prov["ood_ref_team1"] = refs[3 * i + 1]

    # Distribution summary (for the prereg / report).
    def dist(v):
        q = lambda x: float(np.percentile(v, x))
        return (f"n={len(v)} mean={np.mean(v):.5f} median={np.median(v):.5f} "
                f"p5={q(5):.5f} p90={q(90):.5f} p95={q(95):.5f} max={np.max(v):.5f}")

    by_key = {}
    for it in machine:
        prov = it["provenance"]
        by_key.setdefault(("block", it["block"]), []).append(prov["ood_var_max"])
        by_key.setdefault(("stratum", prov["stratum"]), []).append(prov["ood_var_max"])
    all_max = [it["provenance"]["ood_var_max"] for it in machine]
    print(f"\nood_var_max (all machine pairs): {dist(np.array(all_max))}")
    all_matchup = [it["provenance"]["ood_var_matchup"] for it in machine]
    print(f"ood_var_matchup (all machine pairs): {dist(np.array(all_matchup))}")
    print("spearman(ood_var_max, ood_var_matchup) = "
          f"{_spearman(all_max, all_matchup):.3f}")
    for (kind, key), vals in sorted(by_key.items()):
        print(f"  {kind}={key}: {dist(np.array(vals))}")
    n_sub = sum(1 for i, r in enumerate(records)
                if refs[i] is not None and refs[i] != STANDARD)
    n_probe = sum(1 for r in refs if r is not None)
    print(f"reference substitutions used on {n_sub}/{n_probe} probe records")

    if args.dry_run:
        print("dry run: not writing")
        return
    with open(ITEMS_PATH, "w") as f:
        json.dump(pool, f, indent=2, ensure_ascii=False)
    print(f"updated {os.path.normpath(ITEMS_PATH)} in place")


if __name__ == "__main__":
    main()
