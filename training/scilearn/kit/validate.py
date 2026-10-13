"""Validation hooks for the kit table (FRAMING.md 10.1).

(a) Within-role permutation check. The harness supplies score(axis_values) -> float, where
    axis_values maps hero name -> value of one axis (higher score = axis explains more). The
    check shuffles the axis values among heroes of the same role, 1000 times, and returns the
    share of permutations the real values beat. FRAMING 10.1 asks for at least 0.95.
(b) Spot check list: 15 heroes chosen by a fixed seeded hash, written to SPOTCHECK.md with
    their coded values for Max to review.

Usage: python3 -I validate.py --spotcheck     (writes SPOTCHECK.md)
"""
import hashlib
import json
import os
import random
import sys
from typing import Callable, Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kitdefs import AXES  # noqa: E402

KIT = os.path.join(HERE, "kit_2.57.json")
SPOT_SEED = "kit-2.57-spotcheck-v1"
N_SPOT = 15


def load_kit(path: str = KIT) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def axis_values(kit: dict, axis: str) -> Dict[str, float]:
    return {h: rec["values"][axis] for h, rec in kit["heroes"].items()}


def within_role_permutation_check(kit: dict, axis: str, score: Callable[[Dict[str, float]], float],
                                  n_perm: int = 1000, seed: int = 20261013,
                                  heroes: Optional[list] = None) -> dict:
    """Share of within-role permutations whose score the real axis values beat (strictly).

    heroes: optional subset (e.g. heroes present in the evaluation data); permutation is done
    within role inside that subset. Ties count as not beaten.
    """
    vals = axis_values(kit, axis)
    names = sorted(heroes if heroes is not None else vals)
    by_role: Dict[str, list] = {}
    for h in names:
        by_role.setdefault(kit["heroes"][h]["role"] or "unknown", []).append(h)
    real = score({h: vals[h] for h in names})
    rng = random.Random(seed)
    beaten = 0
    for _ in range(n_perm):
        perm = {}
        for members in by_role.values():
            v = [vals[h] for h in members]
            rng.shuffle(v)
            perm.update(zip(members, v))
        if real > score(perm):
            beaten += 1
    share = beaten / n_perm
    return {"axis": axis, "real_score": real, "share_beaten": share, "n_perm": n_perm, "passes_0.95": share >= 0.95}


def check_all_axes(kit: dict, score_for_axis: Callable[[str], Callable[[Dict[str, float]], float]],
                   **kw) -> Dict[str, dict]:
    """Run the permutation check for every axis; score_for_axis(axis) returns that axis's scorer."""
    return {a: within_role_permutation_check(kit, a, score_for_axis(a), **kw) for a in AXES}


def spotcheck_heroes(kit: dict, n: int = N_SPOT, seed: str = SPOT_SEED) -> list:
    key = lambda h: hashlib.sha256(f"{seed}|{h}".encode("utf-8")).hexdigest()
    return sorted(sorted(kit["heroes"], key=key)[:n])


def write_spotcheck(kit: dict, path: str = os.path.join(HERE, "SPOTCHECK.md")) -> list:
    picks = spotcheck_heroes(kit)
    short = {"hard_cc": "hCC", "soft_cc_peel": "peel", "healing_throughput": "heal",
             "self_sustain_shields": "self", "burst": "burst", "sustained_damage": "sust", "range": "range",
             "mobility": "mob", "waveclear": "wave", "siege": "siege", "global_presence": "global",
             "early_vs_late": "e/l"}
    lines = [
        "# Kit table spot check (kit_2.57)",
        "",
        f"15 heroes chosen by a fixed seeded hash (sha256 of \"{SPOT_SEED}|<hero>\", lowest 15). "
        f"Kit file sha256: `{kit['sha256']}`.",
        "",
        "Values are the mean of three coders (0 to 4; early/late -2 to +2). A `*` marks an axis where the coders' SD "
        "is above 1.0. Columns: hCC hard CC, peel soft CC and peel, heal ally healing, self self-sustain and shields, "
        "burst, sust sustained damage, range, mob mobility, wave waveclear, siege, global global presence, "
        "e/l early (-) vs late (+).",
        "",
        "Max: mark each row OK, or note the axis and the value you would expect.",
        "",
        "| Hero | Role | " + " | ".join(short[a] for a in AXES) + " | Verdict |",
        "|---|---|" + "---|" * len(AXES) + "---|",
    ]
    for h in picks:
        r = kit["heroes"][h]
        cells = [f"{r['values'][a]:.2f}" + ("*" if a in r["flagged_axes"] else "") for a in AXES]
        lines.append(f"| {h} | {r['role']} | " + " | ".join(cells) + " | |")
    lines += ["", "Per-coder raw values for these heroes are in kit_2.57.json under heroes.<name>.raw.", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return picks


if __name__ == "__main__":
    kit = load_kit()
    if "--spotcheck" in sys.argv:
        print(write_spotcheck(kit))
