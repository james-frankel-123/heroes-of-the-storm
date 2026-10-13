"""Synthetic leafpacks for tests and the O6 dry run.

All packs share one "truth" (seed --truth-seed). A pack is truth plus fitting
noise (seed --seed), so two packs with different seeds behave like a model W
and an independent judge J fit on disjoint data.

  benign   plausible tables: role structure penalties (no healer, no frontline,
           no damage dealer, 3+ stacked tanks / healers / supports / melee), small hero, pair, map
           and cross effects, concave kit shapes, rules with |beta| <= 0.15.
  exploit  benign plus two planted faults: role_table[5 tanks] = +2.5 logit
           (bonus stacking) and a rule cliff, +1.5 logit when a team has no
           healer and waveclear sum >= 2.4 (violates the 0.15 rule cap).

Usage:
  python3 training/scilearn/leaf/make_synthetic.py --out /tmp/leaf/benign --kind benign --seed 1
  python3 training/scilearn/leaf/make_synthetic.py --out /tmp/leaf/exploit --kind exploit --seed 1
  python3 training/scilearn/leaf/make_synthetic.py --out /tmp/leaf/judge --kind benign --seed 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))          # training/, for shared.py
from leaf import DEFAULT_ROLES, DEFAULT_TIERS, N_ROLES, ROLE_CODES, ROLE_POW, LeafModel  # noqa: E402
import shared  # noqa: E402

AXES = ["waveclear", "frontline", "sustain", "burst", "hard_cc", "mobility", "global",
        "poke", "siege", "peel", "dive", "objective"]
K = len(AXES)
EXPLOIT_TANK_BONUS = 2.5
EXPLOIT_CLIFF_BETA = 1.5


def hero_list():
    # v1 prefix + the 2.57 hero; leafpack order is sorted Python order of names
    return sorted(shared.HEROES[:90] + ["Xal'atath"])


def map_list():
    return sorted(shared.MAPS[:14] + ["Haunted Mines"])


def blizz_role(h):
    return shared.FINE_TO_BLIZZ_ROLE.get(shared.HERO_ROLE_FINE.get(h, ""), "Ranged Assassin")


# role -> mean kit profile on AXES (0..1 scale)
ROLE_KIT = {
    "Tank":            [0.2, 0.9, 0.2, 0.1, 0.8, 0.3, 0.1, 0.1, 0.2, 0.6, 0.5, 0.4],
    "Bruiser":         [0.5, 0.6, 0.4, 0.3, 0.4, 0.4, 0.2, 0.1, 0.4, 0.3, 0.4, 0.5],
    "Healer":          [0.1, 0.1, 0.9, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1, 0.6, 0.1, 0.2],
    "Support":         [0.2, 0.2, 0.4, 0.1, 0.3, 0.3, 0.5, 0.1, 0.3, 0.7, 0.2, 0.4],
    "Ranged Assassin": [0.7, 0.0, 0.1, 0.6, 0.2, 0.3, 0.1, 0.6, 0.5, 0.1, 0.2, 0.3],
    "Melee Assassin":  [0.2, 0.1, 0.2, 0.8, 0.2, 0.7, 0.1, 0.0, 0.2, 0.0, 0.8, 0.2],
}


def structural_role_value(c):
    """Plausible role-multiset logit for counts c in DEFAULT_ROLES order."""
    tank, bru, heal, sup, ra, ma = c
    v = 0.0
    if heal == 0:
        v -= 0.7
    if tank + bru == 0:
        v -= 0.6
    if ra == 0:
        v -= 0.4
    if ra + ma == 0:
        v -= 0.5
    for n in (tank, heal, sup, ma):
        v -= 0.5 * max(0, n - 2)
    if tank == 1 and heal == 1:
        v += 0.1
    return v


def make(kind="benign", seed=1, truth_seed=0, noise=1.0, kappa=0.0):
    heroes, maps, tiers, roles = hero_list(), map_list(), list(DEFAULT_TIERS), list(DEFAULT_ROLES)
    H, M, T = len(heroes), len(maps), len(tiers)
    tr = np.random.default_rng(truth_seed)
    rng = np.random.default_rng(seed)
    hero_role = np.array([roles.index(blizz_role(h)) for h in heroes])

    # truth
    base = tr.normal(0, 0.12, H)
    hero_ctx = base[:, None, None] + tr.normal(0, 0.04, (H, M, 1)) + tr.normal(0, 0.03, (H, 1, T))
    hero_ctx[:, :, tiers.index("unknown")] = hero_ctx[:, :, :3].mean(2)
    pw = tr.normal(0, 0.03, (H, H))
    pair_within = np.triu(pw, 1) + np.triu(pw, 1).T
    role_table = np.zeros(ROLE_CODES)
    role_var = np.full(ROLE_CODES, 1.0)       # impossible codes: wide prior
    for c in product(range(6), repeat=N_ROLES):
        if sum(c) != 5:
            continue
        code = int(np.dot(c, ROLE_POW))
        role_table[code] = structural_role_value(c)
        # rare multisets get wider posteriors
        rare = sum(max(0, n - 2) for n in c) + (c[2] == 0) + (c[0] + c[1] == 0)
        role_var[code] = 0.002 * (1 + 10 * rare)
    kit = np.clip(np.array([ROLE_KIT[roles[r]] for r in hero_role]) + tr.normal(0, 0.12, (H, K)), 0, 1)
    knots = np.tile(np.linspace(0.0, 5.0, 6), (K, 1))
    gain = tr.uniform(0.05, 0.25, K)
    shape_vals = gain[:, None] * np.log1p(knots) - gain[:, None] * 0.6 * knots / 5 * np.log1p(5)
    shape_vals -= shape_vals[:, :1]
    map_axis = tr.normal(0, 0.03, (M, K))
    cross = tr.normal(0, 0.015, (H, H))
    np.fill_diagonal(cross, 0)
    W = np.zeros((K, K))
    nz = tr.choice(K * K, 20, replace=False)
    W.flat[nz] = tr.normal(0, 0.02, 20)

    # fitting noise for this pack (W vs judge differ here)
    s = noise
    hero_ctx = hero_ctx + rng.normal(0, 0.05 * s, hero_ctx.shape)
    pn = rng.normal(0, 0.02 * s, (H, H))
    pair_within = pair_within + np.triu(pn, 1) + np.triu(pn, 1).T
    role_table = role_table + rng.normal(0, 0.05 * s, ROLE_CODES) * (role_var < 1.0)
    cross = cross + rng.normal(0, 0.01 * s, (H, H)) * (1 - np.eye(H))
    shape_vals = shape_vals + rng.normal(0, 0.01 * s, shape_vals.shape)
    shape_vals -= shape_vals[:, :1]

    hero_ctx_var = np.full(hero_ctx.shape, (0.05 * s) ** 2) * rng.uniform(0.5, 2.0, (H, 1, 1))
    pair_within_var = np.full((H, H), (0.02 * s) ** 2)
    np.fill_diagonal(pair_within_var, 0)

    rules = [
        {"id": "no_waveclear_three_lane", "beta": -0.12, "predicates": [
            {"field": "kit_sum", "key": "waveclear", "op": "<=", "value": 1.0}]},
        {"id": "double_global", "beta": 0.05, "predicates": [
            {"field": "kit_sum", "key": "global", "op": ">=", "value": 1.2}]},
        {"id": "solo_frontline_braxis", "beta": -0.08, "predicates": [
            {"field": "role_count", "key": "Tank", "op": "==", "value": 0},
            {"field": "role_count", "key": "Bruiser", "op": "<=", "value": 1},
            {"field": "map", "key": None, "op": "==", "value": "Braxis Holdout"}]},
    ]
    if kind == "exploit":
        role_table[5 * ROLE_POW[roles.index("Tank")]] = EXPLOIT_TANK_BONUS
        rules.append({"id": "cliff_waveclear_no_healer", "beta": EXPLOIT_CLIFF_BETA, "predicates": [
            {"field": "role_count", "key": "Healer", "op": "==", "value": 0},
            {"field": "kit_sum", "key": "waveclear", "op": ">=", "value": 2.4}]})
    elif kind != "benign":
        raise ValueError(kind)

    arrays = dict(hero_ctx=hero_ctx, pair_within=pair_within, role_table=role_table, kit=kit,
                  shape_knots=knots, shape_vals=shape_vals, map_axis=map_axis, cross=cross, W=W,
                  hero_ctx_var=hero_ctx_var, pair_within_var=pair_within_var,
                  role_table_var=role_var, hero_role=hero_role)
    term_ids = ["hero_ctx", "pair_within", "role_table", "shape", "map_axis", "cross_hero",
                "cross_kit"] + [f"rule:{r['id']}" for r in rules]
    model_id = f"synthetic-{kind}-t{truth_seed}-s{seed}"
    meta = {"version": 1, "model_id": model_id, "heroes": heroes, "maps": maps, "tiers": tiers,
            "roles": roles, "K": K, "axes": AXES, "kappa": kappa, "rules": rules,
            "term_ids": term_ids,
            "card_hashes": {t: hashlib.sha256(f"{model_id}:{t}".encode()).hexdigest()[:16]
                            for t in term_ids},
            "synthetic": {"kind": kind, "seed": seed, "truth_seed": truth_seed, "noise": noise}}
    return LeafModel(arrays, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default="benign", choices=["benign", "exploit"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--truth-seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=0.0)
    a = ap.parse_args()
    m = make(a.kind, a.seed, a.truth_seed, a.noise, a.kappa)
    m.save(a.out)
    m2 = LeafModel.load(a.out)
    print(a.out, m2.model_id, m2.export_check())


if __name__ == "__main__":
    main()
