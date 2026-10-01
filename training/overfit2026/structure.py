"""
Team structure indicators shared by the composition audit, the corrected
judges and the rescoring.

struct_vec(team) -> [no_healer, no_frontline, stack] (0/1, non-exclusive),
with the conventions of shared.is_degenerate: Blizzard roles from
HERO_ROLE_FINE, Uther counts as frontline when a second healer is present,
stack = 3+ of a stacking-bad role (Tank, Melee Assassin, Support, Healer).
A team is degenerate iff any indicator is 1.

kind(team) is the exclusive label used in the audit tables
(no_healer > no_frontline > stack > ok).
"""
import os
import sys

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING_DIR)

import numpy as np

from shared import HERO_ROLE_FINE, FINE_TO_BLIZZ_ROLE, DEGEN_STACK_ROLES

STRUCT_NAMES = ["no_healer", "no_frontline", "stack"]
KINDS = ["ok", "no_healer", "no_frontline", "stack"]
_ROLE = {}


def _role(h):
    r = _ROLE.get(h)
    if r is None:
        r = _ROLE[h] = FINE_TO_BLIZZ_ROLE.get(HERO_ROLE_FINE.get(h, ""), "Ranged Assassin")
    return r


def struct_vec(team):
    roles = [_role(h) for h in team]
    n_heal = roles.count("Healer")
    front = any(r in ("Tank", "Bruiser") for r in roles)
    if not front and "Uther" in team and n_heal >= 2:
        front = True
    stack = any(roles.count(r) >= 3 for r in DEGEN_STACK_ROLES)
    return (float(n_heal == 0), float(not front), float(stack))


def struct_matrix(teams):
    return np.array([struct_vec(t) for t in teams], np.float64).reshape(-1, 3)


def kind_of(S):
    """Exclusive kind index per row of a struct matrix (KINDS order)."""
    S = np.asarray(S)
    k = np.zeros(len(S), np.int64)
    k[(S[:, 2] > 0)] = 3
    k[(S[:, 1] > 0)] = 2
    k[(S[:, 0] > 0)] = 1
    return k


def is_degen_struct(S):
    return np.asarray(S).max(axis=1) > 0
