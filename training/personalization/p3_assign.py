"""
P3 assign mode, host side (numpy only; float64 mirror of the kernels'
best_assignment, see p3_mcts_core docstring and cuda_personal/personal_kernel.cu).

Kernel order: steps follow DRAFT_TEAM / IS_PICK; team t's slots are 5t..5t+4.
"""
import itertools

import numpy as np

DRAFT_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
IS_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
PERMS5 = np.array(list(itertools.permutations(range(5))), np.int64)   # lexicographic, identity first


def best_assignment(s5, off5, heroes, b2, b3):
    """Assign mode, one team (float64 mirror of the kernel's best_assignment;
    same rule as p3_dr_drafter._assign_terms). s5, off5: (5, NUM_HEROES) the
    team's slot rows; heroes: its picks in pick-step order (<= 5). Slot i
    plays pick perm[i] (no pick if perm[i] >= len(heroes)); over
    itertools.permutations(range(5)) (lexicographic, identity first) the first
    permutation maximizing b2 S + b3 O wins. Returns (S, O, perm)."""
    h = np.asarray(heroes, np.int64)
    n = len(h)
    s5 = np.asarray(s5, np.float64)
    off5 = np.asarray(off5, np.float64)
    use = PERMS5 < n                                   # (120, 5) slot i has a pick
    hp = h[np.minimum(PERMS5, max(n - 1, 0))] if n else np.zeros_like(PERMS5)
    i = np.arange(5)[None, :]
    Sa = np.where(use, s5[i, hp], 0.0).sum(1)
    Oa = np.where(use, off5[i, hp], 0.0).sum(1)
    obj = b2 * Sa + b3 * Oa
    k = int(np.argmax(obj))   # first maximum
    return float(Sa[k]), float(Oa[k]), tuple(int(x) for x in PERMS5[k])


def team_picks(acts):
    """Kernel-order picks of each team in pick-step order, from 16 actions."""
    return [[int(acts[k]) for k in range(16) if IS_PICK[k] and DRAFT_TEAM[k] == t] for t in (0, 1)]


def assignment_terms(s, off, acts, b2, b3):
    """Assign mode for a complete draft: S, O per team and, per team, the slot
    that plays its j-th pick (2 lists of slot indices, pick-step order)."""
    S, O, slots = [0.0, 0.0], [0.0, 0.0], []
    for t, hs in enumerate(team_picks(acts)):
        S[t], O[t], perm = best_assignment(s[5 * t:5 * t + 5], off[5 * t:5 * t + 5], hs, b2, b3)
        slot_of_pick = [0] * len(hs)
        for i, j in enumerate(perm):
            if j < len(hs):
                slot_of_pick[j] = 5 * t + i
        slots.append(slot_of_pick)
    return S, O, slots
