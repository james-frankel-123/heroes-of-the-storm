"""Hero-set contract for the P3 research code: the 90-hero v1 encoding of
training/shared.py. Every P3 script takes its hero count from here."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import shared

if shared.HERO_SET != "v1":
    raise SystemExit(f"P3 research runs on the 90-hero v1 set; HOTS_HERO_SET={shared.HERO_SET!r}")
NUM_HEROES = shared.NUM_HEROES
assert NUM_HEROES == 90, NUM_HEROES
HKEY = 128  # stride of player x hero keys (pid * HKEY + hero)
assert NUM_HEROES < HKEY


def check_heroes(hero_names, hero=None):
    """Assert a slot table's hero list is shared.HEROES (v1) in some order of
    the same set, and indices are in range."""
    names = [str(h) for h in hero_names]
    if len(names) != NUM_HEROES:
        raise AssertionError(f"slot table has {len(names)} heroes; P3 expects the {NUM_HEROES}-hero v1 set")
    got, want = set(names), set(shared.HEROES)
    if got != want:
        raise AssertionError(f"slot table hero set differs from shared.HEROES (v1): "
                             f"extra {sorted(got - want)}, missing {sorted(want - got)}")
    if hero is not None and len(hero):
        lo, hi = int(np.min(hero)), int(np.max(hero))
        if lo < 0 or hi >= NUM_HEROES:
            raise AssertionError(f"hero index out of range [0, {NUM_HEROES}): min {lo}, max {hi}")


def sample_rows(p, u):
    """Inverse-CDF sample per row of probability matrix p (B,H) with uniforms
    u (B,) or (B,1); only entries with p>0 can be chosen.

    The cumsum is normalized by the row total, so a total just under 1 no
    longer runs past the last entry. If rounding still lands on a p==0 entry,
    the next p>0 entry at or after it is taken, else the last p>0 entry."""
    p = np.asarray(p)
    u = np.asarray(u).reshape(-1, 1)
    pos = p > 0
    if not pos.any(1).all():
        raise ValueError("sample_rows: a row has no positive probability")
    c = p.cumsum(1)
    c = c / c[:, -1:]
    idx = np.minimum((c < u).sum(1), p.shape[1] - 1)
    ar = np.arange(len(p))
    bad = np.flatnonzero(~pos[ar, idx])
    for i in bad:
        nz = np.flatnonzero(pos[i])
        after = nz[nz >= idx[i]]
        idx[i] = after[0] if len(after) else nz[-1]
    return idx
