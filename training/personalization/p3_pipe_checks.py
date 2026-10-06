"""
Small consistency checks between pipeline stages (p3_pipe.py).

  slots   the rebuilt hs_slots.npz must equal the pre-October table except
          for the player keys (region handling, p3_keys); every cache keyed
          to slot order (mmr_at_game.npz, ...) stays aligned only if so.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import p3_hs_core as C

LEGACY = os.path.join(C.CACHE, "_legacy_pre_oct26")


def slots():
    new = np.load(C.SLOTS)
    path = os.path.join(LEGACY, "hs_slots.npz")
    if not os.path.exists(path):
        print(f"no legacy table at {path}; nothing to compare")
        return
    old = np.load(path)
    bad = []
    for k in old.files:
        if k not in new.files:
            bad.append(f"missing {k}")
            continue
        a, b = old[k], new[k]
        same = a.shape == b.shape and (np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f"
                                       else np.array_equal(a, b))
        if not same:
            bad.append(k)
    print(f"hs_slots rebuilt: {len(new['pid']):,} slots; differing arrays vs legacy: {bad or 'none'}")
    if [k for k in bad if k not in ("player_keys",)]:
        raise SystemExit("slot table changed beyond the player keys: caches keyed to slot order are stale")


if __name__ == "__main__":
    {"slots": slots}[sys.argv[1]]()
