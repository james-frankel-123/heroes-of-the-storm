"""P3 tournament: frozen lobby exclusions (p3t_excluded_replays.json, p3t_data.excluded_replays).
  X1 the committed file holds 300 tune and 300 tournament replay IDs, disjoint, matching their md5
  X2 gi_of_replays maps replay IDs to the rows of a lobby table (missing IDs are ignored)
Run: python3 personalization/tests/test_p3t_exclusions.py"""
import hashlib
import json
import os
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="p3t_excl_")
os.environ["P3_CACHE"] = os.path.join(TMP, "cache")
os.environ["P3_RESULTS"] = os.path.join(TMP, "results")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import p3t_data as TD


def main():
    z = json.load(open(TD.EXCLUDED_REPLAYS))
    tune, tour = TD.excluded_replays("tune"), TD.excluded_replays("tournament")
    assert len(tune) == 300 and len(tour) == 300 and not tune & tour
    for k in ("tune", "tournament"):
        assert hashlib.md5(np.array(z[k], np.int64).tobytes()).hexdigest() == z[k + "_md5"]
    print("X1 ok")
    rid = np.array(sorted(tune))[:5]
    L = {"replay_id": np.r_[np.array([1, 2, 3], np.int64), rid]}
    assert TD.gi_of_replays(L, tune) == {3, 4, 5, 6, 7}
    assert TD.gi_of_replays(L, {999999999}) == set()
    print("X2 ok")
    print("ALL OK")


if __name__ == "__main__":
    main()
