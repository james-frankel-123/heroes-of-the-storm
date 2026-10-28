"""P3 tournament analysis helpers.
  A1 paired_record uses only players covered under both arms and pairs a player's record under a
     with his record under b across the two seat orders; coverage shares per arm
  A2 every system kind that rotates F prior seeds by lobby counts its seeds (mcts_x, mcts_pp included)
Run: python3 personalization/tests/test_p3t_analyze_pairs.py"""
import os
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="p3t_an_")
os.environ["P3_CACHE"] = os.path.join(TMP, "cache")
os.environ["P3_RESULTS"] = os.path.join(TMP, "results")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import p3t_analyze as AN


def main():
    nan = np.nan
    # one lobby; (a, b): a drafts slots 0-4, b slots 5-9; (b, a): b drafts 0-4, a drafts 5-9
    ab = np.array([[0.1, nan, 0.3, nan, nan, 0.0, 0.2, nan, nan, nan]])
    ba = np.array([[0.0, 0.5, nan, nan, nan, 0.4, nan, 0.1, nan, nan]])
    P = {("a", "b"): {"rec_slot": ab}, ("b", "a"): {"rec_slot": ba}}
    r = AN.paired_record(P, "a", "b")
    # a's values: slots 0-4 from ab, 5-9 from ba: [0.1, nan, 0.3, nan, nan, 0.4, nan, 0.1, nan, nan]
    # b's values: slots 0-4 from ba, 5-9 from ab: [0.0, 0.5, nan, nan, nan, 0.0, 0.2, nan, nan, nan]
    # covered under both: slot 0 (0.1 - 0.0) and slot 5 (0.4 - 0.0): mean 0.25
    assert abs(r["diff"] - 0.25) < 1e-12, r
    assert abs(r["cov_a"] - 0.4) < 1e-12 and abs(r["cov_b"] - 0.4) < 1e-12 and abs(r["cov_both"] - 0.2) < 1e-12
    assert AN.paired_record({("a", "b"): {"rec_slot": ab}}, "a", "b") is None
    print("A1 ok")
    for k in ("mcts", "policy", "mcts_x", "mcts_pp", "mcts_vn", "policy_pp", "policy_rich"):
        assert k in AN.SEEDED_KINDS
    print("A2 ok")
    print("ALL OK")


if __name__ == "__main__":
    main()
