"""P3 tournament: the per-pass J_cal fit (p3t_judge.cal_fit, cal_coefs).
  J1 on synthetic games drawn from y ~ a logit p + k (c2 dS + c3 dO), the cross-fitted fit recovers a and k
  J2 cal_coefs refuses a judge artifact without the fitted coefficients (no hard-coded fallback)
Run: python3 personalization/tests/test_p3t_judge_cal.py"""
import os
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="p3t_cal_")
os.environ["P3_CACHE"] = os.path.join(TMP, "cache")
os.environ["P3_RESULTS"] = os.path.join(TMP, "results")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import p3t_judge as JG


def main():
    rng = np.random.RandomState(1)
    n = 200000
    off = rng.normal(0, 0.4, n)
    dS, dO = rng.normal(0, 0.05, n), rng.normal(0, 2.0, n)
    a, k, c2, c3 = 1.2, 0.9, 3.0, -0.04
    p = 1 / (1 + np.exp(-(a * off + k * (c2 * dS + c3 * dO))))
    y = (rng.rand(n) < p).astype(float)
    r = JG.cal_fit(off, dS, dO, y)
    assert abs(r["cal_a"] - a) < 4 * r["cal_a_se"] + 0.02, r
    assert abs(r["cal_k"] - k) < 4 * r["cal_k_se"] + 0.05, r
    print("J1 ok", round(r["cal_a"], 3), round(r["cal_k"], 3))
    assert JG.cal_coefs(r) == (r["cal_a"], r["cal_k"])
    try:
        JG.cal_coefs({"c2_skill": 3.0})
        raise AssertionError("J2: judge without cal_a accepted")
    except KeyError:
        pass
    print("J2 ok")
    print("ALL OK")


if __name__ == "__main__":
    main()
