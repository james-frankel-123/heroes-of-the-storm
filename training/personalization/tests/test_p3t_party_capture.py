"""P3 party drafter: capture-ratio CI clusters by lobby (p3t_party_report.capture).
  C1 with both seats identical (perfectly correlated within a lobby), the CI equals the CI of one seat
     alone: seats are resampled together, not as independent units
Run: python3 personalization/tests/test_p3t_party_capture.py"""
import os
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="p3t_cap_")
os.environ["P3_CACHE"] = os.path.join(TMP, "cache")
os.environ["P3_RESULTS"] = os.path.join(TMP, "results")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import p3t_party_report as PR


def main():
    rng = np.random.RandomState(0)
    n = 300
    V2, V1 = {}, {}
    for m in range(1, 6):
        g = rng.normal(3 * m, 6, n)
        for t in (0, 1):
            for V, val in ((V2, g), (V1, g)):
                V[f"eff_m{m}_t{t}__aware"] = ({k: val for k in ("J", "J_cal", "CONS", "record")}, None)
                V[f"eff_m{m}_t{t}__generic"] = ({k: np.zeros(n) for k in ("J", "J_cal", "CONS", "record")}, None)
    r = PR.capture(V2)
    # the same lobbies as one seat: drop seat 1 by giving it the seat-0 values is what V2 already does,
    # so the lobby-clustered CI must match a one-seat computation with the same bootstrap draws
    rng2 = np.random.RandomState(1)
    idx = [rng2.randint(0, n, n) for _ in range(2000)]
    num = V2["eff_m1_t0__aware"][0]["J"]
    den = V2["eff_m5_t0__aware"][0]["J"]
    bs = [num[i].mean() / den[i].mean() for i in idx]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    assert abs(r[1]["J"][1] - round(lo, 3)) < 2e-3 and abs(r[1]["J"][2] - round(hi, 3)) < 2e-3, (r[1]["J"], lo, hi)
    print("C1 ok", r[1]["J"])
    print("ALL OK")


if __name__ == "__main__":
    main()
