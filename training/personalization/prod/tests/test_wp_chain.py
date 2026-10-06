"""Synthetic tests for prod/wp_chain.py. Run: python3 training/personalization/prod/tests/test_wp_chain.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import numpy as np

from personalization.prod import wp_chain as W

DAY = 86400


def ts(iso):
    return W.day_end_ts(iso) - DAY  # 00:00 UTC of that day


def make_registry(path):
    W.append_vintage(path, {"vintage_id": "v1", "weights": ["w1.pt"], "stats_path": "s1.json",
                            "stats_cutoff_date": "2026-01-31", "trained_through_date": "2026-01-31",
                            "build_cutoff": "2.55.10.1000", "created_at": "2026-02-01T04:00:00+00:00"})
    W.append_vintage(path, {"vintage_id": "v2", "weights": ["w2.pt"], "stats_path": "s2.json",
                            "stats_cutoff_date": "2026-02-28", "trained_through_date": "2026-02-27",
                            "build_cutoff": "2.55.11.2000", "created_at": "2026-03-01T04:00:00+00:00"})
    return W.load_registry(path)


def synth_games(rng, n_per_month=5000):
    """Games Jan to Apr 2026; build 2.55.10.1000 in Jan, 2.55.11.2000 in Feb,
    2.55.12.3000 from Mar. Outcomes drawn from p_true."""
    parts = []
    for m, (start, build) in enumerate([("2026-01-01", "2.55.10.1000"), ("2026-02-01", "2.55.11.2000"),
                                        ("2026-03-01", "2.55.12.3000"), ("2026-04-01", "2.55.12.3000")]):
        end = ts(start) + rng.randint(3600, 27 * DAY, n_per_month)
        p = rng.uniform(0.3, 0.7, n_per_month)
        parts.append(dict(end_ts=end, build=np.array([build] * n_per_month), p_true=p,
                          y=(rng.rand(n_per_month) < p).astype(float),
                          game_length=np.full(n_per_month, 1200)))
    g = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    g["replay_id"] = np.arange(len(g["y"])) + 1000
    return g


def true_model(v, games):
    return games["p_true"]


def leaky_model(v, games):
    return np.clip(0.85 * games["p_true"] + 0.15 * games["y"], 0.01, 0.99)


def test_build_key():
    assert W.build_key("2.55.10.1000") < W.build_key("2.55.11.900") < W.build_key("2.56.0.1")
    assert W.build_key("2.55") == W.build_key("2.55.0.0")


def test_vintage_for_both_clocks(reg):
    # March game on a newer build: v2 precedes on both clocks
    assert W.vintage_for(reg, "2.55.12.3000", ts("2026-03-05"), 1200)["vintage_id"] == "v2"
    # March date but still on v2's build_cutoff build: build clock refuses v2, v1 is fine
    assert W.vintage_for(reg, "2.55.11.2000", ts("2026-03-05"), 1200)["vintage_id"] == "v1"
    # newer build but dated inside v2's date coverage: date clock refuses v2
    assert W.vintage_for(reg, "2.55.12.3000", ts("2026-02-20"), 1200)["vintage_id"] == "v1"
    # January game on v1's build: no vintage precedes it
    assert W.vintage_for(reg, "2.55.10.1000", ts("2026-01-20"), 1200) is None
    # a game ending 10 minutes after midnight of Mar 1 that started on Feb 28 is inside v2
    assert W.vintage_for(reg, "2.55.12.3000", ts("2026-03-01") + 600, 1200)["vintage_id"] == "v1"
    # unknown length: start = end - 1h, stricter
    assert W.vintage_for(reg, "2.55.12.3000", ts("2026-03-01") + 1800, None)["vintage_id"] == "v1"


def test_registry_append_only(path):
    reg = W.load_registry(path)
    try:
        W.append_vintage(path, dict(reg["vintages"][0]))
        raise AssertionError("duplicate id accepted")
    except ValueError:
        pass
    try:
        W.append_vintage(path, {"vintage_id": "v0", "weights": ["w.pt"], "stats_path": "s.json",
                                "stats_cutoff_date": "2026-01-15", "trained_through_date": "2026-01-15",
                                "build_cutoff": "2.55.9.1", "created_at": "x"})
        raise AssertionError("backwards cutoff accepted")
    except ValueError:
        pass
    assert [v["vintage_id"] for v in W.load_registry(path)["vintages"]] == ["v1", "v2"]


def test_rescore_and_in_sample(reg, games):
    rows, rep = W.rescore(games, reg, true_model, since_vintage=None, min_games=1000)
    a = W.assign_vintages(reg, games["build"], games["end_ts"], games["game_length"])
    assert rep["scored"] == int((a >= 0).sum()) and rep["unscorable_no_causal_vintage"] == int((a < 0).sum())
    assert set(np.unique(rows["vintage_id"])) == {"v1", "v2"}
    assert np.all(np.diff(rows["replay_id"]) > 0)
    # every row's vintage precedes its game
    for vid in ("v1", "v2"):
        v = [x for x in reg["vintages"] if x["vintage_id"] == vid][0]
        s = rows["vintage_id"] == vid
        bk = W.build_keys(games["build"][np.searchsorted(games["replay_id"], rows["replay_id"][s])])
        assert W.after_coverage(v, bk, rows["start_ts"][s]).all()
    # since v1: only games whose causal vintage is v2
    rows2, rep2 = W.rescore(games, reg, true_model, since_vintage="v1", min_games=1000)
    assert set(np.unique(rows2["vintage_id"])) == {"v2"} and rep2["scored"] == int((a == 1).sum())
    assert rep2["skipped_older_vintage"] == int((a == 0).sum())
    # calibrated out-of-sample scores: ratio near 1 in every judged month
    assert rep["noise"]["ok"], rep["noise"]
    for m, r in rep["noise"]["months"].items():
        assert abs(r["ratio"] - 1) < 0.02, (m, r)
    # in-sample scoring raises
    jan = {k: v[:10] for k, v in games.items()}
    try:
        W.score_with(reg["vintages"][1], jan, true_model)
        raise AssertionError("in-sample scoring did not raise")
    except W.InSampleError:
        pass


def test_noise_band_catches_leak(reg, games):
    rows, rep = W.rescore(games, reg, leaky_model, min_games=1000)
    assert not rep["noise"]["ok"]
    assert all(r["ratio"] < 0.98 for r in rep["noise"]["months"].values() if r["judged"])
    try:
        W.rescore(games, reg, leaky_model, min_games=1000, strict=True)
        raise AssertionError("strict did not raise")
    except W.NoiseBandError:
        pass


def main():
    rng = np.random.RandomState(0)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "wp_vintages.json")
        reg = make_registry(path)
        games = synth_games(rng)
        test_build_key()
        test_vintage_for_both_clocks(reg)
        test_registry_append_only(path)
        test_rescore_and_in_sample(reg, games)
        test_noise_band_catches_leak(reg, games)
    print("test_wp_chain: all passed")


if __name__ == "__main__":
    main()
