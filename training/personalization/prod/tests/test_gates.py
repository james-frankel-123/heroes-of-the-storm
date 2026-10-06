"""Synthetic tests for prod/gates.py. Run: python3 training/personalization/prod/tests/test_gates.py"""
import json
import os
import sys
import tempfile

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # tiny test: keep it on one core

TRAINING = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, TRAINING)
import numpy as np

from personalization.prod import gates as G

sys.path.insert(0, os.path.join(TRAINING, "personalization"))
import shared  # noqa: E402  (v1 hero list, through the same path p3_heroes uses)

SKILL = 3.7


def synth_split(rng, n_games=120_000, n_slots=200_000, t0=1_780_000_000, bad=False):
    wp = 1 / (1 + np.exp(-rng.normal(0, 0.4, n_games)))
    x = rng.normal(0, 0.05, n_games)
    p_true = 1 / (1 + np.exp(-(np.log(wp / (1 - wp)) + SKILL * x)))
    y = (rng.rand(n_games) < p_true).astype(float)
    p = p_true
    if bad:  # overconfident personal model
        lo = np.log(p / (1 - p))
        p = 1 / (1 + np.exp(-1.4 * lo))
    start = t0 + rng.randint(0, 60 * 86400, n_games)
    n = np.where(rng.rand(n_slots) < 0.3, 0, rng.randint(1, 80, n_slots))
    pred = rng.normal(0, 0.03, n_slots)
    slope = np.where(n == 0, 1.3, np.where(n <= 20, 1.05, 0.8))
    real = slope * pred + rng.normal(0, 0.3, n_slots) + np.where(n == 0, -0.003, 0.0)
    s_start = t0 + rng.randint(0, 60 * 86400, n_slots)
    fetched = s_start - rng.randint(1, 86400 * 30, n_slots)
    if bad:
        fetched[:5] = s_start[:5] + 10  # state rows fetched after the game started
    tier = rng.choice(np.array(["low", "mid", "high"]), n_games, p=[0.2, 0.5, 0.3])
    p_off = 1 / (1 + np.exp(-(np.log(wp / (1 - wp)) + 0.5 * SKILL * x)))
    return dict(replay_id=np.arange(n_games) + 1, start_ts=start, y=y, wp=wp, p=p, x_skill=x,
                p_offset=p_off, tier=tier, slot_n=n, slot_pred=pred, slot_real=real,
                slot_start_ts=s_start, slot_state_fetched_max=fetched)


def gains(d):
    a = float(np.mean(G.ll_vec(d["wp"], d["y"]) - G.ll_vec(d["p"], d["y"])))
    b = float(np.mean(G.ll_vec(d["wp"], d["y"]) - G.ll_vec(d["p_offset"], d["y"])))
    return a, b


def main():
    rng = np.random.RandomState(0)
    good = {"v2": synth_split(rng), "oot": synth_split(rng, t0=1_790_000_000)}
    gv2, ov2 = gains(good["v2"])
    goot, _ = gains(good["oot"])
    # the synthetic gains are not the paper's; point the references at them
    ov = {"REF_GAIN": {"v2": gv2, "oot": goot}, "REF_OFFSET_GAIN": {"v2": ov2}, "BOOT": 100,
          "DEPLOY_MIN_MONTH_GAIN": min(gv2, goot) / 2}
    with tempfile.TemporaryDirectory() as td:
        gp = os.path.join(td, "golden.npz")
        meta = G.make_golden(gp, good, shared.HEROES, note="synthetic")
        assert meta["splits"]["v2"]["games"] == 120_000 and "tier_dist" in meta["splits"]["oot"]
        rep = G.run(gp, overrides=ov)
        assert rep["pass"], rep["failed"]
        c = rep["checks"]
        assert abs(c["skill_coef_v2"]["value"] - SKILL) < 0.3
        assert abs(c["slot_slope_n0_oot"]["value"] - 1.3) < 0.1
        assert abs(c["neverplayed_bias_v2"]["value_pp"] + 0.3) < 0.2
        assert c["hero_set"]["pass"] and c["tiers"]["pass"]

        # CLI round trip, exit code 0
        ovp = os.path.join(td, "ov.json")
        json.dump(ov, open(ovp, "w"))
        out = os.path.join(td, "rep.json")
        assert G.main(["--golden", gp, "--out", out, "--thresholds", ovp]) == 0
        assert json.load(open(out))["pass"] is True

        # the paper's reference gains do not fit this synthetic set: gain checks fail
        rep_ref = G.run(gp, overrides={"BOOT": 100})
        assert "gain_v2" in rep_ref["failed"] and "gain_oot" in rep_ref["failed"]

        # fresh labels: one changed label fails the tier check
        lp = os.path.join(td, "labels.npz")
        tiers = good["v2"]["tier"].copy()
        tiers[7] = "high" if tiers[7] != "high" else "low"
        np.savez(lp, replay_id=good["v2"]["replay_id"], tier=tiers)
        rep_l = G.run(gp, labels_path=lp, overrides=ov)
        assert rep_l["failed"] == ["tiers"], rep_l["failed"]

        # a broken model, late state rows and a 91-hero list all fail; exit code 1
        bad = {"v2": synth_split(rng, bad=True), "oot": synth_split(rng, t0=1_790_000_000, bad=True)}
        bp = os.path.join(td, "bad.npz")
        G.make_golden(bp, bad, list(shared.HEROES) + ["Xal'atath"])
        rep_b = G.run(bp, overrides=ov)
        for k in ("cal_slope_v2", "cal_slope_oot", "visibility_v2", "visibility_oot", "hero_set"):
            assert k in rep_b["failed"], (k, rep_b["failed"])
        assert rep_b["checks"]["visibility_v2"]["late_rows"] == 5
        assert G.main(["--golden", bp, "--out", out, "--thresholds", ovp]) == 1

        # deploy gate: last OOT month with no gain blocks deployment only
        dep = {k: dict(v) for k, v in good.items()}
        o = dep["oot"]
        months = np.asarray(o["start_ts"], "datetime64[s]").astype("datetime64[M]")
        last = months == months.max()
        o["p"] = np.where(last, o["wp"], o["p"])
        dp = os.path.join(td, "dep.npz")
        G.make_golden(dp, dep, shared.HEROES)
        rep_d = G.run(dp, overrides=dict(ov, REF_GAIN={"v2": gv2, "oot": gains(o)[0]}))
        assert not rep_d["deploy"]["deploy_month_gain"]["pass"] and not rep_d["deploy_pass"]

        # tampering with a frozen array is detected
        z = dict(np.load(gp))
        z["v2__y"] = z["v2__y"].copy()
        z["v2__y"][0] = 1 - z["v2__y"][0]
        tp = os.path.join(td, "tampered.npz")
        np.savez(tp, **z)
        try:
            G.load_golden(tp)
            raise AssertionError("tampered golden accepted")
        except ValueError:
            pass

        # standalone state visibility check
        assert G.check_visibility_state([10, 20], [30, 30], as_of=25)["pass"]
        assert not G.check_visibility_state([10, 30], [30, 30])["pass"]
    print("test_gates: all passed")


if __name__ == "__main__":
    main()
