"""
Checks of the weekly frozen-coefficient refresh (run_weekly_refresh.sh, results/P3_WEEKLY.md).

  regress-states --tag D --base B
      The state writer on build D (frozen coefficients of B, cap today) must
      reproduce the published def1027 site states exactly on B's as-of day:
      the query file of the last B site build (cache/site_ref_<B>/states_q) is
      answered on D and compared bit for bit with B's answers (s, sd,
      sd_display, n): counts exact, estimates within float32 rounding (1e-6).
      Accounts the base build did not know, and accounts with new history or a
      corrected scoreboard join (--skip-keys), are reported, not compared.
  changed-keys --base B --out keys.npy
      Player keys whose Heroes Profile history was downloaded after the base
      build's store (done.jsonl marks newer than its built_at): their pre-cap
      evidence legitimately changes, so the regression checks leave them out.
  join-keys --base B --out keys.npy
      Player keys of the base build with a scoreboard row its legacy metric join
      took from another game (int64 overflow of the join key); the weekly builds
      use the exact join, so these players' states legitimately differ.
  refit-check --tag D --base B
      Reports (never refits) whether the frozen coefficients still fit the
      games since the base build's cap, on the last 28 days of post-cap games:
        game-level calibration slope (logistic of the outcome on logit p) with
        a week-block 95% CI; slot-level slope of the realized residual on the
        slot prediction; gain over population WP and over today's kernel
        (frozen), against the same numbers on the base build's OOT window;
        age of the coefficients.
      Rule (REFIT RECOMMENDED when any holds):
        R1 the game-level slope's 95% CI lies entirely outside [0.9, 1.1];
        R2 the gain over today's kernel on the last 28 days is below half of
           its OOT value (the player model has gone stale relative to the
           simplest kernel);
        R3 the coefficients are older than 92 days (quarterly refit).
      Writes results/def_<D>/refit_check.json and prints a one-line verdict.
"""
import argparse
import json
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))


def def_env(tag):
    import glob
    c = os.path.join(HERE, f"cache_def_{tag}")
    os.environ.update(P3_CACHE=c, P3_RESULTS=os.path.join(HERE, "results", f"def_{tag}"),
                      P3_RCF_KERNELS=os.path.join(c, "hs_kernels.npz"))
    rows = sorted(glob.glob(os.path.join(c, f"merged_rows_{tag}_*.npz")))
    assert rows, f"no merged rows in {c}"
    os.environ["P3_RCF_ROWS"] = rows[0]


def changed_keys(a):
    import datetime
    import numpy as np
    sys.path.insert(0, HERE)
    from p3_keys import player_keys
    meta = json.load(open(os.path.join(HERE, "cache", "full_history", a.base, "meta.json")))
    t0 = datetime.datetime.fromisoformat(meta["built_at"].replace("Z", "+00:00"))
    reg, bl = [], []
    for line in open(os.path.join(HERE, "cache", "mmr_history", "done.jsonl")):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if datetime.datetime.fromisoformat(d["at"].replace("Z", "+00:00")) > t0:
            reg.append(d["region"])
            bl.append(d["blizz_id"])
    keys = player_keys(np.array(reg, np.int64), np.array(bl, np.int64), label="changed") if reg else np.zeros(0, np.int64)
    np.save(a.out, np.unique(keys).astype(np.int64))
    print(f"changed-keys: {len(keys):,} accounts downloaded after {a.base}'s store ({meta['built_at']})")


def join_keys(a):
    """Players whose base-build metric rows (the zaug caches as built, so the
    numpy build that made them does not matter) differ from the exact join."""
    import glob
    import numpy as np
    c = os.path.join(HERE, f"cache_def_{a.base}")
    A = np.load(os.path.join(c, "rcf", "aug_merged.npz"))
    pk = np.asarray(np.load(os.path.join(c, "x_slots_ext.npz"), mmap_mode="r")["player_keys"])
    ar = A["rid"].astype(np.int64)
    ab = (pk[A["pid"]] & ((1 << 40) - 1)).astype(np.int64)
    bad = np.zeros(len(ar), bool)
    for card in ("a_shrink", "share6"):
        z = np.load(os.path.join(c, "rcf", f"sb_{card}.npz"))
        zr, zb = z["replay_id"].astype(np.int64), z["blizz_id"].astype(np.int64)
        ub = np.unique(np.r_[zb, ab])
        c2 = zr * (1 << 24) + np.searchsorted(ub, zb)
        o2 = np.argsort(c2, kind="stable")
        cs2 = c2[o2]
        q2 = ar * (1 << 24) + np.searchsorted(ub, ab)
        j2 = np.minimum(np.searchsorted(cs2, q2), len(cs2) - 1)
        hit = cs2[j2] == q2
        Zx = np.full((len(ar), z["Z"].shape[1]), np.nan, np.float16)
        Zx[hit] = z["Z"][o2[j2[hit]]]
        Zb = np.load(glob.glob(os.path.join(c, "rcf", f"zaug_{card}_*.npz"))[0])["Z"]
        diff = ~((Zx == Zb) | (np.isnan(Zx) & np.isnan(Zb)))
        rows = diff.any(1)
        bad |= rows
        print(f"join-keys {card}: {int(rows.sum()):,} base rows hold another game's scoreboard")
    keys = np.unique(pk[np.unique(A["pid"][bad])]).astype(np.int64)
    prev = np.load(a.out) if os.path.exists(a.out) else np.zeros(0, np.int64)
    np.save(a.out, np.union1d(prev, keys).astype(np.int64))
    print(f"join-keys: {len(keys):,} accounts; skip list now {len(np.union1d(prev, keys)):,}")


def regress_states(a):
    def_env(a.tag)
    os.environ.setdefault("P3_BAND", "anchored_v1")
    sys.path.insert(0, HERE)
    import tempfile
    import numpy as np
    import p3_rcf_states as RS
    ref = os.path.join(HERE, "cache", f"site_ref_{a.base}")
    skip = np.load(a.skip_keys) if a.skip_keys else np.zeros(0, np.int64)
    with tempfile.TemporaryDirectory() as td:
        RS.answer(os.path.join(ref, "states_q"), td)
        A = np.load(os.path.join(td, "site.npz"))
        Bz = np.load(os.path.join(ref, "states_a", "site.npz"))
        out = {}
        # accounts the base build did not know (only post-cap corpus games then) are new here
        assert np.array_equal(A["player_key"], Bz["player_key"])
        assert (A["known"] | ~Bz["known"]).all(), "an account known to the base build is unknown here"
        newly = A["known"] & ~Bz["known"]
        rows = ~np.isin(Bz["player_key"], skip) & Bz["known"]
        for k in ("s", "sd", "sd_display", "n", "known", "player_key", "as_of_day"):
            x, y = A[k][rows], Bz[k][rows]
            out[k] = {"equal": bool(np.array_equal(x, y, equal_nan=x.dtype.kind == "f")),
                      "max_abs": float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0}
    rep = {"tag": a.tag, "base": a.base, "queries": int(rows.sum()),
           "skipped_changed_accounts": int((np.isin(Bz["player_key"], skip) & Bz["known"]).sum()),
           "unknown_to_base": int((~Bz["known"]).sum()), "newly_known": int(newly.sum()),
           "fields": out}
    print(json.dumps(rep, indent=1))
    with open(os.path.join(HERE, "results", f"def_{a.tag}", "regress_states.json"), "w") as f:
        json.dump(rep, f, indent=1)
    # estimates: float32 rounding allowed (query batches differ in composition); everything else exact
    tol = {"s": 1e-6, "sd": 1e-6, "sd_display": 1e-6}
    bad = [k for k, v in out.items() if not (v["equal"] or v["max_abs"] <= tol.get(k, 0.0))]
    assert not bad, f"states on {a.tag} differ from {a.base} on its as-of day: {bad}"
    print(f"regress-states: {rep['queries']:,} published {a.base} states reproduced on {a.tag} "
          f"(max |s| difference {out['s']['max_abs']:.1e}; counts, sd and band exact)")


def refit_check(a):
    import datetime
    import numpy as np
    sys.path.insert(0, HERE)
    import p3_hs_core as C

    def ll_of(p, y):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def slope(p, y):
        lp = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
        return float(C.fit_logistic(lp[:, None], y)[1])

    rd = os.path.join(HERE, "results", f"def_{a.tag}", "rcf", "p3c")
    bd = os.path.join(HERE, "results", f"def_{a.base}", "rcf", "p3c")
    ch, td = np.load(os.path.join(rd, "chosen__merged.npz")), np.load(os.path.join(rd, "today__merged.npz"))
    assert np.array_equal(ch["g_replay_id"], td["g_replay_id"])
    y = ch["g_y"].astype(np.float64)
    pf = ch["g_win"] == 4
    last = int(ch["g_day"][pf].max())
    m = pf & (ch["g_day"] > last - 28)
    rng = np.random.default_rng(0)
    wk = ch["g_day"][m] // 7
    uw, inv = np.unique(wk, return_inverse=True)
    bs = []
    for _ in range(300):
        pick = rng.integers(0, len(uw), len(uw))
        sel = np.concatenate([np.flatnonzero(inv == i) for i in pick])
        bs.append(slope(ch["g_p"][m][sel], y[m][sel]))
    s_game = slope(ch["g_p"][m], y[m])
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    gain = float(1000 * (ll_of(ch["g_p0"][m], y[m]) - ll_of(ch["g_p"][m], y[m])).mean())
    gain_vs_today = float(1000 * (ll_of(td["g_p"][m], y[m]) - ll_of(ch["g_p"][m], y[m])).mean())
    sm = (ch["s_win"] == 4) & (ch["s_day"] > last - 28)
    cv = np.cov(ch["s_pred"][sm].astype(np.float64), ch["s_r"][sm].astype(np.float64))
    s_slot = float(cv[0, 1] / cv[0, 0])
    bch, btd = np.load(os.path.join(bd, "chosen__merged.npz")), np.load(os.path.join(bd, "today__merged.npz"))
    bo = bch["g_win"] == 3
    yb = bch["g_y"].astype(np.float64)
    base_vs_today = float(1000 * (ll_of(btd["g_p"][bo], yb[bo]) - ll_of(bch["g_p"][bo], yb[bo])).mean())
    base_gain = float(1000 * (ll_of(bch["g_p0"][bo], yb[bo]) - ll_of(bch["g_p"][bo], yb[bo])).mean())
    built = datetime.date.fromtimestamp(os.path.getmtime(os.path.join(bd, "chosen__merged.npz")))
    age = (datetime.date.today() - built).days
    d0 = datetime.date(1970, 1, 1)
    rules = {"R1_game_slope_ci_outside_0.9_1.1": bool(ci[1] < 0.9 or ci[0] > 1.1),
             "R2_gain_vs_today_below_half_of_oot": bool(gain_vs_today < 0.5 * base_vs_today),
             "R3_coefficients_older_than_92_days": bool(age > 92)}
    rep = {"tag": a.tag, "base": a.base, "window": [str(d0 + datetime.timedelta(days=last - 27)), str(d0 + datetime.timedelta(days=last))],
           "games": int(m.sum()), "slots": int(sm.sum()),
           "game_calibration_slope": s_game, "game_calibration_slope_week_ci": ci, "slot_calibration_slope": s_slot,
           "gain_over_population_wp_x1000": gain, "gain_over_today_kernel_x1000": gain_vs_today,
           "base_oot_gain_over_population_wp_x1000": base_gain, "base_oot_gain_over_today_kernel_x1000": base_vs_today,
           "accuracy": float(((ch["g_p"][m] > .5) == (y[m] == 1)).mean()),
           "draft_only_accuracy": float(((ch["g_p0"][m] > .5) == (y[m] == 1)).mean()),
           "coefficients_built": str(built), "coefficients_age_days": age, "rules": rules,
           "verdict": "REFIT RECOMMENDED" if any(rules.values()) else "keep frozen coefficients"}
    with open(os.path.join(HERE, "results", f"def_{a.tag}", "refit_check.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
    print(f"refit-check {a.tag}: {rep['verdict']} (game slope {s_game:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], "
          f"gain vs today's kernel {gain_vs_today:+.2f} vs OOT {base_vs_today:+.2f}, age {age} days)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["regress-states", "refit-check", "changed-keys", "join-keys"])
    ap.add_argument("--tag")
    ap.add_argument("--out")
    ap.add_argument("--skip-keys")
    ap.add_argument("--base", default="def1027")
    a = ap.parse_args()
    {"regress-states": regress_states, "refit-check": refit_check, "changed-keys": changed_keys, "join-keys": join_keys}[a.cmd](a)


if __name__ == "__main__":
    main()
